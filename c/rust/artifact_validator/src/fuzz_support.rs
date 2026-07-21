//! Shared mutation-fuzz machinery: a small deterministic PRNG (no external
//! `rand` crate -- this crate is std-only), byte-mutation operators, seed
//! corpora, and the three campaign runners (`quant_format`, `safetensors`,
//! `mode15`) shared by the fast `cargo test` smoke run
//! (`tests/fuzz_mutation.rs`) and the full evidence-gathering campaign
//! (`src/bin/fuzz_report.rs`).
//!
//! DETERMINISM: every campaign takes an explicit `u64` seed and touches no
//! other source of randomness (no `/dev/urandom`, no `SystemTime`) so a run
//! is exactly reproducible -- required by the design's "deterministic seed"
//! constraint and just generally good practice for a fuzz harness whose
//! whole job is to produce trustworthy counts.
//!
//! PANIC POLICY: every call into this crate's own parsing code
//! (`quant_format::infer_*`, `safetensors::parse_*`, `mode15::open_structural`
//! / `verify_*`) is wrapped in `std::panic::catch_unwind`. A panic is
//! recorded (input + message) rather than silently ignored, and every
//! campaign runner's caller (`tests/fuzz_mutation.rs`, `fuzz_report.rs`)
//! treats a nonempty panic list as a hard failure. See `json_mini.rs`'s doc
//! comment for the one thing `catch_unwind` can NOT save us from (a stack
//! overflow from unbounded JSON nesting) -- that is handled separately, by
//! `json_mini`'s own depth cap, not by anything here.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{mode15, quant_format, safetensors};

// ------------------------------------------------------------------- PRNG --

/// splitmix64 -- small, fast, deterministic, no external crate. Good enough
/// statistical quality for mutation selection (this is a fuzz harness, not
/// a cryptographic use case).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // avoid the degenerate seed==0 producing a short early cycle
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn gen_byte(&mut self) -> u8 {
        self.next_u64() as u8
    }
    pub fn gen_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
    /// Uniform in `[lo, hi)`; returns `lo` if the range is empty.
    pub fn gen_range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            return lo;
        }
        lo + (self.next_u64() as usize) % (hi - lo)
    }
    pub fn gen_i64_signed_ish(&mut self, magnitude: i64) -> i64 {
        let m = magnitude.max(1) as u64;
        (self.next_u64() % (2 * m + 1)) as i64 - m as i64
    }
}

// -------------------------------------------------------- byte mutations --

/// Applies ONE mutation operator to `base`, returning a fresh `Vec<u8>`
/// (never mutates in place -- keeps every call site trivially reusable
/// across many iterations from the same seed corpus entry).
pub fn mutate(rng: &mut Rng, base: &[u8]) -> Vec<u8> {
    let mut buf = base.to_vec();
    if buf.is_empty() {
        return buf;
    }
    match rng.gen_range(0, 4) {
        0 => {
            // single byte flip
            let idx = rng.gen_range(0, buf.len());
            buf[idx] = rng.gen_byte();
        }
        1 => {
            // small cluster of byte flips
            let max_n = buf.len().clamp(1, 8);
            let n = rng.gen_range(1, max_n + 1);
            for _ in 0..n {
                let idx = rng.gen_range(0, buf.len());
                buf[idx] = rng.gen_byte();
            }
        }
        2 => {
            // truncate to a random shorter (or equal) length, including 0
            let new_len = rng.gen_range(0, buf.len() + 1);
            buf.truncate(new_len);
        }
        _ => {
            // truncate, then flip a byte inside whatever remains (compound)
            let new_len = rng.gen_range(0, buf.len() + 1);
            buf.truncate(new_len);
            if !buf.is_empty() {
                let idx = rng.gen_range(0, buf.len());
                buf[idx] = rng.gen_byte();
            }
        }
    }
    buf
}

/// Flips exactly one byte strictly inside `[range.0, range.1)`, leaving the
/// total length (and everything outside the range) untouched -- used to
/// isolate "same-size payload-only corruption" from "corrupted a
/// size/header field", see `mode15`'s campaign below.
pub fn mutate_within_range(rng: &mut Rng, base: &[u8], range: (usize, usize)) -> Vec<u8> {
    let mut buf = base.to_vec();
    let (s, e) = range;
    if e > s && e <= buf.len() {
        let idx = s + rng.gen_range(0, e - s);
        let old = buf[idx];
        let mut nb = rng.gen_byte();
        if nb == old {
            nb = nb.wrapping_add(1); // guarantee an actual change
        }
        buf[idx] = nb;
    }
    buf
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

// ================================================================= quant ==

#[derive(Debug, Default)]
pub struct QuantFormatStats {
    pub iterations: usize,
    pub strict_ok: usize,
    pub strict_unknown: usize,
    pub strict_ambiguous: usize,
    pub strict_invalid_dims_or_overflow: usize,
    /// Strict said `UnknownByteSize`; the C-equivalent ternary silently
    /// defaulted to Int2 anyway. THE core divergence this ternary produces.
    pub divergence_unknown_silently_int2: usize,
    /// Strict said `AmbiguousByteSize` (multiple candidates coincide);
    /// the ternary silently picked one with no way for a caller to know.
    pub divergence_ambiguous_silently_resolved: usize,
    pub panics: Vec<String>,
}

/// Generates one adversarial `(nbytes, o, i)` case. Mixes: exact matches to
/// a real candidate size (so the corpus isn't 100% garbage), near-miss
/// (off by a small delta), pure noise, and extreme dimensions (to exercise
/// the overflow-guarding path).
fn gen_quant_case(rng: &mut Rng) -> (i64, i64, i64) {
    let o = 1 + (rng.next_u64() % 4096) as i64;
    let i = 1 + (rng.next_u64() % 4096) as i64;
    let bucket = rng.gen_range(0, 10);
    let nbytes = match bucket {
        0..=3 => {
            // exact match to one of the three candidates
            let which = rng.gen_range(0, 3);
            let fmts = [quant_format::QuantFormat::Int8, quant_format::QuantFormat::Int4, quant_format::QuantFormat::Int2];
            fmts[which].expected_bytes(o, i).unwrap_or(0)
        }
        4..=6 => {
            // near miss: a real candidate size plus/minus a small delta
            let which = rng.gen_range(0, 3);
            let fmts = [quant_format::QuantFormat::Int8, quant_format::QuantFormat::Int4, quant_format::QuantFormat::Int2];
            let base = fmts[which].expected_bytes(o, i).unwrap_or(0);
            base + rng.gen_i64_signed_ish(4)
        }
        7 => {
            // pure noise
            rng.next_u64() as i64 & 0x7FFF_FFFF
        }
        8 => {
            // extreme dims, small nbytes -- probes the overflow-guard path
            return (rng.gen_i64_signed_ish(16), i64::MAX / 2 + rng.gen_i64_signed_ish(8), i64::MAX / 2);
        }
        _ => {
            // zero/negative dims -- probes NonPositiveDims
            return (rng.next_u64() as i64 & 0xFFFF, rng.gen_i64_signed_ish(2), rng.gen_i64_signed_ish(2));
        }
    };
    (nbytes, o, i)
}

pub fn run_quant_format_campaign(iterations: usize, seed: u64) -> QuantFormatStats {
    let mut rng = Rng::new(seed);
    let mut stats = QuantFormatStats { iterations, ..Default::default() };

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // silence default panic-to-stderr printing during the campaign

    for iter in 0..iterations {
        let (nbytes, o, i) = gen_quant_case(&mut rng);

        let strict = catch_unwind(AssertUnwindSafe(|| quant_format::infer_strict(nbytes, o, i)));
        let lenient = catch_unwind(AssertUnwindSafe(|| quant_format::infer_c_lenient(nbytes, o, i)));

        match strict {
            Err(e) => stats.panics.push(format!(
                "iter {iter}: infer_strict panicked: {} (nbytes={nbytes}, o={o}, i={i})",
                panic_message(e)
            )),
            Ok(strict_result) => match lenient {
                Err(e) => stats.panics.push(format!(
                    "iter {iter}: infer_c_lenient panicked: {} (nbytes={nbytes}, o={o}, i={i})",
                    panic_message(e)
                )),
                Ok(lenient_result) => match strict_result {
                    Ok(_fmt) => stats.strict_ok += 1, // agreement is guaranteed by construction, see quant_format.rs doc comment
                    Err(quant_format::QuantFormatError::UnknownByteSize { .. }) => {
                        stats.strict_unknown += 1;
                        if lenient_result == Some(quant_format::QuantFormat::Int2) {
                            stats.divergence_unknown_silently_int2 += 1;
                        }
                    }
                    Err(quant_format::QuantFormatError::AmbiguousByteSize { .. }) => {
                        stats.strict_ambiguous += 1;
                        if lenient_result.is_some() {
                            stats.divergence_ambiguous_silently_resolved += 1;
                        }
                    }
                    Err(quant_format::QuantFormatError::NonPositiveDims { .. })
                    | Err(quant_format::QuantFormatError::SizeOverflow) => {
                        stats.strict_invalid_dims_or_overflow += 1;
                    }
                },
            },
        }
    }

    std::panic::set_hook(prev_hook);
    stats
}

// ============================================================ safetensors ==

#[derive(Debug, Default)]
pub struct SafetensorsStats {
    pub iterations: usize,
    pub strict_accept: usize,
    pub strict_reject: usize,
    pub lenient_clean_reject: usize,
    pub lenient_header_unbounded: usize,
    pub lenient_tensor_clean_reject: usize,
    pub lenient_tensor_unguarded_gap: usize,
    pub lenient_tensor_raw_examined: usize,
    /// Of the "raw" (individually well-formed, un-cross-validated) tensors
    /// the lenient/C-equivalent mode silently accepted, how many actually
    /// violate an invariant the strict validator enforces -- the core
    /// safetensors-side divergence count.
    pub divergences: usize,
    pub divergence_reasons: HashMap<&'static str, usize>,
    pub panics: Vec<String>,
}

fn seed_corpus_safetensors() -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    // 1) one small F32 tensor
    {
        let json = br#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut blob = Vec::new();
        blob.extend_from_slice(&(json.len() as u64).to_le_bytes());
        blob.extend_from_slice(json);
        blob.extend(std::iter::repeat_n(0x11u8, 24));
        out.push(blob);
    }
    // 2) multiple tensors + a __metadata__ entry + a BF16 dtype
    {
        let json = br#"{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[10],"data_offsets":[8,18]}}"#;
        let mut blob = Vec::new();
        blob.extend_from_slice(&(json.len() as u64).to_le_bytes());
        blob.extend_from_slice(json);
        blob.extend(std::iter::repeat_n(0x22u8, 18));
        out.push(blob);
    }
    // 3) a packed-quantized-style U8 blob (shape=[nbytes], matching how
    //    mode15_container.py / the engine's own .qs sidecars actually store
    //    packed weights: dtype U8, shape=[len(blob)])
    {
        let json = br#"{"e.gate_proj.weight":{"dtype":"U8","shape":[100],"data_offsets":[0,100]}}"#;
        let mut blob = Vec::new();
        blob.extend_from_slice(&(json.len() as u64).to_le_bytes());
        blob.extend_from_slice(json);
        blob.extend(std::iter::repeat_n(0x33u8, 100));
        out.push(blob);
    }
    out
}

pub fn run_safetensors_campaign(iterations: usize, seed: u64) -> SafetensorsStats {
    let mut rng = Rng::new(seed);
    let corpus = seed_corpus_safetensors();
    let mut stats = SafetensorsStats { iterations, ..Default::default() };

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    for iter in 0..iterations {
        let base = &corpus[rng.gen_range(0, corpus.len())];
        let mutated = mutate(&mut rng, base);

        let strict = catch_unwind(AssertUnwindSafe(|| safetensors::parse_strict(&mutated)));
        let lenient = catch_unwind(AssertUnwindSafe(|| safetensors::parse_c_lenient(&mutated)));

        let strict_result = match strict {
            Ok(r) => r,
            Err(e) => {
                stats.panics.push(format!("iter {iter}: parse_strict panicked: {} (len={})", panic_message(e), mutated.len()));
                continue;
            }
        };
        match &strict_result {
            Ok(_) => stats.strict_accept += 1,
            Err(_) => stats.strict_reject += 1,
        }

        let lenient_result = match lenient {
            Ok(r) => r,
            Err(e) => {
                stats.panics.push(format!("iter {iter}: parse_c_lenient panicked: {} (len={})", panic_message(e), mutated.len()));
                continue;
            }
        };

        match lenient_result {
            safetensors::LenientHeaderOutcome::CleanReject(_) => stats.lenient_clean_reject += 1,
            safetensors::LenientHeaderOutcome::HeaderLenUnbounded { .. } => stats.lenient_header_unbounded += 1,
            safetensors::LenientHeaderOutcome::Parsed { data_len, tensors } => {
                for t in tensors {
                    match t {
                        safetensors::TensorFieldsOutcome::CleanReject { .. } => stats.lenient_tensor_clean_reject += 1,
                        safetensors::TensorFieldsOutcome::UnguardedGap { .. } => stats.lenient_tensor_unguarded_gap += 1,
                        safetensors::TensorFieldsOutcome::Raw(raw) => {
                            stats.lenient_tensor_raw_examined += 1;
                            if let Some(reason) = raw.violates_strict_invariant(data_len) {
                                stats.divergences += 1;
                                *stats.divergence_reasons.entry(reason).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    std::panic::set_hook(prev_hook);
    stats
}

// ================================================================ mode15 ==

#[derive(Debug, Default)]
pub struct Mode15Stats {
    pub iterations: usize,
    pub structural_accept: usize,
    pub structural_reject: usize,
    /// Same-size mutations confined to the payload region (header/size
    /// fields untouched): structural validation accepts every one of these
    /// by construction (it doesn't look at payload content) -- tallied to
    /// make that fact an explicit, checked count rather than an assumption.
    pub payload_only_mutations: usize,
    pub payload_only_structural_accept: usize,
    /// Of those structurally-accepted, same-size, payload-corrupted blobs,
    /// how many did whole-tensor CRC32 verification catch. This is the
    /// checksum's positive-control value-add number.
    pub payload_only_crc_caught: usize,
    pub panics: Vec<String>,
}

fn seed_corpus_mode15() -> Vec<Vec<u8>> {
    vec![
        mode15::build_valid_fixture(16, 4, 4, 7),
        mode15::build_valid_fixture(37, 6, 8, 3),
        mode15::build_valid_fixture(0, 0, 4, 0), // degenerate zero-row tensor
        mode15::build_valid_fixture(256, 8, 256, 200),
    ]
}

pub fn run_mode15_campaign(iterations: usize, seed: u64) -> Mode15Stats {
    let mut rng = Rng::new(seed);
    let corpus = seed_corpus_mode15();
    let mut stats = Mode15Stats { iterations, ..Default::default() };

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    for iter in 0..iterations {
        let base = &corpus[rng.gen_range(0, corpus.len())];

        // Alternate between two mutation strategies each iteration: general
        // byte/truncation mutation (exercises structural validation broadly)
        // and payload-only same-size mutation (isolates the checksum's
        // marginal value -- see Mode15Stats doc comments).
        let payload_only = rng.gen_bool();

        let mutated = if payload_only {
            // Need a valid open first to know the payload range on `base`.
            let hdr = match mode15::open_structural(base, None, None, None) {
                Ok(h) => h,
                Err(_) => continue, // seed corpus entries are always valid; unreachable in practice
            };
            if hdr.payload_range.1 == hdr.payload_range.0 {
                continue; // zero-row fixture has no payload bytes to corrupt
            }
            stats.payload_only_mutations += 1;
            mutate_within_range(&mut rng, base, hdr.payload_range)
        } else {
            mutate(&mut rng, base)
        };

        let open_result = catch_unwind(AssertUnwindSafe(|| mode15::open_structural(&mutated, None, None, None)));
        let hdr = match open_result {
            Ok(Ok(h)) => h,
            Ok(Err(_)) => {
                stats.structural_reject += 1;
                continue;
            }
            Err(e) => {
                stats.panics.push(format!(
                    "iter {iter}: open_structural panicked: {} (len={})",
                    panic_message(e),
                    mutated.len()
                ));
                continue;
            }
        };
        stats.structural_accept += 1;
        if payload_only {
            stats.payload_only_structural_accept += 1;
            let crc_result = catch_unwind(AssertUnwindSafe(|| mode15::verify_tensor_crc(&hdr, &mutated)));
            match crc_result {
                Ok(Err(_)) => stats.payload_only_crc_caught += 1,
                Ok(Ok(())) => {} // extraordinarily rare CRC32 collision, or the mutation happened to be a no-op
                Err(e) => stats.panics.push(format!(
                    "iter {iter}: verify_tensor_crc panicked: {} (len={})",
                    panic_message(e),
                    mutated.len()
                )),
            }
        }
    }

    std::panic::set_hook(prev_hook);
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_for_a_fixed_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn small_campaigns_run_without_panicking() {
        let q = run_quant_format_campaign(2_000, 1);
        assert!(q.panics.is_empty(), "quant_format panics: {:?}", q.panics);
        let s = run_safetensors_campaign(2_000, 2);
        assert!(s.panics.is_empty(), "safetensors panics: {:?}", s.panics);
        let m = run_mode15_campaign(2_000, 3);
        assert!(m.panics.is_empty(), "mode15 panics: {:?}", m.panics);
    }

    #[test]
    fn quant_format_campaign_actually_finds_the_known_divergence() {
        // With 2000 iterations across the mixed generator, the "unknown
        // byte size silently -> Int2" bucket should be nonempty -- if this
        // ever goes to zero it means gen_quant_case stopped generating
        // unknown-size cases, which would silently defeat the whole point.
        let q = run_quant_format_campaign(2_000, 7);
        assert!(q.strict_unknown > 0, "expected some UnknownByteSize cases in the mix");
        assert_eq!(q.divergence_unknown_silently_int2, q.strict_unknown);
    }

    #[test]
    fn mode15_campaign_demonstrates_checksum_value_add() {
        let m = run_mode15_campaign(4_000, 9);
        assert!(m.payload_only_mutations > 0);
        // Structural validation, by design, cannot see payload-only corruption:
        assert_eq!(m.payload_only_structural_accept, m.payload_only_mutations);
        // CRC32 should catch the overwhelming majority (all but an
        // astronomically unlikely hash collision or a mutation that
        // happened to land back on the original byte).
        let miss_rate = 1.0 - (m.payload_only_crc_caught as f64 / m.payload_only_structural_accept as f64);
        assert!(miss_rate < 0.01, "CRC32 caught only {:.2}% of payload-only mutations", 100.0 * (1.0 - miss_rate));
    }
}
