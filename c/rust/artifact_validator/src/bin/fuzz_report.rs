//! Full evidence-gathering mutation-fuzz campaign (`cargo run --release
//! --bin fuzz_report`). This is the binary whose output is the actual
//! migration-trigger evidence: per-corpus iteration counts, how often the
//! strict validator rejects mutated input, and -- the point of the whole
//! exercise -- how often a faithful Rust reimplementation of glm.c's/st.h's
//! CURRENT logic would have silently accepted an input the strict validator
//! correctly flagged as malformed.
//!
//! Iteration count defaults to 300_000 per corpus (900_000 total) -- inside
//! the design's ~100k-1M range -- and is overridable via the `FUZZ_ITERS` env
//! var (e.g. `FUZZ_ITERS=1000000 cargo run --release --bin fuzz_report`).
//! Deterministic: same iteration count + same (fixed, hardcoded) seeds
//! always reproduces the same counts.

use artifact_validator::fuzz_support::{run_mode15_campaign, run_quant_format_campaign, run_safetensors_campaign};
use std::time::Instant;

fn iters_from_env() -> usize {
    std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(300_000)
}

fn main() {
    let iters = iters_from_env();
    println!("==================================================================");
    println!(" artifact_validator :: migration-trigger-evidence fuzz campaign");
    println!(" iterations per corpus: {iters}  (3 corpora, {} total)", iters * 3);
    println!("==================================================================\n");

    let t0 = Instant::now();

    // ---- quant_format --------------------------------------------------
    let qt0 = Instant::now();
    let q = run_quant_format_campaign(iters, 0x5151_5151_5151_5151);
    let qdt = qt0.elapsed();
    println!("---- 1. quant_format (glm.c's int8/int4/int2 ternary, c/glm.c:857,1206,1319) ----");
    println!("iterations                              : {}", q.iterations);
    println!("strict: exactly-one-match (Ok)          : {}", q.strict_ok);
    println!("strict: UnknownByteSize (Err)           : {}", q.strict_unknown);
    println!("  -> C-equivalent ternary silently chose Int2 for ALL of these: {}", q.divergence_unknown_silently_int2);
    println!("strict: AmbiguousByteSize (Err)         : {}", q.strict_ambiguous);
    println!("  -> C-equivalent ternary silently resolved ALL of these too : {}", q.divergence_ambiguous_silently_resolved);
    println!("strict: non-positive dims / overflow    : {}", q.strict_invalid_dims_or_overflow);
    println!("panics (MUST be 0)                      : {}", q.panics.len());
    if !q.panics.is_empty() {
        for p in q.panics.iter().take(10) {
            println!("  PANIC: {p}");
        }
    }
    println!("wall time                               : {:.2}s\n", qdt.as_secs_f64());

    // ---- safetensors -----------------------------------------------------
    let st0 = Instant::now();
    let s = run_safetensors_campaign(iters, 0x5A5A_5A5A_5A5A_5A5A);
    let sdt = st0.elapsed();
    println!("---- 2. safetensors header (c/st.h's st_init/st_read_f32/st_read_raw) ----");
    println!("iterations                              : {}", s.iterations);
    println!("strict accept / reject                  : {} / {}", s.strict_accept, s.strict_reject);
    println!("lenient (C-equivalent): header-level clean reject : {}", s.lenient_clean_reject);
    println!("lenient: header length unbounded-claim bucket     : {}", s.lenient_header_unbounded);
    println!("lenient: per-tensor clean reject (bad dtype)      : {}", s.lenient_tensor_clean_reject);
    println!("lenient: per-tensor unguarded-gap (bad arity/etc) : {}", s.lenient_tensor_unguarded_gap);
    println!("lenient: per-tensor raw fields examined           : {}", s.lenient_tensor_raw_examined);
    println!("  -> of those, VIOLATE a strict invariant (silently mishandled by st.h): {}", s.divergences);
    if !s.divergence_reasons.is_empty() {
        let mut reasons: Vec<_> = s.divergence_reasons.iter().collect();
        reasons.sort_by(|a, b| b.1.cmp(a.1));
        for (reason, count) in reasons {
            println!("       [{count:>8}]  {reason}");
        }
    }
    println!("panics (MUST be 0)                      : {}", s.panics.len());
    if !s.panics.is_empty() {
        for p in s.panics.iter().take(10) {
            println!("  PANIC: {p}");
        }
    }
    println!("wall time                               : {:.2}s\n", sdt.as_secs_f64());

    // ---- mode15 ------------------------------------------------------------
    let mt0 = Instant::now();
    let m = run_mode15_campaign(iters, 0x1234_5678_9ABC_DEF0);
    let mdt = mt0.elapsed();
    println!("---- 3. Mode-1.5 / MH01 (c/mode15_reader.c port; NOT wired into glm.c today) ----");
    println!("iterations                              : {}", m.iterations);
    println!("structural accept / reject               : {} / {}", m.structural_accept, m.structural_reject);
    println!("same-size payload-only mutations         : {}", m.payload_only_mutations);
    println!("  -> structural validation accepts ALL of them (blind to payload content): {}", m.payload_only_structural_accept);
    println!("  -> whole-tensor CRC32 catches            : {}", m.payload_only_crc_caught);
    if m.payload_only_structural_accept > 0 {
        let catch_rate = 100.0 * m.payload_only_crc_caught as f64 / m.payload_only_structural_accept as f64;
        println!("     (CRC32 catch rate on same-size corruption: {catch_rate:.4}% -- this is what a checksum buys \
                   that safetensors' format, with no checksum field at all, structurally cannot provide, in ANY language)");
    }
    println!("panics (MUST be 0)                      : {}", m.panics.len());
    if !m.panics.is_empty() {
        for p in m.panics.iter().take(10) {
            println!("  PANIC: {p}");
        }
    }
    println!("wall time                               : {:.2}s\n", mdt.as_secs_f64());

    let total_panics = q.panics.len() + s.panics.len() + m.panics.len();
    println!("==================================================================");
    println!(" TOTAL wall time: {:.2}s  |  TOTAL panics across all campaigns: {}", t0.elapsed().as_secs_f64(), total_panics);
    println!("==================================================================");

    if total_panics > 0 {
        eprintln!("\nFAIL: {total_panics} panic(s) recorded -- see above. This would contradict this crate's core claim.");
        std::process::exit(1);
    }
}
