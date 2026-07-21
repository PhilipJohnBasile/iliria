//! `cargo test`-integrated mutation-fuzz smoke run: a FAST (tens of
//! thousands of iterations per corpus, seconds not minutes), fully
//! deterministic pass that must ALWAYS succeed as part of the normal test
//! suite. It exists to make "the strict validators never panic on
//! adversarial input" a property `cargo test` actively checks on every run,
//! not just a claim in a doc comment.
//!
//! The full evidence-gathering campaign (hundreds of thousands of
//! iterations per corpus, per the design's ~100k-1M range) lives in
//! `src/bin/fuzz_report.rs` (`cargo run --release --bin fuzz_report`) --
//! deliberately NOT part of `cargo test`, so the normal test suite stays
//! fast; this file and that binary share all their actual logic via
//! `src/fuzz_support.rs`.

use artifact_validator::fuzz_support::{run_mode15_campaign, run_quant_format_campaign, run_safetensors_campaign};

const SMOKE_ITERS: usize = 20_000;
const SEED: u64 = 0xC0FFEE_u64;

#[test]
fn quant_format_mutation_smoke() {
    let stats = run_quant_format_campaign(SMOKE_ITERS, SEED);
    assert!(
        stats.panics.is_empty(),
        "infer_strict/infer_c_lenient panicked on {} adversarial inputs, e.g.: {:?}",
        stats.panics.len(),
        &stats.panics[..stats.panics.len().min(5)]
    );
    assert_eq!(stats.iterations, SMOKE_ITERS);
    println!(
        "[quant_format] iters={} strict_ok={} unknown={} (silently->Int2: {}) ambiguous={} (silently resolved: {}) invalid_dims/overflow={}",
        stats.iterations,
        stats.strict_ok,
        stats.strict_unknown,
        stats.divergence_unknown_silently_int2,
        stats.strict_ambiguous,
        stats.divergence_ambiguous_silently_resolved,
        stats.strict_invalid_dims_or_overflow
    );
}

#[test]
fn safetensors_mutation_smoke() {
    let stats = run_safetensors_campaign(SMOKE_ITERS, SEED.wrapping_add(1));
    assert!(
        stats.panics.is_empty(),
        "parse_strict/parse_c_lenient panicked on {} adversarial inputs, e.g.: {:?}",
        stats.panics.len(),
        &stats.panics[..stats.panics.len().min(5)]
    );
    assert_eq!(stats.iterations, SMOKE_ITERS);
    println!(
        "[safetensors] iters={} strict_accept={} strict_reject={} lenient_raw_examined={} divergences={} reasons={:?}",
        stats.iterations,
        stats.strict_accept,
        stats.strict_reject,
        stats.lenient_tensor_raw_examined,
        stats.divergences,
        stats.divergence_reasons
    );
}

#[test]
fn mode15_mutation_smoke() {
    let stats = run_mode15_campaign(SMOKE_ITERS, SEED.wrapping_add(2));
    assert!(
        stats.panics.is_empty(),
        "open_structural/verify_tensor_crc panicked on {} adversarial inputs, e.g.: {:?}",
        stats.panics.len(),
        &stats.panics[..stats.panics.len().min(5)]
    );
    assert_eq!(stats.iterations, SMOKE_ITERS);
    println!(
        "[mode15] iters={} structural_accept={} structural_reject={} payload_only_mutations={} crc_caught={}",
        stats.iterations,
        stats.structural_accept,
        stats.structural_reject,
        stats.payload_only_mutations,
        stats.payload_only_crc_caught
    );
}
