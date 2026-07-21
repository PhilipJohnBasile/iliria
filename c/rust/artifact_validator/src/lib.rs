//! # artifact_validator
//!
//! A standalone, `std`-only, dependency-free Rust crate that STRICTLY parses
//! and validates iliria's two on-disk artifact formats:
//!
//!   - the **safetensors** container header (`safetensors.rs`) -- what
//!     `c/st.h`'s `st_init`/`st_read_f32`/`st_read_raw` read;
//!   - the **Mode-1.5 ("MH01")** compressed tensor blob container
//!     (`mode15.rs`) -- what `c/mode15_reader.h`/`c/mode15_reader.c` define
//!     and validate (that C reader is already thorough; see `mode15.rs`'s
//!     doc comment for why this crate ports it anyway);
//!
//! plus a standalone reimplementation of the int8/int4/int2 format-inference
//! ternary from `c/glm.c`'s `qt_from_disk`/`expert_load` (`quant_format.rs`).
//!
//! ## Why this crate exists
//!
//! This is a migration-TRIGGER-EVIDENCE exercise, not a production
//! dependency. The standing policy for iliria (a C engine; GLM-5.2 744B
//! MoE inference) is: "Rust owns untrusted-input parsing + artifact
//! validation, C keeps the compute kernels" -- but only once concrete
//! parser-fuzz findings justify paying for a boundary rewrite. This crate:
//!
//!   1. builds a STRICT, panic-free, typed validator for both formats
//!      (`quant_format::QuantFormat`, an explicit enum, is the concrete
//!      version of "the tagged-format-enum idea the C side needs" --
//!      unknown/ambiguous input is a typed `Err`, never a silent
//!      fallthrough default);
//!   2. mutation-fuzzes it (see `fuzz_support.rs`, `tests/fuzz_mutation.rs`,
//!      `src/bin/fuzz_report.rs`) against a faithful Rust reimplementation
//!      of the CURRENT C logic, to produce actual counts of which malformed
//!      inputs the C loader would silently mishandle;
//!   3. reports those counts honestly as the trigger evidence, including
//!      where the evidence does NOT justify a rewrite (see the crate's
//!      accompanying report to the reviewer for the full go/no-go).
//!
//! NOT linked into the engine: no FFI, no `cdylib`/`staticlib` target,
//! nothing in `c/glm.c` or any other C file was modified to build or use
//! this crate. `cargo build`/`cargo test` need no network access (see
//! `Cargo.toml`: zero `[dependencies]`).

pub mod crc32;
pub mod json_mini;
pub mod mode15;
pub mod quant_format;
pub mod safetensors;

pub mod fuzz_support;
