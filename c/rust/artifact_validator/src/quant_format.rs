//! The tagged-format-enum idea, standalone: a STRICT typed inference of
//! which quantized packing (int8 / int4 / int2) an expert weight tensor's
//! on-disk byte count implies, plus a LENIENT function that is a faithful,
//! line-for-line port of `glm.c`'s actual current behavior, for differential
//! fuzzing against the strict version.
//!
//! THE C CODE THIS MIRRORS (three call sites, byte-identical logic):
//! ```text
//! c/glm.c:857  (qt_from_disk):
//!     int fmt = (nb==(int64_t)O*I)?1 : (nb==(int64_t)O*((I+1)/2))?2 : 3;
//! c/glm.c:1206 (expert_load, mmap fast path):
//!     int fmt=(nb==(int64_t)OO[k]*II[k])?1:(nb==(int64_t)OO[k]*((II[k]+1)/2))?2:3;
//! c/glm.c:1319 (expert_load, slab path):
//!     int fmt = (nb==(int64_t)OO[k]*II[k])?1 : (nb==(int64_t)OO[k]*((II[k]+1)/2))?2 : 3;
//! ```
//!
//! In all three, `nb` is the tensor's byte length as read straight from the
//! safetensors JSON header's `data_offsets` diff (`c/st.h:140`,
//! `t->nbytes = b0 - a0`) -- i.e. UNTRUSTED, attacker/corruption-controlled
//! input -- while `O`/`I` are the engine's own expected matrix dimensions
//! (config-derived, validated elsewhere by the CKR() macros in
//! `c/glm.c:834-845`). The `else` branch is unconditional: any `nb` that
//! matches neither the int8 nor the int4 candidate size is silently treated
//! as int2, with NO check that `nb == O*ceil(I/4)` (the actual int2 size,
//! `c/glm.c:121`'s `qt_bytes()` formula for fmt==3). Downstream, `qalloc(nb)`
//! (`c/glm.c:859`) allocates exactly `nb` bytes, while consumers like
//! `embed_row` (`c/glm.c:1031-1036`) index that buffer using `O`/`I`, not
//! `nb` -- so an `nb` smaller than `O*ceil(I/4)` is a live heap-OOB-read
//! path, not merely a theoretical one.

/// Which packing a quantized `[O, I]` weight tensor is stored in. `Int2` is
/// spelled out explicitly here (never a bare fallthrough default) precisely
/// so "we don't actually know" has to be its own typed error instead of
/// silently becoming this variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Int8,
    Int4,
    Int2,
}

impl QuantFormat {
    pub fn name(self) -> &'static str {
        match self {
            QuantFormat::Int8 => "int8",
            QuantFormat::Int4 => "int4",
            QuantFormat::Int2 => "int2",
        }
    }

    /// The exact byte count this format implies for an `[o, i]` tensor,
    /// mirroring `c/glm.c:117-123`'s `qt_bytes()` (weights only, NOT
    /// counting the separate per-row f32 scale sidecar -- callers here only
    /// ever compare against the *weight* tensor's own `nbytes`, matching
    /// what glm.c's ternary itself compares against). `None` on overflow
    /// (checked i128 arithmetic; `o`/`i` are untrusted-until-validated in
    /// this crate's callers, so this must never wrap or panic).
    pub fn expected_bytes(self, o: i64, i: i64) -> Option<i64> {
        if o < 0 || i < 0 {
            return None;
        }
        let o = o as i128;
        let i = i as i128;
        let elems_per_row = match self {
            QuantFormat::Int8 => i,
            QuantFormat::Int4 => (i + 1) / 2,
            QuantFormat::Int2 => (i + 3) / 4,
        };
        let total = o.checked_mul(elems_per_row)?;
        i64::try_from(total).ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantFormatError {
    /// O or I is not a positive dimension -- not a valid tensor shape at all.
    NonPositiveDims { o: i64, i: i64 },
    /// int128 overflow computing a candidate size for these dims (defense in
    /// depth: O/I here have already been through CKR()'s bounds in the real
    /// engine, but this validator does not assume its caller did that).
    SizeOverflow,
    /// `nbytes` matched none of the three candidate sizes -- exactly the
    /// input class glm.c's ternary would silently accept as Int2 anyway.
    UnknownByteSize {
        nbytes: i64,
        expected_int8: i64,
        expected_int4: i64,
        expected_int2: i64,
    },
    /// `nbytes` matched MORE than one candidate size (possible for small `i`,
    /// e.g. i==1 makes all three formulas coincide) -- a real ambiguity
    /// glm.c's ternary resolves silently and arbitrarily (whichever branch
    /// is checked first wins), never surfaced to a caller.
    AmbiguousByteSize { nbytes: i64, matches: Vec<QuantFormat> },
}

/// STRICT inference: `nbytes` must equal EXACTLY ONE of the three candidate
/// sizes for `[o, i]`, or this is a typed `Err` -- never a silent default.
pub fn infer_strict(nbytes: i64, o: i64, i: i64) -> Result<QuantFormat, QuantFormatError> {
    if o <= 0 || i <= 0 {
        return Err(QuantFormatError::NonPositiveDims { o, i });
    }
    let candidates = [QuantFormat::Int8, QuantFormat::Int4, QuantFormat::Int2];
    let mut sizes = [0i64; 3];
    for (k, f) in candidates.iter().enumerate() {
        sizes[k] = f.expected_bytes(o, i).ok_or(QuantFormatError::SizeOverflow)?;
    }
    let matches: Vec<QuantFormat> = candidates
        .iter()
        .zip(sizes.iter())
        .filter(|(_, &sz)| sz == nbytes)
        .map(|(f, _)| *f)
        .collect();
    match matches.len() {
        0 => Err(QuantFormatError::UnknownByteSize {
            nbytes,
            expected_int8: sizes[0],
            expected_int4: sizes[1],
            expected_int2: sizes[2],
        }),
        1 => Ok(matches[0]),
        _ => Err(QuantFormatError::AmbiguousByteSize { nbytes, matches }),
    }
}

/// LENIENT, C-EQUIVALENT mimic of glm.c's actual current ternary (see the
/// three call sites quoted in the module doc comment -- this is a
/// line-for-line port, including the unconditional `else`). Deliberately
/// infallible: this function's whole point is to reproduce a real bug, not
/// to be a good validator. `o`/`i` are assumed already-positive (matching
/// the ternary's own assumption that the caller's config-derived O/I are
/// sane; this fn is only ever fed the same o/i the strict version was, in
/// this crate's fuzz harness, so that assumption is meaningful there).
///
/// Returns `None` only for the case this crate cannot faithfully model
/// without literally replicating C's wraparound `int64_t` multiplication
/// (o*i overflowing i64) -- everywhere else it always returns a format,
/// exactly like the C, which never fails either.
pub fn infer_c_lenient(nbytes: i64, o: i64, i: i64) -> Option<QuantFormat> {
    let int8_sz = (o as i128).checked_mul(i as i128)?;
    let int4_sz = (o as i128).checked_mul(((i as i128) + 1) / 2)?;
    let nbytes = nbytes as i128;
    if nbytes == int8_sz {
        Some(QuantFormat::Int8)
    } else if nbytes == int4_sz {
        Some(QuantFormat::Int4)
    } else {
        Some(QuantFormat::Int2) // <-- the unconditional fallback itself
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_accepts_exact_int8() {
        assert_eq!(infer_strict(6, 2, 3), Ok(QuantFormat::Int8));
    }
    #[test]
    fn strict_accepts_exact_int4() {
        // O=2, I=3 -> int4 bytes = 2*ceil(3/2) = 2*2 = 4
        assert_eq!(infer_strict(4, 2, 3), Ok(QuantFormat::Int4));
    }
    #[test]
    fn strict_accepts_exact_int2() {
        // O=2, I=5 -> int2 bytes = 2*ceil(5/4) = 2*2 = 4; int8=10, int4=2*3=6
        assert_eq!(infer_strict(4, 2, 5), Ok(QuantFormat::Int2));
    }
    #[test]
    fn strict_rejects_unknown_size() {
        let e = infer_strict(999, 2, 3).unwrap_err();
        assert!(matches!(e, QuantFormatError::UnknownByteSize { .. }));
    }
    #[test]
    fn strict_flags_ambiguous_small_i() {
        // i==1: int8=O*1, int4=O*ceil(1/2)=O*1, int2=O*ceil(1/4)=O*1 -- all coincide.
        let e = infer_strict(5, 5, 1).unwrap_err();
        match e {
            QuantFormatError::AmbiguousByteSize { matches, .. } => assert_eq!(matches.len(), 3),
            other => panic!("expected AmbiguousByteSize, got {other:?}"),
        }
    }
    #[test]
    fn lenient_defaults_unknown_to_int2() {
        // Same "unknown" input the strict version rejects above:
        assert_eq!(infer_c_lenient(999, 2, 3), Some(QuantFormat::Int2));
    }
    #[test]
    fn lenient_matches_strict_on_exact_sizes() {
        for (nb, o, i) in [(6i64, 2i64, 3i64), (4, 2, 3)] {
            assert_eq!(
                infer_c_lenient(nb, o, i).unwrap(),
                infer_strict(nb, o, i).unwrap()
            );
        }
    }
    #[test]
    fn strict_never_panics_on_huge_dims() {
        // Would overflow i64 multiplication if done naively -- must be a
        // typed error, not a panic (this crate builds with
        // overflow-checks=true even in release, see Cargo.toml).
        let r = infer_strict(1, i64::MAX, i64::MAX);
        assert!(r.is_err());
    }
    #[test]
    fn lenient_never_panics_on_huge_dims() {
        // i128 comfortably holds the product of two i64::MAX values (its
        // range is roughly (i64::MAX)^2 * 5), so this does NOT hit
        // infer_c_lenient's own overflow guard -- it just falls through to
        // its unconditional `else`, exactly like the real C ternary would.
        // This is a nice illustration in itself: strict correctly reports
        // "this size doesn't even fit back into an i64" (SizeOverflow,
        // see strict_never_panics_on_huge_dims above), while the lenient
        // C-equivalent mimic confidently answers "Int2" regardless -- no
        // panic either way, just a confident wrong-flavor answer from the
        // lenient side.
        let r = infer_c_lenient(1, i64::MAX, i64::MAX);
        assert_eq!(r, Some(QuantFormat::Int2));
    }
}
