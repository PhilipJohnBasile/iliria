//! Strict parser+validator for the safetensors container header, alongside a
//! LENIENT mode that faithfully mirrors `c/st.h`'s actual (unaudited-for-
//! cross-field-consistency) behavior, for differential fuzzing.
//!
//! ON-DISK LAYOUT (standard safetensors, as `c/st.h:119-145`'s `st_init`
//! reads it): 8 bytes LE `u64 header_len`, then `header_len` bytes of UTF-8
//! JSON (one object; each key except `"__metadata__"` maps to
//! `{"dtype": str, "shape": [u64...], "data_offsets": [u64, u64]}`), then
//! raw tensor data starting at `8 + header_len`, where a given tensor's
//! bytes are `data[data_offsets[0]..data_offsets[1]]`.
//!
//! THE TWO GAPS THIS MODULE'S STRICT/LENIENT SPLIT IS BUILT TO SURFACE
//! (both confirmed by reading `c/st.h`, not assumed):
//!
//! 1. `c/st.h:140`: `t->nbytes = b0 - a0;` -- NEVER checked `b0 >= a0`.
//!
//! 2. `c/st.h:136-141`: `numel` (product of `shape`) and `nbytes` (from
//!    `data_offsets`) are stored side by side with NO cross-check that
//!    `nbytes == numel * dtype_byte_size`. This matters because
//!    `st_read_f32` (`c/st.h:183-198`) then behaves as follows:
//!
//! ```text
//! malloc()s exactly `nbytes` bytes into `raw`, reads `nbytes` bytes into
//! it (internally consistent, bounded by the tensor's own declared byte
//! range) -- then:
//!   * BF16/F16 (c/st.h:191,193): walks p[i] for i in [0, numel) over
//!     that `raw` buffer -- if numel*2 > nbytes, this reads PAST `raw`
//!     (heap OOB read).
//!   * F32 (c/st.h:189): memcpy(out, raw, t->nbytes) where `out` is a
//!     CALLER-supplied buffer sized for `numel` floats (e.g. ld(),
//!     c/glm.c:878-882) -- if nbytes > numel*4, this OVERFLOWS `out`
//!     (heap OOB WRITE, the more severe direction).
//! ```
//!
//! Both directions are driven purely by one tensor's own JSON header entry
//! being internally inconsistent -- no engine-wide trace needed to see the
//! bug, and no cooperation from any other tensor's entry required.
//!
//! Also modeled (header-length has no cap before allocating it, `c/st.h:123`)
//! as its own `LenientOutcome` bucket -- see that type's doc comment for why
//! it is kept separate from `Accepted` rather than pretending to reproduce
//! the exact C crash.

use crate::json_mini::{self, JsonError};

/// Sanity cap this crate's STRICT validator enforces on the declared header
/// length before it will even attempt to slice/UTF-8-validate it. Chosen
/// generously (real GLM-5.2 safetensors headers with ~120k tensor entries,
/// per `c/st.h:36-38`'s own comment, are plausibly tens of MB of JSON) --
/// this is a backstop against a clearly-absurd claim, not a tight bound.
/// `c/st.h` itself has NO such cap at all; see `LenientOutcome::HeaderLenUnbounded`.
pub const SANE_HEADER_LEN_CAP: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl DType {
    pub fn byte_size(self) -> u64 {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    /// Full canonical safetensors dtype spelling set (for documentation /
    /// completeness) -- NOT the same as what this engine actually accepts,
    /// see `engine_supported`.
    pub fn from_str_any(s: &str) -> Option<DType> {
        match s {
            "F64" => Some(DType::F64),
            "F32" => Some(DType::F32),
            "F16" => Some(DType::F16),
            "BF16" => Some(DType::BF16),
            "I64" => Some(DType::I64),
            "I32" => Some(DType::I32),
            "I16" => Some(DType::I16),
            "I8" => Some(DType::I8),
            "U8" => Some(DType::U8),
            "BOOL" => Some(DType::Bool),
            _ => None,
        }
    }

    /// EXACTLY `c/st.h:49-56`'s `st_dtype_code()` whitelist (BF16/F16/F32/U8/I8
    /// -> codes 0/1/2/3/3; anything else -> `exit(1)`, fail-closed). This is
    /// what the strict validator holds tensors to, since "valid for this
    /// engine" is the property that actually matters here.
    pub fn engine_supported(s: &str) -> Option<DType> {
        match s {
            "BF16" => Some(DType::BF16),
            "F16" => Some(DType::F16),
            "F32" => Some(DType::F32),
            "U8" => Some(DType::U8),
            "I8" => Some(DType::I8),
            _ => None,
        }
    }

    /// `c/st.h`'s own internal dtype code space (0=BF16,1=F16,2=F32,3=U8/I8)
    /// -- collapses U8/I8 (both map to 3 in the C; byte size is 1 either
    /// way, so the collapse is lossless for this module's purposes).
    pub fn from_engine_code(code: i32) -> Option<DType> {
        match code {
            0 => Some(DType::BF16),
            1 => Some(DType::F16),
            2 => Some(DType::F32),
            3 => Some(DType::U8),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub begin: u64,
    pub end: u64,
}

impl TensorEntry {
    pub fn nbytes(&self) -> u64 {
        self.end - self.begin // caller-validated begin<=end before construction
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsHeader {
    pub header_len: u64,
    /// Bytes available after the header for tensor data (`blob_len - 8 -
    /// header_len`); every tensor's `end` must be `<=` this.
    pub data_len: u64,
    pub tensors: Vec<TensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetensorsError {
    TooShortForLengthPrefix { have: usize },
    HeaderLenExceedsCap { hlen: u64, cap: u64 },
    HeaderLenOverflow { hlen: u64 },
    TruncatedHeader { need: u64, have: u64 },
    HeaderNotUtf8,
    JsonSyntax(JsonError),
    HeaderNotObject,
    MissingField { tensor: String, field: &'static str },
    WrongFieldType { tensor: String, field: &'static str },
    UnsupportedDType { tensor: String, dtype: String },
    ShapeOverflow { tensor: String },
    BadOffsets { tensor: String, begin: u64, end: u64 },
    OffsetsOutOfBounds { tensor: String, begin: u64, end: u64, data_len: u64 },
    SizeMismatch { tensor: String, declared_bytes: u64, expected_bytes: u128 },
    DuplicateTensorName(String),
}

/// STRICT parse: every check described in this module's doc comment is
/// enforced; any violation is a distinct typed `Err`, never a panic.
pub fn parse_strict(blob: &[u8]) -> Result<SafetensorsHeader, SafetensorsError> {
    if blob.len() < 8 {
        return Err(SafetensorsError::TooShortForLengthPrefix { have: blob.len() });
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&blob[0..8]);
    let header_len = u64::from_le_bytes(len_bytes);

    if header_len > SANE_HEADER_LEN_CAP {
        return Err(SafetensorsError::HeaderLenExceedsCap {
            hlen: header_len,
            cap: SANE_HEADER_LEN_CAP,
        });
    }
    let need = 8u64
        .checked_add(header_len)
        .ok_or(SafetensorsError::HeaderLenOverflow { hlen: header_len })?;
    if (blob.len() as u64) < need {
        return Err(SafetensorsError::TruncatedHeader {
            need,
            have: blob.len() as u64,
        });
    }
    // Safe: header_len <= SANE_HEADER_LEN_CAP (<< usize::MAX on any platform
    // this crate runs on) and `need <= blob.len()` just verified above.
    let header_end = 8usize + header_len as usize;
    let header_bytes = &blob[8..header_end];
    let data_len = (blob.len() - header_end) as u64;

    let text = std::str::from_utf8(header_bytes).map_err(|_| SafetensorsError::HeaderNotUtf8)?;
    let root = json_mini::parse(text.as_bytes()).map_err(SafetensorsError::JsonSyntax)?;
    let obj = root.as_obj().ok_or(SafetensorsError::HeaderNotObject)?;

    let mut tensors = Vec::with_capacity(obj.len());
    let mut seen_names: Vec<&str> = Vec::with_capacity(obj.len());

    for (key, val) in obj {
        if key == "__metadata__" {
            continue; // matches c/st.h:131's `if (!strcmp(name,"__metadata__")) continue;`
        }
        if seen_names.contains(&key.as_str()) {
            return Err(SafetensorsError::DuplicateTensorName(key.clone()));
        }

        let dtype_val = val
            .obj_get("dtype")
            .ok_or_else(|| SafetensorsError::MissingField { tensor: key.clone(), field: "dtype" })?;
        let dtype_str = dtype_val
            .as_str()
            .ok_or_else(|| SafetensorsError::WrongFieldType { tensor: key.clone(), field: "dtype" })?;
        let dtype = DType::engine_supported(dtype_str).ok_or_else(|| SafetensorsError::UnsupportedDType {
            tensor: key.clone(),
            dtype: dtype_str.to_string(),
        })?;

        let shape_val = val
            .obj_get("shape")
            .ok_or_else(|| SafetensorsError::MissingField { tensor: key.clone(), field: "shape" })?;
        let shape_arr = shape_val
            .as_arr()
            .ok_or_else(|| SafetensorsError::WrongFieldType { tensor: key.clone(), field: "shape" })?;
        let mut shape = Vec::with_capacity(shape_arr.len());
        for el in shape_arr {
            let d = el
                .as_u64_exact()
                .ok_or_else(|| SafetensorsError::WrongFieldType { tensor: key.clone(), field: "shape" })?;
            shape.push(d);
        }
        let numel: u128 = shape.iter().try_fold(1u128, |acc, &d| acc.checked_mul(d as u128)).ok_or_else(
            || SafetensorsError::ShapeOverflow { tensor: key.clone() },
        )?;

        let offsets_val = val.obj_get("data_offsets").ok_or_else(|| SafetensorsError::MissingField {
            tensor: key.clone(),
            field: "data_offsets",
        })?;
        let offsets_arr = offsets_val.as_arr().ok_or_else(|| SafetensorsError::WrongFieldType {
            tensor: key.clone(),
            field: "data_offsets",
        })?;
        if offsets_arr.len() != 2 {
            return Err(SafetensorsError::WrongFieldType { tensor: key.clone(), field: "data_offsets" });
        }
        let begin = offsets_arr[0]
            .as_u64_exact()
            .ok_or_else(|| SafetensorsError::WrongFieldType { tensor: key.clone(), field: "data_offsets" })?;
        let end = offsets_arr[1]
            .as_u64_exact()
            .ok_or_else(|| SafetensorsError::WrongFieldType { tensor: key.clone(), field: "data_offsets" })?;

        if begin > end {
            return Err(SafetensorsError::BadOffsets { tensor: key.clone(), begin, end });
        }
        if end > data_len {
            return Err(SafetensorsError::OffsetsOutOfBounds { tensor: key.clone(), begin, end, data_len });
        }
        let declared_bytes = end - begin;
        let expected_bytes = numel
            .checked_mul(dtype.byte_size() as u128)
            .ok_or_else(|| SafetensorsError::ShapeOverflow { tensor: key.clone() })?;
        if (declared_bytes as u128) != expected_bytes {
            return Err(SafetensorsError::SizeMismatch {
                tensor: key.clone(),
                declared_bytes,
                expected_bytes,
            });
        }

        seen_names.push(key.as_str());
        tensors.push(TensorEntry { name: key.clone(), dtype, shape, begin, end });
    }

    Ok(SafetensorsHeader { header_len, data_len, tensors })
}

// ---------------------------------------------------------------- lenient --

/// One non-metadata tensor entry's fields as `c/st.h`'s `st_init` would
/// actually extract them: NO cross-validation between `dtype`, `shape`
/// (`numel`), and `data_offsets` (`nbytes`) -- exactly `st_tensor`'s fields
/// (`c/st.h:20-27`) before any consumer touches them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTensorFields {
    pub name: String,
    pub dtype_code: i32, // st_dtype_code()'s return value
    pub begin: i64,      // (int64_t)data_offsets[0] -- a C cast, may truncate/saturate an out-of-range double
    pub end: i64,
    pub nbytes: i64,     // b0 - a0, wrapping i64 sub, exactly c/st.h:140 -- may be NEGATIVE, never checked there
    pub numel: i64,      // wrapping i64 product of shape dims, exactly c/st.h:137 -- may overflow/wrap, never checked
}

impl RawTensorFields {
    /// Does this raw (unguarded) extraction violate an invariant the STRICT
    /// parser enforces? `Some(reason)` if so -- this is the actual
    /// differential check the fuzz harness runs per tensor, phrased without
    /// needing to re-invoke the whole strict parser.
    pub fn violates_strict_invariant(&self, data_len: u64) -> Option<&'static str> {
        if self.begin < 0 || self.end < 0 {
            return Some("negative offset (data_offsets entry cast from a negative/huge JSON number)");
        }
        if self.begin > self.end {
            return Some("begin > end (b0 < a0, c/st.h:140 never checks this)");
        }
        if (self.end as u64) > data_len {
            return Some("data_offsets end exceeds the tensor-data region (no bounds check in c/st.h)");
        }
        if self.numel < 0 {
            return Some("numel wrapped negative (int64_t overflow in c/st.h:137's shape product, never checked)");
        }
        let dtype = match DType::from_engine_code(self.dtype_code) {
            Some(d) => d,
            None => return Some("unrecognized dtype code (should be unreachable: st_dtype_code fails closed)"),
        };
        match (self.numel as i128).checked_mul(dtype.byte_size() as i128) {
            None => Some("numel*dtype_size overflow"),
            Some(expected) if expected != self.nbytes as i128 => {
                Some("nbytes != numel*dtype_size (c/st.h never cross-checks this -- the core finding)")
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorFieldsOutcome {
    /// `st_dtype_code()` (`c/st.h:49-56`) would `exit(1)` -- fail-closed,
    /// not itself a gap.
    CleanReject { name: String, reason: &'static str },
    /// `c/st.h` (or its own, unaudited-by-this-crate `json.h` dependency)
    /// has no guard covering this shape at all -- e.g. `data_offsets`/
    /// `shape` with the wrong arity (`c/st.h:134-137` indexes `kids[0]`/
    /// `kids[1]` with no length check). This crate does NOT claim to know
    /// the exact resulting C-side memory effect (that depends on
    /// `json.h`'s own internal representation, out of this exercise's read
    /// scope) -- only that no guard exists.
    UnguardedGap { name: String, reason: &'static str },
    Raw(RawTensorFields),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LenientHeaderOutcome {
    /// e.g. truncated length prefix / truncated header bytes -- `pread`
    /// short-reads, `st_init` calls `perror`+`exit(1)`. Fail-closed, not a
    /// gap.
    CleanReject(String),
    /// `c/st.h:123`: `char *hdr = malloc(hlen + 1);` has NO cap and the
    /// `malloc` return is never NULL-checked before the following `pread`
    /// -- modeled as its own bucket rather than actually attempting a
    /// multi-gigabyte allocation just to "prove" the crash.
    HeaderLenUnbounded { hlen: u64 },
    /// `data_len` is carried alongside the per-tensor outcomes so a caller
    /// can run `RawTensorFields::violates_strict_invariant(data_len)`
    /// without re-deriving it (it's a per-HEADER quantity, computed once).
    Parsed { data_len: u64, tensors: Vec<TensorFieldsOutcome> },
}

/// LENIENT, `c/st.h`-equivalent extraction: mirrors `st_init`'s actual
/// behavior (`c/st.h:119-155`) field for field, with none of the strict
/// parser's cross-checks.
pub fn parse_c_lenient(blob: &[u8]) -> LenientHeaderOutcome {
    if blob.len() < 8 {
        return LenientHeaderOutcome::CleanReject("pread hlen: short file (< 8 bytes), exit(1)".into());
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&blob[0..8]);
    let hlen = u64::from_le_bytes(len_bytes);

    // c/st.h has no cap at all; we refuse to actually allocate/scan
    // arbitrarily large claimed headers ourselves, matching the module doc
    // comment's rationale.
    const MODEL_LIMIT: u64 = 64 * 1024 * 1024; // generous vs. real GLM headers; see safetensors.rs module doc
    if hlen > MODEL_LIMIT {
        return LenientHeaderOutcome::HeaderLenUnbounded { hlen };
    }
    let need = match 8u64.checked_add(hlen) {
        Some(n) => n,
        None => return LenientHeaderOutcome::HeaderLenUnbounded { hlen },
    };
    if (blob.len() as u64) < need {
        return LenientHeaderOutcome::CleanReject("pread hdr: short read, exit(1)".into());
    }
    let header_end = 8usize + hlen as usize;
    let header_bytes = &blob[8..header_end];
    let data_len = (blob.len() - header_end) as u64;

    // c/st.h hands json_parse a NUL-terminated C string with no UTF-8
    // validation of its own; we still need decodable text to tokenize, so
    // this uses a lossy conversion (never fails) rather than treating
    // invalid UTF-8 as a hard reject the way the STRICT parser does --
    // deliberately more permissive here, matching a raw C byte scanner.
    let text = String::from_utf8_lossy(header_bytes);
    let root = match json_mini::parse(text.as_bytes()) {
        Ok(v) => v,
        Err(_) => {
            return LenientHeaderOutcome::CleanReject(
                "json_parse: malformed JSON (this crate does not model c/json.h's own unaudited parser \
                 internals byte for byte; treated as a reject, which is the charitable assumption)"
                    .into(),
            )
        }
    };
    let obj = match root.as_obj() {
        Some(o) => o,
        None => return LenientHeaderOutcome::CleanReject("top-level JSON value is not an object".into()),
    };

    let mut out = Vec::with_capacity(obj.len());
    for (key, val) in obj {
        if key == "__metadata__" {
            continue;
        }
        let name = key.clone();

        let dtype_str = match val.obj_get("dtype").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                out.push(TensorFieldsOutcome::UnguardedGap {
                    name,
                    reason: "dtype field missing/non-string (json_get/->str usage in c/st.h is unchecked here)",
                });
                continue;
            }
        };
        let dtype_code = match DType::engine_supported(dtype_str) {
            Some(d) => match d {
                DType::BF16 => 0,
                DType::F16 => 1,
                DType::F32 => 2,
                DType::U8 | DType::I8 => 3,
                _ => unreachable!("engine_supported only returns the 5 engine dtypes"),
            },
            None => {
                out.push(TensorFieldsOutcome::CleanReject {
                    name,
                    reason: "st_dtype_code(): unrecognized dtype string -> exit(1), fail-closed",
                });
                continue;
            }
        };

        let shape_arr = match val.obj_get("shape").and_then(|v| v.as_arr()) {
            Some(a) => a,
            None => {
                out.push(TensorFieldsOutcome::UnguardedGap {
                    name,
                    reason: "shape field missing/non-array",
                });
                continue;
            }
        };
        // c/st.h:137: `for (k<shp->len) numel *= (int64_t)shp->kids[k]->num;`
        // -- wrapping i64 multiply, no overflow check, no type check on
        // kids[k] (assumed numeric). We model the wraparound (see
        // RawTensorFields doc comment for the standard UB caveat) and bail
        // to UnguardedGap only if an entry isn't numeric at all (a case
        // c/st.h's json.h would need SOME representation for that this
        // crate does not claim to know).
        let mut numel: i64 = 1;
        let mut shape_ok = true;
        for el in shape_arr {
            match el.as_i64_c_cast() {
                Some(d) => numel = numel.wrapping_mul(d),
                None => {
                    shape_ok = false;
                    break;
                }
            }
        }
        if !shape_ok {
            out.push(TensorFieldsOutcome::UnguardedGap {
                name,
                reason: "a shape entry is not numeric",
            });
            continue;
        }

        let offsets_arr = match val.obj_get("data_offsets").and_then(|v| v.as_arr()) {
            Some(a) => a,
            None => {
                out.push(TensorFieldsOutcome::UnguardedGap {
                    name,
                    reason: "data_offsets field missing/non-array",
                });
                continue;
            }
        };
        // c/st.h:136 indexes kids[0]/kids[1] with NO length check at all.
        if offsets_arr.len() < 2 {
            out.push(TensorFieldsOutcome::UnguardedGap {
                name,
                reason: "data_offsets has fewer than 2 entries -- c/st.h:136 indexes kids[0]/kids[1] unconditionally",
            });
            continue;
        }
        let (Some(a0), Some(b0)) = (offsets_arr[0].as_i64_c_cast(), offsets_arr[1].as_i64_c_cast()) else {
            out.push(TensorFieldsOutcome::UnguardedGap {
                name,
                reason: "a data_offsets entry is not numeric",
            });
            continue;
        };
        let nbytes = b0.wrapping_sub(a0); // c/st.h:140, no b0>=a0 check

        out.push(TensorFieldsOutcome::Raw(RawTensorFields {
            name,
            dtype_code,
            begin: a0,
            end: b0,
            nbytes,
            numel,
        }));
    }
    LenientHeaderOutcome::Parsed { data_len, tensors: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_valid_blob() -> Vec<u8> {
        // One F32 tensor, shape [2,2] (4 elems -> 16 bytes), plus a
        // __metadata__ entry that should be skipped entirely.
        let json = br#"{"__metadata__":{"format":"test"},"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut blob = Vec::new();
        blob.extend_from_slice(&(json.len() as u64).to_le_bytes());
        blob.extend_from_slice(json);
        blob.extend(std::iter::repeat(0xABu8).take(16)); // tensor payload
        blob
    }

    #[test]
    fn strict_accepts_well_formed_blob() {
        let blob = build_valid_blob();
        let hdr = parse_strict(&blob).expect("should parse");
        assert_eq!(hdr.tensors.len(), 1);
        assert_eq!(hdr.tensors[0].name, "w");
        assert_eq!(hdr.tensors[0].dtype, DType::F32);
        assert_eq!(hdr.tensors[0].nbytes(), 16);
    }

    #[test]
    fn strict_rejects_size_mismatch() {
        let json = br#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,15]}}"#; // 15, not 16
        let mut blob = Vec::new();
        blob.extend_from_slice(&(json.len() as u64).to_le_bytes());
        blob.extend_from_slice(json);
        blob.extend(std::iter::repeat(0u8).take(15));
        let err = parse_strict(&blob).unwrap_err();
        assert!(matches!(err, SafetensorsError::SizeMismatch { .. }));
    }

    #[test]
    fn strict_rejects_begin_after_end() {
        let json = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[10,0]}}"#;
        let mut blob = Vec::new();
        blob.extend_from_slice(&(json.len() as u64).to_le_bytes());
        blob.extend_from_slice(json);
        let err = parse_strict(&blob).unwrap_err();
        assert!(matches!(err, SafetensorsError::BadOffsets { .. }));
    }

    #[test]
    fn strict_rejects_offsets_out_of_bounds() {
        let json = br#"{"w":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}}"#;
        let mut blob = Vec::new();
        blob.extend_from_slice(&(json.len() as u64).to_le_bytes());
        blob.extend_from_slice(json);
        // only supply 4 trailing bytes, nowhere near 400
        blob.extend(std::iter::repeat(0u8).take(4));
        let err = parse_strict(&blob).unwrap_err();
        assert!(matches!(err, SafetensorsError::OffsetsOutOfBounds { .. }));
    }

    #[test]
    fn strict_rejects_unrecognized_dtype() {
        let json = br#"{"w":{"dtype":"FP8_E4M3","shape":[1],"data_offsets":[0,1]}}"#;
        let mut blob = Vec::new();
        blob.extend_from_slice(&(json.len() as u64).to_le_bytes());
        blob.extend_from_slice(json);
        blob.push(0);
        let err = parse_strict(&blob).unwrap_err();
        assert!(matches!(err, SafetensorsError::UnsupportedDType { .. }));
    }

    #[test]
    fn lenient_silently_accepts_the_same_size_mismatch_strict_rejects() {
        let json = br#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,15]}}"#; // same as above
        let mut blob = Vec::new();
        blob.extend_from_slice(&(json.len() as u64).to_le_bytes());
        blob.extend_from_slice(json);
        blob.extend(std::iter::repeat(0u8).take(15));

        assert!(matches!(parse_strict(&blob), Err(SafetensorsError::SizeMismatch { .. })));

        match parse_c_lenient(&blob) {
            LenientHeaderOutcome::Parsed { data_len, tensors } => {
                assert_eq!(tensors.len(), 1);
                match &tensors[0] {
                    TensorFieldsOutcome::Raw(raw) => {
                        assert_eq!(raw.violates_strict_invariant(data_len), Some(
                            "nbytes != numel*dtype_size (c/st.h never cross-checks this -- the core finding)"
                        ));
                    }
                    other => panic!("expected Raw, got {other:?}"),
                }
            }
            other => panic!("expected Parsed, got {other:?}"),
        }
    }

    #[test]
    fn strict_never_panics_on_truncated_input() {
        for n in 0..12 {
            let blob = vec![0xFFu8; n];
            let _ = parse_strict(&blob); // must not panic
        }
    }

    #[test]
    fn lenient_never_panics_on_truncated_input() {
        for n in 0..12 {
            let blob = vec![0xFFu8; n];
            let _ = parse_c_lenient(&blob); // must not panic
        }
    }

    #[test]
    fn header_len_unbounded_bucket_triggers_on_huge_claim() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&u64::MAX.to_le_bytes());
        match parse_c_lenient(&blob) {
            LenientHeaderOutcome::HeaderLenUnbounded { hlen } => assert_eq!(hlen, u64::MAX),
            other => panic!("expected HeaderLenUnbounded, got {other:?}"),
        }
        assert!(matches!(
            parse_strict(&blob),
            Err(SafetensorsError::HeaderLenExceedsCap { .. })
        ));
    }
}
