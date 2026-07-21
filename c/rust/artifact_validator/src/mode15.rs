//! Strict parser+validator for the Mode-1.5 ("MH01") compressed tensor blob
//! container, ported field-for-field from `c/mode15_reader.c`'s
//! `m15_open_structural()` (that file's own doc comment calls out the
//! validation ORDER as intentionally mirroring `mode15_container.py`'s
//! `parse_tensor_blob()`; this port preserves the same order for the same
//! reason -- so behavior on a given corrupt blob is predictable relative to
//! both reference implementations, not just internally self-consistent).
//!
//! IMPORTANT CONTEXT this crate's report leans on: as of this reading,
//! `c/glm.c` has ZERO references to `mode15`/`m15_`/`crc`/`checksum` --
//! `mode15_reader.h`'s own file header says as much ("Read compressed
//! container header, glm.c, inert"). The hardened, CRC-checked reader this
//! module mirrors is real, tested, and unused in production today; the
//! actual hot path is `c/st.h` + `c/glm.c`'s int8/int4/int2 ternary
//! (`quant_format.rs`/`safetensors.rs` in this crate), which has NO
//! checksum at all. This module exists to (a) prove the STRICT validator
//! itself is panic-free on adversarial MH01 input, and (b) quantify, as a
//! positive control, how much a checksum (which this format has and
//! safetensors does not) actually buys you against same-size payload
//! corruption that no structural check alone can catch -- see
//! `tests/fuzz_mutation.rs` and `src/bin/fuzz_report.rs`.
//!
//! ON-DISK LAYOUT (`c/tools/mode15_container.py`'s module docstring,
//! `c/mode15_reader.h:52-86`):
//!   offset  0: 4B   magic "MH01"
//!   offset  4: u32  O                 rows
//!   offset  8: u32  I                 symbols/row
//!   offset 12: u32  rows_per_block
//!   offset 16: u32  n_blocks          ceil(O/rows_per_block), 0 if O==0
//!   offset 20: u32  tensor_crc32      CRC32 over offset 24..end
//!   offset 24: 8B   huff_lengths      16 x 4-bit lengths, 0=absent
//!   offset 32: (O+1)*4B row_offsets   u32 LE cumulative payload offsets
//!   offset 32+(O+1)*4: n_blocks*4B block_crc32   u32 LE, per block
//!   offset ...: payload               row_offsets[O] bytes

use crate::crc32::Crc32State;

pub const TENSOR_HEADER_LEN: usize = 24;
pub const LENGTHS_BYTES: usize = 8;
pub const NSYM: usize = 16;

/// Caller-supplied trust bounds, mirroring `M15Caps` (`c/mode15_reader.h:273-278`).
/// 0 in a field = uncapped for that field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mode15Caps {
    pub max_o: u32,
    pub max_i: u32,
    pub max_n_blocks: u32,
    pub max_payload_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode15Error {
    TruncatedHeader,
    BadMagic,
    VersionMismatch,
    OMismatch { got: u32, expected: u32 },
    IMismatch { got: u32, expected: u32 },
    CapExceeded(&'static str),
    BadRowsPerBlock,
    BlockCountMismatch { got: u32, expected: u32 },
    TruncatedLengths,
    BadLengthTable,
    TruncatedRowOffsets,
    RowOffsetsInvalid,
    TruncatedBlockCrc,
    LengthMismatch { got: u64, expected: u64 },
    TensorCrcMismatch { got: u32, want: u32 },
    BlockCrcMismatch { block: u32, got: u32, want: u32 },
    BlockIndexOutOfRange { block: u32, n_blocks: u32 },
    /// Defense-in-depth-only: every arithmetic/indexing step above this is
    /// meant to be proven in-bounds before use; this variant exists so a
    /// mistake in that reasoning surfaces as a typed `Err` instead of a
    /// panic. Should be unreachable in practice.
    InternalBoundsCheckFailed(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mode15Header {
    pub o: u32,
    pub i: u32,
    pub rows_per_block: u32,
    pub n_blocks: u32,
    pub tensor_crc32: u32,
    pub lengths: [u8; NSYM],
    pub maxlen: u8,
    pub n_present: u8,
    pub row_offsets: Vec<u32>, // O+1 entries, owned (mirrors M15Reader's copy-at-open lifetime contract)
    pub block_crc32: Vec<u32>, // n_blocks entries, owned
    /// Byte range within the ORIGINAL blob this header was parsed from --
    /// this crate has no separate lifetime contract to enforce (no FFI,
    /// nothing borrows across a function-call boundary the way `M15Reader`
    /// does), so we simply re-slice `blob` on demand via `payload()`.
    pub payload_range: (usize, usize),
}

impl Mode15Header {
    pub fn payload<'a>(&self, blob: &'a [u8]) -> Option<&'a [u8]> {
        blob.get(self.payload_range.0..self.payload_range.1)
    }
    pub fn payload_len(&self) -> u32 {
        (self.payload_range.1 - self.payload_range.0) as u32
    }
}

fn u32le(blob: &[u8], idx: usize) -> Option<u32> {
    let end = idx.checked_add(4)?;
    let s = blob.get(idx..end)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// Port of `m15_lengths_valid()` (`c/mode15_reader.c:97-117`) -- see that
/// function's doc comment / `c/mode15_reader.h`'s "LENGTH TABLE VALIDITY"
/// section for the full rationale (n_present==0/1/>=2 are three genuinely
/// different valid shapes, not a simplification).
fn lengths_valid(lengths: &[u8; NSYM]) -> Option<(u8, u8)> {
    let mut n_present: u32 = 0;
    let mut maxlen: u32 = 0;
    let mut sole_len: i32 = -1;
    let mut kraft_sum: u32 = 0;
    for &raw in lengths.iter() {
        let l = raw as u32;
        if l == 0 {
            continue;
        }
        if l > 15 {
            return None;
        }
        n_present += 1;
        if l > maxlen {
            maxlen = l;
        }
        if n_present == 1 {
            sole_len = l as i32;
        }
        kraft_sum = kraft_sum.wrapping_add(1u32 << (15 - l));
    }
    if n_present == 0 {
        return Some((1, 0));
    }
    if n_present == 1 {
        if sole_len != 1 {
            return None;
        }
        return Some((1, 1));
    }
    if kraft_sum != (1u32 << 15) {
        return None;
    }
    Some((maxlen as u8, n_present as u8))
}

/// STRICT structural open: magic/version, length-table validity, row_offsets
/// monotonicity, every truncation/size/cap check -- port of
/// `m15_open_structural()` (`c/mode15_reader.c:145-266`). Does NOT verify
/// any CRC32 (that is `verify_tensor_crc`/`verify_block_crc` below,
/// deliberately separate -- see this module's doc comment and
/// `c/mode15_reader.h`'s VERIFICATION MODEL section for why).
pub fn open_structural(
    blob: &[u8],
    expect_o: Option<u32>,
    expect_i: Option<u32>,
    caps: Option<&Mode15Caps>,
) -> Result<Mode15Header, Mode15Error> {
    if blob.len() < TENSOR_HEADER_LEN {
        return Err(Mode15Error::TruncatedHeader);
    }
    if blob[0] != b'M' || blob[1] != b'H' {
        return Err(Mode15Error::BadMagic);
    }
    if blob[2] != b'0' || blob[3] != b'1' {
        return Err(Mode15Error::VersionMismatch);
    }

    let o = u32le(blob, 4).ok_or(Mode15Error::TruncatedHeader)?;
    let i = u32le(blob, 8).ok_or(Mode15Error::TruncatedHeader)?;
    let rows_per_block = u32le(blob, 12).ok_or(Mode15Error::TruncatedHeader)?;
    let n_blocks = u32le(blob, 16).ok_or(Mode15Error::TruncatedHeader)?;
    let tensor_crc32 = u32le(blob, 20).ok_or(Mode15Error::TruncatedHeader)?;

    if let Some(e) = expect_o {
        if o != e {
            return Err(Mode15Error::OMismatch { got: o, expected: e });
        }
    }
    if let Some(e) = expect_i {
        if i != e {
            return Err(Mode15Error::IMismatch { got: i, expected: e });
        }
    }
    if let Some(c) = caps {
        if c.max_o != 0 && o > c.max_o {
            return Err(Mode15Error::CapExceeded("max_o"));
        }
        if c.max_i != 0 && i > c.max_i {
            return Err(Mode15Error::CapExceeded("max_i"));
        }
        if c.max_n_blocks != 0 && n_blocks > c.max_n_blocks {
            return Err(Mode15Error::CapExceeded("max_n_blocks"));
        }
    }

    let expect_n_blocks: u32 = if o == 0 {
        0
    } else {
        if rows_per_block == 0 {
            return Err(Mode15Error::BadRowsPerBlock);
        }
        let o64 = o as u64;
        let rpb64 = rows_per_block as u64;
        o64.div_ceil(rpb64) as u32 // always fits: result <= o <= u32::MAX
    };
    if n_blocks != expect_n_blocks {
        return Err(Mode15Error::BlockCountMismatch { got: n_blocks, expected: expect_n_blocks });
    }

    let mut off: u64 = TENSOR_HEADER_LEN as u64;
    if (blob.len() as u64) < off + LENGTHS_BYTES as u64 {
        return Err(Mode15Error::TruncatedLengths);
    }
    let mut lengths = [0u8; NSYM];
    for k in 0..LENGTHS_BYTES {
        let idx = usize::try_from(off).map_err(|_| Mode15Error::InternalBoundsCheckFailed("off->usize"))? + k;
        let b = *blob.get(idx).ok_or(Mode15Error::TruncatedLengths)?;
        lengths[2 * k] = b & 0xF;
        lengths[2 * k + 1] = b >> 4;
    }
    off += LENGTHS_BYTES as u64;

    let (maxlen, n_present) = lengths_valid(&lengths).ok_or(Mode15Error::BadLengthTable)?;
    if n_present == 0 && o != 0 && i != 0 {
        return Err(Mode15Error::BadLengthTable);
    }

    let ro_bytes = (o as u64 + 1) * 4; // o: u32, so o+1 <= 2^32, *4 <= 2^34 -- comfortably fits u64
    if (blob.len() as u64) < off + ro_bytes {
        return Err(Mode15Error::TruncatedRowOffsets);
    }
    let ro_start = usize::try_from(off).map_err(|_| Mode15Error::InternalBoundsCheckFailed("ro_start"))?;
    off += ro_bytes;

    let mut row_offsets: Vec<u32> = Vec::with_capacity(o as usize + 1);
    {
        let mut prev: u32 = 0;
        let mut bad = false;
        for r in 0..=o {
            let idx = ro_start
                .checked_add((r as usize).checked_mul(4).ok_or(Mode15Error::InternalBoundsCheckFailed("ro idx mul"))?)
                .ok_or(Mode15Error::InternalBoundsCheckFailed("ro idx add"))?;
            let cur = u32le(blob, idx).ok_or(Mode15Error::TruncatedRowOffsets)?;
            if r > 0 && cur < prev {
                bad = true;
            }
            row_offsets.push(cur);
            prev = cur;
        }
        if row_offsets.first() != Some(&0) {
            bad = true;
        }
        if bad {
            return Err(Mode15Error::RowOffsetsInvalid);
        }
    }

    let payload_len: u32 = *row_offsets.get(o as usize).ok_or(Mode15Error::InternalBoundsCheckFailed("payload_len"))?;
    if let Some(c) = caps {
        if c.max_payload_len != 0 && (payload_len as u64) > c.max_payload_len {
            return Err(Mode15Error::CapExceeded("max_payload_len"));
        }
    }

    let bc_bytes = (n_blocks as u64) * 4; // n_blocks: u32, *4 fits u64 comfortably
    if (blob.len() as u64) < off + bc_bytes {
        return Err(Mode15Error::TruncatedBlockCrc);
    }
    let bc_start = usize::try_from(off).map_err(|_| Mode15Error::InternalBoundsCheckFailed("bc_start"))?;
    off += bc_bytes;

    let expected_total = off + payload_len as u64;
    if blob.len() as u64 != expected_total {
        return Err(Mode15Error::LengthMismatch { got: blob.len() as u64, expected: expected_total });
    }
    let payload_start = usize::try_from(off).map_err(|_| Mode15Error::InternalBoundsCheckFailed("payload_start"))?;
    let payload_end = payload_start
        .checked_add(payload_len as usize)
        .ok_or(Mode15Error::InternalBoundsCheckFailed("payload_end"))?;
    if payload_end > blob.len() {
        return Err(Mode15Error::InternalBoundsCheckFailed("payload_end > blob.len()"));
    }

    let mut block_crc32: Vec<u32> = Vec::with_capacity(n_blocks as usize);
    for b in 0..n_blocks {
        let idx = bc_start
            .checked_add((b as usize).checked_mul(4).ok_or(Mode15Error::InternalBoundsCheckFailed("bc idx mul"))?)
            .ok_or(Mode15Error::InternalBoundsCheckFailed("bc idx add"))?;
        block_crc32.push(u32le(blob, idx).ok_or(Mode15Error::TruncatedBlockCrc)?);
    }

    Ok(Mode15Header {
        o,
        i,
        rows_per_block,
        n_blocks,
        tensor_crc32,
        lengths,
        maxlen,
        n_present,
        row_offsets,
        block_crc32,
        payload_range: (payload_start, payload_end),
    })
}

/// Whole-tensor CRC32, port of `m15_verify_tensor_once()`
/// (`c/mode15_reader.c:305-338`): reconstructs the exact original index
/// bytes (lengths + row_offsets + block_crc32, re-serialized from the
/// OWNED, already-validated fields -- not from any surviving pointer into
/// the original blob) then hashes that plus the payload.
pub fn verify_tensor_crc(hdr: &Mode15Header, blob: &[u8]) -> Result<(), Mode15Error> {
    let payload = hdr.payload(blob).ok_or(Mode15Error::InternalBoundsCheckFailed("payload range"))?;

    let mut idx_bytes: Vec<u8> = Vec::with_capacity(LENGTHS_BYTES + hdr.row_offsets.len() * 4 + hdr.block_crc32.len() * 4);
    for k in 0..LENGTHS_BYTES {
        let lo = hdr.lengths[2 * k] & 0xF;
        let hi = hdr.lengths[2 * k + 1] & 0xF;
        idx_bytes.push(lo | (hi << 4));
    }
    for &v in &hdr.row_offsets {
        idx_bytes.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &hdr.block_crc32 {
        idx_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let mut st = Crc32State::new();
    st.update(&idx_bytes);
    st.update(payload);
    let got = st.finish();
    if got != hdr.tensor_crc32 {
        return Err(Mode15Error::TensorCrcMismatch { got, want: hdr.tensor_crc32 });
    }
    Ok(())
}

/// Single block's own CRC32, port of `m15_verify_block_once()`
/// (`c/mode15_reader.c:340-361`) minus the per-block verified-cache (this
/// crate has no long-lived reader object to cache state in; every call
/// re-hashes, which is fine at fuzz-campaign scale).
pub fn verify_block_crc(hdr: &Mode15Header, blob: &[u8], block_idx: u32) -> Result<(), Mode15Error> {
    if block_idx >= hdr.n_blocks {
        return Err(Mode15Error::BlockIndexOutOfRange { block: block_idx, n_blocks: hdr.n_blocks });
    }
    let payload = hdr.payload(blob).ok_or(Mode15Error::InternalBoundsCheckFailed("payload range"))?;

    let a = (block_idx as u64) * (hdr.rows_per_block as u64);
    let b = ((block_idx as u64) + 1) * (hdr.rows_per_block as u64);
    let b = b.min(hdr.o as u64);
    let r0 = usize::try_from(a).map_err(|_| Mode15Error::InternalBoundsCheckFailed("r0"))?;
    let r1 = usize::try_from(b).map_err(|_| Mode15Error::InternalBoundsCheckFailed("r1"))?;

    let s0 = *hdr.row_offsets.get(r0).ok_or(Mode15Error::InternalBoundsCheckFailed("row_offsets[r0]"))?;
    let s1 = *hdr.row_offsets.get(r1).ok_or(Mode15Error::InternalBoundsCheckFailed("row_offsets[r1]"))?;
    if s0 > s1 {
        return Err(Mode15Error::InternalBoundsCheckFailed("s0 > s1"));
    }
    let span = payload
        .get(s0 as usize..s1 as usize)
        .ok_or(Mode15Error::InternalBoundsCheckFailed("payload span"))?;

    let got = crate::crc32::crc32(span);
    let want = *hdr.block_crc32.get(block_idx as usize).ok_or(Mode15Error::InternalBoundsCheckFailed("block_crc32[idx]"))?;
    if got != want {
        return Err(Mode15Error::BlockCrcMismatch { block: block_idx, got, want });
    }
    Ok(())
}

/// Builds a small, STRUCTURALLY VALID MH01 blob (bogus but internally
/// consistent Huffman-length table: a single present symbol, which is a
/// legal degenerate codebook per `lengths_valid`'s n_present==1 rule) --
/// used as fuzz-campaign seed material. This does not need to be a REAL
/// Huffman-encoded payload (row content is opaque bytes to this validator;
/// see `c/mode15_reader.h`'s own SCOPE note: "does NOT decode Huffman
/// bitstreams").
pub fn build_valid_fixture(o: u32, i: u32, rows_per_block: u32, payload_seed: u8) -> Vec<u8> {
    let n_blocks = if o == 0 { 0 } else { o.div_ceil(rows_per_block) };
    // Degenerate 1-symbol codebook: symbol 0 present with length 1, matching
    // `lengths_valid`'s explicit single-leaf exception.
    let mut lengths = [0u8; NSYM];
    lengths[0] = 1;
    let lengths_bytes = {
        let mut lb = [0u8; LENGTHS_BYTES];
        for k in 0..LENGTHS_BYTES {
            lb[k] = (lengths[2 * k] & 0xF) | ((lengths[2 * k + 1] & 0xF) << 4);
        }
        lb
    };

    // One byte of payload per row (arbitrary but deterministic content --
    // this validator never interprets payload bytes, only their length and
    // checksum).
    let row_offsets: Vec<u32> = (0..=o).collect();
    let payload: Vec<u8> = (0..o).map(|r| payload_seed.wrapping_add(r as u8)).collect();

    let mut block_crc32 = Vec::with_capacity(n_blocks as usize);
    for b in 0..n_blocks {
        let r0 = (b * rows_per_block) as usize;
        let r1 = (((b + 1) * rows_per_block).min(o)) as usize;
        block_crc32.push(crate::crc32::crc32(&payload[r0..r1]));
    }

    let mut idx_bytes = Vec::new();
    idx_bytes.extend_from_slice(&lengths_bytes);
    for &v in &row_offsets {
        idx_bytes.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &block_crc32 {
        idx_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut st = Crc32State::new();
    st.update(&idx_bytes);
    st.update(&payload);
    let tensor_crc32 = st.finish();

    let mut blob = Vec::new();
    blob.extend_from_slice(b"MH01");
    blob.extend_from_slice(&o.to_le_bytes());
    blob.extend_from_slice(&i.to_le_bytes());
    blob.extend_from_slice(&rows_per_block.to_le_bytes());
    blob.extend_from_slice(&n_blocks.to_le_bytes());
    blob.extend_from_slice(&tensor_crc32.to_le_bytes());
    blob.extend_from_slice(&idx_bytes);
    blob.extend_from_slice(&payload);
    blob
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_fixture_opens_and_verifies() {
        let blob = build_valid_fixture(10, 4, 4, 7);
        let hdr = open_structural(&blob, None, None, None).expect("structural open");
        verify_tensor_crc(&hdr, &blob).expect("tensor crc");
        for b in 0..hdr.n_blocks {
            verify_block_crc(&hdr, &blob, b).expect("block crc");
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut blob = build_valid_fixture(4, 2, 2, 1);
        blob[0] = b'X';
        assert_eq!(open_structural(&blob, None, None, None), Err(Mode15Error::BadMagic));
    }

    #[test]
    fn rejects_version_mismatch() {
        let mut blob = build_valid_fixture(4, 2, 2, 1);
        blob[3] = b'2';
        assert_eq!(open_structural(&blob, None, None, None), Err(Mode15Error::VersionMismatch));
    }

    #[test]
    fn rejects_truncated_header() {
        let blob = vec![0u8; 10];
        assert_eq!(open_structural(&blob, None, None, None), Err(Mode15Error::TruncatedHeader));
    }

    #[test]
    fn structural_open_ignores_payload_corruption_but_crc_catches_it() {
        let mut blob = build_valid_fixture(16, 4, 4, 3);
        let hdr_before = open_structural(&blob, None, None, None).expect("open before corruption");
        verify_tensor_crc(&hdr_before, &blob).expect("crc ok before corruption");

        // Flip one payload byte WITHOUT changing any length/offset field --
        // total blob size is unchanged, so this is exactly the "same-size
        // corruption" class this exercise cares about.
        let (ps, pe) = hdr_before.payload_range;
        assert!(pe > ps, "fixture must have nonempty payload for this test");
        blob[ps] ^= 0xFF;

        // Structural validation alone (no format has a choice here --
        // MH01's OWN structural fields are untouched) still succeeds:
        let hdr_after = open_structural(&blob, None, None, None).expect("structural fields untouched");
        assert_eq!(hdr_after, hdr_before);
        // But CRC verification -- the layer safetensors doesn't have at all -- catches it:
        assert!(matches!(verify_tensor_crc(&hdr_after, &blob), Err(Mode15Error::TensorCrcMismatch { .. })));
    }

    #[test]
    fn zero_row_tensor_is_valid_with_empty_length_table() {
        let blob = build_valid_fixture(0, 0, 4, 0);
        let hdr = open_structural(&blob, None, None, None).expect("O=0 is a legal degenerate tensor");
        assert_eq!(hdr.n_blocks, 0);
        assert_eq!(hdr.payload_len(), 0);
    }

    #[test]
    fn caps_reject_oversized_declared_o() {
        let blob = build_valid_fixture(100, 4, 4, 0);
        let caps = Mode15Caps { max_o: 10, ..Default::default() };
        assert_eq!(
            open_structural(&blob, None, None, Some(&caps)),
            Err(Mode15Error::CapExceeded("max_o"))
        );
    }

    #[test]
    fn never_panics_across_all_truncation_lengths() {
        let full = build_valid_fixture(20, 6, 4, 5);
        for n in 0..=full.len() {
            let _ = open_structural(&full[..n], None, None, None); // must not panic
        }
    }
}
