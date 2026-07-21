"""mode15_container.py -- shared format + codec for the mode-1.5 OFFLINE
ENTROPY-ENCODED container (registration: c/bench-m5max/mode15-lossless-
the pipeline registration; binding: bit-exact lossless, canonical Huffman
per the codec race, kill-shots both PASSED -> build authorized).

This module is the Python-side implementation of the EXACT bitstream
format c/codec_row.h / c/codec_row_huff.h define, vectorized with numpy so
it is fast enough to encode/decode real per-expert tensors (O=2048/6144
rows, I=2048/6144 symbols/row) instead of a per-symbol Python loop, which
would be many orders of magnitude too slow for a ~19,200-expert container.

FORMAT-COMPATIBILITY CONTRACT (read this before touching the bit-level
code below): codec_row_huff.h ships only a per-symbol LENGTH table on disk
(8 bytes, 16 x 4-bit lengths -- "HUFF_TABLE_BYTES" in
tools/measure_expert_entropy.py's own convention) and has EVERY reader
(this module, a future C loader, the Metal GPU decoder) reconstruct the
canonical codewords from those lengths independently via a FIXED,
deterministic algorithm. That means:
  - the LENGTH-CHOICE algorithm (which lengths a given histogram produces)
    does NOT need to bit-for-bit match codec_row_huff.h's huff_build_
    lengths() -- ANY optimal (or even suboptimal-but-valid, Kraft-
    satisfying) length assignment decodes correctly under the canonical
    reconstruction, so this module reuses the repo's own existing
    heapq-based tree-depth computation (measure_expert_entropy.py's
    huffman_code_lengths -- already written, tested, and used by the
    entropy census) rather than reimplementing codec_row_huff.h's
    O(n^2) repeated-min-merge from scratch;
  - the CANONICAL CODE ASSIGNMENT (huff_canonical_codes below) and the
    WIDE-LUT FILL (build_lut below) MUST match codec_row_huff.h's
    huff_canonical_codes()/huff_build() BIT FOR BIT -- these are ported
    line-for-line from the C (same stable sort key, same bit-reversal,
    same LUT-fill loop), because this is the part that actually determines
    which bits this encoder writes. A future C/Metal reader that rebuilds
    its own LUT from this module's stored 8-byte length table via the
    SAME canonical-code algorithm will therefore reconstruct byte-
    identical codewords to what this encoder used, without this container
    ever having to ship a full LUT/code table on disk.
  - bit order: codewords are bit-reversed once at build time and packed
    LSB-first per byte (codec_row.h's CodecBitW/CodecBitR convention) --
    ported below in the vectorized encode_tensor()/decode_tensor().

See gate_m15_g1.py and tests/test_mode15_container.py for the
bit-exactness verification (including an independent C cross-check
against the ACTUAL codec_row_huff.h functions, not just this module's own
self-consistency).

ON-DISK TENSOR BLOB LAYOUT (one blob replaces one EXPERT tensor's raw
bytes in the safetensors shard; dtype U8, shape=[len(blob)]; everything
else -- non-expert tensors, and every tensor's own .qs per-row F32 scales
-- is copied verbatim, byte-identical, per codec_row.h's own comment that
scales "stay raw"):

    offset  0: 4s   magic            b"MH01" (also the per-tensor format
                                      version tag: bump on any layout change)
    offset  4: u32  O                rows
    offset  8: u32  I                symbols/row
    offset 12: u32  rows_per_block   checksum block granularity used
    offset 16: u32  n_blocks         ceil(O / rows_per_block)
    offset 20: u32  tensor_crc32     zlib.crc32 over everything from
                                     offset 24 to the end of payload
                                     (lengths + row_offsets + block CRCs +
                                     payload) -- whole-tensor fail-closed
                                     gate, catches corruption ANYWHERE in
                                     the structural metadata, not just the
                                     payload.
    offset 24: 8B   huff_lengths     16 x 4-bit canonical code lengths,
                                     packed 2/byte (low nibble=even symbol,
                                     matching the codec's own nibble
                                     convention), 0 = symbol absent.
    offset 32: (O+1)*4 B  row_offsets   u32 LE cumulative payload-byte
                                     offsets, row_offsets[0]==0,
                                     row_offsets[O] == total payload bytes.
    offset 32+(O+1)*4: n_blocks*4 B  block_crc32   u32 LE zlib.crc32 of
                                     block b's OWN payload byte range
                                     (rows [b*rows_per_block,
                                     min(O,(b+1)*rows_per_block)) ) --
                                     lets a reader localize/verify a
                                     SPECIFIC block (e.g. the exact GPU
                                     threadgroup batch, rows_per_block
                                     defaults to
                                     ILI_ROWS_PER_TG=256 from
                                     tests/test_metal_row_decode.mm)
                                     without hashing the whole tensor.
    offset ...: payload              row_offsets[O] bytes, row r's own
                                     bitstream at
                                     payload[row_offsets[r]:row_offsets[r+1]]

CONTAINER-LEVEL MANIFEST (mode15-container-manifest.json, one per outdir):
schema/format version, source model dir + config, rows_per_block, the
per-shard PROVENANCE + RESUMABILITY LEDGER (source shard sha256, tool git
commit + dirty flag, per-shard tensor/byte counts, timestamps) -- see
encode_mode15_container.py. This satisfies the registration's hardening
amendment ("block checksums, container versioning, corruption/truncation
recovery") as three DISTINCT things: block_crc32/tensor_crc32 (corruption
detection), format/schema_version fields at both the container-manifest
and per-tensor-magic level (versioning), and the shard ledger's
source_sha256 + tool_commit (provenance/audit trail) -- not one field
doing triple duty.
"""

from __future__ import annotations

import os
import struct
import sys
import zlib

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
import quant_container as qc  # noqa: E402

# Reuse the repo's own optimal-Huffman-length computation (heapq-based tree
# depths; already written + tested, used by the entropy census). See the
# module docstring above for why matching ITS internal tie-break is not
# required for format compatibility -- only huff_canonical_codes/build_lut
# below (ported from codec_row_huff.h) need to match bit-for-bit.
_mee = qc._load_module(
    "measure_expert_entropy", os.path.join(TOOLS_DIR, "measure_expert_entropy.py")
)
huffman_code_lengths = _mee.huffman_code_lengths

NSYM = 16                       # int4 alphabet size, codec_row.h's CODEC_NSYM
HUFF_MAXLEN = 15                # n-1 for a 16-leaf tree, codec_row_huff.h's HUFF_MAXLEN
MAGIC = b"MH01"
TENSOR_HEADER_LEN = 24          # magic..tensor_crc32
LENGTHS_BYTES = 8               # 16 x 4-bit lengths, HUFF_TABLE_BYTES convention
_HEADER_STRUCT = struct.Struct("<4sIIIII")   # magic,O,I,rows_per_block,n_blocks,tensor_crc32
assert _HEADER_STRUCT.size == TENSOR_HEADER_LEN

CONTAINER_FORMAT = "mode15-huffman-row-v1"
CONTAINER_SCHEMA_VERSION = 1
DEFAULT_ROWS_PER_BLOCK = 256    # ties the checksum/fault-isolation unit to
                                 # tests/test_metal_row_decode.mm's
                                 # ILI_ROWS_PER_TG (one GPU threadgroup's
                                 # row batch) -- a physically meaningful
                                 # granularity in the actual decode kernel,
                                 # not an arbitrary number.


class Mode15FormatError(ValueError):
    """Structural parse failure: bad magic, truncated blob, or a length/
    row-offsets/block-count mismatch. Always fail-closed -- never return
    partially-decoded data."""


class Mode15ChecksumError(ValueError):
    """Structurally well-formed blob, but a CRC32 (whole-tensor or a
    specific block) does not match -- corruption, not truncation."""


# ------------------------------------------------------- nibble I/O ----
def unpack_nibbles(raw: np.ndarray, O: int, I: int) -> np.ndarray:
    """Packed int4 bytes (1-D uint8, O*ceil(I/2) bytes) -> (O,I) uint8
    nibble symbols. Low nibble=even column, high nibble=odd column --
    matches glm.c pack_int4 / codec_row.h's codec_unpack_nibbles /
    quant_container.dequant's own bit-manipulation exactly."""
    row_bytes = (I + 1) // 2
    b = raw.reshape(O, row_bytes)
    out = np.empty((O, row_bytes * 2), dtype=np.uint8)
    out[:, 0::2] = b & 0xF
    out[:, 1::2] = b >> 4
    return np.ascontiguousarray(out[:, :I])


def pack_nibbles(sym: np.ndarray, I: int) -> np.ndarray:
    """(O,I) uint8 nibble symbols -> packed int4 bytes (O, ceil(I/2)),
    inverse of unpack_nibbles (codec_row.h's codec_pack_nibbles)."""
    O = sym.shape[0]
    row_bytes = (I + 1) // 2
    out = np.zeros((O, row_bytes), dtype=np.uint8)
    n_pairs = I // 2
    out[:, :n_pairs] = (sym[:, 0:2 * n_pairs:2] & 0xF) | (sym[:, 1:2 * n_pairs:2] << 4)
    if I % 2:
        out[:, n_pairs] = sym[:, I - 1] & 0xF
    return out


# ------------------------------------------------- canonical huffman ----
def _bitrev(x: int, length: int) -> int:
    r = 0
    for _ in range(length):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def huff_canonical_codes(lengths: np.ndarray) -> np.ndarray:
    """EXACT port of codec_row_huff.h's huff_canonical_codes(): stable
    order by (length ascending, symbol ascending), classic canonical-code
    assignment, each codeword bit-reversed for LSB-first streaming. Must
    match the C function bit-for-bit -- see module docstring."""
    order = sorted((s for s in range(NSYM) if lengths[s] > 0),
                   key=lambda s: (int(lengths[s]), s))
    code_rev = np.zeros(NSYM, dtype=np.uint32)
    code = 0
    prev_len = 0
    for s in order:
        L = int(lengths[s])
        code <<= (L - prev_len)
        code_rev[s] = _bitrev(code, L)
        code += 1
        prev_len = L
    return code_rev


def build_lut(lengths: np.ndarray, code_rev: np.ndarray):
    """Port of codec_row_huff.h's huff_build() LUT-fill loop. Returns
    (maxlen, lut_sym uint8[1<<maxlen], lut_len uint8[1<<maxlen])."""
    present = [s for s in range(NSYM) if lengths[s] > 0]
    maxlen = max((int(lengths[s]) for s in present), default=0)
    if maxlen <= 0:
        maxlen = 1  # matches C: cb->maxlen = maxlen>0 ? maxlen : 1
    lut_size = 1 << maxlen
    lut_sym = np.zeros(lut_size, dtype=np.uint8)
    lut_len = np.zeros(lut_size, dtype=np.uint8)
    if len(present) == 1:
        s = present[0]
        lut_sym[:] = s
        lut_len[:] = 1
        return maxlen, lut_sym, lut_len
    for s in present:
        L = int(lengths[s])
        base = int(code_rev[s])
        rest = maxlen - L
        block = 1 << rest
        hi = np.arange(block, dtype=np.uint32) << L
        idx = base | hi
        lut_sym[idx] = s
        lut_len[idx] = L
    return maxlen, lut_sym, lut_len


def pack_lengths(lengths: np.ndarray) -> bytes:
    """16 x 4-bit lengths -> 8 bytes, low nibble=even symbol (matches the
    codec's universal nibble convention)."""
    out = bytearray(LENGTHS_BYTES)
    for k in range(LENGTHS_BYTES):
        lo = int(lengths[2 * k]) & 0xF
        hi = int(lengths[2 * k + 1]) & 0xF
        out[k] = lo | (hi << 4)
    return bytes(out)


def unpack_lengths(data: bytes) -> np.ndarray:
    lengths = np.zeros(NSYM, dtype=np.uint8)
    for k in range(LENGTHS_BYTES):
        b = data[k]
        lengths[2 * k] = b & 0xF
        lengths[2 * k + 1] = b >> 4
    return lengths


# ------------------------------------------------ vectorized encode ----
def encode_tensor(nibbles: np.ndarray, lengths: np.ndarray, code_rev: np.ndarray):
    """(O,I) uint8 nibble symbols + codebook -> (payload bytes, row_offsets
    uint64[O+1]). Vectorized ACROSS ROWS: one Python-level iteration per
    symbol COLUMN (I iterations), each doing O(1) numpy ops over all O
    rows at once -- an O*I-fold reduction in interpreter overhead vs a
    naive per-symbol loop, which is what makes encoding real O=2048/6144-
    row expert tensors tractable.

    Per-row bitstreams are independently byte-aligned (each row's stream
    starts at bit 0 of a fresh byte and is zero-padded to a byte at the
    end) -- matches codec_row_huff.h's huff_encode_row(), which calls
    cbw_init/cbw_finish ONCE per row, and is required for O(1) random
    access via row_offsets.
    """
    O, I = nibbles.shape
    if O == 0:
        return b"", np.zeros(1, dtype=np.uint64)
    L = lengths[nibbles].astype(np.uint32)          # (O,I) codeword bit-lengths
    C = code_rev[nibbles]                            # (O,I) bit-reversed codewords
    if I == 0:
        row_offsets = np.zeros(O + 1, dtype=np.uint64)
        return b"", row_offsets

    start = np.cumsum(L, axis=1, dtype=np.uint32) - L    # (O,I) start bit offset in row
    total_bits = (start[:, -1] + L[:, -1]).astype(np.uint64)
    row_bytes = ((total_bits + 7) // 8).astype(np.uint64)
    row_offsets = np.zeros(O + 1, dtype=np.uint64)
    row_offsets[1:] = np.cumsum(row_bytes)
    total_payload = int(row_offsets[-1])
    max_row_bytes = int(row_bytes.max())

    buf = np.zeros((O, max_row_bytes + 2), dtype=np.uint8)  # +2 headroom, see byte-span note below
    byte_pos = (start >> 3).astype(np.int64)
    bit_off = (start & 7).astype(np.uint32)
    rows_idx = np.arange(O)

    for i in range(I):
        ci = C[:, i]
        bo = bit_off[:, i]
        shifted = (ci << bo).astype(np.uint32)   # up to (HUFF_MAXLEN-1)+7 = 21 bits -> spans <=3 bytes
        bp = byte_pos[:, i]
        buf[rows_idx, bp] += (shifted & 0xFF).astype(np.uint8)
        buf[rows_idx, bp + 1] += ((shifted >> 8) & 0xFF).astype(np.uint8)
        buf[rows_idx, bp + 2] += ((shifted >> 16) & 0xFF).astype(np.uint8)
        # Safe: for every symbol, (start+L-1)//8 <= row's own final
        # row_bytes-1 (prefix sums are non-decreasing and the LAST symbol
        # hits exactly row_bytes-1), and per-row sums of disjoint,
        # non-overlapping codeword bits never exceed 255 -- see module's
        # design notes / tests/test_mode15_container.py for the argument
        # and a regression fuzz test.

    col_idx = np.arange(max_row_bytes + 2)[None, :]
    mask = col_idx < row_bytes[:, None]
    payload = buf[mask]
    assert payload.shape[0] == total_payload
    return payload.tobytes(), row_offsets


# ------------------------------------------------ vectorized decode ----
def decode_tensor(payload: np.ndarray, row_offsets: np.ndarray, O: int, I: int,
                   maxlen: int, lut_sym: np.ndarray, lut_len: np.ndarray) -> np.ndarray:
    """Inverse of encode_tensor: payload (1-D uint8) + row_offsets ->
    (O,I) uint8 nibble symbols. Vectorized the same way (one Python-level
    iteration per symbol column, O(1) numpy ops over all O rows), directly
    mirroring codec_row_huff.h's huff_decode_row() / the Metal
    huff_decode_rows kernel's per-row bit-walk (refill -> peek maxlen bits
    -> LUT resolves symbol+consumed-length -> drop)."""
    out = np.zeros((O, I), dtype=np.uint8)
    if O == 0 or I == 0:
        return out
    pos = row_offsets[:-1].astype(np.int64).copy()
    end = row_offsets[1:].astype(np.int64)
    acc = np.zeros(O, dtype=np.uint32)
    nbits = np.zeros(O, dtype=np.int64)
    mask = np.uint32((1 << maxlen) - 1)
    payload = np.asarray(payload, dtype=np.uint8)
    # ceil(HUFF_MAXLEN/8)+1 rounds always suffices to reach nbits>=maxlen
    # from any valid starting nbits in [0, maxlen) -- see cbr_refill's C
    # while-loop, mirrored here as a bounded-round vectorized version.
    n_refill_rounds = (HUFF_MAXLEN + 7) // 8 + 1

    for i in range(I):
        for _ in range(n_refill_rounds):
            need = (nbits < maxlen) & (pos < end)
            if not need.any():
                break
            nb = np.zeros(O, dtype=np.uint32)
            nb[need] = payload[pos[need]].astype(np.uint32)
            acc = np.where(need, acc | (nb << nbits.astype(np.uint32)), acc)
            pos = np.where(need, pos + 1, pos)
            nbits = np.where(need, nbits + 8, nbits)
        window = (acc & mask).astype(np.int64)
        sym = lut_sym[window]
        length = lut_len[window].astype(np.int64)
        out[:, i] = sym
        acc = np.right_shift(acc, length.astype(np.uint32))
        nbits = nbits - length
    return out


# --------------------------------------------------- codebook build ----
def build_codebook(nibbles: np.ndarray):
    """(O,I) uint8 -> (lengths[16] uint8, code_rev[16] uint32, maxlen,
    lut_sym, lut_len). One codebook per TENSOR (= one gate/up/down
    projection of one expert), built from that tensor's own aggregate
    histogram -- codec_row.h's own definition of "projection": "A tensor
    (one gate/up/down projection of one expert) is O rows ... each row is
    encoded independently against a CODEBOOK SHARED across the whole
    projection." This also keeps the encoder single-pass/per-shard
    resumable: no cross-shard/cross-tensor histogram aggregation is
    needed (see encode_mode15_container.py's docstring for the
    alternative coarser-grouping option this deliberately avoids)."""
    counts = np.bincount(nibbles.ravel(), minlength=NSYM).astype(np.int64)
    lengths = huffman_code_lengths(counts)
    lengths = np.asarray(lengths, dtype=np.uint8)
    code_rev = huff_canonical_codes(lengths)
    maxlen, lut_sym, lut_len = build_lut(lengths, code_rev)
    return lengths, code_rev, maxlen, lut_sym, lut_len


# --------------------------------------------------- tensor blob I/O ----
def make_tensor_blob(nibbles: np.ndarray, rows_per_block: int = DEFAULT_ROWS_PER_BLOCK) -> bytes:
    """Full encode of one expert weight tensor -> the on-disk blob
    (magic+header+lengths+row_offsets+block_crc32s+payload), per the
    module docstring's layout."""
    O, I = nibbles.shape
    lengths, code_rev, maxlen, _, _ = build_codebook(nibbles)
    payload, row_offsets = encode_tensor(nibbles, lengths, code_rev)
    n_blocks = max(1, -(-O // rows_per_block)) if O > 0 else 0
    block_crc = np.zeros(n_blocks, dtype=np.uint32)
    for b in range(n_blocks):
        r0, r1 = b * rows_per_block, min(O, (b + 1) * rows_per_block)
        s0, s1 = int(row_offsets[r0]), int(row_offsets[r1])
        block_crc[b] = zlib.crc32(payload[s0:s1]) & 0xFFFFFFFF

    lengths_bytes = pack_lengths(lengths)
    row_offsets_bytes = row_offsets.astype("<u4").tobytes()
    block_crc_bytes = block_crc.astype("<u4").tobytes()
    body = lengths_bytes + row_offsets_bytes + block_crc_bytes + payload
    tensor_crc32 = zlib.crc32(body) & 0xFFFFFFFF
    header = _HEADER_STRUCT.pack(MAGIC, O, I, rows_per_block, n_blocks, tensor_crc32)
    return header + body


def parse_tensor_blob(blob: bytes, expect_O: int | None = None, expect_I: int | None = None,
                       verify_checksums: bool = True):
    """Structural parse + (optionally) full checksum verification of a
    tensor blob. Fail-closed: any truncation, magic mismatch, or checksum
    mismatch raises rather than returning partial/best-effort data.

    Returns a dict: O, I, rows_per_block, n_blocks, lengths(np.uint8[16]),
    row_offsets(np.uint64[O+1]), block_crc32(np.uint32[n_blocks]),
    payload(np.uint8 1-D), plus the derived code_rev/maxlen/lut_sym/lut_len
    codebook (rebuilt from `lengths` via the SAME canonical-code path a
    C/Metal reader would use).
    """
    if len(blob) < TENSOR_HEADER_LEN:
        raise Mode15FormatError(f"truncated: blob is {len(blob)} bytes, "
                                 f"need >= {TENSOR_HEADER_LEN} for the header")
    magic, O, I, rows_per_block, n_blocks, tensor_crc32 = _HEADER_STRUCT.unpack_from(blob, 0)
    if magic != MAGIC:
        raise Mode15FormatError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if expect_O is not None and O != expect_O:
        raise Mode15FormatError(f"O mismatch: blob says {O}, expected {expect_O}")
    if expect_I is not None and I != expect_I:
        raise Mode15FormatError(f"I mismatch: blob says {I}, expected {expect_I}")
    expect_n_blocks = max(1, -(-O // rows_per_block)) if O > 0 else 0
    if n_blocks != expect_n_blocks:
        raise Mode15FormatError(f"n_blocks mismatch: blob says {n_blocks}, "
                                 f"expected {expect_n_blocks} for O={O} rows_per_block={rows_per_block}")

    off = TENSOR_HEADER_LEN
    if len(blob) < off + LENGTHS_BYTES:
        raise Mode15FormatError("truncated: missing lengths table")
    lengths = unpack_lengths(blob[off:off + LENGTHS_BYTES])
    off += LENGTHS_BYTES

    ro_bytes = (O + 1) * 4
    if len(blob) < off + ro_bytes:
        raise Mode15FormatError("truncated: missing row_offsets index")
    row_offsets = np.frombuffer(blob, dtype="<u4", count=O + 1, offset=off).astype(np.uint64)
    off += ro_bytes

    bc_bytes = n_blocks * 4
    if len(blob) < off + bc_bytes:
        raise Mode15FormatError("truncated: missing block checksums")
    block_crc32 = np.frombuffer(blob, dtype="<u4", count=n_blocks, offset=off).copy()
    off += bc_bytes

    payload_len = int(row_offsets[O]) if O > 0 else 0
    expected_total = off + payload_len
    if len(blob) != expected_total:
        raise Mode15FormatError(
            f"length mismatch: blob is {len(blob)} bytes, expected exactly "
            f"{expected_total} (truncated if shorter, trailing garbage if longer) "
            "-- fail-closed on ANY deviation, not just short reads")
    payload = np.frombuffer(blob, dtype=np.uint8, count=payload_len, offset=off)

    if verify_checksums:
        body = blob[TENSOR_HEADER_LEN:]
        got = zlib.crc32(body) & 0xFFFFFFFF
        if got != tensor_crc32:
            raise Mode15ChecksumError(
                f"whole-tensor CRC32 mismatch: got {got:#010x}, stored {tensor_crc32:#010x}")
        for b in range(n_blocks):
            r0, r1 = b * rows_per_block, min(O, (b + 1) * rows_per_block)
            s0, s1 = int(row_offsets[r0]), int(row_offsets[r1])
            got_b = zlib.crc32(payload[s0:s1].tobytes()) & 0xFFFFFFFF
            if got_b != int(block_crc32[b]):
                raise Mode15ChecksumError(
                    f"block {b} (rows [{r0},{r1})) CRC32 mismatch: "
                    f"got {got_b:#010x}, stored {int(block_crc32[b]):#010x}")

    code_rev = huff_canonical_codes(lengths)
    maxlen, lut_sym, lut_len = build_lut(lengths, code_rev)
    return {
        "O": O, "I": I, "rows_per_block": rows_per_block, "n_blocks": n_blocks,
        "tensor_crc32": tensor_crc32, "lengths": lengths, "row_offsets": row_offsets,
        "block_crc32": block_crc32, "payload": payload,
        "code_rev": code_rev, "maxlen": maxlen, "lut_sym": lut_sym, "lut_len": lut_len,
    }


def verify_block(blob: bytes, block_idx: int) -> bool:
    """Check ONE block's checksum without verifying the whole tensor --
    supports localizing which block a corruption lives in, and lets a
    caller prove per-block checksums catch damage independent of the
    whole-tensor gate (see tests/test_mode15_container.py)."""
    parsed = parse_tensor_blob(blob, verify_checksums=False)
    O, rows_per_block = parsed["O"], parsed["rows_per_block"]
    row_offsets, payload = parsed["row_offsets"], parsed["payload"]
    r0 = block_idx * rows_per_block
    r1 = min(O, r0 + rows_per_block)
    s0, s1 = int(row_offsets[r0]), int(row_offsets[r1])
    got = zlib.crc32(payload[s0:s1].tobytes()) & 0xFFFFFFFF
    return got == int(parsed["block_crc32"][block_idx])


def write_cross_check_fixture(path: str, nibbles: np.ndarray) -> None:
    """Write a 'MHFX' fixture for tools/mode15_cross_check.c: encodes
    `nibbles` with this module's own codebook-build + encode_tensor, then
    dumps [magic][O][I][8B lengths][original nibbles][row_offsets][payload]
    so the C program can decode via the REAL codec_row_huff.h functions
    and compare against the known-original symbols. See
    mode15_cross_check.c's file header for why this independent check
    matters (this module's own round trip only proves internal self-
    consistency, not agreement with the actual C/Metal decode path)."""
    O, I = nibbles.shape
    lengths, code_rev, _maxlen, _lut_sym, _lut_len = build_codebook(nibbles)
    payload, row_offsets = encode_tensor(nibbles, lengths, code_rev)
    with open(path, "wb") as f:
        f.write(b"MHFX")
        f.write(struct.pack("<II", O, I))
        f.write(pack_lengths(lengths))
        f.write(np.ascontiguousarray(nibbles, dtype=np.uint8).tobytes())
        f.write(row_offsets.astype("<u4").tobytes())
        f.write(payload)


def decode_blob_to_packed_bytes(blob: bytes, expect_O=None, expect_I=None) -> bytes:
    """Full round trip: parse+verify a tensor blob, decode every row, and
    repack to the ORIGINAL int4-packed byte layout for a byte-exact
    comparison against the source. This is the function gate_m15_g1.py's
    bit-exactness check calls."""
    parsed = parse_tensor_blob(blob, expect_O=expect_O, expect_I=expect_I, verify_checksums=True)
    nibbles = decode_tensor(parsed["payload"], parsed["row_offsets"], parsed["O"], parsed["I"],
                             parsed["maxlen"], parsed["lut_sym"], parsed["lut_len"])
    return pack_nibbles(nibbles, parsed["I"]).tobytes()
