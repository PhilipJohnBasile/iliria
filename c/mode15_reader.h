/* mode15_reader.h -- structural reader for mode-1.5 (canonical-Huffman
 * entropy-coded) tensor-blob containers, as defined by
 * tools/mode15_container.py. Registration: c/bench-m5max/mode15-lossless-
 * the pipeline registration. Integration plan: c/bench-m5max/
 * the Mode-15 integration design notes step 3 ("Read compressed container header,
 * glm.c, inert") and §1a's OWNERSHIP AMENDMENT (2026-07-18, appended
 * below this header's own hardening-round history note).
 *
 * HARDENING ROUND (2026-07-18, post external-review pass): this
 * header replaces the prior single-call, fully-borrowed M15BlobInfo API
 * with an open/close-lifecycle M15Reader, in response to 8 findings, all
 * accepted. Each is called out at its relevant section below; this
 * paragraph is the index:
 *   (1) OWNERSHIP contract (this section, "LIFETIME CONTRACT")
 *   (2)+(6) verify-once bitmap + structural/full open split ("VERIFICATION MODEL")
 *   (3) length-table validation ("LENGTH TABLE VALIDITY")
 *   (4) CRC caveat ("THREAT MODEL")
 *   (5) caller-supplied caps (M15Caps, below)
 *   (7) hot-path unchecked accessor (m15_get_row_span_unchecked)
 *   (8) error-code discipline (M15_NODISCARD, corrupt-data vs
 *       programmer-error split, m15_is_programmer_error(), cleared
 *       outputs on entry)
 *
 * SCOPE (unchanged). A PURE STRUCTURAL PARSER: magic/version sniff,
 * length-table validity, row_offsets index, CRC32 (lazy or eager). Does
 * NOT decode Huffman bitstreams, does NOT build a HuffCodebook/LUT -- that
 * stays codec_row_huff.h's job (this header does not include codec_row.h/
 * codec_row_huff.h; plain C11 + <stdatomic.h> only, no engine dependency).
 * A caller decodes row `r` by taking the (ptr,len) span m15_get_row_span()
 * (or its unchecked twin) hands back, plus this module's validated
 * `lengths` field, into codec_row_huff.h's huff_canonical_codes()/
 * huff_build-style LUT construction and huff_decode_row() -- see
 * tests/test_mode15_reader.c for the exact wiring.
 *
 * LIFETIME CONTRACT (finding 1 -- READ BEFORE USING THIS MODULE).
 *   - INDEX IS OWNED. m15_open_structural()/m15_open_full() copy the
 *     validated row_offsets and block_crc32 tables (small: O(O) and
 *     O(n_blocks) uint32_t entries -- a few KB even for a 6144-row
 *     tensor) into memory THIS MODULE allocates and owns. Once open
 *     returns M15_OK, the CALLER'S `blob` BUFFER MAY BE FREED, REUSED, OR
 *     MUTATED WITHOUT AFFECTING `O`/`I`/`rows_per_block`/`n_blocks`/
 *     `lengths`/`m15_row_offset()`/`m15_block_crc32()` -- all copied,
 *     none of it re-reads `blob`. Verified in
 *     tests/test_mode15_reader.c's index-copy-lifetime test (free the
 *     source buffer after open; index accessors still agree).
 *   - PAYLOAD IS STILL BORROWED. `r->payload` is a raw pointer INTO the
 *     caller's original `blob` buffer -- NOT copied (copying megabytes of
 *     compressed weight bytes on every expert load would defeat the
 *     entire "decode straight into the consumer buffer" design point,
 *     the Mode-15 integration design notes §1c). THE CALLER MUST KEEP THE
 *     UNDERLYING BYTES `blob` POINTED TO AT OPEN TIME PINNED, IMMUTABLE,
 *     AND MAPPED FOR THE ENTIRE LIFETIME OF THE M15Reader (i.e. until the
 *     matching m15_close(), or until the reader is discarded) -- any
 *     m15_get_row_span()/m15_get_row_span_unchecked() pointer this module
 *     hands back is a view into THAT SAME memory, valid on the exact same
 *     terms. In the real integration (design doc §1a/§1b): the
 *     pread-into-staging-buffer that produced `blob` must not be reused
 *     for a different expert's bytes (nor grown/reallocated) until every
 *     decode dispatch reading from it has been submitted (ordinary GPU
 *     command-buffer submission already implies this: `setBuffer:`
 *     retains the MTLBuffer, and the staging bytes must outlive encoding,
 *     same requirement any zero-copy GPU input buffer already has today).
 *     This module does NOT solve cross-thread staging-buffer recycling
 *     races by itself -- it only guarantees the INDEX survives; getting
 *     the payload's pin/lifetime right at the call site is the
 *     integration's job, called out explicitly (not silently assumed)
 *     because a borrowed multi-megabyte buffer read from several async
 *     load paths (design doc §1b: pipe worker / omp-parallel-for /
 *     speculative pilot / PIN-REPIN) is exactly the TOCTOU/use-after-
 *     free/SIGBUS shape a reviewer should be suspicious of by default.
 *   - m15_close() frees the OWNED arrays (row_offsets/block_crc32/
 *     block_state) and zeroes the M15Reader; it does NOT touch `blob`
 *     (never owned it) and is safe to call on a zero-initialized or
 *     already-closed reader (idempotent).
 *
 * VERIFICATION MODEL (findings 2 + 6).
 *   Two DISTINCT things, kept deliberately separate (mirrors §1d of the
 *   design doc's "two different failure classes, kept deliberately
 *   separate" framing, applied one level down):
 *     - STRUCTURAL validity (magic/version, length-table Kraft validity,
 *       row_offsets monotonicity, every truncation/size check, caps) --
 *       ALWAYS fully checked by BOTH m15_open_structural() and
 *       m15_open_full(), and REQUIRED for M15_OK: a structurally invalid
 *       blob never produces a live reader, verify policy or not.
 *     - DATA integrity (whole-tensor / per-block CRC32) -- hashing every
 *       byte of a compressed expert tensor at open time would fault in
 *       the entire (possibly multi-megabyte, possibly mmap'd-cold)
 *       payload before a single row is ever needed; real usage only
 *       needs the ~1-3 blocks a given GEMV pass actually touches. So:
 *         - m15_open_structural(): NO payload bytes are hashed at open.
 *           `r->tensor_verified==0` and every `block_state[b]` starts
 *           M15_BLOCK_UNVERIFIED.
 *         - m15_verify_block_once(r, b): hashes block b's own payload
 *           range AT MOST ONCE per reader -- cached in `block_state[b]`
 *           (M15_BLOCK_GOOD/M15_BLOCK_BAD), so N calls against the same
 *           block after the first do zero CRC work and return the cached
 *           verdict. Intended call shape, one verify per block before
 *           decoding any of its rows: for each block -> verify_block_once
 *           -> for each row in that block -> get_row_span[_unchecked] ->
 *           decode (mirrors the container's own block granularity, tied
 *           to one GPU threadgroup's row batch, mode15_container.py's
 *           DEFAULT_ROWS_PER_BLOCK comment).
 *         - m15_verify_tensor_once(r): whole-tensor CRC32, cached in
 *           `r->tensor_verified` the same way -- for callers that want
 *           the single strongest fail-closed gate without paying for
 *           every block individually.
 *         - m15_open_full(): convenience = m15_open_structural() +
 *           m15_verify_tensor_once() + m15_verify_block_once() for EVERY
 *           block, eagerly, at open time -- the G1-style offline
 *           full-container gate's use case (bit-exactness proof before a
 *           benchmark/ship claim), not the hot per-expert-load path.
 *
 * LENGTH TABLE VALIDITY (finding 3). A packed 4-bit nibble alone (values
 * 0..15) is NOT sufficient: codec_row_huff.h's huff_canonical_codes()/
 * huff_build() apply NO validation of their own and will silently build a
 * WRONG (not merely absent) codebook from an invalid length table --
 * oversubscribed lengths collide codewords (two symbols get the same
 * code; huff_canonical_codes has no way to detect this, later symbols'
 * codes simply overwrite earlier ones' LUT entries), and undersubscribed
 * (incomplete) lengths leave LUT holes at len=0, which huff_decode_row()'s
 * FIXED n-iteration loop cannot detect landing on (it decodes symbol 0
 * with zero bits consumed and silently repeats). Neither failure mode
 * crashes or errors in codec_row_huff.h -- both just produce wrong
 * decoded weights, invisibly, exactly the "hidden corruption reaching the
 * model's numerical output" class of bug a reader sitting in front of the
 * decoder should catch. m15_open_structural()/m15_open_full() therefore
 * validate the length table (m15_lengths_valid() in the .c file) against
 * codec_row_huff.h's ACTUAL two decode conventions -- read from that
 * header, not invented independently:
 *   - n_present==0 valid ONLY when O==0 or I==0 (the table can never be
 *     consulted); rejected (M15_ERR_BAD_LENGTH_TABLE) for any tensor with
 *     real rows and real per-row symbols.
 *   - n_present==1: codec_row_huff.h's explicit single-symbol convention
 *     is exactly a 1-bit code (its huff_build_lengths(): "if(n==1){
 *     len_out[leaf_of[0]]=1; ... }") -- NOT a Kraft-sum requirement (a
 *     lone symbol's Kraft weight is 2^-1, not 1: the standard single-leaf
 *     exception). Any other length for the sole present symbol is
 *     rejected.
 *   - n_present>=2: a real canonical-Huffman build (mode15_container.py's
 *     huffman_code_lengths / codec_row_huff.h's huff_build_lengths, both
 *     genuine greedy tree merges) always yields a COMPLETE prefix code:
 *     sum over present symbols of 2^-len[s] == 1 EXACTLY (integer
 *     arithmetic, scaled by 2^15 -- lengths are dyadic, no floating point
 *     needed for an exact pass/fail gate). Both oversubscribed (sum>1)
 *     and incomplete (sum<1) are rejected.
 *   - Any length > 15 is rejected too (HUFF_MAXLEN, codec_row_huff.h) --
 *     unreachable via THIS format's packed-nibble encoding (a 4-bit field
 *     cannot exceed 15), kept anyway as defense in depth against a future
 *     wire-format change, not exercised by today's fixtures.
 *
 * THREAT MODEL / CRC CAVEAT (finding 4). CRC32 (whole-tensor or
 * per-block) detects ACCIDENTAL CORRUPTION (bitrot, truncated writes,
 * torn reads) -- it is NOT a cryptographic integrity check and provides
 * NO protection against a deliberately hostile actor who can compute a
 * matching CRC32 for arbitrary modified bytes (trivial: CRC32 is linear
 * and unkeyed). This is an accepted, DOCUMENTED gap, not a bug to fix
 * here: mode-1.5 containers are single-user local files on the same
 * machine that runs the engine, in the same trust domain as the
 * uncompressed containers, model weights, and the engine binary itself
 * (none of which are integrity-protected against a hostile local actor
 * today either) -- adding a keyed MAC/signature here would protect
 * against a threat model this codebase does not otherwise defend against
 * anywhere in the load path, and is out of scope for this sidecar.
 *
 * Caller-supplied caps (finding 5): a corrupt (not necessarily hostile --
 * bitrot in the header itself, before any CRC has even been consulted,
 * is exactly the case truncation checks alone don't fully bound) header
 * can claim an enormous O/I/n_blocks/payload_len; without a cap, the
 * O(O) row_offsets copy+validate or the O(n_blocks) block table copy
 * would scale with whatever the header claims, not with what the caller
 * actually expects. M15Caps (0 = uncapped per field) is checked BEFORE
 * any O(n) work -- see mode15_reader.c's ordering comment. The engine is
 * expected to pass real expected-shape caps (Cfg's hidden/moe_inter
 * bounds), not leave this uncapped in production.
 *
 * Style matched to codec_row_huff.h (dense header comments) but split
 * declaration/definition (mode15_reader.h + mode15_reader.c) rather than
 * header-only `static` functions, matching backend_cuda.h/backend_metal.h.
 */
#ifndef ILIRIA_MODE15_READER_H
#define ILIRIA_MODE15_READER_H

#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/* portable nodiscard (finding 8): every fallible entry point below is
 * annotated so a caller that drops the return status (as opposed to
 * merely deferring on `out`) gets a compiler warning, not silence. */
#if defined(__GNUC__) || defined(__clang__)
#define M15_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER) && _MSC_VER >= 1700
#define M15_NODISCARD _Check_return_
#else
#define M15_NODISCARD
#endif

#define M15_LENGTHS_BYTES     8    /* 16 x 4-bit lengths, codec_row.h's CODEC_NSYM=16 convention */
#define M15_NSYM              16
#define M15_TENSOR_HEADER_LEN 24   /* magic..tensor_crc32, mode15_container.py's TENSOR_HEADER_LEN */

/* Error-code discipline (finding 8): two DISTINCT classes, contiguous
 * ranges so a single threshold comparison (m15_is_programmer_error())
 * tells them apart. CORRUPT-DATA class describes a property of the BLOB
 * BYTES (or a mismatch against the caller's stated expectation of what
 * tensor this should be) -- always a legitimate candidate for the
 * fail-closed fallback-mirror path (design doc §1d). PROGRAMMER-ERROR
 * class describes API MISUSE independent of any blob's contents -- MUST
 * NEVER be silently routed into "corrupt container, try the mirror"
 * fallback logic, because that would mask a real bug in the calling
 * (engine) code as if it were a data problem. */
typedef enum {
    M15_OK = 0,

    /* ---- corrupt-data class (blob bytes, or blob-vs-expectation) ---- */
    M15_ERR_TRUNCATED_HEADER,       /* blob shorter than M15_TENSOR_HEADER_LEN */
    M15_ERR_BAD_MAGIC,              /* first 2 bytes != "MH" -- not a mode-1.5 blob at all */
    M15_ERR_VERSION_MISMATCH,       /* "MH" family ok, version suffix != supported ("01") */
    M15_ERR_O_MISMATCH,             /* expect_O given (>=0) and blob's O disagrees -- wrong tensor mapped here */
    M15_ERR_I_MISMATCH,             /* expect_I given (>=0) and blob's I disagrees */
    M15_ERR_CAP_EXCEEDED,           /* O/I/n_blocks/payload_len exceeds a caller-supplied M15Caps bound */
    M15_ERR_BAD_ROWS_PER_BLOCK,     /* rows_per_block == 0 with O > 0 (would be div-by-zero) */
    M15_ERR_BLOCK_COUNT_MISMATCH,   /* stored n_blocks != ceil(O/rows_per_block) */
    M15_ERR_TRUNCATED_LENGTHS,      /* blob too short for the 8B lengths table */
    M15_ERR_BAD_LENGTH_TABLE,       /* Kraft-oversubscribed/incomplete, wrong single-symbol length, or all-zero on a non-degenerate tensor (see LENGTH TABLE VALIDITY above) */
    M15_ERR_TRUNCATED_ROW_OFFSETS,  /* blob too short for the (O+1)*4B row_offsets table */
    M15_ERR_ROW_OFFSETS_INVALID,    /* row_offsets[0]!=0 or non-monotonic */
    M15_ERR_TRUNCATED_BLOCK_CRC,    /* blob too short for the n_blocks*4B block_crc32 table */
    M15_ERR_LENGTH_MISMATCH,        /* blob size != header-implied total size (truncated payload or trailing garbage) */
    M15_ERR_TENSOR_CRC_MISMATCH,    /* whole-tensor CRC32 (offset 24..end) mismatch */
    M15_ERR_BLOCK_CRC_MISMATCH,     /* one block's own CRC32 mismatch (bad_block_idx out-param set, if requested) */

    /* ---- programmer-error class (API misuse, independent of blob) ---- */
    M15_ERR_BAD_ARGUMENT,           /* NULL blob/out pointer, or a row/block index out of range */
    M15_ERR_NOT_OPEN,               /* reader used before a successful m15_open_*() or after m15_close() */
    M15_ERR_ALLOC_FAILED,           /* malloc() failed for the owned index arrays -- environment/resource
                                      * exhaustion, not a property of the blob: a fallback-mirror retry
                                      * against a DIFFERENT file will not fix an out-of-memory process, so
                                      * this is bucketed with programmer-error (propagate/abort), not
                                      * corrupt-data (try the mirror) */
} M15Status;

/* First programmer-error-class code; every value >= this threshold is
 * programmer error, everything below it (and M15_OK) is corrupt-data
 * class or success. Prefer m15_is_programmer_error() over comparing
 * against this directly -- it's the documented, stable entry point. */
#define M15_FIRST_PROGRAMMER_ERROR M15_ERR_BAD_ARGUMENT

/* nonzero iff `st` is a programmer-error-class code (API misuse) rather
 * than a corrupt-data-class code -- engine fallback logic MUST branch on
 * this before treating any non-OK status as "try the uncompressed
 * mirror" (finding 8): a programmer error must propagate/abort, never be
 * swallowed as if the container were merely corrupt. */
static inline int m15_is_programmer_error(M15Status st){
    return st >= M15_FIRST_PROGRAMMER_ERROR;
}

/* Per-block lazy-verification cache state (finding 2). */
#define M15_BLOCK_UNVERIFIED 0
#define M15_BLOCK_GOOD       1
#define M15_BLOCK_BAD        2

/* Caller-supplied trust bounds (finding 5), checked before any work that
 * scales with a header-claimed size. 0 in any field = uncapped for that
 * field. Pass NULL to m15_open_*() for fully uncapped (matches this
 * module's pre-hardening behavior; the real engine integration is
 * expected to always pass real caps derived from Cfg's hidden/moe_inter,
 * not rely on the uncapped default in production). */
typedef struct {
    uint32_t max_O;
    uint32_t max_I;
    uint32_t max_n_blocks;
    uint64_t max_payload_len;
} M15Caps;

/* An open mode-1.5 tensor reader. TRANSPARENT (fields visible, not an
 * opaque handle) specifically so m15_get_row_span_unchecked() below can
 * be `static inline` in this header and genuinely inline at ~57k-opens-
 * plus-per-row-decode call volumes without relying on LTO -- but treat
 * every field as READ-ONLY from outside mode15_reader.c except through
 * the documented accessors (`block_state` in particular is mutated only
 * via m15_verify_block_once()). See the file header's LIFETIME CONTRACT
 * for what is owned (row_offsets, block_crc32, block_state -- freed by
 * m15_close()) versus borrowed (`payload`, into the caller's own blob,
 * never freed by this module). Zero-initialize (or rely on m15_open_*'s
 * own on-entry clear) before first use; `is_open` guards
 * use-before-open/after-close (M15_ERR_NOT_OPEN). */
typedef struct {
    int      is_open;
    uint32_t O, I;
    uint32_t rows_per_block;
    uint32_t n_blocks;
    uint32_t tensor_crc32;          /* as stored in the header (informational) */
    uint8_t  lengths[M15_NSYM];     /* VALIDATED (see LENGTH TABLE VALIDITY) canonical-code lengths, 0=absent */
    uint8_t  maxlen;                /* max code length actually used, <=15 */
    uint8_t  n_present;             /* count of present symbols; 0 only possible when O==0 or I==0 */
    uint32_t *row_offsets;          /* OWNED, O+1 entries, host-native (copied+validated at open) */
    uint32_t *block_crc32;          /* OWNED, n_blocks entries, host-native */
    atomic_uchar *block_state;      /* OWNED, n_blocks entries: M15_BLOCK_{UNVERIFIED,GOOD,BAD} */
    int       tensor_verified;      /* whole-tensor CRC32 checked and passed (m15_verify_tensor_once) */
    const uint8_t *payload;         /* BORROWED -- see LIFETIME CONTRACT. row_offsets[O] bytes. */
    uint32_t payload_len;
} M15Reader;

/* STRUCTURAL open (findings 2+6): header/magic/version, length-table
 * validity, row_offsets monotonicity, every truncation/size/cap check --
 * ALL enforced. NO payload bytes are hashed (tensor_verified starts 0,
 * every block_state[] starts M15_BLOCK_UNVERIFIED). `out` is cleared and
 * `out->is_open` set false on entry, before any validation (finding 8) --
 * safe to inspect even if the call fails, always M15_ERR_NOT_OPEN-safe.
 * On M15_OK, `out` owns row_offsets/block_crc32/block_state (m15_close()
 * required); on any other return, no allocation is leaked and `out`
 * remains closed. `caps` may be NULL (uncapped). expect_O/expect_I < 0
 * means "don't check" (mirrors mode15_container.py's expect_O:
 * int|None=None). */
M15Status m15_open_structural(const uint8_t *blob, size_t blob_len,
                               int64_t expect_O, int64_t expect_I,
                               const M15Caps *caps,
                               M15Reader *out) M15_NODISCARD;

/* m15_open_structural() + eager m15_verify_tensor_once() + eager
 * m15_verify_block_once() for EVERY block -- the G1-style full-container
 * offline gate's shape (bit-exactness proof before a benchmark/ship
 * claim), not the per-expert-load hot path (finding 6). On
 * M15_ERR_BLOCK_CRC_MISMATCH, *out_bad_block (if non-NULL) is set to the
 * first failing block's index (blocks checked in order 0..n_blocks-1,
 * matching mode15_container.py's parse_tensor_blob loop). `out` is
 * populated (and usable, including for further m15_verify_block_once()
 * calls on OTHER blocks) if the structural phase succeeded even when a
 * later CRC check is what actually fails -- check the return status,
 * not just "did I get a reader," when M15_OK matters to you. */
M15Status m15_open_full(const uint8_t *blob, size_t blob_len,
                         int64_t expect_O, int64_t expect_I,
                         const M15Caps *caps,
                         M15Reader *out,
                         uint32_t *out_bad_block) M15_NODISCARD;

/* Frees the owned arrays and zeroes *r (is_open becomes false). Safe on
 * a zero-initialized (never opened) or already-closed reader -- always
 * call this exactly once per successful-or-not m15_open_*() to avoid
 * leaking the owned index arrays on a structurally-valid-but-CRC-failed
 * m15_open_full() (which still populates `out`, per that function's own
 * doc comment). Does not touch `blob` (never owned it). */
void m15_close(M15Reader *r);

/* Whole-tensor CRC32, verified AT MOST ONCE per reader (cached in
 * `r->tensor_verified`) -- repeat calls after a M15_OK are a no-op
 * returning M15_OK immediately, no re-hash (finding 2). */
M15Status m15_verify_tensor_once(M15Reader *r) M15_NODISCARD;

/* Verifies ONLY block `block_idx`'s own CRC32, AT MOST ONCE per reader
 * per block (cached in `r->block_state[block_idx]` -- finding 2): a
 * repeat call after either a cached M15_OK (good) or a cached
 * M15_ERR_BLOCK_CRC_MISMATCH (bad) returns the SAME status immediately,
 * no re-hash either way (see tests/test_mode15_reader.c's verify-once
 * test: corrupting the block's bytes AFTER a cached-good verify does NOT
 * change the second call's answer -- that is the point, not a bug).
 * Requires `r` to be open (M15_ERR_NOT_OPEN otherwise) and
 * block_idx < r->n_blocks (M15_ERR_BAD_ARGUMENT otherwise). */
M15Status m15_verify_block_once(M15Reader *r, uint32_t block_idx) M15_NODISCARD;

/* Raw cached verification state for block_idx (M15_BLOCK_UNVERIFIED/
 * GOOD/BAD) without triggering a verify -- introspection only (an
 * engine-side g_m15_crc_fail_block-style counter can poll this instead of
 * re-deriving it). block_idx must be < r->n_blocks. */
int m15_block_verify_state(const M15Reader *r, uint32_t block_idx);

/* CHECKED row lookup: row `row`'s own compressed bitstream span within
 * `r->payload`. Requires `r` open (M15_ERR_NOT_OPEN) and row < r->O
 * (M15_ERR_BAD_ARGUMENT otherwise, in particular always rejecting every
 * row on a zero-row reader). Does NOT itself verify any CRC -- pair with
 * m15_verify_block_once() for the block `row` belongs to first (see the
 * VERIFICATION MODEL section's intended call shape). Clears *out_ptr/
 * *out_len on entry (finding 8). */
M15Status m15_get_row_span(const M15Reader *r, uint32_t row,
                            const uint8_t **out_ptr, uint32_t *out_len) M15_NODISCARD;

/* UNCHECKED hot-path row lookup (finding 7): no is_open check, no bounds
 * check, no re-validation of anything -- a direct read of the OWNED,
 * pre-decoded (host-native, no per-call little-endian unpack)
 * row_offsets array. For the post-open, per-row inner loop of a decode
 * pass ONLY, where `row` is already known valid (e.g. iterating
 * `0..r->O` or a block's own r0..r1 from m15_block_row_range()) -- use
 * the checked m15_get_row_span() at any boundary where `row` did not
 * just come from this reader's own O/n_blocks. Passing row >= r->O is
 * undefined behavior (one uint32_t OOB read on `row_offsets[row+1]`) --
 * exactly the cost/safety trade this function exists to make available,
 * not a bug. */
static inline void m15_get_row_span_unchecked(const M15Reader *r, uint32_t row,
                                               const uint8_t **out_ptr, uint32_t *out_len){
    uint32_t s0 = r->row_offsets[row];
    uint32_t s1 = r->row_offsets[row + 1];
    *out_ptr = r->payload + s0;
    *out_len = s1 - s0;
}

/* Block b's row range [r0,r1), clamped to O -- the row-index inputs
 * m15_get_row_span[_unchecked]() expects for iterating one block's rows.
 * No validation (block_idx should be < r->n_blocks); pairs with the
 * unchecked accessor above for a fully validation-free hot loop once a
 * block has been verified. */
static inline void m15_block_row_range(const M15Reader *r, uint32_t block_idx, uint32_t *r0, uint32_t *r1){
    uint64_t a = (uint64_t)block_idx * r->rows_per_block;
    uint64_t b = (uint64_t)(block_idx + 1) * r->rows_per_block;
    if(b > r->O) b = r->O;
    *r0 = (uint32_t)a;
    *r1 = (uint32_t)b;
}

/* Decoded accessor for the owned row_offsets array (idx in [0,O]).
 * Equivalent to r->row_offsets[idx] -- kept as a named function (not just
 * direct field access) for symmetry with m15_block_crc32() and so
 * callers outside this translation unit have one obviously-correct way
 * to read it. No bounds checking. */
static inline uint32_t m15_row_offset(const M15Reader *r, uint32_t idx){ return r->row_offsets[idx]; }
static inline uint32_t m15_block_crc32(const M15Reader *r, uint32_t idx){ return r->block_crc32[idx]; }

/* zlib-compatible CRC32 (IEEE 802.3 reflected poly 0xEDB88320, matching
 * mode15_container.py's zlib.crc32() bit-for-bit -- CRC32("123456789")
 * == 0xCBF43926, the standard check value, verified in
 * tests/test_mode15_reader.c). See the THREAT MODEL note above for what
 * this does and does not protect against. */
uint32_t m15_crc32(const uint8_t *data, size_t len);

/* Human-readable status string (errno-style; never NULL). */
const char *m15_strerror(M15Status st);

#ifdef __cplusplus
}
#endif

#endif /* ILIRIA_MODE15_READER_H */
