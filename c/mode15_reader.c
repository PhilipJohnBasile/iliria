/* mode15_reader.c -- implementation of mode15_reader.h. Plain C11 +
 * <stdatomic.h>, zero engine dependencies. See mode15_reader.h's file
 * header for the full design rationale, the LIFETIME CONTRACT (owned
 * index / borrowed payload), the VERIFICATION MODEL (structural vs. lazy
 * vs. full), LENGTH TABLE VALIDITY, and the error-code discipline this
 * file implements.
 *
 * Validation order in m15_open_structural() mirrors
 * mode15_container.py's parse_tensor_blob() step for step (truncated
 * header -> magic -> O/I expectation -> caps -> n_blocks expectation ->
 * truncated lengths -> length-table validity -> truncated row_offsets ->
 * row_offsets copy+monotonicity -> payload-length cap -> truncated
 * block_crc -> total length) so behavior on a given corrupted blob is
 * predictable relative to the Python reference this format is defined
 * by, not just internally self-consistent. Caps are checked as early as
 * each becomes knowable, specifically BEFORE the O(n) work they bound
 * (finding 5): max_O/max_I/max_n_blocks right after the header is parsed
 * (before the row_offsets copy or any block work), max_payload_len right
 * after row_offsets makes payload_len knowable (before the block_crc
 * copy or any CRC hashing).
 */
#include "mode15_reader.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------- little-endian --
 * Explicit byte-order decode, NOT a pointer cast: the blob has no
 * alignment guarantee relative to its own start once staged into an
 * arbitrary-offset buffer (the Mode-15 integration design notes §1a: "arbitrary
 * safetensors sub-page offset") -- casting to `const uint32_t*` and
 * dereferencing would be an unaligned access through a mismatched type
 * (undefined behavior in C). Mirrors tools/mode15_cross_check.c's own
 * read_u32le(). Used only during the one-time open-time parse; once
 * copied into `M15Reader.row_offsets`/`.block_crc32` (host-native
 * uint32_t, no per-access decode needed), this helper is never consulted
 * again for that reader -- see m15_get_row_span_unchecked()'s direct
 * array read in the header. */
static uint32_t m15_rd_u32le(const uint8_t *p){
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

/* ------------------------------------------------------------- CRC32 ---
 * zlib-compatible (IEEE 802.3 reflected, poly 0xEDB88320, init/xorout
 * 0xFFFFFFFF) -- matches mode15_container.py's zlib.crc32() bit-for-bit.
 * See mode15_reader.h's THREAT MODEL note for what this does and does not
 * protect against (corruption, not tampering).
 *
 * Split init/update/final so m15_verify_tensor_once() can feed it the
 * RECONSTRUCTED index bytes (re-serialized from the owned, already-
 * validated row_offsets/block_crc32/lengths -- see that function) and
 * then the borrowed payload as two separate ranges without materializing
 * one giant contiguous buffer spanning both. The 256-entry table is
 * (re)computed fresh on the caller's stack on every m15_crc32_update()
 * call rather than cached in a file-scope static: this can be called
 * concurrently from several of expert_load's call sites (design doc §1b),
 * and a lazily-initialized shared static table would need real
 * synchronization to be race-free; recomputing it (a few hundred cheap
 * shift/xor ops) is immaterial next to the multi-KB-to-MB scan it
 * precedes, and keeps this dependency-free. */
static void m15_crc32_init_state(uint32_t *state){ *state = 0xFFFFFFFFu; }
static void m15_crc32_update(uint32_t *state, const uint8_t *data, size_t len){
    uint32_t table[256];
    for(uint32_t i=0;i<256;i++){
        uint32_t c=i;
        for(int k=0;k<8;k++) c = (c&1u) ? (0xEDB88320u ^ (c>>1)) : (c>>1);
        table[i]=c;
    }
    uint32_t crc = *state;
    for(size_t i=0;i<len;i++) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc>>8);
    *state = crc;
}
static uint32_t m15_crc32_final(uint32_t state){ return state ^ 0xFFFFFFFFu; }

uint32_t m15_crc32(const uint8_t *data, size_t len){
    uint32_t state; m15_crc32_init_state(&state);
    m15_crc32_update(&state, data, len);
    return m15_crc32_final(state);
}

/* --------------------------------------------------- length-table check
 * (finding 3). See mode15_reader.h's LENGTH TABLE VALIDITY section for
 * the full rationale -- this validates EXACTLY the two conventions
 * codec_row_huff.h's huff_canonical_codes()/huff_build() rely on, read
 * from that header, not invented independently:
 *   - n_present==0: always structurally "valid" here (the caller gates
 *     whether an empty table is ACCEPTABLE on O==0||I==0 -- this
 *     function doesn't know O/I).
 *   - n_present==1: codec_row_huff.h's huff_build_lengths() special-cases
 *     a lone symbol to an explicit length of 1 (NOT a Kraft-sum
 *     requirement -- see the single-leaf exception discussion in the
 *     header). Any other length for the sole symbol is invalid.
 *   - n_present>=2: a real Huffman merge always yields a COMPLETE prefix
 *     code -- sum of 2^-len[s] over present symbols equals 1 exactly.
 *     Checked with exact integer arithmetic (all terms are dyadic:
 *     scale by 2^15, HUFF_MAXLEN, so every term is an exact integer). */
static int m15_lengths_valid(const uint8_t lengths[M15_NSYM], uint8_t *out_maxlen, uint8_t *out_n_present){
    int n_present = 0, maxlen = 0, sole_len = -1;
    uint32_t kraft_sum = 0;   /* units of 2^-15 */
    for(int s=0;s<M15_NSYM;s++){
        int L = lengths[s];
        if(L == 0) continue;
        if(L > 15) return 0;             /* HUFF_MAXLEN -- unreachable via the packed 4-bit format today, kept as defense in depth */
        n_present++;
        if(L > maxlen) maxlen = L;
        if(n_present == 1) sole_len = L;
        kraft_sum += (1u << (15 - L));
    }
    if(n_present == 0){ *out_maxlen = 1; *out_n_present = 0; return 1; }
    if(n_present == 1){
        if(sole_len != 1) return 0;
        *out_maxlen = 1; *out_n_present = 1; return 1;
    }
    if(kraft_sum != (1u << 15)) return 0;   /* oversubscribed (>1) or incomplete (<1) */
    *out_maxlen = (uint8_t)maxlen; *out_n_present = (uint8_t)n_present;
    return 1;
}

const char *m15_strerror(M15Status st){
    switch(st){
        case M15_OK:                       return "ok";
        case M15_ERR_TRUNCATED_HEADER:      return "truncated: blob shorter than the 24-byte tensor header";
        case M15_ERR_BAD_MAGIC:             return "bad magic: not a mode-1.5 (\"MH..\") blob";
        case M15_ERR_VERSION_MISMATCH:      return "magic family ok but version suffix unsupported by this reader build";
        case M15_ERR_O_MISMATCH:            return "O (row count) does not match caller's expectation";
        case M15_ERR_I_MISMATCH:            return "I (symbols/row) does not match caller's expectation";
        case M15_ERR_CAP_EXCEEDED:          return "O/I/n_blocks/payload_len exceeds a caller-supplied cap";
        case M15_ERR_BAD_ROWS_PER_BLOCK:    return "rows_per_block is 0 with O>0 (would be division by zero)";
        case M15_ERR_BLOCK_COUNT_MISMATCH:  return "n_blocks does not match ceil(O/rows_per_block)";
        case M15_ERR_TRUNCATED_LENGTHS:     return "truncated: missing the 8-byte canonical-length table";
        case M15_ERR_BAD_LENGTH_TABLE:      return "length table is not a valid canonical-Huffman codebook (see LENGTH TABLE VALIDITY)";
        case M15_ERR_TRUNCATED_ROW_OFFSETS: return "truncated: missing (part of) the row_offsets index";
        case M15_ERR_ROW_OFFSETS_INVALID:   return "row_offsets is not a valid monotonic index (corruption)";
        case M15_ERR_TRUNCATED_BLOCK_CRC:   return "truncated: missing (part of) the block_crc32 table";
        case M15_ERR_LENGTH_MISMATCH:       return "blob length does not match the header-implied total size";
        case M15_ERR_TENSOR_CRC_MISMATCH:   return "whole-tensor CRC32 mismatch (corruption)";
        case M15_ERR_BLOCK_CRC_MISMATCH:    return "a block's CRC32 mismatch (corruption)";
        case M15_ERR_BAD_ARGUMENT:          return "bad argument (NULL pointer or index out of range)";
        case M15_ERR_NOT_OPEN:              return "reader used before open (or after close)";
        case M15_ERR_ALLOC_FAILED:          return "malloc failed for the owned index arrays";
    }
    return "unknown M15Status";
}

M15Status m15_open_structural(const uint8_t *blob, size_t blob_len,
                               int64_t expect_O, int64_t expect_I,
                               const M15Caps *caps,
                               M15Reader *out){
    if(!blob || !out) return M15_ERR_BAD_ARGUMENT;
    memset(out, 0, sizeof(*out));
    out->is_open = 0;

    if(blob_len < M15_TENSOR_HEADER_LEN) return M15_ERR_TRUNCATED_HEADER;

    if(blob[0]!='M' || blob[1]!='H') return M15_ERR_BAD_MAGIC;
    if(blob[2]!='0' || blob[3]!='1') return M15_ERR_VERSION_MISMATCH;

    uint32_t O              = m15_rd_u32le(blob+4);
    uint32_t I              = m15_rd_u32le(blob+8);
    uint32_t rows_per_block = m15_rd_u32le(blob+12);
    uint32_t n_blocks       = m15_rd_u32le(blob+16);
    uint32_t tensor_crc32   = m15_rd_u32le(blob+20);

    if(expect_O >= 0 && (int64_t)O != expect_O) return M15_ERR_O_MISMATCH;
    if(expect_I >= 0 && (int64_t)I != expect_I) return M15_ERR_I_MISMATCH;

    /* CAPS (finding 5), as early as each field is known -- BEFORE any
     * O(n) work (the row_offsets copy below is O(O), gated by max_O/
     * max_n_blocks here; the eventual block_crc copy is O(n_blocks),
     * same gate). */
    if(caps){
        if(caps->max_O && O > caps->max_O) return M15_ERR_CAP_EXCEEDED;
        if(caps->max_I && I > caps->max_I) return M15_ERR_CAP_EXCEEDED;
        if(caps->max_n_blocks && n_blocks > caps->max_n_blocks) return M15_ERR_CAP_EXCEEDED;
    }

    uint32_t expect_n_blocks;
    if(O == 0){
        expect_n_blocks = 0;   /* matches Python's `... if O > 0 else 0`: rows_per_block is
                                 * never consulted (and so never divided by) when O==0 */
    } else {
        if(rows_per_block == 0) return M15_ERR_BAD_ROWS_PER_BLOCK;
        expect_n_blocks = (uint32_t)(((uint64_t)O + rows_per_block - 1) / rows_per_block);
    }
    if(n_blocks != expect_n_blocks) return M15_ERR_BLOCK_COUNT_MISMATCH;

    uint64_t off = M15_TENSOR_HEADER_LEN;

    if((uint64_t)blob_len < off + M15_LENGTHS_BYTES) return M15_ERR_TRUNCATED_LENGTHS;
    uint8_t lengths[M15_NSYM];
    for(int k=0;k<M15_LENGTHS_BYTES;k++){
        uint8_t b = blob[off+(size_t)k];
        lengths[2*k]   = (uint8_t)(b & 0xF);
        lengths[2*k+1] = (uint8_t)(b >> 4);
    }
    off += M15_LENGTHS_BYTES;

    uint8_t maxlen=1, n_present=0;
    if(!m15_lengths_valid(lengths, &maxlen, &n_present)) return M15_ERR_BAD_LENGTH_TABLE;
    if(n_present == 0 && O != 0 && I != 0) return M15_ERR_BAD_LENGTH_TABLE;  /* empty table but rows exist that would need it */

    uint64_t ro_bytes = ((uint64_t)O + 1) * 4;
    if((uint64_t)blob_len < off + ro_bytes) return M15_ERR_TRUNCATED_ROW_OFFSETS;
    const uint8_t *row_offsets_src = blob + off;
    off += ro_bytes;

    /* OWNED copy + monotonicity validation, one pass (finding 1's
     * ownership + the pre-existing row_offsets hardening, done together
     * since both need the same per-entry decode). After this point,
     * row_offsets no longer depends on `blob` at all. */
    uint32_t *row_offsets = (uint32_t*)malloc(((size_t)O + 1) * sizeof(uint32_t));
    if(!row_offsets) return M15_ERR_ALLOC_FAILED;
    {
        uint32_t prev = 0;
        int bad = 0;
        for(uint32_t r=0; r<=O; r++){
            uint32_t cur = m15_rd_u32le(row_offsets_src + (size_t)r*4);
            if(r>0 && cur<prev) bad = 1;
            row_offsets[r] = cur;
            prev = cur;
        }
        if(row_offsets[0] != 0) bad = 1;
        if(bad){ free(row_offsets); return M15_ERR_ROW_OFFSETS_INVALID; }
    }

    uint32_t payload_len = row_offsets[O];
    if(caps && caps->max_payload_len && (uint64_t)payload_len > caps->max_payload_len){
        free(row_offsets); return M15_ERR_CAP_EXCEEDED;
    }

    uint64_t bc_bytes = (uint64_t)n_blocks * 4;
    if((uint64_t)blob_len < off + bc_bytes){ free(row_offsets); return M15_ERR_TRUNCATED_BLOCK_CRC; }
    const uint8_t *block_crc_src = blob + off;
    off += bc_bytes;

    uint64_t expected_total = off + (uint64_t)payload_len;
    if((uint64_t)blob_len != expected_total){ free(row_offsets); return M15_ERR_LENGTH_MISMATCH; }
    const uint8_t *payload = blob + off;

    uint32_t *block_crc = NULL;
    if(n_blocks > 0){
        block_crc = (uint32_t*)malloc((size_t)n_blocks * sizeof(uint32_t));
        if(!block_crc){ free(row_offsets); return M15_ERR_ALLOC_FAILED; }
        for(uint32_t b=0;b<n_blocks;b++) block_crc[b] = m15_rd_u32le(block_crc_src + (size_t)b*4);
    }

    atomic_uchar *block_state = NULL;
    if(n_blocks > 0){
        block_state = (atomic_uchar*)malloc((size_t)n_blocks * sizeof(atomic_uchar));
        if(!block_state){ free(row_offsets); free(block_crc); return M15_ERR_ALLOC_FAILED; }
        for(uint32_t b=0;b<n_blocks;b++) atomic_init(&block_state[b], (unsigned char)M15_BLOCK_UNVERIFIED);
    }

    out->O = O; out->I = I; out->rows_per_block = rows_per_block; out->n_blocks = n_blocks;
    out->tensor_crc32 = tensor_crc32;
    memcpy(out->lengths, lengths, sizeof(lengths));
    out->maxlen = maxlen; out->n_present = n_present;
    out->row_offsets = row_offsets;
    out->block_crc32 = block_crc;
    out->block_state = block_state;
    out->tensor_verified = 0;
    out->payload = payload;
    out->payload_len = payload_len;
    out->is_open = 1;
    return M15_OK;
}

M15Status m15_open_full(const uint8_t *blob, size_t blob_len,
                         int64_t expect_O, int64_t expect_I,
                         const M15Caps *caps,
                         M15Reader *out,
                         uint32_t *out_bad_block){
    M15Status st = m15_open_structural(blob, blob_len, expect_O, expect_I, caps, out);
    if(st != M15_OK) return st;

    M15Status vt = m15_verify_tensor_once(out);
    if(vt != M15_OK) return vt;   /* `out` remains open (per this function's own doc comment) -- caller should still m15_close() it */

    for(uint32_t b=0; b<out->n_blocks; b++){
        M15Status vb = m15_verify_block_once(out, b);
        if(vb != M15_OK){
            if(out_bad_block) *out_bad_block = b;
            return vb;
        }
    }
    return M15_OK;
}

void m15_close(M15Reader *r){
    if(!r) return;
    free(r->row_offsets);
    free(r->block_crc32);
    free((void*)r->block_state);
    memset(r, 0, sizeof(*r));
}

/* Reconstructs the exact original whole-tensor CRC32 input
 * (lengths_bytes + row_offsets_bytes + block_crc_bytes, all
 * re-serialized LOSSLESSLY from the OWNED, already-validated fields --
 * NOT from `blob`, which this reader never keeps a pointer to beyond
 * m15_open_*()'s own call -- plus the borrowed `payload`) so this can be
 * called long after open, independent of whether the original header
 * bytes are still around (finding 1's ownership contract: only payload
 * needs to still be alive). See mode15_reader.h's VERIFICATION MODEL. */
M15Status m15_verify_tensor_once(M15Reader *r){
    if(!r || !r->is_open) return M15_ERR_NOT_OPEN;
    if(r->tensor_verified) return M15_OK;

    size_t idx_bytes = (size_t)M15_LENGTHS_BYTES + ((size_t)r->O + 1)*4 + (size_t)r->n_blocks*4;
    uint8_t *idx = (uint8_t*)malloc(idx_bytes);
    if(!idx) return M15_ERR_ALLOC_FAILED;

    size_t p = 0;
    for(int k=0;k<M15_LENGTHS_BYTES;k++){
        uint8_t lo = (uint8_t)(r->lengths[2*k] & 0xF), hi = (uint8_t)(r->lengths[2*k+1] & 0xF);
        idx[p++] = (uint8_t)(lo | (hi<<4));
    }
    for(uint32_t i=0;i<=r->O;i++){
        uint32_t v = r->row_offsets[i];
        idx[p++]=(uint8_t)(v&0xFF); idx[p++]=(uint8_t)((v>>8)&0xFF);
        idx[p++]=(uint8_t)((v>>16)&0xFF); idx[p++]=(uint8_t)((v>>24)&0xFF);
    }
    for(uint32_t b=0;b<r->n_blocks;b++){
        uint32_t v = r->block_crc32[b];
        idx[p++]=(uint8_t)(v&0xFF); idx[p++]=(uint8_t)((v>>8)&0xFF);
        idx[p++]=(uint8_t)((v>>16)&0xFF); idx[p++]=(uint8_t)((v>>24)&0xFF);
    }

    uint32_t state; m15_crc32_init_state(&state);
    m15_crc32_update(&state, idx, p);
    m15_crc32_update(&state, r->payload, r->payload_len);
    uint32_t got = m15_crc32_final(state);
    free(idx);

    if(got != r->tensor_crc32) return M15_ERR_TENSOR_CRC_MISMATCH;
    r->tensor_verified = 1;
    return M15_OK;
}

M15Status m15_verify_block_once(M15Reader *r, uint32_t block_idx){
    if(!r || !r->is_open) return M15_ERR_NOT_OPEN;
    if(block_idx >= r->n_blocks) return M15_ERR_BAD_ARGUMENT;

    unsigned char cur = atomic_load_explicit(&r->block_state[block_idx], memory_order_relaxed);
    if(cur == M15_BLOCK_GOOD) return M15_OK;
    if(cur == M15_BLOCK_BAD) return M15_ERR_BLOCK_CRC_MISMATCH;

    uint32_t r0, r1;
    m15_block_row_range(r, block_idx, &r0, &r1);
    uint32_t s0 = r->row_offsets[r0];
    uint32_t s1 = r->row_offsets[r1];
    uint32_t got = m15_crc32(r->payload + s0, (size_t)(s1 - s0));
    uint32_t want = r->block_crc32[block_idx];

    if(got == want){
        atomic_store_explicit(&r->block_state[block_idx], (unsigned char)M15_BLOCK_GOOD, memory_order_relaxed);
        return M15_OK;
    }
    atomic_store_explicit(&r->block_state[block_idx], (unsigned char)M15_BLOCK_BAD, memory_order_relaxed);
    return M15_ERR_BLOCK_CRC_MISMATCH;
}

int m15_block_verify_state(const M15Reader *r, uint32_t block_idx){
    if(!r || !r->is_open || block_idx >= r->n_blocks) return M15_BLOCK_UNVERIFIED;
    return atomic_load_explicit(&r->block_state[block_idx], memory_order_relaxed);
}

M15Status m15_get_row_span(const M15Reader *r, uint32_t row,
                            const uint8_t **out_ptr, uint32_t *out_len){
    if(out_ptr) *out_ptr = NULL;
    if(out_len) *out_len = 0;
    if(!r || !out_ptr || !out_len) return M15_ERR_BAD_ARGUMENT;
    if(!r->is_open) return M15_ERR_NOT_OPEN;
    if(row >= r->O) return M15_ERR_BAD_ARGUMENT;
    m15_get_row_span_unchecked(r, row, out_ptr, out_len);
    return M15_OK;
}
