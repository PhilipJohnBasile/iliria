/* tests/test_mode15_reader.c -- unit tests for mode15_reader.h/.c against
 * SYNTHETIC mode-1.5 blobs (tests/gen_mode15_reader_fixtures.py, built on
 * tools/mode15_container.py's real encoder). Rewritten for the 2026-07-18
 * hardening round (external review, 8 findings) -- see
 * mode15_reader.h's file header for the full rationale of every behavior
 * tested here.
 *
 * Layers (cheapest first, same shape as tests/test_codec_row_huff.c):
 *   1. happy path (structural + full open), including end-to-end decode
 *      through the real codec_row_huff.h functions.
 *   2. zero-row edge.
 *   3. truncation.
 *   4. index-copy lifetime (finding 1): free the source buffer after
 *      open, index accessors still agree.
 *   5. verify-once bitmap semantics (finding 2): both cached-good and
 *      cached-bad are sticky (don't re-hash on a later call even if the
 *      underlying bytes change out from under a cached-good verdict --
 *      that staleness IS the point of "verify exactly once", not a bug).
 *   6. structural-vs-full open behavior (finding 6): a corrupted payload
 *      byte passes m15_open_structural (never hashes payload) but fails
 *      m15_open_full / an explicit m15_verify_tensor_once.
 *   7. whole-tensor + per-block CRC corruption/localization.
 *   8. version vs. magic-family mismatch.
 *   9. row_offsets monotonicity hardening.
 *  10. length-table validity (finding 3): all-zero on a non-degenerate
 *      tensor, Kraft-oversubscribed, Kraft-incomplete, wrong
 *      single-symbol length, and the valid single-symbol positive case.
 *  11. caps rejection (finding 5): each of max_O/max_I/max_n_blocks/
 *      max_payload_len independently, plus a generous-caps positive case.
 *  12. O/I mismatch, CRC32 standard check value, programmer-error class.
 *
 * Fixtures come from argv[1] (see tools/build_mode15_sidecar.sh). No
 * engine dependency: only mode15_reader.h + codec_row.h/codec_row_huff.h
 * (the latter two ONLY for test 1's end-to-end decode proof, exactly like
 * mode15_cross_check.c -- mode15_reader.c itself never includes them).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../mode15_reader.h"
#include "../codec_row.h"
#include "../codec_row_huff.h"

static int g_fails = 0;
static void fail(const char *msg){ fprintf(stderr, "test_mode15_reader FAILED: %s\n", msg); g_fails++; }
#define CHECK(cond, msg) do{ if(!(cond)) fail(msg); }while(0)

/* must match tests/gen_mode15_reader_fixtures.py's HAPPY_* constants */
#define ROWS_PER_BLOCK 8
#define HAPPY_O 37
#define HAPPY_I 64
#define HAPPY_N_BLOCKS 5

static uint8_t *slurp(const char *path, size_t *len_out){
    FILE *f = fopen(path, "rb");
    if(!f){ fprintf(stderr, "cannot open fixture: %s\n", path); exit(2); }
    if(fseek(f, 0, SEEK_END)!=0){ fprintf(stderr, "fseek failed: %s\n", path); exit(2); }
    long sz = ftell(f);
    if(sz < 0 || fseek(f, 0, SEEK_SET)!=0){ fprintf(stderr, "ftell/fseek failed: %s\n", path); exit(2); }
    uint8_t *buf = (uint8_t*)malloc((size_t)sz > 0 ? (size_t)sz : 1);
    if(sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz){ fprintf(stderr, "short read: %s\n", path); exit(2); }
    fclose(f);
    *len_out = (size_t)sz;
    return buf;
}
static uint8_t *clone(const uint8_t *src, size_t len){
    uint8_t *c = (uint8_t*)malloc(len > 0 ? len : 1);
    memcpy(c, src, len);
    return c;
}

/* fixed offsets within any tensor blob */
#define LENGTHS_OFF (M15_TENSOR_HEADER_LEN)
#define ROW_OFFSETS_OFF (M15_TENSOR_HEADER_LEN + M15_LENGTHS_BYTES)

/* faithful copy of codec_row_huff.h's huff_build() LUT-fill loop only
 * (not the length-choice/canonical-order logic, which comes from the
 * REAL huff_canonical_codes() below) -- same gap tools/mode15_cross_
 * check.c's own build_lut_from_lengths() exists to bridge. */
static void build_lut_from_lengths(HuffCodebook *cb){
    int maxlen=0, npres=0;
    for(int s=0;s<CODEC_NSYM;s++){ if(cb->len[s]>0){ npres++; if(cb->len[s]>maxlen) maxlen=cb->len[s]; } }
    cb->n_present = npres;
    cb->maxlen = maxlen>0 ? maxlen : 1;
    size_t lutn = (size_t)1u<<cb->maxlen;
    cb->lut = (HuffLutEnt*)malloc(lutn*sizeof(HuffLutEnt));
    memset(cb->lut, 0, lutn*sizeof(HuffLutEnt));
    if(npres==1){
        uint8_t s=0; for(int t=0;t<CODEC_NSYM;t++) if(cb->len[t]>0) s=(uint8_t)t;
        for(size_t i=0;i<lutn;i++){ cb->lut[i].sym=s; cb->lut[i].len=1; }
        return;
    }
    for(int s=0;s<CODEC_NSYM;s++){
        if(cb->len[s]==0) continue;
        int L=cb->len[s]; uint32_t base=cb->code_rev[s]; int rest=cb->maxlen-L;
        size_t block=(size_t)1u<<rest;
        for(size_t hi=0; hi<block; hi++){ size_t idx=base|(hi<<L); cb->lut[idx].sym=(uint8_t)s; cb->lut[idx].len=(uint8_t)L; }
    }
}

/* --------------------------------------------------------------- tests */

static void test_happy_path(const uint8_t *happy, size_t happy_len, const uint8_t *orig, size_t orig_len){
    M15Reader r;
    M15Status st = m15_open_structural(happy, happy_len, -1, -1, NULL, &r);
    CHECK(st==M15_OK, "happy path: m15_open_structural should return M15_OK");
    if(st != M15_OK){ fprintf(stderr, "  status=%s\n", m15_strerror(st)); return; }

    CHECK(r.O==HAPPY_O, "happy path: O should be 37");
    CHECK(r.I==HAPPY_I, "happy path: I should be 64");
    CHECK(r.rows_per_block==ROWS_PER_BLOCK, "happy path: rows_per_block should be 8");
    CHECK(r.n_blocks==HAPPY_N_BLOCKS, "happy path: n_blocks should be ceil(37/8)=5");
    CHECK(orig_len == (size_t)r.O*r.I, "happy path: .orig size should be O*I");
    CHECK(r.tensor_verified==0, "happy path (structural): tensor must NOT be verified yet");
    for(uint32_t b=0;b<r.n_blocks;b++)
        CHECK(m15_block_verify_state(&r,b)==M15_BLOCK_UNVERIFIED, "happy path (structural): every block starts UNVERIFIED");

    for(uint32_t b=0;b<r.n_blocks;b++)
        CHECK(m15_verify_block_once(&r,b)==M15_OK, "happy path: every block's CRC32 must verify clean");

    HuffCodebook cb; memset(&cb, 0, sizeof(cb));
    memcpy(cb.len, r.lengths, CODEC_NSYM);
    huff_canonical_codes(cb.len, cb.code_rev);
    build_lut_from_lengths(&cb);

    uint8_t decoded[HAPPY_I];
    size_t mismatch_rows = 0;
    for(uint32_t row=0; row<r.O; row++){
        const uint8_t *span_ptr=NULL, *span_ptr2=NULL; uint32_t span_len=0, span_len2=0;
        CHECK(m15_get_row_span(&r, row, &span_ptr, &span_len)==M15_OK, "happy path: m15_get_row_span failed on a valid row");
        m15_get_row_span_unchecked(&r, row, &span_ptr2, &span_len2);
        CHECK(span_ptr==span_ptr2 && span_len==span_len2, "happy path: checked and unchecked row-span accessors must agree");
        huff_decode_row(&cb, span_ptr, span_len, (int)r.I, decoded);
        if(memcmp(decoded, orig + (size_t)row*r.I, r.I) != 0) mismatch_rows++;
    }
    CHECK(mismatch_rows==0, "happy path: every row must decode bit-exact to the original nibbles");

    { const uint8_t *p; uint32_t l;
      CHECK(m15_get_row_span(&r, r.O, &p, &l)==M15_ERR_BAD_ARGUMENT, "happy path: row==O must be rejected as out of range"); }

    huff_free(&cb);
    m15_close(&r);
    CHECK(r.is_open==0, "happy path: m15_close must clear is_open");
}

static void test_open_full(const uint8_t *happy, size_t happy_len){
    M15Reader r; uint32_t bad_block = 0xFFFFFFFFu;
    M15Status st = m15_open_full(happy, happy_len, HAPPY_O, HAPPY_I, NULL, &r, &bad_block);
    CHECK(st==M15_OK, "open_full: clean fixture must fully verify OK");
    if(st==M15_OK){
        CHECK(r.tensor_verified==1, "open_full: tensor_verified must be set");
        for(uint32_t b=0;b<r.n_blocks;b++)
            CHECK(m15_block_verify_state(&r,b)==M15_BLOCK_GOOD, "open_full: every block must be cached GOOD");
        m15_close(&r);
    }
}

static void test_zero_row(const char *fixdir){
    char blob_path[1024];
    snprintf(blob_path, sizeof(blob_path), "%s/zero_row_I64.bin", fixdir);
    size_t blob_len; uint8_t *blob = slurp(blob_path, &blob_len);

    M15Reader r;
    M15Status st = m15_open_structural(blob, blob_len, -1, -1, NULL, &r);
    CHECK(st==M15_OK, "zero-row: should parse cleanly");
    if(st==M15_OK){
        CHECK(r.O==0, "zero-row: O should be 0");
        CHECK(r.n_blocks==0, "zero-row: n_blocks should be 0");
        CHECK(r.payload_len==0, "zero-row: payload_len should be 0");
        const uint8_t *p; uint32_t l;
        CHECK(m15_get_row_span(&r, 0, &p, &l)==M15_ERR_BAD_ARGUMENT, "zero-row: row 0 must be rejected (no valid rows exist)");
        CHECK(m15_verify_tensor_once(&r)==M15_OK, "zero-row: whole-tensor CRC (over an empty body) must still verify");
        m15_close(&r);
    }
    free(blob);
}

static void test_truncation(const uint8_t *happy, size_t happy_len){
    size_t cuts[] = { 0, 1, M15_TENSOR_HEADER_LEN, happy_len/2, happy_len-1 };
    for(size_t i=0;i<sizeof(cuts)/sizeof(cuts[0]);i++){
        M15Reader r;
        M15Status st = m15_open_structural(happy, cuts[i], -1, -1, NULL, &r);
        if(st == M15_OK){ fail("truncation: a cut-down blob must never parse as M15_OK"); m15_close(&r); }
    }
}

static void test_index_copy_lifetime(const uint8_t *happy, size_t happy_len){
    /* finding 1: the index must be independent of the source buffer.
     * malloc a private copy, open against it, FREE that copy, and check
     * every owned/copied field via the reader's own accessors -- never
     * touching r.payload (which is still borrowed, and would legitimately
     * dangle after this free; that is documented, expected, and not what
     * this test is about). */
    uint8_t *src = clone(happy, happy_len);
    M15Reader r;
    M15Status st = m15_open_structural(src, happy_len, -1, -1, NULL, &r);
    CHECK(st==M15_OK, "index-copy lifetime: open must succeed before the free");
    if(st != M15_OK){ free(src); return; }

    free(src);   /* the whole source buffer, header+index+payload, is gone now */

    CHECK(r.O==HAPPY_O && r.I==HAPPY_I && r.rows_per_block==ROWS_PER_BLOCK && r.n_blocks==HAPPY_N_BLOCKS,
          "index-copy lifetime: scalar header fields must survive freeing the source");
    CHECK(m15_row_offset(&r, 0)==0, "index-copy lifetime: row_offsets[0] must survive freeing the source");
    for(uint32_t row=0; row<=r.O; row++){
        /* just touching every copied entry is the point: if row_offsets
         * secretly still aliased the freed `src`, ASan would flag this
         * read as a use-after-free. */
        (void)m15_row_offset(&r, row);
    }
    for(uint32_t b=0;b<r.n_blocks;b++) (void)m15_block_crc32(&r, b);

    /* r.payload is now dangling by design -- do not dereference it.
     * m15_close() only frees OWNED arrays, never touches payload. */
    m15_close(&r);
}

static void test_verify_once_caching(const uint8_t *happy, size_t happy_len){
    /* (a) cached-GOOD is sticky: verify block 0 clean, THEN corrupt its
     * bytes, THEN verify again -- must still report M15_OK (stale, by
     * design: "verifies exactly once" per finding 2, not "verifies every
     * time and happens to agree"). */
    {
        uint8_t *blob = clone(happy, happy_len);
        M15Reader r;
        CHECK(m15_open_structural(blob, happy_len, -1, -1, NULL, &r)==M15_OK, "verify-once(good): open must succeed");
        CHECK(m15_verify_block_once(&r, 0)==M15_OK, "verify-once(good): first verify of block 0 must pass");
        CHECK(m15_block_verify_state(&r,0)==M15_BLOCK_GOOD, "verify-once(good): state must be cached GOOD");

        uint32_t r0,r1; m15_block_row_range(&r, 0, &r0, &r1);
        uint32_t s0 = m15_row_offset(&r, r0);
        ((uint8_t*)r.payload)[s0] ^= 0xFF;   /* corrupt one payload byte inside block 0's own range */

        CHECK(m15_verify_block_once(&r, 0)==M15_OK,
              "verify-once(good): second call must return the CACHED verdict (stale-good), not re-hash");
        m15_close(&r);
        free(blob);
    }
    /* (b) cached-BAD is sticky: corrupt first, verify (fails), REPAIR the
     * bytes, verify again -- must still report the cached failure. */
    {
        uint8_t *blob = clone(happy, happy_len);
        M15Reader r;
        CHECK(m15_open_structural(blob, happy_len, -1, -1, NULL, &r)==M15_OK, "verify-once(bad): open must succeed");

        uint32_t r0,r1; m15_block_row_range(&r, 0, &r0, &r1);
        uint32_t s0 = m15_row_offset(&r, r0);
        uint8_t original_byte = ((uint8_t*)r.payload)[s0];
        ((uint8_t*)r.payload)[s0] ^= 0xFF;

        CHECK(m15_verify_block_once(&r, 0)==M15_ERR_BLOCK_CRC_MISMATCH, "verify-once(bad): first verify must catch the corruption");
        CHECK(m15_block_verify_state(&r,0)==M15_BLOCK_BAD, "verify-once(bad): state must be cached BAD");

        ((uint8_t*)r.payload)[s0] = original_byte;   /* repair -- bytes are genuinely correct again now */
        CHECK(m15_verify_block_once(&r, 0)==M15_ERR_BLOCK_CRC_MISMATCH,
              "verify-once(bad): second call must return the CACHED verdict (stale-bad), not re-hash");
        m15_close(&r);
        free(blob);
    }
}

static void test_structural_vs_full(const uint8_t *happy, size_t happy_len){
    uint8_t *corrupt = clone(happy, happy_len);
    corrupt[happy_len - 1] ^= 0xFF;   /* last byte of blob == last byte of the final row's bitstream */

    M15Reader r;
    CHECK(m15_open_structural(corrupt, happy_len, -1, -1, NULL, &r)==M15_OK,
          "structural-vs-full: m15_open_structural must succeed even with a corrupted payload byte (it never hashes payload)");
    CHECK(m15_verify_tensor_once(&r)==M15_ERR_TENSOR_CRC_MISMATCH,
          "structural-vs-full: an explicit m15_verify_tensor_once() must catch the corruption");
    m15_close(&r);

    M15Reader r2; uint32_t bad_block=0xFFFFFFFFu;
    M15Status st2 = m15_open_full(corrupt, happy_len, -1, -1, NULL, &r2, &bad_block);
    CHECK(st2==M15_ERR_TENSOR_CRC_MISMATCH, "structural-vs-full: m15_open_full must fail on the corrupted payload byte");
    m15_close(&r2);

    free(corrupt);
}

static void test_block_crc_localization(const uint8_t *happy, size_t happy_len){
    uint8_t *corrupt = clone(happy, happy_len);
    corrupt[happy_len - 1] ^= 0xFF;   /* lands in the last row -> last block */

    M15Reader r;
    CHECK(m15_open_structural(corrupt, happy_len, -1, -1, NULL, &r)==M15_OK, "block localization: structural open must succeed");

    int n_bad = 0, bad_idx = -1;
    for(uint32_t b=0; b<r.n_blocks; b++){
        if(m15_verify_block_once(&r, b) != M15_OK){ n_bad++; bad_idx=(int)b; }
    }
    CHECK(n_bad==1, "block localization: exactly one block's CRC32 must fail");
    CHECK(bad_idx==(int)(r.n_blocks-1), "block localization: the corrupted block must be the last one");

    m15_close(&r);
    free(corrupt);
}

static void test_version_and_magic(const uint8_t *happy, size_t happy_len){
    uint8_t *bad_magic = clone(happy, happy_len);
    bad_magic[0]='X'; bad_magic[1]='X';
    M15Reader r;
    CHECK(m15_open_structural(bad_magic, happy_len, -1, -1, NULL, &r)==M15_ERR_BAD_MAGIC, "bad magic family must be M15_ERR_BAD_MAGIC");
    free(bad_magic);

    uint8_t *bad_version = clone(happy, happy_len);
    bad_version[2]='0'; bad_version[3]='2';
    CHECK(m15_open_structural(bad_version, happy_len, -1, -1, NULL, &r)==M15_ERR_VERSION_MISMATCH,
          "recognized family + unsupported version suffix must be M15_ERR_VERSION_MISMATCH");
    free(bad_version);
}

static void test_row_offsets_hardening(const uint8_t *happy, size_t happy_len){
    uint8_t *corrupt = clone(happy, happy_len);
    size_t ro_off = ROW_OFFSETS_OFF;
    corrupt[ro_off + 2*4 + 0]=0; corrupt[ro_off + 2*4 + 1]=0;
    corrupt[ro_off + 2*4 + 2]=0; corrupt[ro_off + 2*4 + 3]=0;

    M15Reader r;
    CHECK(m15_open_structural(corrupt, happy_len, -1, -1, NULL, &r)==M15_ERR_ROW_OFFSETS_INVALID,
          "row_offsets hardening: a decreasing entry must be M15_ERR_ROW_OFFSETS_INVALID");
    free(corrupt);
}

static void test_o_i_mismatch(const uint8_t *happy, size_t happy_len){
    M15Reader r;
    CHECK(m15_open_structural(happy, happy_len, 999, -1, NULL, &r)==M15_ERR_O_MISMATCH, "O mismatch must be caught");
    CHECK(m15_open_structural(happy, happy_len, -1, 999, NULL, &r)==M15_ERR_I_MISMATCH, "I mismatch must be caught");
    M15Status ok = m15_open_structural(happy, happy_len, HAPPY_O, HAPPY_I, NULL, &r);
    CHECK(ok==M15_OK, "matching expect_O/expect_I must still succeed");
    if(ok==M15_OK) m15_close(&r);
}

static void test_crc32_check_value(void){
    uint32_t v = m15_crc32((const uint8_t*)"123456789", 9);
    CHECK(v==0xCBF43926u, "m15_crc32: standard check value for \"123456789\" must be 0xCBF43926");
}

static void test_caps_rejection(const uint8_t *happy, size_t happy_len){
    M15Reader r;
    M15Caps caps;

    caps = (M15Caps){0}; caps.max_O = HAPPY_O - 1;
    CHECK(m15_open_structural(happy, happy_len, -1, -1, &caps, &r)==M15_ERR_CAP_EXCEEDED, "caps: max_O below the real O must reject");

    caps = (M15Caps){0}; caps.max_I = HAPPY_I - 1;
    CHECK(m15_open_structural(happy, happy_len, -1, -1, &caps, &r)==M15_ERR_CAP_EXCEEDED, "caps: max_I below the real I must reject");

    caps = (M15Caps){0}; caps.max_n_blocks = HAPPY_N_BLOCKS - 1;
    CHECK(m15_open_structural(happy, happy_len, -1, -1, &caps, &r)==M15_ERR_CAP_EXCEEDED, "caps: max_n_blocks below the real n_blocks must reject");

    caps = (M15Caps){0}; caps.max_payload_len = 1;
    CHECK(m15_open_structural(happy, happy_len, -1, -1, &caps, &r)==M15_ERR_CAP_EXCEEDED, "caps: max_payload_len of 1 must reject a real payload");

    caps = (M15Caps){0}; caps.max_O=HAPPY_O; caps.max_I=HAPPY_I; caps.max_n_blocks=HAPPY_N_BLOCKS; caps.max_payload_len=1u<<20;
    M15Status ok = m15_open_structural(happy, happy_len, -1, -1, &caps, &r);
    CHECK(ok==M15_OK, "caps: generous (exact-fit) caps must still succeed");
    if(ok==M15_OK) m15_close(&r);
}

/* ------------------------------------------------- length-table tests
 * (finding 3). All of these patch ONLY the 8-byte lengths field on a
 * clone of the happy fixture: length-table validity is independent of
 * row_offsets/payload consistency (m15_open_structural never cross-
 * checks the two), so this is a safe, minimal way to construct each
 * scenario without hand-building a whole new blob. */
static void patch_lengths(uint8_t *blob, const uint8_t lengths[16]){
    for(int k=0;k<M15_LENGTHS_BYTES;k++){
        uint8_t lo = lengths[2*k] & 0xF, hi = lengths[2*k+1] & 0xF;
        blob[LENGTHS_OFF + k] = (uint8_t)(lo | (hi<<4));
    }
}

static void test_length_table_all_zero(const uint8_t *happy, size_t happy_len){
    uint8_t *blob = clone(happy, happy_len);
    uint8_t lengths[16] = {0};
    patch_lengths(blob, lengths);
    M15Reader r;
    CHECK(m15_open_structural(blob, happy_len, -1, -1, NULL, &r)==M15_ERR_BAD_LENGTH_TABLE,
          "length table: all-zero on a non-degenerate (O>0,I>0) tensor must be rejected");
    free(blob);
}

static void test_length_table_oversubscribed(const uint8_t *happy, size_t happy_len){
    uint8_t *blob = clone(happy, happy_len);
    uint8_t lengths[16] = {0};
    lengths[0]=1; lengths[1]=1; lengths[2]=1;   /* three symbols at length 1: only 2 codewords exist at length 1 */
    patch_lengths(blob, lengths);
    M15Reader r;
    CHECK(m15_open_structural(blob, happy_len, -1, -1, NULL, &r)==M15_ERR_BAD_LENGTH_TABLE,
          "length table: Kraft-oversubscribed (3 symbols at length 1) must be rejected");
    free(blob);
}

static void test_length_table_incomplete(const uint8_t *happy, size_t happy_len){
    uint8_t *blob = clone(happy, happy_len);
    uint8_t lengths[16] = {0};
    lengths[0]=2; lengths[1]=2;   /* Kraft sum = 2^-2+2^-2 = 0.5 != 1: valid-but-incomplete, not a real Huffman output */
    patch_lengths(blob, lengths);
    M15Reader r;
    CHECK(m15_open_structural(blob, happy_len, -1, -1, NULL, &r)==M15_ERR_BAD_LENGTH_TABLE,
          "length table: Kraft-incomplete (2 symbols at length 2, sum=0.5) must be rejected");
    free(blob);
}

static void test_length_table_single_symbol(const uint8_t *happy, size_t happy_len){
    /* negative: wrong length for the sole present symbol */
    {
        uint8_t *blob = clone(happy, happy_len);
        uint8_t lengths[16] = {0};
        lengths[5] = 3;   /* codec_row_huff.h's single-symbol convention requires length 1, not 3 */
        patch_lengths(blob, lengths);
        M15Reader r;
        CHECK(m15_open_structural(blob, happy_len, -1, -1, NULL, &r)==M15_ERR_BAD_LENGTH_TABLE,
              "length table: a lone present symbol at length != 1 must be rejected");
        free(blob);
    }
    /* positive: correct single-symbol table must be accepted */
    {
        uint8_t *blob = clone(happy, happy_len);
        uint8_t lengths[16] = {0};
        lengths[5] = 1;
        patch_lengths(blob, lengths);
        M15Reader r;
        M15Status st = m15_open_structural(blob, happy_len, -1, -1, NULL, &r);
        CHECK(st==M15_OK, "length table: a lone present symbol at length 1 must be ACCEPTED");
        if(st==M15_OK){
            CHECK(r.n_present==1, "length table: n_present must be 1");
            CHECK(r.maxlen==1, "length table: maxlen must be 1");
            CHECK(r.lengths[5]==1, "length table: symbol 5's length must round-trip as 1");
            m15_close(&r);
        }
        free(blob);
    }
}

/* ------------------------------------------------------------ programmer-error class */
static void test_programmer_error_classification(void){
    CHECK(m15_is_programmer_error(M15_ERR_BAD_ARGUMENT), "BAD_ARGUMENT must classify as programmer error");
    CHECK(m15_is_programmer_error(M15_ERR_NOT_OPEN), "NOT_OPEN must classify as programmer error");
    CHECK(m15_is_programmer_error(M15_ERR_ALLOC_FAILED), "ALLOC_FAILED must classify as programmer error (propagate, don't try the mirror)");
    CHECK(!m15_is_programmer_error(M15_OK), "M15_OK must not classify as programmer error");
    CHECK(!m15_is_programmer_error(M15_ERR_TENSOR_CRC_MISMATCH), "TENSOR_CRC_MISMATCH must classify as corrupt-data, not programmer error");
    CHECK(!m15_is_programmer_error(M15_ERR_BAD_LENGTH_TABLE), "BAD_LENGTH_TABLE must classify as corrupt-data, not programmer error");
    CHECK(!m15_is_programmer_error(M15_ERR_CAP_EXCEEDED), "CAP_EXCEEDED must classify as corrupt-data, not programmer error");

    M15Reader r; memset(&r, 0, sizeof(r));
    const uint8_t *p; uint32_t l;
    CHECK(m15_get_row_span(&r, 0, &p, &l)==M15_ERR_NOT_OPEN, "unopened reader: m15_get_row_span must report NOT_OPEN");
    CHECK(m15_verify_tensor_once(&r)==M15_ERR_NOT_OPEN, "unopened reader: m15_verify_tensor_once must report NOT_OPEN");
    CHECK(m15_verify_block_once(&r, 0)==M15_ERR_NOT_OPEN, "unopened reader: m15_verify_block_once must report NOT_OPEN");
}

int main(int argc, char **argv){
    if(argc != 2){ fprintf(stderr, "usage: %s <fixture-dir>\n", argv[0]); return 2; }
    const char *fixdir = argv[1];

    test_crc32_check_value();
    test_programmer_error_classification();

    char happy_path[1024], orig_path[1024];
    snprintf(happy_path, sizeof(happy_path), "%s/happy_O37_I64_rpb8.bin", fixdir);
    snprintf(orig_path, sizeof(orig_path), "%s/happy_O37_I64_rpb8.orig", fixdir);
    size_t happy_len, orig_len;
    uint8_t *happy = slurp(happy_path, &happy_len);
    uint8_t *orig  = slurp(orig_path, &orig_len);

    test_happy_path(happy, happy_len, orig, orig_len);
    test_open_full(happy, happy_len);
    test_zero_row(fixdir);
    test_truncation(happy, happy_len);
    test_index_copy_lifetime(happy, happy_len);
    test_verify_once_caching(happy, happy_len);
    test_structural_vs_full(happy, happy_len);
    test_block_crc_localization(happy, happy_len);
    test_version_and_magic(happy, happy_len);
    test_row_offsets_hardening(happy, happy_len);
    test_o_i_mismatch(happy, happy_len);
    test_caps_rejection(happy, happy_len);
    test_length_table_all_zero(happy, happy_len);
    test_length_table_oversubscribed(happy, happy_len);
    test_length_table_incomplete(happy, happy_len);
    test_length_table_single_symbol(happy, happy_len);

    free(happy); free(orig);

    if(g_fails){ fprintf(stderr, "test_mode15_reader: %d failure(s)\n", g_fails); return 1; }
    puts("test_mode15_reader: ok");
    return 0;
}
