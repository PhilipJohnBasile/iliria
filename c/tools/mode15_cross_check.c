/* mode15_cross_check.c -- independent cross-check that mode15_container.py's
 * Python encoder produces a bitstream the REAL codec_row_huff.h decode path
 * (the same functions tests/test_metal_row_decode.mm validated bit-exact
 * against the Metal GPU kernel) can decode correctly.
 *
 * WHY THIS EXISTS. mode15_container.py's own decode_tensor()/parse_tensor_
 * blob() round-trip test (tests/test_mode15_container.py) only proves the
 * Python encoder and Python decoder agree with EACH OTHER -- a bug shared
 * by both (e.g. a wrong tie-break in the canonical-code assignment) could
 * still round-trip internally while producing a bitstream the ACTUAL C/
 * Metal decoder disagrees with. This program instead feeds the Python
 * encoder's OUTPUT into the REAL, UNMODIFIED codec_row_huff.h functions
 * (huff_canonical_codes -- the bit-reversal + canonical assignment that
 * determines which bits get written -- and huff_decode_row -- the exact
 * bit-walk tests/test_metal_row_decode.mm already proved bit-identical to
 * the Metal huff_decode_rows kernel) and checks the decoded symbols match
 * the known original nibbles. Per that file's own header comment: "a GPU/
 * CPU decode mismatch [there] is a Metal-side bug, not a data-generation
 * artifact" -- by the same transitive argument, if THIS program's decode
 * (via the identical C functions) matches, the Metal kernel would decode
 * this container's bitstream correctly too, without needing a Metal run
 * here.
 *
 * codec_row_huff.h does not expose a "build codebook from a lengths table"
 * entry point (huff_build() only accepts raw symbol COUNTS, deriving
 * lengths itself via huff_build_lengths() -- but our on-disk container
 * ships LENGTHS, not counts, precisely so any reader can skip that step).
 * The LUT-fill loop below is a faithful, unmodified-logic copy of
 * huff_build()'s own LUT construction (codec_row_huff.h lines ~108-133,
 * this file's own comment marks where) -- NOT a reimplementation of the
 * bit-reversal/canonical-order logic (that part calls the REAL
 * huff_canonical_codes() directly, unmodified). This file makes NO edits
 * to codec_row.h/codec_row_huff.h (out of scope per this file's "NO engine
 * changes" boundary -- those headers are shared with the future engine
 * decode path and tests/test_metal_row_decode.mm).
 *
 * Fixture format (written by mode15_container.write_cross_check_fixture(),
 * magic "MHFX" -- distinct from the container's own per-tensor "MH01" blob
 * format; this is a throwaway test fixture, not a container artifact):
 *   [4B]            magic "MHFX"
 *   [u32 LE]        O
 *   [u32 LE]        I
 *   [8B]            canonical Huffman lengths, packed 4-bit (mode15_container's pack_lengths)
 *   [O*I bytes]     original nibble symbols, row-major, one byte per symbol (0..15)
 *   [(O+1)*4 bytes] row_offsets, u32 LE
 *   [row_offsets[O] bytes] payload (the actual encoded bitstream)
 *
 * Usage: ./mode15_cross_check <fixture.bin>
 * Exit 0 = every row decoded bit-identical to the original. Exit 1 = any
 * mismatch or malformed fixture (fail-closed, prints details to stderr).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../codec_row.h"
#include "../codec_row_huff.h"

static uint32_t read_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Faithful copy of codec_row_huff.h's huff_build() LUT-construction logic
 * (see that function's lines building cb->maxlen/cb->lut from cb->len,
 * cb->code_rev) -- NOT the length-choice or canonical-order logic, which
 * this program calls directly from the header, unmodified. Kept here only
 * because huff_build() itself starts from raw counts, not a lengths table. */
static void build_lut_from_lengths(HuffCodebook *cb) {
    int maxlen = 0, npres = 0;
    for (int s = 0; s < CODEC_NSYM; s++) {
        if (cb->len[s] > 0) {
            npres++;
            if (cb->len[s] > maxlen) maxlen = cb->len[s];
        }
    }
    cb->n_present = npres;
    cb->maxlen = maxlen > 0 ? maxlen : 1;
    size_t lutn = (size_t)1u << cb->maxlen;
    cb->lut = (HuffLutEnt *)malloc(lutn * sizeof(HuffLutEnt));
    memset(cb->lut, 0, lutn * sizeof(HuffLutEnt));
    if (npres == 1) {
        uint8_t s = 0;
        for (int t = 0; t < CODEC_NSYM; t++) if (cb->len[t] > 0) s = (uint8_t)t;
        for (size_t i = 0; i < lutn; i++) { cb->lut[i].sym = s; cb->lut[i].len = 1; }
        return;
    }
    for (int s = 0; s < CODEC_NSYM; s++) {
        if (cb->len[s] == 0) continue;
        int L = cb->len[s];
        uint32_t base = cb->code_rev[s];
        int rest = cb->maxlen - L;
        size_t block = (size_t)1u << rest;
        for (size_t hi = 0; hi < block; hi++) {
            size_t idx = base | (hi << L);
            cb->lut[idx].sym = (uint8_t)s;
            cb->lut[idx].len = (uint8_t)L;
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture.bin>\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 0) { fprintf(stderr, "ftell failed\n"); return 2; }
    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (fread(buf, 1, (size_t)fsize, f) != (size_t)fsize) {
        fprintf(stderr, "short read on fixture\n"); return 2;
    }
    fclose(f);

    if (fsize < 16 || memcmp(buf, "MHFX", 4) != 0) {
        fprintf(stderr, "bad fixture magic\n"); return 1;
    }
    uint32_t O = read_u32le(buf + 4);
    uint32_t I = read_u32le(buf + 8);
    const uint8_t *lengths_packed = buf + 12;
    size_t off = 20;

    uint64_t nib_bytes = (uint64_t)O * (uint64_t)I;
    if ((uint64_t)fsize < off + nib_bytes) { fprintf(stderr, "truncated: nibbles\n"); return 1; }
    const uint8_t *orig_nibbles = buf + off;
    off += nib_bytes;

    uint64_t ro_bytes = (uint64_t)(O + 1) * 4;
    if ((uint64_t)fsize < off + ro_bytes) { fprintf(stderr, "truncated: row_offsets\n"); return 1; }
    const uint8_t *ro_raw = buf + off;
    off += ro_bytes;

    uint32_t *row_offsets = (uint32_t *)malloc((size_t)(O + 1) * sizeof(uint32_t));
    for (uint32_t r = 0; r <= O; r++) row_offsets[r] = read_u32le(ro_raw + (size_t)r * 4);

    uint64_t payload_len = row_offsets[O];
    if ((uint64_t)fsize < off + payload_len) { fprintf(stderr, "truncated: payload\n"); return 1; }
    const uint8_t *payload = buf + off;

    /* unpack the 8-byte 4-bit length table (mode15_container.pack_lengths convention) */
    HuffCodebook cb;
    memset(&cb, 0, sizeof(cb));
    for (int k = 0; k < 8; k++) {
        uint8_t b = lengths_packed[k];
        cb.len[2 * k] = b & 0xF;
        cb.len[2 * k + 1] = b >> 4;
    }
    huff_canonical_codes(cb.len, cb.code_rev);   /* REAL codec_row_huff.h function, unmodified */
    build_lut_from_lengths(&cb);                 /* faithful copy of huff_build()'s LUT loop only */

    fprintf(stderr, "mode15_cross_check: O=%u I=%u maxlen=%d n_present=%d lut_bytes=%d\n",
            O, I, cb.maxlen, cb.n_present, 1 << cb.maxlen);

    uint8_t *decoded = (uint8_t *)malloc(I ? I : 1);
    uint64_t n_row_mismatches = 0, n_symbol_mismatches = 0;
    for (uint32_t r = 0; r < O; r++) {
        uint32_t start = row_offsets[r], end = row_offsets[r + 1];
        huff_decode_row(&cb, payload + start, end - start, (int)I, decoded);  /* REAL decode function */
        const uint8_t *orig_row = orig_nibbles + (uint64_t)r * I;
        if (memcmp(decoded, orig_row, I) != 0) {
            n_row_mismatches++;
            for (uint32_t i = 0; i < I; i++) if (decoded[i] != orig_row[i]) n_symbol_mismatches++;
            if (n_row_mismatches <= 5) {
                fprintf(stderr, "  MISMATCH row %u\n", r);
            }
        }
    }

    fprintf(stderr, "rows checked=%u row_mismatches=%llu symbol_mismatches=%llu\n",
            O, (unsigned long long)n_row_mismatches, (unsigned long long)n_symbol_mismatches);

    free(decoded);
    free(row_offsets);
    huff_free(&cb);
    free(buf);

    if (n_row_mismatches != 0) {
        fprintf(stderr, "mode15_cross_check: FAIL -- codec_row_huff.h's REAL decoder disagrees "
                        "with the Python encoder's output (this is a format-compatibility bug, "
                        "not noise)\n");
        return 1;
    }
    printf("mode15_cross_check: PASS -- codec_row_huff.h's real huff_canonical_codes()+"
           "huff_decode_row() reconstruct the Python encoder's bitstream bit-exactly (O=%u rows)\n", O);
    return 0;
}
