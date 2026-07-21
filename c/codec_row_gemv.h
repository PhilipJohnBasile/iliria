/* codec_row_gemv.h -- mode-3 GEMV integration seam, SKETCH ONLY (reference
 * scalar C, no Metal/SIMD -- matches this file's explicit scope).
 *
 * STATUS (per the codec-race results, this study's own
 * verdict): none of the four row codecs (codec_row_huff.h, codec_row_rans.h,
 * codec_row_fse.h, codec_row_bitplane.h) clear the mode-3 decode-speed
 * budget as scalar C on this hardware (36-50x too slow at the easier
 * miss-path bar, even 18-way threaded). This header is therefore a DESIGN
 * SKETCH of what the fusion seam would look like IF a future SIMD/GPU
 * decoder closes that gap -- not a production integration, and deliberately
 * NOT included from glm.c or wired into the Makefile's engine build.
 *
 * WHAT THIS REPLACES, EXACTLY
 * ----------------------------
 * Today's int4 nibble unpack (`(int)(byte&0xF)-8` for the even/low-nibble
 * element, `(int)(byte>>4)-8` for the odd/high-nibble element, then a
 * per-output-row F32 scale multiply) appears in glm.c in two call sites,
 * both doing the identical two-line dequant against a flat, fixed-stride
 * packed array:
 *
 *   1. matmul_i4() (glm.c ~L276-310): the batched SIMD GEMV,
 *      y[S,O] = x[S,I] @ W^T, W int4-packed (2 values/byte) + scale[O].
 *      The AVX2/NEON blocks unpack 16 nibbles/iteration straight from
 *      `w[i>>1]` where `w = q4 + o*rb`, `rb=(I+1)/2` -- i.e. row `o` is
 *      found by a FIXED STRIDE multiply, because the container packs every
 *      row at the same width.
 *   2. qt_addrow() / qt_matvec_rows() (glm.c ~L1322-1348): the ABSORPTION
 *      row-wise accessors (`acc[0..I) += coef*W[row,:]` and
 *      `y[j] = W[r0+j,:]*x` respectively), same nibble unpack, one row at
 *      a time via `w = t->q4 + row*((I+1)/2)`.
 *
 * Mode-3 fuses a row DECODE into this exact loop shape instead of the
 * nibble unpack -- same call site, same per-row scale multiply, same
 * float32 accumulation, same signed dequant convention (decoded symbols
 * are 0..15, mapped to [-8,7] via -8, IDENTICAL to qt_addrow's
 * `(b&0xF)-8`). The only thing that changes is where the int4 VALUES come
 * from: instead of a fixed-stride flat array (`q4 + row*rb`), a row lives
 * at a VARIABLE offset (`payload + row_offsets[row]`, length
 * `row_offsets[row+1]-row_offsets[row]`) -- still O(1) addressable, just
 * via an index instead of a stride multiply. That is the entire reason
 * row_offsets[O+1] exists (codec_row.h's RowOffsets): mode-3 needs
 * per-row random access, and a shared/projection-level codebook (also in
 * codec_row.h's design) means the row itself carries no header beyond its
 * own compressed bits, so `payload+row_offsets[row]` is the whole story.
 */
#ifndef ILIRIA_CODEC_ROW_GEMV_H
#define ILIRIA_CODEC_ROW_GEMV_H

#include "codec_row.h"

/* Codec-agnostic decode step: any of codec_row_{huff,rans,fse,bitplane}.h's
 * (codebook, decode_row) pair fits this shape via a one-line adapter --
 * exactly the adapter pattern codec_race.c already uses for its benchmark
 * driver (e.g. `rans_dec_adapt`). `out_symbols` receives `n` values in
 * 0..15, the SAME representation qt_addrow already dequantizes from -- the
 * codec is swappable behind this seam without touching the GEMV loop. */
typedef void (*RowDecodeFn)(const void *codebook, const uint8_t *row_stream,
                             size_t row_stream_len, int n, uint8_t *out_symbols);

typedef struct {
    const void     *codebook;     /* shared per-projection codebook, opaque here */
    RowDecodeFn     decode;       /* e.g. rans_decode_row via an adapter, see codec_race.c */
    const uint8_t  *payload;      /* concatenated compressed rows for this projection */
    const uint32_t *row_offsets;  /* [O+1], cumulative bytes into `payload` */
} CodedTensor;

#define GEMV_CHUNK 16

/* decode_row_into_registers: decode ONE row, in principle CHUNK-at-a-time
 * so each chunk is immediately consumed by the multiply-accumulate before
 * the next chunk is decoded -- "into registers" in the sense that a real
 * fused kernel keeps one GEMV_CHUNK-wide burst of decoded symbols live in
 * vector registers for exactly one FMA burst, never materializing a whole
 * row in memory (rows run up to 6144 symbols -- far more than register
 * width, and materializing the whole row first is exactly the separate
 * decode-then-dequant PASS structure n1's mode-1/mode-2 already ruled out
 * as too slow; mode-3's entire premise is fusing the two).
 *
 * HONEST LIMITATION OF THIS REFERENCE: decoding a partial row and PAUSING
 * mid-stream (returning control after GEMV_CHUNK symbols, then resuming
 * later from the same codec state) needs each codec's decode loop
 * restructured into an explicit resumable step -- straightforward in
 * principle for all four codecs (rANS/tANS already carry an explicit
 * `state`; Huffman/bitplane already carry an explicit bit-reader position,
 * so "pause" is just "stop the loop and keep that struct around") but is a
 * real per-codec rewrite this sketch does not attempt, per the design's own
 * scope (header + reference scalar implementation only). What this
 * reference does instead: decode the WHOLE row up front into a reusable
 * scratch buffer (still one malloc per tensor, not per row -- see
 * matmul_i4_coded_row_ref), then walk it in GEMV_CHUNK bursts for the FMA,
 * which shows the intended LOOP SHAPE at the GEMV call site accurately even
 * though the decode call itself isn't chunked yet. */
static void decode_row_into_registers(const CodedTensor *t, int row, int I,
                                       uint8_t *row_scratch /* size >= I */){
    size_t off = t->row_offsets[row];
    size_t len = t->row_offsets[row+1] - off;
    t->decode(t->codebook, t->payload + off, len, I, row_scratch);
}

/* mode-3 counterpart of glm.c's matmul_i4() (~L276-310): same shape
 * (y, x, scale, S, I, O), same per-(s,o) accumulation and per-o scale
 * multiply -- the nibble-unpack inner loop is replaced by a decode-once-
 * per-row + chunked multiply-accumulate. `q4` (flat packed array) becomes
 * a CodedTensor (payload + row_offsets + shared codebook). */
static void matmul_i4_coded_row_ref(float *y, const float *x, const CodedTensor *t,
                                     const float *scale, int S, int I, int O){
    uint8_t *row_scratch = (uint8_t*)malloc((size_t)I);
    for(int o=0; o<O; o++){
        /* decode ONCE per output row, exactly like matmul_i4 derives its
         * `w = q4 + o*rb` pointer once per `o` and reuses it across `s` */
        decode_row_into_registers(t, o, I, row_scratch);
        float sc = scale[o];
        for(int s=0; s<S; s++){
            const float *xs = x + (int64_t)s*I;
            float a = 0.0f;
            int i = 0;
            /* the chunked multiply-accumulate a real fused kernel would
             * pair with a resumable per-chunk decode (see doc comment
             * above): GEMV_CHUNK symbols consumed per burst. */
            for(; i+GEMV_CHUNK<=I; i+=GEMV_CHUNK){
                for(int k=0;k<GEMV_CHUNK;k++){
                    int v = (int)row_scratch[i+k] - 8; /* SAME dequant as qt_addrow's (b&0xF)-8 */
                    a += xs[i+k]*(float)v;
                }
            }
            for(; i<I; i++){ int v=(int)row_scratch[i]-8; a += xs[i]*(float)v; }
            y[(int64_t)s*O+o] = a*sc;
        }
    }
    free(row_scratch);
}

/* mode-3 counterpart of glm.c's qt_addrow() (~L1322-1333):
 * acc[0..I) += coef * W[row,:], W row-coded. `row_scratch` is caller-
 * allocated (size >= I) and meant to be reused across calls in a loop,
 * matching qt_addrow's own per-call cost (no allocation in the hot path). */
static void qt_addrow_coded_ref(const CodedTensor *t, int row, int I, float coef, float *acc,
                                 uint8_t *row_scratch){
    decode_row_into_registers(t, row, I, row_scratch);
    for(int i=0;i<I;i++) acc[i] += coef*((int)row_scratch[i]-8);
}

/* mode-3 counterpart of glm.c's qt_matvec_rows() (~L1334-1348):
 * y[0..n) = W[r0+j,:] . x, W row-coded. */
static void qt_matvec_rows_coded_ref(const CodedTensor *t, int r0, int n, int I,
                                      const float *x, const float *scale, float *y,
                                      uint8_t *row_scratch){
    for(int j=0;j<n;j++){
        int row = r0+j;
        decode_row_into_registers(t, row, I, row_scratch);
        double a = 0;
        for(int i=0;i<I;i++) a += (double)((int)row_scratch[i]-8) * x[i];
        y[j] = (float)(a*scale[row]);
    }
}

#endif /* ILIRIA_CODEC_ROW_GEMV_H */
