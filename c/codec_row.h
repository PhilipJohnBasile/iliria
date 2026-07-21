/* codec_row.h -- shared plumbing for the row-independent int4 entropy-codec
 * race (docs/PERFORMANCE_THEORY.md n1: "n1_codec_race_requirement").
 *
 * Context: the census (c/bench-m5max/new-math-20260715/) measured H(Q)=2.94
 * bits/weight over the shipped int4 expert codes and a 64KB-block rANS ratio
 * of 0.7379 -- but mode-3 (decode fused into the GEMV dequant loop) needs
 * ROW-INDEPENDENT decode: a GEMV consumes one output row at a time (gate/up
 * rows ~3KB raw = 6144 int4 symbols, down rows ~1KB raw = 2048 symbols) and
 * cannot wait for a 64KB block spanning many rows. This header (+
 * codec_row_huff.h / codec_row_rans.h / codec_row_fse.h / codec_row_bitplane.h)
 * implements four EXACT (lossless, bit-identical) row-coded candidates and
 * measures which is fastest to decode per raw byte produced -- the actual
 * deciding question for mode-3, not the smallest container.
 *
 * Coding unit: ONE OUTPUT ROW. A "tensor" (one gate/up/down projection of one
 * expert) is O rows of I symbols each; each row is encoded independently
 * against a CODEBOOK SHARED across the whole projection (built once from the
 * aggregate symbol histogram), plus a row_offsets[O+1] index (cumulative
 * compressed-byte offsets) for O(1) random-access to any row. Per-row F32
 * quantization scales are UNCHANGED (stay raw, exactly as today's container
 * and as the census's own accounting: "Per-row F32 scales stay raw").
 *
 * All four codecs share:
 *   - the symbol alphabet: 4-bit nibbles, values 0..15 (glm.c's int4 codes,
 *     stored 2/byte: low nibble = even index, high nibble = odd index --
 *     see pack_int4/qt_addrow in glm.c and unpack_nibbles in
 *     tools/measure_expert_entropy.py);
 *   - a caller-allocated-buffer discipline: decode functions NEVER allocate
 *     (matches how a real GEMV dequant loop must work -- decode into a
 *     pre-sized stack/temporary, not a fresh heap block per row); encode
 *     functions take a capacity-bounded output buffer and return bytes used;
 *   - an LSB-first bit-packing convention (fast shift/mask, standard for
 *     rANS/FSE-style codecs; the canonical-Huffman codec bit-reverses its
 *     codewords once at codebook-build time so its *stream* order is still
 *     classical MSB-first canonical order -- see codec_row_huff.h).
 *
 * Single-header, `static`-function style matching tier.h/json.h/grammar.h.
 */
#ifndef ILIRIA_CODEC_ROW_H
#define ILIRIA_CODEC_ROW_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CODEC_NSYM 16          /* int4 alphabet size */

/* ---------------------------------------------------------- nibble I/O --
 * Matches glm.c's pack_int4 / tools/measure_expert_entropy.py's
 * unpack_nibbles: low nibble = even symbol index, high nibble = odd. */
static void codec_unpack_nibbles(const uint8_t *packed, int n, uint8_t *sym){
    int i=0;
    for(; i+1<n; i+=2){ uint8_t b=packed[i>>1]; sym[i]=(uint8_t)(b&0xF); sym[i+1]=(uint8_t)(b>>4); }
    if(i<n) sym[i]=(uint8_t)(packed[i>>1]&0xF);
}
static void codec_pack_nibbles(const uint8_t *sym, int n, uint8_t *packed){
    int i=0;
    for(; i+1<n; i+=2) packed[i>>1]=(uint8_t)((sym[i]&0xF)|((sym[i+1]&0xF)<<4));
    if(i<n) packed[i>>1]=(uint8_t)(sym[i]&0xF);
}

/* ------------------------------------------------------------ histogram --*/
static void codec_histogram(const uint8_t *sym, int n, uint32_t counts[CODEC_NSYM]){
    memset(counts, 0, CODEC_NSYM*sizeof(uint32_t));
    for(int i=0;i<n;i++) counts[sym[i]]++;
}

static double codec_entropy_bits(const uint32_t counts[CODEC_NSYM]){
    uint64_t total=0; for(int s=0;s<CODEC_NSYM;s++) total+=counts[s];
    if(!total) return 0.0;
    double h=0.0, n=(double)total;
    for(int s=0;s<CODEC_NSYM;s++) if(counts[s]){ double p=counts[s]/n; h -= p*log2(p); }
    return h;
}

/* ---- integer frequency quantizer: scale a count vector to sum exactly to
 * `m_total` (a power of two), every present symbol kept >= 1. Port of
 * tools/measure_expert_entropy.py's quantize_freqs (repair-the-largest-
 * bucket algorithm) -- used by both the rANS and FSE codebook builders so
 * their shared tables are internally consistent. */
static void codec_quantize_freqs(const uint32_t counts[CODEC_NSYM], uint32_t m_total,
                                  uint32_t freqs_out[CODEC_NSYM]){
    uint64_t total=0; for(int s=0;s<CODEC_NSYM;s++) total+=counts[s];
    memset(freqs_out, 0, CODEC_NSYM*sizeof(uint32_t));
    if(!total) return;
    int64_t sum=0;
    for(int s=0;s<CODEC_NSYM;s++){
        if(counts[s]==0) continue;
        double f = (double)counts[s] / (double)total * (double)m_total;
        uint32_t fi = (uint32_t)(f + 0.5);
        if(fi < 1) fi = 1;
        freqs_out[s] = fi;
        sum += fi;
    }
    int64_t diff = (int64_t)m_total - sum;
    if(diff == 0) return;
    /* repair onto the largest buckets first, never dropping any present
     * symbol below 1 (mirrors the Python reference's order-by-argsort loop,
     * done here as a simple bounded fixed-point search over ~16 buckets) */
    int order[CODEC_NSYM]; int npres=0;
    for(int s=0;s<CODEC_NSYM;s++) if(freqs_out[s]>0) order[npres++]=s;
    /* selection sort descending by freq (npres<=16: trivial cost) */
    for(int a=0;a<npres;a++){ int best=a;
        for(int b=a+1;b<npres;b++) if(freqs_out[order[b]]>freqs_out[order[best]]) best=b;
        int t=order[a]; order[a]=order[best]; order[best]=t; }
    int guard=0;
    while(diff != 0 && npres>0){
        int s = order[guard % npres];
        int step = (diff>0) ? 1 : -1;
        if((int64_t)freqs_out[s] + step >= 1){ freqs_out[s]=(uint32_t)((int64_t)freqs_out[s]+step); diff -= step; }
        guard++;
        if(guard > 1000000) break; /* unreachable in practice; defensive only */
    }
}

/* --------------------------------------------------------- bit writer ---
 * LSB-first: bw_put(w, code, len) appends `len` bits (value `code`, len<=24
 * per call so the 32-bit accumulator never overflows against the <8 leftover
 * bits from the previous call); byte 0 of the stream holds the earliest-
 * written bits in bit0..bit7 order. Caller supplies the output buffer
 * (capacity-bounded, no internal allocation). */
typedef struct { uint8_t *buf; size_t cap; size_t pos; uint32_t acc; int nbits; } CodecBitW;

static void cbw_init(CodecBitW *w, uint8_t *buf, size_t cap){
    w->buf=buf; w->cap=cap; w->pos=0; w->acc=0; w->nbits=0;
}
static void cbw_put(CodecBitW *w, uint32_t code, int len){
    if(len<=0) return;
    uint32_t mask = (len>=32) ? 0xFFFFFFFFu : ((1u<<len)-1u);
    w->acc |= (code & mask) << w->nbits;
    w->nbits += len;
    while(w->nbits >= 8){
        if(w->pos < w->cap) w->buf[w->pos] = (uint8_t)(w->acc & 0xFF);
        w->pos++;
        w->acc >>= 8;
        w->nbits -= 8;
    }
}
/* flush the partial byte (zero-padded); returns total bytes written */
static size_t cbw_finish(CodecBitW *w){
    if(w->nbits > 0){
        if(w->pos < w->cap) w->buf[w->pos] = (uint8_t)(w->acc & 0xFF);
        w->pos++;
        w->acc=0; w->nbits=0;
    }
    return w->pos;
}

/* --------------------------------------------------------- bit reader ---
 * Mirrors CodecBitW: acc holds `nbits` valid low bits (bits at position
 * >= nbits are always 0 by construction/induction, which is exactly the
 * zero-padding a wide-LUT peek near end-of-stream needs). */
typedef struct { const uint8_t *buf; size_t len; size_t pos; uint64_t acc; int nbits; } CodecBitR;

static void cbr_init(CodecBitR *r, const uint8_t *buf, size_t len){
    r->buf=buf; r->len=len; r->pos=0; r->acc=0; r->nbits=0;
}
static void cbr_refill(CodecBitR *r, int need){
    while(r->nbits < need && r->pos < r->len){
        r->acc |= (uint64_t)r->buf[r->pos++] << r->nbits;
        r->nbits += 8;
    }
}
/* peek `k` bits without consuming (k<=32; zero-padded past end of stream) */
static uint32_t cbr_peek(CodecBitR *r, int k){
    uint64_t mask = (k>=64) ? ~0ULL : ((1ULL<<k)-1ULL);
    return (uint32_t)(r->acc & mask);
}
static void cbr_drop(CodecBitR *r, int k){ r->acc >>= k; r->nbits -= k; }
/* read+consume k bits in one call (refills first) */
static uint32_t cbr_get(CodecBitR *r, int k){
    cbr_refill(r, k);
    uint32_t v = cbr_peek(r, k);
    cbr_drop(r, k);
    return v;
}

/* bit-reversal of the low `len` bits of x (len<=16; used once per symbol at
 * codebook-build time, not in the decode hot loop) */
static uint32_t codec_bitrev(uint32_t x, int len){
    uint32_t r=0;
    for(int i=0;i<len;i++){ r=(r<<1)|(x&1u); x>>=1; }
    return r;
}

/* highbit(x) = floor(log2(x)), x>=1 (used by the FSE table builder) */
static int codec_highbit32(uint32_t x){
    int r=0; while(x>1){ x>>=1; r++; } return r;
}

/* ---------------------------------------------------- row_offsets index --
 * O(1) random access to row r's compressed bytes within a projection's
 * concatenated payload: start=row_offsets[r], len=row_offsets[r+1]-start.
 * Growable append helper used by the benchmark harness while encoding a
 * whole projection's rows in sequence. */
typedef struct {
    uint32_t *offsets; /* [n_rows+1], offsets[0]==0 */
    uint32_t n_rows;
    uint32_t cap_rows;
} RowOffsets;

static void row_offsets_init(RowOffsets *ro, uint32_t expect_rows){
    ro->cap_rows = expect_rows>0?expect_rows:1;
    ro->offsets = (uint32_t*)malloc((ro->cap_rows+1)*sizeof(uint32_t));
    ro->offsets[0]=0;
    ro->n_rows=0;
}
static void row_offsets_push(RowOffsets *ro, uint32_t row_bytes){
    if(ro->n_rows+1 >= ro->cap_rows){
        ro->cap_rows = ro->cap_rows*2+1;
        ro->offsets = (uint32_t*)realloc(ro->offsets, (ro->cap_rows+1)*sizeof(uint32_t));
    }
    ro->offsets[ro->n_rows+1] = ro->offsets[ro->n_rows] + row_bytes;
    ro->n_rows++;
}
static void row_offsets_free(RowOffsets *ro){ free(ro->offsets); ro->offsets=NULL; }
/* fixed per-projection cost of shipping the index itself */
static size_t row_offsets_index_bytes(uint32_t n_rows){ return (size_t)(n_rows+1)*sizeof(uint32_t); }

#endif /* ILIRIA_CODEC_ROW_H */
