/* codec_row_rans.h -- interleaved static rANS (range Asymmetric Numeral
 * System) over the 16 int4 symbols, RANS_LANES-way interleaved (default 4;
 * the "2-4 way" the codec race calls for -- recompile with a different
 * RANS_LANES to trade lane-header overhead for less inter-symbol ILP).
 *
 * Parameters mirror tools/measure_expert_entropy.py's reference rANS exactly
 * (M_BITS=12 quantized-frequency precision, RANS_L=1<<16 renormalization
 * bound, 16-bit renorm words) so the achieved ratio is directly comparable
 * to the census's already-measured 0.7379 (64KB-block) / 0.7442 (4KB-block)
 * numbers -- this header answers what that ratio costs at ROW granularity
 * and what single-lane vs. interleaved decode buys in GB/s.
 *
 * Interleaving: row position i is assigned to lane (i % RANS_LANES). A
 * SINGLE combined loop over i=n-1..0 (encode) / k=0..n-1 (decode) advances
 * whichever lane's state that position belongs to; consecutive loop
 * iterations touch INDEPENDENT lane states (no data dependency until you
 * wrap back to the same lane RANS_LANES steps later), which is what lets
 * the CPU pipeline RANS_LANES decode steps concurrently instead of
 * serializing on one state's dependency chain -- the actual mechanism an
 * "interleaved rANS" decode is faster than a single-lane one.
 *
 * Row layout: [RANS_LANES x uint32 final state][RANS_LANES x uint16 substream
 * byte length][substream_0][substream_1]...[substream_{L-1}], each substream
 * the time-reversed renormalization words for that lane (identical LIFO
 * convention to the Python reference's rans_encode/rans_decode, just
 * replicated across independent lanes). Codebook (quantized frequency table
 * + cumulative table + slot->symbol decode LUT) is shared per projection.
 */
#ifndef ILIRIA_CODEC_ROW_RANS_H
#define ILIRIA_CODEC_ROW_RANS_H

#include "codec_row.h"

#ifndef RANS_LANES
#define RANS_LANES 4
#endif
#define RANS_MBITS 12
#define RANS_MTOTAL (1u << RANS_MBITS)
#define RANS_L (1u << 16)

typedef struct {
    uint32_t freq[CODEC_NSYM];
    uint32_t cum[CODEC_NSYM+1];
    uint8_t  slot2sym[RANS_MTOTAL];
} RansCodebook;

static void rans_build(const uint32_t counts[CODEC_NSYM], RansCodebook *cb){
    codec_quantize_freqs(counts, RANS_MTOTAL, cb->freq);
    cb->cum[0]=0;
    for(int s=0;s<CODEC_NSYM;s++) cb->cum[s+1]=cb->cum[s]+cb->freq[s];
    for(int s=0;s<CODEC_NSYM;s++)
        for(uint32_t slot=cb->cum[s]; slot<cb->cum[s+1]; slot++) cb->slot2sym[slot]=(uint8_t)s;
}

#define RANS_HEADER_BYTES ((size_t)RANS_LANES*sizeof(uint32_t) + (size_t)RANS_LANES*sizeof(uint16_t))

/* returns bytes written, or (size_t)-1 if `cap` is too small */
static size_t rans_encode_row(const RansCodebook *cb, const uint8_t *sym, int n,
                               uint8_t *out, size_t cap){
    if(n<=0) return 0;
    const int L = RANS_LANES;
    uint32_t x[RANS_LANES];
    uint16_t *words[RANS_LANES]; size_t nwords[RANS_LANES], capw[RANS_LANES];
    for(int l=0;l<L;l++){
        x[l]=RANS_L;
        capw[l]=(size_t)(n/L)+8;
        words[l]=(uint16_t*)malloc(capw[l]*sizeof(uint16_t));
        nwords[l]=0;
    }
    /* NOTE: computed in 64-bit. x_max_base=2^20 and f can reach M_TOTAL=2^12
     * (a single symbol taking the whole table, e.g. a degenerate/near-zero-
     * entropy row) -- their product is exactly 2^32, which silently wraps to
     * 0 in 32-bit arithmetic and turns `while(x>=xmax)` into an infinite
     * loop (caught by the fuzz test's near-degenerate-distribution case,
     * not by the typical near-uniform quantized-weight shapes). */
    uint64_t x_max_base = ((uint64_t)RANS_L >> RANS_MBITS) << 16;
    for(int i=n-1;i>=0;i--){
        int l = i % L;
        uint8_t s = sym[i];
        uint32_t f = cb->freq[s];
        uint64_t xmax = x_max_base * (uint64_t)f;
        while((uint64_t)x[l] >= xmax){
            if(nwords[l]==capw[l]){ capw[l]=capw[l]*2+2; words[l]=(uint16_t*)realloc(words[l], capw[l]*sizeof(uint16_t)); }
            words[l][nwords[l]++] = (uint16_t)(x[l] & 0xFFFF);
            x[l] >>= 16;
        }
        x[l] = ((x[l]/f) << RANS_MBITS) + (x[l]%f) + cb->cum[s];
    }
    size_t total = RANS_HEADER_BYTES;
    for(int l=0;l<L;l++) total += nwords[l]*sizeof(uint16_t);
    if(total > cap){
        for(int l=0;l<L;l++) free(words[l]);
        return (size_t)-1;
    }
    size_t pos=0;
    for(int l=0;l<L;l++){ memcpy(out+pos, &x[l], sizeof(uint32_t)); pos+=sizeof(uint32_t); }
    for(int l=0;l<L;l++){ uint16_t lb=(uint16_t)(nwords[l]*sizeof(uint16_t)); memcpy(out+pos, &lb, sizeof(uint16_t)); pos+=sizeof(uint16_t); }
    for(int l=0;l<L;l++){
        for(size_t k=0;k<nwords[l];k++){
            uint16_t w = words[l][nwords[l]-1-k];
            memcpy(out+pos, &w, sizeof(uint16_t)); pos+=sizeof(uint16_t);
        }
        free(words[l]);
    }
    return pos;
}

/* decode: caller-allocated sym_out[n], no internal allocation */
static void rans_decode_row(const RansCodebook *cb, const uint8_t *payload, size_t paylen,
                             int n, uint8_t *sym_out){
    if(n<=0) return;
    const int L = RANS_LANES;
    uint32_t x[RANS_LANES]; uint16_t lens[RANS_LANES];
    size_t pos=0;
    for(int l=0;l<L;l++){ memcpy(&x[l], payload+pos, sizeof(uint32_t)); pos+=sizeof(uint32_t); }
    for(int l=0;l<L;l++){ memcpy(&lens[l], payload+pos, sizeof(uint16_t)); pos+=sizeof(uint16_t); }
    const uint8_t *lane_ptr[RANS_LANES]; size_t lane_len[RANS_LANES], lane_pos[RANS_LANES];
    for(int l=0;l<L;l++){ lane_ptr[l]=payload+pos; lane_len[l]=lens[l]; lane_pos[l]=0; pos+=lens[l]; }
    (void)paylen;
    for(int k=0;k<n;k++){
        int l = k % L;
        uint32_t slot = x[l] & (RANS_MTOTAL-1);
        uint8_t s = cb->slot2sym[slot];
        sym_out[k] = s;
        x[l] = cb->freq[s]*(x[l]>>RANS_MBITS) + slot - cb->cum[s];
        while(x[l] < RANS_L && lane_pos[l]+2<=lane_len[l]){
            uint16_t w; memcpy(&w, lane_ptr[l]+lane_pos[l], sizeof(uint16_t)); lane_pos[l]+=2;
            x[l] = (x[l]<<16) | w;
        }
    }
}

#endif /* ILIRIA_CODEC_ROW_RANS_H */
