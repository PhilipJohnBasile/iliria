/* codec_row_bitplane.h -- branchless bitplane + exception format, the
 * "GPU-oriented exact" candidate: a k-bit (k<4) BASE PLANE covering a
 * contiguous window of the int4 alphabet that captures most of the
 * projection's probability mass, a 1-bit-per-symbol EXCEPTION BITMASK, and a
 * densely packed, RANK-INDEXED exception-value array for the symbols that
 * fall outside the window.
 *
 * Per-position decode work is UNIFORM regardless of whether a position is an
 * exception (the property that matters for GPU lanes / SIMD, where a
 * data-dependent branch or a data-dependent STREAM-WIDTH advance causes
 * divergence/serialization): every position reads a fixed k base bits, a
 * fixed 1 mask bit, and does an UNCONDITIONAL indexed gather into the
 * exception-value array at the running popcount-rank of exceptions seen so
 * far (a plain incrementing counter, not a scan) -- then selects between
 * the gathered exception value and the reconstructed base-window value with
 * a bitmask select (`(v & sel) | (base & ~sel)`), not a branch. This is
 * exactly the shape a real CUDA/Metal port would keep: the reference
 * decoder here is scalar C (per the design's "no Metal" scope) but every
 * operation in the per-symbol loop is either fixed-stride or a predicated
 * gather+select, nothing a GPU warp would diverge on.
 *
 * Codebook = (base_bits k, bias) chosen PER PROJECTION from the aggregate
 * histogram: search k in {1,2,3} and all valid window offsets for the
 * window [bias, bias+2^k) that minimizes modeled total bytes
 * (n*(k+1) bits for base+mask, plus 4 bits per predicted exception).
 *
 * Row layout: [uint16 n_exceptions]
 *             [base-plane: ceil(n*k/8) bytes, k bits/symbol]
 *             [exception bitmask: ceil(n/8) bytes, 1 bit/symbol]
 *             [exception values: ceil(n_exceptions/2) bytes, packed nibbles,
 *              in row order -- addressed by RANK, not consumed serially]
 */
#ifndef ILIRIA_CODEC_ROW_BITPLANE_H
#define ILIRIA_CODEC_ROW_BITPLANE_H

#include "codec_row.h"

typedef struct {
    int base_bits;  /* k in {1,2,3}, or 4 = RAW PASSTHROUGH fallback (no
                     * window helps enough: ship the row's native 4-bit
                     * codes unchanged, exactly the container's existing
                     * format, ratio==1.0 -- see build() below) */
    int bias;       /* window is [bias, bias + 2^k) subset of [0,16); unused when base_bits==4 */
} BitplaneCodebook;

/* Chooses (k, bias) by modeled total bytes against the aggregate histogram,
 * comparing every k in {1,2,3} and window offset against the TRUE raw cost
 * (total*4/8 bytes, zero header) -- so a distribution that is too flat for
 * any narrow window to pay for its own mask+exception overhead (e.g. this
 * codec race's own H~2.9-bits/weight synthetic data, closer to uniform than
 * to the sharply peaked shape this codec wants) correctly FALLS BACK to
 * base_bits=4 (raw passthrough) instead of silently picking an arbitrary,
 * worse-than-raw window. A codec whose stored ratio can exceed 1.0 (expand
 * the data) is not a valid entry in a compression race regardless of decode
 * speed -- this fallback is what keeps it honest. */
static void bitplane_build(const uint32_t counts[CODEC_NSYM], BitplaneCodebook *cb){
    uint64_t total=0; for(int s=0;s<CODEC_NSYM;s++) total += counts[s];
    if(total==0){ cb->base_bits=4; cb->bias=0; return; }
    double raw_cost = (double)total*4.0/8.0;
    double best_cost = raw_cost;
    int best_k=4, best_bias=0;
    for(int k=1;k<=3;k++){
        int width = 1<<k;
        for(int lo=0; lo+width<=CODEC_NSYM; lo++){
            uint64_t in_window=0;
            for(int v=lo; v<lo+width; v++) in_window += counts[v];
            uint64_t n_exc = (total>in_window) ? (total-in_window) : 0;
            double cost = (double)total*(double)(k+1)/8.0 + (double)n_exc*4.0/8.0;
            if(cost < best_cost){ best_cost=cost; best_k=k; best_bias=lo; }
        }
    }
    cb->base_bits=best_k; cb->bias=best_bias;
}

/* returns bytes written, or (size_t)-1 if `cap` is too small */
static size_t bitplane_encode_row(const BitplaneCodebook *cb, const uint8_t *sym, int n,
                                   uint8_t *out, size_t cap){
    if(n<=0) return 0;
    if(cb->base_bits==4){ /* raw passthrough: native packed nibbles, zero header, ratio==1.0 */
        size_t need = ((size_t)n+1)/2;
        if(need > cap) return (size_t)-1;
        codec_pack_nibbles(sym, n, out);
        return need;
    }
    int k=cb->base_bits, bias=cb->bias, width=1<<k;
    size_t base_bytes = ((size_t)n*(size_t)k+7)/8;
    size_t mask_bytes = ((size_t)n+7)/8;
    size_t off_base=2, off_mask=off_base+base_bytes, off_exc=off_mask+mask_bytes;
    if(off_exc > cap) return (size_t)-1;
    CodecBitW bw_base, bw_mask;
    cbw_init(&bw_base, out+off_base, base_bytes);
    cbw_init(&bw_mask, out+off_mask, mask_bytes);
    uint8_t *exc_area = out+off_exc;
    size_t exc_area_cap = cap-off_exc;
    uint32_t n_exc=0;
    for(int i=0;i<n;i++){
        uint8_t s = sym[i];
        int in_window = (s>=(uint8_t)bias) && (s<(uint8_t)(bias+width));
        uint32_t code = in_window ? (uint32_t)(s-(uint8_t)bias) : 0u;
        cbw_put(&bw_base, code, k);
        cbw_put(&bw_mask, in_window?0u:1u, 1);
        if(!in_window){
            size_t byte_idx = (size_t)(n_exc>>1);
            if(byte_idx >= exc_area_cap) return (size_t)-1;
            if((n_exc&1u)==0) exc_area[byte_idx] = (uint8_t)(s & 0xF);
            else exc_area[byte_idx] = (uint8_t)(exc_area[byte_idx] | (uint8_t)(s<<4));
            n_exc++;
        }
    }
    cbw_finish(&bw_base);
    cbw_finish(&bw_mask);
    size_t exc_bytes = ((size_t)n_exc+1)/2;
    uint16_t n_exc16 = (uint16_t)n_exc;
    memcpy(out, &n_exc16, sizeof(uint16_t));
    return off_exc + exc_bytes;
}

/* decode: caller-allocated sym_out[n], no internal allocation. Every
 * position does the same fixed-cost work (see file doc comment); the only
 * genuine branch is a defensive bounds check on the speculative exception-
 * area gather, which a real over-allocated GPU buffer would drop entirely. */
static void bitplane_decode_row(const BitplaneCodebook *cb, const uint8_t *payload, size_t paylen,
                                 int n, uint8_t *sym_out){
    if(n<=0) return;
    if(cb->base_bits==4){ codec_unpack_nibbles(payload, n, sym_out); return; }
    int k=cb->base_bits, bias=cb->bias;
    size_t base_bytes = ((size_t)n*(size_t)k+7)/8;
    size_t mask_bytes = ((size_t)n+7)/8;
    size_t off_base=2, off_mask=off_base+base_bytes, off_exc=off_mask+mask_bytes;
    CodecBitR br_base, br_mask;
    cbr_init(&br_base, payload+off_base, base_bytes);
    cbr_init(&br_mask, payload+off_mask, mask_bytes);
    const uint8_t *exc_area = payload+off_exc;
    size_t exc_area_len = (paylen>off_exc) ? (paylen-off_exc) : 0;
    uint32_t rank=0;
    for(int i=0;i<n;i++){
        uint32_t code = cbr_get(&br_base, k);
        uint32_t is_exc = cbr_get(&br_mask, 1);
        size_t byte_idx = (size_t)(rank>>1);
        uint8_t packed_byte = (byte_idx < exc_area_len) ? exc_area[byte_idx] : 0;
        uint8_t exc_val = ((rank&1u)==0) ? (uint8_t)(packed_byte & 0xF) : (uint8_t)(packed_byte >> 4);
        uint8_t base_val = (uint8_t)(code + (uint32_t)bias);
        uint8_t sel = (uint8_t)0 - (uint8_t)is_exc;         /* 0x00 or 0xFF, no branch */
        sym_out[i] = (uint8_t)((exc_val & sel) | (base_val & (uint8_t)~sel));
        rank += is_exc;
    }
}

#endif /* ILIRIA_CODEC_ROW_BITPLANE_H */
