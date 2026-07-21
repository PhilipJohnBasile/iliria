/* codec_row_huff.h -- canonical Huffman over the 16 int4 symbols, decoded via
 * a single wide lookup table (peek maxlen bits, one table hit resolves both
 * the symbol and how many bits it actually consumed).
 *
 * Codebook is a PROJECTION-LEVEL SHARED canonical Huffman code (built once
 * from the aggregate histogram of all rows in a projection/band group); rows
 * carry no per-row table, only their own codeword bitstream. Max code length
 * for a 16-symbol alphabet is always <= 15 (n-1 for n leaves), which is why
 * the repo's existing Python reference (tools/measure_expert_entropy.py)
 * stores lengths in a 4-bit field -- no length-limiting pass is needed here
 * either; a plain Huffman build is always decodable with a <=15-bit LUT.
 *
 * Bit order: canonical codes are assigned in the classical sense (shorter
 * codes first, lexicographic tie-break by symbol), then bit-reversed once at
 * build time so they can be streamed through codec_row.h's LSB-first
 * CodecBitW/CodecBitR primitives while still behaving as a normal MSB-first
 * prefix code in actual stream order (see codec_row.h's bit-writer doc
 * comment for why the reversal is required for prefix-freeness to survive
 * LSB-first packing).
 */
#ifndef ILIRIA_CODEC_ROW_HUFF_H
#define ILIRIA_CODEC_ROW_HUFF_H

#include "codec_row.h"

#define HUFF_MAXLEN 15          /* n-1 for a 16-leaf tree: worst case exact */
#define HUFF_LUT_SIZE (1u<<HUFF_MAXLEN)

typedef struct {
    uint8_t  sym;
    uint8_t  len;   /* 0 = unused table slot (only possible if alphabet is empty) */
} HuffLutEnt;

typedef struct {
    uint8_t  len[CODEC_NSYM];       /* canonical code length per symbol, 0=absent */
    uint32_t code_rev[CODEC_NSYM];  /* bit-reversed canonical code, LSB-first-stream form */
    int      maxlen;                /* actual max length used by THIS codebook, <= HUFF_MAXLEN */
    int      n_present;
    HuffLutEnt *lut;                 /* size 1<<maxlen, malloc'd */
} HuffCodebook;

/* ---- length build: standard heap-free approach (alphabet is only 16
 * symbols, so an O(n^2) repeated-min merge is simpler and just as fast as a
 * heap here, and easier to verify against the Python reference's heapq
 * version). Ties broken by picking the earliest-index candidates first,
 * same effective tie behavior as Python's heapq on (count, index) tuples. */
static void huff_build_lengths(const uint32_t counts[CODEC_NSYM], uint8_t len_out[CODEC_NSYM]){
    memset(len_out, 0, CODEC_NSYM);
    /* node[i]: weight + depth-so-far for i<npresent (leaves), merged nodes appended */
    int64_t weight[2*CODEC_NSYM];
    int     leaf_of[2*CODEC_NSYM];   /* -1 for internal nodes */
    int     left[2*CODEC_NSYM], right[2*CODEC_NSYM];
    int     alive[2*CODEC_NSYM];
    int n=0;
    for(int s=0;s<CODEC_NSYM;s++){
        if(counts[s]==0) continue;
        weight[n]=(int64_t)counts[s]; leaf_of[n]=s; left[n]=-1; right[n]=-1; alive[n]=1; n++;
    }
    if(n==0) return;
    if(n==1){ len_out[leaf_of[0]]=1; return; }
    int total_nodes=n;
    int root=-1;
    int remaining=n;
    while(remaining>1){
        int a=-1,b=-1;
        for(int i=0;i<total_nodes;i++){ if(!alive[i]) continue; if(a<0||weight[i]<weight[a]){ b=a; a=i; } else if(b<0||weight[i]<weight[b]) b=i; }
        int nn=total_nodes++;
        weight[nn]=weight[a]+weight[b]; leaf_of[nn]=-1; left[nn]=a; right[nn]=b; alive[nn]=1;
        alive[a]=0; alive[b]=0;
        remaining--;
        root=nn;
    }
    /* iterative depth walk (explicit stack; max depth <=15 anyway but avoid recursion) */
    int stack_node[2*CODEC_NSYM], stack_depth[2*CODEC_NSYM]; int sp=0;
    stack_node[sp]=root; stack_depth[sp]=0; sp++;
    while(sp>0){
        sp--; int node=stack_node[sp], depth=stack_depth[sp];
        if(leaf_of[node]>=0){ len_out[leaf_of[node]] = (uint8_t)depth; continue; }
        stack_node[sp]=left[node]; stack_depth[sp]=depth+1; sp++;
        stack_node[sp]=right[node]; stack_depth[sp]=depth+1; sp++;
    }
}

/* canonical code assignment from lengths (shorter first, tie-break by symbol
 * value ascending -- same convention as tools/measure_expert_entropy.py's
 * canonical_codes) then bit-reversed for LSB-first streaming. */
static void huff_canonical_codes(const uint8_t len[CODEC_NSYM], uint32_t code_rev_out[CODEC_NSYM]){
    int order[CODEC_NSYM], n=0;
    for(int s=0;s<CODEC_NSYM;s++) if(len[s]>0) order[n++]=s;
    /* stable insertion sort by (len, symbol): n<=16, trivial */
    for(int a=1;a<n;a++){ int key=order[a]; int b=a-1;
        while(b>=0 && (len[order[b]]>len[key] || (len[order[b]]==len[key] && order[b]>key))){ order[b+1]=order[b]; b--; }
        order[b+1]=key; }
    uint32_t code=0; int prev_len=0;
    memset(code_rev_out, 0, CODEC_NSYM*sizeof(uint32_t));
    for(int i=0;i<n;i++){
        int s=order[i]; int L=len[s];
        code <<= (L-prev_len);
        code_rev_out[s] = codec_bitrev(code, L);
        code++;
        prev_len=L;
    }
}

static void huff_build(const uint32_t counts[CODEC_NSYM], HuffCodebook *cb){
    huff_build_lengths(counts, cb->len);
    huff_canonical_codes(cb->len, cb->code_rev);
    int maxlen=0, npres=0;
    for(int s=0;s<CODEC_NSYM;s++) if(cb->len[s]>0){ npres++; if(cb->len[s]>maxlen) maxlen=cb->len[s]; }
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
        int L=cb->len[s];
        uint32_t base=cb->code_rev[s];
        int rest = cb->maxlen - L;
        size_t block = (size_t)1u<<rest;
        /* every combination of the top `rest` bits, low L bits fixed to base */
        for(size_t hi=0; hi<block; hi++){
            size_t idx = base | (hi << L);
            cb->lut[idx].sym=(uint8_t)s;
            cb->lut[idx].len=(uint8_t)L;
        }
    }
}
static void huff_free(HuffCodebook *cb){ free(cb->lut); cb->lut=NULL; }

/* stored bits for a row under this codebook (for sizing without encoding) */
static uint64_t huff_row_bits(const HuffCodebook *cb, const uint8_t *sym, int n){
    uint64_t bits=0; for(int i=0;i<n;i++) bits += cb->len[sym[i]];
    return bits;
}

/* encode: returns bytes written (<=cap), or (size_t)-1 if it would overflow cap */
static size_t huff_encode_row(const HuffCodebook *cb, const uint8_t *sym, int n,
                               uint8_t *out, size_t cap){
    CodecBitW w; cbw_init(&w, out, cap);
    for(int i=0;i<n;i++){ uint8_t s=sym[i]; cbw_put(&w, cb->code_rev[s], cb->len[s]); }
    size_t used = cbw_finish(&w);
    if(used > cap) return (size_t)-1;
    return used;
}

/* decode: caller-allocated `sym_out[n]`, no internal allocation (mode-3 hot
 * path discipline -- see codec_row.h's file-level doc comment) */
static void huff_decode_row(const HuffCodebook *cb, const uint8_t *payload, size_t paylen,
                            int n, uint8_t *sym_out){
    CodecBitR r; cbr_init(&r, payload, paylen);
    int maxlen = cb->maxlen;
    for(int i=0;i<n;i++){
        cbr_refill(&r, maxlen);
        uint32_t window = cbr_peek(&r, maxlen);
        HuffLutEnt e = cb->lut[window];
        sym_out[i] = e.sym;
        cbr_drop(&r, e.len);
    }
}

#endif /* ILIRIA_CODEC_ROW_HUFF_H */
