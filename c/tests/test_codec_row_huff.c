/* Roundtrip-exactness + fuzz tests for codec_row_huff.h (canonical Huffman
 * + wide LUT decode). Every codec test in this family follows the same
 * shape: known small vectors, edge cases (empty/singleton/boundary-size
 * rows), a battery of synthetic symbol distributions at real row sizes
 * (gate/up ~6144 symbols, down ~2048), and a bounded seeded fuzz loop --
 * "reference" here is the roundtrip itself (decode(encode(x))==x, bit-
 * identical) plus cross-checks against the entropy bound. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../codec_row.h"
#include "../codec_row_huff.h"

static int g_fails = 0;
static void fail(const char *msg){ fprintf(stderr, "codec_row_huff test failed: %s\n", msg); g_fails++; }

static unsigned g_rng = 0xC0FFEEu;
static unsigned xrand(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>17; g_rng^=g_rng<<5; return g_rng; }

static void gen_dist(double p[16], int shape){
    double tot=0;
    for(int i=0;i<16;i++){
        double v;
        if(shape==0) v = 1.0;
        else if(shape==1) v = exp(-((i-7.5)*(i-7.5))/(2*2.0*2.0));
        else if(shape==2) v = (i==7) ? 1000.0 : 1.0;
        else if(shape==3) v = (i==3) ? 1.0 : 0.0;
        else v = (double)(xrand()%1000+1);
        p[i]=v; tot+=v;
    }
    for(int i=0;i<16;i++) p[i]/=tot;
}
static void sample_symbols(const double p[16], int n, uint8_t *out){
    double cdf[17]; cdf[0]=0;
    for(int i=0;i<16;i++) cdf[i+1]=cdf[i]+p[i];
    for(int i=0;i<n;i++){
        double u = (xrand()%1000000)/1000000.0 * cdf[16];
        int s=0; while(s<15 && cdf[s+1]<u) s++;
        out[i]=(uint8_t)s;
    }
}

static void roundtrip_one(const uint8_t *sym, int n, const char *ctx){
    uint32_t counts[16]; codec_histogram(sym, n, counts);
    if(n==0) counts[0]=1; /* codebook always needs something to build on */
    HuffCodebook cb; huff_build(counts, &cb);
    size_t cap = (size_t)n*4+256;
    uint8_t *out = (uint8_t*)malloc(cap);
    size_t used = huff_encode_row(&cb, sym, n, out, cap);
    if(used == (size_t)-1){ fail(ctx); free(out); huff_free(&cb); return; }
    uint8_t *dec = (uint8_t*)malloc(n?(size_t)n:1);
    huff_decode_row(&cb, out, used, n, dec);
    if(n>0 && memcmp(sym, dec, n)!=0) fail(ctx);
    /* stored bits should never exceed 4 bits/symbol (worse than raw would
     * defeat the whole point) and should track the entropy estimate to
     * within a small constant (canonical Huffman's well-known <=1 extra
     * bit/symbol redundancy bound, generously bounded here at +1.05 bits to
     * leave slack for the near-degenerate/skewed synthetic shapes). */
    if(n>0){
        uint64_t bits = huff_row_bits(&cb, sym, n);
        double bits_per_sym = (double)bits / n;
        if(bits_per_sym > 4.0 + 1e-9) fail(ctx);
        double h = codec_entropy_bits(counts);
        if(bits_per_sym > h + 1.05 + 1e-9) fail(ctx);
    }
    free(out); free(dec); huff_free(&cb);
}

static void test_known_vectors(void){
    uint8_t v1[8] = {0,0,0,0,0,0,0,1};       /* heavily skewed: symbol 0 dominant */
    roundtrip_one(v1, 8, "known skewed vector");
    uint8_t v2[16]; for(int i=0;i<16;i++) v2[i]=(uint8_t)i;   /* one of each symbol */
    roundtrip_one(v2, 16, "known one-of-each vector");
    uint8_t v3[1] = {7};
    roundtrip_one(v3, 1, "known singleton vector");
}

static void test_edge_sizes(void){
    int ns[] = {0,1,2,3,4,5,7,8,15,16,17,63,64,65,1023,1024,2047,2048,3071,3072,6143,6144,6145};
    for(size_t i=0;i<sizeof(ns)/sizeof(ns[0]);i++){
        int n=ns[i];
        uint8_t *sym = (uint8_t*)malloc(n?(size_t)n:1);
        for(int j=0;j<n;j++) sym[j]=(uint8_t)(j%16);
        roundtrip_one(sym, n, "edge size, cycling alphabet");
        for(int j=0;j<n;j++) sym[j]=3;
        roundtrip_one(sym, n, "edge size, all-same symbol");
        free(sym);
    }
}

static void test_synthetic_shapes_fuzz(void){
    /* real row sizes: gate/up ~6144 symbols, down ~2048 symbols */
    int row_lens[] = {2048, 6144};
    for(int shape=0; shape<5; shape++){
        for(int ri=0; ri<2; ri++){
            for(int trial=0; trial<25; trial++){
                double p[16]; gen_dist(p, shape);
                int n = row_lens[ri];
                uint8_t *sym = (uint8_t*)malloc((size_t)n);
                sample_symbols(p, n, sym);
                roundtrip_one(sym, n, "synthetic shape fuzz");
                free(sym);
            }
        }
    }
}

static void test_random_length_fuzz(void){
    for(int trial=0; trial<400; trial++){
        double p[16]; gen_dist(p, (int)(xrand()%5));
        int n = 1 + (int)(xrand()%6200);
        uint8_t *sym = (uint8_t*)malloc((size_t)n);
        sample_symbols(p, n, sym);
        roundtrip_one(sym, n, "random-length fuzz");
        free(sym);
    }
}

int main(void){
    test_known_vectors();
    test_edge_sizes();
    test_synthetic_shapes_fuzz();
    test_random_length_fuzz();
    if(g_fails){ fprintf(stderr, "codec_row_huff: %d failure(s)\n", g_fails); return 1; }
    puts("codec_row_huff tests: ok");
    return 0;
}
