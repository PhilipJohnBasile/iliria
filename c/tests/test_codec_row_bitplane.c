/* Roundtrip-exactness + fuzz tests for codec_row_bitplane.h (branchless
 * bitplane + rank-indexed exception format). Includes a check that the
 * chosen (base_bits, bias) window is internally consistent (bias+2^k<=16)
 * across every codebook the fuzz loop builds, and an explicit worst-case
 * "everything is an exception" shape (near-uniform distribution) to make
 * sure the rank-gather path is exercised at n_exceptions close to n. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../codec_row.h"
#include "../codec_row_bitplane.h"

static int g_fails = 0;
static void fail(const char *msg){ fprintf(stderr, "codec_row_bitplane test failed: %s\n", msg); g_fails++; }

static unsigned g_rng = 0xABCD1234u;
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
    if(n==0) counts[0]=1;
    BitplaneCodebook cb; bitplane_build(counts, &cb);
    if(cb.base_bits<1 || cb.base_bits>4 || cb.bias<0 ||
       (cb.base_bits<4 && cb.bias+(1<<cb.base_bits)>16)) fail(ctx);
    size_t cap = (size_t)n*4+512;
    uint8_t *out = (uint8_t*)malloc(cap);
    size_t used = bitplane_encode_row(&cb, sym, n, out, cap);
    if(used == (size_t)-1){ fail(ctx); free(out); return; }
    uint8_t *dec = (uint8_t*)malloc(n?(size_t)n:1);
    bitplane_decode_row(&cb, out, used, n, dec);
    if(n>0 && memcmp(sym, dec, n)!=0) fail(ctx);
    free(out); free(dec);
}

static void test_known_vectors(void){
    uint8_t v1[8] = {0,0,0,0,0,0,0,1};
    roundtrip_one(v1, 8, "known skewed vector");
    uint8_t v2[16]; for(int i=0;i<16;i++) v2[i]=(uint8_t)i;
    roundtrip_one(v2, 16, "known one-of-each vector (every value is an exception under any k<4 window)");
    uint8_t v3[1] = {7};
    roundtrip_one(v3, 1, "known singleton vector");
}

/* a genuinely flat (uniform) aggregate histogram must fall back to raw
 * passthrough (base_bits==4), landing at EXACTLY ceil(n/2) stored bytes --
 * not an arbitrary worse-than-raw window. This is the regression case for
 * the bug where the search's running-best defaulted to k=1 and never
 * actually competed against a true raw cost, silently keeping a bad choice
 * whenever no k in {1,2,3} looked good (originally caught by the codec_race
 * benchmark showing a 1.50x stored ratio -- an expansion -- on the near-
 * uniform-ish real synthetic corpus, not by this unit test in isolation). */
static void test_uniform_falls_back_to_raw_passthrough(void){
    uint32_t counts[16]; for(int s=0;s<16;s++) counts[s]=1000;
    BitplaneCodebook cb; bitplane_build(counts, &cb);
    if(cb.base_bits != 4) fail("uniform histogram did not fall back to raw passthrough");
    int n = 6144;
    uint8_t *sym = (uint8_t*)malloc((size_t)n);
    unsigned rr=99; for(int i=0;i<n;i++){ rr^=rr<<13; rr^=rr>>17; rr^=rr<<5; sym[i]=(uint8_t)(rr%16); }
    size_t cap = (size_t)n*4+256;
    uint8_t *out = (uint8_t*)malloc(cap);
    size_t used = bitplane_encode_row(&cb, sym, n, out, cap);
    size_t expect = ((size_t)n+1)/2;
    if(used != expect) fail("raw passthrough did not land at exactly ceil(n/2) bytes");
    uint8_t *dec = (uint8_t*)malloc((size_t)n);
    bitplane_decode_row(&cb, out, used, n, dec);
    if(memcmp(sym,dec,n)!=0) fail("raw passthrough roundtrip mismatch");
    free(sym); free(out); free(dec);
}

static void test_all_exceptions_worst_case(void){
    /* uniform-ish alternating pattern: no small window covers much mass, so
     * most/all positions land outside it -- exercises n_exceptions near n */
    int ns[] = {1,2,7,64,1000,2048,6144};
    for(size_t i=0;i<sizeof(ns)/sizeof(ns[0]);i++){
        int n = ns[i];
        uint8_t *sym = (uint8_t*)malloc((size_t)n);
        for(int j=0;j<n;j++) sym[j]=(uint8_t)(j%16);
        roundtrip_one(sym, n, "all-exceptions worst case");
        free(sym);
    }
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
    test_uniform_falls_back_to_raw_passthrough();
    test_all_exceptions_worst_case();
    test_edge_sizes();
    test_synthetic_shapes_fuzz();
    test_random_length_fuzz();
    if(g_fails){ fprintf(stderr, "codec_row_bitplane: %d failure(s)\n", g_fails); return 1; }
    puts("codec_row_bitplane tests: ok");
    return 0;
}
