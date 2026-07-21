/* Verifies the mode-3 GEMV integration seam sketch (codec_row_gemv.h)
 * against a direct-dequant reference computation (no codec involved at
 * all) -- i.e. that fusing decode into the multiply-accumulate loop
 * produces the SAME numeric result as today's raw nibble-unpack dequant,
 * for all three drop-in replacement targets (matmul_i4, qt_addrow,
 * qt_matvec_rows). Uses rANS as the concrete codec behind the seam (any of
 * the four would do; rANS is the codec race's own winner among codecs that
 * actually compress -- see codec_row_gemv.h's file doc comment). O is
 * deliberately not a multiple of GEMV_CHUNK to exercise the tail path. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../codec_row.h"
#include "../codec_row_rans.h"
#include "../codec_row_gemv.h"

static int g_fails = 0;
static void fail(const char *msg){ fprintf(stderr, "codec_row_gemv test failed: %s\n", msg); g_fails++; }

static void rans_dec_adapt(const void *cb, const uint8_t *pl, size_t pl_len, int n, uint8_t *out){
    rans_decode_row((const RansCodebook*)cb, pl, pl_len, n, out);
}

static unsigned g_rng = 424242u;
static unsigned xrand(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>17; g_rng^=g_rng<<5; return g_rng; }

int main(void){
    int O = 37, I = 6144, S = 3; /* O not a multiple of GEMV_CHUNK=16: exercises the tail */

    uint8_t *sym = (uint8_t*)malloc((size_t)O*I);
    float *scale = (float*)malloc(sizeof(float)*O);
    for(int o=0;o<O;o++){
        for(int i=0;i<I;i++) sym[(size_t)o*I+i] = (uint8_t)(xrand()%16);
        scale[o] = 0.5f + (xrand()%1000)/1000.0f;
    }
    float *x = (float*)malloc(sizeof(float)*S*I);
    for(int k=0;k<S*I;k++) x[k] = ((int)(xrand()%2000)-1000)/1000.0f;

    /* reference: direct dequant, no codec at all -- same convention as glm.c's qt_addrow */
    float *y_ref = (float*)malloc(sizeof(float)*S*O);
    for(int o=0;o<O;o++)
        for(int s=0;s<S;s++){
            double a=0;
            for(int i=0;i<I;i++) a += (double)((int)sym[(size_t)o*I+i]-8) * x[(size_t)s*I+i];
            y_ref[(size_t)s*O+o] = (float)(a*scale[o]);
        }

    uint32_t counts[16]; memset(counts,0,sizeof(counts));
    for(long k=0;k<(long)O*I;k++) counts[sym[k]]++;
    RansCodebook cb; rans_build(counts, &cb);

    RowOffsets ro; row_offsets_init(&ro, (uint32_t)O);
    size_t cap = (size_t)O*I*2+4096;
    uint8_t *payload = (uint8_t*)malloc(cap);
    size_t pos=0;
    for(int o=0;o<O;o++){
        size_t used = rans_encode_row(&cb, sym+(size_t)o*I, I, payload+pos, cap-pos);
        if(used==(size_t)-1){ fail("encode overflow"); return 1; }
        row_offsets_push(&ro, (uint32_t)used);
        pos += used;
    }

    CodedTensor t;
    t.codebook = &cb; t.decode = rans_dec_adapt; t.payload = payload; t.row_offsets = ro.offsets;

    /* matmul_i4_coded_row_ref vs reference */
    {
        float *y_coded = (float*)malloc(sizeof(float)*S*O);
        matmul_i4_coded_row_ref(y_coded, x, &t, scale, S, I, O);
        double worst_rel = 0;
        for(int k=0;k<S*O;k++){
            double d = fabs((double)y_coded[k]-(double)y_ref[k]);
            double rel = d / (fabs((double)y_ref[k]) > 1e-9 ? fabs((double)y_ref[k]) : 1.0);
            if(rel>worst_rel) worst_rel=rel;
        }
        /* float32 accumulation over I=6144 terms vs. the double-precision
         * reference: a small relative gap is expected precision noise, not
         * a bug (glm.c's real matmul_i4 also accumulates in plain float). */
        if(worst_rel > 1e-3) fail("matmul_i4_coded_row_ref diverges from direct-dequant reference");
        free(y_coded);
    }

    /* qt_addrow_coded_ref vs reference (exact: no cross-i accumulation) */
    {
        float *acc = (float*)calloc((size_t)I, sizeof(float));
        uint8_t *scratch = (uint8_t*)malloc((size_t)I);
        float coef = 0.37f;
        int row = 5;
        qt_addrow_coded_ref(&t, row, I, coef, acc, scratch);
        for(int i=0;i<I;i++){
            double want = coef*((int)sym[(size_t)row*I+i]-8);
            if(fabs(want-(double)acc[i]) > 1e-4){ fail("qt_addrow_coded_ref mismatch"); break; }
        }
        free(acc); free(scratch);
    }

    /* qt_matvec_rows_coded_ref vs reference (both double-accumulated internally) */
    {
        uint8_t *scratch = (uint8_t*)malloc((size_t)I);
        float yv[4];
        int r0 = 10, n = 4;
        qt_matvec_rows_coded_ref(&t, r0, n, I, x, scale, yv, scratch);
        for(int j=0;j<n;j++){
            double a=0;
            for(int i=0;i<I;i++) a += (double)((int)sym[(size_t)(r0+j)*I+i]-8) * x[i];
            double want = a*scale[r0+j];
            if(fabs(want-(double)yv[j]) > 1e-3) fail("qt_matvec_rows_coded_ref mismatch");
        }
        free(scratch);
    }

    free(sym); free(scale); free(x); free(y_ref); free(payload); row_offsets_free(&ro);

    if(g_fails){ fprintf(stderr, "codec_row_gemv: %d failure(s)\n", g_fails); return 1; }
    puts("codec_row_gemv tests: ok");
    return 0;
}
