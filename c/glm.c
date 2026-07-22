/* Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE. */
/* Motore GLM-5.2 (architettura glm_moe_dsa) in C puro.
 * Stadio B: replica fedele del forward di transformers (modeling_glm_moe_dsa.py):
 *   - attenzione MLA (q/kv-LoRA, RoPE interleaved parziale)
 *   - router sigmoid + noaux_tc (n_group=1) con routed_scaling_factor
 *   - shared expert + expert routed in streaming dal disco (per-expert)
 *   - primi first_k_dense_replace layer densi
 * Il DSA indexer e' un NO-OP per seq <= index_topk (seleziona tutte le key): qui si usa
 * attenzione causale densa -> output identico all'oracolo su prompt corti.
 *
 * QUANTIZZAZIONE: gli expert (streaming) e la parte DENSA residente (attenzione, lm_head,
 * embed, mlp densa, shared expert) sono tenuti in int8 per-riga + scala (dequant-on-use).
 * E' cio' che fa entrare GLM-5.2 nei 15 GB: ~17B param residenti a int4 ~= 8.7 GB.
 * Norme/router/bias restano f32 (piccoli e sensibili).
 *
 * Validazione: stessi token id di ref_glm.json (oracolo transformers, c/tools/make_glm_oracle.py).
 *   build: make glm   run: SNAP=./glm_tiny ./glm <cap> <expert_bits> <dense_bits>
 *   TF=1 -> teacher-forcing (valida il prefill su tutta la sequenza)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <signal.h>                               /* fail-loud crash handler (SIGSEGV/SIGBUS/SIGABRT/SIGILL/SIGFPE) */
#include <pthread.h>                              /* thread I/O del PILOTA */
#include <stdatomic.h>                            /* PIPE ready-flags/job queue + PILOT_REAL cross-layer handshake */
#include <sched.h>                                /* sched_yield: PIPE spin / PILOT barrier */
#include <unistd.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#include <sys/mman.h>                             /* mlock: inchioda le pagine in RAM / wire pages into RAM */
#include <sys/stat.h>                             /* fstat per mmap degli shard (ILI_MMAP) */
#endif
#include "st.h"
#include "tok.h"
#include "tier.h"
#include "grammar.h"                              /* metodo F: draft grammaticali (#48) */
/* Legacy build-flag compatibility: -DCOLI_CUDA / -DCOLI_METAL still select the backends. */
#if defined(COLI_CUDA) && !defined(ILI_CUDA)
#define ILI_CUDA COLI_CUDA
#endif
#if defined(COLI_METAL) && !defined(ILI_METAL)
#define ILI_METAL COLI_METAL
#endif
/* Env lookup with silent legacy fallback: ILI_<name> > COLI_<name> > FA_<name>.
 * Canonical spelling is ILI_*; the legacy prefixes keep old setups working. */
static const char *ili_env(const char *name){
    char k[96]; const char *v;
    snprintf(k,sizeof k,"ILI_%s",name); if((v=getenv(k))) return v;
    snprintf(k,sizeof k,"COLI_%s",name); if((v=getenv(k))) return v;
    snprintf(k,sizeof k,"FA_%s",name);   return getenv(k);
}
#ifdef _OPENMP
#include <omp.h>                                   /* per-thread attention scratch */
#else
static inline int omp_get_max_threads(void){ return 1; }
static inline int omp_get_thread_num(void){ return 0; }
#endif
#ifdef ILI_CUDA
#include "backend_cuda.h"
#endif
#ifdef ILI_METAL
#include "backend_metal.h"
static int g_metal_enabled;
static int g_metal_gemm_min=16;   /* ILI_METAL_GEMM_MIN: min rows to send a matmul_qt GEMM to GPU */
static int g_metal_prefill=0;     /* ILI_METAL_PREFILL=1: S>4 batch attention on GPU (the prefill-I/O study) */
static int g_mm_forcecpu=0;       /* force the CPU path in matmul_qt: the S-row prefill projections must stay
                                   * byte-exact with the historical per-token loop unless ILI_METAL_PREFILL=1
                                   * (Metal GEMM rounding forks greedy output on the standard 3 prompts) */
/* routing precalcolata dalla GPU (layer CB): moe() la usa e salta la FASE A */
static const int *g_pre_idx; static const float *g_pre_w; static const int *g_pre_keff;
static const float *g_pre_sh;   /* output dello shared expert gia' calcolato su GPU */
#endif
#ifdef ILI_MODE15
/* Mode-1.5 (Huffman-compressed expert-tensor container) CPU decode support --
 * OPT-IN (mirrors ILI_CUDA/ILI_METAL above): a build WITHOUT -DILI_MODE15 never
 * sees these symbols at all, so the fail-loud guard below (mode15_blob()/
 * mode15_unsupported(), unconditionally present) remains the ONLY behavior on an
 * "MH01" tensor -- exactly today's shipped behavior, byte-for-byte. See
 * the Mode-15 integration design notes step 4 ("Wire CPU decode into
 * expert_load ... correctness milestone, not a performance one") -- this is that
 * step. codec_row.h/codec_row_huff.h are header-only `static`-function modules
 * (safe to include unconditionally like st.h/json.h); mode15_reader.h/.c is
 * split declaration/definition (matching backend_cuda.h/backend_metal.h's own
 * style per that header's file comment) -- there is no separate Makefile object
 * for it today, so its .c is pulled in here the same way tests/test_idot.c
 * already pulls in the whole of glm.c: one translation unit, no new link step. */
#include "codec_row.h"
#include "codec_row_huff.h"
#include "mode15_reader.h"
#include "mode15_reader.c"
#endif
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){            /* somma orizzontale di 8 float */
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>                             /* Apple Silicon / aarch64: kernel NEON */
#elif defined(__VSX__)
#include <altivec.h>                              /* POWER8+ (ppc64le): kernel VSX */
#undef vector                                     /* igiene: si usano __vector/__bool espliciti */
#undef pixel
#undef bool
#endif
#ifdef __APPLE__
#include <mach/mach.h>                            /* host_statistics64: MemAvailable di macOS */
#endif

typedef struct {
    int hidden, n_layers, n_heads, n_experts, topk, moe_inter, dense_inter;
    int first_dense, q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head, n_shared, vocab;
    int n_group, topk_group, norm_topk;
    int stop_ids[8], n_stop;                     /* eos_token_id dal config (GLM-5.2 ne ha 3!) */
    int index_topk, index_nh, index_hd;          /* DSA lightning indexer */
    int8_t idx_type[128];                        /* per layer: 1=full (calcola), 0=shared (riusa) */
    float eps, theta, attn_scale, routed_scale;
} Cfg;

/* tensore [O,I] in uno di tre formati:
 *   fmt=0 F32   -> qf
 *   fmt=1 INT8  -> q8 (1 byte/param) + scala per riga
 *   fmt=2 INT4  -> q4 (2 valori per byte, impacchettati) + scala per riga
 * INT4 e' cio' che fa stare la densa residente nei 15 GB (0.5 byte/param). */
/* fmt: 0 F32, 1 INT8, 2 INT4 (2/byte), 3 INT2 (4/byte). q4 ospita sia int4 che int2 packed. */
typedef struct {
    int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I;
#ifdef ILI_CUDA
    IliCudaTensor *cuda;
#endif
    int cuda_eligible, cuda_failed, cuda_device;  /* resident tensor, never a reused expert slot */
} QT;
static int64_t qt_bytes(const QT *t){    /* byte residenti del tensore */
    int64_t n=(int64_t)t->O*t->I;
    if(t->fmt==0) return n*4;
    if(t->fmt==1) return n + (int64_t)t->O*4;
    if(t->fmt==3) return (int64_t)t->O*((t->I+3)/4) + (int64_t)t->O*4;
    return (int64_t)t->O*((t->I+1)/2) + (int64_t)t->O*4;
}

typedef struct {
    float *in_ln, *post_ln;
    /* MLA (densa, quantizzata) */
    QT q_a, q_b, kv_a, kv_b, o; float *q_a_ln, *kv_a_ln;
    int sparse;
    /* dense mlp (sparse==0) */
    QT gate_proj, up_proj, down_proj;
    /* moe (sparse==1) */
    float *router, *router_bias;                 /* router f32 (sensibile) */
    QT sh_gate, sh_up, sh_down;                  /* shared expert */
} Layer;

/* slot di un expert: pesi quantizzati + scale. Nel container pre-quantizzato g/u/d sono
 * VISTE dentro `slab` (una sola pread coalescente); nel fallback hanno buffer propri.
 * slab_cap/fslab_cap: capienza allocata — gli slot ws[] sono riusati TRA layer e gli
 * expert non hanno tutti la stessa taglia (layer MTP int8 = 2x i layer int4). */
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used; } ESlot;

typedef struct {
    float **Lc, **Rc, **Ic;
    int *kv_start, max_t;
    int disk_nrec;
    char disk_path[2048];
} KVState;

typedef struct {
    Cfg c; shards S;
    int ebits, dbits;                            /* bit expert / bit densa */
    QT embed, lm_head; float *final_norm;
    Layer *L;
    /* KV-cache MLA COMPRESSA: per token si tiene solo il latente normato [kv_lora] e
     * k_rot [qk_rope] (576 vs 32768 valori/token). k_nope e value si ricostruiscono al
     * volo con kv_b. E' cio' che rende gestibile il contesto su 15 GB (64 teste, no GQA). */
    float **Lc, **Rc; int max_t;                 /* alias della KVState attiva */
    int *kv_start;                               /* prima pos valida nella KV del layer (MTP: parziale) */
    KVState *kv;
    ESlot **ecache; int *ecn; int ecap;          /* LRU expert per-layer */
    ESlot ws[64];                                /* working set del layer corrente (load paralleli) */
    ESlot **pin; int *npin;                      /* HOT-STORE: expert pinnati in RAM (mai evicted) */
    uint32_t **eusage;                           /* contatori persistenti (per STATS/PIN) */
    uint32_t **eheat;                            /* calore recente per promotion/demotion live */
    /* DSA lightning indexer (attivo solo se i pesi out-idx-* sono presenti) */
    int has_dsa;
    QT *ix_wq, *ix_wk, *ix_wp;                   /* per layer FULL: wq_b, wk, weights_proj */
    float **ix_knw, **ix_knb;                    /* k_norm (LayerNorm, eps 1e-6) */
    float **Ic;                                  /* alias KVState: cache indexer [max_t*hd] */
    int *dsa_sel, *dsa_nsel; int dsa_scap;       /* selezione per posizione del batch corrente */
    /* testa MTP (layer n_layers, stile DeepSeek-V3): draft nativi ad alta acceptance */
    int has_mtp; Layer mtpL; QT eh_proj;
    float *enorm, *hnorm, *mtp_norm;
    float *hlast, *h_all;                        /* hidden pre-norm: ultima pos / tutte le pos batch */
    uint64_t mtp_prop, mtp_acc;                  /* statistica acceptance */
    int **eroute; int *enr;                      /* metodo C: routing dell'ULTIMO token per layer */
    uint64_t eclock, hits, miss, ereq;
    uint64_t gpu_expert_calls; int gpu_expert_count; int64_t gpu_expert_bytes;
    uint64_t n_fw, n_emit;                       /* metodo E: forward di decode / token emessi */
    double t_edisk, t_emm, t_attn, t_kvb, t_head;/* profiling: dove va il tempo (sempre attivo) */
    int64_t resident_bytes;
    /* ---- streaming-causality instrumentation (c/bench-m5max/factorial-streaming-causality-
     * the format spec). t_stall_exposed is CONSUMER-BLOCKED CRITICAL-PATH time only -- separate from
     * t_edisk above (unchanged service-time series, still reported on its own line). See
     * pipe_wait_timed() near the PIPE pool. */
    double t_stall_exposed;
    int64_t io_bytes_requested, io_bytes_read;     /* expert-fetch path, absolute, whole run */
    uint64_t io_reads_attempted, io_reads_completed;
    /* [IOKIND] Step-0 diagnostic (#1438 deep-offload reconciliation): io_bytes_read/
     * io_reads_completed above stay byte-identical (blended weight+scale) -- these are an
     * ADDITIVE per-tensor-kind split, main (non-mmap, non-mode15) expert_load() path only.
     * See io_kind_done() near io_read_done() for the accounting. */
    int64_t io_bytes_weight, io_bytes_scale;
    uint64_t io_reads_weight, io_reads_scale;
    uint64_t n_pipe_waits, n_pipe_waits_blocked;   /* occupancy proxy: waits that found ready==0 */
} Model;

static void usage_save(Model *m);        /* cache che impara: definita accanto a stats_dump */

/* Workload-specific decayed hotsets (Phase 2).
 * ILI_HOTSET_PROFILE selects a named usage file (.fa_usage.<profile>).
 * ILI_HOTSET_DECAY enables multiplicative decay (e.g., 98 = 0.98).
 * ILI_HOTSET_DECAY_INTERVAL applies decay after N cumulative selections. */
static int g_hotset_decay_factor = 0;    /* 0 = disabled, 1-99 = percentage factor */
static int64_t g_hotset_decay_interval = 0; /* selections between decay applications */
static int64_t g_hotset_selections = 0;   /* cumulative selections since last decay */
static char g_hotset_profile[64] = "";    /* validated profile name */

#ifdef ILI_CUDA
static int g_cuda_enabled;
static double g_cuda_expert_gb;
static int g_cuda_dense;
static int g_cuda_devices[ILI_CUDA_MAX_DEVICES], g_cuda_ndev, g_cuda_rr;
static int64_t g_cuda_dense_projected[ILI_CUDA_MAX_DEVICES];
static void qt_cuda_reset(QT *t){
    if(t->cuda){ ili_cuda_tensor_free(t->cuda); t->cuda=NULL; }
    t->cuda_failed=0;
}
static int qt_cuda_upload(QT *t){
    const void *weights = t->fmt==0 ? (const void*)t->qf
                        : t->fmt==1 ? (const void*)t->q8 : (const void*)t->q4;
    return ili_cuda_tensor_upload(&t->cuda,weights,t->s,t->fmt,t->I,t->O,t->cuda_device);
}
static void cuda_stats_print(void){
    size_t n=0,b=0; ili_cuda_stats(-1,&n,&b);
    fprintf(stderr,"[CUDA] resident set: %zu tensors, %.2f GB VRAM\n",n,b/1e9);
    if(g_cuda_ndev>1) for(int i=0;i<g_cuda_ndev;i++){
        ili_cuda_stats(g_cuda_devices[i],&n,&b);
        fprintf(stderr,"[CUDA]   device %d: %zu tensors, %.2f GB\n",g_cuda_devices[i],n,b/1e9);
    }
}
static int parse_cuda_devices(const char *list, int *out){
    if(!list||!*list) return 0;
    int n=0; const char *p=list;
    while(*p){
        char *end=NULL; long v=strtol(p,&end,10);
        if(end==p||v<0||v>INT_MAX||n>=ILI_CUDA_MAX_DEVICES) return 0;
        for(int i=0;i<n;i++) if(out[i]==(int)v) return 0;
        out[n++]=(int)v; p=end;
        while(*p==' '||*p=='\t') p++;
        if(!*p) break;
        if(*p++!=',') return 0;
        while(*p==' '||*p=='\t') p++;
        if(!*p) return 0;
    }
    return n;
}
#endif
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);   /* macOS: ru_maxrss in BYTE */
#else
    return r.ru_maxrss/(1024.0*1024.0);          /* Linux: in KB */
#endif
}
static float *falloc(int64_t n){
    /* guardia anti-wrap (report PR #25): n assurdo da file modello ostili non deve
     * diventare una malloc piccola. Niente calloc: il memset nel percorso caldo costa. */
    if(n<0 || (uint64_t)n > SIZE_MAX/sizeof(float)){ fprintf(stderr,"falloc: n=%lld is out of range\n",(long long)n); exit(1); }
    float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }

/* y[S,O] = x[S,I] @ W^T, W[O,I] f32 */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}
/* y[S,O] = x[S,I] @ W^T con W quantizzato int8 per-riga + scala[O] (dequant-on-use) */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            __m256 acc=_mm256_setzero_ps();
            for(;i+8<=I;i+=8){ __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w+i)));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), _mm256_cvtepi32_ps(wi), acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+8<=I;i+=8){ int16x8_t w16=vmovl_s8(vld1_s8(w+i));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),   vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++) a+=xs[i]*(float)w[i]; y[(int64_t)s*O+o]=a*sc; } }
}
/* y[S,O] = x[S,I] @ W^T con W int4 impacchettato (2 valori/byte) + scala[O]. */
static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));   /* 8 byte=16 nibble */
                __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i nib=_mm_unpacklo_epi8(lo,hi);                                       /* nibble in ordine */
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m4=vdup_n_u8(0x0F); const int8x8_t b8=vdup_n_s8(8);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint8x8_t by=vld1_u8(w+(i>>1));               /* 8 byte=16 nibble */
                uint8x8x2_t z=vzip_u8(vand_u8(by,m4), vshr_n_u8(by,4));        /* nibble in ordine */
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]),b8));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]),b8));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi; }
            if(i<I){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8; a += xs[i]*(float)lo; }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* y[S,O] = x[S,I] @ W^T con W int2 impacchettato (4 valori/byte) + scala[O]. nibble 2-bit -> [-2,1]. */
static void matmul_i2(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O){
    int rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q2+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m2=_mm_set1_epi8(0x03); const __m256i b2=_mm256_set1_epi32(2);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_cvtsi32_si128(*(const int*)(w+(i>>2)));    /* 4 byte=16 valori */
                __m128i p0=_mm_and_si128(by,m2), p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
                __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2), p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
                __m128i lo=_mm_unpacklo_epi8(p0,p1), hi=_mm_unpacklo_epi8(p2,p3);
                __m128i nib=_mm_unpacklo_epi16(lo,hi);                                      /* 16 valori in ordine */
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b2));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b2));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m2v=vdup_n_u8(3); const int8x8_t b2v=vdup_n_s8(2);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint32_t wd; memcpy(&wd, w+(i>>2), 4);        /* 4 byte=16 valori */
                uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v),              vand_u8(vshr_n_u8(by,2),m2v));
                uint8x8x2_t z23=vzip_u8(vand_u8(vshr_n_u8(by,4),m2v), vshr_n_u8(by,6));
                uint16x4x2_t zz=vzip_u16(vreinterpret_u16_u8(z01.val[0]), vreinterpret_u16_u8(z23.val[0]));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[0]),b2v));  /* 16 valori in ordine */
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[1]),b2v));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++){ uint8_t byte=w[i>>2]; int sh=(i&3)*2; a += xs[i]*(float)((int)((byte>>sh)&3)-2); }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* ---- KERNEL INTERI (IDOT): attivazioni quantizzate a int8 per riga (absmax/127,
 * stile Q8_0), prodotto scalare INTERO via maddubs/madd AVX2 — niente conversione
 * f32 dei pesi nel ciclo caldo. ~2-3x sui matmul quantizzati; errore aggiunto ~0.3%
 * RMS per matmul (attivazione int8), IDOT=0 torna al percorso f32 esatto. */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#define IDOT_KERNEL "avx512-vnni"
#elif defined(__AVX2__)
#define IDOT_KERNEL "avx2"
#elif defined(__ARM_NEON)
#define IDOT_KERNEL "neon"
#elif defined(__VSX__)
#define IDOT_KERNEL "vsx"
#else
#define IDOT_KERNEL "scalar"
#endif
static int g_idot=1;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int g_i4s=1;   /* SDOT presente: int4 IDOT conviene anche a S=1 (decode). Misurato
                       * su Apple M-series: +14%%, expert-matmul -16%%. EN: with SDOT, int4
                       * IDOT pays even at S=1 (decode); measured on Apple M-series. */
#elif defined(__VSX__)
static int g_i4s=1;   /* POWER8 vec_msum: qui il fallback f32 e' SCALARE, quindi l'IDOT
                       * int4 conviene anche a S=1. Misurato su POWER8 S824 (vedi PR).
                       * EN: on VSX the f32 fallback is plain scalar C, so int4 IDOT
                       * pays even at S=1. Measured on a POWER8 S824 (see PR). */
#else
static int g_i4s=2;   /* senza SDOT / altrove: soglia originale (misura AVX2 dell'autore).
                       * EN: without SDOT / elsewhere: original threshold (author's AVX2). */
#endif
static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}
#ifdef __AVX2__
static inline int hsum256_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
#endif
/* dot int8·int8: trucco del segno (|w| unsigned × x·sign(w) signed). Sicuro:
 * coppie <= 128*127*2 = 32512 < 32767, accumulo s32 fino a I=16384. */
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* VNNI: vpdpbusd u8*s8 -> s32 directly, 64 bytes/iter, no 16-bit intermediate.
     * AVX-512 has no vpsignb: |w| via abs, sign folded into x with a mask-negate
     * (w==0 -> product 0 either way). |x|<=127 (qrow_i8), |w|<=128 as u8: each
     * s32 lane adds <= 4*128*127, safe up to I=16384 like the AVX2 bound. */
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m512i wv=_mm512_loadu_si512((const void*)(w+i));
        __m512i xv=_mm512_loadu_si512((const void*)(x+i));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVX2__)
    __m256i acc=_mm256_setzero_si256(); const __m256i ones=_mm256_set1_epi16(1);
    for(;i+32<=I;i+=32){
        __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    /* ARM: SDOT nativo se disponibile (Apple Silicon: sempre); altrimenti vmull/vpadal.
     * Stesso bound anti-overflow del trucco AVX2: coppie <= 128*127*2 = 32512 < 32767. */
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+16<=I;i+=16){
        int8x16_t wv=vld1q_s8(w+i), xv=vld1q_s8(x+i);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,wv,xv);
#else
        int16x8_t p=vmull_s8(vget_low_s8(wv),vget_low_s8(xv));
        p=vmlal_s8(p,vget_high_s8(wv),vget_high_s8(xv));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#elif defined(__VSX__)
    /* POWER8: vec_msum (s8 x u8 -> s32) somma i prodotti byte DIRETTAMENTE in lane
     * s32, 16 byte/iter: il bound anti-saturazione a 16 bit di maddubs qui non serve.
     * Stesso trucco del segno (|w| u8 per x*sign(w) s8), ma |w| via select+sub MODULO
     * e non vec_abs: -128 deve diventare 128 u8, non saturare a 127.
     * EN: vec_msum accumulates byte products straight into s32 lanes; |w| is built
     * with a modulo subtract select instead of vec_abs so w=-128 wraps to 128 (u8)
     * rather than saturating to 127. |x|<=127 from qrow_i8, so x negation is safe. */
    __vector signed int acc=vec_splats(0);
    const __vector signed char vz=vec_splats((signed char)0);
    for(;i+16<=I;i+=16){
        __vector signed char wv=vec_xl(0,(const signed char*)(w+i));
        __vector signed char xv=vec_xl(0,(const signed char*)(x+i));
        __vector __bool char neg=vec_cmplt(wv,vz);
        __vector signed char xs=vec_sel(xv,vec_sub(vz,xv),neg);
        __vector unsigned char wa=(__vector unsigned char)vec_sel(wv,vec_sub(vz,wv),neg);
        acc=vec_msum(xs,wa,acc);
    }
    sum=vec_extract(acc,0)+vec_extract(acc,1)+vec_extract(acc,2)+vec_extract(acc,3);
#endif
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}
/* dot int4(packed)·int8: nibble -> int8 [-8,7] al volo, poi stesso trucco */
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* 32 bytes = 64 nibbles -> int8 in [-8,7], one vpdpbusd per 64 values.
     * 256-bit unpack leaves values in per-128-lane order [0-15][32-47]/[16-31][48-63];
     * dot pairing is order-invariant, so permute x's 128-bit blocks to match
     * instead of re-ordering w (one vpermq per iter, off the critical unpack path). */
    const __m256i m4v=_mm256_set1_epi8(0x0F);
    const __m512i b8v=_mm512_set1_epi8(8);
    const __m512i xidx=_mm512_setr_epi64(0,1,4,5,2,3,6,7);
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        __m256i z0=_mm256_unpacklo_epi8(lo,hi), z1=_mm256_unpackhi_epi8(lo,hi);
        __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(z0),z1,1),b8v);
        __m512i xv=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i)));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVX2__)
    const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi8(8);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));   /* 16 byte = 32 nibble */
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);   /* in ordine */
        __m256i wv=_mm256_sub_epi8(_mm256_set_m128i(n1,n0),b8);
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));                          /* 16 byte = 32 nibble */
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4)); /* nibble in ordine */
        int8x16_t w0=vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q);
        int8x16_t w1=vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q);
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,w0,x0); acc=vdotq_s32(acc,w1,x1);
#else
        int16x8_t p=vmull_s8(vget_low_s8(w0),vget_low_s8(x0));      /* |w|<=8: nessun overflow */
        p=vmlal_s8(p,vget_high_s8(w0),vget_high_s8(x0));
        acc=vpadalq_s16(acc,p);
        p=vmull_s8(vget_low_s8(w1),vget_low_s8(x1));
        p=vmlal_s8(p,vget_high_s8(w1),vget_high_s8(x1));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#elif defined(__VSX__)
    /* 16 byte = 32 nibble. vec_mergeh/vec_mergel su ppc64le (GCC) interallacciano come
     * unpacklo/unpackhi x86 (verificato empiricamente su POWER8): i nibble escono in
     * ordine di memoria. |w|<=8 dopo il -8, quindi stesso trucco del segno di dot_i8i8.
     * EN: vec_mergeh/l on ppc64le interleave like x86 unpacklo/hi (verified on POWER8),
     * so nibbles come out in memory order; then the same sign trick as dot_i8i8. */
    const __vector unsigned char m4v=vec_splats((unsigned char)0x0F);
    const __vector unsigned char sh4=vec_splats((unsigned char)4);
    const __vector signed char b8v=vec_splats((signed char)8);
    const __vector signed char vz=vec_splats((signed char)0);
    __vector signed int acc=vec_splats(0);
    for(;i+32<=I;i+=32){
        __vector unsigned char by=vec_xl(0,w4+(i>>1));               /* 16 byte = 32 nibble */
        __vector unsigned char lo=vec_and(by,m4v), hi=vec_sr(by,sh4);
        __vector signed char w0=vec_sub((__vector signed char)vec_mergeh(lo,hi),b8v);
        __vector signed char w1=vec_sub((__vector signed char)vec_mergel(lo,hi),b8v);
        __vector signed char x0=vec_xl(0,(const signed char*)(x+i));
        __vector signed char x1=vec_xl(0,(const signed char*)(x+i+16));
        __vector __bool char n0=vec_cmplt(w0,vz), n1=vec_cmplt(w1,vz);
        acc=vec_msum(vec_sel(x0,vec_sub(vz,x0),n0),
                     (__vector unsigned char)vec_sel(w0,vec_sub(vz,w0),n0),acc);
        acc=vec_msum(vec_sel(x1,vec_sub(vz,x1),n1),
                     (__vector unsigned char)vec_sel(w1,vec_sub(vz,w1),n1),acc);
    }
    sum=vec_extract(acc,0)+vec_extract(acc,1)+vec_extract(acc,2)+vec_extract(acc,3);
#endif
    for(;i+1<I;i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}
static void matmul_q_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                          const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i8i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
static void matmul_i4_idot(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                           const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}

typedef struct { int8_t *xq; size_t xq_cap; float *sx; size_t sx_cap; } QScratch;
static _Thread_local QScratch g_qscratch;
static void quant_scratch(size_t xn, size_t sn, int8_t **xq, float **sx){
    if(xn>g_qscratch.xq_cap){
        int8_t *p=realloc(g_qscratch.xq,xn);
        if(!p){ fprintf(stderr,"OOM quant scratch\n"); exit(1); }
        g_qscratch.xq=p; g_qscratch.xq_cap=xn;
    }
    if(sn>g_qscratch.sx_cap){
        float *p=realloc(g_qscratch.sx,sn*sizeof(float));
        if(!p){ fprintf(stderr,"OOM quant scales\n"); exit(1); }
        g_qscratch.sx=p; g_qscratch.sx_cap=sn;
    }
    *xq=g_qscratch.xq; *sx=g_qscratch.sx;
}

static void matmul_qt(float *y, const float *x, QT *w, int S){
#ifdef ILI_METAL
    /* Large row-batches (prefill: kv_b reconstruction, o_proj, dense MLP, step_all logits)
     * amortize Metal's ~5ms submit latency; small-S decode matmuls stay on CPU (NEON wins).
     * Weights must be registered (all dense QT allocs are, via qalloc). */
    if(g_metal_enabled && !g_mm_forcecpu && S>=g_metal_gemm_min && (w->fmt==1||w->fmt==2) && !omp_in_parallel()){
        const void *wp = w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(ili_metal_gemm(y,x,wp,w->s,w->fmt,S,w->I,w->O)) return;
    }
#endif
#ifdef ILI_CUDA
    /* The CUDA backend owns persistent copies only for model-resident tensors.
     * Streaming expert slots are reused for different IDs and must never enter
     * this cache. Nested OpenMP calls stay on CPU because each device context
     * intentionally owns one synchronous scratch stream in this stage. */
    if(g_cuda_enabled && w->cuda_eligible && !w->cuda_failed && !omp_in_parallel()){
        const void *weights = w->fmt==0 ? (const void*)w->qf
                            : w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(ili_cuda_matmul(&w->cuda,y,x,weights,w->s,w->fmt,S,w->I,w->O,w->cuda_device)) return;
        w->cuda_failed=1;
        fprintf(stderr,"[CUDA] tensor [%d,%d] on device %d disabled after an error; falling back to CPU\n",
            w->O,w->I,w->cuda_device);
    }
#endif
    if(w->fmt==0){ matmul(y,x,w->qf,S,w->I,w->O); return; }
    /* int8 IDOT vince sempre (1.4-2.5x). int4 IDOT: l'autore su AVX2 trovo' che a S=1
     * non ripaga (soglia S>=2); ma su ARM/SDOT il singolo token CONVIENE (vedi g_i4s /
     * PR #9 per il gemello VNNI). Soglia configurabile con I4S.
     * EN: int8 IDOT always wins (1.4-2.5x). int4 IDOT: on AVX2 the author found S=1 didn't
     * pay (S>=2 gate); on ARM/SDOT single-token DOES pay (see g_i4s / PR #9 for the VNNI
     * twin). Threshold configurable via I4S. */
    if(g_idot && (w->fmt==1 || (w->fmt==2 && S>=g_i4s))){
        int I=w->I; int8_t *xq; float *sx;
        if(S<0 || I<0 || (size_t)S>SIZE_MAX/(size_t)(I?I:1)){ fprintf(stderr,"matmul_qt: shape overflow\n"); exit(1); }
        quant_scratch((size_t)S*I,(size_t)S,&xq,&sx);
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        if(w->fmt==1) matmul_q_idot(y,xq,sx,w->q8,w->s,S,I,w->O);
        else matmul_i4_idot(y,xq,sx,w->q4,w->s,S,I,w->O);
        return;
    }
    if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==3) matmul_i2(y,x,w->q4,w->s,S,w->I,w->O);
    else matmul_i4(y,x,w->q4,w->s,S,w->I,w->O);
}

/* quantizza w[O,I] f32 -> int8 q[O,I] + scala[O] simmetrica per riga */
static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        int8_t *qr=q+(int64_t)o*I;
        for(int i=0;i<I;i++){ int v=(int)lrintf(wr[i]/s); if(v>qmax)v=qmax; if(v<-qmax-1)v=-qmax-1; qr[i]=(int8_t)v; }
    }
}
/* quantizza w[O,I] f32 -> int4 impacchettato (2/byte) + scala[O].
 * bits<=4: valori in [-qmax-1,qmax] stanno in un nibble [-8,7]; memorizzati come v+8 (0..15). */
static void pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1, rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        uint8_t *qr=q4+(int64_t)o*rb;
        for(int i=0;i<I;i+=2){
            int v0=(int)lrintf(wr[i]/s); if(v0>qmax)v0=qmax; if(v0<-8)v0=-8;
            int v1=0; if(i+1<I){ v1=(int)lrintf(wr[i+1]/s); if(v1>qmax)v1=qmax; if(v1<-8)v1=-8; }
            qr[i>>1] = (uint8_t)((v0+8) | ((v1+8)<<4));
        }
    }
}

/* quantizza w[O,I] f32 -> int2 impacchettato (4/byte) + scala[O]. valori nibble 2-bit in [-2,1]. */
static void pack_int2(const float *w, uint8_t *q2, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1, rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        uint8_t *qr=q2+(int64_t)o*rb;
        for(int i=0;i<I;i+=4){ uint8_t byte=0;
            for(int k=0;k<4 && i+k<I;k++){ int v=(int)lrintf(wr[i+k]/s); if(v>qmax)v=qmax; if(v<-2)v=-2; byte|=(uint8_t)((v+2)<<(k*2)); }
            qr[i>>2]=byte;
        }
    }
}

static int g_nopack=0;   /* NOPACK=1 -> tiene i valori <=4bit in contenitore int8 (per validare il packing) */
static int g_drop=0;     /* DROP=1 -> scarta le pagine expart dopo l'uso. Default 0: le lascia in
                          * page-cache (buff/cache, NON RSS) come L2 gratuito -> sfrutta lo
                          * sbilanciamento del routing MoE (pochi expert "caldi" riusati). */
static int g_prefetch=0; /* PREFETCH=1 -> riabilita il WILLNEED cross-layer (metodo C). Default
                          * OFF: i load VERI in parallelo lo hanno reso superfluo, e sotto
                          * pressione di memoria il readahead speculativo veniva rievictato. */
static int g_direct=0;   /* DIRECT=1 -> O_DIRECT sugli slab expert. Default OFF: su questo host
                          * (VHDX su NVMe DRAM-less, latenza serializzata ~60ms/req) il buffered
                          * liscio e' risultato il migliore; su NVMe veri DIRECT=1 rende di piu'. */
static float g_temp=-1;  /* TEMP: temperatura di sampling sui TOKEN. <0 = auto (1.0 in chat/testo,
                          * 0=greedy in validazione). 0 = greedy puro. */
static float g_nuc=0.95f;/* NUCLEUS: top-p sul vocabolario (default dal generation_config GLM-5.2) */
static int g_topk=0;     /* TOPK=n -> usa n expert/token invece di config (ricerca: meno disco) */
static float g_topp=0;   /* TOPP=p (0..1) -> top-p adattivo: tieni gli expert fino a peso cumulato p */
static int g_spec=1;     /* metodo C: SPEC=0 disabilita il prefetch speculativo cross-layer */
static int g_draft=0;    /* metodo E: DRAFT=n token auto-speculati per forward via n-gram lookup
                          * (0=off). LOSSLESS: verifica = output identico al greedy. Default OFF:
                          * misurato sul run reale (2026-07-03) acceptance ~5% -> ogni draft
                          * rifiutato paga comunque i suoi expert dal disco = ~3x piu' lento.
                          * Opt-in (DRAFT=4) per testi ripetitivi dove l'acceptance e' alta. */
/* metodo F (#48): GRAMMAR=<file.gbnf> -> terza sorgente di draft, la grammatica stessa.
 * Nei workload a output vincolato (JSON/NDJSON, function calling) i byte FORZATI dalla
 * grammatica (chiavi, punteggiatura, valori enum) sono draft gratuiti ad acceptance ~1:
 * nessuna testa, nessuna lookup table, e si aggancia anche dove la testa MTP int4 non
 * parte (#8). MAI un vincolo sul sampling: solo proposte, la verifica batch-union
 * decide — grammatica sbagliata = draft rifiutati, output identico.
 * GRAMMAR_DRAFT=n (default 24) limita i token forzati per forward. */
static Grammar g_gram; static GrState g_gst;
static Tok *g_gr_T=NULL;
static int g_gr_on=0;     /* grammatica caricata e walker vivo */
static int g_gr_armed=0;  /* lazy: parte dal primo byte ammesso dalla radice (salta i preamboli) */
static int g_gr_max=24;
static uint64_t g_gr_prop=0, g_gr_acc=0;
static int g_looka=0;    /* LOOKA=1: misura (solo contatori, zero effetti) quanto il routing MoE
                          * e' predicibile IN ANTICIPO — la domanda che decide se un prefetch
                          * pilotato dal router puo' riempire i tempi morti del disco.
                          * [0] token precedente, stesso layer (cio' che usa gia' SPEC/PREFETCH)
                          * [1] ingresso del layer -> routing dello STESSO layer (salta l'attention)
                          * [2] post-attention del layer L -> routing di L+1 (un residuo MoE e
                          *     un'attention di anticipo: il punto dove il prefetch avrebbe
                          *     un intero giro di disco per lavorare in ombra). */
static int64_t la_hit[3], la_tot[3];
static int la_pred[2][130][16]; static signed char la_val[2][130];
static int g_pilot=0;    /* PILOT=1: prefetch pilotato dal router (vedi pilot_prefetch) */
static int g_pilot_k=8;  /* PILOT_K=k: prefetcha solo le prime k predizioni per posizione */
/* Aligned allocator for dense QT weights/scales: under METAL, page-align + register so the
 * GPU reads them zero-copy (no upload duplicate). Plain malloc otherwise. */
static void *qalloc(size_t n){
#ifdef ILI_METAL
    if(g_metal_enabled){ void *p; size_t r=(n+16383)&~(size_t)16383;
        if(posix_memalign(&p,16384,r)){fprintf(stderr,"OOM qalloc\n");exit(1);}
        ili_metal_register(p,r); return p; }
#endif
    return malloc(n);
}
static float *qsalloc(int O){ return (float*)qalloc((size_t)O*sizeof(float)); }
static int g_pilot_real=0;/* PILOT_REAL=1: il pilota fa LOAD VERI cross-layer dentro ecache[L+1]
                          * (non il semplice WILLNEED). Implica PILOT=1. Default OFF: hint-only. */
/* Handshake main<->pilota per il load-vero cross-layer. Invariante di sicurezza in DUE parti:
 *  1) Percorso MATMUL (moe): il pilota scrive SOLO ecache[layer] con layer > g_cur_moe_layer;
 *     il matmul in moe() legge SOLO ecache[layer]==g_cur_moe_layer, e la barriera a inizio moe()
 *     aspetta l'eventuale load in volo su QUEL layer. Quindi NESSUNO slot mezzo-caricato viene
 *     mai matmul-ato: il matmul e il pilota non toccano mai lo stesso layer contemporaneamente.
 *  2) Percorso SCAN (pilot_prefetch, anch'esso sul MAIN): la scansione di residenza gira sul
 *     layer FUTURO (lnext = layer corrente + 1), esattamente il layer che il pilota sta scrivendo
 *     -> QUI i due thread toccano davvero la stessa ecache. Percio' quella scansione prende
 *     g_pilot_mx (lo stesso lock del worker): letture e pubblicazione degli slot sono serializzate,
 *     niente torn read di ecn[]/eid. Il pilota non altera MAI il valore di un expert, solo QUALE
 *     expert e' residente: con un load andato a buon fine l'output resta byte-identico all'OFF. */
static pthread_mutex_t g_pilot_mx=PTHREAD_MUTEX_INITIALIZER;
static _Atomic int g_cur_moe_layer=-1;   /* massimo layer moe in cui il MAIN e' entrato (per forward) */
static _Atomic int g_pilot_inflight=-1;  /* layer che il worker sta REAL-caricando adesso (-1 = idle) */
static _Atomic long g_pilot_loads=0;     /* load cross-layer VERI completati (banda spesa) */
static _Atomic long g_pilot_drops=0;     /* predizioni scartate perche' il main possiede gia' il layer */
/* sceglie il formato da `bits`: >=16 f32, 5..8 int8, <=4 int4-packed */
static void qt_alloc(QT *t, int O, int I, int bits){
    t->O=O; t->I=I; t->qf=NULL; t->q8=NULL; t->q4=NULL; t->s=NULL;
    if(bits>=16){ t->fmt=0; t->qf=falloc((int64_t)O*I); }
    else if(bits>=5 || g_nopack){ t->fmt=1; t->q8=qalloc((int64_t)O*I); t->s=qsalloc(O); }
    else if(bits>=3){ t->fmt=2; t->q4=qalloc((int64_t)O*((I+1)/2)); t->s=qsalloc(O); }
    else { t->fmt=3; t->q4=qalloc((int64_t)O*((I+3)/4)); t->s=qsalloc(O); }
}
static void qt_fill(QT *t, const float *w, int bits){
    if(t->fmt==0) memcpy(t->qf, w, (int64_t)t->O*t->I*sizeof(float));
    else if(t->fmt==1) quantize_rows(w, t->q8, t->s, t->O, t->I, bits);
    else if(t->fmt==3) pack_int2(w, t->q4, t->s, t->O, t->I, bits);
    else pack_int4(w, t->q4, t->s, t->O, t->I, bits);
}

static void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps); for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}
/* LayerNorm classica (media+varianza, weight+bias) — usata dal k_norm dell'indexer DSA */
static void layernorm(float *v, const float *w, const float *b, int n, float eps){
    double mu=0; for(int i=0;i<n;i++) mu+=v[i]; mu/=n;
    double var=0; for(int i=0;i<n;i++){ double d=v[i]-mu; var+=d*d; } var/=n;
    float r=1.f/sqrtf((float)var+eps);
    for(int i=0;i<n;i++) v[i]=((float)(v[i]-mu))*r*w[i]+b[i];
}
static void softmax(float *x,int n){ float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++) x[i]/=s; }
static inline float sigmoidf(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf(float x){ return x/(1.f+expf(-x)); }

/* RoPE interleaved su un vettore di dimensione qk_rope a posizione pos */
static void rope_interleave(float *v, int pos, const Cfg *c){
    int half = c->qk_rope/2; float in[256]; memcpy(in,v,c->qk_rope*sizeof(float));
    for(int j=0;j<half;j++){
        float inv = powf(c->theta, -2.0f*j/c->qk_rope);
        float ang = pos*inv, cs=cosf(ang), sn=sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        v[j]      = a*cs - b*sn;
        v[half+j] = b*cs + a*sn;
    }
}

/* ---------- config ---------- */
static jval* cfg_root(const char *snap, char **arena){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    return json_parse(b,arena);
}
static int gi(jval*r,const char*k){ jval*v=json_get(r,k); return v?(int)v->num:0; }
static void load_cfg(Cfg *c, const char *snap){
    char *ar=NULL; jval *r=cfg_root(snap,&ar);
    c->hidden=gi(r,"hidden_size"); c->n_layers=gi(r,"num_hidden_layers");
    c->n_heads=gi(r,"num_attention_heads"); c->n_experts=gi(r,"n_routed_experts");
    c->topk=gi(r,"num_experts_per_tok"); c->moe_inter=gi(r,"moe_intermediate_size");
    c->dense_inter=gi(r,"intermediate_size"); c->first_dense=gi(r,"first_k_dense_replace");
    c->q_lora=gi(r,"q_lora_rank"); c->kv_lora=gi(r,"kv_lora_rank");
    c->qk_nope=gi(r,"qk_nope_head_dim"); c->qk_rope=gi(r,"qk_rope_head_dim");
    c->v_head=gi(r,"v_head_dim"); c->n_shared=gi(r,"n_shared_experts"); c->vocab=gi(r,"vocab_size");
    c->n_group=gi(r,"n_group"); c->topk_group=gi(r,"topk_group");
    jval *nt=json_get(r,"norm_topk_prob"); c->norm_topk=(nt&&nt->t==J_BOOL)?nt->boolean:0;
    jval *ep=json_get(r,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-5f;
    jval *rs=json_get(r,"routed_scaling_factor"); c->routed_scale=rs?(float)rs->num:1.f;
    jval *rp=json_get(r,"rope_parameters"); jval *th=rp?json_get(rp,"rope_theta"):NULL;
    c->theta = th?(float)th->num:10000.f;
    /* token di stop: GLM-5.2 ne ha TRE (endoftext, user, observation). Fermarsi solo sul
     * primo = generare spazzatura invisibile dopo la fine del turno (5-10x token sprecati). */
    c->n_stop=0;
    jval *eo=json_get(r,"eos_token_id");
    if(eo){ if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
            else if(eo->t==J_ARR) for(int i=0;i<eo->len && c->n_stop<8;i++)
                c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num; }
    /* DSA lightning indexer: parametri + tipo per-layer (lista esplicita o formula freq/offset) */
    c->index_topk=gi(r,"index_topk"); c->index_nh=gi(r,"index_n_heads"); c->index_hd=gi(r,"index_head_dim");
    { jval *it=json_get(r,"indexer_types");
      int freq=gi(r,"index_topk_freq"); if(freq<1) freq=1;
      jval *of=json_get(r,"index_skip_topk_offset"); int off=of?(int)of->num:2;
      for(int i=0;i<c->n_layers && i<128;i++){
          if(it && it->t==J_ARR && i<it->len && it->kids[i]->str)
              c->idx_type[i] = !strcmp(it->kids[i]->str,"full");
          else { int v=i-off+1; if(v<0) v=0; c->idx_type[i] = (v%freq)==0; }
      } }
    c->qk_head=c->qk_nope+c->qk_rope;
    c->attn_scale = 1.f / sqrtf((float)c->qk_head);
    if(c->n_group!=1){ fprintf(stderr,"this engine requires n_group=1 (GLM-5.2)\n"); exit(1); }
    /* VALIDAZIONE (report PR #25): il config.json arriva da mirror non fidati — dimensioni
     * ostili non devono superare questo punto. Un solo choke point protegge ogni alloc a valle. */
    #define CKR(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){ \
        fprintf(stderr,"config: %s=%d is outside [%d,%d]\n",name,(int)(v),(int)(lo),(int)(hi)); exit(1); }
    CKR("hidden_size",c->hidden,1,1<<20)         CKR("num_hidden_layers",c->n_layers,1,128)
    CKR("num_attention_heads",c->n_heads,1,1024) CKR("n_routed_experts",c->n_experts,1,4096)
    CKR("num_experts_per_tok",c->topk,1,64)      CKR("moe_intermediate_size",c->moe_inter,1,1<<20)
    CKR("intermediate_size",c->dense_inter,1,1<<24) CKR("first_k_dense_replace",c->first_dense,0,c->n_layers)
    CKR("q_lora_rank",c->q_lora,0,1<<20)         CKR("kv_lora_rank",c->kv_lora,1,1<<20)
    /* qk_rope <= 256: rope_interleave() copies qk_rope floats into a fixed `float in[256]`
     * stack buffer on EVERY forward; index_n_heads <= 64: the DSA indexer writes index_nh
     * scores into a fixed `float w32[64]` stack buffer (attention()). This choke point must
     * guarantee every fixed-size stack array downstream, or a valid-but-larger config
     * smashes the stack instead of failing loudly here (2026-07-21 bug pass). */
    CKR("qk_nope_head_dim",c->qk_nope,1,1<<16)   CKR("qk_rope_head_dim",c->qk_rope,1,256)
    CKR("v_head_dim",c->v_head,1,1<<16)          CKR("n_shared_experts",c->n_shared,0,64)
    CKR("vocab_size",c->vocab,1,1<<24)           CKR("index_topk",c->index_topk,0,1<<20)
    CKR("index_n_heads",c->index_nh,0,64)        CKR("index_head_dim",c->index_hd,0,1<<16)
    #undef CKR
    free(ar);
}

/* ---- mode-1.5 (Huffman-compressed) container guard ------------------------------
 * tools/mode15_container.py / c/mode15_reader.h define a SEPARATE on-disk format for
 * entropy-coded expert weights: a 4-byte magic "MH01" + header, replacing an expert
 * tensor's raw quantized bytes with a canonical-Huffman bitstream (see bench-m5max/
 * the Mode-15 lossless-pipeline registration). Its payload size is DATA-DEPENDENT, so it
 * can (and typically does) land in the `else` bucket of the byte-count format heuristic
 * used below (qt_from_disk / expert_load's `fmt=...?1:...?2:3`) exactly like a genuine
 * packed-int2 tensor would -- byte count alone cannot tell them apart.
 * THIS BUILD HAS NO MODE-1.5 DECODER WIRED IN (c/mode15_reader.h + c/codec_row_huff.h
 * exist and are tested standalone -- tests/test_mode15_reader.c -- but are not yet
 * called from this file: the Mode-15 integration design notes step 3+ is that
 * wiring, not done here). Silently treating a mode-1.5 blob as fmt==3 would hand its
 * compressed bytes straight to matmul_i2 -- numeric garbage, no error at all. Sniff the
 * magic BEFORE trusting the byte-count heuristic and abort loudly instead, naming the
 * tensor, so this reads as a container/build mismatch, not a mystery accuracy bug. */
static int mode15_blob(const uint8_t *p, int64_t n){
    return n>=4 && p[0]=='M' && p[1]=='H' && p[2]=='0' && p[3]=='1';
}
static void mode15_unsupported(const char *tensor_name){
    fprintf(stderr,
        "FATAL: '%s' is a mode-1.5 Huffman-compressed tensor (magic \"MH01\") but this "
        "engine build has no mode-1.5 decoder wired in (see c/mode15_reader.h, "
        "c/codec_row_huff.h, the Mode-15 integration design notes). Refusing to run "
        "matmul_i2 on compressed bytes -- rebuild with mode-1.5 decode support, or serve "
        "this model from an uncompressed (raw int4/int2) container instead.\n",
        tensor_name);
}

/* ---- legacy (non-mode-1.5) format determination -- fail-closed on unknown size ----
 * qt_from_disk/expert_load's byte-count format heuristic used to be an unconditional
 * `(nb==O*I)?1 : (nb==O*ceil(I/2))?2 : 3` -- ANY byte count that matched NEITHER int8
 * nor int4 fell through to the `: 3` (int2) bucket UNCHECKED, even if it didn't match
 * int2's own expected size either. A truncated/corrupted/wrong-shape tensor (or simply
 * a container built for a different bit-width than the engine was told to expect) would
 * therefore be silently treated as valid int2 data -- garbage weights, no error, the
 * exact same failure shape the mode-1.5 guard above exists to prevent for compressed
 * tensors, just left open here for legacy ones. legacy_fmt_from_size() replaces the
 * ternary with a real membership test against all three known formats' EXACT expected
 * byte counts; legacy_fmt_unknown() is the fail-closed diagnostic when none match. */
static int legacy_fmt_from_size(int64_t nb, int64_t O, int64_t I, int *out_fmt){
    if(nb==(int64_t)O*I){ *out_fmt=1; return 1; }              /* int8: 1 byte/param */
    if(nb==(int64_t)O*((I+1)/2)){ *out_fmt=2; return 1; }      /* int4: 2 params/byte */
    if(nb==(int64_t)O*((I+3)/4)){ *out_fmt=3; return 1; }      /* int2: 4 params/byte */
    return 0;
}
static void legacy_fmt_unknown(const char *tensor_name, int64_t nb, int64_t O, int64_t I){
    fprintf(stderr,
        "FATAL: '%s' has an on-disk size of %lld bytes, which matches NONE of the known "
        "quantized formats for its declared shape [O=%lld,I=%lld]: int8 expects exactly "
        "%lld bytes, int4 expects %lld, int2 expects %lld. Refusing to guess -- a previous "
        "version of this format heuristic silently treated any unrecognized size as int2 "
        "(the ternary's unconditional final bucket), which would hide a truncated, "
        "corrupted, or wrong-shape/wrong-bitwidth tensor as if it were valid weights. "
        "Rebuild this container, check for truncation/corruption, or confirm the engine "
        "was given the shape/bit-width the container was actually built with.\n",
        tensor_name, (long long)nb, (long long)O, (long long)I,
        (long long)((int64_t)O*I), (long long)((int64_t)O*((I+1)/2)), (long long)((int64_t)O*((I+3)/4)));
}

/* KNOWN GAP, investigated and deliberately NOT closed here (2026-07-20 robustness
 * pass): unlike the mode-1.5 path above (m15_verify_tensor_once's whole-tensor
 * CRC32, checked before any row is trusted), legacy int8/int4/int2 tensor reads
 * (st_read_raw() in qt_from_disk's `else` branch, the mmap fast path, and
 * expert_load's main pread path, all below/above) have NO integrity check at
 * all: a same-size bit-flip (bitrot, torn write, a wrong-but-plausible file) is
 * undetectable -- legacy_fmt_from_size() above only catches a WRONG SIZE, never
 * corruption within a size that happens to be correct. Checked (not assumed):
 * the safetensors header this engine reads carries only dtype/shape/
 * data_offsets per tensor (st_init(), st.h) -- no per-tensor hash field exists
 * anywhere in the container or a sidecar to key an opt-in verification off of.
 * The repo's only other sha256/checksum machinery (tools/provenance_manifest.py)
 * hashes the BINARY/generated-source/config.json for run-reproducibility
 * tracking -- a different concern, not per-tensor weight bytes, and not read by
 * this engine at load time. Adding real protection here would mean extending
 * the container FORMAT itself (a new per-tensor hash field, a sidecar manifest,
 * or similar) -- explicitly out of scope for this pass (a data format change,
 * not a code-robustness one). Left for later, together with whatever schema
 * change is chosen. */

#ifdef ILI_MODE15
/* Distinct from mode15_unsupported() above on PURPOSE: that message means "no
 * decoder is compiled into this build at all" (a build-configuration fact,
 * true regardless of this tensor's bytes); this one fires only when decode
 * support IS compiled in but this SPECIFIC blob's bytes were rejected (bad
 * magic/version/shape, truncation, or a CRC32 mismatch -- mode15_reader.h's
 * M15Status corrupt-data class). Conflating the two would hide a real
 * data-integrity event (corruption/bitrot/wrong-file -- design doc §1d's
 * "data-integrity failures" class, exactly what g_m15_crc_fail_* is meant to
 * count) behind a message that reads like a missing-feature build issue.
 * Both still fail CLOSED the same way (fprintf + exit/return, never a
 * partially-decoded buffer reaching matmul) -- only the diagnostic differs. */
static void mode15_decode_failed(const char *tensor_name){
    fprintf(stderr,
        "FATAL: '%s' is a mode-1.5 Huffman-compressed tensor (magic \"MH01\") but this "
        "build's decoder REJECTED it (bad magic/version/declared shape, a truncated "
        "container, or a CRC32 mismatch -- see c/mode15_reader.h's M15Status). This is a "
        "data-integrity event (corruption, bitrot, or a wrong/mismatched file), not a "
        "missing-decoder build issue -- see the Mode-15 integration design notes §1d. "
        "Fail-closed: refusing to run matmul_i2 on unverified/undecodable bytes.\n",
        tensor_name);
}

/* codec_row_huff.h (as committed today) exposes only huff_build() (raw symbol
 * COUNTS -> lengths, derived internally via huff_build_lengths() -> LUT) and
 * huff_canonical_codes() (lengths -> canonical codes) as real entry points --
 * there is no "build a codebook's LUT from an already-chosen lengths table"
 * function, because mode15_reader.h's own doc comment already flags this same
 * gap ("codec_row_huff.h's ACTUAL two decode conventions", citing only
 * huff_build_lengths()/huff_canonical_codes()). A mode-1.5 container ships
 * LENGTHS on the wire (never raw counts -- the whole point is a decoder never
 * re-derives them from a histogram), so this decode path needs exactly that
 * missing step. tools/mode15_cross_check.c and tests/test_mode15_reader.c
 * already hit this same gap and each carry their own faithful, unmodified-
 * logic copy of huff_build()'s LUT-fill loop (both named build_lut_from_
 * lengths(), both with a comment pointing at this exact limitation) -- this
 * is that same well-precedented workaround a third time, scoped `static` to
 * this file, NOT a modification of codec_row_huff.h itself (kept deliberately
 * out of scope for this change -- see this function's call site for why).
 * Returns 1 on success, 0 on OOM (2026-07-20 robustness pass: the malloc()
 * here used to be unchecked, followed unconditionally by memset() -- under
 * real memory pressure that is a NULL-pointer write, i.e. a SIGSEGV with no
 * diagnostic, instead of the controlled "OOM" failure every other allocation
 * in this file's mode-1.5 path already reports and fails closed on). */
static int mode15_huff_lut_from_lengths(HuffCodebook *cb){
    int maxlen=0, npres=0;
    for(int s=0;s<CODEC_NSYM;s++){ if(cb->len[s]>0){ npres++; if(cb->len[s]>maxlen) maxlen=cb->len[s]; } }
    cb->n_present=npres;
    cb->maxlen = maxlen>0 ? maxlen : 1;
    size_t lutn=(size_t)1u<<cb->maxlen;
    cb->lut=(HuffLutEnt*)malloc(lutn*sizeof(HuffLutEnt));
    if(!cb->lut){ fprintf(stderr,"OOM mode-1.5 huffman LUT\n"); return 0; }
    memset(cb->lut,0,lutn*sizeof(HuffLutEnt));
    if(npres==1){
        uint8_t s=0; for(int t=0;t<CODEC_NSYM;t++) if(cb->len[t]>0) s=(uint8_t)t;
        for(size_t i=0;i<lutn;i++){ cb->lut[i].sym=s; cb->lut[i].len=1; }
        return 1;
    }
    for(int s=0;s<CODEC_NSYM;s++){
        if(cb->len[s]==0) continue;
        int L=cb->len[s]; uint32_t base=cb->code_rev[s]; int rest=cb->maxlen-L;
        size_t block=(size_t)1u<<rest;
        for(size_t hi=0; hi<block; hi++){ size_t idx=base|(hi<<L); cb->lut[idx].sym=(uint8_t)s; cb->lut[idx].len=(uint8_t)L; }
    }
    return 1;
}

/* Decodes ONE already-in-memory mode-1.5 blob (`blob[0..nbytes)`, the WHOLE
 * on-disk tensor span -- magic+header+lengths+row_offsets+block_crc32+payload,
 * mode15_reader.h's LIFETIME CONTRACT) into `out_q4`, which the caller must have
 * already sized `O*((I+1)/2)` bytes -- fmt=2 (int4, 2 symbols/byte) layout,
 * identical to what a LEGACY int4 tensor already occupies (codec_row.h's
 * codec_pack_nibbles: low nibble = even column index, high nibble = odd),
 * so matmul_i4 cannot tell a decoded tensor from a legacy one afterward.
 *
 * Fail-CLOSED (design doc §1d "data-integrity failures" class): expect_O/
 * expect_I are passed straight to m15_open_structural so a container whose
 * declared shape disagrees with what THIS tensor's caller expects is rejected
 * exactly like a truncated/corrupt blob, not decoded against the wrong shape.
 * m15_verify_tensor_once() gates every payload byte behind its whole-tensor
 * CRC32 BEFORE a single row is decoded -- simplest correct choice for this
 * milestone (design doc step 4 is a correctness proof, not the lazy per-block
 * burst-decode path future perf work may switch to). Returns 1 on success, 0
 * on ANY failure (bad magic/version/shape, truncation, CRC mismatch, row
 * decode short-count) -- callers MUST treat 0 exactly like "decode
 * unavailable" (mode15_unsupported()+abort), never proceed with a partially-
 * filled out_q4. */
static int mode15_decode_i4(const uint8_t *blob, int64_t nbytes, int64_t O, int64_t I, uint8_t *out_q4){
    if(O<0 || I<0 || O>UINT32_MAX || I>UINT32_MAX || nbytes<0) return 0;
    M15Caps caps; memset(&caps,0,sizeof(caps)); caps.max_O=(uint32_t)O; caps.max_I=(uint32_t)I;
    M15Reader r;
    if(m15_open_structural(blob,(size_t)nbytes,O,I,&caps,&r)!=M15_OK) return 0;
    if(m15_verify_tensor_once(&r)!=M15_OK){ m15_close(&r); return 0; }

    HuffCodebook cb; memset(&cb,0,sizeof(cb));
    memcpy(cb.len, r.lengths, CODEC_NSYM);
    huff_canonical_codes(cb.len, cb.code_rev);        /* REAL codec_row_huff.h function, unmodified */
    if(!mode15_huff_lut_from_lengths(&cb)){ m15_close(&r); return 0; }  /* LUT-fill; OOM fails closed */

    int rb = (int)((I+1)/2);
    uint8_t *sym = (uint8_t*)malloc((size_t)(I>0?I:1));
    int ok = (sym!=NULL);
    if(ok){
        for(uint32_t row=0; row<r.O; row++){
            const uint8_t *ptr; uint32_t len;
            if(m15_get_row_span(&r,row,&ptr,&len)!=M15_OK){ ok=0; break; }
            huff_decode_row(&cb, ptr, len, (int)I, sym);
            codec_pack_nibbles(sym, (int)I, out_q4 + (int64_t)row*rb);
        }
    }
    free(sym);
    huff_free(&cb);
    m15_close(&r);
    return ok;
}

/* pread's WHOLE compressed blob (nbytes, at off, on fd) into a fresh
 * transient staging buffer, decodes it via mode15_decode_i4(), then frees the
 * staging buffer -- the "read compressed, decode straight into the consumer
 * buffer" shape design doc §1c describes, minus its buffer-reuse optimization
 * (out of scope for this correctness milestone: qt_from_disk's dense tensors
 * load once per model-load, and expert_load's mode-1.5 branch is the CPU
 * decode step design doc step 4 explicitly expects to be far below the
 * eventual GPU-decode throughput target -- a plain malloc/free per tensor load
 * is proportionate here, not a hot-path concern). Returns 1/0 exactly like
 * mode15_decode_i4(). */
static int mode15_pread_decode_i4(int fd, int64_t off, int64_t nbytes, int64_t O, int64_t I, uint8_t *out_q4){
    if(nbytes<=0) return 0;
    uint8_t *staging=(uint8_t*)malloc((size_t)nbytes);
    if(!staging) return 0;
    int ok = (pread(fd,staging,(size_t)nbytes,off)==nbytes) && mode15_decode_i4(staging,nbytes,O,I,out_q4);
    free(staging);
    return ok;
}
#endif /* ILI_MODE15 */

/* costruisce un QT [O,I] dal disco in `t` (buffer riusabili tra chiamate).
 *  - se esiste `name.qs`: pesi GIA' quantizzati nel container (U8 qdata + F32 scala) -> letti diretti
 *  - altrimenti: tensore pieno (f32/bf16) -> quantizzato a runtime a `bits` (oracolo tiny / pesi pieni)
 * drop=1 -> fadvise DONTNEED (streaming expert). */
static void qt_from_disk(Model *m, const char *name, int O, int I, int bits, int drop, QT *t){
    char sn[300]; snprintf(sn,sizeof(sn),"%s.qs",name);
    if(st_has(&m->S,sn)){
        int64_t nb=st_nbytes(&m->S,name);
        /* mode-1.5 detection -- checked HERE, BEFORE fmt is even computed: a peek pread of
         * just this tensor's own first 4 bytes, straight off disk, ahead of the
         * byte-count fmt heuristic below, ahead of qalloc()'s buffer sizing, and ahead of
         * t->fmt ever being assigned (a prior version of this guard checked AFTER t->fmt
         * was already committed and a buffer already allocated for it -- correct in
         * effect here because of the unconditional exit(1) two lines later, but the wrong
         * shape for a guard whose whole job is "never let dispatch/allocation commit to
         * unvalidated bytes"). See mode15_blob()/mode15_unsupported() above. */
        st_tensor *tt=st_find(&m->S,name);
        int is_m15=0;
        if(tt && nb>=4){
            uint8_t peek[4]={0};
            ssize_t got=pread(tt->fd,peek,sizeof(peek),tt->off);
            is_m15 = mode15_blob(peek,got);
        }
        if(is_m15){
#ifdef ILI_MODE15
            /* decode support compiled in: CPU-decode the whole blob straight into this
             * tensor's fmt=2 (int4-packed) buffer -- same shape/allocation convention as
             * the legacy fmt==2 branch below, so nothing downstream (matmul_i4, qt_bytes,
             * the CUDA dense-tensor path in qt_load) can tell the difference afterward. */
            if(t->fmt!=2||!t->q4){ t->fmt=2; t->O=O; t->I=I; t->q4=qalloc((size_t)O*((I+1)/2)); t->s=qsalloc(O); }
            if(!mode15_pread_decode_i4(tt->fd,tt->off,nb,O,I,t->q4)){ mode15_decode_failed(name); exit(1); }
#else
            /* no decode support in this build: preserve the fail-loud guard exactly --
             * never let a compressed blob reach the byte-count fmt heuristic below. */
            mode15_unsupported(name); exit(1);
#endif
        } else {
            int fmt;  /* int8 / int4 / int2, determined from byte count -- see legacy_fmt_from_size() */
            if(!legacy_fmt_from_size(nb,O,I,&fmt)){ legacy_fmt_unknown(name,nb,O,I); exit(1); }
            if(fmt==1){ if(t->fmt!=1||!t->q8){ t->fmt=1; t->O=O; t->I=I; t->q8=qalloc(nb); t->s=qsalloc(O); } st_read_raw(&m->S,name,t->q8,drop); }
            else      { if(t->fmt!=fmt||!t->q4){ t->fmt=fmt; t->O=O; t->I=I; t->q4=qalloc(nb); t->s=qsalloc(O); } st_read_raw(&m->S,name,t->q4,drop); }
        }
        st_read_f32(&m->S,sn,t->s,drop);
    } else {
        if(!t->qf && !t->q8 && !t->q4) qt_alloc(t,O,I,bits);
        if(t->fmt==0) st_read_f32(&m->S,name,t->qf,drop);
        else { float *tmp=falloc((int64_t)O*I); st_read_f32(&m->S,name,tmp,drop); qt_fill(t,tmp,bits); free(tmp); }
    }
}
static QT qt_load(Model *m, const char *name, int O, int I, int bits){
    QT t; memset(&t,0,sizeof(t)); qt_from_disk(m,name,O,I,bits,0,&t);
#ifdef ILI_CUDA
    if(g_cuda_enabled&&g_cuda_dense){
        t.cuda_eligible=1;
        int slot=g_cuda_rr++%g_cuda_ndev; t.cuda_device=g_cuda_devices[slot];
        g_cuda_dense_projected[slot]+=qt_bytes(&t);
    }
#endif
    return t;
}
static float *ld(Model *m, const char *name){   /* tensore 1D f32 residente (norme/bias) */
    int64_t n=st_numel(&m->S,name); if(n<0){fprintf(stderr,"missing %s\n",name);exit(1);}
    float *p=(float*)qalloc((size_t)n*sizeof(float));   /* registrato per la GPU sotto METAL */
    st_read_f32(&m->S,name,p,0); return p;
}

static void model_init(Model *m, const char *snap, int cap, int ebits, int dbits){
    memset(m,0,sizeof(*m)); m->ebits=ebits; m->dbits=dbits;
    load_cfg(&m->c,snap); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[256]; int H=c->n_heads, D=c->hidden;
    /* embed e lm_head sono il confine I/O: tenerli ad alta precisione (come i quant dynamic
     * reali). A bf16 ~1.9GB su GLM reale: trascurabile. dbits>=8 -> qui f32; piu' basso -> dbits. */
    int io_bits = dbits>=8 ? 16 : dbits;
    m->embed   = qt_load(m,"model.embed_tokens.weight", c->vocab, D, io_bits);
    m->lm_head = qt_load(m,"lm_head.weight", c->vocab, D, io_bits);
    m->final_norm = ld(m,"model.norm.weight");
    m->L=calloc(c->n_layers,sizeof(Layer));
    int NR=c->n_layers+1;                        /* +1: riga del layer MTP */
    m->ecap=cap; m->ecache=calloc(NR,sizeof(ESlot*)); m->ecn=calloc(NR,sizeof(int));
    m->eroute=calloc(NR,sizeof(int*)); m->enr=calloc(NR,sizeof(int));
    m->pin=calloc(NR,sizeof(ESlot*)); m->npin=calloc(NR,sizeof(int));
    m->eusage=calloc(NR,sizeof(uint32_t*)); m->eheat=calloc(NR,sizeof(uint32_t*));
    m->kv=calloc(1,sizeof(KVState));
    m->kv_start=m->kv->kv_start=calloc(NR,sizeof(int));
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        #define P(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
        l->in_ln=ld(m,P("input_layernorm.weight"));
        l->post_ln=ld(m,P("post_attention_layernorm.weight"));
        l->q_a   = qt_load(m,P("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
        l->q_a_ln= ld(m,P("self_attn.q_a_layernorm.weight"));
        l->q_b   = qt_load(m,P("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
        l->kv_a  = qt_load(m,P("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
        l->kv_a_ln= ld(m,P("self_attn.kv_a_layernorm.weight"));
        l->kv_b  = qt_load(m,P("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
        l->o     = qt_load(m,P("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
        l->sparse = (i >= c->first_dense);
        if(!l->sparse){
            l->gate_proj = qt_load(m,P("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
            l->up_proj   = qt_load(m,P("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
            l->down_proj = qt_load(m,P("mlp.down_proj.weight"), D, c->dense_inter, dbits);
        } else {
            l->router=ld(m,P("mlp.gate.weight"));
            l->router_bias=ld(m,P("mlp.gate.e_score_correction_bias"));
            int sI=c->moe_inter*c->n_shared;
            l->sh_gate = qt_load(m,P("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,P("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,P("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));      /* metodo C: ultimo routing del layer */
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
        }
        #undef P
    }
    /* testa MTP (layer n_layers): presente solo se convertita con --mtp */
    {
        /* MTP attiva SOLO se il set e' COMPLETO (i tensori vivono su 3 shard: durante la
         * conversione parziale ne esiste solo una parte). MTP=0 la disabilita comunque. */
        const char *req[]={"eh_proj.weight","enorm.weight","hnorm.weight","shared_head.norm.weight",
            "input_layernorm.weight","post_attention_layernorm.weight",
            "self_attn.q_a_proj.weight","self_attn.q_b_proj.weight","self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight","self_attn.o_proj.weight","mlp.gate.weight",
            "mlp.shared_experts.gate_proj.weight","mlp.shared_experts.down_proj.weight",
            "mlp.experts.0.gate_proj.weight","mlp.experts.255.down_proj.weight"};
        char mn[256]; m->has_mtp=1;
        for(unsigned q=0;q<sizeof(req)/sizeof(req[0]);q++){
            snprintf(mn,sizeof(mn),"model.layers.%d.%s",c->n_layers,req[q]);
            if(!st_has(&m->S,mn)){ m->has_mtp=0; break; }
        }
        if(getenv("MTP") && atoi(getenv("MTP"))==0) m->has_mtp=0;
        if(m->has_mtp){
            int i=c->n_layers; Layer *l=&m->mtpL;
            #define PM(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
            l->in_ln=ld(m,PM("input_layernorm.weight"));
            l->post_ln=ld(m,PM("post_attention_layernorm.weight"));
            l->q_a   = qt_load(m,PM("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
            l->q_a_ln= ld(m,PM("self_attn.q_a_layernorm.weight"));
            l->q_b   = qt_load(m,PM("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
            l->kv_a  = qt_load(m,PM("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
            l->kv_a_ln= ld(m,PM("self_attn.kv_a_layernorm.weight"));
            l->kv_b  = qt_load(m,PM("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
            l->o     = qt_load(m,PM("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
            l->sparse=1;
            l->router=ld(m,PM("mlp.gate.weight"));
            l->router_bias=ld(m,PM("mlp.gate.e_score_correction_bias"));
            int sI=c->moe_inter*c->n_shared;
            l->sh_gate = qt_load(m,PM("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,PM("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,PM("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
            m->eh_proj = qt_load(m,PM("eh_proj.weight"), D, 2*D, dbits);
            m->enorm=ld(m,PM("enorm.weight")); m->hnorm=ld(m,PM("hnorm.weight"));
            m->mtp_norm=ld(m,PM("shared_head.norm.weight"));
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->kv_start[i]=-1;                    /* KV MTP: parte dalla prima posizione di decode */
            #undef PM
        }
    }
    /* DSA lightning indexer: attivo SOLO se i pesi (conversione --indexer) ci sono per
     * TUTTI i layer full. Auto-rilevamento come per MTP: niente flag, niente passi extra. */
    {
        m->has_dsa = (c->index_topk>0 && c->index_nh>0 && c->index_hd>0 && c->index_hd<=256);
        char inm[300];
        for(int i=0;i<c->n_layers && m->has_dsa;i++){
            if(!c->idx_type[i]) continue;
            snprintf(inm,sizeof(inm),"model.layers.%d.self_attn.indexer.wq_b.weight",i);
            if(!st_has(&m->S,inm)) m->has_dsa=0;
        }
        if(getenv("DSA") && atoi(getenv("DSA"))==0) m->has_dsa=0;
        if(m->has_dsa){
            m->ix_wq=calloc(c->n_layers,sizeof(QT)); m->ix_wk=calloc(c->n_layers,sizeof(QT));
            m->ix_wp=calloc(c->n_layers,sizeof(QT));
            m->ix_knw=calloc(c->n_layers,sizeof(float*)); m->ix_knb=calloc(c->n_layers,sizeof(float*));
            for(int i=0;i<c->n_layers;i++){
                if(!c->idx_type[i]) continue;
                #define PI(s) (snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.indexer." s,i),nm)
                m->ix_wq[i]=qt_load(m,PI("wq_b.weight"), c->index_nh*c->index_hd, c->q_lora, dbits);
                m->ix_wk[i]=qt_load(m,PI("wk.weight"), c->index_hd, D, dbits);
                m->ix_wp[i]=qt_load(m,PI("weights_proj.weight"), c->index_nh, D, dbits);
                m->ix_knw[i]=ld(m,PI("k_norm.weight")); m->ix_knb[i]=ld(m,PI("k_norm.bias"));
                #undef PI
            }
            fprintf(stderr,"[DSA] indexer active: top-%d sparse attention beyond %d context tokens\n",
                c->index_topk, c->index_topk);
        }
    }
    m->hlast=falloc(D); m->h_all=falloc((int64_t)64*D);

    /* byte della parte DENSA residente (embed+lm_head+attn+mlp densa+shared+norme) */
    int64_t rb=qt_bytes(&m->embed)+qt_bytes(&m->lm_head);
    for(int i=0;i<c->n_layers;i++){ Layer *l=&m->L[i];
        rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
        if(!l->sparse) rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
        else rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down);
    }
    if(m->has_mtp){ Layer *l=&m->mtpL;
        rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
        rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down)+qt_bytes(&m->eh_proj);
    }
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        rb+=qt_bytes(&m->ix_wq[i])+qt_bytes(&m->ix_wk[i])+qt_bytes(&m->ix_wp[i]);
    m->resident_bytes=rb;
}

/* embed: dequantizza la riga del token (scala per-riga) in x[hidden] */
static void embed_row(Model *m, int tok, float *x){
    int D=m->c.hidden; QT *e=&m->embed;
    if(e->fmt==0){ memcpy(x, e->qf+(int64_t)tok*D, D*sizeof(float)); return; }
    if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)tok*D; float s=e->s[tok];
        for(int i=0;i<D;i++) x[i]=(float)q[i]*s; return; }
    if(e->fmt==2){ const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); float s=e->s[tok];   /* int4 */
        for(int i=0;i<D;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
            if(i+1<D) x[i+1]=(float)((int)(byte>>4)-8)*s; }
        return; }
    const uint8_t *q=e->q4+(int64_t)tok*((D+3)/4); float s=e->s[tok];   /* int2 */
    for(int i=0;i<D;i++){ uint8_t byte=q[i>>2]; int sh=(i&3)*2; x[i]=(float)((int)((byte>>sh)&3)-2)*s; }
}

/* ILI_MMAP=1: gli expert diventano VISTE dentro mmap dei file safetensors (niente pread,
 * niente slab, niente copia: la page cache del kernel E' la cache). Le mappe sono
 * registrate con Metal (newBufferWithBytesNoCopy su pagine file-backed, come llama.cpp),
 * quindi la GPU legge gli stessi byte. Fallback allo slab path su disallineamento. */
static int g_mmap=0;
static struct { int fd; void *base; size_t len; } g_maps[512]; static int g_nmaps;
static pthread_mutex_t g_map_mtx = PTHREAD_MUTEX_INITIALIZER;   /* expert_load e' OMP-parallel */
static void *map_of_fd(int fd){
    pthread_mutex_lock(&g_map_mtx);
    for(int i=0;i<g_nmaps;i++) if(g_maps[i].fd==fd){ void *b=g_maps[i].base; pthread_mutex_unlock(&g_map_mtx); return b; }
    void *base=NULL; struct stat st;
    if(g_nmaps<512 && fstat(fd,&st)==0){
        size_t len=((size_t)st.st_size+16383)&~(size_t)16383;
        void *p=mmap(NULL,len,PROT_READ,MAP_SHARED,fd,0);
        if(p!=MAP_FAILED){
            base=p; g_maps[g_nmaps].fd=fd; g_maps[g_nmaps].base=p; g_maps[g_nmaps].len=len; g_nmaps++;
#ifdef ILI_METAL
            if(g_metal_enabled) ili_metal_register(p,len);
#endif
        }
    }
    pthread_mutex_unlock(&g_map_mtx);
    return base;
}

/* ============ streaming-causality instrumentation: read-completion latency ============
 * Records one sample per completed expert fetch (read-completion latency, i.e. the real
 * disk-service time for that fetch, INCLUDING any Stage-1 throttle delay -- see
 * io_delay_inject() below once the throttle lever lands). Design constraint (pre-registered
 * <1% overhead bound): NO locks, NO allocation, NO formatting, NO histogram update on the
 * hot path. Each OS thread that ever calls expert_load (pipe workers, OMP parallel-for
 * workers, the PILOT thread, or the main thread on a direct/serial load) lazily claims one
 * fixed-size ring buffer via a single one-time atomic fetch-add (amortized, not hot-path);
 * after that, every record is a private array write + counter bump, no synchronization at
 * all. Percentiles are computed AFTER the run (iolat_percentiles), by which point every
 * writer thread is idle. */
#define IOLAT_MAX_THREADS 64
#define IOLAT_CAP 8192   /* per-thread ring; wraps on long runs, keeping the most recent
                          * samples -- adequate for steady-state p50/p95/p99 estimation */
typedef struct { double lat[IOLAT_CAP]; uint32_t n, next; } IOLatBuf;
static IOLatBuf g_iolat[IOLAT_MAX_THREADS];
static _Atomic int g_iolat_nthreads = 0;
/* [IOKIND] per-thread latency-SUM (not a ring: we only need totals, not percentiles) for
 * the weight-pread vs scale-pread split -- see io_kind_done() below io_read_done(). Shares
 * g_iolat_nthreads' slot claim (t_iolat_slot, declared below) so it inherits the exact same
 * thread-safety argument as g_iolat[] itself: each OS thread ever calling expert_load owns
 * exactly one slot for its whole lifetime and only ever writes its own index -- no lock, no
 * atomic-double needed. */
static double g_iolat_wsum[IOLAT_MAX_THREADS], g_iolat_ssum[IOLAT_MAX_THREADS];
static void iolat_reset(void){
    int n = g_iolat_nthreads; if(n>IOLAT_MAX_THREADS) n=IOLAT_MAX_THREADS;
    for(int i=0;i<n;i++){ g_iolat[i].n=0; g_iolat[i].next=0; g_iolat_wsum[i]=0; g_iolat_ssum[i]=0; }
}
static _Thread_local int t_iolat_slot = -1;
static inline void iolat_record(double sec){
    if(t_iolat_slot < 0){
        int s = atomic_fetch_add_explicit(&g_iolat_nthreads, 1, memory_order_relaxed);
        t_iolat_slot = (s < IOLAT_MAX_THREADS) ? s : (IOLAT_MAX_THREADS - 1); /* overflow: share last slot */
    }
    IOLatBuf *b = &g_iolat[t_iolat_slot];
    b->lat[b->next] = sec; b->next = (b->next + 1) % IOLAT_CAP;
    if(b->n < IOLAT_CAP) b->n++;
}
static int cmp_double(const void *a, const void *b){
    double x = *(const double*)a, y = *(const double*)b; return x<y ? -1 : x>y ? 1 : 0;
}
/* Aggregates every thread's ring into one sorted buffer and reports p50/p95/p99 + n.
 * Called once at end-of-run (profile_print); allocation/sort here is fine, it is off
 * the per-read hot path entirely. */
static void iolat_percentiles(double *p50, double *p95, double *p99, uint64_t *n_out){
    int nthreads = atomic_load_explicit(&g_iolat_nthreads, memory_order_relaxed);
    if(nthreads > IOLAT_MAX_THREADS) nthreads = IOLAT_MAX_THREADS;
    uint64_t total = 0;
    for(int i=0;i<nthreads;i++) total += g_iolat[i].n;
    *p50=*p95=*p99=0; *n_out=total;
    if(!total) return;
    double *all = malloc(total*sizeof(double)); uint64_t k=0;
    for(int i=0;i<nthreads;i++){ IOLatBuf *b=&g_iolat[i]; for(uint32_t j=0;j<b->n;j++) all[k++]=b->lat[j]; }
    qsort(all, total, sizeof(double), cmp_double);
    uint64_t i50=(uint64_t)(0.50*(total-1)), i95=(uint64_t)(0.95*(total-1)), i99=(uint64_t)(0.99*(total-1));
    *p50=all[i50]; *p95=all[i95]; *p99=all[i99];
    free(all);
}
/* ============ streaming-causality instrumentation: IO-throttle lever (Stage 1) ============
 * ILI_IO_DELAY_US: injects a deterministic delay into the expert-fetch path AFTER the real
 * pread(s) complete and BEFORE the load is exposed to the consumer (io_read_done() below is
 * called immediately before every `s->eid=eid; return 0;` in expert_load) -- i.e. it reduces
 * SERVICE CAPACITY only. It never changes: the number of expert fetches issued (routing/
 * selection is pure compute upstream of this, untouched by a sleep call), their grouping/
 * coalescing (the contig-vs-3-way-pread decision in expert_load is made and executed BEFORE
 * this runs), or their issue order (io_read_done() fires from the same call sites, in the
 * same order, delay on or off). Default 0 = off, zero behavior change. See
 * the stage-0 validation record for the single-variable proof (io_trace_log() below,
 * diffed on vs off on the same run). Calibration against iostat is Stage 1 proper and out
 * of scope here -- this commit builds the clean lever + the proof it is single-variable. */
static int g_io_delay_us=0;        /* legacy lever: applies to EVERY expert fetch, all phases */
static int g_io_delay_decode_us=0; /* Stage-2 v3 lever: applies ONLY once g_decode_phase=1 */
static int g_decode_phase=0;       /* 0 = load/prefill; run_replay sets 1 at the decode boundary */
/* a2 overlap falsifier: env-gated per-layer miss-completion stagger vs compute window. NO-OP off. */
static FILE *g_a2f=NULL; static int g_a2_on=0, g_a2_steps=1, g_a2_step=0;
static double g_a2_issue=0, g_a2_comp[64], g_a2_load_end=0, g_a2_cs=0;
static int g_a2_nmiss=0, g_a2_nb=0; static int64_t g_a2_bytes=0;
static FILE *g_moe_dump=NULL;                      /* ILI_MOE_INPUT_DUMP: per-position expert-input capture (gate 3) */
static int g_moe_dump_layers[8], g_moe_dump_nlayers=0;   /* ILI_MOE_INPUT_DUMP_LAYERS filter; empty=all layers */
static void io_delay_inject(void){
    int total = g_io_delay_us + (g_decode_phase ? g_io_delay_decode_us : 0);
    if(total<=0) return;
    struct timespec ts={ total/1000000, (long)(total%1000000)*1000L };
    nanosleep(&ts,NULL);
}
/* Marks one expert fetch as ACTUALLY COMPLETED (as opposed to merely attempted -- see
 * io_reads_attempted at the top of expert_load): records its read-completion latency
 * sample (t_read0 was stamped right after the real pread(s) were sized/looked up, so this
 * duration is dominated by the real I/O) and adds its bytes to io_bytes_read. Called at
 * every successful-return point in expert_load's real (non-fallback) path. */
static inline void io_read_done(Model *m, int64_t bytes, double t_read0){
    io_delay_inject();  /* Stage-1 throttle lever: AFTER the real read, BEFORE exposure below */
    double dt = now_s()-t_read0;
    iolat_record(dt);
    __atomic_fetch_add(&m->io_bytes_read,bytes,__ATOMIC_RELAXED);
    __atomic_fetch_add(&m->io_reads_completed,1,__ATOMIC_RELAXED);
}

/* [IOKIND] Step-0 diagnostic (#1438 deep-offload reconciliation; see RESULTS.md under
 * c/bench-m5max/step0-iokind-diag): per-tensor-kind split of the same expert-fetch path
 * io_read_done() already accounts for, called IN ADDITION to (never instead of) it --
 * io_bytes_read/io_reads_completed/iolat_record (and any Stage-1 io_delay_inject())
 * stay byte-identical; this purely ADDS a breakdown. kind_scale=0 -> weight-pread(s)
 * (the coalesced ~19MB contig pread, or its rare non-contig 3-way fallback);
 * kind_scale=1 -> one of the 3 small .qs scale-preads. `bytes` is the LOGICAL tensor
 * byte count (tw[k]->nbytes / tq[k]->nbytes), the same basis req_bytes/io_bytes_read
 * already use -- NOT the physical (possibly 4K-padded) O_DIRECT transfer size -- so
 * weight+scale bytes reduce back to req_bytes exactly, a cheap sanity check. `dt` is
 * the wall-clock duration of JUST that one pread() call (measured at each call site),
 * deliberately EXCLUDING any Stage-1 io_delay_inject() sleep (that lever fires once,
 * after all of an expert's reads, from io_read_done -- attributing it to one kind would
 * be arbitrary; the blended io_read_done sample above still includes it unchanged).
 * Thread-safety mirrors iolat_record exactly (see g_iolat_wsum/g_iolat_ssum's own
 * comment above): bytes/count are shared Model fields -> atomic fetch-add (same idiom
 * as io_bytes_read); latency-sum is written only into this thread's own slot. */
static inline void io_kind_done(Model *m, int kind_scale, int64_t bytes, double dt){
    if(t_iolat_slot < 0){       /* mirror iolat_record's lazy one-time slot claim */
        int s = atomic_fetch_add_explicit(&g_iolat_nthreads, 1, memory_order_relaxed);
        t_iolat_slot = (s < IOLAT_MAX_THREADS) ? s : (IOLAT_MAX_THREADS - 1);
    }
    if(kind_scale){
        g_iolat_ssum[t_iolat_slot] += dt;
        __atomic_fetch_add(&m->io_bytes_scale,bytes,__ATOMIC_RELAXED);
        __atomic_fetch_add(&m->io_reads_scale,1,__ATOMIC_RELAXED);
    } else {
        g_iolat_wsum[t_iolat_slot] += dt;
        __atomic_fetch_add(&m->io_bytes_weight,bytes,__ATOMIC_RELAXED);
        __atomic_fetch_add(&m->io_reads_weight,1,__ATOMIC_RELAXED);
    }
}
/* Reduces every thread's [IOKIND] latency-sum slot into two grand totals. Called once at
 * end-of-run (profile_print), same post-idle timing convention as iolat_percentiles. */
static void io_kind_latency_totals(double *wsum, double *ssum){
    int n = g_iolat_nthreads; if(n>IOLAT_MAX_THREADS) n=IOLAT_MAX_THREADS;
    double w=0, s=0;
    for(int i=0;i<n;i++){ w+=g_iolat_wsum[i]; s+=g_iolat_ssum[i]; }
    *wsum=w; *ssum=s;
}

#ifdef ILI_MODE15
/* ILI_MODE15_TRACE=<path>: DEBUG-ONLY, env-gated per-call trace of
 * expert_load_mode15() -- appends one "ENTER ..." line BEFORE any allocation/
 * pread/decode, and a matching "DONE ..." line right before a successful
 * return. Off by default (empty path -> one cheap string check, no I/O);
 * mirrors ILI_IO_TRACE's own append-only, mutex-guarded, fopen-per-call
 * shape (io_trace_log(), below this file's expert_load()) so it stays safe
 * to enable under the OMP-parallel expert-load loop too.
 *
 * Added specifically to help pin down WHERE a future occurrence of the
 * "silent death on a later turn" bug lands (c/scripts/mode15_e2e_certify.sh's
 * own "KNOWN ISSUE" note: a mode-1.5 `ili chat` session that already
 * completed prior turns could die with NO diagnostic output partway through
 * a later turn -- root-caused to something specific to REPEATED
 * expert_load_mode15() calls against long-lived ESlot/slab state across
 * turns). An ENTER line with no matching DONE line -- combined with the
 * fail-loud signal handler installed in main() and the `ili` wrapper's own
 * fix to always surface the engine's captured stderr on ANY session death --
 * turns "somewhere in expert_load_mode15, at some point, for some expert"
 * into an exact, reproducible (layer,eid) call and the slab/fslab capacity
 * this ESlot carried into that call. */
static char g_m15_trace_path[512]="";
static pthread_mutex_t g_m15_trace_mtx = PTHREAD_MUTEX_INITIALIZER;
static void m15_trace_log(const char *phase, int layer, int eid,
                           int64_t dtot, int64_t ftot, int64_t slab_cap, int64_t fslab_cap){
    if(!g_m15_trace_path[0]) return;
    pthread_mutex_lock(&g_m15_trace_mtx);
    FILE *f=fopen(g_m15_trace_path,"a");
    if(f){
        fprintf(f,"%s layer=%d eid=%d dtot=%lld ftot=%lld slab_cap_before=%lld fslab_cap_before=%lld\n",
                phase,layer,eid,(long long)dtot,(long long)ftot,(long long)slab_cap,(long long)fslab_cap);
        fflush(f); fclose(f);
    }
    pthread_mutex_unlock(&g_m15_trace_mtx);
}

/* Loads one expert's 3 tensors when at least one of them is a mode-1.5 (MH01)
 * blob -- split out of expert_load() itself so the existing g_mmap zero-copy
 * path and the existing contiguous/non-contiguous coalesced-pread fast path
 * (below, in expert_load proper) stay COMPLETELY UNTOUCHED for the common
 * (legacy, non-mode-1.5) case: neither can be reused here, by design
 * (the Mode-15 integration design notes §1a: "nothing downstream can consume [MH01]
 * bytes in place" -- g_mmap's zero-copy view and the raw contiguous pread both
 * assume on-disk bytes ARE the final in-memory representation, which is false
 * for a compressed blob). Per tensor, independently: MH01 -> pread the whole
 * compressed blob into a transient staging buffer, then CPU-decode
 * (mode15_pread_decode_i4) straight into this expert's slab position; legacy
 * (this expert's OTHER tensors, when only SOME of the three are mode-1.5 --
 * a real, tested mix: see tests/test_mode15_engine_guard.c's gate-only fixture)
 * -> an ordinary pread, byte-identical to expert_load's own main path. Scale
 * (.qs) tensors are NEVER compressed (design doc §1c: "stay raw") -- read
 * exactly like expert_load's own main path already does.
 *
 * Correctness milestone (design doc step 4), not a performance one: no
 * attempt is made to preserve the coalesced single-pread optimization (a
 * mixed compressed/legacy expert has no fixed-width relationship between
 * on-disk and in-slab byte offsets, so that trick does not apply here) or to
 * avoid the GPU zero-copy path's throughput (mode-1.5 always takes the
 * pread+decode path, same as any tensor that already fails today's `okm`
 * alignment check). Returns 0 on success (s->eid updated to `eid`), -1 on
 * failure -- same fatal-vs-speculative contract as expert_load itself: fatal=1
 * exits the process on any I/O/OOM/decode error (byte-for-byte the same
 * failure class expert_load's own main path already exits on); fatal=0
 * abandons the load and returns -1 without touching s->eid. */
static int expert_load_mode15(Model *m, int layer, int eid, ESlot *s, st_tensor *tw[3], st_tensor *tq[3],
                               const int is_m15[3], const char nm[][288], int I, int D, int fatal){
    int64_t req_bytes = tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes+tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes;
    double t_read0 = now_s();
    __atomic_fetch_add(&m->io_reads_attempted,1,__ATOMIC_RELAXED);
    __atomic_fetch_add(&m->io_bytes_requested,req_bytes,__ATOMIC_RELAXED);

    int OO[3]={I,I,D}, II[3]={D,D,I};
    int64_t dec_sz[3], dtot=0;
    for(int k=0;k<3;k++){ dec_sz[k]=(int64_t)OO[k]*((II[k]+1)/2); dtot+=dec_sz[k]; }
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    /* ENTER, before any (re)alloc/pread/decode -- see m15_trace_log()'s own
     * doc comment above for why: an ENTER with no matching DONE (below) is
     * this call's exact (layer,eid), captured at the one point that survives
     * a death anywhere later in this function. */
    m15_trace_log("ENTER",layer,eid,dtot,ftot,s->slab_cap,s->fslab_cap);

    /* slab/fslab (re)alloc: same grow-only convention + ILI_METAL zero-copy
     * registration as expert_load's own main path below, just sized off `dtot`
     * (the DECODED int4-packed total) instead of `wtot` (the raw on-disk
     * total) -- for a legacy tensor these are the same number (nbytes==
     * O*ceil(I/2) already, or expert_load_mode15 would never have been
     * called with that tensor un-flagged -- see the size check in the fill
     * loop below); for an actually-compressed tensor dtot > its own on-disk
     * bytes (that size reduction is the entire point of mode-1.5). */
    if(!s->slab || dtot+8192 > s->slab_cap){
#ifdef ILI_METAL
        if(s->slab && g_metal_enabled) ili_metal_unregister(s->slab);
        compat_aligned_free(s->slab);
        size_t need=((size_t)dtot+8192+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->slab,16384,need)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=need;
        if(g_metal_enabled) ili_metal_register(s->slab,need);
#else
        compat_aligned_free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,dtot+8192)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=dtot+8192;
#endif
    }
    if(!s->fslab || ftot > s->fslab_cap){
#ifdef ILI_METAL
        if(s->fslab && g_metal_enabled) ili_metal_unregister(s->fslab);
        free(s->fslab);
        size_t fb=(((size_t)ftot*sizeof(float))+16383)&~(size_t)16383;
        if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
           posix_memalign((void**)&s->fslab,16384,fb)){
            fprintf(stderr,"OOM fslab\n"); if(fatal) exit(1);
            compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0;
            s->fslab=NULL; s->fslab_cap=0; return -1;
        }
        s->fslab_cap=ftot;
        if(g_metal_enabled) ili_metal_register(s->fslab,fb);
#else
        free(s->fslab);
        if(fatal){ s->fslab=falloc(ftot); }
        else {
            if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
               !(s->fslab=malloc((size_t)ftot*sizeof(float)))){
                fprintf(stderr,"OOM fslab\n");
                compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0;
                s->fslab=NULL; s->fslab_cap=0; return -1;
            }
        }
        s->fslab_cap=ftot;
#endif
    }

    int64_t pos[3]; { int64_t o=0; for(int k=0;k<3;k++){ pos[k]=o; o+=dec_sz[k]; } }
    for(int k=0;k<3;k++){
        uint8_t *dst=s->slab+pos[k];
        if(is_m15[k]){
            if(!mode15_pread_decode_i4(tw[k]->fd,tw[k]->off,tw[k]->nbytes,OO[k],II[k],dst)){
                mode15_decode_failed(nm[k]); if(fatal) exit(1); return -1;
            }
        } else {
            /* legacy sibling inside a mixed expert: must already be plain fmt=2
             * int4-packed bytes (GLM quantizes all 3 tensors of one expert to
             * the same bit-width) -- a size mismatch here means a container
             * that isn't what mode-1.5 was designed against, not something
             * safe to guess through. */
            if(tw[k]->nbytes!=dec_sz[k]){
                fprintf(stderr,"mode-1.5: %s is a legacy (uncompressed) tensor but its size "
                        "(%lld bytes) doesn't match the int4-packed shape this expert's "
                        "mode-1.5 sibling(s) imply (expected %lld)\n",
                        nm[k], (long long)tw[k]->nbytes, (long long)dec_sz[k]);
                if(fatal) exit(1); return -1;
            }
            if(pread(tw[k]->fd,dst,(size_t)dec_sz[k],tw[k]->off)!=dec_sz[k]){
                perror("pread expert (mode-1.5 sibling)"); if(fatal) exit(1); return -1;
            }
        }
    }
    float *fp[3]; int64_t fo=0;                  /* scale (piccole) -- mai compresse */
    for(int k=0;k<3;k++){
        if(pread(tq[k]->fd,(char*)(s->fslab+fo),tq[k]->nbytes,tq[k]->off)!=tq[k]->nbytes){
            perror("pread qs"); if(fatal) exit(1); return -1;
        }
        fp[k]=s->fslab+fo; fo+=tq[k]->nbytes/4;
    }
    if(g_drop){
        for(int k=0;k<3;k++){
            posix_fadvise(tw[k]->fd,tw[k]->off,tw[k]->nbytes,POSIX_FADV_DONTNEED);
            posix_fadvise(tq[k]->fd,tq[k]->off,tq[k]->nbytes,POSIX_FADV_DONTNEED);
        }
    }
    QT *qt[3]={&s->g,&s->u,&s->d};
    for(int k=0;k<3;k++){
        qt[k]->fmt=2; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+pos[k]); qt[k]->q4=s->slab+pos[k]; qt[k]->s=fp[k];
    }
    io_read_done(m,req_bytes,t_read0);
    s->eid=eid;
    m15_trace_log("DONE",layer,eid,dtot,ftot,s->slab_cap,s->fslab_cap);
    return 0;
}
#endif /* ILI_MODE15 */

/* carica un expert nello slot. Container pre-quantizzato: le 3 matrici sono contigue nel
 * file -> UNA pread coalescente da ~19 MB dentro `slab` (+ le scale in fslab); i QT sono
 * viste dentro lo slab (zero copie). Fallback per modelli non quantizzati (oracolo tiny).
 * THREAD-SAFE su slot distinti (pread posizionale, st_find read-only). */
/* Load one expert's weights into slot `s`. Returns 0 on success, -1 on failure.
 * fatal=1 (all main / on-demand / REPIN / pin callers): preserve the original
 * exit-on-error contract byte-for-byte — any missing tensor, OOM, short read or
 * pread error aborts the process. fatal=0 (speculative pilot only): the same
 * errors instead abandon the load and return -1 without touching s->eid, so a
 * mispredicted cross-layer prefetch can never kill the server. */
static int expert_load(Model *m, int layer, int eid, ESlot *s, int fatal){
#ifdef ILI_CUDA
    /* A live REPIN may reuse a GPU-enabled pinned slot for a different expert.
     * Keep its tier assignment, but invalidate the old device weights. */
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
    char nm[3][288]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[300]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(!st_has(&m->S,qn)){                       /* fallback: tensori pieni, quantizza a runtime.
                                                  * Reachable ONLY for unquantized models (no .qs);
                                                  * GLM always has .qs, so the pilot never hits it. */
        qt_from_disk(m,nm[0],I,D,b,g_drop,&s->g);
        qt_from_disk(m,nm[1],I,D,b,g_drop,&s->u);
        qt_from_disk(m,nm[2],D,I,b,g_drop,&s->d);
        s->eid=eid; return 0;
    }
    st_tensor *tw[3], *tq[3];
    for(int k=0;k<3;k++){
        tw[k]=st_find(&m->S,nm[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm[k]); tq[k]=st_find(&m->S,qn);
        if(!tw[k]||!tq[k]){ fprintf(stderr,"missing %s\n",nm[k]); if(fatal) exit(1); return -1; }
    }
    /* mode-1.5 detection -- checked HERE, as early as the tensors are known to exist, and
     * strictly BEFORE anything below commits to a format: before req_bytes/io-attempt
     * accounting, before g_mmap's map_of_fd()/CPU pre-touch, before slab/fslab
     * (re)allocation, before the coalesced or per-tensor pread, and before any qt[]->fmt
     * is ever assigned -- i.e. before dispatch selection, buffer allocation, or any
     * output, not merely "before the matmul call". A peek of each tensor's own first 4
     * bytes (3 tiny preads, independent of the multi-MB read that follows in the
     * legitimate case) is enough to identify a "MH01"-tagged expert almost instantly
     * instead of after a wasted full read. See mode15_blob()/mode15_unsupported() above. */
    int is_m15[3]={0,0,0}, any_m15=0;
    for(int k=0;k<3;k++){
        if(tw[k]->nbytes>=4){
            uint8_t peek[4]={0};
            ssize_t got=pread(tw[k]->fd,peek,sizeof(peek),tw[k]->off);
            if(mode15_blob(peek,got)){ is_m15[k]=1; any_m15=1; }
        }
    }
    if(any_m15){
#ifdef ILI_MODE15
        /* decode support compiled in: hand off to the dedicated mode-1.5 load path
         * (expert_load_mode15, defined above expert_load) -- it owns req_bytes/
         * io-attempt accounting, slab/fslab sizing, and the fill/return for this call
         * entirely; the g_mmap fast path and the coalesced-pread path below are never
         * reached for this expert (see expert_load_mode15's own comment for why
         * neither applies to a compressed tensor). */
        return expert_load_mode15(m,layer,eid,s,tw,tq,is_m15,nm,I,D,fatal);
#else
        /* no decode support in this build: preserve the fail-loud guard exactly --
         * never let a compressed blob reach req_bytes accounting, g_mmap, slab
         * allocation, or any pread/fmt-dispatch below. */
        for(int k=0;k<3;k++) if(is_m15[k]){ mode15_unsupported(nm[k]); if(fatal) exit(1); return -1; }
#endif
    }
    /* streaming-causality instrumentation: bytes REQUESTED counts every attempt (this point),
     * bytes READ / the latency sample only land on an actual successful completion below --
     * see the two "read done" blocks at the mmap-fastpath and main-path returns. */
    int64_t req_bytes = tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes+tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes;
    double t_read0 = now_s();
    __atomic_fetch_add(&m->io_reads_attempted,1,__ATOMIC_RELAXED);
    __atomic_fetch_add(&m->io_bytes_requested,req_bytes,__ATOMIC_RELAXED);
    if(g_mmap){
        void *bw[3],*bq[3]; int okm=1;
        int OO[3]={I,I,D}, II[3]={D,D,I};
        int fmt[3]={0,0,0};
        for(int k=0;k<3;k++){
            bw[k]=map_of_fd(tw[k]->fd); bq[k]=map_of_fd(tq[k]->fd);
            if(!bw[k]||!bq[k]||((tw[k]->off)&3)||((tq[k]->off)&3)) okm=0;
            /* mode-1.5 already ruled out for all 3 tensors above -- this is ONLY the
             * legacy int8/int4/int2 byte-count check (legacy_fmt_from_size()). An
             * unrecognized size disqualifies the mmap fast path exactly like a bad
             * alignment already does: falls through to the main (non-mmap) path below,
             * which re-derives fmt the same way and fails CLOSED there with the real
             * diagnostic -- never let qt[]->fmt commit to a guessed value under a
             * zero-copy view into file-backed memory. */
            else if(!legacy_fmt_from_size(tw[k]->nbytes,OO[k],II[k],&fmt[k])) okm=0;
        }
        if(okm){
            QT *qt[3]={&s->g,&s->u,&s->d};
            for(int k=0;k<3;k++){
                qt[k]->fmt=fmt[k]; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
                qt[k]->q8=(int8_t*)((char*)bw[k]+tw[k]->off); qt[k]->q4=(uint8_t*)((char*)bw[k]+tw[k]->off);
                qt[k]->s=(float*)((char*)bq[k]+tq[k]->off);
            }
            /* CPU pre-touch: fault the pages in HERE (cheap, parallel, overlapped with the
             * resident-experts GPU submit) so the GPU never demand-faults file-backed pages
             * (measured catastrophic). madvise starts async readahead, the touch guarantees
             * residency. This is pread's I/O without the copy and without the slab. */
            for(int k=0;k<3;k++){
                char *p=(char*)bw[k]+tw[k]->off; size_t n=(size_t)tw[k]->nbytes;
                madvise((void*)((uintptr_t)p & ~16383UL), n+16384, MADV_WILLNEED);
                volatile char acc=0;
                for(size_t i=0;i<n;i+=4096) acc+=p[i];
                acc+=p[n-1]; (void)acc;
                char *q=(char*)bq[k]+tq[k]->off; size_t nq=(size_t)tq[k]->nbytes;
                for(size_t i=0;i<nq;i+=4096) acc+=q[i];
            }
            io_read_done(m,req_bytes,t_read0);
            s->eid=eid; return 0;
        }
    }
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    /* rialloca se lo slot (riusato tra layer) e' troppo piccolo per QUESTO expert:
     * pread oltre la mappatura = short-read o CORRUZIONE silenziosa dei vicini */
    if(!s->slab || wtot+8192 > s->slab_cap){
#ifdef ILI_METAL
        /* page-align + zero-copy wrap: the GPU reads this slab in place (unified memory) */
        if(s->slab && g_metal_enabled) ili_metal_unregister(s->slab);
        compat_aligned_free(s->slab);
        size_t need=((size_t)wtot+8192+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->slab,16384,need)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=need;
        if(g_metal_enabled) ili_metal_register(s->slab,need);
#else
        compat_aligned_free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,wtot+8192)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=wtot+8192;
#endif
    }
    if(!s->fslab || ftot > s->fslab_cap){
#ifdef ILI_METAL
        /* page-align + register: the GPU reads the scales in place (unified memory).
         * Honours `fatal` exactly like the CPU arm below — a speculative pilot load
         * that hits OOM must unwind into a clean hidden slot, never exit(). */
        if(s->fslab && g_metal_enabled) ili_metal_unregister(s->fslab);
        free(s->fslab);
        size_t fb=(((size_t)ftot*sizeof(float))+16383)&~(size_t)16383;
        if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
           posix_memalign((void**)&s->fslab,16384,fb)){
            fprintf(stderr,"OOM fslab\n"); if(fatal) exit(1);
            compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0;  /* clean, hidden slot (eid stays -1) */
            s->fslab=NULL; s->fslab_cap=0; return -1;
        }
        s->fslab_cap=ftot;
        if(g_metal_enabled) ili_metal_register(s->fslab,fb);
#else
        free(s->fslab);
        if(fatal){ s->fslab=falloc(ftot); }          /* main path: byte-identical exit-on-OOM */
        else {                                        /* speculative pilot: checked alloc, never exit() */
            /* replicate falloc's anti-wrap guard + malloc (no zeroing/alignment) */
            if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
               !(s->fslab=malloc((size_t)ftot*sizeof(float)))){
                fprintf(stderr,"OOM fslab\n");
                compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0; /* leave a clean, hidden slot (eid stays -1) */
                s->fslab=NULL; s->fslab_cap=0; return -1;
            }
        }
        s->fslab_cap=ftot;
#endif
    }
    int ord[3]={0,1,2};                          /* ordina per offset nel file */
    for(int a=0;a<3;a++) for(int bb=a+1;bb<3;bb++) if(tw[ord[bb]]->off<tw[ord[a]]->off){ int t=ord[a]; ord[a]=ord[bb]; ord[bb]=t; }
    int contig = tw[ord[0]]->fd==tw[ord[1]]->fd && tw[ord[1]]->fd==tw[ord[2]]->fd
              && tw[ord[0]]->off+tw[ord[0]]->nbytes==tw[ord[1]]->off
              && tw[ord[1]]->off+tw[ord[1]]->nbytes==tw[ord[2]]->off;
    int64_t pos[3]; int done=0;
    if(contig){
        int64_t off0=tw[ord[0]]->off;
        int dfd = g_direct ? st_direct_fd(&m->S, tw[ord[0]]->fd) : -1;
        if(dfd>=0){                              /* O_DIRECT: offset/len allineati a 4K */
            int64_t base=off0 & ~4095LL, need=(off0-base)+wtot;
            int64_t len=(need+4095)&~4095LL;
            double t_w0=now_s();                 /* [IOKIND] weight-pread (O_DIRECT coalesced) */
            ssize_t r=pread(dfd, s->slab, len, base);
            if(r>=need){
                io_kind_done(m,0,wtot,now_s()-t_w0);
                pos[ord[0]]=off0-base; pos[ord[1]]=pos[ord[0]]+tw[ord[0]]->nbytes;
                pos[ord[2]]=pos[ord[1]]+tw[ord[1]]->nbytes; done=1;
            }
        }
        if(!done){                               /* fallback bufferizzato */
            double t_w0=now_s();                 /* [IOKIND] weight-pread (buffered coalesced) */
            if(pread(tw[ord[0]]->fd, s->slab, wtot, off0)!=wtot){ perror("pread expert"); if(fatal) exit(1); return -1; }
            io_kind_done(m,0,wtot,now_s()-t_w0);
            pos[ord[0]]=0; pos[ord[1]]=tw[ord[0]]->nbytes; pos[ord[2]]=tw[ord[0]]->nbytes+tw[ord[1]]->nbytes; done=1;
        }
    }
    if(!done){                                   /* non contigui: 3 pread bufferizzate */
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a];
            double t_w0=now_s();                 /* [IOKIND] weight-pread (non-contig fallback, per-tensor) */
            if(pread(tw[k]->fd, s->slab+o, tw[k]->nbytes, tw[k]->off)!=tw[k]->nbytes){ perror("pread expert"); if(fatal) exit(1); return -1; }
            io_kind_done(m,0,tw[k]->nbytes,now_s()-t_w0);
            pos[k]=o; o+=tw[k]->nbytes; }
    }
    float *fp[3]; int64_t fo=0;                  /* scale (piccole) */
    for(int k=0;k<3;k++){
        double t_s0=now_s();                     /* [IOKIND] scale-pread */
        if(pread(tq[k]->fd, (char*)(s->fslab+fo), tq[k]->nbytes, tq[k]->off)!=tq[k]->nbytes){ perror("pread qs"); if(fatal) exit(1); return -1; }
        io_kind_done(m,1,tq[k]->nbytes,now_s()-t_s0);
        fp[k]=s->fslab+fo; fo+=tq[k]->nbytes/4; }
    if(g_drop){                                  /* scarta subito le pagine: evita che la page
                                                  * cache in pressione strangoli il throughput */
        posix_fadvise(tw[ord[0]]->fd, tw[ord[0]]->off, wtot, POSIX_FADV_DONTNEED);
        for(int k=0;k<3;k++) posix_fadvise(tq[k]->fd, tq[k]->off, tq[k]->nbytes, POSIX_FADV_DONTNEED);
    }
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
    for(int k=0;k<3;k++){
        /* mode-1.5 already ruled out for all 3 tensors above, before slab allocation or
         * either pread branch even ran -- this is ONLY the legacy int8/int4/int2
         * byte-count check now (legacy_fmt_from_size()): an unrecognized size fails
         * CLOSED here, before qt[]->fmt ever commits to a guessed value, exactly like
         * the mmap fast path above and qt_from_disk now do -- never the silent-int2-
         * default the old unconditional ternary fell through to. */
        int fmt;
        if(!legacy_fmt_from_size(tw[k]->nbytes,OO[k],II[k],&fmt)){
            legacy_fmt_unknown(nm[k],tw[k]->nbytes,OO[k],II[k]); if(fatal) exit(1); return -1;
        }
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+pos[k]); qt[k]->q4=s->slab+pos[k]; qt[k]->s=fp[k];
    }
    io_read_done(m,req_bytes,t_read0);
    s->eid=eid; return 0;
}

/* ============================ PIPE: load ‖ matmul ============================
 * Overlap NVMe expert-weight loads with expert matmul. A small persistent pool
 * of I/O worker pthreads runs the misses' pread (expert_load) into distinct
 * ws[] slabs and sets a per-slot `ready` flag; the MAIN thread walks the block's
 * experts in order, waiting on ready[q] only for the expert it needs right now,
 * and does all matmul_qt on itself (matmul_qt parallelises internally via OpenMP
 * and checks !omp_in_parallel() for GPU dispatch — so it must stay off the omp
 * team and off these I/O threads).
 *
 * Cross-generation safety is provided by a single generation-tagged, lock-free
 * cursor `cur = (gen<<8) | index`. The main thread is the sole writer of `gen`
 * (monotonic bump, so no ABA); workers grab jobs by CAS-advancing the low 8-bit
 * index. THE INVARIANT: a worker reads eids[i]/layer only AFTER its winning CAS,
 * and that CAS's comparand carries the generation — so if `cur`'s gen advanced
 * (a new batch was published), the CAS fails and the worker re-reads, seeing the
 * new generation. A straggler preempted anywhere (wake gap, post-cursor) can
 * therefore NEVER grab a wrong-generation job or read torn batch state: its
 * first act is a gen-checked CAS. dispatch publishes all batch state with
 * relaxed stores and then RELEASE-stores `cur`; each worker ACQUIRE-loads `cur`,
 * so the ready[] reset + eids[]/njobs/layer are visible before any worker acts.
 * The per-expert pipe_wait(ready[q]) in the matmul loop makes every grabbed job
 * complete before the block ends, so no grab outlives its generation — which is
 * why the old `active` counter AND the end-of-block drain barrier are gone (both
 * were redundant with those per-slot waits + the gen-tagged cursor). The mutex/
 * condvar exist ONLY to park/wake idle workers, never for correctness. Gated
 * behind PIPE=1; OFF => the original blocking-load + serial-matmul path runs
 * byte-identically. */
static int g_pipe=0;      /* PIPE=1: async expert-load pipeline (default OFF) */
static int g_pipe_nw=8;   /* PIPE_WORKERS=n: I/O worker threads (disk-parallel reads) */
typedef struct {
    _Atomic uint64_t cur;                         /* (gen<<8)|index; gen main-only, index 0..njobs (≤64) */
    _Atomic int njobs;                            /* current batch job count */
    _Atomic int eids[64];                         /* current batch expert ids */
    _Atomic int layer;                            /* current batch layer */
    _Atomic int ready[64];                        /* per-slot load-done flag */
    pthread_mutex_t mx; pthread_cond_t cv;        /* ONLY for parking/waking idle workers */
    Model *m;
    pthread_t th[16]; int nw; int started;
} PipePool;
static PipePool g_pp;

static void *pipe_worker(void *arg){
    (void)arg; PipePool *p=&g_pp; uint64_t seen=0;
    for(;;){
        pthread_mutex_lock(&p->mx);
        while((atomic_load_explicit(&p->cur,memory_order_relaxed)>>8)==seen)
            pthread_cond_wait(&p->cv,&p->mx);
        pthread_mutex_unlock(&p->mx);
        for(;;){
            uint64_t c=atomic_load_explicit(&p->cur,memory_order_acquire);
            seen=c>>8;
            uint32_t i=(uint32_t)(c & 0xFF);
            if(i >= (uint32_t)atomic_load_explicit(&p->njobs,memory_order_relaxed))
                break;                                /* batch drained → re-park */
            if(atomic_compare_exchange_weak_explicit(&p->cur,&c,c+1,
                    memory_order_acq_rel,memory_order_relaxed)){
                int L  =atomic_load_explicit(&p->layer,memory_order_relaxed);
                int eid=atomic_load_explicit(&p->eids[i],memory_order_relaxed); /* AFTER winning CAS */
                expert_load(p->m,L,eid,&p->m->ws[i],1);  /* needed-now load: fatal on I/O error (matches serial path) */
                atomic_store_explicit(&p->ready[i],1,memory_order_release);
            }
            /* CAS failed → another worker advanced index (or gen advanced): re-loop */
        }
    }
    return NULL;
}
static void pipe_init(Model *m){
    if(g_pp.started) return;
    g_pp.m=m; g_pp.nw=g_pipe_nw; if(g_pp.nw>16) g_pp.nw=16; if(g_pp.nw<1) g_pp.nw=1;
    atomic_store(&g_pp.cur,0); atomic_store(&g_pp.njobs,0);
    pthread_mutex_init(&g_pp.mx,NULL); pthread_cond_init(&g_pp.cv,NULL);
    /* pthread_create CAN fail (EAGAIN under thread/rlimit pressure). If NO worker starts,
     * a later pipe_dispatch would enqueue jobs nobody ever runs and pipe_wait() would spin
     * on ready[q] forever -- a silent mid-token hang. Count live workers; with zero, clear
     * g_pipe so the caller takes the synchronous blocking-load path (2026-07-21 bug pass). */
    int live=0;
    for(int i=0;i<g_pp.nw;i++) if(pthread_create(&g_pp.th[live],NULL,pipe_worker,NULL)==0) live++;
    if(live<1){
        fprintf(stderr,"[PIPE] no I/O worker thread could be started; PIPE disabled, "
                       "falling back to synchronous expert loads\n");
        g_pipe=0; return;                        /* started stays 0 */
    }
    if(live<g_pp.nw)
        fprintf(stderr,"[PIPE] only %d/%d I/O worker threads started\n",live,g_pp.nw);
    g_pp.nw=live;
    g_pp.started=1;
}
/* enqueue `njobs` loads (slots ws[0..njobs)); returns immediately, workers run ahead.
 * Order is load-bearing: write all batch state RELAXED, then RELEASE-store cur to
 * publish it, then wake parked workers. */
static void pipe_dispatch(Model *m,int layer,const int *eids,int njobs){
    g_pp.m=m;
    atomic_store_explicit(&g_pp.njobs,njobs,memory_order_relaxed);
    atomic_store_explicit(&g_pp.layer,layer,memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.eids[q],eids[q],memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.ready[q],0,memory_order_relaxed); /* reset BEFORE publish */
    uint64_t g=(atomic_load_explicit(&g_pp.cur,memory_order_relaxed)>>8)+1;
    atomic_store_explicit(&g_pp.cur,(g<<8),memory_order_release);                          /* PUBLISH */
    pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv); pthread_mutex_unlock(&g_pp.mx);
}
static inline void pipe_wait(int q){
    while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) sched_yield();
}
/* EXPOSED-STALL measurement: consumer-blocked critical-path time only. If ready[q] is
 * ALREADY set the instant we check, this job's fetch was fully hidden behind whatever the
 * consumer did between dispatch and this call (compute, or waiting on an earlier job in the
 * same block) -- returns 0, exactly the Stage-0 "fully overlapped -> ~zero exposed stall"
 * requirement. Otherwise the consumer genuinely cannot proceed: times ONLY the remaining
 * wait, so overlap already "spent" waiting on an earlier job in the block is never double-
 * counted onto this one. n_waits/n_blocked is a queue-occupancy proxy (fraction of waits
 * that actually had to block), reported alongside, not folded into the stall sum. */
static inline double pipe_wait_timed(int q, uint64_t *n_waits, uint64_t *n_blocked){
    (*n_waits)++;
    if(atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) return 0.0;
    (*n_blocked)++;
    double t0=now_s();
    while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) sched_yield();
    return now_s()-t0;
}

/* prefetch asincrono dei pesi di un expert (e delle sue scale .qs): avvia il readahead
 * cosi' le letture sincrone successive trovano la page-cache calda. */
static void expert_prefetch(Model *m, int layer, int eid){
    char nm[300];
    const char *suf[3]={"gate_proj.weight","up_proj.weight","down_proj.weight"};
    for(int k=0;k<3;k++){
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s",layer,eid,suf[k]); st_prefetch(&m->S,nm);
        char qs[320]; snprintf(qs,sizeof(qs),"%s.qs",nm); st_prefetch(&m->S,qs);
    }
}

/* ILI_IO_TRACE=<path>: DEBUG-ONLY proof log for the throttle lever's single-variable claim
 * (deliverable 2, the stage-0 validation record). Off by default (empty path -> a
 * single cheap string check, no I/O). When set, appends one line per dispatched miss-block:
 * "<layer> <nmiss> <eid0>,<eid1>,...\n" -- the exact set/order of expert ids requested,
 * captured at dispatch, BEFORE any read or delay happens. Run the identical trace twice
 * (ILI_IO_DELAY_US=0 vs >0) and diff the two trace files: routing/selection is pure compute
 * upstream of the read path, so a sleep call in expert_load cannot feed back into it --
 * byte-identical trace files are the mechanical, run-to-run proof that the lever perturbs
 * timing only, never request count, grouping, or order. */
static char g_io_trace_path[512]="";
static pthread_mutex_t g_io_trace_mtx = PTHREAD_MUTEX_INITIALIZER;
static void io_trace_log(int layer, const int *eids, int n){
    if(!g_io_trace_path[0]) return;
    pthread_mutex_lock(&g_io_trace_mtx);
    FILE *f=fopen(g_io_trace_path,"a");
    if(f){
        fprintf(f,"%d %d ",layer,n);
        for(int i=0;i<n;i++) fprintf(f,"%s%d",i?",":"",eids[i]);
        fprintf(f,"\n"); fclose(f);
    }
    pthread_mutex_unlock(&g_io_trace_mtx);
}

/* ---- helper per l'ABSORPTION: accesso per-riga ai QT quantizzati ---- */
/* acc[0..I) += coef * W[row,:] (dequant al volo) */
static void qt_addrow(const QT *t, int row, float coef, float *acc){
    int I=t->I;
    if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=coef*w[i]; return; }
    float c=coef*t->s[row];
    if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=c*(float)w[i]; return; }
    if(t->fmt==2){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
        for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc[i]+=c*((int)(b&0xF)-8); acc[i+1]+=c*((int)(b>>4)-8); }
        if(I&1){ uint8_t b=w[I>>1]; acc[I-1]+=c*((int)(b&0xF)-8); } return; }
    const uint8_t *w=t->q4+(int64_t)row*((I+3)/4);
    for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; acc[i]+=c*((int)((b>>((i&3)*2))&3)-2); }
}
/* y[0..n) = W[r0+j,:]·x  (matvec su una FETTA di righe del QT) */
static void qt_matvec_rows(const QT *t, int r0, int n, const float *x, float *y){
    int I=t->I;
    for(int j=0;j<n;j++){ int row=r0+j; double a=0;
        if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) a+=(double)w[i]*x[i]; }
        else if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; float s=t->s[row];
            float acc=0; for(int i=0;i<I;i++) acc+=(float)w[i]*x[i]; a=acc*s; }
        else if(t->fmt==2){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2); float s=t->s[row]; float acc=0;
            for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
            if(I&1){ uint8_t b=w[I>>1]; acc+=((int)(b&0xF)-8)*x[I-1]; } a=acc*s; }
        else { const uint8_t *w=t->q4+(int64_t)row*((I+3)/4); float s=t->s[row]; float acc=0;
            for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; acc+=((int)((b>>((i&3)*2))&3)-2)*x[i]; } a=acc*s; }
        y[j]=(float)a;
    }
}
static int g_absorb=-1;   /* ABSORB: -1 auto (decode S<=4), 0 mai, 1 sempre (test) */
static int g_dsa_force=0; /* DSA_FORCE=1: selezione sempre attiva (test: top-min(k,T)=denso) */
/* Extracted for direct unit-testability (c/tests/test_dsa_gate_regression.c, via the same
 * `#define main ...; #include "../glm.c"` trick tests/test_idot.c already uses) -- pinning
 * the S>4 Metal-prefill-attention gate's real activation condition, regression-tested
 * against the bug it replaced: the OLD gate used `!dsel`, where dsel/dnsel alias
 * m->dsa_sel/m->dsa_nsel -- buffers allocated ONCE and never freed between layers or
 * requests, so that pointer is non-NULL for the rest of the process as soon as any FULL DSA
 * layer has ever run, regardless of whether selection is actually restricting anything on
 * THIS call. Returns true iff DSA top-k selection MAY be restricting attention on this call
 * (in which case the Metal S>4 kernel below, which only ever computes plain dense causal
 * attention, must NOT be used) -- kept OUTSIDE any #ifdef ILI_METAL guard so it stays
 * testable in non-Metal builds too. */
static inline int dsa_gate_blocks_metal_prefill(int has_dsa, int n_layers, int layer,
                                                 int dsa_force, int pos_base, int S, int index_topk){
    return has_dsa && layer<n_layers && (dsa_force || (pos_base+S) > index_topk);
}
static int cmp_fdesc(const void *a,const void *b){
    float x=*(const float*)a, y=*(const float*)b; return x<y?1:x>y?-1:0; }

/* attenzione MLA con KV-cache compressa, su token nuovi x[S,hidden], pos_base = pos del primo */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    Cfg *c=&m->c; int H=c->n_heads, D=c->hidden, qh=c->qk_head, vh=c->v_head;
    int kvb_dim=H*(c->qk_nope+vh), Tk=pos_base+S;
    double ta0=now_s();
#ifdef ILI_METAL
    /* Fused decode attention on GPU: whole layer in one command buffer (keeps the GPU hot).
     * S<=4 absorption path with st0==0, DSA selection inactive, and GLM-5.2 int4 dims. */
    if(g_metal_enabled && S<=4 && (g_absorb==1||(g_absorb<0&&S<=4)) && m->kv_start[layer]==0
       && D==6144 && H==64 && c->q_lora==2048 && c->kv_lora==512 && c->qk_nope==192
       && c->qk_rope==64 && vh==256 && l->kv_b.fmt==2){
        /* Gate on the SAME condition as the S>4 prefill path (dsa_gate_blocks_metal_prefill):
         * once (pos_base+S) > index_topk, DSA selection restricts attention on EVERY layer --
         * SHARED indexer layers (idx_type==0) reuse the last FULL layer's top-k list in the
         * CPU path (see the dsel/dnsel reuse in the CPU code below). The previous gate also
         * tested c->idx_type[layer], which let SHARED layers keep taking this GPU kernel
         * (plain dense causal attention) past index_topk -- silently diverging from the CPU
         * reference exactly like the prefill-gate bug that helper's comment documents
         * (2026-07-21 bug pass). */
        int sel_active = dsa_gate_blocks_metal_prefill(m->has_dsa,c->n_layers,layer,g_dsa_force,pos_base,S,c->index_topk);
        if(!sel_active){
            if(m->has_dsa && layer<c->n_layers && c->idx_type[layer]){   /* index keys for future selection */
                for(int s=0;s<S;s++){ int pos=pos_base+s; float *kd=m->Ic[layer]+(int64_t)pos*c->index_hd;
                    matmul_qt(kd, x+(int64_t)s*D, &m->ix_wk[layer], 1);
                    layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], c->index_hd, 1e-6f);
                    rope_interleave(kd, pos, c); }
            }
            #define WP_(q) ((q).fmt==1?(const void*)(q).q8:(const void*)(q).q4)
            int ok = ili_metal_attn_decode(x,
                WP_(l->q_a), l->q_a.s, l->q_a.fmt, l->q_a_ln,
                WP_(l->q_b), l->q_b.s, l->q_b.fmt,
                WP_(l->kv_a), l->kv_a.s, l->kv_a.fmt, l->kv_a_ln,
                WP_(l->kv_b), l->kv_b.s, l->kv_b.fmt,
                WP_(l->o), l->o.s, l->o.fmt,
                m->Lc[layer], m->Rc[layer], S, pos_base, m->kv_start[layer], c->eps, c->theta, c->attn_scale, out);
            #undef WP_
            if(ok){ m->t_attn += now_s()-ta0; return; }
        }
    }
#endif
    float *ctx=falloc((int64_t)S*H*vh);
    float *Q=falloc((int64_t)S*H*qh);                  /* query (roped) dei token nuovi */
    float *QR=falloc((int64_t)S*c->q_lora), *comp=falloc((int64_t)S*(c->kv_lora+c->qk_rope));
    /* 1) proiezioni q_a/q_b/kv_a dei token nuovi come GEMM a S righe (prefill: auto-dispatch
     * Metal a S>=g_metal_gemm_min dentro matmul_qt; per riga il percorso CPU e' identico al
     * vecchio loop per-token). Poi per token: norme+RoPE e scrittura in cache.
     * QR tiene il residuo q_a per TUTTE le posizioni: serve anche all'indexer DSA. */
#ifdef ILI_METAL
    g_mm_forcecpu = !g_metal_prefill;   /* byte-exact default: GPU projections only with ILI_METAL_PREFILL=1 */
#endif
    matmul_qt(QR, x, &l->q_a, S);
    matmul_qt(comp, x, &l->kv_a, S);
    for(int s=0;s<S;s++){
        float *qresid=QR+(int64_t)s*c->q_lora;
        rmsnorm(qresid, qresid, l->q_a_ln, c->q_lora, c->eps);
    }
    matmul_qt(Q, QR, &l->q_b, S);
#ifdef ILI_METAL
    g_mm_forcecpu = 0;
#endif
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        float *qfull=Q+(int64_t)s*H*qh;
        for(int h=0;h<H;h++) rope_interleave(qfull+(int64_t)h*qh+c->qk_nope, pos, c);
        const float *cs=comp+(int64_t)s*(c->kv_lora+c->qk_rope);
        float *Ldst=m->Lc[layer]+(int64_t)pos*c->kv_lora, *Rdst=m->Rc[layer]+(int64_t)pos*c->qk_rope;
        memcpy(Ldst, cs, c->kv_lora*sizeof(float));
        rmsnorm(Ldst, Ldst, l->kv_a_ln, c->kv_lora, c->eps);     /* latente normato */
        memcpy(Rdst, cs+c->kv_lora, c->qk_rope*sizeof(float));
        rope_interleave(Rdst, pos, c);                            /* k_rot roped, condiviso fra teste */
    }
    /* ---- DSA lightning indexer ----
     * Layer FULL: k_idx dei token nuovi in cache + selezione top-k per query (riusata
     * dai layer SHARED successivi). Selezione attiva solo con contesto > index_topk
     * (o DSA_FORCE=1 per il test: selezionare TUTTO deve dare l'output denso esatto). */
    const int *dsel=NULL, *dnsel=NULL; int dtopk=0;
    if(m->has_dsa && layer<c->n_layers && m->kv_start[layer]==0){
        int nh=c->index_nh, hd=c->index_hd; dtopk=c->index_topk;
        if(c->idx_type[layer]){
            /* k dell'indexer in un solo GEMM a S righe (le posizioni sono contigue in Ic) */
#ifdef ILI_METAL
            g_mm_forcecpu = !g_metal_prefill;
#endif
            matmul_qt(m->Ic[layer]+(int64_t)pos_base*hd, x, &m->ix_wk[layer], S);
#ifdef ILI_METAL
            g_mm_forcecpu = 0;
#endif
            for(int s=0;s<S;s++){
                int pos=pos_base+s;
                float *kd=m->Ic[layer]+(int64_t)pos*hd;
                layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], hd, 1e-6f);
                rope_interleave(kd, pos, c);                 /* primi qk_rope dim, interleaved */
            }
            if((int64_t)S*dtopk > m->dsa_scap){
                free(m->dsa_sel); free(m->dsa_nsel);
                m->dsa_scap=(int64_t)S*dtopk;
                m->dsa_sel=malloc((size_t)m->dsa_scap*sizeof(int));
                m->dsa_nsel=malloc((size_t)S*sizeof(int));
            }
            #pragma omp parallel for schedule(dynamic,1)
            for(int s=0;s<S;s++){
                int pos=pos_base+s, nk=pos+1;
                if(nk<=dtopk && !g_dsa_force){ m->dsa_nsel[s]=0; continue; }
                int keep = nk<dtopk ? nk : dtopk;
                float *qi=falloc((int64_t)nh*hd);
                matmul_qt(qi, QR+(int64_t)s*c->q_lora, &m->ix_wq[layer], 1);
                for(int h=0;h<nh;h++) rope_interleave(qi+(int64_t)h*hd, pos, c);
                float w32[64];
                matmul_qt(w32, x+(int64_t)s*D, &m->ix_wp[layer], 1);
                float wsc=1.f/sqrtf((float)nh), rs=1.f/sqrtf((float)hd);
                float *isc=falloc(nk);
                for(int t=0;t<nk;t++){
                    const float *kt=m->Ic[layer]+(int64_t)t*hd;
                    float a=0;
                    for(int h=0;h<nh;h++){ const float *qhp=qi+(int64_t)h*hd;
                        float d0=0; for(int i=0;i<hd;i++) d0+=qhp[i]*kt[i];
                        d0*=rs; if(d0>0) a+=w32[h]*d0;       /* ReLU sullo score, poi peso */
                    }
                    isc[t]=a*wsc;
                }
                /* top-keep: soglia via qsort desc, poi scan in ordine di posizione */
                float *tmp=falloc(nk); memcpy(tmp,isc,nk*sizeof(float));
                qsort(tmp,nk,sizeof(float),cmp_fdesc);
                float thr=tmp[keep-1];
                int *dst=m->dsa_sel+(int64_t)s*dtopk, nd=0;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]>thr) dst[nd++]=t;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]==thr) dst[nd++]=t;
                m->dsa_nsel[s]=nd;
                free(qi); free(isc); free(tmp);
            }
        }
        if(m->dsa_nsel){ dsel=m->dsa_sel; dnsel=m->dsa_nsel; }
    }
    /* WEIGHT ABSORPTION (DeepSeek): per S piccoli (decode/verifica MTP) NON si ricostruisce
     * k/v per ogni token del contesto. Per linearita':
     *   q·k_nope_t = (W_K^hT q_nope)·L_t      ctx^h = W_V^h (Σ_t a_t L_t)
     * costo per step ~O(T·kv_lora) invece di O(T·H·(nope+vh)) del matmul kvb_all. */
    int absorb = g_absorb==1 || (g_absorb<0 && S<=4);
    if(absorb && c->kv_lora<=512){
        int kvl=c->kv_lora, r0v=c->qk_nope;      /* offset righe V dentro il blocco di testa */
        /* The full-context path can exceed 8192 scores when DSA selection is absent
         * (including MTP layers). Keep one correctly-sized slice per OMP worker so a
         * long context cannot overwrite the worker stack. */
        int64_t sc_cap=Tk-m->kv_start[layer];
        float *sc_all=falloc((int64_t)omp_get_max_threads()*sc_cap);
        #pragma omp parallel for collapse(2) schedule(static)
        for(int s=0;s<S;s++) for(int h=0;h<H;h++){
            int pos=pos_base+s;
            const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;
            const float *qr=qp+c->qk_nope;
            int rbase=h*(c->qk_nope+vh);
            float qabs[512]; memset(qabs,0,kvl*sizeof(float));
            for(int d=0;d<c->qk_nope;d++) qt_addrow(&l->kv_b, rbase+d, qp[d], qabs);
            float *sc=sc_all+(int64_t)omp_get_thread_num()*sc_cap;
            int st0=m->kv_start[layer];
            int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;    /* DSA: lista top-k o range pieno */
            const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
            int nt = ns ? ns : pos+1-st0;
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=m->Lc[layer]+(int64_t)t*kvl;
                const float *kr=m->Rc[layer]+(int64_t)t*c->qk_rope;
                float a=0; for(int i=0;i<kvl;i++) a+=qabs[i]*Lt[i];
                for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
                sc[jj]=a*c->attn_scale;
            }
            softmax(sc,nt);
            float clat[512]; memset(clat,0,kvl*sizeof(float));
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=m->Lc[layer]+(int64_t)t*kvl;
                float a=sc[jj]; for(int i=0;i<kvl;i++) clat[i]+=a*Lt[i]; }
            qt_matvec_rows(&l->kv_b, rbase+r0v, vh, clat, ctx+((int64_t)s*H+h)*vh);
        }
        matmul_qt(out, ctx, &l->o, S);
        free(ctx); free(Q); free(QR); free(comp); free(sc_all);
        m->t_attn += now_s()-ta0;
        return;
    }
#ifdef ILI_METAL
    /* Prefill attention on GPU (ILI_METAL_PREFILL=1): the S>4 score/softmax/AV loop is
     * the TTFT wall (the prefill-I/O study). Absorption-form kernels over the
     * compressed KV — no kvb_all reconstruction. Falls through to the CPU path on any
     * missing precondition. Q/Lc/Rc for the new tokens are already written above.
     * `dsel` above is non-NULL as soon as ANY earlier FULL DSA layer has ever run in this
     * process: m->dsa_sel/dsa_nsel are model-lifetime buffers (allocated once, never freed
     * between layers or requests), so the pointer only says "the buffer was allocated at
     * some point", not "selection is restricting attention on THIS call" — for SCORE mode
     * (has_dsa=1 models, e.g. GLM-5.2: layer 0 is FULL) that makes it non-NULL from
     * the very first layer of the very first request onward, permanently. The GPU kernel
     * below computes plain dense causal attention, which is exactly what the CPU selected
     * path *also* computes whenever selection cannot restrict anything: dsa_nsel[s] is 0
     * (dense range, see the nk<=dtopk early-out a few lines up) for every position s with
     * pos_base+s+1 <= index_topk — true for every SCORE-mode request shorter than
     * index_topk (2048; typical ctx+cont here is 50-300). So gate on the real activation
     * condition (mirrors `sel_active` in the decode path just above / layer_forward's
     * fused-CB gate) instead of the stale "was it ever allocated" pointer. g_dsa_force
     * (DSA_FORCE=1, test-only) always routes through the selection code path even when it
     * would be a no-op, so it must still steer here to CPU to keep that self-test meaningful. */
    int dsa_may_select = dsa_gate_blocks_metal_prefill(m->has_dsa,c->n_layers,layer,g_dsa_force,pos_base,S,c->index_topk);
    if(g_metal_prefill && g_metal_enabled && S>4 && !dsa_may_select && m->kv_start[layer]==0
       && D==6144 && H==64 && c->kv_lora==512 && c->qk_nope==192 && c->qk_rope==64
       && vh==256 && l->kv_b.fmt==2 && !omp_in_parallel()){
        if(ili_metal_attn_prefill(Q, (const void*)l->kv_b.q4, l->kv_b.s, l->kv_b.fmt,
                                   m->Lc[layer], m->Rc[layer], S, pos_base, c->attn_scale, ctx)){
            matmul_qt(out, ctx, &l->o, S);
            free(ctx); free(Q); free(QR); free(comp);
            m->t_attn += now_s()-ta0;
            return;
        }
    }
#endif
    /* 2) ricostruzione di k_nope+value per TUTTI i token 0..Tk-1 (un solo matmul su kv_b) */
    /* MEMORY BOUND (2026-07-21 bug pass): this buffer is Tk*n_heads*(qk_nope+v_head)*4
     * bytes with Tk = the TOTAL context so far -- O(CTX) per chunk per layer, NOT
     * O(PREFILL_CHUNK) (~112 KB/token at GLM-5.2 dims: 64*448*4). ILI_PREFILL_CHUNK bounds
     * the x/nrm/tmp/Q/ctx activations but NOT this reconstruction, so raising CTX raises
     * the transient prefill peak linearly. run_serve clamps CTX against this exact term
     * via ctx_clamp_for_ram() (ILI_CTX_FORCE=1 overrides). */
    double tk0=now_s();
    int stL=m->kv_start[layer];
    float *kvb_all=falloc((int64_t)Tk*kvb_dim);
    matmul_qt(kvb_all+(int64_t)stL*kvb_dim, m->Lc[layer]+(int64_t)stL*c->kv_lora, &l->kv_b, Tk-stL);
    m->t_kvb += now_s()-tk0;
    /* 3) attenzione causale: score = q_pass·k_nope + q_rot·k_rot */
    int64_t sc_cap=Tk-stL;
    float *sc_all=falloc((int64_t)omp_get_max_threads()*sc_cap);
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s;
        const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;          /* [qk_nope | qk_rope] */
        const float *qr=qp+c->qk_nope;
        float *sc=sc_all+(int64_t)omp_get_thread_num()*sc_cap;
        int st0=m->kv_start[layer];
        int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;        /* DSA: lista top-k o range pieno */
        const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
        int nt = ns ? ns : pos+1-st0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            const float *kn=kvb_all+(int64_t)t*kvb_dim+(int64_t)h*(c->qk_nope+vh);
            const float *kr=m->Rc[layer]+(int64_t)t*c->qk_rope;
            float a=0; for(int d=0;d<c->qk_nope;d++) a+=qp[d]*kn[d];
            for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
            sc[jj]=a*c->attn_scale;
        }
        softmax(sc,nt);
        float *cx=ctx+((int64_t)s*H+h)*vh; for(int d=0;d<vh;d++) cx[d]=0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            const float *vv=kvb_all+(int64_t)t*kvb_dim+(int64_t)h*(c->qk_nope+vh)+c->qk_nope;
            float a=sc[jj]; for(int d=0;d<vh;d++) cx[d]+=a*vv[d]; }
    }
    matmul_qt(out, ctx, &l->o, S);
    free(ctx); free(Q); free(QR); free(comp); free(kvb_all); free(sc_all);
    m->t_attn += now_s()-ta0;
}

/* MoE GLM su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, + shared expert).
 * BATCH-UNION: per S>1 (prefill, verifica MTP) ogni expert UNICO del batch viene caricato
 * una volta sola e moltiplicato per tutte le posizioni che lo usano (pesi letti 1 volta);
 * lo shared expert e' un unico matmul a S righe. Per posizione l'accumulo resta
 * nell'ordine (routed nel loro ordine di union, poi shared). */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out){
    if(g_pilot_real){   /* barriera cross-layer: prendi possesso di QUESTO layer e aspetta
                         * l'eventuale load-pilota in volo sullo stesso layer (dopodiche' il
                         * worker droppa ogni nuovo load <= layer -> ecache[layer] e' stabile
                         * per tutto il resolve/matmul/promozione qui sotto). */
        for(;;){
            pthread_mutex_lock(&g_pilot_mx);
            atomic_store_explicit(&g_cur_moe_layer,layer,memory_order_release);
            int inf=atomic_load_explicit(&g_pilot_inflight,memory_order_acquire);
            pthread_mutex_unlock(&g_pilot_mx);
            if(inf!=layer) break;
            sched_yield();
        }
    }
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    float *logit=falloc(E), *choice=falloc(E);
    int sI=c->moe_inter*c->n_shared;
    /* ---- FASE A: routing di tutte le S posizioni ---- */
    int *idxs=malloc((size_t)S*K*sizeof(int)); float *ws=malloc((size_t)S*K*sizeof(float));
    int *keff=malloc(S*sizeof(int));
#ifdef ILI_METAL
    if(g_pre_idx){                               /* routing gia' calcolata dal layer CB (GPU) */
        memcpy(idxs,g_pre_idx,(size_t)S*K*sizeof(int));
        memcpy(ws,g_pre_w,(size_t)S*K*sizeof(float));
        memcpy(keff,g_pre_keff,(size_t)S*sizeof(int));
        for(int s=0;s<S;s++){
            m->ereq+=keff[s];
            for(int kk=0;kk<keff[s];kk++){
                m->eusage[layer][idxs[(int64_t)s*K+kk]]++;
                if(m->eheat[layer][idxs[(int64_t)s*K+kk]]<UINT32_MAX) m->eheat[layer][idxs[(int64_t)s*K+kk]]++;
                g_hotset_selections++;
            }
            for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
        }
    } else
#endif
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*D;
        matmul(logit, xs, l->router, 1, D, E);
        for(int e=0;e<E;e++){ logit[e]=sigmoidf(logit[e]); choice[e]=logit[e]+l->router_bias[e]; }
        int *idx=idxs+(int64_t)s*K; float *w=ws+(int64_t)s*K;
        int Ksel = g_topk>0 ? (g_topk<K?g_topk:K) : K;
        for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
            idx[kk]=best; w[kk]=logit[best];
        }
        int Ke=Ksel;
        if(g_topp>0 && g_topp<1.f){
            for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
                while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
            float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
            float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=g_topp*tot){ Ke=kk+1; break; } }
        }
        keff[s]=Ke; m->ereq+=Ke;
        for(int kk=0;kk<Ke;kk++){
            m->eusage[layer][idx[kk]]++;
            if(m->eheat[layer][idx[kk]]<UINT32_MAX) m->eheat[layer][idx[kk]]++;
            g_hotset_selections++;
        }
        if(c->norm_topk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
        for(int kk=0;kk<Ke;kk++) w[kk]*=c->routed_scale;
        if(g_moe_dump){
            int want = g_moe_dump_nlayers==0;
            for(int z=0;z<g_moe_dump_nlayers && !want;z++) if(g_moe_dump_layers[z]==layer) want=1;
            if(want){
                int32_t hdr[4]={layer,g_a2_step,D,Ke};
                fwrite(hdr,sizeof(int32_t),4,g_moe_dump);
                fwrite(x+(int64_t)s*D,sizeof(float),D,g_moe_dump);
                for(int kk=0;kk<Ke;kk++){ int32_t id=idx[kk];
                    fwrite(&id,sizeof(int32_t),1,g_moe_dump); fwrite(&w[kk],sizeof(float),1,g_moe_dump); }
            }
        }
        for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
    }
    if(g_looka && S==1 && layer<c->n_layers){
        int Ke=keff[0];
        if(m->enr[layer]>0){                       /* [0] vs routing del token precedente */
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<m->enr[layer];z++)
                if(m->eroute[layer][z]==idxs[kk]){ la_hit[0]++; break; }
            la_tot[0]+=Ke;
        }
        for(int kind=0;kind<2;kind++) if(la_val[kind][layer]){   /* [1]/[2] vs predizioni */
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<K;z++)
                if(la_pred[kind][layer][z]==idxs[kk]){ la_hit[1+kind]++; break; }
            la_tot[1+kind]+=Ke; la_val[kind][layer]=0;
        }
    }
    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];
    /* ---- FASE B: union degli expert del batch ---- */
    int *uniq=malloc((size_t)E*sizeof(int)); int nu=0;
    unsigned char seen[E]; memset(seen,0,(size_t)E);
    for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){
        int e=idxs[(int64_t)s*K+kk];
        if(!seen[e]){ seen[e]=1; uniq[nu++]=e; }
    }
    /* ---- FASE C/D: risolvi (pin/cache/disco) e calcola, a blocchi di 64 unici ---- */
    float *xg=falloc((int64_t)S*D), *gg=falloc((int64_t)S*I), *uu=falloc((int64_t)S*I), *hh=falloc((int64_t)S*D);
    int *rows=malloc(S*sizeof(int)); float *rw=malloc(S*sizeof(float));
    int shared_on_gpu=0; (void)shared_on_gpu;   /* set by the Metal path when Phase E was fused */
    for(int base=0;base<nu;base+=64){
        int nb = nu-base<64 ? nu-base : 64;
        ESlot *use[64]; int missk[64]; int qof[64]; int nmiss=0;
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; use[j]=NULL; qof[j]=-1;
            ESlot *P=m->pin[layer];
            for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ m->hits++; use[j]=&P[z]; break; }
            if(!use[j]){ ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
                for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ m->hits++; Sl[z].used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED); use[j]=&Sl[z]; break; } }
            if(!use[j]){ qof[j]=nmiss; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++; }
        }
#ifdef ILI_METAL
        /* GPU/disk OVERLAP: submit the RESIDENT experts (pin/LRU hits, + shared expert on
         * the first block) to the GPU BEFORE loading the missed experts from disk, so the
         * preads run while the GPU computes; the missed subset follows in a second submit.
         * Per-subset CPU fallback on unresolved slab / bad fmt / GPU fault.
         *
         * MIXED-FORMAT SAFETY: a 64-expert unique-union block can span experts of DIFFERENT
         * quantization formats (a mixed int4/int2 container demotes experts individually, not
         * per-layer), but ili_metal_moe_block[_begin]() takes ONE scalar fmt for its whole
         * batch. MB_BUILD's PF argument filters to exactly one format per call, so the block is
         * partitioned into up to 4 per-format sub-blocks (one Metal submit per format actually
         * present) instead of ever letting one Metal call span more than one format. An expert
         * whose own g/u/d tensors disagree in format, or whose format's sub-block fails (bad
         * slab / GPU fault), is simply never marked `handled` and falls through to the ordinary
         * per-expert CPU loop below -- never a silent format-mismatched GPU submit. */
        int is_miss[64]={0}; int handled[64]={0};
        IliMetalMoeHandle *mh[4]={0,0,0,0};
        int mh_n[4]={0,0,0,0}; int mh_jidx[4][65]; int mh_sh[4]={0,0,0,0};
        int nbb=0, Rtot=0, sh_in=0; int jidx[65];
        const void *MG[65],*MU[65],*MD[65]; const float *MGS[65],*MUS[65],*MDS[65];
        int xoffb[65],nrb[65];
        float *mxg=NULL; int *mrows=NULL; float *mrw=NULL;
        /* subset builder: experts with is_miss==WANTMISS and (self-consistent) format PF,
         * + the shared expert when TRY_SH and its own g/u/d format is also PF. */
        #define MB_BUILD(WANTMISS, TRY_SH, PF) do{ \
            nbb=0; Rtot=0; sh_in=0; \
            for(int j=0;j<nb;j++){ if(is_miss[j]!=(WANTMISS)) continue; \
                int eid=uniq[base+j]; ESlot *e=use[j]; \
                if(e->g.fmt!=(PF) || e->u.fmt!=e->g.fmt || e->d.fmt!=e->g.fmt) continue; \
                int cnt=0; \
                for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++) \
                    if(idxs[(int64_t)s*K+kk]==eid){ cnt++; break; } \
                if(!cnt) continue; \
                MG[nbb]=e->g.fmt==1?(const void*)e->g.q8:(const void*)e->g.q4; \
                MU[nbb]=e->u.fmt==1?(const void*)e->u.q8:(const void*)e->u.q4; \
                MD[nbb]=e->d.fmt==1?(const void*)e->d.q8:(const void*)e->d.q4; \
                MGS[nbb]=e->g.s; MUS[nbb]=e->u.s; MDS[nbb]=e->d.s; \
                xoffb[nbb]=Rtot; nrb[nbb]=cnt; jidx[nbb]=j; Rtot+=cnt; nbb++; \
            } \
            if((TRY_SH) && c->n_shared==1 && sI==I && l->sh_gate.fmt==(PF) && \
               l->sh_up.fmt==(PF) && l->sh_down.fmt==(PF)){ \
                MG[nbb]=(PF)==1?(const void*)l->sh_gate.q8:(const void*)l->sh_gate.q4; \
                MU[nbb]=(PF)==1?(const void*)l->sh_up.q8  :(const void*)l->sh_up.q4; \
                MD[nbb]=(PF)==1?(const void*)l->sh_down.q8:(const void*)l->sh_down.q4; \
                MGS[nbb]=l->sh_gate.s; MUS[nbb]=l->sh_up.s; MDS[nbb]=l->sh_down.s; \
                xoffb[nbb]=Rtot; nrb[nbb]=S; jidx[nbb]=-1; Rtot+=S; nbb++; sh_in=1; \
            } \
            int p=0; \
            for(int j=0;j<nb;j++){ if(is_miss[j]!=(WANTMISS)) continue; \
                int eid=uniq[base+j]; ESlot *e=use[j]; \
                if(e->g.fmt!=(PF) || e->u.fmt!=e->g.fmt || e->d.fmt!=e->g.fmt) continue; \
                for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++) \
                    if(idxs[(int64_t)s*K+kk]==eid){ \
                        memcpy(mxg+(int64_t)p*D, x+(int64_t)s*D, D*sizeof(float)); \
                        mrows[p]=s; mrw[p]=ws[(int64_t)s*K+kk]; p++; break; } } \
            if(sh_in) for(int s=0;s<S;s++){ \
                memcpy(mxg+(int64_t)p*D, x+(int64_t)s*D, D*sizeof(float)); \
                mrows[p]=s; mrw[p]=1.0f; p++; } \
        }while(0)
        if(g_metal_enabled){
            for(int q=0;q<nmiss;q++) is_miss[missk[q]]=1;
            mxg=falloc((int64_t)(nb+1)*S*D);
            mrows=malloc((size_t)(nb+1)*S*sizeof(int)); mrw=malloc((size_t)(nb+1)*S*sizeof(float));
            for(int pf=0; pf<4; pf++){
                MB_BUILD(0, base==0 && !g_pre_sh, pf);
                if(nbb>0){
                    double t0=now_s();
                    mh[pf]=ili_metal_moe_block_begin(nbb,D,I,pf,MG,MU,MD,MGS,MUS,MDS,mxg,xoffb,nrb,mrows,mrw);
                    m->t_emm += now_s()-t0;
                    if(mh[pf]){ mh_n[pf]=nbb; memcpy(mh_jidx[pf],jidx,(size_t)nbb*sizeof(int)); mh_sh[pf]=sh_in; }
                }
            }
        }
#endif
        /* Expert loads run HERE, after the resident-experts GPU submit above: under METAL the
         * preads overlap the GPU compute (that submit is async). With METAL off the submit block
         * is a no-op / compiled out, so this sits exactly where dev put it and CPU behaviour is
         * unchanged. */
        if(nmiss){
            int eids[64]; for(int q=0;q<nmiss;q++) eids[q]=uniq[base+missk[q]];
            io_trace_log(layer,eids,nmiss);   /* debug proof-of-single-variable, ILI_IO_TRACE only */
            if(g_pipe && !g_pp.started) pipe_init(m);   /* may clear g_pipe if no worker could start */
            if(g_pipe){                            /* PIPE: launch loads async, matmul overlaps them */
                double t0=now_s();
                pipe_dispatch(m,layer,eids,nmiss);
                m->t_edisk += now_s()-t0;           /* dispatch only; real reads hide behind matmul */
            } else { double t0=now_s();             /* ORIGINALE: blocking parallel load */
                if(g_a2_on){ g_a2_issue=t0; g_a2_nmiss=(nmiss>64?64:nmiss); }
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++){ expert_load(m,layer,uniq[base+missk[q]],&m->ws[q],1);
                    if(g_a2_on && q<64) g_a2_comp[q]=now_s(); }
                double ddt=now_s()-t0; m->t_edisk += ddt;
                if(g_a2_on){ g_a2_load_end=now_s(); g_a2_nb=nb; g_a2_bytes=(int64_t)g_a2_nmiss*18915328; }
                /* No PIPE = no overlap mechanism at all here: this whole span is, by
                 * construction, consumer-blocked-because-data-unavailable, so it is fully
                 * exposed stall too (t_stall_exposed mirrors t_edisk exactly in this mode). */
                m->t_stall_exposed += ddt; }
        }
        /* I/O ASINCRONO: readahead (WILLNEED) del blocco SUCCESSIVO mentre calcoliamo
         * questo — il kernel legge in background, le pread dopo trovano cache calda */
        if(base+64<nu){
            int nb2 = nu-(base+64)<64 ? nu-(base+64) : 64;
            for(int j=0;j<nb2;j++){ int eid=uniq[base+64+j]; int found=0;
                ESlot *P=m->pin[layer];
                for(int z=0;z<m->npin[layer] && !found;z++) if(P[z].eid==eid) found=1;
                ESlot *Sl=m->ecache[layer];
                for(int z=0;z<m->ecn[layer] && !found;z++) if(Sl[z].eid==eid) found=1;
                if(!found) expert_prefetch(m,layer,eid);
            }
        }
#ifdef ILI_METAL
        if(g_metal_enabled){
            /* PIPE drain. Two reasons this barrier is mandatory here, and not optional:
             *  1) MB_BUILD(1) hands the missed experts' slabs straight to the GPU — a slot still
             *     being pread by an I/O worker would be matmul-ed half-loaded.
             *  2) PIPE's only drain barrier is the per-expert pipe_wait() in the CPU matmul loop
             *     below, which the handled[] skip can bypass entirely for a GPU-done expert.
             *     Without this, a still-writing worker would race the end-of-block LRU swap that
             *     recycles ws[].
             * pipe_wait() is an idempotent spin on ready[q], so the per-expert waits below stay
             * correct (and free) when a subset falls back to the CPU. */
            if(g_pipe && nmiss){ double tw=now_s();
                for(int q=0;q<nmiss;q++)
                    m->t_stall_exposed += pipe_wait_timed(q,&m->n_pipe_waits,&m->n_pipe_waits_blocked);
                m->t_edisk += now_s()-tw; }
            for(int pf=0; pf<4; pf++){
                MB_BUILD(1, 0, pf);                           /* missed experts, now loaded */
                if(nbb>0){
                    double t0=now_s();
                    if(ili_metal_moe_block(nbb,D,I,pf,MG,MU,MD,MGS,MUS,MDS,mxg,xoffb,nrb,mrows,mrw,out,S))
                        for(int t=0;t<nbb;t++) if(jidx[t]>=0) handled[jidx[t]]=1;
                    m->t_emm += now_s()-t0;
                }
            }
            for(int pf=0; pf<4; pf++){
                if(!mh[pf]) continue;
                double t0=now_s();
                int ok=ili_metal_moe_block_end(mh[pf],out);
                m->t_emm += now_s()-t0; mh[pf]=NULL;
                if(ok){
                    for(int t=0;t<mh_n[pf];t++) if(mh_jidx[pf][t]>=0) handled[mh_jidx[pf][t]]=1;
                    if(mh_sh[pf]) shared_on_gpu=1;
                }
                /* on failure those residents are simply left un-handled -> CPU loop below */
            }
            free(mxg); free(mrows); free(mrw);
        }
        #undef MB_BUILD
#endif
        if(g_a2_on) g_a2_cs=now_s();
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
            /* Drain this miss's async load BEFORE the nr==0 early-exit below: every
             * dispatched slot must be waited before the end-of-block LRU swap can reuse
             * its ws[] slab, so correctness does not depend on the nr>=1 routing invariant.
             * Stays ABOVE the METAL skip: a subset that fell back to the CPU still needs its
             * slot drained here, and under METAL the block-level drain above already ran (this
             * spin is then a no-op). */
            if(g_pipe && qof[j]>=0){ double tw=now_s();
                m->t_stall_exposed += pipe_wait_timed(qof[j],&m->n_pipe_waits,&m->n_pipe_waits_blocked);
                m->t_edisk += now_s()-tw; }
#ifdef ILI_METAL
            /* skip experts already computed on GPU by one of the per-format sub-blocks above */
            if(g_metal_enabled && handled[j]) continue;
#endif
            int nr=0;                                 /* righe (posizioni) che usano questo expert */
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
#ifdef ILI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
#endif
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
            matmul_qt(gg, xg, &e->g, nr);
            matmul_qt(uu, xg, &e->u, nr);
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
            matmul_qt(hh, gg, &e->d, nr);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            m->t_emm += now_s()-t0;
        }
        /* No drain barrier: the per-expert pipe_wait(qof[j]) above (issued for every
         * dispatched miss slot, before the nr==0 skip) already waited on all ws[] loads
         * for this block, so they are complete before the LRU swap — and the gen-tagged
         * cursor keeps any still-spinning worker off a wrong-generation slot. */
        { ESlot *Sl=m->ecache[layer]; int *nn=&m->ecn[layer];   /* promozione LRU (swap buffer) */
          int promo = nmiss<m->ecap ? nmiss : m->ecap;
          for(int a=0;a<promo;a++){ int q=nmiss-1-a; ESlot *dst;
              if(*nn<m->ecap) dst=&Sl[(*nn)++];
              else { int lru=0; for(int z=1;z<*nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; dst=&Sl[lru]; }
              ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp; dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED); }
        }
    }
    if(g_a2_on && g_a2f && g_a2_nmiss>0 && g_a2_step<g_a2_steps){
        double ce=now_s();
        fprintf(g_a2f,"{\"step\":%d,\"layer\":%d,\"nmiss\":%d,\"nb\":%d,",g_a2_step,layer,g_a2_nmiss,g_a2_nb);
        fprintf(g_a2f,"\"load_stall_ms\":%.4f,\"compute_ms\":%.4f,\"comp_off_ms\":[",
                1000.0*(g_a2_load_end-g_a2_issue), 1000.0*(ce-g_a2_cs));
        for(int q=0;q<g_a2_nmiss;q++) fprintf(g_a2f,"%s%.4f",q?",":"",1000.0*(g_a2_comp[q]-g_a2_issue));
        fprintf(g_a2f,"],\"bytes\":%lld}\n",(long long)g_a2_bytes); fflush(g_a2f);
        g_a2_nmiss=0;
    }
    /* ---- FASE E: shared expert, un matmul a S righe (skipped se fuso nel blocco GPU) ---- */
    float *sg=falloc((int64_t)S*sI), *su=falloc((int64_t)S*sI);
#ifdef ILI_METAL
    if(g_pre_sh){ for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=g_pre_sh[z]; shared_on_gpu=1; }
#endif
    if(!shared_on_gpu){
        matmul_qt(sg, x, &l->sh_gate, S);
        matmul_qt(su, x, &l->sh_up,   S);
        for(int64_t z=0;z<(int64_t)S*sI;z++) sg[z]=siluf(sg[z])*su[z];
        matmul_qt(hh, sg, &l->sh_down, S);
        for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=hh[z];
    }
    free(logit); free(choice); free(idxs); free(ws); free(keff); free(uniq);
    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw); free(sg); free(su);
}

static void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out){
    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);
    matmul_qt(g, x, &l->gate_proj, S);
    matmul_qt(u, x, &l->up_proj,   S);
    for(int64_t i=0;i<(int64_t)S*I;i++) g[i]=siluf(g[i])*u[i];
    matmul_qt(out, g, &l->down_proj, S);
    free(g); free(u);
}

/* LOOKA: predice il top-K del router del layer `target` dallo stato h (residual stream),
 * usando la STESSA pipeline del routing vero (post_ln -> router -> sigmoid+bias, top-K).
 * kind 0 = stesso layer saltando l'attention, kind 1 = layer successivo. */
static void la_predict(Model *m, int target, const float *h, int kind){
    Cfg *c=&m->c; Layer *l=&m->L[target]; int D=c->hidden, E=c->n_experts, K=c->topk;
    float *nrm=falloc(D), *ch=falloc(E);
    rmsnorm(nrm,h,l->post_ln,D,c->eps);
    matmul(ch,nrm,l->router,1,D,E);
    for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
    int *pred=la_pred[kind][target];
    for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(pred[j]==e){tk=1;break;}
            if(!tk && ch[e]>bv){bv=ch[e];best=e;} }
        pred[kk]=best; }
    la_val[kind][target]=1;
    free(nrm); free(ch);
}

/* PILOTA: prefetch guidato dal router. Predice il top-K del layer L+1 dallo stato
 * post-attention di L (recall misurato 71.6% su GLM-5.2, vs 41.3% del token precedente)
 * e lancia il WILLNEED degli expert mancanti MENTRE il MoE di L legge i suoi: il disco
 * lavora nei tempi morti del calcolo invece di aspettare il routing vero. Con MTP attiva
 * predice per TUTTE le posizioni del draft: la speculazione pilota anche l'I/O.
 * PILOT_K limita alle prime k predizioni (la testa del ranking e' piu' affidabile
 * della coda: meno banda sprecata sulle predizioni sbagliate).
 *
 * I WILLNEED partono da un THREAD I/O dedicato: con la coda disco satura la submit
 * del fadvise BLOCCA (~0.5ms x 169k chiamate = +92s/48 token, misurato) — inline
 * il pilota costava piu' di quanto rendesse. Ring lock-free 1P/1C; pieno = scarta
 * (un hint perso non e' un errore). */
static struct { int l,e; } pilot_q[4096];
static volatile unsigned pilot_w=0, pilot_r=0;
static Model *pilot_m=NULL;
/* PILOT_REAL: load VERO dell'expert predetto dentro la LRU del layer FUTURO. Vedi
 * l'invariante di sicurezza accanto a g_pilot_real. Il pread (lento) gira FUORI dal lock;
 * il lock protegge solo la scelta/pubblicazione dello slot e l'handshake col main. */
static void pilot_realload(Model *m, int layer, int eid){
    pthread_mutex_lock(&g_pilot_mx);
    if(layer <= atomic_load_explicit(&g_cur_moe_layer,memory_order_acquire)){
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
        pthread_mutex_unlock(&g_pilot_mx); return;      /* il main possiede gia' questo layer */
    }
    ESlot *P=m->pin[layer];                             /* gia' residente (pin o ecache)? skip */
    for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ pthread_mutex_unlock(&g_pilot_mx); return; }
    ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
    for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ pthread_mutex_unlock(&g_pilot_mx); return; }
    int slot,isnew;                                     /* cresci se c'e' posto, altrimenti LRU */
    if(nn<m->ecap){ slot=nn; isnew=1; }
    else { int lru=0; for(int z=1;z<nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; slot=lru; isnew=0; }
    ESlot *dst=&Sl[slot];
    dst->eid=-1;                                        /* nascondi dagli scan-hint mentre carica */
    atomic_store_explicit(&g_pilot_inflight,layer,memory_order_release);
    pthread_mutex_unlock(&g_pilot_mx);

    int rc=expert_load(m,layer,eid,dst,0);              /* pread VERO — fuori dal lock, sovrapposto al compute; fatal=0: un errore su una speculazione NON deve uccidere il server */

    pthread_mutex_lock(&g_pilot_mx);
    if(rc==0){
        dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED);
        if(isnew) m->ecn[layer]=slot+1;                 /* pubblica lo slot SOLO ora che eid e' valido */
        atomic_fetch_add_explicit(&g_pilot_loads,1,memory_order_relaxed);
    } else {
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed); /* load fallito: slot resta nascosto (eid=-1), mai pubblicato */
    }
    atomic_store_explicit(&g_pilot_inflight,-1,memory_order_release);
    pthread_mutex_unlock(&g_pilot_mx);
    if(rc!=0)                                            /* mai swallow silenzioso: logga (una riga) e prosegui */
        fprintf(stderr,"[PILOT] load speculativo abbandonato: layer %d expert %d (I/O error/short read) — nessun impatto sull'output\n",layer,eid);
}
static void *pilot_worker(void *arg){
    (void)arg;
    for(;;){
        unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
        unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
        if(r==w){ usleep(200); continue; }
        if(g_pilot_real) pilot_realload(pilot_m, pilot_q[r&4095].l, pilot_q[r&4095].e);
        else             expert_prefetch(pilot_m, pilot_q[r&4095].l, pilot_q[r&4095].e);
        __atomic_store_n(&pilot_r,r+1,__ATOMIC_RELEASE);
    }
    return NULL;
}
static void pilot_prefetch(Model *m, int lnext, const float *x, int S){
    Cfg *c=&m->c; Layer *l=&m->L[lnext]; int D=c->hidden, E=c->n_experts;
    int K = g_pilot_k<c->topk ? g_pilot_k : c->topk;
    if(!pilot_m){ pilot_m=m; pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
    float *nrm=falloc(D), *ch=falloc(E);
    for(int s=0;s<S;s++){
        rmsnorm(nrm, x+(int64_t)s*D, l->post_ln, D, c->eps);
        matmul(ch, nrm, l->router, 1, D, E);
        for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
        for(int kk=0;kk<K;kk++){
            int best=0; for(int e=1;e<E;e++) if(ch[e]>ch[best]) best=e;
            ch[best]=-2e30f;
            /* Residency scan of the FUTURE layer lnext under g_pilot_mx: with
             * PILOT_REAL=1 the pilot worker mutates ecache[lnext]/ecn[lnext]
             * concurrently, so read them under the same lock (Option A). Decide
             * under the lock, then enqueue AFTER unlocking — the pilot_q ring is
             * lock-free (pilot_w/pilot_r atomics, not g_pilot_mx) so there is no
             * re-entrant double-lock, and the worker re-checks residency under the
             * lock anyway, making a racing redundant enqueue harmless. */
            int found=0;
            pthread_mutex_lock(&g_pilot_mx);
            ESlot *P=m->pin[lnext];
            for(int z=0;z<m->npin[lnext] && !found;z++) if(P[z].eid==best) found=1;
            ESlot *Sl=m->ecache[lnext];
            for(int z=0;z<m->ecn[lnext] && !found;z++) if(Sl[z].eid==best) found=1;
            pthread_mutex_unlock(&g_pilot_mx);
            if(!found){
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    pilot_q[w&4095].l=lnext; pilot_q[w&4095].e=best;
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                }
            }
        }
    }
    free(nrm); free(ch);
}

/* forward di UN layer (usato dai 78 principali e dal layer MTP) */
static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_spec && g_prefetch && l->sparse && m->enr[li]>0)
        for(int z=0;z<m->enr[li];z++) expert_prefetch(m,li,m->eroute[li][z]);
    if(g_looka && S==1 && li<c->n_layers && l->sparse) la_predict(m,li,x,0);
#ifdef ILI_METAL
    /* FULL-LAYER CB: in_ln + attention + residuo + post_ln + shared expert + router/top-K
     * in un solo submit GPU; la CPU legge il routing e fa solo resolve/disk/expert-CB.
     * Fallback: qualsiasi condizione mancante -> percorso CPU intero qui sotto. */
    if(g_metal_enabled && S<=4 && li<c->n_layers && l->sparse
       && (g_absorb==1||(g_absorb<0&&S<=4)) && m->kv_start[li]==0
       && D==6144 && c->n_heads==64 && c->q_lora==2048 && c->kv_lora==512
       && c->qk_nope==192 && c->qk_rope==64 && c->v_head==256 && l->kv_b.fmt==2
       && c->n_experts==256 && c->topk==8 && c->n_shared==1 && c->moe_inter==2048){
        /* Same gate as attention()'s decode path / dsa_gate_blocks_metal_prefill: past
         * index_topk, selection restricts SHARED indexer layers too (they reuse the last
         * FULL layer's top-k list on the CPU path) -- the old `c->idx_type[li]` term let
         * shared layers keep running this fused GPU layer (dense attention) and silently
         * diverge from the CPU reference (2026-07-21 bug pass). */
        int sel_active = dsa_gate_blocks_metal_prefill(m->has_dsa,c->n_layers,li,g_dsa_force,pos_base,S,c->index_topk);
        if(!sel_active){
            static float *linrm,*lnrm,*lsh,*lw; static int *lidx,*lkeff;
            if(!linrm){ linrm=falloc(4*(int64_t)D); lnrm=falloc(4*(int64_t)D); lsh=falloc(4*(int64_t)D);
                        lidx=malloc(4*8*sizeof(int)); lw=malloc(4*8*sizeof(float)); lkeff=malloc(4*sizeof(int)); }
            int Ksel = g_topk>0 ? (g_topk<8?g_topk:8) : 8;
            float tp = (g_topp>0 && g_topp<1.f) ? g_topp : 0.f;
            double ta0=now_s();
            #define WP_(q) ((q).fmt==1?(const void*)(q).q8:(const void*)(q).q4)
            int ok = ili_metal_layer_decode(x, l->in_ln, l->post_ln,
                WP_(l->q_a), l->q_a.s, l->q_a.fmt, l->q_a_ln,
                WP_(l->q_b), l->q_b.s, l->q_b.fmt,
                WP_(l->kv_a), l->kv_a.s, l->kv_a.fmt, l->kv_a_ln,
                WP_(l->kv_b), l->kv_b.s, l->kv_b.fmt,
                WP_(l->o), l->o.s, l->o.fmt,
                WP_(l->sh_gate), l->sh_gate.s, l->sh_gate.fmt,
                WP_(l->sh_up),   l->sh_up.s,   l->sh_up.fmt,
                WP_(l->sh_down), l->sh_down.s, l->sh_down.fmt,
                l->router, l->router_bias,
                c->n_experts, c->topk, Ksel, tp, c->norm_topk, c->routed_scale,
                m->Lc[li], m->Rc[li], S, pos_base, m->kv_start[li],
                c->eps, c->theta, c->attn_scale,
                linrm, lnrm, lsh, lidx, lw, lkeff);
            #undef WP_
            if(ok){
                m->t_attn += now_s()-ta0;
                if(m->has_dsa && c->idx_type[li]){            /* index key per selezioni future */
                    for(int s=0;s<S;s++){ int pos=pos_base+s;
                        float *kd=m->Ic[li]+(int64_t)pos*c->index_hd;
                        matmul_qt(kd, linrm+(int64_t)s*D, &m->ix_wk[li], 1);
                        layernorm(kd, m->ix_knw[li], m->ix_knb[li], c->index_hd, 1e-6f);
                        rope_interleave(kd, pos, c);
                    }
                }
                if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
                if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse) la_predict(m,li+1,x,1);
                g_pre_idx=lidx; g_pre_w=lw; g_pre_keff=lkeff; g_pre_sh=lsh;
                moe(m,l,li,lnrm,S,tmp);
                g_pre_idx=NULL; g_pre_w=NULL; g_pre_keff=NULL; g_pre_sh=NULL;
                for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
                return;
            }
        }
    }
#endif
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
    attention(m,l,li,nrm,S,pos_base,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
    if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse) la_predict(m,li+1,x,1);
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->post_ln, D, c->eps);
    if(l->sparse) moe(m,l,li,nrm,S,tmp); else dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
}
static void layers_forward(Model *m, float *x, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_pilot_real){   /* nuovo forward: il possesso-layer riparte da -1 (i layer si rifanno da 0) */
        pthread_mutex_lock(&g_pilot_mx);
        atomic_store_explicit(&g_cur_moe_layer,-1,memory_order_release);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    for(int i=0;i<c->n_layers;i++){
        /* progresso su stderr per i batch grossi (prefill): il primo byte di risposta
         * puo' arrivare dopo MINUTI di streaming — al buio sembra un blocco. */
        if(S>=8 && (i%4==0 || i==c->n_layers-1))
            fprintf(stderr,"[prefill] layer %d/%d · %d token\n", i+1, c->n_layers, S);
        layer_forward(m,&m->L[i],i,x,S,pos_base,nrm,tmp);
    }
    free(nrm); free(tmp);
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    KVState *k=m->kv;
    if(k->Lc){ for(int i=0;i<c->n_layers+1;i++){
#ifdef ILI_METAL
        if(g_metal_enabled){ ili_metal_unregister(k->Lc[i]); ili_metal_unregister(k->Rc[i]); }
#endif
        free(k->Lc[i]); free(k->Rc[i]); } free(k->Lc); free(k->Rc); }
    if(k->Ic){ for(int i=0;i<c->n_layers;i++) free(k->Ic[i]); free(k->Ic); k->Ic=NULL; }
    if(m->has_dsa){
        k->Ic=calloc(c->n_layers,sizeof(float*));
        for(int i=0;i<c->n_layers;i++) if(c->idx_type[i]) k->Ic[i]=falloc((int64_t)max_t*c->index_hd);
    }
    k->max_t=max_t;
    int NR=c->n_layers+1;                        /* riga extra: KV del layer MTP */
    k->Lc=calloc(NR,sizeof(float*)); k->Rc=calloc(NR,sizeof(float*));
    for(int i=0;i<NR;i++){ k->Lc[i]=falloc((int64_t)max_t*c->kv_lora);
        k->Rc[i]=falloc((int64_t)max_t*c->qk_rope);
#ifdef ILI_METAL
        /* page-align + register Lc/Rc for zero-copy GPU attention. falloc isn't 16K-aligned,
         * so re-allocate aligned and register the exact byte length. */
        if(g_metal_enabled){
            size_t lb=(((size_t)max_t*c->kv_lora*sizeof(float))+16383)&~(size_t)16383;
            size_t rb=(((size_t)max_t*c->qk_rope*sizeof(float))+16383)&~(size_t)16383;
            free(k->Lc[i]); free(k->Rc[i]); void *lp,*rp;
            if(posix_memalign(&lp,16384,lb)||posix_memalign(&rp,16384,rb)){fprintf(stderr,"OOM kv\n");exit(1);}
            k->Lc[i]=lp; k->Rc[i]=rp;
            ili_metal_register(k->Lc[i],lb); ili_metal_register(k->Rc[i],rb);
        }
#endif
    }
    m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic; m->max_t=k->max_t; m->kv_start=k->kv_start;
}

static void kv_bind(Model *m, KVState *k){
    m->kv=k; m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic;
    m->max_t=k->max_t; m->kv_start=k->kv_start;
}

static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base);
static float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    if(m->has_mtp && S>=2 && g_draft>0) mtp_absorb(m, ids+1, x, S-1, pos_base);
    float *last=falloc(D); rmsnorm(last, x+(int64_t)(S-1)*D, m->final_norm, D, c->eps);
    double th0=now_s();
    float *logit=falloc(c->vocab); matmul_qt(logit,last,&m->lm_head,1);
    m->t_head += now_s()-th0;
    free(x); free(last); return logit;
}

/* come step(), ma ritorna i logits di TUTTE le S posizioni [S,vocab] (per la verifica spec) */
static float *step_all(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->h_all) memcpy(m->h_all, x, (int64_t)S*D*sizeof(float));   /* hidden di TUTTE le pos (S<=64) */
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    float *lo=falloc((int64_t)S*c->vocab), *row=falloc(D);
    for(int s=0;s<S;s++){ rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo+(int64_t)s*c->vocab, row, &m->lm_head, 1); }
    free(x); free(row); return lo;
}

/* METODO E — prompt-lookup: cerca l'occorrenza piu' recente dell'ultimo bigramma nel
 * contesto e propone i token che la seguirono. Zero pesi extra, zero costo: e' solo
 * un'ipotesi che il modello verifichera'. */
static int ngram_draft(const int *ids, int len, int G, int *draft){
    if(len<4 || G<1) return 0;
    int a=ids[len-2], b=ids[len-1];
    for(int i=len-3;i>=1;i--)
        if(ids[i-1]==a && ids[i]==b){
            int n=0; for(int j=i+1;j<len && n<G;j++) draft[n++]=ids[j];
            return n;
        }
    return 0;
}

/* METODO MTP: propone fino a G draft con la testa multi-token nativa di GLM-5.2.
 * Input: next_tok (appena emesso, posizione kv) e hlast (hidden pre-norm della pos kv-1).
 * Catena DeepSeek-V3: h' = Layer78( eh_proj[ enorm(emb(tok)) ; hnorm(h) ] ),
 * draft = argmax(lm_head(shared_head.norm(h'))). La KV del layer MTP vive alla riga n_layers
 * ed e' valida da kv_start (niente prefill: finestra di solo-decode, basta per il draft). */
static int mtp_argmax(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}
static int mtp_draft(Model *m, int next_tok, int kv, int G, int *draft){
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    int p=kv-1; if(p<0||G<1) return 0;
    if(m->kv_start[li]<0 || m->kv_start[li]>p) m->kv_start[li]=p;
    float *x=falloc(D), *cat=falloc(2*D), *hx=falloc(D), *nrm=falloc(D), *tmp=falloc(D);
    float *row=falloc(D), *logit=falloc(c->vocab), *h=falloc(D);
    memcpy(h, m->hlast, D*sizeof(float));
    int tok=next_tok, n=0;
    int prenorm = getenv("MTP_PRENORM")!=NULL;
    for(int g=0; g<G; g++){
        int pos=p+g; if(pos+2>=m->max_t) break;
        embed_row(m, tok, x);
        rmsnorm(x, x, m->enorm, D, c->eps);
        if(g==0 && !prenorm) rmsnorm(h, h, m->final_norm, D, c->eps);  /* h vero: post model.norm */
        rmsnorm(h, h, m->hnorm, D, c->eps);
        if(getenv("MTP_SWAP")){ memcpy(cat, h, D*sizeof(float)); memcpy(cat+D, x, D*sizeof(float)); }
        else { memcpy(cat, x, D*sizeof(float)); memcpy(cat+D, h, D*sizeof(float)); }
        matmul_qt(hx, cat, &m->eh_proj, 1);
        double n_eh=0; for(int d=0;d<D;d++) n_eh+=hx[d]*hx[d];
        int dbg = getenv("MTP_DEBUG") && atoi(getenv("MTP_DEBUG"))>=2;
        int t_pre=-1;
        if(dbg){ rmsnorm(row, hx, m->mtp_norm, D, c->eps); matmul_qt(logit, row, &m->lm_head, 1);
                 t_pre=mtp_argmax(logit, c->vocab); }
        layer_forward(m, &m->mtpL, li, hx, 1, pos, nrm, tmp);
        double n_post=0; for(int d=0;d<D;d++) n_post+=hx[d]*hx[d];
        rmsnorm(row, hx, m->mtp_norm, D, c->eps);
        matmul_qt(logit, row, &m->lm_head, 1);
        int t2=mtp_argmax(logit, c->vocab);
        if(dbg) fprintf(stderr,"[mtp2] pos=%d in_tok=%d ||eh||=%.1f ||post||=%.1f pre_blk=%d post_blk=%d\n",
                        pos, tok, sqrt(n_eh), sqrt(n_post), t_pre, t2);
        draft[n++]=t2; tok=t2; memcpy(h, hx, D*sizeof(float));
    }
    free(x); free(cat); free(hx); free(nrm); free(tmp); free(row); free(logit); free(h);
    return n;
}
/* assorbe nella KV della testa MTP le coppie VERIFICATE (emb(token@pos+1), h_vero@pos):
 * next_ids[i] = token alla posizione pos_base+i+1; x[i] = hidden VERO a pos_base+i.
 * Un solo passaggio batch del layer MTP (il batch-union rende economici gli expert). */
static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base){
    if(!m->has_mtp || S<1) return;
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    if(m->kv_start[li]<0 || m->kv_start[li]>pos_base) m->kv_start[li]=pos_base;
    float *hx=falloc((int64_t)S*D), *cat=falloc(2*D), *e=falloc(D), *hn=falloc(D), *hf=falloc(D);
    int prenorm = getenv("MTP_PRENORM")!=NULL;
    for(int i=0;i<S;i++){
        embed_row(m,next_ids[i],e);
        rmsnorm(e,e,m->enorm,D,c->eps);
        if(prenorm) rmsnorm(hn,x+(int64_t)i*D,m->hnorm,D,c->eps);
        else { rmsnorm(hf,x+(int64_t)i*D,m->final_norm,D,c->eps);   /* vLLM: h POST model.norm */
               rmsnorm(hn,hf,m->hnorm,D,c->eps); }
        if(getenv("MTP_SWAP")){ memcpy(cat,hn,D*sizeof(float)); memcpy(cat+D,e,D*sizeof(float)); }
        else { memcpy(cat,e,D*sizeof(float)); memcpy(cat+D,hn,D*sizeof(float)); }
        matmul_qt(hx+(int64_t)i*D, cat, &m->eh_proj, 1);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    layer_forward(m,&m->mtpL,li,hx,S,pos_base,nrm,tmp);
    free(hx); free(cat); free(e); free(hn); free(hf); free(nrm); free(tmp);
}

static inline int argmax_v(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}

/* ---- METODO F: draft grammaticale (#48) ----
 * gr_feed consuma i byte di ogni token EMESSO e tiene il walker in sync con l'output;
 * grammar_draft propone lo span FORZATO successivo (un solo byte legale per posizione)
 * gia' tokenizzato. Il confine di tokenizzazione non e' garantito coincidere con quello
 * del modello: la verifica assorbe la differenza (al peggio l'ultimo draft e' rifiutato). */
static void grammar_setup(Tok *T){
    const char *gf=getenv("GRAMMAR"); if(!gf||!*gf) return;
    FILE *f=fopen(gf,"rb");
    if(!f){ fprintf(stderr,"[GRAMMAR] cannot open %s\n",gf); return; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *txt=malloc((size_t)n+1);
    if(!txt || fread(txt,1,(size_t)n,f)!=(size_t)n){
        fprintf(stderr,"[GRAMMAR] failed to read %s\n",gf); fclose(f); free(txt); return; }
    fclose(f); txt[n]=0;
    if(gr_parse(&g_gram,txt)){ fprintf(stderr,"[GRAMMAR] %s: %s\n",gf,g_gram.err); free(txt); return; }
    free(txt);
    gr_state_init(&g_gst,&g_gram);
    if(!g_gst.alive){ fprintf(stderr,"[GRAMMAR] %s: grammar cannot be evaluated (left recursion?)\n",gf); return; }
    if(getenv("GRAMMAR_DRAFT")) g_gr_max=atoi(getenv("GRAMMAR_DRAFT"));
    if(g_gr_max<1) g_gr_max=1;
    if(g_gr_max>48) g_gr_max=48;
    g_gr_T=T; g_gr_on=1;
    fprintf(stderr,"[GRAMMAR] %s: %d rules, forced span capped at %d tokens/forward\n",gf,g_gram.n,g_gr_max);
}
/* stato pulito all'inizio di ogni RISPOSTA (non tra i \x02MORE, che continuano) */
static void grammar_reset(void){
    if(!g_gr_on) return;
    gr_state_init(&g_gst,&g_gram); g_gr_armed=0;
    if(!g_gst.alive) g_gr_on=0;
}
/* consuma i byte di un token emesso. Preambolo (prima dell'arming): ignorato.
 * Desync dopo l'arming: si riarma in attesa del prossimo inizio valido — al peggio
 * i draft vengono rifiutati dalla verifica, l'output non cambia MAI. */
static void gr_feed(int t){
    if(!g_gr_on||!g_gr_T) return;
    char b[64]; int n=tok_decode(g_gr_T,&t,1,b,63);
    for(int i=0;i<n;i++){
        int r=gr_accept(&g_gst,(unsigned char)b[i]);
        if(r==1){ g_gr_armed=1; continue; }
        if(r<0){ g_gr_on=0; return; }                 /* walker spento: fine dei draft */
        if(!g_gr_armed) continue;                     /* preambolo: aspetta l'inizio */
        gr_state_init(&g_gst,&g_gram); g_gr_armed=0;  /* desync: riparti dalla radice */
        if(!g_gst.alive){ g_gr_on=0; return; }
        if(gr_accept(&g_gst,(unsigned char)b[i])==1) g_gr_armed=1;
    }
}
/* propone lo span forzato come token (max cap); 0 se la grammatica dirama qui */
static int grammar_draft(int *draft, int cap){
    if(!g_gr_on||!g_gr_armed||!g_gr_T||cap<1) return 0;
    if(g_gr_prop>=32 && g_gr_acc*2<g_gr_prop){        /* guardia adattiva, come per MTP:
        acceptance sotto il 50% = tokenizzazione fuori asse, meglio spegnersi */
        g_gr_on=0;
        fprintf(stderr,"[GRAMMAR] %.0f%% acceptance after %llu proposals: grammar drafts disabled\n",
            100.0*g_gr_acc/g_gr_prop,(unsigned long long)g_gr_prop);
        return 0;
    }
    char fb[512]; int nb=gr_forced(&g_gst,fb,(int)sizeof fb-1);
    if(nb<=0) return 0;
    int g=tok_encode(g_gr_T,fb,nb,draft,cap);
    return g>0?g:0;
}

/* ---- SAMPLING (temperatura + nucleus) con verifica speculativa LOSSLESS ----
 * Il draft (MTP/n-gram) e' DETERMINISTICO (argmax della testa): q = massa puntuale.
 * Rejection sampling di Leviathan: accetta il draft x_d con prob p(x_d); al rifiuto
 * ricampiona da p con x_d azzerato e rinormalizzato. La distribuzione risultante e'
 * ESATTAMENTE p: la speculazione resta invisibile all'output anche col sampling. */
static uint64_t g_rng=0x9E3779B97F4A7C15ULL;
static inline double rndu(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>7; g_rng^=g_rng<<17;
    return (double)(g_rng>>11)*(1.0/9007199254740992.0); }
static float *g_pbuf=NULL; static int *g_pidx=NULL;   /* buffer riusati (decode single-thread) */
static int cmp_pdesc(const void *a,const void *b){
    float pa=g_pbuf[*(const int*)a], pb=g_pbuf[*(const int*)b];
    return pa<pb ? 1 : pa>pb ? -1 : 0; }
/* costruisce in g_pbuf la distribuzione target: softmax(lo/temp) troncata a top-p g_nuc */
static void dist_build(const float *lo, int V){
    if(!g_pbuf){ g_pbuf=falloc(V); g_pidx=malloc(V*sizeof(int)); }
    float mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double s=0; float invt=1.f/(g_temp>1e-4f?g_temp:1e-4f);
    for(int i=0;i<V;i++){ g_pbuf[i]=expf((lo[i]-mx)*invt); s+=g_pbuf[i]; }
    for(int i=0;i<V;i++) g_pbuf[i]/=(float)s;
    if(g_nuc>0 && g_nuc<1.f){
        for(int i=0;i<V;i++) g_pidx[i]=i;
        qsort(g_pidx,V,sizeof(int),cmp_pdesc);
        double cum=0; int keep=V;
        for(int i=0;i<V;i++){ cum+=g_pbuf[g_pidx[i]]; if(cum>=g_nuc){ keep=i+1; break; } }
        double s2=0; for(int i=keep;i<V;i++) g_pbuf[g_pidx[i]]=0;
        for(int i=0;i<keep;i++) s2+=g_pbuf[g_pidx[i]];
        for(int i=0;i<keep;i++) g_pbuf[g_pidx[i]]/=(float)s2;
    }
}
/* campiona da g_pbuf; ban>=0 -> quel token e' escluso (rinormalizzando al volo) */
static int dist_sample(int V, int ban){
    double z = 1.0 - (ban>=0 ? g_pbuf[ban] : 0.0); if(z<=1e-12) z=1e-12;
    double u = rndu()*z, cum=0;
    for(int i=0;i<V;i++){ if(i==ban) continue; cum+=g_pbuf[i]; if(cum>=u) return i; }
    for(int i=V-1;i>=0;i--) if(i!=ban && g_pbuf[i]>0) return i;
    return 0;
}
/* prossimo token dai logits: greedy se g_temp<=0, altrimenti sampling.
 * ban = token escluso perche' rifiutato dalla verifica speculativa precedente. */
static int pick_tok(const float *lo, int V, int ban){
    if(g_temp<=0) return argmax_v(lo,V);
    dist_build(lo,V);
    return dist_sample(V,ban);
}

/* stop-set attivo (popolato da run_text/run_serve dal config; vuoto in validazione,
 * dove si genera un numero fisso di token da confrontare con l'oracolo) */
static int g_stop[9], g_nstop=0;
static inline int is_stop(int t){ for(int i=0;i<g_nstop;i++) if(t==g_stop[i]) return 1; return 0; }
static void stops_arm(const Cfg *c, int tok_eos){
    g_nstop=0;
    for(int i=0;i<c->n_stop;i++) g_stop[g_nstop++]=c->stop_ids[i];
    if(tok_eos>=0 && !is_stop(tok_eos)) g_stop[g_nstop++]=tok_eos;
    fprintf(stderr,"[stop] %d stop tokens:",g_nstop);
    for(int i=0;i<g_nstop;i++) fprintf(stderr," %d",g_stop[i]);
    fprintf(stderr,"\n");
}

/* decode greedy con SELF-SPECULATION n-gram: LOSSLESS (output identico al greedy puro).
 * Ogni forward verifica fino a g_draft token proposti dal contesto: i token accettati
 * costano UNA sola passata sui pesi -> disco e banda RAM ammortizzati su piu' token.
 * all: storia token (capacita' >= kv+n_new+g_draft+2), kv = token gia' in KV.
 * logit = logits della posizione kv-1 (dal prefill); viene liberato qui.
 * emit(tok,ud) per ogni token emesso. Ritorna i token emessi; *kv_out = nuova kv. */
static int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit,
                       void (*emit)(int,void*), void *ud, int *kv_out){
    Cfg *c=&m->c; int V=c->vocab; int emitted=0, done=0;
    int draft[64]; if(g_draft>63) g_draft=63;
    int carry_ban=-1;                    /* token rifiutato dalla verifica: escluso dal resample */
    while(emitted<n_new && !done){
        int next=pick_tok(logit,V,carry_ban); carry_ban=-1; free(logit); logit=NULL;
        if((eos>=0 && next==eos) || is_stop(next)) break;
        emit(next,ud); all[kv]=next; emitted++; m->n_emit++;
        gr_feed(next);                                  /* il walker segue l'output emesso */
        if(emitted>=n_new) break;                       /* l'ultimo token non serve forwardarlo */
        int g = 0, gsrc = 0;                            /* sorgente: 1=grammatica 2=MTP/n-gram */
        if(g_gr_on){                                    /* metodo F: prima la grammatica — dove
                                                         * forza, l'acceptance e' ~1 (#48) */
            g=grammar_draft(draft,g_gr_max);
            if(g>0) gsrc=1;
        }
        if(!g && g_draft>0){
            /* auto-off adattivo: draft che non vengono mai accettati = solo tassa disco */
            if(m->has_mtp && m->mtp_prop>=24 && m->mtp_acc*10 < m->mtp_prop){
                g_draft=0;
                fprintf(stderr,"[MTP] %.0f%% acceptance after %llu proposals: drafts disabled\n",
                    100.0*m->mtp_acc/m->mtp_prop, (unsigned long long)m->mtp_prop);
            }
        }
        if(!g && g_draft>0){
            if(m->has_mtp){ g=mtp_draft(m,next,kv,g_draft,draft); m->mtp_prop+=g; if(g)gsrc=2; }
            else { g=ngram_draft(all,kv+1,g_draft,draft); if(g)gsrc=2; }
        }
        if(g>n_new-emitted) g=n_new-emitted;
        if(kv+1+g+1>m->max_t) g=m->max_t-kv-2;
        if(g<0) g=0;
        if(gsrc==1) g_gr_prop+=(uint64_t)g;
        int S=1+g; int batch[64]; batch[0]=next; memcpy(batch+1,draft,g*sizeof(int));
        float *lo=step_all(m,batch,S,kv); m->n_fw++;
        int k=0;                                        /* verifica: accetta finche' coincide */
        if(g>0 && getenv("MTP_DEBUG")){ int veri=argmax_v(lo,V);
            fprintf(stderr,"[mtpdbg] draft0=%d verified=%d %s\n", draft[0], veri, draft[0]==veri?"HIT":"miss"); }
        while(k<g && emitted<n_new){
            int accept;
            if(g_temp<=0) accept = (argmax_v(lo+(int64_t)k*V,V)==draft[k]);
            else { dist_build(lo+(int64_t)k*V,V);          /* rejection sampling: p(draft) */
                   accept = (rndu() < g_pbuf[draft[k]]); }
            if(!accept){ if(g_temp>0) carry_ban=draft[k]; break; }
            if((eos>=0 && draft[k]==eos) || is_stop(draft[k])){ done=1; break; }
            emit(draft[k],ud); all[kv+1+k]=draft[k]; emitted++; m->n_emit++;
            gr_feed(draft[k]); k++;
        }
        if(gsrc==1) g_gr_acc+=(uint64_t)k;
        else if(gsrc==2 && m->has_mtp) m->mtp_acc+=k;
        if(m->has_mtp && k>=1) mtp_absorb(m, all+kv+1, m->h_all, k, kv);   /* KV MTP in sync coi verificati */
        /* hlast deve corrispondere all'ultima posizione ACCETTATA (kv+k), non a fine batch */
        if(m->h_all && k<S-1) memcpy(m->hlast, m->h_all+(int64_t)k*m->c.hidden, m->c.hidden*sizeof(float));
        kv += 1+k;                                      /* KV oltre kv e' stantia: verra' sovrascritta */
        logit=falloc(V); memcpy(logit, lo+(int64_t)k*V, V*sizeof(float)); free(lo);
    }
    if(logit) free(logit);
    if(kv_out) *kv_out=kv;
    return emitted;
}

/* emit callback: accumula in un array (validazione) */
typedef struct { int *dst; int n; } EmitStore;
static void emit_store(int t, void *ud){ EmitStore *e=(EmitStore*)ud; e->dst[e->n++]=t; }
/* emit callback: detokenizza e stampa in streaming (chat/run), con heartbeat */
typedef struct { Tok *T; Model *m; double t0; int count; int quiet; } EmitStream;
static void emit_stream(int t, void *ud){
    EmitStream *e=(EmitStream*)ud; char dec[64];
    int dn=tok_decode(e->T,&t,1,dec,63); dec[dn]=0; fputs(dec,stdout); fflush(stdout);
    if(!e->quiet && ++e->count%16==0){ double tt=e->m->hits+e->m->miss;
        fprintf(stderr,"\n[t=%d  RSS %.2f GB  hit %.0f%%  %.2f tok/s  %.2f tok/fw]\n", e->count,
            rss_gb(), tt?100.0*e->m->hits/tt:0.0, e->count/(now_s()-e->t0),
            e->m->n_fw?(double)e->m->n_emit/e->m->n_fw:1.0); }
}

/* teacher-forcing: un solo forward su ids[S], argmax per posizione in pred[S] */
static void forward_all(Model *m, const int *ids, int S, int *pred){
    Cfg *c=&m->c; int D=c->hidden;
    kv_alloc(m,S);
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,0);
    float *lo=falloc(c->vocab);
    float *row=falloc(D);   /* heap, not a fixed stack array: hidden may legally exceed 8192 (CKR) */
    for(int s=0;s<S;s++){
        rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo, row, &m->lm_head, 1);
        int best=0; float bv=lo[0]; for(int i=1;i<c->vocab;i++) if(lo[i]>bv){bv=lo[i];best=i;}
        pred[s]=best;
    }
    free(x); free(lo); free(row);
}

/* log-prob (log-softmax) del token target dato il vettore di logit; *am=1 se e' l'argmax */
static double logprob_target(const float *lo, int V, int target, int *am){
    float mx=lo[0]; int best=0; for(int i=1;i<V;i++){ if(lo[i]>mx){mx=lo[i];best=i;} }
    double se=0; for(int i=0;i<V;i++) se+=exp((double)lo[i]-mx);
    if(am)*am=(best==target);
    return (double)(lo[target]-mx) - log(se);
}
static void profile_print(Model *m, double elapsed);
/* modalita' SCORING per i benchmark (stile lm-eval, log-likelihood):
 * input: file con righe "<ctxlen> <contlen> <id0> .. <id_{T-1}>"  (T=ctxlen+contlen)
 * output: riga "<logprob_continuazione> <contlen> <greedy 0/1>" per richiesta.
 * Un solo forward per richiesta (teacher-forcing): niente generazione -> fattibile a bassa velocita'. */
static void run_score(Model *m, const char *path){
    Cfg *c=&m->c; int D=c->hidden;
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    int maxT=1; { char *ln=NULL; size_t cp=0;
        while(getline(&ln,&cp,f)>0){ int a,b; if(sscanf(ln,"%d %d",&a,&b)==2 && a+b>maxT) maxT=a+b; }
        free(ln); }
    kv_alloc(m,maxT);
    float *x=falloc((int64_t)maxT*D), *lo=falloc(c->vocab), *row=falloc(D);
    int *ids=malloc(maxT*sizeof(int));
    rewind(f); char *ln=NULL; size_t cp=0; int nreq=0; double t0=now_s();
    while(getline(&ln,&cp,f)>0){
        char *p=ln; int ctxlen=strtol(p,&p,10), contlen=strtol(p,&p,10), T=ctxlen+contlen;
        if(T<=0||ctxlen<1){ printf("0 0 0\n"); fflush(stdout); continue; }
        for(int i=0;i<T;i++) ids[i]=strtol(p,&p,10);
        double treq0=now_s();
        for(int s=0;s<T;s++) embed_row(m, ids[s], x+(int64_t)s*D);
        layers_forward(m,x,T,0);
        double lp=0; int greedy=1;
        for(int pos=ctxlen-1; pos<T-1; pos++){
            rmsnorm(row, x+(int64_t)pos*D, m->final_norm, D, c->eps);
            matmul_qt(lo,row,&m->lm_head,1);
            int am; lp += logprob_target(lo,c->vocab,ids[pos+1],&am); if(!am) greedy=0;
        }
        double treq_dt=now_s()-treq0;
        printf("%.6f %d %d\n", lp, contlen, greedy); fflush(stdout);
        /* T (=S passed to layers_forward) is printed alongside the per-request latency:
         * SCORE requests batch the whole ctx+cont in one forward (S=T, almost always >4),
         * so this is the number that decides whether a given request could even reach the
         * S>4 Metal prefill gate (glm.c attention(), g_metal_prefill && S>4 && ...). */
        if(++nreq%5==0) fprintf(stderr,"[score %d req | %.1fs | last T=%d %.3fs/fwd | RSS %.2f GB | hit %.0f%%]\n",
            nreq, now_s()-t0, T, treq_dt, rss_gb(), (m->hits+m->miss)?100.0*m->hits/(m->hits+m->miss):0.0);
    }
    double dt=now_s()-t0;
    fprintf(stderr,"[score DONE] %d req | %.1fs total | %.3fs/req avg\n",
        nreq, dt, nreq?dt/nreq:0.0);
    profile_print(m,dt);
    free(ln); free(ids); free(x); free(lo); free(row); fclose(f);
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out){
    kv_alloc(m,np+n_new+g_draft+2);
    for(int i=0;i<np;i++) out[i]=prompt[i];
    float *logit=step(m,prompt,np,0);
    EmitStore es={out+np,0};
    spec_decode(m,out,np,n_new,-1,logit,emit_store,&es,NULL);
}

static void profile_print(Model *m, double elapsed){
    double accounted=m->t_edisk+m->t_emm+m->t_attn+m->t_head;
    printf("PROFILE: expert-disk %.3fs | expert-matmul %.3fs | attention %.3fs "
           "(including kvb %.3fs) | lm_head %.3fs | other %.3fs\n",
        m->t_edisk,m->t_emm,m->t_attn,m->t_kvb,m->t_head,elapsed-accounted);
    /* ---- streaming-causality instrumentation (c/bench-m5max/factorial-streaming-causality-
     * the format spec). Every line above is UNCHANGED (byte-identical to before this instrument
     * landed -- downstream parsers keep working); everything below is new, additive,
     * clearly-labeled absolute per-run (and, where n_emit>0, per-token) metrics.
     * t_edisk above is the async SERVICE-time series -- explicitly overlapping, and, per
     * the wall-sum identity below, NEVER added into that sum (would double-count overlap). */
    { int64_t ntok=(int64_t)m->n_emit;
      double p50=0,p95=0,p99=0; uint64_t nlat=0; iolat_percentiles(&p50,&p95,&p99,&nlat);
      double hitpct=(m->hits+m->miss)?100.0*m->hits/(double)(m->hits+m->miss):0.0;
      printf("STALL-EXPOSED: %.3fs (consumer-blocked critical-path only, excludes overlapped "
             "service) | pipe-waits %llu blocked %llu (occupancy %.1f%%)\n",
          m->t_stall_exposed,(unsigned long long)m->n_pipe_waits,(unsigned long long)m->n_pipe_waits_blocked,
          m->n_pipe_waits?100.0*m->n_pipe_waits_blocked/(double)m->n_pipe_waits:0.0);
      if(ntok>0) printf("STALL-EXPOSED/TOKEN: %.4f ms/token (n=%lld decode tokens)\n",
          1000.0*m->t_stall_exposed/ntok,(long long)ntok);
      printf("IO-BYTES: requested %lld | read %lld | reads attempted %llu completed %llu | "
             "hits %llu misses %llu (%.1f%% hit)\n",
          (long long)m->io_bytes_requested,(long long)m->io_bytes_read,
          (unsigned long long)m->io_reads_attempted,(unsigned long long)m->io_reads_completed,
          (unsigned long long)m->hits,(unsigned long long)m->miss,hitpct);
      if(ntok>0) printf("IO-BYTES/TOKEN: requested %.1f B/token | read %.1f B/token\n",
          (double)m->io_bytes_requested/ntok,(double)m->io_bytes_read/ntok);
      printf("IO-LATENCY: read-completion p50 %.4fms p95 %.4fms p99 %.4fms (n=%llu samples)\n",
          p50*1000.0,p95*1000.0,p99*1000.0,(unsigned long long)nlat);
      /* [IOKIND] Step-0 diagnostic (#1438 deep-offload reconciliation): per-tensor-kind
       * split of the blended IO-BYTES/IO-LATENCY lines above -- main (non-mmap, non-
       * mode15) expert_load() path only. lat_sum_s is a SUM of real per-pread durations
       * (see io_kind_done()'s own comment for exactly what it excludes/includes); it is
       * NOT the same quantity as STALL-EXPOSED above (that is consumer-blocked
       * critical-path time only) -- summed pread latency can overlap real compute under
       * OMP-parallel expert loading, so do not read lat_sum_s as directly-exposed wall
       * time without checking the overlap caveat in this diagnostic's own RESULTS.md. */
      { double wsum=0,ssum=0; io_kind_latency_totals(&wsum,&ssum);
        printf("[IOKIND] weight: n=%llu bytes=%lld lat_sum_s=%.6f | scale: n=%llu bytes=%lld lat_sum_s=%.6f\n",
            (unsigned long long)m->io_reads_weight,(long long)m->io_bytes_weight,wsum,
            (unsigned long long)m->io_reads_scale,(long long)m->io_bytes_scale,ssum);
      }
      { double t_compute=m->t_emm+m->t_attn+m->t_head;
        double other=elapsed-t_compute-m->t_stall_exposed;
        printf("WALL-SUM: compute %.3fs + exposed-stall %.3fs + other %.3fs = wall %.3fs "
               "| residual(other) %.1f%% of wall (async service/read-latency/disk-activity "
               "above are OVERLAPPING telemetry, never included in this sum)\n",
            t_compute,m->t_stall_exposed,other,elapsed,elapsed>0?100.0*other/elapsed:0.0);
      }
    }
#ifdef ILI_METAL
    if(g_metal_enabled){ uint64_t ok=0,fb=0,ex=0; double su=0,gp=0,sc=0;
        ili_metal_moe_counts(&ok,&fb,&ex); ili_metal_moe_times(&su,&gp,&sc);
        { uint64_t aok=0; double aw=0,ak=0; ili_metal_attn_counts(&aok,&aw,&ak);
          if(aok){ double ks=0,gs=0; ili_metal_attn_lat(&ks,&gs);
          printf("METAL-ATTN: layer GPU %llu | gpu-wall %.2fs (kernel %.2fs | cpu-sched %.2fs gpu-sched %.2fs)\n",(unsigned long long)aok,aw,ak,ks,gs); } }
        printf("METAL: blocchi GPU %llu | fallback CPU %llu | expert su GPU %llu | setup %.2fs gpu-wall %.2fs (kernel %.2fs) scatter %.2fs\n",
               (unsigned long long)ok,(unsigned long long)fb,(unsigned long long)ex,su,gp,ili_metal_moe_kernel_time(),sc);
        { uint64_t gok=0; double gw=0,gk=0; ili_metal_gemm_counts(&gok,&gw,&gk);
          if(gok) printf("METAL-GEMM: calls %llu | gpu-wall %.2fs (kernel %.2fs)\n",(unsigned long long)gok,gw,gk); } }
#endif
}

/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ. */
/* Stage-2 v3 measurement-boundary reset (the stage-2 v3 registration): counters ONLY -- never
 * KV, expert-cache, routing, RNG, model, or treatment state. Call sites guarantee loader
 * idleness (PIPE off => the OMP parallel-for in step() has joined; PIPE on is REJECTED in
 * warmup mode below). IOLAT rings: n/next zeroed per claimed ring; g_iolat_nthreads is
 * deliberately NOT reset (worker threads keep their thread-local slot for later samples). */
static void measure_boundary_reset(Model *m){
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0;
    m->t_edisk=m->t_emm=m->t_attn=m->t_kvb=m->t_head=0;
    m->t_stall_exposed=0; m->io_bytes_requested=m->io_bytes_read=0;
    m->io_reads_attempted=m->io_reads_completed=m->n_pipe_waits=m->n_pipe_waits_blocked=0;
    m->io_bytes_weight=m->io_bytes_scale=0; m->io_reads_weight=m->io_reads_scale=0;  /* [IOKIND] */
    iolat_reset();
}

/* forward decls: kv_disk_* are defined below run_replay (serve section) */
static void kv_disk_reset(Model *m);
static void kv_disk_append(Model *m, const int *hist, int len);
static int kv_disk_load(Model *m, int *hist, int maxctx);

/* ===== Stage-2 v4: immutable 8K KV snapshots (the stage-2 v4 registration) =====
 * A snapshot is the byte-exact fp32 serialization of the post-prefill KV state (reusing the
 * proven kv_disk_* path that serve-mode resume already validated bit-exact), plus a sidecar
 * .prov file binding provenance. LOAD restores KV in place of the live 8K prefill; the expert
 * LRU is process-local and never persisted, so every arm starts with a FRESH expert cache --
 * exactly the registered "KV-restored context, cache established by 32 warmup tokens." */
static uint64_t v4_prompt_hash(const int *ids, int n){
    uint64_t h=1469598103934665603ULL;               /* FNV-1a over the prompt token ids */
    for(int i=0;i<n;i++){ uint32_t t=(uint32_t)ids[i];
        for(int b=0;b<4;b++){ h^=(t>>(b*8))&0xff; h*=1099511628211ULL; } }
    return h;
}
static void v4_prov_write(const char *snap_path, Model *m, const int *ids, int nprompt, int nrec){
    char pp[2200]; snprintf(pp,sizeof(pp),"%s.prov",snap_path);
    FILE *f=fopen(pp,"w"); if(!f){ perror(pp); exit(2); }
    Cfg *c=&m->c;
    fprintf(f,"{\"source_commit\":\"%s\",\"n_layers\":%d,\"kv_lora\":%d,\"qk_rope\":%d,"
              "\"has_dsa\":%d,\"vocab\":%d,\"prompt_hash\":\"%016llx\",\"prompt_count\":%d,"
              "\"kv_records\":%d}\n",
        ili_env("V4_SRC_COMMIT")?ili_env("V4_SRC_COMMIT"):"unset",
        c->n_layers,c->kv_lora,c->qk_rope,m->has_dsa,c->vocab,
        (unsigned long long)v4_prompt_hash(ids,nprompt),nprompt,nrec);
    fclose(f);
}
static void v4_prov_verify(const char *snap_path, Model *m, const int *ids, int nprompt, int nrec_expected){
    char pp[2200]; snprintf(pp,sizeof(pp),"%s.prov",snap_path);
    FILE *f=fopen(pp,"rb"); if(!f){ fprintf(stderr,"V4: missing provenance %s\n",pp); exit(2); }
    char buf[4096]; size_t n=fread(buf,1,sizeof(buf)-1,f); buf[n]=0; fclose(f);
    Cfg *c=&m->c; long v; char hx[32];
    #define REQ(key,want) do{ char *p=strstr(buf,"\""key"\":"); if(!p){fprintf(stderr,"V4 prov: missing %s\n",key);exit(2);} \
        v=strtol(p+strlen(key)+3,NULL,10); if(v!=(want)){fprintf(stderr,"V4 prov mismatch %s: %ld != %d (fail closed)\n",key,v,(int)(want));exit(2);} }while(0)
    REQ("n_layers",c->n_layers); REQ("kv_lora",c->kv_lora); REQ("qk_rope",c->qk_rope);
    REQ("has_dsa",m->has_dsa); REQ("vocab",c->vocab); REQ("prompt_count",nprompt); REQ("kv_records",nrec_expected);
    #undef REQ
    char *ph=strstr(buf,"\"prompt_hash\":\""); if(!ph){fprintf(stderr,"V4 prov: missing prompt_hash\n");exit(2);}
    snprintf(hx,sizeof(hx),"%016llx",(unsigned long long)v4_prompt_hash(ids,nprompt));
    if(strncmp(ph+15,hx,16)!=0){ fprintf(stderr,"V4 prov: prompt_hash mismatch (fail closed)\n"); exit(2); }
}

static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\n"); return; }
    int warm=0, meas=0;   /* Stage-2 v3: ILI_REPLAY_WARMUP=32 ILI_REPLAY_MEASURE=256 */
    { const char *w=ili_env("REPLAY_WARMUP"), *g=ili_env("REPLAY_MEASURE");
      if(w) warm=atoi(w); if(g) meas=atoi(g); }
    if(warm>0){
        /* fail closed: the trace must contain EXACTLY prefill + warm + meas decode tokens */
        if(meas<=0){ fprintf(stderr,"REPLAY v3: REPLAY_WARMUP requires REPLAY_MEASURE\n"); exit(2); }
        if(nfull-np != warm+meas){
            fprintf(stderr,"REPLAY v3: trace shape mismatch: decode tokens %d != warmup %d + measured %d\n",
                    nfull-np, warm, meas); exit(2); }
        if(g_pipe){ fprintf(stderr,"REPLAY v3: PIPE must be OFF for the measurement-boundary reset "
                                    "(loader-idle guarantee)\n"); exit(2); }
    }
    kv_alloc(m,nfull+2);
    const char *snap=ili_env("KV_SNAPSHOT");
    int save_mode = snap && ili_env("REPLAY_SAVE") && atoi(ili_env("REPLAY_SAVE"));
    float *logit=NULL;
    if(snap && !save_mode){
        /* v4 LOAD: restore post-prefill KV from the immutable snapshot (NO live prefill). */
        v4_prov_verify(snap, m, full, np, np-1);
        snprintf(m->kv->disk_path,sizeof(m->kv->disk_path),"%s",snap);
        int *hbuf=calloc(nfull+2,sizeof(int));
        int loaded=kv_disk_load(m, hbuf, nfull+2);
        if(loaded != np-1){ fprintf(stderr,"V4 LOAD: restored %d KV records != expected %d (fail closed)\n",loaded,np-1); exit(2); }
        for(int p=0;p<loaded;p++) if(hbuf[p]!=full[p]){
            fprintf(stderr,"V4 LOAD: token history mismatch at %d (%d!=%d, fail closed)\n",p,hbuf[p],full[p]); exit(2); }
        free(hbuf);
        fprintf(stderr,"V4: restored %d-token KV snapshot (no live prefill); expert cache FRESH\n",loaded);
    } else {
        logit=step(m,full,np-1,0); free(logit);       /* 8K prefill: g_decode_phase=0, decode lever OFF */
        if(save_mode){
            snprintf(m->kv->disk_path,sizeof(m->kv->disk_path),"%s",snap);
            kv_disk_reset(m); kv_disk_append(m, full, np-1);
            v4_prov_write(snap, m, full, np, np-1);
            fprintf(stderr,"V4 SAVE: wrote %d-record KV snapshot %s (+ .prov); exiting (setup artifact)\n",np-1,snap);
            return;
        }
    }
    g_decode_phase=1;                                 /* v3 treatment boundary: decode lever active */
    measure_boundary_reset(m);
    int i=np-1;
    if(warm>0){
        double tw0=now_s();
        g_a2_step=g_a2_steps;   /* a2: never log warmup tokens */
        for(int w=0;w<warm;w++,i++){ logit=step(m,full+i,1,i); free(logit); }
        fprintf(stderr,"REPLAY v3: %d warmup decode tokens under treatment in %.3fs (excluded)\n",
                warm, now_s()-tw0);
        measure_boundary_reset(m);                    /* loaders idle: !g_pipe enforced above */
    }
    double t0=now_s(); int steps=0;
    int argmax_hash_on = ili_env("REPLAY_ARGMAX_HASH") && atoi(ili_env("REPLAY_ARGMAX_HASH"));
    uint64_t amh=1469598103934665603ULL;   /* FNV-1a over per-step greedy argmax (parity fingerprint) */
    FILE *ldump = ili_env("REPLAY_LOGIT_DUMP") ? fopen(ili_env("REPLAY_LOGIT_DUMP"),"w") : NULL;
    g_moe_dump = ili_env("MOE_INPUT_DUMP") ? fopen(ili_env("MOE_INPUT_DUMP"),"wb") : NULL;
    if(g_moe_dump && ili_env("MOE_INPUT_DUMP_LAYERS")){
        char lb[256]; snprintf(lb,sizeof lb,"%s",ili_env("MOE_INPUT_DUMP_LAYERS"));
        for(char *t=strtok(lb,","); t && g_moe_dump_nlayers<8; t=strtok(NULL,","))
            g_moe_dump_layers[g_moe_dump_nlayers++]=atoi(t);
    }
    for(; i<nfull-1; i++){
        g_a2_step=steps;
        logit=step(m,full+i,1,i);
        if(argmax_hash_on){ int am=0; float best=logit[0];
            for(int v=1;v<m->c.vocab;v++) if(logit[v]>best){best=logit[v];am=v;}
            for(int b=0;b<4;b++){ amh^=((uint32_t)am>>(b*8))&0xff; amh*=1099511628211ULL; } }
        if(ldump){
            /* determinism-probe capture (observation only): top-8 ids+raw logits by rank, plus an
             * FNV-1a fingerprint over the raw float bytes of the FULL vocab logit vector. */
            int tid[8]; float tlg[8]; int V=m->c.vocab;
            for(int k=0;k<8;k++){ tid[k]=-1; tlg[k]=-3.4e38f; }
            for(int v=0;v<V;v++){ float x=logit[v];
                if(x>tlg[7]){ int k=7; while(k>0 && x>tlg[k-1]){ tlg[k]=tlg[k-1]; tid[k]=tid[k-1]; k--; }
                    tlg[k]=x; tid[k]=v; } }
            uint64_t fp=1469598103934665603ULL; const unsigned char *pb=(const unsigned char*)logit;
            for(size_t bb=0; bb<(size_t)V*sizeof(float); bb++){ fp^=pb[bb]; fp*=1099511628211ULL; }
            fprintf(ldump,"%d %d %.9g %016llx",steps,tid[0],(double)tlg[0],(unsigned long long)fp);
            for(int k=0;k<8;k++) fprintf(ldump," %d:%.9g",tid[k],(double)tlg[k]);
            fputc('\n',ldump);
        }
        free(logit); steps++;
    }
    if(ldump) fclose(ldump);
    if(g_moe_dump){ fclose(g_moe_dump); g_moe_dump=NULL; }
    if(warm>0 && steps!=meas){ fprintf(stderr,"REPLAY v3: measured %d != required %d\n",steps,meas); exit(2); }
    if(argmax_hash_on) printf("REPLAY argmax-hash: %016llx (n=%d)\n",(unsigned long long)amh,steps);
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
#ifdef ILI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
}

/* generazione reale: tokenizza PROMPT, prefill + decode greedy con stop su EOS,
 * detokenizza e stampa il testo in streaming. */
static void run_text(Model *m, const char *snap, const char *prompt, int ngen){
    Cfg *c=&m->c; char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos);
    grammar_setup(&T);                   /* metodo F: GRAMMAR=file.gbnf (#48) */
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int cap=(int)strlen(prompt)+16; int *pids=malloc(cap*sizeof(int));
    int np=tok_encode(&T,prompt,(int)strlen(prompt),pids,cap);
    if(np<1){ fprintf(stderr,"prompt is empty after tokenization\n"); return; }
    printf("prompt: %d tokens | generating up to %d (EOS stop=%d) | n-gram draft=%d\n", np, ngen, eos, g_draft);
    fputs(prompt,stdout); fflush(stdout);
    kv_alloc(m, np+ngen+g_draft+2);
    int *all=malloc((np+ngen+g_draft+2)*sizeof(int)); memcpy(all,pids,np*sizeof(int));
    double t=now_s();
    float *logit=step(m,pids,np,0);
    EmitStream es={&T,m,t,0,0};
    grammar_reset();
    int produced=spec_decode(m,all,np,ngen,eos,logit,emit_stream,&es,NULL);
    double dt=now_s()-t;
    double tot=m->hits+m->miss;
    uint64_t output_hash=14695981039346656037ULL;
    for(int i=np;i<np+produced;i++){
        output_hash^=(uint32_t)all[i];
        output_hash*=1099511628211ULL;
    }
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    printf("\n---\n%d tokens in %.2fs (%.2f tok/s) | expert hit rate %.1f%% | RSS %.2f GB\n",
        produced, dt, produced/dt, tot?100.0*m->hits/tot:0.0, rss_gb());
    printf("experts loaded/token: %.1f (per-layer %.2f across %d; baseline topk=%d) | TOPK=%d TOPP=%.2f\n",
        produced?(double)m->ereq/produced:0.0, (produced&&nsp)?(double)m->ereq/produced/nsp:0.0, nsp, c->topk, g_topk, g_topp);
    printf("speculation: %.2f tokens/forward (%llu forwards per %llu tokens) | MTP acceptance %.0f%% (%llu/%llu)\n",
        m->n_fw?(double)m->n_emit/m->n_fw:1.0, (unsigned long long)m->n_fw, (unsigned long long)m->n_emit,
        m->mtp_prop?100.0*m->mtp_acc/m->mtp_prop:0.0, (unsigned long long)m->mtp_acc, (unsigned long long)m->mtp_prop);
    printf("output token hash: %016llx\n",(unsigned long long)output_hash);
    if(g_gr_prop) printf("grammar: %.0f%% acceptance (%llu/%llu forced drafts)\n",
        100.0*g_gr_acc/g_gr_prop, (unsigned long long)g_gr_acc, (unsigned long long)g_gr_prop);
#ifdef ILI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    profile_print(m,dt);
    if(g_pilot_real) printf("PILOT_REAL: %ld load cross-layer completati, %ld scartati (main gia' sul layer) | PILOT_K=%d\n",
        (long)atomic_load_explicit(&g_pilot_loads,memory_order_relaxed),
        (long)atomic_load_explicit(&g_pilot_drops,memory_order_relaxed), g_pilot_k);
    if(g_looka){
        const char *nm[3]={"previous token (=SPEC prefetch)","layer input, skip attention","next layer (one step ahead)"};
        printf("LOOKAHEAD routing — recall of true experts in predicted top-8:\n");
        for(int i=0;i<3;i++) printf("  %-38s %5.1f%%  (%lld/%lld)\n", nm[i],
            la_tot[i]?100.0*la_hit[i]/la_tot[i]:0.0, (long long)la_hit[i], (long long)la_tot[i]);
    }
    free(pids); free(all);
    usage_save(m);
}

/* modalita' SERVE (per la CLI 'ili'): carica il modello UNA volta, poi CHAT conversazionale.
 * KV-cache PERSISTENTE tra i turni: la storia resta in cache, si fa il prefill solo dei
 * token NUOVI -> il modello RICORDA la conversazione e non ri-processa il passato (lossless,
 * piu' umano, piu' veloce). Template chat GLM con token speciali (CHAT_TEMPLATE=0 -> grezzo).
 * Protocollo: "\x01\x01" "READY" "\x01\x01\n" dopo il load; risposta in streaming; "\x01\x01" "END" "\x01\x01\n" a fine turno.
 * ":reset" (riga "\x02RESET") azzera la memoria. EOF -> esce. */
/* ---- RFC: RE-PIN A CALDO / LIVE RE-PIN (opt-in, REPIN=n, default OFF) ----
 * Upstream fa AUTOPIN allo START (dalla storia .fa_usage). Questo aggiunge un re-pin
 * TRA I TURNI: nel punto sicuro dopo la risposta scambia i pin peggiori con i non-pinnati
 * piu' caldi, cosi' l'hot-store insegue il carico VIVO senza un profilo a parte. Isteresi
 * 25% (+4) contro il ping-pong; max 4 scambi/passata (~20 MB di disco l'uno). Una heat
 * map separata decade a ogni passata: la storia persistente .fa_usage resta intatta.
 * EN: upstream AUTOPINs at START (from .fa_usage). This adds a between-turns re-pin: at
 * the safe point after the reply, swap the worst pins for the hottest unpinned, so the
 * hot-store tracks the LIVE workload without a separate profile. 25% (+4) hysteresis vs
 * ping-pong; max 4 swaps/pass (~20 MB disk each). A separate decaying heat map keeps
 * persistent .fa_usage intact while adapting to the current workload. */
static int g_repin=0;
static uint64_t g_last_repin=0;
typedef struct { long gain; int l, slot, eid; } RepinCand;
static int repin_pick(Model *m, RepinCand *out, int maxc){
    Cfg *c=&m->c; int nb=0;
    for(int l=0;l<c->n_layers;l++){
        if(!m->npin || m->npin[l]<1 || !m->eheat[l]) continue;
        ESlot *P=m->pin[l]; int ids[4096], zp, eu; long g;
        int np=m->npin[l]; if(np>4096) np=4096;
        for(int z=0;z<np;z++) ids[z]=P[z].eid;
        if(!tier_pick_swap(m->eheat[l],c->n_experts,ids,np,&zp,&eu,&g)) continue;
        if(nb<maxc){ out[nb].gain=g; out[nb].l=l; out[nb].slot=zp; out[nb].eid=eu; nb++; }
        else { int w=0; for(int b=1;b<maxc;b++) if(out[b].gain<out[w].gain) w=b;
               if(g>out[w].gain){ out[w].gain=g; out[w].l=l; out[w].slot=zp; out[w].eid=eu; } }
    }
    return nb;
}
static void repin_pass(Model *m){
    if(g_repin<=0) return;
    if(m->n_emit - g_last_repin < (uint64_t)g_repin) return;
    g_last_repin = m->n_emit;
    RepinCand cd[4]; int nb=repin_pick(m,cd,4);
    for(int b=0;b<nb;b++){
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        int old=s->eid;
        uint32_t old_heat=m->eheat[cd[b].l][old], new_heat=m->eheat[cd[b].l][cd[b].eid];
#ifdef ILI_CUDA
        int gpu=s->g.cuda_eligible;
        int64_t old_gpu=gpu ? (int64_t)ili_cuda_tensor_bytes(s->g.cuda)
                             +(int64_t)ili_cuda_tensor_bytes(s->u.cuda)
                             +(int64_t)ili_cuda_tensor_bytes(s->d.cuda) : 0;
#endif
        double t0=now_s();
        expert_load(m,cd[b].l,cd[b].eid,s,1);       /* disk -> RAM, same resident slot */
        const char *tier="RAM";
#ifdef ILI_CUDA
        if(gpu){                                  /* refresh the same VRAM slot now, not lazily */
            if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                int64_t now_gpu=(int64_t)ili_cuda_tensor_bytes(s->g.cuda)
                               +(int64_t)ili_cuda_tensor_bytes(s->u.cuda)
                               +(int64_t)ili_cuda_tensor_bytes(s->d.cuda);
                m->gpu_expert_bytes+=now_gpu-old_gpu; tier="VRAM";
            } else {
                qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                m->gpu_expert_count--; m->gpu_expert_bytes-=old_gpu;
                fprintf(stderr,"[REPIN] VRAM upload failed; slot downgraded to RAM\n");
            }
        }
#endif
        fprintf(stderr,"[REPIN] %s layer %d: evict %d (heat=%u) <- admit %d (heat=%u) in %.0f ms\n",
            tier,cd[b].l,old,old_heat,cd[b].eid,new_heat,(now_s()-t0)*1e3);
    }
    for(int l=0;l<m->c.n_layers;l++) if(m->eheat[l]) tier_decay(m->eheat[l],m->c.n_experts);
}
/* ---- KV SU DISCO: la conversazione si riapre CALDA (KVSAVE=0 disattiva) ----
 * Il re-prefill di una chat riaperta costa ore su questo disco; la KV compressa MLA
 * costa ~182 KB/token. File <SNAP>/.ili_kv append-only: header (magic + dimensioni +
 * nrec) e un record per posizione [tok i32][Lc+Rc dei 78 layer][Ic DSA]. A fine turno
 * si appendono SOLO le posizioni nuove e si riscrive nrec per ultimo: un crash a meta'
 * append lascia nrec vecchio = file coerente. La riga KV del layer MTP non si salva:
 * al resume kv_start=-1 e la finestra di draft riparte da sola. */
static int g_kvsave=1;
#define KV_MAGIC "COLIKV1\0"   /* legacy on-disk magic: kept so existing KV files resume */
static void kv_hdr(Model *m, int32_t *h, int nrec){
    Cfg *c=&m->c; int nic=0;
    for(int i=0;i<c->n_layers;i++) if(m->Ic && m->Ic[i]) nic++;
    h[0]=c->n_layers; h[1]=c->kv_lora; h[2]=c->qk_rope;
    h[3]=m->has_dsa?c->index_hd:0; h[4]=nic; h[5]=c->vocab; h[6]=nrec; h[7]=0;
}
static void kv_disk_truncate(Model *m, int nrec){
    if(!g_kvsave) return;
    KVState *k=m->kv;
    FILE *f=fopen(k->disk_path,"r+b");
    if(!f){ k->disk_nrec=0; return; }
    k->disk_nrec=nrec;
    int32_t nr=nrec; fseek(f,8+6*4,SEEK_SET); fwrite(&nr,4,1,f); fclose(f);
}
static void kv_disk_reset(Model *m){ kv_disk_truncate(m,0); }
static void kv_disk_append(Model *m, const int *hist, int len){
    KVState *k=m->kv;
    if(!g_kvsave || len<=k->disk_nrec) return;
    Cfg *c=&m->c;
    int ok=1;                                    /* 2026-07-21 bug pass: every write is checked.
                                                  * The crash-safety design here is "dati prima,
                                                  * contatore poi" -- but that only holds if a
                                                  * FAILED data write (disk full: records are
                                                  * ~180 KB/token) never lets the counter commit.
                                                  * On any fwrite/fflush failure, leave the OLD
                                                  * nrec (proven consistent) and warn once. */
    FILE *f=fopen(k->disk_path,"r+b");
    if(!f){ f=fopen(k->disk_path,"wb"); if(!f) return;
        int32_t h[8]; kv_hdr(m,h,0);
        if(fwrite(KV_MAGIC,1,8,f)!=8 || fwrite(h,4,8,f)!=8) ok=0; }
    int64_t rec = 4 + (int64_t)c->n_layers*(c->kv_lora+c->qk_rope)*4;
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i]) rec+=(int64_t)c->index_hd*4;
    fseek(f, 8+8*4 + (int64_t)k->disk_nrec*rec, SEEK_SET);
    for(int p=k->disk_nrec;p<len && ok;p++){
        int32_t tk=hist[p]; if(fwrite(&tk,4,1,f)!=1) ok=0;
        for(int i=0;i<c->n_layers && ok;i++){
            if(fwrite(m->Lc[i]+(int64_t)p*c->kv_lora, 4, c->kv_lora, f)!=(size_t)c->kv_lora ||
               fwrite(m->Rc[i]+(int64_t)p*c->qk_rope, 4, c->qk_rope, f)!=(size_t)c->qk_rope) ok=0;
        }
        if(m->has_dsa) for(int i=0;i<c->n_layers && ok;i++) if(m->Ic[i])
            if(fwrite(m->Ic[i]+(int64_t)p*c->index_hd, 4, c->index_hd, f)!=(size_t)c->index_hd) ok=0;
    }
    if(fflush(f)!=0) ok=0;                       /* dati prima, contatore poi */
    if(!ok){
        fprintf(stderr,"[KV] append to %s FAILED (disk full?): keeping the previous %d-record "
                       "count; the conversation stays resumable up to there\n",
                k->disk_path, k->disk_nrec);
        fclose(f); return;
    }
    int32_t nr=len; fseek(f,8+6*4,SEEK_SET);
    if(fwrite(&nr,4,1,f)!=1 || fflush(f)!=0){
        fprintf(stderr,"[KV] record-count update of %s FAILED: keeping the previous %d-record "
                       "count\n", k->disk_path, k->disk_nrec);
        fclose(f); return;
    }
    fclose(f);
    k->disk_nrec=len;
}
static int kv_disk_load(Model *m, int *hist, int maxctx){
    if(!g_kvsave) return 0;
    KVState *k=m->kv;
    Cfg *c=&m->c;
    FILE *f=fopen(k->disk_path,"rb"); if(!f) return 0;
    char mg[8]; int32_t h[8], w[8]; kv_hdr(m,w,0);
    if(fread(mg,1,8,f)!=8 || memcmp(mg,KV_MAGIC,8) || fread(h,4,8,f)!=8 ||
       h[0]!=w[0]||h[1]!=w[1]||h[2]!=w[2]||h[3]!=w[3]||h[4]!=w[4]||h[5]!=w[5]){
        fprintf(stderr,"[KV] ignoring .ili_kv from a different model or version\n"); fclose(f); return 0; }
    int nrec=h[6];
    if(nrec<1){ fclose(f); return 0; }
    if(nrec>=maxctx-8-g_draft){
        fprintf(stderr,"[KV] saved conversation (%d tokens) exceeds the context: starting over\n",nrec);
        fclose(f); return 0; }
    double t0=now_s();
    for(int p=0;p<nrec;p++){
        int32_t tk; if(fread(&tk,4,1,f)!=1){ nrec=p; break; } hist[p]=tk;
        for(int i=0;i<c->n_layers;i++){
            if(fread(m->Lc[i]+(int64_t)p*c->kv_lora, 4, c->kv_lora, f)!=(size_t)c->kv_lora ||
               fread(m->Rc[i]+(int64_t)p*c->qk_rope, 4, c->qk_rope, f)!=(size_t)c->qk_rope){ nrec=p; goto out; }
        }
        if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i])
            if(fread(m->Ic[i]+(int64_t)p*c->index_hd, 4, c->index_hd, f)!=(size_t)c->index_hd){ nrec=p; goto out; }
    }
out:
    fclose(f);
    if(nrec>0){
        if(m->has_mtp) m->kv_start[c->n_layers]=-1;    /* la finestra MTP riparte da sola */
        fprintf(stderr,"[KV] resumed conversation from disk: %d tokens in %.1fs (no re-prefill)\n",
            nrec, now_s()-t0);
    }
    k->disk_nrec=nrec;
    return nrec;
}

/* tail: 1 iff hist[len] holds an EMITTED-but-unforwarded token (spec_decode's NGEN-truncation
 * exit stores the last picked token at all[kv] without forwarding it, so `len` excludes it --
 * see the \x02MORE handler, 2026-07-21 bug pass). */
typedef struct { KVState kv; int *hist, len, first, tail; } ServeCtx;
static double kv_pool_bytes(Model *m, int max_ctx);
static int ctx_clamp_for_ram(Model *m, double ram_gb, int max_ctx);

static void serve_ctx_init(Model *m, ServeCtx *s, const char *snap, int slot, int maxctx){
    s->kv.kv_start=calloc(m->c.n_layers+1,sizeof(int));
    if(m->has_mtp) s->kv.kv_start[m->c.n_layers]=-1;
    kv_bind(m,&s->kv); kv_alloc(m,maxctx);
    s->hist=malloc(maxctx*sizeof(int)); s->first=1;
    if(slot==0) snprintf(s->kv.disk_path,sizeof(s->kv.disk_path),"%s/.ili_kv",snap);
    else snprintf(s->kv.disk_path,sizeof(s->kv.disk_path),"%s/.ili_kv.%d",snap,slot);
    { /* one-time migration: adopt a legacy .ili_kv file under the new name */
        FILE *ex=fopen(s->kv.disk_path,"rb");
        if(ex) fclose(ex);
        else {
            char legacy[sizeof(s->kv.disk_path)];
            if(slot==0) snprintf(legacy,sizeof legacy,"%s/.coli_kv",snap);
            else snprintf(legacy,sizeof legacy,"%s/.coli_kv.%d",snap,slot);
            FILE *lf=fopen(legacy,"rb");
            if(lf){ fclose(lf);
                if(rename(legacy,s->kv.disk_path)==0)
                    fprintf(stderr,"[KV] migrated %s -> %s\n",legacy,s->kv.disk_path);
            }
        }
    }
    s->len=kv_disk_load(m,s->hist,maxctx); if(s->len>0) s->first=0;
}

static void serve_ctx_free(Model *m, ServeCtx *s){
    KVState *k=&s->kv; int NR=m->c.n_layers+1;
    if(k->Lc) for(int i=0;i<NR;i++){ free(k->Lc[i]); free(k->Rc[i]); }
    if(k->Ic) for(int i=0;i<m->c.n_layers;i++) free(k->Ic[i]);
    free(k->Lc); free(k->Rc); free(k->Ic); free(k->kv_start); free(s->hist);
}

static void run_serve(Model *m, const char *snap){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos);
    grammar_setup(&T);                   /* metodo F: GRAMMAR=file.gbnf (#48) */
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):256;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    { /* clamp CTX against the O(context) per-chunk prefill transient (kvb_all in attention())
       * plus the per-slot KV pool -- see ctx_clamp_for_ram()'s comment for the exact formula.
       * Without this, raising CTX silently re-creates the mid-prefill OOM-kill that
       * ILI_PREFILL_CHUNK was added to fix. ILI_CTX_FORCE=1 skips the clamp. */
      const char *force=ili_env("CTX_FORCE");
      if(!(force&&atoi(force))){
          int c2=ctx_clamp_for_ram(m, getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0, maxctx);
          if(c2<maxctx){
              fprintf(stderr,"[RAM] CTX %d -> %d: KV pool + O(context) prefill transient would "
                             "exceed the RAM budget (ILI_CTX_FORCE=1 to override)\n",maxctx,c2);
              maxctx=c2;
          }
      }
    }
    int templ=getenv("CHAT_TEMPLATE")?atoi(getenv("CHAT_TEMPLATE")):1;
    g_kvsave = getenv("KVSAVE")?atoi(getenv("KVSAVE")):1;
    int nctx=getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
    if(nctx<1||nctx>16){ fprintf(stderr,"KV_SLOTS must be between 1 and 16\n"); exit(2); }
    KVState *initial=m->kv; free(initial->kv_start); free(initial);
    ServeCtx *ctx=calloc(nctx,sizeof(ServeCtx));
    for(int i=0;i<nctx;i++) serve_ctx_init(m,&ctx[i],snap,i,maxctx);
    int active=0; ServeCtx *sc=&ctx[0]; kv_bind(m,&sc->kv);
    fprintf(stderr,"[KV] context slots: %d x %d tokens, projected pool %.2f GB\n",
        nctx,maxctx,kv_pool_bytes(m,maxctx)/1e9);
    #define hist  (sc->hist)
    #define len   (sc->len)
    #define first (sc->first)
    char *line=NULL; size_t cap=0; ssize_t nr; char *buf=malloc(1<<16);
    printf("\x01\x01" "READY" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout);
    while((nr=getline(&line,&cap,stdin))>0){
        if(nr>0 && line[nr-1]=='\n') line[--nr]=0;
        if(!strcmp(line,"\x02RESET")){ len=0; first=1; sc->tail=0; if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
            kv_disk_reset(m);
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        if(!strcmp(line,"\x02MORE")){                /* continua la risposta troncata da NGEN */
            if(len<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
            uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
            float *logit;
            if(sc->tail){
                /* NGEN-truncated turn: hist[len] is the last STREAMED token, emitted but never
                 * forwarded (spec_decode's truncation exit). Forward it for real so the
                 * continuation starts AFTER it. The old code re-forwarded hist[len-1] and let
                 * spec_decode re-pick position len -- under TEMP>0 (serve default 0.7) that
                 * re-SAMPLES the already-displayed token, silently forking the model's history
                 * from the visible transcript at every :more boundary (2026-07-21 bug pass). */
                logit=step(m,hist+len,1,len); len+=1; sc->tail=0;
            } else {
                /* turn ended without an unforwarded tail (natural stop / draft-path truncation):
                 * re-forward the last position to re-derive its logits, as before. */
                logit=step(m,hist+len-1,1,len-1);
            }
            int cur=ngen; if(len+cur+g_draft+2>=maxctx) cur=maxctx-len-g_draft-2;
            EmitStream es={&T,m,now_s(),0,1};
            int prod=0, len0=len;
            if(cur>0){ prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len);
                       /* exactly one more emitted token than history positions advanced ==
                        * spec_decode left a fresh unforwarded tail at hist[len] */
                       sc->tail=(prod>0 && prod==len-len0+1); }
            else free(logit);
            double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
            double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
            printf("\n\x01\x01" "END" "\x01\x01\n");
            printf("STAT %d %.2f %.1f %.2f\n", prod, prod/tdt, (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb());
            fflush(stdout); kv_disk_append(m,hist,len); repin_pass(m); continue; }   /* RFC: re-pin a caldo tra i turni / live re-pin between turns */
        if(nr<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        /* API mode: an exact, length-prefixed prompt. Unlike the interactive
         * line protocol this accepts newlines. The tokenized prompt is matched
         * against hist so the common KV prefix survives stateless HTTP turns.
         * Per-request generation controls follow the byte count:
         *   \x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]\n<prompt>\n */
        char *raw=NULL, *input=line;
        int input_n=(int)nr, raw_mode=0, req_ngen=ngen, prompt_tokens=0;
        float base_temp=g_temp, base_nuc=g_nuc;
        if(!strncmp(line,"\x02PROMPT ",8)){
            unsigned long long nb=0; double rt=0, rp=0; int slot=0;
            int nf=sscanf(line+8,"%llu %d %lf %lf %d",&nb,&req_ngen,&rt,&rp,&slot);
            if(nf<4 || nb>(16u<<20) || req_ngen<1 || rt<0 || rt>2 || rp<=0 || rp>1 ||
               slot<0 || slot>=nctx){
                /* Header rejected. A parsed byte count is a promise: the client already
                 * wrote that payload into the pipe. Drain it (plus the delimiter) or the
                 * payload gets replayed as interactive lines and every later frame reads
                 * someone else's bytes — the server then waits forever for its reply. */
                if(nf>=1 && nb>0){
                    char sink[1<<16]; unsigned long long left=nb;
                    while(left){
                        size_t want = left>sizeof(sink) ? sizeof(sink) : (size_t)left;
                        size_t got=fread(sink,1,want,stdin); if(!got) break; left-=got;
                    }
                    int delim=fgetc(stdin); if(delim!='\n' && delim!=EOF) ungetc(delim,stdin);
                }
                printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;
            }
            active=slot; sc=&ctx[active]; kv_bind(m,&sc->kv);
            raw=malloc((size_t)nb+1); if(!raw){fprintf(stderr,"OOM raw prompt\n");exit(1);}
            if(fread(raw,1,(size_t)nb,stdin)!=(size_t)nb){free(raw);break;}
            int delim=fgetc(stdin); if(delim!='\n' && delim!=EOF) ungetc(delim,stdin);
            if(memchr(raw,0,(size_t)nb)){free(raw); printf("\x01\x01" "END" "\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;}
            raw[nb]=0; input=raw; input_n=(int)nb; raw_mode=1;
            if(req_ngen>ngen) req_ngen=ngen;
            g_temp=(float)rt; g_nuc=(float)rp;
        } else { active=0; sc=&ctx[0]; kv_bind(m,&sc->kv); }
        int bl=0, k=0;                           /* costruisce/tokenizza il turno */
        /* template UFFICIALE GLM-5.2 (chat_template.jinja): niente \n dopo i ruoli, e dopo
         * <|assistant|> serve SEMPRE il blocco think — <think></think> lo DISATTIVA (nothink):
         * col template sbagliato il modello farfuglia e non emette mai lo stop. THINK=1 lo abilita. */
        const char *tk = getenv("THINK")&&atoi(getenv("THINK"))? "<think>" : "<think></think>";
        if(raw_mode){
            int *tmp=malloc(maxctx*sizeof(int)); if(!tmp){fprintf(stderr,"OOM raw tokens\n");exit(1);}
            prompt_tokens=tok_encode(&T,input,input_n,tmp,maxctx-8-g_draft);
            int old_len=len, prefix=0;
            while(prefix<old_len && prefix<prompt_tokens && hist[prefix]==tmp[prefix]) prefix++;
            if(prefix<old_len){
                len=prefix;
                if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
                kv_disk_truncate(m,len);           /* il prossimo append sovrascrive solo la coda */
            }
            k=prompt_tokens-len;
            if(k>0) memcpy(hist+len,tmp+len,k*sizeof(int));
            fprintf(stderr,"[API] KV slot %d prefix %d/%d token, prefill %d\n",
                active,len,prompt_tokens,k);
            free(tmp);
        } else {
            if(templ){ if(first) bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop>");
                       bl+=snprintf(buf+bl,(1<<16)-bl,"<|user|>%s<|assistant|>%s",input,tk); }
            else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",input);
            /* snprintf returns the NEEDED length, not what fit: a pasted line >64 KB pushes
             * bl past buf's 1<<16 bytes and tok_encode below would read bl bytes -- a heap
             * over-read. Clamp to the written content (2026-07-21 bug pass). */
            if(bl>(1<<16)-1) bl=(1<<16)-1;
            k=tok_encode(&T,buf,bl,hist+len,maxctx-len); prompt_tokens=k;
            if(len+k+8+g_draft>=maxctx){ len=0; first=1; kv_disk_reset(m);
                bl=0; if(templ){ bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop><|user|>%s<|assistant|>%s",input,tk); }
                else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",input);
                if(bl>(1<<16)-1) bl=(1<<16)-1;   /* same NEEDED-length clamp as above */
                k=tok_encode(&T,buf,bl,hist,maxctx); if(k>maxctx-8-g_draft) k=maxctx-8-g_draft;
                prompt_tokens=k;
            }
        }
        if(prompt_tokens<1){ free(raw); g_temp=base_temp; g_nuc=base_nuc;
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb()); fflush(stdout); continue; }
        first=0;
        int cur=req_ngen; if(len+k+cur+g_draft+2>=maxctx) cur=maxctx-len-k-g_draft-2;
        uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
        float *logit;
        /* Per-turn PROFILE (gate 0, the prefill-I/O study): reset the cumulative
         * counters at turn start and report where the prefill wall time went. On
         * stderr: in serve mode stdout is the protocol stream to the API server. */
        m->t_edisk=m->t_emm=m->t_attn=m->t_kvb=m->t_head=0;
    m->t_stall_exposed=0; m->io_bytes_requested=m->io_bytes_read=0;
    m->io_reads_attempted=m->io_reads_completed=m->n_pipe_waits=m->n_pipe_waits_blocked=0;
    m->io_bytes_weight=m->io_bytes_scale=0; m->io_reads_weight=m->io_reads_scale=0;  /* [IOKIND] */
        if(k>0){
            /* Chunked prefill: one step() per <=pfc tokens instead of a single S=k
             * forward. The KV outcome is identical (causal attention appends to the
             * same cache and every row is computed independently), but the transient
             * activations drop from O(k) to O(pfc): a monolithic 40K-token forward
             * allocates ~8 GB of x/nrm/tmp/Q/ctx scratch on the 744B geometry — the
             * 2026-07-14 serve freeze between 5K and 40K tokens (RSS shoved over the
             * budget into swap on a disk that was also running a container build).
             * Per-chunk progress goes to stderr so a minutes-long prefill is visibly
             * alive. ILI_PREFILL_CHUNK=n tunes it; 0 restores the monolithic call.
             * (With MTP drafts on, a chunk boundary skips one absorb pair; drafts are
             * a warmup heuristic, not output, and serve runs DRAFT=0.) */
            int pfc = ili_env("PREFILL_CHUNK") ? atoi(ili_env("PREFILL_CHUNK")) : 2048;
            if(pfc<=0 || pfc>k) pfc=k;
            int done=0;
            for(;;){
                int cs = k-done<pfc ? k-done : pfc;
                if(done) free(logit);
                logit=step(m,hist+len,cs,len); len+=cs; done+=cs;
                if(done>=k) break;
                fprintf(stderr,"[API] prefill %d/%d tok\n",done,k);
            }
            double tpf=now_s()-tt0, acc=m->t_edisk+m->t_emm+m->t_attn+m->t_head;
            fprintf(stderr,"[API] PROFILE prefill %d tok in %.1fs: expert-disk %.3fs | expert-matmul %.3fs | "
                "attention %.3fs (kvb %.3fs) | lm_head %.3fs | other %.3fs\n",
                k,tpf,m->t_edisk,m->t_emm,m->t_attn,m->t_kvb,m->t_head,tpf-acc);
        }
        else logit=step(m,hist+len-1,1,len-1);   /* prompt identico/prefisso: rigenera i logits */
        EmitStream es={&T,m,now_s(),0,1};
        int prod=0, len0=len;
        grammar_reset();                         /* nuova risposta = nuovo documento (MORE invece continua) */
        if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len);
        else free(logit);
        /* tail bookkeeping for \x02MORE: emitted count exceeding the history-position advance
         * by one means spec_decode's truncation exit left hist[len] emitted-but-unforwarded */
        sc->tail=(cur>0 && prod>0 && prod==len-len0+1);
        double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
        double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
        printf("%s\x01\x01" "END" "\x01\x01\n",raw_mode?"":"\n");
        printf("STAT %d %.2f %.1f %.2f %d %d\n", prod, prod/tdt,
            (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb(), prompt_tokens, prod>=cur);
        fflush(stdout);
        free(raw); g_temp=base_temp; g_nuc=base_nuc;
        usage_save(m);                   /* la cache che impara: storia aggiornata a ogni turno */
        kv_disk_append(m,hist,len);      /* KV su disco: il prossimo avvio riparte da qui */
    }
    free(line); free(buf);
    usage_save(m);
    #undef hist
    #undef len
    #undef first
    for(int i=0;i<nctx;i++) serve_ctx_free(m,&ctx[i]);
    free(ctx); m->kv=NULL; m->Lc=m->Rc=m->Ic=NULL; m->kv_start=NULL; m->max_t=0;
}

static int *read_arr(jval*o,const char*k,int*n){ jval*a=json_get(o,k); int*r=malloc(a->len*sizeof(int));
    for(int i=0;i<a->len;i++) r[i]=(int)a->kids[i]->num; *n=a->len; return r; }

/* byte residenti di un tensore [O,I] al numero di bit dato (specchio di qt_bytes) */
static int64_t tbytes(int O,int I,int bits){
    if(bits>=16) return (int64_t)O*I*4;
    if(bits>=5)  return (int64_t)O*I + (int64_t)O*4;
    return (int64_t)O*((I+1)/2) + (int64_t)O*4;
}
/* byte VERI di un expert: dal container se pre-quantizzato, altrimenti stima da ebits.
 *
 * ROOT CAUSE (2026-07-20 robustness pass, found by an external second-opinion review while
 * chasing the mode15_e2e_certify.sh "silent death on a later turn" report): for a
 * mode-1.5 (MH01) gate/up/down tensor, st_nbytes() below returns the COMPRESSED
 * on-disk size, but expert_load_mode15() always decodes it to the FULL int4-packed
 * size before it is ever resident (the Mode-15 integration design notes's whole point: the
 * decoded bytes are byte-identical to a legacy int4 tensor's layout). Every caller of
 * this probe (expert_avail(), cap_for_ram()'s per-layer LRU sizing AND its AUTO-RAISE)
 * assumes the return value IS the real resident-per-expert footprint -- so on a
 * mode-1.5 container this under-counts by ~1/ratio (measured ~1.33x at this
 * container's ~0.75 compression ratio), and cap_for_ram() auto-raises the per-layer
 * cache to an expert COUNT the RAM budget cannot actually hold once experts are
 * genuinely resident in decoded form. The cache starts EMPTY and fills gradually
 * across MANY turns of one persistent process (:reset deliberately keeps it warm --
 * see run_serve()'s own kv_disk_reset() vs. the cache arrays it never touches), so
 * this never shows up on an isolated first turn or a single `ili run` (too few
 * distinct experts touched yet) -- only once enough turns' worth of distinct experts
 * accumulate does real RSS cross the actual physical ceiling, at which point the
 * kernel's own OOM killer ends the process with SIGKILL: uncatchable by any
 * userspace handler (see ili_install_fatal_handlers() above) and easy to miss as
 * "no OOM trace" unless the unified log is checked specifically for a memorystatus/
 * jetsam kill, not just dmesg/crash-report locations. Fixed by peeking each weight
 * tensor's own magic (mode15_blob(), already used identically by expert_load() and
 * qt_from_disk() above) and substituting the DECODED size (O*ceil(I/2), the exact
 * dec_sz[k] formula expert_load_mode15() itself uses) for a detected MH01 blob,
 * instead of trusting its on-disk byte count. */
static int64_t expert_bytes_probe(Model *m, int ebits){
    Cfg *c=&m->c; int64_t eb=0; char nm[256];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.gate_proj.weight",c->first_dense);
    if(st_nbytes(&m->S,nm)>0){
        const char *suf[3]={"gate_proj","up_proj","down_proj"};
        /* [O,I] per tensor -- same convention as expert_load_mode15's own OO[]/II[]:
         * gate/up are [moe_inter,hidden], down is [hidden,moe_inter]. */
        int OO[3]={c->moe_inter,c->moe_inter,c->hidden}, II[3]={c->hidden,c->hidden,c->moe_inter};
        for(int k=0;k<3;k++){
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight",c->first_dense,suf[k]);
            st_tensor *t=st_find(&m->S,nm);
            int64_t wbytes=t?t->nbytes:0;
            if(t && wbytes>=4){
                uint8_t peek[4]={0};
                ssize_t got=pread(t->fd,peek,sizeof(peek),t->off);
                if(mode15_blob(peek,got)) wbytes=(int64_t)OO[k]*((II[k]+1)/2);
            }
            eb+=wbytes;
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight.qs",c->first_dense,suf[k]);
            int64_t q=st_nbytes(&m->S,nm); if(q>0) eb+=q;
        }
    }
    if(eb<=0) eb = tbytes(c->moe_inter,c->hidden,ebits)*2 + tbytes(c->hidden,c->moe_inter,ebits);
    return eb;
}

/* scarica su file l'istogramma d'uso degli expert: righe "layer eid count" (per PIN).
 * Include la riga MTP (layer n_layers). Scrittura atomica (tmp+rename): viene chiamata
 * anche a ogni turno di serve e il processo puo' morire in qualsiasi momento. */
static void stats_dump_q(Model *m, const char *path, int quiet){
    char tmp[2100]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE *f=fopen(tmp,"w"); if(!f){ if(!quiet) perror(tmp); return; }
    /* Write recognized header — usage_load and pin_load skip these lines.
     * The decay_progress field persists the selection-count window state. */
    fprintf(f, "# fa_hotset_v1\n");
    if(g_hotset_profile[0]) fprintf(f, "# profile: %s\n", g_hotset_profile);
    if(g_hotset_decay_factor > 0) fprintf(f, "# decay_factor: %d\n", g_hotset_decay_factor);
    if(g_hotset_decay_interval > 0) fprintf(f, "# decay_interval: %lld\n", (long long)g_hotset_decay_interval);
    fprintf(f, "# decay_progress: %lld\n", (long long)g_hotset_selections);
    Cfg *c=&m->c; int64_t tot=0, nz=0;
    for(int i=0;i<=c->n_layers;i++){ if(!m->eusage[i]) continue;
        for(int e=0;e<c->n_experts;e++) if(m->eusage[i][e]){ fprintf(f,"%d %d %u\n",i,e,m->eusage[i][e]); tot+=m->eusage[i][e]; nz++; } }
    fclose(f); rename(tmp,path);
    if(!quiet) fprintf(stderr,"[STATS] %lld selections across %lld distinct experts -> %s\n",(long long)tot,(long long)nz,path);
}
static void stats_dump(Model *m, const char *path){ stats_dump_q(m,path,0); }

/* CACHE CHE IMPARA: istogramma d'uso PERSISTENTE in <SNAP>/.fa_usage.
 * Caricato all'avvio (i contatori ripartono dalla storia), salvato a ogni turno:
 * piu' usi iliria, meglio l'auto-pin conosce i TUOI expert caldi. */
static char g_usage_path[2100]="";
static int64_t usage_load(Model *m, const char *path){
    FILE *f=fopen(path,"r"); if(!f) return 0;
    Cfg *c=&m->c; int l,e; uint32_t cnt; int64_t tot=0;
    char line[2100];
    while(fgets(line, sizeof(line), f)){
        if(hotset_is_header(line)) continue;  /* skip header comments */
        if(sscanf(line, "%d %d %u", &l, &e, &cnt) == 3)
            if(l>=0&&l<=c->n_layers&&e>=0&&e<c->n_experts&&m->eusage[l]){ m->eusage[l][e]+=cnt; tot+=cnt; }
    }
    fclose(f); return tot;
}
static void usage_save(Model *m){
    /* Apply decay if the selection-count window has been reached.
     * Decay only eusage (the persistent counters), not eheat (session-only). */
    if(g_hotset_decay_factor > 0 && g_hotset_decay_interval > 0
       && g_hotset_selections >= g_hotset_decay_interval){
        Cfg *cc = &m->c;
        for(int i=0; i<=cc->n_layers; i++){
            if(!m->eusage[i]) continue;
            hotset_decay_fixed(m->eusage[i], cc->n_experts, g_hotset_decay_factor);
        }
        g_hotset_selections = 0;  /* reset counter after decay */
    }
    if(g_usage_path[0]) stats_dump_q(m,g_usage_path,1);
}

/* HOT-STORE ("iliria cache"): carica in RAM, UNA VOLTA e per sempre, i top expert
 * per frequenza d'uso misurata (file STATS di un run precedente), entro un budget in GB.
 * Ogni hit evita una lettura dal disco lento. */
/* MLOCK: inchioda in RAM fisica gli expert pinnati cosi' il compressore di memoria di
 * macOS non li comprime/evacua (visto: RSS reale < residente previsto -> "hit" lenti).
 * -1 = auto (ON su macOS dove serve e RLIMIT_MEMLOCK e' permissivo; OFF altrove, dove
 * il limite e' spesso minuscolo e va alzato a mano), 0 = off, 1 = force.
 * EN: MLOCK: wire pinned experts into physical RAM so macOS's memory compressor can't
 * compress/evict them (we saw actual RSS < intended resident -> slow "hits"). -1 = auto
 * (ON on macOS where it matters and RLIMIT_MEMLOCK is permissive; OFF elsewhere, where the
 * limit is often tiny and must be raised by hand), 0 = off, 1 = force. */
static int g_mlock=-1;
static int mem_should_wire(void){
    if(g_mlock>=0) return g_mlock;
#if defined(__APPLE__)
    return 1;                                     /* macOS: default ON */
#else
    return 0;                                     /* Linux/altri: opt-in via MLOCK=1 / opt-in */
#endif
}
/* Inchioda [addr,addr+len) in RAM fisica. No-op fuori da POSIX (Windows ecc.).
 * EN: wire [addr,addr+len) into physical RAM. No-op off POSIX (Windows, etc.). */
static int mem_wire(void *addr, size_t len){
#if defined(__APPLE__) || defined(__linux__)
    return mlock(addr, len);
#else
    (void)addr; (void)len; return 0;
#endif
}
/* Inchioda tutti gli slab degli expert pinnati (pesi + scale). Non fatale se fallisce.
 * EN: wire all pinned-expert slabs (weights + scales). Non-fatal on failure. */
static void pin_wire(Model *m){
    if(!mem_should_wire()) return;
    Cfg *c=&m->c; double t0=now_s(); int64_t wired=0; long failed=0;
    for(int i=0;i<c->n_layers;i++) for(int z=0;z<m->npin[i];z++){
        ESlot *s=&m->pin[i][z];
        if(s->slab){  if(mem_wire(s->slab, s->slab_cap)==0) wired+=s->slab_cap; else failed++; }
        if(s->fslab){ size_t fl=(size_t)s->fslab_cap*sizeof(float);
                      if(mem_wire(s->fslab, fl)==0) wired+=fl; else failed++; }
    }
    if(failed)
        fprintf(stderr,"[PIN] mlock: %.1f GB wired, %ld allocations failed "
            "(raise the limit: ulimit -l unlimited) in %.0fs\n", wired/1e9, failed, now_s()-t0);
    else
        fprintf(stderr,"[PIN] mlock: %.1f GB wired in physical RAM "
            "(no compression) in %.0fs\n", wired/1e9, now_s()-t0);
}

static void pin_load(Model *m, const char *statspath, double gb){
    FILE *f=fopen(statspath,"r"); if(!f){ perror(statspath); return; }
    typedef struct { int l,e; uint32_t c; } Rec;
    Cfg *c=&m->c; int cap=(c->n_layers+1)*c->n_experts;
    Rec *r=malloc((size_t)cap*sizeof(Rec)); int n=0;
    int l,e; uint32_t cnt;
    char line[2100];
    while(n<cap && fgets(line, sizeof(line), f)){
        if(hotset_is_header(line)) continue;  /* skip header comments */
        if(sscanf(line, "%d %d %u", &l, &e, &cnt) == 3){
            int ok = l>=0 && e>=0 && e<c->n_experts &&
                     ((l<c->n_layers && m->L[l].sparse) || (l==c->n_layers && m->has_mtp));
            if(ok) r[n++]=(Rec){l,e,cnt};
        }
    }
    fclose(f);
    for(int a=0;a<n;a++){ int best=a;                       /* selection sort parziale, poi taglio */
        for(int b=a+1;b<n;b++) if(r[b].c>r[best].c) best=b;
        Rec t=r[a]; r[a]=r[best]; r[best]=t;
        if(a>4095) break;                                    /* bastano i top ~4k */
    }
    int64_t eb=expert_bytes_probe(m,m->ebits);
    int npin=(int)(gb*1e9/eb); if(npin>n) npin=n; if(npin>4096) npin=4096;
    if(npin<1){ free(r); return; }
    int *cnt_l=calloc(c->n_layers+1,sizeof(int));   /* +1: riga MTP */
    for(int a=0;a<npin;a++) cnt_l[r[a].l]++;
    for(int i=0;i<=c->n_layers;i++) if(cnt_l[i]) m->pin[i]=calloc(cnt_l[i],sizeof(ESlot));
    double t0=now_s();
    #pragma omp parallel for schedule(dynamic,1)
    for(int a=0;a<npin;a++){
        int li=r[a].l, slot;
        #pragma omp critical
        slot=m->npin[li]++;
        expert_load(m,li,r[a].e,&m->pin[li][slot],1);
    }
    m->resident_bytes += (int64_t)npin*eb;
    fprintf(stderr,"[PIN] hot store: %d experts in RAM (%.1f GB) loaded in %.0fs from %s\n",
        npin, npin*eb/1e9, now_s()-t0, statspath);
#ifdef ILI_CUDA
    if(g_cuda_enabled && g_cuda_expert_gb>0){
        double remaining[ILI_CUDA_MAX_DEVICES]={0}, placed_b[ILI_CUDA_MAX_DEVICES]={0};
        int placed_n[ILI_CUDA_MAX_DEVICES]={0};
        double budget=g_cuda_expert_gb*1e9, safe_total=0;
        for(int i=0;i<g_cuda_ndev;i++){
            size_t free_b=0,total_b=0;
            if(ili_cuda_mem_info(g_cuda_devices[i],&free_b,&total_b)){
                /* Dense tensors are assigned round-robin and upload lazily.
                 * Reserve their projected footprint plus 2 GB per device. */
                remaining[i]=(double)free_b-(double)g_cuda_dense_projected[i]-2e9;
                if(remaining[i]<0) remaining[i]=0;
                safe_total+=remaining[i];
            }
        }
        if(budget>safe_total) budget=safe_total;
        for(int a=0;a<npin && m->gpu_expert_bytes<budget;a++){
            int li=r[a].l;
            for(int z=0;z<m->npin[li];z++) if(m->pin[li][z].eid==r[a].e){
                ESlot *s=&m->pin[li][z];
                int64_t need=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
                if(m->gpu_expert_bytes+need>budget) break;
                int tried[ILI_CUDA_MAX_DEVICES]={0}, placed=0;
                for(int attempt=0;attempt<g_cuda_ndev && !placed;attempt++){
                    int best=-1;
                    for(int i=0;i<g_cuda_ndev;i++) if(!tried[i] && remaining[i]>=need &&
                        (best<0||placed_b[i]<placed_b[best])) best=i;
                    if(best<0) break;
                    tried[best]=1;
                    s->g.cuda_device=s->u.cuda_device=s->d.cuda_device=g_cuda_devices[best];
                    s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=1;
                    if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                        int64_t actual=(int64_t)ili_cuda_tensor_bytes(s->g.cuda)
                                      +(int64_t)ili_cuda_tensor_bytes(s->u.cuda)
                                      +(int64_t)ili_cuda_tensor_bytes(s->d.cuda);
                        m->gpu_expert_count++; m->gpu_expert_bytes+=actual;
                        remaining[best]-=actual; placed_b[best]+=actual; placed_n[best]++;
                        placed=1;
                    } else {
                        qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                        s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                        remaining[best]=0;             /* device rejected its projected capacity */
                    }
                }
                break;
            }
        }
        fprintf(stderr,"[CUDA] hot expert tier: %d/%d experts, VRAM %.2f GB (total budget %.1f GB)\n",
            m->gpu_expert_count,npin,m->gpu_expert_bytes/1e9,g_cuda_expert_gb);
        for(int i=0;i<g_cuda_ndev;i++) fprintf(stderr,"[CUDA]   device %d: %d experts, %.2f GB\n",
            g_cuda_devices[i],placed_n[i],placed_b[i]/1e9);
    }
#endif
    pin_wire(m);                                   /* inchioda in RAM (no compressione) / wire in RAM (no compression) */
    free(r); free(cnt_l);
}

static double g_mem_avail_boot=0;   /* MemAvailable all'avvio, prima di caricare il modello */
/* RAM disponibile ADESSO (GB): e' il tetto vero, non il totale. Linux: MemAvailable
 * da /proc/meminfo. macOS: pagine free+inactive+purgeable da host_statistics64
 * (stessa semantica: recuperabili senza swap). Senza questo ramo il fallback
 * "assumo 8 GB" castrava la cache expert proprio sulle macchine con piu' RAM. */
static double mem_available_gb(void){
#ifdef __APPLE__
    mach_msg_type_number_t cnt=HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm;
    if(host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&cnt)!=KERN_SUCCESS) return 0;
    return ((double)vm.free_count+(double)vm.inactive_count+(double)vm.purgeable_count)
           * (double)sysconf(_SC_PAGESIZE) / 1e9;
#elif defined(_WIN32)
    double total, avail;
    compat_meminfo(&total, &avail);
    return avail;
#else
    FILE *f=fopen("/proc/meminfo","r"); if(!f) return 0;
    char ln[256]; double kb=0;
    while(fgets(ln,sizeof(ln),f)) if(sscanf(ln,"MemAvailable: %lf",&kb)==1) break;
    fclose(f); return kb/1e6;
#endif
}

static int kv_slot_count(void){
    if(!getenv("SERVE")) return 1;
    return getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
}

static double kv_pool_bytes(Model *m, int max_ctx){
    Cfg *c=&m->c; double one=(double)(c->n_layers+1)*max_ctx*(c->kv_lora+c->qk_rope)*4.0;
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        one+=(double)max_ctx*c->index_hd*4.0;
    int slots=kv_slot_count(); if(slots<1||slots>16) slots=1;
    return one*slots;
}

/* byte disponibili per gli expert (pin + LRU) nel budget — specchio del conto di cap_for_ram */
static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int64_t eb=expert_bytes_probe(m,ebits);
    if(ram_gb<=0){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    double slack = 1.2e9 + 2.5e9 + 64.0*(double)eb
        + kv_pool_bytes(m,max_ctx)
        + (double)max_ctx*c->n_heads*(c->qk_nope+c->v_head)*4.0;
    return ram_gb*1e9 - (double)m->resident_bytes - slack;
}

/* Largest CTX the RAM budget can actually sustain (2026-07-21 bug pass). The binding term is
 * attention()'s S>4 prefill reconstruction: kvb_all = CTX*n_heads*(qk_nope+v_head)*4 transient
 * bytes PER CHUNK (Tk = total context so far, NOT the PREFILL_CHUNK size) -- ~112 KB/token at
 * GLM-5.2 dims (64*448*4) -- plus the per-slot KV pool (kv_pool_bytes: all KV_SLOTS), both
 * linear in CTX. Everything else (resident weights, ws[64] slabs, activations+page-cache
 * reserve) is CTX-independent and mirrored from cap_for_ram's own slack accounting. So:
 *   fit = (budget - resident - 1.2e9 - 2.5e9 - 64*expert_bytes) /
 *         (kv_pool_bytes(m,1) + n_heads*(qk_nope+v_head)*4)          [bytes/token]
 * Floor of 1024 keeps the tool usable on tiny machines (run_serve still warns loudly);
 * ILI_CTX_FORCE=1 at the call site skips the clamp entirely. Deliberately NOT a tiling
 * rewrite of the prefill path -- this only stops CTX from silently overcommitting. */
static int ctx_clamp_for_ram(Model *m, double ram_gb, int max_ctx){
    Cfg *c=&m->c;
    if(max_ctx<1) return max_ctx;
    if(ram_gb<=0){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    int64_t eb=expert_bytes_probe(m,m->ebits);
    double fixed = 1.2e9 + 2.5e9 + 64.0*(double)eb + (double)m->resident_bytes;
    double per_tok = kv_pool_bytes(m,1)                                   /* KV pool, all slots */
                   + (double)c->n_heads*(c->qk_nope+c->v_head)*4.0;       /* kvb_all transient  */
    double left = ram_gb*1e9 - fixed;
    int fit = (left>0 && per_tok>0) ? (int)(left/per_tok) : 0;
    if(fit<1024) fit=1024;
    return max_ctx<fit ? max_ctx : fit;
}

/* clampa la cache expert a un budget RAM (GB): cap t.c. residente + cache + slack <= budget.
 * ram_gb<=0 -> budget AUTO = 88% della RAM disponibile adesso (lascia respiro a OS+wrapper:
 * sforare = OOM-kill del kernel a meta' generazione, molto peggio di una cache piu' piccola). */
static void cap_for_ram(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    if(m->has_mtp) nsp+=2;                       /* riga cache MTP: conta ~doppia (expert int8 = 2x int4) */
    int64_t eb=expert_bytes_probe(m,ebits);
    int auto_b = ram_gb<=0;
    if(auto_b){ ram_gb = g_mem_avail_boot*0.88;   /* misurata PRIMA del load: il residente gia'
                                                   * allocato viene sottratto sotto, non due volte */
        if(ram_gb<4){ fprintf(stderr,"[RAM] MemAvailable is unreadable or too low; assuming 8 GB\n"); ram_gb=8; } }
    /* slack ONESTO, non forfettario (l'OOM del 2026-07-04 veniva da qui):
     *  ws[64] slab del working-set (si materializzano TUTTI nel prefill batch-union),
     *  KV cache a max_ctx, kvb_all della ricostruzione k/v in attention,
     *  attivazioni+logits+overhead ~1.2 GB */
    double ws_b  = 64.0*(double)eb;
    double kv_b  = kv_pool_bytes(m,max_ctx);
    double kvb_b = (double)max_ctx*c->n_heads*(c->qk_nope+c->v_head)*4.0;
    /* RISERVA PAGE-CACHE (misurato 2026-07-06): strangolarla fa crollare le pread
     * buffered da ~800 a ~180 MB/s — gli ultimi GB di LRU rendono MENO di quanto
     * costino in banda disco persa. 2.5 GB restano SEMPRE al kernel. */
    double pc_b  = 2.5e9;
    double slack = 1.2e9 + pc_b + ws_b + kv_b + kvb_b;
    double avail = ram_gb*1e9 - (double)m->resident_bytes - slack;
    int capmax = (avail>0 && nsp>0) ? (int)(avail/((double)nsp*eb)) : 0;
    if(capmax<1) capmax=1;
    if(capmax < m->ecap){
        fprintf(stderr,"[RAM_GB=%.1f%s] resident %.1f GB + reserve %.1f GB (ws %.1f, KV %dx%d %.1f, kvb %.1f), "
            "experts %.1f MB x %d layers -> cap lowered %d->%d (projected peak %.1f GB)\n",
            ram_gb,auto_b?" auto":"",m->resident_bytes/1e9,slack/1e9,ws_b/1e9,
            kv_slot_count(),max_ctx,kv_b/1e9,kvb_b/1e9,
            eb/1e6, nsp, m->ecap, capmax,
            (m->resident_bytes + (double)capmax*nsp*eb + slack)/1e9);
        m->ecap=capmax;
    } else {
        /* AUTO-RAISE (issue #12): il budget consente PIU' cache di quella chiesta.
         * Senza questo, una macchina da 128 GB girava con la LRU di una da 16
         * (cap=8 di default in ili): hit 23-28% con decine di GB inutilizzati.
         * Tetto a n_experts: oltre, ogni layer avrebbe slot che non puo' riempire.
         * CAP_RAISE=0 ripristina il comportamento fisso. */
        int raise_on = getenv("CAP_RAISE")?atoi(getenv("CAP_RAISE")):1;
        int newcap = capmax>c->n_experts ? c->n_experts : capmax;
        if(raise_on && newcap>m->ecap){
            int grew=1;
            for(int i=0;i<=c->n_layers && grew;i++) if(m->ecache[i]){
                /* never `p=realloc(p,...)`: on failure that clobbers the pointer (leak) and a
                 * later moe() resolve would deref NULL with ecn[i]>0 (2026-07-21 bug pass).
                 * On failure keep the old array+cap -- layers already grown just carry unused
                 * extra capacity, harmless because m->ecap stays the old minimum. */
                ESlot *ns=realloc(m->ecache[i],(size_t)newcap*sizeof(ESlot));
                if(!ns){ grew=0; break; }
                memset(ns+m->ecap,0,(size_t)(newcap-m->ecap)*sizeof(ESlot));
                m->ecache[i]=ns;
            }
            if(grew){
                fprintf(stderr,"[RAM_GB=%.1f%s] cap raised %d->%d: budget allows it "
                    "(projected peak %.1f GB; set CAP_RAISE=0 to disable)\n",
                    ram_gb, auto_b?" auto":"", m->ecap, newcap,
                    (m->resident_bytes + (double)newcap*nsp*eb + slack)/1e9);
                m->ecap=newcap;
            } else
                fprintf(stderr,"[RAM_GB=%.1f%s] cap raise %d->%d aborted: realloc failed; "
                    "keeping cap=%d\n", ram_gb, auto_b?" auto":"", m->ecap, newcap, m->ecap);
        } else
            fprintf(stderr,"[RAM_GB=%.1f%s] cap=%d ok (projected peak %.1f GB)\n", ram_gb, auto_b?" auto":"", m->ecap,
                (m->resident_bytes + (double)m->ecap*nsp*eb + slack)/1e9);
    }
}

/* Safeguard (deliverable 3b, the factorial-streaming causality spec): one
 * machine-parseable "EFFECTIVE-FLAGS:"-prefixed line PER flag, printed once at startup
 * after every env-alias resolution AND master-switch gating has already run (i.e. this is
 * resolved-config ground truth, not an echo of what was requested) -- REQUESTED env != the
 * EFFECTIVE value the engine actually acted on, and this is the one place both stay in
 * sync automatically, by construction, because it just reads the variables the rest of
 * main() already resolved (g_metal_enabled, g_metal_prefill, m->has_dsa, g_io_delay_us,
 * g_pipe), it does not recompute them from env a second time.
 * c/scripts/provenance.sh and c/tools/eval_glm.py capture these lines into their run
 * manifests (grep "^EFFECTIVE-FLAGS:"). Call AFTER model_init so has_dsa reflects whether
 * the model actually shipped indexer weights, not just the request. */
static void print_effective_flags(Model *m){
    const char *metal_req = ili_env("METAL");
    const char *prefill_req = ili_env("METAL_PREFILL");
    const char *dsa_req = ili_env("DSA");
    const char *io_delay_req = ili_env("IO_DELAY_US");
    const char *io_delay_dec_req = ili_env("IO_DELAY_DECODE_US");
    const char *pipe_req = ili_env("PIPE");
    int metal_eff = 0, prefill_eff = 0;
    const char *metal_reason = "";
#ifdef ILI_METAL
    metal_eff = g_metal_enabled;
    prefill_eff = g_metal_enabled && g_metal_prefill;
#else
    metal_reason = " reason=not_compiled";
#endif
    const char *prefill_reason =
        (prefill_req && atoi(prefill_req) && !prefill_eff) ?
            (metal_eff ? " reason=unknown" : " reason=master_disabled") : "";
    const char *dsa_reason =
        (dsa_req && atoi(dsa_req)==0 && !m->has_dsa) ? " reason=disabled_by_env" :
        (!m->has_dsa ? " reason=model_lacks_dsa_weights" : "");
    printf("EFFECTIVE-FLAGS: METAL requested=%s effective=%d%s\n",
        metal_req?metal_req:"unset", metal_eff, metal_reason);
    printf("EFFECTIVE-FLAGS: METAL_PREFILL requested=%s effective=%d%s\n",
        prefill_req?prefill_req:"unset", prefill_eff, prefill_reason);
    printf("EFFECTIVE-FLAGS: DSA requested=%s effective=%d%s\n",
        dsa_req?dsa_req:"unset", m->has_dsa, dsa_reason);
    printf("EFFECTIVE-FLAGS: IO_DELAY_US requested=%s effective=%d\n",
        io_delay_req?io_delay_req:"unset", g_io_delay_us);
    printf("EFFECTIVE-FLAGS: IO_DELAY_DECODE_US requested=%s effective=%d\n",
        io_delay_dec_req?io_delay_dec_req:"unset", g_io_delay_decode_us);
    printf("EFFECTIVE-FLAGS: PREFILL_DELAY_US effective=%d\n", g_io_delay_us);
    printf("EFFECTIVE-FLAGS: DECODE_DELAY_US effective=%d\n", g_io_delay_us+g_io_delay_decode_us);
    printf("EFFECTIVE-FLAGS: PIPE requested=%s effective=%d\n",
        pipe_req?pipe_req:"unset", g_pipe);
}

/* ---- fail-LOUD crash diagnostic (2026-07-20 robustness pass) --------------------
 * Found: a mode-1.5 `ili chat` session that had already completed one or more
 * PRIOR turns could die with ZERO diagnostic output partway through a LATER turn
 * -- no FATAL, no crash report, no OOM trace, the process just stops (see
 * c/scripts/mode15_e2e_certify.sh's own "KNOWN ISSUE" note). Root cause turned out
 * to be expert_bytes_probe() under-counting mode-1.5 resident bytes (see that
 * function's own comment) -- fixed below -- but a real OOM-kill is SIGKILL, which
 * no userspace handler can ever catch (this handler exists for every OTHER fatal
 * signal, and as defense in depth against whatever else might crash this process).
 * Whatever the fault, an actual SIGSEGV/SIGBUS/SIGABRT/SIGILL/SIGFPE should never
 * be allowed to end this process with NOTHING written to stderr: install a minimal
 * handler that writes ONE line naming the signal -- write(2,...) on a
 * PRECOMPUTED-LENGTH literal only (no strlen(): not on Apple's async-signal-safe
 * list, unlike write/sigaction/raise -- see sigaction(2)), no malloc/stdio/locks,
 * so it stays safe to call even if the fault happened mid-allocation or
 * mid-fprintf -- then restores the DEFAULT disposition and re-raises, so the OS's
 * own coredump/crash-reporting behavior for the re-raised signal is otherwise
 * unchanged; this is purely additive. Paired with the `ili` wrapper's own fix to
 * always surface the engine's captured stderr tail on ANY session death
 * (previously only shown on an initial-load failure, never on a later-turn one --
 * see ili's cmd_chat()), a future occurrence of this same class of bug can no
 * longer look "silent" even if its root cause is still open.
 * KNOWN LIMITATIONS (not addressed here, out of scope for this pass): does not
 * chain to a pre-existing handler, so building this engine under
 * -fsanitize=address/undefined (not this codebase's default build) would lose the
 * sanitizer's own richer report in favor of this one-line message; does not use
 * an alternate signal stack (sigaltstack), so a stack-overflow SIGSEGV on a
 * near-exhausted stack (main thread or an OMP worker) may still fail to run this
 * handler at all -- a per-OMP-thread altstack is real additional complexity, left
 * for later. */
static void ili_fatal_signal_handler(int sig){
    static const char m_segv[]="\nFATAL: iliria engine crashed: SIGSEGV (segmentation fault)\n";
    static const char m_bus[] ="\nFATAL: iliria engine crashed: SIGBUS (bus error)\n";
    static const char m_abrt[]="\nFATAL: iliria engine crashed: SIGABRT (abort)\n";
    static const char m_ill[] ="\nFATAL: iliria engine crashed: SIGILL (illegal instruction)\n";
    static const char m_fpe[] ="\nFATAL: iliria engine crashed: SIGFPE (floating-point exception)\n";
    static const char m_dflt[]="\nFATAL: iliria engine crashed: unexpected fatal signal\n";
    const char *msg; size_t len;
    switch(sig){
        case SIGSEGV: msg=m_segv; len=sizeof(m_segv)-1; break;
        case SIGBUS:  msg=m_bus;  len=sizeof(m_bus)-1;  break;
        case SIGABRT: msg=m_abrt; len=sizeof(m_abrt)-1; break;
        case SIGILL:  msg=m_ill;  len=sizeof(m_ill)-1;  break;
        case SIGFPE:  msg=m_fpe;  len=sizeof(m_fpe)-1;  break;
        default:      msg=m_dflt; len=sizeof(m_dflt)-1; break;
    }
    ssize_t wr = write(2, msg, len); (void)wr;   /* best-effort; nothing sane to do if this fails too */
    signal(sig, SIG_DFL);
    raise(sig);   /* re-raise with the default disposition: coredump/crash-report behavior unchanged */
}
static void ili_install_fatal_handlers(void){
    struct sigaction sa; memset(&sa,0,sizeof(sa));
    sa.sa_handler=ili_fatal_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags=0;   /* these signals never return to the faulting code -- SA_RESTART is moot */
    int sigs[]={SIGSEGV,SIGBUS,SIGABRT,SIGILL,SIGFPE};
    for(size_t i=0;i<sizeof(sigs)/sizeof(sigs[0]);i++)
        if(sigaction(sigs[i],&sa,NULL)!=0) perror("sigaction (fatal-signal handler install)");
}

int main(int argc, char **argv){
    /* ---- Permanent OpenMP hot-thread tuning. The per-expert matmul regions are
     * tiny and back-to-back; with the default passive wait policy libgomp parks
     * the worker team between regions and the re-wake latency dominates. Keeping
     * the threads hot (active spin) collapses that overhead — measured matmul
     * time 66.9s -> 20.9s on the Zen5 build, with no change to numerical output.
     *
     * libgomp reads the OMP_ / GOMP_ vars in a CONSTRUCTOR that runs before
     * main(), so setenv() here and continuing would be too late (verified:
     * setenv-in-main is ignored by the already-initialised runtime). Instead, on
     * first entry seed the winning defaults — respecting anything the user
     * already set (overwrite=0) — then re-exec self once so a fresh libgomp
     * constructor picks them up. The ILI_OMP_TUNED sentinel guards the exec so
     * we re-exec at most once. Fully overridable: any explicit OMP_/GOMP_ env the
     * user sets wins (overwrite=0), pre-setting ILI_OMP_TUNED=1 skips the
     * re-exec entirely (runs with whatever policy the environment already has),
     * and ILI_NO_OMP_TUNE=1 is a documented kill-switch that disables the whole
     * re-exec + tuning path (distinct from the internal ILI_OMP_TUNED sentinel).
     *
     * Must remain the FIRST statement in main(): argv is passed verbatim to execv(). */
    if(!ili_env("OMP_TUNED") && !ili_env("NO_OMP_TUNE")){
        setenv("OMP_WAIT_POLICY","active",0);  /* keep the team hot across the tiny per-expert matmul regions */
        setenv("GOMP_SPINCOUNT","200000",0);   /* spin briefly, then yield so long disk waits don't burn a core */
        setenv("OMP_PROC_BIND","close",0);     /* pack the team onto adjacent cores for cache locality */
        setenv("OMP_DYNAMIC","FALSE",0);       /* fixed team size: no per-region thread-count churn */
        setenv("ILI_OMP_TUNED","1",1);
#ifdef __linux__
        fprintf(stderr,"[OMP] hot-thread tuning: re-exec once (ILI_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/self/exe", argv);         /* returns only on failure -> fall through and run untuned */
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
    ili_install_fatal_handlers();   /* fail-LOUD on SIGSEGV/SIGBUS/SIGABRT/SIGILL/SIGFPE -- see above */
    const char *snap=getenv("SNAP"); if(!snap){fprintf(stderr,"SNAP=<dir>\n");return 1;}
    g_nopack = getenv("NOPACK")?1:0;
    g_drop = getenv("DROP")?1:0;
    g_prefetch = getenv("PREFETCH")?atoi(getenv("PREFETCH")):0;
    g_mmap = ili_env("MMAP")?atoi(ili_env("MMAP")):0;
    if(g_mmap) fprintf(stderr,"[MMAP] expert = viste zero-copy nei file (page cache = cache)\n");
    g_topk = getenv("TOPK")?atoi(getenv("TOPK")):0;
    g_topp = getenv("TOPP")?atof(getenv("TOPP")):0;
    g_mlock  = getenv("MLOCK")?atoi(getenv("MLOCK")):-1;   /* -1 auto (ON macOS), 0 off, 1 force / auto (ON macOS), 0 off, 1 force */
    g_spec = getenv("SPEC")?atoi(getenv("SPEC")):1;
    g_draft = getenv("DRAFT")?atoi(getenv("DRAFT")):-1;   /* -1 = auto: 3 se MTP, 0 senza */
    g_looka = getenv("LOOKA")?atoi(getenv("LOOKA")):0;    /* 1 = misura predicibilita' routing */
    g_pilot = getenv("PILOT")?atoi(getenv("PILOT")):0;    /* 1 = prefetch pilotato dal router */
    g_pilot_real = getenv("PILOT_REAL")?atoi(getenv("PILOT_REAL")):0; /* default OFF: load VERI cross-layer (value-preserving prefetch); PILOT_REAL=1 opta in */
    if(g_pilot_real) g_pilot=1;                           /* PILOT_REAL implica il pilota attivo */
    /* Default K: hint-only PILOT keeps 8 (WILLNEED hints are free, no eviction).
     * Under PILOT_REAL the speculative loads are REAL and create LRU eviction
     * pressure, so at ~28% mispredict a large K thrashes the cache — default to 6
     * (best-measured this session) unless the user set PILOT_K explicitly. */
    g_pilot_k = getenv("PILOT_K")?atoi(getenv("PILOT_K")):(g_pilot_real?6:8);
    if(g_pilot_k<1) g_pilot_k=1;
    g_pipe = getenv("PIPE")?atoi(getenv("PIPE")):0;       /* default OFF: overlap expert load ‖ matmul (byte-identical; reorders I/O). PIPE=1 opts in */
    g_pipe_nw = getenv("PIPE_WORKERS")?atoi(getenv("PIPE_WORKERS")):8; /* I/O worker threads */
    if(g_pipe_nw<1) g_pipe_nw=1;
    g_direct = getenv("DIRECT")?atoi(getenv("DIRECT")):0;
    /* Stage-1 throttle lever (deliverable 2) + its debug proof log -- see the two comment
     * blocks at io_delay_inject()/io_trace_log() above expert_prefetch/moe(). Both default
     * off: zero behavior change unless explicitly requested. */
    g_io_delay_us = ili_env("IO_DELAY_US") ? atoi(ili_env("IO_DELAY_US")) : 0;
    if(g_io_delay_us<0) g_io_delay_us=0;
    g_io_delay_decode_us = ili_env("IO_DELAY_DECODE_US") ? atoi(ili_env("IO_DELAY_DECODE_US")) : 0;
    if(ili_env("A2_TRACE")){ g_a2f=fopen(ili_env("A2_TRACE"),"w"); g_a2_on=(g_a2f!=NULL);
        g_a2_steps = ili_env("A2_STEPS")?atoi(ili_env("A2_STEPS")):1; }
    if(g_io_delay_decode_us<0) g_io_delay_decode_us=0;
    { const char *iotr=ili_env("IO_TRACE");
      if(iotr) strncpy(g_io_trace_path,iotr,sizeof(g_io_trace_path)-1); }
#ifdef ILI_MODE15
    { const char *m15tr=ili_env("MODE15_TRACE");
      if(m15tr) strncpy(g_m15_trace_path,m15tr,sizeof(g_m15_trace_path)-1); }
#endif
    g_idot = getenv("IDOT")?atoi(getenv("IDOT")):1;        /* 0 = kernel f32 esatti (A/B) */
    g_repin = getenv("REPIN")?atoi(getenv("REPIN")):0;     /* RFC: re-pin ogni n token emessi (0=off) / live re-pin every n emitted tokens (0=off) */
    g_absorb = getenv("ABSORB")?atoi(getenv("ABSORB")):-1; /* -1 auto: assorbita per S<=4 */
    g_dsa_force = getenv("DSA_FORCE")?atoi(getenv("DSA_FORCE")):0;
    g_temp = getenv("TEMP")?atof(getenv("TEMP")):-1;       /* -1 = auto (1.0 chat/testo, greedy altrove) */
    g_nuc  = getenv("NUCLEUS")?atof(getenv("NUCLEUS")):0.90f;  /* piu' stretto dell'ufficiale 0.95: la coda int4 e' rumore */
    if(getenv("SEED")) g_rng = (uint64_t)atoll(getenv("SEED"))*0x9E3779B97F4A7C15ULL+1;
    else { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); g_rng ^= (uint64_t)ts.tv_nsec<<20 ^ (uint64_t)getpid(); }
    if(g_draft>63) g_draft=63;                             /* -1 = auto, risolto dopo model_init */
    int cap  = argc>1?atoi(argv[1]):64;
    int ebits= argc>2?atoi(argv[2]):8;
    int dbits= argc>3?atoi(argv[3]):ebits;
    if(getenv("SERVE") && (kv_slot_count()<1 || kv_slot_count()>16)){
        fprintf(stderr,"KV_SLOTS must be between 1 and 16\n"); return 2;
    }
#ifdef ILI_CUDA
    if(ili_env("CUDA") && atoi(ili_env("CUDA"))){
        const char *one=ili_env("GPU"), *many=ili_env("GPUS");
        if(one&&many){ fprintf(stderr,"use ILI_GPU or ILI_GPUS, not both\n"); return 2; }
        if(many) g_cuda_ndev=parse_cuda_devices(many,g_cuda_devices);
        else if(one) g_cuda_ndev=parse_cuda_devices(one,g_cuda_devices);
        else { g_cuda_ndev=1; g_cuda_devices[0]=0; }
        if(g_cuda_ndev<1){ fprintf(stderr,"invalid ILI_GPUS: use a list such as 0,1,2\n"); return 2; }
        g_cuda_enabled=ili_cuda_init(g_cuda_devices,g_cuda_ndev);
        if(!g_cuda_enabled){ fprintf(stderr,"[CUDA] requested backend is unavailable\n"); return 2; }
    }
    g_cuda_dense=getenv("CUDA_DENSE")?atoi(getenv("CUDA_DENSE")):0;
    g_cuda_expert_gb=getenv("CUDA_EXPERT_GB")?atof(getenv("CUDA_EXPERT_GB")):0;
    if((ili_env("GPU")||ili_env("GPUS"))&&!g_cuda_enabled){ fprintf(stderr,"ILI_GPU(S) requires ILI_CUDA=1\n"); return 2; }
    if(g_cuda_dense&&!g_cuda_enabled){ fprintf(stderr,"CUDA_DENSE requires ILI_CUDA=1\n"); return 2; }
    if(g_cuda_expert_gb>0 && !g_cuda_enabled){ fprintf(stderr,"CUDA_EXPERT_GB requires ILI_CUDA=1\n"); return 2; }
    if(g_cuda_enabled) fprintf(stderr,"[CUDA] mode: routed experts%s\n",g_cuda_dense?" + resident dense tensors":" only (resident dense on CPU)");
#else
    if((ili_env("CUDA") && atoi(ili_env("CUDA"))) ||
       ili_env("GPU") || ili_env("GPUS") ||
       (getenv("CUDA_DENSE") && atoi(getenv("CUDA_DENSE"))) ||
       (getenv("CUDA_EXPERT_GB") && atof(getenv("CUDA_EXPERT_GB"))>0)){
        fprintf(stderr,"CUDA was requested, but this binary is CPU-only; rebuild with: make CUDA=1\n");
        return 2;
    }
#endif
#ifdef ILI_METAL
    if(ili_env("METAL") && atoi(ili_env("METAL"))){
        g_metal_enabled = ili_metal_init();
        if(!g_metal_enabled){ fprintf(stderr,"[METAL] backend requested but not available\n"); return 2; }
        fprintf(stderr,"[METAL] mode: batched routed experts on GPU (unified-memory zero-copy)\n");
        if(ili_env("METAL_SPIN") && atoi(ili_env("METAL_SPIN"))){ ili_metal_spin_start(); fprintf(stderr,"[METAL] keep-alive spinner ON\n"); }
        if(ili_env("METAL_GEMM_MIN")) g_metal_gemm_min=atoi(ili_env("METAL_GEMM_MIN"));
        if(ili_env("METAL_PREFILL")) g_metal_prefill=atoi(ili_env("METAL_PREFILL"));
        if(g_metal_prefill) fprintf(stderr,"[METAL] prefill attention (S>4) on GPU: ON\n");
    }
#else
    if(ili_env("METAL") && atoi(ili_env("METAL"))){
        fprintf(stderr,"METAL was requested, but this binary has no Metal backend; rebuild with: make METAL=1\n");
        return 2;
    }
#endif
    printf("== GLM C engine (glm_moe_dsa), cache=%d experts/layer | experts@%d-bit dense@%d-bit | idot: " IDOT_KERNEL " ==\n", cap, ebits, dbits);
    g_mem_avail_boot = mem_available_gb();
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    if(g_draft<0) g_draft = m.has_mtp ? 3 : 0;
    if(getenv("DSA_TOPK")) m.c.index_topk=atoi(getenv("DSA_TOPK"));   /* override per test */
    print_effective_flags(&m);
    printf("loaded in %.2fs | resident dense: %.2f MB | layers=%d experts=%d | MTP %s (draft=%d)\n",
           now_s()-t0, m.resident_bytes/(1024.0*1024.0), m.c.n_layers, m.c.n_experts,
           m.has_mtp?(g_draft>0?"ACTIVE":"disabled"):"absent", g_draft);
    /* anche su stderr: e' il canale che le UI (ili) mostrano all'utente */
    fprintf(stderr,"[MTP] %s (draft=%d)\n",
            m.has_mtp?(g_draft>0?"active: native speculative decoding":"disabled"):"absent", g_draft);
    if(!strncmp(snap,"/mnt/",5))
        fprintf(stderr,"WARNING: the model is on %s (slow 9p/Windows filesystem; fadvise is ineffective).\n"
                       "         Keep it on ext4 (for example, /home/...) for memory efficiency and speed.\n", snap);
    /* HOT-STORE: PIN=<statsfile> [PIN_GB=g] -> top expert per frequenza fissi in RAM.
     * Va PRIMA di cap_for_ram: i pinnati contano nel residente. */
    if(getenv("PIN")) pin_load(&m, getenv("PIN"), getenv("PIN_GB")?atof(getenv("PIN_GB")):10.0);
    /* CACHE CHE IMPARA: l'uso degli expert si accumula in <SNAP>/.fa_usage tra le sessioni;
     * all'avvio i piu' usati vengono auto-pinnati in RAM (meta' del budget expert: il pin
     * conosce la TUA storia, la LRU si adatta alla sessione). AUTOPIN=0 disattiva.
     * ILI_HOTSET_PROFILE selects a workload-specific file (.fa_usage.<profile>). */
    { const char *profile_env = ili_env("HOTSET_PROFILE");
      if(profile_env && hotset_validate_profile(profile_env)){
          strncpy(g_hotset_profile, profile_env, sizeof(g_hotset_profile)-1);
          g_hotset_profile[sizeof(g_hotset_profile)-1] = '\0';
      }
      /* Read decay config */
      { const char *decay_env = ili_env("HOTSET_DECAY");
        if(decay_env){
            int d = atoi(decay_env);
            if(d > 0 && d < 100) g_hotset_decay_factor = d;
        }
      }
      { const char *interval_env = ili_env("HOTSET_DECAY_INTERVAL");
        if(interval_env){
            int64_t iv = atoll(interval_env);
            if(iv > 0) g_hotset_decay_interval = iv;
        }
      }
      double ram_env = getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx = getenv("CTX")?atoi(getenv("CTX")):4096;   /* stesso default di run_serve */
      hotset_usage_path(g_usage_path, sizeof(g_usage_path), snap, g_hotset_profile);
      int64_t hist = usage_load(&m,g_usage_path);
      if(hist>0) fprintf(stderr,"[USAGE] expert history: %lld selections (%s)\n",(long long)hist,g_usage_path);
      int autopin = getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
      if(!getenv("PIN") && autopin && hist>=5000){
          /* quota pin proporzionale alla FIDUCIA nella storia: con pochi dati il pin
           * sbaglia expert e ruba slot alla LRU adattiva; a regime (>=200k selezioni,
           * qualche ora di chat) arriva a meta' del budget expert. */
          double conf = (double)hist/200000.0; if(conf>1) conf=1;
          double pin_gb = expert_avail(&m,ram_env,ebits,est_ctx)*0.5*conf/1e9;
          if(pin_gb>=0.5) pin_load(&m, g_usage_path, pin_gb);
      }
      /* SEMPRE: senza clamp la LRU cresce fino a cap*76 layer = decine di GB -> OOM-kill.
       * RAM_GB assente o <=0 = budget automatico da MemAvailable. */
      cap_for_ram(&m, ram_env, ebits, est_ctx); }
    const char *stats=getenv("STATS");   /* STATS=<file> -> istogramma uso expert a fine run */

    /* modo scoring per benchmark: SCORE=<requests.txt> -> log-likelihood per riga */
    if(getenv("SCORE")){ run_score(&m, getenv("SCORE")); if(stats) stats_dump(&m,stats); return 0; }

    /* modo serve persistente per la CLI 'ili': SERVE=1 */
    if(getenv("SERVE")){ run_serve(&m, snap); if(stats) stats_dump(&m,stats); return 0; }

    /* modo testo reale: PROMPT="..." [NGEN=n] -> tokenizza, genera, detokenizza */
    if(getenv("PROMPT")){
        int ngen=getenv("NGEN")?atoi(getenv("NGEN")):64;
        run_text(&m, snap, getenv("PROMPT"), ngen);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    /* altrimenti: validazione contro l'oracolo (ref_glm.json) */
    const char *refpath=getenv("REF")?getenv("REF"):"ref_glm.json";
    FILE *f=fopen(refpath,"rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    char *ar=NULL; jval *ref=json_parse(b,&ar);
    int np,nfull; int *prompt=read_arr(ref,"prompt_ids",&np); int *full=read_arr(ref,"full_ids",&nfull);
    int n_new=nfull-np;
    /* L'oracolo (ref_glm.json in repo) e' del modello TINY: contro il 744B da' 0/20
     * garantito su OGNI piattaforma (prompt-token tiny = spazzatura per il modello vero).
     * Non e' un bug del motore — vedi #76. */
    { int maxid=0; for(int i=0;i<nfull;i++) if(full[i]>maxid) maxid=full[i];
      if(m.c.vocab>1000 && maxid<1000 && !getenv("REF_FORCE")){
        fprintf(stderr,"ERRORE: ref_glm.json e' l'oracolo del modello TINY (token max %d, ma il tuo vocab e' %d).\n"
                       "        Self-test motore:  SNAP=./glm_tiny TF=1 ./glm 64 16 16   (atteso 32/32)\n"
                       "        Prova reale:       PROMPT=\"Ciao\" NGEN=32 SNAP=<modello> ./glm 64\n"
                       "        REF_FORCE=1 per eseguire comunque il confronto (senza senso).\n", maxid, m.c.vocab);
        return 1;
      } }

    if(getenv("REPLAY")){
        run_replay(&m,full,nfull,np);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    if(getenv("TF")){
        int *tf=read_arr(ref,"tf_pred",&(int){0});
        int *pred=malloc(nfull*sizeof(int)); double tt=now_s();
        forward_all(&m, full, nfull, pred); double tdt=now_s()-tt;
        int ok=0; for(int i=0;i<nfull;i++){
            if(pred[i]==tf[i]) ok++;
            else fprintf(stderr,"[ORACLE] mismatch pos=%d expected=%d got=%d\n",i,tf[i],pred[i]);
        }
        printf("PREFILL (teacher-forcing) C vs oracle: %d/%d positions | %.1f pos/s\n",
            ok,nfull,nfull/tdt);
        if(ok<nfull) fprintf(stderr,
            "[ORACLE] %d/%d mismatches — run: TF=1 DEBUG_LOGITS=1 for top-5 logit dump\n",
            nfull-ok,nfull);
        profile_print(&m,tdt);
#ifdef ILI_CUDA
        if(g_cuda_enabled) cuda_stats_print();
#endif
        return 0;
    }
    int *out=malloc((np+n_new)*sizeof(int));
    double t=now_s(); generate(&m,prompt,np,n_new,out); double dt=now_s()-t;
    int match=0;
    printf("\nReference (oracle): "); for(int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nGLM C engine      : "); for(int i=np;i<nfull;i++){ printf("%d ", out[i]); if(out[i]==full[i])match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double tot=m.hits+m.miss;
    printf("N-gram speculation (DRAFT=%d): %.2f tokens/forward (%llu forwards per %llu tokens)\n",
        g_draft, m.n_fw?(double)m.n_emit/m.n_fw:1.0, (unsigned long long)m.n_fw, (unsigned long long)m.n_emit);
    printf("Expert cache hit rate: %.1f%% (hit=%llu miss=%llu) | RSS: %.2f GB | %.1f tok/s\n",
           tot?100.0*m.hits/tot:0.0, (unsigned long long)m.hits, (unsigned long long)m.miss, rss_gb(), n_new/dt);
    profile_print(&m,dt);
#ifdef ILI_CUDA
    if(m.gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m.gpu_expert_count,m.gpu_expert_bytes/1e9,(unsigned long long)m.gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    if(g_looka){
        const char *nm[3]={"previous token (=SPEC prefetch)","layer input, skip attention","next layer (one step ahead)"};
        printf("LOOKAHEAD routing — recall of true experts in predicted top-8:\n");
        for(int i=0;i<3;i++) printf("  %-38s %5.1f%%  (%lld/%lld)\n", nm[i],
            la_tot[i]?100.0*la_hit[i]/la_tot[i]:0.0, (long long)la_hit[i], (long long)la_tot[i]);
    }
    if(stats) stats_dump(&m,stats);
    return 0;
}
