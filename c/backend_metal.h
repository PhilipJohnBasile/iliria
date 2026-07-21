/* Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE. */
#ifndef ILI_BACKEND_METAL_H
#define ILI_BACKEND_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Apple-GPU (Metal) backend for iliria. Apple Silicon has one GPU and unified
 * memory, so there is no device list and no host<->device copy: resident weights
 * are read zero-copy from the RAM they already occupy. The shader is compiled at
 * runtime (newLibraryWithSource:), so no Xcode / offline metal compiler is needed.
 */

/* Opaque, persistent GPU handle for one resident quantized tensor. */
typedef struct IliMetalTensor IliMetalTensor;

/* Returns 1 if a Metal device is available and pipelines compiled, else 0. */
int  ili_metal_init(void);
void ili_metal_shutdown(void);
int  ili_metal_available(void);
/* Bytes of unified memory in use by wrapped tensors, and their count. */
void ili_metal_stats(size_t *tensor_count, size_t *tensor_bytes);
int  ili_metal_mem_info(size_t *used_bytes, size_t *total_bytes);

/*
 * y[S,O] = (x[S,I] @ W[O,I]^T) * scale[o].
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4(packed), 3=int2(packed).
 * The first successful call wraps W and its row scales in GPU-visible buffers;
 * later calls reuse them (weights are assumed stable at the same address).
 * Returns 1 on success, 0 if Metal is unavailable or fmt is invalid.
 */
int ili_metal_matmul(IliMetalTensor **tensor,
                      float *y, const float *x,
                      const void *weights, const float *scales,
                      int fmt, int S, int I, int O);

void   ili_metal_tensor_free(IliMetalTensor *tensor);
size_t ili_metal_tensor_bytes(const IliMetalTensor *tensor);

/*
 * Register a page-aligned host allocation (expert slab / scale slab) so the batched
 * MoE path can read it zero-copy: the backend wraps it once in an MTLBuffer
 * (newBufferWithBytesNoCopy) and resolves any pointer inside [base,base+len) to a GPU
 * address. Call after (re)allocating a slab; call unregister before freeing it.
 * base must be aligned to 16384 (Apple page) and len a multiple of it.
 */
void ili_metal_spin_start(void);   /* ILI_METAL_SPIN=1 keep-alive experiment */
void ili_metal_spin_stop(void);
void ili_metal_register(void *base, size_t len);
void ili_metal_unregister(void *base);

/*
 * Fused decode (S=1) attention for one layer, entirely on the GPU in one command buffer:
 * q_a -> rmsnorm -> q_b -> RoPE ; kv_a -> latent rmsnorm@pos + krot RoPE@pos (cache write) ;
 * MLA absorption core ; o_proj. Weights (q_a/q_b/kv_a/kv_b/o) and the Lc/Rc caches must be
 * registered (page-aligned) for zero-copy resolve. GLM-5.2 dims compiled in. Handles st0==0
 * full-range only. Returns 1 on success, 0 to signal CPU fallback.
 */
/*
 * Full decode layer in ONE command buffer: in_ln -> attention -> residual -> post_ln ->
 * shared expert -> router+top-K (exact phase-A semantics). x updated in place; nrm_out
 * is the expert input; sh_out the shared-expert output; idx/w/keff the routing.
 * Returns 0 -> caller runs the whole layer on the CPU path.
 */
int ili_metal_layer_decode(float *x,
    const float *in_ln, const float *post_ln,
    const void *qa_w, const float *qa_s, int qa_fmt, const float *qa_ln,
    const void *qb_w, const float *qb_s, int qb_fmt,
    const void *kva_w, const float *kva_s, int kva_fmt, const float *kva_ln,
    const void *kvb_w, const float *kvb_s, int kvb_fmt,
    const void *o_w, const float *o_s, int o_fmt,
    const void *shg_w, const float *shg_s, int shg_fmt,
    const void *shu_w, const float *shu_s, int shu_fmt,
    const void *shd_w, const float *shd_s, int shd_fmt,
    const float *router_w, const float *router_bias,
    int E, int K, int Ksel, float topp, int normk, float rscale,
    float *Lc, float *Rc, int S, int pos_base, int st0,
    float eps, float theta, float ascale,
    float *inrm_out, float *nrm_out, float *sh_out, int *idx_out, float *w_out, int *keff_out);

int ili_metal_gemm(float *y, const float *x, const void *weights, const float *scales,
                    int fmt, int S, int I, int O);   /* large-batch sync GEMM; 0 -> CPU */
void ili_metal_attn_counts(uint64_t *ok, double *wall, double *kernel);
void ili_metal_attn_lat(double *ksched, double *gsched);
/* Roofline harness: GPU-time counters for ili_metal_gemm (the prefill "projections" path
 * -- o_proj/dense-MLP/kv_b-reconstruction/logits, see matmul_qt in glm.c), previously
 * uninstrumented. Same ok/wall/kernel shape as ili_metal_attn_counts. */
void ili_metal_gemm_counts(uint64_t *ok, double *wall, double *kernel);
int ili_metal_attn_decode(const float *x,
    const void *qa_w, const float *qa_s, int qa_fmt, const float *qa_ln,
    const void *qb_w, const float *qb_s, int qb_fmt,
    const void *kva_w, const float *kva_s, int kva_fmt, const float *kva_ln,
    const void *kvb_w, const float *kvb_s, int kvb_fmt,
    const void *o_w, const float *o_s, int o_fmt,
    float *Lc, float *Rc, int S, int pos_base, int st0, float eps, float theta, float ascale, float *out);

/* Prefill attention (S>4, GLM-5.2 dims, full KV from position 0, no DSA selection):
 * absorption-form score/softmax/AV over the compressed cache for a whole query batch.
 * Q is the roped query block [S, H*QH]; Lc/Rc must already hold all S new positions;
 * ctx receives [S, H*VH] (the o-projection stays with the caller). 0 -> CPU fallback. */
int ili_metal_attn_prefill(const float *Q,
    const void *kvb_w, const float *kvb_s, int kvb_fmt,
    float *Lc, float *Rc, int S, int pos_base, float ascale, float *ctx);

/* Diagnostics: GPU blocks executed, CPU-fallback blocks, experts run on GPU. */
void ili_metal_moe_counts(uint64_t *ok, uint64_t *fb, uint64_t *experts);
void ili_metal_moe_times(double *setup, double *gpu, double *scatter);
double ili_metal_moe_kernel_time(void);

/*
 * Batched routed-expert SwiGLU for one MoE block, in ONE command buffer.
 * For each expert e in [0,nb): computes hh_e[nr_e, D] = down( silu(gate(xg_e)) * up(xg_e) )
 * and scatter-adds rw * hh_e into out. All experts share the command buffer so the
 * ~150us Metal launch latency is paid once per block, not per matmul.
 *
 *  D           = hidden size, Iinter = moe intermediate size
 *  g/u/d[e]    = pointers to expert e's gate/up/down quantized weights (in RAM slabs)
 *  gs/us/ds[e] = pointers to expert e's per-row scales
 *  fmt         = quant format (shared across experts)
 *  xg          = packed activations [total_rows, D]; xoff[e] = row offset of expert e
 *  nr[e]       = rows for expert e; rows[]/rw[] map packed rows back to out positions
 *  out         = [S, D] accumulate target
 * Returns 1 on success, 0 to signal the caller to fall back to the CPU path.
 *
 * Two opt-in fusions (default off, both env-gated "initially" -- see docs/performance-theory.json
 * entry a1-fused-eight-expert-layer-kernel for the broader per-layer fusion this is a step toward):
 *
 *  ILI_GPU_REDUCE=1    Replace the CPU scatter-add (out[rows[r],d] += rw[r]*hh[r,d]) and its
 *                       CPU-side read of hh with a Metal reduction kernel in the SAME command
 *                       buffer as the gate/up/down GEMVs: out[s,d] += sum over r with rows[r]==s
 *                       of rw[r]*hh[r,d]. `out` is seeded with its own pre-call contents (so it
 *                       composes with whatever the caller already accumulated -- residual, shared
 *                       expert, earlier blocks) and read back once the block completes. hh never
 *                       becomes CPU-visible in this path.
 *  ILI_FUSED_GATEUP=1  Replace the gate-GEMV + up-GEMV + barrier + moe_silu + barrier sequence
 *                       with one kernel that computes both dot products per row and combines them
 *                       (silu(gate*scale_g)*(up*scale_u)) before ever writing to memory: no
 *                       up-output buffer, no separate silu dispatch, one fewer barrier. The down
 *                       projection is unchanged. fmt 1 (int8) and 2 (int4) only; fmt 3 (int2) and
 *                       any future compressed-row decode are an explicit, commented seam in the
 *                       kernel and fall back to the CPU path rather than silently misreading bytes
 *                       (see the ILI_CPU_MOE_ALL mixed-container gate in run_container_gates.sh).
 *
 * Both flags are read fresh (not cached) so a single process can exercise every combination; the
 * cost of doing so is one getenv+atoi per block call, negligible next to a GPU dispatch. Neither
 * flag touches ili_metal_moe_block_begin/_end (the disk-overlap async path): those keep today's
 * CPU scatter and 3-kernel gate/up/silu unconditionally in this initial cut.
 */
int ili_metal_moe_block(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw,
                         float *out, int S);

/* Counters for the two opt-in fusions above: calls = ili_metal_moe_block invocations that took
 * the fused path, dispatches = total Metal kernel dispatches issued by those calls, submits =
 * total command-buffer commits (always 1 per call -- both fusions stay inside the single
 * gate/up/down command buffer, they never add a second submit). All 0 if never triggered. */
void ili_metal_moe_fused_stats(uint64_t *calls, uint64_t *dispatches, uint64_t *submits);

/*
 * Async two-phase variant: begin encodes+commits the block (own scratch, no wait) and
 * returns a handle, so the CPU can load missed experts from disk WHILE the GPU computes
 * the resident ones; end waits, checks for GPU faults, scatter-adds into out, and frees
 * the handle. begin returns NULL (nothing submitted) on unresolved slab / bad fmt / R==0;
 * end returns 0 on GPU fault (caller redoes those experts on CPU).
 */
typedef struct IliMetalMoeHandle IliMetalMoeHandle;
IliMetalMoeHandle* ili_metal_moe_block_begin(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw);
int ili_metal_moe_block_end(IliMetalMoeHandle *h, float *out);

#ifdef __cplusplus
}
#endif

#endif
