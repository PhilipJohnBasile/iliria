// PRE-FALSIFIER for docs/performance-theory.json a4-mla-native-tiled-dense-attention.
//
// The a4 entry's own notes say this must run FIRST, before any kernel rewrite: "benchmark
// the exact QK^T shapes (576xT for the score stage...) via a tiled-GEMM or tensor-primitive
// path on real M5 hardware. Kill the rewrite outright if the score stage cannot beat the
// CURRENT kernel by >=2.5x on this microbench alone." This tool does exactly that, in
// isolation from the rest of the attention chain and from any real model
// (SYNTHETIC data only; no /path/to/models reads):
//
//   pf_score_naive: the CURRENT score-stage kernel's exact per-thread compute pattern --
//     one thread per (row,t) output score, a single scalar 576-wide dot (see a_score,
//     backend_metal.mm ~line 153-162; causal masking is omitted here since this measures
//     the raw score-stage GEMM primitive on the literal dense shape the a4 entry specifies,
//     Qabs[S*64,576] x Kc[576,T] -- causal skipping is an orthogonal, separately-applicable
//     optimization that would help both arms roughly equally).
//   pf_score_tiled: a hand-tiled, threadgroup-shared-memory-blocked GEMM (BM=BN=16, BK=64,
//     9 k-chunks since 576=9*64) -- the "tiled QK^T" the entry asks to compare against.
//
// Both compute the identical GEMM Qabs[rows,576] @ Kc[T,576]^T -> scores[rows,T] (Kc stored
// row-major [T,576], matching the real Lc/Rc per-token layout, i.e. this is literally
// "A @ B^T", the standard attention QK^T form) on synthetic random data, at every
// (S,T) in {4,64,256} x {1024,4096,16384} the entry specifies (rows = S*64).
//
// Reports, per shape and in aggregate: the measured naive-vs-tiled factor, held up plainly
// against the entry's own >=2.5x kill criterion. These are fixture/synthetic numbers on
// isolated kernels -- directional evidence for the kill/no-kill decision, not a production
// benchmark of the full attention chain.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static const char *SHADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Mirrors a_score's exact per-thread pattern: one thread per (row,t), one 576-wide scalar
// dot, no causal skip (dense, matching the entry's literal Qabs[rows,576] x Kc[576,T] shape).
kernel void pf_score_naive(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
                           device float* S [[buffer(2)]], constant int& ROWS [[buffer(3)]],
                           constant int& TT [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
  int row = (int)gid / TT, t = (int)gid % TT;
  if (row >= ROWS) return;
  device const float* q = Q + (long)row*576;
  device const float* k = K + (long)t*576;
  float a = 0.0f;
  for (int i = 0; i < 576; i++) a += q[i]*k[i];
  S[(long)row*TT + t] = a;
}

// Hand-tiled, threadgroup-shared-memory-blocked GEMM: BM=BN=16 output tile, BK=64 k-chunk
// (576 = 9*64, so no k-remainder handling is needed; BK=64 rather than a smaller chunk
// halves the barrier count for the same total work, measured to matter more than the tile
// shape itself for this shape family). Each of the 256 threads/threadgroup owns exactly one
// output element for the whole kernel body; the tile amortizes each Q/K element it loads
// across all 16 output rows/cols that reuse it (16x fewer global loads than the naive
// kernel for the same output tile).
kernel void pf_score_tiled(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
                           device float* S [[buffer(2)]], constant int& ROWS [[buffer(3)]],
                           constant int& TT [[buffer(4)]],
                           uint2 tgpos [[threadgroup_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]]) {
  threadgroup float As[16][64];
  threadgroup float Bs[64][16];
  int row = (int)tgpos.y*16 + (int)lid.y;
  int col = (int)tgpos.x*16 + (int)lid.x;
  int flat = (int)lid.y*16 + (int)lid.x;
  float acc = 0.0f;
  for (int k0 = 0; k0 < 576; k0 += 64) {
    for (int e = flat; e < 1024; e += 256) {          // As is 16x64=1024 elems, 4/thread
      int ar = e/64, ac = e%64;
      As[ar][ac] = Q[(long)((int)tgpos.y*16+ar)*576 + k0+ac];
    }
    for (int e = flat; e < 1024; e += 256) {          // Bs is 64x16=1024 elems, 4/thread
      int br = e/16, bc = e%16;
      Bs[br][bc] = K[(long)((int)tgpos.x*16+bc)*576 + k0+br];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < 64; kk++) acc += As[lid.y][kk]*Bs[kk][lid.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < ROWS && col < TT) S[(long)row*TT + col] = acc;
}
)METAL";

struct Ctx {
  id<MTLDevice> dev; id<MTLCommandQueue> q;
  id<MTLComputePipelineState> naive, tiled;
};

static Ctx setup() {
  Ctx c{};
  c.dev = MTLCreateSystemDefaultDevice();
  if (!c.dev) { fprintf(stderr, "no Metal device\n"); exit(1); }
  c.q = [c.dev newCommandQueue];
  NSError *err = nil;
  id<MTLLibrary> lib = [c.dev newLibraryWithSource:[NSString stringWithUTF8String:SHADER] options:nil error:&err];
  if (!lib) { fprintf(stderr, "shader compile failed: %s\n", err ? [[err localizedDescription] UTF8String] : "?"); exit(1); }
  c.naive = [c.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pf_score_naive"] error:&err];
  c.tiled = [c.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pf_score_tiled"] error:&err];
  if (!c.naive || !c.tiled) { fprintf(stderr, "pipeline creation failed\n"); exit(1); }
  return c;
}

// Runs `iters` timed dispatches (after `warmup` untimed ones) of one kernel; returns the
// mean GPU-side time in seconds (cb.GPUEndTime-cb.GPUStartTime, this file's own convention
// for isolating kernel cost from CPU-side encode/submit latency).
static double time_kernel(Ctx&c, id<MTLComputePipelineState> pso, bool tiled,
                          id<MTLBuffer> Q, id<MTLBuffer> K, id<MTLBuffer> Sbuf,
                          int rows, int T, int warmup, int iters) {
  auto dispatch_once = [&]() -> id<MTLCommandBuffer> {
    id<MTLCommandBuffer> cb = [c.q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:K offset:0 atIndex:1]; [e setBuffer:Sbuf offset:0 atIndex:2];
    [e setBytes:&rows length:4 atIndex:3]; [e setBytes:&T length:4 atIndex:4];
    if (tiled) {
      [e dispatchThreadgroups:MTLSizeMake((size_t)T/16,(size_t)rows/16,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
    } else {
      [e dispatchThreads:MTLSizeMake((size_t)rows*T,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    [e endEncoding]; [cb commit];
    return cb;
  };
  for (int i = 0; i < warmup; i++) { id<MTLCommandBuffer> cb = dispatch_once(); [cb waitUntilCompleted]; }
  double total = 0.0;
  for (int i = 0; i < iters; i++) {
    id<MTLCommandBuffer> cb = dispatch_once();
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
      fprintf(stderr, "cmdbuf error: %s\n", cb.error ? [[cb.error localizedDescription] UTF8String] : "?");
      exit(1);
    }
    total += [cb GPUEndTime] - [cb GPUStartTime];
  }
  return total / iters;
}

struct ShapeResult { int S, T, rows; double t_naive, t_tiled, factor; double max_abs_diff, ref_max; };

static ShapeResult run_shape(Ctx&c, int S, int T) {
  int rows = S*64;
  @autoreleasepool {
    std::vector<float> Qh((size_t)rows*576), Kh((size_t)T*576);
    srand(1000u + (unsigned)S*97u + (unsigned)T);
    for (auto&v : Qh) v = ((rand()%2000)-1000)/1000.f;
    for (auto&v : Kh) v = ((rand()%2000)-1000)/1000.f;
    id<MTLBuffer> Q = [c.dev newBufferWithBytes:Qh.data() length:Qh.size()*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> K = [c.dev newBufferWithBytes:Kh.data() length:Kh.size()*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> Sn = [c.dev newBufferWithLength:(size_t)rows*T*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> St = [c.dev newBufferWithLength:(size_t)rows*T*4 options:MTLResourceStorageModeShared];

    // Correctness first: same GEMM, must agree (order-of-summation differences only).
    { id<MTLCommandBuffer> cb=[c.q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
      [e setComputePipelineState:c.naive]; [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:K offset:0 atIndex:1];
      [e setBuffer:Sn offset:0 atIndex:2]; [e setBytes:&rows length:4 atIndex:3]; [e setBytes:&T length:4 atIndex:4];
      [e dispatchThreads:MTLSizeMake((size_t)rows*T,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb=[c.q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
      [e setComputePipelineState:c.tiled]; [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:K offset:0 atIndex:1];
      [e setBuffer:St offset:0 atIndex:2]; [e setBytes:&rows length:4 atIndex:3]; [e setBytes:&T length:4 atIndex:4];
      [e dispatchThreadgroups:MTLSizeMake((size_t)T/16,(size_t)rows/16,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; }
    const float *an=(const float*)[Sn contents], *at=(const float*)[St contents];
    double max_abs=0, refmax=0; size_t n=(size_t)rows*T;
    for (size_t i=0;i<n;i++) { max_abs=fmax(max_abs,fabs((double)an[i]-at[i])); refmax=fmax(refmax,fabs((double)an[i])); }

    int warmup=2, iters=5;
    double t_naive = time_kernel(c, c.naive, false, Q, K, Sn, rows, T, warmup, iters);
    double t_tiled = time_kernel(c, c.tiled, true,  Q, K, St, rows, T, warmup, iters);
    return { S, T, rows, t_naive, t_tiled, t_naive/t_tiled, max_abs, refmax };
  }
}

int main(void) {
  Ctx c = setup();
  printf("Pre-falsifier (docs/performance-theory.json a4-mla-native-tiled-dense-attention):\n");
  printf("current score-stage kernel (pf_score_naive) vs tiled QK^T (pf_score_tiled), synthetic data.\n");
  printf("Kill criterion: tiled must beat naive by >=2.5x, else a4's kernel rewrite does not proceed.\n\n");
  int S_vals[3] = {4,64,256};
  int T_vals[3] = {1024,4096,16384};
  std::vector<ShapeResult> results;
  bool correctness_ok = true;
  for (int S : S_vals) for (int T : T_vals) {
    ShapeResult r = run_shape(c, S, T);
    results.push_back(r);
    double relerr = r.max_abs_diff / (r.ref_max + 1e-9);
    bool ok = relerr < 1e-3;
    correctness_ok &= ok;
    printf("S=%-4d T=%-6d rows=%-6d  naive=%9.5f ms  tiled=%9.5f ms  factor=%6.2fx  [naive-vs-tiled relerr=%.2e %s]\n",
           r.S, r.T, r.rows, r.t_naive*1e3, r.t_tiled*1e3, r.factor, relerr, ok?"ok":"*** MISMATCH");
  }
  double sum_naive=0, sum_tiled=0;
  for (auto&r : results) { sum_naive+=r.t_naive; sum_tiled+=r.t_tiled; }
  double agg_factor = sum_naive/sum_tiled;
  printf("\n==================== PRE-FALSIFIER RESULT ====================\n");
  for (auto&r : results) {
    printf("  S=%-4d T=%-6d : %6.2fx  (kill line 2.5x: %s)\n", r.S, r.T, r.factor, r.factor>=2.5 ? "CLEARS" : "MISSES");
  }
  printf("  ---------------------------------------------------------\n");
  printf("  aggregate (total naive time / total tiled time): %.2fx\n", agg_factor);
  printf("  KILL CRITERION (score-stage >= 2.5x): %s\n", agg_factor>=2.5 ? "CLEARS -- pre-falsifier does NOT kill the rewrite" : "MISSES -- pre-falsifier KILLS the rewrite per the a4 entry's own rule");
  printf("  (fixture/synthetic numbers on isolated kernels, directional -- not a full-chain benchmark)\n");
  printf("================================================================\n");
  if (!correctness_ok) { printf("\ncorrectness FAILED: naive and tiled kernels disagree beyond tolerance\n"); return 1; }
  return 0;
}
