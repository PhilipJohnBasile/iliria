/* Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE. */
// Kernel-correctness test for the Metal backend: ili_metal_matmul vs CPU reference
// (dequant->f32 MAC * per-row scale) for f32/int8/int4/int2 across real GLM shapes.
#include "../backend_metal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

enum { F32=0, I8=1, I4=2, I2=3 };

static void cpu_ref(int fmt, const void *W, const float *s, const float *x,
                    float *y, int S, int I, int O) {
  const int8_t *q8 = (const int8_t*)W; const uint8_t *q4 = (const uint8_t*)W;
  const float *qf = (const float*)W;
  int rb4=(I+1)/2, rb2=(I+3)/4;
  for (int o=0;o<O;o++) for (int si=0;si<S;si++){
    const float *xr = x + (size_t)si*I; float acc=0;
    for (int i=0;i<I;i++){
      float w;
      if (fmt==I8) w=(float)q8[(size_t)o*I+i];
      else if (fmt==I4){ uint8_t b=q4[(size_t)o*rb4+(i>>1)]; int v=(i&1)?(b>>4):(b&0xF); w=(float)(v-8); }
      else if (fmt==I2){ uint8_t b=q4[(size_t)o*rb2+(i>>2)]; int v=(b>>(2*(i&3)))&0x3; w=(float)(v-2); }
      else w=qf[(size_t)o*I+i];
      acc += w*xr[i];
    }
    y[(size_t)si*O+o]=acc*s[o];
  }
}

static int run(int fmt, int O, int I, int S, const char *name) {
  int rb4=(I+1)/2, rb2=(I+3)/4;
  size_t wn = (fmt==I8)?(size_t)O*I : (fmt==I4)?(size_t)O*rb4 : (fmt==I2)?(size_t)O*rb2 : (size_t)O*I*sizeof(float);
  std::vector<uint8_t> W(wn); std::vector<float> Wf;
  srand(99);
  if (fmt==F32){ Wf.resize((size_t)O*I); for(auto&v:Wf) v=((rand()%2000)-1000)/1000.f; }
  else for(auto&b:W) b=(uint8_t)((fmt==I8)?((rand()%255)-127):(rand()&0xFF));
  const void *Wp = (fmt==F32)?(const void*)Wf.data():(const void*)W.data();
  std::vector<float> s(O), x((size_t)S*I), yr((size_t)S*O), yg((size_t)S*O);
  for(auto&v:s) v=(fmt==F32)?1.0f:(0.01f+(rand()%100)/10000.f);
  for(auto&v:x) v=((rand()%2000)-1000)/1000.f;
  cpu_ref(fmt, Wp, s.data(), x.data(), yr.data(), S, I, O);
  IliMetalTensor *t=nullptr;
  if (!ili_metal_matmul(&t, yg.data(), x.data(), Wp, s.data(), fmt, S, I, O)) {
    printf("  %-22s FAIL (matmul returned 0)\n", name); return 1; }
  double maxabs=0, ymax=0;
  for(size_t i=0;i<(size_t)S*O;i++){ maxabs=fmax(maxabs,fabs(yg[i]-yr[i])); ymax=fmax(ymax,fabs(yr[i])); }
  double nerr=maxabs/(ymax+1e-9);
  int ok = nerr < 1e-4;
  printf("  %-22s nerr=%.2e  %s\n", name, nerr, ok?"ok":"*** MISMATCH");
  ili_metal_tensor_free(t);
  return ok?0:1;
}

static float deq4(const uint8_t* w,int i){ uint8_t b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); return (float)(v-8); }
static float deq2(const uint8_t* w,int i){ uint8_t b=w[i>>2]; int v=(b>>(2*(i&3)))&0x3; return (float)(v-2); }
static float deqw(int fmt,const uint8_t* w,int i){ return fmt==3 ? deq2(w,i) : deq4(w,i); }
static int rowbytes(int fmt,int dim){ return fmt==3 ? (dim+3)/4 : (dim+1)/2; }
static size_t roundpg(size_t n){ size_t p=16384; return ((n+p-1)/p)*p; }

// Validate ili_metal_moe_block against a CPU reference (gate/up/silu/down + weighted scatter-add).
// fmt: 2=int4 (2 values/byte), 3=int2 (4 values/byte) -- exercises moe_gemv's vectorized branch
// for whichever format is passed, so the same harness covers both packed widths.
static int run_moe(const std::vector<int>& nrv, const char* name, int fmt=2) {
  const int D=6144, I=2048; int rbG=rowbytes(fmt,D), rbD=rowbytes(fmt,I), nb=(int)nrv.size();
  int R=0; std::vector<int> xoff(nb),nr(nrv); for(int e=0;e<nb;e++){ xoff[e]=R; R+=nrv[e]; }
  srand(2024+nb+fmt);
  // per-expert page-aligned slab [Wg|Wu|Wd] and fslab [Sg|Su|Sd]; register both.
  std::vector<void*> slab(nb), fslab(nb);
  std::vector<const void*> g(nb),u(nb),d(nb); std::vector<const float*> gs(nb),us(nb),ds(nb);
  size_t wlen=roundpg((size_t)I*rbG*2 + (size_t)D*rbD), flen=roundpg(((size_t)I*2+D)*sizeof(float));
  for(int e=0;e<nb;e++){
    posix_memalign(&slab[e],16384,wlen); posix_memalign(&fslab[e],16384,flen);
    uint8_t* sp=(uint8_t*)slab[e]; for(size_t i=0;i<(size_t)I*rbG*2+(size_t)D*rbD;i++) sp[i]=(uint8_t)(rand()&0xFF);
    float* fp=(float*)fslab[e]; for(size_t i=0;i<(size_t)I*2+D;i++) fp[i]=0.01f+(rand()%50)/50000.f;
    g[e]=sp; u[e]=sp+(size_t)I*rbG; d[e]=sp+(size_t)I*rbG*2;
    gs[e]=fp; us[e]=fp+I; ds[e]=fp+2*I;
    ili_metal_register(slab[e],wlen); ili_metal_register(fslab[e],flen);
  }
  std::vector<float> xg((size_t)R*D); for(auto&v:xg) v=((rand()%2000)-1000)/1000.f;
  std::vector<int> rows(R); std::vector<float> rw(R);
  for(int gr=0;gr<R;gr++){ rows[gr]=0; rw[gr]=0.1f+(rand()%100)/100.f; }   // decode: all -> position 0
  int S=1;
  // CPU reference
  std::vector<float> refout((size_t)S*D,0.f), gg(I),uu(I),hh(D);
  for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++){ int gr=xoff[e]+r; const float* xr=&xg[(size_t)gr*D];
    const uint8_t* wg=(const uint8_t*)g[e]; const uint8_t* wu=(const uint8_t*)u[e]; const uint8_t* wd=(const uint8_t*)d[e];
    for(int o=0;o<I;o++){ float a=0; for(int k=0;k<D;k++) a+=deqw(fmt,wg+(size_t)o*rbG,k)*xr[k]; gg[o]=a*gs[e][o]; }
    for(int o=0;o<I;o++){ float a=0; for(int k=0;k<D;k++) a+=deqw(fmt,wu+(size_t)o*rbG,k)*xr[k]; uu[o]=a*us[e][o]; }
    for(int o=0;o<I;o++){ float v=gg[o]; gg[o]=(v/(1.f+expf(-v)))*uu[o]; }
    for(int o=0;o<D;o++){ float a=0; for(int k=0;k<I;k++) a+=deqw(fmt,wd+(size_t)o*rbD,k)*gg[k]; hh[o]=a*ds[e][o]; }
    float* os=&refout[(size_t)rows[gr]*D]; for(int o=0;o<D;o++) os[o]+=rw[gr]*hh[o];
  }
  auto normalized_error=[&](const std::vector<float>& got){
    double maxabs=0,ymax=0;
    for(size_t i=0;i<got.size();i++){
      maxabs=fmax(maxabs,fabs(got[i]-refout[i]));
      ymax=fmax(ymax,fabs(refout[i]));
    }
    return maxabs/(ymax+1e-9);
  };

  std::vector<float> gout((size_t)S*D,0.f);
  int ok = ili_metal_moe_block(nb,D,I,fmt,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
                                xg.data(),xoff.data(),nr.data(),rows.data(),rw.data(),gout.data(),S);
  double nerr=normalized_error(gout); int pass = ok && nerr<1e-4;
  printf("  %-22s sync R=%d nerr=%.2e  %s\n", name, R, nerr, pass?"ok":"*** MISMATCH");

  // Exercise the persistent async slot twice for every shape.  The second begin
  // can succeed only if the first end completed cleanup and released the busy
  // flag; it also reuses the same grow-only scratch buffers and handle storage.
  for(int cycle=0;cycle<2;cycle++){
    std::vector<float> async_out((size_t)S*D,0.f);
    IliMetalMoeHandle *h=ili_metal_moe_block_begin(
        nb,D,I,fmt,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
        xg.data(),xoff.data(),nr.data(),rows.data(),rw.data());
    int async_ok=h && ili_metal_moe_block_end(h,async_out.data());
    double async_nerr=normalized_error(async_out);
    int async_pass=async_ok && async_nerr<1e-4;
    printf("  %-22s async%d R=%d nerr=%.2e  %s\n",name,cycle+1,R,async_nerr,
           async_pass?"ok":"*** MISMATCH");
    pass &= async_pass;
  }
  for(int e=0;e<nb;e++){ ili_metal_unregister(slab[e]); ili_metal_unregister(fslab[e]); free(slab[e]); free(fslab[e]); }
  return pass?0:1;
}

static void setenv01(const char* name, int on){ setenv(name, on?"1":"0", 1); }
static float deq_g8_g4(int fmt, const uint8_t* w, int i) {
  if (fmt==1) return (float)((const int8_t*)w)[i];
  uint8_t b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); return (float)(v-8);
}

// ---- ILI_GPU_REDUCE / ILI_FUSED_GATEUP kernel-agreement, tiny synthetic fixture ----
// Same per-expert reference math as run_moe() above (gate/up dequant+scale -> silu*mul ->
// down dequant+scale -> weighted scatter), checked against ili_metal_moe_block under all
// four combinations of the two opt-in fusions. D/I are deliberately small and, in the second
// call site below, deliberately non-multiples of 8/4 (the SIMD group width / int4 pack width)
// so both the vectorized and scalar-remainder arms of the new kernels get exercised -- no
// disk, no real model weights, everything is synthetic and in-memory.
static int run_fusion_agreement(int fmt, int D, int I, const std::vector<int>& nrv, const char* name) {
  int rbG=(D+1)/2, rbD=(I+1)/2, nb=(int)nrv.size();
  int R=0; std::vector<int> xoff(nb), nr(nrv); for(int e=0;e<nb;e++){ xoff[e]=R; R+=nrv[e]; }
  srand(3141+nb+fmt+D+I);
  std::vector<void*> slab(nb), fslab(nb);
  std::vector<const void*> g(nb),u(nb),d(nb); std::vector<const float*> gs(nb),us(nb),ds(nb);
  size_t glen = fmt==1 ? (size_t)I*D : (size_t)I*rbG;      // one gate/up matrix: [I rows, D cols]
  size_t dlen = fmt==1 ? (size_t)D*I : (size_t)D*rbD;      // down matrix: [D rows, I cols]
  size_t wlen=roundpg(glen*2+dlen), flen=roundpg(((size_t)I*2+D)*sizeof(float));
  for(int e=0;e<nb;e++){
    posix_memalign(&slab[e],16384,wlen); posix_memalign(&fslab[e],16384,flen);
    uint8_t* sp=(uint8_t*)slab[e]; for(size_t i=0;i<glen*2+dlen;i++) sp[i]=(uint8_t)(rand()&0xFF);
    float* fp=(float*)fslab[e]; for(size_t i=0;i<(size_t)I*2+D;i++) fp[i]=0.01f+(rand()%50)/50000.f;
    g[e]=sp; u[e]=sp+glen; d[e]=sp+glen*2;
    gs[e]=fp; us[e]=fp+I; ds[e]=fp+2*I;
    ili_metal_register(slab[e],wlen); ili_metal_register(fslab[e],flen);
  }
  std::vector<float> xg((size_t)R*D); for(auto&v:xg) v=((rand()%2000)-1000)/1000.f;
  std::vector<int> rows(R); std::vector<float> rw(R);
  for(int gr=0;gr<R;gr++){ rows[gr]=0; rw[gr]=0.1f+(rand()%100)/100.f; }   // decode: all -> position 0
  int S=1;
  int rbg = fmt==1 ? D : rbG, rbd = fmt==1 ? I : rbD;
  std::vector<float> refout((size_t)S*D,0.f), gg(I),uu(I),hh(D);
  for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++){ int gr=xoff[e]+r; const float* xr=&xg[(size_t)gr*D];
    const uint8_t* wg=(const uint8_t*)g[e]; const uint8_t* wu=(const uint8_t*)u[e]; const uint8_t* wd=(const uint8_t*)d[e];
    for(int o=0;o<I;o++){ float a=0; for(int k=0;k<D;k++) a+=deq_g8_g4(fmt,wg+(size_t)o*rbg,k)*xr[k]; gg[o]=a*gs[e][o]; }
    for(int o=0;o<I;o++){ float a=0; for(int k=0;k<D;k++) a+=deq_g8_g4(fmt,wu+(size_t)o*rbg,k)*xr[k]; uu[o]=a*us[e][o]; }
    for(int o=0;o<I;o++){ float v=gg[o]; gg[o]=(v/(1.f+expf(-v)))*uu[o]; }
    for(int o=0;o<D;o++){ float a=0; for(int k=0;k<I;k++) a+=deq_g8_g4(fmt,wd+(size_t)o*rbd,k)*gg[k]; hh[o]=a*ds[e][o]; }
    float* os=&refout[(size_t)rows[gr]*D]; for(int o=0;o<D;o++) os[o]+=rw[gr]*hh[o];
  }
  int fail=0;
  for(int fu=0; fu<=1; fu++) for(int rd=0; rd<=1; rd++){
    setenv01("ILI_FUSED_GATEUP", fu); setenv01("ILI_GPU_REDUCE", rd);
    std::vector<float> got((size_t)S*D,0.f);
    int ok = ili_metal_moe_block(nb,D,I,fmt,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
                                  xg.data(),xoff.data(),nr.data(),rows.data(),rw.data(),got.data(),S);
    double maxabs=0,ymax=0;
    for(size_t i=0;i<got.size();i++){ maxabs=fmax(maxabs,fabs(got[i]-refout[i])); ymax=fmax(ymax,fabs(refout[i])); }
    double nerr=maxabs/(ymax+1e-9);
    int pass = ok && nerr<1e-4;
    printf("  %-28s fmt=%d fused=%d reduce=%d R=%-3d nerr=%.2e  %s\n",name,fmt,fu,rd,R,nerr,pass?"ok":"*** MISMATCH");
    fail |= !pass;
  }
  setenv01("ILI_FUSED_GATEUP",0); setenv01("ILI_GPU_REDUCE",0);
  for(int e=0;e<nb;e++){ ili_metal_unregister(slab[e]); ili_metal_unregister(fslab[e]); free(slab[e]); free(fslab[e]); }
  return fail;
}

// fmt=3 (int2) must still refuse (return 0, CPU fallback) with both fusions requested --
// the routed shader has no int2 branch (this is the seam the task explicitly leaves for the
// parallel int2 worktree), and the mixed int4/int2 container hazard fixed in
// run_container_gates.sh (ILI_CPU_MOE_ALL) is exactly what silently accepting fmt=3 here
// would reintroduce.
static int run_fusion_fmt3_refusal(void) {
  const int D=64, I=16, nb=2;
  std::vector<int> nrv={1,1}; std::vector<int> xoff={0,1}, nr=nrv;
  std::vector<uint8_t> junk(64,0);
  const void* g[2]={junk.data(),junk.data()}; const void* u[2]={junk.data(),junk.data()}; const void* d[2]={junk.data(),junk.data()};
  std::vector<float> sc(64,1.0f); const float* gs[2]={sc.data(),sc.data()}; const float* us[2]={sc.data(),sc.data()}; const float* ds[2]={sc.data(),sc.data()};
  std::vector<float> xg(2*D,0.f); std::vector<int> rows={0,0}; std::vector<float> rw={1.f,1.f};
  std::vector<float> out(D,0.f);
  setenv01("ILI_FUSED_GATEUP",1); setenv01("ILI_GPU_REDUCE",1);
  int ok = ili_metal_moe_block(nb,D,I,/*fmt=*/3,g,u,d,gs,us,ds,xg.data(),xoff.data(),nr.data(),
                                rows.data(),rw.data(),out.data(),1);
  setenv01("ILI_FUSED_GATEUP",0); setenv01("ILI_GPU_REDUCE",0);
  int pass = (ok==0);
  printf("  %-28s fmt=3 (int2) with both fusions requested -> %s\n","fmt3 refusal",
         pass?"correctly refused (CPU fallback)":"*** ACCEPTED (would misread int2 as int4!)");
  return pass?0:1;
}

// ---- ILI_GPU_REDUCE / ILI_FUSED_GATEUP microbench: 10k repeated forced-resident block
// calls (all experts pre-registered in memory before the timed loop -- no disk, no misses, in
// the spirit of docs/performance-theory.json entry a1-fused-eight-expert-layer-kernel's own
// falsifier design), reporting exclusive per-call wall time and dispatch/submission counts for
// today's path vs each fusion. These numbers are DIRECTIONAL ONLY: tiny synthetic dims on one
// machine, not the 744B model -- the real verdict is a later, separate run against that model.
static void bench_fusion(int D, int I, int nb, int rows_per_expert, int NITER, const char *scale_label) {
  const int fmt=2;                                // 8 routed experts/token, matching top-8 decode
  int rbG=(D+1)/2, rbD=(I+1)/2;
  std::vector<int> nrv(nb,rows_per_expert), xoff(nb), nr(nrv); int R=0;
  for(int e=0;e<nb;e++){ xoff[e]=R; R+=nrv[e]; }
  srand(999);
  std::vector<void*> slab(nb), fslab(nb);
  std::vector<const void*> g(nb),u(nb),d(nb); std::vector<const float*> gs(nb),us(nb),ds(nb);
  size_t wlen=roundpg((size_t)I*rbG*2+(size_t)D*rbD), flen=roundpg(((size_t)I*2+D)*sizeof(float));
  for(int e=0;e<nb;e++){
    posix_memalign(&slab[e],16384,wlen); posix_memalign(&fslab[e],16384,flen);
    uint8_t* sp=(uint8_t*)slab[e]; for(size_t i=0;i<(size_t)I*rbG*2+(size_t)D*rbD;i++) sp[i]=(uint8_t)(rand()&0xFF);
    float* fp=(float*)fslab[e]; for(size_t i=0;i<(size_t)I*2+D;i++) fp[i]=0.01f+(rand()%50)/50000.f;
    g[e]=sp; u[e]=sp+(size_t)I*rbG; d[e]=sp+(size_t)I*rbG*2;
    gs[e]=fp; us[e]=fp+I; ds[e]=fp+2*I;
    ili_metal_register(slab[e],wlen); ili_metal_register(fslab[e],flen);   // registered ONCE: forced-resident for the whole bench
  }
  std::vector<float> xg((size_t)R*D); for(auto&v:xg) v=((rand()%2000)-1000)/1000.f;
  std::vector<int> rows(R); std::vector<float> rw(R);
  for(int gr=0;gr<R;gr++){ rows[gr]=0; rw[gr]=0.1f+(rand()%100)/100.f; }
  const int S=1;
  int any_fail=0;

  auto run_config=[&](int fu,int rd,const char* label)->double{
    setenv01("ILI_FUSED_GATEUP",fu); setenv01("ILI_GPU_REDUCE",rd);
    std::vector<float> out((size_t)S*D,0.f);
    for(int i=0;i<50;i++){ std::fill(out.begin(),out.end(),0.f);            // warm up scratch-buffer growth
      ili_metal_moe_block(nb,D,I,fmt,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
                            xg.data(),xoff.data(),nr.data(),rows.data(),rw.data(),out.data(),S); }
    uint64_t c0=0,disp0=0,sub0=0; ili_metal_moe_fused_stats(&c0,&disp0,&sub0);
    int ok_all=1;
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<NITER;i++){ std::fill(out.begin(),out.end(),0.f);
      int ok=ili_metal_moe_block(nb,D,I,fmt,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
                                  xg.data(),xoff.data(),nr.data(),rows.data(),rw.data(),out.data(),S);
      ok_all &= ok; }
    auto t1=std::chrono::steady_clock::now();
    uint64_t c1=0,disp1=0,sub1=0; ili_metal_moe_fused_stats(&c1,&disp1,&sub1);
    double us_per = std::chrono::duration<double,std::micro>(t1-t0).count()/NITER;
    if (fu||rd) {
      double dcall=(double)(disp1-disp0)/NITER, scall=(double)(sub1-sub0)/NITER;
      printf("  %-32s %8.2f us/call   dispatches/call=%.2f (measured)   submits/call=%.2f (measured)\n",
             label, us_per, dcall, scall);
    } else {
      printf("  %-32s %8.2f us/call   dispatches/call=4 (from moe_submit source: gate,up,silu,down)   submits/call=1 (from moe_finish source)\n",
             label, us_per);
    }
    if(!ok_all){ printf("    *** at least one call in this configuration returned 0 (unexpected CPU fallback)\n"); any_fail=1; }
    return us_per;
  };

  printf("Metal fused-kernel microbench [%s] (forced-resident, %d iters, nb=%d D=%d I=%d fmt=int4):\n",
         scale_label, NITER, nb, D, I);
  printf("  fixture numbers are DIRECTIONAL ONLY (synthetic weights, one machine) -- not the 744B verdict.\n");
  double base    = run_config(0,0,"current (3-kernel + CPU scatter)");
  double gu_only = run_config(1,0,"ILI_FUSED_GATEUP only");
  double rd_only = run_config(0,1,"ILI_GPU_REDUCE only");
  double both    = run_config(1,1,"both fused");
  printf("  exclusive layer-time delta, both-fused vs current: %+.1f%%  (%.2f -> %.2f us/call)\n",
         100.0*(both-base)/base, base, both);
  printf("  exclusive layer-time delta, fused-gateup only:     %+.1f%%  (%.2f -> %.2f us/call)\n",
         100.0*(gu_only-base)/base, base, gu_only);
  printf("  exclusive layer-time delta, gpu-reduce only:       %+.1f%%  (%.2f -> %.2f us/call)\n",
         100.0*(rd_only-base)/base, base, rd_only);
  if(any_fail) printf("  *** bench had unexpected fallbacks; treat the timings above as unreliable\n");
  setenv01("ILI_FUSED_GATEUP",0); setenv01("ILI_GPU_REDUCE",0);
  for(int e=0;e<nb;e++){ ili_metal_unregister(slab[e]); ili_metal_unregister(fslab[e]); free(slab[e]); free(fslab[e]); }
}

// ---- fused decode attention vs a CPU reference replicating glm.c's exact math ----
// GLM-5.2 dims (hardcoded in the backend): hidden=6144 H=64 q_lora=2048 kv_lora=512
// nope=192 rope=64 vh=256; theta=10000 ascale=1/16 eps=1e-5.
enum { TH=6144, THH=64, TQL=2048, TKVL=512, TNOPE=192, TROPE=64, TVH=256, TQH=256, TROWSH=448 };
static void t_rms(float*o,const float*x,const float*w,int n,float eps){ double ms=0; for(int i=0;i<n;i++) ms+=(double)x[i]*x[i];
  float r=1.f/sqrtf((float)(ms/n)+eps); for(int i=0;i<n;i++) o[i]=x[i]*r*w[i]; }
static void t_rope(float*v,int pos,float th){ int hl=TROPE/2; float in[TROPE]; memcpy(in,v,sizeof(in));
  for(int j=0;j<hl;j++){ float inv=powf(th,-2.f*j/TROPE), a=in[2*j], b=in[2*j+1], cs=cosf(pos*inv), sn=sinf(pos*inv);
    v[j]=a*cs-b*sn; v[hl+j]=b*cs+a*sn; } }
static void t_gemv4(float*y,const float*x,const uint8_t*w,const float*sc,int O,int I){ int rb=(I+1)/2;
  for(int o=0;o<O;o++){ const uint8_t*r=w+(size_t)o*rb; float a=0;
    for(int i=0;i<I;i++){ uint8_t b=r[i>>1]; int v=(i&1)?(b>>4):(b&0xF); a+=(float)(v-8)*x[i]; } y[o]=a*sc[o]; } }
struct TW { uint8_t*w; float*s; size_t wb, sb; };
static TW t_mkw(int O,int I){ TW t; int rb=(I+1)/2;
  t.wb=((size_t)O*rb+16383)&~(size_t)16383; t.sb=((size_t)O*4+16383)&~(size_t)16383;
  posix_memalign((void**)&t.w,16384,t.wb); posix_memalign((void**)&t.s,16384,t.sb);
  for(size_t i=0;i<(size_t)O*rb;i++) t.w[i]=(uint8_t)(rand()&0xFF);
  for(int i=0;i<O;i++) t.s[i]=0.01f+(rand()%40)/40000.f;
  ili_metal_register(t.w,t.wb); ili_metal_register(t.s,t.sb); return t; }
static int run_attn(int S, int pos_base, const char* name){
  const float eps=1e-5f, theta=10000.f, ascale=1.f/16.f;
  srand(4242+S+pos_base);
  TW qa=t_mkw(TQL,TH), qb=t_mkw(THH*TQH,TQL), kva=t_mkw(TKVL+TROPE,TH), kvb=t_mkw(THH*TROWSH,TKVL), o=t_mkw(TH,THH*TVH);
  std::vector<float> qaln(TQL), kvaln(TKVL);
  for(auto&v:qaln) v=0.5f+(rand()%1000)/1000.f; for(auto&v:kvaln) v=0.5f+(rand()%1000)/1000.f;
  int T=pos_base+S; size_t lcb=(((size_t)T*TKVL*4)+16383)&~(size_t)16383, rcb=(((size_t)T*TROPE*4)+16383)&~(size_t)16383;
  float *Lc,*Rc; posix_memalign((void**)&Lc,16384,lcb); posix_memalign((void**)&Rc,16384,rcb);
  ili_metal_register(Lc,lcb); ili_metal_register(Rc,rcb);
  // pre-existing cache history [0,pos_base): random normed latents + roped krot
  for(int t=0;t<pos_base;t++){ for(int i=0;i<TKVL;i++) Lc[(size_t)t*TKVL+i]=((rand()%2000)-1000)/1500.f;
    for(int i=0;i<TROPE;i++) Rc[(size_t)t*TROPE+i]=((rand()%2000)-1000)/1500.f; }
  std::vector<float> x((size_t)S*TH); for(auto&v:x) v=((rand()%2000)-1000)/1000.f;
  std::vector<float> Lr((size_t)T*TKVL), Rr((size_t)T*TROPE);   // reference cache copies
  memcpy(Lr.data(),Lc,(size_t)pos_base*TKVL*4); memcpy(Rr.data(),Rc,(size_t)pos_base*TROPE*4);
  // CPU reference: mirrors glm.c attention() absorb branch (per new token, then per head)
  std::vector<float> Q((size_t)S*THH*TQH), ref((size_t)S*TH);
  for(int s=0;s<S;s++){ int pos=pos_base+s;
    std::vector<float> qr(TQL), comp(TKVL+TROPE);
    t_gemv4(qr.data(),&x[(size_t)s*TH],qa.w,qa.s,TQL,TH); t_rms(qr.data(),qr.data(),qaln.data(),TQL,eps);
    t_gemv4(&Q[(size_t)s*THH*TQH],qr.data(),qb.w,qb.s,THH*TQH,TQL);
    for(int h=0;h<THH;h++) t_rope(&Q[(size_t)s*THH*TQH+(size_t)h*TQH+TNOPE],pos,theta);
    t_gemv4(comp.data(),&x[(size_t)s*TH],kva.w,kva.s,TKVL+TROPE,TH);
    t_rms(&Lr[(size_t)pos*TKVL],comp.data(),kvaln.data(),TKVL,eps);
    memcpy(&Rr[(size_t)pos*TROPE],&comp[TKVL],TROPE*4); t_rope(&Rr[(size_t)pos*TROPE],pos,theta);
  }
  int rb=(TKVL+1)/2;
  for(int s=0;s<S;s++){ int pos=pos_base+s; std::vector<float> ctx((size_t)THH*TVH);
    for(int h=0;h<THH;h++){ int rbase=h*TROWSH;
      const float* qp=&Q[(size_t)s*THH*TQH+(size_t)h*TQH]; const float* qro=qp+TNOPE;
      std::vector<float> qabs(TKVL,0);
      for(int d=0;d<TNOPE;d++){ const uint8_t*r=kvb.w+(size_t)(rbase+d)*rb; float sc=kvb.s[rbase+d];
        for(int i=0;i<TKVL;i++){ uint8_t b=r[i>>1]; int v=(i&1)?(b>>4):(b&0xF); qabs[i]+=qp[d]*(float)(v-8)*sc; } }
      std::vector<float> a(pos+1);
      for(int t=0;t<=pos;t++){ const float*Lt=&Lr[(size_t)t*TKVL]; const float*Rt=&Rr[(size_t)t*TROPE];
        float v=0; for(int i=0;i<TKVL;i++) v+=qabs[i]*Lt[i]; for(int d=0;d<TROPE;d++) v+=qro[d]*Rt[d]; a[t]=v*ascale; }
      float mx=-1e30f; for(float v:a) mx=fmaxf(mx,v); float sum=0; for(float&v:a){ v=expf(v-mx); sum+=v; } for(float&v:a) v/=sum;
      std::vector<float> cl(TKVL,0);
      for(int t=0;t<=pos;t++){ const float*Lt=&Lr[(size_t)t*TKVL]; for(int i=0;i<TKVL;i++) cl[i]+=a[t]*Lt[i]; }
      for(int j=0;j<TVH;j++){ const uint8_t*r=kvb.w+(size_t)(rbase+TNOPE+j)*rb; float sc=kvb.s[rbase+TNOPE+j];
        float v=0; for(int i=0;i<TKVL;i++){ uint8_t b=r[i>>1]; int vv=(i&1)?(b>>4):(b&0xF); v+=cl[i]*(float)(vv-8)*sc; }
        ctx[(size_t)h*TVH+j]=v; } }
    t_gemv4(&ref[(size_t)s*TH],ctx.data(),o.w,o.s,TH,THH*TVH);
  }
  std::vector<float> got((size_t)S*TH);
  int ok=ili_metal_attn_decode(x.data(), qa.w,qa.s,2,qaln.data(), qb.w,qb.s,2,
        kva.w,kva.s,2,kvaln.data(), kvb.w,kvb.s,2, o.w,o.s,2,
        Lc,Rc,S,pos_base,0,eps,theta,ascale,got.data());
  double ma=0,ym=0; for(size_t i=0;i<ref.size();i++){ ma=fmax(ma,fabs(got[i]-ref[i])); ym=fmax(ym,fabs(ref[i])); }
  // also verify the cache write-back (Lc/Rc for the new positions)
  double mc=0; for(int s=0;s<S;s++){ int pos=pos_base+s;
    for(int i=0;i<TKVL;i++) mc=fmax(mc,fabs(Lc[(size_t)pos*TKVL+i]-Lr[(size_t)pos*TKVL+i]));
    for(int i=0;i<TROPE;i++) mc=fmax(mc,fabs(Rc[(size_t)pos*TROPE+i]-Rr[(size_t)pos*TROPE+i])); }
  double nerr=ma/(ym+1e-9);
  int pass = ok && nerr<2e-4 && mc<1e-4;
  printf("  %-24s nerr=%.2e cache=%.2e  %s\n", name, nerr, mc, pass?"ok":"*** MISMATCH");
  auto freew=[&](TW&t){ ili_metal_unregister(t.w); ili_metal_unregister(t.s); free(t.w); free(t.s); };
  freew(qa); freew(qb); freew(kva); freew(kvb); freew(o);
  ili_metal_unregister(Lc); ili_metal_unregister(Rc); free(Lc); free(Rc);
  return pass?0:1;
}

// Repeat the identical fused-attention call `reps` times (same weights, same x, same cache
// history reset before every call) and require every output AND every newly-written cache
// slice to be bit-identical to the first repetition. This is what actually pins down the
// a_rope threadgroup-barrier fix: run_attn()'s numeric-tolerance check would pass even if a
// race only occasionally corrupted a value, since a single run could get lucky. A live race
// would show up here as a nonzero divergence count.
static int run_attn_determinism(int S, int pos_base, int reps, const char* name){
  const float eps=1e-5f, theta=10000.f, ascale=1.f/16.f;
  srand(9001+S+pos_base);
  TW qa=t_mkw(TQL,TH), qb=t_mkw(THH*TQH,TQL), kva=t_mkw(TKVL+TROPE,TH), kvb=t_mkw(THH*TROWSH,TKVL), o=t_mkw(TH,THH*TVH);
  std::vector<float> qaln(TQL), kvaln(TKVL);
  for(auto&v:qaln) v=0.5f+(rand()%1000)/1000.f; for(auto&v:kvaln) v=0.5f+(rand()%1000)/1000.f;
  int T=pos_base+S; size_t lcb=(((size_t)T*TKVL*4)+16383)&~(size_t)16383, rcb=(((size_t)T*TROPE*4)+16383)&~(size_t)16383;
  std::vector<float> Lc0((size_t)pos_base*TKVL), Rc0((size_t)pos_base*TROPE);
  for(int t=0;t<pos_base;t++){ for(int i=0;i<TKVL;i++) Lc0[(size_t)t*TKVL+i]=((rand()%2000)-1000)/1500.f;
    for(int i=0;i<TROPE;i++) Rc0[(size_t)t*TROPE+i]=((rand()%2000)-1000)/1500.f; }
  std::vector<float> x((size_t)S*TH); for(auto&v:x) v=((rand()%2000)-1000)/1000.f;

  float *Lc,*Rc; posix_memalign((void**)&Lc,16384,lcb); posix_memalign((void**)&Rc,16384,rcb);
  ili_metal_register(Lc,lcb); ili_metal_register(Rc,rcb);
  std::vector<float> first_out((size_t)S*TH), first_L((size_t)S*TKVL), first_R((size_t)S*TROPE), got((size_t)S*TH);
  int divergences=0;
  for(int r=0;r<reps;r++){
    memcpy(Lc,Lc0.data(),(size_t)pos_base*TKVL*4); memcpy(Rc,Rc0.data(),(size_t)pos_base*TROPE*4);
    int ok=ili_metal_attn_decode(x.data(), qa.w,qa.s,2,qaln.data(), qb.w,qb.s,2,
          kva.w,kva.s,2,kvaln.data(), kvb.w,kvb.s,2, o.w,o.s,2,
          Lc,Rc,S,pos_base,0,eps,theta,ascale,got.data());
    if(!ok){ printf("  %-24s FAIL (attn_decode returned 0 at rep %d)\n",name,r); return 1; }
    if(r==0){
      first_out=got;
      memcpy(first_L.data(),Lc+(size_t)pos_base*TKVL,(size_t)S*TKVL*4);
      memcpy(first_R.data(),Rc+(size_t)pos_base*TROPE,(size_t)S*TROPE*4);
    } else {
      bool same = memcmp(got.data(),first_out.data(),got.size()*sizeof(float))==0 &&
                  memcmp(Lc+(size_t)pos_base*TKVL,first_L.data(),(size_t)S*TKVL*4)==0 &&
                  memcmp(Rc+(size_t)pos_base*TROPE,first_R.data(),(size_t)S*TROPE*4)==0;
      if(!same) divergences++;
    }
  }
  int pass = divergences==0;
  printf("  %-24s reps=%d divergences=%d  %s\n",name,reps,divergences,pass?"ok":"*** NONDETERMINISTIC");
  auto freew=[&](TW&t){ ili_metal_unregister(t.w); ili_metal_unregister(t.s); free(t.w); free(t.s); };
  freew(qa); freew(qb); freew(kva); freew(kvb); freew(o);
  ili_metal_unregister(Lc); ili_metal_unregister(Rc); free(Lc); free(Rc);
  return pass?0:1;
}

int main(void) {
  if (!ili_metal_init()) { printf("Metal unavailable (skipping)\n"); return 0; }
  printf("Metal backend kernel tests:\n");
  int fail=0;
  fail |= run(I8, 2048,6144,1, "int8 gate/up S=1");
  fail |= run(I4, 2048,6144,1, "int4 gate/up S=1");
  fail |= run(I4, 6144,2048,1, "int4 down S=1");
  fail |= run(I2, 2048,6144,1, "int2 gate/up S=1");
  fail |= run(F32,1024,6144,1, "f32  S=1");
  fail |= run(I8, 2048,6144,4, "int8 gate/up S=4");
  fail |= run(I4, 2048,6144,7, "int4 gate/up S=7 (odd)");
  fail |= run(I4, 2050,6146,3, "int4 non-mult-4 dims");
  printf("Metal batched moe_block tests:\n");
  fail |= run_moe({1,1,1,1,1,1,1,1}, "moe decode nb=8");
  fail |= run_moe({3,1,4,2,1,5},     "moe ragged nb=6");
  fail |= run_moe({1,1,1,1,1,1,1,1}, "moe int2 decode nb=8", 3);
  fail |= run_moe({3,1,4,2,1,5},     "moe int2 ragged nb=6", 3);
  printf("Metal ILI_GPU_REDUCE / ILI_FUSED_GATEUP agreement (tiny synthetic fixture):\n");
  fail |= run_fusion_agreement(1, 128,32, {1,1,1,1,1,1,1,1}, "fused int8 decode nb=8");
  fail |= run_fusion_agreement(2, 128,32, {1,1,1,1,1,1,1,1}, "fused int4 decode nb=8");
  fail |= run_fusion_agreement(2, 128,32, {3,1,4,2,1,5},     "fused int4 ragged nb=6");
  fail |= run_fusion_agreement(2, 130,37, {2,1,3},           "fused int4 non-mult-8 dims");
  fail |= run_fusion_fmt3_refusal();
  printf("Metal large-batch gemm test:\n");
  { // registered int4 weights, S=64: ili_metal_gemm vs cpu_ref
    srand(77); int O=2048,I=6144,S=64,rb=(I+1)/2;
    size_t wb=(((size_t)O*rb)+16383)&~(size_t)16383, sb2=(((size_t)O*4)+16383)&~(size_t)16383;
    uint8_t*W; float*Sc; posix_memalign((void**)&W,16384,wb); posix_memalign((void**)&Sc,16384,sb2);
    for(size_t i=0;i<(size_t)O*rb;i++) W[i]=(uint8_t)(rand()&0xFF);
    for(int i=0;i<O;i++) Sc[i]=0.01f+(rand()%50)/50000.f;
    ili_metal_register(W,wb); ili_metal_register(Sc,sb2);
    std::vector<float> x((size_t)S*I), yr((size_t)S*O), yg((size_t)S*O);
    for(auto&v:x) v=((rand()%2000)-1000)/1000.f;
    cpu_ref(I4,W,Sc,x.data(),yr.data(),S,I,O);
    // Roofline harness counters: ili_metal_gemm_counts must be zero before the first call
    // and reflect exactly one more GPU submission, with a positive kernel time, after it
    // (docs/performance-theory.json p2-m5-gpu-tensor-path-probe -- the "projections" kernel
    // class GPU-time counter this entry's harness needs, added because none existed).
    uint64_t gok0=0; double gw0=0,gk0=0; ili_metal_gemm_counts(&gok0,&gw0,&gk0);
    int ok=ili_metal_gemm(yg.data(),x.data(),W,Sc,2,S,I,O);
    double ma=0,ym=0; for(size_t i=0;i<yr.size();i++){ ma=fmax(ma,fabs(yg[i]-yr[i])); ym=fmax(ym,fabs(yr[i])); }
    int pass = ok && ma/(ym+1e-9)<1e-4;
    printf("  gemm S=64 int4          nerr=%.2e  %s\n", ma/(ym+1e-9), pass?"ok":"*** MISMATCH");
    fail |= !pass;
    uint64_t gok1=0; double gw1=0,gk1=0; ili_metal_gemm_counts(&gok1,&gw1,&gk1);
    int counters_ok = ok && (gok1==gok0+1) && (gw1>gw0) && (gk1>0.0);
    printf("  gemm counters (ok=%llu wall=%.2es kernel=%.2es) %s\n",
           (unsigned long long)gok1, gw1, gk1, counters_ok?"ok":"*** MISMATCH");
    fail |= !counters_ok;
    ili_metal_unregister(W); ili_metal_unregister(Sc); free(W); free(Sc);
  }
  printf("Metal fused attention tests:\n");
  fail |= run_attn(1, 0,   "attn S=1 pos=0");
  fail |= run_attn(1, 37,  "attn S=1 pos=37");
  fail |= run_attn(4, 12,  "attn S=4 pos=12 (MTP)");
  fail |= run_attn(3, 0,   "attn S=3 pos=0");
  printf("Metal RoPE determinism (a_rope threadgroup-barrier fix):\n");
  fail |= run_attn_determinism(1, 0,  1000, "attn S=1 pos=0   x1000");
  fail |= run_attn_determinism(1, 37, 1000, "attn S=1 pos=37  x1000");
  fail |= run_attn_determinism(4, 12, 1000, "attn S=4 pos=12  x1000");
  printf(fail? "metal backend tests: FAILED\n" : "metal backend tests: ok\n");
  // Informational only (directional timings) -- never affects the exit code. Two scales: the
  // "tiny fixture" dims used for agreement above, and GLM-5.2's real D/I (still synthetic
  // random weights, still no model/disk I/O) so a reader can see whether any speedup is a
  // function of problem size before the real, later run against the 744B model.
  bench_fusion(128,32,     8, 1, 10000, "tiny fixture, R=8 (top-8 decode)");
  bench_fusion(6144,2048,  8, 1, 2000,  "real GLM-5.2 D/I, R=8 (top-8 decode), synthetic weights");
  bench_fusion(6144,2048,  64,4, 1000,  "real GLM-5.2 D/I, R=256 (batched/prefill-shaped block), synthetic weights");
  ili_metal_shutdown();
  return fail;
}
