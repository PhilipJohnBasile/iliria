// Numerics validation for the MLA-native tiled/online-softmax prefill kernel
// (ILI_ONLINE_ATTN=1; docs/performance-theory.json a4-mla-native-tiled-dense-attention).
//
// Compares, on tiny SYNTHETIC fixtures (no real model, no /path/to/models reads):
//   (1) a CPU reference that mirrors a_qabs -> a_score -> a_smax -> a_clat -> a_ctx exactly
//       (the same math backend_metal.mm's existing chunked kernel chain implements);
//   (2) ili_metal_attn_prefill with ILI_ONLINE_ATTN unset (today's default, chunked path);
//   (3) ili_metal_attn_prefill with ILI_ONLINE_ATTN=1 (the new tiled/online-softmax path).
// Metrics follow c/tools/compare_layer_captures.py's conventions exactly: max_abs (worst
// position, i.e. worst query row here), rms/nrms/cosine (mean over positions, nrms
// normalized by the CPU reference's own RMS), plus the flattened cosine. Fixtures cross
// the kernel's 256-token tile boundary (single tile, straddling one boundary, multi-tile
// with a partial last tile) since that recurrence is exactly what's under test.
// ili_metal_attn_prefill re-reads its env gate on every call (see backend_metal.mm), so
// both paths run in one process without a restart.
#include "../backend_metal.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

enum { H=64, QH=256, NOPE=192, ROPE=64, KVL=512, VH=256, ROWSH=448 };

static float deq4(const uint8_t* base, int row, int i, const float* scale){
  const uint8_t* w = base + (size_t)row*((KVL+1)/2);
  uint8_t b = w[i>>1]; int v = (i&1) ? (b>>4) : (b&0xF);
  return float(v-8)*scale[row];
}

struct KVB { uint8_t *w; float *s; size_t wbytes, sbytes; };
static size_t roundpg(size_t n){ size_t p=16384; return (n+p-1)/p*p; }
static KVB make_kvb(unsigned seed){
  KVB k; k.wbytes=roundpg((size_t)H*ROWSH*((KVL+1)/2)); k.sbytes=roundpg((size_t)H*ROWSH*4);
  posix_memalign((void**)&k.w,16384,k.wbytes); posix_memalign((void**)&k.s,16384,k.sbytes);
  srand(seed);
  for(size_t i=0;i<(size_t)H*ROWSH*((KVL+1)/2);i++) k.w[i]=(uint8_t)(rand()&0xFF);
  for(int i=0;i<H*ROWSH;i++) k.s[i]=0.01f+(rand()%40)/40000.f;
  ili_metal_register(k.w,k.wbytes); ili_metal_register(k.s,k.sbytes);
  return k;
}
static void free_kvb(KVB&k){ ili_metal_unregister(k.w); ili_metal_unregister(k.s); free(k.w); free(k.s); }

// CPU reference: a_qabs -> a_score -> a_smax -> a_clat -> a_ctx, the same arithmetic order
// as backend_metal.mm's chunked kernel chain (see those kernels' comments for the shapes).
static void cpu_ref(const KVB&kvb, const std::vector<float>&Q, const std::vector<float>&Lc,
                    const std::vector<float>&Rc, int S, int pos_base, float ascale,
                    std::vector<float>&ctx){
  ctx.assign((size_t)S*H*VH,0.f);
  for(int s=0;s<S;s++){
    int pos=pos_base+s;
    const float* qrow=&Q[(size_t)s*H*QH];
    for(int h=0;h<H;h++){
      const float* qp=qrow+(size_t)h*QH; const float* qr=qp+NOPE;
      float qabs[KVL]; memset(qabs,0,sizeof(qabs));
      for(int d=0;d<NOPE;d++){ int row=h*ROWSH+d; float qd=qp[d];
        for(int i=0;i<KVL;i++) qabs[i]+=qd*deq4(kvb.w,row,i,kvb.s); }
      std::vector<float> sc(pos+1);
      for(int t=0;t<=pos;t++){
        const float* Lt=&Lc[(size_t)t*KVL]; const float* Rt=&Rc[(size_t)t*ROPE];
        float a=0; for(int i=0;i<KVL;i++) a+=qabs[i]*Lt[i]; for(int d=0;d<ROPE;d++) a+=qr[d]*Rt[d];
        sc[t]=a*ascale;
      }
      float mx=-1e30f; for(float v : sc) mx=std::fmax(mx,v);
      float sum=0; for(float&v : sc){ v=std::exp(v-mx); sum+=v; } for(float&v : sc) v/=sum;
      float clat[KVL]; memset(clat,0,sizeof(clat));
      for(int t=0;t<=pos;t++){ const float* Lt=&Lc[(size_t)t*KVL]; float w=sc[t]; for(int i=0;i<KVL;i++) clat[i]+=w*Lt[i]; }
      float* co=&ctx[((size_t)s*H+h)*VH];
      for(int j=0;j<VH;j++){ int row=h*ROWSH+NOPE+j; float a=0; for(int i=0;i<KVL;i++) a+=clat[i]*deq4(kvb.w,row,i,kvb.s); co[j]=a; }
    }
  }
}

// ---- metrics, following c/tools/compare_layer_captures.py's conventions exactly:
// max_abs = worst row's max(|a-b|); rms/nrms/cosine = mean over rows (nrms normalized by
// the REFERENCE row's own RMS); cosine_flat = cosine over the whole flattened vector. ----
struct Metrics { double max_abs, rms_mean, nrms_mean, cosine_mean, cosine_flat; };
static Metrics compare_metrics(const std::vector<float>&ref, const std::vector<float>&cmp, int S, int D){
  Metrics m{0,0,0,0,0};
  double sum_rms=0, sum_nrms=0, sum_cos=0;
  for(int s=0;s<S;s++){
    const float* a=&ref[(size_t)s*D]; const float* b=&cmp[(size_t)s*D];
    double max_abs=0, sq=0, refsq=0, dot=0, na=0, nb=0;
    for(int i=0;i<D;i++){ double d=(double)a[i]-b[i]; max_abs=std::fmax(max_abs,std::fabs(d));
      sq+=d*d; refsq+=(double)a[i]*a[i]; dot+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i]; }
    double rms=std::sqrt(sq/D), refrms=std::sqrt(refsq/D);
    double nrms = refrms>0 ? rms/refrms : (rms==0 ? 0.0 : HUGE_VAL);
    double cosine = (na>0 && nb>0) ? dot/(std::sqrt(na)*std::sqrt(nb)) : NAN;
    m.max_abs=std::fmax(m.max_abs,max_abs); sum_rms+=rms; sum_nrms+=nrms; sum_cos+=cosine;
  }
  m.rms_mean=sum_rms/S; m.nrms_mean=sum_nrms/S; m.cosine_mean=sum_cos/S;
  double dot=0,na=0,nb=0;
  for(size_t i=0;i<ref.size();i++){ dot+=(double)ref[i]*cmp[i]; na+=(double)ref[i]*ref[i]; nb+=(double)cmp[i]*cmp[i]; }
  m.cosine_flat = (na>0 && nb>0) ? dot/(std::sqrt(na)*std::sqrt(nb)) : NAN;
  return m;
}

static int run_case(int S, int pos_base, const char* name){
  unsigned seed = 1000u+(unsigned)S*7u+(unsigned)pos_base;
  int T = pos_base+S;
  KVB kvb = make_kvb(seed);
  std::vector<float> Q((size_t)S*H*QH), Lc((size_t)T*KVL), Rc((size_t)T*ROPE);
  srand(seed+1);
  auto randf=[&]{ return ((rand()%2000)-1000)/1000.f; };
  for(auto&v:Q) v=randf(); for(auto&v:Lc) v=randf(); for(auto&v:Rc) v=randf();
  float ascale = 1.f/16.f;

  // page-align + register Lc/Rc (ili_metal_attn_prefill resolves pointers inside
  // registered slabs, exactly like the real engine's Lc[layer]/Rc[layer] caches).
  size_t lcb=roundpg((size_t)T*KVL*4), rcb=roundpg((size_t)T*ROPE*4);
  float *Lcp,*Rcp; posix_memalign((void**)&Lcp,16384,lcb); posix_memalign((void**)&Rcp,16384,rcb);
  memcpy(Lcp,Lc.data(),(size_t)T*KVL*4); memcpy(Rcp,Rc.data(),(size_t)T*ROPE*4);
  ili_metal_register(Lcp,lcb); ili_metal_register(Rcp,rcb);

  std::vector<float> ref, chunked((size_t)S*H*VH,0.f), online((size_t)S*H*VH,0.f);
  cpu_ref(kvb, Q, Lc, Rc, S, pos_base, ascale, ref);

  setenv("ILI_ONLINE_ATTN","0",1);
  int ok_chunked = ili_metal_attn_prefill(Q.data(), kvb.w, kvb.s, 2, Lcp, Rcp, S, pos_base, ascale, chunked.data());
  setenv("ILI_ONLINE_ATTN","1",1);
  int ok_online  = ili_metal_attn_prefill(Q.data(), kvb.w, kvb.s, 2, Lcp, Rcp, S, pos_base, ascale, online.data());
  setenv("ILI_ONLINE_ATTN","0",1);

  int D=H*VH;
  Metrics mc  = compare_metrics(ref, chunked, S, D);
  Metrics mo  = compare_metrics(ref, online,  S, D);
  Metrics moc = compare_metrics(chunked, online, S, D);

  printf("  %-40s S=%-3d T=%-5d chunked_ok=%d online_ok=%d\n", name, S, T, ok_chunked, ok_online);
  printf("    vs CPU  chunked:      max_abs=%.3e nrms=%.3e cosine=%.9f cosine_flat=%.9f\n",
         mc.max_abs, mc.nrms_mean, mc.cosine_mean, mc.cosine_flat);
  printf("    vs CPU  online:       max_abs=%.3e nrms=%.3e cosine=%.9f cosine_flat=%.9f\n",
         mo.max_abs, mo.nrms_mean, mo.cosine_mean, mo.cosine_flat);
  printf("    online  vs chunked:   max_abs=%.3e nrms=%.3e cosine=%.9f cosine_flat=%.9f\n",
         moc.max_abs, moc.nrms_mean, moc.cosine_mean, moc.cosine_flat);

  int pass = ok_chunked && ok_online
           && mc.nrms_mean<1e-4 && mo.nrms_mean<1e-4
           && mc.cosine_mean>0.999999 && mo.cosine_mean>0.999999;
  printf("    %s\n", pass?"ok":"*** MISMATCH");

  ili_metal_unregister(Lcp); ili_metal_unregister(Rcp); free(Lcp); free(Rcp);
  free_kvb(kvb);
  return pass?0:1;
}

int main(void){
  if(!ili_metal_init()){ printf("Metal unavailable (skipping)\n"); return 0; }
  printf("Online-softmax attention (ILI_ONLINE_ATTN=1) numerics validation:\n");
  int fail=0;
  fail |= run_case(3,    0,    "S=3  pos_base=0     (single tile)");
  fail |= run_case(4,    250,  "S=4  pos_base=250   (crosses 1 tile boundary)");
  fail |= run_case(6,    600,  "S=6  pos_base=600   (multi-tile, partial last)");
  fail |= run_case(2,    1022, "S=2  pos_base=1022  (straddles a 4-tile boundary)");
  fail |= run_case(8,    4090, "S=8  pos_base=4090  (~16 tiles, partial last)");
  printf(fail? "online-attn numerics: FAILED\n" : "online-attn numerics: ok\n");
  ili_metal_shutdown();
  return fail;
}
