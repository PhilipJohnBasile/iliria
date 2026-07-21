#!/usr/bin/env python3
"""Add an opt-in Metal 4 submission path to the generated M5 Max MoE backend.

This patch deliberately leaves the legacy Metal path intact.  A build must
define ILI_METAL4 and the process must set ILI_METAL4_MOE=1 before the new
path is considered.  Any availability, object-creation, residency, or encoding
setup failure falls back to the existing command-buffer implementation.

The Metal 4 path uses the existing ``moe_gemv`` and ``moe_silu`` pipelines,
dispatch dimensions, dispatch order, and CPU scatter loop.  The only data-path
change is moving the five ``setBytes`` integers for each GEMV into the existing
shared metadata buffer so an MTL4ArgumentTable can bind their GPU addresses.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_text(text: str) -> str:
    text = replace_once(
        text,
        "#include <algorithm>\n",
        """#include <algorithm>

// Keep normal and older-SDK builds free of Metal 4 symbols.  The production
// path additionally requires the runtime ILI_METAL4_MOE=1 opt-in.
#if defined(ILI_METAL4) && ILI_METAL4 && \\
    defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \\
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
#define ILI_METAL4_COMPILED 1
#include <dispatch/dispatch.h>
#else
#define ILI_METAL4_COMPILED 0
#endif
""",
        "Metal 4 compile guard",
    )

    text = replace_once(
        text,
        """static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
""",
        """static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static int g_metal4_requested=0;
#if ILI_METAL4_COMPILED
static int g_metal4_moe=0;
static id<MTL4CommandQueue> g_queue4;
static uint64_t g_metal4_ok=0, g_metal4_legacy_fallback=0, g_metal4_failed=0;
#endif
""",
        "Metal 4 queue state",
    )

    text = replace_once(
        text,
        '''extern "C" void ili_metal_moe_counts(uint64_t *ok, uint64_t *fb, uint64_t *ex) {
  if(ok)*ok=g_moe_ok; if(fb)*fb=g_moe_fb; if(ex)*ex=g_moe_experts;
}
''',
        '''extern "C" void ili_metal_moe_counts(uint64_t *ok, uint64_t *fb, uint64_t *ex) {
  if(ok)*ok=g_moe_ok; if(fb)*fb=g_moe_fb; if(ex)*ex=g_moe_experts;
#if ILI_METAL4_COMPILED
  if(g_metal4_requested) fprintf(stderr,"[metal4] blocks %llu | legacy fallback %llu | failed %llu\\n",
      (unsigned long long)g_metal4_ok,(unsigned long long)g_metal4_legacy_fallback,
      (unsigned long long)g_metal4_failed);
#endif
}
''',
        "Metal 4 execution counters",
    )

    text = replace_once(
        text,
        """struct MoeSlot {
  id<MTLBuffer> xg=nil, gg=nil, uu=nil, hh=nil, meta=nil;
  size_t xg_cap=0, gg_cap=0, uu_cap=0, hh_cap=0, meta_cap=0;
};
""",
        """struct MoeSlot {
  id<MTLBuffer> xg=nil, gg=nil, uu=nil, hh=nil, meta=nil;
  size_t xg_cap=0, gg_cap=0, uu_cap=0, hh_cap=0, meta_cap=0;
#if ILI_METAL4_COMPILED
  id<MTL4CommandAllocator> allocator4=nil;
  id<MTL4CommandBuffer> command_buffer4=nil;
  id<MTL4ArgumentTable> argument_table4=nil;
  id<MTLResidencySet> residency4=nil;
  MTL4CommitOptions *commit_options4=nil;
  MTL4CommitFeedbackHandler __strong feedback_handler4=nil;
  dispatch_semaphore_t feedback_ready4=nullptr;
  __strong NSError *feedback_error4=nil;
  CFTimeInterval gpu_start4=0.0, gpu_end4=0.0;
  bool inflight4=false, disabled4=false;
  bool residency4_set=false;  // persistent residency published once
  bool args4_set=false;       // reserved
  __unsafe_unretained id<MTLBuffer> seen4[390]={}; // buffers already in residency
  int seen4_n=0;
  __unsafe_unretained id<MTLBuffer> seen4_scratch4[5]={}; // track scratch buffer identity
#endif
};
""",
        "Metal 4 per-slot state",
    )

    text = replace_once(
        text,
        """extern "C" int ili_metal_init(void) {
  if (g_dev) return 1;
  g_unretained_cb = ili_env("METAL_UNRETAINED") && atoi(ili_env("METAL_UNRETAINED"));
""",
        """extern "C" int ili_metal_init(void) {
  if (g_dev) return 1;
  g_unretained_cb = ili_env("METAL_UNRETAINED") && atoi(ili_env("METAL_UNRETAINED"));
  g_metal4_requested = ili_env("METAL4_MOE") && atoi(ili_env("METAL4_MOE"));
""",
        "Metal 4 runtime opt-in",
    )

    text = replace_once(
        text,
        """      fprintf(stderr, "[metal] pipeline failed\\n"); g_dev = nil; return 0; }
  }
  return 1;
}

extern "C" void ili_metal_register(void *base, size_t len) {
""",
        """      fprintf(stderr, "[metal] pipeline failed\\n"); g_dev = nil; return 0; }
#if ILI_METAL4_COMPILED
    if(g_metal4_requested){
      if (@available(macOS 26.0, *)) g_queue4=[g_dev newMTL4CommandQueue];
      if(g_queue4){
        g_metal4_moe=1;
        fprintf(stderr,"[metal4] MoE submission enabled (opt-in)\\n");
      } else {
        fprintf(stderr,"[metal4] unavailable; using legacy Metal submission\\n");
      }
    }
#else
    if(g_metal4_requested)
      fprintf(stderr,"[metal4] build lacks ILI_METAL4/macOS 26 SDK support; using legacy Metal submission\\n");
#endif
  }
  return 1;
}

extern "C" void ili_metal_register(void *base, size_t len) {
""",
        "Metal 4 queue initialization",
    )

    text = replace_once(
        text,
        '''extern "C" void ili_metal_shutdown(void) { ili_metal_spin_stop(); g_gemv=nil; g_queue=nil; g_dev=nil; g_tensor_count=g_tensor_bytes=0; }
''',
        '''extern "C" void ili_metal_shutdown(void) {
  ili_metal_spin_stop();
#if ILI_METAL4_COMPILED
  auto clear4=[](MoeSlot *slot){
    slot->allocator4=nil; slot->command_buffer4=nil; slot->argument_table4=nil;
    slot->residency4=nil; slot->commit_options4=nil; slot->feedback_handler4=nil;
    slot->feedback_ready4=nullptr; slot->feedback_error4=nil;
    slot->gpu_start4=slot->gpu_end4=0.0; slot->inflight4=false; slot->disabled4=false;
    slot->residency4_set=false; slot->args4_set=false; slot->seen4_n=0;
    memset(slot->seen4_scratch4,0,sizeof(slot->seen4_scratch4));
  };
  clear4(&g_moe_sync); clear4(&g_moe_async);
  g_moe_async_busy.store(false,std::memory_order_release);
  g_queue4=nil; g_metal4_moe=0; g_metal4_requested=0;
  g_metal4_ok=g_metal4_legacy_fallback=g_metal4_failed=0;
#endif
  g_gemv=nil; g_queue=nil; g_dev=nil; g_tensor_count=g_tensor_bytes=0;
}
''',
        "Metal 4 shutdown cleanup",
    )

    metal4_helpers = r'''
#if ILI_METAL4_COMPILED
// Lazily create one independent Metal 4 command arena for each existing MoE
// scratch slot.  The sync and async slots can be in flight at the same time.
static int metal4_slot_init(MoeSlot *slot) {
  if(!g_metal4_moe || !g_queue4 || !slot || slot->disabled4) return 0;
  if(slot->allocator4 && slot->command_buffer4 && slot->argument_table4 &&
     slot->residency4 && slot->commit_options4 && slot->feedback_ready4 &&
     slot->feedback_handler4) return 1;

  NSError *table_error=nil, *residency_error=nil;
  slot->allocator4=[g_dev newCommandAllocator];
  slot->command_buffer4=[g_dev newCommandBuffer];
  MTL4ArgumentTableDescriptor *td=[MTL4ArgumentTableDescriptor new];
  td.maxBufferBindCount=10; td.initializeBindings=YES;
  slot->argument_table4=[g_dev newArgumentTableWithDescriptor:td error:&table_error];
  MTLResidencySetDescriptor *rd=[MTLResidencySetDescriptor new];
  rd.initialCapacity=395; // at most 390 resolved slabs + five scratch/meta buffers
  slot->residency4=[g_dev newResidencySetWithDescriptor:rd error:&residency_error];
  slot->commit_options4=[MTL4CommitOptions new];
  slot->feedback_ready4=dispatch_semaphore_create(0);
  if(!slot->allocator4 || !slot->command_buffer4 || !slot->argument_table4 ||
     !slot->residency4 || !slot->commit_options4 || !slot->feedback_ready4 ||
     table_error || residency_error){
    NSError *error=table_error?table_error:residency_error;
    fprintf(stderr,"[metal4] slot init failed: %s; falling back to legacy Metal\n",
            error?[[error localizedDescription]UTF8String]:"?");
    slot->allocator4=nil; slot->command_buffer4=nil; slot->argument_table4=nil;
    slot->residency4=nil; slot->commit_options4=nil; slot->feedback_ready4=nullptr;
    slot->disabled4=true;
    return 0;
  }

  // MTL4CommitOptions consumes its handlers on each commit.  Keep one copied
  // block per slot, and add it back immediately before every submission.
  slot->feedback_handler4=^(id<MTL4CommitFeedback> feedback){
    slot->feedback_error4=feedback.error;
    slot->gpu_start4=feedback.GPUStartTime;
    slot->gpu_end4=feedback.GPUEndTime;
    dispatch_semaphore_signal(slot->feedback_ready4);
  };
  if(slot->feedback_handler4) return 1;
  fprintf(stderr,"[metal4] feedback handler creation failed; falling back to legacy Metal\n");
  slot->allocator4=nil; slot->command_buffer4=nil; slot->argument_table4=nil;
  slot->residency4=nil; slot->commit_options4=nil; slot->feedback_ready4=nullptr;
  slot->disabled4=true;
  return 0;
}

// Returns 1 only after a Metal 4 command buffer has been committed, 0 when it
// is safe to retry through legacy Metal, and -1 for an invariant violation
// where touching this slot through the legacy path would race in-flight work.
static int metal4_moe_submit(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr, int R,
                         MoeSlot *slot) {
  if(!g_metal4_moe || !g_dev || !slot ||
     (fmt!=1 && fmt!=2 && fmt!=3) || nb<1 || nb>65) return 0;
  if(slot->inflight4){
    fprintf(stderr,"[metal4] refusing to reuse an in-flight MoE slot\n");
    return -1;
  }
  if(!metal4_slot_init(slot)) return 0;
  double ts_start=mnow();

  slot->xg=ensure(slot->xg,&slot->xg_cap,(size_t)R*D*4);
  slot->gg=ensure(slot->gg,&slot->gg_cap,(size_t)R*Iinter*4);
  slot->uu=ensure(slot->uu,&slot->uu_cap,(size_t)R*Iinter*4);
  slot->hh=ensure(slot->hh,&slot->hh_cap,(size_t)R*D*4);
  if(!slot->xg || !slot->gg || !slot->uu || !slot->hh) return 0;

  uint64_t ag[65],au[65],ad[65],sgv[65],suv[65],sdv[65];
  id<MTLBuffer> use[390]; int nuse=0;
  auto add_use=[&](id<MTLBuffer> b){
    for(int i=0;i<nuse;i++) if(use[i]==b) return;
    use[nuse++]=b;
  };
  for(int ex=0;ex<nb;ex++){
    id<MTLBuffer> b;
    if(!(b=resolve(g[ex],&ag[ex]))) return 0; add_use(b);
    if(!(b=resolve(u[ex],&au[ex]))) return 0; add_use(b);
    if(!(b=resolve(d[ex],&ad[ex]))) return 0; add_use(b);
    if(!(b=resolve(gs[ex],&sgv[ex]))) return 0; add_use(b);
    if(!(b=resolve(us[ex],&suv[ex]))) return 0; add_use(b);
    if(!(b=resolve(ds[ex],&sdv[ex]))) return 0; add_use(b);
  }

  auto A=[](size_t v){ return (v+255)&~(size_t)255; };
  size_t oag=0, oau=A(oag+(size_t)nb*8), oad=A(oau+(size_t)nb*8);
  size_t osg=A(oad+(size_t)nb*8), osu=A(osg+(size_t)nb*8), osd=A(osu+(size_t)nb*8);
  size_t oerow=A(osd+(size_t)nb*8);
  size_t oc_gate=A(oerow+(size_t)R*4), oc_up=A(oc_gate+5*4);
  size_t oc_down=A(oc_up+5*4), meta_need=A(oc_down+5*4);
  slot->meta=ensure(slot->meta,&slot->meta_cap,meta_need);
  if(!slot->meta) return 0;
  char *mp=(char*)[slot->meta contents];
  int *erow=(int*)(mp+oerow);
  for(int ex=0;ex<nb;ex++) for(int r=0;r<nr[ex];r++) erow[xoff[ex]+r]=ex;
  memcpy(mp+oag,ag,(size_t)nb*8); memcpy(mp+oau,au,(size_t)nb*8); memcpy(mp+oad,ad,(size_t)nb*8);
  memcpy(mp+osg,sgv,(size_t)nb*8); memcpy(mp+osu,suv,(size_t)nb*8); memcpy(mp+osd,sdv,(size_t)nb*8);
  int gate_args[5]={Iinter,D,D,fmt,R*Iinter};
  int up_args[5]={Iinter,D,D,fmt,R*Iinter};
  int down_args[5]={D,Iinter,Iinter,fmt,R*D};
  memcpy(mp+oc_gate,gate_args,sizeof(gate_args));
  memcpy(mp+oc_up,up_args,sizeof(up_args));
  memcpy(mp+oc_down,down_args,sizeof(down_args));
  memcpy([slot->xg contents],xg,(size_t)R*D*4);

  // Persistent residency: publish once on first submission, add only new buffers afterwards.
  // Avoids the ~8s per-block removeAll/addAll/commit churn that dominated setup time.
  int added_new4=false;
  if(!slot->residency4_set){
    slot->residency4_set=true;
    __unsafe_unretained id<MTLAllocation> allocations[395]; int nalloc=0;
    for(int i=0;i<nuse;i++) allocations[nalloc++]=(id<MTLAllocation>)use[i];
    allocations[nalloc++]=(id<MTLAllocation>)slot->xg;
    allocations[nalloc++]=(id<MTLAllocation>)slot->gg;
    allocations[nalloc++]=(id<MTLAllocation>)slot->uu;
    allocations[nalloc++]=(id<MTLAllocation>)slot->hh;
    allocations[nalloc++]=(id<MTLAllocation>)slot->meta;
    [slot->residency4 addAllocations:allocations count:(NSUInteger)nalloc];
    for(int i=0;i<nuse;i++) slot->seen4[slot->seen4_n++]=use[i];
    [slot->residency4 commit];
  } else {
    for(int i=0;i<nuse;i++){
      bool found=false;
      for(int j=0;j<slot->seen4_n;j++) if(use[i]==slot->seen4[j]){found=true;break;}
      if(!found){
        id<MTLAllocation> alloc=(id<MTLAllocation>)use[i];
        [slot->residency4 addAllocations:&alloc count:1];
        if(slot->seen4_n<390) slot->seen4[slot->seen4_n++]=use[i];
        added_new4=true;
      }
    }
    // Check if scratch buffers changed (ensure may have reallocated between tests)
    id<MTLBuffer> scratch4[]={slot->xg,slot->gg,slot->uu,slot->hh,slot->meta};
    for(int i=0;i<5;i++){
      if(!slot->seen4_scratch4[i] || slot->seen4_scratch4[i]!=scratch4[i]){
        id<MTLAllocation> alloc=(id<MTLAllocation>)scratch4[i];
        [slot->residency4 addAllocations:&alloc count:1];
        slot->seen4_scratch4[i]=scratch4[i];
        added_new4=true;
      }
    }
    if(added_new4) [slot->residency4 commit];
  }

  id<MTL4CommandBuffer> cb=slot->command_buffer4;
  [cb beginCommandBufferWithAllocator:slot->allocator4];
  [cb useResidencySet:slot->residency4];
  id<MTL4ComputeCommandEncoder> e=[cb computeCommandEncoder];
  if(!e){
    [cb endCommandBuffer]; [slot->allocator4 reset];
    return 0;
  }
  MTLGPUAddress ma=slot->meta.gpuAddress;

  auto gemv=[&](size_t wo,size_t so,id<MTLBuffer> xin,id<MTLBuffer> y,
                size_t constants_offset,int NT){
    id<MTL4ArgumentTable> table=slot->argument_table4;
    [e setComputePipelineState:g_moe_gemv];
    [e setArgumentTable:table];
    [table setAddress:ma+wo atIndex:0]; [table setAddress:ma+so atIndex:1];
    [table setAddress:ma+oerow atIndex:2]; [table setAddress:xin.gpuAddress atIndex:3];
    [table setAddress:y.gpuAddress atIndex:4];
    for(NSUInteger i=0;i<5;i++) [table setAddress:ma+constants_offset+i*4 atIndex:5+i];
    [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1)
         threadsPerThreadgroup:MTLSizeMake(128,1,1)];
  };
  gemv(oag,osg,slot->xg,slot->gg,oc_gate,R*Iinter);            // gate
  gemv(oau,osu,slot->xg,slot->uu,oc_up,R*Iinter);              // up
  [e barrierAfterEncoderStages:MTLStageDispatch
           beforeEncoderStages:MTLStageDispatch
             visibilityOptions:MTL4VisibilityOptionDevice];
  {
    id<MTL4ArgumentTable> table=slot->argument_table4;
    [e setComputePipelineState:g_moe_silu];
    [e setArgumentTable:table];
    [table setAddress:slot->gg.gpuAddress atIndex:0];
    [table setAddress:slot->uu.gpuAddress atIndex:1];
  }
  [e dispatchThreads:MTLSizeMake((size_t)R*Iinter,1,1)
       threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e barrierAfterEncoderStages:MTLStageDispatch
           beforeEncoderStages:MTLStageDispatch
             visibilityOptions:MTL4VisibilityOptionDevice];
  gemv(oad,osd,slot->gg,slot->hh,oc_down,R*D);                 // down
  g_t_setup+=mnow()-ts_start;
  [e endEncoding]; [cb endCommandBuffer];

  slot->feedback_error4=nil; slot->gpu_start4=0.0; slot->gpu_end4=0.0;
  [slot->commit_options4 addFeedbackHandler:slot->feedback_handler4];
  slot->inflight4=true;
  [g_queue4 commit:&cb count:1 options:slot->commit_options4];
  return 1;
}

// Wait, preserve the legacy error/fallback contract, and execute the exact
// existing CPU scatter-add after Metal reports completion.
static int metal4_moe_finish(MoeSlot *slot, int nb, int R, int D,
                             const int *rows, const float *rw, float *out) {
  if(!slot || !slot->inflight4) return 0;
  double t0=mnow();
  dispatch_semaphore_wait(slot->feedback_ready4,DISPATCH_TIME_FOREVER);
  double ts_gpu=mnow(); g_t_gpu+=ts_gpu-t0;
  if(slot->gpu_end4>=slot->gpu_start4) g_t_kernel+=slot->gpu_end4-slot->gpu_start4;
  NSError *error=slot->feedback_error4;
  slot->feedback_error4=nil;
  slot->inflight4=false;
  [slot->allocator4 reset];
  if(error){
    fprintf(stderr,"[metal4] moe_block error (nb=%d R=%d): %s\n",nb,R,
            [[error localizedDescription]UTF8String]);
    g_metal4_failed++; g_moe_fb++; return 0;
  }
  const float *hh=(const float*)[slot->hh contents];
  for(int gr=0;gr<R;gr++){ float *os=out+(size_t)rows[gr]*D, w=rw[gr]; const float *hr=hh+(size_t)gr*D;
    for(int dd=0;dd<D;dd++) os[dd]+=w*hr[dd]; }
  g_t_scatter+=mnow()-ts_gpu;
  g_metal4_ok++; g_moe_ok++; g_moe_experts+=nb;
  return 1;
}
#endif

'''
    sync_marker = '''extern "C" int ili_metal_moe_block(int nb, int D, int Iinter, int fmt,
'''
    text = replace_once(
        text,
        sync_marker,
        metal4_helpers + sync_marker,
        "Metal 4 MoE helpers",
    )

    text = replace_once(
        text,
        """    std::lock_guard<std::mutex> lk(g_moe_sync_mtx);
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,&g_moe_sync);
    if (!cb) return 0;
    return moe_finish(cb,g_moe_sync.hh,nb,R,D,rows,rw,out);
""",
        """    std::lock_guard<std::mutex> lk(g_moe_sync_mtx);
#if ILI_METAL4_COMPILED
    int submitted4=g_metal4_moe?metal4_moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,&g_moe_sync):0;
    if(submitted4>0)
      return metal4_moe_finish(&g_moe_sync,nb,R,D,rows,rw,out);
    if(submitted4<0) return 0;
    if(g_metal4_moe) g_metal4_legacy_fallback++;
#endif
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,&g_moe_sync);
    if (!cb) return 0;
    return moe_finish(cb,g_moe_sync.hh,nb,R,D,rows,rw,out);
""",
        "synchronous Metal 4 dispatch",
    )

    text = replace_once(
        text,
        """struct IliMetalMoeHandle {
  id<MTLCommandBuffer> cb; id<MTLBuffer> hh;
  int *rows=nullptr; float *rwv=nullptr; size_t cap=0;
  int nb=0, R=0, D=0;
};
""",
        """struct IliMetalMoeHandle {
  id<MTLCommandBuffer> cb; id<MTLBuffer> hh;
  int *rows=nullptr; float *rwv=nullptr; size_t cap=0;
  int nb=0, R=0, D=0;
#if ILI_METAL4_COMPILED
  bool metal4=false;
#endif
};
""",
        "asynchronous Metal 4 handle flag",
    )

    text = replace_once(
        text,
        """    if(g_moe_async_busy.exchange(true,std::memory_order_acq_rel)) return nullptr;
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,&g_moe_async);
    if (!cb) { g_moe_async_busy.store(false,std::memory_order_release); return nullptr; }
    async_handle_ensure((size_t)R);
    IliMetalMoeHandle *h=&g_async_handle;
    h->cb=cb; h->hh=g_moe_async.hh;
""",
        """    if(g_moe_async_busy.exchange(true,std::memory_order_acq_rel)) return nullptr;
    id<MTLCommandBuffer> cb=nil;
#if ILI_METAL4_COMPILED
    int submit_result4=g_metal4_moe?metal4_moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,&g_moe_async):0;
    if(submit_result4<0) { g_moe_async_busy.store(false,std::memory_order_release); return nullptr; }
    bool submitted4=submit_result4>0;
    if(!submitted4){ if(g_metal4_moe) g_metal4_legacy_fallback++;
#endif
      cb=moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,&g_moe_async);
#if ILI_METAL4_COMPILED
    }
#endif
#if ILI_METAL4_COMPILED
    if(!submitted4 && !cb) { g_moe_async_busy.store(false,std::memory_order_release); return nullptr; }
#else
    if(!cb) { g_moe_async_busy.store(false,std::memory_order_release); return nullptr; }
#endif
    async_handle_ensure((size_t)R);
    IliMetalMoeHandle *h=&g_async_handle;
    h->cb=cb; h->hh=g_moe_async.hh;
#if ILI_METAL4_COMPILED
    h->metal4=submitted4;
#endif
""",
        "asynchronous Metal 4 submission",
    )

    text = replace_once(
        text,
        """  int ok;
  @autoreleasepool { ok = moe_finish(h->cb,h->hh,h->nb,h->R,h->D,h->rows,h->rwv,out); }
  h->cb=nil; h->hh=nil;
""",
        """  int ok;
  @autoreleasepool {
#if ILI_METAL4_COMPILED
    if(h->metal4) ok=metal4_moe_finish(&g_moe_async,h->nb,h->R,h->D,h->rows,h->rwv,out);
    else
#endif
      ok=moe_finish(h->cb,h->hh,h->nb,h->R,h->D,h->rows,h->rwv,out);
  }
  h->cb=nil; h->hh=nil;
#if ILI_METAL4_COMPILED
  h->metal4=false;
#endif
""",
        "asynchronous Metal 4 completion",
    )

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(patch_text(args.source.read_text()))


if __name__ == "__main__":
    main()
