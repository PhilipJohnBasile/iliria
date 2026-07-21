#!/usr/bin/env python3
"""Add an opt-in, lifecycle-safe global persistent Metal state prototype.

This transformation runs after patch_m5max_metal4.py.  With
ILI_METAL_PERSISTENT_STATE=1:

1.  Every slab registration receives a monotonically increasing generation and a
    heap `SlabLife` node that owns one extra strong reference to the slab's
    no-copy MTLBuffer.
2.  Before a MoE command buffer is committed (legacy Metal and Metal 4 paths),
    one in-flight reference is taken on every distinct non-pinned slab it reads;
    a completion handler on that command buffer releases the references.
3.  `ili_metal_unregister` on a slab with live in-flight references defers the
    actual release: the node moves to a bounded retirement queue drained by
    completion handlers, and unregister blocks until the references reach zero
    (bounded, because completed handlers always fire for committed command
    buffers).  The engine frees or reuses the host backing immediately after
    unregister returns, so returning early would be a use-after-free.
4.  `ili_metal_register_pinned` marks a slab as process-lifetime pinned: it
    skips per-command-buffer refcounting (immutable fast path) and its wrap is
    only dropped at shutdown.
5.  Registered expert slabs are published into one generation-tagged global
    MTLResidencySet (Metal 4 builds).  Old snapshots retire into a bounded list
    swept on command-buffer completion; when the list is at capacity the
    refresh fails cleanly into the per-slot residency fallback.
6.  Any failure (allocation, API error, slab vanished between resolve and
    commit) falls back to the existing non-persistent path.

The default path (ILI_METAL_PERSISTENT_STATE unset) is behavior-identical, and
the production `make mac-fast` build never applies this transform at all.
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
    # ---- includes -------------------------------------------------------
    text = replace_once(
        text,
        "#include <atomic>\n#include <algorithm>\n",
        "#include <atomic>\n#include <algorithm>\n"
        "#include <condition_variable>\n#include <chrono>\n#include <new>\n",
        "lifecycle includes",
    )

    # ---- globals --------------------------------------------------------
    # The lifecycle opt-in flag lives OUTSIDE the Metal 4 compile guard so the
    # slab refcounting is exercised on legacy-Metal lab builds too.
    text = replace_once(
        text,
        """static int g_metal4_requested=0;
#if ILI_METAL4_COMPILED
""",
        """static int g_metal4_requested=0;
static int g_metal_persistent_state=0;
#if ILI_METAL4_COMPILED
""",
        "persistent Metal opt-in flag",
    )

    text = replace_once(
        text,
        """static int g_metal4_moe=0;
static id<MTL4CommandQueue> g_queue4;
""",
        """static int g_metal4_moe=0;
static id<MTL4CommandQueue> g_queue4;
static id<MTLResidencySet> g_global_residency4=nil;
static uint64_t g_global_residency_gen4=0, g_global_residency_refresh4=0;
static uint64_t g_global_residency_allocs4=0;
static double g_global_residency_refresh_sec4=0.0;
static NSMutableArray *g_retired_global_residency4=nil;
""",
        "persistent Metal globals",
    )

    text = replace_once(
        text,
        """  g_metal4_requested = ili_env("METAL4_MOE") && atoi(ili_env("METAL4_MOE"));
""",
        """  g_metal4_requested = ili_env("METAL4_MOE") && atoi(ili_env("METAL4_MOE"));
  /* ILI_METAL_PERSISTENT_STATE=1 (legacy alias: COLI_METAL_PERSISTENT_STATE) */
  g_metal_persistent_state = ili_env("METAL_PERSISTENT_STATE") &&
      atoi(ili_env("METAL_PERSISTENT_STATE"));
  if(g_metal_persistent_state)
    fprintf(stderr,"[metal] persistent-state slab lifecycle enabled (opt-in)\\n");
""",
        "persistent Metal runtime opt-in",
    )

    text = replace_once(
        text,
        """        fprintf(stderr,"[metal4] MoE submission enabled (opt-in)\\n");
""",
        """        fprintf(stderr,"[metal4] MoE submission enabled (opt-in)\\n");
        if(g_metal_persistent_state)
          fprintf(stderr,"[metal4] global persistent residency prototype enabled (opt-in)\\n");
""",
        "persistent Metal startup status",
    )

    # ---- slab lifecycle: struct, registry, refcounting -------------------
    text = replace_once(
        text,
        "struct Slab { void *base; size_t len; id<MTLBuffer> buf; };\n",
        """// Lifecycle node for one slab registration generation.  Allocated only when
// ILI_METAL_PERSISTENT_STATE=1.  Every in-flight command buffer that reads the
// slab holds one reference; the extra strong `buf` keeps the no-copy MTLBuffer
// alive until the last reference is dropped by a completion handler.  All
// fields are guarded by g_slab_mtx (refs included: completion handlers take it).
struct SlabLife {
  id<MTLBuffer> buf=nil;
  uint64_t gen=0;
  uint32_t refs=0;
  bool pinned=false;    // process-lifetime: immutable fast path, never refcounted
  bool zombie=false;    // unregistered while still referenced
  bool waited=false;    // a blocked unregister owns this node's destruction
};
struct Slab { void *base; size_t len; id<MTLBuffer> buf; SlabLife *life; uint64_t gen; };
""",
        "slab lifecycle struct",
    )

    lifecycle_helpers = r'''
// ---- persistent-state slab lifecycle (ILI_METAL_PERSISTENT_STATE=1) ----
enum { ILI_SLAB_RETIRE_MAX=256, ILI_SNAP_RETIRE_MAX=16 };
static std::condition_variable g_slab_cv;
static std::vector<SlabLife*> g_retired_slab_lives;   // zombies awaiting refs==0
static uint64_t g_slab_refs_acquired=0, g_slab_deferred_unregisters=0, g_slab_unregister_waits=0;
static bool g_slab_untracked_warned=false;

// Release the per-command-buffer references taken by slab_refs_acquire.  Runs
// inside Metal completion handlers; zombie nodes are destroyed here unless a
// blocked unregister owns their destruction.
static void slab_refs_release_all(SlabLife *const *arr, int n){
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  for(int i=0;i<n;i++){
    SlabLife *L=arr[i];
    if(L->refs>0) L->refs--;
    if(L->refs==0 && L->zombie && !L->waited){
      auto it=std::find(g_retired_slab_lives.begin(),g_retired_slab_lives.end(),L);
      if(it!=g_retired_slab_lives.end()) g_retired_slab_lives.erase(it);
      L->buf=nil; delete L;
    }
  }
  g_slab_cv.notify_all();
}

// Take one in-flight reference per distinct tracked slab a command buffer reads.
// Returns 1 with *out_arr==NULL when nothing needs tracking (persistent state
// off, or every slab pinned/untracked); 1 with a malloc'd array of held nodes on
// success; 0 (nothing held) when a slab vanished between resolve and commit or
// malloc failed -- the caller must abandon the submission (CPU fallback).
static int slab_refs_acquire(id<MTLBuffer> const *use, int nuse, SlabLife ***out_arr, int *out_n){
  *out_arr=nullptr; *out_n=0;
  if(!g_metal_persistent_state || nuse<=0) return 1;
  SlabLife **arr=(SlabLife**)malloc(sizeof(SlabLife*)*(size_t)nuse);
  if(!arr){ fprintf(stderr,"[metal] slab ref array alloc failed; falling back\n"); return 0; }
  int n=0;
  {
    std::lock_guard<std::mutex> lk(g_slab_mtx);
    for(int i=0;i<nuse;i++){
      const Slab *hit=nullptr;
      for(const auto &s:g_slabs) if(s.buf==use[i]){ hit=&s; break; }
      if(!hit){                        // unregistered between resolve and commit
        for(int j=0;j<n;j++) if(arr[j]->refs>0) arr[j]->refs--;
        g_slab_cv.notify_all();
        free(arr);
        return 0;
      }
      if(!hit->life){                  // registered outside this persistent epoch: unsafe, refuse
        if(!g_slab_untracked_warned){ g_slab_untracked_warned=true;
          fprintf(stderr,"[metal] slab without lifecycle node; refusing GPU submission\n"); }
        for(int j=0;j<n;j++) if(arr[j]->refs>0) arr[j]->refs--;
        g_slab_cv.notify_all();
        free(arr);
        return 0;
      }
      if(hit->life->pinned) continue;  // immutable fast path: no refcounting
      hit->life->refs++;
      arr[n++]=hit->life;
    }
    g_slab_refs_acquired+=(uint64_t)n;
  }
  if(n==0){ free(arr); return 1; }
  *out_arr=arr; *out_n=n;
  return 1;
}

// Owns the acquired references until a completion handler is attached; any early
// return between acquire and commit releases them (clean non-persistent fallback).
struct SlabRefGuard {
  SlabLife **arr; int n;
  ~SlabRefGuard(){ if(arr){ slab_refs_release_all(arr,n); free(arr); } }
  void disarm(){ arr=nullptr; n=0; }
};

// Retire one lifecycle node under g_slab_mtx.  wait_for_refs=true blocks until
// no in-flight command buffer references the slab; the wait is bounded because
// completed handlers always fire for committed command buffers.  The caller may
// free or reuse the host backing immediately after a waited retire returns.
static void slab_life_retire_locked(std::unique_lock<std::mutex> &lk, SlabLife *life, bool wait_for_refs){
  if(!life) return;
  if(life->pinned){
    // Process-lifetime contract: never drop the no-copy wrap before shutdown.
    life->zombie=true;
    g_retired_slab_lives.push_back(life);
    fprintf(stderr,"[metal] unregister of pinned slab deferred to shutdown\n");
    return;
  }
  if(life->refs==0){ life->buf=nil; delete life; return; }
  life->zombie=true;
  g_retired_slab_lives.push_back(life);
  g_slab_deferred_unregisters++;
  if(g_retired_slab_lives.size()>ILI_SLAB_RETIRE_MAX)
    fprintf(stderr,"[metal] retired slab queue above cap (%zu); completions will drain it\n",
        g_retired_slab_lives.size());
  if(!wait_for_refs) return;           // same-base refresh: backing stays valid
  life->waited=true;
  g_slab_unregister_waits++;
  int warned=0;
  while(life->refs>0){
    if(g_slab_cv.wait_for(lk,std::chrono::seconds(5))==std::cv_status::timeout && !warned){
      warned=1;
      fprintf(stderr,"[metal] unregister still waiting on %u in-flight slab refs\n",life->refs);
    }
  }
  auto it=std::find(g_retired_slab_lives.begin(),g_retired_slab_lives.end(),life);
  if(it!=g_retired_slab_lives.end()) g_retired_slab_lives.erase(it);
  life->buf=nil; delete life;
}

// Lab introspection for stress tests.
extern "C" void ili_metal_lab_lifecycle_stats(uint64_t *refs_acquired, uint64_t *deferred,
                                               uint64_t *waits, uint64_t *retired_pending){
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  if(refs_acquired)*refs_acquired=g_slab_refs_acquired;
  if(deferred)*deferred=g_slab_deferred_unregisters;
  if(waits)*waits=g_slab_unregister_waits;
  if(retired_pending)*retired_pending=(uint64_t)g_retired_slab_lives.size();
}

'''
    text = replace_once(
        text,
        'extern "C" void ili_metal_register(void *base, size_t len) {\n',
        lifecycle_helpers + 'extern "C" void ili_metal_register(void *base, size_t len) {\n',
        "slab lifecycle helpers",
    )

    old_registry = '''extern "C" void ili_metal_register(void *base, size_t len) {
  if (!g_dev || !base) return;
  id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:base length:len
                              options:g_res_opts deallocator:nil];
  if (!b) return;
  uintptr_t key=(uintptr_t)base;
  std::lock_guard<std::mutex> lk(g_slab_mtx);   // called from parallel expert_load threads
  auto it=std::lower_bound(g_slabs.begin(),g_slabs.end(),key,
      [](const Slab& s, uintptr_t v){ return (uintptr_t)s.base<v; });
  if(it!=g_slabs.end() && it->base==base){
    it->len=len; it->buf=b; g_slab_gen.fetch_add(1,std::memory_order_release); return;
  }
  g_slabs.insert(it,{base,len,b});
  g_slab_gen.fetch_add(1,std::memory_order_release);
}
extern "C" void ili_metal_unregister(void *base) {
  uintptr_t key=(uintptr_t)base;
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  auto it=std::lower_bound(g_slabs.begin(),g_slabs.end(),key,
      [](const Slab& s, uintptr_t v){ return (uintptr_t)s.base<v; });
  if(it!=g_slabs.end() && it->base==base){
    it->buf=nil; g_slabs.erase(it); g_slab_gen.fetch_add(1,std::memory_order_release);
  }
}
'''
    new_registry = '''static void m5_slab_register(void *base, size_t len, int pinned) {
  if (!g_dev || !base) return;
  id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:base length:len
                              options:g_res_opts deallocator:nil];
  if (!b) return;
  SlabLife *life=nullptr;
  if(g_metal_persistent_state){
    life=new(std::nothrow) SlabLife;
    if(!life){
      // An untracked slab could be freed under an in-flight command buffer, so
      // refuse the registration: resolve() misses it and the block runs on the
      // CPU / non-persistent fallback instead.
      fprintf(stderr,"[metal] slab lifecycle node alloc failed; slab not registered (CPU fallback)\\n");
      return;
    }
    life->buf=b; life->pinned=pinned!=0;
  }
  uintptr_t key=(uintptr_t)base;
  std::unique_lock<std::mutex> lk(g_slab_mtx);   // called from parallel expert_load threads
  uint64_t gen=g_slab_gen.fetch_add(1,std::memory_order_release)+1;
  if(life) life->gen=gen;
  auto it=std::lower_bound(g_slabs.begin(),g_slabs.end(),key,
      [](const Slab& s, uintptr_t v){ return (uintptr_t)s.base<v; });
  if(it!=g_slabs.end() && it->base==base){
    SlabLife *old=it->life;
    it->len=len; it->buf=b; it->life=life; it->gen=gen;
    // Same-base refresh reuses the caller's allocation, so the old wrap retires
    // without blocking; in-flight readers keep the old MTLBuffer alive via refs.
    slab_life_retire_locked(lk,old,false);
    return;
  }
  g_slabs.insert(it,{base,len,b,life,gen});
}
extern "C" void ili_metal_register(void *base, size_t len) { m5_slab_register(base,len,0); }
// Explicit process-lifetime pin: resolution skips in-flight refcounting for this
// slab (immutable fast path).  Only pinned when the caller says so.
extern "C" void ili_metal_register_pinned(void *base, size_t len) { m5_slab_register(base,len,1); }
extern "C" void ili_metal_unregister(void *base) {
  uintptr_t key=(uintptr_t)base;
  std::unique_lock<std::mutex> lk(g_slab_mtx);
  auto it=std::lower_bound(g_slabs.begin(),g_slabs.end(),key,
      [](const Slab& s, uintptr_t v){ return (uintptr_t)s.base<v; });
  if(it==g_slabs.end() || it->base!=base) return;
  SlabLife *life=it->life;
  it->buf=nil; g_slabs.erase(it); g_slab_gen.fetch_add(1,std::memory_order_release);
  // The engine frees or reuses the backing right after unregister returns, so
  // block until no in-flight command buffer references this slab.
  slab_life_retire_locked(lk,life,true);
}
'''
    text = replace_once(text, old_registry, new_registry, "lifecycle-aware slab registry")

    # ---- lab-only fault injection for command-buffer allocation -----------
    text = replace_once(
        text,
        "static id<MTLCommandBuffer> m5_command_buffer(){\n",
        """// Lab-only fault injection: force the next N main-queue command buffers to be
// nil so tests can drive the allocation-failure fallback deterministically.
static std::atomic<int> g_lab_fail_cb{0};
extern "C" void ili_metal_lab_fail_next_command_buffer(int n){ g_lab_fail_cb.store(n,std::memory_order_relaxed); }
static id<MTLCommandBuffer> m5_command_buffer(){
  if(g_lab_fail_cb.load(std::memory_order_relaxed)>0 &&
     g_lab_fail_cb.fetch_sub(1,std::memory_order_relaxed)>0) return nil;
""",
        "command-buffer fault injection",
    )

    # ---- legacy Metal MoE path: refs across commit ------------------------
    text = replace_once(
        text,
        """    if(!(b=resolve(ds[e],&sdv[e]))) {g_moe_fb++; return nil;} add_use(b);
  }
""",
        """    if(!(b=resolve(ds[e],&sdv[e]))) {g_moe_fb++; return nil;} add_use(b);
  }
  SlabLife **held_refs=nullptr; int nheld=0;
  if(!slab_refs_acquire(use,nuse,&held_refs,&nheld)){ g_moe_fb++; return nil; }
  SlabRefGuard ref_guard{held_refs,nheld};
""",
        "legacy MoE ref acquisition",
    )

    text = replace_once(
        text,
        """  id<MTLCommandBuffer> cb=m5_command_buffer(); id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  for(int i=0;i<nuse;i++) [e useResource:use[i] usage:MTLResourceUsageRead];
""",
        """  id<MTLCommandBuffer> cb=m5_command_buffer(); id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  // Bail while ref_guard is still armed: a nil cb would make addCompletedHandler
  // a silent no-op and leak the refs, hanging every later unregister of them.
  if(!cb || !e){ g_moe_fb++; return nil; }
  for(int i=0;i<nuse;i++) [e useResource:use[i] usage:MTLResourceUsageRead];
""",
        "legacy MoE nil command-buffer bail",
    )

    text = replace_once(
        text,
        """  g_t_setup += mnow() - ts_start;
  [e endEncoding];[cb commit];
  return cb;
""",
        """  g_t_setup += mnow() - ts_start;
  if(held_refs){
    SlabLife **cb_refs=held_refs; int cb_n=nheld;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done){
      (void)done; slab_refs_release_all(cb_refs,cb_n); free(cb_refs);
    }];
    ref_guard.disarm();
  }
  [e endEncoding];[cb commit];
  return cb;
""",
        "legacy MoE ref release on completion",
    )

    # ---- Metal 4 MoE path: refs across commit -----------------------------
    text = replace_once(
        text,
        """    if(!(b=resolve(ds[ex],&sdv[ex]))) return 0; add_use(b);
  }
""",
        """    if(!(b=resolve(ds[ex],&sdv[ex]))) return 0; add_use(b);
  }
  SlabLife **held_refs=nullptr; int nheld=0;
  if(!slab_refs_acquire(use,nuse,&held_refs,&nheld)) return 0;
  SlabRefGuard ref_guard{held_refs,nheld};
""",
        "Metal 4 MoE ref acquisition",
    )

    text = replace_once(
        text,
        """  slot->feedback_error4=nil; slot->gpu_start4=0.0; slot->gpu_end4=0.0;
  [slot->commit_options4 addFeedbackHandler:slot->feedback_handler4];
""",
        """  slot->feedback_error4=nil; slot->gpu_start4=0.0; slot->gpu_end4=0.0;
  [slot->commit_options4 addFeedbackHandler:slot->feedback_handler4];
  if(held_refs){
    SlabLife **cb_refs=held_refs; int cb_n=nheld;
    [slot->commit_options4 addFeedbackHandler:^(id<MTL4CommitFeedback> fb4){
      (void)fb4; slab_refs_release_all(cb_refs,cb_n); free(cb_refs);
    }];
    ref_guard.disarm();
  }
""",
        "Metal 4 MoE ref release on completion",
    )

    # ---- Metal 4 slot state: in-flight snapshot tracking ------------------
    text = replace_once(
        text,
        "  __unsafe_unretained id<MTLBuffer> seen4_scratch4[5]={}; // track scratch buffer identity\n",
        """  __unsafe_unretained id<MTLBuffer> seen4_scratch4[5]={}; // track scratch buffer identity
  __unsafe_unretained id<MTLResidencySet> snapshot_inflight4=nil; // global snapshot bound to an in-flight cb (guarded by g_slab_mtx)
""",
        "Metal 4 in-flight snapshot slot field",
    )

    text = replace_once(
        text,
        "    slot->residency4_set=false; slot->args4_set=false; slot->seen4_n=0;\n",
        """    slot->residency4_set=false; slot->args4_set=false; slot->seen4_n=0;
    slot->snapshot_inflight4=nil;
""",
        "Metal 4 shutdown snapshot slot reset",
    )

    # ---- shutdown: drain lifecycle refs, reset persistent globals ---------
    text = replace_once(
        text,
        """  g_queue4=nil; g_metal4_moe=0; g_metal4_requested=0;
  g_metal4_ok=g_metal4_legacy_fallback=g_metal4_failed=0;
""",
        """  g_queue4=nil; g_metal4_moe=0; g_metal4_requested=0;
  g_global_residency4=nil; g_global_residency_gen4=0;
  g_global_residency_refresh4=g_global_residency_allocs4=0;
  g_global_residency_refresh_sec4=0.0;
  [g_retired_global_residency4 removeAllObjects]; g_retired_global_residency4=nil;
  g_metal4_ok=g_metal4_legacy_fallback=g_metal4_failed=0;
""",
        "persistent Metal 4 shutdown",
    )

    text = replace_once(
        text,
        """  g_gemv=nil; g_queue=nil; g_dev=nil; g_tensor_count=g_tensor_bytes=0;
}
""",
        """  // Drain lifecycle refs: completed handlers always fire for committed command
  // buffers, so this wait is bounded and cannot deadlock at shutdown.  Nodes a
  // blocked unregister waits on are destroyed by that unregister, not here.
  {
    std::unique_lock<std::mutex> lk(g_slab_mtx);
    g_slab_cv.wait(lk,[]{
      for(SlabLife *L:g_retired_slab_lives) if(L->refs>0) return false;
      return true;
    });
    for(size_t i=g_retired_slab_lives.size();i-- > 0;){
      SlabLife *L=g_retired_slab_lives[i];
      if(L->waited) continue;
      L->buf=nil; delete L;
      g_retired_slab_lives.erase(g_retired_slab_lives.begin()+(long)i);
    }
    g_slab_refs_acquired=g_slab_deferred_unregisters=g_slab_unregister_waits=0;
    g_metal_persistent_state=0;
  }
  g_gemv=nil; g_queue=nil; g_dev=nil; g_tensor_count=g_tensor_bytes=0;
}
""",
        "lifecycle shutdown drain",
    )

    # ---- global residency snapshot (Metal 4), bounded retirement ----------
    marker = """// Lazily create one independent Metal 4 command arena for each existing MoE
// scratch slot.  The sync and async slots can be in flight at the same time.
static int metal4_slot_init(MoeSlot *slot) {
"""
    helper = r'''// Retired snapshots stay strongly retained here until neither slot has them
// bound to an in-flight command buffer (Metal 4 command buffers do not retain
// residency sets).  Called with g_slab_mtx held.
static void metal4_snapshot_sweep_locked(void) {
  if(!g_retired_global_residency4) return;
  for(NSInteger i=(NSInteger)[g_retired_global_residency4 count]-1;i>=0;i--){
    id<MTLResidencySet> s=g_retired_global_residency4[(NSUInteger)i];
    if(s==g_moe_sync.snapshot_inflight4 || s==g_moe_async.snapshot_inflight4) continue;
    [g_retired_global_residency4 removeObjectAtIndex:(NSUInteger)i];
  }
}

// Build a global residency snapshot only when the registered slab generation
// changes.  Old snapshots retire into a bounded list swept on completion; if the
// list is at capacity the refresh fails cleanly into per-slot residency.
static id<MTLResidencySet> metal4_global_residency_snapshot(void) {
  if(!g_metal_persistent_state || !g_metal4_moe || !g_dev) return nil;
  uint64_t wanted=g_slab_gen.load(std::memory_order_acquire);
  if(g_global_residency4 && g_global_residency_gen4==wanted) return g_global_residency4;

  double t0=mnow();
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  wanted=g_slab_gen.load(std::memory_order_relaxed);
  if(g_global_residency4 && g_global_residency_gen4==wanted) return g_global_residency4;
  metal4_snapshot_sweep_locked();
  if(g_retired_global_residency4 &&
     (int)[g_retired_global_residency4 count]>=ILI_SNAP_RETIRE_MAX){
    fprintf(stderr,"[metal4] retired residency snapshots at cap (%d); per-slot residency until drain\n",
        ILI_SNAP_RETIRE_MAX);
    return nil;
  }

  MTLResidencySetDescriptor *rd=[MTLResidencySetDescriptor new];
  rd.initialCapacity=(NSUInteger)(g_slabs.size()+64);
  NSError *error=nil;
  id<MTLResidencySet> fresh=[g_dev newResidencySetWithDescriptor:rd error:&error];
  if(!fresh || error){
    fprintf(stderr,"[metal4] global residency snapshot failed: %s; using per-slot residency\n",
        error?[[error localizedDescription]UTF8String]:"?");
    return nil;
  }
  for(const auto &slab:g_slabs){
    id<MTLAllocation> allocation=(id<MTLAllocation>)slab.buf;
    if(allocation) [fresh addAllocations:&allocation count:1];
  }
  [fresh commit];
  if(g_global_residency4){
    if(!g_retired_global_residency4) g_retired_global_residency4=[NSMutableArray array];
    [g_retired_global_residency4 addObject:g_global_residency4];
  }
  g_global_residency4=fresh; g_global_residency_gen4=wanted;
  g_global_residency_refresh4++; g_global_residency_allocs4=(uint64_t)g_slabs.size();
  g_global_residency_refresh_sec4+=mnow()-t0;
  fprintf(stderr,"[metal4] persistent residency refresh %llu: %zu slabs in %.3f ms\n",
      (unsigned long long)g_global_residency_refresh4,g_slabs.size(),(mnow()-t0)*1000.0);
  return g_global_residency4;
}

'''
    text = replace_once(text, marker, helper + marker, "persistent residency helper")

    old_block = r'''  // Persistent residency: publish once on first submission, add only new buffers afterwards.
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
'''
    new_block = r'''  id<MTLResidencySet> global_residency4=metal4_global_residency_snapshot();
  const bool use_global_residency4=(g_metal_persistent_state && global_residency4!=nil);
  int added_new4=false;
  if(use_global_residency4){
    // Expert/scales live in the generation-tagged global set.  The slot set owns
    // only five growable scratch/metadata buffers and changes only on reallocation.
    id<MTLBuffer> scratch4[]={slot->xg,slot->gg,slot->uu,slot->hh,slot->meta};
    for(int i=0;i<5;i++){
      if(!slot->seen4_scratch4[i] || slot->seen4_scratch4[i]!=scratch4[i]){
        id<MTLAllocation> alloc=(id<MTLAllocation>)scratch4[i];
        [slot->residency4 addAllocations:&alloc count:1];
        slot->seen4_scratch4[i]=scratch4[i];
        added_new4=true;
      }
    }
    if(!slot->residency4_set || added_new4){
      [slot->residency4 commit]; slot->residency4_set=true;
    }
  } else if(!slot->residency4_set){
    // Original per-slot behavior remains the fallback and the default.
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
'''
    text = replace_once(text, old_block, new_block, "global versus slot residency policy")

    text = replace_once(
        text,
        """  [cb useResidencySet:slot->residency4];
""",
        """  if(use_global_residency4){
    [cb useResidencySet:global_residency4];
    // Record the binding so the retirement sweep keeps this snapshot strongly
    // retained until this command buffer completes (MTL4 does not retain it).
    std::lock_guard<std::mutex> snap_lk(g_slab_mtx);
    slot->snapshot_inflight4=global_residency4;
    if(global_residency4!=g_global_residency4){
      if(!g_retired_global_residency4) g_retired_global_residency4=[NSMutableArray array];
      if(![g_retired_global_residency4 containsObject:global_residency4])
        [g_retired_global_residency4 addObject:global_residency4];
    }
  }
  [cb useResidencySet:slot->residency4];
""",
        "global residency binding",
    )

    text = replace_once(
        text,
        """  slot->inflight4=false;
  [slot->allocator4 reset];
""",
        """  slot->inflight4=false;
  [slot->allocator4 reset];
  if(g_metal_persistent_state){
    std::lock_guard<std::mutex> snap_lk(g_slab_mtx);
    slot->snapshot_inflight4=nil;
    metal4_snapshot_sweep_locked();
  }
""",
        "snapshot release on completion",
    )

    text = replace_once(
        text,
        """      (unsigned long long)g_metal4_failed);
""",
        """      (unsigned long long)g_metal4_failed);
  if(g_metal_persistent_state) fprintf(stderr,
      "[metal4] persistent refresh %llu | slabs %llu | refresh %.3f s\\n",
      (unsigned long long)g_global_residency_refresh4,
      (unsigned long long)g_global_residency_allocs4,g_global_residency_refresh_sec4);
""",
        "persistent residency counters",
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
