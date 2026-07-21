#!/usr/bin/env python3
"""Generate an M5 Max-specialized Metal backend without modifying upstream source.

Optimizations applied:
1. Keep the registered unified-memory slab registry ordered for logarithmic
   interval lookup instead of scanning every resident slab.
2. Add a generation-safe exact-pointer cache for repeated dense/expert weight
   and scale resolution without taking the registry mutex.
3. Chain the reusable two-slot MoE scratch/metadata pool transformation.

RoPE sin/cos caching (once per decode position, reused across layers and heads)
and the a_rope threadgroup-barrier race fix now live directly in upstream
backend_metal.mm, so this generator no longer needs to patch them in.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.source.read_text()

    text = replace_once(
        text,
        '#include <mutex>\n',
        '#include <mutex>\n#include <atomic>\n#include <algorithm>\n',
        'atomic and algorithm includes',
    )

    text = replace_once(
        text,
        'static std::vector<Slab> g_slabs;\nstatic std::mutex g_slab_mtx;   // expert_load registers slabs from parallel OpenMP threads\n',
        '''static std::vector<Slab> g_slabs;  // always ordered by host base address
static std::mutex g_slab_mtx;   // expert_load registers slabs from parallel OpenMP threads
static std::atomic<uint64_t> g_slab_gen{1};
enum { RESOLVE_CACHE_N=8192 };
struct ResolveCache {
  uintptr_t key=0; uint64_t addr=0; __unsafe_unretained id<MTLBuffer> buf=nil; uint64_t gen=0;
};
static thread_local ResolveCache g_resolve_cache[RESOLVE_CACHE_N];
static inline size_t resolve_cache_index(uintptr_t u){
  uint64_t x=(uint64_t)u; x^=x>>17; x*=0xed5ad4bbU; x^=x>>23;
  return (size_t)x&(RESOLVE_CACHE_N-1);
}
''',
        'resolver declarations',
    )

    old_registry = '''extern "C" void ili_metal_register(void *base, size_t len) {
  if (!g_dev || !base) return;
  id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:base length:len
                              options:g_res_opts deallocator:nil];
  if (!b) return;
  std::lock_guard<std::mutex> lk(g_slab_mtx);   // called from parallel expert_load threads
  for (auto &s : g_slabs) if (s.base == base) { s.len = len; s.buf = b; return; }
  g_slabs.push_back({base, len, b});
}
extern "C" void ili_metal_unregister(void *base) {
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  for (size_t i=0;i<g_slabs.size();i++) if (g_slabs[i].base==base) { g_slabs[i].buf=nil; g_slabs.erase(g_slabs.begin()+i); return; }
}
// Resolve a host pointer inside a registered slab to (buffer, gpuAddress). Returns nil if unknown.
static id<MTLBuffer> resolve(const void *p, uint64_t *addr) {
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  uintptr_t u=(uintptr_t)p;
  for (auto &s : g_slabs) { uintptr_t b=(uintptr_t)s.base;
    if (u>=b && u<b+s.len) { *addr = (uint64_t)[s.buf gpuAddress] + (u-b); return s.buf; } }
  return nil;
}
'''
    new_registry = '''extern "C" void ili_metal_register(void *base, size_t len) {
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
// Exact pointers repeat across layers/tokens, so the common path is a lock-free TLS hash.
// A miss uses predecessor interval lookup in the ordered registry: O(log resident slabs).
static id<MTLBuffer> resolve(const void *p, uint64_t *addr) {
  uintptr_t u=(uintptr_t)p;
  uint64_t gen=g_slab_gen.load(std::memory_order_acquire);
  ResolveCache &ce=g_resolve_cache[resolve_cache_index(u)];
  if(ce.gen==gen && ce.key==u && ce.buf){ *addr=ce.addr; return ce.buf; }

  std::lock_guard<std::mutex> lk(g_slab_mtx);
  gen=g_slab_gen.load(std::memory_order_relaxed);
  auto it=std::upper_bound(g_slabs.begin(),g_slabs.end(),u,
      [](uintptr_t v, const Slab& s){ return v<(uintptr_t)s.base; });
  if(it==g_slabs.begin()) return nil;
  --it;
  uintptr_t b=(uintptr_t)it->base;
  if(u>=b && u<b+it->len){
    uint64_t exact=(uint64_t)[it->buf gpuAddress]+(u-b);
    ce.key=u; ce.addr=exact; ce.buf=it->buf; ce.gen=gen;
    *addr=exact; return it->buf;
  }
  return nil;
}
'''
    text = replace_once(text, old_registry, new_registry, 'ordered slab resolver')

    args.output.write_text('// Generated by tools/gen_m5max_backend.py; do not edit directly.\n' + text)
    pool_patcher = Path(__file__).with_name('patch_m5max_moe_pool.py')
    subprocess.run(
        [sys.executable, str(pool_patcher), str(args.output), str(args.output)],
        check=True,
    )


if __name__ == '__main__':
    main()
