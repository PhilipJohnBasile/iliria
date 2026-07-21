#!/usr/bin/env python3
"""Remove remaining per-MoE-block C++ heap allocations in generated Metal code.

The public API caps a block at 65 experts. Address/resource metadata therefore
fits fixed stack arrays. The engine permits only one async resident submission,
so its handle and copied row/weight arrays can be safely reused behind the
existing atomic busy guard.
"""

from __future__ import annotations

import argparse
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
        '''  std::vector<uint64_t> ag(nb),au(nb),ad(nb),sgv(nb),suv(nb),sdv(nb);
  std::vector<id<MTLBuffer>> use; use.reserve(nb*2);
  auto add_use=[&](id<MTLBuffer> b){ for(auto&x:use) if(x==b) return; use.push_back(b); };
''',
        '''  if(nb<1 || nb>65) { g_moe_fb++; return nil; }
  uint64_t ag[65],au[65],ad[65],sgv[65],suv[65],sdv[65];
  id<MTLBuffer> use[390]; int nuse=0;
  auto add_use=[&](id<MTLBuffer> b){
    for(int i=0;i<nuse;i++) if(use[i]==b) return;
    use[nuse++]=b;
  };
''',
        'fixed address and resource metadata',
    )

    old_meta = '''  std::vector<int> erow(R); for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++) erow[xoff[e]+r]=e;
  auto A=[](size_t v){ return (v+255)&~(size_t)255; };
  size_t oag=0, oau=A(oag+(size_t)nb*8), oad=A(oau+(size_t)nb*8);
  size_t osg=A(oad+(size_t)nb*8), osu=A(osg+(size_t)nb*8), osd=A(osu+(size_t)nb*8);
  size_t oerow=A(osd+(size_t)nb*8), meta_need=A(oerow+(size_t)R*4);
  slot->meta=ensure(slot->meta,&slot->meta_cap,meta_need);
  char *mp=(char*)[slot->meta contents];
  memcpy(mp+oag,ag.data(),(size_t)nb*8); memcpy(mp+oau,au.data(),(size_t)nb*8); memcpy(mp+oad,ad.data(),(size_t)nb*8);
  memcpy(mp+osg,sgv.data(),(size_t)nb*8); memcpy(mp+osu,suv.data(),(size_t)nb*8); memcpy(mp+osd,sdv.data(),(size_t)nb*8);
  memcpy(mp+oerow,erow.data(),(size_t)R*4);
'''
    new_meta = '''  auto A=[](size_t v){ return (v+255)&~(size_t)255; };
  size_t oag=0, oau=A(oag+(size_t)nb*8), oad=A(oau+(size_t)nb*8);
  size_t osg=A(oad+(size_t)nb*8), osu=A(osg+(size_t)nb*8), osd=A(osu+(size_t)nb*8);
  size_t oerow=A(osd+(size_t)nb*8), meta_need=A(oerow+(size_t)R*4);
  slot->meta=ensure(slot->meta,&slot->meta_cap,meta_need);
  char *mp=(char*)[slot->meta contents];
  int *erow=(int*)(mp+oerow);
  for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++) erow[xoff[e]+r]=e;
  memcpy(mp+oag,ag,(size_t)nb*8); memcpy(mp+oau,au,(size_t)nb*8); memcpy(mp+oad,ad,(size_t)nb*8);
  memcpy(mp+osg,sgv,(size_t)nb*8); memcpy(mp+osu,suv,(size_t)nb*8); memcpy(mp+osd,sdv,(size_t)nb*8);
'''
    text = replace_once(text, old_meta, new_meta, 'direct metadata packing')
    text = replace_once(
        text,
        '  for(auto&b:use) [e useResource:b usage:MTLResourceUsageRead];\n',
        '  for(int i=0;i<nuse;i++) [e useResource:use[i] usage:MTLResourceUsageRead];\n',
        'fixed resource iteration',
    )

    old_handle = '''struct IliMetalMoeHandle {
  id<MTLCommandBuffer> cb; id<MTLBuffer> hh;
  std::vector<int> rows; std::vector<float> rwv;
  int nb, R, D;
};
'''
    new_handle = '''struct IliMetalMoeHandle {
  id<MTLCommandBuffer> cb; id<MTLBuffer> hh;
  int *rows=nullptr; float *rwv=nullptr; size_t cap=0;
  int nb=0, R=0, D=0;
};
static IliMetalMoeHandle g_async_handle;
static void async_handle_ensure(size_t count){
  if(g_async_handle.cap>=count) return;
  size_t cap=g_async_handle.cap?g_async_handle.cap:64;
  while(cap<count) cap*=2;
  int *rows=(int*)realloc(g_async_handle.rows,cap*sizeof(int));
  float *rwv=(float*)realloc(g_async_handle.rwv,cap*sizeof(float));
  if(!rows||!rwv){ fprintf(stderr,"[metal] async handle OOM (%zu rows)\\n",cap); abort(); }
  g_async_handle.rows=rows; g_async_handle.rwv=rwv; g_async_handle.cap=cap;
}
'''
    text = replace_once(text, old_handle, new_handle, 'reusable async handle')

    old_begin = '''    IliMetalMoeHandle *h = new IliMetalMoeHandle();
    h->cb=cb; h->hh=g_moe_async.hh; h->rows.assign(rows,rows+R); h->rwv.assign(rw,rw+R);
    h->nb=nb; h->R=R; h->D=D;
    return h;
'''
    new_begin = '''    async_handle_ensure((size_t)R);
    IliMetalMoeHandle *h=&g_async_handle;
    h->cb=cb; h->hh=g_moe_async.hh;
    memcpy(h->rows,rows,(size_t)R*sizeof(int)); memcpy(h->rwv,rw,(size_t)R*sizeof(float));
    h->nb=nb; h->R=R; h->D=D;
    return h;
'''
    text = replace_once(text, old_begin, new_begin, 'async handle acquisition')

    old_end = '''  @autoreleasepool { ok = moe_finish(h->cb,h->hh,h->nb,h->R,h->D,h->rows.data(),h->rwv.data(),out); }
  h->cb=nil; h->hh=nil; delete h;
  g_moe_async_busy.store(false,std::memory_order_release);
'''
    new_end = '''  @autoreleasepool { ok = moe_finish(h->cb,h->hh,h->nb,h->R,h->D,h->rows,h->rwv,out); }
  h->cb=nil; h->hh=nil;
  g_moe_async_busy.store(false,std::memory_order_release);
'''
    text = replace_once(text, old_end, new_end, 'async handle release')

    args.output.write_text(text)


if __name__ == '__main__':
    main()
