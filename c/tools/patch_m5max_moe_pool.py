#!/usr/bin/env python3
"""Patch the generated M5 Max backend with reusable MoE Metal scratch slots.

The engine can have one resident-expert async submission outstanding while a
separate missed-expert submission runs synchronously. Two grow-only slots avoid
reallocating large activation buffers and seven small metadata buffers for every
MoE block without allowing either submission to overwrite the other's inputs.
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
        'static id<MTLBuffer> g_gg, g_uu, g_hh, g_xg; static size_t g_gg_cap, g_uu_cap, g_hh_cap, g_xg_cap;\n',
        '''struct MoeSlot {
  id<MTLBuffer> xg=nil, gg=nil, uu=nil, hh=nil, meta=nil;
  size_t xg_cap=0, gg_cap=0, uu_cap=0, hh_cap=0, meta_cap=0;
};
static MoeSlot g_moe_sync, g_moe_async;
static std::mutex g_moe_sync_mtx;
static std::atomic<bool> g_moe_async_busy{false};
''',
        'MoE slot declarations',
    )

    old_sig = '''static id<MTLCommandBuffer> moe_submit(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr, int R,
                         id<MTLBuffer> xg_buf, id<MTLBuffer> gg_buf, id<MTLBuffer> uu_buf, id<MTLBuffer> hh_buf) {
  if (!g_dev || (fmt != 1 && fmt != 2 && fmt != 3)) return nil;
  double ts_start = mnow();
'''
    new_sig = '''static id<MTLCommandBuffer> moe_submit(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr, int R,
                         MoeSlot *slot) {
  if (!g_dev || (fmt != 1 && fmt != 2 && fmt != 3)) return nil;
  double ts_start = mnow();
  slot->xg=ensure(slot->xg,&slot->xg_cap,(size_t)R*D*4);
  slot->gg=ensure(slot->gg,&slot->gg_cap,(size_t)R*Iinter*4);
  slot->uu=ensure(slot->uu,&slot->uu_cap,(size_t)R*Iinter*4);
  slot->hh=ensure(slot->hh,&slot->hh_cap,(size_t)R*D*4);
  id<MTLBuffer> xg_buf=slot->xg, gg_buf=slot->gg, uu_buf=slot->uu, hh_buf=slot->hh;
'''
    text = replace_once(text, old_sig, new_sig, 'moe_submit signature')

    old_meta = '''  std::vector<int> erow(R); for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++) erow[xoff[e]+r]=e;
  auto shb=[&](const void*p,size_t n){ return [g_dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared]; };
  id<MTLBuffer> bag=shb(ag.data(),nb*8), bau=shb(au.data(),nb*8), bad=shb(ad.data(),nb*8);
  id<MTLBuffer> bsg=shb(sgv.data(),nb*8), bsu=shb(suv.data(),nb*8), bsd=shb(sdv.data(),nb*8);
  id<MTLBuffer> berow=shb(erow.data(),R*4);
  memcpy([xg_buf contents], xg, (size_t)R*D*4);

  id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  for(auto&b:use) [e useResource:b usage:MTLResourceUsageRead];
  auto gemv=[&](id<MTLBuffer> wa,id<MTLBuffer> sa,id<MTLBuffer> xin,id<MTLBuffer> y,int O,int K,int Kin){
    int NT=R*O;
    [e setComputePipelineState:g_moe_gemv];
    [e setBuffer:wa offset:0 atIndex:0];[e setBuffer:sa offset:0 atIndex:1];[e setBuffer:berow offset:0 atIndex:2];
    [e setBuffer:xin offset:0 atIndex:3];[e setBuffer:y offset:0 atIndex:4];
    [e setBytes:&O length:4 atIndex:5];[e setBytes:&K length:4 atIndex:6];[e setBytes:&Kin length:4 atIndex:7];[e setBytes:&fmt length:4 atIndex:8];
    [e setBytes:&NT length:4 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; };
  gemv(bag,bsg,xg_buf,gg_buf,Iinter,D,D);                     // gate
  gemv(bau,bsu,xg_buf,uu_buf,Iinter,D,D);                     // up
'''
    new_meta = '''  std::vector<int> erow(R); for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++) erow[xoff[e]+r]=e;
  auto A=[](size_t v){ return (v+255)&~(size_t)255; };
  size_t oag=0, oau=A(oag+(size_t)nb*8), oad=A(oau+(size_t)nb*8);
  size_t osg=A(oad+(size_t)nb*8), osu=A(osg+(size_t)nb*8), osd=A(osu+(size_t)nb*8);
  size_t oerow=A(osd+(size_t)nb*8), meta_need=A(oerow+(size_t)R*4);
  slot->meta=ensure(slot->meta,&slot->meta_cap,meta_need);
  char *mp=(char*)[slot->meta contents];
  memcpy(mp+oag,ag.data(),(size_t)nb*8); memcpy(mp+oau,au.data(),(size_t)nb*8); memcpy(mp+oad,ad.data(),(size_t)nb*8);
  memcpy(mp+osg,sgv.data(),(size_t)nb*8); memcpy(mp+osu,suv.data(),(size_t)nb*8); memcpy(mp+osd,sdv.data(),(size_t)nb*8);
  memcpy(mp+oerow,erow.data(),(size_t)R*4);
  memcpy([xg_buf contents], xg, (size_t)R*D*4);

  id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  for(auto&b:use) [e useResource:b usage:MTLResourceUsageRead];
  auto gemv=[&](size_t wo,size_t so,id<MTLBuffer> xin,id<MTLBuffer> y,int O,int K,int Kin){
    int NT=R*O;
    [e setComputePipelineState:g_moe_gemv];
    [e setBuffer:slot->meta offset:wo atIndex:0];[e setBuffer:slot->meta offset:so atIndex:1];[e setBuffer:slot->meta offset:oerow atIndex:2];
    [e setBuffer:xin offset:0 atIndex:3];[e setBuffer:y offset:0 atIndex:4];
    [e setBytes:&O length:4 atIndex:5];[e setBytes:&K length:4 atIndex:6];[e setBytes:&Kin length:4 atIndex:7];[e setBytes:&fmt length:4 atIndex:8];
    [e setBytes:&NT length:4 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; };
  gemv(oag,osg,xg_buf,gg_buf,Iinter,D,D);                     // gate
  gemv(oau,osu,xg_buf,uu_buf,Iinter,D,D);                     // up
'''
    text = replace_once(text, old_meta, new_meta, 'MoE metadata pooling')
    text = replace_once(
        text,
        '  gemv(bad,bsd,gg_buf,hh_buf,D,Iinter,Iinter);                // down\n',
        '  gemv(oad,osd,gg_buf,hh_buf,D,Iinter,Iinter);                // down\n',
        'down GEMV metadata offsets',
    )

    old_sync = '''    g_xg = ensure(g_xg,&g_xg_cap,(size_t)R*D*4);
    g_gg = ensure(g_gg,&g_gg_cap,(size_t)R*Iinter*4);
    g_uu = ensure(g_uu,&g_uu_cap,(size_t)R*Iinter*4);
    g_hh = ensure(g_hh,&g_hh_cap,(size_t)R*D*4);
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);
    if (!cb) return 0;
    return moe_finish(cb,g_hh,nb,R,D,rows,rw,out);
'''
    new_sync = '''    std::lock_guard<std::mutex> lk(g_moe_sync_mtx);
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,&g_moe_sync);
    if (!cb) return 0;
    return moe_finish(cb,g_moe_sync.hh,nb,R,D,rows,rw,out);
'''
    text = replace_once(text, old_sync, new_sync, 'synchronous MoE slot')

    old_begin = '''    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0 || !g_dev) return nullptr;
    id<MTLBuffer> bxg=[g_dev newBufferWithLength:(size_t)R*D*4 options:g_res_opts];
    id<MTLBuffer> bgg=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:g_res_opts];
    id<MTLBuffer> buu=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:g_res_opts];
    id<MTLBuffer> bhh=[g_dev newBufferWithLength:(size_t)R*D*4 options:g_res_opts];
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,bxg,bgg,buu,bhh);
    if (!cb) return nullptr;
    IliMetalMoeHandle *h = new IliMetalMoeHandle();
    h->cb=cb; h->hh=bhh; h->rows.assign(rows,rows+R); h->rwv.assign(rw,rw+R);
'''
    new_begin = '''    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0 || !g_dev) return nullptr;
    if(g_moe_async_busy.exchange(true,std::memory_order_acq_rel)) return nullptr;
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,&g_moe_async);
    if (!cb) { g_moe_async_busy.store(false,std::memory_order_release); return nullptr; }
    IliMetalMoeHandle *h = new IliMetalMoeHandle();
    h->cb=cb; h->hh=g_moe_async.hh; h->rows.assign(rows,rows+R); h->rwv.assign(rw,rw+R);
'''
    text = replace_once(text, old_begin, new_begin, 'asynchronous MoE slot')

    text = replace_once(
        text,
        '  h->cb=nil; h->hh=nil; delete h;\n  return ok;\n',
        '  h->cb=nil; h->hh=nil; delete h;\n  g_moe_async_busy.store(false,std::memory_order_release);\n  return ok;\n',
        'asynchronous slot release',
    )

    args.output.write_text(text)


if __name__ == '__main__':
    main()
