#!/usr/bin/env python3
"""Add low-overhead ordered expert-route tracing to the generated M5 engine.

The trace is opt-in through ILI_ROUTE_TRACE. It records every
selected route in the exact token/rank order produced by the router. Each fixed
24-byte little-endian record contains:

    event_id:u64, moe_call_id:u64, layer:u16, batch_row:u16,
    route_rank:u16, expert_id:u16

Version 2 also stores the engine's exact q4 expert byte size and current pin/LRU
budget in the header, removing rounded-unit assumptions from offline simulation.
The source glm.c remains untouched.
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
    marker = """/* MoE GLM su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, + shared expert).
"""
    helper = r'''/* ---- M5 ordered expert-route trace ------------------------------------------
 * Opt-in only: ILI_ROUTE_TRACE=/path/file.
 * Fixed little-endian format keeps the hot-path overhead bounded and makes the
 * trace independent of compiler struct padding. Tracing is single-writer because
 * layer/MoE execution is owned by the inference thread.
 */
enum { M5_ROUTE_RECORD_SIZE=24, M5_ROUTE_BUFFER_RECORDS=4096, M5_ROUTE_HEADER_SIZE=40 };
static int64_t expert_bytes_probe(Model *m, int ebits); /* defined by RAM planner below */
typedef struct {
    FILE *fp;
    unsigned char buf[M5_ROUTE_RECORD_SIZE*M5_ROUTE_BUFFER_RECORDS];
    size_t used;
    uint64_t event_id, call_id, max_events;
    int initialized, disabled, sync_each_call;
} M5RouteTrace;
static M5RouteTrace g_m5_route_trace;

static inline void m5_route_put16(unsigned char *p, uint16_t v){
    p[0]=(unsigned char)(v&255u); p[1]=(unsigned char)(v>>8);
}
static inline void m5_route_put32(unsigned char *p, uint32_t v){
    for(int i=0;i<4;i++) p[i]=(unsigned char)(v>>(8*i));
}
static inline void m5_route_put64(unsigned char *p, uint64_t v){
    for(int i=0;i<8;i++) p[i]=(unsigned char)(v>>(8*i));
}
static void m5_route_trace_flush(void){
    M5RouteTrace *t=&g_m5_route_trace;
    if(!t->fp || !t->used) return;
    if(fwrite(t->buf,1,t->used,t->fp)!=t->used){
        fprintf(stderr,"[route-trace] write failed; tracing disabled\n");
        t->disabled=1;
    }
    t->used=0;
}
static void m5_route_trace_close(void){
    M5RouteTrace *t=&g_m5_route_trace;
    if(!t->fp) return;
    m5_route_trace_flush();
    fprintf(stderr,"[route-trace] wrote %llu route events across %llu MoE calls\n",
        (unsigned long long)t->event_id,(unsigned long long)t->call_id);
    fclose(t->fp); t->fp=NULL;
}
static void m5_route_trace_init(Model *m){
    M5RouteTrace *t=&g_m5_route_trace;
    if(t->initialized) return;
    t->initialized=1;
    const char *path=ili_env("ROUTE_TRACE");
    if(!path||!*path) { t->disabled=1; return; }
    const char *mx=ili_env("ROUTE_TRACE_MAX_EVENTS");
    if(mx&&*mx){ char *e=NULL; unsigned long long v=strtoull(mx,&e,10); if(e!=mx&&!*e)t->max_events=(uint64_t)v; }
    const char *sy=ili_env("ROUTE_TRACE_SYNC");
    t->sync_each_call=sy&&atoi(sy)!=0;
    t->fp=fopen(path,"wb");
    if(!t->fp){ fprintf(stderr,"[route-trace] cannot open %s; tracing disabled\n",path); t->disabled=1; return; }
    setvbuf(t->fp,NULL,_IOFBF,1<<20);

    /* Header v2 uses q4-equivalent slots. MTP experts are int8 and therefore count
     * as two units in both cache-layer and pinned-unit fields, matching cap_for_ram. */
    uint32_t cache_units=0,pinned_units=0,flags=m->has_mtp?1u:0u;
    for(int i=0;i<m->c.n_layers;i++) if(m->L[i].sparse){
        cache_units++; pinned_units+=(uint32_t)m->npin[i];
    }
    if(m->has_mtp){ cache_units+=2; pinned_units+=(uint32_t)(2*m->npin[m->c.n_layers]); }
    uint64_t expert_bytes=(uint64_t)expert_bytes_probe(m,m->ebits);
    unsigned char h[M5_ROUTE_HEADER_SIZE]={0}; memcpy(h,"FAROUTE1",8);
    m5_route_put32(h+8,2); m5_route_put32(h+12,M5_ROUTE_RECORD_SIZE);
    m5_route_put64(h+16,expert_bytes); m5_route_put32(h+24,cache_units);
    m5_route_put32(h+28,(uint32_t)m->ecap); m5_route_put32(h+32,pinned_units);
    m5_route_put32(h+36,flags);
    if(fwrite(h,1,sizeof(h),t->fp)!=sizeof(h)){ fclose(t->fp); t->fp=NULL; t->disabled=1; return; }
    atexit(m5_route_trace_close);
    fprintf(stderr,"[route-trace] enabled: %s | expert=%llu bytes | pin-units=%u | LRU=%u x %u\n",
        path,(unsigned long long)expert_bytes,pinned_units,(unsigned)m->ecap,cache_units);
}
static void m5_route_trace_emit(Model *m,int layer,int S,int K,const int *idxs,const int *keff){
    M5RouteTrace *t=&g_m5_route_trace;
    if(!t->initialized) m5_route_trace_init(m);
    if(t->disabled||!t->fp) return;
    uint64_t call=t->call_id++;
    for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++){
        if(t->max_events && t->event_id>=t->max_events){ t->disabled=1; m5_route_trace_flush(); return; }
        if(t->used+M5_ROUTE_RECORD_SIZE>sizeof(t->buf)) m5_route_trace_flush();
        if(t->disabled) return;
        unsigned char *r=t->buf+t->used;
        m5_route_put64(r,t->event_id++); m5_route_put64(r+8,call);
        m5_route_put16(r+16,(uint16_t)layer); m5_route_put16(r+18,(uint16_t)s);
        m5_route_put16(r+20,(uint16_t)k); m5_route_put16(r+22,(uint16_t)idxs[(int64_t)s*K+k]);
        t->used+=M5_ROUTE_RECORD_SIZE;
    }
    if(t->sync_each_call){ m5_route_trace_flush(); fflush(t->fp); }
}

'''
    text = replace_once(text, marker, helper + marker, "route trace helper insertion")

    hook = """    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];
"""
    replacement = hook + "    m5_route_trace_emit(m,layer,S,K,idxs,keff);\n"
    return replace_once(text, hook, replacement, "ordered route trace hook")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(patch_text(args.source.read_text()))


if __name__ == "__main__":
    main()
