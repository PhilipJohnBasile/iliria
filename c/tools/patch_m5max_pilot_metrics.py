#!/usr/bin/env python3
"""Add opt-in PILOT and blocked-I/O instrumentation to generated M5 Max C.

The production source stays unchanged.  The patch is applied only by the lab build
and is activated at runtime with ILI_PILOT_METRICS=1 (or ILI_PILOT_METRICS=1).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def sub_once(text: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    out, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return out


def patch_text(text: str) -> str:
    text = sub_once(
        text,
        r"(typedef struct \{ int eid; QT g,u,d; uint8_t \*slab; float \*fslab;\s*"
        r"int64_t slab_cap, fslab_cap; uint64_t used;)( \} ESlot;)",
        r"\1 unsigned char pilot_prefetched;\2",
        "ESlot pilot marker",
        re.S,
    )

    counter_anchor = (
        "static _Atomic long g_pilot_drops=0;     /* predizioni scartate perche' il main possiede gia' il layer */\n"
    )
    counters = counter_anchor + r'''/* Opt-in measurement counters.  All cross-thread values use integer nanoseconds
 * and relaxed atomics; they never influence scheduling or model output. */
static int g_pilot_metrics=0;
static _Atomic uint64_t g_pm_predicted=0, g_pm_enqueued=0, g_pm_queue_full=0;
static _Atomic uint64_t g_pm_resident_skip=0, g_pm_race_skip=0;
static _Atomic uint64_t g_pm_useful=0, g_pm_wasted=0, g_pm_evictions=0;
static _Atomic uint64_t g_pm_late=0, g_pm_load_ns=0, g_pm_barrier_ns=0;
static _Atomic uint64_t g_pm_pipe_wait_ns=0, g_pm_pipe_wait_calls=0;
static inline uint64_t m5_metric_now_ns(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000ULL+(uint64_t)t.tv_nsec;
}
static inline void m5_metric_add(_Atomic uint64_t *v,uint64_t n){
    if(g_pilot_metrics) atomic_fetch_add_explicit(v,n,memory_order_relaxed);
}
'''
    if counter_anchor not in text:
        raise RuntimeError("pilot counter anchor not found")
    text = text.replace(counter_anchor, counters, 1)

    text = sub_once(
        text,
        r"static inline void pipe_wait\(int q\)\{\s*"
        r"while\(!atomic_load_explicit\(&g_pp\.ready\[q\],memory_order_acquire\)\) sched_yield\(\);\s*\}",
        r'''static inline void pipe_wait(int q){
    uint64_t t0=g_pilot_metrics?m5_metric_now_ns():0;
    while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) sched_yield();
    if(t0){ m5_metric_add(&g_pm_pipe_wait_ns,m5_metric_now_ns()-t0);
            m5_metric_add(&g_pm_pipe_wait_calls,1); }
}''',
        "pipe wait timing",
        re.S,
    )

    text = sub_once(
        text,
        r"static void moe\(Model \*m, Layer \*l, int layer, float \*x, int S, float \*out\)\{\s*"
        r"if\(g_pilot_real\)\{",
        r'''static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out){
    uint64_t m5_barrier0=(g_pilot_real&&g_pilot_metrics)?m5_metric_now_ns():0;
    if(g_pilot_real){''',
        "MoE barrier start",
        re.S,
    )
    text = sub_once(
        text,
        r"(\n\s*}\n\s*Cfg \*c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;)",
        r'''\n    }
    if(m5_barrier0) m5_metric_add(&g_pm_barrier_ns,m5_metric_now_ns()-m5_barrier0);
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;''',
        "MoE barrier end",
        re.S,
    )

    hit_pattern = (
        r"for\(int z=0;z<nn;z\+\+\) if\(Sl\[z\]\.eid==eid\)\{ m->hits\+\+; "
        r"Sl\[z\]\.used=\(uint64_t\)__atomic_add_fetch\(&m->eclock,1,__ATOMIC_RELAXED\); "
        r"use\[j\]=&Sl\[z\]; break; \}"
    )
    hit_repl = r'''for(int z=0;z<nn;z++) if(Sl[z].eid==eid){
                    m->hits++;
                    if(Sl[z].pilot_prefetched){
                        Sl[z].pilot_prefetched=0;
                        m5_metric_add(&g_pm_useful,1);
                    }
                    Sl[z].used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED);
                    use[j]=&Sl[z]; break;
                }'''
    text = sub_once(text, hit_pattern, hit_repl, "pilot useful hit")

    promote_pattern = (
        r"ESlot tmp=\*dst; \*dst=m->ws\[q\]; m->ws\[q\]=tmp; "
        r"dst->used=\(uint64_t\)__atomic_add_fetch\(&m->eclock,1,__ATOMIC_RELAXED\);"
    )
    promote_repl = r'''if(dst->pilot_prefetched) m5_metric_add(&g_pm_wasted,1);
              ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp;
              dst->pilot_prefetched=0; m->ws[q].pilot_prefetched=0;
              dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED);'''
    text = sub_once(text, promote_pattern, promote_repl, "normal LRU pilot eviction")

    text = sub_once(
        text,
        r"if\(layer <= atomic_load_explicit\(&g_cur_moe_layer,memory_order_acquire\)\)\{\s*"
        r"atomic_fetch_add_explicit\(&g_pilot_drops,1,memory_order_relaxed\);",
        r'''if(layer <= atomic_load_explicit(&g_cur_moe_layer,memory_order_acquire)){
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
        m5_metric_add(&g_pm_late,1);''',
        "late pilot drop",
        re.S,
    )

    text = sub_once(
        text,
        r"for\(int z=0;z<m->npin\[layer\];z\+\+\) if\(P\[z\]\.eid==eid\)\{ pthread_mutex_unlock\(&g_pilot_mx\); return; \}",
        r'''for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){
        m5_metric_add(&g_pm_race_skip,1); pthread_mutex_unlock(&g_pilot_mx); return;
    }''',
        "pilot pinned race skip",
    )
    text = sub_once(
        text,
        r"for\(int z=0;z<nn;z\+\+\) if\(Sl\[z\]\.eid==eid\)\{ pthread_mutex_unlock\(&g_pilot_mx\); return; \}",
        r'''for(int z=0;z<nn;z++) if(Sl[z].eid==eid){
        m5_metric_add(&g_pm_race_skip,1); pthread_mutex_unlock(&g_pilot_mx); return;
    }''',
        "pilot cache race skip",
    )

    text = sub_once(
        text,
        r"ESlot \*dst=&Sl\[slot\];\s*dst->eid=-1;",
        r'''ESlot *dst=&Sl[slot];
    if(!isnew){
        m5_metric_add(&g_pm_evictions,1);
        if(dst->pilot_prefetched) m5_metric_add(&g_pm_wasted,1);
    }
    dst->pilot_prefetched=0;
    dst->eid=-1;''',
        "pilot eviction accounting",
        re.S,
    )

    text = sub_once(
        text,
        r"int rc=expert_load\(m,layer,eid,dst,0\);",
        r'''uint64_t m5_load0=g_pilot_metrics?m5_metric_now_ns():0;
    int rc=expert_load(m,layer,eid,dst,0);
    if(m5_load0) m5_metric_add(&g_pm_load_ns,m5_metric_now_ns()-m5_load0);''',
        "pilot load timing",
    )
    text = sub_once(
        text,
        r"(if\(rc==0\)\{\s*dst->used=\(uint64_t\)__atomic_add_fetch\(&m->eclock,1,__ATOMIC_RELAXED\);)",
        r'''if(rc==0){
        dst->pilot_prefetched=1;
        dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED);''',
        "pilot publish marker",
        re.S,
    )

    predicted_anchor = "            ch[best]=-2e30f;\n"
    if predicted_anchor not in text:
        raise RuntimeError("pilot prediction anchor not found")
    text = text.replace(
        predicted_anchor,
        predicted_anchor + "            m5_metric_add(&g_pm_predicted,1);\n",
        1,
    )

    queue_pattern = r'''            if\(!found\)\{\s*
                unsigned w=__atomic_load_n\(&pilot_w,__ATOMIC_RELAXED\);\s*
                if\(w-__atomic_load_n\(&pilot_r,__ATOMIC_ACQUIRE\)<4096\)\{\s*
                    pilot_q\[w&4095\]\.l=lnext; pilot_q\[w&4095\]\.e=best;\s*
                    __atomic_store_n\(&pilot_w,w\+1,__ATOMIC_RELEASE\);\s*
                }\s*
            }'''
    queue_repl = r'''            if(found){
                m5_metric_add(&g_pm_resident_skip,1);
            } else {
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    pilot_q[w&4095].l=lnext; pilot_q[w&4095].e=best;
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                    m5_metric_add(&g_pm_enqueued,1);
                } else m5_metric_add(&g_pm_queue_full,1);
            }'''
    text = sub_once(text, queue_pattern, queue_repl, "pilot queue accounting", re.S)

    env_pattern = r"(if\(g_pilot_k<1\) g_pilot_k=1;)"
    env_repl = r'''\1
    { const char *pm=ili_env("PILOT_METRICS");
      g_pilot_metrics=pm&&atoi(pm)!=0; }'''
    text = sub_once(text, env_pattern, env_repl, "metrics environment")

    profile_anchor = "#endif\n}\n\n/* Fixed-token decode benchmark:"
    report = r'''#endif
    if(g_pilot_metrics){
        uint64_t predicted=atomic_load_explicit(&g_pm_predicted,memory_order_relaxed);
        uint64_t enqueued=atomic_load_explicit(&g_pm_enqueued,memory_order_relaxed);
        uint64_t useful=atomic_load_explicit(&g_pm_useful,memory_order_relaxed);
        uint64_t wasted=atomic_load_explicit(&g_pm_wasted,memory_order_relaxed);
        uint64_t loads=(uint64_t)atomic_load_explicit(&g_pilot_loads,memory_order_relaxed);
        printf("PILOT-METRICS: predicted %llu | enqueued %llu | resident-skip %llu | race-skip %llu | queue-full %llu\n",
            (unsigned long long)predicted,(unsigned long long)enqueued,
            (unsigned long long)atomic_load_explicit(&g_pm_resident_skip,memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&g_pm_race_skip,memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&g_pm_queue_full,memory_order_relaxed));
        printf("PILOT-OUTCOME: loads %llu | useful %llu | wasted %llu | late %llu | evictions %llu | precision %.1f%%\n",
            (unsigned long long)loads,(unsigned long long)useful,(unsigned long long)wasted,
            (unsigned long long)atomic_load_explicit(&g_pm_late,memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&g_pm_evictions,memory_order_relaxed),
            loads?100.0*(double)useful/(double)loads:0.0);
        printf("PILOT-TIME: load %.3fs | layer-barrier %.3fs | blocked-pipe %.3fs (%llu waits)\n",
            atomic_load_explicit(&g_pm_load_ns,memory_order_relaxed)/1e9,
            atomic_load_explicit(&g_pm_barrier_ns,memory_order_relaxed)/1e9,
            atomic_load_explicit(&g_pm_pipe_wait_ns,memory_order_relaxed)/1e9,
            (unsigned long long)atomic_load_explicit(&g_pm_pipe_wait_calls,memory_order_relaxed));
    }
}

/* Fixed-token decode benchmark:'''
    if profile_anchor not in text:
        raise RuntimeError("profile report anchor not found")
    text = text.replace(profile_anchor, report, 1)

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(patch_text(args.source.read_text()))


if __name__ == "__main__":
    main()
