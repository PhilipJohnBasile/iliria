#!/usr/bin/env python3
"""Add opt-in per-layer decode stall tracing to generated M5 Max C.

This is the arrival-order-overlap (a2, docs/performance-theory.json) go/no-go
INSTRUMENT: a falsifier needs real timestamps, not a model, to answer "how
much of today's miss latency is already hidden behind resident-expert
compute." Activated at runtime with ILI_STALL_TRACE=1 (or COLI_/FA_ legacy
prefixes via ili_env, same convention as every other opt-in feature here).
The production source (glm.c) stays untouched -- this patch is applied only
to the generated/lab engine variant, same convention as
patch_m5max_route_trace.py and patch_m5max_pilot_metrics.py.

For every DECODE (S==1) MoE layer call, moe() now prints one compact stderr
line (see m5_stall_trace_emit below for the exact fields) with:
  - route-ready        : router output resolved, union of unique experts built
  - resident_start/finish : the CPU compute window this block's RESIDENT
                        (hit) experts ran in
  - miss_issue         : when the block's missed-expert reads were issued
                        (shared timestamp: today's engine issues a block's
                        misses as one batch, whether via PIPE's async
                        dispatch or the blocking parallel-for)
  - miss_complete_max  : the SLOWEST missed expert's read completion (tracked
                        per-expert, not just per-block)
  - reduction_start    : the first down-projection (the step whose output
                        gets weighted-accumulated into `out`) -- see "Two
                        anchor variants" below for where this is (and is
                        not) resolvable
  - exposed_stall_ms   : max(0, miss_complete_max - resident_finish) -- the
                        derived number the a2 falsifier's kill/go rule reads

All five timing fields plus exposed_stall_ms are the layer-opportunity
matrix's "pending instrumentation" columns
(bench-m5max/offline-replay-20260715/layer-opportunity-matrix.csv);
tools/parse_stall_trace.py aggregates repeated STALL_TRACE lines into that
matrix's column set (written to a NEW file -- the v1 is never edited).

Two anchor variants, chosen automatically (read, don't guess, which applies):
gen_m5max_engine.py's own main() unconditionally chains
tools/patch_m5max_grouped_cpu_moe.py, which REWRITES the per-expert compute
loop this patch targets into a call to a new m5_cpu_moe_subset() helper that
processes an entire block's resident+miss experts together (grouped,
tiled, OpenMP-parallel across experts at once) -- the m5max chain's actual,
always-applied structure, not an edge case. That structure has no per-
expert compute boundary to hook a per-expert "first down-projection" instant
from, so under it reduction_start is honestly reported as unavailable
(-1 in the emitted line) and resident_start/finish bracket the WHOLE
block's CPU compute call (hit+miss fused -- grouping computes them
together), not a hit-only window. Applied directly to a pristine glm.c (no
grouped-CPU-MoE rewrite ever ran), the original per-expert loop is still
intact, and this patch instruments it at full per-expert, hit-vs-miss
granularity, including a real reduction_start. Both variants share one
sentinel convention (a captured timestamp is a real value >0; anything
never captured this block prints as -1 in the emitted line) and the exact
same STALL_TRACE line format, so tools/parse_stall_trace.py needs no
knowledge of which variant produced a given trace.

Mixed-format guard (glm.c's MB_BUILD, see "never submit a mixed-format
expert block to Metal as one scalar fmt"): a 64-expert union block can now
resolve on Metal per-format-sub-block rather than all-or-nothing, so the
old block-level `metal_done`/`cpu_res`/`cpu_miss` booleans this patch used
to key off of are gone from glm.c, replaced by a per-expert `handled[]`
mask (an expert is `handled` iff some per-format Metal sub-block actually
computed it; anything left over -- format-inconsistent, an unrepresented
format, or a failed GPU submit -- falls through to the CPU). This patch
does not need to read `handled[]` itself (glm.c's own `#ifdef ILI_METAL
if(g_metal_enabled && handled[j]) continue;` skip is untouched, part of
the anchor text): it only needs a stand-in for "did metal fully resolve
this block, i.e. is there nothing left for this instrument to time" --
variant 1 tracks that live as `m5_any_cpu` (set the first time the CPU
loop actually reaches compute for some j; never set at all iff every j
was skipped, the exact `handled[j]==1 for all j` case the old
`metal_done` captured); variant 2 reads it off m5_cpu_moe_subset()'s own
return value (routed-row count, 0 iff its mask filtered out every
expert). Same chain shape, same rewrite, carried into
patch_m5max_grouped_cpu_moe.py + fix_m5max_grouped_build.py: the M5 max
generated engine's m5_cpu_moe_subset() now takes the `handled` mask
directly (as `m5_mask`) instead of the old per-category
(is_miss/take_res/take_miss) selection.

Scope, stated not hidden: only the CPU compute path is instrumented; a
block every expert of which is resolved by a Metal command buffer emits no
line for that layer/token (nothing for this instrument to time -- see
`m5_any_cpu` / `m5_moe_rows` above). The timestamp model here does not
(yet) map onto the Metal begin/end block structure itself (a separate,
future extension). Fixture validation (no GPU) always takes the CPU path,
so this is the falsifier's primary, fully-covered regime today.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


HELPER = r'''/* ---- M5 stall trace (opt-in, a2 arrival-order-overlap falsifier) ------------
 * ILI_STALL_TRACE=1: for every DECODE (S==1) MoE layer call that resolves on
 * the CPU compute path, print one compact stderr line with route-ready,
 * resident-expert compute start/finish, per-miss read issue + completion,
 * and reduction-start, as milliseconds relative to route-ready, plus the
 * derived metric this instrument exists to answer:
 *     exposed_stall_ms = max(0, last_miss_complete - resident_finish) * 1000
 * i.e. how much of the slowest miss's read is NOT already hidden behind the
 * resident experts' compute. Zero overhead when unset (one cached env check
 * per process). Read-only timing around the EXISTING miss-resolve /
 * dispatch / compute path -- no scheduling, dispatch order, or output change
 * of any kind. One line per LAYER per TOKEN: S==1 decode always has exactly
 * one <=64-expert block (nu<=topk<=64), tagged by m->n_fw (stable across a
 * token's layers; incremented once the whole token's forward returns, so
 * every layer of one token shares one value).
 *
 * Sentinel convention: a timestamp field is captured (>0, a real
 * CLOCK_MONOTONIC reading) or not (0, printed as -1 in the emitted line --
 * see tools/patch_m5max_stall_trace.py module docstring, "Two anchor
 * variants," for when resident_start/finish and reduction_start are and are
 * not resolvable at per-expert granularity). */
static int g_stall_trace=0, g_stall_trace_init=0;
static void m5_stall_trace_init(void){
    if(g_stall_trace_init) return;
    g_stall_trace_init=1;
    const char *v=ili_env("STALL_TRACE");
    g_stall_trace=v&&atoi(v)!=0;
    if(g_stall_trace) fprintf(stderr,"[stall-trace] enabled: one STALL_TRACE line per decode layer per token to stderr\n");
}
static void m5_stall_trace_emit(Model *m,int layer,int nu,int nhit,int nmiss,
        double route_ready,double resident_start,double resident_finish,
        double miss_issue,const double *miss_complete,double reduction_start){
    double miss_complete_max=0; int have_miss=0;
    for(int q=0;q<nmiss;q++){
        if(!have_miss || miss_complete[q]>miss_complete_max) miss_complete_max=miss_complete[q];
        have_miss=1;
    }
    int have_resident = resident_start>0 && resident_finish>0;
    double rf = have_resident ? resident_finish : route_ready;  /* no resident window: nothing hides the miss */
    double exposed_ms = (have_miss && (miss_complete_max-rf)>0) ? (miss_complete_max-rf)*1000.0 : 0.0;
    fprintf(stderr,
        "STALL_TRACE fwd=%llu layer=%d nu=%d nhit=%d nmiss=%d "
        "resident_start_ms=%.4f resident_finish_ms=%.4f miss_issue_ms=%.4f "
        "miss_complete_max_ms=%.4f reduction_start_ms=%.4f exposed_stall_ms=%.4f\n",
        (unsigned long long)m->n_fw,layer,nu,nhit,nmiss,
        resident_start>0?(resident_start-route_ready)*1000.0:-1.0,
        resident_finish>0?(resident_finish-route_ready)*1000.0:-1.0,
        miss_issue>0?(miss_issue-route_ready)*1000.0:-1.0,
        have_miss?(miss_complete_max-route_ready)*1000.0:-1.0,
        reduction_start>0?(reduction_start-route_ready)*1000.0:-1.0,
        exposed_ms);
}

'''

# ---- Anchor A: right after the union-of-unique-experts loop (FASE B), before
# FASE C/D's per-64-block resolve loop begins -- this IS "route-ready": the
# router's output for this token/layer is fully resolved here. Untouched by
# gen_m5max_engine.py / patch_m5max_grouped_cpu_moe.py / patch_m5max_route_trace.py /
# patch_m5max_pilot_metrics.py in either anchor variant.
ROUTE_READY_ANCHOR = """        if(!seen[e]){ seen[e]=1; uniq[nu++]=e; }
    }
    /* ---- FASE C/D: risolvi (pin/cache/disco) e calcola, a blocchi di 64 unici ---- */
"""
ROUTE_READY_REPL = """        if(!seen[e]){ seen[e]=1; uniq[nu++]=e; }
    }
    m5_stall_trace_init();
    int m5_trace_on = g_stall_trace && S==1;
    double m5_route_ready = m5_trace_on ? now_s() : 0;
    /* ---- FASE C/D: risolvi (pin/cache/disco) e calcola, a blocchi di 64 unici ---- */
"""

# ---- Anchor B: right after hit/miss resolution, immediately before the
# (unconditional) "#ifdef ILI_METAL" MB_BUILD/mixed-format-guard region --
# declares this block's trace scratch (block-scoped, so a fresh set of
# timestamps every base+=64 iteration; S==1 decode only ever runs this
# once). The "did metal fully resolve this block" stand-in for the old
# `metal_done` (see module docstring) is declared per-variant instead of
# here (m5_any_cpu in COMPUTE_REPL_PLAIN, m5_moe_rows in
# COMPUTE_REPL_GROUPED) since only one variant's compute anchor ever
# applies to a given source -- declaring both here would leave the unused
# one an unused-variable warning. Untouched by every transform in both
# anchor variants.
SCRATCH_ANCHOR = """            if(!use[j]){ qof[j]=nmiss; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++; }
        }
#ifdef ILI_METAL
"""
SCRATCH_REPL = """            if(!use[j]){ qof[j]=nmiss; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++; }
        }
        double m5_miss_issue_ts=0, m5_miss_complete_ts[64]={0};
        double m5_resident_start_ts=0, m5_resident_finish_ts=0, m5_reduction_start_ts=0;
#ifdef ILI_METAL
"""

# ---- Anchor C: the miss dispatch block (PIPE async vs blocking parallel-for).
# miss_issue is one shared timestamp (both paths issue the whole block's
# misses at once); miss_complete is captured per-expert here for the
# non-PIPE blocking loop (each omp iteration owns a distinct q, so no lock is
# needed). Untouched by patch_m5max_grouped_cpu_moe.py in either variant.
DISPATCH_ANCHOR = """        if(nmiss){
            int eids[64]; for(int q=0;q<nmiss;q++) eids[q]=uniq[base+missk[q]];
            io_trace_log(layer,eids,nmiss);   /* debug proof-of-single-variable, ILI_IO_TRACE only */
            if(g_pipe){                            /* PIPE: launch loads async, matmul overlaps them */
                if(!g_pp.started) pipe_init(m);
                double t0=now_s();
                pipe_dispatch(m,layer,eids,nmiss);
                m->t_edisk += now_s()-t0;           /* dispatch only; real reads hide behind matmul */
            } else { double t0=now_s();             /* ORIGINALE: blocking parallel load */
                if(g_a2_on){ g_a2_issue=t0; g_a2_nmiss=(nmiss>64?64:nmiss); }
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++){ expert_load(m,layer,uniq[base+missk[q]],&m->ws[q],1);
                    if(g_a2_on && q<64) g_a2_comp[q]=now_s(); }
                double ddt=now_s()-t0; m->t_edisk += ddt;
                if(g_a2_on){ g_a2_load_end=now_s(); g_a2_nb=nb; g_a2_bytes=(int64_t)g_a2_nmiss*18915328; }
                /* No PIPE = no overlap mechanism at all here: this whole span is, by
                 * construction, consumer-blocked-because-data-unavailable, so it is fully
                 * exposed stall too (t_stall_exposed mirrors t_edisk exactly in this mode). */
                m->t_stall_exposed += ddt; }
        }
"""
DISPATCH_REPL = """        if(nmiss){
            int eids[64]; for(int q=0;q<nmiss;q++) eids[q]=uniq[base+missk[q]];
            io_trace_log(layer,eids,nmiss);   /* debug proof-of-single-variable, ILI_IO_TRACE only */
            if(m5_trace_on) m5_miss_issue_ts=now_s();
            if(g_pipe){                            /* PIPE: launch loads async, matmul overlaps them */
                if(!g_pp.started) pipe_init(m);
                double t0=now_s();
                pipe_dispatch(m,layer,eids,nmiss);
                m->t_edisk += now_s()-t0;           /* dispatch only; real reads hide behind matmul */
            } else { double t0=now_s();             /* ORIGINALE: blocking parallel load */
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++){ expert_load(m,layer,uniq[base+missk[q]],&m->ws[q],1);
                    if(m5_trace_on) m5_miss_complete_ts[q]=now_s(); }
                double ddt=now_s()-t0; m->t_edisk += ddt;
                /* No PIPE = no overlap mechanism at all here: this whole span is, by
                 * construction, consumer-blocked-because-data-unavailable, so it is fully
                 * exposed stall too (t_stall_exposed mirrors t_edisk exactly in this mode). */
                m->t_stall_exposed += ddt; }
        }
"""

# ---- Anchor D, variant 1: a PRISTINE glm.c (no grouped-CPU-MoE rewrite ever
# ran) -- the original serial per-expert compute loop, now gated per-expert
# by the mixed-format guard's `handled[]` mask (not the removed
# is_miss/cpu_res/cpu_miss booleans -- see module docstring) instead of a
# block-level `if(!metal_done)` around the whole loop (glm.c no longer has
# one; every j is visited, `handled[j]` just makes GPU-done ones a no-op).
# Full per-expert, hit-vs-miss granularity: pipe_wait's own return is a
# miss's true completion under PIPE=1; resident_start/finish bracket only
# the qof[j]<0 (resident/hit) iterations; reduction_start is the first
# down-projection across every expert in the block, hit or miss;
# m5_any_cpu is set the first time ANY expert (hit or miss) actually
# reaches compute, standing in for the old `!metal_done` at the emit gate.
COMPUTE_ANCHOR_PLAIN = """        if(g_a2_on) g_a2_cs=now_s();
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
            /* Drain this miss's async load BEFORE the nr==0 early-exit below: every
             * dispatched slot must be waited before the end-of-block LRU swap can reuse
             * its ws[] slab, so correctness does not depend on the nr>=1 routing invariant.
             * Stays ABOVE the METAL skip: a subset that fell back to the CPU still needs its
             * slot drained here, and under METAL the block-level drain above already ran (this
             * spin is then a no-op). */
            if(g_pipe && qof[j]>=0){ double tw=now_s();
                m->t_stall_exposed += pipe_wait_timed(qof[j],&m->n_pipe_waits,&m->n_pipe_waits_blocked);
                m->t_edisk += now_s()-tw; }
#ifdef ILI_METAL
            /* skip experts already computed on GPU by one of the per-format sub-blocks above */
            if(g_metal_enabled && handled[j]) continue;
#endif
            int nr=0;                                 /* righe (posizioni) che usano questo expert */
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
#ifdef ILI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
#endif
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
            matmul_qt(gg, xg, &e->g, nr);
            matmul_qt(uu, xg, &e->u, nr);
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
            matmul_qt(hh, gg, &e->d, nr);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            m->t_emm += now_s()-t0;
        }
"""
COMPUTE_REPL_PLAIN = """        int m5_any_cpu=0;
        if(g_a2_on) g_a2_cs=now_s();
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
            /* Drain this miss's async load BEFORE the nr==0 early-exit below: every
             * dispatched slot must be waited before the end-of-block LRU swap can reuse
             * its ws[] slab, so correctness does not depend on the nr>=1 routing invariant.
             * Stays ABOVE the METAL skip: a subset that fell back to the CPU still needs its
             * slot drained here, and under METAL the block-level drain above already ran (this
             * spin is then a no-op). */
            if(g_pipe && qof[j]>=0){ double tw=now_s(); pipe_wait(qof[j]);
                double tc=now_s(); m->t_edisk += tc-tw;
                if(m5_trace_on) m5_miss_complete_ts[qof[j]]=tc; }
#ifdef ILI_METAL
            /* skip experts already computed on GPU by one of the per-format sub-blocks above */
            if(g_metal_enabled && handled[j]) continue;
#endif
            int nr=0;                                 /* righe (posizioni) che usano questo expert */
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
#ifdef ILI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
#endif
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
            if(m5_trace_on) m5_any_cpu=1;
            if(m5_trace_on && qof[j]<0){ if(!m5_resident_start_ts) m5_resident_start_ts=t0; m5_resident_finish_ts=t0; }
            matmul_qt(gg, xg, &e->g, nr);
            matmul_qt(uu, xg, &e->u, nr);
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
            if(m5_trace_on && !m5_reduction_start_ts) m5_reduction_start_ts=now_s();
            matmul_qt(hh, gg, &e->d, nr);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            m->t_emm += now_s()-t0;
            if(m5_trace_on && qof[j]<0) m5_resident_finish_ts=now_s();
        }
        if(m5_trace_on && m5_any_cpu) m5_stall_trace_emit(m,layer,nb,nb-nmiss,nmiss,
            m5_route_ready,m5_resident_start_ts,m5_resident_finish_ts,
            m5_miss_issue_ts,m5_miss_complete_ts,m5_reduction_start_ts);
"""

# ---- Anchor D, variant 2: post patch_m5max_grouped_cpu_moe.py (the ALWAYS-
# applied structure of the real m5max chain, since gen_m5max_engine.py's own
# main() chains that patch unconditionally). The per-expert loop is gone,
# replaced by one call to m5_cpu_moe_subset() that processes the block's
# resident+miss experts together (grouped/tiled/OpenMP-parallel across
# experts at once, or a per-expert fallback INSIDE that helper when grouping
# is unsupported/disabled -- either way, moe() itself has no per-expert hook
# any more). m5_cpu_moe_subset() now takes the mixed-format guard's
# `handled[]` mask directly (as m5_mask, when Metal is enabled and not
# forcing all-CPU) rather than the old per-category take_res/take_miss
# selection. miss_complete is captured in the hoisted upfront pipe_wait
# drain loop (also moved by that same patch); resident_start/finish bracket
# the WHOLE block's CPU compute call; reduction_start has no per-expert
# instant to bind to here and is honestly left uncaptured (-1 in the
# emitted line) -- see the module docstring's "Two anchor variants".
# m5_moe_rows captures the call's own return (routed-row count) as the
# stand-in for the old `!metal_done` at the emit gate: 0 iff the mask
# filtered out every expert this block (metal handled all of it).
COMPUTE_ANCHOR_GROUPED = """        /* Drain every dispatched miss before any CPU consumer or LRU promotion. */
        if(g_pipe) for(int j=0;j<nb;j++) if(qof[j]>=0){
            double tw=now_s(); pipe_wait(qof[j]); m->t_edisk += now_s()-tw;
        }
        {
            const int *m5_mask=NULL;
#ifdef ILI_METAL
            if(g_metal_enabled && !m5_cpu_all_experts()) m5_mask=handled;
#endif
            double t0=now_s();
            m5_cpu_moe_subset(use,uniq+base,nb,m5_mask,
                              x,S,D,I,K,idxs,ws,keff,out);
            m->t_emm += now_s()-t0;
        }
"""
COMPUTE_REPL_GROUPED = """        /* Drain every dispatched miss before any CPU consumer or LRU promotion. */
        if(g_pipe) for(int j=0;j<nb;j++) if(qof[j]>=0){
            double tw=now_s(); pipe_wait(qof[j]);
            double tc=now_s(); m->t_edisk += tc-tw;
            if(m5_trace_on) m5_miss_complete_ts[qof[j]]=tc;
        }
        int m5_moe_rows=0;
        {
            const int *m5_mask=NULL;
#ifdef ILI_METAL
            if(g_metal_enabled && !m5_cpu_all_experts()) m5_mask=handled;
#endif
            double t0=now_s();
            if(m5_trace_on) m5_resident_start_ts=t0;
            m5_moe_rows=m5_cpu_moe_subset(use,uniq+base,nb,m5_mask,
                              x,S,D,I,K,idxs,ws,keff,out);
            m->t_emm += now_s()-t0;
            if(m5_trace_on) m5_resident_finish_ts=now_s();
        }
        if(m5_trace_on && m5_moe_rows>0) m5_stall_trace_emit(m,layer,nb,nb-nmiss,nmiss,
            m5_route_ready,m5_resident_start_ts,m5_resident_finish_ts,
            m5_miss_issue_ts,m5_miss_complete_ts,m5_reduction_start_ts);
"""


def patch_text(text: str) -> str:
    marker = """/* MoE GLM su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, + shared expert).
"""
    text = replace_once(text, marker, HELPER + marker, "stall trace helper insertion")
    text = replace_once(text, ROUTE_READY_ANCHOR, ROUTE_READY_REPL, "route-ready capture")
    text = replace_once(text, SCRATCH_ANCHOR, SCRATCH_REPL, "block trace scratch")
    text = replace_once(text, DISPATCH_ANCHOR, DISPATCH_REPL, "miss dispatch timing")

    has_plain = COMPUTE_ANCHOR_PLAIN in text
    has_grouped = COMPUTE_ANCHOR_GROUPED in text
    if has_plain and has_grouped:
        raise RuntimeError(
            "per-expert compute anchor: BOTH the plain and grouped-CPU-MoE "
            "forms matched -- ambiguous input, refusing to guess")
    if has_plain:
        text = replace_once(text, COMPUTE_ANCHOR_PLAIN, COMPUTE_REPL_PLAIN,
                            "per-expert compute + emit (plain glm.c form)")
    elif has_grouped:
        text = replace_once(text, COMPUTE_ANCHOR_GROUPED, COMPUTE_REPL_GROUPED,
                            "per-expert compute + emit (grouped-CPU-MoE form)")
    else:
        raise RuntimeError(
            "per-expert compute anchor: neither the plain glm.c form nor the "
            "patch_m5max_grouped_cpu_moe.py-rewritten form was found -- "
            "source does not match either known moe() shape")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(patch_text(args.source.read_text()))


if __name__ == "__main__":
    main()
