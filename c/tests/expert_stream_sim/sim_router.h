/* sim_router.h -- memoryless synthetic router + the decode-step driver that
 * ties SimCache + sim_expert_load + NvmeSim together.
 *
 * The decode-step driver (sim_layer_step) reproduces glm.c's moe() resolve
 * loop (~L1860-2061) at decode granularity (one token, S=1, so the union of
 * routed experts per layer is at most topk<=n_experts -- this engine's real
 * per-block chunk size is 64, and n_experts=256 for GLM-5.2, so a single
 * decode step's unique-expert union NEVER needs the multi-block chunking
 * moe() does for large prefill batches; that chunking (the outer `for(base=
 * 0;base<nu;base+=64)` loop) is therefore or out of scope here -- see the
 * file-level note in test_expert_stream_sim.c's header for the explicit
 * scope statement). Order of operations, each mirrored 1:1 against a cited
 * glm.c line range:
 *   1. dedupe routed experts into a unique set                  (~L1850-1855)
 *   2. bump usage/heat for every uniquely-routed expert          (~L1788,1816)
 *   3. resolve each: pin hit / LRU hit / miss                    (~L1862-1868)
 *   4. dispatch all misses across the worker pool                (~L1948-1966)
 *   5. sequential per-expert wait-then-compute walk in resolve
 *      order, mirroring pipe_wait_timed()'s "0 if already ready,
 *      else the residual wait" semantics                        (~L2020-2049)
 *   6. promote up to min(nmiss,ecap) loaded experts into the LRU,
 *      in REVERSE dispatch order, never promoting a failed load  (~L2055-2060)
 */
#ifndef SIM_ROUTER_H
#define SIM_ROUTER_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sim_cache.h"
#include "sim_expert_io.h"
#include "sim_nvme.h"

/* ---- tiny deterministic PRNG (xorshift32, same construction as
 * c/tests/test_idot.c's rng_state/xr()) -- reproducible fuzzing without any
 * libc PRNG-quality/seeding surprises. ---- */
typedef struct { uint32_t state; } SimRng;
static void sim_rng_seed(SimRng *r, uint32_t seed){ r->state = seed?seed:0x9e3779b9u; }
static uint32_t sim_rng_next(SimRng *r){
    r->state ^= r->state<<13; r->state ^= r->state>>17; r->state ^= r->state<<5;
    return r->state;
}
static double sim_rng_uniform01(SimRng *r){ return (sim_rng_next(r)&0xFFFFFFu)/(double)0x1000000u; }

typedef enum { SIM_ROUTE_UNIFORM=0, SIM_ROUTE_ZIPF=1 } SimRouteMode;

/* MEMORYLESS BY CONSTRUCTION: every sim_router_pick() call draws `topk`
 * distinct experts i.i.d. from a FIXED marginal distribution, with NO
 * reference to any previous call's result. This stands in for the project's
 * settled finding that GLM-5.2's real routing is memoryless across tokens
 * (task framing: "cross-token caching is proven dead"). "Memoryless" here
 * means i.i.d. draws from a STABLE marginal -- compatible with, and in fact
 * required to usefully test, a persistently skewed per-expert popularity
 * profile (exactly what the real engine's .fa_usage histogram + pin/
 * hot-store mechanism bets exists). SIM_ROUTE_UNIFORM is the null model (no
 * exploitable skew -- LRU hit rate should track cache-capacity/n_experts and
 * nothing else); SIM_ROUTE_ZIPF is the positive control (skew exists --
 * pin+LRU hit rate should measurably beat the capacity-only baseline).
 * ASSUMPTION (explicit, out of scope to remove): this does NOT reimplement
 * GLM-5.2's real router (a learned gate matmul + top-k over 256 logits,
 * glm.c's moe() ~L1780-1820) -- only its two claimed statistical properties
 * (memoryless in time; possibly skewed in the marginal) are modeled. */
typedef struct {
    int n_experts, topk;
    SimRouteMode mode;
    double zipf_s;
    double *weight;
} SimRouter;

static void sim_router_init(SimRouter *rt, int n_experts, int topk, SimRouteMode mode, double zipf_s){
    rt->n_experts=n_experts; rt->topk=topk; rt->mode=mode; rt->zipf_s = zipf_s>0?zipf_s:1.0;
    rt->weight=malloc((size_t)n_experts*sizeof(double));
    for(int i=0;i<n_experts;i++)
        rt->weight[i] = (mode==SIM_ROUTE_ZIPF) ? 1.0/pow((double)(i+1), rt->zipf_s) : 1.0;
}
static void sim_router_free(SimRouter *rt){ free(rt->weight); rt->weight=NULL; }

/* Weighted sampling WITHOUT replacement, O(topk*n_experts) -- fine at
 * n_experts=256. Writes exactly min(topk,n_experts) ids into out[]. */
static int sim_router_pick(SimRouter *rt, SimRng *rng, int *out){
    int n=rt->n_experts, k=rt->topk<n?rt->topk:n;
    double *w = malloc((size_t)n*sizeof(double));
    int    *id= malloc((size_t)n*sizeof(int));
    memcpy(w, rt->weight, (size_t)n*sizeof(double));
    for(int i=0;i<n;i++) id[i]=i;
    int remaining=n;
    for(int pick=0; pick<k; pick++){
        double total=0; for(int i=0;i<remaining;i++) total+=w[i];
        double u = total>0 ? sim_rng_uniform01(rng)*total : 0.0;
        int sel=remaining-1; double acc=0;
        for(int i=0;i<remaining;i++){ acc+=w[i]; if(u<=acc){ sel=i; break; } }
        out[pick]=id[sel];
        w[sel]=w[remaining-1]; id[sel]=id[remaining-1]; remaining--;
    }
    free(w); free(id);
    return k;
}

/* Populates layer `l`'s pin (hot-store) set by actually LOADING each id from
 * `nvme` (fatal=1 semantics) -- mirrors the real engine's AUTOPIN-at-startup
 * behavior (glm.c reads the persisted .fa_usage histogram and loads each
 * chosen expert once into m->pin[] before serving a single token). All-or-
 * nothing: if ANY load fails, NO pins are committed and 0 is returned --
 * mirrors the fatal=1 contract (a real failed startup pin-load calls
 * exit(1) before the server ever comes up, so there is no such thing as a
 * "partially populated hot-store" to fall back to). */
static int sim_cache_populate_pins(SimCache *c, NvmeSim *nvme, SimFaultProgram *faults,
                                    IoByteCounters *io, double t_issue,
                                    const ExpertShardSizes *shard, const ExpertShape *shape,
                                    int l, const int *eids, int n){
    SimSlot *tmp = malloc((size_t)(n>0?n:1)*sizeof(SimSlot));
    for(int z=0;z<n;z++){
        ExpertLoadResult r = sim_expert_load(nvme, 0, t_issue, io, faults, l, eids[z], shard, 1,
                                              shape->O_gate,shape->I_gate,shape->O_up,shape->I_up,
                                              shape->O_down,shape->I_down);
        if(!r.ok){ free(tmp); return 0; }
        tmp[z].eid=eids[z]; tmp[z].used=0; tmp[z].bytes=r.declared_bytes;
    }
    free(c->pin[l]);
    c->pin[l]=tmp; c->npin[l]=n;
    return 1;
}

/* ---- runtime: global counters across a whole simulated session ---- */
typedef struct {
    int n_workers;
    double per_expert_compute_s;  /* consumer compute time charged per resolved expert
                                    * (hit or miss) -- mirrors the matmul_qt(gate)/matmul_qt(up)/
                                    * matmul_qt(down) span per expert in moe(), glm.c ~L2042-2049 */
    int coalesced_reads;          /* mirrors glm.c's `contig` pread-coalescing fast path */
    double t_now;                 /* virtual clock */
    IoByteCounters io;
    uint64_t n_pipe_waits, n_pipe_waits_blocked;
    double t_stall_exposed;
    uint64_t n_fatal_errors;        /* critical-path load failures -- real engine: exit(1) here */
    uint64_t n_format_findings;     /* format-inference mismatches, oracle-only detection */
    uint64_t n_corruption_findings; /* content corruption, oracle-only detection (see sim_expert_io.h) */
} SimRuntime;

static void sim_runtime_init(SimRuntime *rt, int n_workers, double per_expert_compute_s, int coalesced_reads){
    memset(rt,0,sizeof(*rt));
    rt->n_workers = n_workers<1?1:n_workers;
    rt->per_expert_compute_s = per_expert_compute_s>=0 ? per_expert_compute_s : 0.0;
    rt->coalesced_reads = coalesced_reads;
}

typedef struct {
    int layer, nu, nmiss;
    int fatal;               /* a critical-path miss failed to load this step */
    int any_short_read;
    int any_format_finding;
    int any_corrupted;
    double stall_added_s;    /* stall exposed BY THIS STEP ALONE (not cumulative) */
} SimLayerStepSummary;

/* Resolves one layer's routed-expert set for one decode step against
 * `cache`, loading misses from `nvme` (subject to `faults`), and advances
 * rt->t_now by the consumer's wait+compute time. `shard`/O_*various/I_*
 * describe the per-expert tensor shapes (see expert_shard_sizes_glm52_int4).
 * `summary` may be NULL. */
static void sim_layer_step(SimRuntime *rt, SimCache *cache, NvmeSim *nvme, SimFaultProgram *faults,
                           const ExpertShardSizes *shard,
                           int64_t O_g,int64_t I_g,int64_t O_u,int64_t I_u,int64_t O_d,int64_t I_d,
                           int layer, const int *routed, int n_routed, SimLayerStepSummary *summary){
    int NE = cache->n_experts;
    int *uniq       = malloc((size_t)NE*sizeof(int));
    unsigned char *seen = calloc((size_t)NE,1);
    int *is_hit     = malloc((size_t)NE*sizeof(int));
    int *miss_pos   = malloc((size_t)NE*sizeof(int));   /* miss_pos[q] = uniq-index of q-th miss */
    ExpertLoadResult *mres = malloc((size_t)NE*sizeof(ExpertLoadResult));
    int nu=0;

    for(int i=0;i<n_routed;i++){
        int e=routed[i];
        if(e>=0 && e<NE && !seen[e]){ seen[e]=1; uniq[nu++]=e; }
    }
    for(int j=0;j<nu;j++) sim_cache_note_routed(cache, layer, uniq[j]);

    int nmiss=0;
    for(int j=0;j<nu;j++){
        int64_t b=0;
        if(sim_cache_lookup(cache, layer, uniq[j], &b)){ is_hit[j]=1; }
        else { is_hit[j]=0; miss_pos[nmiss++]=j; }
    }

    for(int q=0;q<nmiss;q++){
        int eid=uniq[miss_pos[q]];
        int w = q % rt->n_workers;
        mres[q] = sim_expert_load(nvme, w, rt->t_now, &rt->io, faults, layer, eid, shard,
                                   rt->coalesced_reads, O_g,I_g,O_u,I_u,O_d,I_d);
        if(mres[q].format_finding) rt->n_format_findings++;
        if(mres[q].corrupted)      rt->n_corruption_findings++;
    }

    double t_cursor = rt->t_now;
    double stall_before = rt->t_stall_exposed;
    int fatal=0, any_short=0;
    { int q=0;
      for(int j=0;j<nu;j++){
          if(!is_hit[j]){
              rt->n_pipe_waits++;
              if(mres[q].t_complete <= t_cursor){
                  /* fully hidden behind earlier compute in this block -- 0 exposed stall,
                   * exactly pipe_wait_timed()'s "ready[q] already set" fast path */
              } else {
                  rt->n_pipe_waits_blocked++;
                  rt->t_stall_exposed += mres[q].t_complete - t_cursor;
                  t_cursor = mres[q].t_complete;
              }
              if(!mres[q].ok){ fatal=1; rt->n_fatal_errors++; }
              if(mres[q].short_read) any_short=1;
              q++;
          }
          t_cursor += rt->per_expert_compute_s;
      }
    }
    rt->t_now = t_cursor;

    { int promo = nmiss < cache->ecap ? nmiss : cache->ecap;
      for(int a=0;a<promo;a++){
          int qq = nmiss-1-a;
          if(!mres[qq].ok) continue;              /* never promote a failed load */
          int eid = uniq[miss_pos[qq]];
          sim_cache_promote(cache, layer, eid, mres[qq].declared_bytes);
      }
    }

    if(summary){
        summary->layer=layer; summary->nu=nu; summary->nmiss=nmiss; summary->fatal=fatal;
        summary->any_short_read=any_short;
        int fmt_f=0, corr_f=0;
        for(int q=0;q<nmiss;q++){ if(mres[q].format_finding) fmt_f=1; if(mres[q].corrupted) corr_f=1; }
        summary->any_format_finding=fmt_f; summary->any_corrupted=corr_f;
        summary->stall_added_s = rt->t_stall_exposed - stall_before;
    }

    free(uniq); free(seen); free(is_hit); free(miss_pos); free(mres);
}

/* Human-readable summary in the same field names/order as glm.c's own
 * IO-BYTES / STALL-EXPOSED report lines (~L2747-2759), so results here are
 * directly eyeball-comparable to a real run's printout. */
static void sim_report(FILE *f, SimCache *cache, SimRuntime *rt){
    uint64_t tot = cache->hits+cache->misses;
    double hitpct = tot ? 100.0*(double)cache->hits/(double)tot : 0.0;
    fprintf(f, "Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n",
        hitpct, (unsigned long long)cache->hits, (unsigned long long)cache->misses);
    fprintf(f, "STALL-EXPOSED: %.6fs (consumer-blocked critical-path only) | pipe-waits %llu blocked %llu (occupancy %.1f%%)\n",
        rt->t_stall_exposed, (unsigned long long)rt->n_pipe_waits, (unsigned long long)rt->n_pipe_waits_blocked,
        rt->n_pipe_waits ? 100.0*(double)rt->n_pipe_waits_blocked/(double)rt->n_pipe_waits : 0.0);
    fprintf(f, "IO-BYTES: requested %lld | read %lld | reads attempted %llu completed %llu | hits %llu misses %llu (%.1f%% hit)\n",
        (long long)rt->io.bytes_requested, (long long)rt->io.bytes_read,
        (unsigned long long)rt->io.reads_attempted, (unsigned long long)rt->io.reads_completed,
        (unsigned long long)cache->hits, (unsigned long long)cache->misses, hitpct);
    fprintf(f, "FUZZ FINDINGS: fatal-critical-path-errors=%llu format-inference-mismatches(oracle-only)=%llu corruption(oracle-only)=%llu\n",
        (unsigned long long)rt->n_fatal_errors, (unsigned long long)rt->n_format_findings,
        (unsigned long long)rt->n_corruption_findings);
}

#endif /* SIM_ROUTER_H */
