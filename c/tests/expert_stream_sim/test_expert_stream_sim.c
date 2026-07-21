/* test_expert_stream_sim.c -- test + fuzz suite for the expert-stream/cache
 * simulator (sim_nvme.h, sim_cache.h, sim_expert_io.h, sim_router.h).
 *
 * WHY THIS EXISTS: iliria (c/glm.c) streams GLM-5.2's 744B-parameter int4
 * MoE experts from NVMe on demand (measured 13.3-13.5 GB/s); decode is
 * weight-read-bandwidth-bound and expert routing is memoryless across
 * tokens. This suite is a CPU-only, from-scratch model of that streaming +
 * cache subsystem's LOGIC (not its numerics -- no real matmul, no real
 * safetensors bytes, no GPU, no ~359GB container), built by reading
 * c/glm.c + c/st.h + c/tier.h and re-implementing the algorithms, so the
 * logic and its failure handling can be validated deterministically without
 * touching the real loader/matmul path another agent is concurrently
 * editing. Every sim_*.h file cites the exact glm.c line ranges its
 * behavior is ported from.
 *
 * SCOPE, STATED EXPLICITLY:
 *   MODELED (and tested below): pin (hot-store) + per-layer LRU cache with
 *   the engine's exact lookup order/eviction algorithm; a synthetic NVMe
 *   backend calibrated to the measured 13.3-13.5 GB/s with multi-worker
 *   bandwidth sharing; the PIPE-style async-dispatch + sequential per-expert
 *   deadline/stall accounting (mirrors pipe_wait_timed/t_stall_exposed);
 *   byte accounting (requested vs read, attempted vs completed) with the
 *   engine's exact success/failure asymmetry; a memoryless (i.i.d.,
 *   optionally Zipf-skewed) synthetic router; the REPIN hot-store swap
 *   policy (ported from tier.h); fault injection for short reads, wrong-
 *   declared-size shards, content corruption, and stragglers/out-of-order
 *   completions.
 *   NOT MODELED (explicit, deliberate scope cut -- read but not
 *   reimplemented): the real router's gate matmul + top-k (only its two
 *   claimed statistical properties -- memoryless in time, possibly skewed
 *   in the marginal -- are modeled); cross-layer PILOT/PILOT_REAL
 *   speculative prefetch (a same-token, cross-LAYER prediction mechanism,
 *   distinct from the cross-TOKEN memorylessness this task is about; it
 *   would need a router-internal "prediction recall" parameter we have not
 *   attempted to reproduce); real tensor VALUES/matmul numerics; real
 *   safetensors file I/O (all "bytes" here are logical counts + metadata,
 *   never actual buffers of that size -- this keeps the simulator fast and
 *   is appropriate for a logic/robustness validator, not a numerics one);
 *   the >64-unique-expert block-chunking moe() does for large prefill
 *   batches (n_experts=256 for GLM-5.2 means a single DECODE step, this
 *   this test's stated focus, never needs more than one such chunk, since
 *   nu<=n_experts always).
 *
 * TAXONOMY (borrowed from this repo's own
 * c/bench-m5max/iliria-streaming-evidence/RESULTS.md fuzz-gate convention):
 * every fault-injection test below classifies its outcome as one of
 *   - caught (short_read / fatal):        engine-visible, on the real critical path this is fatal
 *   - oracle-only (format_finding / corrupted): NOT visible to the real engine's own checks
 *     (verified by reading expert_load: only byte-COUNT is ever checked, never content, and
 *     the format-inference ternary has an unconditional else-fallback -- see sim_expert_io.h),
 *     but IS visible to this harness's independent oracle, which is exactly the deliverable:
 *     "the simulator must detect and surface these, never silently miscompute."
 *   - silent-WRONG (must never happen): a fault that produces ok==1 with NEITHER short_read
 *     NOR format_finding NOR corrupted set, i.e. something bad happened and NOTHING caught it.
 *     No test below should ever observe this; if one does, that is the finding to escalate.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sim_nvme.h"
#include "sim_cache.h"
#include "sim_expert_io.h"
#include "sim_router.h"

static int g_checks=0, g_fail=0;
#define CHECK(cond, ...) do{ \
    g_checks++; \
    if(!(cond)){ g_fail++; fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
                 fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); } \
}while(0)

/* ================= (a) synthetic NVMe: bandwidth/latency timing ================= */

static void test_nvme_single_read_timing_hand_computed(void){
    /* Fully independent, hand-computed check (not just re-running the
     * production formula): 1 GB/s, 1 MB read, zero latency -> exactly 1ms. */
    NvmeSim n; nvme_init(&n, 1e9, 0.0, 1);
    NvmeCompletion c = nvme_issue(&n, 0, 0.0, 1000000, 0.0);
    CHECK(fabs(c.t_complete-0.001) < 1e-12, "1MB@1GB/s should take exactly 1ms, got %.9fs", c.t_complete);
}

static void test_nvme_default_bandwidth_matches_measured_expert_transfer(void){
    /* One full GLM-5.2 expert (18,915,328 B) at the task-measured 13.3-13.5
     * GB/s should land in a tight, plausible millisecond range. */
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    int64_t total = expert_shard_total_bytes(&sh);
    NvmeSim lo, hi;
    nvme_init(&lo, SIM_NVME_BW_LOW_BPS, 0.0, 1);
    nvme_init(&hi, SIM_NVME_BW_HIGH_BPS, 0.0, 1);
    NvmeCompletion clo = nvme_issue(&lo, 0, 0.0, total, 0.0);
    NvmeCompletion chi = nvme_issue(&hi, 0, 0.0, total, 0.0);
    CHECK(chi.t_complete < clo.t_complete, "higher bandwidth must transfer faster");
    CHECK(clo.t_complete > 0.00135 && clo.t_complete < 0.00145,
        "expected ~1.40-1.42ms at 13.3GB/s, got %.6fs", clo.t_complete);
    CHECK(chi.t_complete > 0.00138 && chi.t_complete < 0.00143,
        "expected ~1.40ms at 13.5GB/s, got %.6fs", chi.t_complete);
}

static void test_nvme_worker_bandwidth_sharing(void){
    NvmeSim n; nvme_init(&n, 1e9, 0.0, 2);   /* 1GB/s total, 2 workers -> 0.5GB/s each */
    NvmeCompletion c0 = nvme_issue(&n, 0, 0.0, 500000000, 0.0); /* 0.5GB */
    NvmeCompletion c1 = nvme_issue(&n, 1, 0.0, 500000000, 0.0);
    CHECK(fabs(c0.t_complete-1.0)<1e-9, "worker0: 0.5GB @ 0.5GB/s should take 1.0s, got %.6f", c0.t_complete);
    CHECK(fabs(c1.t_complete-1.0)<1e-9, "worker1: 0.5GB @ 0.5GB/s should take 1.0s, got %.6f", c1.t_complete);
}

static void test_nvme_fifo_ordering_per_worker(void){
    NvmeSim n; nvme_init(&n, 1e9, 0.0, 1);
    NvmeCompletion c1 = nvme_issue(&n, 0, 0.0, 300000000, 0.0); /* 0.3s */
    CHECK(fabs(c1.t_complete-0.3)<1e-9, "first read should complete at 0.3s, got %.6f", c1.t_complete);
    /* second read issued at the SAME virtual instant, same worker: must queue
     * behind the first (FIFO, one pread at a time per worker), not overlap it. */
    NvmeCompletion c2 = nvme_issue(&n, 0, 0.0, 100000000, 0.0); /* 0.1s of transfer */
    CHECK(fabs(c2.t_complete-0.4)<1e-9, "second read must queue behind the first: expected 0.4s, got %.6f", c2.t_complete);
}

static void test_nvme_cross_worker_out_of_order(void){
    /* This is the mechanism behind fuzz item (d)'s "out-of-order completions":
     * a big job on worker0 and a small job on worker1, dispatched at the
     * same instant, complete out of dispatch order -- real, not an artifact
     * (see sim_nvme.h file header). */
    NvmeSim n; nvme_init(&n, 1e9, 0.0, 2);
    NvmeCompletion big   = nvme_issue(&n, 0, 0.0, 400000000, 0.0); /* worker0: 0.8s */
    NvmeCompletion small = nvme_issue(&n, 1, 0.0, 50000000,  0.0); /* worker1: 0.1s */
    CHECK(small.t_complete < big.t_complete,
        "worker1's smaller job must finish first despite being issued after in index order (big=%.3f small=%.3f)",
        big.t_complete, small.t_complete);
}

/* ================= byte-model fidelity: cross-check against glm.c's own constant ================= */

static void test_byte_model_matches_engine_hardcoded_constant(void){
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    int64_t total = expert_shard_total_bytes(&sh);
    /* glm.c ~L1962 (a2-overlap falsifier): g_a2_bytes=(int64_t)g_a2_nmiss*18915328
     * for this EXACT GLM-5.2 shape (D=6144 hidden, moe_inter=2048, int4). */
    CHECK(total==18915328, "expert byte total must match glm.c's own hardcoded per-expert constant 18915328, got %lld",
        (long long)total);
}

/* ================= (b) cache + eviction ================= */

static void test_cache_pin_always_hits_and_never_evicted(void){
    SimCache c; sim_cache_init(&c, 2, 64, 2 /* tiny ecap */);
    int pins[1] = {5};
    sim_cache_set_pins(&c, 0, pins, 1);
    /* Fill + overflow the LRU cache on layer 0 with unrelated experts. */
    for(int i=0;i<10;i++) sim_cache_promote(&c, 0, 100+i, 1000);
    int64_t b=-1;
    CHECK(sim_cache_lookup(&c,0,5,&b)==1, "pinned expert 5 must still hit after 10 LRU churns");
    sim_cache_free(&c);
}

static void test_cache_lru_eviction_basic(void){
    SimCache c; sim_cache_init(&c, 1, 64, 2);
    CHECK(sim_cache_promote(&c,0,1,10)==-1, "first promotion into empty cache: no eviction expected");
    CHECK(sim_cache_promote(&c,0,2,10)==-1, "second promotion, still room: no eviction expected");
    CHECK(sim_cache_lookup(&c,0,1,NULL)==1, "expert 1 must be resident");     /* refresh 1 -> most recent */
    int evicted = sim_cache_promote(&c,0,3,10);                              /* cache full: must evict LRU */
    CHECK(evicted==2, "expected eviction of expert 2 (least recently used), got %d", evicted);
    CHECK(sim_cache_lookup(&c,0,1,NULL)==1, "expert 1 should still be resident (was just refreshed)");
    CHECK(sim_cache_lookup(&c,0,2,NULL)==0, "expert 2 should have been evicted");
    CHECK(sim_cache_lookup(&c,0,3,NULL)==1, "expert 3 should be resident (just promoted)");
    sim_cache_free(&c);
}

static void test_cache_lru_shared_clock_is_structural_not_behavioral(void){
    /* See sim_cache.h file header: the clock field is SHARED across layers
     * (structurally faithful to glm.c's single m->eclock), but since every
     * eviction comparison in glm.c (and here) only ever scans one layer's
     * own array, heavy unrelated activity on OTHER layers must never change
     * which slot a given layer evicts. This test interleaves layer-1 hits
     * between layer-0 promotions and checks layer-0's eviction choice is
     * unchanged from the no-interleaving case. */
    SimCache c; sim_cache_init(&c, 2, 64, 2);
    sim_cache_promote(&c,0,1,10);
    sim_cache_promote(&c,0,2,10);
    sim_cache_lookup(&c,0,1,NULL);            /* refresh layer0's expert 1 */
    /* Heavy unrelated layer-1 traffic in between, to advance the shared clock a lot. */
    sim_cache_promote(&c,1,900,10);
    sim_cache_promote(&c,1,901,10);
    for(int i=0;i<50;i++) sim_cache_lookup(&c,1,900,NULL);
    int evicted = sim_cache_promote(&c,0,3,10);   /* layer0 eviction decision */
    CHECK(evicted==2, "layer-0 eviction choice must be unaffected by interleaved layer-1 clock activity, got evict=%d", evicted);
    sim_cache_free(&c);
}

static void test_cache_promo_cap_reverse_order_via_layer_step(void){
    /* ecap=2, 5 simultaneous misses in one decode step. glm.c ~L2055-2060:
     * promo=min(nmiss,ecap)=2, promoted in REVERSE dispatch order (last
     * miss first) -- so only the LAST 2 of the 5 routed experts should end
     * up resident; the first 3 are computed for this step and then lost. */
    SimCache c; sim_cache_init(&c, 1, 256, 2);
    NvmeSim nvme; nvme_init(&nvme, SIM_NVME_DEFAULT_BW_BPS, 0.0, 4);
    SimRuntime rt; sim_runtime_init(&rt, 4, 0.0001, 1);
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    ExpertShape shp = expert_shape_glm52();
    int routed[5] = {10,20,30,40,50};
    sim_layer_step(&rt,&c,&nvme,NULL,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down,
                   0, routed, 5, NULL);
    CHECK(sim_cache_lookup(&c,0,50,NULL)==1, "expert 50 (last dispatched) must be resident");
    CHECK(sim_cache_lookup(&c,0,40,NULL)==1, "expert 40 (2nd-last dispatched) must be resident");
    CHECK(sim_cache_lookup(&c,0,30,NULL)==0, "expert 30 must NOT be resident (promo cap reached before it)");
    CHECK(sim_cache_lookup(&c,0,20,NULL)==0, "expert 20 must NOT be resident (promo cap reached before it)");
    CHECK(sim_cache_lookup(&c,0,10,NULL)==0, "expert 10 must NOT be resident (promo cap reached before it)");
    sim_cache_free(&c);
}

/* ================= REPIN hot-store swap policy (ported from tier.h) ================= */

static void test_repin_swap_hysteresis(void){
    SimCache c; sim_cache_init(&c, 1, 16, 1);
    int pins[2] = {0,1};
    sim_cache_set_pins(&c,0,pins,2);
    /* pin 0 cold (heat 1), pin 1 hotter (heat 5); unpinned expert 9 hottest (heat 40). */
    c.heat[0][0]=1; c.heat[0][1]=5; c.heat[0][9]=40;
    int out_l[4], out_old[4], out_new[4];
    int nb = sim_repin_pass(&c, 4, out_l, out_old, out_new);
    CHECK(nb==1, "expected exactly one swap candidate, got %d", nb);
    if(nb==1){
        CHECK(out_l[0]==0 && out_old[0]==0 && out_new[0]==9,
            "expected layer0 to swap out pin 0 (coldest) for expert 9 (hottest unpinned), got l=%d old=%d new=%d",
            out_l[0], out_old[0], out_new[0]);
    }
    /* Hysteresis: heat 40 vs cold-pin heat 1 clears 1+(1>>2)+4=5 easily. Now
     * try a case that must NOT clear the margin: cold=10, hot=13 (13 <= 10+2+4=16). */
    SimCache c2; sim_cache_init(&c2, 1, 16, 1);
    sim_cache_set_pins(&c2,0,pins,2);
    c2.heat[0][0]=10; c2.heat[0][1]=20; c2.heat[0][9]=13;
    int nb2 = sim_repin_pass(&c2, 4, NULL,NULL,NULL);
    CHECK(nb2==0, "swap must be refused when the hysteresis margin isn't cleared (13 <= 10+2+4), got nb=%d", nb2);
    sim_cache_free(&c); sim_cache_free(&c2);
}

static void test_repin_decay_halves_heat(void){
    SimCache c; sim_cache_init(&c, 1, 4, 1);
    int pins[1]={0}; sim_cache_set_pins(&c,0,pins,1);
    c.heat[0][0]=100; c.heat[0][1]=51; c.heat[0][2]=1; c.heat[0][3]=0;
    sim_repin_pass(&c, 0, NULL,NULL,NULL);  /* max_swaps=0: no swap, decay still runs unconditionally */
    CHECK(c.heat[0][0]==50, "heat 100 must decay to 50, got %u", c.heat[0][0]);
    CHECK(c.heat[0][1]==25, "heat 51 must decay to 25 (integer >>1), got %u", c.heat[0][1]);
    CHECK(c.heat[0][2]==0,  "heat 1 must decay to 0, got %u", c.heat[0][2]);
    sim_cache_free(&c);
}

/* ================= (c) prefetch DEADLINE behavior: does a late load stall decode, by how much? ================= */

/* Shared fixture: a backend where ONLY gate_w carries bytes, sized so its
 * transfer takes exactly 0.1s at 1GB/s with zero base latency -- clean,
 * hand-verifiable numbers for the deadline arithmetic below. */
static ExpertShardSizes deadline_fixture_shard(void){
    ExpertShardSizes s; memset(&s,0,sizeof(s));
    s.gate_w=100000000; /* 0.1s @ 1GB/s */
    return s;
}

static void test_prefetch_deadline_fully_hidden_zero_stall(void){
    SimCache c; sim_cache_init(&c,1,256,4);
    sim_cache_promote(&c,0,100,1);                 /* expert 100 pre-resident: a HIT */
    NvmeSim nvme; nvme_init(&nvme, 1e9, 0.0, 1);
    SimRuntime rt; sim_runtime_init(&rt, 1, 0.5 /* 0.5s compute/expert */, 1);
    ExpertShardSizes sh = deadline_fixture_shard();
    ExpertShape shp = expert_shape_glm52();
    int routed[2] = {100 /* hit, resolved first */, 1 /* miss, 0.1s load */};
    SimLayerStepSummary sum;
    sim_layer_step(&rt,&c,&nvme,NULL,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down,
                   0, routed, 2, &sum);
    CHECK(sum.stall_added_s==0.0, "0.1s load fully hidden behind a preceding 0.5s compute must add 0 stall, got %.6f", sum.stall_added_s);
    sim_cache_free(&c);
}

static void test_prefetch_deadline_late_full_stall_quantified(void){
    SimCache c; sim_cache_init(&c,1,256,4);
    sim_cache_promote(&c,0,100,1);
    NvmeSim nvme; nvme_init(&nvme, 1e9, 0.0, 1);
    SimRuntime rt; sim_runtime_init(&rt, 1, 0.5, 1);
    ExpertShardSizes sh = deadline_fixture_shard();
    ExpertShape shp = expert_shape_glm52();
    /* Same load, same size -- but the MISS is resolved FIRST this time (no
     * preceding compute to hide behind): the ENTIRE 0.1s load must show up
     * as exposed stall, quantified exactly. */
    int routed[2] = {1, 100};
    SimLayerStepSummary sum;
    sim_layer_step(&rt,&c,&nvme,NULL,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down,
                   0, routed, 2, &sum);
    CHECK(fabs(sum.stall_added_s-0.1)<1e-9, "late load with nothing to hide behind must expose exactly its own 0.1s duration, got %.6f", sum.stall_added_s);
    sim_cache_free(&c);
}

static void test_prefetch_deadline_partial_overlap_quantified(void){
    SimCache c; sim_cache_init(&c,1,256,4);
    sim_cache_promote(&c,0,100,1);
    NvmeSim nvme; nvme_init(&nvme, 1e9, 0.0, 1);
    SimRuntime rt; sim_runtime_init(&rt, 1, 0.03 /* only 0.03s of compute precedes the miss */, 1);
    ExpertShardSizes sh = deadline_fixture_shard();
    ExpertShape shp = expert_shape_glm52();
    int routed[2] = {100, 1};   /* hit first (0.03s compute), then the 0.1s miss */
    SimLayerStepSummary sum;
    sim_layer_step(&rt,&c,&nvme,NULL,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down,
                   0, routed, 2, &sum);
    /* 0.1s load, 0.03s hidden behind the preceding hit's compute -> 0.07s must be exposed. */
    CHECK(fabs(sum.stall_added_s-0.07)<1e-9, "partial overlap: expected exactly 0.07s exposed stall, got %.6f", sum.stall_added_s);
    sim_cache_free(&c);
}

static void test_prefetch_deadline_occupancy_proxy(void){
    /* All-hit step: 0 pipe waits fire (no misses to wait on). All-cold-miss
     * step with zero compute overlap: every wait blocks (100% occupancy). */
    SimCache c; sim_cache_init(&c,1,256,8);
    sim_cache_promote(&c,0,1,1); sim_cache_promote(&c,0,2,1);
    NvmeSim nvme; nvme_init(&nvme, 1e9, 0.0, 1);
    SimRuntime rt; sim_runtime_init(&rt, 1, 0.0, 1);
    ExpertShardSizes sh = deadline_fixture_shard();
    ExpertShape shp = expert_shape_glm52();
    int hits[2]={1,2};
    sim_layer_step(&rt,&c,&nvme,NULL,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down, 0, hits,2, NULL);
    CHECK(rt.n_pipe_waits==0, "an all-hit step must never invoke a pipe wait, got %llu", (unsigned long long)rt.n_pipe_waits);
    int misses[3]={10,11,12};
    sim_layer_step(&rt,&c,&nvme,NULL,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down, 0, misses,3, NULL);
    CHECK(rt.n_pipe_waits==3, "3 cold misses must invoke exactly 3 pipe waits, got %llu", (unsigned long long)rt.n_pipe_waits);
    CHECK(rt.n_pipe_waits_blocked==3, "with zero compute-per-expert, every one of those 3 waits must block, got %llu", (unsigned long long)rt.n_pipe_waits_blocked);
    sim_cache_free(&c);
}

/* ================= (d) FUZZING: never silently miscompute ================= */

static void test_fuzz_short_read_is_caught_and_fatal_on_critical_path(void){
    SimFaultProgram fp; sim_fault_init(&fp);
    sim_fault_add(&fp, 0, 42, SIM_TENSOR_DOWN_W, SIM_FAULT_SHORT_READ, 100 /* deliver only 100B */);
    NvmeSim nvme; nvme_init(&nvme, SIM_NVME_DEFAULT_BW_BPS, 0.0, 2);
    IoByteCounters io; memset(&io,0,sizeof(io));
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    ExpertShape shp = expert_shape_glm52();
    ExpertLoadResult r = sim_expert_load(&nvme,0,0.0,&io,&fp, 0,42,&sh,1,
        shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down);
    CHECK(r.ok==0, "a short read must be DETECTED (ok==0), never silently accepted");
    CHECK(r.short_read==1, "short_read flag must be set");
    CHECK(io.bytes_requested==expert_shard_total_bytes(&sh), "requested bytes must count the FULL declared size even on failure, got %lld want %lld",
        (long long)io.bytes_requested, (long long)expert_shard_total_bytes(&sh));
    CHECK(io.bytes_read==0, "read bytes must stay 0 on a failed load, got %lld", (long long)io.bytes_read);
    CHECK(io.reads_attempted==1 && io.reads_completed==0,
        "attempted must be 1, completed must be 0 on failure (got attempted=%llu completed=%llu)",
        (unsigned long long)io.reads_attempted,(unsigned long long)io.reads_completed);
    CHECK(fp.rules[0].hit_count==1, "the fault rule must have actually fired exactly once");

    /* Now drive the SAME fault through the full decode-step driver and
     * confirm the critical-path "fatal" signal fires and the failed expert
     * is NEVER promoted into the cache (glm.c never publishes a half/failed
     * load into ecache -- s->eid is only set on the successful-return path). */
    SimCache c; sim_cache_init(&c,1,256,4);
    SimRuntime rt; sim_runtime_init(&rt,2,0.0,1);
    SimFaultProgram fp2; sim_fault_init(&fp2);
    sim_fault_add(&fp2, 0, 42, SIM_TENSOR_DOWN_W, SIM_FAULT_SHORT_READ, 100);
    int routed[1]={42};
    SimLayerStepSummary sum;
    sim_layer_step(&rt,&c,&nvme,&fp2,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down, 0, routed,1,&sum);
    CHECK(sum.fatal==1, "a critical-path short-read load must be flagged fatal (matches glm.c's fatal=1 exit(1) contract)");
    CHECK(rt.n_fatal_errors==1, "runtime fatal-error counter must increment");
    CHECK(sim_cache_is_resident(&c,0,42)==0, "a FAILED load must never become resident in the cache");
    sim_fault_free(&fp); sim_fault_free(&fp2); sim_cache_free(&c);
}

static void test_fuzz_wrong_size_shard_silent_but_oracle_catches(void){
    /* Bogus declared size that matches NONE of int8/int4/int2's formulas for
     * gate_proj's (O=2048,I=6144): int8=12582912, int4=6291456, int2=3145728.
     * 6291457 (int4+1) matches none. */
    SimFaultProgram fp; sim_fault_init(&fp);
    sim_fault_add(&fp, 0, 7, SIM_TENSOR_GATE_W, SIM_FAULT_WRONG_DECLARED_SIZE, 6291457);
    NvmeSim nvme; nvme_init(&nvme, SIM_NVME_DEFAULT_BW_BPS, 0.0, 1);
    IoByteCounters io; memset(&io,0,sizeof(io));
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    ExpertShape shp = expert_shape_glm52();
    ExpertLoadResult r = sim_expert_load(&nvme,0,0.0,&io,&fp, 0,7,&sh,1,
        shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down);
    CHECK(r.ok==1, "faithful path mirrors the real engine: a wrong-size-but-fully-delivered shard does NOT fail the load (ok==1) -- this IS the gap");
    CHECK(r.inferred_fmt[0]==3, "faithful ternary must fall through to fmt=3 (int2) exactly like glm.c's unconditional else, got %d", r.inferred_fmt[0]);
    CHECK(r.format_finding==1, "the independent oracle MUST flag this as a format mismatch (never silently miscompute), but the engine's own checks would not");
    sim_fault_free(&fp);
}

static void test_fuzz_corruption_silent_but_oracle_catches(void){
    SimFaultProgram fp; sim_fault_init(&fp);
    sim_fault_add(&fp, 0, 9, SIM_TENSOR_UP_W, SIM_FAULT_CORRUPT, 0);
    NvmeSim nvme; nvme_init(&nvme, SIM_NVME_DEFAULT_BW_BPS, 0.0, 1);
    IoByteCounters io; memset(&io,0,sizeof(io));
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    ExpertShape shp = expert_shape_glm52();
    ExpertLoadResult r = sim_expert_load(&nvme,0,0.0,&io,&fp, 0,9,&sh,1,
        shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down);
    CHECK(r.ok==1, "glm.c's expert_load has NO content checksum (verified by reading it): same-size corruption must NOT fail the load");
    CHECK(r.corrupted==1, "the fault program's ground truth (known only because THIS harness injected it) must mark this corrupted");
    sim_fault_free(&fp);
}

static void test_fuzz_pin_population_short_read_is_fatal_all_or_nothing(void){
    SimCache c; sim_cache_init(&c,1,256,4);
    NvmeSim nvme; nvme_init(&nvme, SIM_NVME_DEFAULT_BW_BPS, 0.0, 1);
    IoByteCounters io; memset(&io,0,sizeof(io));
    SimFaultProgram fp; sim_fault_init(&fp);
    sim_fault_add(&fp, 0, 22, SIM_TENSOR_GATE_S, SIM_FAULT_SHORT_READ, 4);
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    ExpertShape shp = expert_shape_glm52();
    int want_pins[3] = {20,21,22};   /* 22's load will fail */
    int ok = sim_cache_populate_pins(&c,&nvme,&fp,&io,0.0,&sh,&shp, 0, want_pins, 3);
    CHECK(ok==0, "AUTOPIN-style startup population must report failure (fatal-at-startup) when any pin load fails");
    CHECK(c.npin[0]==0, "on failure, NO pins may be committed (all-or-nothing, mirrors fatal=1 exit-before-serving)");
    CHECK(sim_cache_is_resident(&c,0,20)==0 && sim_cache_is_resident(&c,0,21)==0,
        "even pins that loaded fine before the failing one must not be left half-committed");
    sim_cache_free(&c); sim_fault_free(&fp);
}

/* ---- out-of-order / stale completion safety ----
 * The real PIPE pool (glm.c ~L1327-1432) tolerates worker threads completing
 * jobs out of dispatch order via a GENERATION-TAGGED cursor: a worker only
 * ever commits a result if its CAS-won (job-index, generation) pair is still
 * current; a straggler from an OLD generation whose job index has since
 * been recycled for a NEW batch must never overwrite that slot. This
 * simulator's own decode-step driver (sim_layer_step) is single-threaded and
 * fully drains every dispatched read before returning, so it cannot
 * reproduce the race by construction -- instead we isolate and directly
 * test the INVARIANT itself with a minimal, focused generation-tagged
 * commit primitive, mirroring the real cursor's contract precisely. */
typedef struct { int eid; uint64_t gen; int published; } GenTaggedSlot;
static int gen_slot_try_commit(GenTaggedSlot *slot, uint64_t completion_gen, int completion_eid, uint64_t current_gen){
    if(completion_gen != current_gen) return 0;    /* stale generation: reject, exactly the real CAS invariant */
    slot->eid=completion_eid; slot->gen=current_gen; slot->published=1;
    return 1;
}
static void test_generation_tag_stale_completion_rejected(void){
    GenTaggedSlot slot; memset(&slot,0,sizeof(slot));
    uint64_t current_gen = 1;
    /* Job A dispatched under generation 1 (a straggler -- imagine a huge
     * injected latency), does NOT complete yet. */
    /* Meanwhile generation advances (a new block was dispatched, reusing
     * the same slot index), and job B under generation 2 completes first. */
    current_gen = 2;
    int commB = gen_slot_try_commit(&slot, 2, /*eid=*/777, current_gen);
    CHECK(commB==1, "an on-time, current-generation completion must commit");
    CHECK(slot.eid==777, "slot must reflect B's expert id after B commits");
    /* NOW the generation-1 straggler A finally "arrives", tagged with the
     * OLD generation. It must be rejected, and must NOT clobber B's data. */
    int commA = gen_slot_try_commit(&slot, 1, /*eid=*/555, current_gen);
    CHECK(commA==0, "a stale (old-generation) straggler completion must be REJECTED, never silently applied");
    CHECK(slot.eid==777, "slot must still hold B's data -- a stale straggler must never overwrite a recycled slot");
}

/* ================= (e) byte accounting ================= */

static void test_byte_accounting_identity_success_and_failure_mix(void){
    SimCache c; sim_cache_init(&c,1,256,64);
    NvmeSim nvme; nvme_init(&nvme, SIM_NVME_DEFAULT_BW_BPS, 0.0, 4);
    SimRuntime rt; sim_runtime_init(&rt,4,0.00001,1);
    SimFaultProgram fp; sim_fault_init(&fp);
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    ExpertShape shp = expert_shape_glm52();
    int64_t one_expert = expert_shard_total_bytes(&sh);

    /* 5 experts fault-free, 3 experts short-read (all distinct eids so each
     * is a genuine, independent miss -- no cache interaction confounds the
     * count). */
    int clean[5]  = {1,2,3,4,5};
    int broken[3] = {6,7,8};
    for(int i=0;i<3;i++) sim_fault_add(&fp, 0, broken[i], SIM_TENSOR_GATE_W, SIM_FAULT_SHORT_READ, 10);

    sim_layer_step(&rt,&c,&nvme,&fp,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down, 0, clean,5, NULL);
    sim_layer_step(&rt,&c,&nvme,&fp,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down, 0, broken,3, NULL);

    CHECK(rt.io.reads_attempted==8, "8 distinct never-before-seen experts must trigger 8 attempted reads, got %llu",
        (unsigned long long)rt.io.reads_attempted);
    CHECK(rt.io.reads_completed==5, "only the 5 fault-free loads must count as completed, got %llu",
        (unsigned long long)rt.io.reads_completed);
    CHECK(rt.io.bytes_requested==8*one_expert, "requested bytes must count all 8 attempts at full declared size, got %lld want %lld",
        (long long)rt.io.bytes_requested, (long long)(8*one_expert));
    CHECK(rt.io.bytes_read==5*one_expert, "read bytes must count only the 5 successful loads, got %lld want %lld",
        (long long)rt.io.bytes_read, (long long)(5*one_expert));
    /* Reconciliation identity: every byte "lost" between requested and read
     * must be fully accounted for by the failed attempts' full declared size. */
    CHECK(rt.io.bytes_requested - rt.io.bytes_read == 3*one_expert,
        "requested-read must equal exactly the 3 failed attempts' declared bytes, got %lld want %lld",
        (long long)(rt.io.bytes_requested-rt.io.bytes_read), (long long)(3*one_expert));
    CHECK(rt.io.reads_attempted - rt.io.reads_completed == 3,
        "attempted-completed must equal the 3 failed attempts, got %llu",
        (unsigned long long)(rt.io.reads_attempted-rt.io.reads_completed));
    sim_cache_free(&c); sim_fault_free(&fp);
}

/* ================= integration: memoryless routing + cache over a session ================= */

static double run_session_hit_rate(SimRouteMode mode, double zipf_s, uint32_t seed,
                                    int n_experts, int topk, int ecap, int n_layers, int n_steps){
    SimCache c; sim_cache_init(&c, n_layers, n_experts, ecap);
    NvmeSim nvme; nvme_init(&nvme, SIM_NVME_DEFAULT_BW_BPS, 0.0002, 8);
    SimRuntime rt; sim_runtime_init(&rt, 8, 0.00002, 1);
    SimRouter router; sim_router_init(&router, n_experts, topk, mode, zipf_s);
    SimRng rng; sim_rng_seed(&rng, seed);
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    ExpertShape shp = expert_shape_glm52();
    int *picks = malloc((size_t)topk*sizeof(int));
    for(int step=0; step<n_steps; step++){
        for(int l=0; l<n_layers; l++){
            sim_router_pick(&router, &rng, picks);
            sim_layer_step(&rt,&c,&nvme,NULL,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down,
                           l, picks, topk, NULL);
        }
    }
    uint64_t tot = c.hits+c.misses;
    double hitpct = tot ? 100.0*(double)c.hits/(double)tot : 0.0;
    free(picks); sim_router_free(&router); sim_cache_free(&c);
    return hitpct;
}

static void test_integration_zipf_skew_beats_uniform_baseline(void){
    int n_experts=256, topk=8, ecap=16, n_layers=4, n_steps=400;
    double uniform_hit = run_session_hit_rate(SIM_ROUTE_UNIFORM, 1.0, 12345, n_experts,topk,ecap,n_layers,n_steps);
    double zipf_hit     = run_session_hit_rate(SIM_ROUTE_ZIPF,    1.2, 12345, n_experts,topk,ecap,n_layers,n_steps);
    fprintf(stdout, "[integration] uniform-routing hit rate: %.2f%%  |  zipf-skewed hit rate: %.2f%%  (ecap=%d/%d experts)\n",
        uniform_hit, zipf_hit, ecap, n_experts);
    /* Null-model sanity: with NO exploitable skew, hit rate should be small
     * and in the ballpark of ecap/n_experts (=6.25%) -- not near-zero, not huge. */
    CHECK(uniform_hit>1.0 && uniform_hit<20.0,
        "uniform (null model) hit rate should be a small capacity-driven number near ecap/n_experts=6.25%%, got %.2f%%", uniform_hit);
    /* Positive control: real skew must be CAPTURED by the LRU, clearly
     * beating the capacity-only baseline -- this is the actual "does the
     * cache logic work" claim. */
    CHECK(zipf_hit > uniform_hit*2.0,
        "skewed routing should let the cache capture popularity structure, expected zipf hit rate > 2x uniform's (%.2f%% vs %.2f%%)",
        zipf_hit, uniform_hit);
}

static void test_integration_full_session_report(void){
    /* End-to-end smoke test that also exercises sim_report()'s formatting,
     * with a few pins, real cache pressure, and a couple of faults sprinkled
     * on eids guaranteed to be routed (since we hand-supply routed[] here
     * rather than going through the random router, for determinism). */
    SimCache c; sim_cache_init(&c,3,256,8);
    int pins[2]={0,1};
    sim_cache_set_pins(&c,0,pins,2);
    NvmeSim nvme; nvme_init(&nvme, SIM_NVME_DEFAULT_BW_BPS, 0.0002, 8);
    SimRuntime rt; sim_runtime_init(&rt, 8, 0.00003, 1);
    SimFaultProgram fp; sim_fault_init(&fp);
    sim_fault_add(&fp, 2, 199, SIM_TENSOR_DOWN_S, SIM_FAULT_SHORT_READ, 8);
    ExpertShardSizes sh = expert_shard_sizes_glm52_int4();
    ExpertShape shp = expert_shape_glm52();

    int step_routes[6][4] = {
        {0,1,2,3}, {0,1,4,5}, {2,3,6,7}, {0,1,2,8}, {9,10,11,12}, {0,1,2,3}
    };
    for(int s=0;s<6;s++)
        for(int l=0;l<3;l++)
            sim_layer_step(&rt,&c,&nvme,NULL,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down,
                           l, step_routes[s], 4, NULL);
    /* One deliberately-fatal step on layer 2. */
    int fault_route[1]={199};
    SimLayerStepSummary sum;
    sim_layer_step(&rt,&c,&nvme,&fp,&sh,shp.O_gate,shp.I_gate,shp.O_up,shp.I_up,shp.O_down,shp.I_down,
                   2, fault_route,1,&sum);
    CHECK(sum.fatal==1, "the sprinkled fault on eid 199 must surface as fatal on its critical-path step");

    printf("\n---- test_integration_full_session_report: sim_report() output ----\n");
    sim_report(stdout, &c, &rt);
    printf("---------------------------------------------------------------------\n");

    CHECK(c.hits+c.misses>0, "session must have recorded some lookups");
    CHECK(rt.io.bytes_requested>=rt.io.bytes_read, "requested must never be less than read (accounting sanity)");
    CHECK(rt.n_fatal_errors==1, "exactly one fatal error expected from the sprinkled fault");
    sim_cache_free(&c); sim_fault_free(&fp);
}

int main(void){
    test_nvme_single_read_timing_hand_computed();
    test_nvme_default_bandwidth_matches_measured_expert_transfer();
    test_nvme_worker_bandwidth_sharing();
    test_nvme_fifo_ordering_per_worker();
    test_nvme_cross_worker_out_of_order();

    test_byte_model_matches_engine_hardcoded_constant();

    test_cache_pin_always_hits_and_never_evicted();
    test_cache_lru_eviction_basic();
    test_cache_lru_shared_clock_is_structural_not_behavioral();
    test_cache_promo_cap_reverse_order_via_layer_step();

    test_repin_swap_hysteresis();
    test_repin_decay_halves_heat();

    test_prefetch_deadline_fully_hidden_zero_stall();
    test_prefetch_deadline_late_full_stall_quantified();
    test_prefetch_deadline_partial_overlap_quantified();
    test_prefetch_deadline_occupancy_proxy();

    test_fuzz_short_read_is_caught_and_fatal_on_critical_path();
    test_fuzz_wrong_size_shard_silent_but_oracle_catches();
    test_fuzz_corruption_silent_but_oracle_catches();
    test_fuzz_pin_population_short_read_is_fatal_all_or_nothing();
    test_generation_tag_stale_completion_rejected();

    test_byte_accounting_identity_success_and_failure_mix();

    test_integration_zipf_skew_beats_uniform_baseline();
    test_integration_full_session_report();

    printf("\nexpert_stream_sim: %d/%d checks passed\n", g_checks-g_fail, g_checks);
    return g_fail?1:0;
}
