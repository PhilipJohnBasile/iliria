/* Stage-0 exposed-stall timer validation (deliverable 5, c/bench-m5max/
 * the factorial-streaming causality spec): "Tiny fixture with DELIBERATELY inserted read
 * delays and KNOWN compute/IO overlap. The timer must ... satisfy ... ~0%, ~50%, ~100%
 * overlap ... (~full stall, ~half stall, ~zero stall) within stated tolerance."
 *
 * Isolated-primitive design (documented, not a silent scope narrowing): this test drives
 * the REAL production pipe pool (pipe_init/pipe_dispatch/pipe_wait_timed) and the REAL
 * expert_load() I/O path (against tests/tiny_mixed_moe_fixture.py, an on-disk fixture with
 * real .qs sidecars so the instrumented read path in expert_load actually runs, exactly
 * like the deliverable-1 commit's own smoke test) with g_io_delay_us injecting a large,
 * known delay so real disk time is negligible noise by comparison. What it does NOT do is
 * route the "known compute overlap" through the full model forward pass: c/glm.c's real
 * overlap opportunity (concurrent expert loads racing a block's own per-expert matmul loop,
 * or Metal GPU submission racing disk reads) is not precisely controllable to an exact
 * target fraction -- it depends on real matmul timing, thread scheduling, and (for the
 * Metal path) real GPU submission latency, none of which can be dialed to "exactly 50%" on
 * demand. Stage 0's own purpose is to validate the TIMER's correctness under KNOWN overlap,
 * decoupled from model/kernel performance realism -- so this test supplies the "known
 * compute" as an explicit, controllable sleep in ITS OWN main(), between the real
 * pipe_dispatch() and the real pipe_wait_timed() call, and asserts what the REAL,
 * unmodified instrumentation (glm.c's pipe_wait_timed(), exactly as wired into moe()'s two
 * call sites) reports. No new production code path is added for this: the sleep lives here,
 * in the test, never in glm.c.
 *
 * See the stage-0 validation record for the full write-up, exact commands, and results
 * this file's output feeds into.
 */
#define main ili_glm_main_unused
#include "../glm.c"
#undef main

#include <stdlib.h>

static int failures=0;
#define CHECKF(desc, cond) do{ \
    if(!(cond)){ fprintf(stderr,"FAIL %s\n",desc); failures++; } \
    else fprintf(stderr,"ok   %s\n",desc); \
}while(0)

/* One trial: dispatch a real (tiny) expert load with g_io_delay_us injected, sleep
 * compute_frac*delay_s to simulate KNOWN concurrent compute, then time the real
 * pipe_wait_timed(). Returns the measured exposed-stall seconds. */
static double one_trial(Model *m, int layer, int eid, double delay_s, double compute_frac){
    if(!g_pp.started) pipe_init(m);
    int eids[1]={eid};
    pipe_dispatch(m, layer, eids, 1);
    if(compute_frac>0){
        struct timespec ts; double sleep_s=compute_frac*delay_s;
        ts.tv_sec=(time_t)sleep_s; ts.tv_nsec=(long)((sleep_s-(double)ts.tv_sec)*1e9);
        nanosleep(&ts,NULL);
    }
    uint64_t nw=0, nb=0;
    return pipe_wait_timed(0,&nw,&nb);
}

int main(void){
    const char *snap = getenv("STAGE0_FIXTURE");
    if(!snap){ fprintf(stderr,"STAGE0_FIXTURE=<tiny_mixed_moe fixture dir> required\n"); return 2; }
    Model m; model_init(&m, snap, 64, 8, 8);

    const double DELAY_S = 0.030;      /* 30ms: >> this fixture's real (sub-ms) read time */
    const double TOL_ABS_S = 0.010;    /* +/-10ms absolute tolerance: generous vs. sched_yield
                                        * spin-wait / nanosleep scheduling jitter on a busy
                                        * dev machine, tight vs. DELAY_S=30ms (33% relative
                                        * at the endpoints, tighter in the middle where the
                                        * expected value itself is smaller-relative-error-
                                        * sensitive) -- see the stage-0 validation record for the
                                        * actual observed numbers this bound was set against. */
    g_io_delay_us = (int)(DELAY_S*1e6);

    printf("Stage-0 exposed-stall overlap validation: delay=%.1fms tolerance=+/-%.1fms\n",
        DELAY_S*1e3, TOL_ABS_S*1e3);

    /* layer 0 in this fixture is DENSE (first_k_dense_replace=1, see tiny_mixed_moe_fixture.py)
     * -- it has no expert weights at all, so expert_load would exit(1) (fatal=1, missing
     * tensor). Layer 1 is the first MoE (sparse) layer; expert ids 0..7 are all valid there
     * (N_ROUTED_EXPERTS=8). Three DISTINCT expert ids across the three trials so no trial
     * benefits from a previous trial's now-warm slot/page-cache state. */
    const int LAYER=1;

    /* ~0% overlap: no compute between dispatch and wait -> expect ~FULL stall (~DELAY_S). */
    double s0 = one_trial(&m, LAYER, 0, DELAY_S, 0.0);
    printf("  0%% overlap: exposed-stall=%.4fms (expect ~%.1fms)\n", s0*1e3, DELAY_S*1e3);
    CHECKF("0% overlap recovers ~full stall", fabs(s0-DELAY_S) < TOL_ABS_S);

    /* ~50% overlap: compute for half the delay -> expect ~HALF stall. */
    double s50 = one_trial(&m, LAYER, 1, DELAY_S, 0.5);
    printf("  50%% overlap: exposed-stall=%.4fms (expect ~%.1fms)\n", s50*1e3, DELAY_S*1e3*0.5);
    CHECKF("50% overlap recovers ~half stall", fabs(s50-DELAY_S*0.5) < TOL_ABS_S);

    /* ~100% overlap: compute for the FULL delay -> expect ~ZERO stall (fully hidden). */
    double s100 = one_trial(&m, LAYER, 2, DELAY_S, 1.0);
    printf("  100%% overlap: exposed-stall=%.4fms (expect ~0ms)\n", s100*1e3);
    CHECKF("100% overlap recovers ~zero stall", s100 < TOL_ABS_S);

    /* Monotonicity, independent of the absolute tolerances above: more overlap must never
     * produce MORE exposed stall. */
    CHECKF("monotone: stall(0%) >= stall(50%) >= stall(100%)", s0 >= s50 && s50 >= s100);

    if(failures){ fprintf(stderr,"\n%d FAILURE(S)\n",failures); return 1; }
    fprintf(stderr,"\nStage-0 overlap validation: all checks pass\n");
    return 0;
}
