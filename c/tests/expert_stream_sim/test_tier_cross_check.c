/* test_tier_cross_check.c -- proves sim_cache.h's ported sim_tier_pick_swap()/
 * sim_tier_decay() are byte-for-byte behaviorally identical to the REAL,
 * production c/tier.h they were ported from.
 *
 * This is the ONE file in this simulator directory that reaches outside it
 * (a read-only #include of ../../tier.h) -- deliberately kept separate from
 * the main suite (test_expert_stream_sim.c, `make test`) and its own `make
 * cross-check` target, so the primary deliverable stays fully standalone
 * (see sim_nvme.h's file header) while this fidelity proof is still one
 * command away. tier.h is a tiny, already-standalone, dependency-free header
 * (own tests/test_tier.c already exists) well outside the loader/matmul
 * path -- including it read-only carries negligible risk and, unlike hand-
 * reviewing the port, actually PROVES the port matches on many random cases
 * rather than merely asserting it by eye.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sim_cache.h"
#include "../../tier.h"

static uint32_t rng_state=0x2545F491u;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }

static int g_checks=0, g_fail=0;
#define CHECK(cond, ...) do{ \
    g_checks++; \
    if(!(cond)){ g_fail++; fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
                 fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); } \
}while(0)

int main(void){
    enum { NEXPERT=64, NTRIALS=20000 };
    uint32_t heat[NEXPERT];
    int pinned[16];

    for(int t=0;t<NTRIALS;t++){
        for(int e=0;e<NEXPERT;e++) heat[e]=xr()%2000;
        int npin = 1 + (int)(xr()%16);
        /* distinct pinned ids */
        unsigned char used[NEXPERT]; memset(used,0,sizeof(used));
        for(int z=0;z<npin;z++){
            int e; do{ e=xr()%NEXPERT; } while(used[e]);
            used[e]=1; pinned[z]=e;
        }

        int slot_real, eid_real; long gain_real;
        int slot_sim,  eid_sim;  long gain_sim;
        int ret_real = tier_pick_swap(heat, NEXPERT, pinned, npin, &slot_real, &eid_real, &gain_real);
        int ret_sim  = sim_tier_pick_swap(heat, NEXPERT, pinned, npin, &slot_sim, &eid_sim, &gain_sim);

        CHECK(ret_real==ret_sim, "trial %d: return value mismatch real=%d sim=%d", t, ret_real, ret_sim);
        if(ret_real && ret_sim){
            CHECK(slot_real==slot_sim, "trial %d: slot mismatch real=%d sim=%d", t, slot_real, slot_sim);
            CHECK(eid_real==eid_sim,   "trial %d: eid mismatch real=%d sim=%d", t, eid_real, eid_sim);
            CHECK(gain_real==gain_sim, "trial %d: gain mismatch real=%ld sim=%ld", t, gain_real, gain_sim);
        }

        uint32_t heat_real[NEXPERT], heat_sim[NEXPERT];
        memcpy(heat_real, heat, sizeof(heat));
        memcpy(heat_sim,  heat, sizeof(heat));
        tier_decay(heat_real, NEXPERT);
        sim_tier_decay(heat_sim, NEXPERT);
        CHECK(memcmp(heat_real,heat_sim,sizeof(heat_real))==0, "trial %d: decay output mismatch", t);
    }

    printf("test_tier_cross_check: %d/%d checks passed (sim_tier_pick_swap/sim_tier_decay vs real c/tier.h, %d random trials)\n",
        g_checks-g_fail, g_checks, NTRIALS);
    return g_fail?1:0;
}
