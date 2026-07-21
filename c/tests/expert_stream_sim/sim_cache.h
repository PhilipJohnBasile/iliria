/* sim_cache.h -- expert cache/eviction model: pin (hot-store) + per-layer LRU.
 *
 * Ported from the ALGORITHM (not the code -- no #include of glm.c anywhere
 * in this simulator) read out of:
 *
 *   - lookup order + LRU-hit clock bump: glm.c's moe(), "FASE C/D" resolve
 *     loop, ~L1862-1868:
 *       ESlot *P=m->pin[layer];
 *       for(z<m->npin[layer]) if(P[z].eid==eid){ m->hits++; use[j]=&P[z]; break; }
 *       if(!use[j]){ ESlot *Sl=m->ecache[layer]; ...
 *         for(z<nn) if(Sl[z].eid==eid){ m->hits++;
 *             Sl[z].used=(uint64_t)__atomic_add_fetch(&m->eclock,1,...); use[j]=&Sl[z]; break; } }
 *       if(!use[j]){ ... m->miss++; }
 *     i.e. pin is checked BEFORE the LRU cache, and only an LRU-cache hit
 *     bumps `used` (a pin hit never touches any clock, because a pin slot is
 *     never an eviction candidate).
 *
 *   - promotion + eviction: glm.c ~L2055-2060:
 *       int promo = nmiss<m->ecap ? nmiss : m->ecap;
 *       for(a<promo){ q=nmiss-1-a;               // reverse order: LAST miss first
 *         if(*nn<m->ecap) dst=&Sl[(*nn)++];
 *         else { lru=0; for(z=1..*nn) if(Sl[z].used<Sl[lru].used) lru=z; dst=&Sl[lru]; }
 *         ...; dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,...); }
 *     i.e. append while there is room, else evict the slot with the MINIMUM
 *     `used` via a LINEAR SCAN (not a heap/intrusive list) -- O(ecap) per
 *     promotion, and when nmiss>ecap in one 64-expert block, only `ecap` of
 *     the misses are promoted at all (in reverse dispatch order); the rest
 *     are computed from m->ws[] this block and never enter the LRU.
 *
 *   - the LRU clock is a SINGLE counter SHARED ACROSS ALL LAYERS (glm.c's
 *     `m->eclock`, one field on Model, bumped by every layer's hits and
 *     promotions), reproduced here as one `clock` field on SimCache rather
 *     than one per layer. VERIFIED (grep of every `.used`/`eclock` site in
 *     glm.c, all 5 occurrences): every eviction comparison only ever scans
 *     one layer's own `Sl[]`/`m->ecache[layer]` array (moe()'s promotion
 *     ~L2121, pilot_realload's own-layer scan ~L2207) -- nothing compares
 *     `used` ACROSS layers. So sharing one counter vs. giving each layer its
 *     own independent monotonic counter is BEHAVIORALLY EQUIVALENT for every
 *     eviction decision either can make: interleaved bumps from other layers
 *     change a layer's own timestamps' absolute magnitude but never their
 *     relative order, which is all eviction ever looks at. We still mirror
 *     the shared-field structure exactly (not the "obvious" per-layer
 *     version) for structural exactness and in case a future engine change
 *     ever does add a cross-layer comparison, but we do NOT claim -- and
 *     test_cache_lru_shared_clock_is_structural_not_behavioral() below
 *     demonstrates -- that today's code makes any decision a per-layer clock
 *     would not also make.
 *
 *   - REPIN swap policy: c/tier.h's tier_pick_swap()/tier_decay() are already
 *     a tiny, self-contained, dependency-free header (no glm.c coupling,
 *     already has its own tests/test_tier.c). This file REIMPLEMENTS the
 *     same hysteresis formula (sim_tier_pick_swap below) byte-for-byte
 *     instead of #include-ing tier.h, purely so this simulator directory has
 *     NO #include edge into any file under c/ at all -- it can never be broken by, or
 *     even accused of touching, files someone else is concurrently editing.
 *     test_tier_cross_check.c (a separate, clearly-optional file/target)
 *     cross-validates this port against the REAL tier.h on many random
 *     cases, read-only, to prove the port isn't subtly wrong.
 */
#ifndef SIM_CACHE_H
#define SIM_CACHE_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int      eid;    /* -1 = empty/hidden slot */
    uint64_t used;   /* logical LRU timestamp, drawn from SimCache.clock */
    int64_t  bytes;  /* resident payload size (heterogeneous sizes allowed --
                       * e.g. the real engine's MTP row is int8 (~2x an int4
                       * layer's bytes) reusing the same ws[]/ecache slots) */
} SimSlot;

typedef struct {
    int n_layers, n_experts, ecap;
    SimSlot  **ecache; int *ecn;    /* [layer][0..ecn) LRU cache, ecn<=ecap */
    SimSlot  **pin;     int *npin;  /* [layer][0..npin) hot-store, evicted only via sim_repin_pass */
    uint32_t **heat;                /* [layer][expert] live decaying heat (repin candidate score) */
    uint32_t **usage;               /* [layer][expert] persistent lifetime usage counter */
    uint64_t clock;                 /* SHARED across ALL layers -- see file header */
    uint64_t hits, misses;
} SimCache;

static void sim_cache_init(SimCache *c, int n_layers, int n_experts, int ecap){
    memset(c,0,sizeof(*c));
    c->n_layers=n_layers; c->n_experts=n_experts; c->ecap=ecap;
    c->ecache=calloc((size_t)n_layers,sizeof(SimSlot*));
    c->ecn   =calloc((size_t)n_layers,sizeof(int));
    c->pin   =calloc((size_t)n_layers,sizeof(SimSlot*));
    c->npin  =calloc((size_t)n_layers,sizeof(int));
    c->heat  =calloc((size_t)n_layers,sizeof(uint32_t*));
    c->usage =calloc((size_t)n_layers,sizeof(uint32_t*));
    for(int l=0;l<n_layers;l++){
        c->ecache[l]=malloc((size_t)(ecap>0?ecap:1)*sizeof(SimSlot));
        for(int z=0;z<ecap;z++){ c->ecache[l][z].eid=-1; c->ecache[l][z].used=0; c->ecache[l][z].bytes=0; }
        c->heat[l]=calloc((size_t)n_experts,sizeof(uint32_t));
        c->usage[l]=calloc((size_t)n_experts,sizeof(uint32_t));
    }
}

static void sim_cache_free(SimCache *c){
    for(int l=0;l<c->n_layers;l++){
        free(c->ecache[l]); free(c->pin[l]); free(c->heat[l]); free(c->usage[l]);
    }
    free(c->ecache); free(c->ecn); free(c->pin); free(c->npin); free(c->heat); free(c->usage);
}

/* Sets layer `l`'s pin (hot-store) set from a caller-owned id list (copied
 * in). Never evicted except via sim_repin_pass() below. */
static void sim_cache_set_pins(SimCache *c, int l, const int *eids, int n){
    free(c->pin[l]);
    c->pin[l]=malloc((size_t)(n>0?n:1)*sizeof(SimSlot));
    for(int z=0;z<n;z++){ c->pin[l][z].eid=eids[z]; c->pin[l][z].used=0; c->pin[l][z].bytes=0; }
    c->npin[l]=n;
}

/* Records that expert `eid` on layer `l` was SELECTED by routing (whether it
 * then hits or misses). Mirrors glm.c incrementing eusage[]/eheat[] once per
 * routed selection (~L1788,1816), upstream of and independent from the
 * cache lookup below. */
static void sim_cache_note_routed(SimCache *c, int l, int eid){
    if(c->usage[l][eid] < UINT32_MAX) c->usage[l][eid]++;
    if(c->heat[l][eid]  < UINT32_MAX) c->heat[l][eid]++;
}

/* Cache lookup for one routed expert: pin checked first, then per-layer LRU
 * (bumping its `used` clock on hit) -- exact order + clock semantics of
 * glm.c ~L1863-1868. Returns 1 on hit (writes *bytes_out if non-NULL), 0 on
 * miss (counted into SimCache.misses; caller must then load + promote). */
static int sim_cache_lookup(SimCache *c, int l, int eid, int64_t *bytes_out){
    SimSlot *P=c->pin[l];
    for(int z=0;z<c->npin[l];z++) if(P[z].eid==eid){
        c->hits++; if(bytes_out) *bytes_out=P[z].bytes; return 1;
    }
    SimSlot *L=c->ecache[l];
    for(int z=0;z<c->ecn[l];z++) if(L[z].eid==eid){
        c->hits++; L[z].used=++c->clock; if(bytes_out) *bytes_out=L[z].bytes; return 1;
    }
    c->misses++;
    return 0;
}

/* Residency probe WITHOUT counting a hit/miss or bumping any clock -- mirrors
 * the next-block WILLNEED readahead residency check at glm.c ~L1972-1977,
 * which is explicitly a hint scan, never a real lookup. */
static int sim_cache_is_resident(SimCache *c, int l, int eid){
    SimSlot *P=c->pin[l]; for(int z=0;z<c->npin[l];z++) if(P[z].eid==eid) return 1;
    SimSlot *L=c->ecache[l]; for(int z=0;z<c->ecn[l];z++) if(L[z].eid==eid) return 1;
    return 0;
}

/* Promotes one just-loaded (layer,eid) into the LRU cache: append while
 * there is room, else evict the slot with the MINIMUM `used` (linear scan)
 * -- glm.c ~L2055-2060. Returns the evicted eid, or -1 if this was a plain
 * append (no eviction occurred). Caller is responsible for the promo-cap +
 * reverse-dispatch-order edge case when nmiss>ecap in one block (that lives
 * at the call site in sim_router.h's sim_layer_step, exactly where it lives
 * in glm.c's moe() -- there is no such block-level abstraction in the real
 * engine either, so none is invented here). */
static int sim_cache_promote(SimCache *c, int l, int eid, int64_t bytes){
    SimSlot *L=c->ecache[l]; int *nn=&c->ecn[l];
    if(*nn < c->ecap){
        SimSlot *dst=&L[(*nn)++];
        dst->eid=eid; dst->bytes=bytes; dst->used=++c->clock;
        return -1;
    }
    int lru=0; for(int z=1;z<*nn;z++) if(L[z].used<L[lru].used) lru=z;
    int evicted=L[lru].eid;
    L[lru].eid=eid; L[lru].bytes=bytes; L[lru].used=++c->clock;
    return evicted;
}

/* ---- REPIN hot-store swap policy: reimplementation of c/tier.h ---- */

/* Byte-for-byte port of tier_pick_swap() (c/tier.h ~L10-27): pick the
 * coldest pinned expert and the hottest unpinned one; refuse the swap unless
 * the hot one clears the cold one by >25% + 4 (fixed-sample hysteresis
 * margin, prevents ping-pong on noise). */
static int sim_tier_pick_swap(const uint32_t *heat, int nexpert, const int *pinned, int npin,
                               int *slot, int *eid, long *gain){
    if(!heat || !pinned || npin<1 || nexpert<1) return 0;
    int cold=0;
    for(int z=1;z<npin;z++) if(heat[pinned[z]]<heat[pinned[cold]]) cold=z;
    int hot=-1; uint32_t fh=0;
    for(int e=0;e<nexpert;e++){
        int resident=0;
        for(int z=0;z<npin;z++) if(pinned[z]==e){ resident=1; break; }
        if(!resident && heat[e]>fh){ fh=heat[e]; hot=e; }
    }
    if(hot<0) return 0;
    uint32_t fc=heat[pinned[cold]];
    if(fh<=fc+(fc>>2)+4) return 0;
    *slot=cold; *eid=hot; *gain=(long)fh-(long)fc;
    return 1;
}
/* Byte-for-byte port of tier_decay() (c/tier.h ~L29-31). */
static void sim_tier_decay(uint32_t *heat, int nexpert){ for(int e=0;e<nexpert;e++) heat[e]>>=1; }

/* One REPIN pass across all layers: swap at most `max_swaps` worst-pin /
 * hottest-unpinned pairs (picking the single best-gain candidate per layer,
 * then globally the top `max_swaps` by gain -- mirrors repin_pick()/
 * repin_pass() at glm.c ~L3019-3069), then decay every layer's heat
 * unconditionally. This function only performs the cache-state bookkeeping
 * swap; the caller is responsible for "loading" the newly-pinned expert's
 * bytes via the I/O layer (glm.c's repin_pass does that with a real,
 * fatal=1 expert_load -- see sim_expert_io.h). Returns the number of swaps
 * performed; out_l/out_old_eid/out_new_eid (each length >= max_swaps, may be
 * NULL) receive the swaps performed, in order. */
static int sim_repin_pass(SimCache *c, int max_swaps, int *out_l, int *out_old_eid, int *out_new_eid){
    typedef struct { long gain; int l, slot, eid; } Cand;
    Cand cd[64]; int nb=0; if(max_swaps>64) max_swaps=64; if(max_swaps<0) max_swaps=0;
    for(int l=0;l<c->n_layers;l++){
        if(c->npin[l]<1) continue;
        int ids[4096]; int np=c->npin[l]; if(np>4096) np=4096;
        for(int z=0;z<np;z++) ids[z]=c->pin[l][z].eid;
        int zp,eu; long g;
        if(!sim_tier_pick_swap(c->heat[l],c->n_experts,ids,np,&zp,&eu,&g)) continue;
        if(nb<max_swaps){ cd[nb].gain=g; cd[nb].l=l; cd[nb].slot=zp; cd[nb].eid=eu; nb++; }
        else if(max_swaps>0){
            int w=0; for(int b=1;b<max_swaps;b++) if(cd[b].gain<cd[w].gain) w=b;
            if(g>cd[w].gain){ cd[w].gain=g; cd[w].l=l; cd[w].slot=zp; cd[w].eid=eu; }
        }
    }
    for(int b=0;b<nb;b++){
        int old=c->pin[cd[b].l][cd[b].slot].eid;
        c->pin[cd[b].l][cd[b].slot].eid=cd[b].eid;
        if(out_l)       out_l[b]=cd[b].l;
        if(out_old_eid) out_old_eid[b]=old;
        if(out_new_eid) out_new_eid[b]=cd[b].eid;
    }
    for(int l=0;l<c->n_layers;l++) sim_tier_decay(c->heat[l],c->n_experts);
    return nb;
}

#endif /* SIM_CACHE_H */
