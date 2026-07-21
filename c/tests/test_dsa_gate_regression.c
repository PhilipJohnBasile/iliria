/* SCORE-gate regression (deliverable 4a, c/bench-m5max/factorial-streaming-causality-
 * the format spec): pins dsa_gate_blocks_metal_prefill()'s real activation condition -- the S>4
 * Metal-prefill-attention gate in attention() (glm.c) must ENGAGE (i.e. the predicate must
 * be FALSE, "selection cannot restrict") for a has_dsa model at short context, and must NOT
 * engage under DSA_FORCE=1 or once context exceeds index_topk.
 *
 * Practicality note (why this tests the extracted predicate rather than a tiny-fixture
 * run): the S>4 Metal kernel's own precondition hardcodes GLM-5.2's REAL attention
 * dimensions (D==6144 && H==64 && kv_lora==512 && qk_nope==192 && qk_rope==64 && vh==256 &&
 * kv_b.fmt==2 -- see attention()), so no truly "tiny" fixture can ever reach it; only a
 * fixture built at those exact dims can (c/tests/test_score_greedy_parity.py does exactly
 * that, for deliverable 4b, and asserts a real METAL-ATTN engagement count there). Testing
 * the extracted predicate directly is the "whatever level is practical" the spec allows for
 * 4a, and it is a REAL regression test: dsa_gate_blocks_metal_prefill() is the literal
 * function attention() calls, not a reimplementation, via the `#define main ...;
 * #include "../glm.c"` trick tests/test_idot.c already established for this codebase.
 *
 * "Must FAIL against the pre-fix !dsel logic": old_buggy_gate_allows_metal() below
 * reproduces that exact pre-fix boolean (the gate ENGAGED iff `!dsel`, i.e. iff the
 * dsa_sel/dsa_nsel buffers had NEVER been allocated in this process) applied to the same
 * scenario as test 1 -- a has_dsa model where layer 0 is FULL DSA (so the buffers get
 * allocated at kv_alloc time, unconditionally, the very first time ANY request runs) and a
 * short SCORE-mode context. Under the OLD logic that scenario permanently blocks the Metal
 * gate (ever_allocated=1 for the rest of the process); under the FIXED logic
 * (dsa_gate_blocks_metal_prefill) it correctly engages. Reverting the fix would flip test
 * 1's assertion from PASS to FAIL -- this file documents and checks that divergence
 * directly rather than merely asserting the new behavior in isolation. */
#define main ili_glm_main_unused
#include "../glm.c"
#undef main

/* Mirrors the OLD (pre-fix, e567f80's parent) gate boolean EXACTLY: the S>4 Metal kernel
 * engaged iff `!dsel`, where dsel aliased m->dsa_sel and was non-NULL as soon as it had
 * EVER been allocated (kv_alloc allocates it unconditionally for any has_dsa model, the
 * first time any request runs, and it is never freed between layers or requests). So
 * "engages" reduces to "the buffers were never allocated in this process yet". */
static int old_buggy_gate_allows_metal(int dsa_buffers_ever_allocated){
    return !dsa_buffers_ever_allocated;   /* old condition was `!dsel` in the if(...) */
}

static int failures=0;
#define CHECK(desc,got,want) do{ \
    if((got)!=(want)){ fprintf(stderr,"FAIL %s: got=%d want=%d\n",desc,(int)(got),(int)(want)); failures++; } \
    else fprintf(stderr,"ok   %s\n",desc); \
}while(0)

int main(void){
    /* Test 1: has_dsa model, layer 0 (FULL DSA, mirrors GLM-5.2), SHORT context
     * (pos_base+S <= index_topk) -- selection is a documented no-op (dsa_nsel[s]==0, the
     * dense-range early-out), so the gate MUST engage: dsa_may_select must be FALSE. This
     * is the exact SCORE-mode scenario e567f80 fixed (ctx+cont ~50-300 tokens << index_topk
     * 2048). */
    {
        int has_dsa=1, n_layers=78, layer=0, dsa_force=0, pos_base=0, S=8, index_topk=2048;
        int dsa_may_select = dsa_gate_blocks_metal_prefill(has_dsa,n_layers,layer,dsa_force,pos_base,S,index_topk);
        CHECK("short-context has_dsa request: gate engages (predicate false)", dsa_may_select, 0);

        /* The regression proof: under the OLD !dsel logic, this exact scenario is blocked
         * FOREVER once the DSA buffers are ever allocated -- which, for a has_dsa model,
         * happens unconditionally on the very first request. A has_dsa model always
         * allocates them, so ever_allocated=1 here (matching real runtime behavior for
         * every has_dsa model past its first request, i.e. essentially always). */
        int dsa_buffers_ever_allocated = has_dsa;  /* kv_alloc: unconditional for has_dsa models */
        int old_gate_allows_metal = old_buggy_gate_allows_metal(dsa_buffers_ever_allocated);
        CHECK("pre-fix !dsel logic WOULD wrongly block this exact scenario", old_gate_allows_metal, 0);
        /* -> new (fixed) predicate says "engage" (dsa_may_select==0 -> !dsa_may_select==1);
         *    old predicate says "block" (old_gate_allows_metal==0). Reverting
         *    dsa_gate_blocks_metal_prefill() to the old logic flips the FIRST assertion
         *    above from PASS (0) to FAIL (would report 1, blocked) -- this is the
         *    "must fail against the pre-fix logic" requirement, checked mechanically. */
        if(!dsa_may_select == !!old_gate_allows_metal){
            fprintf(stderr,"FAIL regression-bite: fixed and pre-fix predicates agree on the "
                            "bug scenario (expected them to DISAGREE: fixed=engage, old=block)\n");
            failures++;
        } else fprintf(stderr,"ok   regression-bite: fixed predicate disagrees with pre-fix logic on the bug scenario (as required)\n");
    }

    /* Test 2: DSA_FORCE=1 must ALWAYS route to CPU (gate must NOT engage), regardless of
     * context length -- g_dsa_force exists specifically to keep the selection-vs-dense
     * self-test meaningful (top-min(k,T)=dense), so it must never take the Metal shortcut. */
    {
        int has_dsa=1, n_layers=78, layer=0, dsa_force=1, pos_base=0, S=8, index_topk=2048;
        int dsa_may_select = dsa_gate_blocks_metal_prefill(has_dsa,n_layers,layer,dsa_force,pos_base,S,index_topk);
        CHECK("DSA_FORCE=1: gate does NOT engage (predicate true) even at short context", dsa_may_select, 1);
    }

    /* Test 3: long context (pos_base+S > index_topk) -- selection genuinely restricts, so
     * the gate must NOT engage (Metal kernel only computes dense attention, which would be
     * wrong here). */
    {
        int has_dsa=1, n_layers=78, layer=0, dsa_force=0, pos_base=3000, S=8, index_topk=2048;
        int dsa_may_select = dsa_gate_blocks_metal_prefill(has_dsa,n_layers,layer,dsa_force,pos_base,S,index_topk);
        CHECK("long-context (pos_base+S > index_topk): gate does NOT engage", dsa_may_select, 1);
    }

    /* Test 4: has_dsa=0 (no DSA weights / DSA=0 override) -- gate must always engage
     * (nothing to restrict, ever), independent of context length. */
    {
        int has_dsa=0, n_layers=78, layer=0, dsa_force=0, pos_base=3000, S=8, index_topk=2048;
        int dsa_may_select = dsa_gate_blocks_metal_prefill(has_dsa,n_layers,layer,dsa_force,pos_base,S,index_topk);
        CHECK("has_dsa=0: gate always engages regardless of context", dsa_may_select, 0);
    }

    /* Test 5: layer >= n_layers (MTP head layer, e.g. layer==n_layers for the draft head)
     * is out of the DSA indexer's range by construction -- gate must engage. */
    {
        int has_dsa=1, n_layers=78, layer=78, dsa_force=0, pos_base=3000, S=8, index_topk=2048;
        int dsa_may_select = dsa_gate_blocks_metal_prefill(has_dsa,n_layers,layer,dsa_force,pos_base,S,index_topk);
        CHECK("layer>=n_layers (MTP head): gate always engages", dsa_may_select, 0);
    }

    if(failures){ fprintf(stderr,"\n%d FAILURE(S)\n",failures); return 1; }
    fprintf(stderr,"\nSCORE-gate regression: all checks pass\n");
    return 0;
}
