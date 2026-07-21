/* tests/test_mode15_slab_reuse.c -- regression tests around the "silent death
 * on a LATER turn" robustness gap found by c/scripts/mode15_e2e_certify.sh's
 * first live run against the real 744B Mode-1.5 container (2026-07-20): a
 * mode-1.5 `ili chat` session that had already completed one or more PRIOR
 * turns could die with ZERO diagnostic output (no FATAL, no crash report, no
 * OOM trace) partway through a LATER turn, always on the same prompt,
 * reproduced at two different turn indices (3rd, 5th). The identical prompt
 * always succeeds as an isolated first turn of a fresh session, as a
 * one-shot `ili run`, and on the int4 (non-mode-1.5) container at ANY turn
 * position. Decode itself was independently proven correct (single-turn
 * mode-1.5 output token-identical to int4, 5/5) -- so this is not a decode-
 * correctness bug.
 *
 * VERIFIED ROOT CAUSE (found by an external second-opinion review, confirmed
 * live against the real container -- see test 3 and expert_bytes_probe()'s
 * own comment in glm.c): cap_for_ram()'s per-layer LRU cache AUTO-RAISE
 * under-counted a mode-1.5 expert's RESIDENT (decoded) bytes as its smaller
 * COMPRESSED on-disk bytes, over-provisioning the cache by ~1/ratio (~1.33x
 * measured) relative to what the RAM budget can hold once experts are
 * genuinely resident -- confirmed live: this container's own startup log
 * read "cap raised 8->104 (projected peak 113.5 GB)" before the fix and
 * "cap raised 8->68 (projected peak 113.0 GB)" after, same 114 GB budget.
 * The cache starts EMPTY and fills gradually across turns (:reset never
 * clears it), so this cannot show up on an isolated first turn -- only once
 * enough distinct experts accumulate does real RSS cross the actual ceiling,
 * at which point the kernel's OOM killer silently ends the process (SIGKILL,
 * uncatchable by any userspace handler). test_expert_bytes_probe_mode15_
 * matches_legacy() (test 3, below) is the direct regression test for this;
 * it was verified to FAIL against the pre-fix expert_bytes_probe().
 *
 * ORIGINAL HYPOTHESIS (recorded in mode15_e2e_certify.sh's own header
 * comment, investigated first): a robustness gap in something specific to
 * REPEATED expert_load_mode15() calls against LONG-LIVED ESlot/slab state
 * across multiple turns of one persistent process -- exactly the shape
 * `ili chat`/`ili serve` exercise and a single `ili run` never does.
 * Tests 1+2 below target exactly that mechanism; both PASS against the real
 * code -- i.e. the ESlot/slab-swap ownership machinery itself checks out as
 * sound. They are kept as genuine ownership/lifecycle invariant coverage
 * (no existing mode-1.5 test exercised the ws[]<->ecache[layer] swap idiom
 * even once before this file), not because they reproduce the reported
 * incident -- the verified root cause is the cap-sizing bug above.
 *
 * This is a code-level regression test, not a live-model repro (the actual
 * bug surfaced only against a real 744B container over many cold-prefill
 * minutes per turn -- infeasible for `make test-c`). It drives the REAL
 * expert_load()/expert_load_mode15()/expert_bytes_probe() code (the
 * "#define main ...; #include "../glm.c"" trick this suite already uses,
 * see test_mode15_decode.c) far past what a single manual multi-turn
 * session would exercise, targeting things a persistent process does that
 * a single `ili run` never does, and that no existing mode-1.5 test
 * exercises even once:
 *
 *   1. test_repeated_reuse_same_slot(): the SAME physical ESlot (same
 *      slab/fslab buffers) is handed to expert_load_mode15() many times in a
 *      row for DIFFERENT experts with a DIFFERENT mode-1.5/legacy tensor mix
 *      each time (real containers can compress some experts' tensors and
 *      leave others legacy int4) -- glm.c's own ESlot comment describes
 *      exactly this: "slot di un expert ... gli slot ws[] sono riusati TRA
 *      layer". Every one of many reuse cycles is checked byte-exact against
 *      a freshly-recomputed reference, not just the first, so a bug that
 *      only corrupts the Nth reuse of a slot cannot hide behind a test that
 *      only ever loads once.
 *
 *   2. test_ws_ecache_swap_idiom(): the exact ws[]<->ecache[layer]
 *      struct-swap PROMOTION idiom glm.c's decode loop performs after every
 *      block of loaded experts ("promozione LRU (swap buffer)"):
 *          ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp;
 *      A whole-struct assignment hands slab/fslab buffer OWNERSHIP back and
 *      forth between a transient "working set" slot and a persistent
 *      per-layer cache slot -- this is the literal mechanism a multi-turn
 *      session exercises thousands of times (every cache miss, every
 *      layer, every token) that a single `ili run` prefill+decode barely
 *      touches and test_mode15_decode.c never touches at all (it only ever
 *      expert_load()s into a single freshly-zeroed ESlot). This test
 *      performs the swap explicitly, repeatedly, with alternating experts,
 *      and checks that (a) decode stays byte-exact after inheriting a
 *      recycled buffer via swap rather than fresh allocation, and (b) the
 *      QT views (g/u/d q4/q8/s pointers) always point INSIDE the slot that
 *      now legitimately owns them -- a cross-slot-pointer bug here is
 *      exactly the shape of a use-after-free/aliasing hazard that would
 *      surface as exactly this bug's reported symptom (a later, unrelated
 *      read into a since-reused/freed buffer -- silent corruption or a
 *      crash with no diagnostic printed anywhere near the actual fault).
 *
 *   3. test_expert_bytes_probe_mode15_matches_legacy(): the VERIFIED root
 *      cause -- expert_bytes_probe() must report the same resident-byte
 *      estimate for a mode-1.5 expert as for a legacy one of identical
 *      decoded shape/content, not the mode-1.5 tensor's smaller compressed
 *      on-disk size. See this test's own doc comment (right before its
 *      definition) and expert_bytes_probe()'s comment in glm.c for the full
 *      chain from this under-count to cap_for_ram()'s over-provisioned
 *      cache to a real, silent, kernel OOM-kill across enough turns.
 */
#define main ili_glm_main_unused
#include "../glm.c"
#undef main

#include <errno.h>

static int failures=0;
#define CHECK(desc,cond) do{ \
    if(!(cond)){ fprintf(stderr,"FAIL %s\n",desc); failures++; } \
    else fprintf(stderr,"ok   %s\n",desc); \
}while(0)

/* ---- fixture helpers (same conventions as test_mode15_decode.c) -------- */
typedef struct { const char *name; const char *dtype; int64_t shape0; const void *data; int64_t nbytes; } FakeTensor;

static void write_fake_shard(const char *path, const FakeTensor *ts, int n){
    char hdr[65536]; int hp=0;
    hdr[hp++]='{';
    int64_t off=0;
    for(int i=0;i<n;i++){
        int64_t a=off, b=off+ts[i].nbytes; off=b;
        hp += snprintf(hdr+hp, sizeof(hdr)-(size_t)hp,
            "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
            i?",":"", ts[i].name, ts[i].dtype, (long long)ts[i].shape0, (long long)a, (long long)b);
    }
    hdr[hp++]='}';
    uint64_t hlen=(uint64_t)hp;
    FILE *f=fopen(path,"wb");
    if(!f){ perror(path); exit(2); }
    if(fwrite(&hlen,8,1,f)!=1){ perror("fwrite hlen"); exit(2); }
    if(fwrite(hdr,1,(size_t)hp,f)!=(size_t)hp){ perror("fwrite hdr"); exit(2); }
    for(int i=0;i<n;i++)
        if(ts[i].nbytes>0 && fwrite(ts[i].data,1,(size_t)ts[i].nbytes,f)!=(size_t)ts[i].nbytes){ perror("fwrite data"); exit(2); }
    fclose(f);
}
static void mkdir_or_die(const char *path){
    if(mkdir(path,0700)!=0 && errno!=EEXIST){ perror(path); exit(2); }
}

/* deterministic pseudo-random nibble matrix -- same LCG as
 * test_mode15_decode.c's fill_sym() (kept as an independent copy, matching
 * this suite's existing convention of small self-contained test files). */
static void fill_sym(uint8_t *sym, uint32_t O, uint32_t I, uint32_t seed){
    uint32_t x=seed?seed:1;
    for(uint64_t i=0;i<(uint64_t)O*I;i++){
        x = x*1103515245u + 12345u;
        sym[i] = (uint8_t)((x>>16) & 0xF);
    }
}

/* Encodes sym[O*I] into a fresh, valid mode-1.5 (MH01) blob -- identical
 * logic to test_mode15_decode.c's build_mode15_blob() (independent copy, see
 * fill_sym() above for why). */
static size_t build_mode15_blob(const uint8_t *sym, uint32_t O, uint32_t I,
                                 uint32_t rows_per_block, uint8_t **out_blob){
    uint32_t n_blocks = O>0 ? (O + rows_per_block - 1)/rows_per_block : 0;
    uint32_t counts[CODEC_NSYM]={0};
    for(uint64_t i=0;i<(uint64_t)O*I;i++) counts[sym[i]]++;
    HuffCodebook cb; memset(&cb,0,sizeof(cb));
    huff_build(counts,&cb);

    uint32_t *row_offsets=(uint32_t*)malloc(((size_t)O+1)*sizeof(uint32_t));
    size_t cap=(size_t)O*I*2+1024;
    uint8_t *payload=(uint8_t*)malloc(cap);
    size_t p=0; row_offsets[0]=0;
    for(uint32_t r=0;r<O;r++){
        size_t used=huff_encode_row(&cb, sym+(size_t)r*I, (int)I, payload+p, cap-p);
        if(used==(size_t)-1){ free(row_offsets); free(payload); huff_free(&cb); *out_blob=NULL; return 0; }
        p+=used; row_offsets[r+1]=(uint32_t)p;
    }
    uint32_t payload_len=(uint32_t)p;

    uint32_t *block_crc=(uint32_t*)malloc((n_blocks>0?n_blocks:1)*sizeof(uint32_t));
    for(uint32_t b=0;b<n_blocks;b++){
        uint32_t r0=b*rows_per_block, r1=r0+rows_per_block; if(r1>O) r1=O;
        block_crc[b]=m15_crc32(payload+row_offsets[r0], (size_t)(row_offsets[r1]-row_offsets[r0]));
    }

    uint8_t lengths_packed[8]={0};
    for(int k=0;k<8;k++){
        uint8_t lo=cb.len[2*k]&0xF, hi=cb.len[2*k+1]&0xF;
        lengths_packed[k]=(uint8_t)(lo|(hi<<4));
    }

    size_t idx_bytes=8+((size_t)O+1)*4+(size_t)n_blocks*4;
    uint8_t *idx=(uint8_t*)malloc(idx_bytes);
    size_t q=0;
    memcpy(idx+q,lengths_packed,8); q+=8;
    for(uint32_t r=0;r<=O;r++){ uint32_t v=row_offsets[r];
        idx[q++]=(uint8_t)v; idx[q++]=(uint8_t)(v>>8); idx[q++]=(uint8_t)(v>>16); idx[q++]=(uint8_t)(v>>24); }
    for(uint32_t b=0;b<n_blocks;b++){ uint32_t v=block_crc[b];
        idx[q++]=(uint8_t)v; idx[q++]=(uint8_t)(v>>8); idx[q++]=(uint8_t)(v>>16); idx[q++]=(uint8_t)(v>>24); }

    uint32_t crc_state; m15_crc32_init_state(&crc_state);
    m15_crc32_update(&crc_state, idx, idx_bytes);
    m15_crc32_update(&crc_state, payload, payload_len);
    uint32_t tensor_crc32=m15_crc32_final(crc_state);

    size_t total=24+idx_bytes+payload_len;
    uint8_t *blob=(uint8_t*)malloc(total);
    size_t o=0;
    blob[o++]='M'; blob[o++]='H'; blob[o++]='0'; blob[o++]='1';
#define PUTU32(v) do{ uint32_t _v=(v); blob[o++]=(uint8_t)_v; blob[o++]=(uint8_t)(_v>>8); \
                       blob[o++]=(uint8_t)(_v>>16); blob[o++]=(uint8_t)(_v>>24); }while(0)
    PUTU32(O); PUTU32(I); PUTU32(rows_per_block); PUTU32(n_blocks); PUTU32(tensor_crc32);
#undef PUTU32
    memcpy(blob+o,idx,idx_bytes); o+=idx_bytes;
    memcpy(blob+o,payload,payload_len); o+=payload_len;

    free(row_offsets); free(payload); free(block_crc); free(idx); huff_free(&cb);
    *out_blob=blob;
    return total;
}

/* ------------------------------------------------------------------------
 * Fixture: NX experts (layer 0, eid 0..NX-1), the SAME (moe_inter,hidden)
 * shape for all of them -- matching the real model, where I/D are
 * architecture-wide constants in expert_load_mode15()'s own dtot/ftot
 * derivation, never per-expert -- but a DIFFERENT mode-1.5/legacy
 * composition per expert (some all-compressed, some mixed, some all-
 * legacy) and independently randomized content, so reusing one ESlot
 * across all of them repeatedly exercises every combination of "what was
 * in this buffer before vs. what is loading into it now" a real
 * multi-turn, LRU-churned cache slot can present. */
#define NX      4u
#define MOE     16u
#define HID     24u
#define ROWBUF  256   /* > max(MOE*ceil(HID/2), HID*ceil(MOE/2)) == 192 */
static uint8_t g_gate_sym[NX][MOE*HID], g_up_sym[NX][MOE*HID], g_down_sym[NX][HID*MOE];
static const int g_gate_m15[NX] = {1,1,0,1};   /* every expert has >=1 mode-1.5 tensor, */
static const int g_up_m15[NX]   = {1,0,0,1};   /* every expert has >=1 legacy tensor, and */
static const int g_down_m15[NX] = {1,0,1,0};   /* no two experts share the same 3-way pattern */

static void build_fixture(const char *dir){
    FakeTensor ts[NX*6]; int nt=0;
    uint8_t *blobs[NX*3]; int nblob=0;
    uint8_t packed[NX*3][ROWBUF]; int npacked=0;
    float qs[NX*3][32];
    char nmbuf[NX*6][160];

    for(uint32_t e=0;e<NX;e++){
        fill_sym(g_gate_sym[e], MOE, HID, 0x1000u+e*7u+1u);
        fill_sym(g_up_sym[e],   MOE, HID, 0x2000u+e*11u+1u);
        fill_sym(g_down_sym[e], HID, MOE, 0x3000u+e*13u+1u);

        struct { const uint8_t *sym; uint32_t O,I; int is_m15; const char *suf; } tt[3] = {
            { g_gate_sym[e], MOE, HID, g_gate_m15[e], "gate_proj" },
            { g_up_sym[e],   MOE, HID, g_up_m15[e],   "up_proj"   },
            { g_down_sym[e], HID, MOE, g_down_m15[e], "down_proj" },
        };
        for(int k=0;k<3;k++){
            snprintf(nmbuf[nt],160,"model.layers.0.mlp.experts.%u.%s.weight",e,tt[k].suf);
            if(tt[k].is_m15){
                uint8_t *blob; size_t blen=build_mode15_blob(tt[k].sym,tt[k].O,tt[k].I,/*rows_per_block=*/4,&blob);
                if(!blob){ fprintf(stderr,"build_mode15_blob failed (expert %u tensor %d)\n",e,k); exit(2); }
                blobs[nblob++]=blob;
                ts[nt].name=nmbuf[nt]; ts[nt].dtype="U8"; ts[nt].shape0=(int64_t)blen;
                ts[nt].data=blob; ts[nt].nbytes=(int64_t)blen;
            } else {
                int rb=(int)((tt[k].I+1)/2);
                for(uint32_t r=0;r<tt[k].O;r++)
                    codec_pack_nibbles(tt[k].sym+(size_t)r*tt[k].I,(int)tt[k].I,packed[npacked]+(size_t)r*rb);
                int64_t nb=(int64_t)tt[k].O*rb;
                ts[nt].name=nmbuf[nt]; ts[nt].dtype="U8"; ts[nt].shape0=nb;
                ts[nt].data=packed[npacked]; ts[nt].nbytes=nb;
                npacked++;
            }
            nt++;
        }
        for(int k=0;k<3;k++){
            uint32_t O = tt[k].O; int qi=(int)(e*3)+k;
            for(uint32_t i=0;i<O;i++) qs[qi][i]=1.0f+0.01f*(float)(i+e);
            snprintf(nmbuf[nt],160,"model.layers.0.mlp.experts.%u.%s.weight.qs",e,tt[k].suf);
            ts[nt].name=nmbuf[nt]; ts[nt].dtype="F32"; ts[nt].shape0=(int64_t)O;
            ts[nt].data=qs[qi]; ts[nt].nbytes=(int64_t)O*4;
            nt++;
        }
    }
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    write_fake_shard(shard, ts, nt);
    for(int i=0;i<nblob;i++) free(blobs[i]);
}

/* recompute expert e's exact expected int4-packed bytes fresh (never cached
 * across calls), so a bug that corrupts only, say, the 15th reuse of a slot
 * cannot hide behind a stale/precomputed comparison. */
static void reference_bytes(uint32_t e, uint8_t *ref_g, uint8_t *ref_u, uint8_t *ref_d){
    int grb=(int)((HID+1)/2), drb=(int)((MOE+1)/2);
    for(uint32_t r=0;r<MOE;r++){
        codec_pack_nibbles(g_gate_sym[e]+(size_t)r*HID,(int)HID,ref_g+(size_t)r*grb);
        codec_pack_nibbles(g_up_sym[e]+(size_t)r*HID,  (int)HID,ref_u+(size_t)r*grb);
    }
    for(uint32_t r=0;r<HID;r++)
        codec_pack_nibbles(g_down_sym[e]+(size_t)r*MOE,(int)MOE,ref_d+(size_t)r*drb);
}

/* ==================================================================== test 1 ===
 * The SAME long-lived ESlot, reloaded with a DIFFERENT expert (different
 * mode-1.5/legacy mix, different content) many times in a row -- the LRU
 * cache-slot-reuse pattern a multi-turn session exercises thousands of
 * times and a single `ili run` prefill+decode barely touches. */
static void test_repeated_reuse_same_slot(const char *dir){
    Model m; memset(&m,0,sizeof(m));
    m.c.hidden=(int)HID; m.c.moe_inter=(int)MOE; m.ebits=2;
    st_init(&m.S, dir);

    ESlot slot; memset(&slot,0,sizeof(slot));
    int cycles=6;   /* 6 passes x NX=4 experts = 24 expert_load_mode15 calls reusing
                     * ONE physical slab/fslab pair -- well past the documented
                     * repro's 3rd/5th-turn window, deliberately, so a bug that only
                     * manifests after "enough" reuse still gets caught here. */
    int all_ok=1;
    int64_t want_dtot=(int64_t)MOE*((HID+1)/2)*2 + (int64_t)HID*((MOE+1)/2);
    int64_t want_ftot=(int64_t)MOE+MOE+HID;

    for(int c=0;c<cycles && all_ok;c++){
        for(uint32_t e=0;e<NX;e++){
            int rc=expert_load(&m, /*layer=*/0, /*eid=*/(int)e, &slot, /*fatal=*/1);
            if(rc!=0){ fprintf(stderr,"cycle %d expert %u: expert_load returned %d (want 0)\n",c,e,rc); all_ok=0; break; }
            if(slot.eid!=(int)e){ fprintf(stderr,"cycle %d expert %u: slot.eid==%d after load\n",c,e,slot.eid); all_ok=0; break; }

            if(!slot.slab || slot.slab_cap<want_dtot){
                fprintf(stderr,"cycle %d expert %u: slab NULL or slab_cap too small (%lld < %lld)\n",
                        c,e,(long long)slot.slab_cap,(long long)want_dtot); all_ok=0; break; }
            if(!slot.fslab || slot.fslab_cap<want_ftot){
                fprintf(stderr,"cycle %d expert %u: fslab NULL or fslab_cap too small (%lld < %lld)\n",
                        c,e,(long long)slot.fslab_cap,(long long)want_ftot); all_ok=0; break; }

            uint8_t ref_g[MOE*((HID+1)/2)], ref_u[MOE*((HID+1)/2)], ref_d[HID*((MOE+1)/2)];
            reference_bytes(e, ref_g, ref_u, ref_d);
            if(memcmp(slot.g.q4,ref_g,sizeof(ref_g))!=0){ fprintf(stderr,"cycle %d expert %u: gate_proj MISMATCH\n",c,e); all_ok=0; break; }
            if(memcmp(slot.u.q4,ref_u,sizeof(ref_u))!=0){ fprintf(stderr,"cycle %d expert %u: up_proj MISMATCH\n",c,e); all_ok=0; break; }
            if(memcmp(slot.d.q4,ref_d,sizeof(ref_d))!=0){ fprintf(stderr,"cycle %d expert %u: down_proj MISMATCH\n",c,e); all_ok=0; break; }
        }
    }
    CHECK("repeated reuse: 24 expert_load_mode15 calls into the SAME long-lived "
          "ESlot (alternating mode-1.5/legacy composition) each decode byte-exact, "
          "and slab/fslab capacity bookkeeping stays sane every time -- not just "
          "on the first load", all_ok);
}

/* ==================================================================== test 2 ===
 * The exact ws[]<->ecache[layer] struct-swap PROMOTION idiom (glm.c,
 * "promozione LRU (swap buffer)"):
 *     ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp;
 * -- a whole-struct assignment that hands slab/fslab buffer OWNERSHIP back
 * and forth between a transient scratch slot and a persistent cache slot.
 * This is the literal mechanism a multi-turn session performs on every
 * cache miss of every layer of every token; no existing mode-1.5 test
 * performs this swap even once. */
static void test_ws_ecache_swap_idiom(const char *dir){
    Model m; memset(&m,0,sizeof(m));
    m.c.hidden=(int)HID; m.c.moe_inter=(int)MOE; m.ebits=2;
    st_init(&m.S, dir);

    ESlot ws_slot, cache_slot;
    memset(&ws_slot,0,sizeof(ws_slot));
    memset(&cache_slot,0,sizeof(cache_slot));

    int all_ok=1;
    int rounds=10;
    for(int r=0;r<rounds && all_ok;r++){
        uint32_t e=(uint32_t)(r % (int)NX);

        int rc=expert_load(&m, 0, (int)e, &ws_slot, 1);
        if(rc!=0){ fprintf(stderr,"round %d: expert_load into ws_slot returned %d\n",r,rc); all_ok=0; break; }

        /* promote: swap ws_slot's freshly-loaded buffers into cache_slot,
         * byte-for-byte the same idiom glm.c's own decode loop performs.
         * cache_slot's PRIOR occupant (round r-1's expert, or a zeroed slot
         * on round 0) moves into ws_slot -- the recycled buffer the NEXT
         * expert_load() call below must correctly grow-or-reuse. */
        ESlot tmp=cache_slot; cache_slot=ws_slot; ws_slot=tmp;

        if(cache_slot.eid!=(int)e){ fprintf(stderr,"round %d: cache_slot.eid==%d, want %d\n",r,cache_slot.eid,(int)e); all_ok=0; break; }

        uint8_t ref_g[MOE*((HID+1)/2)], ref_u[MOE*((HID+1)/2)], ref_d[HID*((MOE+1)/2)];
        reference_bytes(e, ref_g, ref_u, ref_d);
        if(memcmp(cache_slot.g.q4,ref_g,sizeof(ref_g))!=0){ fprintf(stderr,"round %d: cache_slot gate_proj MISMATCH after swap\n",r); all_ok=0; break; }
        if(memcmp(cache_slot.u.q4,ref_u,sizeof(ref_u))!=0){ fprintf(stderr,"round %d: cache_slot up_proj MISMATCH after swap\n",r); all_ok=0; break; }
        if(memcmp(cache_slot.d.q4,ref_d,sizeof(ref_d))!=0){ fprintf(stderr,"round %d: cache_slot down_proj MISMATCH after swap\n",r); all_ok=0; break; }

        /* the QT views (g/u/d q4/q8/s pointers) must point INSIDE cache_slot's
         * OWN slab after the swap, never into ws_slot's -- a cross-slot
         * pointer here is exactly a use-after-free/aliasing hazard once
         * ws_slot's buffer is reallocated/overwritten by the next
         * expert_load() call, and would surface as silent corruption or an
         * unexplained crash far from the actual fault -- this bug's
         * reported symptom. Compared as uintptr_t, not raw pointer relational
         * ops: if a view pointed into a DIFFERENT allocation (ws_slot's, or
         * anything else) -- exactly the failure this checks for -- the two
         * pointers would not share an array/object and </>= on them would be
         * undefined behavior, not just a wrong answer. */
        uintptr_t slab0=(uintptr_t)cache_slot.slab, slab1=slab0+(uintptr_t)cache_slot.slab_cap;
        uintptr_t pg=(uintptr_t)cache_slot.g.q4, pu=(uintptr_t)cache_slot.u.q4, pd=(uintptr_t)cache_slot.d.q4;
        int inside = pg>=slab0 && pg<slab1 && pu>=slab0 && pu<slab1 && pd>=slab0 && pd<slab1;
        if(!inside){ fprintf(stderr,"round %d: a QT view points outside cache_slot.slab (cross-slot aliasing)\n",r); all_ok=0; break; }
    }
    CHECK("ws<->ecache struct-swap promotion idiom: 10 rounds of load-then-swap "
          "(alternating experts/compression mixes) all decode byte-exact and keep "
          "every QT view pointing inside its OWN slot's slab after the swap",
          all_ok);
}

/* ==================================================================== test 3 ===
 * expert_bytes_probe() must estimate the SAME resident-byte count for a
 * mode-1.5-compressed expert as for an equivalent LEGACY (raw int4) one of
 * identical decoded shape/content -- this IS the actual root cause an external
 * second-opinion review found while chasing this bug (2026-07-20):
 * expert_bytes_probe() used to sum st_nbytes() unconditionally, which for a
 * mode-1.5 tensor is the COMPRESSED on-disk size, not the DECODED resident
 * size expert_load_mode15() actually allocates. cap_for_ram() (which drives
 * the per-layer LRU cache's AUTO-RAISE) trusted that under-count,
 * provisioning a cache with ~1/ratio (measured ~1.33x on the real
 * container) too many expert SLOTS for the RAM budget once they are
 * genuinely resident in decoded form -- confirmed live against the real
 * 744B container: its own startup log read "cap raised 8->104 (projected
 * peak 113.5 GB)" before this fix and "cap raised 8->68 (projected peak
 * 113.0 GB)" after, same 114 GB budget, same container. The cache starts
 * EMPTY and fills gradually across MANY turns (:reset never clears it), so
 * this never shows up on an isolated first turn -- only once enough
 * distinct experts accumulate does real RSS cross the actual physical
 * ceiling, at which point the kernel's own OOM killer silently ends the
 * process (SIGKILL, uncatchable by any userspace handler, including the
 * fail-loud signal handler installed in main()) -- this bug's exact
 * reported shape. */
static void test_expert_bytes_probe_mode15_matches_legacy(const char *dir_m15, const char *dir_legacy){
    uint8_t gate_sym[MOE*HID], up_sym[MOE*HID], down_sym[HID*MOE];
    fill_sym(gate_sym, MOE, HID, 0xAAAAu);
    fill_sym(up_sym,   MOE, HID, 0xBBBBu);
    fill_sym(down_sym, HID, MOE, 0xCCCCu);
    float qs_g[MOE], qs_u[MOE], qs_d[HID];
    for(uint32_t i=0;i<MOE;i++){ qs_g[i]=1.0f; qs_u[i]=1.0f; }
    for(uint32_t i=0;i<HID;i++) qs_d[i]=1.0f;

    /* mode-1.5 variant: all three tensors compressed */
    uint8_t *gate_blob=NULL,*up_blob=NULL,*down_blob=NULL;
    size_t gate_len=build_mode15_blob(gate_sym,MOE,HID,4,&gate_blob);
    size_t up_len  =build_mode15_blob(up_sym,  MOE,HID,4,&up_blob);
    size_t down_len=build_mode15_blob(down_sym,HID,MOE,4,&down_blob);
    CHECK("bytes-probe test: all 3 mode15 blobs built", gate_blob && up_blob && down_blob);

    char shard_m15[512]; snprintf(shard_m15,sizeof(shard_m15),"%s/model.safetensors",dir_m15);
    FakeTensor ts_m15[6] = {
        { "model.layers.0.mlp.experts.0.gate_proj.weight", "U8", (int64_t)gate_len, gate_blob, (int64_t)gate_len },
        { "model.layers.0.mlp.experts.0.up_proj.weight",   "U8", (int64_t)up_len,   up_blob,   (int64_t)up_len },
        { "model.layers.0.mlp.experts.0.down_proj.weight", "U8", (int64_t)down_len, down_blob, (int64_t)down_len },
        { "model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32", MOE, qs_g, (int64_t)sizeof(qs_g) },
        { "model.layers.0.mlp.experts.0.up_proj.weight.qs",   "F32", MOE, qs_u, (int64_t)sizeof(qs_u) },
        { "model.layers.0.mlp.experts.0.down_proj.weight.qs", "F32", HID, qs_d, (int64_t)sizeof(qs_d) },
    };
    write_fake_shard(shard_m15, ts_m15, 6);
    free(gate_blob); free(up_blob); free(down_blob);

    /* legacy variant: same content, raw int4-packed (uncompressed) */
    int grb=(int)((HID+1)/2), drb=(int)((MOE+1)/2);
    uint8_t gate_packed[MOE*((HID+1)/2)], up_packed[MOE*((HID+1)/2)], down_packed[HID*((MOE+1)/2)];
    for(uint32_t r=0;r<MOE;r++){
        codec_pack_nibbles(gate_sym+(size_t)r*HID,(int)HID,gate_packed+(size_t)r*grb);
        codec_pack_nibbles(up_sym+(size_t)r*HID,  (int)HID,up_packed+(size_t)r*grb);
    }
    for(uint32_t r=0;r<HID;r++)
        codec_pack_nibbles(down_sym+(size_t)r*MOE,(int)MOE,down_packed+(size_t)r*drb);

    char shard_legacy[512]; snprintf(shard_legacy,sizeof(shard_legacy),"%s/model.safetensors",dir_legacy);
    FakeTensor ts_legacy[6] = {
        { "model.layers.0.mlp.experts.0.gate_proj.weight", "U8", (int64_t)sizeof(gate_packed), gate_packed, (int64_t)sizeof(gate_packed) },
        { "model.layers.0.mlp.experts.0.up_proj.weight",   "U8", (int64_t)sizeof(up_packed),   up_packed,   (int64_t)sizeof(up_packed) },
        { "model.layers.0.mlp.experts.0.down_proj.weight", "U8", (int64_t)sizeof(down_packed), down_packed, (int64_t)sizeof(down_packed) },
        { "model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32", MOE, qs_g, (int64_t)sizeof(qs_g) },
        { "model.layers.0.mlp.experts.0.up_proj.weight.qs",   "F32", MOE, qs_u, (int64_t)sizeof(qs_u) },
        { "model.layers.0.mlp.experts.0.down_proj.weight.qs", "F32", HID, qs_d, (int64_t)sizeof(qs_d) },
    };
    write_fake_shard(shard_legacy, ts_legacy, 6);

    Model m_m15; memset(&m_m15,0,sizeof(m_m15));
    m_m15.c.hidden=(int)HID; m_m15.c.moe_inter=(int)MOE; m_m15.c.first_dense=0; m_m15.ebits=2;
    st_init(&m_m15.S, dir_m15);

    Model m_legacy; memset(&m_legacy,0,sizeof(m_legacy));
    m_legacy.c.hidden=(int)HID; m_legacy.c.moe_inter=(int)MOE; m_legacy.c.first_dense=0; m_legacy.ebits=2;
    st_init(&m_legacy.S, dir_legacy);

    int64_t eb_m15    = expert_bytes_probe(&m_m15, 2);
    int64_t eb_legacy = expert_bytes_probe(&m_legacy, 2);

    fprintf(stderr,"bytes-probe: mode15=%lld legacy=%lld\n",(long long)eb_m15,(long long)eb_legacy);
    CHECK("expert_bytes_probe: mode-1.5 and legacy fixtures of IDENTICAL decoded shape "
          "report the SAME resident-byte estimate (the actual root cause: mode-1.5 used "
          "to report its smaller COMPRESSED on-disk size, undercounting real RAM use and "
          "letting cap_for_ram() over-provision the per-layer expert cache)",
          eb_m15 == eb_legacy);
}

int main(void){
    char root[] = "/tmp/test_mode15_slab_reuse.XXXXXX";
    if(!mkdtemp(root)){ perror("mkdtemp"); return 2; }
    mkdir_or_die(root);
    build_fixture(root);

    test_repeated_reuse_same_slot(root);
    test_ws_ecache_swap_idiom(root);

    char root2[] = "/tmp/test_mode15_bytes_probe.XXXXXX";
    if(!mkdtemp(root2)){ perror("mkdtemp"); return 2; }
    char dm15[600], dleg[600];
    snprintf(dm15,sizeof(dm15),"%s/mode15",root2); snprintf(dleg,sizeof(dleg),"%s/legacy",root2);
    mkdir_or_die(dm15); mkdir_or_die(dleg);
    test_expert_bytes_probe_mode15_matches_legacy(dm15, dleg);
    char f2[700]; snprintf(f2,sizeof(f2),"%s/model.safetensors",dm15); unlink(f2);
    snprintf(f2,sizeof(f2),"%s/model.safetensors",dleg); unlink(f2);
    rmdir(dm15); rmdir(dleg); rmdir(root2);

    char f1[700]; snprintf(f1,sizeof(f1),"%s/model.safetensors",root); unlink(f1);
    rmdir(root);

    if(failures){ fprintf(stderr,"\n%d FAILURE(S)\n",failures); return 1; }
    fprintf(stderr,"\nmode-1.5 slab-reuse-across-turns regression: all checks pass\n");
    return 0;
}
