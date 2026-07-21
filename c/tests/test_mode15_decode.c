/* tests/test_mode15_decode.c -- round-trip correctness test for the mode-1.5
 * (Huffman-compressed) CPU decode path wired into glm.c's qt_from_disk() and
 * expert_load() (the Mode-15 integration design notes step 4: "Wire CPU
 * decode into expert_load ... correctness milestone, not a performance one").
 * Built with -DILI_MODE15 (see Makefile) so the real mode15_decode_i4()/
 * mode15_pread_decode_i4()/expert_load_mode15() code compiles into this test
 * binary -- reusing the SAME "#define main ...; #include "../glm.c"" trick
 * tests/test_idot.c and tests/test_mode15_engine_guard.c already establish for
 * this codebase, so this exercises the REAL engine code, not a reimplementation.
 *
 * WHAT THIS PROVES (small, in-memory fixtures -- no GPU, no real model):
 *   1. qt_from_disk(): a dense tensor stored as a valid mode-1.5 (MH01)
 *      container decodes to EXACTLY the same int4-packed bytes (byte-for-byte
 *      memcmp) as the same tensor stored raw/uncompressed, and matmul_i4 on
 *      the decoded weights produces bit-identical output to matmul_i4 on the
 *      raw-reference weights.
 *   2. expert_load(): a MIXED expert (gate_proj mode-1.5-compressed, up_proj/
 *      down_proj legacy int4) decodes/loads correctly through the real
 *      coalesced-expert-load entry point -- same byte-exact + matmul-exact
 *      proof, for both the decoded tensor AND its legacy siblings (proving
 *      the "only some of the three tensors are compressed" mix the fail-loud
 *      guard's own test fixture already exercises still works end-to-end).
 *   3. expert_load(): an all-legacy expert (no mode-1.5 anywhere) still loads
 *      correctly in an ILI_MODE15-compiled binary -- the "legacy shards load
 *      byte-for-byte unchanged" requirement, checked specifically under the
 *      build that has decode support compiled in (tests/test_
 *      mode15_engine_guard.c already checks this for the OTHER build, without
 *      -DILI_MODE15).
 *   4. qt_from_disk(): a mode-1.5 blob with one corrupted payload byte (bad
 *      whole-tensor CRC32) is REJECTED -- exit(1) via mode15_unsupported(),
 *      the exact same fail-closed guard behavior as "decode unavailable" --
 *      never silently decoded into garbage weights.
 *
 * The mode-1.5 container itself is built HERE, in pure C, via the real
 * codec_row_huff.h functions (huff_build/huff_encode_row) and the real
 * mode15_reader.c CRC32 primitives (m15_crc32_init_state/_update/_final,
 * pulled in transitively through glm.c's own "#ifdef ILI_MODE15 #include
 * mode15_reader.c" -- see glm.c near the ILI_METAL includes) -- no Python
 * tooling dependency, matching this test's "small fixtures, unit-test the
 * decode path" scope.
 */
#define main ili_glm_main_unused
#include "../glm.c"
#undef main

#include <sys/wait.h>
#include <errno.h>

static int failures=0;
#define CHECK(desc,cond) do{ \
    if(!(cond)){ fprintf(stderr,"FAIL %s\n",desc); failures++; } \
    else fprintf(stderr,"ok   %s\n",desc); \
}while(0)

/* ============================================================ blob construction === */

/* Encodes sym[O*I] (each byte in [0,15], row-major) into a fresh, fully valid
 * mode-1.5 (MH01) container blob -- exact on-disk format mode15_reader.c
 * parses (magic/O/I/rows_per_block/n_blocks/tensor_crc32 header, packed
 * lengths, row_offsets, block_crc32, payload). Builds ONE shared canonical
 * Huffman codebook from this tensor's own symbol histogram (huff_build --
 * the real codec_row_huff.h entry point, not a reimplementation) and encodes
 * every row against it (huff_encode_row). Returns the blob length and hands
 * back a malloc'd buffer via *out_blob (caller frees). */
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

/* deterministic pseudo-random nibble matrix (diverse enough to force a real
 * multi-symbol canonical Huffman tree, not the single-symbol degenerate case) */
static void fill_sym(uint8_t *sym, uint32_t O, uint32_t I, uint32_t seed){
    uint32_t x=seed?seed:1;
    for(uint64_t i=0;i<(uint64_t)O*I;i++){
        x = x*1103515245u + 12345u;
        sym[i] = (uint8_t)((x>>16) & 0xF);
    }
}

/* ============================================================ fixture I/O === */
typedef struct { const char *name; const char *dtype; int64_t shape0; const void *data; int64_t nbytes; } FakeTensor;

static void write_fake_shard(const char *path, const FakeTensor *ts, int n){
    char hdr[16384]; int hp=0;
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

/* ================================================================ fork+capture === */
typedef void (*child_fn)(const char *dir);
static void run_child_and_capture(child_fn fn, const char *dir, char *out, size_t outcap, int *exit_code, int *by_signal){
    int p[2];
    if(pipe(p)!=0){ perror("pipe"); exit(2); }
    fflush(NULL);
    pid_t pid=fork();
    if(pid<0){ perror("fork"); exit(2); }
    if(pid==0){
        close(p[0]); dup2(p[1],STDERR_FILENO); close(p[1]);
        fn(dir);
        fprintf(stderr,"UNEXPECTED: decode path did not abort -- fn returned normally\n");
        fflush(stderr); _exit(42);
    }
    close(p[1]);
    size_t total=0; ssize_t n;
    while(total+1<outcap && (n=read(p[0], out+total, outcap-1-total))>0) total+=(size_t)n;
    out[total]=0; close(p[0]);
    int status=0; waitpid(pid,&status,0);
    *by_signal=WIFSIGNALED(status);
    *exit_code=WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

/* ==================================================================== test 1 ===
 * qt_from_disk(): dense mode-1.5 tensor decodes byte-exact + matmul-exact. */
#define T1_O 20u
#define T1_I 48u
static void test_qt_from_disk_dense_roundtrip(const char *dir){
    uint8_t sym[T1_O*T1_I]; fill_sym(sym, T1_O, T1_I, 0xA5A5u);
    uint8_t *blob; size_t blob_len = build_mode15_blob(sym, T1_O, T1_I, /*rows_per_block=*/8, &blob);
    CHECK("test1: blob built", blob!=NULL && blob_len>24);

    float qs[T1_O]; for(uint32_t i=0;i<T1_O;i++) qs[i]=1.0f+0.01f*(float)i;

    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    FakeTensor ts[2] = {
        { "test.dense.weight",    "U8",  (int64_t)blob_len, blob, (int64_t)blob_len },
        { "test.dense.weight.qs", "F32", T1_O, qs, (int64_t)sizeof(qs) },
    };
    write_fake_shard(shard, ts, 2);

    Model m; memset(&m,0,sizeof(m));
    st_init(&m.S, dir);
    QT t; memset(&t,0,sizeof(t));
    qt_from_disk(&m, "test.dense.weight", (int)T1_O, (int)T1_I, /*bits=*/2, /*drop=*/0, &t);

    CHECK("test1: fmt inferred as int4 (fmt==2)", t.fmt==2);
    CHECK("test1: O matches", t.O==(int)T1_O);
    CHECK("test1: I matches", t.I==(int)T1_I);

    int rb=(int)((T1_I+1)/2);
    uint8_t ref_q4[T1_O*((T1_I+1)/2)];
    for(uint32_t r=0;r<T1_O;r++) codec_pack_nibbles(sym+(size_t)r*T1_I, (int)T1_I, ref_q4+(size_t)r*rb);

    CHECK("test1: decoded q4 bytes are byte-exact vs raw-packed reference",
          t.q4 && memcmp(t.q4, ref_q4, sizeof(ref_q4))==0);
    CHECK("test1: decoded scales are byte-exact vs written .qs",
          t.s && memcmp(t.s, qs, sizeof(qs))==0);

    float x[T1_I]; for(uint32_t i=0;i<T1_I;i++) x[i]=0.1f*(float)(i%7)-0.3f;
    float y_dec[T1_O], y_ref[T1_O];
    matmul_i4(y_dec, x, t.q4, t.s, /*S=*/1, (int)T1_I, (int)T1_O);
    matmul_i4(y_ref, x, ref_q4, qs, /*S=*/1, (int)T1_I, (int)T1_O);
    CHECK("test1: matmul_i4(decoded) bit-identical to matmul_i4(raw reference)",
          memcmp(y_dec, y_ref, sizeof(y_dec))==0);
    int all_finite=1; for(uint32_t i=0;i<T1_O;i++) if(!isfinite(y_dec[i])) all_finite=0;
    CHECK("test1: matmul_i4(decoded) output is finite", all_finite);

    free(blob);
}

/* ==================================================================== test 2 ===
 * expert_load(): mixed expert -- gate_proj mode-1.5, up_proj/down_proj legacy int4. */
#define T2_MOE 12u   /* moe_inter */
#define T2_HID 16u   /* hidden */
static void test_expert_load_mixed_roundtrip(const char *dir){
    uint8_t gate_sym[T2_MOE*T2_HID]; fill_sym(gate_sym, T2_MOE, T2_HID, 0x1111u);
    uint8_t up_sym[T2_MOE*T2_HID];   fill_sym(up_sym,   T2_MOE, T2_HID, 0x2222u);
    uint8_t down_sym[T2_HID*T2_MOE]; fill_sym(down_sym, T2_HID, T2_MOE, 0x3333u);

    uint8_t *gate_blob; size_t gate_blob_len = build_mode15_blob(gate_sym, T2_MOE, T2_HID, /*rows_per_block=*/4, &gate_blob);
    CHECK("test2: gate blob built", gate_blob!=NULL);

    int up_rb=(int)((T2_HID+1)/2), down_rb=(int)((T2_MOE+1)/2);
    uint8_t up_packed[T2_MOE*((T2_HID+1)/2)], down_packed[T2_HID*((T2_MOE+1)/2)];
    for(uint32_t r=0;r<T2_MOE;r++) codec_pack_nibbles(up_sym+(size_t)r*T2_HID, (int)T2_HID, up_packed+(size_t)r*up_rb);
    for(uint32_t r=0;r<T2_HID;r++) codec_pack_nibbles(down_sym+(size_t)r*T2_MOE, (int)T2_MOE, down_packed+(size_t)r*down_rb);

    float qs_g[T2_MOE], qs_u[T2_MOE], qs_d[T2_HID];
    for(uint32_t i=0;i<T2_MOE;i++){ qs_g[i]=1.0f+0.02f*(float)i; qs_u[i]=0.9f+0.03f*(float)i; }
    for(uint32_t i=0;i<T2_HID;i++) qs_d[i]=1.1f+0.01f*(float)i;

    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    FakeTensor ts[6] = {
        { "model.layers.0.mlp.experts.0.gate_proj.weight", "U8", (int64_t)gate_blob_len, gate_blob, (int64_t)gate_blob_len },
        { "model.layers.0.mlp.experts.0.up_proj.weight",   "U8", (int64_t)sizeof(up_packed),   up_packed,   (int64_t)sizeof(up_packed) },
        { "model.layers.0.mlp.experts.0.down_proj.weight", "U8", (int64_t)sizeof(down_packed), down_packed, (int64_t)sizeof(down_packed) },
        { "model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32", T2_MOE, qs_g, (int64_t)sizeof(qs_g) },
        { "model.layers.0.mlp.experts.0.up_proj.weight.qs",   "F32", T2_MOE, qs_u, (int64_t)sizeof(qs_u) },
        { "model.layers.0.mlp.experts.0.down_proj.weight.qs", "F32", T2_HID, qs_d, (int64_t)sizeof(qs_d) },
    };
    write_fake_shard(shard, ts, 6);

    Model m; memset(&m,0,sizeof(m));
    m.c.hidden=(int)T2_HID; m.c.moe_inter=(int)T2_MOE; m.ebits=2;
    st_init(&m.S, dir);
    ESlot slot; memset(&slot,0,sizeof(slot));
    int rc=expert_load(&m, /*layer=*/0, /*eid=*/0, &slot, /*fatal=*/1);

    CHECK("test2: expert_load returns 0 (mixed mode-1.5/legacy expert)", rc==0);
    CHECK("test2: slot.eid updated", slot.eid==0);
    CHECK("test2: gate_proj fmt==2 (decoded int4)", slot.g.fmt==2);
    CHECK("test2: up_proj fmt==2 (legacy int4)",    slot.u.fmt==2);
    CHECK("test2: down_proj fmt==2 (legacy int4)",  slot.d.fmt==2);
    CHECK("test2: gate_proj shape O==moe_inter,I==hidden", slot.g.O==(int)T2_MOE && slot.g.I==(int)T2_HID);
    CHECK("test2: down_proj shape O==hidden,I==moe_inter", slot.d.O==(int)T2_HID && slot.d.I==(int)T2_MOE);

    {
        uint8_t gate_ref[T2_MOE*((T2_HID+1)/2)];
        int grb=(int)((T2_HID+1)/2);
        for(uint32_t r=0;r<T2_MOE;r++) codec_pack_nibbles(gate_sym+(size_t)r*T2_HID, (int)T2_HID, gate_ref+(size_t)r*grb);
        CHECK("test2: gate_proj (mode-1.5-decoded) q4 is byte-exact vs raw-packed reference",
              memcmp(slot.g.q4, gate_ref, sizeof(gate_ref))==0);
    }
    CHECK("test2: up_proj (legacy sibling) q4 byte-exact vs its own raw-packed bytes",
          memcmp(slot.u.q4, up_packed, sizeof(up_packed))==0);
    CHECK("test2: down_proj (legacy sibling) q4 byte-exact vs its own raw-packed bytes",
          memcmp(slot.d.q4, down_packed, sizeof(down_packed))==0);
    CHECK("test2: gate_proj scales byte-exact", memcmp(slot.g.s, qs_g, sizeof(qs_g))==0);
    CHECK("test2: up_proj scales byte-exact",   memcmp(slot.u.s, qs_u, sizeof(qs_u))==0);
    CHECK("test2: down_proj scales byte-exact", memcmp(slot.d.s, qs_d, sizeof(qs_d))==0);

    float xg[T2_HID]; for(uint32_t i=0;i<T2_HID;i++) xg[i]=0.05f*(float)(i%5)-0.1f;
    float y_dec[T2_MOE], y_ref[T2_MOE];
    matmul_i4(y_dec, xg, slot.g.q4, slot.g.s, 1, (int)T2_HID, (int)T2_MOE);
    {
        uint8_t gate_ref[T2_MOE*((T2_HID+1)/2)];
        int grb=(int)((T2_HID+1)/2);
        for(uint32_t r=0;r<T2_MOE;r++) codec_pack_nibbles(gate_sym+(size_t)r*T2_HID, (int)T2_HID, gate_ref+(size_t)r*grb);
        matmul_i4(y_ref, xg, gate_ref, qs_g, 1, (int)T2_HID, (int)T2_MOE);
    }
    CHECK("test2: matmul_i4(gate_proj decoded) bit-identical to matmul_i4(raw reference)",
          memcmp(y_dec, y_ref, sizeof(y_dec))==0);

    free(gate_blob);
}

/* ==================================================================== test 3 ===
 * expert_load(): all-legacy expert still works in an ILI_MODE15-compiled binary. */
#define T3_MOE 10u
#define T3_HID 14u
static void test_expert_load_all_legacy_under_mode15_build(const char *dir){
    uint8_t gate_sym[T3_MOE*T3_HID]; fill_sym(gate_sym, T3_MOE, T3_HID, 0x4444u);
    uint8_t up_sym[T3_MOE*T3_HID];   fill_sym(up_sym,   T3_MOE, T3_HID, 0x5555u);
    uint8_t down_sym[T3_HID*T3_MOE]; fill_sym(down_sym, T3_HID, T3_MOE, 0x6666u);

    int grb=(int)((T3_HID+1)/2), drb=(int)((T3_MOE+1)/2);
    uint8_t gate_packed[T3_MOE*((T3_HID+1)/2)], up_packed[T3_MOE*((T3_HID+1)/2)], down_packed[T3_HID*((T3_MOE+1)/2)];
    for(uint32_t r=0;r<T3_MOE;r++){
        codec_pack_nibbles(gate_sym+(size_t)r*T3_HID, (int)T3_HID, gate_packed+(size_t)r*grb);
        codec_pack_nibbles(up_sym+(size_t)r*T3_HID,   (int)T3_HID, up_packed+(size_t)r*grb);
    }
    for(uint32_t r=0;r<T3_HID;r++)
        codec_pack_nibbles(down_sym+(size_t)r*T3_MOE, (int)T3_MOE, down_packed+(size_t)r*drb);

    float qs_g[T3_MOE], qs_u[T3_MOE], qs_d[T3_HID];
    for(uint32_t i=0;i<T3_MOE;i++){ qs_g[i]=1.0f; qs_u[i]=1.0f; }
    for(uint32_t i=0;i<T3_HID;i++) qs_d[i]=1.0f;

    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    FakeTensor ts[6] = {
        { "model.layers.0.mlp.experts.0.gate_proj.weight", "U8", (int64_t)sizeof(gate_packed), gate_packed, (int64_t)sizeof(gate_packed) },
        { "model.layers.0.mlp.experts.0.up_proj.weight",   "U8", (int64_t)sizeof(up_packed),   up_packed,   (int64_t)sizeof(up_packed) },
        { "model.layers.0.mlp.experts.0.down_proj.weight", "U8", (int64_t)sizeof(down_packed), down_packed, (int64_t)sizeof(down_packed) },
        { "model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32", T3_MOE, qs_g, (int64_t)sizeof(qs_g) },
        { "model.layers.0.mlp.experts.0.up_proj.weight.qs",   "F32", T3_MOE, qs_u, (int64_t)sizeof(qs_u) },
        { "model.layers.0.mlp.experts.0.down_proj.weight.qs", "F32", T3_HID, qs_d, (int64_t)sizeof(qs_d) },
    };
    write_fake_shard(shard, ts, 6);

    Model m; memset(&m,0,sizeof(m));
    m.c.hidden=(int)T3_HID; m.c.moe_inter=(int)T3_MOE; m.ebits=2;
    st_init(&m.S, dir);
    ESlot slot; memset(&slot,0,sizeof(slot));
    int rc=expert_load(&m, 0, 0, &slot, 1);

    CHECK("test3: legacy-only expert still loads (rc==0) under an ILI_MODE15 build", rc==0);
    CHECK("test3: gate_proj bytes untouched (byte-exact vs written raw bytes)",
          memcmp(slot.g.q4, gate_packed, sizeof(gate_packed))==0);
    CHECK("test3: up_proj bytes untouched",   memcmp(slot.u.q4, up_packed, sizeof(up_packed))==0);
    CHECK("test3: down_proj bytes untouched", memcmp(slot.d.q4, down_packed, sizeof(down_packed))==0);
}

/* ==================================================================== test 4 ===
 * qt_from_disk(): a mode-1.5 blob with a corrupted payload byte must fail
 * CLOSED (mode15_unsupported()+exit(1)), never decode into wrong weights. */
static void child_qt_from_disk_corrupt(const char *dir){
    Model m; memset(&m,0,sizeof(m));
    st_init(&m.S, dir);
    QT t; memset(&t,0,sizeof(t));
    qt_from_disk(&m, "test.dense.weight", (int)T1_O, (int)T1_I, 2, 0, &t);
}
static void test_corrupted_mode15_blob_fails_closed(const char *dir){
    uint8_t sym[T1_O*T1_I]; fill_sym(sym, T1_O, T1_I, 0x7777u);
    uint8_t *blob; size_t blob_len = build_mode15_blob(sym, T1_O, T1_I, 8, &blob);
    CHECK("test4: blob built", blob!=NULL);
    blob[blob_len-1] ^= 0xFF;   /* corrupt the last payload byte -> whole-tensor CRC32 must now fail */

    float qs[T1_O]; for(uint32_t i=0;i<T1_O;i++) qs[i]=1.0f;
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    FakeTensor ts[2] = {
        { "test.dense.weight",    "U8",  (int64_t)blob_len, blob, (int64_t)blob_len },
        { "test.dense.weight.qs", "F32", T1_O, qs, (int64_t)sizeof(qs) },
    };
    write_fake_shard(shard, ts, 2);

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_qt_from_disk_corrupt, dir, out, sizeof(out), &code, &sig);
    fprintf(stderr, "--- captured child stderr ---\n%s-----------------------------\n", out);
    CHECK("test4: corrupted blob -> child did not crash by signal", !sig);
    CHECK("test4: corrupted blob -> child exit code is 1 (fail-closed, not silently decoded)", code==1);
    CHECK("test4: diagnostic names the tensor", strstr(out,"test.dense.weight")!=NULL);
    CHECK("test4: diagnostic identifies mode-1.5 format", strstr(out,"mode-1.5")!=NULL);
    CHECK("test4: diagnostic is the DECODE-FAILED message (data-integrity event), "
          "not the no-decoder-compiled-in message -- this build has decode support "
          "compiled in, so a corrupted blob must be distinguishable from a missing "
          "decoder", strstr(out,"data-integrity event")!=NULL);
    CHECK("test4: diagnostic does NOT claim the decoder is missing (it is compiled in "
          "-- this is a CRC rejection, not a build-configuration issue)",
          strstr(out,"no mode-1.5 decoder wired in")==NULL);

    free(blob);
}

int main(void){
    char root[] = "/tmp/test_mode15_decode.XXXXXX";
    if(!mkdtemp(root)){ perror("mkdtemp"); return 2; }
    char d1[600],d2[600],d3[600],d4[600];
    snprintf(d1,sizeof(d1),"%s/dense_roundtrip",root);
    snprintf(d2,sizeof(d2),"%s/mixed_expert",root);
    snprintf(d3,sizeof(d3),"%s/legacy_expert",root);
    snprintf(d4,sizeof(d4),"%s/corrupt_dense",root);
    mkdir_or_die(d1); mkdir_or_die(d2); mkdir_or_die(d3); mkdir_or_die(d4);

    test_qt_from_disk_dense_roundtrip(d1);
    test_expert_load_mixed_roundtrip(d2);
    test_expert_load_all_legacy_under_mode15_build(d3);
    test_corrupted_mode15_blob_fails_closed(d4);

    char f1[700],f2[700],f3[700],f4[700];
    snprintf(f1,sizeof(f1),"%s/model.safetensors",d1); unlink(f1);
    snprintf(f2,sizeof(f2),"%s/model.safetensors",d2); unlink(f2);
    snprintf(f3,sizeof(f3),"%s/model.safetensors",d3); unlink(f3);
    snprintf(f4,sizeof(f4),"%s/model.safetensors",d4); unlink(f4);
    rmdir(d1); rmdir(d2); rmdir(d3); rmdir(d4); rmdir(root);

    if(failures){ fprintf(stderr,"\n%d FAILURE(S)\n",failures); return 1; }
    fprintf(stderr,"\nmode-1.5 decode round-trip: all checks pass\n");
    return 0;
}
