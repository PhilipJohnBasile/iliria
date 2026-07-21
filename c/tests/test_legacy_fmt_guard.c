/* tests/test_legacy_fmt_guard.c -- unit tests for the legacy (non-mode-1.5) int8/
 * int4/int2 format-determination fix in glm.c's qt_from_disk() and expert_load()
 * (both the mmap zero-copy fast path and the main pread path).
 *
 * BUG THIS REPLACES: the byte-count format heuristic used to be an unconditional
 * ternary `(nb==O*I)?1 : (nb==O*ceil(I/2))?2 : 3` -- ANY size that matched NEITHER
 * int8 NOR int4 fell through to the `: 3` (int2) bucket UNCHECKED, even if it did
 * not match int2's own expected size either. A truncated, corrupted, or wrong-
 * shape/wrong-bitwidth legacy tensor was therefore silently treated as valid int2
 * data -- garbage weights, no error, ever, at all three sites (qt_from_disk, the
 * g_mmap zero-copy fast path, and expert_load's main pread path). The fix
 * (legacy_fmt_from_size()/legacy_fmt_unknown() in glm.c) replaces the ternary with
 * a real membership test against all three formats' EXACT expected byte counts and
 * fails CLOSED (fprintf diagnostic + exit(1)/return -1, same fatal-vs-speculative
 * contract as every other fatal load error in this codebase) on anything else.
 *
 * WHAT THIS PROVES:
 *   1. qt_from_disk(): a dense tensor whose size matches none of int8/int4/int2
 *      is REJECTED (exit(1), diagnostic naming the tensor + observed-vs-expected
 *      sizes), not silently decoded as int2.
 *   2. qt_from_disk(): legacy int8 (fmt=1) AND int2 (fmt=3) dense tensors --
 *      whose sizes DO match -- still load byte-for-byte unchanged (previously
 *      only int4/fmt=2 had any test coverage anywhere in this suite).
 *   3. expert_load() (default, non-mmap path): one bogus-sized tensor among an
 *      otherwise-legacy-int4 expert's three is REJECTED, naming that specific
 *      tensor, not silently treated as int2.
 *   4. expert_load()'s g_mmap=1 zero-copy fast path: the SAME bogus-sized expert
 *      is REJECTED (falls through to the main path, which fails closed there)
 *      rather than committing a guessed format to a zero-copy view straight into
 *      file-backed memory -- the single highest-risk site for this class of bug,
 *      since there the "decoded" bytes ARE the raw file bytes, read directly by
 *      matmul with no intervening buffer at all.
 *
 * Uses the same "#define main ...; #include "../glm.c"" trick as tests/test_
 * mode15_engine_guard.c (exercises the real engine code, not a reimplementation).
 * No -DILI_MODE15 needed: this bug and its fix are orthogonal to mode-1.5.
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

/* ============================================================ fixture I/O === */
typedef struct { const char *name; const char *dtype; int64_t shape0; const void *data; int64_t nbytes; } FakeTensor;

/* Plain writer (offsets NOT guaranteed 4-byte aligned) -- fine for every test
 * except the g_mmap fast-path one, which requires alignment to even reach the
 * code path under test (map_of_fd()+`(off)&3` check) -- see
 * write_fake_shard_aligned() below for that one. */
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

/* Same format, but prepends a throwaway "__metadata__" padding field (st_init
 * already special-cases and skips this key when building the tensor table --
 * an existing, supported no-op, not reliance on unverified whitespace
 * tolerance in this codebase's own tiny JSON parser) with a tuned-length pad
 * string so that data_start (8+hlen) lands on a 4-byte boundary. Every tensor
 * byte length in `ts` must ALSO already be a multiple of 4 (caller's
 * responsibility -- true for the aligned test's own fixture) so alignment is
 * preserved for every subsequent tensor too, not just the first. */
static void write_fake_shard_aligned(const char *path, const FakeTensor *ts, int n){
    for(int pad=0; pad<4; pad++){
        char hdr[16384]; int hp=0;
        hp += snprintf(hdr+hp,sizeof(hdr)-(size_t)hp,"{\"__metadata__\":{\"pad\":\"%.*s\"}", pad, "xxxx");
        int64_t off=0;
        for(int i=0;i<n;i++){
            int64_t a=off, b=off+ts[i].nbytes; off=b;
            hp += snprintf(hdr+hp, sizeof(hdr)-(size_t)hp,
                ",\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
                ts[i].name, ts[i].dtype, (long long)ts[i].shape0, (long long)a, (long long)b);
        }
        hdr[hp++]='}';
        int64_t data_start = 8+hp;
        if((data_start & 3)==0){
            uint64_t hlen=(uint64_t)hp;
            FILE *f=fopen(path,"wb");
            if(!f){ perror(path); exit(2); }
            if(fwrite(&hlen,8,1,f)!=1){ perror("fwrite hlen"); exit(2); }
            if(fwrite(hdr,1,(size_t)hp,f)!=(size_t)hp){ perror("fwrite hdr"); exit(2); }
            for(int i=0;i<n;i++)
                if(ts[i].nbytes>0 && fwrite(ts[i].data,1,(size_t)ts[i].nbytes,f)!=(size_t)ts[i].nbytes){ perror("fwrite data"); exit(2); }
            fclose(f);
            return;
        }
    }
    fprintf(stderr,"write_fake_shard_aligned: could not find a 4-byte-aligned padding in 4 tries (should be impossible)\n");
    exit(2);
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
        fprintf(stderr,"UNEXPECTED: format guard did not abort -- fn returned normally\n");
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
 * qt_from_disk(): a dense tensor whose size matches NONE of int8/int4/int2 must
 * be REJECTED, not silently treated as int2. */
#define D1_O 8
#define D1_I 8   /* int8=64B, int4=32B, int2=16B -- 50B matches none of the three */
#define D1_BOGUS_NBYTES 50
static void child_qt_from_disk_bogus_size(const char *dir){
    Model m; memset(&m,0,sizeof(m));
    st_init(&m.S, dir);
    QT t; memset(&t,0,sizeof(t));
    qt_from_disk(&m, "test.dense.weight", D1_O, D1_I, /*bits=*/2, /*drop=*/0, &t);
}
static void test_qt_from_disk_rejects_unknown_size(const char *dir){
    uint8_t junk[D1_BOGUS_NBYTES]; for(int i=0;i<D1_BOGUS_NBYTES;i++) junk[i]=(uint8_t)(i*7+3);
    float qs[D1_O]; for(int i=0;i<D1_O;i++) qs[i]=1.0f;
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    FakeTensor ts[2] = {
        { "test.dense.weight",    "U8",  D1_BOGUS_NBYTES, junk, D1_BOGUS_NBYTES },
        { "test.dense.weight.qs", "F32", D1_O, qs, (int64_t)sizeof(qs) },
    };
    write_fake_shard(shard, ts, 2);

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_qt_from_disk_bogus_size, dir, out, sizeof(out), &code, &sig);
    fprintf(stderr, "--- captured child stderr ---\n%s-----------------------------\n", out);
    CHECK("test1: bogus-size dense tensor -> child did not crash by signal", !sig);
    CHECK("test1: bogus-size dense tensor -> child exit code is 1 (fail-closed, not silent int2)", code==1);
    CHECK("test1: diagnostic names the tensor", strstr(out,"test.dense.weight")!=NULL);
    CHECK("test1: diagnostic states no known format matched", strstr(out,"matches NONE of the known")!=NULL);
    CHECK("test1: diagnostic gives the observed size", strstr(out,"50 bytes")!=NULL);
    CHECK("test1: diagnostic gives the int8-expected size", strstr(out,"int8 expects exactly 64")!=NULL);
}

/* ==================================================================== test 2 ===
 * qt_from_disk(): legacy int8 (fmt=1) and int2 (fmt=3) dense tensors -- whose
 * sizes DO match -- still load correctly (previously only int4/fmt=2 had any
 * coverage in this suite). */
#define D2_O 6
#define D2_I 10
static void test_qt_from_disk_legacy_int8_loads(const char *dir){
    int64_t nb = (int64_t)D2_O*D2_I;   /* int8: 1 byte/param, exact */
    uint8_t raw[D2_O*D2_I]; for(int i=0;i<D2_O*D2_I;i++) raw[i]=(uint8_t)(int8_t)(i-30);
    float qs[D2_O]; for(int i=0;i<D2_O;i++) qs[i]=0.5f+0.1f*(float)i;
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    FakeTensor ts[2] = {
        { "test.int8.weight",    "U8",  nb,   raw, nb },
        { "test.int8.weight.qs", "F32", D2_O, qs,  (int64_t)sizeof(qs) },
    };
    write_fake_shard(shard, ts, 2);

    Model m; memset(&m,0,sizeof(m));
    st_init(&m.S, dir);
    QT t; memset(&t,0,sizeof(t));
    qt_from_disk(&m, "test.int8.weight", D2_O, D2_I, /*bits=*/8, /*drop=*/0, &t);

    CHECK("test2a: legacy int8 tensor -> fmt inferred as 1", t.fmt==1);
    CHECK("test2a: legacy int8 tensor -> O/I match", t.O==D2_O && t.I==D2_I);
    CHECK("test2a: legacy int8 tensor -> q8 bytes byte-exact vs written raw bytes",
          t.q8 && memcmp(t.q8, raw, (size_t)nb)==0);
    CHECK("test2a: legacy int8 tensor -> scales byte-exact", t.s && memcmp(t.s, qs, sizeof(qs))==0);
}
static void test_qt_from_disk_legacy_int2_loads(const char *dir){
    int64_t rb = (D2_I+3)/4;
    int64_t nb = (int64_t)D2_O*rb;   /* int2: 4 params/byte, exact */
    uint8_t raw[D2_O*((D2_I+3)/4)]; for(int i=0;i<(int)sizeof(raw);i++) raw[i]=(uint8_t)(i*13+1);
    float qs[D2_O]; for(int i=0;i<D2_O;i++) qs[i]=1.5f+0.2f*(float)i;
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    FakeTensor ts[2] = {
        { "test.int2.weight",    "U8",  nb,   raw, nb },
        { "test.int2.weight.qs", "F32", D2_O, qs,  (int64_t)sizeof(qs) },
    };
    write_fake_shard(shard, ts, 2);

    Model m; memset(&m,0,sizeof(m));
    st_init(&m.S, dir);
    QT t; memset(&t,0,sizeof(t));
    qt_from_disk(&m, "test.int2.weight", D2_O, D2_I, /*bits=*/2, /*drop=*/0, &t);

    CHECK("test2b: legacy int2 tensor -> fmt inferred as 3", t.fmt==3);
    CHECK("test2b: legacy int2 tensor -> O/I match", t.O==D2_O && t.I==D2_I);
    CHECK("test2b: legacy int2 tensor -> q4 bytes byte-exact vs written raw bytes",
          t.q4 && memcmp(t.q4, raw, (size_t)nb)==0);
    CHECK("test2b: legacy int2 tensor -> scales byte-exact", t.s && memcmp(t.s, qs, sizeof(qs))==0);
}

/* ==================================================================== test 3 ===
 * expert_load() (default, g_mmap=0): one bogus-sized tensor among an otherwise-
 * legacy-int4 expert must be REJECTED, naming that specific tensor. */
#define E_MOE 8
#define E_HID 8   /* up_proj [E_MOE,E_HID]: int8=64B, int4=32B, int2=16B -- 50B matches none */
static void child_expert_load_bogus_size(const char *dir){
    Model m; memset(&m,0,sizeof(m));
    m.c.hidden=E_HID; m.c.moe_inter=E_MOE; m.ebits=2;
    st_init(&m.S, dir);
    ESlot slot; memset(&slot,0,sizeof(slot));
    expert_load(&m, /*layer=*/0, /*eid=*/0, &slot, /*fatal=*/1);
}
/* Tensor ORDER matters for the mmap_aligned case: up_proj's bogus 50-byte size
 * is deliberately NOT a multiple of 4, so whatever tensor comes AFTER it in the
 * file would land at a misaligned offset -- which would disqualify the g_mmap
 * fast path via the PRE-EXISTING alignment check (`(off)&3`), not the NEW size
 * check this test exists to isolate. Writing up_proj LAST sidesteps that: every
 * tensor before it has a 4-byte-multiple length (32 bytes each), so all of
 * THEIR offsets stay aligned, and up_proj's own offset is also aligned (its
 * predecessors' lengths sum to a multiple of 4) even though its own length
 * isn't -- which is fine, since nothing follows it in the file. */
static void write_expert_shard_one_bogus(const char *path, int mmap_aligned){
    uint8_t gate[E_MOE*((E_HID+1)/2)]; for(unsigned i=0;i<sizeof(gate);i++) gate[i]=(uint8_t)(i+1);
    uint8_t down[E_HID*((E_MOE+1)/2)]; for(unsigned i=0;i<sizeof(down);i++) down[i]=(uint8_t)(i+9);
    uint8_t up_bogus[50]; for(int i=0;i<50;i++) up_bogus[i]=(uint8_t)(i*3+1);   /* matches no format for [E_MOE,E_HID] */
    float qs_g[E_MOE], qs_u[E_MOE], qs_d[E_HID];
    for(int i=0;i<E_MOE;i++){ qs_g[i]=1.0f; qs_u[i]=1.0f; }
    for(int i=0;i<E_HID;i++) qs_d[i]=1.0f;

    FakeTensor ts[6] = {
        { "model.layers.0.mlp.experts.0.gate_proj.weight",    "U8",  (int64_t)sizeof(gate), gate, (int64_t)sizeof(gate) },
        { "model.layers.0.mlp.experts.0.down_proj.weight",    "U8",  (int64_t)sizeof(down), down, (int64_t)sizeof(down) },
        { "model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32", E_MOE, qs_g, (int64_t)sizeof(qs_g) },
        { "model.layers.0.mlp.experts.0.down_proj.weight.qs", "F32", E_HID, qs_d, (int64_t)sizeof(qs_d) },
        { "model.layers.0.mlp.experts.0.up_proj.weight.qs",   "F32", E_MOE, qs_u, (int64_t)sizeof(qs_u) },
        { "model.layers.0.mlp.experts.0.up_proj.weight",      "U8",  50, up_bogus, 50 },
    };
    if(mmap_aligned) write_fake_shard_aligned(path, ts, 6);
    else             write_fake_shard(path, ts, 6);
}
static void test_expert_load_rejects_unknown_size_main_path(const char *dir){
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    write_expert_shard_one_bogus(shard, /*mmap_aligned=*/0);

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_expert_load_bogus_size, dir, out, sizeof(out), &code, &sig);
    fprintf(stderr, "--- captured child stderr ---\n%s-----------------------------\n", out);
    CHECK("test3: bogus-size expert tensor (main path) -> child did not crash by signal", !sig);
    CHECK("test3: bogus-size expert tensor (main path) -> child exit code is 1", code==1);
    CHECK("test3: diagnostic names up_proj specifically", strstr(out,"up_proj")!=NULL);
    CHECK("test3: diagnostic states no known format matched", strstr(out,"matches NONE of the known")!=NULL);
}

/* ==================================================================== test 4 ===
 * expert_load()'s g_mmap=1 zero-copy fast path: the same bogus-sized expert must
 * ALSO fail closed (falls through to the main path, which rejects it there) --
 * NEVER commit a guessed format to a zero-copy view into file-backed memory.
 * g_mmap is set inside the FORKED CHILD only, so this never leaks into the
 * parent test process or any other test. */
static void child_expert_load_bogus_size_mmap(const char *dir){
    g_mmap = 1;
    Model m; memset(&m,0,sizeof(m));
    m.c.hidden=E_HID; m.c.moe_inter=E_MOE; m.ebits=2;
    st_init(&m.S, dir);
    ESlot slot; memset(&slot,0,sizeof(slot));
    expert_load(&m, 0, 0, &slot, 1);
}
static void test_expert_load_rejects_unknown_size_mmap_path(const char *dir){
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    write_expert_shard_one_bogus(shard, /*mmap_aligned=*/1);

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_expert_load_bogus_size_mmap, dir, out, sizeof(out), &code, &sig);
    fprintf(stderr, "--- captured child stderr (g_mmap=1) ---\n%s-----------------------------\n", out);
    CHECK("test4: bogus-size expert tensor (g_mmap=1) -> child did not crash by signal", !sig);
    CHECK("test4: bogus-size expert tensor (g_mmap=1) -> child exit code is 1 (never a zero-copy view onto a guessed format)", code==1);
    CHECK("test4: diagnostic names up_proj specifically", strstr(out,"up_proj")!=NULL);
    CHECK("test4: diagnostic states no known format matched", strstr(out,"matches NONE of the known")!=NULL);
}

int main(void){
    char root[] = "/tmp/test_legacy_fmt_guard.XXXXXX";
    if(!mkdtemp(root)){ perror("mkdtemp"); return 2; }
    char d1[600],d2a[600],d2b[600],d3[600],d4[600];
    snprintf(d1,sizeof(d1),"%s/bogus_dense",root);
    snprintf(d2a,sizeof(d2a),"%s/legacy_int8",root);
    snprintf(d2b,sizeof(d2b),"%s/legacy_int2",root);
    snprintf(d3,sizeof(d3),"%s/bogus_expert_main",root);
    snprintf(d4,sizeof(d4),"%s/bogus_expert_mmap",root);
    mkdir_or_die(d1); mkdir_or_die(d2a); mkdir_or_die(d2b); mkdir_or_die(d3); mkdir_or_die(d4);

    test_qt_from_disk_rejects_unknown_size(d1);
    test_qt_from_disk_legacy_int8_loads(d2a);
    test_qt_from_disk_legacy_int2_loads(d2b);
    test_expert_load_rejects_unknown_size_main_path(d3);
    test_expert_load_rejects_unknown_size_mmap_path(d4);

    char f[700];
    snprintf(f,sizeof(f),"%s/model.safetensors",d1); unlink(f);
    snprintf(f,sizeof(f),"%s/model.safetensors",d2a); unlink(f);
    snprintf(f,sizeof(f),"%s/model.safetensors",d2b); unlink(f);
    snprintf(f,sizeof(f),"%s/model.safetensors",d3); unlink(f);
    snprintf(f,sizeof(f),"%s/model.safetensors",d4); unlink(f);
    rmdir(d1); rmdir(d2a); rmdir(d2b); rmdir(d3); rmdir(d4); rmdir(root);

    if(failures){ fprintf(stderr,"\n%d FAILURE(S)\n",failures); return 1; }
    fprintf(stderr,"\nlegacy format guard: all checks pass\n");
    return 0;
}
