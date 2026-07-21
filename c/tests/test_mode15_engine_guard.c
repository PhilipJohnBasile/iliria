/* Unit test for the mode-1.5 (Huffman-compressed) container fail-loud guard added to
 * glm.c's shard loaders (qt_from_disk, expert_load). Context: tools/mode15_container.py
 * defines an on-disk format for entropy-coded expert-weight tensors -- a 4-byte magic
 * "MH01" followed by a header, replacing an expert's raw quantized bytes. Before this
 * guard, qt_from_disk/expert_load inferred a tensor's quantization format PURELY from its
 * byte count (`fmt=(nb==O*I)?1:(nb==O*ceil(I/2))?2:3`); a mode-1.5 blob's compressed size
 * is data-dependent and would fall through into the fmt==3 (packed-int2) bucket, handing
 * its Huffman-coded bytes straight to matmul_i2 -- silent numeric garbage, no error. This
 * build has no mode-1.5 decoder wired in (c/mode15_reader.h + c/codec_row_huff.h are built
 * and tested standalone -- see tests/test_mode15_reader.c -- but not yet called from
 * glm.c, per the Mode-15 integration design notes step 3+), so the only correct
 * behavior on seeing "MH01" bytes today is to abort loudly, not guess.
 *
 * This test proves, against the REAL glm.c code (the `#define main ...; #include
 * "../glm.c"` trick tests/test_idot.c already established for this codebase -- not a
 * reimplementation):
 *   1. expert_load()'s primary (pread) hot path aborts -- exit code 1, diagnostic naming
 *      the tensor on stderr -- when it meets a "MH01"-tagged expert tensor.
 *   2. qt_from_disk() aborts the same way for a "MH01"-tagged dense tensor.
 *   3. expert_load() still loads a LEGACY (uncompressed, int4-packed) expert shard
 *      exactly as before, and the loaded weights are usable by the real matmul_i4 kernel
 *      -- the guard must not false-positive on ordinary containers.
 * expert_load/qt_from_disk call exit(1) directly on this path (this codebase's existing
 * fatal-load-error idiom -- see the "missing tensor"/OOM messages elsewhere in glm.c), so
 * tests 1+2 fork a child and capture its stderr through a pipe rather than calling them
 * in-process, which would tear down this test binary too. */
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

/* ============================================================ fixture construction === */
typedef struct { const char *name; const char *dtype; int64_t shape0; const void *data; int64_t nbytes; } FakeTensor;

/* Writes a minimal-but-valid safetensors shard: st_init()'s own expected format (8-byte
 * LE header length + JSON header + raw tensor bytes, header maps name -> {dtype, shape,
 * data_offsets}) containing exactly ts[0..n). Tensor bytes are written back-to-back in
 * array order, so callers control contiguity (real mode-1.5/quantized containers keep an
 * expert's 3 weight tensors contiguous -- expert_load's "one coalesced pread" fast path --
 * by listing them first, scales after). */
static void write_fake_shard(const char *path, const FakeTensor *ts, int n){
    char hdr[8192]; int hp=0;
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

/* ================================================================== small fixtures ==== */
#define FIX_O 8   /* moe_inter -- kept tiny: this test proves the GUARD, not model math */
#define FIX_D 8   /* hidden */

/* A "MH01"-tagged blob: real containers follow with a header + canonical-Huffman
 * payload (tools/mode15_container.py), but the guard fires on the 4-byte magic alone
 * (this build has no decoder to validate anything past it) -- so a minimal blob of
 * TENSOR_HEADER_LEN-ish size is a faithful, minimal trigger for THIS guard specifically.
 * Content past the magic is deliberately arbitrary. */
static const uint8_t MODE15_BLOB[24] = { 'M','H','0','1', 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 };
static const uint8_t LEGACY_INT4_ZERO[FIX_O*((FIX_D+1)/2)] = {0}; /* O*ceil(I/2) int4-packed bytes */
static float QS_ONES[FIX_O];

static void fill_qs_ones(void){ for(int i=0;i<FIX_O;i++) QS_ONES[i]=1.0f; }

/* model.layers.0.mlp.experts.0.{gate,up,down}_proj.weight(.qs) shard, with gate_proj's
 * raw weight tensor swapped for a mode-1.5 blob or left as a legacy int4 tensor depending
 * on `gate_is_mode15`. */
static void write_expert_shard(const char *path, int gate_is_mode15){
    FakeTensor ts[6] = {
        { "model.layers.0.mlp.experts.0.gate_proj.weight", "U8", gate_is_mode15?24:(int64_t)sizeof(LEGACY_INT4_ZERO),
          gate_is_mode15?(const void*)MODE15_BLOB:(const void*)LEGACY_INT4_ZERO,
          gate_is_mode15?24:(int64_t)sizeof(LEGACY_INT4_ZERO) },
        { "model.layers.0.mlp.experts.0.up_proj.weight", "U8", (int64_t)sizeof(LEGACY_INT4_ZERO),
          LEGACY_INT4_ZERO, (int64_t)sizeof(LEGACY_INT4_ZERO) },
        { "model.layers.0.mlp.experts.0.down_proj.weight", "U8", (int64_t)sizeof(LEGACY_INT4_ZERO),
          LEGACY_INT4_ZERO, (int64_t)sizeof(LEGACY_INT4_ZERO) },
        { "model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32", FIX_O, QS_ONES, (int64_t)sizeof(QS_ONES) },
        { "model.layers.0.mlp.experts.0.up_proj.weight.qs",   "F32", FIX_O, QS_ONES, (int64_t)sizeof(QS_ONES) },
        { "model.layers.0.mlp.experts.0.down_proj.weight.qs", "F32", FIX_O, QS_ONES, (int64_t)sizeof(QS_ONES) },
    };
    write_fake_shard(path, ts, 6);
}

static void write_dense_shard(const char *path){
    FakeTensor ts[2] = {
        { "test.dense.weight",     "U8",  24,    MODE15_BLOB, 24 },
        { "test.dense.weight.qs",  "F32", FIX_O, QS_ONES, (int64_t)sizeof(QS_ONES) },
    };
    write_fake_shard(path, ts, 2);
}

/* ===================================================================== fork+capture ==== */
typedef void (*child_fn)(const char *dir);

/* Runs `fn(dir)` in a forked child with stderr redirected through a pipe. `fn` is expected
 * to call exit(1) via the guard (glm.c's existing fatal-load-error idiom) -- if it instead
 * RETURNS, the child exits 42 so that outcome is distinguishable from "guard fired". */
static void run_child_and_capture(child_fn fn, const char *dir, char *out, size_t outcap, int *exit_code, int *by_signal){
    int p[2];
    if(pipe(p)!=0){ perror("pipe"); exit(2); }
    fflush(NULL);
    pid_t pid=fork();
    if(pid<0){ perror("fork"); exit(2); }
    if(pid==0){
        close(p[0]);
        dup2(p[1], STDERR_FILENO);
        close(p[1]);
        fn(dir);
        fprintf(stderr, "UNEXPECTED: guard did not abort -- fn returned normally\n");
        fflush(stderr);
        _exit(42);
    }
    close(p[1]);
    size_t total=0; ssize_t n;
    while(total+1<outcap && (n=read(p[0], out+total, outcap-1-total))>0) total+=(size_t)n;
    out[total]=0;
    close(p[0]);
    int status=0;
    waitpid(pid,&status,0);
    *by_signal = WIFSIGNALED(status);
    *exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static void child_expert_load_mode15(const char *dir){
    Model m; memset(&m,0,sizeof(m));
    m.c.hidden=FIX_D; m.c.moe_inter=FIX_O; m.ebits=2;
    st_init(&m.S, dir);
    ESlot slot; memset(&slot,0,sizeof(slot));
    int rc=expert_load(&m, /*layer=*/0, /*eid=*/0, &slot, /*fatal=*/1);
    (void)rc; /* only reached if the guard failed to abort -- caller reports that */
}

static void child_qt_from_disk_mode15(const char *dir){
    Model m; memset(&m,0,sizeof(m));
    st_init(&m.S, dir);
    QT t; memset(&t,0,sizeof(t));
    qt_from_disk(&m, "test.dense.weight", FIX_O, FIX_D, /*bits=*/2, /*drop=*/0, &t);
}

/* ========================================================================= tests ====== */

static void test_expert_load_aborts_on_mode15(const char *dir){
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    write_expert_shard(shard, /*gate_is_mode15=*/1);

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_expert_load_mode15, dir, out, sizeof(out), &code, &sig);

    fprintf(stderr, "--- captured child stderr ---\n%s-----------------------------\n", out);
    CHECK("expert_load: mode-1.5 tensor -> child did not crash by signal", !sig);
    CHECK("expert_load: mode-1.5 tensor -> child exit code is 1 (exit(1), not 42/other)", code==1);
    CHECK("expert_load: diagnostic names the tensor", strstr(out,"gate_proj")!=NULL);
    CHECK("expert_load: diagnostic identifies mode-1.5 format", strstr(out,"mode-1.5")!=NULL);
    CHECK("expert_load: diagnostic names the magic", strstr(out,"MH01")!=NULL);
    CHECK("expert_load: diagnostic names the required decoder", strstr(out,"mode15_reader.h")!=NULL);
    CHECK("expert_load: diagnostic calls out the at-risk kernel", strstr(out,"matmul_i2")!=NULL);
}

static void test_qt_from_disk_aborts_on_mode15(const char *dir){
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    write_dense_shard(shard);

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_qt_from_disk_mode15, dir, out, sizeof(out), &code, &sig);

    fprintf(stderr, "--- captured child stderr ---\n%s-----------------------------\n", out);
    CHECK("qt_from_disk: mode-1.5 tensor -> child did not crash by signal", !sig);
    CHECK("qt_from_disk: mode-1.5 tensor -> child exit code is 1", code==1);
    CHECK("qt_from_disk: diagnostic names the tensor", strstr(out,"test.dense.weight")!=NULL);
    CHECK("qt_from_disk: diagnostic identifies mode-1.5 format", strstr(out,"mode-1.5")!=NULL);
}

static void test_legacy_expert_load_still_works(const char *dir){
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    write_expert_shard(shard, /*gate_is_mode15=*/0);

    Model m; memset(&m,0,sizeof(m));
    m.c.hidden=FIX_D; m.c.moe_inter=FIX_O; m.ebits=2;
    st_init(&m.S, dir);
    ESlot slot; memset(&slot,0,sizeof(slot));
    int rc=expert_load(&m, /*layer=*/0, /*eid=*/0, &slot, /*fatal=*/1);

    CHECK("legacy shard: expert_load returns 0 (no abort)", rc==0);
    CHECK("legacy shard: slot.eid updated to the loaded expert", slot.eid==0);
    CHECK("legacy shard: gate_proj inferred as fmt==2 (int4)", slot.g.fmt==2);
    CHECK("legacy shard: up_proj inferred as fmt==2 (int4)",   slot.u.fmt==2);
    CHECK("legacy shard: down_proj inferred as fmt==2 (int4)", slot.d.fmt==2);
    CHECK("legacy shard: gate_proj shape O==moe_inter", slot.g.O==FIX_O);
    CHECK("legacy shard: gate_proj shape I==hidden",    slot.g.I==FIX_D);

    /* Prove the loaded weights are actually USABLE by the real kernel matmul_i2 would have
     * silently corrupted for a mode-1.5 tensor -- not just that expert_load returned 0. */
    float x[FIX_D]; for(int i=0;i<FIX_D;i++) x[i]=1.0f;
    float y[FIX_O];
    matmul_i4(y, x, slot.g.q4, slot.g.s, /*S=*/1, /*I=*/slot.g.I, /*O=*/slot.g.O);
    int all_finite=1; for(int i=0;i<FIX_O;i++) if(!isfinite(y[i])) all_finite=0;
    CHECK("legacy shard: matmul_i4 on the loaded weights produces finite output", all_finite);
}

int main(void){
    fill_qs_ones();

    char root[] = "/tmp/test_mode15_engine_guard.XXXXXX";
    if(!mkdtemp(root)){ perror("mkdtemp"); return 2; }

    char d1[600], d2[600], d3[600];
    snprintf(d1,sizeof(d1),"%s/mode15_expert",root);
    snprintf(d2,sizeof(d2),"%s/legacy_expert",root);
    snprintf(d3,sizeof(d3),"%s/mode15_dense",root);
    mkdir_or_die(d1); mkdir_or_die(d2); mkdir_or_die(d3);

    test_expert_load_aborts_on_mode15(d1);
    test_qt_from_disk_aborts_on_mode15(d3);
    test_legacy_expert_load_still_works(d2);

    /* best-effort cleanup -- not the point of the test, but leave /tmp tidy */
    char f1[700],f2[700],f3[700];
    snprintf(f1,sizeof(f1),"%s/model.safetensors",d1); unlink(f1);
    snprintf(f2,sizeof(f2),"%s/model.safetensors",d2); unlink(f2);
    snprintf(f3,sizeof(f3),"%s/model.safetensors",d3); unlink(f3);
    rmdir(d1); rmdir(d2); rmdir(d3); rmdir(root);

    if(failures){ fprintf(stderr,"\n%d FAILURE(S)\n",failures); return 1; }
    fprintf(stderr,"\nmode-1.5 engine guard: all checks pass\n");
    return 0;
}
