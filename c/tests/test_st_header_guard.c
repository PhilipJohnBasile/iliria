/* tests/test_st_header_guard.c -- unit tests for the safetensors header
 * validation added to st_init() (c/st.h): a Rust fuzzer (c/rust/artifact_validator,
 * 9M iterations) found that a malformed header had no backstop at parse time,
 * only surfacing (if ever) as an OOB read/write deep inside st_read_f32()'s
 * BF16/F16 element walk or F32 memcpy, and only for whichever tensor a given run
 * actually happens to touch.
 *
 * BUGS THIS REPLACES (none of these were checked before):
 *   1. `nbytes = b0 - a0` never checked b0>=a0 (or a0>=0) -- a reversed or
 *      negative data_offsets pair produced a negative/garbage nbytes.
 *   2. `numel` (shape product) was never cross-checked against `nbytes` for the
 *      declared dtype's element size -- st_read_f32's BF16/F16 arm walks `numel`
 *      elements over an `nbytes`-sized buffer (OOB READ if numel*elsize>nbytes);
 *      its F32 arm memcpy's `nbytes` bytes into a caller buffer sized for `numel`
 *      floats (OOB WRITE if nbytes>numel*4).
 *   3. A tensor's absolute byte range was never checked against the shard
 *      file's actual physical size -- pread()'s own return-value check would
 *      eventually catch a read that runs off the true end of file, but only
 *      for a tensor this specific run actually reads, and only at that later
 *      read call, not at parse time for every declared tensor up front.
 *
 * Fix (st.h): st_init() now validates, for every declared tensor, before ever
 * storing it: (a) 0<=a0<=b0, (b) nbytes == numel * dtype_size(dtype) exactly,
 * (c) the tensor's absolute end offset does not exceed the shard's real file
 * size -- fail-closed (fprintf naming the tensor + observed-vs-expected, then
 * exit(1)), mirroring st_dtype_code()'s existing exit(1)-on-unknown-dtype
 * discipline.
 *
 * Uses write_raw_shard() (full, independent control over declared shape/
 * data_offsets vs. physically-written bytes) to construct each malformed case
 * directly -- a well-behaved higher-level tensor-array writer can't produce an
 * inconsistent header by construction, which is exactly the point: these are
 * hand-crafted bad inputs, not accidents a normal writer would ever emit.
 *
 * Lean test (like tests/test_st.c): includes st.h directly, no glm.c/Model
 * dependency -- st_init/st_find/st_read_f32/st_read_raw are fully self-
 * contained.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/wait.h>
#include <errno.h>

#include "../st.h"

static int failures=0;
#define CHECK(desc,cond) do{ \
    if(!(cond)){ fprintf(stderr,"FAIL %s\n",desc); failures++; } \
    else fprintf(stderr,"ok   %s\n",desc); \
}while(0)

static void mkdir_or_die(const char *path){
    if(mkdir(path,0700)!=0 && errno!=EEXIST){ perror(path); exit(2); }
}

/* Writes a MINIMAL, single-tensor safetensors file with full, independent
 * control over the header's declared dtype/shape/data_offsets vs. how many
 * bytes are actually physically written after it -- needed to construct
 * deliberately inconsistent headers (reversed offsets, numel/nbytes mismatch,
 * a declared end past the true file size) that a well-formed higher-level
 * writer could never produce by construction. */
static void write_raw_shard(const char *path, const char *tensor_name, const char *dtype,
                             int64_t shape0, int64_t off_a, int64_t off_b,
                             const void *data, int64_t data_len){
    char hdr[2048];
    int hp = snprintf(hdr, sizeof(hdr),
        "{\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}}",
        tensor_name, dtype, (long long)shape0, (long long)off_a, (long long)off_b);
    uint64_t hlen = (uint64_t)hp;
    FILE *f = fopen(path, "wb");
    if(!f){ perror(path); exit(2); }
    if(fwrite(&hlen,8,1,f)!=1){ perror("fwrite hlen"); exit(2); }
    if(fwrite(hdr,1,(size_t)hp,f)!=(size_t)hp){ perror("fwrite hdr"); exit(2); }
    if(data_len>0 && fwrite(data,1,(size_t)data_len,f)!=(size_t)data_len){ perror("fwrite data"); exit(2); }
    fclose(f);
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
        fprintf(stderr,"UNEXPECTED: st_init did not abort -- fn returned normally\n");
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
static void child_st_init(const char *dir){
    shards S;
    st_init(&S, dir);
    (void)S;
}

/* ==================================================================== test 1 ===
 * reject reversed data_offsets (b0 < a0). */
static void test_reject_reversed_offsets(const char *dir){
    uint8_t data[100]; memset(data,0xAB,sizeof(data));
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    write_raw_shard(shard, "bad.reversed", "U8", /*shape0=*/100, /*off_a=*/100, /*off_b=*/50, data, sizeof(data));

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_st_init, dir, out, sizeof(out), &code, &sig);
    fprintf(stderr, "--- captured child stderr ---\n%s-----------------------------\n", out);
    CHECK("test1: reversed offsets -> child did not crash by signal", !sig);
    CHECK("test1: reversed offsets -> child exit code is 1", code==1);
    CHECK("test1: diagnostic names the tensor", strstr(out,"bad.reversed")!=NULL);
    CHECK("test1: diagnostic states invalid data_offsets", strstr(out,"invalid data_offsets")!=NULL);
}

/* ==================================================================== test 2 ===
 * reject negative a0 (companion to test 1 -- same check, other half). */
static void test_reject_negative_a0(const char *dir){
    uint8_t data[100]; memset(data,0xCD,sizeof(data));
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    write_raw_shard(shard, "bad.negative", "U8", /*shape0=*/10, /*off_a=*/-10, /*off_b=*/10, data, sizeof(data));

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_st_init, dir, out, sizeof(out), &code, &sig);
    fprintf(stderr, "--- captured child stderr ---\n%s-----------------------------\n", out);
    CHECK("test2: negative a0 -> child did not crash by signal", !sig);
    CHECK("test2: negative a0 -> child exit code is 1", code==1);
    CHECK("test2: diagnostic names the tensor", strstr(out,"bad.negative")!=NULL);
    CHECK("test2: diagnostic states invalid data_offsets", strstr(out,"invalid data_offsets")!=NULL);
}

/* ==================================================================== test 3 ===
 * reject numel/nbytes mismatch: F32 tensor declares shape=[10] (implying 40
 * bytes) but data_offsets span only 32 bytes -- internally consistent
 * start<=end, and well within the physical file, but numel*elemsize != nbytes.
 * This is exactly the gap that let st_read_f32's F32 memcpy(out,raw,nbytes)
 * OOB-write past a caller buffer sized for numel=10 floats. */
static void test_reject_numel_mismatch(const char *dir){
    uint8_t data[64]; memset(data,0xEF,sizeof(data));   /* file itself is plenty big */
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    write_raw_shard(shard, "bad.mismatch", "F32", /*shape0=*/10, /*off_a=*/0, /*off_b=*/32, data, sizeof(data));

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_st_init, dir, out, sizeof(out), &code, &sig);
    fprintf(stderr, "--- captured child stderr ---\n%s-----------------------------\n", out);
    CHECK("test3: numel/nbytes mismatch -> child did not crash by signal", !sig);
    CHECK("test3: numel/nbytes mismatch -> child exit code is 1", code==1);
    CHECK("test3: diagnostic names the tensor", strstr(out,"bad.mismatch")!=NULL);
    CHECK("test3: diagnostic states size mismatch", strstr(out,"size mismatch")!=NULL);
    CHECK("test3: diagnostic gives observed nbytes", strstr(out,"nbytes=32")!=NULL);
    CHECK("test3: diagnostic gives expected (numel x dtype_size)", strstr(out,"= 40")!=NULL);
}

/* ==================================================================== test 4 ===
 * reject a declared end past the shard's true physical file size: internally
 * consistent (nbytes==numel*elemsize) and non-reversed, but the file itself was
 * only ever written much shorter than the header claims. This is the "within-
 * file-but-wrong" residual the fuzzer flagged: a tensor whose declared range
 * would exceed the REAL file, caught here eagerly at parse time for every
 * tensor, not lazily only if/when some run happens to pread() it. */
static void test_reject_end_exceeds_file_size(const char *dir){
    uint8_t data[8]; memset(data,0x11,sizeof(data));   /* file only has 8 real data bytes ... */
    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    /* ... but the header claims a 64-byte U8 tensor (shape=[64], internally
     * self-consistent: nbytes=64-0=64=numel(64)*elemsize(1)) */
    write_raw_shard(shard, "bad.toolong", "U8", /*shape0=*/64, /*off_a=*/0, /*off_b=*/64, data, sizeof(data));

    char out[4096]; int code=0, sig=0;
    run_child_and_capture(child_st_init, dir, out, sizeof(out), &code, &sig);
    fprintf(stderr, "--- captured child stderr ---\n%s-----------------------------\n", out);
    CHECK("test4: end exceeds file size -> child did not crash by signal", !sig);
    CHECK("test4: end exceeds file size -> child exit code is 1", code==1);
    CHECK("test4: diagnostic names the tensor", strstr(out,"bad.toolong")!=NULL);
    CHECK("test4: diagnostic states exceeds file size", strstr(out,"exceeds file size")!=NULL);
}

/* ==================================================================== test 5 ===
 * a valid shard -- multiple tensors, spanning EVERY dtype st.h supports
 * (BF16, F16, F32, U8) -- must still load and read back byte/value-exact,
 * unchanged by the new validation. */
typedef struct { const char *name; const char *dtype; int64_t shape0; const void *data; int64_t nbytes; } FakeTensor;
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
static void test_valid_shard_still_loads(const char *dir){
    uint16_t bf16[4] = {0x3f80, 0xc020, 0x0000, 0x4000};  /* 1.0, -2.5, 0.0, 2.0 */
    uint16_t f16[3]  = {0x3c00, 0xc100, 0x0001};            /* 1.0, -2.5, smallest subnormal */
    float f32[5]     = {1.5f, -3.25f, 0.0f, 100.0f, -0.001f};
    uint8_t u8[6]     = {0,1,2,253,254,255};

    char shard[512]; snprintf(shard,sizeof(shard),"%s/model.safetensors",dir);
    FakeTensor ts[4] = {
        { "t.bf16", "BF16", 4, bf16, (int64_t)sizeof(bf16) },
        { "t.f16",  "F16",  3, f16,  (int64_t)sizeof(f16) },
        { "t.f32",  "F32",  5, f32,  (int64_t)sizeof(f32) },
        { "t.u8",   "U8",   6, u8,   (int64_t)sizeof(u8) },
    };
    write_fake_shard(shard, ts, 4);

    shards S;
    st_init(&S, dir);   /* must NOT abort -- this is a well-formed shard */

    CHECK("test5: st_find finds t.bf16", st_has(&S,"t.bf16"));
    CHECK("test5: st_find finds t.f16",  st_has(&S,"t.f16"));
    CHECK("test5: st_find finds t.f32",  st_has(&S,"t.f32"));
    CHECK("test5: st_find finds t.u8",   st_has(&S,"t.u8"));

    float out[5];
    int64_t n;
    n = st_read_f32(&S, "t.bf16", out, 0);
    CHECK("test5: bf16 numel==4", n==4);
    CHECK("test5: bf16 values decode correctly", out[0]==1.0f && out[1]==-2.5f && out[2]==0.0f && out[3]==2.0f);

    n = st_read_f32(&S, "t.f16", out, 0);
    CHECK("test5: f16 numel==3", n==3);
    CHECK("test5: f16 values decode correctly", out[0]==1.0f && out[1]==-2.5f && out[2]>0.0f);

    n = st_read_f32(&S, "t.f32", out, 0);
    CHECK("test5: f32 numel==5", n==5);
    CHECK("test5: f32 values byte-exact", memcmp(out, f32, sizeof(f32))==0);

    uint8_t u8out[6];
    st_read_raw(&S, "t.u8", u8out, 0);
    CHECK("test5: u8 bytes byte-exact", memcmp(u8out, u8, sizeof(u8))==0);

    CHECK("test5: st_nbytes(t.u8) matches what was written", st_nbytes(&S,"t.u8")==(int64_t)sizeof(u8));
}

int main(void){
    char root[] = "/tmp/test_st_header_guard.XXXXXX";
    if(!mkdtemp(root)){ perror("mkdtemp"); return 2; }
    char d1[600],d2[600],d3[600],d4[600],d5[600];
    snprintf(d1,sizeof(d1),"%s/reversed",root);
    snprintf(d2,sizeof(d2),"%s/negative",root);
    snprintf(d3,sizeof(d3),"%s/mismatch",root);
    snprintf(d4,sizeof(d4),"%s/toolong",root);
    snprintf(d5,sizeof(d5),"%s/valid",root);
    mkdir_or_die(d1); mkdir_or_die(d2); mkdir_or_die(d3); mkdir_or_die(d4); mkdir_or_die(d5);

    test_reject_reversed_offsets(d1);
    test_reject_negative_a0(d2);
    test_reject_numel_mismatch(d3);
    test_reject_end_exceeds_file_size(d4);
    test_valid_shard_still_loads(d5);

    char f[700];
    snprintf(f,sizeof(f),"%s/model.safetensors",d1); unlink(f);
    snprintf(f,sizeof(f),"%s/model.safetensors",d2); unlink(f);
    snprintf(f,sizeof(f),"%s/model.safetensors",d3); unlink(f);
    snprintf(f,sizeof(f),"%s/model.safetensors",d4); unlink(f);
    snprintf(f,sizeof(f),"%s/model.safetensors",d5); unlink(f);
    rmdir(d1); rmdir(d2); rmdir(d3); rmdir(d4); rmdir(d5); rmdir(root);

    if(failures){ fprintf(stderr,"\n%d FAILURE(S)\n",failures); return 1; }
    fprintf(stderr,"\nst_init header guard: all checks pass\n");
    return 0;
}
