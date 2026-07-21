/* iobench_ab.c -- A/B microbenchmark harness for the two GPU-free NVMe I/O
 * levers surfaced by a triage of glm.c but never validated on the M5 Max:
 *
 *   (a) DIRECT=1 (F_NOCACHE/O_DIRECT) vs buffered read bandwidth+latency,
 *       at glm.c's g_direct fast path (glm.c ~677-679, ~1286-1298): a SINGLE
 *       pread of the expert's contiguous gate+up+down weight block, 4K-base/
 *       len-aligned, on a twin fd opened O_RDONLY|O_DIRECT (Linux) or
 *       O_RDONLY+F_NOCACHE (macOS, via compat_open_direct) -- exactly
 *       mirroring st.h's st_open_fd/st_direct_fd twin-fd convention.
 *       Default block size 18,915,328 bytes (18.9153 MB) is the real,
 *       measured, constant int4 per-expert byte count for GLM-5.2
 *       (gate_proj+up_proj+down_proj combined; K=8 routing width, 256
 *       experts) -- see docs/PERFORMANCE_THEORY.md and
 *       the depth-prefetch trace verdict section 1. It is
 *       already an exact multiple of 4096 (4618 * 4096).
 *       NOTE ON SCOPE: the three small ".qs" quantization-scale reads
 *       (glm.c:1307-1310) are ALWAYS buffered regardless of g_direct --
 *       they do not participate in the DIRECT-vs-buffered branch at all,
 *       so they are intentionally out of scope here (measuring them would
 *       not inform the g_direct decision either way).
 *
 *   (b) single- vs concurrent-dual-stream aggregate NVMe bandwidth: resolves
 *       the bandwidth half of the pre-registered depth-prefetch REOPEN GATE
 *       (the depth-prefetch trace verdict, "REOPEN GATE" (b)):
 *       whether prefetch traffic ADDED on top of demand traffic behaves like
 *       the trace's own optimistic ~44 GB/s implied concurrent bandwidth, or
 *       is capped near the pessimistic 13.5 GB/s device-throughput
 *       reference. Production's own demand-read concurrency is exactly
 *       K=8 threads reading missed experts in one `omp parallel for`
 *       (glm.c's expert_load callers) -- i.e. today's "single stream" IS
 *       already 8-way concurrent. A hypothetical depth-1 prefetch stream
 *       running alongside it would make that "16 concurrent reads sharing
 *       one pipe" -- TRACE-VERDICT.md's own phrasing for the exact unmeasured
 *       quantity. --threads (default 8) sets the per-stream width so
 *       --streams 1 vs --streams 2 directly measures that transition.
 *       This is a raw hardware-concurrency probe, NOT a full demand-priority
 *       production A/B (that fuller test -- reopen gate part (b) in full --
 *       needs real routing/prediction and is out of scope for this cheap
 *       extension; this tool answers the narrower, cheaper question first).
 *
 * Design:
 *   - Randomized A/B ordering: the {direct, streams} condition matrix x reps
 *     is flattened into a trial list and Fisher-Yates shuffled (seeded xorshift64*,
 *     --seed or time-based) before execution, so thermal/background drift
 *     cannot bias one condition over another.
 *   - Fresh random (4K-aligned-at-read-time) offsets are drawn for EVERY
 *     trial from the whole file, never reused across trials -- reusing the
 *     same bytes across conditions would let an earlier buffered pass warm
 *     the page cache for a later "direct" pass (F_NOCACHE/O_DIRECT bypass
 *     caching on their OWN reads but do not evict pages already resident --
 *     see docs/history.md's O_DIRECT note -- so a stale reused offset set
 *     would silently bias the comparison).
 *   - Reports median + bootstrap 95% CI (percentile method, dependency-free)
 *     per condition, for both aggregate GB/s and per-read latency.
 *
 * NVMe-quiet guard: refuses the real (large-file) run if a process matching
 * --guard-pattern (default "gate_m15_g1") is detected via `pgrep -fl`, fail-
 * CLOSED on any check failure -- mirrors the fail-closed doctrine of
 * c/scripts/quiesce_check.sh and the generic foreign-process detection added
 * to c/tools/timing_watchdog.py (commit 0052b7b), scoped here to the one
 * process a human operator identified as currently streaming the model
 * container from NVMe. The guard runs BEFORE the target file is even opened,
 * so it also gates a real-mode invocation against a nonexistent path (used
 * by --selftest as a live positive control).
 *
 * Usage:
 *   ./iobench_ab --selftest
 *       Safe at any time, any NVMe load: logic + tiny mock-read self-check
 *       against a private scratch file. Does not touch the real model
 *       container and does not require the NVMe-quiet guard to pass.
 *
 *   ./iobench_ab <real_expert_shard_file> [options]
 *       Runs the real A/B benchmark. Refuses (nonzero exit) if the NVMe-quiet
 *       guard trips. Options:
 *         --blk BYTES            main block size (default 18915328)
 *         --threads N             threads per stream (default 8)
 *         --reads N               reads per thread per trial (default 8)
 *         --reps N                repetitions per condition (default 11)
 *         --seed N                RNG seed (default: time-based, printed)
 *         --guard-pattern STR     pgrep -f pattern (default "gate_m15_g1")
 *
 * Build: gcc/clang -O2 -fopenmp iobench_ab.c -o iobench_ab   (or `make iobench_ab`)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <dirent.h>
#include "compat.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ============================ RNG (xorshift64*) ============================
 * One seeded stream drives EVERYTHING (offset draws, trial shuffle, bootstrap
 * resampling) in strict execution order: a given --seed reproduces an entire
 * session byte-for-byte, and we sidestep platform rand()/RAND_MAX limits
 * iobench.c's own comment flags (RAND_MAX=32767 on Windows). */
typedef uint64_t rng_t;
static uint64_t xorshift64star(rng_t *state){
    uint64_t x=*state;
    x^=x>>12; x^=x<<25; x^=x>>27;
    *state=x;
    return x*0x2545F4914F6CDD1DULL;
}
static rng_t rng_seed(uint64_t seed){ return seed?seed:88172645463325252ULL; }

static void shuffle_int(int *arr,int n,rng_t *st){
    for(int i=n-1;i>0;i--){
        int j=(int)(xorshift64star(st)%(uint64_t)(i+1));
        int t=arr[i]; arr[i]=arr[j]; arr[j]=t;
    }
}

/* A single planned read: which file (index into a FileSet, below) and at what
 * byte offset within it. Used so run_trial's read loop is agnostic to
 * whether its reads come from one small file (existing single-file mode) or
 * are spread across an entire multi-hundred-GB directory of shards
 * (cold-sweep mode, added to defeat page-cache contamination -- see
 * build_cold_pool below). */
typedef struct { int file_idx; off_t off; } BlockRef;

static void shuffle_blockref(BlockRef *arr,int64_t n,rng_t *st){
    for(int64_t i=n-1;i>0;i--){
        int64_t j=(int64_t)(xorshift64star(st)%(uint64_t)(i+1));
        BlockRef t=arr[i]; arr[i]=arr[j]; arr[j]=t;
    }
}

/* ============================ stats ============================ */
static int cmp_double(const void *a,const void *b){
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}
static double median_of(double *arr,int n){        /* sorts arr in place */
    if(n<=0) return NAN;
    qsort(arr,(size_t)n,sizeof(double),cmp_double);
    return (n%2)? arr[n/2] : 0.5*(arr[n/2-1]+arr[n/2]);
}
/* Percentile bootstrap 95% CI on the median, B resamples. Dependency-free,
 * honest about small-n: n<2 collapses to a degenerate point CI. */
static void bootstrap_median_ci(const double *data,int n,rng_t *st,int B,double *lo,double *hi){
    if(n<2){ *lo=*hi=(n==1?data[0]:NAN); return; }
    double *resample=malloc(sizeof(double)*(size_t)n);
    double *meds=malloc(sizeof(double)*(size_t)B);
    for(int b=0;b<B;b++){
        for(int i=0;i<n;i++) resample[i]=data[xorshift64star(st)%(uint64_t)n];
        meds[b]=median_of(resample,n);
    }
    qsort(meds,(size_t)B,sizeof(double),cmp_double);
    int lo_idx=(int)(0.025*B), hi_idx=(int)(0.975*B);
    if(hi_idx>=B) hi_idx=B-1;
    if(lo_idx<0) lo_idx=0;
    *lo=meds[lo_idx]; *hi=meds[hi_idx];
    free(resample); free(meds);
}

/* ============================ direct-fd open ============================
 * Exactly mirrors st.h's st_open_fd twin-fd convention (st.h:82-88). */
static int open_direct_twin(const char *path){
#ifdef O_DIRECT
    return open(path,COMPAT_O_RDONLY|O_DIRECT);
#elif defined(__APPLE__)
    return compat_open_direct(path);
#else
    (void)path; return -1;
#endif
}

/* ============================ page-cache residency probe ============================
 * Unprivileged (no root): mmap a SMALL sampled window of the candidate block
 * on its BUFFERED fd and call mincore(). macOS's unified buffer cache backs
 * file-backed mmap with the SAME pages read()/pread() populate for that
 * vnode, so this reflects the file's real page-cache state -- including
 * residency left over from something else entirely (e.g. gate_m15_g1's own
 * multi-hour read pass over this exact directory just before this tool
 * runs), not just pages this process touched. Used to SKIP candidates that
 * would silently turn a "cold NVMe" trial into a cache-hit trial regardless
 * of the direct/buffered arm (F_NOCACHE/O_DIRECT bypass caching on THEIR OWN
 * read but cannot evict pages already resident -- docs/history.md's
 * O_DIRECT note). Only samples 3 pages (first/middle/last) per block to
 * keep the check cheap relative to an 18.9 MB real read; this is a
 * transparency/credibility aid, not a safety gate, so it fails OPEN
 * (treated as "assume cold") if the probe itself can't run -- unlike
 * nvme_quiet_guard, which fails closed. Returns 1=looks resident,
 * 0=looks cold, -1=probe failed (caller treats as cold). */
static int block_looks_resident(int fd_buf,off_t off,long blk){
    long pagesize=sysconf(_SC_PAGESIZE);
    if(pagesize<=0) return -1;
    off_t mapoff=off & ~(pagesize-1);
    size_t extra=(size_t)(off-mapoff);
    size_t maplen=extra+(size_t)blk;
    void *m=mmap(NULL,maplen,PROT_READ,MAP_PRIVATE,fd_buf,mapoff);
    if(m==MAP_FAILED) return -1;
    size_t npages=(maplen+(size_t)pagesize-1)/(size_t)pagesize;
    char *vec=malloc(npages);
    int resident=-1;
    if(vec){
#ifdef __APPLE__
        int rc=mincore(m,maplen,vec);
#else
        int rc=mincore(m,maplen,(unsigned char*)vec);
#endif
        if(rc==0){
            size_t idxs[3]={0,npages/2,npages-1};
            resident=0;
            for(int k=0;k<3;k++){ size_t pi=idxs[k]; if(pi<npages && (vec[pi]&1)){ resident=1; break; } }
        }
        free(vec);
    }
    munmap(m,maplen);
    return resident;
}

/* ============================ NVMe-quiet guard ============================
 * Runs `pgrep -fl <pattern>` via fork/exec (no shell -- no injection surface
 * even though the pattern is trusted local-operator input). Returns:
 *   1 = match found (contents written to out)   0 = no match   -1 = check itself failed */
static int run_pgrep(const char *pattern,char *out,size_t out_sz){
    int pfd[2];
    if(pipe(pfd)!=0) return -1;
    pid_t pid=fork();
    if(pid<0){ close(pfd[0]); close(pfd[1]); return -1; }
    if(pid==0){
        close(pfd[0]);
        dup2(pfd[1],1); dup2(pfd[1],2); close(pfd[1]);
        execlp("pgrep","pgrep","-fl",pattern,(char*)NULL);
        _exit(127);
    }
    close(pfd[1]);
    size_t got=0; ssize_t r;
    if(out&&out_sz) out[0]=0;
    while(out && got+1<out_sz && (r=read(pfd[0],out+got,out_sz-1-got))>0) got+=(size_t)r;
    if(out) out[got]=0;
    char junk[256]; while((r=read(pfd[0],junk,sizeof(junk)))>0){}
    close(pfd[0]);
    int status=0;
    if(waitpid(pid,&status,0)<0) return -1;
    if(!WIFEXITED(status)) return -1;
    int code=WEXITSTATUS(status);
    if(code==0) return 1;
    if(code==1) return 0;
    return -1;                                     /* pgrep usage error / exec failed (127) */
}
/* Fail-CLOSED, mirroring c/scripts/quiesce_check.sh's doctrine ("missing or
 * unparseable telemetry is a failure, never a pass") and the generic
 * foreign-process net added to c/tools/timing_watchdog.py (commit 0052b7b) --
 * scoped here to the one known heavy-NVMe-reader the caller identified.
 * Returns 1 (clear, safe to proceed) or 0 (refused / could not verify). */
static int nvme_quiet_guard(const char *pattern){
    char buf[4096];
    int rc=run_pgrep(pattern,buf,sizeof buf);
    if(rc<0){
        fprintf(stderr,
          "[nvme-guard] FAIL-CLOSED: could not verify NVMe quiescence (the pgrep\n"
          "[nvme-guard] check itself failed to run). Refusing the real benchmark\n"
          "[nvme-guard] out of caution. Check manually with:\n"
          "[nvme-guard]     pgrep -fl %s\n",pattern);
        return 0;
    }
    if(rc==1){
        fprintf(stderr,
          "[nvme-guard] REFUSED: a process matching '%s' is running RIGHT NOW --\n"
          "[nvme-guard] it is reading the model container from NVMe. Bandwidth\n"
          "[nvme-guard] numbers measured concurrently with it are MEANINGLESS.\n"
          "[nvme-guard] Matched:\n%s"
          "[nvme-guard] Check manually with:\n"
          "[nvme-guard]     pgrep -fl %s\n"
          "[nvme-guard] Re-run this benchmark once that command prints nothing.\n",
          pattern,buf,pattern);
        return 0;
    }
    fprintf(stderr,"[nvme-guard] clear -- no process matching '%s' (checked: pgrep -fl %s).\n",pattern,pattern);
    return 1;
}

/* ============================ trial engine ============================ */
#define MAX_STREAMS 2

typedef struct { int direct; int streams; const char *label; } Cond;

typedef struct {
    double gbps;                    /* aggregate useful-byte GB/s, all streams */
    double per_stream_gbps[MAX_STREAMS];
    int64_t bytes;                  /* useful bytes (production accounting: unaligned tensor size, glm.c:1192) */
    double dt;
    double *lat_ms;                 /* caller-owned buffer, length >= streams*threads*reads_per_thread */
    int n_lat;                      /* entries actually written (== streams*threads*reads_per_thread) */
    int n_fail;                     /* short/failed reads (marked -1 in lat_ms) */
} TrialResult;

/* Runs one trial: `streams` concurrent groups of `threads` OpenMP workers
 * each, `reads` pread's per worker, of `blk` USEFUL bytes at the offset
 * `refs[gtid*reads+k]` names (file index + byte offset -- see BlockRef).
 * Offsets are ALWAYS pre-built by the caller (gen_single_file_refs for a
 * lone small file, build_cold_pool for a cache-defeating cross-shard sweep)
 * so this function has no opinion on where its reads come from. direct=1
 * reads via fds_dir[file_idx] with glm.c's exact 4K base/len alignment math
 * (glm.c:1287-1290); direct=0 reads via fds_buf[file_idx] at the raw
 * (unaligned) offset, matching the buffered fallback path (glm.c:1296-1298).
 * Bandwidth is credited on `blk` (useful) bytes in both arms, matching
 * production's own io_bytes_requested accounting (glm.c:1192) -- the aligned
 * arm's few extra padding bytes actually transferred are not double-counted
 * as throughput, so neither arm is penalized/favored by the alignment tax. */
static int run_trial(const BlockRef *refs,const int *fds_buf,const int *fds_dir,
                      int direct,int streams,int threads,int reads,long blk,TrialResult *out){
    int total_threads=streams*threads;
    int64_t stream_bytes[MAX_STREAMS]; for(int s=0;s<streams;s++) stream_bytes[s]=0;
    int fail_count=0;
    double t0=now();
    #pragma omp parallel num_threads(total_threads) reduction(+:fail_count)
    {
        int gtid=0;
#ifdef _OPENMP
        gtid=omp_get_thread_num();
#endif
        int stream_id = (threads>0)? (gtid/threads) : 0;
        if(stream_id>=streams) stream_id=streams-1; /* defensive: no-OpenMP single-thread fallback */
        /* +8192 (not +4096): when blk itself is NOT 4096-aligned (a custom
         * --blk), both the (o-base) remainder and the round-up padding can
         * independently approach 4095 bytes, so the aligned `len` computed
         * below can reach blk+8190 in the worst case. +8192 matches glm.c's
         * own slack for the identical reason (s->slab_cap=wtot+8192,
         * glm.c:1244) and is exact/sufficient; the default blk (18915328,
         * already a multiple of 4096) never needs more than +4096 of it. */
        size_t cap=(size_t)blk+8192;
        void *buf=NULL;
        if(posix_memalign(&buf,4096,cap)!=0) buf=NULL;
        int64_t local_bytes=0;
        for(int k=0;k<reads;k++){
            int idx=gtid*reads+k;
            int file_idx=refs[idx].file_idx;
            int fd = direct? fds_dir[file_idx] : fds_buf[file_idx];
            double s0=now();
            ssize_t r;
            int ok;
            if(!buf || fd<0){ r=-1; ok=0; }
            else if(direct){
                off_t o=refs[idx].off;
                off_t base=o & ~4095LL;
                int64_t need=(o-base)+blk;
                int64_t len=(need+4095)&~4095LL;
                r=pread(fd,buf,(size_t)len,base);
                ok=(r>=need);
            } else {
                r=pread(fd,buf,(size_t)blk,refs[idx].off);
                ok=(r==blk);
            }
            double s1=now();
            if(out->lat_ms) out->lat_ms[idx]=(s1-s0)*1000.0;
            if(ok) local_bytes+=blk; else fail_count++;
        }
        if(buf) compat_aligned_free(buf);
        #pragma omp atomic
        stream_bytes[stream_id]+=local_bytes;
    }
    double dt=now()-t0;
    int64_t total_bytes=0; for(int s=0;s<streams;s++) total_bytes+=stream_bytes[s];
    out->dt=dt; out->bytes=total_bytes; out->gbps=(double)total_bytes/1e9/dt;
    out->n_lat=total_threads*reads; out->n_fail=fail_count;
    for(int s=0;s<streams;s++) out->per_stream_gbps[s]=(double)stream_bytes[s]/1e9/dt;
    return 0;
}

/* Single-file offset generation (the ORIGINAL v1 behavior): fresh random
 * offset per read, drawn with replacement from one file's whole range. Fine
 * for a file much larger than any one trial's footprint, or for the
 * self-test's tiny mock file where cache-cleanliness isn't the point --
 * NOT sufficient by itself to defeat page cache at real model-container
 * scale (a single ~2-5 GB shard fits trivially in 128+ GB RAM after a
 * handful of trials). See build_cold_pool for the cache-defeating version. */
static int gen_single_file_refs(BlockRef *refs,int n,off_t filesize,long blk,rng_t *rngst){
    if(filesize<=blk+4096) return -2;
    off_t range=filesize-blk-4096;
    if(range<1) range=1;
    for(int i=0;i<n;i++){ refs[i].file_idx=0; refs[i].off=(off_t)(xorshift64star(rngst)%(uint64_t)range); }
    return 0;
}

/* ============================ selftest ============================ */
static int selftest_fail(const char *msg){ fprintf(stderr,"[selftest] FAIL: %s\n",msg); return 1; }

static int run_selftest(const char *argv0){
    int failures=0;
    printf("[selftest] iobench_ab logic + small-mock-read self-check (no real model file touched)\n");

    /* 1. RNG determinism: same seed -> identical stream. */
    { rng_t a=rng_seed(42), b=rng_seed(42); int ok=1;
      for(int i=0;i<1000;i++) if(xorshift64star(&a)!=xorshift64star(&b)){ ok=0; break; }
      if(!ok) failures+=selftest_fail("xorshift64star not reproducible for same seed");
      else printf("[selftest] PASS: RNG reproducible under fixed seed (1000 draws)\n"); }

    /* 2. shuffle_int is a permutation (same multiset in/out), and reproducible under fixed seed. */
    { int a[9]={0,0,0,1,1,1,2,2,2}, b[9]; memcpy(b,a,sizeof a);
      rng_t sa=rng_seed(7), sb=rng_seed(7);
      shuffle_int(a,9,&sa); shuffle_int(b,9,&sb);
      int ok=1; for(int i=0;i<9;i++) if(a[i]!=b[i]){ ok=0; break; }
      if(!ok) failures+=selftest_fail("shuffle_int not reproducible for same seed");
      int cnt[3]={0,0,0}; for(int i=0;i<9;i++) cnt[a[i]]++;
      if(cnt[0]!=3||cnt[1]!=3||cnt[2]!=3) failures+=selftest_fail("shuffle_int changed the multiset (not a permutation)");
      if(ok && cnt[0]==3 && cnt[1]==3 && cnt[2]==3) printf("[selftest] PASS: shuffle_int is a reproducible permutation\n"); }

    /* 3. median_of on known arrays. */
    { double odd[5]={5,1,4,2,3}; double m1=median_of(odd,5);
      double even[4]={4,1,3,2}; double m2=median_of(even,4);
      if(fabs(m1-3.0)>1e-12) failures+=selftest_fail("median_of wrong for odd-length array (expected 3)");
      if(fabs(m2-2.5)>1e-12) failures+=selftest_fail("median_of wrong for even-length array (expected 2.5)");
      if(fabs(m1-3.0)<=1e-12 && fabs(m2-2.5)<=1e-12) printf("[selftest] PASS: median_of correct on known arrays (3.0, 2.5)\n"); }

    /* 4. bootstrap CI: constant data collapses to a point; CI must bracket the true median for varied data. */
    { double c[6]={7,7,7,7,7,7}; rng_t st=rng_seed(1); double lo,hi;
      bootstrap_median_ci(c,6,&st,500,&lo,&hi);
      if(fabs(lo-7.0)>1e-9 || fabs(hi-7.0)>1e-9) failures+=selftest_fail("bootstrap CI on constant data did not collapse to the constant");
      double v[7]={1,2,3,4,5,6,7}; rng_t st2=rng_seed(2); double lo2,hi2;
      bootstrap_median_ci(v,7,&st2,2000,&lo2,&hi2);
      if(!(lo2<=4.0 && hi2>=4.0)) failures+=selftest_fail("bootstrap CI did not bracket the true median of a simple known array");
      else printf("[selftest] PASS: bootstrap_median_ci sane (constant-collapse + brackets true median 4.0, got [%.3f,%.3f])\n",lo2,hi2); }

    /* 5. NVMe guard, LIVE positive control: gate_m15_g1 was confirmed running via
     * `ps -p 54302` immediately before this tool was built (PID 54302, reading
     * the GLM-5.2 model container). If it is still running, the guard MUST
     * refuse; if it has since finished, we only check the "clear" arm instead
     * and say so -- either outcome is a legitimate self-test result, never a
     * silent skip. */
    { char buf[4096]; int rc=run_pgrep("gate_m15_g1",buf,sizeof buf);
      if(rc<0) failures+=selftest_fail("pgrep check machinery itself failed (fork/exec/pgrep problem)");
      else if(rc==1) printf("[selftest] PASS: guard correctly detects the LIVE gate_m15_g1 process (real positive control):\n%s",buf);
      else printf("[selftest] INFO: gate_m15_g1 is no longer running (finished since this worktree started) -- positive control unavailable this run; guard's negative arm is checked next.\n"); }
    { char buf[256]; int rc=run_pgrep("definitely_not_a_real_process_zzz_9f8e7d",buf,sizeof buf);
      if(rc<0) failures+=selftest_fail("pgrep check machinery failed on the no-match control");
      else if(rc==1) failures+=selftest_fail("guard false-positived on a pattern that should never match");
      else printf("[selftest] PASS: guard correctly reports clear for a pattern that matches nothing\n"); }

    /* 6. CLI-level integration: invoke the ACTUAL compiled binary in "real"
     * mode against a nonexistent path. The guard must fire and refuse BEFORE
     * the (nonexistent) file is ever opened -- proving open() failure is not
     * what's producing the nonzero exit. Only meaningful while gate_m15_g1 is
     * actually running (checked above); skipped-but-reported otherwise. */
    { char buf[4096]; int rc=run_pgrep("gate_m15_g1",buf,sizeof buf);
      if(rc==1){
          const char *scratch_dir=getenv("TMPDIR"); if(!scratch_dir || !*scratch_dir) scratch_dir="/tmp";
          char outpath[1024]; snprintf(outpath,sizeof outpath,"%s/.iobench_ab_selftest_out",scratch_dir);
          char cmd[1200];
          snprintf(cmd,sizeof cmd,"%s /nonexistent/path/should/not/be/opened --guard-pattern gate_m15_g1 >%s 2>&1",argv0,outpath);
          int sys_rc=system(cmd);
          int exited_nonzero = WIFEXITED(sys_rc) && WEXITSTATUS(sys_rc)!=0;
          FILE *f=fopen(outpath,"r"); int saw_refused=0;
          if(f){ char line[512]; while(fgets(line,sizeof line,f)) if(strstr(line,"REFUSED")){ saw_refused=1; break; } fclose(f); remove(outpath); }
          if(!exited_nonzero || !saw_refused) failures+=selftest_fail("CLI real-mode invocation did not refuse-before-open against a live gate_m15_g1 + nonexistent path");
          else printf("[selftest] PASS: CLI real-mode invocation refuses (guard-before-open) against a nonexistent path while gate_m15_g1 is live\n");
      } else printf("[selftest] INFO: gate_m15_g1 not running -- skipping the guard-before-open CLI integration check (would need a live blocker to be meaningful)\n"); }

    /* 7. End-to-end tiny mock read: private scratch file, small blocks, single-
     * AND dual-stream, buffered AND direct arms -- exercises the full
     * offset-draw -> pread -> latency -> aggregation pipeline for real,
     * at safe/tiny scale. Written under a temp scratch dir, never the repo. */
    {
        const char *scratch_dir=getenv("TMPDIR"); if(!scratch_dir || !*scratch_dir) scratch_dir="/tmp";
        char path[1024]; snprintf(path,sizeof path,"%s/iobench_ab_selftest_mock.bin",scratch_dir);
        long mock_blk=262144;                         /* 256 KiB mock "expert" block -- NOT the real 18.9 MB size */
        off_t mock_filesize=(off_t)mock_blk*64;        /* 16 MiB scratch file: plenty for a handful of tiny reads */
        FILE *f=fopen(path,"wb");
        int wrote_ok=0;
        if(f){
            char *zeros=calloc(1,(size_t)mock_blk);
            wrote_ok=1;
            for(int i=0;i<64 && wrote_ok;i++) if(fwrite(zeros,1,(size_t)mock_blk,f)!=(size_t)mock_blk) wrote_ok=0;
            free(zeros); fclose(f);
        }
        if(!f || !wrote_ok) failures+=selftest_fail("could not create the tiny mock scratch file");
        else {
            int fd_buf=open(path,O_RDONLY);
            int fd_dir=open_direct_twin(path);
            if(fd_buf<0) failures+=selftest_fail("could not open the mock scratch file");
            else {
                rng_t st=rng_seed(123);
                int combos_ok=1;
                int direct_arms = (fd_dir>=0)?2:1;
                int fds_buf1[1]={fd_buf}, fds_dir1[1]={fd_dir};
                for(int direct=0; direct<direct_arms && combos_ok; direct++){
                    for(int streams=1; streams<=2 && combos_ok; streams++){
                        int threads=2, reads=4;
                        int nreads_trial=streams*threads*reads;
                        TrialResult tr; memset(&tr,0,sizeof tr);
                        tr.lat_ms=malloc(sizeof(double)*(size_t)nreads_trial);
                        BlockRef *refs=malloc(sizeof(BlockRef)*(size_t)nreads_trial);
                        int rc=gen_single_file_refs(refs,nreads_trial,mock_filesize,mock_blk,&st);
                        if(rc==0) rc=run_trial(refs,fds_buf1,fds_dir1,direct,streams,threads,reads,mock_blk,&tr);
                        if(rc!=0){ combos_ok=0; }
                        else {
                            int64_t expect_max=(int64_t)streams*threads*reads*mock_blk;
                            if(tr.bytes<=0 || tr.bytes>expect_max || !(tr.gbps>0.0) || !isfinite(tr.gbps) || tr.n_fail>0) combos_ok=0;
                        }
                        free(refs);
                        free(tr.lat_ms);
                    }
                }
                if(!combos_ok) failures+=selftest_fail("mock read trial(s) produced an invalid/failed result (direct x{1,2} streams)");
                else printf("[selftest] PASS: end-to-end mock read pipeline OK across buffered%s x {single,dual}-stream (256KiB blocks, private scratch file)\n",
                            (fd_dir>=0)?"+direct":" (direct twin-fd unavailable on this path -- buffered-only checked)");
            }
            if(fd_buf>=0) close(fd_buf);
            if(fd_dir>=0) close(fd_dir);
        }
        remove(path);
    }

    if(failures==0){ printf("[selftest] ALL PASS\n"); return 0; }
    fprintf(stderr,"[selftest] %d check(s) FAILED\n",failures);
    return 1;
}

/* ============================ multi-file discovery + cold pool ============================
 * Everything below exists for ONE reason: a single expert shard (~2-5 GB) is
 * trivially smaller than this machine's RAM, so v1's single-file, fresh-
 * draw-with-replacement sampling (gen_single_file_refs, still used for
 * --selftest and quick single-file checks) inevitably goes page-cache-warm
 * within a session -- confirmed in practice (a --reads 2 --reps 7 run against
 * one 2.69 GB shard measured 47-58 GB/s, i.e. DRAM, not NVMe). Cold-sweep
 * mode instead spreads reads, WITHOUT REPLACEMENT, across an entire
 * directory of shards sized well past RAM, so no byte-range is ever touched
 * twice in a session -- the only argument that doesn't depend on eviction
 * timing at all. */
typedef struct {
    char **paths;
    int *fd_buf;
    int *fd_dir;
    off_t *sizes;
    int n;
} FileSet;

static int cmp_str_ptr(const void *a,const void *b){
    const char *const *pa=(const char *const *)a,*const *pb=(const char *const *)b;
    return strcmp(*pa,*pb);
}

/* Non-recursive discovery of *.safetensors directly inside `dir`, sorted for
 * determinism, buffered+direct fds opened eagerly. Skips (stderr note, not a
 * hard failure) any file too small to hold one blk+4096-byte block. Returns
 * 0 with fs->n>=1 on success, <0 if nothing usable was found. */
static int build_fileset_from_dir(const char *dir,long blk,FileSet *fs){
    DIR *d=opendir(dir);
    if(!d) return -1;
    int cap=256,n=0;
    char **names=malloc(sizeof(char*)*(size_t)cap);
    struct dirent *e;
    while((e=readdir(d))){
        size_t len=strlen(e->d_name);
        static const char suf[]=".safetensors";
        size_t suflen=sizeof(suf)-1;
        if(len<=suflen || strcmp(e->d_name+len-suflen,suf)!=0) continue;
        if(n==cap){ cap*=2; names=realloc(names,sizeof(char*)*(size_t)cap); }
        char full[2048]; snprintf(full,sizeof full,"%s/%s",dir,e->d_name);
        names[n++]=strdup(full);
    }
    closedir(d);
    if(n==0){ free(names); return -2; }
    qsort(names,(size_t)n,sizeof(char*),cmp_str_ptr);

    fs->paths=malloc(sizeof(char*)*(size_t)n);
    fs->fd_buf=malloc(sizeof(int)*(size_t)n);
    fs->fd_dir=malloc(sizeof(int)*(size_t)n);
    fs->sizes=malloc(sizeof(off_t)*(size_t)n);
    fs->n=0;
    for(int i=0;i<n;i++){
        int fb=open(names[i],COMPAT_O_RDONLY);
        if(fb<0){ fprintf(stderr,"[note] skipping %s (open failed: %s)\n",names[i],strerror(errno)); free(names[i]); continue; }
        off_t sz=lseek(fb,0,SEEK_END);
        if(sz<blk+4096){ fprintf(stderr,"[note] skipping %s (too small: %lld bytes)\n",names[i],(long long)sz); close(fb); free(names[i]); continue; }
        int fdir=open_direct_twin(names[i]);
        int idx=fs->n++;
        fs->paths[idx]=names[i]; fs->fd_buf[idx]=fb; fs->fd_dir[idx]=fdir; fs->sizes[idx]=sz;
    }
    free(names);
    if(fs->n==0) return -3;
    return 0;
}

static void fileset_free(FileSet *fs){
    for(int i=0;i<fs->n;i++){ close(fs->fd_buf[i]); if(fs->fd_dir[i]>=0) close(fs->fd_dir[i]); free(fs->paths[i]); }
    free(fs->paths); free(fs->fd_buf); free(fs->fd_dir); free(fs->sizes);
}

/* Divides every file into non-overlapping blk-spaced grid slots (+ a <4096B
 * random jitter per slot, so offsets look like production's real arbitrary
 * tensor offsets rather than perfectly page-aligned -- jitter can never
 * cause overlap since consecutive slots are a full blk apart and blk >>
 * 4096), pools every file's slots into one array, and shuffles the WHOLE
 * cross-file pool once. Returns the pool's total slot count (*pool_out is
 * NULL/0 if no file had any usable slot). No residency filtering happens
 * here -- see take_clean_blocks, which filters lazily as trials consume the
 * pool so the mincore cost scales with the session, not the container. */
static int64_t build_cold_pool(FileSet *fs,long blk,rng_t *rngst,BlockRef **pool_out){
    int64_t total=0;
    for(int i=0;i<fs->n;i++){ int64_t usable=(int64_t)fs->sizes[i]-blk-4096; if(usable>0) total+=usable/blk; }
    if(total<=0){ *pool_out=NULL; return 0; }
    BlockRef *pool=malloc(sizeof(BlockRef)*(size_t)total);
    int64_t p=0;
    for(int i=0;i<fs->n;i++){
        int64_t usable=(int64_t)fs->sizes[i]-blk-4096; if(usable<=0) continue;
        int64_t nslots=usable/blk;
        for(int64_t g=0; g<nslots; g++){
            uint32_t jitter=(uint32_t)(xorshift64star(rngst)%4096ULL);
            pool[p].file_idx=i; pool[p].off=(off_t)g*blk+(off_t)jitter; p++;
        }
    }
    shuffle_blockref(pool,total,rngst);
    *pool_out=pool;
    return total;
}

/* Advances *cursor forward through the (already-shuffled, never-reused-once-
 * consumed) pool, filling out[0..need) with blocks that do NOT look page-
 * cache-resident (mincore-sampled), skipping and counting any that do.
 * Returns the number actually filled -- less than `need` only if the pool
 * runs out, which the caller must treat as fatal for the session (it means
 * the requested working set exceeded what the discovered files can supply
 * once already-resident candidates are excluded). */
static int64_t take_clean_blocks(const BlockRef *pool,int64_t pool_n,int64_t *cursor,
                                  const FileSet *fs,long blk,int64_t need,
                                  BlockRef *out,int64_t *skipped_resident){
    int64_t filled=0;
    while(filled<need && *cursor<pool_n){
        BlockRef cand=pool[(*cursor)++];
        int r=block_looks_resident(fs->fd_buf[cand.file_idx],cand.off,blk);
        if(r==1){ if(skipped_resident) (*skipped_resident)++; continue; }
        out[filled++]=cand;
    }
    return filled;
}

/* ============================ CLI / real run ============================ */
static void usage(const char *argv0){
    fprintf(stderr,
      "usage:\n"
      "  %s --selftest\n"
      "  %s <real_expert_shard_file> [--blk BYTES] [--threads N] [--reads N]\n"
      "     [--reps N] [--seed N] [--guard-pattern STR]\n"
      "  %s <container_directory> [same options] [--min-working-set-gb N]\n"
      "     COLD-SWEEP mode (auto-selected when the path is a directory): discovers\n"
      "     every *.safetensors shard inside it and samples WITHOUT REPLACEMENT\n"
      "     across the whole set (each byte-range read at most once this session --\n"
      "     the real page-cache-defeat mechanism), auto-sizing --reps so the total\n"
      "     working set clears --min-working-set-gb (default 150, i.e. > typical\n"
      "     unified-memory sizes) unless --reps is given explicitly. Also runs a\n"
      "     lazy mincore()-sampled pre-filter that drops any candidate block that\n"
      "     already looks page-cache-resident (e.g. left over from an unrelated\n"
      "     recent full-container read pass).\n"
      "\n"
      "defaults: blk=18915328 (18.9153 MB, real GLM-5.2 int4 gate+up+down expert size)\n"
      "          threads=8 (per stream, matches production K=8 routing concurrency)\n"
      "          reads=8 (per thread per trial)  reps=11 (single-file) / auto (directory)\n"
      "          guard-pattern=gate_m15_g1  min-working-set-gb=150 (directory mode only)\n"
      "\n"
      "CRITICAL: refuses to run against a real target while --guard-pattern is detected\n"
      "running (default check: `pgrep -fl gate_m15_g1`) -- concurrent heavy NVMe\n"
      "readers make bandwidth numbers meaningless. See docs/history.md's O_DIRECT\n"
      "note and the depth-prefetch trace verdict's reopen gate.\n",
      argv0,argv0,argv0);
}

int main(int argc,char**argv){
    if(argc>=2 && !strcmp(argv[1],"--selftest")) return run_selftest(argv[0]);
    if(argc<2 || !strcmp(argv[1],"--help")){ usage(argv[0]); return (argc<2)?2:0; }

    const char *path=argv[1];
    long blk=18915328;                                /* exact, real, 4618*4096 -- see header comment */
    int threads=8, reads=8, reps=11;
    unsigned long seed_arg=0; int have_seed=0, have_reps=0;
    const char *guard_pattern="gate_m15_g1";
    double min_working_set_gb=150.0;                  /* directory/cold-sweep mode only; > typical unified-memory sizes */

    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--blk") && i+1<argc) blk=atol(argv[++i]);
        else if(!strcmp(argv[i],"--threads") && i+1<argc) threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--reads") && i+1<argc) reads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--reps") && i+1<argc){ reps=atoi(argv[++i]); have_reps=1; }
        else if(!strcmp(argv[i],"--seed") && i+1<argc){ seed_arg=strtoul(argv[++i],NULL,10); have_seed=1; }
        else if(!strcmp(argv[i],"--guard-pattern") && i+1<argc) guard_pattern=argv[++i];
        else if(!strcmp(argv[i],"--min-working-set-gb") && i+1<argc) min_working_set_gb=atof(argv[++i]);
        else { fprintf(stderr,"unknown option: %s\n",argv[i]); usage(argv[0]); return 2; }
    }
    if(blk<=0 || threads<=0 || reads<=0 || reps<=0){ fprintf(stderr,"all of --blk/--threads/--reads/--reps must be positive\n"); return 2; }

    /* NVMe-quiet guard -- MUST pass before the target is even opened. */
    if(!nvme_quiet_guard(guard_pattern)) return 1;

    struct stat pathstat;
    if(stat(path,&pathstat)!=0){ perror(path); return 1; }
    int is_dir=S_ISDIR(pathstat.st_mode);

    FileSet fs; memset(&fs,0,sizeof fs);
    if(is_dir){
        int rc=build_fileset_from_dir(path,blk,&fs);
        if(rc!=0){ fprintf(stderr,"no usable *.safetensors shard found under %s (rc=%d)\n",path,rc); return 1; }
    } else {
        int fb=open(path,COMPAT_O_RDONLY);
        if(fb<0){ perror(path); return 1; }
        off_t sz=lseek(fb,0,SEEK_END);
        if(sz<0){ perror("lseek"); close(fb); return 1; }
        if(sz<=blk*2+8192){ fprintf(stderr,"file too small (%lld bytes) for blk=%ld\n",(long long)sz,blk); close(fb); return 1; }
        int fdir=open_direct_twin(path);
        fs.n=1;
        fs.paths=malloc(sizeof(char*)); fs.paths[0]=strdup(path);
        fs.fd_buf=malloc(sizeof(int)); fs.fd_buf[0]=fb;
        fs.fd_dir=malloc(sizeof(int)); fs.fd_dir[0]=fdir;
        fs.sizes=malloc(sizeof(off_t)); fs.sizes[0]=sz;
    }
    int direct_available=0; for(int i=0;i<fs.n;i++) if(fs.fd_dir[i]>=0){ direct_available=1; break; }
    if(!direct_available) fprintf(stderr,"[note] O_DIRECT/F_NOCACHE unavailable for this target -- direct arm will be skipped, buffered-only.\n");
#ifndef _OPENMP
    fprintf(stderr,"[note] built without OpenMP (libomp not found at build time) -- concurrency will be SERIALIZED; "
                   "aggregate numbers will not reflect real hardware concurrency. See Makefile's libomp warning.\n");
#endif

    rng_t rngst = rng_seed(have_seed? seed_arg : (uint64_t)time(NULL)^0x9E3779B97F4A7C15ULL);
    unsigned long printed_seed = have_seed? seed_arg : (unsigned long)rngst;

    Cond all_conds[4]={ {0,1,"buffered/single"}, {1,1,"direct/single"}, {0,2,"buffered/dual"}, {1,2,"direct/dual"} };
    int n_conds=0; Cond active[4];
    for(int i=0;i<4;i++){ if(all_conds[i].direct && !direct_available) continue; active[n_conds++]=all_conds[i]; }
    int64_t reads_per_rep_cycle=0; for(int c=0;c<n_conds;c++) reads_per_rep_cycle += (int64_t)active[c].streams*threads*reads;

    int64_t total_container_bytes=0; for(int i=0;i<fs.n;i++) total_container_bytes+=fs.sizes[i];

    BlockRef *pool=NULL; int64_t pool_n=0, pool_cursor=0, skipped_resident=0;
    if(is_dir){
        double target_bytes=min_working_set_gb*1e9;
        int64_t auto_reps=(int64_t)ceil(target_bytes/blk/(double)reads_per_rep_cycle);
        if(auto_reps<1) auto_reps=1;
        if(!have_reps) reps=(int)auto_reps;
        else {
            double actual_gb=(double)reps*(double)reads_per_rep_cycle*blk/1e9;
            if(actual_gb*1e9 < target_bytes)
                fprintf(stderr,
                  "[WARNING] --reps %d explicitly given: working set ~%.2f GB, UNDER your own\n"
                  "[WARNING] --min-working-set-gb=%.0f floor. A working set this size risks page-cache\n"
                  "[WARNING] contamination on a large-RAM machine (exactly what a 2.69 GB single-shard,\n"
                  "[WARNING] --reads 2 --reps 7 run measured: 47-58 GB/s = DRAM, not NVMe). Omit --reps\n"
                  "[WARNING] to auto-size it, or raise it yourself.\n",
                  reps,actual_gb,min_working_set_gb);
        }
        pool_n=build_cold_pool(&fs,blk,&rngst,&pool);
        printf("[iobench_ab] COLD-SWEEP mode: dir=%s\n",path);
        printf("[iobench_ab] discovered %d usable shard(s), total container %.2f GB, cold-pool capacity %lld blocks (~%.2f GB)\n",
               fs.n,(double)total_container_bytes/1e9,(long long)pool_n,(double)pool_n*blk/1e9);
        printf("[iobench_ab] cache-defeat method: WITHOUT-REPLACEMENT sampling across the whole discovered\n"
               "[iobench_ab]   shard set (each byte-range read AT MOST ONCE this session -- non-overlapping\n"
               "[iobench_ab]   blk-spaced grid + <4096B jitter per file, pool shuffled once) + a lazy\n"
               "[iobench_ab]   mincore()-sampled pre-filter (3 pages/block) dropping any candidate that\n"
               "[iobench_ab]   already looks page-cache-resident (e.g. left over from a prior read pass).\n");
    } else {
        int64_t usable=(int64_t)fs.sizes[0]-blk-4096;
        int64_t file_pool_blocks = usable>0 ? usable/blk : 0;
        double planned_gb=(double)reps*reads_per_rep_cycle*blk/1e9;
        printf("[iobench_ab] SINGLE-FILE mode: file=%s size=%.2f GB\n",path,(double)fs.sizes[0]/1e9);
        printf("[iobench_ab] planned session volume ~%.2f GB vs this file's own %.2f GB "
               "(%lld non-overlapping %.1fMB slots available)\n",
               planned_gb,(double)fs.sizes[0]/1e9,(long long)file_pool_blocks,blk/1e6);
        if(planned_gb*1e9 > (double)fs.sizes[0]*0.5)
            fprintf(stderr,
              "[WARNING] single-file mode against a file this size cannot defeat page cache -- once\n"
              "[WARNING] touched, this file's bytes stay resident for the rest of the session (F_NOCACHE/\n"
              "[WARNING] O_DIRECT bypass caching on THEIR OWN read but cannot evict pages already resident,\n"
              "[WARNING] docs/history.md's O_DIRECT note). Point at a DIRECTORY of many shards (cold-sweep\n"
              "[WARNING] mode, auto-sized past RAM) for a genuinely cold, credible measurement.\n");
    }

    printf("[iobench_ab] seed=%lu blk=%ld (%.4f MB) threads/stream=%d reads/thread=%d reps=%d conditions=%d\n",
           printed_seed, blk, blk/1e6, threads, reads, reps, n_conds);
    double planned_gb_total=(double)reps*reads_per_rep_cycle*blk/1e9;
    printf("[iobench_ab] planned session working set ~%.2f GB across %lld reads\n",
           planned_gb_total,(long long)(reps*reads_per_rep_cycle));

    int total_trials=n_conds*reps;
    int *order=malloc(sizeof(int)*(size_t)total_trials);
    { int t=0; for(int c=0;c<n_conds;c++) for(int r=0;r<reps;r++) order[t++]=c; }
    shuffle_int(order,total_trials,&rngst);

    double **gbps_samples=malloc(sizeof(double*)*(size_t)n_conds);
    double **lat_pool=malloc(sizeof(double*)*(size_t)n_conds);
    int *lat_n=calloc((size_t)n_conds,sizeof(int));
    int max_reads_per_trial=0;
    for(int c=0;c<n_conds;c++){ int r=active[c].streams*threads*reads; if(r>max_reads_per_trial) max_reads_per_trial=r; }
    for(int c=0;c<n_conds;c++){
        gbps_samples[c]=malloc(sizeof(double)*(size_t)reps);
        lat_pool[c]=malloc(sizeof(double)*(size_t)reps*(size_t)max_reads_per_trial);
    }

    int aborted=0;
    for(int t=0;t<total_trials && !aborted;t++){
        int ci=order[t]; Cond c=active[ci];
        TrialResult tr; memset(&tr,0,sizeof tr);
        int nreads_trial=c.streams*threads*reads;
        BlockRef *refs=malloc(sizeof(BlockRef)*(size_t)nreads_trial);
        int refs_ok;
        if(is_dir){
            int64_t filled=take_clean_blocks(pool,pool_n,&pool_cursor,&fs,blk,nreads_trial,refs,&skipped_resident);
            refs_ok=(filled==nreads_trial);
            if(!refs_ok) fprintf(stderr,"[trial %d/%d] cond=%s ABORTING SESSION: cold pool exhausted (%lld/%d clean blocks available)\n",
                                  t+1,total_trials,c.label,(long long)filled,nreads_trial);
        } else {
            refs_ok=(gen_single_file_refs(refs,nreads_trial,fs.sizes[0],blk,&rngst)==0);
        }
        if(!refs_ok){ free(refs); aborted=1; break; }
        tr.lat_ms=malloc(sizeof(double)*(size_t)nreads_trial);
        int rc=run_trial(refs,fs.fd_buf,fs.fd_dir,c.direct,c.streams,threads,reads,blk,&tr);
        free(refs);
        if(rc!=0){ fprintf(stderr,"[trial %d/%d] cond=%s FAILED (rc=%d)\n",t+1,total_trials,c.label,rc); free(tr.lat_ms); continue; }
        int rep_idx = 0; for(int k=0;k<t;k++) if(order[k]==ci) rep_idx++;
        gbps_samples[ci][rep_idx]=tr.gbps;
        for(int i=0;i<tr.n_lat;i++) if(tr.lat_ms[i]>=0) lat_pool[ci][lat_n[ci]++]=tr.lat_ms[i];
        printf("[trial %d/%d] cond=%-16s gbps=%.3f dt=%.3fs bytes=%.3fGB fail=%d\n",
               t+1,total_trials,c.label,tr.gbps,tr.dt,(double)tr.bytes/1e9,tr.n_fail);
        free(tr.lat_ms);
    }
    if(aborted){
        fprintf(stderr,"[iobench_ab] session aborted before completion -- results below only cover completed trials.\n");
    }
    if(is_dir){
        printf("[iobench_ab] mincore pre-filter: skipped %lld candidate block(s) as already page-cache-resident "
               "(out of %lld pool blocks touched)\n",(long long)skipped_resident,(long long)pool_cursor);
    }

    printf("\n================ A/B RESULTS (median + 95%% bootstrap CI, n=%d/condition) ================\n",reps);
    for(int c=0;c<n_conds;c++){
        double *g=malloc(sizeof(double)*(size_t)reps); memcpy(g,gbps_samples[c],sizeof(double)*(size_t)reps);
        double gmed=median_of(g,reps); double glo,ghi;
        bootstrap_median_ci(gbps_samples[c],reps,&rngst,2000,&glo,&ghi);
        double *l=malloc(sizeof(double)*(size_t)lat_n[c]); memcpy(l,lat_pool[c],sizeof(double)*(size_t)lat_n[c]);
        double lmed=median_of(l,lat_n[c]); double llo,lhi;
        bootstrap_median_ci(lat_pool[c],lat_n[c],&rngst,2000,&llo,&lhi);
        printf("%-16s  bandwidth: %6.3f GB/s  [%.3f, %.3f]   per-read latency: %7.3f ms  [%.3f, %.3f]  (n_lat=%d)\n",
               active[c].label,gmed,glo,ghi,lmed,llo,lhi,lat_n[c]);
        printf("RESULT,%s,gbps,%.6f,%.6f,%.6f,%d\n",active[c].label,gmed,glo,ghi,reps);
        printf("RESULT,%s,latency_ms,%.6f,%.6f,%.6f,%d\n",active[c].label,lmed,llo,lhi,lat_n[c]);
        free(g); free(l);
    }
    /* Explicit scaling read-out for the reopen-gate question (recomputed from
     * fresh copies since median_of sorts in place). */
    {
        double *bs=malloc(sizeof(double)*(size_t)reps), *bd=NULL;
        memcpy(bs,gbps_samples[0],sizeof(double)*(size_t)reps);
        double single_med=median_of(bs,reps);
        free(bs);
        int dual_idx=-1; for(int c=0;c<n_conds;c++) if(active[c].streams==2 && active[c].direct==all_conds[0].direct) dual_idx=c;
        if(dual_idx>=0){
            bd=malloc(sizeof(double)*(size_t)reps);
            memcpy(bd,gbps_samples[dual_idx],sizeof(double)*(size_t)reps);
            double dual_med=median_of(bd,reps);
            free(bd);
            printf("\n[reopen-gate read-out] buffered single-stream (%d-way) median %.3f GB/s vs dual-stream (%d-way) median %.3f GB/s "
                   "-> scaling factor %.3fx (1.0x = no marginal headroom / pessimistic-bound regime; ~2.0x = full linear scaling / optimistic-bound regime)\n",
                   threads,single_med,2*threads,dual_med,dual_med/single_med);
        }
    }

    for(int c=0;c<n_conds;c++){ free(gbps_samples[c]); free(lat_pool[c]); }
    free(gbps_samples); free(lat_pool); free(lat_n); free(order);
    if(pool) free(pool);
    fileset_free(&fs);
    return aborted?1:0;
}
