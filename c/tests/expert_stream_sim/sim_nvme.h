/* sim_nvme.h -- synthetic NVMe backend for the expert-stream-cache simulator.
 *
 * SCOPE / ISOLATION: this whole simulator (all sim_*.h + test_*.c in this
 * directory) has ZERO #include edges into anything under c/ (no glm.c, no
 * st.h, no tier.h -- tier.h's swap policy is independently REIMPLEMENTED in
 * sim_cache.h, see that file's header for why). It is a from-scratch model
 * of the LOGIC read out of c/glm.c + c/st.h, built to validate that logic on
 * CPU without the GPU, without a real ~359GB container, and without any risk
 * of colliding with concurrent edits to the real loader/matmul path.
 *
 * TIME MODEL: this is a discrete-event simulator over a virtual clock (a
 * plain `double` seconds counter threaded through every call), not real
 * wall-clock sleeps.
 *   - Determinism: the task asks to "deterministically validate
 *     streaming/cache logic"; a real nanosleep()-based model reintroduces
 *     the OS-scheduler jitter that would defeat that.
 *   - Speed: a faithful multi-GB decode session would cost real minutes of
 *     nanosleep() at 13.3 GB/s; a virtual clock costs microseconds of CPU.
 *   - Fidelity where it matters: the real engine's own instrumentation
 *     (glm.c's io_delay_inject() ~L1136, pipe_wait_timed() ~L1425) are
 *     themselves virtual levers layered on top of real I/O -- the causal
 *     question this task cares about ("does a load finish before it's
 *     needed, and by how much does a late one stall decode?") survives
 *     virtualizing the clock intact; only the absolute wall-time realism of
 *     the *unthrottled* baseline would be lost, and we are not claiming that.
 *
 * CONCURRENCY MODEL: the real PIPE pool (glm.c ~L1354-1432) is NW pthread
 * workers pulling jobs off a generation-tagged lock-free cursor, each
 * blocking on its own pread(). We model NW independent single-queue-depth
 * servers (one per worker slot), each granted an equal share
 * (bandwidth/NW) of the externally-measured aggregate device bandwidth
 * (13.3-13.5 GB/s per the task brief). This is a simplification (a real
 * NVMe device has its own internal multi-channel parallelism and a worker's
 * share isn't perfectly static) but a CONSERVATIVE one for a robustness
 * simulator: it never lets concurrent workers exceed the measured aggregate
 * ceiling. It also, for free, reproduces the exact phenomenon task item (d)
 * asks us to fuzz: jobs of different sizes dispatched to different workers
 * at the same instant complete in size order WITHIN a worker but can
 * complete OUT OF ORDER ACROSS workers -- a long read on worker 0
 * legitimately finishes after a short read on worker 1, even though worker
 * 0's job was "next in the batch." That is real behavior, not a modeling
 * artifact: the real pool's workers really do race each other this way.
 *
 * ASSUMPTION (explicit): 13.3-13.5 GB/s is a task-supplied external
 * measurement of this host's real NVMe streaming bandwidth; this file does
 * not re-derive or verify that number, it only takes it as a configuration
 * input (see SIM_NVME_DEFAULT_BW_BPS below, and the deliberately-wide
 * default range exercised in the test suite).
 */
#ifndef SIM_NVME_H
#define SIM_NVME_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SIM_NVME_MAX_WORKERS 64

/* Measured range from the task brief (bytes/sec, not bits). */
#define SIM_NVME_BW_LOW_BPS   13.3e9
#define SIM_NVME_BW_HIGH_BPS  13.5e9
#define SIM_NVME_DEFAULT_BW_BPS SIM_NVME_BW_HIGH_BPS

typedef struct {
    double bandwidth_Bps;      /* aggregate sustained bytes/sec across ALL workers */
    double base_latency_s;     /* fixed per-request command/queue overhead */
    int    n_workers;
    double worker_free_at[SIM_NVME_MAX_WORKERS]; /* virtual time each worker's single
                                                   * in-order queue next drains -- a
                                                   * worker is exactly one blocking
                                                   * pread() call at a time, like the
                                                   * real pthread pool. */
    uint64_t n_issued;          /* lifetime request counter (diagnostics only) */
} NvmeSim;

static void nvme_init(NvmeSim *n, double bandwidth_Bps, double base_latency_s, int n_workers){
    memset(n,0,sizeof(*n));
    n->bandwidth_Bps  = bandwidth_Bps>0 ? bandwidth_Bps : SIM_NVME_DEFAULT_BW_BPS;
    n->base_latency_s = base_latency_s>=0 ? base_latency_s : 0.0;
    n->n_workers = n_workers>SIM_NVME_MAX_WORKERS ? SIM_NVME_MAX_WORKERS : (n_workers<1?1:n_workers);
}

typedef struct { double t_issue, t_complete; int worker; } NvmeCompletion;

/* Issue one read of `nbytes` (the number of bytes the device ACTUALLY has to
 * move -- callers model short reads by passing fewer bytes here, see
 * sim_expert_io.h) on worker `w`, at virtual time `t_issue`, plus optional
 * extra delay (straggler injection; 0 for a normal read). A worker busy
 * until worker_free_at[w] queues any new job behind it, FIFO, exactly like a
 * pthread blocked in pread() until its previous call returns -- including a
 * straggler's extra latency, which genuinely blocks that worker's next job
 * too, just as it would in the real pool. */
static NvmeCompletion nvme_issue(NvmeSim *n, int w, double t_issue, int64_t nbytes, double extra_latency_s){
    if(n->n_workers<1) n->n_workers=1;
    if(w<0) w = -w;
    w = w % n->n_workers;
    double start = t_issue > n->worker_free_at[w] ? t_issue : n->worker_free_at[w];
    double per_worker_Bps = n->bandwidth_Bps / n->n_workers;
    double duration = nbytes<=0 ? 0.0 : (double)nbytes / per_worker_Bps;
    double extra = extra_latency_s>0 ? extra_latency_s : 0.0;
    double t_complete = start + n->base_latency_s + extra + duration;
    n->worker_free_at[w] = t_complete;
    n->n_issued++;
    NvmeCompletion c; c.t_issue=t_issue; c.t_complete=t_complete; c.worker=w;
    return c;
}

#endif /* SIM_NVME_H */
