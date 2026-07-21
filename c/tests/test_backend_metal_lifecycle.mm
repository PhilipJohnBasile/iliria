// Lifecycle stress test for the persistent-state lab backend
// (backend_metal_m5max_lab.mm = generated backend + patch_m5max_persistent_state).
//
// Exercises, with async MoE command buffers in flight:
//   1. register/unregister churn of expert slabs from a concurrent thread, with
//      reallocation (free + fresh posix_memalign) after every unregister --
//      run under MallocScribble=1 so a use-after-free of a no-copy slab shows up
//      as a numeric mismatch or a crash;
//   2. deterministic deferred-unregister: unregister a slab that an in-flight
//      command buffer references and assert the release was deferred/blocked;
//   3. fallback semantics: submitting against an unregistered slab returns 0
//      (CPU fallback) and recovers after re-registration;
//   4. the pinned fast path (ili_metal_register_pinned takes no in-flight refs);
//   5. shutdown drains without deadlock (SIGALRM watchdog).
//
// With ILI_METAL_PERSISTENT_STATE unset the concurrent-churn phases are skipped
// (they are exactly the unsafe pattern the lifecycle work fixes) and the test
// verifies the default path is unaffected.
#include "../backend_metal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h>
#include <signal.h>

// Lab-only entry points added by tools/patch_m5max_persistent_state.py.
extern "C" void ili_metal_register_pinned(void *base, size_t len);
extern "C" void ili_metal_lab_lifecycle_stats(uint64_t *refs_acquired, uint64_t *deferred,
                                               uint64_t *waits, uint64_t *retired_pending);
extern "C" void ili_metal_lab_fail_next_command_buffer(int n);

enum { D = 2048, II = 768, NB = 16, NRQ = 4, FMT = 1 /* int8 */ };
static const size_t PG = 16384;

static int g_persistent = 0;

static void watchdog(int) {
  fprintf(stderr, "lifecycle test: WATCHDOG TIMEOUT (deadlock?)\n");
  _exit(3);
}

static size_t roundpg(size_t n) { return ((n + PG - 1) / PG) * PG; }

// One expert's weights (int8 gate/up [II,D], down [D,II]) + scales, in two
// page-aligned registered slabs, mirroring how the engine lays experts out.
struct Expert {
  uint8_t *slab = nullptr;   // wlen bytes: [Wg | Wu | Wd]
  float *fslab = nullptr;    // flen bytes: [Sg | Su | Sd]
  size_t wlen = 0, flen = 0;
};
static size_t wlen_bytes() { return roundpg((size_t)II * D * 2 + (size_t)D * II); }
static size_t flen_bytes() { return roundpg(((size_t)II * 2 + D) * sizeof(float)); }

static Expert expert_alloc() {
  Expert e; e.wlen = wlen_bytes(); e.flen = flen_bytes();
  if (posix_memalign((void **)&e.slab, PG, e.wlen) ||
      posix_memalign((void **)&e.fslab, PG, e.flen)) { fprintf(stderr, "OOM\n"); exit(1); }
  return e;
}
static void expert_fill_random(Expert &e) {
  for (size_t i = 0; i < (size_t)II * D * 2 + (size_t)D * II; i++)
    e.slab[i] = (uint8_t)((rand() % 255) - 127);
  for (size_t i = 0; i < (size_t)II * 2 + D; i++)
    e.fslab[i] = 0.01f + (rand() % 50) / 50000.f;
}

// Pointer table shared with the churn thread (guarded by mu).
struct Table {
  std::mutex mu;
  Expert ex[NB];
  const void *g[NB], *u[NB], *d[NB];
  const float *gs[NB], *us[NB], *ds[NB];
  void refresh_ptrs(int e) {
    uint8_t *sp = ex[e].slab; float *fp = ex[e].fslab;
    g[e] = sp; u[e] = sp + (size_t)II * D; d[e] = sp + (size_t)II * D * 2;
    gs[e] = fp; us[e] = fp + II; ds[e] = fp + 2 * II;
  }
};

// CPU reference for one expert's contribution rows (int8 SwiGLU), from explicit
// weight pointers so it can be evaluated against a content snapshot.
static void expert_ref_rows(const uint8_t *wg, const uint8_t *wu, const uint8_t *wd,
                            const float *sg, const float *su, const float *sd,
                            const float *xrows, int nrows, float *hh /*[nrows,D]*/) {
  std::vector<float> gg(II);
  for (int r = 0; r < nrows; r++) {
    const float *xr = xrows + (size_t)r * D;
    for (int o = 0; o < II; o++) {
      const int8_t *rg = (const int8_t *)wg + (size_t)o * D;
      const int8_t *ru = (const int8_t *)wu + (size_t)o * D;
      float a = 0, b = 0;
      for (int k = 0; k < D; k++) { a += (float)rg[k] * xr[k]; b += (float)ru[k] * xr[k]; }
      a *= sg[o]; b *= su[o];
      gg[o] = (a / (1.f + expf(-a))) * b;
    }
    float *hr = hh + (size_t)r * D;
    for (int o = 0; o < D; o++) {
      const int8_t *rd = (const int8_t *)wd + (size_t)o * II;
      float a = 0;
      for (int k = 0; k < II; k++) a += (float)rd[k] * gg[k];
      hr[o] = a * sd[o];
    }
  }
}

static double normalized_error(const float *got, const float *ref, size_t n) {
  double ma = 0, ym = 0;
  for (size_t i = 0; i < n; i++) { ma = fmax(ma, fabs(got[i] - ref[i])); ym = fmax(ym, fabs(ref[i])); }
  return ma / (ym + 1e-9);
}

// ---------------------------------------------------------------------------
// Phase 1: async submissions vs concurrent unregister/realloc churn.
// The churn thread replaces slabs with byte-identical copies at new addresses
// (register new -> unregister old -> free old), so the reference output never
// changes; any GPU read of a freed slab shows up as a mismatch or a crash.
// ---------------------------------------------------------------------------
static int phase_churn(Table &T) {
  const int R = NB * NRQ, ITERS = 200;
  std::vector<int> xoff(NB), nr(NB, NRQ), rows(R, 0);
  for (int e = 0; e < NB; e++) xoff[e] = e * NRQ;
  std::vector<float> xg((size_t)R * D), rw(R);
  for (auto &v : xg) v = ((rand() % 2000) - 1000) / 1000.f;
  for (auto &v : rw) v = 0.1f + (rand() % 100) / 100.f;

  // Reference (contents are churn-invariant): sum over experts of rw * hh rows.
  std::vector<float> refout(D, 0.f), hh((size_t)NRQ * D);
  for (int e = 0; e < NB; e++) {
    expert_ref_rows((const uint8_t *)T.g[e], (const uint8_t *)T.u[e], (const uint8_t *)T.d[e],
                    T.gs[e], T.us[e], T.ds[e], &xg[(size_t)xoff[e] * D], NRQ, hh.data());
    for (int r = 0; r < NRQ; r++) {
      int gr = xoff[e] + r;
      for (int o = 0; o < D; o++) refout[o] += rw[gr] * hh[(size_t)r * D + o];
    }
  }

  std::atomic<bool> stop{false};
  std::atomic<int> churns{0};
  std::thread churner;
  if (g_persistent) {
    churner = std::thread([&] {
      int victim = 0;
      while (!stop.load(std::memory_order_acquire)) {
        Expert fresh = expert_alloc();
        Expert old;
        {
          std::lock_guard<std::mutex> lk(T.mu);
          old = T.ex[victim];
          memcpy(fresh.slab, old.slab, old.wlen);      // byte-identical contents
          memcpy(fresh.fslab, old.fslab, old.flen);
          ili_metal_register(fresh.slab, fresh.wlen);
          ili_metal_register(fresh.fslab, fresh.flen);
          T.ex[victim] = fresh;
          T.refresh_ptrs(victim);
        }
        // Deferred unregister under test: blocks while any in-flight command
        // buffer still references the old slab, then the backing is freed
        // (MallocScribble poisons it) and immediately reused by the allocator.
        ili_metal_unregister(old.slab);
        ili_metal_unregister(old.fslab);
        free(old.slab);
        free(old.fslab);
        churns.fetch_add(1, std::memory_order_relaxed);
        victim = (victim + 1) % NB;
      }
    });
  }

  int ok_iters = 0, fallbacks = 0, mismatches = 0;
  std::vector<float> out(D);
  const void *g[NB]; const void *u[NB]; const void *d[NB];
  const float *gs[NB]; const float *us[NB]; const float *ds[NB];
  for (int it = 0; it < ITERS; it++) {
    {
      std::lock_guard<std::mutex> lk(T.mu);   // snapshot pointers racing the churn
      memcpy(g, T.g, sizeof(g)); memcpy(u, T.u, sizeof(u)); memcpy(d, T.d, sizeof(d));
      memcpy(gs, T.gs, sizeof(gs)); memcpy(us, T.us, sizeof(us)); memcpy(ds, T.ds, sizeof(ds));
    }
    memset(out.data(), 0, sizeof(float) * D);
    IliMetalMoeHandle *h = ili_metal_moe_block_begin(
        NB, D, II, FMT, g, u, d, gs, us, ds, xg.data(), xoff.data(), nr.data(),
        rows.data(), rw.data());
    if (!h) { fallbacks++; continue; }         // slab vanished pre-commit: clean fallback
    if (!ili_metal_moe_block_end(h, out.data())) { fallbacks++; continue; }
    double nerr = normalized_error(out.data(), refout.data(), D);
    if (nerr < 1e-4) ok_iters++;
    else { mismatches++; fprintf(stderr, "  iter %d MISMATCH nerr=%.2e\n", it, nerr); }
  }
  stop.store(true, std::memory_order_release);
  if (churner.joinable()) churner.join();

  int pass = mismatches == 0 && ok_iters > ITERS / 2;
  printf("  churn: %d ok / %d fallback / %d mismatch / %d slab churns  %s\n",
         ok_iters, fallbacks, mismatches, churns.load(), pass ? "ok" : "*** FAIL");
  return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Phase 2: deterministic deferred unregister while a command buffer is in
// flight.  Only meaningful with the lifecycle enabled.
// ---------------------------------------------------------------------------
static int phase_deferred(Table &T) {
  if (!g_persistent) { printf("  deferred: skipped (persistent state off)\n"); return 0; }
  const int NRB = 32, R = NB * NRB;            // bigger block: several ms on the GPU
  std::vector<int> xoff(NB), nr(NB, NRB), rows(R, 0);
  for (int e = 0; e < NB; e++) xoff[e] = e * NRB;
  std::vector<float> xg((size_t)R * D), rw(R, 0.25f), out(D);
  for (auto &v : xg) v = ((rand() % 2000) - 1000) / 1000.f;

  uint64_t waits0 = 0, def0 = 0;
  ili_metal_lab_lifecycle_stats(nullptr, &def0, &waits0, nullptr);
  int deferred_seen = 0, attempts = 0;
  for (attempts = 0; attempts < 10 && !deferred_seen; attempts++) {
    IliMetalMoeHandle *h = ili_metal_moe_block_begin(
        NB, D, II, FMT, T.g, T.u, T.d, T.gs, T.us, T.ds, xg.data(), xoff.data(),
        nr.data(), rows.data(), rw.data());
    if (!h) return fprintf(stderr, "  deferred: begin failed\n"), 1;
    int victim = attempts % NB;
    Expert old = T.ex[victim];
    Expert fresh = expert_alloc();
    memcpy(fresh.slab, old.slab, old.wlen);
    memcpy(fresh.fslab, old.fslab, old.flen);
    ili_metal_register(fresh.slab, fresh.wlen);
    ili_metal_register(fresh.fslab, fresh.flen);
    T.ex[victim] = fresh; T.refresh_ptrs(victim);
    std::thread un([&] {                        // must block until the cb completes
      ili_metal_unregister(old.slab);
      ili_metal_unregister(old.fslab);
      free(old.slab); free(old.fslab);
    });
    memset(out.data(), 0, sizeof(float) * D);
    int end_ok = ili_metal_moe_block_end(h, out.data());
    un.join();
    if (!end_ok) return fprintf(stderr, "  deferred: end failed\n"), 1;
    uint64_t waits1 = 0, def1 = 0;
    ili_metal_lab_lifecycle_stats(nullptr, &def1, &waits1, nullptr);
    if (def1 > def0 && waits1 > waits0) deferred_seen = 1;
  }
  printf("  deferred: unregister-in-flight deferred+blocked after %d attempt(s)  %s\n",
         attempts, deferred_seen ? "ok" : "*** FAIL");
  return deferred_seen ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Phase 3: unresolved slab -> clean fallback (returns 0), then recovery.
// ---------------------------------------------------------------------------
static int phase_fallback(Table &T) {
  const int R = NB;
  std::vector<int> xoff(NB), nr(NB, 1), rows(R, 0);
  for (int e = 0; e < NB; e++) xoff[e] = e;
  std::vector<float> xg((size_t)R * D, 0.5f), rw(R, 1.0f), out(D, 0.f);
  ili_metal_unregister(T.ex[0].slab);
  int r_unreg = ili_metal_moe_block(NB, D, II, FMT, T.g, T.u, T.d, T.gs, T.us, T.ds,
                                     xg.data(), xoff.data(), nr.data(), rows.data(),
                                     rw.data(), out.data(), 1);
  ili_metal_register(T.ex[0].slab, T.ex[0].wlen);
  int r_rereg = ili_metal_moe_block(NB, D, II, FMT, T.g, T.u, T.d, T.gs, T.us, T.ds,
                                     xg.data(), xoff.data(), nr.data(), rows.data(),
                                     rw.data(), out.data(), 1);
  int pass = (r_unreg == 0 && r_rereg == 1);
  printf("  fallback: unregistered->%d re-registered->%d  %s\n", r_unreg, r_rereg,
         pass ? "ok" : "*** FAIL");
  return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Phase 3b: forced-nil command buffer.  The submission must fail into the CPU
// fallback WITHOUT leaking the acquired slab refs -- a leaked ref would make
// the following unregister block forever (the watchdog turns that into a
// failure) and would leave retired nodes pending at shutdown.
// ---------------------------------------------------------------------------
static int phase_failcb(Table &T) {
  const int R = NB;
  std::vector<int> xoff(NB), nr(NB, 1), rows(R, 0);
  for (int e = 0; e < NB; e++) xoff[e] = e;
  std::vector<float> xg((size_t)R * D, 0.5f), rw(R, 1.0f), out(D, 0.f);
  ili_metal_lab_fail_next_command_buffer(1);
  int r_fail = ili_metal_moe_block(NB, D, II, FMT, T.g, T.u, T.d, T.gs, T.us, T.ds,
                                    xg.data(), xoff.data(), nr.data(), rows.data(),
                                    rw.data(), out.data(), 1);
  // Refs must have been released by the bail-out path: unregistering a slab the
  // failed submission had acquired must return immediately, not block forever.
  ili_metal_unregister(T.ex[1].slab);
  ili_metal_register(T.ex[1].slab, T.ex[1].wlen);
  int r_ok = ili_metal_moe_block(NB, D, II, FMT, T.g, T.u, T.d, T.gs, T.us, T.ds,
                                  xg.data(), xoff.data(), nr.data(), rows.data(),
                                  rw.data(), out.data(), 1);
  uint64_t pending = 0;
  ili_metal_lab_lifecycle_stats(nullptr, nullptr, nullptr, &pending);
  int pass = (r_fail == 0 && r_ok == 1 && pending == 0);
  printf("  failcb: nil-cb->%d recovered->%d retired-pending=%llu  %s\n", r_fail, r_ok,
         (unsigned long long)pending, pass ? "ok" : "*** FAIL");
  return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Phase 4: pinned slabs take no in-flight refs (immutable fast path).
// ---------------------------------------------------------------------------
static int phase_pinned() {
  Expert e = expert_alloc();
  srand(7);
  expert_fill_random(e);
  ili_metal_register_pinned(e.slab, e.wlen);
  ili_metal_register_pinned(e.fslab, e.flen);
  const void *g[1] = { e.slab }, *u[1] = { e.slab + (size_t)II * D },
             *d[1] = { e.slab + (size_t)II * D * 2 };
  const float *gs[1] = { e.fslab }, *us[1] = { e.fslab + II }, *ds[1] = { e.fslab + 2 * II };
  int xoff[1] = { 0 }, nr[1] = { 1 }, rows[1] = { 0 };
  std::vector<float> xg(D, 0.25f), out(D, 0.f);
  float rw[1] = { 1.0f };
  uint64_t refs0 = 0, refs1 = 0;
  ili_metal_lab_lifecycle_stats(&refs0, nullptr, nullptr, nullptr);
  int ok = ili_metal_moe_block(1, D, II, FMT, g, u, d, gs, us, ds, xg.data(), xoff, nr,
                                rows, rw, out.data(), 1);
  ili_metal_lab_lifecycle_stats(&refs1, nullptr, nullptr, nullptr);
  std::vector<float> hh(D), ref(D);
  expert_ref_rows((const uint8_t *)g[0], (const uint8_t *)u[0], (const uint8_t *)d[0],
                  gs[0], us[0], ds[0], xg.data(), 1, hh.data());
  for (int o = 0; o < D; o++) ref[o] = hh[o];
  double nerr = normalized_error(out.data(), ref.data(), D);
  int no_refs = !g_persistent || refs1 == refs0;
  int pass = ok && nerr < 1e-4 && no_refs;
  printf("  pinned: ok=%d nerr=%.2e refs_delta=%llu  %s\n", ok, nerr,
         (unsigned long long)(refs1 - refs0), pass ? "ok" : "*** FAIL");
  // Unregister of a pinned slab defers the wrap release to shutdown; the backing
  // itself is process-lifetime, so it is freed only after ili_metal_shutdown().
  ili_metal_unregister(e.slab);
  ili_metal_unregister(e.fslab);
  // e.slab / e.fslab intentionally leaked until after shutdown (see main).
  return pass ? 0 : 1;
}

int main(void) {
  signal(SIGALRM, watchdog);
  alarm(900);                                   // any hang (deadlock) fails hard
  g_persistent = getenv("ILI_METAL_PERSISTENT_STATE") &&
                 atoi(getenv("ILI_METAL_PERSISTENT_STATE"));
  if (!ili_metal_init()) { printf("Metal unavailable (skipping)\n"); return 0; }
  printf("Metal lifecycle stress tests (persistent state %s):\n",
         g_persistent ? "ON" : "off");
  srand(20260714);
  Table T;
  for (int e = 0; e < NB; e++) {
    T.ex[e] = expert_alloc();
    expert_fill_random(T.ex[e]);
    ili_metal_register(T.ex[e].slab, T.ex[e].wlen);
    ili_metal_register(T.ex[e].fslab, T.ex[e].flen);
    T.refresh_ptrs(e);
  }

  int fail = 0;
  fail |= phase_churn(T);
  fail |= phase_deferred(T);
  fail |= phase_fallback(T);
  fail |= phase_failcb(T);
  fail |= phase_pinned();

  uint64_t pending = 0;
  ili_metal_lab_lifecycle_stats(nullptr, nullptr, nullptr, &pending);
  printf("  retired-pending before shutdown: %llu\n", (unsigned long long)pending);

  // Shutdown must drain in-flight refs and return (watchdog catches deadlock).
  for (int e = 0; e < NB; e++) { ili_metal_unregister(T.ex[e].slab); ili_metal_unregister(T.ex[e].fslab); }
  ili_metal_shutdown();
  printf("  shutdown: returned (no deadlock)  ok\n");
  for (int e = 0; e < NB; e++) { free(T.ex[e].slab); free(T.ex[e].fslab); }

  printf(fail ? "metal lifecycle tests: FAILED\n" : "metal lifecycle tests: ok\n");
  return fail;
}
