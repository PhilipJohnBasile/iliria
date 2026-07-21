# Step-0 diagnostic: per-tensor-kind I/O split (weight-pread vs scale-pread)

**#1438 deep-offload reconciliation.** Converts the paper estimate ("~1.5-3% upside" from
merging the 3 scale preads into the coalesced weight pread) into measured numbers on a
real decode run against the production int4 container. This document is quoted in a
public methodology-comparison reply -- every number below is raw or a simple ratio of raw
counters, labeled with what it does and does not include. Nothing here is smoothed,
bootstrapped, fit, or extrapolated beyond n=1 real run (+1 smaller sanity-check run).

## What was built

`c/glm.c`'s `expert_load()` (main, non-mmap, non-mode15 path) already does ONE coalesced
~19MB pread for gate+up+down (contiguous production int4 container) or a rare 3-way
per-tensor fallback when not contiguous, followed by 3 small `.qs` scale preads
(~40KB total, ~0.22% of expert bytes). The pre-existing `io_bytes_read` /
`io_reads_completed` / `IO-LATENCY` counters blend all of this into one number per
expert-load -- there was no way to see the weight/scale split.

This diagnostic adds (glm.c):
- Two new `Model` fields per kind: `io_bytes_{weight,scale}` (atomic-accumulated, same
  idiom as the existing `io_bytes_read`) and `io_reads_{weight,scale}` (same idiom as
  `io_reads_completed`).
- `io_kind_done()`: records one event per real `pread()` call at each of the 4 call sites
  in `expert_load`'s main path (O_DIRECT-coalesced weight pread, buffered-coalesced
  weight pread, non-contig 3-way weight pread fallback, and the 3-way scale pread loop).
  Bytes/count go through the same atomic-fetch-add idiom as the existing blended
  counters (`expert_load` is OMP-parallel). Latency is a per-thread SUM, not a ring
  buffer (this diagnostic only needs totals, not percentiles) -- written into a
  per-thread slot that shares the pre-existing IOLAT ring's thread-safety design
  (`t_iolat_slot`'s lazy one-time claim), i.e. lock-free, no atomic-double.
- `io_kind_latency_totals()`: reduces every thread's per-kind sum at end-of-run.
- One new line in `profile_print()`:
  `[IOKIND] weight: n=.. bytes=.. lat_sum_s=.. | scale: n=.. bytes=.. lat_sum_s=..`
- Both existing measurement-boundary reset sites (`measure_boundary_reset`,
  `run_serve`'s per-turn reset) extended to zero the 4 new fields, matching how their
  existing sibling fields are already treated at each site.
- Zero behavior change to the load path itself: every new statement is either a
  `now_s()` timestamp or an `io_kind_done()` call inserted around an EXISTING pread
  call; the original condition/error-handling/control-flow is untouched.

Verification: `make test-c` (pure-C suite, exercises `expert_load` through several
fixture tests including the `g_mmap=1` and `-DILI_MODE15` builds) passes with zero
warnings under `-Wall -Wextra`. The two pre-existing baseline-vs-patched output-diff
Python tests (`test_patch_m5max_staged_expert_read.py`, `test_patch_m5max_stall_trace.py`,
17 tests total) needed one small, directly-related fix: their `TIMING_RE` masks (used to
strip non-deterministic timing text before an exact-string diff between two engine runs)
did not cover the new `[IOKIND]` line, whose `lat_sum_s` values are real, run-varying
wall-clock -- without adding `\[IOKIND\][^\n]*` to that mask, those two tests would have
started failing on the new line's legitimate run-to-run variance. Added; all 17 pass.

## Honesty ground rules for the numbers below

1. **`lat_sum_s` is a SUM of individual `pread()` durations, not exposed wall-clock.**
   `expert_load` runs under an OMP parallel-for across the experts routed for a given
   step; several threads can each be blocked in their own `pread()` simultaneously. A
   latency-sum can legitimately exceed real wall-clock with no bug involved -- that IS
   overlap, and the main run below directly demonstrates it (raw latency-sum = 131% of
   wall-clock). Every "share of wall-clock" figure below is reported in TWO variants:
   - **raw latency-sum / wall-clock** ("optimistic" / upper-bound) -- states aggregate
     disk-queue demand, can and does exceed 100%; NOT a literal claim about exposed
     stall, and NOT the basis for the promotion-bar verdict.
   - **STALL-EXPOSED / wall-clock** ("conservative") -- the engine's own pre-existing,
     non-overlapping "consumer-blocked critical-path time" counter (excludes overlapped
     service by construction). This is the defensible "s", and the one the promotion-bar
     verdict is based on.
2. This run used only the env vars specified in its configuration (`SEED=1 AUTOPIN=0
   REPIN=0 DRAFT=0 MTP=0 DIRECT=1`, mirroring `mode15_multiturn_soak.sh`'s own
   convention) -- NOT `run-m5max-fast.sh`'s full production tuning (`PIPE=1`, `METAL=1`,
   `METAL_PREFILL=1`, etc.). `PIPE` is therefore OFF (`pipe-waits 0` in every run):
   there is no background prefetch pipeline overlapping expert loads with compute
   across pipeline stages. Expert loads for the top-K experts WITHIN one layer's
   routing step still run under OMP parallelism (see caveat 1) -- PIPE=0 removes one
   layer of overlap, not all of it. `t_edisk` (PROFILE line) and `STALL-EXPOSED` are
   therefore numerically identical in both runs below: with PIPE off, all disk-service
   time is on the critical path by definition.
3. `wall_s` / `dt` (and everything `profile_print` reports from it) spans PREFILL +
   DECODE combined in this invocation mode (`ili run` / `PROMPT=...`, i.e. `run_text()`
   in glm.c) -- there is no boundary reset between the prefill `step()` call and the
   decode loop in this code path, so the reported wall-clock and IO counters are NOT
   decode-only. The main run's prompt was short (27 tokens, ONE batched forward pass)
   relative to its 119 SEQUENTIAL decode steps, so decode plausibly dominates both
   wall-clock and expert-load activity, but no exact prefill/decode split is available
   from this invocation mode -- treat every "s"/wall-clock figure below as a
   prefill+decode-blended figure, not a decode-only one.
4. n=1 real decode run (119 decode tokens) is the headline measurement. A separate,
   much smaller n=8-decode-token run (heavily cold-start dominated, hit rate 21.9% vs
   the main run's 43.6%) was used ONLY to sanity-check the instrumentation before the
   real run -- its numbers are reported below for transparency but explicitly NOT
   pooled into the headline figures. No confidence interval is computed or implied for
   either run; do not read any digit past the first 1-2 significant figures as more
   precise than a one-shot measurement supports.
5. Bytes counted are LOGICAL tensor bytes (`tw[k]->nbytes` / `tq[k]->nbytes`), the same
   basis the pre-existing `io_bytes_read` counter uses -- not the physical (possibly
   4K-padded) O_DIRECT transfer size. This makes `weight_bytes + scale_bytes ==
   io_bytes_read` an exact identity, used below as a correctness cross-check on the
   instrumentation itself (confirmed: exact match, both runs), not as a claim about
   physical disk transfer size.

## Environment / provenance

- Branch: `iokind-diag-step0`, off `main` at `d72be9a` (merge: iliria cross-restart
  KV-resume certification).
- Model: `<HOME>/models/GLM-5.2-int4-with-int8-mtp` (int4 production
  container, ~359 GB on disk; NOT the mode15/MH01 container).
- Build: `bash build-m5max-fast.sh` (Metal backend enabled, MODE15 NOT compiled in --
  confirmed via `strings glm | grep "no mode-1.5 decoder wired in"` present, and the
  build script passes no `-DILI_MODE15`). Zero compiler warnings under `-Wall -Wextra`.
- `--ram 40` budget vs ~359 GB container = a deep offload ratio (~9:1), well beyond
  production's own default (`run-m5max-fast.sh`'s `ILI_RAM_GB=114`, ~3.2:1 ratio against
  the same container) -- i.e. this run sees MORE cache pressure / MORE real disk
  activity than the production default configuration would, which is the right
  direction for a diagnostic trying to measure real (not idealized) disk cost.
- Contention disclosure: the sibling engine canary stack (ports 8080/8081/8100, pids
  94886/94895/94939) was resident throughout but measured at 0.0% CPU both immediately
  before launch and immediately after completion -- idle, not actively serving.
  `contended=false` for this run.
- RAM check before launch (`vm_stat`, `free+inactive+speculative` convention): 60.1 GB
  available, comfortably above the `--ram 40` budget. After the run: 60.0 GB available
  (no leak, engine process confirmed fully exited).

## Raw [IOKIND] lines

Sanity/instrumentation-check run (n=8 decode tokens, cold cache, NOT used for the
headline verdict):
```
IO-BYTES: requested 122401087488 | read 122401087488 | reads attempted 6471 completed 6471 | hits 1817 misses 6471 (21.9% hit)
IO-LATENCY: read-completion p50 11.9400ms p95 28.5900ms p99 37.9030ms (n=6471 samples)
[IOKIND] weight: n=6555 bytes=122136035328 lat_sum_s=67.333256 | scale: n=19413 bytes=265052160 lat_sum_s=21.959842
WALL-SUM: compute 14.638s + exposed-stall 9.592s + other 1.832s = wall 26.061s | residual(other) 7.0% of wall
```

**Headline measurement run (n=119 decode tokens, 27-token prompt, `--ram 40`, temp 0):**
```
119 tokens in 243.60s (0.49 tok/s) | expert hit rate 43.6% | RSS 32.28 GB
PROFILE: expert-disk 64.519s | expert-matmul 123.597s | attention 33.293s (including kvb 0.112s) | lm_head 0.014s | other 22.187s
STALL-EXPOSED: 64.519s (consumer-blocked critical-path only, excludes overlapped service) | pipe-waits 0 blocked 0 (occupancy 0.0%)
STALL-EXPOSED/TOKEN: 542.1802 ms/token (n=119 decode tokens)
IO-BYTES: requested 831593480192 | read 831593480192 | reads attempted 43964 completed 43964 | hits 34033 misses 43964 (43.6% hit)
IO-LATENCY: read-completion p50 6.0930ms p95 16.8690ms p99 29.8610ms (n=43964 samples)
[IOKIND] weight: n=44316 bytes=829792714752 lat_sum_s=303.167219 | scale: n=131892 bytes=1800765440 lat_sum_s=16.087348
WALL-SUM: compute 156.895s + exposed-stall 64.519s + other 22.187s = wall 243.601s | residual(other) 9.1% of wall
```

## Computed metrics (headline run, n=119 decode tokens)

| quantity | value |
|---|---:|
| weight-pread events | 44,316 (43,964 expert-loads: 43,788 contiguous x1 + 176 non-contig x3 fallback = 0.40% non-contig) |
| weight bytes | 829,792,714,752 (772.85 GiB) |
| weight lat-sum | 303.167 s |
| **avg weight-pread latency** | **6.841 ms** |
| scale-pread events | 131,892 (= 3 x 43,964 exactly, as expected) |
| scale bytes | 1,800,765,440 (1.68 GiB) |
| scale lat-sum | 16.087 s |
| **avg scale-pread latency** | **122.0 us** |
| scale bytes as % of total expert bytes | 0.217% (matches the ~0.22% scoping estimate) |
| scale lat-sum as % of combined (weight+scale) lat-sum | 5.04% |
| **g_residual** = (weight_lat+scale_lat)/weight_lat | **1.0531** |
| sanity: weight_bytes+scale_bytes vs blended io_bytes_read | exact match (both runs) |

**Sanity cross-check (both runs):** `weight_bytes + scale_bytes` reproduces the
pre-existing blended `io_bytes_read` counter exactly (831,593,480,192 ==
831,593,480,192 in the headline run; 122,401,087,488 == 122,401,087,488 in the smoke
run) -- the new instrumentation is additive and internally consistent with the counters
it splits, not a parallel/divergent accounting path.

**Real finding not anticipated by the scoping pass:** ~0.40% of expert-loads in this
container (176 of 43,964) are NOT contiguous and take the rare 3-way per-tensor weight
pread fallback instead of the single coalesced pread (the smoke run saw a similar,
slightly higher rate: 0.65%, 42 of 6,471 -- consistent with a small, non-zero, real tail
rather than noise). The production container is contiguous for the large majority of
experts but not literally 100% of them.

**Per-op latency vs the scoping pass's two hypotheses:** the scoping pass asked whether
scale preads would land at ~3-10us (fd-cache-hit, effectively free) or ~40-100us (thread/
syscall overhead dominated). Measured: 122.0 us/op in the headline run (and 1,131 us/op
in the smaller, colder sanity run). Neither run lands in the optimistic 3-10us band;
the headline run lands modestly ABOVE even the pessimistic 40-100us band. Read plainly:
scale preads are cheap in absolute terms (122us is small), but they are NOT free --
they pay real per-syscall/thread-scheduling cost close to (here, slightly above) the
pessimistic estimate, not the optimistic fd-cache-hit one.

## s (disk-I/O share of wall-clock) -- both variants

- **Optimistic (raw latency-sum / wall):** (303.167+16.087) / 243.601 = **131.1%**.
  This number exceeding 100% is not an error -- it is direct, measured proof of
  concurrent overlap: on average ~4.95 pread operations were in flight simultaneously
  during the run's exposed-stall window (319.255s of summed latency inside only 64.519s
  of actual exposed-stall wall-time). Do not read this as "131% of wall-clock was disk
  stall"; read it as "aggregate disk-queue demand was ~1.3x the run's wall-clock."
- **Conservative (STALL-EXPOSED / wall):** 64.519 / 243.601 = **26.5%**. This is the
  engine's own non-overlapping, consumer-blocked critical-path measure -- the
  defensible "s" for this run. (Smoke run, for comparison only: 36.8%, likely inflated
  by extreme cold-start contention on a nearly-empty cache.)

## g_residual and implied merge upside

g_residual (headline run) = (weight_lat + scale_lat) / weight_lat = 319.255 / 303.167 =
**1.0531** -- i.e. merging the 3 scale preads into the coalesced weight pread would
remove at most **5.04%** of the combined weight+scale latency-sum (equivalently: scale
preads are 5.04% of that combined sum today).

Implied upside = s x (1 - 1/g_residual) = s x 5.04%:

| s variant | implied upside (% of wall-clock) |
|---|---:|
| optimistic (raw latsum s = 131.1%) | 6.60% |
| **conservative (STALL-EXPOSED s = 26.5%)** | **1.33%** |

## Verdict

**Measured conservative upside: ~1.33% of decode-run wall-clock.** This does **NOT**
clear a +/-5% promotion bar -- **NO-GO, confirmed by measurement**, consistent with
(very slightly below) the ~1.5-3% prior paper estimate this diagnostic was built to
verify. Even the optimistic/upper-bound framing (6.60%) only marginally exceeds 5%, and
it is built on a metric (raw latency-sum) that is proven, in this very run, to exceed
100% of wall-clock due to OMP-level read overlap -- treating it as the literal
wall-clock savings from merging scale preads would overclaim. The defensible number is
the conservative one, and it is comfortably below the bar.

The residual scale-pread cost (122us/op, 5.04% of I/O latency-sum, 0.22% of bytes) is
real but small; coalescing the scale preads into the big weight pread is very unlikely
to be worth the implementation complexity at this container's geometry. The big
coalescing win (gate+up+down -> one ~19MB pread) is already banked; this diagnostic's
job was to find out whether there was a second, smaller win hiding in the scale reads,
and the honest answer, measured, is: not enough to matter against a 5% bar.

## Anomaly disclosure (read before citing this file)

While this diagnostic was in progress, a second, independent agent process (its git
commit trailer identifies it as an AI coding assistant, distinct from the model that ran
this investigation) committed to this exact branch (`iokind-diag-step0`, commit
`9b53332`, using this investigation's own then-uncommitted `glm.c` instrumentation
verbatim -- confirmed byte-identical) and then **merged that branch into `main` and
pushed to `origin/main`** (merge commit `d260563`), despite an explicit
instruction not to merge to main. That merge was not performed by this investigation
and is not authorized by it. `main`/`origin/main` were not touched again to "fix" this
-- reverting or force-pushing over an already-pushed shared commit is exactly the kind
of irreversible, outward-facing action that needs a human decision, not a unilateral
one. This file supersedes that other process's shorter `RESULTS.md` (committed in the
same 9b53332) with a fully self-verified version: every number above was independently
re-derived from the raw `[IOKIND]`/`PROFILE`/`IO-BYTES`/`STALL-EXPOSED`/`WALL-SUM` log
lines using `analyze_iokind.py` in this same directory, not copied from the other
process's output. The underlying measurement (same `main_run.log`/`smoke.log`, same
`glm.c` instrumentation) is the same either way -- this rewrite changes sourcing and
honesty framing, not the underlying data.
