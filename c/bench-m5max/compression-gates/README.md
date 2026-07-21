# Compression campaign gates — run order and status

**NOTE (public repo):** this is a dated operational runbook (2026-07-18) for
the tiered int4/int2 mixed-precision container campaign. The registration
docs it points to (`../the compression-campaign registration,
`../the compression-gate threshold record), most of the gate scripts named below
(`gate1_integrity.sh`, `gate2_v2_tail_spec.py`, `gate2_5_weight_proxy.py`,
`gate3_v2_activation.py`, `gate4b_tf_drift.py`), and `../campaign-log.md` are
part of the private research history and are not shipped in this repo (only
`gate_m15_g1.py`, an unrelated Mode-1.5 gate, ships under this directory).
This runbook also predates the campaign's actual outcome: Gate 3 later
**FAILED** on the built container (long-generation collapse) and the
container was deleted — see `STORY.md` §8 and `DEFENSIBLE-ASSETS.md` (b)2 for
the final, certified verdict this file's "tomorrow" never reached.

Binding specs (read these first, do not alter): `../compression-campaign-
the campaign registration (esp. "Reviewer amendments (2026-07-18)", items 1-8) and
`../the compression-gate threshold record. This file is operational glue only —
it records *what to run, in what order, and whether the engine needs to be
idle for it* — it does not re-derive or restate the registered thresholds.

## Status as of tonight (2026-07-18, container conversion in progress)

| Script | Status | Certifies |
|---|---|---|
| `gate1_integrity.sh` | built, smoke-tested | container integrity (amendment 1: CONTAINER cert only) |
| `gate2_decoder_correctness.py` | built, smoke-tested | superseded numeric spec — see gate2_v2 |
| `gate2_v2_tail_spec.py` | built, `--selftest-only` green | int2-fitted tail distribution (amendment 2) |
| `gate2_5_weight_proxy.py` | renamed from `gate3_activation_error.py` | weight-space proxy ONLY — never gate 3 (amendment 3) |
| `gate3_v2_activation.py` | built, `compare --fixture-selftest` green; `capture` is a spec stub | real activation comparison (amendment 3) |
| the gate-3 capture spec | written by `gate3_v2_activation.py capture` | the glm.c patch gate 3's capture side needs — **needs human approval** |
| `gate4b_tf_drift.py` | built, `--selftest-only` green | teacher-forced continuation drift, PRIMARY long-horizon safeguard (amendment 4) |
| gate4 (SCORE quality) | pre-existing: `tools/eval_glm.py` | original registration's Quality Gate — unchanged by the amendments |
| gate5 (throughput) | pre-existing pieces, no single harness | throughput + amendment 5's new required fields (not yet assembled) |

None of tonight's new/changed scripts have been run against the real
container beyond header-only shard discovery (machine discipline: the
conversion owns the disk; "no real-model reads beyond completed-shard
headers" tonight). Every numeric code path is fixture/selftest-verified
against synthetic data instead — run `--selftest-only` / `--fixture-
selftest` / `capture` (no args) on any of the three new scripts to re-
verify before trusting them tomorrow.

## Tomorrow's exact run order

Numbered steps are sequential dependencies (each needs the previous one's
*output*, not necessarily a human waiting around). All commands below are
given relative to `c/` (this project's C engine + tooling root — the same
anchor `tools/eval_glm.py` and `run-m5max-fast.sh` are normally invoked
from); the three new gate scripts happen to live in
`bench-m5max/compression-gates/` so their own paths are written from
there. **"engine idle"** means specifically: no `glm`/`glm_m5max`/`ili`
process alive (checked by `tools/quant_container.py`'s `engine_busy()` /
`require_idle()`, the same guard `measure_expert_quant_error.py` already
uses) — the container converter is a *different* process and is NOT "the
engine" in this sense, but steps 1-4 still share its disk, so nothing else
heavy should run concurrently with it either.

1. **Conversion resume** — `bash bench-m5max/resume-conversion.sh` (or the
   `nohup` form logged in `bench-m5max/conversion.log`'s `STOPPED-0940:`
   lines). Engine idle:
   **not applicable** (no inference engine involved; this is the
   converter itself — the thing the rest of tonight's work was staying
   out of the way of). Runs until all 164 shards exist.

2. **gate1** — `gate1_integrity.sh <outdir>`. Engine idle: **not
   required** (no inference engine involved at all; safe to run even
   while the converter is still writing new shards per its own docstring
   — reads only shards already present).

3. **gate2_v2 calibrate** — `gate2_v2_tail_spec.py --calibrate --outdir
   <outdir>`. Freezes `gate2_v2_limits.json` from shards `out-00028`
   through whatever the window resolves to (recorded verbatim in the
   output — see the script's `--calib-start`/`--calib-end`, default
   28-46). **Commit `gate2_v2_limits.json` immediately after this step**
   — it is the frozen reference every later holdout run depends on; treat
   it like any other frozen-epoch artifact (never silently re-derive it
   once holdout checks exist against it). Engine idle: **not required**
   (numpy + disk reads only), but this DOES compete for disk I/O — best
   run once the converter is idle/done, or throttled if run concurrently.

4. **gate2_v2 holdout** — `gate2_v2_tail_spec.py --holdout --outdir
   <outdir>`. Requires shards *outside* the calibration window to exist,
   i.e. requires step 1 to have progressed past `--calib-end` (tonight's
   window ends at shard 46 of 164 — holdout will correctly report
   `FAIL_INSUFFICIENT_POPULATION` if run before that, which is expected,
   not a bug). Engine idle: **not required**, same disk-sharing caveat as
   step 3.

5. **[engine addition approval]** — human/reviewer step. Read
   the gate-3 capture spec (re-generate first via `gate3_v2_activation.py
   capture` to confirm it's still current against `glm.c` — it re-checks
   itself). If approved: apply the ~19-line patch, rebuild, and — per this
   campaign's own epoch discipline (the compression-campaign registration,
   "Baseline provenance") — **freeze a new epoch** for the patched binary
   (it must never be silently called epoch-06). Engine idle: **the engine
   is being rebuilt**, i.e. by definition no engine process should be
   running during this step.

6. **gate3 capture** — run the newly-approved-and-built binary in REPLAY
   mode with `ILI_MOE_INPUT_DUMP` set (exact invocation in
   the gate-3 capture spec), against the BASELINE (int4-only) container,
   at early/mid/late positions per amendment 3. Engine idle: **REQUIRED
   both before and during** — this step *is* an engine run (the one place
   in this whole list that directly executes the engine binary before
   gate 4/4b), so it needs the machine quiet exactly like any other timed
   engine run in this campaign's discipline (AC power, no thermal warning,
   no foreign engine/compiler process, etc. — see the launch protocol
   entries in `../campaign-log.md` for the standing checklist).

7. **gate3 compare** — `gate3_v2_activation.py compare --capture-baseline
   <bin from step 6> --outdir <candidate container>`. Engine idle: **not
   required** (pure numpy against the capture file + container tensors,
   no engine process involved).

8. **gate4 SCORE** — `tools/eval_glm.py` (existing harness; unchanged by
   the amendments), deterministic CPU SCORE path, n=100 paired vs the
   61.3%/62.5% replicated baseline, per the original registration's Gate
   4. Engine idle: **REQUIRED** (this runs the engine; expect multiple
   hours — the last comparable run was 3h14m for 120 questions per
   `../campaign-log.md`).

9. **gate4b TF-drift** — two REPLAY runs (baseline binary, candidate-
   aware binary) with `ILI_REPLAY_LOGIT_DUMP` set on the SAME frozen
   2K-4K token continuation, then `gate4b_tf_drift.py --baseline
   <dump> --candidate <dump>`. Engine idle: **REQUIRED for the two REPLAY
   runs**; the comparison step itself afterward needs no engine.

10. **cache-storage question** (amendment 5: "FIRST determine whether the
    expert cache stores compressed or expanded weights"). Engine idle:
    **not required** — this is a code-reading question, not a run.
    **Lead from tonight** (stated as a lead, not a verified conclusion —
    confirming it properly is tomorrow's step, not preempted here):
    `glm.c:137-142`'s `ESlot` struct comment says gate/up/down (`g,u,d`)
    are "VISTE dentro `slab`" (views inside `slab`, a `uint8_t*` buffer)
    — i.e. the resident cache appears to hold the quantized/COMPRESSED
    bytes directly (one coalesced pread into `slab`, `QT` views with the
    right stride/format over it), not expanded float weights. There is
    also a separate `fslab` (float slab) whose exact role ("nel fallback
    hanno buffer propri" / "in the fallback they have their own buffers")
    needs checking before concluding anything — confirm by reading
    `expert_load()` (glm.c:1163) and how `slab` vs `fslab` are actually
    populated and read from in `moe()`'s matmul calls. If confirmed
    compressed: per amendment 5, this implies a possible NONLINEAR
    daily-driver residency gain (more experts fit per GB of cache) that
    gate 5 should report on, not just raw hit-rate.

11. **gate5** — throughput gate. Both registered configs (daily-driver
    RAM_GB=114 and streaming-stress RAM_GB=2) are the SAME driver script,
    `run-m5max-fast.sh`, via its `ILI_RAM_GB` override (default 114;
    confirmed tonight: `run-m5max-fast.sh:9,120`) — not two different
    scripts. `tools/parse_stall_trace.py` looks like the existing piece
    for pulling exposed-stall/IOLAT numbers back out of engine output,
    but tonight's search found **no single existing harness that runs
    baseline-vs-candidate on the frozen matrix and emits one paired
    verdict** — that assembly (or confirmation one already exists
    elsewhere and was missed) is itself part of this step, stated plainly
    rather than assumed done. Per amendment 5, whatever runs this must
    ALSO record: resident expert count, effective cached source bytes,
    expanded cache bytes, hit rate, miss bytes/token, decode time — framed
    by step 10's answer. Predeclared bar unchanged: >= +15% median paired
    throughput on the frozen matrix. Engine idle: **REQUIRED** (the
    longest-running step in this list; two configs x baseline/candidate).

## Not in this run order (registered, but explicitly deferred/separate)

- **Amendment 1** (container-vs-runtime split): a framing rule for how to
  read gates 1-4/step 11 above, not a step of its own — until the Metal
  row-decoder (or some other GPU path) actually consumes this container's
  bitstream, step 11's throughput numbers certify the **CPU-decoded**
  mixed-container path, and must be labeled that way, never quietly
  presented as validating a GPU decode that doesn't exist yet.
- **Amendment 6** (ActiveFlow deadline/byte-aware falsifier) and
  **amendment 8** (future case E): registered but explicitly not blocking
  case D's gates above; not scheduled here.
- **Amendment 7** (self-fork localization): narrowed, not final; not
  blocking compression, not a step here.
- 3x2K free-generation smokes (amendment 4's SECONDARY check): repetition/
  corruption/formatting only, run whenever convenient after step 9; never
  a substitute for gate4b's PRIMARY teacher-forced verdict.

## Re-verifying tonight's deliverables (no real paths touched)

```
python3 gate2_v2_tail_spec.py --selftest-only
python3 gate3_v2_activation.py capture            # re-check + rewrite the spec stub
python3 gate3_v2_activation.py compare --fixture-selftest
python3 gate4b_tf_drift.py --selftest-only
```
