# iliria cross-restart KV-resume: certified TTFT ABBA

Status: **FINAL.** The full run (n=6/arm, Section 4) completed 2026-07-20 and
is the certified number cited from `STORY.md` §7 and this project's README.

## 1. What this certifies

iliria persists attention KV state to disk (`<SNAP>/.ili_kv`, `c/glm.c`
`kv_disk_append`/`kv_disk_load`) and reloads it across a full engine **process
restart**. This is a different mechanism from the sibling engine's own certified
147.8x KV-resume number (`the sibling engine's kv-resume results
kv_resume_abba_results.json`), which measures a same-process, in-RAM prompt
cache on the sibling engine's Qwen3-32B and explicitly excludes process
spawn / model load from its TTFT. iliria's own prior number for its own
mechanism (3.07s vs 436.65s, "142x", `docs/PERFORMANCE_THEORY.md`
`serve_kv_persistence_ttft_speedup_x`) was retired as uncitable: N=1, the cold
and resumed observations used different prompts, and it never stated whether
TTFT included process spawn or model load.

This harness (`run_kv_resume_abba.py`, same directory) ports the sibling engine's harness
DESIGN -- same-prompt ABBA, randomized block order, proof of the mechanism
actually firing (not assumed), bootstrap CI -- to iliria's own cross-restart
mechanism, fixing the same three flaws that retired the old number.

## 2. Design

**Prompt.** One fixed, deterministic, sha256-pinned synthetic prompt (tiled
filler text, `build_target_prompt`), sent byte-identical to both arms on
every trial. Actual token count is read off the engine's own telemetry
(`[API] KV slot S prefix P/T token, prefill K` -- see below), not assumed from
a local tokenizer.

**Arms.**
- **COLD**: `.ili_kv` (slot 0) absent before engine start. Fresh process,
  full prefill of every prompt token.
- **RESUME**: `.ili_kv` (slot 0) is a copy of a canonical **primed** file --
  produced once, up front, by a real engine run that submitted the identical
  target prompt and was verified (by directly parsing the on-disk KV header,
  not by trusting a timing assumption) to have persisted at least the full
  prompt's KV rows before being copied aside. Fresh process, `.ili_kv`
  present, engine loads it at startup (`kv_disk_load`, before READY), then
  the identical prompt is resubmitted; iliria's own prefix-match (`glm.c`
  run_serve raw-mode: walks the incoming tokenized prompt against the
  resident `hist[]` array token-for-token) finds a full-length match and
  prefills zero new tokens.

Unlike the sibling engine's same-process design (one server held constant across all
trials, cache flushed/primed by an untimed request before each trial),
iliria's KV is disk-persisted and requires a genuine process restart to
exercise the claim under test -- so every trial in both arms restarts the
engine process from scratch. That difference is also why iliria's canonical
primed `.ili_kv` does not need re-priming per resume trial the way the sibling engine's
in-RAM cache needs re-priming per resume request: the same bytes on disk
reproduce the same resumed state every time; the harness simply copies the
one primed file into place before each resume trial and deletes it before
each cold trial.

**TTFT definition** (verbatim from the harness's own `TTFT_DEFINITION`
constant, carried into every JSON result so no reader has to guess):

> TTFT = wall-clock seconds from immediately-before-subprocess.Popen of the
> iliria engine to the harness's own on_text callback's FIRST invocation
> with non-empty decoded text. IDENTICAL measurement code and definition for
> both arms. INCLUDED for both arms alike: process spawn; one-time
> model-weight load; the per-slot `.ili_kv` disk load performed at startup,
> BEFORE READY; receipt of READY; transmission of the prompt; the engine's
> own prefix-match against resident KV history; and the first decode step.
> This is DELIBERATELY DIFFERENT from the sibling engine's own TTFT_DEFINITION, which
> excludes process spawn and model load because the sibling engine never restarts its
> process. iliria's claim is specifically about surviving a restart, so
> process spawn and model load are symmetric, paid-in-both-arms costs,
> deliberately left inside the timed window rather than excluded -- excluding
> them would silently reintroduce the exact ambiguity that retired the old
> 142x number.

**Consequence of this definition worth stating plainly**: because model load
is included and is a substantial fixed cost paid by both arms, this harness's
speedup ratio is not comparable in magnitude to the sibling engine's 147.8x (which factors
model load out entirely). A smaller ratio here is an expected, correctly
documented result of measuring the harder, more complete cross-restart claim
iliria actually makes -- not a weaker result.

**Randomization.** Near-verbatim port of the sibling engine's `build_abba_schedule`:
4-trial blocks, each independently `[cold,resume,resume,cold]` or
`[resume,cold,cold,resume]` by a seeded fair coin per block -- classic ABBA
counterbalancing that cancels a linear drift (thermal ramp, contention
ramping up) between arms regardless of which arm leads a given block.

**Resume-hit proof (not assumed).** Every trial's engine stderr is parsed for
two telemetry lines the engine itself prints, independent of anything this
harness assumes about which arm is running:
- `[KV] resumed conversation from disk: N tokens in T s (no re-prefill)` --
  present iff `kv_disk_load` found a valid, non-empty `.ili_kv` at startup.
- `[API] KV slot S prefix P/T token, prefill K` -- P = length of the resident
  KV prefix that matched the incoming prompt, T = the prompt's own token
  count, K = T-P = tokens that actually needed prefilling.

A genuine resume hit is `prefix == prompt_tokens, prefill == 0`; a genuine
cold run is `prefix == 0, prefill == prompt_tokens`. Each trial record's
`resume_hit_proof.matches_arm_expectation` field is this comparison, computed
from the engine's own report, never from the schedule alone.

**Token-identity proof.** Both arms decode at `temperature=0` (pure greedy,
`glm.c`: `if(g_temp<=0) return argmax_v(...)`), so the same prompt run through
the same weights on both arms should produce byte-identical output. The
harness sha256's every trial's generated text and reports whether the set of
cold-arm hashes equals the set of resume-arm hashes.

**Bootstrap CI.** Near-verbatim port of the sibling engine's `bootstrap_median_ci` /
`bootstrap_speedup_ci` (stdlib-only percentile bootstrap, seeded
`random.Random`, resample-with-replacement on each arm's raw TTFT samples
independently, then on the ratio of medians).

**Contention disclosure.** Two sibling engine Qwen3-32B servers (ports 8080,
8081) and their router (port 8100) were live, expected, user-owned processes
for the full duration of this run -- explicitly not waited out or touched.
Every trial record's `contention` field states
what else was running at that trial's start. Absolute TTFTs in this run are
not contention-free; the cold/resume **ratio** (both arms interleaved by
ABBA through the same contention regime) is the headline number for exactly
that reason.

## 3. How this fixes each flaw that retired the old 142x number

| Old flaw (142x, N=1) | This harness |
|---|---|
| Cold and resume observations used **different prompts** | One fixed, sha256-pinned prompt, byte-identical, submitted verbatim in both arms on every trial |
| **N=1**, no interval | N pairs (see Section 4) with a percentile bootstrap CI on both arms and on the ratio |
| **Ambiguous TTFT accounting** -- unclear whether process/model/KV load were included | One explicit `TTFT_DEFINITION` string, stated in full in Section 2 above and carried verbatim into the JSON output, identical code path for both arms |
| Resume was **assumed**, not proven | `resume_hit_proof` parses the engine's own `[API] KV slot ... prefix/prefill` and `[KV] resumed ...` telemetry per trial; mismatches are counted and reported, not hidden |
| No ordering control | Randomized ABBA/BAAB block schedule (seeded, reproducible), not a fixed cold-phase-then-resume-phase order |

## 4. Results

**Full run 2026-07-20 (n=6 per arm, 3 randomized ABBA/BAAB blocks, order
CRRCRCCRRCCR, 414-token prompt, temp 0, 12/12 trials ok):**

| metric | value |
|---|---:|
| cold median TTFT | **153.92s** [145.99, 163.49] |
| resume median TTFT | **22.56s** [19.18, 30.97] |
| **speedup (cold/resume)** | **6.8x  CI [5.0x, 8.0x]** (percentile bootstrap) |
| resume-hit proof | **12/12** `matches_expectation=True` (cold: `prefix=0, prefill=414`; resume: `prefix=414, prefill=0` — engine's own `[API]`/`[KV]` counters) |
| token identity across arms | **True** (byte-identical completions, greedy) |
| `.ili_kv` load time (resume arm) | ~4.0-5.8s of the 22.6s |

Pilot (same design, N=1, earlier + heavier contention): cold 352.17s /
resume 96.42s — consistent direction, superseded by the full run above.

**Contention annotation: every trial ran `contended=true`** — the sibling engine
drafter-canary stack (two Qwen3-32B servers + router) was live on the same
GPU throughout, by design (real-workload conditions). Ratio remains valid
(both arms equally contended); absolute TTFTs are contention-inflated and
NOT clean-machine numbers. What resume actually eliminates is the ~131s
contended prefill (prefill 414 -> 0); the remaining ~22.6s is dominated by
process spawn + 744B model load, which no KV cache can remove.

## 5. Comparison to the sibling engine 147.8x number (not the same claim)

| | the sibling engine (certified) | iliria (this harness) |
|---|---:|---:|
| mechanism | same-process in-RAM prompt cache | cross-restart, disk-persisted `.ili_kv` |
| engine | Qwen3-32B-4bit (dense, MLX) | GLM-5.2 744B MoE int4 (NVMe-streamed experts) |
| process restarts between trials? | no -- one process, whole run | **yes -- every trial, both arms** |
| TTFT includes process spawn/model load? | no (explicitly excluded) | **yes (explicitly included)** |
| cold median | 6.792s | 153.92s [145.99, 163.49] |
| resume median | 0.046s | 22.56s [19.18, 30.97] |
| speedup | 147.8x [137.1, 157.8] | **6.8x [5.0, 8.0]** |

These are two different, individually-honest numbers for two different
mechanisms on two different engines. Neither substitutes for the other;
citing the sibling engine's 147.8x for iliria's own cross-restart claim (or vice versa)
would misattribute a sibling engine's result.

## 6. Provenance

- Harness: `c/bench-m5max/ili-kv-resume-abba/run_kv_resume_abba.py`
- Raw results: `c/bench-m5max/ili-kv-resume-abba/results/kv_resume_abba_results.json`,
  `results/trials.jsonl`, `results/pilot.json`
- Branch / commit: development history not preserved in this repository's
  single-commit public release.
- `glm` engine binary: built in this worktree from `c/glm.c`
  (confirmed byte-identical to `<repo>` main checkout's
  `glm.c` at run time) via `build-m5max-fast.sh` (native Apple Silicon,
  Metal backend, `-mcpu=native` CPU fallback; prefill matmuls stay on the
  CPU path by default -- `g_mm_forcecpu` -- so greedy decode is not subject to
  the documented Metal-GEMM-rounding nondeterminism unless
  `ILI_METAL_PREFILL=1` is set, which this run does not set).
- The user's pre-existing real `.ili_kv` (slot 0, 1.3GB, in active use) was
  backed up before this harness touched the model directory and restored
  afterward: `.ili_kv.PRE-HARNESS-BACKUP-20260720-163646`. Slot 1
  (`.ili_kv.1`) was never touched (this harness only ever uses slot 0).
