# L2 numerical-validation capture plan: CPU vs Metal-prefill attention output

Status: **plan + ready-to-apply patch, not yet run**. No engine changes were made
(glm.c and backend_metal.mm are untouched on disk); the quality bench owns the
engine tonight. This document is what to run tomorrow, once the engine is free
(`bench-m5max/campaign-state.json`'s `queued_after_gates` lists "numerical kernel
validation on tiny fixture" right after the serve ABBA matrix).

## The question

`run-m5max-serve.sh` documents `ILI_METAL_PREFILL=1` as accepting "rounding-
variance greedy forks" against the `ILI_METAL_PREFILL=0` byte-exact CPU default
(measured 6-9x TTFT win, the prefill-I/O study §7, ruling 2026-07-14). That
acceptance was qualitative. This is the next tier of rigor: actually measure how
far the two paths diverge at the attention-output level, across a grid of
sequence lengths, context depths, and layer depths, with a fixed metric set,
so "accepts the rounding-variance forks" becomes a measured bound instead of an
assumption.

## What was read to answer "what's the cheapest capture point"

`c/glm.c`: `attention()` (line 1355) and its single caller, `layer_forward()`
(line 1981). `c/backend_metal.mm`: every Metal command-buffer path (`ili_metal_attn_prefill`,
`ili_metal_attn_decode`, `ili_metal_layer_decode`) ends in
`[e endEncoding]; [cb commit]; [cb waitUntilCompleted];` before control returns to
`glm.c` -- i.e. every GPU path is synchronous by construction; there is no
outstanding async work to race when reading its output back on the CPU.

Existing debug/dump hooks surveyed and found NOT to fit:
- `MTP_DEBUG` (glm.c ~2176, ~2382): speculative-decode draft/verify accounting,
  unrelated subsystem.
- `SCORE`/`SERVE` env checks (glm.c ~3448): top-level run-mode selection, not a
  data dump.
- `ILI_ROUTE_TRACE` (`tools/patch_m5max_route_trace.py`): MoE expert-routing
  trace (which experts, not attention tensors) -- wrong subsystem, and it is
  itself a **generated-file patch**, never applied to `glm.c` (see "Why a patch,
  not an edit" below) -- the precedent this plan follows.

Conclusion: no existing hook captures attention output. A tiny new one is needed.

## The capture point

`glm.c`'s per-layer forward function calls `attention()` and adds its result into
the residual stream immediately after:

```c
static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    ...
    attention(m,l,li,nrm,S,pos_base,tmp);              // <-- capture tmp here
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];  //     before this residual add
    ...
}
```

This ONE call site is the cheapest possible capture point, for three reasons:

1. **It is the only call site.** `attention()` has exactly one caller
   (`layer_forward()`), used for both the 78 main layers and the MTP layer.
2. **It is path-agnostic.** Whether `attention()` internally takes the CPU
   per-token loop, the CPU absorption path, the fused Metal4 decode-absorption
   path, or the Metal-prefill batched path (`ILI_METAL_PREFILL=1`, `S>4`), every
   branch writes its result into the SAME `out` parameter (bound to `tmp` at the
   call site) before returning. The capture hook does not need to know or care
   which branch ran.
3. **It needs no extra synchronization.** Confirmed above: every Metal path
   already blocks on `waitUntilCompleted` before `attention()` returns, so `tmp`
   is fully resident, CPU-readable data by the time the hook would run.

**Scoping note for a clean comparison:** the `ILI_METAL4_MOE=1` fused-decode
fast path (`layer_forward()`'s own `g_metal_enabled && S<=4 ...` branch, lines
1990-2038) bypasses `attention()`/`tmp` entirely for eligible decode steps,
computing attention internally as part of a fused GPU command buffer. This
feature is off by default (`ILI_METAL4_MOE` unset in `run-m5max-serve.sh`,
`metal4=0` unless `ab-m5max-k6-matrix.sh`'s `AXIS=metal4|pstate`) and is a
SEPARATE axis from `ILI_METAL_PREFILL`. Captures for this study should be taken
with `ILI_METAL4_MOE=0` (the default) so every S value -- including S=1 and S=4,
which would otherwise sometimes hit the fused fast path -- always flows through
the one instrumented call site.

## The patch

`scripts/capture_layer_outputs.patch` -- a ready-to-apply unified diff against
`c/glm.c`. **glm.c itself is untouched**; this follows the same convention as the
`tools/gen_m5max_*.py` / `tools/patch_m5max_*.py` generated-file workflow, where
`glm.c` is the pristine source of truth that lab variants are generated FROM
(`gen_m5max_engine.py glm.c glm_m5max.c`) and then patched (`patch_m5max_*.py
glm_m5max.c glm_m5max.c`) -- never edited in place. This capture hook is written
as a literal `git apply`-compatible diff rather than another
`tools/patch_m5max_*.py` script because it is meant to be applied directly to
whichever build is at hand (`glm.c` for a stock build, or `glm_m5max.c` if
validating the M5-Max lab variant too -- the same hunks apply cleanly to either,
since `gen_m5max_engine.py`'s transforms do not touch `attention()`'s call site
or signature). Apply it, build normally, capture, then revert:

```sh
cd c/
git apply scripts/capture_layer_outputs.patch   # or: patch -p1 < scripts/capture_layer_outputs.patch
make glm METAL=1                                 # normal build, opt-in env vars only
git checkout -- glm.c                            # revert when done capturing
```

Verified tonight WITHOUT building or running the engine: `git apply --check
scripts/capture_layer_outputs.patch` applies cleanly against the current
`glm.c`, and the patched source passes a standalone `cc -fsyntax-only -DILI_METAL`
frontend parse with zero new diagnostics versus the unpatched file (the only
errors either file produces this way are pre-existing, unrelated to this hook:
`omp_in_parallel` needs `-fopenmp`/`omp.h`, not part of this syntax check).

### Env vars it adds (opt-in, zero behavior/overhead change when unset)

| var | meaning |
|---|---|
| `ILI_LAYER_CAPTURE_DIR` | directory to write captures into. Unset = hook fully disabled (one cached check, no allocation). |
| `ILI_LAYER_CAPTURE_LAYERS` | comma-separated layer indices to capture (e.g. `5,39,74`). Unset = capture **every** layer on every call -- fine for a single short prompt, NOT recommended for real S/T combinations (a 16K-token prefill would write 78 files per chunk). Always set this for anything beyond a smoke test. |

### File format (one file per captured call)

Filename: `L<layer>_S<S>_T<pos_base>.bin`, written into `ILI_LAYER_CAPTURE_DIR`.

| offset | field | type |
|---|---|---|
| 0 | magic `"FACAPT1\0"` | 8 bytes |
| 8 | format version (1) | uint32 LE |
| 12 | layer index | uint32 LE |
| 16 | S (tokens in this call) | uint32 LE |
| 20 | pos_base (T: context depth at capture time) | uint32 LE |
| 24 | D (hidden dim; 6144 for GLM-5.2 int4) | uint32 LE |
| 28 | metal_prefill (0=CPU path, 1=Metal-prefill path was active) | uint32 LE |
| 32 | payload: S x D float32, row-major | S*D x 4 bytes |

Read/written by `tools/compare_layer_captures.py`'s `load_capture()` /
`write_capture()` -- the latter also builds synthetic fixtures for the unit
tests, so the format is exercised without the patch ever being applied.

## The grid: S x T x layer, both paths

| axis | values |
|---|---|
| S (sequence length / new tokens in the captured call) | 1, 4, 16, 64, 256 |
| T (`pos_base`: context depth at capture time) | 128, 1024 (1K), 4096 (4K), 16384 (16K) |
| layer (early / mid / late, within the 75 MoE-routed layers 3-77) | 5 (early), 39 (mid), 74 (late) |
| path | CPU (`ILI_METAL_PREFILL=0`), Metal (`=1`) |

5 x 4 x 3 x 2 = **120 capture files** for the full grid (60 CPU + 60 Metal,
paired for 60 comparisons). `ILI_LAYER_CAPTURE_LAYERS=5,39,74` keeps every run
to exactly 3 files regardless of S/T, so a full grid sweep is 40 engine
invocations (one per S x T cell, each producing 3 layer files), run twice (once
per path) -- 80 invocations, or fewer if a single long transcript naturally
passes through multiple T depths in one run (recommended: see below).

**How to hit a specific (S, T) cell:** S is the number of NEW tokens in one
`/v1/chat/completions` turn; T (`pos_base`) is however many tokens already sit in
that KV slot when the turn starts. The cheapest way to hit all 4 T depths in one
session per S value: load a monotonic-history transcript once up to just under
each T target (reusing `scripts/long_ctx_profile.py`'s `build_transcript()` /
`_LineCycler`, which deterministically slices real repo source files -- same
approach `scripts/abba_transcript_driver.py` and `scripts/serve_gate.py` already
use for their own scripted transcripts), then send one more turn whose new
content is padded/trimmed to exactly S tokens. Repeat per S value at each of the
4 depths, once per path (`ILI_METAL_PREFILL=0` then `=1`, two separate
`ILI_LAYER_CAPTURE_DIR`s, e.g. `captures-cpu/` and `captures-metal/`). This
reuses existing, already-tested transcript machinery instead of writing a new
driver -- no new script was needed for this beyond the comparison tool itself.

**Honesty note (matching `scripts/long_ctx_profile.py`'s own instrumentation-
honesty precedent):** captures at T=128/1K happen during what is functionally
still prefill-dominated context; T=16K exercises the depth this campaign's
daily-driver work actually cares about (`docs/roadmap-daily-driver.md`). S=1
captures a decode step; S>1 captures a (possibly chunked, see
`ILI_PREFILL_CHUNK`) prefill step. Nothing here claims S/T are independent in
their engine cost -- only that the grid is a reasonable, deliberately broad
sampling of both.

## The comparison tool

`tools/compare_layer_captures.py` -- loads two captures (single-pair `--a/--b`,
or a whole directory pair with `--dir-a/--dir-b`, matching files by name across
the grid above) and reports, per layer:

- **max-abs** difference (worst single element, aggregated as the worst
  position's worst element across the S rows)
- **RMS** difference and **normalized RMS** (RMS diff / RMS of the reference,
  conventionally the CPU/byte-exact path)
- **cosine similarity** (per position, mean across positions, and once over the
  whole flattened `[S, D]` tensor)
- **top-margin delta**: a PROXY metric, not a real output-token logit margin --
  the capture is an intermediate activation, not `lm_head` output. Per position,
  sort the D-dim vector descending and take `value[0] - value[1]`; report how
  much that gap shrinks (or grows) between the reference and comparison capture,
  worst case across positions. A shrinking margin means whatever downstream
  argmax-like decision depends on this vector got measurably closer to flipping
  -- the honest, stated purpose of borrowing "logit margin" terminology for an
  activation vector.

Unit-tested on synthetic captures (`tests/test_compare_layer_captures.py`):
round-trip fidelity, format-error rejection (bad magic, truncated payload),
identical/perturbed/orthogonal-vector metric sanity checks, the top-margin-delta
sign convention, shape-mismatch rejection, and batch directory pairing
(including files present in only one side).

## What this does NOT claim

- Not a verdict. This is instrumentation + a comparison tool; the actual 120-file
  grid has not been captured (no engine tonight). Running it and reading the
  results is tomorrow's job, gated on the quality bench finishing.
- `top_margin` is a proxy on captured activations, not real decode-time logits;
  see above.
- The M5-Max lab engine variant (`glm_m5max.c`, `gen_m5max_engine.py`'s scratch-
  buffer-reuse transforms) is a SEPARATE numerical-validation question from
  CPU-vs-Metal-prefill; this hook happens to apply cleanly to either source file
  (see "The patch" above) but this plan only scopes the CPU/Metal comparison on
  the stock engine.
