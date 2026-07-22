<p align="center">
  <img src="assets/iliria-logo.png" width="500" alt="iliria — tiny engine, immense model">
</p>

**Tiny engine, immense model.** Run **GLM-5.2 (744B-parameter MoE)** as a local agent and daily driver — in pure C, with zero dependencies, by streaming experts from disk. Primary target: an M5 Max MacBook Pro with 128 GB of unified memory.

**Part of a two-tier local-inference family:** iliria is the deep-reasoning tier — its siblings are **[trailbrake](https://github.com/PhilipJohnBasile/trailbrake)** (the fast MLX tier) and **[racecontrol](https://github.com/PhilipJohnBasile/racecontrol)** (the OpenAI-compatible router that pairs them). Its Apple **Foundation Models** integration — the engines as FM providers, and Apple's on-device/Private Cloud Compute models as router tiers — is **[iliria-fm](https://github.com/PhilipJohnBasile/iliria-fm)**.


## Project status (2026-07-15)

iliria (a fork of JustVugg/colibri — see [NOTICE](NOTICE)) runs GLM-5.2 744B at
**~1.5 tok/s warm decode** on an M5 Max 128 GB, with **prefill now GPU-accelerated 6.3–9.1×**
(a 1,500-token turn's TTFT dropped from 13–25 minutes to ~2–3 minutes;
`ILI_METAL_PREFILL=1` is the shipped default). Most performance claims pass a hash-gated,
frozen-state, 3-held-out-prompt confirmation matrix (`c/ab-m5max-k6-matrix.sh`) before they ship
— the Metal-prefill 6.3–9.1× figure above is the noted exception: a real before/after
measurement, but not yet run through that same matrix (see docs/PERFORMANCE_THEORY.md).
Verdicts so far: PILOT K6 prefetch **reverted** (−6.96%), smarter eviction **dead** (−26%, zero
hit change), Metal 4 **+2%** (provisional), DSA sparse indexer **opt-in only** (disqualifies the
Metal prefill kernel — combined they're 4.9× *slower*), MTP / static pinning / hotset training /
hetero CPU MoE / cross-request expert self-speculation all measured dead; **inter-expert read
coalescing / profile-guided expert layout dead at this block size** (each expert is already ONE
coalesced ~19 MB pread issued concurrently — merging the residual 0.22% scale preads measured
**1.33% of decode wall-clock**, NO-GO vs the ±5% bar; see `c/bench-m5max/step0-iokind-diag/`). The cache-policy
frontier is closed: capacity vs reuse distance is the wall. The first `ili bench` quality
baseline has now run: **62.5% mean acc_norm** (HellaSwag/ARC/MMLU, n=40/task — small-n, see
"Quality benchmark" below). The saliency-aware int4/int2 mixed-precision expert container's
first build failed its long-generation quality gate and was retired (**dead**, see "Roadmap:
agent daily driver" below); a repaired quantizer exists but has not been rebuilt into a
container. Current
plan: [docs/roadmap-daily-driver.md](docs/roadmap-daily-driver.md). CLI is `ili`; the `coli`
symlink is gone, but legacy `COLI_`/`FA_` environment-variable aliases are still accepted for
compatibility.

## The idea

A 744B Mixture-of-Experts model activates only ~40B parameters per token — and only ~11 GB of those change from token to token (the routed experts). So:

- the **dense part** (attention, shared experts, embeddings — ~17B params) stays **resident in RAM at int4** (~9.9 GB);
- the **19,456 routed experts** (75 main-path MoE layers × 256 + the MTP layer's 256; ~19 MB each at int4, per config.json: 78 layers, first 3 dense) live **on disk** (~370 GB) and are **streamed on demand**, with a per-layer LRU cache, a learned pinned hot-store, and the OS page cache as a free L2.

The engine is centered on a single C file (`c/glm.c`) plus small headers. No BLAS,
no Python at runtime, and no GPU is required. The opt-in Metal backend (and a CUDA
backend on Linux) accelerates supported paths without changing the streamed
whole-model design. The model itself is never modified — no fine-tuning, expert
merging, or router recalibration — but the engine is free to reorganize *how* the
same computation runs (kernel choice, quantization tier, sparsity the model already
defines) as long as it clears the quality baseline (`ili bench`) before shipping.

## Quick start

```bash
cd c
./setup.sh                      # checks gcc/OpenMP, builds, self-tests

# chat — RAM budget, expert cache and MTP are all detected automatically:
ILI_MODEL=/path/to/glm52_i4 ./ili chat
```

On the M5 Max, build and run with the measured defaults instead
(`DRAFT=0`, routed-expert `PIPE=1`, 6 performance-core OpenMP threads,
`RAM_GB=114`, PILOT off):

```bash
cd c
make mac-fast
bash run-m5max-fast.sh /path/to/glm52_i4 run "Your prompt"
```

For sustained runs, plug in AC power and select
[High Power](https://support.apple.com/en-us/101613) in System Settings →
Battery (`caffeinate` keeps the Mac awake but does not select that thermal
policy).

Other useful commands (all subcommands of `c/ili`):

```bash
ILI_MODEL=/path/to/glm52_i4 ./ili plan     # Disk/RAM/VRAM placement plan (headers only)
ILI_MODEL=/path/to/glm52_i4 ./ili doctor   # read-only readiness check, --json for automation
ILI_MODEL=/path/to/glm52_i4 ./ili info     # model, RAM, disk, configuration status
./ili run "prompt"                          # one-shot generation
./ili bench                                 # quality benchmarks (MMLU/HellaSwag/ARC)
```

`ili plan` reads only safetensors headers and reports the exact dense/expert
footprint, RAM reserve, safe expert-cache cap, and bounded VRAM hot tier;
`--auto-tier` applies the same plan to `chat`, `run`, and `serve`. `ili doctor`
validates the model directory, tokenizer, headers, engine binary, and placement
budget without starting the engine (warnings exit 0, blockers exit 1).

### Download the model

A pre-converted **GLM-5.2 int4** model for iliria is available on Hugging Face — **use the version with the int8 MTP heads**:

**https://huggingface.co/philipjohnbasile/GLM-5.2-colibri-int4-with-int8-mtp**

(Mirror: [mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp](https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp) — same int8 MTP heads.)

> ⚠️ **The MTP head must be int8.** The original mirror ([jlnsrk/GLM-5.2-colibri-int4](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4)) ships **int4** MTP heads, which give **0% draft acceptance** — speculation silently never engages ([#8](https://github.com/JustVugg/colibri/issues/8), [#102](https://github.com/JustVugg/colibri/issues/102)). Check what you have: `ls -l <model>/out-mtp-*` — int8 (correct): `3527131672 / 5366238584 / 1065950496`; int4: `1765523544 / 2686077736 / 536747200` (replace just those three files from the int8 mirror). Note that on the M5 Max streamed path MTP is measured slower end-to-end and defaults off (`DRAFT=0`) — the int8 head matters on hardware where verification isn't I/O-bound, and keeps the option honest everywhere.

```bash
ILI_MODEL=/path/to/GLM-5.2-colibri-int4-with-int8-mtp ./ili chat
```

Alternatively, convert from the official FP8 release yourself — one shard at a
time, the full 756 GB never needs to exist on disk at once, resumable
(conversion is the only step that needs Python: `pip install torch safetensors
huggingface_hub numpy`):

```bash
./ili convert --model /nvme/glm52_i4     # ~400 GB free on a local NVMe path
```

## Honest numbers (M5 Max, 128 GB unified memory)

| metric | value |
|---|---|
| model on disk (int4 container) | ~370 GB |
| resident RAM (dense, int4) | 9.9 GB |
| load time | ~30 s |
| warm decode (confirmation matrix, 3 held-out coding prompts) | **1.42–1.58 tok/s** |
| typical warm state | ~101 GB RSS · 48.9 GB learned hot store (2,584 experts) · LRU 34/layer |
| non-disk floor (attention + matmul + overhead) | ~0.36 s/token → decode caps ~2.8 tok/s even at 100% hit |
| prefill throughput, pre-kernel → post-kernel (measured) | ~1.6 tok/s → **~9–11 tok/s** (6.3–9.1× TTFT) |
| 1,500-token turn TTFT, pre-kernel → post-kernel (measured) | 821–1,523 s → **130–168 s** |
| 20-turn agent session (~22K ctx), reuse-only (this afternoon) | ~5–7 h |
| 20-turn agent session, reuse + prefill kernel (tonight) | **~1.5–2 h** |
| roadmap target | ≤60 min/session · ~2.0–2.2 tok/s decode |

This is a 744B-parameter model answering correctly on a laptop, not a claim about how it
ranks on quality (see "Quality benchmark" below). Decode is
disk-bound cold (~11 GB of expert reads/token) and capacity-bound warm: the
learned hot store and page cache absorb what fits, and the remaining misses are
reuse-distance misses no cache policy can remove (measured — see verdicts
below). The levers that remain are fewer bytes per expert, I/O/compute overlap,
and not re-prefilling context that hasn't changed.

iliria was originally built and validated on a 12-core/25 GB WSL2 dev box
(~0.05–0.1 tok/s cold — the existence proof). That record, community
benchmarks from a dozen machines, the Windows 11 native port, and the CUDA
backend details live in [docs/history.md](docs/history.md).

## Settled verdicts (confirmation matrix, 2026-07-14)

*Full hypothesis → prediction → measurement → status map, with citations for every number: [docs/PERFORMANCE_THEORY.md](docs/PERFORMANCE_THEORY.md).*

> **K6 verdict:** the controlled 3-held-out-prompt confirmation matrix
> (`c/bench-m5max/k6-matrix-20260714-090059`, frozen usage profile, hash-gated)
> measured **−6.96% median paired throughput** for PILOT K6 lookahead vs off,
> despite +13 hit-rate points — attention contention (+4.8–6.3 s) and the layer
> barrier outweigh the disk savings. Decision gate: **REVERT DEFAULT**; PILOT
> stays opt-in. The prediction itself is at the Belady ceiling — hit rate was
> never the problem; wall time is the objective.

| candidate | matrix result | verdict |
|---|---:|---|
| PILOT K6 router-lookahead prefetch | −6.96% | **reverted**, opt-in only |
| prefetch-aware / smarter eviction | −26%, zero hit change | **dead** — cache-policy frontier closed |
| resident-sketch self-speculative decoding | miss bytes flat in K; net negative w/ RAM charged | **dead** |
| Metal 4 MoE submission | +2.05% | provisional; sub-threshold solo, rides in the stack |
| persistent Metal state | +0.00% | no perf effect; kept for lifecycle-safety infra value |
| **Metal prefill attention (S>4)** | **+6.3–9.1× TTFT**, decode unchanged | **shipped, default on** |
| DSA sparse indexer, combined with the Metal prefill kernel | 4.9× *slower* (CPU-only path) | **opt-in only**; wins standalone CPU prefill >~36K ctx |
| MTP speculative decoding (streamed path) | 1.49 → 0.37 tok/s at D2 | **dead** on M5 Max; `DRAFT=0` default |
| static pinning / hotset training | flat/diffuse | **dead** |
| heterogeneous CPU MoE tail | loss | **dead**, off by default |
| continuous batching (2 concurrent streams) | 1.95× disk traffic vs 1 stream | **dead** — expert overlap barely above chance |

Capacity vs reuse distance is the wall: no replacement policy or predictor buys
wall time on this workload. What's left is the roadmap.

## Roadmap: agent daily driver

The plan is [docs/roadmap-daily-driver.md](docs/roadmap-daily-driver.md), gated step by step.
**Done:** `ili serve` with prefix KV reuse (delta-only prefill, verified exact; `.ili_kv`
warm-restart resume is **6.8× faster than cold**, a certified same-prompt ABBA result — an
earlier N=1 figure of 142× was retired as uncitable, since that cold/resume comparison used
different prompts, see `c/bench-m5max/ili-kv-resume-abba/`), the Metal prefill kernel (shipped,
above), and the first `ili bench` quality baseline (62.5% mean acc_norm, n=40/task — see
"Quality benchmark" below). **Dead:** the saliency-aware int4/int2 mixed-precision expert
container sized by measured per-expert quantization error (predicted +21% decode from offline
simulation; early/salient experts would have stayed int4) — the first build (a defective
quantizer, 280 GB manifest) failed its long-generation collapse gate and was retired; a repaired
quantizer exists but has not been rebuilt into a container, and nothing shipped ever changed the
model's weights. **Queued:** `--topp` adaptive expert routing, arrival-order MoE overlap, and a
final "fight" —
every surviving lever stacked, matrix-tested together, and ablated one at a time to catch
interactions — before anything becomes the shipped default. Realistic endgame: ~2.0–2.2 tok/s
decode and ~45–70 minute sessions. 10–15 tok/s interactive is unreachable on this hardware and
is not the bar a daily driver needs.

## Benchmarking: the confirmation matrix is the house standard

Single-prompt A/Bs are how K6 got called "promising" before losing 7% on
held-out prompts. Any performance claim on the M5 Max path must pass:

```bash
cd c
bash ab-m5max-k6-matrix.sh /path/to/model            # off vs PILOT K6
ILI_K6_AXIS=metal4 bash ab-m5max-k6-matrix.sh /path/to/model   # off vs Metal 4
ILI_K6_AXIS=pstate bash ab-m5max-k6-matrix.sh /path/to/model   # persistent Metal state
```

The matrix runs 3 held-out coding prompts × warm/cold cache × 3 interleaved
ABBA trials per cell, snapshots and restores the persistent `.fa_usage` usage
histogram before every run (a drifting hot-set changes expert placement, whose
kernel-family rounding can flip greedy tokens), gates every run on the exact
output-token hash, and reports median paired throughput. The promotion bar is
**+5% median paired throughput with byte-identical output**; anything inside
±5% is noise. Results land in `c/bench-m5max/` (local artifacts, not
committed). Discipline: AC power, High Power Mode, GPU-heavy apps closed.

The older single-axis `ab-m5max-*.sh` scripts and `make mac-ab-*` targets
still run but are superseded — see
[docs/history.md](docs/history.md#superseded-single-axis-ab-scripts).

## What's implemented

- **Faithful GLM-5.2 (`glm_moe_dsa`) forward** — token-exact against a `transformers` oracle (TF 32/32, greedy 20/20).
- **MLA attention** (q/kv-LoRA, interleaved partial RoPE) with compressed KV-cache: 576 floats/token, 57× smaller.
- **DeepSeek-V3-style sigmoid router** (noaux_tc, routed scaling), shared expert, first-3-dense layers.
- **DSA sparse attention** — GLM-5.2's lightning indexer, per-layer top-2048 causal key selection; validated exact against dense.
- **MLA weight absorption** for decode — no per-token k/v reconstruction; validated exact.
- **Streaming expert tier** — per-layer LRU auto-sized from `MemAvailable`, learned hot-store (`.fa_usage`) auto-pinned at startup, async readahead, batch-union MoE for prefill.
- **Quantization kernels** — int8 / packed int4 / packed int2, per-row scales, AVX2 and Apple SDOT paths, integer-dot routing decided per shape by measurement.
- **Metal backend (opt-in)** — batched routed-expert SwiGLU zero-copy from RAM slabs, fused decode attention, prefill GEMMs; lifecycle-safe persistent residency state (slab generations, in-flight refs); every GPU path falls back to CPU per-block on fault. `make glm METAL=1`, `ILI_METAL=1`.
- **Grouped CPU MoE** — ragged descriptor over pinned/LRU slabs, gate+up fused over one quantized activation (`ILI_CPU_GROUPED_MOE=1`).
- **Native MTP speculative decoding** — GLM-5.2's own draft head, batched verification, rejection sampling under sampling. Off by default on the streamed path (measured slower); requires the int8 head to engage at all. Full story: [docs/history.md](docs/history.md#mtp-speculative-decoding-the-full-story).
- **Grammar-forced speculative drafts** (`GRAMMAR=file.gbnf`) — single-legal-byte spans injected as pre-accepted drafts for JSON/function-calling workloads; verification makes a wrong grammar harmless.
- **KV-cache persistence** — conversations reopen warm across restarts (`.ili_kv`, ~182 KB/token, crash-safe, byte-identical to an uninterrupted session; `KVSAVE=0` disables).
- **True sampling** — temperature + nucleus, defaults 0.7/0.90 tuned for int4 reality.
- **Byte-level BPE tokenizer in C** (GPT-2-style with Unicode-property regex, 320k merges).
- **RAM safety** — honest peak projection (working set, KV, MTP row, buffers) to keep the OOM-killer from firing in normal operation.
- **Offline FP8→int4 converter** — one shard at a time, resumable (`ili convert`).
- **Resource planning** — `ili plan` / `ili doctor`, versioned JSON, shared by CLI, server, and UIs.
- **Route-trace + cache-simulation lab** — ordered route traces, exact LRU/Belady replay, pin sweeps ([docs/m5max-route-trace-cache-sim.md](docs/m5max-route-trace-cache-sim.md)). This is the tooling that closed the cache frontier.
- **CUDA backend (opt-in, Linux)** — resident-tensor and VRAM hot-expert tier; details in [docs/history.md](docs/history.md#experimental-resident-cuda-backend-linux).

Runtime knobs (env or flags): `--temp`, `--topp` (adaptive expert top-p, 30–40% less disk — quality A/B pending, roadmap Step 6), `--ngen`, `--repin N` (live tier adaptation at turn boundaries), `AUTOPIN=0`, `THINK=1` (reasoning block), `DRAFT=n`, `GRAMMAR=g.gbnf`, `PIPE`, `PILOT=1` (opt-in, see verdicts), `DSA=0`, `KVSAVE=0`.

## Serving: OpenAI-compatible API

`ili serve` keeps one model process loaded and exposes a text-only
OpenAI-compatible HTTP API (`/v1/models`, `/v1/chat/completions`, legacy
`/v1/completions`, SSE streaming, usage counts; gateway is Python stdlib only,
inference stays in the C engine). This is the substrate for the roadmap's
agent-harness step.

```bash
cd c
ILI_MODEL=/nvme/glm52_i4 ILI_API_KEY=local-secret ./ili serve \
  --host 127.0.0.1 --port 8000 --model-id glm-5.2-iliria
```

- One generation at a time: a bounded FIFO admission queue (`--max-queue`,
  `--queue-timeout`) returns OpenAI-shaped 429s instead of pretending to run
  unsafe parallel sequences. `GET /health` exposes queue counters.
- `--kv-slots N` (≤16) allocates independent sequence contexts with crash-safe
  persistence files; requests select one with `cache_slot`, or the server picks
  the slot with the longest known prompt prefix. The engine verifies the exact
  token prefix before reusing KV rows. Start small: the roadmap uses 1–2 slots
  (4 slots would take ~29 GB from the expert cache).
- `enable_thinking: true` (or `reasoning_effort`) enables the reasoning block.
  Tools, images, logprobs, and penalties return explicit errors rather than
  being silently ignored.
- Localhost by default; set `ILI_API_KEY` before exposing it. CORS is
  preconfigured for the Vite/Tauri dev origins; add exact origins with
  `--cors-origin`.

On the M5 Max reference box, `c/run-m5max-serve.sh MODEL_DIR` launches serve
with the measured tuning from `run-m5max-fast.sh` plus the roadmap Step 1
KV-reuse defaults (2 slots × 40960-token context, `.ili_kv` persistence).
Agent-harness contract: keep the serve process alive, send monotonic
(append-only) message history, and the engine prefills only each turn's new
tokens. Response-length discipline (roadmap Step 2, all already supported
per request — no server changes needed):

- keep thinking **off** (`ILI_THINK=0`, the default; clients can still opt in
  per request with `enable_thinking`/`reasoning_effort`);
- send per-request `max_tokens`: ~200 for tool-call/short-answer turns, ~400
  for code edits, ~700 only for from-scratch generation (server default is 256);
- system-prompt snippet that shortens replies: *"Reply with unified diffs or
  changed hunks only, never full files. No preamble, no recap, no restating
  the request."*

`c/scripts/serve_gate.py` replays a scripted 10-turn agent transcript through
the API and verifies the gate (delta-only prefill, unchanged decode tok/s).

A browser UI (React + TypeScript, a pure API client) and a Tauri desktop
shell around it exist as separate components and are not part of this
release. The terminal `ili chat` remains the first-class interface.

## Quality benchmark

We had never measured how much int4 costs in accuracy until the first baseline run
(`ili bench`: HellaSwag, ARC-Challenge, MMLU, EleutherAI-style log-likelihood scoring), which
has now run: **62.5% mean acc_norm across the three tasks (n=40/task)**. n=40 is small — this
project's own accounting flags it as too small to distinguish a real effect from noise beyond
the roughest sanity check, and a planned n=100 rerun has not yet produced a committed result.
This baseline becomes the permanent quality gate every future output-changing lever (the
mixed-precision container, `--topp`) must clear before shipping. Prior GLM-5.2 quantization
work says what to expect and what to gate on (length-dependent collapse, saliency-tiered
bit-width):
[docs/experiments/glm52-quantization-and-finetuning-findings.md](docs/experiments/glm52-quantization-and-finetuning-findings.md).

```bash
cd c
./ili bench                                   # hellaswag, arc_challenge, mmlu — 40 questions each
./ili bench hellaswag --limit 200             # one task, more questions
./ili bench mmlu arc_challenge --ram 100      # pick tasks, set a RAM budget
```

If you run this — or run iliria on interesting hardware — **please open an
issue with your numbers**. Every datapoint moves the ceiling; the historical
community-benchmark table is in [docs/history.md](docs/history.md#community-benchmarks-measured-upstream-era).

## Repo layout

```
Makefile                  root build/check entry point
c/
├── glm.c                 single-file GLM engine (reference implementation)
├── st.h, tok.h, json.h   runtime headers
├── backend_metal.mm      opt-in Metal backend
├── backend_cuda.*        opt-in CUDA tier (Linux)
├── ili                  user-facing CLI
├── openai_server.py      OpenAI-compatible HTTP gateway
├── ab-m5max-k6-matrix.sh confirmation-matrix benchmark harness
├── run-m5max-fast.sh     M5 Max helper with the measured defaults
├── tools/                offline conversion, fixtures, trace/cache simulators
├── scripts/              long-running conversion helpers
└── tests/                dependency-free C and Python tests
docs/                     roadmap, experiment records, history
```

Generated M5 Max artifacts (`c/glm_m5max.c`, `c/backend_metal_m5max.mm`, test
binaries, `c/bench-m5max/`) are local build/measurement outputs: the sources of
truth are the generators and patch scripts under `c/tools/` and their tests.
The runtime path intentionally stays flat and readable: `glm.c` plus its small
headers; Python and shell tooling is never a runtime dependency of the engine.

From the repository root, `make`, `make check`, and `make clean` delegate to
the engine Makefile.

## License

Apache 2.0. iliria is a fork of
[JustVugg/colibri](https://github.com/JustVugg/colibri) — see
[NOTICE](NOTICE) for attribution. GLM-5.2 weights are released by Z.ai under
MIT.
