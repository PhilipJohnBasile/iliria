# iliria history — origins, ports, and superseded experiments

This file preserves measurement records and engineering history that no longer
belong in the front-page README. Nothing here is deleted knowledge: these are
real numbers from real machines, plus experiment write-ups whose conclusions
have since been superseded by the M5 Max confirmation matrix
(`c/ab-m5max-k6-matrix.sh`) and the current plan in
[roadmap-daily-driver.md](roadmap-daily-driver.md).

Where a section is superseded, the superseding verdict is stated at the top of
the section. The measurements themselves remain valid records of what was
observed on that hardware, on that date.

---

## Origins: the WSL2 dev box

iliria's engine (as colibri) was originally written and validated on
deliberately humble hardware: a 12-core laptop with 25 GB of RAM and an older
DRAM-less NVMe behind a WSL2 VHDX that measured ~1 GB/s random reads on that
drive. (WSL2 VHDX is not inherently slow: a community 5090 box measured
10.5 GB/s O_DIRECT through one, upstream
[#101](https://github.com/JustVugg/colibri/issues/101).) Every constraint of
that machine is a knob a better machine can turn up.

### Honest numbers (WSL2, 12 cores, 25 GB RAM, NVMe via VHDX)

| metric | value |
|---|---|
| model on disk (int4 container) | ~370 GB |
| resident RAM (dense, int4) | 9.9 GB |
| load time | ~30 s |
| peak RSS during chat | ~20 GB (auto-capped) |
| cold decode cost | ~11 GB disk reads/token (75 layers × 8 experts) |
| disk ceiling (this dev box's drive) | ~1 GB/s → ~0.05–0.1 tok/s cold |
| MTP speculation (int8 head) | 2.2–2.8 tok/forward measured ([#8](https://github.com/JustVugg/colibri/issues/8)) |

This was never fast. It was a 744B frontier-class model answering correctly on
a machine that costs less than one H100 fan — the proof that the streaming
design works at all. The project's performance work has since moved to the
M5 Max (see the README).

### Back-of-envelope hardware predictions (historical)

Decode is disk-bound cold: a cold token costs ~11.4 GB of expert reads; RAM
turns cold reads into free cache hits.

| machine | expected |
|---|---|
| the WSL2 dev box (~1 GB/s, 25 GB RAM) | ~0.05–0.1 tok/s cold — proven baseline |
| native Linux, PCIe4 NVMe (~3–5 GB/s random), 32 GB | ~0.5–1 tok/s |
| PCIe5 NVMe or 2×NVMe RAID0 (~8–12 GB/s), 64 GB (PIN ~40 GB) | ~2–4 tok/s |
| 128–256 GB RAM, 12 cores (hot experts cached) | ~2–4 tok/s — matmul-bound |
| same RAM + 24–32 cores, or AVX-512/VNNI kernels | ~5–15 tok/s — kernel work is the multiplier |

These were estimates. The community benchmarks below and the M5 Max record in
the README are the measurements.

### How to test a new machine (historical procedure, still valid)

```bash
cd c && ./setup.sh                 # build + architecture self-test (expects 32/32)

# 1) measure YOUR disk the way the engine uses it (parallel 19 MB random reads):
gcc -O2 -fopenmp iobench.c -o iobench
./iobench /path/to/glm52_i4/out-00069.safetensors 19 64 8 0   # buffered, 8 threads
./iobench /path/to/glm52_i4/out-00069.safetensors 19 64 8 1   # O_DIRECT (bypass cache)
# Caveat (upstream #86): iobench reads a bounded ~1 GB shard, so buffered reads on a
# big-RAM box report the PAGE CACHE, not the disk. Use the O_DIRECT run for a true
# number, on a shard you haven't touched this session. On macOS there is no O_DIRECT —
# iobench uses F_NOCACHE, which can't evict already-resident pages: reboot or use a
# fresh shard for a real cold read.

# 2) chat; watch the per-turn stats line (tok/s, expert hit-rate, RSS):
COLI_MODEL=/path/to/glm52_i4 ./ili chat

# 3) record expert usage, then pin the hottest experts in your spare RAM:
STATS=stats.txt ./ili chat
PIN=stats.txt PIN_GB=20 ./ili chat        # scale PIN_GB to your free RAM
```

### SSD wear note

Cold starts are heavy on random reads (~11 GB/token), but reads don't
meaningfully wear an SSD — iliria's streaming is read-only. The real concerns
under heavy use are (1) swap traffic if the system runs out of RAM (writes do
wear the drive; the auto-budget is designed to stay clear of swap) and
(2) sustained thermals: hours at full read duty cycle will heat cheaper drives.

## Community benchmarks (measured, upstream-era)

Real numbers from real machines, stock build (`setup.sh`, gcc 13), greedy
decoding, `--ngen 32`, MTP active unless noted:

| machine | disk (iobench, 19 MB × 64, 8 threads) | config | measured |
|---|---|---|---|
| Intel Core Ultra 7 270K Plus (24 threads) · WSL2 · 24 GB RAM · NVMe VHDX ([#2](https://github.com/JustVugg/colibri/issues/2)) | 1.96 GB/s buffered · 2.74 GB/s O_DIRECT | default | 0.07 tok/s · expert hit 3–4% · RSS 14.1 GB |
| 〃 | 〃 | `--topp 0.7` | **0.11 tok/s** · expert hit 11% · RSS 14.7 GB |
| Apple M5 Max (18 cores) · macOS · 128 GB unified · internal SSD ([#4](https://github.com/JustVugg/colibri/issues/4), [#5](https://github.com/JustVugg/colibri/issues/5)) | ~4 GB/s cold (the 14.2 GB/s reading was cache-influenced) | default, MTP off | **1.06 tok/s** · expert hit 23% · RSS 21.8 GB |
| Apple M5 Max · macOS · 128 GB unified · 2 TB SSD · **Metal backend** ([#72](https://github.com/JustVugg/colibri/pull/72), [#87](https://github.com/JustVugg/colibri/issues/87)) | (macOS O_DIRECT figure unreliable) | Metal on · `--ram 96` · 39.7 GB warm pin · MTP off | **1.83 tok/s** · expert hit 66% · warmed 1.11 → 1.83 over the run |
| 〃 · 46.9 GB pin (2.94M-selection history) · `--ram 110`, 1024-token run ([#103](https://github.com/JustVugg/colibri/issues/103)) | 〃 | Metal on (experts + attention) · MTP off | **2.06 tok/s** · hit 72.5% · coherent output (pre-rebase Metal branch) |
| Epyc 9654 ES · Linux · 4x16GB DDR5-4800-rdimm · Samsung PCIe Gen3 x4 NVMe | — | `MTP=1 DIRECT=1` | 0.31 tok/s · expert hit 35% · RSS 21.52 GB |
| Ryzen AI 9 HX 370 (Framework 13) · Arch Linux · 128 GB · WD SN850X, BTRFS zstd ([#12](https://github.com/JustVugg/colibri/issues/12)) | — | int8 MTP head · `--cap 32` · 46.7 GB auto-learned PIN | **0.37 tok/s** · expert hit 66% · MTP acceptance 52% (2.59 tok/fw) · RSS 105 GB |
| Ryzen 9 9950X (32 threads) · Linux · 123 GB · Crucial P3 QLC Gen3 ([#31](https://github.com/JustVugg/colibri/issues/31)) | 1.51 GB/s buffered | default, 2 runs from cold | 0.10 tok/s · hit 53% · profile 66% disk |
| 〃 same machine, model moved to a Samsung 9100 PRO PCIe 5.0 ([#31](https://github.com/JustVugg/colibri/issues/31)) | **8.81 GB/s** O_DIRECT | 〃 (usage history retained) | **0.28 tok/s** · hit 57% · profile flips: 32% disk / **57% matmul** |
| Ryzen AI Max+ 395 (Framework Desktop) · Ubuntu · 128 GB LPDDR5x · Intel Optane 905p PCIe 3.0 ([#39](https://github.com/JustVugg/colibri/issues/39)) | 3.27 GB/s buffered | int8 MTP head · fresh history (pure LRU, auto-raised cap 65) | 0.16 tok/s · hit 57% · profile 49% disk / 47% matmul |
| 〃 five runs later — learned pin 47.6 GB ([#39](https://github.com/JustVugg/colibri/issues/39)) | 〃 | `--temp 0.7 --topp 0.7` | **0.40 tok/s** · hit 71% · fastest non-Apple datapoint |
| Ryzen 7 9800X3D (16T) · WSL2 · 70 GB RAM · Samsung 9100 PRO PCIe 5.0 · RTX 5090 ([#101](https://github.com/JustVugg/colibri/issues/101)) | **10.51 GB/s** O_DIRECT | MTP off · learned pin 24 GB · hit 54% · OMP hot-team on | **0.41 tok/s** · disk-bound (36.5 s disk vs 24.0 s matmul) · **CUDA expert tier ≈ 0%** (AVX-512 CPU matches the 5090) · `--topp 0.7` → **0.52 tok/s** |
| EPYC 7443 (24C/48T, Zen3 AVX2) · Linux · **430 GB RAM** · NVMe RAID-Z1 via TrueNAS VM ([#104](https://github.com/JustVugg/colibri/issues/104)) | ~1 GB/s (VM overhead) | 77.5 GB pin · cap auto-raised to 194/layer · MTP off | **1.00 tok/s** · **hit 98%** · disk eliminated → RAM-bandwidth + matmul bound (no AVX-512/VNNI on Zen3) |

Takeaways from this era: with 24 GB of RAM the engine auto-caps the expert
cache to 2 slots/layer, so decode stays cold even on a fast disk — on
small-RAM machines the RAM cap, not the disk, is the binding constraint.
`--topp 0.7` alone bought a clean 1.6× end-to-end speedup. The Framework 13
rows are the cache thesis proven end-to-end on one machine: 0.29 → 0.37 tok/s
just by giving the cache its RAM (the cap part became automatic on
2026-07-10 — benchmarks recorded before that date ran RAM-capped and should be
rerun). The 9950X pair is the cleanest bottleneck experiment: same machine,
same history, only the disk swapped — ×5.8 disk bandwidth bought ×2.9 tokens
and the profile flipped from 66% disk to 57% matmul. But the crossover depends
on the CPU kernel: the 9800X3D row shows that with OMP hot-team tuning on,
AVX-512 CPU matmul is fast enough that even a 10 GB/s NVMe stays disk-bound —
and there the CUDA expert tier buys ≈ 0%, because the CPU already matches the
5090 on expert matmul. (Honest correction from #101: an earlier version of
that report ran with OMP tuning off, which manufactured a false matmul-bound
crossover and a false +14% for CUDA — neither survived a clean re-run.)

### Early Metal backend datapoint (M4 Max)

Measured on an M4 Max (128 GB, warm cache, MTP on) when the Metal backend first
landed: CPU 0.30 → Metal **0.42 tok/s (~1.4×)**, best config adding `DIRECT=1`
(~3× vs that machine's first cold run). Design points that still hold: Metal's
~5 ms submit latency makes per-matmul dispatch a loss — everything is batched
into few command buffers per layer, and resident experts' GPU work is submitted
*before* the missed experts' disk reads so I/O and compute overlap.
`COLI_METAL_GEMM_MIN` tunes the prefill GEMM row threshold (default 16). Every
GPU path falls back to the CPU per-block on any fault; numerics are
dequant→f32-MAC, deterministic for a fixed backend/configuration but not
byte-identical across kernel families (see the MTP/#100 story below).

## MTP speculative decoding: the full story

> **Current verdict:** MTP is measured dead on the streamed M5 Max path
> (`DRAFT=0` 1.49 tok/s vs `DRAFT=2` 0.37 tok/s vs `DRAFT=4` 0.28 tok/s in a
> controlled whole-model A/B). The M5 helper defaults to `DRAFT=0`. The record
> below explains the mechanism and remains relevant to anyone running iliria
> on hardware where verification is not I/O-bound.

GLM-5.2 ships its own multi-token-prediction head (layer 78) that drafts
tokens the main model verifies in one batched forward. The head must be int8
(the converter does this by default): at int4, draft acceptance collapses to
0–4% and speculation never engages; at int8 it's 39–59% acceptance,
2.2–2.8 tokens/forward (community-measured,
[#8](https://github.com/JustVugg/colibri/issues/8)).

MTP is lossless *in exact arithmetic* — but not byte-identical to
non-speculative greedy in practice
([#100](https://github.com/JustVugg/colibri/issues/100)). This isn't
MTP-specific: iliria's quantized integer kernels are shape-dependent, so any
batched (S>1) or GPU forward rounds slightly differently from the single-token
path, and int4 GLM-5.2 sits close enough to argmax ties that such a rounding
change can flip a token. MTP, the CUDA expert tier, and batched prefill are
three different ways to trip the same sensitivity (community-confirmed in
#100: swapping only the kernel family forks greedy output on 3/5 prompts, with
zero speculation). Every emitted token is still the argmax of a *valid*
forward — the continuation stays correct — it just isn't the same stream. For
byte-exact reproducibility: `DRAFT=0` (no speculation), plus `IDOT=0
COLI_CUDA=0` if you also want kernel-family/GPU independence. Under sampling,
rejection sampling keeps the distribution correct.

Why speculation loses on a streamed MoE: tokens/forward is not the metric that
matters when verification loads the *union* of routed experts. On a cold cache
each verified draft routes to extra experts (~660 → ~1100 expert-loads/token),
so speculation is a net time loss until the cache/pin warms — and on the
M5 Max it never catches up.

A related lineage trap from separate quantization work: changing trunk
precision breaks an existing MTP head's draft acceptance (a distinct mechanism
from the int4-head issue). Re-verify acceptance after any precision change.

## Windows 11 native port (Phase 1 complete, parked)

iliria builds and runs natively on Windows 11 x86-64 with MinGW-w64. The port
adds a `_WIN32` compatibility layer in `c/compat.h` that maps POSIX I/O to the
Windows API (pread → ReadFile+OVERLAPPED, posix_fadvise no-op, aligned
allocation, MoveFileEx rename, GlobalMemoryStatusEx RAM detection). All
platform differences stay in `compat.h`; the engine source is unchanged.

**Toolchain:** GCC via [winlibs](https://winlibs.com/) or MSYS2 MinGW-w64.
Tested with GCC 16.1.0 (x86_64-ucrt-posix-seh).

```powershell
# One-time toolchain install (pick one):
scoop install mingw-winlibs                    # portable, no shell needed
# or: pacman -S mingw-w64-x86_64-gcc make     # via MSYS2

# Build (from c/ directory):
make glm.exe            # GLM-5.2 engine (static, no DLL dependencies)
make olmoe.exe          # OLMoE engine (same shims)
make iobench.exe        # disk I/O benchmark
make test-c             # run C tests
make test-python        # run Python tests (requires python)

# Verify (tiny model, 2.4 MB):
pip install torch transformers safetensors huggingface_hub
python tools/make_glm_oracle.py                # generate tiny oracle
SNAP=./glm_tiny TF=1 ./glm.exe 64 16 16        # expect "32/32 positions"

# Run with real model:
SNAP=D:\glm52_i4 ./glm.exe 64 4 16             # batch inference
python ili chat --model D:\glm52_i4           # interactive chat
python ili serve --model D:\glm52_i4          # OpenAI-compatible API
```

**Status:** Phase 1 complete (compiles, correct, static-linked). O_DIRECT
(Phase 2), GPU via `LoadLibrary` on `coli_cuda.dll` (Phases G0–G2), and
full-model validation are separate, currently unscheduled workstreams. (The
detailed the Windows-port plan referenced by earlier READMEs was an upstream
working document and is not part of this repository.)

## Experimental resident CUDA backend (Linux)

> **Current verdict:** not part of the M5 Max daily-driver path. The measured
> record (upstream #101) shows the CUDA expert tier buys ≈ 0% when the CPU
> matmul is fast — the GPU tier earns its VRAM only when the CPU is the weak
> link. Kept working and opt-in.

iliria includes an opt-in CUDA backend for model-resident tensors. Streaming
experts deliberately remain on the original CPU path: copying an expert from
NVMe to the GPU on every use would only replace the disk bottleneck with a
PCIe bottleneck. Resident quantized tensors are uploaded lazily once and
reused.

```bash
cd c
make cuda-test CUDA=1                  # q8/q4/q2/f32 kernel correctness
make CUDA=1
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=/nvme/glm52_i4 ./glm 64 4 4
```

Requirements: Linux, an NVIDIA driver, and a CUDA Toolkit under
`/usr/local/cuda` (override with `CUDA_HOME=/path/to/cuda`).
`CUDA_ARCH=native` builds for the GPU in the current machine. Requesting CUDA
with a CPU-only binary, an invalid device, or an unavailable runtime fails at
startup instead of silently falling back.

CUDA defaults to an expert-only accelerator: resident dense/attention tensors
stay on CPU because fixture measurements show that moving them does not help
while expert I/O is the bottleneck. `CUDA_DENSE=1` keeps the earlier
all-resident experimental path. A measured `PIN` profile can promote its
hottest experts into a persistent VRAM tier:

```bash
STATS=stats.txt SNAP=/nvme/glm52_i4 ./glm 64 4 4   # collect routing frequencies first
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=16 \
PIN=stats.txt PIN_GB=160 SNAP=/nvme/glm52_i4 ./glm 64 4 4
# multi-GPU expert tier, 96 GB total budget across six devices
COLI_CUDA=1 COLI_GPUS=0,1,2,3,4,5 CUDA_EXPERT_GB=96 \
PIN=stats.txt PIN_GB=160 SNAP=/nvme/glm52_i4 ./glm 64 4 4
```

Selected experts are uploaded during startup, so capacity failures occur
before inference. The budget is clamped against free VRAM after reserving the
projected dense resident set and 2 GB of runtime headroom per device. With
`COLI_GPUS`, `CUDA_EXPERT_GB` is a total budget across the device set. Devices
use independent contexts and synchronous host-staged activation copies — no
P2P/NCCL. The kernels are correctness-first custom kernels, not
cuBLAS/Tensor-Core kernels.

For a reproducible backend A/B without the full checkpoint, generate the
deterministic 313M-parameter `glm_moe_dsa` fixture and run fixed-token replay:

```bash
cd c
python tools/make_glm_bench_model.py --output /nvme/iliria-bench-medium --device cuda
python tools/benchmark_cuda_fixture.py --model /nvme/iliria-bench-medium --gpu 0
```

The fixture has random weights and is not a language model; it exists only to
preserve the real MLA/MoE/streaming shapes for controlled comparison.

The bigger GPU experiment — full expert residency across 6× RTX 5090 — is
recorded in
[experiments/glm52-6x5090-2026-07-12.md](experiments/glm52-6x5090-2026-07-12.md)
(6.84 tok/s single-request decode with disk removed from the decode path).

## The July 2026 M5 Max engineering roadmap (superseded)

> **Superseded 2026-07-14** by the confirmation-matrix verdicts and
> [roadmap-daily-driver.md](roadmap-daily-driver.md). The measurements below
> remain the record; the *recommendations* below do not. Specifically:
> PILOT K6 measured **−6.96%** in the matrix and was reverted; smarter/
> prefetch-aware eviction measured **−26%** with zero hit-rate change (the
> cache-policy frontier is closed — capacity vs reuse distance is the wall);
> Metal 4 with persistent state measured **+2%** (provisional, infra value
> only); workload hotset training measured flat/diffuse and is dead; the fused
> q4 kernel phase was overtaken by the mixed-precision container plan.

### The pre-matrix stable configuration

The best repeatable single-stream setup measured on the M5 Max (128 GB) was
approximately **1.71 tok/s** on a frozen 112-token coding prompt
("Review a Godot 4.7 GDScript controller for bugs and allocations.",
output-token hash `562050532423d626`):

```text
RAM_GB=114
DRAFT=0
PIPE=1, PIPE_WORKERS=8
OMP_NUM_THREADS=6
AUTOPIN=1
PILOT_REAL=0
COLI_METAL4_MOE=0
-O3 -mcpu=native -flto=thin -fno-math-errno
```

Typical state: ~101 GB RSS, a 48.9 GB learned hot store of 2,584 experts, and
an auto-sized LRU cap of 34 experts/layer. (The matrix later replaced this
single-prompt number with the 3-held-out-prompt warm range of 1.42–1.58 tok/s
— the same configuration, honestly averaged.)

### Single-axis A/B results (frozen prompt, pre-matrix)

| Experiment | Frozen result | Decision then | Decision now |
|---|---:|---|---|
| PIPE off | median 1.585 tok/s | slower | unchanged |
| PIPE on | median 1.710 tok/s, same hash | keep; +7.9% | unchanged |
| OpenMP 6 threads | median 1.710 tok/s | keep | unchanged |
| OpenMP 12 threads | median 1.695 tok/s | slower | unchanged |
| OpenMP 18 threads | median 1.665 tok/s | slower | unchanged |
| RAM 114 GB | about 1.72 tok/s | keep | unchanged |
| RAM 118 GB | about 1.69 tok/s | slower (page-cache headroom) | unchanged |
| MTP D0 | 1.49 tok/s | keep | unchanged |
| MTP D2 | 0.37 tok/s | slower | dead |
| MTP D4 | 0.28 tok/s, divergent hash | slower | dead |
| legacy Metal submission | median 1.705 tok/s | keep | unchanged |
| Metal 4 prototype | median 1.570 tok/s, same hash | setup-bound | +2% with persistent state (provisional) |
| PILOT_REAL off | median 1.690 tok/s, 67.5% hit | production default | confirmed default |
| PILOT_REAL K2 | median 1.525 tok/s, 72.5% hit | slower | dead |
| PILOT_REAL K4 | median 1.645 tok/s, 77.0% hit | slower | dead |
| PILOT_REAL K6 | median 1.735 tok/s, 80.8% hit | "promising" | **reverted: matrix measured −6.96%** |

The K6 row is the whole argument for the confirmation matrix: a single frozen
prompt showed +1.5% and 13 extra hit-rate points, and the controlled
3-held-out-prompt matrix (`c/bench-m5max/k6-matrix-20260714-090059`, frozen
usage profile, hash-gated) showed **−6.96% median paired throughput** —
attention contention (+4.8–6.3 s) and the layer barrier outweigh the disk
savings. Hit rate is not the objective; wall time is.

Benchmark receipts live under `c/bench-m5max/` in the dated `pipe-ab`,
`omp-ab`, `metal4-ab`, `mtp-ab`, `pilot-ab`, and `k6-matrix` directories. They
are local measurement artifacts rather than source files.

### Why legacy Metal beat the first Metal 4 prototype

The first Metal 4 prototype improved GPU wall time but rebuilt residency and
argument state for almost every MoE block:

| Path | End-to-end | Metal setup | GPU wall | GPU kernel |
|---|---:|---:|---:|---:|
| legacy | 65.43 s / 1.71 tok/s | 0.26 s | 10.73 s | 6.70 s |
| Metal 4 | 71.26 s / 1.57 tok/s | 8.02 s | 8.13 s | 7.26 s |

Metal 4 saved ~2.6 s of GPU wall time but paid ~7.8 s of avoidable setup —
which made persistent resource state the measured next step. That work landed
(slab generations, in-flight refs — see
[m5max-route-trace-cache-sim.md](m5max-route-trace-cache-sim.md)) and the
matrix measured Metal 4 + persistent state at **+2%** (provisional).

The same Metal 4 run spent 33.37 s in expert disk I/O, 16.30 s in expert
matmul, 17.39 s in attention, 4.18 s elsewhere, and only 0.07 s in scatter —
scatter fusion is not a priority (upstream
[PR #111](https://github.com/JustVugg/colibri/pull/111) also measured a large
regression from moving it to the GPU).

### Phase 2 (superseded): workload-specific decayed hotsets

The plan was per-workload usage profiles (`COLI_HOTSET_PROFILE=coding`,
`.coli_usage.coding` / `.fa_usage.coding`), deterministic fixed-point decay
(`COLI_HOTSET_DECAY`, `COLI_HOTSET_DECAY_INTERVAL`), and startup-only pin
re-ranking. The infrastructure exists (`c/train-m5max-hotset.sh`, profile
files, `c/tier.h` helpers) and the matrix uses the frozen `coding` profile as
its controlled state — but as a *speedup*, hotset training measured
flat/diffuse and is dead. The routing distribution is too diffuse for a
specialized pin ranking to beat the cumulative histogram.

### Phase 3 (superseded): fused q4 expert kernels

The plan was a `moe_pair_swiglu_q4` Metal kernel (one SIMDgroup/output row,
fused gate+up+SiLU, fewer barriers), then an all-resident `down_sum8_q4`, with
canonical route-slot reduction to keep floating-point order independent of
cache timing. [DwarfStar's kernels](https://github.com/antirez/ds4/blob/main/metal/moe.metal)
validated the structural approach (its Q4_K/top-6 specifics don't transfer to
iliria's row-q4/top-8 path). Overtaken by the roadmap's mixed-precision
container: fewer bytes beats faster math while decode is I/O-dominated.

### PILOT lifecycle blockers (context for the K6 revert)

`PILOT_REAL` had six identified correctness/lifecycle blockers (unlocked
`ESlot.eid` publication, residency-dependent accumulation order, discarded
worker handle, unbounded speculative-pread waits, missing generation tags on
queue entries, REPIN interaction). The slab lifecycle part was fixed properly
(generations + in-flight refs, merged); the rest became moot for the default
path when the matrix reverted K6. The prediction itself was measured at the
Belady ceiling — prediction quality was never the problem; wall-time cost was.

### Workload-level lessons that fed the current roadmap

1. **Prefix reuse for agentic coding.** A growing 40K context measured ~200 s
   to prefill uncached vs ~5 s cached — up to 9.7× faster TTFT. This became
   Step 1 of [roadmap-daily-driver.md](roadmap-daily-driver.md).
2. **Continuous batching for aggregate throughput.** One expert load can serve
   several sequences: reference measurements rose 15.8 → 27.1 → 34.6 → 41.1
   total tok/s at batch 1/2/4/8. A multi-client throughput win, not lower
   single-conversation latency; the server still serializes generation.

Other transferable lessons: preserve GPU wired/page-cache/KV headroom;
evaluate long and adjacent real workloads rather than trusting one green hash;
verify the built tensor format rather than review assumptions; never reduce
top-k or precision silently to cross a quality cliff.

### Benchmark discipline (absorbed into the matrix)

Before recording A/B results: AC power, High Power Mode, close GPU-heavy apps.
`caffeinate` cannot prevent emergency low-battery hibernation (one early
confirmation accidentally included a 5.5-hour sleep at 1% battery). Record the
exact environment, cold/warm runs, ABBA order, median throughput, output hash,
profile breakdown, cache hit and expert loads, Metal setup/GPU/kernel/scatter
timers, RSS, and faults/fallbacks. All of this is now automated by
`c/ab-m5max-k6-matrix.sh`, which is the house standard (see the README).

## Superseded single-axis A/B scripts

The dated single-axis helpers under `c/` (`ab-m5max-pipe.sh`, `ab-m5max-omp.sh`,
`ab-m5max-pilot.sh`, `ab-m5max-mtp.sh`, `ab-m5max-metal4.sh`,
`ab-m5max-backend.sh`, `ab-m5max-cpu-moe.sh`, `ab-m5max-pin-sweep.sh`, and the
`make mac-ab-*` targets that wrap them) produced the record above. They still
run, but they measure one frozen prompt and are exactly the methodology that
called K6 "promising." For any performance claim, use
`c/ab-m5max-k6-matrix.sh` instead.
