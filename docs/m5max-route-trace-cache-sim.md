# M5 Max ordered route trace and cache lab

> **Status (2026-07-14):** this lab did its job. The cache-simulation side
> answered its question — LRU is already near the Belady offline optimum for
> this workload, and the live eviction-veto A/B measured **−26% with zero
> hit-rate change**, closing the cache-policy frontier (see the
> [README verdicts](../README.md#settled-verdicts-confirmation-matrix-2026-07-14)).
> The persistent Metal-state prototype has since been hardened and **merged**
> (lifecycle-safe: slab generations, in-flight refs); the confirmation matrix
> measured Metal 4 + persistent state at **+2%** (provisional). The tooling
> below remains current for future trace studies.

This lab separates three questions that cumulative `.fa_usage` data cannot answer:

1. How much temporal locality does the routed-expert stream contain?
2. What is the best static-pin/per-layer-LRU split for a fixed RAM budget?
3. How much of the remaining miss rate is avoidable by a better online predictor?

Every runtime feature in this document is opt-in. The normal `make mac-fast` generated output is unchanged.

## Instrumented build

```bash
cd c
bash build-m5max-lab.sh
```

This runs the production M5 generators and then applies:

- `tools/patch_m5max_route_trace.py` to the generated C engine;
- `tools/patch_m5max_persistent_state.py` after the existing Metal 4 transform.

The lab build still defaults to ordinary execution. No trace file is opened and no persistent Metal state is used unless the corresponding environment variable is set.

The helper explicitly compiles Objective-C++ as C++17 because Apple Clang can otherwise default Objective-C++ to a mode that rejects the backend's existing raw-string Metal shader.

## Ordered route-trace format

Enable tracing:

```bash
ILI_ROUTE_TRACE=/tmp/routes.bin \
  bash run-m5max-fast.sh /path/to/model run "prompt"
```

Optional controls:

```text
ILI_ROUTE_TRACE_MAX_EVENTS=N   stop after N selected routes
ILI_ROUTE_TRACE_SYNC=1         flush after every MoE call for crash diagnosis
```

### Version 2 header

The file begins with a 40-byte little-endian header:

```text
magic[8] = FAROUTE1
version:u32 = 2
record_size:u32 = 24
expert_bytes:u64       exact q4 expert bytes from expert_bytes_probe()
cache_units:u32        q4-equivalent LRU layer units used by cap_for_ram()
lru_per_layer:u32      live m->ecap after RAM admission
pinned_units:u32       q4-equivalent pinned experts already resident
flags:u32              bit 0 = MTP expert tier present
```

The total live expert allowance is therefore exact for the captured run:

```text
total q4-equivalent slots = pinned_units + cache_units × lru_per_layer
```

The simulator still reads older version 1 traces, but those require a fallback RAM estimate or an explicit `--total-expert-slots` value.

### Route records

Each 24-byte record is:

```text
event_id:u64
moe_call_id:u64
layer:u16
batch_row:u16
route_rank:u16
expert_id:u16
```

The writer records selected routes in router order. The engine subsequently forms a first-occurrence union of experts across all rows in the MoE call. The simulator performs the same deduplication before cache lookup, so one expert selected by several MTP/prefill rows counts as one expert load request.

## One-command route study

```bash
bash run-m5max-route-study.sh /path/to/model
```

The harness:

1. builds the instrumented engine;
2. preserves and later restores the existing `.fa_usage` file;
3. captures a deterministic 112-token trace using the stable M5 settings;
4. reads the exact live expert byte size and pin/LRU slot allowance from the trace;
5. simulates static pin sizes `0 8 12 16 20 24 32 48` GB;
6. writes text and CSV reports under `c/bench-m5max/route-study-*`.

### Units

The engine uses decimal units:

```text
PIN_GB=1 means 1,000,000,000 bytes
expert size is reported in decimal MB
```

The simulator uses the same convention. Treating runtime GB values as GiB would overstate pin capacity by about 7.4%.

## Offline simulator

Direct use:

```bash
python3 tools/simulate_m5max_cache.py routes.bin \
  --usage /path/to/model/.fa_usage \
  --pin-gb 0 8 12 16 20 24 32 48 \
  --tokens 112 \
  --csv cache.csv
```

Trace v2 supplies the preferred cache budget automatically. Fallback or diagnostic overrides are available:

```text
--expert-budget-gb 97.5     fallback decimal GB for a v1 trace
--expert-mb 18.916          override exact/fallback expert size
--total-expert-slots N      override total q4-equivalent pin+LRU slots
--lru-per-layer N           hold dynamic capacity fixed
```

### Static policies

`--pin-policy global` matches `pin_load()`: it globally ranks complete `(layer, expert)` tensors by usage count and truncates the ranking by byte budget. Expert ID 17 in layer 3 is distinct from expert ID 17 in layer 40.

`--pin-policy per-layer` is an experimental comparison that allocates an equal base number of pins to each represented layer, then uses global frequency for remaining slots.

### Dynamic policies

For each pin size the simulator derives per-layer LRU capacity from the remaining expert slot budget, unless `--lru-per-layer` is supplied.

It reports:

- static pin hits;
- per-layer LRU hits;
- misses;
- total hit rate;
- Belady offline-optimal hit rate for the same pins and LRU capacity.

The difference between LRU and the offline optimum is an upper bound on the replacement-policy headroom. It is not a predicted performance gain: page cache, parallel I/O, Metal overlap, and memory pressure still affect wall time.

The default study sets `DRAFT=0`, which also prevents the int8 MTP expert tier from loading. The simulator rejects a trace whose header reports MTP because q4 and int8 expert slots require a layer-size-aware pin-selection model.

## Real downward pin/LRU sweep

> **Note:** the `ab-m5max-pin-sweep.sh` on-hardware sweep harness is not included in this release; the offline simulator above answers the same pin/LRU question from a captured trace.

```bash
bash ab-m5max-pin-sweep.sh /path/to/model
```

Defaults:

```text
lru auto 8 12 16 20 24 32 GB
112 generated tokens
one rotated pass
RAM=114 GB
PIPE=1 / 8 workers
OMP=6
MTP, pilot, repin, Metal 4 disabled
```

Useful overrides:

```bash
ILI_PIN_SWEEP_PASSES=3 \
ILI_PIN_SWEEP_VALUES="lru auto 8 12 16 20 24 32" \
  bash ab-m5max-pin-sweep.sh /path/to/model
```

The harness restores the frozen usage file before every case and at exit. It reports throughput, hit rate, disk/expert/attention timers, detected pin size, LRU capacity, and output hashes. It does not change `.m5max-profile.env`.

## Persistent Metal-state prototype

The existing Metal 4 experiment already maintains command allocators, argument tables, and per-slot residency sets. Its first implementation accumulated expert resources separately in both slots and had a fixed 390-buffer tracking array.

The new prototype adds a process-wide generation-tagged residency snapshot:

```bash
ILI_M5_LAB_METAL4=1 bash build-m5max-lab.sh

ILI_METAL4_MOE=1 \
ILI_METAL_PERSISTENT_STATE=1 \
  bash run-m5max-fast.sh /path/to/model run "prompt"
```

When enabled:

- all currently registered slabs are committed into one global `MTLResidencySet`;
- the set is rebuilt only when `g_slab_gen` changes;
- each MoE slot retains a small set containing only `xg`, `gg`, `uu`, `hh`, and metadata;
- prior snapshots retire into a bounded list (`ILI_SNAP_RETIRE_MAX`) that is swept
  when the command buffers referencing them complete; at capacity, refreshes fail
  cleanly into the per-slot residency fallback until the list drains;
- snapshot creation failure falls back to the original per-slot residency code.

The prototype is guarded by all of the following:

1. a build with `ILI_M5_LAB_METAL4=1`;
2. Metal 4 symbols available in the SDK;
3. macOS runtime support;
4. `ILI_METAL4_MOE=1`;
5. `ILI_METAL_PERSISTENT_STATE=1`.

It has since been tested on the target macOS 27 machine (matrix result: +2%,
provisional) and the lifecycle work below is merged; the guards remain so that
CI on macOS 14 — which validates the transformations and legacy compile path
but cannot compile or execute the guarded Metal 4 API body — stays green.

### Slab lifecycle (in-flight references, deferred unregister)

Retaining an `MTLBuffer` or residency object does not by itself prove that externally owned `newBufferWithBytesNoCopy` backing memory cannot be freed or reused while an older command buffer is in flight. With `ILI_METAL_PERSISTENT_STATE=1` the transform therefore adds an explicit lifecycle, active on both the legacy-Metal and Metal 4 MoE submission paths (it does not require a Metal 4 build):

- every `ili_metal_register` receives a monotonically increasing generation and a
  `SlabLife` node holding an extra strong reference to the slab's no-copy wrap;
- before a MoE command buffer is committed, one in-flight reference is taken on
  every distinct non-pinned slab it reads; a completion handler on that command
  buffer releases the references (any failure between acquire and commit releases
  them immediately and falls back to the non-persistent path);
- `ili_metal_unregister` on a slab with live in-flight references defers the
  actual release into a bounded retirement queue drained by completion handlers,
  and blocks until the references reach zero — the engine frees or reuses the
  backing immediately after unregister returns, so returning earlier would be a
  use-after-free. The wait is bounded because completed handlers always fire for
  committed command buffers;
- `ili_metal_register_pinned` marks a slab as process-lifetime pinned: it skips
  per-command-buffer refcounting entirely (immutable fast path) and its wrap is
  released only at shutdown. Nothing is pinned implicitly;
- `ili_metal_shutdown` waits for the retirement queue to drain before tearing
  the device down (no deadlock: the wait is completion-driven).

`make metal-test-lifecycle` builds `backend_metal_m5max_lab.mm` (the full lab
transform chain) and runs `tests/test_backend_metal_lifecycle.mm` under
`MallocScribble=1`, with the lifecycle enabled and disabled: register/unregister
plus reallocation churn against in-flight async MoE dispatches, a deterministic
unregister-while-in-flight case, fallback and pinned-path checks, and a watchdog
that fails the run if shutdown deadlocks. `make metal-test-lifecycle-asan` runs
the same binary under AddressSanitizer.

The attention/layer/gemm paths resolve dense weights and KV caches that the
engine never evicts mid-decode; they remain outside the refcounting scope.

## Promotion requirements

> **Superseded (2026-07-14):** the cache-policy line of work is closed — the
> live A/B measured −26% with zero hit change; capacity vs reuse distance is
> the wall. Persistent Metal state met its checklist and is merged. New
> candidates go through the confirmation matrix (`c/ab-m5max-k6-matrix.sh`),
> not this checklist.

Route tracing may be merged once the trace parser and generated engine remain stable and trace-disabled performance is unchanged.

A cache policy should be promoted only when offline predictions correlate with multiple real A/B points.

Persistent Metal state should be promoted only when it has:

- the same deterministic hash as the corresponding fixed Metal 4 configuration;
- explicit safe backing-memory lifetime across unregister/reallocation;
- no GPU faults or fallback growth;
- no unbounded residency snapshot refreshes;
- lower setup and end-to-end median time;
- stable memory use across long sessions and cache-slot reuse.
