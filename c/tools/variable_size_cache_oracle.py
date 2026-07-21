#!/usr/bin/env python3
"""Trace-only cache-frontier studies: water-filling, variable-size oracle, extents.

Zero shard reads: inputs are the committed ordered route traces, a .fa_usage
histogram (pin ranking), and the 280 GB allocation manifest. Three studies:

WATERFILL (Task: layer-capacity water-filling)
  Per-layer miss-byte curves B_l(C_l) under byte-LRU at UNIFORM int4 sizes,
  computed exactly for every capacity in one pass via Mattson stack
  distances (LRU inclusion property). A resource-allocation DP over
  fixed byte increments (one int4 expert = the only useful step under
  uniform sizes) solves  min sum_l B_l(C_l)  s.t.  sum_l C_l = C_total,
  where C_total equals the engine's current uniform allocation
  (cache_units x lru_per_layer slots from the trace header). Reported
  against equal-per-layer allocation (the engine's current uniform ecap),
  in-sample per trace, plus a pooled allocation (fitted on the sum of all
  traces) evaluated per trace -- the deployable variant.

ORACLE (Task: variable-size offline bound)
  Byte-LRU vs size-aware online policies vs an offline bound proxy, over
  HETEROGENEOUS sizes: int4 18,915,328 B for manifest int4 experts,
  repaired-int2 (groupwise g128/f16 sizing, expert_bytes_group_scales ->
  10,027,008 B, the '~10.1 MB' repaired tier) for the manifest's int2 set.
  Policies per layer at the RAM-matched byte budget (lru_per_layer x int4):
    - byte-LRU (B_current; D-case accounting);
    - size-normalized reuse distance (SNRD): victim maximizes
      size x staleness;
    - GreedyDual-Size, cost = size (provably identical to byte-LRU when
      miss cost is proportional to size -- kept as a cross-check) and
      cost = 1 (optimizes miss COUNT, not bytes; reference only);
    - Belady-by-bytes-saved greedy (offline): at each miss the incoming
      item joins the candidate set and farthest-next-use items are evicted
      until the byte budget holds (an item with no future use is a free
      eviction; the incoming item itself may be bypassed). NON-OPTIMAL:
      offline caching with variable sizes is NP-hard (Chrobak et al. 2012);
      this greedy ignores packing effects (FOO/PFOO, Berger et al. 2018,
      construct tighter flow-based bounds), so it is a strong proxy for
      OPT, not OPT itself. With miss cost proportional to bytes, per-byte
      eviction value reduces to next-use distance, which is exactly what
      this greedy ranks by.
  KILL LINE: (B_LRU - B_oracle)/B_LRU < 2% -> the cache-policy frontier
  stays closed under heterogeneity.

EXTENTS (Task: co-activation hypergraph layout)
  Per layer, cluster the 256 experts into 64 MB extents (3 int4 experts
  per extent) by co-selection frequency: greedy agglomerative merging on
  the co-activation matrix, then first-fit-decreasing packing into
  ceil(256/3) extents. Reports extents-touched-per-layer-selection and
  the theoretical preadv count (1 preadv per touched extent; today's
  engine issues 1 coalesced pread PER missed expert) vs the id-ordered
  layout, on the fit trace (in-sample) and held-out traces.
  KILL LINE: <15% extent reduction (held-out, selection-level) = dead.

Pins are filled once per layout with pin_load() semantics (usage-ranked,
byte-truncated) and FROZEN across policies so every comparison sees the
same pinned tier.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


slotsim = _load_module("simulate_m5max_cache", _TOOLS / "simulate_m5max_cache.py")
qc = _load_module("quant_container", _TOOLS / "quant_container.py")

GB = 1e9
INF = float("inf")
EXTENT_BYTES = 64_000_000  # decimal 64 MB, matching the engine's decimal units


# ------------------------------------------------------------------ trace ----
@dataclass
class Call:
    call_id: int
    layer: int
    experts: list
    n_rows: int


def load_calls(path: Path):
    """-> (metadata, [Call]) in strict trace order; first-occurrence union
    per MoE call (the engine resolves each unique expert once per call)."""
    calls: list[Call] = []
    current: Call | None = None
    rows: set = set()
    with path.open("rb") as f:
        metadata = slotsim.read_trace_header(f, path)
        for rec in slotsim._iter_records(f, path):
            if current is None or rec.call != current.call_id:
                if current is not None:
                    current.n_rows = len(rows)
                    calls.append(current)
                current = Call(rec.call, rec.layer, [], 0)
                rows = set()
            if rec.layer != current.layer:
                raise ValueError(f"{path}: call {rec.call} spans layers")
            rows.add(rec.row)
            if rec.expert not in current.experts:
                current.experts.append(rec.expert)
        if current is not None:
            current.n_rows = len(rows)
            calls.append(current)
    return metadata, calls


def per_layer_streams(calls) -> dict:
    """{layer: [expert, ...]} preserving request order (batch-union)."""
    streams: dict[int, list[int]] = defaultdict(list)
    for call in calls:
        streams[call.layer].extend(call.experts)
    return dict(streams)


# -------------------------------------------------------------------- pins ----
def fill_pins(usage: dict, sizes: dict, pin_bytes: int):
    """pin_load() semantics: rank (layer, expert) by usage count (ties:
    layer, expert), stop at the first pair that no longer fits."""
    ranked = sorted(usage, key=lambda p: (-usage[p], p[0], p[1]))
    pinned, used = {}, 0
    for pair in ranked:
        size = sizes.get(pair)
        if size is None:
            continue
        if used + size > pin_bytes:
            break
        pinned[pair] = size
        used += size
    return pinned, used


# ----------------------------------------------------- WATERFILL: mattson ----
def mattson_stack_distances(stream: list):
    """LRU stack distance per request (INF = first touch), one pass.

    Sequential (per-request) semantics, matching simulate_m5max_cache's
    slot LRU. Uniform sizes: byte-LRU at c*item_bytes == item-LRU at c."""
    stack: list = []
    out = []
    for x in stream:
        try:
            i = stack.index(x)
        except ValueError:
            out.append(INF)
            stack.insert(0, x)
            continue
        out.append(i + 1)
        del stack[i]
        stack.insert(0, x)
    return out


def mattson_miss_curve(stream: list, max_cap: int) -> list:
    """Exact LRU miss counts for every capacity 0..max_cap (LRU inclusion:
    a request at stack distance d hits iff capacity >= d)."""
    hist = [0] * (max_cap + 2)
    for d in mattson_stack_distances(stream):
        if not math.isinf(d) and d <= max_cap:
            hist[int(d)] += 1
    n = len(stream)
    misses = [n] * (max_cap + 1)
    hits = 0
    for c in range(1, max_cap + 1):
        hits += hist[c]
        misses[c] = n - hits
    return misses


def call_distance_lists(calls, layer: int, pinned: dict):
    """Per-call stack-distance lists for one layer (pinned filtered out).

    Every call of the layer yields one list (possibly empty when all its
    experts are pinned) so per-call miss distributions keep call weights."""
    layer_calls = [c for c in calls if c.layer == layer]
    flat = []
    bounds = []
    for c in layer_calls:
        kept = [e for e in c.experts if (layer, e) not in pinned]
        bounds.append(len(kept))
        flat.extend(kept)
    dists = mattson_stack_distances(flat)
    out = []
    pos = 0
    for n in bounds:
        out.append(dists[pos:pos + n])
        pos += n
    return out


def cvar_curves(call_dists: list, max_cap: int, item_bytes: int, alpha=0.95):
    """-> (mean_bytes_per_call[c], cvar_bytes_per_call[c]) for c=0..max_cap.

    Pointer sweep over capacities: a request at finite stack distance d
    stops missing once c >= d; INF requests always miss (compulsory)."""
    import numpy as np

    n_calls = len(call_dists)
    if n_calls == 0:
        z = [0.0] * (max_cap + 1)
        return z, list(z)
    counts = np.array([len(d) for d in call_dists], dtype=np.int64)
    finite = []  # (distance, call_idx)
    for j, dl in enumerate(call_dists):
        for d in dl:
            if not math.isinf(d) and d <= max_cap:
                finite.append((int(d), j))
    finite.sort()
    k_tail = max(1, math.ceil((1 - alpha) * n_calls))
    means, cvars = [], []
    p = 0
    for c in range(max_cap + 1):
        while p < len(finite) and finite[p][0] <= c:
            counts[finite[p][1]] -= 1
            p += 1
        mean = counts.sum() / n_calls * item_bytes
        top = np.partition(counts, n_calls - k_tail)[n_calls - k_tail:]
        cvars.append(float(top.mean()) * item_bytes)
        means.append(float(mean))
    return means, cvars


def dp_allocate(curves: dict, total_units: int, max_per_layer: int = 256):
    """min sum_l B_l(C_l) s.t. sum C_l = total_units, C_l <= max_per_layer.

    curves: {layer: [miss_bytes_at_cap_0, ..., miss_bytes_at_cap_K]}.
    Returns (allocation {layer: units}, total_miss_bytes).
    """
    layers = sorted(curves)
    dp = [0.0] + [INF] * total_units
    choices: list = []
    for layer in layers:
        curve = curves[layer]
        kmax = min(len(curve) - 1, max_per_layer)
        new = [INF] * (total_units + 1)
        choice = [0] * (total_units + 1)
        for u in range(total_units + 1):
            best, bk = INF, 0
            for k in range(0, min(kmax, u) + 1):
                prev = dp[u - k]
                if prev == INF:
                    continue
                v = prev + curve[k]
                if v < best:
                    best, bk = v, k
            new[u] = best
            choice[u] = bk
        dp = new
        choices.append(choice)
    allocation = {}
    u = total_units
    for layer, choice in zip(reversed(layers), reversed(choices)):
        k = choice[u]
        allocation[layer] = k
        u -= k
    if u != 0:
        raise AssertionError("DP failed to spend the full budget")
    return allocation, dp[total_units]


def eval_allocation(curves: dict, allocation: dict) -> float:
    total = 0.0
    for layer, curve in curves.items():
        k = min(allocation.get(layer, 0), len(curve) - 1)
        total += curve[k]
    return total


# ------------------------------------------------------- ORACLE: policies ----
def replay_byte_lru(stream, sizes_of, budget):
    """-> (misses, miss_bytes). Classic per-layer byte-LRU."""
    od: OrderedDict = OrderedDict()
    used = 0
    misses = miss_bytes = 0
    for x in stream:
        if x in od:
            od.move_to_end(x)
            continue
        sz = sizes_of(x)
        misses += 1
        miss_bytes += sz
        if sz > budget:
            continue
        while od and used + sz > budget:
            _, v = od.popitem(last=False)
            used -= v
        od[x] = sz
        used += sz
    return misses, miss_bytes


def replay_snrd(stream, sizes_of, budget):
    """Size-normalized reuse distance: victim maximizes size x staleness."""
    resident: dict = {}  # expert -> [last_t, size]
    used = 0
    misses = miss_bytes = 0
    for t, x in enumerate(stream):
        if x in resident:
            resident[x][0] = t
            continue
        sz = sizes_of(x)
        misses += 1
        miss_bytes += sz
        if sz > budget:
            continue
        while resident and used + sz > budget:
            victim = max(resident, key=lambda e: (t - resident[e][0]) * resident[e][1])
            used -= resident[victim][1]
            del resident[victim]
        resident[x] = [t, sz]
        used += sz
    return misses, miss_bytes


def replay_gds(stream, sizes_of, budget, cost="size"):
    """GreedyDual-Size: H = L + cost/size; evict min H; L = victim's H."""
    import heapq

    heap: list = []  # (H, seq, expert) lazy entries
    entry: dict = {}  # expert -> (H, seq, size)
    used = 0
    L = 0.0
    seq = 0
    misses = miss_bytes = 0

    def credit(sz):
        return (sz if cost == "size" else 1.0) / sz

    for x in stream:
        if x in entry:
            _, _, sz = entry[x]
            h = L + credit(sz)
            seq += 1
            entry[x] = (h, seq, sz)
            heapq.heappush(heap, (h, seq, x))
            continue
        sz = sizes_of(x)
        misses += 1
        miss_bytes += sz
        if sz > budget:
            continue
        while used + sz > budget:
            h, s, e = heapq.heappop(heap)
            cur = entry.get(e)
            if cur is None or cur[1] != s:
                continue  # stale
            L = h
            used -= cur[2]
            del entry[e]
        h = L + credit(sz)
        seq += 1
        entry[x] = (h, seq, sz)
        heapq.heappush(heap, (h, seq, x))
        used += sz
    return misses, miss_bytes


def _next_use_positions(stream):
    n = len(stream)
    next_use = [INF] * n
    last_seen: dict = {}
    for i in range(n - 1, -1, -1):
        x = stream[i]
        next_use[i] = last_seen.get(x, INF)
        last_seen[x] = i
    return next_use


def replay_belady_bytes(stream, sizes_of, budget):
    """Offline Belady-by-bytes-saved greedy WITH admission bypass (see
    module docstring: proxy for OPT, non-optimal under variable sizes).
    Bypass makes this a strictly larger policy class than the engine's
    mandatory-insert cache -- generous to the frontier by design."""
    next_use = _next_use_positions(stream)
    resident: dict = {}  # expert -> [next_use_pos, size]
    used = 0
    misses = miss_bytes = 0
    for i, x in enumerate(stream):
        if x in resident:
            resident[x][0] = next_use[i]
            continue
        sz = sizes_of(x)
        misses += 1
        miss_bytes += sz
        if sz > budget:
            continue
        # candidate set includes the incoming item (self-bypass allowed)
        resident[x] = [next_use[i], sz]
        used += sz
        while used > budget:
            victim = max(resident, key=lambda e: (resident[e][0], resident[e][1]))
            used -= resident[victim][1]
            del resident[victim]
    return misses, miss_bytes


def replay_belady_demand(stream, sizes_of, budget):
    """Offline farthest-next-use with MANDATORY insertion (demand paging,
    byte-capacity) -- the engine's own policy class (every missed expert is
    loaded and cached). The LRU-vs-THIS gap isolates pure REPLACEMENT
    headroom; the bypass variant above adds ADMISSION headroom on top."""
    next_use = _next_use_positions(stream)
    resident: dict = {}
    used = 0
    misses = miss_bytes = 0
    for i, x in enumerate(stream):
        if x in resident:
            resident[x][0] = next_use[i]
            continue
        sz = sizes_of(x)
        misses += 1
        miss_bytes += sz
        if sz > budget:
            continue
        while resident and used + sz > budget:
            victim = max(resident, key=lambda e: (resident[e][0], resident[e][1]))
            used -= resident[victim][1]
            del resident[victim]
        resident[x] = [next_use[i], sz]
        used += sz
    return misses, miss_bytes


POLICIES = {
    "byte_lru": replay_byte_lru,
    "snrd": replay_snrd,
    "gds_cost_size": lambda s, z, b: replay_gds(s, z, b, "size"),
    "gds_cost_1": lambda s, z, b: replay_gds(s, z, b, "unit"),
    "belady_demand": replay_belady_demand,
    "belady_bytes": replay_belady_bytes,
}
OFFLINE = ("belady_bytes", "belady_demand")


def run_policies(streams: dict, sizes: dict, pinned: dict, budget: int):
    """Replay every policy per layer; pinned experts never enter the LRU.

    -> {policy: (requests, pin_hits, misses, miss_bytes)}."""
    out = {}
    for name, fn in POLICIES.items():
        req = pin_hits = misses = 0
        miss_bytes = 0
        for layer, stream in streams.items():
            filtered = [e for e in stream if (layer, e) not in pinned]
            req += len(stream)
            pin_hits += len(stream) - len(filtered)
            m, mb = fn(filtered, lambda e, l=layer: sizes[(l, e)], budget)
            misses += m
            miss_bytes += mb
        out[name] = (req, pin_hits, misses, miss_bytes)
    return out


# ----------------------------------------------------------- EXTENTS ----
def coactivation(calls, layer: int, n_experts: int):
    """Symmetric co-selection count matrix for one layer."""
    import numpy as np

    m = np.zeros((n_experts, n_experts), dtype=np.int64)
    for call in calls:
        if call.layer != layer:
            continue
        ex = call.experts
        for i in range(len(ex)):
            for j in range(i + 1, len(ex)):
                m[ex[i], ex[j]] += 1
                m[ex[j], ex[i]] += 1
    return m


def cluster_extents(coact, cap: int):
    """Greedy agglomerative clustering with cluster size cap, then
    first-fit-decreasing packing into extents of `cap` slots.
    -> extent id per expert (list of ints)."""
    import numpy as np

    n = coact.shape[0]
    clusters = [[i] for i in range(n)]

    def weight(a, b):
        return int(coact[np.ix_(clusters[a], clusters[b])].sum())

    active = set(range(len(clusters)))
    merged = True
    while merged:
        merged = False
        best = (0, None, None)
        act = sorted(active)
        for ai in range(len(act)):
            a = act[ai]
            if len(clusters[a]) >= cap:
                continue
            for b in act[ai + 1:]:
                if len(clusters[a]) + len(clusters[b]) > cap:
                    continue
                w = weight(a, b)
                if w > best[0]:
                    best = (w, a, b)
        if best[1] is not None:
            _, a, b = best
            clusters[a].extend(clusters[b])
            clusters[b] = []
            active.discard(b)
            merged = True
    groups = [clusters[i] for i in sorted(active) if clusters[i]]
    # first-fit-decreasing packing into extents of `cap` expert slots
    groups.sort(key=len, reverse=True)
    bins: list[list[int]] = []
    for g in groups:
        for b in bins:
            if len(b) + len(g) <= cap:
                b.extend(g)
                break
        else:
            bins.append(list(g))
    extent_of = [0] * n
    for ei, b in enumerate(bins):
        for e in b:
            extent_of[e] = ei
    return extent_of, len(bins)


def extents_touched(calls, extent_maps: dict, miss_flags=None):
    """Mean distinct extents per layer-selection + totals.

    extent_maps: {layer: extent_of list}. miss_flags: optional
    {call_index: [bool per expert]} restricting to missed experts."""
    total_calls = 0
    total_extents = 0
    total_experts = 0
    for ci, call in enumerate(calls):
        emap = extent_maps[call.layer]
        experts = call.experts
        if miss_flags is not None:
            experts = [e for e, m in zip(call.experts, miss_flags[ci]) if m]
        if not experts:
            continue
        total_calls += 1
        total_experts += len(experts)
        total_extents += len({emap[e] for e in experts})
    mean = total_extents / total_calls if total_calls else 0.0
    return {"calls": total_calls, "total_extents": total_extents,
            "total_experts": total_experts, "mean_extents_per_call": mean}


def miss_flags_byte_lru(calls, sizes, pinned, budget):
    """Per-call miss flags under per-layer byte-LRU (two-phase batch
    semantics: classify against call-entry state, then insert)."""
    lrus: dict = defaultdict(OrderedDict)
    used: dict = defaultdict(int)
    flags = {}
    for ci, call in enumerate(calls):
        layer = call.layer
        od = lrus[layer]
        fl = []
        missing = []
        for e in call.experts:
            if (layer, e) in pinned:
                fl.append(False)
            elif e in od:
                fl.append(False)
                od.move_to_end(e)
            else:
                fl.append(True)
                missing.append(e)
        for e in missing:
            sz = sizes[(layer, e)]
            if sz > budget:
                continue
            while od and used[layer] + sz > budget:
                _, v = od.popitem(last=False)
                used[layer] -= v
            od[e] = sz
            used[layer] += sz
        flags[ci] = fl
    return flags


# ------------------------------------------------------------------ sizes ----
def build_sizes(cfg, manifest_int2, int4_b, int2_b, mix: str) -> dict:
    routed = [(l, e) for l in range(cfg["first_dense"], cfg["n_layers"])
              for e in range(cfg["n_experts"])]
    if mix == "uniform-int4":
        return {p: int4_b for p in routed}
    if mix == "manifest":
        s = {p: int4_b for p in routed}
        for p in manifest_int2:
            s[p] = int2_b
        return s
    raise ValueError(f"unknown mix {mix}")


# ------------------------------------------------------------------- main ----
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--traces", type=Path, nargs="+", required=True)
    p.add_argument("--usage", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--mode", default="waterfill,oracle,extents")
    p.add_argument("--int4-bytes", type=int, default=None)
    p.add_argument("--int2-bytes", type=int, default=None,
                   help="repaired int2 expert bytes (default: groupwise "
                        "g128/f16 sizing = 10,027,008)")
    p.add_argument("--fit-trace", type=Path, default=None,
                   help="extents: trace to fit co-activation on "
                        "(default: first trace)")
    p.add_argument("--warmup-tokens", type=int, default=32)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    manifest = json.loads(a.manifest.read_text())
    cfg = manifest["model_config"]
    manifest_int2 = set(map(tuple, manifest.get("int2", [])))
    int4_b = a.int4_bytes or qc.expert_total_bytes(cfg, 4)
    # repaired groupwise int2 sizing (g128, f16 scales): the '~10.1 MB' tier
    if a.int2_bytes is not None:
        int2_b = a.int2_bytes
    else:
        int2_b = 0
        for proj in qc.PROJS:
            O, I = qc.expert_shape(cfg, proj)
            n_groups = O * ((I + 127) // 128)
            int2_b += qc.packed_nbytes(O, I, 2) + n_groups * 2

    usage = qc.load_usage(str(a.usage))
    modes = {m.strip() for m in a.mode.split(",") if m.strip()}
    a.outdir.mkdir(parents=True, exist_ok=True)

    metas, calls_by_trace = {}, {}
    for t in a.traces:
        meta, calls = load_calls(t)
        if meta.mtp_present:
            raise SystemExit(f"{t}: MTP tier present; unsupported")
        metas[t] = meta
        calls_by_trace[t] = calls
    meta0 = metas[a.traces[0]]
    if meta0.expert_bytes and meta0.expert_bytes != int4_b:
        raise SystemExit(f"trace header expert_bytes {meta0.expert_bytes} != "
                         f"int4 size {int4_b}")
    pin_bytes = meta0.pinned_units * int4_b
    lru_budget = meta0.lru_per_layer * int4_b
    total_units = meta0.cache_units * meta0.lru_per_layer

    print(f"[oracle] int4 {int4_b:,} B | repaired int2 {int2_b:,} B | "
          f"pin budget {pin_bytes/GB:.3f} GB | LRU budget "
          f"{lru_budget:,} B/layer | total {total_units} units")

    summary: dict = {
        "int4_bytes": int4_b, "int2_bytes_repaired": int2_b,
        "pin_bytes": pin_bytes, "lru_budget_per_layer": lru_budget,
        "total_units": total_units,
        "usage": str(a.usage), "manifest": str(a.manifest),
        "traces": {str(t): {"pinned_units": metas[t].pinned_units,
                            "lru_per_layer": metas[t].lru_per_layer}
                   for t in a.traces},
    }

    if "waterfill" in modes:
        summary["waterfill"] = run_waterfill(
            a, cfg, usage, int4_b, pin_bytes, total_units, calls_by_trace)
    if "oracle" in modes:
        summary["oracle"] = run_oracle(
            a, cfg, usage, manifest_int2, int4_b, int2_b, pin_bytes,
            lru_budget, calls_by_trace)
    if "extents" in modes:
        summary["extents"] = run_extents(
            a, cfg, usage, int4_b, pin_bytes, lru_budget, calls_by_trace)

    with (a.outdir / "oracle-summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[oracle] summary: {a.outdir / 'oracle-summary.json'}")
    return 0


# ------------------------------------------------------------ mode runners ----
LAYER_BANDS = (("early", 3, 27), ("mid", 28, 52), ("late", 53, 77))
CVAR_LAMBDAS = (0.0, 0.5, 1.0, 2.0)


def band_of_layer(layer: int) -> str:
    for name, lo, hi in LAYER_BANDS:
        if lo <= layer <= hi:
            return name
    return "other"


def run_waterfill(a, cfg, usage, int4_b, pin_bytes, total_units, calls_by_trace):
    sizes = build_sizes(cfg, set(), int4_b, int4_b, "uniform-int4")
    pinned, pin_used = fill_pins(usage, sizes, pin_bytes)
    print(f"\n[waterfill] pinned {len(pinned)} experts ({pin_used/GB:.2f} GB), "
          f"budget {total_units} units of {int4_b:,} B")

    layers = sorted({c.layer for calls in calls_by_trace.values() for c in calls})
    # per-call stack distances: per trace for mean curves, pooled for CVaR
    dists_by_trace = {
        t: {layer: call_distance_lists(calls, layer, pinned) for layer in layers}
        for t, calls in calls_by_trace.items()}
    curves_by_trace = {}
    for t, per_layer in dists_by_trace.items():
        curves = {}
        for layer, call_dists in per_layer.items():
            hist = [0] * 258
            n = 0
            for dl in call_dists:
                for d in dl:
                    n += 1
                    if not math.isinf(d) and d <= 256:
                        hist[int(d)] += 1
            misses, hits = [n] * 257, 0
            for c in range(1, 257):
                hits += hist[c]
                misses[c] = n - hits
            curves[layer] = [m * int4_b for m in misses]
        curves_by_trace[t] = curves

    pooled = {layer: [sum(curves_by_trace[t][layer][c] for t in curves_by_trace)
                      for c in range(257)] for layer in layers}
    pooled_alloc, _ = dp_allocate(pooled, total_units)

    uniform = total_units // len(layers)
    rows = []
    out = {"per_trace": {}, "uniform_units_per_layer": uniform}
    for t, curves in curves_by_trace.items():
        b_uni = eval_allocation(curves, {l: uniform for l in curves})
        alloc_t, b_dp = dp_allocate(curves, total_units)
        b_pooled = eval_allocation(curves, pooled_alloc)
        name = Path(t).stem
        out["per_trace"][name] = {
            "B_uniform": b_uni, "B_dp_insample": b_dp,
            "B_dp_pooled": b_pooled,
            "delta_insample_pct": 100 * (b_uni - b_dp) / b_uni if b_uni else 0.0,
            "delta_pooled_pct": 100 * (b_uni - b_pooled) / b_uni if b_uni else 0.0,
        }
        r = out["per_trace"][name]
        print(f"  {name:18s} B_uniform {b_uni/GB:8.3f} GB | DP in-sample "
              f"{b_dp/GB:8.3f} GB ({r['delta_insample_pct']:+.3f}%) | pooled "
              f"{b_pooled/GB:8.3f} GB ({r['delta_pooled_pct']:+.3f}%)")
        for layer in sorted(curves):
            rows.append((name, layer, uniform, alloc_t[layer],
                         pooled_alloc.get(layer, 0),
                         curves[layer][min(uniform, len(curves[layer]) - 1)],
                         curves[layer][min(alloc_t[layer], len(curves[layer]) - 1)]))

    with (a.outdir / "waterfill-allocation.csv").open("w") as f:
        f.write("trace,layer,uniform_units,dp_units_insample,dp_units_pooled,"
                "B_uniform_bytes,B_dp_bytes\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    deltas = [v["delta_pooled_pct"] for v in out["per_trace"].values()]
    out["headline_delta_pooled_pct_median"] = sorted(deltas)[len(deltas) // 2]
    ins = [v["delta_insample_pct"] for v in out["per_trace"].values()]
    out["headline_delta_insample_pct_median"] = sorted(ins)[len(ins) // 2]

    # ---- tail-risk variant: min sum_l E[B_l] + lambda*CVaR_0.95(B_l) ----
    # per-call distributions pooled across traces; CVaR over per-call bytes
    pooled_calls = {
        layer: [dl for t in dists_by_trace
                for dl in dists_by_trace[t][layer]]
        for layer in layers}
    mean_c, cvar_c = {}, {}
    for layer in layers:
        mean_c[layer], cvar_c[layer] = cvar_curves(pooled_calls[layer], 256, int4_b)

    cvar_out = {"lambdas": list(CVAR_LAMBDAS), "allocations": {},
                "band_units": {}, "evaluation": {}}
    cvar_rows = []
    for lam in CVAR_LAMBDAS:
        fcurves = {layer: [mean_c[layer][c] + lam * cvar_c[layer][c]
                           for c in range(257)] for layer in layers}
        alloc, _ = dp_allocate(fcurves, total_units)
        e_total = sum(mean_c[l][min(alloc[l], 256)] for l in layers)
        cvar_total = sum(cvar_c[l][min(alloc[l], 256)] for l in layers)
        bands = {b: 0 for b, _, _ in LAYER_BANDS}
        for l, u in alloc.items():
            bands[band_of_layer(l)] += u
        cvar_out["allocations"][str(lam)] = {str(l): alloc[l] for l in sorted(alloc)}
        cvar_out["band_units"][str(lam)] = bands
        cvar_out["evaluation"][str(lam)] = {
            "mean_bytes_per_call_sum": e_total,
            "cvar95_bytes_per_call_sum": cvar_total,
        }
        for l in sorted(alloc):
            cvar_rows.append((lam, l, band_of_layer(l), alloc[l]))
        print(f"  [cvar] lambda {lam:3.1f}: units early/mid/late "
              f"{bands['early']}/{bands['mid']}/{bands['late']} | "
              f"E {e_total/1e6:.2f} MB/call-sum | CVaR95 {cvar_total/1e6:.2f}")
    uniform_bands = {b: 0 for b, _, _ in LAYER_BANDS}
    for l in layers:
        uniform_bands[band_of_layer(l)] += uniform
    cvar_out["band_units"]["uniform"] = uniform_bands
    cvar_out["evaluation"]["uniform"] = {
        "mean_bytes_per_call_sum": sum(mean_c[l][uniform] for l in layers),
        "cvar95_bytes_per_call_sum": sum(cvar_c[l][uniform] for l in layers),
    }
    with (a.outdir / "waterfill-cvar-allocation.csv").open("w") as f:
        f.write("lambda,layer,band,dp_units\n")
        for r in cvar_rows:
            f.write(",".join(map(str, r)) + "\n")
    out["cvar"] = cvar_out
    return out


def run_oracle(a, cfg, usage, manifest_int2, int4_b, int2_b, pin_bytes,
               lru_budget, calls_by_trace):
    out = {}
    rows = []
    for mix in ("manifest", "uniform-int4"):
        sizes = build_sizes(cfg, manifest_int2, int4_b, int2_b, mix)
        pinned, pin_used = fill_pins(usage, sizes, pin_bytes)
        print(f"\n[oracle:{mix}] pinned {len(pinned)} ({pin_used/GB:.2f} GB), "
              f"LRU {lru_budget:,} B/layer")
        out[mix] = {"pinned_pairs": len(pinned), "pinned_bytes": pin_used,
                    "per_trace": {}}
        for t, calls in calls_by_trace.items():
            streams = per_layer_streams(calls)
            res = run_policies(streams, sizes, pinned, lru_budget)
            name = Path(t).stem
            blob = {}
            for pol, (req, ph, m, mb) in res.items():
                blob[pol] = {"requests": req, "pin_hits": ph, "misses": m,
                             "miss_bytes": mb}
                rows.append((mix, name, pol, req, ph, m, mb))
            b_lru = blob["byte_lru"]["miss_bytes"]
            b_oracle = min(blob[p]["miss_bytes"] for p in OFFLINE)
            b_best_online = min(
                blob[p]["miss_bytes"] for p in POLICIES if p not in OFFLINE)
            gap = (b_lru - b_oracle) / b_lru if b_lru else 0.0
            blob["gap_lru_vs_oracle_pct"] = 100 * gap
            blob["gap_lru_vs_demand_belady_pct"] = (
                100 * (b_lru - blob["belady_demand"]["miss_bytes"]) / b_lru
                if b_lru else 0.0)
            blob["gap_lru_vs_bypass_belady_pct"] = (
                100 * (b_lru - blob["belady_bytes"]["miss_bytes"]) / b_lru
                if b_lru else 0.0)
            blob["best_online"] = min(
                (p for p in POLICIES if p not in OFFLINE),
                key=lambda p: blob[p]["miss_bytes"])
            blob["gap_lru_vs_best_online_pct"] = (
                100 * (b_lru - b_best_online) / b_lru if b_lru else 0.0)
            out[mix]["per_trace"][name] = blob
            print(f"  {name:18s} LRU {b_lru/GB:8.3f} GB | demand-Belady gap "
                  f"{blob['gap_lru_vs_demand_belady_pct']:5.2f}% | +bypass gap "
                  f"{blob['gap_lru_vs_bypass_belady_pct']:5.2f}% | best online "
                  f"{blob['best_online']} "
                  f"({blob['gap_lru_vs_best_online_pct']:+.2f}%)")
        gaps = [v["gap_lru_vs_oracle_pct"]
                for v in out[mix]["per_trace"].values()]
        out[mix]["max_gap_pct"] = max(gaps)
        out[mix]["median_gap_pct"] = sorted(gaps)[len(gaps) // 2]

    with (a.outdir / "oracle-policies.csv").open("w") as f:
        f.write("mix,trace,policy,requests,pin_hits,misses,miss_bytes\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    # sanity: GDS(cost=size) must equal byte-LRU exactly
    for mix, blob in out.items():
        for name, tb in blob["per_trace"].items():
            if tb["gds_cost_size"]["miss_bytes"] != tb["byte_lru"]["miss_bytes"]:
                print(f"  WARNING: GDS(size) != byte-LRU on {mix}/{name}")
    return out


def run_extents(a, cfg, usage, int4_b, pin_bytes, lru_budget, calls_by_trace):
    import numpy as np  # noqa: F401  (coactivation uses numpy)

    cap = EXTENT_BYTES // int4_b  # experts per 64 MB extent (int4: 3)
    n_exp = cfg["n_experts"]
    fit_trace = a.fit_trace or a.traces[0]
    fit_calls = calls_by_trace[fit_trace]
    layers = sorted({c.layer for calls in calls_by_trace.values() for c in calls})

    print(f"\n[extents] {cap} int4 experts per {EXTENT_BYTES/1e6:.0f} MB extent, "
          f"fit trace {Path(fit_trace).stem}")
    clustered = {}
    n_bins = {}
    for layer in layers:
        co = coactivation(fit_calls, layer, n_exp)
        extent_of, bins = cluster_extents(co, cap)
        clustered[layer] = extent_of
        n_bins[layer] = bins
    id_ordered = {layer: [e // cap for e in range(n_exp)] for layer in layers}

    sizes = build_sizes(cfg, set(), int4_b, int4_b, "uniform-int4")
    pinned, _ = fill_pins(usage, sizes, pin_bytes)

    out = {"experts_per_extent": cap,
           "extents_per_layer": math.ceil(n_exp / cap),
           "fit_trace": Path(fit_trace).stem, "per_trace": {}}
    rows = []
    for t, calls in calls_by_trace.items():
        name = Path(t).stem
        held_out = t != fit_trace
        flags = miss_flags_byte_lru(calls, sizes, pinned, lru_budget)
        sel_id = extents_touched(calls, id_ordered)
        sel_cl = extents_touched(calls, clustered)
        miss_id = extents_touched(calls, id_ordered, flags)
        miss_cl = extents_touched(calls, clustered, flags)
        red_sel = 1 - sel_cl["mean_extents_per_call"] / sel_id["mean_extents_per_call"]
        red_miss = (1 - miss_cl["mean_extents_per_call"]
                    / miss_id["mean_extents_per_call"]
                    if miss_id["mean_extents_per_call"] else 0.0)
        out["per_trace"][name] = {
            "held_out": held_out,
            "selection": {"id_ordered": sel_id, "clustered": sel_cl,
                          "extent_reduction_pct": 100 * red_sel},
            "miss_only": {"id_ordered": miss_id, "clustered": miss_cl,
                          "extent_reduction_pct": 100 * red_miss,
                          "preadv_today_per_expert": miss_id["total_experts"],
                          "preadv_clustered_extents": miss_cl["total_extents"]},
        }
        rows.append((name, held_out,
                     sel_id["mean_extents_per_call"],
                     sel_cl["mean_extents_per_call"], 100 * red_sel,
                     miss_id["mean_extents_per_call"],
                     miss_cl["mean_extents_per_call"], 100 * red_miss,
                     miss_id["total_experts"], miss_id["total_extents"],
                     miss_cl["total_extents"]))
        print(f"  {name:18s} {'held-out' if held_out else 'in-sample':9s} "
              f"sel {sel_id['mean_extents_per_call']:.3f} -> "
              f"{sel_cl['mean_extents_per_call']:.3f} (-{100*red_sel:.1f}%) | "
              f"miss {miss_id['mean_extents_per_call']:.3f} -> "
              f"{miss_cl['mean_extents_per_call']:.3f} (-{100*red_miss:.1f}%)")

    with (a.outdir / "extents-summary.csv").open("w") as f:
        f.write("trace,held_out,sel_extents_id,sel_extents_clustered,"
                "sel_reduction_pct,miss_extents_id,miss_extents_clustered,"
                "miss_reduction_pct,miss_experts_total,"
                "miss_extents_total_id,miss_extents_total_clustered\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    held = [v["selection"]["extent_reduction_pct"]
            for v in out["per_trace"].values() if v["held_out"]]
    out["headline_heldout_selection_reduction_pct"] = (
        sorted(held)[len(held) // 2] if held else None)
    return out


if __name__ == "__main__":
    sys.exit(main())
