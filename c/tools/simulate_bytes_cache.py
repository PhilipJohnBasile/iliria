#!/usr/bin/env python3
"""Byte-capacity-aware route-trace cache replay (offline; never touches the engine).

Extends tools/simulate_m5max_cache.py (slot-count LRU, uniform expert size) to
heterogeneous per-expert byte sizes so mixed-precision container layouts can be
adjudicated against the SAME ordered route trace:

  - the pinned tier is filled by BYTE budget exactly like the engine's
    pin_load(): usage-ranked (layer, expert) pairs, truncated at the first
    pair that no longer fits;
  - the per-layer LRU runs in either SLOT-COUNT mode (the CURRENT engine:
    `ecap` ESlot slabs per layer -- a smaller int2 expert does NOT create
    another slot) or BYTE-BUDGET mode (repaired accounting: byte-exact
    eviction from the LRU end until the incoming expert fits);
  - an optional global second-tier page-cache LRU (bytes) models UBC reuse;
    the production launcher runs DIRECT=1 (no page cache), so it defaults
    to 0 bytes and stays inert unless configured;
  - each MoE call is resolved in TWO PHASES against the cache state at call
    entry (classification of the whole batch union first, insertions after),
    matching the engine's batch resolve; this is also what makes the
    per-call miss count M well defined (pre-insertion semantics);
  - decode tokens are segmented from the trace (single-row calls; a new
    token starts when the layer index resets), giving warmup vs steady-state
    B_miss bytes/token and the P(M = misses among the 8 routed experts)
    miss-structure outputs.

The tok/s numbers printed by --bw-gbs are SENSITIVITY ONLY (t = C + B/BW with
a fixed non-disk floor C); B_miss bytes/token is the authoritative output.

Layout presets (see build_layouts): A = uniform int4 @ current slot ecap;
B = manifest int2 set @ current ecap; C = all-cold-tier int2 @ current ecap
(cheaper miss bytes only); D = all-cold-tier int2 @ RAM-matched byte-budget
LRU (adds the capacity-derived hit gain). E-* are groupwise-scale size
sensitivity variants (different tensor format; NOT engine-ready today).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, OrderedDict, defaultdict
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

GB = 1e9


# ------------------------------------------------------------- byte sizing ----
def packed_nbytes(O: int, I: int, bits: int) -> int:
    """Mirror of glm.c qt_alloc / quant_container.packed_nbytes."""
    if bits >= 5:
        return O * I
    if bits >= 3:
        return O * ((I + 1) // 2)
    return O * ((I + 3) // 4)


def expert_bytes_per_row_scales(cfg: dict, bits: int) -> int:
    """One routed expert: 3 packed tensors + per-row F32 scales (shipping format)."""
    total = 0
    for proj in ("gate", "up", "down"):
        if proj == "down":
            O, I = cfg["hidden"], cfg["moe_inter"]
        else:
            O, I = cfg["moe_inter"], cfg["hidden"]
        total += packed_nbytes(O, I, bits) + O * 4
    return total


def expert_bytes_group_scales(cfg: dict, bits: int, group: int, scale_bytes: int) -> int:
    """Hypothetical groupwise-scale format: packed payload + one scale per
    `group` input elements per row (NOT engine-ready; sizing only)."""
    total = 0
    for proj in ("gate", "up", "down"):
        if proj == "down":
            O, I = cfg["hidden"], cfg["moe_inter"]
        else:
            O, I = cfg["moe_inter"], cfg["hidden"]
        n_groups = O * ((I + group - 1) // group)
        total += packed_nbytes(O, I, bits) + n_groups * scale_bytes
    return total


# ------------------------------------------------------------------ trace ----
@dataclass
class Call:
    call_id: int
    layer: int
    experts: list  # first-occurrence union, router order
    n_rows: int
    n_records: int


def load_calls(path: Path):
    """-> (metadata, [Call, ...]) preserving strict trace order."""
    calls: list[Call] = []
    current = None
    rows = set()
    with path.open("rb") as f:
        metadata = slotsim.read_trace_header(f, path)
        for rec in slotsim._iter_records(f, path):
            if current is None or rec.call != current.call_id:
                if current is not None:
                    current.n_rows = len(rows)
                    calls.append(current)
                current = Call(rec.call, rec.layer, [], 0, 0)
                rows = set()
            if rec.layer != current.layer:
                raise ValueError(f"{path}: call {rec.call} spans layers")
            rows.add(rec.row)
            current.n_records += 1
            if rec.expert not in current.experts:
                current.experts.append(rec.expert)
        if current is not None:
            current.n_rows = len(rows)
            calls.append(current)
    return metadata, calls


def segment_tokens(calls):
    """Assign a decode-token index to every decode (single-row) call.

    -> (token_of: {call index -> token id or None}, n_tokens). A new token
    starts at a decode call whose layer is <= the previous decode call's
    layer (the per-token layer walk is strictly increasing).
    """
    token_of = {}
    token = -1
    prev_layer = None
    for i, call in enumerate(calls):
        if call.n_rows != 1:
            token_of[i] = None
            continue
        if prev_layer is None or call.layer <= prev_layer:
            token += 1
        prev_layer = call.layer
        token_of[i] = token
    return token_of, token + 1


# ------------------------------------------------------------------- tiers ----
class LayerLRU:
    """One layer's dynamic tier: slot-count (current engine) or byte budget."""

    __slots__ = ("mode", "budget", "od", "used")

    def __init__(self, mode: str, budget: int):
        if mode not in ("slots", "bytes"):
            raise ValueError(f"bad LRU mode {mode}")
        self.mode = mode
        self.budget = budget
        self.od: OrderedDict = OrderedDict()
        self.used = 0

    def __contains__(self, expert) -> bool:
        return expert in self.od

    def touch(self, expert) -> None:
        self.od.move_to_end(expert)

    def insert(self, expert, size: int, evicted=None) -> None:
        if self.budget <= 0:
            return
        if self.mode == "slots":
            while len(self.od) >= self.budget:
                k, v = self.od.popitem(last=False)
                self.used -= v
                if evicted is not None:
                    evicted.append((k, v))
        else:
            if size > self.budget:
                return  # can never fit; do not flush the tier for it
            while self.od and self.used + size > self.budget:
                k, v = self.od.popitem(last=False)
                self.used -= v
                if evicted is not None:
                    evicted.append((k, v))
        self.od[expert] = size
        self.used += size


class PageCache:
    """Global byte-LRU second tier keyed by (layer, expert). Entries appear
    when bytes are READ FROM DISK (a RAM-tier eviction does not re-insert:
    the file pages entered the cache at read time and simply survive)."""

    __slots__ = ("budget", "od", "used")

    def __init__(self, budget: int):
        self.budget = budget
        self.od: OrderedDict = OrderedDict()
        self.used = 0

    def hit(self, key) -> bool:
        if self.budget <= 0 or key not in self.od:
            return False
        self.od.move_to_end(key)
        return True

    def fill(self, key, size: int) -> None:
        if self.budget <= 0 or size > self.budget:
            return
        if key in self.od:
            self.od.move_to_end(key)
            return
        while self.od and self.used + size > self.budget:
            _, v = self.od.popitem(last=False)
            self.used -= v
        self.od[key] = size
        self.used += size


# -------------------------------------------------------------------- pins ----
def fill_pins(usage: dict, sizes: dict, pin_bytes: int):
    """pin_load() semantics: rank every known (layer, expert) by usage count
    (ties: layer, expert), walk the ranking, stop at the first pair that no
    longer fits the byte budget. -> (pinned {pair: bytes}, used_bytes)."""
    ranked = sorted(usage, key=lambda p: (-usage[p], p[0], p[1]))
    pinned, used = {}, 0
    for pair in ranked:
        size = sizes.get(pair)
        if size is None:
            continue  # outside the routed-expert table
        if used + size > pin_bytes:
            break
        pinned[pair] = size
        used += size
    return pinned, used


# ------------------------------------------------------------------ replay ----
@dataclass
class ReplayResult:
    layout: str
    trace: str
    lru_mode: str
    lru_budget: int
    pinned_pairs: int
    pinned_bytes: int
    requests: int = 0
    pin_hits: int = 0
    lru_hits: int = 0
    page_hits: int = 0
    disk_misses: int = 0
    disk_bytes: int = 0
    dec_requests: int = 0
    dec_pin_hits: int = 0
    dec_lru_hits: int = 0
    dec_page_hits: int = 0
    dec_disk_misses: int = 0
    dec_disk_bytes: int = 0
    n_tokens: int = 0
    per_token_bytes: list = field(default_factory=list)
    per_token_misses: list = field(default_factory=list)
    call_m: list = field(default_factory=list)  # (token, layer, M, disk_bytes, hit_flags)

    @property
    def hit_rate(self) -> float:
        return (self.pin_hits + self.lru_hits) / self.requests if self.requests else 0.0

    @property
    def dec_hit_rate(self) -> float:
        return ((self.dec_pin_hits + self.dec_lru_hits) / self.dec_requests
                if self.dec_requests else 0.0)

    def steady(self, warmup: int):
        """(bytes/token, misses/token, tokens) for decode tokens >= warmup."""
        b = self.per_token_bytes[warmup:]
        m = self.per_token_misses[warmup:]
        if not b:
            return 0.0, 0.0, 0
        return sum(b) / len(b), sum(m) / len(m), len(b)

    def warm(self, warmup: int):
        b = self.per_token_bytes[:warmup]
        m = self.per_token_misses[:warmup]
        if not b:
            return 0.0, 0.0, 0
        return sum(b) / len(b), sum(m) / len(m), len(b)


def replay(calls, token_of, sizes, pinned, lru_mode, lru_budget,
           page_bytes=0, layout="?", trace="?", collect_m=False) -> ReplayResult:
    """Two-phase batch-union replay. RAM-tier residency (pin | LRU) decides
    hit/miss; a page-cache hit still counts as a RAM miss but costs no disk
    bytes. M (miss structure) is classified against the cache state at CALL
    ENTRY: insertions from the same call can neither hit nor evict earlier
    classifications."""
    lrus: dict[int, LayerLRU] = defaultdict(lambda: LayerLRU(lru_mode, lru_budget))
    page = PageCache(page_bytes) if page_bytes > 0 else None
    res = ReplayResult(layout, trace, lru_mode, lru_budget,
                       len(pinned), sum(pinned.values()))
    n_tokens = max((t for t in token_of.values() if t is not None), default=-1) + 1
    res.n_tokens = n_tokens
    res.per_token_bytes = [0] * n_tokens
    res.per_token_misses = [0] * n_tokens

    for idx, call in enumerate(calls):
        layer = call.layer
        lru = lrus[layer]
        token = token_of[idx]
        decode = token is not None

        # phase 1: classify the whole batch union against call-entry state
        lru_hit_experts, missing, flags = [], [], []
        pin_h = lru_h = 0
        for expert in call.experts:
            if (layer, expert) in pinned:
                pin_h += 1
                flags.append(1)
            elif expert in lru:
                lru_h += 1
                lru_hit_experts.append(expert)
                flags.append(1)
            else:
                missing.append(expert)
                flags.append(0)

        # phase 2: apply recency updates and load the misses (router order)
        for expert in lru_hit_experts:
            lru.touch(expert)
        page_h = 0
        disk_b = 0
        for expert in missing:
            size = sizes[(layer, expert)]
            if page is not None and page.hit((layer, expert)):
                page_h += 1
            else:
                disk_b += size
                if page is not None:
                    page.fill((layer, expert), size)
            lru.insert(expert, size)

        n = len(call.experts)
        res.requests += n
        res.pin_hits += pin_h
        res.lru_hits += lru_h
        res.page_hits += page_h
        res.disk_misses += len(missing) - page_h
        res.disk_bytes += disk_b
        if decode:
            res.dec_requests += n
            res.dec_pin_hits += pin_h
            res.dec_lru_hits += lru_h
            res.dec_page_hits += page_h
            res.dec_disk_misses += len(missing) - page_h
            res.dec_disk_bytes += disk_b
            res.per_token_bytes[token] += disk_b
            res.per_token_misses[token] += len(missing) - page_h
            if collect_m:
                res.call_m.append((token, layer, len(missing), disk_b, tuple(flags)))
    return res


# ---------------------------------------------------------- miss structure ----
def miss_structure(res: ReplayResult, warmup: int):
    """Task-B statistics from the collected per-decode-call records."""
    calls = res.call_m
    steady = [c for c in calls if c[0] >= warmup]
    out = {"warmup_tokens": warmup, "n_decode_calls": len(calls),
           "n_steady_calls": len(steady)}

    def pm(rows):
        counts = Counter(m for _, _, m, _, _ in rows)
        total = sum(counts.values())
        return {m: counts.get(m, 0) / total for m in range(9)}, counts, total

    pm_all, counts_all, total_all = pm(calls)
    pm_steady, _, _ = pm(steady)
    out["pm_all"] = pm_all
    out["pm_steady"] = pm_steady
    out["p_le1"] = pm_all[0] + pm_all[1]
    out["p_le2"] = pm_all[0] + pm_all[1] + pm_all[2]
    out["p_le1_steady"] = pm_steady[0] + pm_steady[1]
    out["p_le2_steady"] = pm_steady[0] + pm_steady[1] + pm_steady[2]

    # per-layer P(M)
    by_layer = defaultdict(list)
    for tok, layer, m, b, fl in calls:
        by_layer[layer].append(m)
    out["per_layer_pm"] = {
        layer: {m: c / len(ms) for m, c in Counter(ms).items()}
        for layer, ms in sorted(by_layer.items())
    }

    # all-hit frequency vs independence
    h = res.dec_hit_rate
    out["h"] = h
    out["h8"] = h ** 8
    out["p_m0"] = pm_all[0]
    per_layer_h8 = []
    for layer, ms in by_layer.items():
        hl = 1.0 - sum(ms) / (8 * len(ms))
        per_layer_h8.append(hl ** 8)
    out["mean_layer_h8"] = sum(per_layer_h8) / len(per_layer_h8) if per_layer_h8 else 0.0

    # bytes-missed distribution by M (steady state; warmup mixes regimes)
    import numpy as np
    bytes_by_m = defaultdict(list)
    for c in steady:
        bytes_by_m[c[2]].append(c[3])
    out["bytes_by_m"] = {
        m: {
            "n": len(v),
            "mean": float(np.mean(v)),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
        }
        for m, v in sorted(bytes_by_m.items())
    }
    tok_bytes = np.array(res.per_token_bytes[warmup:], dtype=float)
    out["token_bytes"] = {
        "mean": float(tok_bytes.mean()) if tok_bytes.size else 0.0,
        "p10": float(np.percentile(tok_bytes, 10)) if tok_bytes.size else 0.0,
        "p50": float(np.percentile(tok_bytes, 50)) if tok_bytes.size else 0.0,
        "p90": float(np.percentile(tok_bytes, 90)) if tok_bytes.size else 0.0,
        "p99": float(np.percentile(tok_bytes, 99)) if tok_bytes.size else 0.0,
    }

    # pairwise hit-indicator correlation across the 8 route ranks
    mat = np.array([fl for _, _, _, _, fl in calls if len(fl) == 8], dtype=float)
    def mean_pairwise_r(matrix):
        if matrix.shape[0] < 2:
            return float("nan"), 0
        with np.errstate(invalid="ignore"):
            cm = np.corrcoef(matrix.T)
        iu = np.triu_indices(8, k=1)
        vals = cm[iu]
        vals = vals[~np.isnan(vals)]
        return (float(vals.mean()) if vals.size else float("nan")), int(vals.size)

    out["mean_pairwise_r_pooled"], out["n_pairs_pooled"] = mean_pairwise_r(mat)
    layer_rs = []
    for layer in sorted(by_layer):
        lm = np.array([fl for _, l, _, _, fl in calls
                       if l == layer and len(fl) == 8], dtype=float)
        r, npairs = mean_pairwise_r(lm)
        if not math.isnan(r):
            layer_rs.append(r)
    out["mean_pairwise_r_stratified"] = (
        float(np.mean(layer_rs)) if layer_rs else float("nan"))
    out["n_layers_with_r"] = len(layer_rs)

    # consecutive all-hit-layer run lengths within each token
    per_token_layers = defaultdict(list)
    for tok, layer, m, b, fl in calls:
        per_token_layers[tok].append((layer, m))
    run_hist = Counter()
    covered_ge = Counter()  # layer-calls inside runs of length >= k
    total_layer_calls = 0
    for tok in sorted(per_token_layers):
        seq = [m == 0 for _, m in sorted(per_token_layers[tok])]
        total_layer_calls += len(seq)
        run = 0
        for allhit in seq + [False]:
            if allhit:
                run += 1
            elif run:
                run_hist[run] += 1
                for k in (2, 3, 4):
                    if run >= k:
                        covered_ge[k] += run
                run = 0
    out["run_hist"] = dict(sorted(run_hist.items()))
    out["all_hit_layer_calls"] = sum(l * c for l, c in run_hist.items())
    out["total_layer_calls"] = total_layer_calls
    out["frac_layers_all_hit"] = (out["all_hit_layer_calls"] / total_layer_calls
                                  if total_layer_calls else 0.0)
    out["frac_layers_in_runs_ge2"] = (covered_ge[2] / total_layer_calls
                                      if total_layer_calls else 0.0)
    out["frac_layers_in_runs_ge3"] = (covered_ge[3] / total_layer_calls
                                      if total_layer_calls else 0.0)
    return out


# ----------------------------------------------------------------- layouts ----
def build_layouts(cfg, manifest_int2, forced_layers, names, int4_b, int2_b,
                  groupwise_sizes):
    """-> [(name, sizes {(layer,expert): bytes}, lru_mode, description)]."""
    routed = [(l, e) for l in range(cfg["first_dense"], cfg["n_layers"])
              for e in range(cfg["n_experts"])]
    cold = [p for p in routed if p[0] not in forced_layers]

    def uniform():
        return {p: int4_b for p in routed}

    def manifest():
        s = uniform()
        for p in manifest_int2:
            s[p] = int2_b
        return s

    def all_cold(size):
        s = uniform()
        for p in cold:
            s[p] = size
        return s

    table = {}
    table["A"] = (uniform(), "slots",
                  f"uniform int4 {int4_b} B @ current slot ecap")
    table["B"] = (manifest(), "slots",
                  f"manifest: {len(manifest_int2)} int2 @ {int2_b} B, rest int4 "
                  "@ current slot ecap")
    table["C"] = (all_cold(int2_b), "slots",
                  f"repaired tier: ALL {len(cold)} cold experts int2 @ {int2_b} B "
                  "(fitted-scale, format-compatible) @ current slot ecap")
    table["D"] = (all_cold(int2_b), "bytes",
                  f"repaired tier as C but RAM-matched BYTE-budget LRU")
    for label, size in groupwise_sizes.items():
        table[f"E-{label}-slot"] = (
            all_cold(size), "slots",
            f"groupwise sizing sensitivity: cold experts {size} B @ slot ecap")
        table[f"E-{label}-byte"] = (
            all_cold(size), "bytes",
            f"groupwise sizing sensitivity: cold experts {size} B @ byte LRU")

    out = []
    for name in names:
        if name == "E":
            out.extend((k, *table[k]) for k in table if k.startswith("E-"))
        elif name in table:
            out.append((name, *table[name]))
        else:
            raise SystemExit(f"unknown layout {name} (have {sorted(table)} + E)")
    return out


# -------------------------------------------------------------------- main ----
def md5_of(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("traces", type=Path, nargs="+")
    p.add_argument("--usage", type=Path, required=True,
                   help=".fa_usage histogram that ranks the pin fill")
    p.add_argument("--manifest", type=Path, required=True,
                   help="allocation manifest JSON (model_config, int2 list, "
                        "force_int4_layers)")
    p.add_argument("--layouts", default="A,B,C,D",
                   help="comma list from A,B,C,D,E (E = groupwise sensitivity)")
    p.add_argument("--int4-bytes", type=int, default=None,
                   help="override int4 expert bytes (default: model_config, "
                        "verified against every trace header)")
    p.add_argument("--int2-bytes", type=int, default=None,
                   help="override int2 expert bytes (default: model_config)")
    p.add_argument("--pin-bytes", type=int, default=None,
                   help="pinned-tier byte budget (default: trace pinned_units "
                        "x int4 bytes)")
    p.add_argument("--lru-slots", type=int, default=None,
                   help="slot-mode capacity (default: trace lru_per_layer)")
    p.add_argument("--lru-bytes", type=int, default=None,
                   help="byte-mode per-layer budget (default: lru-slots x int4 "
                        "bytes -- RAM-matched)")
    p.add_argument("--page-cache-bytes", type=int, default=0,
                   help="second-tier page-cache bytes (default 0: DIRECT=1)")
    p.add_argument("--bw-gbs", default="5,8,10,13.3",
                   help="sensitivity disk bandwidths, decimal GB/s")
    p.add_argument("--c-seconds", type=float, default=0.36,
                   help="non-disk seconds per decode token in t = C + B/BW")
    p.add_argument("--warmup-tokens", type=int, default=32,
                   help="decode tokens treated as warmup (excluded from "
                        "steady-state B_miss)")
    p.add_argument("--miss-structure", action="store_true",
                   help="collect P(M), correlations, run lengths per layout")
    p.add_argument("--outdir", type=Path, default=None,
                   help="write CSV outputs + config.json here")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    manifest = json.loads(a.manifest.read_text())
    cfg = manifest["model_config"]
    forced = set(manifest.get("force_int4_layers", []))
    manifest_int2 = set(map(tuple, manifest.get("int2", [])))

    int4_b = a.int4_bytes or expert_bytes_per_row_scales(cfg, 4)
    int2_b = a.int2_bytes or expert_bytes_per_row_scales(cfg, 2)
    groupwise = {
        "g128f16": expert_bytes_group_scales(cfg, 2, 128, 2),
        "g128f32": expert_bytes_group_scales(cfg, 2, 128, 4),
    }

    # trace metadata drives the RAM budgets; verify consistency across traces
    metas = {}
    calls_by_trace = {}
    for tpath in a.traces:
        meta, calls = load_calls(tpath)
        if meta.mtp_present:
            raise SystemExit(f"{tpath}: MTP tier present; unsupported")
        if meta.expert_bytes and meta.expert_bytes != int4_b:
            raise SystemExit(
                f"{tpath}: header expert_bytes {meta.expert_bytes} != int4 "
                f"size {int4_b}; pass --int4-bytes to override")
        metas[tpath] = meta
        calls_by_trace[tpath] = calls

    meta0 = metas[a.traces[0]]
    pin_bytes = a.pin_bytes if a.pin_bytes is not None else meta0.pinned_units * int4_b
    lru_slots = a.lru_slots if a.lru_slots is not None else meta0.lru_per_layer
    lru_bytes = a.lru_bytes if a.lru_bytes is not None else lru_slots * int4_b
    bws = [float(x) for x in a.bw_gbs.split(",") if x]

    usage = slotsim.load_usage(a.usage)
    layouts = build_layouts(cfg, manifest_int2, forced,
                            [s.strip() for s in a.layouts.split(",") if s.strip()],
                            int4_b, int2_b, groupwise)

    print(f"[bytes-sim] int4 {int4_b} B | int2 {int2_b} B | groupwise {groupwise}")
    print(f"[bytes-sim] pin budget {pin_bytes:,} B ({pin_bytes/GB:.3f} GB) | "
          f"LRU {lru_slots} slots | byte-mode {lru_bytes:,} B/layer")
    print(f"[bytes-sim] usage {a.usage} md5 {md5_of(a.usage)[:10]} | "
          f"page-cache {a.page_cache_bytes:,} B | warmup {a.warmup_tokens} tokens")

    if a.outdir:
        a.outdir.mkdir(parents=True, exist_ok=True)

    grid_rows, tok_rows, struct_blobs = [], [], {}
    pinsets_written = set()
    for name, sizes, mode, desc in layouts:
        pinned, pin_used = fill_pins(usage, sizes, pin_bytes)
        budget = lru_slots if mode == "slots" else lru_bytes
        for tpath in a.traces:
            tname = tpath.stem
            token_of, n_tokens = segment_tokens(calls_by_trace[tpath])
            res = replay(calls_by_trace[tpath], token_of, sizes, pinned,
                         mode, budget, a.page_cache_bytes, name, tname,
                         collect_m=a.miss_structure)
            sb, sm, sn = res.steady(a.warmup_tokens)
            wb, wm, wn = res.warm(a.warmup_tokens)
            row = {
                "layout": name, "trace": tname, "lru_mode": mode,
                "lru_budget": budget, "pinned_pairs": res.pinned_pairs,
                "pinned_bytes": res.pinned_bytes,
                "requests": res.requests, "pin_hits": res.pin_hits,
                "lru_hits": res.lru_hits, "page_hits": res.page_hits,
                "disk_misses": res.disk_misses, "disk_bytes": res.disk_bytes,
                "hit_rate": round(res.hit_rate, 6),
                "dec_hit_rate": round(res.dec_hit_rate, 6),
                "dec_disk_bytes": res.dec_disk_bytes,
                "n_decode_tokens": res.n_tokens,
                "warm_tokens": wn, "warm_bytes_per_tok": round(wb, 1),
                "warm_miss_per_tok": round(wm, 3),
                "steady_tokens": sn, "steady_bytes_per_tok": round(sb, 1),
                "steady_miss_per_tok": round(sm, 3),
            }
            grid_rows.append(row)
            print(f"  {name:12s} {tname:16s} {mode:5s} hit {100*res.hit_rate:6.2f}% "
                  f"(decode {100*res.dec_hit_rate:6.2f}%) steady "
                  f"{sb/1e6:8.1f} MB/tok {sm:6.2f} miss/tok")
            for bw in bws:
                t = a.c_seconds + sb / (bw * GB)
                tok_rows.append({
                    "layout": name, "trace": tname, "bw_gbs": bw,
                    "t_s_per_tok": round(t, 4), "tok_s": round(1.0 / t, 4),
                })
            if a.miss_structure and not name.startswith("E-"):
                struct_blobs[(name, tname)] = miss_structure(res, a.warmup_tokens)
            if a.outdir:
                per_tok = a.outdir / f"per-token-{name}-{tname}.csv"
                with per_tok.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["token", "disk_bytes", "disk_misses"])
                    for i, (b, m) in enumerate(
                            zip(res.per_token_bytes, res.per_token_misses)):
                        w.writerow([i, b, m])
        if a.outdir and name not in pinsets_written:
            pinsets_written.add(name)
            with (a.outdir / f"pinset-{name}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer", "expert", "bytes"])
                for (l, e), sz in sorted(pinned.items()):
                    w.writerow([l, e, sz])

    # delta vs layout A at the same trace/BW
    base = {(r["trace"], r["bw_gbs"]): r["tok_s"] for r in tok_rows
            if r["layout"] == "A"}
    for r in tok_rows:
        b = base.get((r["trace"], r["bw_gbs"]))
        r["delta_vs_A_pct"] = round(100 * (r["tok_s"] / b - 1), 2) if b else ""

    if a.outdir:
        with (a.outdir / "layout-grid.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(grid_rows[0]))
            w.writeheader()
            w.writerows(grid_rows)
        with (a.outdir / "tok-s-grid.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tok_rows[0]))
            w.writeheader()
            w.writerows(tok_rows)
        if struct_blobs:
            _write_struct(a.outdir, struct_blobs)
        config = {
            "traces": {str(t): {"md5": md5_of(t),
                                "pinned_units": metas[t].pinned_units,
                                "lru_per_layer": metas[t].lru_per_layer,
                                "expert_bytes": metas[t].expert_bytes}
                       for t in a.traces},
            "usage": {"path": str(a.usage), "md5": md5_of(a.usage)},
            "manifest": {"path": str(a.manifest), "md5": md5_of(a.manifest),
                         "n_int2": len(manifest_int2),
                         "force_int4_layers": sorted(forced)},
            "int4_bytes": int4_b, "int2_bytes": int2_b,
            "groupwise_bytes": groupwise,
            "pin_bytes": pin_bytes, "lru_slots": lru_slots,
            "lru_bytes": lru_bytes, "page_cache_bytes": a.page_cache_bytes,
            "c_seconds": a.c_seconds, "bw_gbs": bws,
            "warmup_tokens": a.warmup_tokens,
            "layouts": {n: d for n, _, _, d in layouts},
            "note": "tok/s grid is sensitivity only (t = C + B_miss/BW); "
                    "B_miss bytes/token is the authoritative output",
        }
        (a.outdir / "config.json").write_text(json.dumps(config, indent=2))
        print(f"[bytes-sim] outputs in {a.outdir}")
    return 0


def _write_struct(outdir: Path, blobs: dict) -> None:
    with (outdir / "missstruct-pm.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "trace", "layer"] + [f"p_m{m}" for m in range(9)])
        for (name, tname), s in sorted(blobs.items()):
            w.writerow([name, tname, "ALL"]
                       + [round(s["pm_all"].get(m, 0.0), 6) for m in range(9)])
            w.writerow([name, tname, "ALL_STEADY"]
                       + [round(s["pm_steady"].get(m, 0.0), 6) for m in range(9)])
            for layer, pm in s["per_layer_pm"].items():
                w.writerow([name, tname, layer]
                           + [round(pm.get(m, 0.0), 6) for m in range(9)])
    with (outdir / "missstruct-summary.csv").open("w", newline="") as f:
        cols = ["layout", "trace", "h_decode", "p_m0", "h^8", "mean_layer_h^8",
                "p_le1", "p_le2", "p_le1_steady", "p_le2_steady",
                "mean_pairwise_r_pooled", "mean_pairwise_r_stratified",
                "frac_layers_all_hit", "frac_layers_in_runs_ge2",
                "frac_layers_in_runs_ge3", "n_decode_calls"]
        w = csv.writer(f)
        w.writerow(cols)
        for (name, tname), s in sorted(blobs.items()):
            w.writerow([name, tname, round(s["h"], 6), round(s["p_m0"], 6),
                        round(s["h8"], 6), round(s["mean_layer_h8"], 6),
                        round(s["p_le1"], 6), round(s["p_le2"], 6),
                        round(s["p_le1_steady"], 6), round(s["p_le2_steady"], 6),
                        round(s["mean_pairwise_r_pooled"], 6),
                        round(s["mean_pairwise_r_stratified"], 6),
                        round(s["frac_layers_all_hit"], 6),
                        round(s["frac_layers_in_runs_ge2"], 6),
                        round(s["frac_layers_in_runs_ge3"], 6),
                        s["n_decode_calls"]])
    with (outdir / "missstruct-bytes-by-m.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "trace", "m", "n_calls", "mean_bytes",
                    "p50_bytes", "p90_bytes", "p99_bytes"])
        for (name, tname), s in sorted(blobs.items()):
            for m, st in s["bytes_by_m"].items():
                w.writerow([name, tname, m, st["n"], round(st["mean"], 1),
                            st["p50"], st["p90"], st["p99"]])
    with (outdir / "missstruct-runlengths.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "trace", "run_length", "n_runs"])
        for (name, tname), s in sorted(blobs.items()):
            for length, n in s["run_hist"].items():
                w.writerow([name, tname, length, n])
    with (outdir / "missstruct-token-bytes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "trace", "mean", "p10", "p50", "p90", "p99"])
        for (name, tname), s in sorted(blobs.items()):
            tb = s["token_bytes"]
            w.writerow([name, tname, round(tb["mean"], 1), tb["p10"],
                        tb["p50"], tb["p90"], tb["p99"]])


if __name__ == "__main__":
    sys.exit(main())
