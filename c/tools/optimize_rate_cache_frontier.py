#!/usr/bin/env python3
"""Counterfactual rate-cache ranking: is error-per-byte greedy leaving
miss-bytes on the table?

Trace-only (zero shard reads). For a stratified sample of demotable,
non-pinned experts, computes the counterfactual marginal cache value

    V_e = B_miss(e at int4) - B_miss(e at int2 repaired)

by one-at-a-time byte-LRU replay at the D-case byte-budget cache
(per-layer budget = lru_per_layer x int4 bytes from the trace header),
holding every OTHER expert at its 280 GB-manifest size and the pin tier
frozen. Because layers are independent given a frozen pin set, each
counterfactual only replays the flipped expert's layer; experts absent
from a layer's request stream have V_e = 0 exactly (shortcut, verified
in tests). V_e is summed over all supplied traces.

Comparison target: the shipped error-per-byte greedy
(allocate_bit_budget.py ranks demotions by err_int2 ascending; with
constant bytes-saved per demotion that IS error-per-byte). Reported:

  - Spearman rank correlation between the greedy demotion priority
    (-err_int2) and the cache-value priority (V_e), tie-averaged ranks;
  - a matched-distortion Pareto test: demote K experts from the sampled
    universe, chosen (a) by the greedy (lowest err) vs (b) by V_e descending
    subject to sum(err) <= the greedy set's sum (feasibility-aware greedy
    knapsack), then FULL byte-LRU replays of both K-demotion layouts
    (identical byte savings, distortion(b) <= distortion(a)); Pareto
    improvement = (B_greedy - B_cache_aware)/B_greedy;
  - an additivity check: sum(V_e) over set (b) vs the replayed delta from
    the all-int4 baseline (one-at-a-time marginals are not guaranteed to
    add; the check quantifies how far they are from doing so).

Pin tier for ALL replays here: usage-ranked fill at the trace-header pin
budget with ALL-INT4 sizes (2,584 pairs -- the engine's live pin config),
frozen across counterfactuals and layouts so every comparison sees the
same pinned set. (The Pareto layouts are all-int4-except-K-demotions, so
the int4 pin fill is also their natural fill.)

KILL LINE (external review): Spearman > 0.95 AND Pareto improvement < 2%
at matched distortion => keep the simple greedy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))

import importlib.util


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vso = _load_module("variable_size_cache_oracle",
                   _TOOLS / "variable_size_cache_oracle.py")
qc = sys.modules["quant_container"]

GB = 1e9


# ------------------------------------------------------------- statistics ----
def rankdata_avg(a) -> np.ndarray:
    """Average ranks (1-based) with tie averaging, no scipy."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    rx, ry = rankdata_avg(x), rankdata_avg(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum()) / denom if denom else 0.0


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum()) / denom if denom else 0.0


# ------------------------------------------------------------ counterfacts ----
def layer_miss_bytes(stream, sizes_of, budget) -> int:
    _, mb = vso.replay_byte_lru(stream, sizes_of, budget)
    return mb


def compute_v(candidates, streams_by_trace, base_size_of, int4_b, int2_b,
              budget):
    """{(layer, expert): V_e summed over traces}; one-at-a-time flips,
    replaying only the flipped expert's layer. The manifest-baseline
    replay of each (trace, layer) is shared across its candidates; experts
    absent from every stream get V_e = 0 without any replay."""
    v = {}
    base_cache: dict = {}

    def baseline(t, layer):
        key = (t, layer)
        if key not in base_cache:
            stream = streams_by_trace[t][layer]
            base_cache[key] = layer_miss_bytes(
                stream, lambda e, l=layer: base_size_of(l, e), budget)
        return base_cache[key]

    for layer, expert in candidates:
        total = 0.0
        for t, streams in streams_by_trace.items():
            stream = streams.get(layer)
            if not stream or expert not in stream:
                continue  # absent: flipping its size cannot change the replay
            in_int2 = base_size_of(layer, expert) == int2_b
            flip_to = int4_b if in_int2 else int2_b

            def flipped(e, l=layer, x=expert, s=flip_to):
                return s if e == x else base_size_of(l, e)

            b_flip = layer_miss_bytes(stream, flipped, budget)
            if in_int2:
                b4, b2 = b_flip, baseline(t, layer)
            else:
                b4, b2 = baseline(t, layer), b_flip
            total += b4 - b2
        v[(layer, expert)] = total
    return v


def full_replay(streams_by_trace, size_of, budget) -> int:
    """Total miss bytes across all layers and traces (streams already
    pin-filtered)."""
    total = 0
    for t, streams in streams_by_trace.items():
        for layer, stream in streams.items():
            total += layer_miss_bytes(
                stream, lambda e, l=layer: size_of(l, e), budget)
    return total


def matched_distortion_pick(cands, v, err, k, err_budget):
    """Top-V_e pick of k experts with sum(err) <= err_budget (feasibility-
    aware greedy: accept a candidate only if the cheapest completion still
    fits the budget). -> list of pairs."""
    by_v = sorted(cands, key=lambda p: (-v[p], err[p]))
    by_err = sorted(cands, key=lambda p: (err[p], p))
    chosen: set = set()
    order: list = []
    err_sum = 0.0
    for p in by_v:
        if len(order) >= k:
            break
        if p in chosen:
            continue
        need = k - len(order) - 1
        cheap = 0.0
        cnt = 0
        for q in by_err:
            if cnt >= need:
                break
            if q == p or q in chosen:
                continue
            cheap += err[q]
            cnt += 1
        if err_sum + err[p] + cheap <= err_budget + 1e-12:
            chosen.add(p)
            order.append(p)
            err_sum += err[p]
    for q in by_err:  # completion path (unreachable in practice; tripwired)
        if len(order) >= k:
            break
        if q in chosen:
            continue
        chosen.add(q)
        order.append(q)
        err_sum += err[q]
    if err_sum > err_budget + 1e-9:
        raise AssertionError("matched-distortion pick exceeded the budget")
    return order


# ------------------------------------------------------------------- main ----
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--traces", type=Path, nargs="+", required=True)
    p.add_argument("--usage", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--error-csv", type=Path, required=True)
    p.add_argument("--error-col", default="err_int2")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--per-layer", type=int, default=40,
                   help="sampled candidates per demotable layer (~2000 total)")
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--int2-bytes", type=int, default=None)
    return p.parse_args(argv)


def read_errors(path: Path, col: str) -> dict:
    errors = {}
    with path.open() as f:
        header = f.readline().rstrip("\n").split(",")
        li, ei, ci = header.index("layer"), header.index("expert"), header.index(col)
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= ci:
                continue
            errors[(int(parts[li]), int(parts[ei]))] = float(parts[ci])
    return errors


def main(argv=None) -> int:
    a = parse_args(argv)
    manifest = json.loads(a.manifest.read_text())
    cfg = manifest["model_config"]
    manifest_int2 = set(map(tuple, manifest.get("int2", [])))
    forced = set(manifest.get("force_int4_layers", []))
    int4_b = qc.expert_total_bytes(cfg, 4)
    if a.int2_bytes is not None:
        int2_b = a.int2_bytes
    else:
        int2_b = 0
        for proj in qc.PROJS:
            O, I = qc.expert_shape(cfg, proj)
            int2_b += qc.packed_nbytes(O, I, 2) + O * ((I + 127) // 128) * 2

    errors = read_errors(a.error_csv, a.error_col)
    usage = qc.load_usage(str(a.usage))

    metas, calls_by_trace = {}, {}
    for t in a.traces:
        meta, calls = vso.load_calls(t)
        metas[t] = meta
        calls_by_trace[t] = calls
    meta0 = metas[a.traces[0]]
    pin_bytes = meta0.pinned_units * int4_b
    budget = meta0.lru_per_layer * int4_b

    # frozen pin tier: all-int4 usage fill (the engine's live pin config)
    int4_sizes = {(l, e): int4_b
                  for l in range(cfg["first_dense"], cfg["n_layers"])
                  for e in range(cfg["n_experts"])}
    pinned, pin_used = vso.fill_pins(usage, int4_sizes, pin_bytes)
    print(f"[rate-cache] pins {len(pinned)} pairs ({pin_used/GB:.2f} GB, "
          f"all-int4 fill, frozen) | LRU {budget:,} B/layer | int2 {int2_b:,} B")

    streams_by_trace = {}
    for t, calls in calls_by_trace.items():
        streams = vso.per_layer_streams(calls)
        streams_by_trace[t] = {
            layer: [e for e in stream if (layer, e) not in pinned]
            for layer, stream in streams.items()}

    # stratified sample: per demotable layer, non-pinned experts with a CSV row
    rng = np.random.default_rng(a.seed)
    demotable_layers = [l for l in range(cfg["first_dense"], cfg["n_layers"])
                        if l not in forced]
    candidates = []
    for layer in demotable_layers:
        pool = [(layer, e) for e in range(cfg["n_experts"])
                if (layer, e) not in pinned and (layer, e) in errors]
        take = min(a.per_layer, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        candidates.extend(pool[i] for i in sorted(idx))
    print(f"[rate-cache] {len(candidates)} sampled candidates across "
          f"{len(demotable_layers)} demotable layers")

    def manifest_size_of(l, e):
        return int2_b if (l, e) in manifest_int2 else int4_b

    # primary operating point: all-int4 (where the greedy ranking starts and
    # where the matched-distortion layouts below are replayed); the manifest
    # point is kept as a stability diagnostic
    v = compute_v(candidates, streams_by_trace, lambda l, e: int4_b,
                  int4_b, int2_b, budget)
    v_manifest = compute_v(candidates, streams_by_trace, manifest_size_of,
                           int4_b, int2_b, budget)
    v_arr = np.array([v[p] for p in candidates])
    err_arr = np.array([errors[p] for p in candidates])
    n_zero = int((v_arr == 0).sum())

    rho = spearman(-err_arr, v_arr)  # greedy priority vs cache-value priority
    r_raw = pearson(-err_arr, v_arr)
    rho_stability = spearman(v_arr, np.array([v_manifest[p] for p in candidates]))
    print(f"[rate-cache] Spearman(greedy priority, V_e) = {rho:.4f} "
          f"(Pearson {r_raw:.4f}); {n_zero} candidates never appear "
          f"(V_e = 0 exact); operating-point stability "
          f"Spearman(V_allint4, V_manifest) = {rho_stability:.4f}")

    # matched-distortion Pareto test. Exact matching (tolerance 0) is
    # DEGENERATE by construction: the greedy set minimizes sum(err) for
    # its K, so with budget = greedy's own sum the only feasible K-set is
    # greedy itself. "Matched" therefore means within a small distortion
    # tolerance eps (the err instrument is nearly flat AND invalidated,
    # docs/PERFORMANCE_THEORY.md expert-quant-error-saliency); we sweep eps
    # and use eps = 1% as the headline (stated in the summary).
    a.outdir.mkdir(parents=True, exist_ok=True)
    b_base = full_replay(streams_by_trace, lambda l, e: int4_b, budget)
    pareto = {}
    eps_headline = 0.01
    for frac in (0.25, 0.50):
        k = int(len(candidates) * frac)
        by_err = sorted(candidates, key=lambda p: (errors[p], p))
        greedy_set = by_err[:k]
        err_greedy = sum(errors[p] for p in greedy_set)

        def layout(int2_set):
            s = set(int2_set)
            return lambda l, e: int2_b if (l, e) in s else int4_b

        b_greedy = full_replay(streams_by_trace, layout(greedy_set), budget)
        sweep = []
        for eps in (0.0, 0.001, 0.005, 0.01, 0.02):
            aware_set = matched_distortion_pick(
                candidates, v, errors, k, err_greedy * (1 + eps))
            b_aware = full_replay(streams_by_trace, layout(aware_set), budget)
            additive_pred = b_base - sum(v[p] for p in aware_set)
            sweep.append({
                "eps": eps,
                "err_sum_aware": sum(errors[p] for p in aware_set),
                "err_sum_ratio": (sum(errors[p] for p in aware_set)
                                  / err_greedy if err_greedy else 1.0),
                "overlap": len(set(greedy_set) & set(aware_set)),
                "B_cache_aware": b_aware,
                "pareto_improvement_pct":
                    100 * (b_greedy - b_aware) / b_greedy if b_greedy else 0.0,
                "additivity_rel_err_pct":
                    100 * (additive_pred - b_aware) / b_aware if b_aware else 0.0,
            })
            s = sweep[-1]
            print(f"[rate-cache] K={k} eps={100*eps:4.1f}%: overlap "
                  f"{s['overlap']}/{k}, Pareto "
                  f"{s['pareto_improvement_pct']:+.3f}% (additivity err "
                  f"{s['additivity_rel_err_pct']:+.2f}%)")
        pareto[f"k{frac}"] = {
            "k": k,
            "err_sum_greedy": err_greedy,
            "B_all_int4": b_base,
            "B_greedy": b_greedy,
            "sweep": sweep,
            "headline_eps": eps_headline,
            "pareto_improvement_pct_at_headline_eps": next(
                s["pareto_improvement_pct"] for s in sweep
                if s["eps"] == eps_headline),
        }

    kill = (rho > 0.95
            and all(p["pareto_improvement_pct_at_headline_eps"] < 2
                    for p in pareto.values()))
    verdict = ("KEEP SIMPLE GREEDY (kill line met)" if kill
               else "greedy ranking does NOT reproduce cache value "
                    "(kill line not met)")
    print(f"[rate-cache] VERDICT: {verdict}")

    with (a.outdir / "rate-frontier-values.csv").open("w") as f:
        f.write("layer,expert,in_manifest_int2,err_int2,v_bytes_allint4,"
                "v_bytes_manifest\n")
        for p in candidates:
            f.write(f"{p[0]},{p[1]},{int(p in manifest_int2)},"
                    f"{errors[p]:.6g},{v[p]:.0f},{v_manifest[p]:.0f}\n")
    summary = {
        "n_candidates": len(candidates),
        "per_layer": a.per_layer,
        "seed": a.seed,
        "n_zero_v": n_zero,
        "spearman_greedy_vs_v": rho,
        "pearson_greedy_vs_v": r_raw,
        "spearman_v_allint4_vs_v_manifest": rho_stability,
        "pareto": pareto,
        "kill_line": "Spearman > 0.95 AND Pareto < 2% => keep greedy",
        "kill_line_met": bool(kill),
        "int4_bytes": int4_b, "int2_bytes_repaired": int2_b,
        "pin_pairs": len(pinned), "pin_bytes_used": pin_used,
        "lru_budget_per_layer": budget,
        "notes": "pins frozen at the all-int4 usage fill; V_e computed at the "
                 "280GB-manifest operating point; Pareto layouts are "
                 "all-int4-minus-K-demotions",
    }
    with (a.outdir / "rate-frontier-summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[rate-cache] outputs: {a.outdir}/rate-frontier-values.csv, "
          f"rate-frontier-summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
