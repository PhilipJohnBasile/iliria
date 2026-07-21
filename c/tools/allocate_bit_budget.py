#!/usr/bin/env python3
"""Global bit-budget allocator for the mixed-precision (int4/int2) container.

GEMQ-style global allocation: given the per-expert reconstruction-error CSV
from measure_expert_quant_error.py and a target container size, keep the
early layers (default 3-27, per Study 2: quality-critical AND untierable by
routing mass) at int4 and demote the LOWEST-error mid/late experts to int2
until the target fits. With every demotion saving the same bytes, greedy
ascending-error IS the optimal knapsack; the code still ranks by
error-per-byte-saved so unequal sizes stay correct.

Routing mass (.fa_usage) is NEVER folded into the ranking (MoPEQ + our #47:
frequency misleads); it is reported as int4-tier mass coverage only.

Prediction: with --traces, replays route traces through the Study-1
byte-budget cache simulator and maps the miss-byte ratio through the
disk model fitted to BOTH measured K6-matrix points (sublinear), printing
predicted tok/s per candidate size.

Usage (CPU-only; safe while the bench runs ONLY if traces/CSV are local):
  python3 c/tools/allocate_bit_budget.py \
      --csv c/bench-m5max/container-20260715/expert-quant-error.csv \
      --model /path/to/models/GLM-5.2-int4-with-int8-mtp \
      --container /path/to/models/GLM-5.2-int4-with-int8-mtp \
      --targets 260,265,270,273,280 \
      --traces c/bench-m5max/overnight-20260714-001239/routes-baseline.bin,... \
      --emit manifest-270.json --emit-gb 270
"""

import argparse
import glob as globmod
import importlib.util
import json
import os
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_container as qc

GB = 1e9

# Study-1 disk model (the offline math-verdict record, tiered_sim2.py): measured
# K6-matrix warm medians per prompt: (tok/s off, hit% off, disk s off, tok/s
# k6, hit% k6, disk s k6) over TOKENS decoded. disk_s = a * miss_bytes^b
# fitted through both points per prompt (sublinear b~0.34-0.51); b=1 linear
# is the optimistic bracket. Cache geometry comes from the trace-v2 metadata
# (pin 2584 int4-slots ~48.9 GB + LRU 34 int4-slots/layer ~643 MB/layer =
# ~97 GB), which reproduces the measured 64.7-66.1% baseline hit rate.
MEASURED = {
    "godot_controller": (1.440, 64.7, 36.31, 1.330, 77.8, 28.62),
    "rust_queue": (1.420, 66.1, 38.46, 1.410, 79.5, 30.87),
    "typescript_websocket": (1.580, 65.8, 33.69, 1.470, 78.9, 28.61),
}
TOKENS = 112


def disk_fits():
    """{prompt: (b, nondisk_s, disk_s_base, tok_s_base)}."""
    import math
    fits = {}
    for prompt, (ts0, h0, d0, ts1, h1, d1) in MEASURED.items():
        m0, m1 = (100 - h0) / 100, (100 - h1) / 100
        b = math.log(d0 / d1) / math.log(m0 / m1)
        fits[prompt] = (b, TOKENS / ts0 - d0, d0, ts0)
    return fits


def simulate_bytes(requests, sizes, pinned, layer_budget):
    """Per-layer byte-capacity LRU with a global pinned set (Study 1 replica).
    -> (lookups, hits, misses, miss_bytes)."""
    n = hits = misses = 0
    miss_bytes = 0
    for layer in sorted(requests):
        cache = OrderedDict()
        used = 0
        for expert in requests[layer]:
            n += 1
            if (layer, expert) in pinned:
                hits += 1
                continue
            if expert in cache:
                hits += 1
                cache.move_to_end(expert)
                continue
            misses += 1
            sz = sizes[(layer, expert)]
            miss_bytes += sz
            if layer_budget <= 0:
                continue
            while cache and used + sz > layer_budget:
                _, vsz = cache.popitem(last=False)
                used -= vsz
            if used + sz <= layer_budget:
                cache[expert] = sz
                used += sz
    return n, hits, misses, miss_bytes


def read_error_csv(path, error_col):
    """-> {(layer, expert): error}."""
    errors = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
        if error_col not in header:
            raise SystemExit(f"column '{error_col}' not in {path} (has: {header})")
        li, ei, ci = header.index("layer"), header.index("expert"), header.index(error_col)
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= ci:
                continue
            errors[(int(parts[li]), int(parts[ei]))] = float(parts[ci])
    return errors


def container_file_bytes(container_dir):
    """stat-only (zero data reads): total bytes of every *.safetensors."""
    return sum(os.path.getsize(p)
               for p in globmod.glob(os.path.join(container_dir, "*.safetensors")))


def allocate(cfg, errors, forced_layers, target_bytes, nonexpert_bytes,
             allow_partial=False):
    """Greedy demotion by error-per-byte-saved. -> dict with the allocation."""
    b4 = qc.expert_total_bytes(cfg, 4)
    b2 = qc.expert_total_bytes(cfg, 2)
    all_experts = qc.routed_experts(cfg)
    demotable = [k for k in all_experts if k[0] not in forced_layers]
    missing = [k for k in demotable if k not in errors]
    if missing and not allow_partial:
        raise SystemExit(
            f"{len(missing)} demotable experts missing from the error CSV "
            f"(first: {missing[0]}); finish the measurement pass or pass "
            "--allow-partial (missing experts then stay int4).")
    ranked = sorted((k for k in demotable if k in errors),
                    key=lambda k: (errors[k] / (b4 - b2), k))
    size = nonexpert_bytes + len(all_experts) * b4
    int2 = []
    for k in ranked:
        if size <= target_bytes:
            break
        int2.append(k)
        size -= b4 - b2
    feasible = size <= target_bytes
    int2_set = set(int2)
    kept = [errors[k] for k in ranked if k not in int2_set]
    return {
        "int2": sorted(int2),
        "size_bytes": size,
        "feasible": feasible,
        "n_experts": len(all_experts),
        "n_int2": len(int2),
        "n_forced": len(all_experts) - len(demotable),
        "n_missing": len(missing),
        "max_demoted_err": max((errors[k] for k in int2), default=0.0),
        "min_kept_err": min(kept, default=float("inf")),
        "bytes_int4": b4, "bytes_int2": b2,
    }


def mass_coverage(usage, int2_set, cfg):
    """Fraction of routed mass landing on int4 experts (reporting only)."""
    if not usage:
        return None
    tot = int4 = 0
    for (l, e), c in usage.items():
        if not (cfg["first_dense"] <= l < cfg["n_layers"]):
            continue
        tot += c
        if (l, e) not in int2_set:
            int4 += c
    return int4 / tot if tot else None


def predict_throughput(traces, usage_path, alloc, cfg, fits):
    """Replay traces through the byte-LRU sim; predicted tok/s vs baseline."""
    sim_path = os.path.join(qc.TOOLS_DIR, "simulate_m5max_cache.py")
    spec = importlib.util.spec_from_file_location("simulate_m5max_cache", sim_path)
    sim = importlib.util.module_from_spec(spec)
    sys.modules["simulate_m5max_cache"] = sim
    spec.loader.exec_module(sim)
    from pathlib import Path

    usage = sim.load_usage(Path(usage_path))
    ranked = sorted(usage, key=lambda p: (-usage[p], p[0], p[1]))
    b4, b2 = alloc["bytes_int4"], alloc["bytes_int2"]
    int2_set = set(map(tuple, alloc["int2"]))
    lines = []
    for tpath in traces:
        requests, sel, calls, meta = sim.load_engine_requests(Path(tpath))
        # calibrated live geometry (tiered_sim2.py): pin slots + per-layer LRU
        # slots from the trace metadata, both expressed in int4 bytes
        pin_budget = meta.pinned_units * b4
        layer_budget = meta.lru_per_layer * b4

        def run(sizes):
            pinned, used = set(), 0.0
            for p in ranked:
                sz = sizes.get(p, b2)
                if used + sz > pin_budget:
                    break
                pinned.add(p)
                used += sz
            return simulate_bytes(requests, sizes, pinned, layer_budget)

        base_sizes = {(l, e): b4 for l in range(cfg["n_layers"] + 1)
                      for e in range(cfg["n_experts"])}
        mix_sizes = {k: (b2 if k in int2_set else b4) for k in base_sizes}
        n0, h0, _, mb0 = run(base_sizes)
        n1, h1, _, mb1 = run(mix_sizes)
        ratio = mb1 / mb0 if mb0 else 1.0
        deltas = sorted(
            100 * ((TOKENS / (nd + d0 * ratio ** b)) / ts0 - 1)
            for (b, nd, d0, ts0) in fits.values())
        toks = sorted(
            TOKENS / (nd + d0 * ratio ** b) for (b, nd, d0, ts0) in fits.values())
        lin = sorted(
            100 * ((TOKENS / (nd + d0 * ratio)) / ts0 - 1)
            for (b, nd, d0, ts0) in fits.values())[1]
        lines.append(
            f"    trace {os.path.basename(tpath)}: hit {100 * h0 / n0:.2f}% -> "
            f"{100 * h1 / n1:.2f}%, miss-byte ratio {ratio:.3f}, "
            f"pred {toks[1]:.2f} tok/s med ({deltas[1]:+.1f}% sublin, {lin:+.1f}% lin)")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True, help="error CSV from measure_expert_quant_error.py")
    ap.add_argument("--model", required=True, help="container dir (for config.json)")
    ap.add_argument("--container", default=None,
                    help="container dir to size non-expert bytes by stat (no data reads)")
    ap.add_argument("--nonexpert-gb", type=float, default=None,
                    help="non-expert container bytes in GB (overrides --container)")
    ap.add_argument("--targets", default="260,265,270,273,280",
                    help="target container sizes in GB, comma-separated")
    ap.add_argument("--force-int4-layers", default="3-27",
                    help="layers pinned int4 regardless of error (default 3-27, Study 2)")
    ap.add_argument("--error-col", default="err_int2",
                    help="CSV column to rank by (default err_int2)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="tolerate missing CSV rows (those experts stay int4)")
    ap.add_argument("--usage", default=None,
                    help=".fa_usage for mass-coverage reporting (default <model>/.fa_usage)")
    ap.add_argument("--usage-coding", default=None)
    ap.add_argument("--traces", default=None,
                    help="route-trace .bin files (comma-separated) for throughput prediction")
    ap.add_argument("--sim-usage", default=None,
                    help="usage histogram for the sim's pin ranking "
                         "(default <model>/.fa_usage.coding, as in Study 1)")
    ap.add_argument("--emit", default=None, help="write the manifest JSON here")
    ap.add_argument("--emit-gb", type=float, default=None,
                    help="target size the emitted manifest is solved for")
    a = ap.parse_args(argv)

    cfg = qc.load_config(a.model)
    errors = read_error_csv(a.csv, a.error_col)
    forced = qc.parse_layers(a.force_int4_layers)
    bad = [l for l in forced if not (cfg["first_dense"] <= l < cfg["n_layers"])]
    if bad:
        raise SystemExit(f"--force-int4-layers outside routed range: {sorted(bad)}")

    b4 = qc.expert_total_bytes(cfg, 4)
    n_all = len(qc.routed_experts(cfg))
    if a.nonexpert_gb is not None:
        nonexpert = a.nonexpert_gb * GB
    elif a.container:
        nonexpert = container_file_bytes(a.container) - n_all * b4
        if nonexpert < 0:
            raise SystemExit("container smaller than all-int4 experts - wrong dir?")
    else:
        nonexpert = 0.0
        print("WARNING: no --container/--nonexpert-gb; targets are expert bytes only",
              file=sys.stderr)

    usage_path = a.usage or os.path.join(a.model, ".fa_usage")
    coding_path = a.usage_coding or os.path.join(a.model, ".fa_usage.coding")
    usage = qc.load_usage(usage_path) if os.path.exists(usage_path) else {}
    coding = qc.load_usage(coding_path) if os.path.exists(coding_path) else {}
    fits = disk_fits()
    traces = [t for t in (a.traces.split(",") if a.traces else []) if t]

    targets = sorted(float(t) for t in a.targets.split(","))
    if a.emit_gb is not None and a.emit_gb not in targets:
        targets.append(a.emit_gb)
        targets.sort()

    print(f"[allocate] {n_all} routed experts (L{cfg['first_dense']}-"
          f"{cfg['n_layers'] - 1}), int4 {b4 / 1e6:.3f} MB / int2 "
          f"{qc.expert_total_bytes(cfg, 2) / 1e6:.3f} MB each; non-expert "
          f"{nonexpert / GB:.1f} GB; all-int4 container "
          f"{(nonexpert + n_all * b4) / GB:.1f} GB; forced int4 layers "
          f"{a.force_int4_layers} ({len(forced)} layers); error col {a.error_col}")

    emitted = None
    for tgt in targets:
        alloc = allocate(cfg, errors, forced, tgt * GB, nonexpert, a.allow_partial)
        int2_set = set(map(tuple, alloc["int2"]))
        cov_g = mass_coverage(usage, int2_set, cfg)
        cov_c = mass_coverage(coding, int2_set, cfg)
        flag = "" if alloc["feasible"] else "  INFEASIBLE (even all-int2 too big)"
        print(f"\n== target {tgt:.0f} GB -> {alloc['size_bytes'] / GB:.1f} GB, "
              f"int2 {alloc['n_int2']}/{n_all - alloc['n_forced']} demotable "
              f"({alloc['n_forced']} forced int4){flag}")
        print(f"    knife edge: max demoted err {alloc['max_demoted_err']:.4g} "
              f"vs min kept err {alloc['min_kept_err']:.4g}")
        if cov_g is not None:
            print(f"    int4-tier mass coverage: general {100 * cov_g:.1f}%"
                  + (f", coding {100 * cov_c:.1f}%" if cov_c is not None else ""))
        if alloc["n_missing"]:
            print(f"    WARNING: {alloc['n_missing']} experts missing from CSV kept int4")
        if traces and alloc["feasible"]:
            for line in predict_throughput(
                    traces,
                    a.sim_usage or os.path.join(a.model, ".fa_usage.coding"),
                    alloc, cfg, fits):
                print(line)
        if a.emit and a.emit_gb is not None and tgt == a.emit_gb:
            emitted = alloc

    if a.emit:
        if a.emit_gb is None:
            raise SystemExit("--emit requires --emit-gb")
        if not emitted["feasible"]:
            raise SystemExit(f"refusing to emit an infeasible manifest for {a.emit_gb} GB")
        manifest = {
            "version": 1,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error_csv": os.path.abspath(a.csv),
            "error_col": a.error_col,
            "target_gb": a.emit_gb,
            "predicted_gb": emitted["size_bytes"] / GB,
            "force_int4_layers": sorted(forced),
            "model_config": cfg,
            "default_bits": 4,
            "int2": [list(k) for k in emitted["int2"]],
        }
        with open(a.emit, "w") as f:
            json.dump(manifest, f)
        print(f"\n[allocate] manifest written: {a.emit} "
              f"({emitted['n_int2']} experts -> int2, "
              f"{emitted['size_bytes'] / GB:.1f} GB predicted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
