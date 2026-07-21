#!/usr/bin/env python3
"""Parse ILI_STALL_TRACE=1 stderr output into the layer-opportunity matrix's
pending-instrumentation columns (bench-m5max/offline-replay-20260715/
layer-opportunity-matrix.csv, produced by tools/layer_opportunity_matrix.py).

WHAT THIS IS: an offline aggregator over STALL_TRACE lines emitted by the
engine patch in tools/patch_m5max_stall_trace.py (ILI_STALL_TRACE=1). Each
input line looks like (one per decode layer per token):

    STALL_TRACE fwd=142 layer=7 nu=6 nhit=4 nmiss=2 resident_start_ms=0.012 \
        resident_finish_ms=0.450 miss_issue_ms=0.015 miss_complete_max_ms=0.980 \
        reduction_start_ms=0.400 exposed_stall_ms=0.530

Any field may read -1.0000 (not captured this sample -- see
tools/patch_m5max_stall_trace.py's sentinel convention: resident_start/finish
and reduction_start are not always resolvable, depending on which of the two
moe() shapes the engine binary was built from). Lines are grouped by `layer`;
per layer this tool computes:

  resident_compute_duration_ms : mean(resident_finish_ms - resident_start_ms)
                                  over samples where BOTH were captured
  final_read_completion_ms     : mean(miss_complete_max_ms) over samples
                                  with nmiss>0 (both relative to route-ready)
  hideable_ms                  : mean per-sample min(resident_duration,
                                  final_read_completion) -- how much of the
                                  miss latency COULD be covered if resident
                                  compute ran for its own natural duration
                                  starting at route-ready (computed
                                  per-sample, then averaged -- not derived
                                  from the two already-averaged columns, to
                                  avoid a Jensen's-inequality mismatch)
  exposed_stall_ms              : mean of the engine's OWN reported
                                  exposed_stall_ms (the most faithful figure:
                                  it is computed engine-side against the true
                                  resident_finish anchor, not reconstructed
                                  here)
  fast_path_eligible             : "true" if this layer's mean exposed_stall_ms
                                  is at or below FAST_PATH_EPSILON_MS, OR
                                  every sample had zero misses; "false"
                                  otherwise; "unknown" if there is no data

These five reuse tools/layer_opportunity_matrix.py's exact column names
(PLACEHOLDER_COLUMNS) and CSV_COLUMNS ordering convention, so a v1 row and a
row from this tool line up field-for-field for every column both files
share. This tool ADDS the exposed-stall distribution and CVaR columns the
2026-07-15 build task also asked for (mean/median/p95/max/cvar95, see
EXTRA_COLUMNS) -- v1 has no such columns, so this is a strict superset, never
edited into v1 (--out defaults to a NEW file next to it, never
layer-opportunity-matrix.csv itself).

CVaR (Conditional Value at Risk) at level alpha is the mean of the worst
(1-alpha) tail of a sample -- e.g. cvar95 is the mean of the worst 5% of
observed exposed_stall_ms values for that layer. With few samples the "worst
5%" can be a single value; this is noted, not hidden (see `n_samples` in the
output and cvar()'s docstring).

Usage:
  ILI_STALL_TRACE=1 <engine> ... 2>trace.log
  python3 c/tools/parse_stall_trace.py --log trace.log \
      --out c/bench-m5max/offline-replay-20260715/layer-opportunity-matrix-stall.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

TRACE_RE = re.compile(r"^STALL_TRACE\s+(.*)$")
FIELD_RE = re.compile(r"(\w+)=(-?[\d.]+)")
INT_FIELDS = {"fwd", "layer", "nu", "nhit", "nmiss"}
FAST_PATH_EPSILON_MS = 0.01  # <= this mean exposed stall counts as "already hidden"


def _load_layer_opportunity_matrix():
    spec = importlib.util.spec_from_file_location(
        "layer_opportunity_matrix", HERE / "layer_opportunity_matrix.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LOM = _load_layer_opportunity_matrix()
EXTRA_COLUMNS = [
    "n_samples", "exposed_stall_mean_ms", "exposed_stall_median_ms",
    "exposed_stall_p95_ms", "exposed_stall_max_ms", "exposed_stall_cvar95_ms",
]
CSV_COLUMNS = LOM.CSV_COLUMNS + EXTRA_COLUMNS


def parse_line(line: str) -> dict | None:
    m = TRACE_RE.match(line.strip())
    if not m:
        return None
    record = {}
    for key, value in FIELD_RE.findall(m.group(1)):
        record[key] = int(value) if key in INT_FIELDS else float(value)
    missing = {"layer", "nhit", "nmiss", "resident_start_ms", "resident_finish_ms",
               "miss_issue_ms", "miss_complete_max_ms", "reduction_start_ms",
               "exposed_stall_ms"} - set(record)
    if missing:
        raise ValueError(f"STALL_TRACE line missing field(s) {sorted(missing)}: {line!r}")
    return record


def parse_logs(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = parse_line(line)
                if record is not None:
                    records.append(record)
    return records


def cvar(values: list[float], alpha: float = 0.95) -> float:
    """Mean of the worst (1-alpha) tail. With < 1/(1-alpha) samples the tail
    is a single (the worst) value -- a valid, if noisy, estimate; callers
    needing a stable CVaR should collect more samples, not distrust this."""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    tail_n = max(1, round(n * (1 - alpha)))
    tail = ordered[n - tail_n:]
    return sum(tail) / len(tail)


def _captured(value: float) -> bool:
    """True if this ms-relative field was actually captured this sample.
    NOTE the sentinel here is -1 (tools/patch_m5max_stall_trace.py's emitted-
    line convention), NOT 0 -- 0.0 is a legitimate "right at route-ready"
    timestamp once converted to ms-relative-to-route-ready, unlike the
    engine's OWN internal raw-seconds fields (where 0 means "unset", since a
    real CLOCK_MONOTONIC reading is never exactly 0)."""
    return value >= 0


def aggregate_layer(records: list[dict]) -> dict:
    exposed = [r["exposed_stall_ms"] for r in records]
    resident_durations = [
        r["resident_finish_ms"] - r["resident_start_ms"] for r in records
        if _captured(r["resident_start_ms"]) and _captured(r["resident_finish_ms"])
    ]
    final_reads = [r["miss_complete_max_ms"] for r in records if r["nmiss"] > 0]
    hideable_samples = []
    for r in records:
        if r["nmiss"] == 0:
            continue
        rd = (r["resident_finish_ms"] - r["resident_start_ms"]
              if _captured(r["resident_start_ms"]) and _captured(r["resident_finish_ms"]) else 0.0)
        hideable_samples.append(min(rd, r["miss_complete_max_ms"]))

    all_zero_miss = all(r["nmiss"] == 0 for r in records)
    mean_exposed = statistics.fmean(exposed) if exposed else 0.0
    fast_path = "true" if (all_zero_miss or mean_exposed <= FAST_PATH_EPSILON_MS) else "false"

    return {
        "resident_compute_duration_ms": (
            round(statistics.fmean(resident_durations), 4) if resident_durations else PENDING_NA),
        "final_read_completion_ms": (
            round(statistics.fmean(final_reads), 4) if final_reads else PENDING_NA),
        "hideable_ms": (
            round(statistics.fmean(hideable_samples), 4) if hideable_samples else PENDING_NA),
        "exposed_stall_ms": round(mean_exposed, 4),
        "fast_path_eligible": fast_path,
        "n_samples": len(records),
        "exposed_stall_mean_ms": round(mean_exposed, 4),
        "exposed_stall_median_ms": round(statistics.median(exposed), 4) if exposed else 0.0,
        "exposed_stall_p95_ms": round(_percentile(exposed, 0.95), 4) if exposed else 0.0,
        "exposed_stall_max_ms": round(max(exposed), 4) if exposed else 0.0,
        "exposed_stall_cvar95_ms": round(cvar(exposed, 0.95), 4) if exposed else 0.0,
    }


PENDING_NA = "no-miss-or-resident-samples"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def build_rows(records: list[dict]) -> list[dict]:
    by_layer: dict[int, list[dict]] = {}
    for r in records:
        by_layer.setdefault(r["layer"], []).append(r)
    rows = []
    for layer in sorted(by_layer):
        agg = aggregate_layer(by_layer[layer])
        row = {col: "" for col in CSV_COLUMNS}
        row["layer"] = layer
        row.update(agg)
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, action="append", required=True,
                        help="stderr log containing STALL_TRACE lines (repeatable)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output CSV path -- MUST NOT be the v1 "
                             "layer-opportunity-matrix.csv (refused if so)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.out.name == "layer-opportunity-matrix.csv":
        raise SystemExit(
            "refusing to write to layer-opportunity-matrix.csv (the v1 file) -- "
            "use a distinct --out name, e.g. layer-opportunity-matrix-stall.csv")
    records = parse_logs(args.log)
    if not records:
        print(f"[parse-stall-trace] no STALL_TRACE lines found in {args.log}")
        sys.exit(1)
    rows = build_rows(records)
    write_csv(rows, args.out)
    print(f"[parse-stall-trace] {len(records)} STALL_TRACE lines across "
          f"{len(rows)} layer(s) -> {args.out}")
    for row in rows:
        print(f"  layer {row['layer']}: n={row['n_samples']} "
              f"exposed_mean={row['exposed_stall_mean_ms']}ms "
              f"exposed_p95={row['exposed_stall_p95_ms']}ms "
              f"cvar95={row['exposed_stall_cvar95_ms']}ms "
              f"fast_path_eligible={row['fast_path_eligible']}")


if __name__ == "__main__":
    main()
