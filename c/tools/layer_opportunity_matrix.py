#!/usr/bin/env python3
"""Assemble the 75-row layer-opportunity matrix (V1) from the committed
byte-capacity-aware replay CSVs at bench-m5max/offline-replay-20260715/.

WHAT THIS IS: a per-layer catalog of which future optimization category looks
most promising for each of the 75 MoE-routed layers (indices 3..77 -- layers
0-2 are dense/non-routed, see missstruct-pm.csv), built ENTIRELY from data the
replay already produced (`missstruct-pm.csv`'s per-layer P(M=0..8)). No new
engine instrumentation, no engine run: this is offline analytics over
already-committed CSVs, same as the replay itself
(the offline-replay results).

WHAT THIS IS NOT: a final verdict. Five of its columns are explicitly marked
'pending instrumentation' because they need real timestamps from a resident-
compute/overlap instrumentation pass that does not exist yet (see the review's
F1/F2 gate structure and campaign-log.md 2026-07-15 11:58:28). The
`layer_class` column is provisional -- a V1 prioritization, not a commitment.

SOURCE DATA SCOPE (documented, not hidden): `missstruct-pm.csv` carries 4
layouts (A/B/C/D) x 3 traces (routes-baseline/coding/shotgun) x 77 layer rows
(75 numeric + ALL/ALL_STEADY). This matrix uses layout A only (today's actual
uniform-int4 container -- see config.json's layouts.A / results.md's "Layouts"
table) because it is the only layout with a layer-independent per-expert byte
size (`int4_bytes` in config.json), which the byte-derivation below depends
on. Layouts B/C/D mix int2/int4 per expert per the allocation manifest, whose
per-layer expert-format split is not exposed in the committed CSVs, so a
byte-accurate per-layer figure for those layouts is left for V2 (would need
the allocation manifest joined in). P(M=*) is averaged across the 3 traces
per layer (equal weight -- the traces are equally realistic daily-driver
samples, see results.md's inputs section); per-trace and per-layout figures
remain directly available in missstruct-pm.csv for anyone who wants to slice
differently.

BYTE-COST DERIVATION (V1, layout A only, expert_bytes = 18,915,328 B/expert,
`int4_bytes` in config.json):
    mean_miss_bytes(layer) = expert_bytes * sum_{m=0}^{8} m * P(M=m | layer)
    p95_miss_bytes(layer)  = expert_bytes * p95_m(layer)
  where p95_m is the smallest m such that the layer's cumulative P(M<=m) >=
  0.95 (the empirical 95th percentile of the per-call miss COUNT, not of the
  byte total directly -- equivalent under layout A's uniform per-expert size).
  This is an expectation over the SAME replay classification the rest of the
  campaign uses (call-entry, pre-insertion, batch-union; see results.md), not
  a new measurement.

CLASSIFICATION THRESHOLDS (V1 -- provisional, chosen from the actual layer-3..
77 distribution under layout A; see the header of layer-matrix-summary.md for
the one-paragraph rationale). Evaluated in this priority order per layer
(first match wins), because a fat right tail matters even when the mean case
looks fine, and a layer only gets called a fusion candidate once its tail
risk has been ruled out:

  1. tail-protection : p95_m(layer)      >= TAIL_P95_M_THRESHOLD   (6)
  2. fusion-first     : p_m0(layer)       >  FUSION_P_M0_THRESHOLD  (0.30)
  3. overlap-first     : p_le2(layer)      >= OVERLAP_P_LE2_THRESHOLD (0.60)
  4. capacity-first    : p_ge4(layer)      >= CAPACITY_P_GE4_THRESHOLD(0.30)
  5. mixed             : none of the above decisively -- an honest V1 "needs
                         case-by-case review" bucket, not a bug. Forcing a
                         4-way partition on every layer would be false
                         precision; see layer-matrix-summary.md for the count.

Rationale in one line per bucket: fusion-first wants "usually zero misses" (a
FULL-LAYER fused command buffer / persistent-Metal-state path pays off only
when it can skip the disk round trip most of the time -- glm.c's `g_metal4_moe`
absorbed-decode path); overlap-first wants "rarely zero but rarely many either"
(1-2 misses are cheap to hide behind the next layer's compute -- the
arrival-order overlap program); capacity-first wants "often 4+ misses" (no
amount of overlap/fusion helps if the working set structurally does not fit --
needs more resident bytes, i.e. the mixed-precision container upgrade);
tail-protection wants "occasionally almost everything misses" (a rare-but-huge
stall that the mean-case bucket doesn't capture and that needs an explicit
worst-case mitigation, e.g. a bounded-wait fallback, regardless of which
mean-case bucket the layer is otherwise in).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_C_DIR = HERE.parent
DEFAULT_REPLAY_DIR = REPO_C_DIR / "bench-m5max" / "offline-replay-20260715"

MAX_M = 8   # GLM-5.2 topk=8: a MoE call can miss at most 8 of its routed experts
FIRST_MOE_LAYER = 3    # layers 0-2 are dense (first_k_dense_replace), never routed

# ---- classification thresholds (V1; see module docstring for derivation) -----------------
TAIL_P95_M_THRESHOLD = 6        # p95(M) >= 6 of 8 experts missing: extreme tail
FUSION_P_M0_THRESHOLD = 0.30    # P(M=0) > 30%: usually nothing to fetch
OVERLAP_P_LE2_THRESHOLD = 0.60  # P(M<=2) >= 60%: usually cheap to hide
CAPACITY_P_GE4_THRESHOLD = 0.30 # P(M>=4) >= 30%: often structurally short of room

PENDING = "pending instrumentation"
PLACEHOLDER_COLUMNS = [
    "resident_compute_duration_ms",
    "final_read_completion_ms",
    "hideable_ms",
    "exposed_stall_ms",
    "fast_path_eligible",
]

CSV_COLUMNS = (
    ["layer", "p_m0", "p_m1", "p_m2", "p_le2", "p_ge4", "p95_m",
     "mean_miss_bytes", "p95_miss_bytes"]
    + PLACEHOLDER_COLUMNS
    + ["layer_class"]
)


def load_pm_rows(replay_dir: Path, layout: str) -> list[dict]:
    path = replay_dir / "missstruct-pm.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [r for r in rows if r["layout"] == layout and r["layer"] not in ("ALL", "ALL_STEADY")]
    if not selected:
        raise ValueError(f"no per-layer rows found for layout {layout!r} in {path}")
    return selected


def expert_bytes_for_layout(replay_dir: Path, layout: str) -> float:
    with open(replay_dir / "config.json", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if layout != "A":
        raise ValueError(
            f"layout {layout!r} has no layer-independent per-expert byte size in the "
            "committed CSVs (mixed int2/int4 per the allocation manifest) -- V1 byte "
            "derivation only supports layout A. See the module docstring.")
    return float(cfg["int4_bytes"])


def per_layer_pm(rows: list[dict]) -> dict[int, list[float]]:
    """layer -> [P(M=0)..P(M=8)] averaged across every trace present for this layout."""
    by_layer: dict[int, list[list[float]]] = {}
    for row in rows:
        layer = int(row["layer"])
        pm = [float(row[f"p_m{m}"]) for m in range(MAX_M + 1)]
        by_layer.setdefault(layer, []).append(pm)
    averaged = {}
    for layer, samples in by_layer.items():
        n = len(samples)
        averaged[layer] = [sum(s[m] for s in samples) / n for m in range(MAX_M + 1)]
    return averaged


def p95_of_m(pm: list[float]) -> int:
    """Smallest m such that cumulative P(M<=m) >= 0.95 (empirical 95th percentile of M)."""
    cumulative = 0.0
    for m, p in enumerate(pm):
        cumulative += p
        if cumulative >= 0.95:
            return m
    return MAX_M


def classify(p_m0: float, p_le2: float, p_ge4: float, p95_m: int) -> str:
    if p95_m >= TAIL_P95_M_THRESHOLD:
        return "tail-protection"
    if p_m0 > FUSION_P_M0_THRESHOLD:
        return "fusion-first"
    if p_le2 >= OVERLAP_P_LE2_THRESHOLD:
        return "overlap-first"
    if p_ge4 >= CAPACITY_P_GE4_THRESHOLD:
        return "capacity-first"
    return "mixed"


def build_matrix(replay_dir: Path, layout: str) -> list[dict]:
    rows = load_pm_rows(replay_dir, layout)
    expert_bytes = expert_bytes_for_layout(replay_dir, layout)
    pm_by_layer = per_layer_pm(rows)

    matrix = []
    for layer in sorted(pm_by_layer):
        pm = pm_by_layer[layer]
        p_m0, p_m1, p_m2 = pm[0], pm[1], pm[2]
        p_le2 = p_m0 + p_m1 + p_m2
        p_ge4 = sum(pm[4:MAX_M + 1])
        p95_m = p95_of_m(pm)
        mean_miss_bytes = expert_bytes * sum(m * pm[m] for m in range(MAX_M + 1))
        p95_miss_bytes = expert_bytes * p95_m
        layer_class = classify(p_m0, p_le2, p_ge4, p95_m)

        record = {
            "layer": layer,
            "p_m0": round(p_m0, 6), "p_m1": round(p_m1, 6), "p_m2": round(p_m2, 6),
            "p_le2": round(p_le2, 6), "p_ge4": round(p_ge4, 6), "p95_m": p95_m,
            "mean_miss_bytes": round(mean_miss_bytes, 1),
            "p95_miss_bytes": round(p95_miss_bytes, 1),
            "layer_class": layer_class,
        }
        for column in PLACEHOLDER_COLUMNS:
            record[column] = PENDING
        matrix.append(record)
    return matrix


def write_csv(matrix: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in matrix:
            writer.writerow(record)


def class_distribution(matrix: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in matrix:
        counts[record["layer_class"]] = counts.get(record["layer_class"], 0) + 1
    return counts


def write_summary(matrix: list[dict], layout: str, path: Path) -> None:
    counts = class_distribution(matrix)
    order = ["fusion-first", "overlap-first", "capacity-first", "tail-protection", "mixed"]
    total = len(matrix)
    p_m0_vals = [r["p_m0"] for r in matrix]
    lines = [
        "# Layer opportunity matrix V1 -- class distribution summary",
        "",
        f"Source: `missstruct-pm.csv`, layout {layout}, {total} MoE-routed layers "
        f"({FIRST_MOE_LAYER}..{FIRST_MOE_LAYER + total - 1}), averaged over 3 traces "
        "(routes-baseline/coding/shotgun). Byte figures use layout A's uniform "
        "`int4_bytes` per-expert size (config.json). Full methodology and thresholds: "
        "`tools/layer_opportunity_matrix.py` module docstring.",
        "",
        "| class | layers | share |",
        "|---|---|---|",
    ]
    for cls in order:
        n = counts.get(cls, 0)
        lines.append(f"| {cls} | {n} | {n / total:.1%} |")
    lines += [
        "",
        f"Mean P(M=0) across all {total} layers: {statistics.mean(p_m0_vals):.3f} -- "
        "consistent with results.md's system-wide finding that a fused all-hit fast "
        "path is premature today; the fusion-first bucket above is the specific "
        "minority of layers where it already looks viable.",
        "",
        "All 5 timing/eligibility columns in the matrix are placeholders "
        f"(`{PENDING}`) pending a resident-compute/overlap instrumentation pass; "
        "`layer_class` is a V1 prioritization from P(M) shape alone, not a final call.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR,
                        help="directory containing missstruct-pm.csv and config.json")
    parser.add_argument("--layout", default="A",
                        help="which missstruct-pm.csv layout to use (V1 supports A only)")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="default: <replay-dir>/layer-opportunity-matrix.csv")
    parser.add_argument("--out-md", type=Path, default=None,
                        help="default: <replay-dir>/layer-matrix-summary.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_csv = args.out_csv or (args.replay_dir / "layer-opportunity-matrix.csv")
    out_md = args.out_md or (args.replay_dir / "layer-matrix-summary.md")

    matrix = build_matrix(args.replay_dir, args.layout)
    write_csv(matrix, out_csv)
    write_summary(matrix, args.layout, out_md)

    counts = class_distribution(matrix)
    print(f"wrote {out_csv} ({len(matrix)} layers) and {out_md}")
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()
