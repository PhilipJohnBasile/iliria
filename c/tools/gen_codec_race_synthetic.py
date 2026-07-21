#!/usr/bin/env python3
"""Synthetic int4-row corpus for the row-independent entropy-codec race
(docs/PERFORMANCE_THEORY.md n1: "n1_codec_race_requirement" /
"n1_row_independent_coding_caveat_for_mode3").

Generates packed int4 nibble data SHAPED LIKE the real census
(c/bench-m5max/new-math-20260715/census-tensors.csv): per (band, proj) cell,
a target symbol entropy is bootstrapped from that cell's own real measured
h_global distribution (~85 real per-tensor entropy values per cell), then a
16-symbol probability vector hitting that target entropy is solved for (a
quantized-Gaussian family, peaked at the center codes the way a uniformly
quantized approximately-Gaussian/Laplacian weight distribution looks,
bisected on sigma) and sampled to produce a full-size synthetic tensor.

Row shapes match the census exactly: gate/up tensors are O=2048 rows of
I=6144 symbols (3072 B packed/row, "gate/up rows ~3KB"); down tensors are
O=6144 rows of I=2048 symbols (1024 B packed/row, "down rows ~1KB") -- both
shapes pack to the SAME 6,291,456 B/tensor the census's own census-
tensors.csv reports for every one of its 768 sampled tensors (confirmed
against the container-design plan's 18,915,328 B/expert total, which solves
exactly at hidden=6144, moe_inter=2048: 1.5*moe_inter*hidden + 8*moe_inter +
4*hidden == 18,915,328). No /path/to/models read, today or ever, by this
script -- the only input is the already-committed census CSV.

Usage:
  python3 c/tools/gen_codec_race_synthetic.py \\
      --outdir c/bench-m5max/codec-race-20260715 --tensors-per-group 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

BANDS = ("early", "mid", "late")
PROJS = ("gate", "up", "down")

# [O, I] per projection -- see module docstring for the derivation.
ROW_SHAPE = {
    "gate": (2048, 6144),
    "up": (2048, 6144),
    "down": (6144, 2048),
}

DEFAULT_CENSUS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "bench-m5max",
    "new-math-20260715", "census-tensors.csv")


def load_census_h(census_csv):
    """{(band,proj): [h_global, ...]} bootstrap pool from the committed census."""
    by_group = {}
    with open(census_csv, newline="") as f:
        for rec in csv.DictReader(f):
            key = (rec["band"], rec["proj"])
            by_group.setdefault(key, []).append(float(rec["h_global"]))
    for key, vals in by_group.items():
        if not vals:
            raise ValueError(f"census csv has no h_global rows for {key}")
    return by_group


def entropy_of(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def dist_for_entropy(target_h, tol=1e-4, max_iter=60):
    """16-symbol probability vector (quantized-Gaussian shape) with entropy
    == target_h within `tol` bits, found by bisection on the Gaussian sigma
    (entropy is monotonic increasing in sigma over the search range: ->0 as
    sigma->0+, ->4.0 as sigma grows). Peaked at symbol 8, matching glm.c's
    signed int4 convention (value = symbol-8, so a real weight distribution's
    mode at value 0 lands exactly on symbol 8) -- centering on a single
    integer (not the midpoint 7.5) also lets sigma->0 collapse to a true
    single-symbol delta (entropy->0), not floor at a 50/50 split between two
    equally-near symbols."""
    xs = np.arange(16.0) - 8.0

    def dist(sigma):
        v = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
        return v / v.sum()

    target_h = min(3.999, max(0.02, target_h))
    lo, hi = 0.06, 200.0
    if entropy_of(dist(lo)) >= target_h:
        return dist(lo)
    if entropy_of(dist(hi)) <= target_h:
        return dist(hi)
    mid = dist(lo)
    for _ in range(max_iter):
        mid_sigma = (lo + hi) / 2.0
        mid = dist(mid_sigma)
        h = entropy_of(mid)
        if abs(h - target_h) <= tol:
            return mid
        if h < target_h:
            lo = mid_sigma
        else:
            hi = mid_sigma
    return mid


def pack_nibbles(symbols):
    """2 symbols/byte, low nibble = even index -- matches glm.c/codec_row.h."""
    n = len(symbols)
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    even = symbols[0::2]
    odd = symbols[1::2]
    if len(odd) < len(even):
        odd = np.concatenate([odd, np.zeros(1, dtype=np.uint8)])
    packed[: len(even)] = (even & 0xF) | (odd << 4)
    return packed


def sample_symbols(rng, p, n):
    cdf = np.cumsum(p)
    u = rng.random(n)
    symbols = np.searchsorted(cdf, u, side="right").astype(np.uint8)
    np.clip(symbols, 0, 15, out=symbols)  # guards the rare cdf[-1]<1.0 fp edge
    return symbols


def gen_tensor(rng, proj, target_h):
    O, I = ROW_SHAPE[proj]
    p = dist_for_entropy(target_h)
    n = O * I
    symbols = sample_symbols(rng, p, n)
    packed = pack_nibbles(symbols)
    achieved_h = entropy_of(np.bincount(symbols, minlength=16) / n)
    return packed.tobytes(), O, I, achieved_h


def generate(census_csv, outdir, tensors_per_group, seed):
    by_group = load_census_h(census_csv)
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)

    index_rows = []
    byte_offset = 0
    corpus_path = os.path.join(outdir, "synthetic-corpus.bin")
    with open(corpus_path, "wb") as fbin:
        for band in BANDS:
            for proj in PROJS:
                h_pool = np.array(by_group[(band, proj)])
                for t in range(tensors_per_group):
                    base_h = float(rng.choice(h_pool))
                    jitter = float(rng.normal(0, 0.01))
                    target_h = base_h + jitter
                    packed, O, I, achieved_h = gen_tensor(rng, proj, target_h)
                    fbin.write(packed)
                    index_rows.append({
                        "band": band, "proj": proj, "tensor_idx": t,
                        "O": O, "I": I, "row_bytes": (I + 1) // 2,
                        "byte_offset": byte_offset, "nbytes": len(packed),
                        "target_h": round(target_h, 6),
                        "achieved_h": round(achieved_h, 6),
                    })
                    byte_offset += len(packed)

    index_path = os.path.join(outdir, "synthetic-index.csv")
    cols = ["band", "proj", "tensor_idx", "O", "I", "row_bytes",
            "byte_offset", "nbytes", "target_h", "achieved_h"]
    with open(index_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in index_rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    max_err = max(abs(r["target_h"] - r["achieved_h"]) for r in index_rows)
    summary = {
        "seed": seed,
        "tensors_per_group": tensors_per_group,
        "n_tensors": len(index_rows),
        "total_bytes": byte_offset,
        "census_csv": os.path.relpath(census_csv),
        "row_shape": ROW_SHAPE,
        "h_target_vs_achieved_max_abs_err_bits": round(max_err, 6),
        "note": ("Synthetic only -- no /path/to/models reads, today or "
                 "ever, by this script. Per-tensor entropy targets are "
                 "bootstrapped from the committed census's own per-"
                 "(band,proj) h_global distribution (c/bench-m5max/"
                 "new-math-20260715/census-tensors.csv); symbol shape is a "
                 "quantized-Gaussian family solved by bisection to hit that "
                 "target entropy exactly -- not a claim that real weights "
                 "are Gaussian-quantized, just a realistic, reproducible "
                 "stand-in with the measured entropy and the measured "
                 "per-band/projection conditional structure."),
    }
    with open(os.path.join(outdir, "synthetic-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary, index_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--census-csv", default=DEFAULT_CENSUS_CSV)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tensors-per-group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260715)
    a = ap.parse_args(argv)

    summary, _ = generate(a.census_csv, a.outdir, a.tensors_per_group, a.seed)
    print(f"[gen] {summary['n_tensors']} synthetic tensors, "
          f"{summary['total_bytes']/1e6:.1f} MB packed, written to {a.outdir}")
    print(f"[gen] max |target_h - achieved_h| = "
          f"{summary['h_target_vs_achieved_max_abs_err_bits']:.4f} bits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
