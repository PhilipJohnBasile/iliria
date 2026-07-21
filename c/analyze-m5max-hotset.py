#!/usr/bin/env python3
"""Analyze iliria's cumulative expert-usage histogram correctly.

The resident object is an (MoE layer, expert id) tensor. Expert id 17 in layer 3
is not the same weight tensor as expert id 17 in layer 40. This tool therefore
keeps that layer dimension when estimating static top-N pin coverage.

The .fa_usage file is cumulative and contains no access order. It cannot predict
LRU hit rate, page-cache behavior, or an end-to-end hit-rate ceiling; those require
an ordered routing trace or a real A/B run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("usage", type=Path, help="Path to .fa_usage or a named hotset file")
    p.add_argument("--experts-per-layer", type=int, default=256)
    p.add_argument(
        "--expert-mib",
        type=float,
        default=18.04,
        help="Approximate raw q4 payload per expert; actual resident-slot memory can be higher",
    )
    p.add_argument(
        "--top",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256],
    )
    return p.parse_args()


def load_usage(path: Path) -> dict[int, Counter[int]]:
    layers: dict[int, Counter[int]] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            raise ValueError(f"{path}:{lineno}: expected 'layer expert count'")
        layer, expert, count = map(int, parts[:3])
        if layer < 0 or expert < 0 or count < 0:
            raise ValueError(f"{path}:{lineno}: negative field")
        layers.setdefault(layer, Counter())[expert] += count
    if not layers:
        raise ValueError(f"{path}: no usage rows")
    return layers


def static_coverage(layers: dict[int, Counter[int]], n: int) -> tuple[int, int]:
    selected = sum(sum(v for _, v in counts.most_common(n)) for counts in layers.values())
    total = sum(sum(counts.values()) for counts in layers.values())
    return selected, total


def minimum_n_for_target(layers: dict[int, Counter[int]], target: float, max_n: int) -> int | None:
    for n in range(1, max_n + 1):
        selected, total = static_coverage(layers, n)
        if total and selected / total >= target:
            return n
    return None


def main() -> None:
    args = parse_args()
    if args.experts_per_layer < 1:
        raise SystemExit("--experts-per-layer must be positive")
    if args.expert_mib <= 0:
        raise SystemExit("--expert-mib must be positive")

    layers = load_usage(args.usage)
    layer_ids = sorted(layers)
    total = sum(sum(c.values()) for c in layers.values())
    unique_pairs = sum(len(c) for c in layers.values())
    possible_pairs = len(layers) * args.experts_per_layer

    print(f"Usage file: {args.usage}")
    print(f"Total selections: {total:,}")
    print(f"MoE layers represented: {len(layers)} ({layer_ids[0]}..{layer_ids[-1]})")
    print(f"Unique (layer, expert) tensors observed: {unique_pairs:,}")
    print(f"Possible pairs in represented layers: {possible_pairs:,}")
    print(f"Observed pair coverage: {100 * unique_pairs / possible_pairs:.1f}%")

    layer_totals = [sum(layers[layer].values()) for layer in layer_ids]
    unique_per_layer = [len(layers[layer]) for layer in layer_ids]
    print("\nPer-layer sample shape")
    print(
        "  selections/layer min/median/max: "
        f"{min(layer_totals):,} / {statistics.median(layer_totals):,.0f} / {max(layer_totals):,}"
    )
    print(
        "  observed experts/layer min/median/max: "
        f"{min(unique_per_layer)} / {statistics.median(unique_per_layer):.0f} / {max(unique_per_layer)}"
    )

    print("\nStatic top-N pin coverage")
    print("  N/layer  pair tensors  raw GiB*  selections  coverage")
    for n in sorted(set(args.top)):
        if n < 1 or n > args.experts_per_layer:
            continue
        selected, _ = static_coverage(layers, n)
        pairs = n * len(layers)
        gib = pairs * args.expert_mib / 1024
        print(f"  {n:7d}  {pairs:12,d}  {gib:8.1f}  {selected:10,d}  {100*selected/total:7.2f}%")

    print("\nStatic coverage targets")
    for target in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        n = minimum_n_for_target(layers, target, args.experts_per_layer)
        if n is None:
            print(f"  {target:5.0%}: not reached")
            continue
        pairs = n * len(layers)
        gib = pairs * args.expert_mib / 1024
        print(f"  {target:5.0%}: top-{n}/layer = {pairs:,} tensors, approximately {gib:.1f} raw GiB*")

    # This view can reveal that a router ID is globally common, but it must not be
    # interpreted as a pin list because each layer owns a different tensor.
    by_id: Counter[int] = Counter()
    for counts in layers.values():
        by_id.update(counts)
    print("\nRouter expert-ID bias (not resident-tensor coverage)")
    for n in (1, 8, 16, 32, 64, 100, 128, 256):
        n = min(n, len(by_id))
        covered = sum(v for _, v in by_id.most_common(n))
        print(f"  top-{n:3d} IDs: {100*covered/total:6.2f}%")
        if n == len(by_id):
            break

    print("\n* Raw q4 expert payload estimate only. Scales, alignment, allocator/slab overhead,")
    print("  shared tensors, KV state, scratch, page cache, and macOS headroom are excluded.")
    print("  A cumulative histogram estimates static pin coverage only. It cannot simulate")
    print("  LRU or prove a total hit-rate ceiling because access order is absent.")


if __name__ == "__main__":
    main()
