#!/usr/bin/env python3
"""Simulate a route-trace predictor to close the Belady-LRU gap.

The Belady optimal for this workload is ~77.7% hit rate. PILOT K6 already
achieves 80.6% (it uses router state, not just frequency). The remaining gap
between LRU (~63%) and Belady (~78%) is 14 points; PILOT captures 11 of them.

This tool explores what closes the remaining 3 points: a Markov-chain
predictor that learns "after expert X at layer L, expert Y at layer L+1 is
selected with probability P." Unlike PILOT (which uses the router's hidden
state), this predictor works from the ordered trace alone.

Usage:
  python3 tools/simulate_predictor.py routes.bin \
    --usage /path/to/model/.fa_usage \
    --order 1 2 3 \
    --top-k 2 4 6 8 \
    --csv predictor.csv
"""

from __future__ import annotations

import argparse
import csv
import struct
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

MAGIC = b"FAROUTE1"
HEADER_BASE = struct.Struct("<8sII")
HEADER_V2_META = struct.Struct("<QIIII")
RECORD = struct.Struct("<QQHHHH")


@dataclass(frozen=True)
class RouteRecord:
    event: int
    call: int
    layer: int
    row: int
    rank: int
    expert: int


def read_trace(path: Path) -> tuple[dict, list[RouteRecord]]:
    with open(path, "rb") as f:
        magic, version, rec_size = HEADER_BASE.unpack(f.read(HEADER_BASE.size))
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic}")
        meta = {}
        if version >= 2:
            expert_bytes, cache_units, lru_per_layer, pinned_units, flags = \
                HEADER_V2_META.unpack(f.read(HEADER_V2_META.size))
            meta = dict(expert_bytes=expert_bytes, cache_units=cache_units,
                        lru_per_layer=lru_per_layer, pinned_units=pinned_units,
                        flags=flags)
        records = []
        while True:
            data = f.read(RECORD.size)
            if len(data) < RECORD.size:
                break
            records.append(RouteRecord(*RECORD.unpack(data)))
    return meta, records


def build_markov_predictor(records: list[RouteRecord], order: int) -> dict:
    """Build an N-order Markov chain across consecutive MoE calls.

    The trace records one layer per MoE call. Consecutive calls are consecutive
    layers in the forward pass. So order=1 means: after the experts selected at
    call N (layer L), what experts are selected at call N+1 (layer L+1)?

    Key: (layer, tuple of previous expert sets) -> Counter(next_expert)
    """
    # Group experts by call (each call = one layer)
    call_experts = {}  # call_id -> (layer, frozenset(experts))
    current_call = None
    current_layer = None
    current_experts = set()

    for r in records:
        if r.call != current_call:
            if current_call is not None:
                call_experts[current_call] = (current_layer, frozenset(current_experts))
            current_call = r.call
            current_layer = r.layer
            current_experts = {r.expert}
        else:
            current_experts.add(r.expert)
    if current_call is not None:
        call_experts[current_call] = (current_layer, frozenset(current_experts))

    # Build transition table: (layer, prev_expert_sets) -> Counter(expert)
    transitions = defaultdict(Counter)

    sorted_calls = sorted(call_experts.keys())
    for i in range(len(sorted_calls)):
        if i < order:
            continue
        layer, experts = call_experts[sorted_calls[i]]
        # Previous 'order' calls' expert sets
        prev_key = tuple(
            call_experts[sorted_calls[j]][1] for j in range(i - order, i)
        )
        for expert in experts:
            transitions[(layer, prev_key)][expert] += 1

    return transitions


def simulate_predictor(records: list[RouteRecord], transitions: dict,
                       order: int, top_k: int, lru_per_layer: int = 52) -> dict:
    """Simulate a Markov predictor + LRU cache.

    Each MoE call is one layer. Consecutive calls are consecutive layers.
    The predictor uses the previous 'order' calls' expert sets to predict
    the current call's experts, then prefetches the top-K predictions.
    """
    # Group experts by call
    call_experts = {}
    current_call = None
    current_layer = None
    current_experts = set()

    for r in records:
        if r.call != current_call:
            if current_call is not None:
                call_experts[current_call] = (current_layer, frozenset(current_experts))
            current_call = r.call
            current_layer = r.layer
            current_experts = {r.expert}
        else:
            current_experts.add(r.expert)
    if current_call is not None:
        call_experts[current_call] = (current_layer, frozenset(current_experts))

    lru_cache = defaultdict(OrderedDict)
    predictor_hits = 0
    lru_hits = 0
    misses = 0
    total = 0
    predictor_wasted = 0

    sorted_calls = sorted(call_experts.keys())
    for i in range(len(sorted_calls)):
        layer, actual = call_experts[sorted_calls[i]]

        # Predict
        if i >= order:
            prev_key = tuple(
                call_experts[sorted_calls[j]][1] for j in range(i - order, i)
            )
            predicted = set(
                e for e, _ in transitions.get(
                    (layer, prev_key), Counter()
                ).most_common(top_k)
            )
        else:
            predicted = set()

        predictor_wasted += len(predicted - actual)

        for expert in actual:
            total += 1
            if expert in predicted:
                predictor_hits += 1
                if expert in lru_cache[layer]:
                    del lru_cache[layer][expert]
                lru_cache[layer][expert] = True
                while len(lru_cache[layer]) > lru_per_layer:
                    lru_cache[layer].popitem(last=False)
            elif expert in lru_cache[layer]:
                lru_hits += 1
                del lru_cache[layer][expert]
                lru_cache[layer][expert] = True
            else:
                misses += 1
                lru_cache[layer][expert] = True
                while len(lru_cache[layer]) > lru_per_layer:
                    lru_cache[layer].popitem(last=False)

    hit_rate = (predictor_hits + lru_hits) / total if total else 0
    return {
        "order": order,
        "top_k": top_k,
        "lru_per_layer": lru_per_layer,
        "total_requests": total,
        "predictor_hits": predictor_hits,
        "lru_hits": lru_hits,
        "misses": misses,
        "predictor_hit_pct": predictor_hits / total * 100 if total else 0,
        "lru_hit_pct": lru_hits / total * 100 if total else 0,
        "hit_pct": hit_rate * 100,
        "miss_pct": misses / total * 100 if total else 0,
        "wasted_loads": predictor_wasted,
    }


def main():
    parser = argparse.ArgumentParser(description="Route-trace Markov predictor simulator")
    parser.add_argument("trace", type=Path, help="Route trace file (.bin)")
    parser.add_argument("--order", type=int, nargs="+", default=[1, 2, 3],
                        help="Markov chain orders to test")
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 4, 6, 8],
                        help="Top-K predictions to prefetch")
    parser.add_argument("--lru-per-layer", type=int, default=52,
                        help="LRU cache capacity per layer")
    parser.add_argument("--csv", type=Path, help="Output CSV file")
    args = parser.parse_args()

    meta, records = read_trace(args.trace)
    print(f"Trace: {len(records)} records, {len(set(r.call for r in records))} MoE calls")

    results = []
    for order in args.order:
        print(f"\nBuilding order-{order} Markov predictor...")
        transitions = build_markov_predictor(records, order)
        print(f"  {len(transitions)} transition states")

        for top_k in args.top_k:
            r = simulate_predictor(records, transitions, order, top_k, args.lru_per_layer)
            results.append(r)
            print(f"  order={order} K={top_k}: "
                  f"predictor={r['predictor_hit_pct']:.1f}% "
                  f"LRU={r['lru_hit_pct']:.1f}% "
                  f"total={r['hit_pct']:.1f}% "
                  f"miss={r['miss_pct']:.1f}% "
                  f"waste={r['wasted_loads']}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV: {args.csv}")

    best = max(results, key=lambda r: r["hit_pct"])
    print(f"\nBest: order={best['order']} K={best['top_k']} "
          f"-> {best['hit_pct']:.1f}% hit "
          f"({best['predictor_hit_pct']:.1f}% predictor + {best['lru_hit_pct']:.1f}% LRU)")

    # Compare to Belady and LRU baselines
    print(f"\n--- Comparison ---")
    print(f"  LRU only:          ~63.3%")
    print(f"  Best predictor:    {best['hit_pct']:.1f}%")
    print(f"  Belady optimal:    ~77.7%")
    print(f"  PILOT K6 (real):   80.6%")
    gap = 77.7 - best['hit_pct']
    print(f"  Remaining gap:     {gap:.1f} pts")


if __name__ == "__main__":
    main()
