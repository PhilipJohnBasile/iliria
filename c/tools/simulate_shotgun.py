#!/usr/bin/env python3
"""Simulate shotgun (speculative broad) prefetch from an ordered route trace.

For each MoE call, instead of loading only the K selected experts, we speculatively
load the top-K most-frequently-selected experts for that layer (from .fa_usage
history). The "shotgun" fires a wide spread; the actual router selections are
the "target." We measure what fraction of actual selections land in the spread.

Then we simulate a two-stage policy:
  1. SHOTGUN: prefetch top-K likely experts for the next N layers during compute
  2. SNIPER: on-demand load for anything the shotgun missed
  3. LEARNING: experts the shotgun keeps hitting right get promoted to pin

The key metric is "effective hit rate" — how many actual expert requests are
already in RAM when needed, divided by total requests.

Usage:
  python3 tools/simulate_shotgun.py routes.bin \
    --usage /path/to/model/.fa_usage \
    --spread-k 8 16 24 32 48 \
    --lookahead 1 2 3 \
    --csv shotgun.csv
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
        magic, version, rec_size = HEADER_BASE.unpack(f.read(16))
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
            data = f.read(24)
            if len(data) < 24:
                break
            event, call, layer, row, rank, expert = RECORD.unpack(data)
            records.append(RouteRecord(event, call, layer, row, rank, expert))
    return meta, records


def load_usage(path: Path) -> dict[int, Counter]:
    """Load .fa_usage into per-layer expert frequency counters."""
    usage = defaultdict(Counter)
    if not path or not path.exists():
        return usage
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            layer, expert, count = int(parts[0]), int(parts[1]), int(parts[2])
            usage[layer][expert] = count
    return usage


def simulate_shotgun(records: list[RouteRecord], usage: dict[int, Counter],
                     spread_k: int, lookahead: int, lru_per_layer: int = 52) -> dict:
    """Simulate shotgun + sniper + LRU cache.

    For each MoE call:
      1. Look at the actual experts selected (batch union — dedup across rows)
      2. The shotgun pre-loaded top-K most-frequent experts for this layer
      3. Anything in the shotgun spread = shotgun hit (no disk stall)
      4. Anything in LRU = LRU hit (no disk stall)
      5. Anything else = miss (disk stall, then added to LRU)
      6. LRU evicts least-recently-used when full
    """
    # Group records by MoE call
    calls = defaultdict(list)
    for r in records:
        calls[r.call].append(r)

    # Precompute shotgun spread per layer (top-K by usage frequency)
    shotgun_spread = {}
    for layer in range(130):
        if layer in usage:
            shotgun_spread[layer] = set(
                e for e, _ in usage[layer].most_common(spread_k)
            )
        else:
            shotgun_spread[layer] = set()

    # Simulate
    lru_cache = defaultdict(OrderedDict)  # layer -> OrderedDict(expert -> True)
    shotgun_hits = 0
    lru_hits = 0
    misses = 0
    total = 0
    shotgun_wasted = 0  # experts loaded but not needed

    for call_id in sorted(calls.keys()):
        # Get actual experts for this call (batch union)
        call_records = calls[call_id]
        layer = call_records[0].layer
        actual_experts = set(r.expert for r in call_records)

        # Shotgun spread for this layer
        spread = shotgun_spread.get(layer, set())

        # Count wasted shotgun loads (in spread but not needed)
        shotgun_wasted += len(spread - actual_experts)

        for expert in actual_experts:
            total += 1
            if expert in spread:
                # Shotgun hit — expert was already prefetched
                shotgun_hits += 1
                # Add to LRU too (it's now resident)
                if expert in lru_cache[layer]:
                    del lru_cache[layer][expert]
                lru_cache[layer][expert] = True
                # Evict if over capacity
                while len(lru_cache[layer]) > lru_per_layer:
                    lru_cache[layer].popitem(last=False)
            elif expert in lru_cache[layer]:
                # LRU hit
                lru_hits += 1
                del lru_cache[layer][expert]
                lru_cache[layer][expert] = True
            else:
                # Miss — load from disk, add to LRU
                misses += 1
                lru_cache[layer][expert] = True
                while len(lru_cache[layer]) > lru_per_layer:
                    lru_cache[layer].popitem(last=False)

    hit_rate = (shotgun_hits + lru_hits) / total if total else 0
    shotgun_hit_rate = shotgun_hits / total if total else 0
    lru_hit_rate = lru_hits / total if total else 0
    miss_rate = misses / total if total else 0

    # Shotgun bandwidth cost: wasted loads per call
    calls_count = len(calls)
    wasted_per_call = shotgun_wasted / calls_count if calls_count else 0

    return {
        "spread_k": spread_k,
        "lookahead": lookahead,
        "lru_per_layer": lru_per_layer,
        "total_requests": total,
        "shotgun_hits": shotgun_hits,
        "lru_hits": lru_hits,
        "misses": misses,
        "shotgun_hit_pct": shotgun_hit_rate * 100,
        "lru_hit_pct": lru_hit_rate * 100,
        "hit_pct": hit_rate * 100,
        "miss_pct": miss_rate * 100,
        "wasted_loads": shotgun_wasted,
        "wasted_per_call": wasted_per_call,
        "calls": calls_count,
    }


def simulate_bomb(records: list[RouteRecord], usage: dict[int, Counter],
                  spread_k: int, lru_per_layer: int = 52,
                  bomb_interval: int = 16) -> dict:
    """Simulate bomb (bulk prewarming) prefetch.

    Every `bomb_interval` tokens, load top-K most-frequent experts for ALL layers
    into the LRU cache in one burst. Then compute freely until the next bomb.

    This is prediction-free — it doesn't look at the router, just at cumulative
    frequency. The "explosion" fills the cache; the LRU evicts what's not used.
    """
    calls = defaultdict(list)
    for r in records:
        calls[r.call].append(r)

    # Precompute bomb spread per layer
    bomb_spread = {}
    for layer in range(130):
        if layer in usage:
            bomb_spread[layer] = set(
                e for e, _ in usage[layer].most_common(spread_k)
            )
        else:
            bomb_spread[layer] = set()

    lru_cache = defaultdict(OrderedDict)
    bomb_hits = 0
    lru_hits = 0
    misses = 0
    total = 0
    bombs_thrown = 0
    bomb_wasted = 0
    tokens_since_bomb = 0

    for call_id in sorted(calls.keys()):
        call_records = calls[call_id]
        layer = call_records[0].layer
        actual_experts = set(r.expert for r in call_records)

        # Throw bomb every N calls (approximating N tokens)
        if tokens_since_bomb >= bomb_interval:
            bombs_thrown += 1
            for bomb_layer in range(130):
                spread = bomb_spread.get(bomb_layer, set())
                bomb_wasted += len(spread - set())  # all are "wasted" until needed
                for expert in spread:
                    cache = lru_cache[bomb_layer]
                    if expert in cache:
                        del cache[expert]
                    cache[expert] = True
                    while len(cache) > lru_per_layer:
                        cache.popitem(last=False)
            tokens_since_bomb = 0

        tokens_since_bomb += 1

        for expert in actual_experts:
            total += 1
            if expert in lru_cache[layer]:
                # Check if it was from the bomb (most recent batch) or LRU
                lru_hits += 1
                del lru_cache[layer][expert]
                lru_cache[layer][expert] = True
            else:
                misses += 1
                lru_cache[layer][expert] = True
                while len(lru_cache[layer]) > lru_per_layer:
                    lru_cache[layer].popitem(last=False)

    hit_rate = lru_hits / total if total else 0
    miss_rate = misses / total if total else 0
    bomb_loads = bombs_thrown * sum(len(s) for s in bomb_spread.values())

    return {
        "mode": "bomb",
        "spread_k": spread_k,
        "bomb_interval": bomb_interval,
        "lru_per_layer": lru_per_layer,
        "total_requests": total,
        "bomb_hits": lru_hits,  # all hits are cache hits (bomb fills cache)
        "lru_hits": lru_hits,
        "misses": misses,
        "hit_pct": hit_rate * 100,
        "miss_pct": miss_rate * 100,
        "bombs_thrown": bombs_thrown,
        "bomb_loads": bomb_loads,
        "bomb_wasted": bomb_wasted,
        "bomb_gb": bomb_loads * 19 / 1024,  # approximate
        "calls": len(calls),
    }


def main():
    parser = argparse.ArgumentParser(description="Shotgun prefetch simulator")
    parser.add_argument("trace", type=Path, help="Route trace file (.bin)")
    parser.add_argument("--usage", type=Path, help=".fa_usage file for frequency data")
    parser.add_argument("--spread-k", type=int, nargs="+", default=[8, 16, 24, 32, 48],
                        help="Shotgun spread sizes to test")
    parser.add_argument("--lookahead", type=int, nargs="+", default=[1, 2, 3],
                        help="Lookahead depths (layers ahead to prefetch)")
    parser.add_argument("--lru-per-layer", type=int, default=52,
                        help="LRU cache capacity per layer")
    parser.add_argument("--csv", type=Path, help="Output CSV file")
    parser.add_argument("--bomb", action="store_true",
                        help="Also simulate bomb (bulk prewarming) mode")
    parser.add_argument("--bomb-intervals", type=int, nargs="+", default=[8, 16, 24, 32],
                        help="Bomb intervals (tokens between explosions)")
    args = parser.parse_args()

    meta, records = read_trace(args.trace)
    usage = load_usage(args.usage)

    print(f"Trace: {len(records)} records, {len(set(r.call for r in records))} MoE calls")
    if usage:
        total_usage = sum(sum(c.values()) for c in usage.values())
        print(f"Usage: {len(usage)} layers, {total_usage} total selections")
    else:
        print("Warning: no usage file — shotgun spread will be empty")

    results = []
    for spread_k in args.spread_k:
        for lookahead in args.lookahead:
            r = simulate_shotgun(records, usage, spread_k, lookahead, args.lru_per_layer)
            results.append(r)
            print(f"  K={spread_k:2d} LA={lookahead}: "
                  f"shotgun={r['shotgun_hit_pct']:.1f}% "
                  f"LRU={r['lru_hit_pct']:.1f}% "
                  f"total={r['hit_pct']:.1f}% "
                  f"miss={r['miss_pct']:.1f}% "
                  f"waste={r['wasted_per_call']:.1f}/call")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV: {args.csv}")

    # Find best shotgun
    best = max(results, key=lambda r: r["hit_pct"])
    print(f"\nBest shotgun: K={best['spread_k']} LA={best['lookahead']} "
          f"-> {best['hit_pct']:.1f}% hit ({best['shotgun_hit_pct']:.1f}% shotgun + {best['lru_hit_pct']:.1f}% LRU)")
    print(f"  Waste: {best['wasted_per_call']:.1f} unnecessary loads per call")

    # Bomb simulation
    if args.bomb:
        print(f"\n--- Bomb (bulk prewarming) ---")
        bomb_results = []
        for spread_k in args.spread_k:
            for interval in args.bomb_intervals:
                r = simulate_bomb(records, usage, spread_k, args.lru_per_layer, interval)
                bomb_results.append(r)
                print(f"  K={spread_k:2d} interval={interval:2d}: "
                      f"hit={r['hit_pct']:.1f}% miss={r['miss_pct']:.1f}% "
                      f"bombs={r['bombs_thrown']} bomb_gb={r['bomb_gb']:.1f}")

        if args.csv:
            csv_path = args.csv.with_suffix(".bomb.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=bomb_results[0].keys())
                writer.writeheader()
                writer.writerows(bomb_results)
            print(f"\nBomb CSV: {csv_path}")

        best_bomb = max(bomb_results, key=lambda r: r["hit_pct"])
        print(f"\nBest bomb: K={best_bomb['spread_k']} interval={best_bomb['bomb_interval']} "
              f"-> {best_bomb['hit_pct']:.1f}% hit, {best_bomb['bomb_gb']:.1f} GB per explosion")

        # Compare
        print(f"\n--- Shotgun vs Bomb ---")
        print(f"  Best shotgun: {best['hit_pct']:.1f}% hit")
        print(f"  Best bomb:    {best_bomb['hit_pct']:.1f}% hit")
        winner = "shotgun" if best['hit_pct'] > best_bomb['hit_pct'] else "bomb"
        print(f"  Winner: {winner}")


if __name__ == "__main__":
    main()
