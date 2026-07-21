#!/usr/bin/env python3
"""Simulate iliria static pin + per-layer LRU policies from an ordered route trace.

Unlike .fa_usage, the route trace preserves request order. The engine resolves one
copy of each unique expert in a MoE call (batch union), so this simulator first
collapses duplicate top-k selections inside each call before applying pin/LRU
semantics.

Trace v2 embeds the exact expert byte size, pinned q4-equivalent units, cache-layer
units, and live LRU capacity. Those values reproduce the engine's current expert
RAM allowance without inferring it from rounded startup text.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

MAGIC = b"FAROUTE1"
HEADER_BASE = struct.Struct("<8sII")
HEADER_V2_META = struct.Struct("<QIIII")
RECORD = struct.Struct("<QQHHHH")


@dataclass(frozen=True)
class TraceMetadata:
    version: int
    record_size: int
    expert_bytes: int = 0
    cache_units: int = 0
    lru_per_layer: int = 0
    pinned_units: int = 0
    flags: int = 0

    @property
    def mtp_present(self) -> bool:
        return bool(self.flags & 1)

    @property
    def total_expert_slots(self) -> int:
        if self.cache_units <= 0:
            return 0
        return self.pinned_units + self.cache_units * self.lru_per_layer


@dataclass(frozen=True)
class RouteRecord:
    event: int
    call: int
    layer: int
    row: int
    rank: int
    expert: int


@dataclass
class SimResult:
    pin_gb_requested: float
    pin_gb_actual: float
    pinned_pairs: int
    lru_per_layer: int
    requests: int
    pin_hits: int
    lru_hits: int
    misses: int
    optimal_hits: int

    @property
    def hits(self) -> int:
        return self.pin_hits + self.lru_hits

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def optimal_rate(self) -> float:
        return self.optimal_hits / self.requests if self.requests else 0.0


def read_trace_header(f: BinaryIO, path: Path | None = None) -> TraceMetadata:
    label = str(path) if path else "route trace"
    raw = f.read(HEADER_BASE.size)
    if len(raw) != HEADER_BASE.size:
        raise ValueError(f"{label}: truncated route-trace header")
    magic, version, record_size = HEADER_BASE.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"{label}: bad route-trace magic {magic!r}")
    if record_size != RECORD.size:
        raise ValueError(f"{label}: unsupported route record size {record_size}")
    if version == 1:
        return TraceMetadata(version, record_size)
    if version == 2:
        extra = f.read(HEADER_V2_META.size)
        if len(extra) != HEADER_V2_META.size:
            raise ValueError(f"{label}: truncated v2 metadata")
        expert_bytes, cache_units, lru, pinned_units, flags = HEADER_V2_META.unpack(extra)
        return TraceMetadata(
            version,
            record_size,
            expert_bytes,
            cache_units,
            lru,
            pinned_units,
            flags,
        )
    raise ValueError(f"{label}: unsupported trace version {version}")


def _iter_records(f: BinaryIO, path: Path) -> Iterator[RouteRecord]:
    expected_event = 0
    while True:
        chunk = f.read(RECORD.size * 8192)
        if not chunk:
            break
        if len(chunk) % RECORD.size:
            raise ValueError(f"{path}: truncated final route record")
        for fields in struct.iter_unpack(RECORD.format, chunk):
            rec = RouteRecord(*fields)
            if rec.event != expected_event:
                raise ValueError(
                    f"{path}: event sequence break at {expected_event}, found {rec.event}"
                )
            expected_event += 1
            yield rec


def iter_trace(path: Path) -> Iterator[RouteRecord]:
    with path.open("rb") as f:
        read_trace_header(f, path)
        yield from _iter_records(f, path)


def load_engine_requests(
    path: Path,
) -> tuple[dict[int, list[int]], int, int, TraceMetadata]:
    """Return per-layer batch-union request streams and trace metadata.

    Records are ordered by call, row and route rank. Within a call/layer the engine
    forms a first-occurrence union, so repeated experts across S rows are one cache
    lookup/load and are collapsed here.
    """

    by_layer: dict[int, list[int]] = defaultdict(list)
    current_key: tuple[int, int] | None = None
    seen: set[int] = set()
    selections = 0
    calls = 0
    with path.open("rb") as f:
        metadata = read_trace_header(f, path)
        for rec in _iter_records(f, path):
            selections += 1
            key = (rec.call, rec.layer)
            if key != current_key:
                current_key = key
                seen.clear()
                calls += 1
            if rec.expert not in seen:
                seen.add(rec.expert)
                by_layer[rec.layer].append(rec.expert)
    return dict(by_layer), selections, calls, metadata


def load_usage(path: Path) -> dict[tuple[int, int], int]:
    usage: dict[tuple[int, int], int] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 3:
            raise ValueError(f"{path}:{lineno}: expected 'layer expert count'")
        layer, expert, count = map(int, parts[:3])
        if layer < 0 or expert < 0 or count < 0:
            raise ValueError(f"{path}:{lineno}: negative field")
        usage[(layer, expert)] = usage.get((layer, expert), 0) + count
    if not usage:
        raise ValueError(f"{path}: no usage records")
    return usage


def usage_from_requests(requests: dict[int, list[int]]) -> dict[tuple[int, int], int]:
    return {
        (layer, expert): count
        for layer, seq in requests.items()
        for expert, count in Counter(seq).items()
    }


def select_pins(
    usage: dict[tuple[int, int], int],
    slots: int,
    policy: str,
    layers: list[int],
) -> set[tuple[int, int]]:
    if slots <= 0:
        return set()
    if policy == "global":
        # This matches pin_load(): all valid (layer, expert) records are globally
        # ranked by cumulative count, then truncated by the byte budget.
        ranked = sorted(usage, key=lambda p: (-usage[p], p[0], p[1]))
        return set(ranked[:slots])
    if policy != "per-layer":
        raise ValueError(f"unknown pin policy: {policy}")

    # Experimental comparison policy: equal base allocation per represented layer,
    # then assign the remainder from the globally hottest unpinned tensors.
    base, _ = divmod(slots, max(1, len(layers)))
    chosen: set[tuple[int, int]] = set()
    next_candidates: list[tuple[int, int]] = []
    for layer in layers:
        ranked = sorted(
            (p for p in usage if p[0] == layer),
            key=lambda p: (-usage[p], p[1]),
        )
        chosen.update(ranked[:base])
        next_candidates.extend(ranked[base:])
    next_candidates.sort(key=lambda p: (-usage[p], p[0], p[1]))
    for pair in next_candidates:
        if len(chosen) >= slots:
            break
        chosen.add(pair)
    return chosen


def simulate_lru(
    requests: dict[int, list[int]],
    pinned: set[tuple[int, int]],
    capacity: int,
) -> tuple[int, int, int]:
    pin_hits = lru_hits = misses = 0
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    for layer in sorted(requests):
        cache = caches[layer]
        for expert in requests[layer]:
            if (layer, expert) in pinned:
                pin_hits += 1
                continue
            if expert in cache:
                lru_hits += 1
                cache.move_to_end(expert)
                continue
            misses += 1
            if capacity <= 0:
                continue
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[expert] = None
    return pin_hits, lru_hits, misses


def simulate_optimal_layer(seq: list[int], pinned_ids: set[int], capacity: int) -> int:
    """Return total hits (pin + dynamic) under Belady MIN for one layer."""

    future: dict[int, deque[int]] = defaultdict(deque)
    for pos, expert in enumerate(seq):
        if expert not in pinned_ids:
            future[expert].append(pos)
    resident: set[int] = set()
    hits = 0
    infinity = len(seq) + 1
    for pos, expert in enumerate(seq):
        if expert in pinned_ids:
            hits += 1
            continue
        q = future[expert]
        if not q or q[0] != pos:
            raise AssertionError("future-position bookkeeping corrupted")
        q.popleft()
        if expert in resident:
            hits += 1
            continue
        if capacity <= 0:
            continue
        if len(resident) >= capacity:
            victim = max(
                resident,
                key=lambda item: future[item][0] if future[item] else infinity,
            )
            resident.remove(victim)
        resident.add(expert)
    return hits


def simulate_optimal(
    requests: dict[int, list[int]],
    pinned: set[tuple[int, int]],
    capacity: int,
) -> int:
    total = 0
    for layer, seq in requests.items():
        pins = {expert for pin_layer, expert in pinned if pin_layer == layer}
        total += simulate_optimal_layer(seq, pins, capacity)
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("trace", type=Path)
    p.add_argument("--usage", type=Path)
    p.add_argument(
        "--pin-gb",
        type=float,
        nargs="+",
        default=[0, 8, 12, 16, 20, 24, 32, 48],
        help="Decimal GB, matching the engine's PIN_GB variable",
    )
    p.add_argument(
        "--expert-budget-gb",
        type=float,
        default=97.5,
        help="Fallback decimal GB budget when the trace has no v2 cache metadata",
    )
    p.add_argument(
        "--expert-mb",
        type=float,
        help="Override expert size in decimal MB; v2 metadata is preferred",
    )
    p.add_argument(
        "--total-expert-slots",
        type=int,
        help="Override total q4-equivalent pin+LRU slots",
    )
    p.add_argument("--pin-policy", choices=("global", "per-layer"), default="global")
    p.add_argument(
        "--lru-per-layer",
        type=int,
        help="Override derived LRU capacity and use this fixed capacity for every layer",
    )
    p.add_argument("--tokens", type=int, default=0, help="Optional emitted-token count")
    p.add_argument("--csv", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.expert_budget_gb < 0:
        raise SystemExit("expert budget must be non-negative")
    if args.expert_mb is not None and args.expert_mb <= 0:
        raise SystemExit("expert size must be positive")
    if args.total_expert_slots is not None and args.total_expert_slots < 0:
        raise SystemExit("total expert slots must be non-negative")

    requests, selections, calls, metadata = load_engine_requests(args.trace)
    if not requests:
        raise SystemExit("trace contains no route requests")
    if metadata.mtp_present:
        raise SystemExit(
            "trace includes int8 MTP expert capacity; rerun with DRAFT=0/MTP=0 "
            "before using the uniform q4 cache simulator"
        )

    usage = load_usage(args.usage) if args.usage else usage_from_requests(requests)
    layers = sorted(set(requests) | {layer for layer, _ in usage})
    load_requests = sum(len(seq) for seq in requests.values())

    if args.expert_mb is not None:
        expert_bytes = args.expert_mb * 1_000_000
        expert_source = "command-line override"
    elif metadata.expert_bytes > 0:
        expert_bytes = float(metadata.expert_bytes)
        expert_source = "trace v2"
    else:
        expert_bytes = 18.916 * 1_000_000
        expert_source = "fallback estimate"

    if args.total_expert_slots is not None:
        total_slots = args.total_expert_slots
        budget_source = "command-line slot override"
    elif metadata.total_expert_slots > 0:
        total_slots = metadata.total_expert_slots
        budget_source = "trace v2 live pin+LRU state"
    else:
        total_slots = math.floor(args.expert_budget_gb * 1_000_000_000 / expert_bytes)
        budget_source = f"{args.expert_budget_gb:.2f} GB fallback"

    capacity_units = metadata.cache_units if metadata.cache_units > 0 else len(layers)
    if capacity_units <= 0:
        raise SystemExit("cannot determine cache-layer units")

    if args.usage is None:
        print("warning: no --usage supplied; pins are trained on the measured trace itself")
    print(f"Trace: {args.trace}")
    print(f"Trace version: {metadata.version}")
    print(f"Route selections: {selections:,}")
    print(f"MoE calls: {calls:,}")
    print(f"Batch-union cache requests: {load_requests:,}")
    print(f"Layers represented: {len(layers)}")
    print(f"Cache budget units: {total_slots:,} q4-equivalent slots ({budget_source})")
    print(f"Cache-layer units: {capacity_units}")
    print(f"Expert payload: {expert_bytes/1_000_000:.6f} MB ({expert_source})")
    print(f"Pin ranking: {args.pin_policy}")

    results: list[SimResult] = []
    for requested in args.pin_gb:
        if requested < 0:
            raise SystemExit("--pin-gb values must be non-negative")
        requested_slots = math.floor(requested * 1_000_000_000 / expert_bytes)
        pin_slots = min(requested_slots, total_slots, len(usage))
        pinned = select_pins(usage, pin_slots, args.pin_policy, layers)
        actual_pin_gb = len(pinned) * expert_bytes / 1_000_000_000
        if args.lru_per_layer is None:
            remaining_slots = max(0, total_slots - len(pinned))
            lru_capacity = remaining_slots // capacity_units
        else:
            lru_capacity = max(0, args.lru_per_layer)
        pin_hits, lru_hits, misses = simulate_lru(requests, pinned, lru_capacity)
        optimal_hits = simulate_optimal(requests, pinned, lru_capacity)
        results.append(
            SimResult(
                requested,
                actual_pin_gb,
                len(pinned),
                lru_capacity,
                load_requests,
                pin_hits,
                lru_hits,
                misses,
                optimal_hits,
            )
        )

    print("\nSimulation")
    print(" req_pin  actual_pin  pairs  LRU/L  pin_hit  lru_hit  miss     hit%  ideal%  gap")
    for r in results:
        gap = 100 * (r.optimal_rate - r.hit_rate)
        print(
            f" {r.pin_gb_requested:7.1f}  {r.pin_gb_actual:10.1f}  {r.pinned_pairs:5d}"
            f"  {r.lru_per_layer:5d}  {r.pin_hits:7d}  {r.lru_hits:7d}  {r.misses:7d}"
            f"  {100*r.hit_rate:7.2f}  {100*r.optimal_rate:7.2f}  {gap:5.2f}"
        )
        if args.tokens:
            print(f"           misses/emitted-token: {r.misses/args.tokens:.2f}")

    winner = max(results, key=lambda r: (r.hit_rate, -r.pin_gb_actual))
    print(
        f"\nBest simulated LRU hit rate: requested {winner.pin_gb_requested:g} GB pin, "
        f"actual {winner.pin_gb_actual:.1f} GB, LRU {winner.lru_per_layer}/layer, "
        f"{100*winner.hit_rate:.2f}%"
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "pin_gb_requested",
                    "pin_gb_actual",
                    "pinned_pairs",
                    "lru_per_layer",
                    "requests",
                    "pin_hits",
                    "lru_hits",
                    "misses",
                    "hit_rate",
                    "optimal_rate",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.pin_gb_requested,
                        r.pin_gb_actual,
                        r.pinned_pairs,
                        r.lru_per_layer,
                        r.requests,
                        r.pin_hits,
                        r.lru_hits,
                        r.misses,
                        r.hit_rate,
                        r.optimal_rate,
                    ]
                )
        print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
