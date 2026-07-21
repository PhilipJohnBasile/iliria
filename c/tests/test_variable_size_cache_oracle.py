from __future__ import annotations

import importlib.util
import itertools
import math
import random
import struct
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vso = load_module("variable_size_cache_oracle",
                  ROOT / "tools" / "variable_size_cache_oracle.py")


def reference_lru_misses(stream, capacity):
    """Straightforward slot-LRU miss count (uniform sizes)."""
    od = OrderedDict()
    misses = 0
    for x in stream:
        if x in od:
            od.move_to_end(x)
            continue
        misses += 1
        if capacity <= 0:
            continue
        if len(od) >= capacity:
            od.popitem(last=False)
        od[x] = None
    return misses


class MattsonTests(unittest.TestCase):
    def test_matches_direct_lru_at_every_capacity(self):
        rng = random.Random(1)
        for trial in range(5):
            stream = [rng.randrange(12) for _ in range(400)]
            curve = vso.mattson_miss_curve(stream, 14)
            for cap in range(15):
                self.assertEqual(curve[cap], reference_lru_misses(stream, cap),
                                 f"trial {trial} cap {cap}")

    def test_stack_distance_basics(self):
        d = vso.mattson_stack_distances([7, 7, 3, 7, 3])
        self.assertTrue(math.isinf(d[0]))
        self.assertEqual(d[1], 1)   # immediate reuse
        self.assertTrue(math.isinf(d[2]))
        self.assertEqual(d[3], 2)   # one distinct item in between
        self.assertEqual(d[4], 2)


class CvarTests(unittest.TestCase):
    def test_hand_computed_distribution(self):
        # 4 calls; distances: [inf, 1], [inf], [2, 2], []
        calls = [[math.inf, 1.0], [math.inf], [2.0, 2.0], []]
        means, cvars = vso.cvar_curves(calls, 3, item_bytes=10, alpha=0.95)
        # capacity 0: misses per call = [2, 1, 2, 0] -> mean 1.25*10
        self.assertAlmostEqual(means[0], 12.5)
        # CVaR tail k = ceil(0.05*4) = 1 -> worst call = 2 misses = 20 bytes
        self.assertAlmostEqual(cvars[0], 20.0)
        # capacity 1: the d=1 request now hits -> [1, 1, 2, 0] mean 10.0
        self.assertAlmostEqual(means[1], 10.0)
        self.assertAlmostEqual(cvars[1], 20.0)
        # capacity 2: d=2 requests hit -> [1, 1, 0, 0] mean 5.0, worst 10
        self.assertAlmostEqual(means[2], 5.0)
        self.assertAlmostEqual(cvars[2], 10.0)

    def test_mean_curve_matches_miss_curve(self):
        rng = random.Random(2)
        stream = [rng.randrange(8) for _ in range(120)]
        # one call per request keeps the two views directly comparable
        calls = [[d] for d in vso.mattson_stack_distances(stream)]
        means, _ = vso.cvar_curves(calls, 10, item_bytes=1)
        curve = vso.mattson_miss_curve(stream, 10)
        for c in range(11):
            self.assertAlmostEqual(means[c] * len(stream), curve[c] * 1.0,
                                   places=6)


class DpAllocateTests(unittest.TestCase):
    def synthetic_curves(self, rng, n_layers=4, kmax=6):
        curves = {}
        for layer in range(n_layers):
            base = rng.randrange(50, 150)
            drop = sorted(rng.randrange(0, base) for _ in range(kmax))[::-1]
            curve = [float(base)] + [float(d) for d in drop]
            # enforce monotone nonincreasing
            for i in range(1, len(curve)):
                curve[i] = min(curve[i], curve[i - 1])
            curves[layer] = curve
        return curves

    def test_matches_brute_force_on_tiny_instances(self):
        rng = random.Random(3)
        for _ in range(10):
            curves = self.synthetic_curves(rng)
            total = 8
            alloc, best = vso.dp_allocate(curves, total)
            self.assertEqual(sum(alloc.values()), total)
            # brute force over all splits
            layers = sorted(curves)
            best_bf = math.inf
            kmax = [len(curves[l]) - 1 for l in layers]
            for split in itertools.product(*(range(k + 1) for k in kmax)):
                if sum(split) != total:
                    continue
                v = sum(curves[l][k] for l, k in zip(layers, split))
                best_bf = min(best_bf, v)
            self.assertAlmostEqual(best, best_bf, places=9)

    def test_never_worse_than_uniform(self):
        rng = random.Random(4)
        for _ in range(10):
            curves = self.synthetic_curves(rng, n_layers=5, kmax=8)
            total = 20
            uniform = total // len(curves)
            _, best = vso.dp_allocate(curves, total)
            b_uni = sum(c[min(uniform, len(c) - 1)] for c in curves.values())
            self.assertLessEqual(best, b_uni + 1e-9)


class PolicyTests(unittest.TestCase):
    def rand_stream(self, rng, n=500, alphabet=20):
        # skewed popularity so caching matters
        weights = [1.0 / (i + 1) for i in range(alphabet)]
        return rng.choices(range(alphabet), weights=weights, k=n)

    def test_byte_lru_equals_slot_lru_under_uniform_sizes(self):
        rng = random.Random(5)
        stream = self.rand_stream(rng)
        for cap in (1, 3, 7):
            m, mb = vso.replay_byte_lru(stream, lambda e: 10, cap * 10)
            self.assertEqual(m, reference_lru_misses(stream, cap))
            self.assertEqual(mb, m * 10)

    def test_snrd_equals_lru_under_uniform_sizes(self):
        rng = random.Random(6)
        stream = self.rand_stream(rng)
        m_lru, _ = vso.replay_byte_lru(stream, lambda e: 10, 50)
        m_snrd, _ = vso.replay_snrd(stream, lambda e: 10, 50)
        self.assertEqual(m_lru, m_snrd)

    def test_gds_cost_size_equals_byte_lru_heterogeneous(self):
        rng = random.Random(7)
        sizes = {e: rng.choice((10, 19)) for e in range(20)}
        for _ in range(5):
            stream = self.rand_stream(rng, n=400)
            m1, b1 = vso.replay_byte_lru(stream, sizes.__getitem__, 100)
            m2, b2 = vso.replay_gds(stream, sizes.__getitem__, 100, "size")
            self.assertEqual((m1, b1), (m2, b2))

    @staticmethod
    def brute_force_optimal_misses(stream, sizes, budget):
        """Exact offline minimum misses WITH admission bypass (lazy OPT):
        DP over resident sets. Tiny instances only."""
        states = {frozenset(): 0}
        for x in stream:
            nxt: dict = {}

            def relax(s, c):
                if s not in nxt or c < nxt[s]:
                    nxt[s] = c

            for s, cost in states.items():
                if x in s:
                    relax(s, cost)
                    continue
                relax(s, cost + 1)  # bypass
                base = s | {x}
                if sum(sizes[e] for e in base) <= budget:
                    relax(base, cost + 1)
                # insert after evicting any subset of residents
                for r in range(1, len(s) + 1):
                    for victims in itertools.combinations(s, r):
                        cand = (s - set(victims)) | {x}
                        if sum(sizes[e] for e in cand) <= budget:
                            relax(frozenset(cand), cost + 1)
            states = nxt
        return min(states.values())

    def test_belady_bytes_dominates_demand_belady_uniform(self):
        # The proxy allows admission bypass, a strictly LARGER class than
        # the committed demand-paging Belady (simulate_optimal_layer), so
        # its hits must be >= on the same stream (generous oracle).
        slotsim = sys.modules["simulate_m5max_cache"]
        rng = random.Random(8)
        for cap in (2, 4, 8):
            stream = self.rand_stream(rng, n=300, alphabet=12)
            m, _ = vso.replay_belady_bytes(stream, lambda e: 1, cap)
            hits_ref = slotsim.simulate_optimal_layer(stream, set(), cap)
            self.assertGreaterEqual(len(stream) - m, hits_ref)

    def test_belady_bytes_bounded_by_true_optimum(self):
        rng = random.Random(88)
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 1}
        for _ in range(6):
            stream = [rng.randrange(5) for _ in range(14)]
            greedy_m, _ = vso.replay_belady_bytes(
                stream, sizes.__getitem__, 3)
            opt_m = self.brute_force_optimal_misses(stream, sizes, 3)
            self.assertGreaterEqual(greedy_m, opt_m)  # greedy is feasible
            # documented non-optimality: gap may exist but stays small here
            self.assertLessEqual(greedy_m - opt_m, 2)

    def test_belady_bypass_beats_mandatory_insert_case(self):
        # cap 1, sequence a b a b a: demand Belady thrashes (5 misses),
        # bypass keeps 'a' resident (3 misses)
        m, _ = vso.replay_belady_bytes([0, 1, 0, 1, 0], lambda e: 1, 1)
        self.assertEqual(m, 3)

    def test_belady_bytes_never_loses_to_lru_on_typical_streams(self):
        rng = random.Random(9)
        sizes = {e: rng.choice((10, 19)) for e in range(20)}
        for _ in range(5):
            stream = self.rand_stream(rng, n=400)
            _, b_lru = vso.replay_byte_lru(stream, sizes.__getitem__, 95)
            _, b_off = vso.replay_belady_bytes(stream, sizes.__getitem__, 95)
            self.assertLessEqual(b_off, b_lru)

    def test_oversized_item_is_a_miss_but_never_cached(self):
        m, mb = vso.replay_byte_lru([1, 1, 1], lambda e: 100, 50)
        self.assertEqual((m, mb), (3, 300))
        m, mb = vso.replay_belady_bytes([1, 1, 1], lambda e: 100, 50)
        self.assertEqual((m, mb), (3, 300))


class PinTests(unittest.TestCase):
    def test_fill_pins_truncates_at_byte_budget(self):
        usage = {(0, 1): 100, (0, 2): 90, (1, 1): 80, (1, 2): 70}
        sizes = {k: 10 for k in usage}
        pinned, used = vso.fill_pins(usage, sizes, 25)
        # ranking: (0,1), (0,2), (1,1)... third no longer fits at 25 B
        self.assertEqual(set(pinned), {(0, 1), (0, 2)})
        self.assertEqual(used, 20)

    def test_fill_pins_stops_at_first_nonfitting(self):
        # pin_load() semantics: BREAK at the first pair that does not fit,
        # even if a later smaller pair would
        usage = {(0, 1): 100, (0, 2): 90, (1, 1): 80}
        sizes = {(0, 1): 10, (0, 2): 30, (1, 1): 5}
        pinned, used = vso.fill_pins(usage, sizes, 15)
        self.assertEqual(set(pinned), {(0, 1)})
        self.assertEqual(used, 10)


class ExtentTests(unittest.TestCase):
    def test_cluster_recovers_planted_structure(self):
        # 9 experts, planted triples that always co-activate
        n = 9
        co = np.zeros((n, n), dtype=np.int64)
        triples = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
        for t in triples:
            for i in t:
                for j in t:
                    if i != j:
                        co[i, j] = 50
        extent_of, bins = vso.cluster_extents(co, 3)
        self.assertEqual(bins, 3)
        for t in triples:
            self.assertEqual(len({extent_of[i] for i in t}), 1)

    def test_extents_touched_hand_case(self):
        calls = [vso.Call(0, 0, [0, 1, 2], 1), vso.Call(1, 0, [0, 3, 6], 1)]
        id_map = {0: [e // 3 for e in range(9)]}
        r = vso.extents_touched(calls, id_map)
        # call 0: experts 0,1,2 -> extent 0 only; call 1: extents 0,1,2
        self.assertEqual(r["total_extents"], 1 + 3)
        self.assertAlmostEqual(r["mean_extents_per_call"], 2.0)

    def test_miss_flag_restriction(self):
        calls = [vso.Call(0, 0, [0, 1, 2], 1)]
        id_map = {0: [e // 3 for e in range(9)]}
        r = vso.extents_touched(calls, id_map, {0: [False, False, False]})
        self.assertEqual(r["calls"], 0)
        self.assertEqual(r["total_extents"], 0)


class TraceRoundtripTests(unittest.TestCase):
    def make_trace(self, path, records):
        with open(path, "wb") as f:
            f.write(struct.pack("<8sII", b"FAROUTE1", 2, 24))
            f.write(struct.pack("<QIIII", 18915328, 75, 34, 0, 0))
            for i, (call, layer, row, rank, expert) in enumerate(records):
                f.write(struct.pack("<QQHHHH", i, call, layer, row, rank, expert))

    def test_load_calls_batch_union(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.bin"
            # call 0 layer 3: rows 0/1 pick overlapping experts
            self.make_trace(p, [
                (0, 3, 0, 0, 5), (0, 3, 0, 1, 9), (0, 3, 1, 0, 5),
                (1, 4, 0, 0, 2), (1, 4, 0, 1, 2),
            ])
            meta, calls = vso.load_calls(p)
            self.assertEqual(meta.lru_per_layer, 34)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0].experts, [5, 9])  # union, router order
            self.assertEqual(calls[0].n_rows, 2)
            self.assertEqual(calls[1].experts, [2])
            streams = vso.per_layer_streams(calls)
            self.assertEqual(streams, {3: [5, 9], 4: [2]})


if __name__ == "__main__":
    unittest.main()
