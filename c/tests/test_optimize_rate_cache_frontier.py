from __future__ import annotations

import importlib.util
import random
import sys
import unittest
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


rcf = load_module("optimize_rate_cache_frontier",
                  ROOT / "tools" / "optimize_rate_cache_frontier.py")
vso = sys.modules["variable_size_cache_oracle"]


class RankStatsTests(unittest.TestCase):
    def test_rankdata_with_ties(self):
        r = rcf.rankdata_avg([10, 20, 20, 30])
        self.assertEqual(r.tolist(), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_perfect_and_inverse(self):
        x = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(rcf.spearman(x, [2, 4, 6, 8, 10]), 1.0)
        self.assertAlmostEqual(rcf.spearman(x, [5, 4, 3, 2, 1]), -1.0)

    def test_spearman_known_value(self):
        # classic example with one swapped pair
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 5, 4]
        self.assertAlmostEqual(rcf.spearman(x, y), 0.9, places=9)

    def test_spearman_constant_input_is_zero(self):
        self.assertEqual(rcf.spearman([1, 1, 1], [1, 2, 3]), 0.0)


class ComputeVTests(unittest.TestCase):
    def two_replay_reference(self, stream, base_size_of, layer, expert,
                             int4_b, int2_b, budget):
        def sized(target):
            return lambda e: target if e == expert else base_size_of(layer, e)
        b4 = rcf.layer_miss_bytes(stream, sized(int4_b), budget)
        b2 = rcf.layer_miss_bytes(stream, sized(int2_b), budget)
        return b4 - b2

    def test_matches_direct_two_replay_difference(self):
        rng = random.Random(11)
        int4_b, int2_b, budget = 19, 10, 60
        for trial in range(10):
            stream = [rng.randrange(8) for _ in range(80)]
            streams = {"t": {5: stream}}
            int2_set = {(5, e) for e in range(8) if rng.random() < 0.4}

            def base(l, e):
                return int2_b if (l, e) in int2_set else int4_b

            cands = [(5, e) for e in range(8)]
            v = rcf.compute_v(cands, streams, base, int4_b, int2_b, budget)
            for (l, e) in cands:
                ref = self.two_replay_reference(
                    stream, base, l, e, int4_b, int2_b, budget)
                self.assertEqual(v[(l, e)], ref, f"trial {trial} expert {e}")

    def test_absent_expert_has_zero_value(self):
        streams = {"t": {5: [0, 1, 0, 1]}}
        v = rcf.compute_v([(5, 7)], streams, lambda l, e: 19, 19, 10, 40)
        self.assertEqual(v[(5, 7)], 0.0)
        # cross-check with the expensive path: a full two-replay difference
        ref = self.two_replay_reference(
            [0, 1, 0, 1], lambda l, e: 19, 5, 7, 19, 10, 40)
        self.assertEqual(ref, 0)

    def test_values_sum_over_traces(self):
        stream_a = [0, 1, 0, 1, 0]
        stream_b = [0, 2, 0, 2]
        streams = {"a": {3: stream_a}, "b": {3: stream_b}}
        v_joint = rcf.compute_v([(3, 0)], streams, lambda l, e: 19, 19, 10, 19)
        v_a = rcf.compute_v([(3, 0)], {"a": {3: stream_a}},
                            lambda l, e: 19, 19, 10, 19)
        v_b = rcf.compute_v([(3, 0)], {"b": {3: stream_b}},
                            lambda l, e: 19, 19, 10, 19)
        self.assertEqual(v_joint[(3, 0)], v_a[(3, 0)] + v_b[(3, 0)])


class MatchedDistortionTests(unittest.TestCase):
    def test_prefers_high_value_within_budget(self):
        cands = [(0, i) for i in range(6)]
        err = {p: 1.0 for p in cands}  # flat distortion: any K matches
        v = {p: float(p[1]) for p in cands}
        pick = rcf.matched_distortion_pick(cands, v, err, 3, 3.0)
        self.assertEqual(sorted(p[1] for p in pick), [3, 4, 5])

    def test_budget_forces_cheap_completion(self):
        cands = [(0, 0), (0, 1), (0, 2), (0, 3)]
        err = {(0, 0): 0.1, (0, 1): 0.1, (0, 2): 0.1, (0, 3): 1.0}
        v = {(0, 0): 0.0, (0, 1): 0.0, (0, 2): 0.0, (0, 3): 100.0}
        # k=2, budget 0.2: the high-V expensive candidate cannot fit
        pick = rcf.matched_distortion_pick(cands, v, err, 2, 0.2)
        self.assertEqual(len(pick), 2)
        self.assertNotIn((0, 3), pick)
        self.assertLessEqual(sum(err[p] for p in pick), 0.2 + 1e-12)

    def test_high_value_taken_when_budget_allows(self):
        cands = [(0, 0), (0, 1), (0, 2), (0, 3)]
        err = {(0, 0): 0.1, (0, 1): 0.1, (0, 2): 0.1, (0, 3): 0.3}
        v = {(0, 0): 1.0, (0, 1): 2.0, (0, 2): 3.0, (0, 3): 100.0}
        pick = rcf.matched_distortion_pick(cands, v, err, 2, 0.4)
        self.assertIn((0, 3), pick)
        self.assertIn((0, 2), pick)


class EndToEndSyntheticTests(unittest.TestCase):
    def test_v_reflects_reuse_frequency(self):
        # expert 0 is requested constantly (high reuse pressure), expert 7
        # once; with everything int4 the layer thrashes, so demoting the
        # hot expert must be worth more miss bytes than demoting the cold one
        stream = [0, 1, 2, 3, 4, 5, 6] * 12 + [7]
        streams = {"t": {9: stream}}
        int4_b, int2_b, budget = 19, 10, 60  # 3 int4 slots for 8 experts
        v = rcf.compute_v([(9, 0), (9, 7)], streams, lambda l, e: int4_b,
                          int4_b, int2_b, budget)
        self.assertGreater(v[(9, 0)], v[(9, 7)])
        self.assertGreaterEqual(v[(9, 7)], 0)


if __name__ == "__main__":
    unittest.main()
