from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from collections import Counter
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


holdout = load_module("prepare_codec_holdout", ROOT / "tools" / "prepare_codec_holdout.py")


class BandOfTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(holdout.band_of(3), "early")
        self.assertEqual(holdout.band_of(27), "early")
        self.assertEqual(holdout.band_of(28), "mid")
        self.assertEqual(holdout.band_of(52), "mid")
        self.assertEqual(holdout.band_of(53), "late")
        self.assertEqual(holdout.band_of(77), "late")

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            holdout.band_of(2)
        with self.assertRaises(ValueError):
            holdout.band_of(78)


class RelativeErrorTests(unittest.TestCase):
    def test_zero_error(self):
        self.assertEqual(holdout.relative_error_pct(100.0, 100.0), 0.0)

    def test_known_value(self):
        self.assertAlmostEqual(holdout.relative_error_pct(100.0, 110.0), 10.0, places=9)
        self.assertAlmostEqual(holdout.relative_error_pct(100.0, 90.0), 10.0, places=9)

    def test_zero_observed_is_defined(self):
        self.assertEqual(holdout.relative_error_pct(0.0, 50.0), 0.0)


class AcceptanceReportTests(unittest.TestCase):
    def test_preregistered_bounds_are_the_stated_values(self):
        # the design's own bounds must be baked into the constants, not just
        # into prose -- this is the "preregistered" part
        self.assertEqual(holdout.MEDIAN_ERR_BOUND_PCT, 2.0)
        self.assertEqual(holdout.P95_ERR_BOUND_PCT, 5.0)

    def test_perfect_predictions_accept(self):
        rows = [{"band": "early", "proj": "gate", "observed": 100.0, "predicted": 100.0}] * 50
        rep = holdout.acceptance_report(rows)
        self.assertTrue(rep["accept"])
        self.assertEqual(rep["median_pct"], 0.0)
        self.assertEqual(rep["p95_pct"], 0.0)

    def test_uniformly_large_error_rejects(self):
        rows = [{"band": "early", "proj": "gate", "observed": 100.0, "predicted": 120.0}] * 50
        rep = holdout.acceptance_report(rows)
        self.assertFalse(rep["accept"])

    def test_boundary_at_exactly_the_bound_accepts(self):
        # all errors sitting exactly AT both bounds (median==2.0%, and every
        # single row is also exactly at the p95 bound of 5.0% since they're
        # all identical) should ACCEPT ("<=", not "<")
        rows = [{"band": "early", "proj": "gate", "observed": 100.0, "predicted": 102.0}] * 100
        rep = holdout.acceptance_report(rows)
        self.assertAlmostEqual(rep["median_pct"], 2.0, places=6)
        self.assertAlmostEqual(rep["p95_pct"], 2.0, places=6)
        self.assertTrue(rep["accept"])

        rows2 = [{"band": "early", "proj": "gate", "observed": 100.0, "predicted": 105.0}] * 100
        rep2 = holdout.acceptance_report(rows2)
        self.assertAlmostEqual(rep2["median_pct"], 5.0, places=6)
        self.assertAlmostEqual(rep2["p95_pct"], 5.0, places=6)
        self.assertFalse(rep2["accept"], "median==5.0%%>bound 2.0%% must reject")

    def test_just_over_the_bound_rejects(self):
        rows = [{"band": "early", "proj": "gate", "observed": 100.0, "predicted": 102.01}] * 100
        rep = holdout.acceptance_report(rows)
        self.assertFalse(rep["accept"])

    def test_empty_rows_does_not_crash_and_does_not_accept(self):
        rep = holdout.acceptance_report([])
        self.assertFalse(rep["accept"])

    def test_by_band_and_by_proj_breakdowns_present(self):
        rows = []
        for band in ("early", "mid", "late"):
            for proj in ("gate", "up", "down"):
                rows.append({"band": band, "proj": proj, "observed": 1000.0, "predicted": 1010.0})
        rep = holdout.acceptance_report(rows)
        self.assertEqual(set(rep["by_band"].keys()), {"early", "mid", "late"})
        self.assertEqual(set(rep["by_proj"].keys()), {"gate", "up", "down"})


class CrossEntropyBytesTests(unittest.TestCase):
    def test_own_table_tracks_entropy_estimate(self):
        # reuse the census's own entropy_bits for a ground truth comparison
        census = load_module("measure_expert_entropy", ROOT / "tools" / "measure_expert_entropy.py")
        counts = np.array([2, 5, 40, 200, 900, 2500, 4800, 6800, 7200, 6800,
                            4800, 2500, 900, 200, 40, 5], dtype=np.int64)
        own_freq = census.quantize_freqs(counts)
        est = holdout.cross_entropy_bytes(counts, own_freq)
        ideal = census.entropy_bits(counts) * int(counts.sum()) / 8.0
        self.assertAlmostEqual(est / ideal, 1.0, delta=0.02)

    def test_mismatched_table_never_cheaper_gibbs_inequality(self):
        census = load_module("measure_expert_entropy", ROOT / "tools" / "measure_expert_entropy.py")
        rng = np.random.default_rng(3)
        for _ in range(20):
            counts = rng.integers(1, 5000, size=16).astype(np.int64)
            own_freq = census.quantize_freqs(counts)
            own_bytes = holdout.cross_entropy_bytes(counts, own_freq)
            foreign_freq = np.roll(own_freq, 3)  # a plausible-looking but wrong table
            foreign_bytes = holdout.cross_entropy_bytes(counts, foreign_freq)
            self.assertGreaterEqual(foreign_bytes, own_bytes - 1e-6)

    def test_empty_counts_is_zero_bytes(self):
        self.assertEqual(holdout.cross_entropy_bytes(np.zeros(16, dtype=np.int64),
                                                       np.full(16, 256, dtype=np.int64)), 0.0)


class StratifiedSampleTests(unittest.TestCase):
    def _mock_counts(self, seed=7, n_experts=64):
        rng = np.random.default_rng(seed)
        counts = Counter()
        for _, lo, hi in holdout.BANDS:
            for l in range(lo, hi + 1):
                for e in range(n_experts):
                    if rng.random() < 0.3:
                        counts[(l, e)] = int(rng.integers(1, 50))
        return counts

    def test_returns_exactly_n_total_unique_experts(self):
        cfg = {"n_experts": 64}
        counts = self._mock_counts()
        chosen = holdout.stratified_sample(cfg, counts, 120, seed=1)
        self.assertEqual(len(chosen), 120)
        self.assertEqual(len(set(chosen)), 120)

    def test_band_split_is_even(self):
        cfg = {"n_experts": 64}
        counts = self._mock_counts()
        chosen = holdout.stratified_sample(cfg, counts, 300, seed=2)
        band_n = Counter(holdout.band_of(l) for (l, e) in chosen)
        self.assertEqual(sum(band_n.values()), 300)
        self.assertTrue(max(band_n.values()) - min(band_n.values()) <= 2)

    def test_includes_both_hot_and_cold_experts(self):
        cfg = {"n_experts": 64}
        counts = self._mock_counts()
        chosen = holdout.stratified_sample(cfg, counts, 120, seed=3)
        hot = sum(1 for p in chosen if counts.get(p, 0) > 0)
        cold = len(chosen) - hot
        self.assertGreater(hot, 0)
        self.assertGreater(cold, 0)

    def test_different_seed_changes_the_draw(self):
        cfg = {"n_experts": 64}
        counts = self._mock_counts()
        chosen_a = holdout.stratified_sample(cfg, counts, 60, seed=10)
        chosen_b = holdout.stratified_sample(cfg, counts, 60, seed=11)
        self.assertNotEqual(chosen_a, chosen_b)

    def test_all_zero_touch_counts_still_returns_full_sample(self):
        # an all-cold pool (empty counts) must not crash or short-change the draw
        cfg = {"n_experts": 64}
        chosen = holdout.stratified_sample(cfg, Counter(), 90, seed=4)
        self.assertEqual(len(chosen), 90)


class FrozenConfigTests(unittest.TestCase):
    """The frozen config is a committed artifact this study produced; check
    its shape is what the holdout script expects (catches drift between the
    two files without needing a real holdout run)."""

    def test_default_frozen_config_loads_and_has_expected_shape(self):
        cfg = holdout.load_frozen_config(holdout.DEFAULT_FROZEN_CONFIG)
        self.assertIn("frozen_quantized_freq16_by_band_proj", cfg)
        tables = cfg["frozen_quantized_freq16_by_band_proj"]
        for band in ("early", "mid", "late"):
            for proj in ("gate", "up", "down"):
                key = f"{band}/{proj}"
                self.assertIn(key, tables)
                freq = tables[key]
                self.assertEqual(len(freq), 16)
                self.assertEqual(sum(freq), 4096)  # M_TOTAL, 2^M_BITS=12

    def test_frozen_seed_differs_from_census_seed(self):
        # the whole point of a holdout: NOT a relabeled rerun of the census draw
        self.assertNotEqual(holdout.HOLDOUT_SEED_DEFAULT, 20260715)


if __name__ == "__main__":
    unittest.main()
