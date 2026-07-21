from __future__ import annotations

import csv
import importlib.util
import os
import sys
import tempfile
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


gen = load_module("gen_codec_race_synthetic", ROOT / "tools" / "gen_codec_race_synthetic.py")


def write_fixture_census(path, rows):
    cols = ["layer", "expert", "band", "proj", "nbytes", "h_global",
            "h_cond_scalebin", "h_tile_4096", "h_tile_16384", "h_tile_65536"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for band, proj, h in rows:
            w.writerow([10, 0, band, proj, 6291456, h, h, h, h, h])


class EntropyDistTests(unittest.TestCase):
    def test_dist_for_entropy_hits_target_across_the_real_range(self):
        # census H spans ~2.67-3.02 bits/weight; check a wider bracket too
        for target in (0.5, 1.0, 2.0, 2.67, 2.9401, 3.02, 3.5, 3.9):
            p = gen.dist_for_entropy(target, tol=1e-4)
            self.assertAlmostEqual(gen.entropy_of(p), target, delta=2e-3)
            self.assertAlmostEqual(float(p.sum()), 1.0, places=9)

    def test_dist_is_peaked_at_center_not_flat(self):
        p = gen.dist_for_entropy(2.94)
        # a quantized-Gaussian shape at H~2.94 (out of a max 4.0 for uniform)
        # must put more mass on the center codes (7,8) than the extremes
        self.assertGreater(p[7] + p[8], p[0] + p[15])

    def test_uniform_distribution_is_the_high_entropy_limit(self):
        p = gen.dist_for_entropy(3.999, tol=1e-4)
        self.assertAlmostEqual(gen.entropy_of(p), 4.0, delta=0.01)


class PackNibblesTests(unittest.TestCase):
    def test_pack_matches_low_even_high_odd_convention(self):
        symbols = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        packed = gen.pack_nibbles(symbols)
        # byte0 = sym0 | sym1<<4, byte1 = sym2 | sym3<<4, byte2 = sym4 (low only)
        self.assertEqual(int(packed[0]), 1 | (2 << 4))
        self.assertEqual(int(packed[1]), 3 | (4 << 4))
        self.assertEqual(int(packed[2]), 5)

    def test_pack_length_is_ceil_half(self):
        for n in (0, 1, 2, 3, 100, 101):
            symbols = np.zeros(n, dtype=np.uint8)
            self.assertEqual(len(gen.pack_nibbles(symbols)), (n + 1) // 2)


class SampleSymbolsTests(unittest.TestCase):
    def test_sample_symbols_stays_in_alphabet(self):
        rng = np.random.default_rng(1)
        p = gen.dist_for_entropy(2.9)
        symbols = gen.sample_symbols(rng, p, 200000)
        self.assertTrue((symbols >= 0).all())
        self.assertTrue((symbols <= 15).all())
        achieved = gen.entropy_of(np.bincount(symbols, minlength=16) / len(symbols))
        self.assertAlmostEqual(achieved, 2.9, delta=0.02)


class EndToEndFixtureTests(unittest.TestCase):
    """Full generate() run against a tiny FIXTURE census (not the real one,
    and never /path/to/models) -- checks the corpus/index byte accounting
    is internally consistent."""

    def test_generate_end_to_end_on_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            census_csv = os.path.join(td, "fixture-census.csv")
            rows = []
            for band in gen.BANDS:
                for proj in gen.PROJS:
                    for h in (2.7, 2.8, 2.9, 3.0):
                        rows.append((band, proj, h))
            write_fixture_census(census_csv, rows)

            outdir = os.path.join(td, "out")
            summary, index_rows = gen.generate(census_csv, outdir, tensors_per_group=1, seed=42)

            self.assertEqual(summary["n_tensors"], len(gen.BANDS) * len(gen.PROJS))
            corpus_path = os.path.join(outdir, "synthetic-corpus.bin")
            self.assertEqual(os.path.getsize(corpus_path), summary["total_bytes"])

            # row_offsets-style byte accounting: offsets must be contiguous
            # and non-overlapping, and every tensor's nbytes must match its
            # declared O/I shape exactly (packed 2 symbols/byte).
            running = 0
            for row in index_rows:
                self.assertEqual(row["byte_offset"], running)
                expected_bytes = (row["O"] * row["I"] + 1) // 2
                self.assertEqual(row["nbytes"], expected_bytes)
                self.assertEqual(row["row_bytes"] * row["O"], row["nbytes"],
                                  "O rows of row_bytes must tile the tensor exactly")
                running += row["nbytes"]
            self.assertEqual(running, summary["total_bytes"])

            # achieved entropy should track the (jittered) target closely at
            # this tensor size (O*I is still in the millions of symbols)
            self.assertLess(summary["h_target_vs_achieved_max_abs_err_bits"], 0.01)

            index_csv_path = os.path.join(outdir, "synthetic-index.csv")
            with open(index_csv_path, newline="") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows), len(index_rows))


if __name__ == "__main__":
    unittest.main()
