"""Unit tests for tools/compare_layer_captures.py on synthetic captures (no engine,
no real glm.c capture hook needed -- the module's own write_capture()/load_capture()
round-trip is the fixture)."""

from __future__ import annotations

import importlib.util
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


clc = load_module("compare_layer_captures", ROOT / "tools" / "compare_layer_captures.py")


def make_capture(rng, S, D, seed_offset=0.0):
    return (rng.standard_normal((S, D)).astype(np.float32) + seed_offset)


class CaptureRoundTripTests(unittest.TestCase):
    def test_write_then_load_preserves_header_fields(self):
        rng = np.random.default_rng(1)
        data = make_capture(rng, S=4, D=6144)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "L005_S0004_T000128.bin"
            clc.write_capture(path, layer=5, S=4, pos_base=128, D=6144, metal_prefill=1, data=data)
            cap = clc.load_capture(path)
        self.assertEqual(cap.layer, 5)
        self.assertEqual(cap.S, 4)
        self.assertEqual(cap.pos_base, 128)
        self.assertEqual(cap.D, 6144)
        self.assertEqual(cap.metal_prefill, 1)
        self.assertEqual(cap.data.shape, (4, 6144))
        np.testing.assert_allclose(cap.data, data.astype(np.float64), rtol=0, atol=0)

    def test_bad_magic_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.bin"
            path.write_bytes(b"NOTAMAGIC" + b"\x00" * 32)
            with self.assertRaises(clc.CaptureFormatError):
                clc.load_capture(path)

    def test_truncated_payload_is_rejected(self):
        rng = np.random.default_rng(2)
        data = make_capture(rng, S=4, D=8)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "L000_S0004_T000000.bin"
            clc.write_capture(path, layer=0, S=4, pos_base=0, D=8, metal_prefill=0, data=data)
            # Truncate a few payload bytes.
            raw = path.read_bytes()
            path.write_bytes(raw[:-3])
            with self.assertRaises(clc.CaptureFormatError):
                clc.load_capture(path)


class CompareMetricsTests(unittest.TestCase):
    def test_identical_captures_are_a_perfect_match(self):
        rng = np.random.default_rng(3)
        data = make_capture(rng, S=8, D=32)
        with tempfile.TemporaryDirectory() as td:
            path_a = Path(td) / "a.bin"
            path_b = Path(td) / "b.bin"
            clc.write_capture(path_a, 10, 8, 512, 32, 0, data)
            clc.write_capture(path_b, 10, 8, 512, 32, 1, data.copy())
            result = clc.compare_captures(clc.load_capture(path_a), clc.load_capture(path_b))
        self.assertEqual(result["max_abs"], 0.0)
        self.assertEqual(result["rms_mean"], 0.0)
        self.assertEqual(result["nrms_mean"], 0.0)
        self.assertAlmostEqual(result["cosine_mean"], 1.0, places=6)
        self.assertAlmostEqual(result["cosine_flat"], 1.0, places=6)
        self.assertAlmostEqual(result["top_margin_delta_worst"], 0.0, places=6)

    def test_small_perturbation_gives_small_but_nonzero_divergence(self):
        rng = np.random.default_rng(4)
        S, D = 4, 6144
        base = make_capture(rng, S, D)
        noise = rng.standard_normal((S, D)).astype(np.float32) * 1e-3
        with tempfile.TemporaryDirectory() as td:
            path_a = Path(td) / "a.bin"
            path_b = Path(td) / "b.bin"
            clc.write_capture(path_a, 5, S, 128, D, 0, base)
            clc.write_capture(path_b, 5, S, 128, D, 1, base + noise)
            result = clc.compare_captures(clc.load_capture(path_a), clc.load_capture(path_b))
        self.assertGreater(result["max_abs"], 0.0)
        self.assertLess(result["max_abs"], 0.05)
        self.assertGreater(result["cosine_mean"], 0.999)
        self.assertLess(result["cosine_mean"], 1.0)

    def test_orthogonal_vectors_have_zero_cosine(self):
        D = 4
        a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            path_a = Path(td) / "a.bin"
            path_b = Path(td) / "b.bin"
            clc.write_capture(path_a, 0, 1, 0, D, 0, a)
            clc.write_capture(path_b, 0, 1, 0, D, 1, b)
            result = clc.compare_captures(clc.load_capture(path_a), clc.load_capture(path_b))
        self.assertAlmostEqual(result["cosine_mean"], 0.0, places=6)

    def test_top_margin_delta_detects_shrinking_margin(self):
        # Reference: a clear top-1 (margin 10 - 1 = 9). Comparison: same top-1 index
        # but the runner-up has grown to nearly tie it (margin shrinks toward 0).
        ref = np.array([[10.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        cmp_close = np.array([[10.0, 9.9, 0.0, 0.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            path_a = Path(td) / "a.bin"
            path_b = Path(td) / "b.bin"
            clc.write_capture(path_a, 0, 1, 0, 4, 0, ref)
            clc.write_capture(path_b, 0, 1, 0, 4, 1, cmp_close)
            result = clc.compare_captures(clc.load_capture(path_a), clc.load_capture(path_b))
        # Reference margin 9.0, comparison margin 0.1 -> delta = 0.1 - 9.0 = -8.9.
        self.assertAlmostEqual(result["top_margin_delta_worst"], -8.9, places=4)

    def test_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path_a = Path(td) / "a.bin"
            path_b = Path(td) / "b.bin"
            clc.write_capture(path_a, 0, 2, 0, 8, 0, np.zeros((2, 8), dtype=np.float32))
            clc.write_capture(path_b, 0, 3, 0, 8, 1, np.zeros((3, 8), dtype=np.float32))
            with self.assertRaises(ValueError):
                clc.compare_captures(clc.load_capture(path_a), clc.load_capture(path_b))


class BatchDiscoveryTests(unittest.TestCase):
    def test_discover_pairs_matches_by_filename_and_reports_unmatched(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as td:
            dir_a = Path(td) / "cpu"
            dir_b = Path(td) / "metal"
            dir_a.mkdir()
            dir_b.mkdir()
            data = make_capture(rng, S=2, D=16)
            clc.write_capture(dir_a / "L005_S0002_T000128.bin", 5, 2, 128, 16, 0, data)
            clc.write_capture(dir_b / "L005_S0002_T000128.bin", 5, 2, 128, 16, 1, data)
            clc.write_capture(dir_a / "L010_S0002_T000128.bin", 10, 2, 128, 16, 0, data)
            clc.write_capture(dir_b / "L099_S0002_T000128.bin", 99, 2, 128, 16, 1, data)

            pairs, only_a, only_b = clc.discover_pairs(dir_a, dir_b)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0].name, "L005_S0002_T000128.bin")
            self.assertEqual(only_a, ["L010_S0002_T000128.bin"])
            self.assertEqual(only_b, ["L099_S0002_T000128.bin"])

    def test_run_batch_produces_one_row_per_matched_pair(self):
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as td:
            dir_a = Path(td) / "cpu"
            dir_b = Path(td) / "metal"
            dir_a.mkdir()
            dir_b.mkdir()
            for layer, S, T in ((5, 1, 128), (39, 16, 1024), (74, 64, 4096)):
                data = make_capture(rng, S, 32)
                clc.write_capture(dir_a / f"L{layer:03d}_S{S:04d}_T{T:06d}.bin", layer, S, T, 32, 0, data)
                clc.write_capture(dir_b / f"L{layer:03d}_S{S:04d}_T{T:06d}.bin", layer, S, T, 32, 1, data)
            rows = clc.run_batch(dir_a, dir_b)
        self.assertEqual(len(rows), 3)
        self.assertEqual(sorted(r["layer"] for r in rows), [5, 39, 74])
        for row in rows:
            self.assertEqual(row["max_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
