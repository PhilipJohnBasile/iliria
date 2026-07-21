from __future__ import annotations

import importlib.util
import math
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


ent = load_module("measure_expert_entropy", ROOT / "tools" / "measure_expert_entropy.py")


class EntropyTests(unittest.TestCase):
    def test_uniform_16_symbols_is_4_bits(self):
        counts = np.full(16, 100)
        self.assertAlmostEqual(ent.entropy_bits(counts), 4.0, places=12)

    def test_single_symbol_is_zero_bits(self):
        counts = np.zeros(16, dtype=int)
        counts[3] = 1000
        self.assertEqual(ent.entropy_bits(counts), 0.0)

    def test_known_binary_entropy(self):
        counts = np.zeros(16, dtype=int)
        counts[0] = 1
        counts[1] = 3
        expected = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
        self.assertAlmostEqual(ent.entropy_bits(counts), expected, places=12)

    def test_tile_entropy_weighted_mean_and_separation(self):
        # two homogeneous halves: global entropy 1 bit, tile entropy 0
        nib = np.array([0] * 64 + [1] * 64, dtype=np.uint8)
        counts = ent.block_counts(nib, 64)
        self.assertEqual(counts.shape, (2, 16))
        self.assertAlmostEqual(ent.tile_entropy(counts), 0.0, places=12)
        self.assertAlmostEqual(
            ent.entropy_bits(counts.sum(axis=0)), 1.0, places=12)

    def test_block_counts_includes_tail(self):
        nib = np.arange(10, dtype=np.uint8) % 16
        counts = ent.block_counts(nib, 4)
        self.assertEqual(counts.shape, (3, 16))
        self.assertEqual(int(counts.sum()), 10)

    def test_conditional_entropy_never_exceeds_global(self):
        rng = np.random.default_rng(7)
        O, per_row = 64, 32
        qs = rng.uniform(0.5, 2.0, O).astype(np.float32)
        # rows with larger scales get more concentrated symbols
        nib = np.empty(O * per_row, dtype=np.uint8)
        for r in range(O):
            spread = 1 + int(qs[r] * 6)
            nib[r * per_row:(r + 1) * per_row] = rng.integers(0, spread, per_row)
        h_cond = ent.scale_bin_conditional_entropy(nib, qs, O)
        h_global = ent.entropy_bits(np.bincount(nib, minlength=16))
        self.assertLessEqual(h_cond, h_global + 1e-9)


class HuffmanTests(unittest.TestCase):
    def test_lengths_satisfy_kraft_with_equality(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            counts = rng.integers(0, 1000, 16)
            if counts.sum() == 0 or (counts > 0).sum() < 2:
                continue
            lens = ent.huffman_code_lengths(counts)
            kraft = sum(2.0 ** -l for l in lens if l > 0)
            self.assertAlmostEqual(kraft, 1.0, places=12)

    def test_optimality_against_entropy_bounds(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            counts = rng.integers(1, 500, 16)
            lens = ent.huffman_code_lengths(counts)
            n = counts.sum()
            avg = float((counts * lens).sum()) / n
            h = ent.entropy_bits(counts)
            self.assertGreaterEqual(avg + 1e-12, h)
            self.assertLess(avg, h + 1.0)  # Huffman redundancy < 1 bit

    def test_roundtrip_random_blocks(self):
        rng = np.random.default_rng(5)
        for skew in (0.5, 1.5, 4.0):
            syms = np.clip(
                np.rint(rng.normal(0, skew, 4096)), -8, 7
            ).astype(np.int8).astype(np.int16) + 8
            syms = syms.astype(np.uint8)
            counts = np.bincount(syms, minlength=16)
            lens = ent.huffman_code_lengths(counts)
            data = ent.huffman_encode(syms, lens)
            back = ent.huffman_decode(data, lens, len(syms))
            self.assertTrue(np.array_equal(back, syms))
            # the stored-size formula matches the emitted stream exactly
            bits = int((counts * lens).sum())
            self.assertEqual(len(data), (bits + 7) // 8)

    def test_single_symbol_block_roundtrip(self):
        syms = np.full(100, 9, dtype=np.uint8)
        counts = np.bincount(syms, minlength=16)
        lens = ent.huffman_code_lengths(counts)
        self.assertEqual(lens[9], 1)
        data = ent.huffman_encode(syms, lens)
        back = ent.huffman_decode(data, lens, len(syms))
        self.assertTrue(np.array_equal(back, syms))

    def test_lengths_fit_four_bits(self):
        # adversarial fibonacci-ish counts drive maximum depth
        counts = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
                           233, 377, 610, 987])
        lens = ent.huffman_code_lengths(counts)
        self.assertLessEqual(int(lens.max()), 15)


class RansTests(unittest.TestCase):
    def test_quantize_freqs_sums_to_m_and_keeps_present_symbols(self):
        rng = np.random.default_rng(6)
        for _ in range(20):
            counts = rng.integers(0, 10000, 16)
            counts[rng.integers(0, 16)] = 0
            if counts.sum() == 0:
                continue
            f = ent.quantize_freqs(counts)
            self.assertEqual(int(f.sum()), ent.M_TOTAL)
            self.assertTrue(np.all(f[counts > 0] >= 1))
            self.assertTrue(np.all(f[counts == 0] == 0))

    def test_roundtrip_gaussian_block(self):
        rng = np.random.default_rng(8)
        syms = (np.clip(np.rint(rng.normal(0, 1.8, 8192)), -8, 7) + 8).astype(np.uint8)
        f = ent.quantize_freqs(np.bincount(syms, minlength=16))
        data = ent.rans_encode(syms, f)
        back = ent.rans_decode(data, f, len(syms))
        self.assertTrue(np.array_equal(back, syms))

    def test_roundtrip_skewed_and_uniform(self):
        rng = np.random.default_rng(9)
        skewed = rng.choice(16, size=4096, p=np.array(
            [0.7] + [0.02] * 15))  # heavy skew
        uniform = rng.integers(0, 16, 4096)
        for syms in (skewed.astype(np.uint8), uniform.astype(np.uint8)):
            f = ent.quantize_freqs(np.bincount(syms, minlength=16))
            back = ent.rans_decode(ent.rans_encode(syms, f), f, len(syms))
            self.assertTrue(np.array_equal(back, syms))

    def test_single_symbol_stream(self):
        syms = np.full(500, 4, dtype=np.uint8)
        f = ent.quantize_freqs(np.bincount(syms, minlength=16))
        back = ent.rans_decode(ent.rans_encode(syms, f), f, len(syms))
        self.assertTrue(np.array_equal(back, syms))

    def test_estimate_tracks_real_size(self):
        rng = np.random.default_rng(10)
        for sigma in (1.0, 1.8, 3.0):
            syms = (np.clip(np.rint(rng.normal(0, sigma, 8192)), -8, 7) + 8
                    ).astype(np.uint8)
            counts = np.bincount(syms, minlength=16)
            est = ent.rans_estimate_bytes(counts)
            real = ent.rans_real_bytes(syms, counts)
            self.assertLess(abs(est - real) / real, 0.01)

    def test_estimate_never_beats_entropy(self):
        rng = np.random.default_rng(11)
        syms = (np.clip(np.rint(rng.normal(0, 1.5, 8192)), -8, 7) + 8
                ).astype(np.uint8)
        counts = np.bincount(syms, minlength=16)
        h_bytes = ent.entropy_bits(counts) * len(syms) / 8
        est_payload = (ent.rans_estimate_bytes(counts) - ent.RANS_TABLE_BYTES
                       - ent.RANS_STATE_BYTES - ent.INDEX_BYTES)
        self.assertGreaterEqual(est_payload + 1, h_bytes)


class UnpackAndSamplingTests(unittest.TestCase):
    def test_unpack_matches_quant_container_dequant_convention(self):
        # low nibble first (even element), matching qc.dequant fmt 2
        raw = np.array([0x21, 0xFF, 0x08], dtype=np.uint8)
        nib = ent.unpack_nibbles(raw)
        self.assertEqual(nib.tolist(), [1, 2, 15, 15, 8, 0])

    def test_stratified_sampling_is_balanced_and_deterministic(self):
        cfg = {"n_experts": 256, "first_dense": 3, "n_layers": 78,
               "hidden": 6144, "moe_inter": 2048}
        s1 = ent.sample_experts(cfg, 256, 42)
        s2 = ent.sample_experts(cfg, 256, 42)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 256)
        self.assertEqual(len(set(s1)), 256)
        bands = {"early": 0, "mid": 0, "late": 0}
        for layer, _ in s1:
            bands[ent.band_of(layer)] += 1
        self.assertEqual(sum(bands.values()), 256)
        self.assertTrue(all(85 <= v <= 86 for v in bands.values()))

    def test_stored_size_formulas_include_metadata(self):
        counts = np.bincount(np.zeros(4096, dtype=np.uint8), minlength=16)
        h = ent.huffman_stored_bytes(counts)
        # 4096 one-bit codes = 512 payload bytes + 8 table + 4 index
        self.assertEqual(h, 512 + ent.HUFF_TABLE_BYTES + ent.INDEX_BYTES)


if __name__ == "__main__":
    unittest.main()
