"""Tests for tools/mode15_container.py -- the mode-1.5 offline entropy-
encoder's shared codec/format module (registration: c/bench-m5max/mode15-
the lossless-pipeline registration; canonical Huffman per the codec race).

Three layers, cheapest-first (matches c/tests/test_codec_row_huff.c's own
shape: known vectors, edge sizes, synthetic-shape fuzz, random-length fuzz):
  1. unit-level checks on the canonical-code/LUT construction and the
     nibble pack/unpack helpers;
  2. round-trip fuzzing of encode_tensor/decode_tensor and the full tensor-
     blob pack/parse (incl. corruption/truncation -- the same properties
     gate_m15_g1.py's harness re-checks against a real container);
  3. an INDEPENDENT cross-check: compile tools/mode15_cross_check.c and run
     it against fixtures this module writes, so a bug shared by this
     module's own encode+decode (which would still pass layer 2's round
     trip) gets caught by disagreement with the REAL, unmodified
     codec_row_huff.h decoder -- see that C file's own header comment for
     the full argument. Skipped (not failed) if no C compiler is found.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # c/
TOOLS_DIR = ROOT / "tools"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m15 = load_module("mode15_container", TOOLS_DIR / "mode15_container.py")


def rng_dist(kind: str, rng: np.random.Generator, n: int = 16) -> np.ndarray:
    if kind == "uniform":
        p = np.ones(n)
    elif kind == "skewed":
        p = np.array([1000.0 if i == 7 else 1.0 for i in range(n)])
    elif kind == "singleton":
        p = np.array([1.0 if i == 3 else 0.0 for i in range(n)])
    elif kind == "two_symbol":
        p = np.array([5.0 if i in (2, 9) else 0.0 for i in range(n)])
    elif kind == "gaussian":
        x = np.arange(n) - 7.5
        p = np.exp(-(x * x) / (2 * 2.0 * 2.0))
    else:
        raise ValueError(kind)
    return p / p.sum()


def sample_nibbles(rng, O, I, dist):
    if O == 0:
        return np.zeros((0, I), dtype=np.uint8)
    p = rng_dist(dist, rng)
    return rng.choice(16, size=(O, I), p=p).astype(np.uint8)


# ------------------------------------------------------ nibble pack/unpack ----
class NibblePackTests(unittest.TestCase):
    def test_known_vector_low_even_high_odd(self):
        sym = np.array([[1, 2, 3, 4, 5]], dtype=np.uint8)
        packed = m15.pack_nibbles(sym, 5)
        self.assertEqual(int(packed[0, 0]), 1 | (2 << 4))
        self.assertEqual(int(packed[0, 1]), 3 | (4 << 4))
        self.assertEqual(int(packed[0, 2]), 5)

    def test_pack_unpack_roundtrip_odd_and_even(self):
        rng = np.random.default_rng(1)
        for I in (1, 2, 3, 4, 5, 6144, 2048, 2047):
            sym = rng.integers(0, 16, size=(4, I), dtype=np.uint8)
            packed = m15.pack_nibbles(sym, I)
            back = m15.unpack_nibbles(packed.reshape(-1), 4, I)
            np.testing.assert_array_equal(sym, back)


# --------------------------------------------------- canonical code build ----
class CanonicalCodeTests(unittest.TestCase):
    def test_two_symbol_alphabet_gets_length_one_codes(self):
        # two present symbols -> Huffman gives both length 1 (0 and 1)
        lengths = np.zeros(16, dtype=np.uint8)
        lengths[2] = 1
        lengths[9] = 1
        code_rev = m15.huff_canonical_codes(lengths)
        # canonical order ascending by symbol among equal lengths: symbol 2 -> code 0, symbol 9 -> code 1
        self.assertEqual(int(code_rev[2]), 0)
        self.assertEqual(int(code_rev[9]), 1)

    def test_singleton_alphabet_lut_matches_c_convention(self):
        lengths = np.zeros(16, dtype=np.uint8)
        lengths[5] = 1
        code_rev = m15.huff_canonical_codes(lengths)
        maxlen, lut_sym, lut_len = m15.build_lut(lengths, code_rev)
        self.assertEqual(maxlen, 1)
        self.assertTrue((lut_sym == 5).all())
        self.assertTrue((lut_len == 1).all())

    def test_lut_covers_every_present_code_prefix(self):
        rng = np.random.default_rng(3)
        counts = rng.integers(1, 1000, size=16)
        lengths = np.asarray(m15.huffman_code_lengths(counts), dtype=np.uint8)
        code_rev = m15.huff_canonical_codes(lengths)
        maxlen, lut_sym, lut_len = m15.build_lut(lengths, code_rev)
        # every present symbol's own codeword, zero-extended to maxlen bits,
        # must resolve to itself in the LUT (sanity check of the fill loop)
        for s in range(16):
            if lengths[s] == 0:
                continue
            self.assertEqual(int(lut_sym[int(code_rev[s])]), s)
            self.assertEqual(int(lut_len[int(code_rev[s])]), int(lengths[s]))

    def test_max_length_never_exceeds_15_for_16_symbol_alphabet(self):
        # Fibonacci-like worst case: counts 1,1,2,3,5,8,... forces the
        # deepest possible Huffman tree for n=16 leaves (depth n-1=15)
        fib = [1, 1]
        while len(fib) < 16:
            fib.append(fib[-1] + fib[-2])
        lengths = m15.huffman_code_lengths(np.array(fib))
        self.assertLessEqual(int(max(lengths)), m15.HUFF_MAXLEN)


# --------------------------------------------------- encode/decode roundtrip --
class EncodeDecodeRoundtripTests(unittest.TestCase):
    def _check_roundtrip(self, nibbles, ctx):
        O, I = nibbles.shape
        lengths, code_rev, maxlen, lut_sym, lut_len = m15.build_codebook(nibbles)
        payload, row_offsets = m15.encode_tensor(nibbles, lengths, code_rev)
        payload_arr = np.frombuffer(payload, dtype=np.uint8)
        decoded = m15.decode_tensor(payload_arr, row_offsets, O, I, maxlen, lut_sym, lut_len)
        np.testing.assert_array_equal(nibbles, decoded, err_msg=ctx)
        # never worse than 4 bits/symbol (raw storage) by more than the
        # unavoidable per-row byte-alignment padding (<1 byte/row)
        if O > 0 and I > 0:
            raw_bits = O * I * 4
            got_bits = int(row_offsets[-1]) * 8
            self.assertLessEqual(got_bits, raw_bits + O * 8, msg=ctx)

    def test_known_small_vectors(self):
        self._check_roundtrip(np.array([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8), "skewed row")
        self._check_roundtrip(np.array([list(range(16))], dtype=np.uint8), "one-of-each row")
        self._check_roundtrip(np.array([[7]], dtype=np.uint8), "singleton row")

    def test_edge_sizes(self):
        rng = np.random.default_rng(42)
        for O in (0, 1, 2, 3, 7, 8, 16, 63, 257):
            for I in (0, 1, 2, 3, 15, 16, 17, 2048, 6144):
                if O * I > 6144 * 300:  # keep the fuzz sweep fast
                    continue
                nib = sample_nibbles(rng, O, I, "gaussian")
                self._check_roundtrip(nib, f"O={O} I={I}")

    def test_real_row_shapes_all_distributions(self):
        rng = np.random.default_rng(99)
        shapes = [("gate/up", 2048, 6144), ("down", 6144, 2048)]
        dists = ["uniform", "skewed", "singleton", "two_symbol", "gaussian"]
        for name, O, I in shapes:
            for dist in dists:
                nib = sample_nibbles(rng, O, I, dist)
                self._check_roundtrip(nib, f"{name}/{dist}")

    def test_random_length_fuzz(self):
        rng = np.random.default_rng(2026)
        for _ in range(20):
            O = int(rng.integers(1, 64))
            I = int(rng.integers(1, 512))
            dist = rng.choice(["uniform", "skewed", "singleton", "gaussian"])
            nib = sample_nibbles(rng, O, I, str(dist))
            self._check_roundtrip(nib, f"fuzz O={O} I={I} dist={dist}")


# ------------------------------------------------------------ tensor blob ----
class TensorBlobTests(unittest.TestCase):
    def test_blob_roundtrip_byte_exact_vs_source(self):
        rng = np.random.default_rng(7)
        for name, O, I, dist in [("gateup", 64, 6144, "gaussian"), ("down", 128, 2048, "skewed")]:
            nib = sample_nibbles(rng, O, I, dist)
            blob = m15.make_tensor_blob(nib)
            src = m15.pack_nibbles(nib, I).tobytes()
            got = m15.decode_blob_to_packed_bytes(blob, expect_O=O, expect_I=I)
            self.assertEqual(src, got, name)

    def test_flipped_payload_byte_caught_by_checksum(self):
        rng = np.random.default_rng(8)
        nib = sample_nibbles(rng, 512, 2048, "gaussian")
        blob = bytearray(m15.make_tensor_blob(nib))
        blob[-1] ^= 0xFF
        with self.assertRaises(m15.Mode15ChecksumError):
            m15.parse_tensor_blob(bytes(blob))

    def test_flipped_row_offsets_byte_caught_by_checksum(self):
        rng = np.random.default_rng(9)
        nib = sample_nibbles(rng, 512, 2048, "gaussian")
        blob = bytearray(m15.make_tensor_blob(nib))
        blob[m15.TENSOR_HEADER_LEN + m15.LENGTHS_BYTES + 4] ^= 0x01
        with self.assertRaises((m15.Mode15ChecksumError, m15.Mode15FormatError)):
            m15.parse_tensor_blob(bytes(blob))

    def test_truncation_fails_closed(self):
        rng = np.random.default_rng(10)
        nib = sample_nibbles(rng, 512, 2048, "gaussian")
        blob = m15.make_tensor_blob(nib)
        for cut in (0, 1, m15.TENSOR_HEADER_LEN, len(blob) // 2, len(blob) - 1):
            with self.assertRaises(m15.Mode15FormatError):
                m15.parse_tensor_blob(blob[:cut])

    def test_bad_magic_fails_closed(self):
        rng = np.random.default_rng(11)
        nib = sample_nibbles(rng, 8, 64, "gaussian")
        blob = bytearray(m15.make_tensor_blob(nib))
        blob[0] ^= 0xFF
        with self.assertRaises(m15.Mode15FormatError):
            m15.parse_tensor_blob(bytes(blob))

    def test_block_checksum_localizes_a_single_corrupted_block(self):
        rng = np.random.default_rng(12)
        rows_per_block = 16
        nib = sample_nibbles(rng, 200, 2048, "gaussian")
        blob = bytearray(m15.make_tensor_blob(nib, rows_per_block=rows_per_block))
        parsed = m15.parse_tensor_blob(bytes(blob))
        n_blocks = parsed["n_blocks"]
        target = n_blocks // 2
        r0 = target * rows_per_block
        payload_off = (m15.TENSOR_HEADER_LEN + m15.LENGTHS_BYTES
                       + (parsed["O"] + 1) * 4 + n_blocks * 4)
        byte_idx = payload_off + int(parsed["row_offsets"][r0])
        blob[byte_idx] ^= 0xFF
        results = [m15.verify_block(bytes(blob), b) for b in range(n_blocks)]
        self.assertEqual([i for i, ok in enumerate(results) if not ok], [target])

    def test_o_i_mismatch_rejected(self):
        rng = np.random.default_rng(13)
        nib = sample_nibbles(rng, 16, 64, "gaussian")
        blob = m15.make_tensor_blob(nib)
        with self.assertRaises(m15.Mode15FormatError):
            m15.parse_tensor_blob(blob, expect_O=17)
        with self.assertRaises(m15.Mode15FormatError):
            m15.parse_tensor_blob(blob, expect_I=65)


# --------------------------------------------- independent C cross-check ----
class CCrossCheckTests(unittest.TestCase):
    """Compiles tools/mode15_cross_check.c (if a C compiler is available)
    and runs it against fixtures from THIS module, decoding via the REAL,
    unmodified codec_row_huff.h functions -- see that file's header for
    why this catches bugs a pure-Python round trip cannot."""

    @classmethod
    def setUpClass(cls):
        cls.cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
        if not cls.cc:
            return
        cls.tmpdir = tempfile.mkdtemp(prefix="mode15_xcheck_")
        cls.binary = os.path.join(cls.tmpdir, "mode15_cross_check")
        src = str(TOOLS_DIR / "mode15_cross_check.c")
        r = subprocess.run([cls.cc, "-O1", "-Wall", "-Wno-unused-function",
                             src, "-o", cls.binary], capture_output=True, text=True)
        if r.returncode != 0:
            cls.compile_error = r.stderr
            cls.binary = None
        else:
            cls.compile_error = None

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "tmpdir", None):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _run_case(self, name, O, I, dist):
        if not self.cc:
            self.skipTest("no C compiler (cc/clang/gcc) found on PATH")
        if not self.binary:
            self.fail(f"mode15_cross_check.c failed to compile:\n{self.compile_error}")
        rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
        nib = sample_nibbles(rng, O, I, dist)
        fixture = os.path.join(self.tmpdir, f"{name}.bin")
        m15.write_cross_check_fixture(fixture, nib)
        r = subprocess.run([self.binary, fixture], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0,
                          f"{name}: mode15_cross_check exited {r.returncode}\n"
                          f"stdout={r.stdout}\nstderr={r.stderr}")
        self.assertIn("PASS", r.stdout)

    def test_gate_up_shape_census_entropy(self):
        self._run_case("gateup", 128, 6144, "gaussian")

    def test_down_shape_census_entropy(self):
        self._run_case("down", 128, 2048, "gaussian")

    def test_skewed_distribution(self):
        self._run_case("skewed", 64, 6144, "skewed")

    def test_singleton_alphabet(self):
        self._run_case("singleton", 32, 2048, "singleton")

    def test_uniform_distribution(self):
        self._run_case("uniform", 32, 6144, "uniform")


if __name__ == "__main__":
    unittest.main()
