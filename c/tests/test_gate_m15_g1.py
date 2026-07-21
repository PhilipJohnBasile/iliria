"""Tests for bench-m5max/compression-gates/gate_m15_g1.py -- the mode-1.5
pipeline's G1 bit-exactness gate. Uses the same tiny synthetic int4
container as test_encode_mode15_container.py (no /path/to/models
access, no engine, no network).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = ROOT / "bench-m5max" / "compression-gates"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qc = load_module("quant_container", ROOT / "tools" / "quant_container.py")
m15 = load_module("mode15_container", ROOT / "tools" / "mode15_container.py")
enc = load_module("encode_mode15_container", ROOT / "tools" / "encode_mode15_container.py")
test_enc = load_module("test_encode_mode15_container", ROOT / "tests" / "test_encode_mode15_container.py")
gate = load_module("gate_m15_g1", GATE_DIR / "gate_m15_g1.py")


class SelftestTests(unittest.TestCase):
    def test_selftest_passes(self):
        report = gate.selftest()
        self.assertEqual(report["bit_exactness"]["status"], "PASS")
        self.assertEqual(report["corruption_truncation"]["status"], "PASS")
        self.assertEqual(report["overall"], "PASS")

    def test_cli_selftest_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.json")
            rc = gate.main(["--selftest", "--out", out])
            self.assertEqual(rc, 0)
            report = json.loads(Path(out).read_text())
            self.assertEqual(report["overall"], "PASS")


class CorruptionBatteryTests(unittest.TestCase):
    def test_every_subtest_passes_on_a_fresh_synthetic_tensor(self):
        result = gate.run_corruption_tests()
        self.assertEqual(result["status"], "PASS")
        for name, r in result["tests"].items():
            self.assertTrue(r["pass"], f"{name}: {r['detail']}")

    def test_result_shape_has_all_five_subtests(self):
        result = gate.run_corruption_tests()
        expected = {"flip_payload_byte", "flip_row_offsets_byte", "truncation_fails_closed",
                    "block_checksum_localizes_corruption", "bad_magic_rejected"}
        self.assertEqual(set(result["tests"]), expected)


class BitExactnessOnRealContainerShapeTests(unittest.TestCase):
    """Builds the SAME tiny int4 fixture + runs the SAME encoder as
    test_encode_mode15_container.py, then points gate_m15_g1 at the
    result -- this is the gate's own "does it actually work end to end"
    proof, independent of encode_mode15_container.py's own --verify-only
    (which only re-checks internal checksum self-consistency, not
    agreement with the true source bytes)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "tiny_i4"
        test_enc.make_int4_container(self.model, np.random.default_rng(31415))
        self.outdir = self.root / "mode15"
        rc = enc.main(["--indir", str(self.model), "--outdir", str(self.outdir),
                       "--min-free-gb", "0", "--allow-busy"])
        self.assertEqual(rc, 0)

    def tearDown(self):
        self.tmp.cleanup()

    def run_gate(self, extra=()):
        out = self.root / "g1-report.json"
        rc = gate.main(["--indir", str(self.model), "--outdir", str(self.outdir),
                        "--out", str(out), *extra])
        return rc, json.loads(out.read_text())

    def test_passes_on_a_correctly_encoded_container(self):
        rc, report = self.run_gate()
        self.assertEqual(rc, 0)
        self.assertEqual(report["overall"], "PASS")
        be = report["bit_exactness"]
        self.assertEqual(be["status"], "PASS")
        # shard 0 + shard 1: layers {3,4} and {5,6}, 4 experts, 3 projs = 24 each
        self.assertEqual(be["n_tensors_checked"], 48)
        self.assertEqual(be["n_failures"], 0)
        self.assertGreater(be["n_rows_checked"], 0)
        self.assertGreater(be["n_blocks_checked"], 0)

    def test_limit_restricts_to_first_n_shards(self):
        rc, report = self.run_gate(["--limit", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(report["bit_exactness"]["shards_checked"]), 1)
        self.assertEqual(report["bit_exactness"]["shards_checked"][0], "out-00000.safetensors")

    def test_detects_a_wrong_but_internally_consistent_blob(self):
        """A blob that is structurally valid and passes its OWN checksums
        (so it is not "corrupted" in the truncation/flip-a-byte sense) but
        was encoded from DIFFERENT nibbles than the true source -- this is
        the failure mode a hypothetical encoder data-plumbing bug (e.g.
        reading the wrong tensor, or an off-by-one row) would produce, and
        must be caught by the byte-for-byte comparison against --indir,
        not just by checksum validation."""
        cfg = qc.load_config(self.model)
        path = self.outdir / "out-00000.safetensors"
        name = "model.layers.3.mlp.experts.0.gate_proj.weight"
        O, I = qc.expert_shape(cfg, "gate_proj")

        rng = np.random.default_rng(999)
        wrong_nibbles = rng.integers(0, 16, size=(O, I), dtype=np.uint8)
        wrong_blob = m15.make_tensor_blob(wrong_nibbles)
        # sanity: this blob is internally self-consistent (passes its own
        # checksums), it is just semantically WRONG vs the true source
        m15.parse_tensor_blob(wrong_blob, expect_O=O, expect_I=I, verify_checksums=True)

        # rewrite the WHOLE shard with the substituted tensor (the wrong
        # blob's encoded length need not match the original -- Huffman
        # size is data-dependent -- so an in-place byte-preserving swap
        # would not be a valid edit; re-saving via qc.st_save recomputes
        # every tensor's offset, exactly like the real encoder would)
        index = qc.st_scan(str(self.outdir))
        tensors = {}
        for tname, entry in index.items():
            if entry["path"] != str(path):
                continue
            tensors[tname] = (np.frombuffer(wrong_blob, dtype=np.uint8).copy()
                              if tname == name else qc.st_read_tensor(entry))
        qc.st_save(str(path), tensors)

        rc, report = self.run_gate()
        self.assertEqual(rc, 1)
        self.assertEqual(report["overall"], "FAIL")
        be = report["bit_exactness"]
        self.assertEqual(be["status"], "FAIL")
        self.assertEqual(be["n_failures"], 1)
        self.assertEqual(be["failures"][0]["tensor"], name)
        self.assertIn("!= source", be["failures"][0]["error"])

    def test_detects_missing_source_shard(self):
        os.remove(self.model / "out-00001.safetensors")
        rc, report = self.run_gate()
        self.assertEqual(rc, 1)
        self.assertEqual(report["overall"], "FAIL")
        self.assertTrue(any("source shard missing" in f.get("error", "")
                            for f in report["bit_exactness"]["failures"]))


if __name__ == "__main__":
    unittest.main()
