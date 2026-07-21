"""scripts/run_layer_capture_grid.py: --dry-run synthesis (no engine, no serve, no HTTP) --
the leg of the capture pipeline scripts/evening_orchestrator.sh's own --dry-run tests exercise
as a subprocess. This file unit-tests the module's grid math and file output directly."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
SCRIPT = C_DIR / "scripts" / "run_layer_capture_grid.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rlcg = load_module("run_layer_capture_grid", SCRIPT)
clc = load_module("compare_layer_captures", C_DIR / "tools" / "compare_layer_captures.py")


class BuildFillerMessageTest(unittest.TestCase):
    def test_scales_with_requested_tokens(self):
        small = rlcg.build_filler_message(1, "tag")
        big = rlcg.build_filler_message(256, "tag")
        self.assertLess(len(small["content"]), len(big["content"]))
        self.assertEqual(small["role"], "user")

    def test_never_negative_padding_for_tiny_token_counts(self):
        # A 1-token message's raw char budget (4) is shorter than the tag prefix itself --
        # must not raise or produce a negative repeat count.
        msg = rlcg.build_filler_message(1, "T128-S1")
        self.assertIsInstance(msg["content"], str)


class DryRunSynthesisTest(unittest.TestCase):
    def test_writes_expected_file_count_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "cpu"
            rc = subprocess.run(
                [sys.executable, str(SCRIPT), "--capture-dir", str(out), "--metal-prefill", "0",
                 "--layers", "5,39,74", "--s-values", "1,4,16,64,256", "--t-values", "128,1024,4096",
                 "--dry-run"],
                capture_output=True, text=True)
            self.assertEqual(rc.returncode, 0, rc.stdout + rc.stderr)

            bins = sorted(p.name for p in out.glob("*.bin"))
            self.assertEqual(len(bins), 3 * 5 * 3)   # layers x S values x T values
            self.assertIn("L005_S0001_T000128.bin", bins)
            self.assertIn("L074_S0256_T004096.bin", bins)

            manifest = clc.load_capture(out / "L005_S0001_T000128.bin")
            self.assertEqual(manifest.layer, 5)
            self.assertEqual(manifest.S, 1)
            self.assertEqual(manifest.pos_base, 128)
            self.assertEqual(manifest.metal_prefill, 0)

    def test_metal_arm_is_labeled_and_perturbed_relative_to_cpu_arm(self):
        with tempfile.TemporaryDirectory() as td:
            cpu_dir, metal_dir = Path(td) / "cpu", Path(td) / "metal"
            for arm, out in ((0, cpu_dir), (1, metal_dir)):
                rc = subprocess.run(
                    [sys.executable, str(SCRIPT), "--capture-dir", str(out),
                     "--metal-prefill", str(arm), "--layers", "5", "--s-values", "4",
                     "--t-values", "128", "--dry-run"],
                    capture_output=True, text=True)
                self.assertEqual(rc.returncode, 0, rc.stdout + rc.stderr)

            cpu_cap = clc.load_capture(cpu_dir / "L005_S0004_T000128.bin")
            metal_cap = clc.load_capture(metal_dir / "L005_S0004_T000128.bin")
            self.assertEqual(cpu_cap.metal_prefill, 0)
            self.assertEqual(metal_cap.metal_prefill, 1)
            # Same shape, deliberately-perturbed (not identical, not wildly different) values --
            # compare_layer_captures.py must be able to run on the pair without error.
            result = clc.compare_captures(cpu_cap, metal_cap)
            self.assertEqual(result["S"], 4)
            self.assertGreater(result["max_abs"], 0.0)

    def test_full_grid_is_comparable_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            cpu_dir, metal_dir = Path(td) / "captures-cpu", Path(td) / "captures-metal"
            for arm, out in ((0, cpu_dir), (1, metal_dir)):
                subprocess.run(
                    [sys.executable, str(SCRIPT), "--capture-dir", str(out),
                     "--metal-prefill", str(arm), "--dry-run"],
                    capture_output=True, text=True, check=True)
            rows = clc.run_batch(cpu_dir, metal_dir)
            self.assertEqual(len(rows), 3 * 5 * 3)


if __name__ == "__main__":
    unittest.main()
