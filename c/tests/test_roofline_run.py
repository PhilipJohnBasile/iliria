"""End-to-end test for scripts/roofline_run.sh --mock (docs/performance-theory.json
p2-m5-gpu-tensor-path-probe): runs the real shell harness -- real quiesce_check.sh gate,
real phase loop, real telemetry sampler, real manifest/report generation -- against the
synthetic mock-log engine (tools/roofline_report.py's own generator), so the whole pipeline
is proven without a real model, real GPU work, or real hardware risk.

Durations and quiesce_check.sh's own sample counts are tuned down via env so this stays a
reasonably fast (roughly under a minute) integration test; QUIESCE_SAMPLES/QUIESCE_DISK_MBS
are quiesce_check.sh's own supported tuning knobs, not a bypass of its logic.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "roofline_run.sh"


def run_mock(result_dir: Path, extra_env=None, extra_args=None):
    env = dict(os.environ)
    env.update({
        "ILI_ROOFLINE_RESULT_DIR": str(result_dir),
        "ILI_ROOFLINE_COLD_S": "3",
        "ILI_ROOFLINE_STEADY_S": "3",
        "ILI_ROOFLINE_SAMPLE_S": "1",
        "ILI_ROOFLINE_MOCK_RUN_S": "0.2",
        "QUIESCE_SAMPLES": "1",       # quiesce_check.sh's own knob: 1x5s instead of 6x5s
    })
    if extra_env:
        env.update(extra_env)
    args = [str(HARNESS), "--mock"] + (extra_args or [])
    return subprocess.run(
        ["bash", *args], env=env, capture_output=True, text=True, timeout=180,
    )


@unittest.skipUnless(sys.platform == "darwin", "roofline_run.sh targets macOS")
class RooflineRunMockTests(unittest.TestCase):
    def test_mock_end_to_end_produces_a_complete_result_dir(self):
        with_tmp = ROOT / "bench-m5max" / "_test-roofline-mock-tmp"
        proc = run_mock(with_tmp)
        try:
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            self.assertTrue((with_tmp / "report.md").exists())
            self.assertTrue((with_tmp / "manifest.json").exists())
            manifest = json.loads((with_tmp / "manifest.json").read_text())
            self.assertTrue(manifest["mock"])
            phases = {r["phase"] for r in manifest["runs"]}
            self.assertEqual(phases, {"cold", "steady"})
            self.assertGreater(len(manifest["runs"]), 0)
            for r in manifest["runs"]:
                self.assertTrue((with_tmp / r["log"]).exists())
                # attempt_id must be embedded in every artifact filename (campaign-state.json
                # convention this harness follows).
                self.assertIn(manifest["attempt_id"], r["log"])

            for phase in ("cold", "steady"):
                csv_path = with_tmp / manifest["telemetry"][phase]
                self.assertTrue(csv_path.exists())
                with csv_path.open(newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertGreater(len(rows), 0, f"expected at least one telemetry sample in {phase}")
                self.assertTrue(all(row["phase"] == phase for row in rows))

            system_dir = with_tmp / "system"
            self.assertTrue(any(p.name.endswith("preflight-quiesce.txt") for p in system_dir.iterdir()))
            self.assertTrue(any(p.name.endswith("postflight-quiesce.txt") for p in system_dir.iterdir()))

            report = (with_tmp / "report.md").read_text()
            self.assertIn("# Dual roofline report", report)
            self.assertIn("## Per-kernel-class roofline table", report)
            self.assertIn("router", report)
        finally:
            import shutil
            shutil.rmtree(with_tmp, ignore_errors=True)

    def test_cold_only_flag_skips_the_steady_phase(self):
        with_tmp = ROOT / "bench-m5max" / "_test-roofline-mock-cold-only-tmp"
        proc = run_mock(with_tmp, extra_args=["--cold-only"])
        try:
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            manifest = json.loads((with_tmp / "manifest.json").read_text())
            phases = {r["phase"] for r in manifest["runs"]}
            self.assertEqual(phases, {"cold"})
        finally:
            import shutil
            shutil.rmtree(with_tmp, ignore_errors=True)

    def test_conflicting_cold_only_and_steady_only_flags_are_rejected_fast(self):
        proc = subprocess.run(
            ["bash", str(HARNESS), "--mock", "--cold-only", "--steady-only"],
            capture_output=True, text=True, timeout=20,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("mutually exclusive", proc.stderr)

    def test_real_mode_without_model_dir_fails_fast_without_mock(self):
        proc = subprocess.run(
            ["bash", str(HARNESS)], capture_output=True, text=True, timeout=20,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("MODEL_DIR is required", proc.stderr)


if __name__ == "__main__":
    unittest.main()
