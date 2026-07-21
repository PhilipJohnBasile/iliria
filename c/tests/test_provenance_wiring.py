"""--mock/--dry-run proof that the executable-provenance system (scripts/provenance.sh) is
actually wired into the existing engine-sequencing scripts, not just present as a standalone
tool. Drives the REAL scripts/roofline_run.sh --mock and scripts/evening_orchestrator.sh
--dry-run (same fixture conventions as tests/test_roofline_run.py and
tests/test_evening_orchestrator.py -- deliberately not importing those modules, to keep this
proof self-contained and independent of their fixture internals) and asserts a
provenance-<attempt_id>.json actually lands next to each script's own results, with the
--mock/--dry-run binary substitution (./ili, since no real ./glm runs) this repo's dry-run
convention requires.

scripts/run_abba_matrix.sh and scripts/run_container_gates.sh are also wired (see their own
diffs) but are not driven here: run_abba_matrix.sh --dry-run's own proof already costs several
real quiesce_check.sh calls per its module docstring (multiple minutes), and
run_container_gates.sh has no dry-run mode and no existing test scaffold at all (it drives a
real ~1h+ gate against real hardware/model directories) -- both were verified by manual
end-to-end invocation during development instead; see the task report for what was run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]


def write_script(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


@unittest.skipUnless(
    sys.platform == "darwin",
    "scripts/roofline_run.sh refuses to run on non-macOS (its own preflight prints "
    "'FATAL: this harness targets macOS' and exits 1); EveningOrchestratorProvenanceWiringTest "
    "below is unaffected and still runs on CI.",
)
class RooflineRunProvenanceWiringTest(unittest.TestCase):
    def test_mock_run_drops_a_provenance_manifest_using_ili_as_the_binary(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "roofline-out"
            env = dict(os.environ)
            env.update({
                "ILI_ROOFLINE_RESULT_DIR": str(out_dir),
                "ILI_ROOFLINE_COLD_S": "2",
                "ILI_ROOFLINE_STEADY_S": "2",
                "ILI_ROOFLINE_SAMPLE_S": "1",
                "ILI_ROOFLINE_MOCK_RUN_S": "0.1",
                "QUIESCE_SAMPLES": "1",
            })
            proc = subprocess.run(
                ["bash", str(C_DIR / "scripts" / "roofline_run.sh"), "--mock"],
                env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

            manifests = list(out_dir.glob("provenance-*.json"))
            self.assertEqual(len(manifests), 1,
                             f"expected exactly one provenance manifest in {out_dir}")
            manifest = json.loads(manifests[0].read_text())
            self.assertEqual(manifest["binary_path"], str(C_DIR / "ili"))
            self.assertTrue(manifest["binary_sha256"])
            # roofline_run.sh passes --prompt "${prompts[*]}" -- the fixed 3-prompt set joined;
            # only its hash is in the manifest, but a hash is only meaningful if it was
            # actually computed (never null) from something.
            self.assertIsNotNone(manifest["prompt_or_dataset_hash"])
            self.assertEqual(manifest["prompt_or_dataset_source"], "inline")
            self.assertEqual(manifest["quiesce"]["skipped"], False)


class EveningOrchestratorProvenanceWiringTest(unittest.TestCase):
    def test_dry_run_drops_a_provenance_manifest_using_ili_as_the_binary(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            bench_dir = tmp_path / "bench-m5max"
            bench_dir.mkdir()
            evening_out = bench_dir / "evening-out"
            gates_log = bench_dir / "gates.log"
            gates_log.write_text(
                "[gates 12:00:00] gate A PASS (no obvious collapse; human should still skim outputs)\n")
            state_json = bench_dir / "campaign-state.json"
            state_json.write_text(json.dumps({
                "gate_b_comparator_correction": {
                    "b0_command": "cd c && echo TEST-B0 && nohup sleep 1 >/dev/null 2>&1 &"}}))
            quiesce_pass = write_script(tmp_path / "quiesce_pass.sh", "#!/bin/bash\nexit 0\n")

            # A test-owned stand-in for the armed run_container_gates.sh process, reaped by a
            # background thread the instant it exits: `kill -0` on an un-reaped zombie PID
            # still reports "alive" on POSIX (the slot stays occupied until the parent calls
            # wait()), which would make wait_for_gates()'s poll loop hang forever -- see
            # tests/test_evening_orchestrator.py's start_fake_gates_chain() for the same
            # pattern and its full explanation.
            gates_proc = subprocess.Popen(["sleep", "0.4"])
            threading.Thread(target=gates_proc.wait, daemon=True).start()
            try:
                env = dict(os.environ)
                env.update({
                    "ILI_EVENING_ATTEMPT_ID": "wiring-proof",
                    "ILI_EVENING_CAMPAIGN_LOG": str(bench_dir / "campaign-log.md"),
                    "ILI_EVENING_STATUS_FILE": str(bench_dir / "evening-status.json"),
                    "ILI_EVENING_OUT": str(evening_out),
                    "ILI_EVENING_GATES_LOG": str(gates_log),
                    "ILI_EVENING_GATES_POLL_S": "1",
                    "ILI_EVENING_GATES_PID": str(gates_proc.pid),
                    "ILI_EVENING_QUIESCE_BIN": str(quiesce_pass),
                    "ILI_EVENING_STATE_JSON": str(state_json),
                    "ILI_EVENING_MARKER_FILE": str(tmp_path / "marker"),
                })
                proc = subprocess.run(
                    ["bash", str(C_DIR / "scripts" / "evening_orchestrator.sh"), "--dry-run"],
                    cwd=str(C_DIR), env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            finally:
                if gates_proc.poll() is None:
                    gates_proc.kill()
                gates_proc.wait(timeout=5)

            manifest_path = evening_out / "provenance-wiring-proof.json"
            self.assertTrue(manifest_path.is_file(),
                            f"expected {manifest_path} to exist; dir has: {list(evening_out.iterdir())}")
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["binary_path"], str(C_DIR / "ili"))
            self.assertEqual(manifest["quiesce"]["pass"], True)


if __name__ == "__main__":
    unittest.main()
