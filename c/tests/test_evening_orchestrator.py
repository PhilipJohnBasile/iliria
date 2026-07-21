"""scripts/evening_orchestrator.sh: --dry-run proof of the state machine + logging contract.

Every test here runs the REAL script (subprocess, always with --dry-run) against a fully
isolated fixture environment (its own campaign-log.md / evening-status.json / marker file /
campaign-state.json, all under a tempdir via the ILI_EVENING_* overrides the script itself
documents) and a test-owned "gates chain" process (a short-lived `sleep`, never a real
run_container_gates.sh) -- so nothing here ever touches this machine's real, currently-running
overnight campaign state. quiesce_check.sh and run_abba_matrix.sh are ALSO always replaced by
tiny fixture scripts (ILI_EVENING_QUIESCE_BIN / ILI_EVENING_ABBA_BIN): the real
quiesce_check.sh reads live system telemetry (disk/CPU/thermal/battery) and takes tens of
real seconds per call, which would make these tests both slow and flaky (its pass/fail would
depend on this machine's actual state at test-run time, not on the scenario under test); ABBA's
own --dry-run is already proven elsewhere (commit 4df7ca2) and re-driving it here would mean
several more real quiesce_check.sh calls. What IS exercised for real: the capture-patch
apply/revert cycle against the real c/glm.c (git-only, fast, reverted in a finally block and
independently verified clean), and scripts/run_layer_capture_grid.py's real --dry-run mode
(synthesizes capture files, no serve/HTTP), so the numerical-capture leg of the pipeline is
proven end to end without an engine.

No real engine command (`make`, `ili run/serve/bench`, `glm`) is ever invoked by this suite.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
SCRIPT = C_DIR / "scripts" / "evening_orchestrator.sh"


def write_script(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def glm_c_is_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", "glm.c"],
                         cwd=str(C_DIR), capture_output=True, text=True, check=True)
    return out.stdout.strip() == ""


def force_revert_glm_c() -> None:
    subprocess.run(["git", "checkout", "--", "glm.c"], cwd=str(C_DIR), check=False)


class EveningOrchestratorTestBase(unittest.TestCase):
    def setUp(self):
        self.assertTrue(glm_c_is_clean(), "glm.c must be clean before the test starts")

        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.bench_dir = self.tmp_path / "bench-m5max"
        self.bench_dir.mkdir()

        self.campaign_log = self.bench_dir / "campaign-log.md"
        self.status_file = self.bench_dir / "evening-status.json"
        self.evening_out = self.bench_dir / "evening-out"
        self.marker_file = self.tmp_path / "marker-shard-reads-ok"
        self.gates_log = self.bench_dir / "gates.log"

        self.state_json = self.bench_dir / "campaign-state.json"
        self.state_json.write_text(json.dumps({
            "gate_b_comparator_correction": {
                "b0_command": "cd c && echo TEST-FIXTURE-B0-COMMAND && nohup sleep 1 >/dev/null 2>&1 &"
            }
        }))

        self.quiesce_pass = write_script(self.tmp_path / "quiesce_pass.sh", "#!/bin/bash\nexit 0\n")
        self.quiesce_fail = write_script(self.tmp_path / "quiesce_fail.sh", "#!/bin/bash\nexit 1\n")
        self.abba_pass = write_script(self.tmp_path / "abba_pass.sh",
                                      "#!/bin/bash\necho fake-abba ok \"$@\"\nexit 0\n")
        self.abba_fail = write_script(self.tmp_path / "abba_fail.sh",
                                      "#!/bin/bash\necho fake-abba boom >&2\nexit 1\n")

        self._gates_proc = None

    def tearDown(self):
        if self._gates_proc is not None and self._gates_proc.poll() is None:
            self._gates_proc.kill()
            self._gates_proc.wait(timeout=5)
        if not glm_c_is_clean():
            force_revert_glm_c()
            self.fail("glm.c was left modified after the test -- force-reverted; this is a bug "
                      "in the capture step's cleanup, not a false pass")
        self.tmp.cleanup()

    def start_fake_gates_chain(self, sleep_s: float = 0.4) -> int:
        """A test-owned stand-in for the armed run_container_gates.sh process: just a `sleep`
        the orchestrator will poll via kill -0, exactly like the real thing -- NEVER pgrep-
        discovered (every test passes ILI_EVENING_GATES_PID explicitly), so this suite can
        never latch onto this machine's real, currently-running gates chain.

        Reaped by a background thread the instant it exits: `kill -0` on an un-reaped zombie
        PID still reports "alive" on POSIX (the slot stays occupied until the parent calls
        wait()), which would make the orchestrator's poll loop hang forever waiting for a PID
        that has, in every meaningful sense, already exited. This is purely an artifact of this
        process being OUR direct child for test purposes -- the real run_container_gates.sh is
        never a child of evening_orchestrator.sh (it is reparented to launchd, which reaps it
        immediately), so this reaping concern does not exist in production, only here."""
        proc = subprocess.Popen(["sleep", str(sleep_s)])
        threading.Thread(target=proc.wait, daemon=True).start()
        self._gates_proc = proc
        return proc.pid

    def base_env(self, **overrides) -> dict:
        env = dict(os.environ)
        env.update({
            "ILI_EVENING_ATTEMPT_ID": "test-attempt",
            "ILI_EVENING_CAMPAIGN_LOG": str(self.campaign_log),
            "ILI_EVENING_STATUS_FILE": str(self.status_file),
            "ILI_EVENING_OUT": str(self.evening_out),
            "ILI_EVENING_GATES_LOG": str(self.gates_log),
            "ILI_EVENING_GATES_POLL_S": "1",
            "ILI_EVENING_QUIESCE_BIN": str(self.quiesce_pass),
            "ILI_EVENING_QUIESCE_RETRIES": "2",
            "ILI_EVENING_QUIESCE_INTERVAL_S": "0",
            "ILI_EVENING_ABBA_BIN": str(self.abba_pass),
            "ILI_EVENING_STATE_JSON": str(self.state_json),
            "ILI_EVENING_MARKER_FILE": str(self.marker_file),
        })
        env.update(overrides)
        return env

    def run_orchestrator(self, env: dict, timeout: float = 90) -> subprocess.CompletedProcess:
        # start_new_session=True puts the orchestrator (and anything it launches, e.g. the
        # --dry-run B0 mock) in its own process group, so a timeout here can be cleaned up by
        # killing that whole group -- scoped strictly to this test's own descendants -- rather
        # than risking a broad/pattern-based kill that could reach unrelated processes on a
        # shared machine.
        proc = subprocess.Popen(["bash", str(SCRIPT), "--dry-run"], cwd=str(C_DIR), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            raise AssertionError(
                f"evening_orchestrator.sh --dry-run did not finish within {timeout}s "
                f"(process group killed); stdout so far:\n{stdout}\nstderr so far:\n{stderr}")
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)

    def read_status(self) -> dict:
        return json.loads(self.status_file.read_text())

    def read_campaign_log(self) -> str:
        return self.campaign_log.read_text() if self.campaign_log.exists() else ""


class GateAPassedTest(EveningOrchestratorTestBase):
    def test_gate_a_passed_skips_recovery_sequence_and_exits_clean(self):
        pid = self.start_fake_gates_chain()
        self.gates_log.write_text(
            "[gates 12:00:00] gate A PASS (no obvious collapse; human should still skim outputs)\n")
        env = self.base_env(ILI_EVENING_GATES_PID=str(pid))

        result = self.run_orchestrator(env)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        status = self.read_status()
        self.assertEqual(status["status"], "passed")
        self.assertEqual(status["step"], "gate_a_passed")
        self.assertEqual(status["attempt_id"], "test-attempt")
        self.assertIsNotNone(status["finished"])

        log = self.read_campaign_log()
        self.assertIn("gate A PASSED", log)
        self.assertIn("skipping numerical captures and ABBA", log)
        for absent in ("numerical_captures", "quiesce", "abba_matrix", "b0_launch", "marker_file"):
            self.assertNotIn(absent, log, f"'{absent}' must never run on the gate-A-passed path")
        self.assertFalse(self.marker_file.exists(),
                         "marker file must NOT be written on the gate-A-passed path")


class GateIndeterminateTest(EveningOrchestratorTestBase):
    def test_indeterminate_verdict_takes_neither_branch(self):
        pid = self.start_fake_gates_chain()
        # Neither "GATE A FAIL" nor "gate A PASS" -- e.g. the chain aborted before gate A ran.
        self.gates_log.write_text("[gates 12:00:00] ABORT: engine still busy after wait\n")
        env = self.base_env(ILI_EVENING_GATES_PID=str(pid))

        result = self.run_orchestrator(env)

        self.assertEqual(result.returncode, 1)
        status = self.read_status()
        self.assertEqual(status["status"], "failed")
        log = self.read_campaign_log()
        self.assertIn("INDETERMINATE", log)
        self.assertFalse(self.marker_file.exists())
        for absent in ("numerical_captures", "quiesce", "abba_matrix", "b0_launch"):
            self.assertNotIn(absent, log)

    def test_missing_gates_log_is_indeterminate(self):
        pid = self.start_fake_gates_chain()
        # gates_log deliberately never written.
        env = self.base_env(ILI_EVENING_GATES_PID=str(pid))

        result = self.run_orchestrator(env)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.read_status()["status"], "failed")
        self.assertIn("gates log not found", result.stdout)
        self.assertIn("INDETERMINATE", self.read_campaign_log())


class GateAFailedRecoverySequenceTest(EveningOrchestratorTestBase):
    """The full gate-A-FAILED recovery sequence: numerical captures -> quiesce -> ABBA -> B0 ->
    marker, all succeeding. This is the only scenario that reaches every step, so it also
    carries the strongest correctness burden for the capture patch's apply/rebuild-skip/
    revert cycle against the real glm.c."""

    def test_full_recovery_sequence_all_steps_pass(self):
        pid = self.start_fake_gates_chain()
        self.gates_log.write_text(
            "[gates 12:00:00] GATE A FAIL: short output or heavy repetition detected -- "
            "SKIPPING gate B (bench) to save 8h. Human review required: foo\n")
        env = self.base_env(ILI_EVENING_GATES_PID=str(pid))

        result = self.run_orchestrator(env, timeout=120)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        status = self.read_status()
        self.assertEqual(set(status.keys()), {"step", "status", "started", "finished", "attempt_id"})
        self.assertEqual(status["step"], "complete")
        self.assertEqual(status["status"], "passed")
        self.assertIsNone(status["started"])
        self.assertIsNotNone(status["finished"])
        self.assertEqual(status["attempt_id"], "test-attempt")

        log = self.read_campaign_log()
        self.assertIn("gate A FAILED (registered prediction)", log)
        for step in ("numerical_captures", "quiesce", "abba_matrix", "b0_launch", "marker_file"):
            self.assertIn(f"step '{step}' done (ok)", log, f"expected step '{step}' to pass:\n{log}")
        self.assertIn("B0 command resolved: cd c && echo TEST-FIXTURE-B0-COMMAND", log)
        self.assertRegex(log, r"B0 launched \(pid=\d+, dry_run=1\)")
        self.assertIn("marker file written", log)

        self.assertTrue(self.marker_file.exists())

        capture_out = self.evening_out / "layer-capture"
        self.assertTrue((capture_out / "summary.csv").exists(), "compare_layer_captures.py must have run")
        cpu_files = list((capture_out / "captures-cpu").glob("*.bin"))
        metal_files = list((capture_out / "captures-metal").glob("*.bin"))
        # 3 layers x 5 S-values x 3 T-values (the default FAST grid) = 45 files per arm.
        self.assertEqual(len(cpu_files), 45)
        self.assertEqual(len(metal_files), 45)

        # glm.c must have been reverted (tearDown() also asserts this suite-wide; check it
        # explicitly here too since this is the test that actually exercises the patch).
        self.assertTrue(glm_c_is_clean())


class QuiesceFailurePathTest(EveningOrchestratorTestBase):
    def test_quiesce_exhausts_retries_then_fails_closed_but_still_launches_b0_and_marker(self):
        pid = self.start_fake_gates_chain()
        self.gates_log.write_text("[gates 12:00:00] GATE A FAIL: short output\n")
        env = self.base_env(ILI_EVENING_GATES_PID=str(pid),
                            ILI_EVENING_QUIESCE_BIN=str(self.quiesce_fail))

        result = self.run_orchestrator(env, timeout=120)

        self.assertEqual(result.returncode, 1)
        status = self.read_status()
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["step"], "complete")

        log = self.read_campaign_log()
        self.assertIn("step 'numerical_captures' done (ok)", log)
        self.assertIn("quiesce FAILED after 2 attempts", log)
        self.assertIn("step 'quiesce' FAILED", log)
        self.assertIn("step 'abba_matrix' SKIPPED", log)
        # Fail-closed exception: B0 + marker still run.
        self.assertIn("step 'b0_launch' done (ok)", log)
        self.assertIn("step 'marker_file' done (ok)", log)
        self.assertTrue(self.marker_file.exists())

        attempts = sorted((self.evening_out / "quiesce").glob("attempt-*.log"))
        self.assertEqual(len(attempts), 2, "must retry exactly QUIESCE_RETRIES times, no more/less")


class AbbaFailurePathTest(EveningOrchestratorTestBase):
    def test_abba_failure_fails_closed_but_still_launches_b0_and_marker(self):
        pid = self.start_fake_gates_chain()
        self.gates_log.write_text("[gates 12:00:00] GATE A FAIL: short output\n")
        env = self.base_env(ILI_EVENING_GATES_PID=str(pid),
                            ILI_EVENING_ABBA_BIN=str(self.abba_fail))

        result = self.run_orchestrator(env, timeout=120)

        self.assertEqual(result.returncode, 1)
        status = self.read_status()
        self.assertEqual(status["status"], "failed")

        log = self.read_campaign_log()
        self.assertIn("step 'numerical_captures' done (ok)", log)
        self.assertIn("step 'quiesce' done (ok)", log)
        self.assertIn("step 'abba_matrix' FAILED", log)
        # Fail-closed exception: B0 + marker still run.
        self.assertIn("step 'b0_launch' done (ok)", log)
        self.assertIn("step 'marker_file' done (ok)", log)
        self.assertTrue(self.marker_file.exists())


class CapturePatchDoesNotApplyTest(EveningOrchestratorTestBase):
    def test_patch_that_no_longer_applies_aborts_captures_but_still_runs_b0_and_marker(self):
        pid = self.start_fake_gates_chain()
        self.gates_log.write_text("[gates 12:00:00] GATE A FAIL: short output\n")
        broken_patch = self.tmp_path / "broken.patch"
        broken_patch.write_text(
            "diff --git a/c/glm.c b/c/glm.c\n"
            "index 0000000..1111111 100644\n"
            "--- a/c/glm.c\n"
            "+++ b/c/glm.c\n"
            "@@ -1,3 +1,4 @@\n"
            "+this line is not context that exists anywhere in glm.c, on purpose\n"
            " line two that also will not match\n"
            " line three that also will not match\n"
            " line four that also will not match\n")
        env = self.base_env(ILI_EVENING_GATES_PID=str(pid),
                            ILI_EVENING_CAPTURE_PATCH=str(broken_patch))

        result = self.run_orchestrator(env, timeout=120)

        self.assertEqual(result.returncode, 1)
        log = self.read_campaign_log()
        self.assertIn("numerical_captures ABORTED", log)
        self.assertIn("no longer applies", log)
        self.assertIn("step 'quiesce' SKIPPED", log)
        self.assertIn("step 'abba_matrix' SKIPPED", log)
        self.assertIn("step 'b0_launch' done (ok)", log)
        self.assertIn("step 'marker_file' done (ok)", log)
        self.assertTrue(self.marker_file.exists())
        # Never applied in the first place -- still clean.
        self.assertTrue(glm_c_is_clean())


if __name__ == "__main__":
    unittest.main()
