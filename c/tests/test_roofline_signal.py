"""roofline_run.sh signal handling: a TERM mid-run must EXIT (non-zero) and NOT resume.

Pins a second reviewer's re-review blocking finding. `trap cleanup EXIT INT TERM` (a handler that does not
exit) let bash RESUME the script after the handler returned, so a signalled campaign released the
exclusivity lock, killed the watchdog + telemetry, then KEPT launching engine trials to completion
and reported success (rc 0) -- exactly the unprotected-measurement failure the whole system exists
to prevent. The fix is `trap 'exit 143' TERM` / `trap 'exit 130' INT` with cleanup on the EXIT trap
(which still fires when exiting from a signal, so teardown happens exactly once).

--mock only (no engine). Slow-ish: the driver's preflight quiesce samples for ~30s before the cold
phase, so reaching a mid-phase signal takes ~40s; one test, generous timeouts.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
ROOFLINE = C_DIR / "scripts" / "roofline_run.sh"


@unittest.skipUnless(
    sys.platform == "darwin",
    "scripts/roofline_run.sh refuses to run on non-macOS (its own preflight prints "
    "'FATAL: this harness targets macOS' and exits 1), so the cold phase this test waits "
    "for never starts.",
)
class SignalExitsNoResume(unittest.TestCase):
    def test_term_mid_run_exits_nonzero_and_does_not_resume(self):
        with tempfile.TemporaryDirectory() as d:
            lockdir = os.path.join(d, "lock")
            logf = os.path.join(d, "run.log")
            env = dict(os.environ,
                       ILI_TIMING_LOCK_DIR=lockdir,
                       ILI_ROOFLINE_RESULT_DIR=os.path.join(d, "out"),
                       ILI_ROOFLINE_COLD_S="120", ILI_ROOFLINE_STEADY_S="0",
                       ILI_ROOFLINE_SAMPLE_S="2", ILI_ROOFLINE_MOCK_RUN_S="1")
            with open(logf, "w") as lf:
                p = subprocess.Popen(["bash", str(ROOFLINE), "--mock", "--cold-only"],
                                     stdout=lf, stderr=subprocess.STDOUT, env=env)
            try:
                # Wait (<=90s) for the cold phase to actually start (past the ~30s quiesce window).
                deadline = time.monotonic() + 90
                started = False
                while time.monotonic() < deadline and p.poll() is None:
                    with open(logf) as fh:
                        if "phase=cold starting" in fh.read():
                            started = True
                            break
                    time.sleep(0.5)
                self.assertTrue(started, "cold phase never started -- cannot exercise a mid-run signal")
                time.sleep(2)  # let a trial or two launch
                p.send_signal(signal.SIGTERM)
                # the trap defers until the in-flight ~1s mock trial returns, then `exit 143`
                rc = p.wait(timeout=25)
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()
            with open(logf) as fh:
                log = fh.read()
            self.assertNotEqual(rc, 0, "driver must exit NON-zero on TERM (pre-fix it resumed to rc 0)")
            self.assertNotIn("roofline run complete", log,
                             "driver RESUMED to completion after TERM -- the signal trap did not exit")
            self.assertFalse(os.path.isdir(lockdir), "the exclusivity lock must be released by cleanup on TERM")


if __name__ == "__main__":
    unittest.main()
