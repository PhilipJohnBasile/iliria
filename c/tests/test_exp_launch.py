"""c/scripts/exp_launch.sh: process-group ownership + monitor-mode restore.

Pins the 2026-07-15 fixes and the file's core guarantees (its own docstring cited a
c/tests/test_exp_launch.py that did not exist -- this makes the claim true):
  - exp_launch_restore_m restores monitor mode IN the calling shell. The old exp_launch_end was
    invoked via <(...) (a subshell), so its `set +m` never propagated and monitor mode leaked ON.
  - under `set -m`, a backgrounded pipeline leader gets its OWN process group (pgid == its pid),
    distinct from the calling shell's group.
  - exp_launch_cancel kills the WHOLE group (leader + child) and verifies no descendant survives.

Shell-only; no engine, no model, no network.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
EXP = C_DIR / "scripts" / "exp_launch.sh"


def run_bash(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


class MonitorMode(unittest.TestCase):
    def test_restore_m_clears_monitor_mode_in_shell(self):
        out = run_bash(
            f'source "{EXP}"; '
            f'exp_launch_begin; case "$-" in *m*) echo BEGIN_ON;; esac; '
            f'exp_launch_restore_m; case "$-" in *m*) echo STILL_ON;; *) echo RESTORED_OFF;; esac'
        ).stdout
        self.assertIn("BEGIN_ON", out)
        self.assertIn("RESTORED_OFF", out)
        self.assertNotIn("STILL_ON", out)


class OwnGroup(unittest.TestCase):
    def test_backgrounded_leader_gets_own_pgid(self):
        out = run_bash(
            f'source "{EXP}"; shellpgid=$(ps -o pgid= -p $$ | tr -d " "); '
            f'exp_launch_begin; sleep 5 & p=$!; exp_launch_restore_m; '
            f'lpgid=$(ps -o pgid= -p $p | tr -d " "); '
            f'echo "pid=$p pgid=$lpgid shell=$shellpgid"; kill $p 2>/dev/null'
        ).stdout
        m = re.search(r"pid=(\d+) pgid=(\d+) shell=(\d+)", out)
        self.assertIsNotNone(m, out)
        pid, pgid, shell = m.groups()
        self.assertEqual(pid, pgid, "leader pgid should equal its own pid under set -m")
        self.assertNotEqual(pgid, shell, "leader group should differ from the shell's group")


class CancelGroup(unittest.TestCase):
    def test_cancel_kills_whole_group(self):
        # Leader = bash running sequential sleeps; its sleep child shares the leader's group, so a
        # correct cancel (kill -TERM -PGID) must reap BOTH, not just the leaf.
        out = run_bash(
            f'source "{EXP}"; exp_launch_begin; '
            f"bash -c 'sleep 30; sleep 30' & p=$!; exp_launch_restore_m; "
            f'pgid=$(ps -o pgid= -p $p | tr -d " "); '
            f'exp_launch_cancel "$pgid" "" 5 >/dev/null 2>&1; echo cancel_rc=$?; '
            f'pgrep -g "$pgid" >/dev/null 2>&1 && echo GROUP_ALIVE || echo GROUP_CLEAN'
        ).stdout
        self.assertIn("cancel_rc=0", out)
        self.assertIn("GROUP_CLEAN", out)
        self.assertNotIn("GROUP_ALIVE", out)


if __name__ == "__main__":
    unittest.main()
