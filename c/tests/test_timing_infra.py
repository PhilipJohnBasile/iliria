"""c/tools/timing_watchdog.py + c/scripts/timing_lock.sh: unit + integration tests pinning the
2026-07-15 fixes to the timing-exclusivity infrastructure (second-review pass, VERDICT: BROKEN).

Each test pins one of the six bugs an adversarial review DEMONSTRATED by executing the code, so
a regression re-opens a specific, named hole:
  bug1  read_thermal_speed_limit_pct: 'No thermal warning' output (no CPU_Speed_Limit line) is
        NOMINAL (100), not a telemetry gap -- the old code hard-killed EVERY run at sample 1.
  bug2  read_disk_mbs: iostat must use -c 2 and parse the live delta (last line), not the single
        since-boot report (~3719 MB/s constant) that falsely breached disk every run.
  bug3  timing_lock.sh stale-reap: concurrent acquirers must not both reap+grant the same stale
        lock (was 29/30 double-grants); the reap-mutex serializes reclaim.
  bug5  read_on_ac_power + classify_sample: on-battery is a HARD blocker; AC telemetry gap fails
        closed; thermal/pageout are HARD, cpu/disk are SUSTAINED (the severity split).
  bug6  unexpected_processes: the watchdog must skip its OWN pid (its argv carries the analysis
        pattern -> self-match -> self-kill) and honor --exclude-pids.
  bug4  own_pgids is a SET: an allowlisted per-trial experiment pgid is non-foreign only while
        listed; read_pgid_file re-reads it; --no-terminate marks INVALID without killing.

No engine, no model, no network. Watchdog telemetry functions are tested by monkeypatching the
module-level _run; the lock is exercised as a black box via its standalone CLI in a tempdir.
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
WATCHDOG = C_DIR / "tools" / "timing_watchdog.py"
LOCK_SH = C_DIR / "scripts" / "timing_lock.sh"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wd = load_module("timing_watchdog", WATCHDOG)


class ThermalParse(unittest.TestCase):
    """bug1: present-but-no-CPU_Speed_Limit is nominal (100); only empty output is a gap."""

    def setUp(self):
        self._orig = wd._run

    def tearDown(self):
        wd._run = self._orig

    def test_no_thermal_warning_is_nominal_100(self):
        wd._run = lambda *a, **k: "Note: No thermal warning level has been recorded\n"
        self.assertEqual(wd.read_thermal_speed_limit_pct(), 100)

    def test_empty_output_is_gap_none(self):
        wd._run = lambda *a, **k: ""
        self.assertIsNone(wd.read_thermal_speed_limit_pct())

    def test_throttled_value_parsed(self):
        wd._run = lambda *a, **k: "CPU_Speed_Limit = 80\n"
        self.assertEqual(wd.read_thermal_speed_limit_pct(), 80)


class DiskParse(unittest.TestCase):
    """bug2: iostat -c 2, parse the live last-line delta (not the since-boot single report)."""

    def setUp(self):
        self._orig = wd._run
        self.calls = []

    def tearDown(self):
        wd._run = self._orig

    def test_uses_c2_and_parses_last_line(self):
        two_report = ("          disk0\n    KB/t  tps  MB/s\n"
                      "   22.55   12   3719.88\n    9.10    3   0.42\n")

        def fake(cmd, *a, **k):
            self.calls.append(cmd)
            return two_report

        wd._run = fake
        self.assertAlmostEqual(wd.read_disk_mbs(), 0.42, places=2)
        cmd = self.calls[0]
        self.assertEqual(cmd[cmd.index("-c") + 1], "2", "must request 2 reports for a live delta")

    def test_empty_is_none(self):
        wd._run = lambda *a, **k: ""
        self.assertIsNone(wd.read_disk_mbs())


class AcPower(unittest.TestCase):
    """bug5: AC power read; hard-block on battery, fail-closed on empty."""

    def setUp(self):
        self._orig = wd._run

    def tearDown(self):
        wd._run = self._orig

    def test_ac(self):
        wd._run = lambda *a, **k: "Now drawing from 'AC Power'\n"
        self.assertTrue(wd.read_on_ac_power())

    def test_battery(self):
        wd._run = lambda *a, **k: "Now drawing from 'Battery Power'\n"
        self.assertFalse(wd.read_on_ac_power())

    def test_empty_none(self):
        wd._run = lambda *a, **k: ""
        self.assertIsNone(wd.read_on_ac_power())


class Classify(unittest.TestCase):
    """AC-loss / thermal / pageout / telemetry-gap are HARD; cpu / disk are INFORMATIONAL only
    (recorded, never enforced -- streaming-coupled). classify_sample now returns just the hard list."""

    def _sample(self, **kw):
        d = dict(timestamp="t", foreign_cpu_pct=0.0, disk_mbs=0.0, pageout_rate=0.0,
                 thermal_speed_limit_pct=100, on_ac_power=True, unexpected_processes=[], errors=[])
        d.update(kw)
        return wd.Sample(**d)

    def test_battery_is_hard(self):
        hard = wd.classify_sample(self._sample(on_ac_power=False), wd.Thresholds())
        self.assertTrue(any("BATTERY" in h for h in hard))

    def test_thermal_is_hard(self):
        hard = wd.classify_sample(self._sample(thermal_speed_limit_pct=70), wd.Thresholds())
        self.assertTrue(any("thermal" in h for h in hard))

    def test_pageout_is_hard(self):
        hard = wd.classify_sample(self._sample(pageout_rate=999.0), wd.Thresholds())
        self.assertTrue(any("pageout" in h.lower() for h in hard))

    def test_cpu_is_informational_not_enforced(self):
        # aggregate foreign CPU is recorded but must NOT produce a hard reason (kernel_task-coupled)
        self.assertEqual(wd.classify_sample(self._sample(foreign_cpu_pct=99999.0), wd.Thresholds()), [])

    def test_disk_is_informational_not_enforced(self):
        self.assertEqual(wd.classify_sample(self._sample(disk_mbs=99999.0), wd.Thresholds()), [])

    def test_telemetry_gap_is_hard(self):
        hard = wd.classify_sample(self._sample(errors=["disk gap"]), wd.Thresholds())
        self.assertTrue(any("disk gap" in h for h in hard))


class Foreign(unittest.TestCase):
    """bug6 + bug4: own_pid skip (self-kill guard), exclude_pids, allowlist pgids (own_pgids set)."""

    def _rows(self):
        u = os.getuid()
        return [
            wd.ProcRow(os.getpid(), 111, u, 5.0, "python3 timing_watchdog.py --python-analysis-pattern entropy"),
            wd.ProcRow(4242, 222, u, 5.0, "python3 measure_expert_entropy.py"),
            wd.ProcRow(7000, 333, u, 5.0, "glm"),
        ]

    def test_own_pid_skipped_self_kill_guard(self):
        out = wd.unexpected_processes(self._rows(), frozenset({999}), os.getpid(), frozenset(),
                                      re.compile("^glm$"), re.compile("^cc$"), re.compile("entropy"))
        self.assertFalse(any(str(os.getpid()) in o for o in out))

    def test_foreign_still_caught(self):
        out = wd.unexpected_processes(self._rows(), frozenset({999}), os.getpid(), frozenset(),
                                      re.compile("^glm$"), re.compile("^cc$"), re.compile("entropy"))
        self.assertTrue(any("4242" in o for o in out), "foreign py-analysis must be flagged")
        self.assertTrue(any("7000" in o for o in out), "foreign glm must be flagged")

    def test_exclude_pids(self):
        # own_pid (clears row 0) + exclude_pids (clears rows 1,2) together remove every match
        out = wd.unexpected_processes(self._rows(), frozenset({999}), os.getpid(),
                                      frozenset({4242, 7000}),
                                      re.compile("^glm$"), re.compile("^cc$"), re.compile("entropy"))
        self.assertEqual(out, [])

    def test_allowlist_pgid_makes_non_foreign(self):
        # glm in pgid 333 is foreign UNLESS 333 is in own_pgids (the per-trial allowlist)
        out = wd.unexpected_processes(self._rows(), frozenset({333}), 1, frozenset(),
                                      re.compile("^glm$"), re.compile("^zzz$"), re.compile("zzz"))
        self.assertFalse(any("7000" in o for o in out))

    def test_foreign_cpu_excludes_allowed_own_and_excluded(self):
        u = os.getuid()
        rows = [wd.ProcRow(10, 100, u, 30.0, "x"), wd.ProcRow(11, 200, u, 40.0, "y"),
                wd.ProcRow(12, 300, u, 50.0, "z")]
        # own_pgids={100} drops pid10; own_pid=11 drops pid11; exclude={12} drops pid12 -> 0
        self.assertEqual(wd.foreign_cpu_pct(rows, frozenset({100}), 11, frozenset({12})), 0.0)
        # nothing excluded -> 30+40+50
        self.assertEqual(wd.foreign_cpu_pct(rows, frozenset({999}), 0, frozenset()), 120.0)


class ForeignUserProc(unittest.TestCase):
    """The GENERIC contamination net (replaces the disabled aggregate-CPU gate): any user-owned
    process outside the allowed groups that persists or spikes -- regardless of NAME -- is caught,
    while root/system (kernel_task = the engine's own I/O) and benign operator procs (ghostty, node) are not. This is the positive-control logic the real-machine test exercises."""

    def _rows(self, me):
        return [
            wd.ProcRow(100, 100, me, 50.0, "some_unanticipated_tool --x"),      # foreign user, spike
            wd.ProcRow(200, 200, 0,  90.0, "kernel_task"),                      # root uid -> excluded
            wd.ProcRow(300, 300, me, 0.5, "idle"),                             # below 3% floor -> excluded
            wd.ProcRow(400, 400, me, 40.0, "/Applications/Ghostty.app ghostty"),  # benign -> excluded
        ]

    def test_generic_catch_root_and_benign_excluded(self):
        me = os.getuid()
        benign = re.compile("ghostty|node", re.IGNORECASE)
        f = wd.foreign_user_processes(self._rows(me), frozenset({999}), 1, frozenset(), me, 3.0, benign)
        self.assertEqual({r.pid for r in f}, {100}, "only the unanticipated user tool survives")

    def test_own_pgid_exclude_pids_and_uid_filter(self):
        me = os.getuid()
        rows = [wd.ProcRow(10, 100, me, 50.0, "a"), wd.ProcRow(11, 200, me, 50.0, "b"),
                wd.ProcRow(12, 300, 999999, 50.0, "c")]  # uid != me -> excluded (system-owned)
        f = wd.foreign_user_processes(rows, frozenset({100}), 1, frozenset({11}), me, 3.0, None)
        self.assertEqual(f, [], "own_pgid, exclude_pids, and uid-mismatch each remove a row")

    def test_tracker_instant_spike_confirms_on_sample1(self):
        t = wd.ForeignProcTracker(required=2, instant_pcpu=25.0)
        self.assertEqual([r.pid for r in t.update([wd.ProcRow(1, 1, 501, 50.0, "hog")])], [1])

    def test_tracker_low_cpu_needs_persistence(self):
        t = wd.ForeignProcTracker(required=2, instant_pcpu=25.0)
        low = [wd.ProcRow(1, 1, 501, 5.0, "slow")]
        self.assertEqual(t.update(low), [])                    # sample 1: not yet
        self.assertEqual([r.pid for r in t.update(low)], [1])  # sample 2: confirmed (persisted)

    def test_tracker_oneshot_blip_ignored(self):
        t = wd.ForeignProcTracker(required=2, instant_pcpu=25.0)
        self.assertEqual(t.update([wd.ProcRow(9, 9, 501, 5.0, "ls")]), [])
        self.assertEqual(t.update([]), [])                     # gone -> streak reset, never confirmed


class PgidFile(unittest.TestCase):
    """bug4: allowlist file re-read each sample; missing / garbage tolerated as empty."""

    def test_reads_ints(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("111 222\n333\nnot_an_int\n")
            p = f.name
        try:
            self.assertEqual(wd.read_pgid_file(p), frozenset({111, 222, 333}))
        finally:
            os.unlink(p)

    def test_missing_is_empty(self):
        self.assertEqual(wd.read_pgid_file("/no/such/file"), frozenset())

    def test_none_is_empty(self):
        self.assertEqual(wd.read_pgid_file(None), frozenset())


class LockCli(unittest.TestCase):
    """bug3: reap-mutex prevents concurrent double-grant of a stale lock; acquire/busy/release."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lockdir = os.path.join(self.tmp, "lock")
        self.env = dict(os.environ, ILI_TIMING_LOCK_DIR=self.lockdir)

    def tearDown(self):
        subprocess.run(["rm", "-rf", self.tmp])

    def _acquire(self, attempt):
        return subprocess.run(
            ["bash", str(LOCK_SH), "acquire", "--attempt-id", attempt, "--pid", str(os.getpid())],
            env=self.env, capture_output=True, text=True).returncode

    def _seed_stale(self):
        subprocess.run(["rm", "-rf", self.lockdir, self.lockdir + ".reap"])
        os.makedirs(self.lockdir)
        with open(os.path.join(self.lockdir, "meta.json"), "w") as f:
            json.dump({"attempt_id": "stale", "driver_pid": 999999, "pgid": 999999,
                       "started_at": "x", "host": "h", "driver_start_time": "old"}, f)

    def test_concurrent_stale_reap_single_winner(self):
        for rnd in range(10):
            self._seed_stale()
            with cf.ThreadPoolExecutor(max_workers=4) as ex:
                rcs = list(ex.map(lambda n: self._acquire(f"r{rnd}-{n}"), range(4)))
            self.assertEqual(rcs.count(0), 1, f"round {rnd}: winners={rcs.count(0)} -- double-grant?")

    def test_busy_then_release(self):
        subprocess.run(["rm", "-rf", self.lockdir, self.lockdir + ".reap"])
        script = (
            f'source "{LOCK_SH}"; timing_lock_acquire A {os.getpid()} >/dev/null 2>&1; echo held=$?; '
            f'bash "{LOCK_SH}" acquire --attempt-id B --pid {os.getpid()} >/dev/null 2>&1; echo second=$?; '
            f'timing_lock_release >/dev/null 2>&1; echo rel=$?'
        )
        out = subprocess.run(["bash", "-c", script], env=self.env, capture_output=True, text=True).stdout
        self.assertIn("held=0", out)
        self.assertIn("second=1", out)   # BUSY while a live holder exists
        self.assertIn("rel=0", out)


@unittest.skipIf(
    os.environ.get("ILI_CI"),
    "real end-to-end watchdog runs (--duration-s 4 and 20, blocking subprocess.run) sampling "
    "live thermal/disk/AC-power/process telemetry: meant to validate the nice-19 long-encode "
    "watchdog on the actual dev machine, not a shared ephemeral CI runner where that telemetry "
    "is meaningless and the wall-clock cost (24s+ per matrix job) contributes to the runner "
    "dying (see ci.yml); stays authoritative in the local full suite.",
)
class WatchdogRun(unittest.TestCase):
    """bug1/2/5/6 end-to-end: a clean short run does NOT false-invalidate; --no-terminate marks
    INVALID + drops the sentinel WITHOUT killing the offending process."""

    def test_clean_run_valid(self):
        # Non-matching process patterns + huge cpu/disk/pageout thresholds isolate this to the
        # telemetry PARSES (bug1/bug2): the only way it can invalidate is a thermal/disk/ac gap.
        with tempfile.TemporaryDirectory() as d:
            outj = os.path.join(d, "wd.json")
            spgid = subprocess.run(["ps", "-o", "pgid=", "-p", str(os.getpid())],
                                   capture_output=True, text=True).stdout.strip()
            rc = subprocess.run(
                ["python3", str(WATCHDOG), "run", "--pgid", spgid, "--out-json", outj,
                 "--duration-s", "4", "--interval-s", "2",
                 "--glm-pattern", "^__nomatch__$", "--compiler-pattern", "^__nomatch__$",
                 "--python-analysis-pattern", "__nomatch__",
                 "--foreign-cpu-pct", "100000", "--foreign-disk-mbs", "100000",
                 "--pageout-rate", "100000", "--foreign-proc-min-cpu", "100000"],
                capture_output=True, text=True).returncode
            with open(outj) as fh:
                data = json.load(fh)
            self.assertEqual(rc, 0, data.get("reasons"))
            self.assertTrue(data["timing_environment_valid"], data.get("reasons"))
            self.assertGreaterEqual(data["sample_count"], 1)

    def test_no_terminate_marks_but_does_not_kill(self):
        with tempfile.TemporaryDirectory() as d:
            outj = os.path.join(d, "wd.json")
            sleeper = subprocess.Popen(["sleep", "30"])
            try:
                rc = subprocess.run(
                    ["python3", str(WATCHDOG), "run", "--pgid", "1", "--out-json", outj,
                     "--result-dir", d, "--duration-s", "20", "--interval-s", "2",
                     "--compiler-pattern", "^sleep$", "--no-terminate",
                     "--foreign-cpu-pct", "100000", "--foreign-disk-mbs", "100000",
                     "--pageout-rate", "100000", "--foreign-proc-min-cpu", "100000"],
                    capture_output=True, text=True).returncode
                self.assertEqual(rc, 1)
                self.assertIsNone(sleeper.poll(), "sleep was killed -- --no-terminate not honored")
                self.assertTrue(os.path.exists(os.path.join(d, "TIMING_INVALID.json")))
                with open(outj) as fh:
                    self.assertFalse(json.load(fh)["timing_environment_valid"])
            finally:
                sleeper.terminate()
                sleeper.wait()


if __name__ == "__main__":
    unittest.main()
