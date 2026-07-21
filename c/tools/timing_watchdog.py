#!/usr/bin/env python3
"""Lightweight timing-environment watchdog (Iliria timing-exclusivity system, alongside
scripts/timing_lock.sh and scripts/exp_launch.sh).

A pre-run quiesce_check.sh pass proves the machine WAS quiet at t=0; it says nothing about
minute 27. This tool samples ONLY process/system counters -- `ps`, `iostat`, `vm_stat`,
`pmset -g therm` -- every ``--interval-s`` seconds (5-10s) for the whole duration of a timing
experiment, and reacts the moment contamination appears instead of only gating the start.
Deliberately does NOT scan the repo or read any model/container file: it must stay cheap
enough to run continuously for a 40+ minute steady-state phase without itself perturbing the
measurement it is watching over.

Two severities (reviewer-requested distinction):
  HARD blockers (single sample, no debounce): an unexpected `glm`/compiler(cc/clang/make/gcc)
  /python-analysis process running OUTSIDE the watched experiment's own process group; loss of
  AC power (on battery); thermal throttling; swap/pageout activity; or a telemetry read that
  fails outright (fail-closed: an unmeasured interval is not evidence of cleanliness -- mirrors
  scripts/quiesce_check.sh's own stated philosophy for the same class of gate). Each is a
  deliberate, meaningful STATE event, not noise -- it invalidates on the first sample it sees.
  SUSTAINED thresholds (require --consecutive-required samples in a row): foreign aggregate
  CPU% and total disk MB/s. These are continuous metrics with real transient noise -- a
  1-sample mds/Spotlight indexing blip must NOT invalidate a 40-minute run -- so a metric must
  breach its threshold on `--consecutive-required` CONSECUTIVE samples before it counts; a
  single clean sample resets that metric's streak to zero.

On invalidation this tool does not merely flag and keep sampling -- continuing a run that is
already known-contaminated wastes the rest of its wall-clock budget for nothing. It, in
order: (1) atomically writes timing_environment_valid=false plus the EXACT offending sample
and reason(s) to --out-json; (2) drops a TIMING_INVALID.json sentinel into --result-dir if
given, so partial outputs already on disk are unambiguously labeled invalid without being
deleted; (3) terminates the watched experiment's whole process group (SIGTERM, wait, SIGKILL
if needed, verify no descendants survive -- the same protocol as scripts/exp_launch.sh's
exp_launch_cancel, reimplemented natively here via os.killpg rather than shelled out to that
script -- see "why two implementations" below); (4) ONLY once that verification confirms no
descendants remain does it release the driver's timing_lock.sh lease on the driver's behalf
(scripts/timing_lock.sh release-attempt, authorized by a matching --attempt-id -- never
merely "the current lock"), since the driver may not get a chance to run its own exit trap if
its child is what just got force-killed out from under it.

Why two implementations of the same kill-group protocol (this file's terminate_process_group
and exp_launch.sh's exp_launch_cancel): the protocol itself (SIGTERM group, wait, SIGKILL
group, verify, mark) is small, stable, and fully specified -- reimplementing it natively in
each language (os.killpg here, `kill -TERM -PGID` there) is simpler and more robust than a
Python-to-bash subprocess hop for something this size. The LOCK, by contrast, has real
persistent shared state (the lock directory's on-disk metadata schema) that every caller must
manipulate identically -- that one is always invoked through the single canonical
scripts/timing_lock.sh, in every language, rather than re-implemented.

Calibration mode (``calibrate`` subcommand): thresholds should come from a machine's measured
noise floor, not a guess. It runs three short phases and reports observed min/mean/p95/max
per counter: (a) idle -- nothing but this tool running; (b) watchdog-only -- identical to (a),
reported separately so the watchdog's OWN overhead is visible on its own rather than
conflated with ambient OS noise; (c) with-benchmark -- a synthetic CPU-burning child in its
own process group, to confirm a legitimately busy (but valid) experiment's OWN load is
correctly excluded from "foreign" and does not itself trip the thresholds. Defaults shipped
in this file are principled starting points, not a validated result of running this against
any specific machine's real overnight idle window -- run `calibrate` for real on the actual
timing machine before trusting the defaults for an unattended multi-hour campaign.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
C_DIR = TOOLS_DIR.parent
DEFAULT_TIMING_LOCK_SH = C_DIR / "scripts" / "timing_lock.sh"

# Mirrors scripts/quiesce_check.sh gate 1's own analytics-process pattern verbatim -- an
# "unexpected python-analysis process" means the same thing here as it does in the pre-flight
# gate, not a second, silently-diverging definition of the same idea.
PY_ANALYSIS_PATTERN_DEFAULT = "measure_expert|entropy|build_mixed|simulate_bytes|quant_error|long_ctx_profile"
COMPILER_PATTERN_DEFAULT = r"^(cc|clang|make|gcc|g\+\+|c\+\+)$"
GLM_PATTERN_DEFAULT = r"^glm$"          # exact-name match, mirrors `pgrep -x glm`
SAMPLES_RING_LIMIT = 720                 # ~2h of history at 10s cadence; bounds file growth


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def write_json_atomic(path: str | Path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# ---- process/system counter collection (impure: shells out to ps/iostat/vm_stat/pmset) -----

@dataclass
class ProcRow:
    pid: int
    pgid: int
    uid: int
    pcpu: float
    command: str


def list_processes() -> list[ProcRow] | None:
    out = _run(["ps", "-Ao", "pid,pgid,uid,pcpu,command"])
    if not out.strip():
        return None
    rows = []
    for line in out.strip().splitlines()[1:]:
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            rows.append(ProcRow(int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), parts[4]))
        except ValueError:
            continue
    return rows


def foreign_cpu_pct(rows: list[ProcRow], own_pgids: frozenset[int], own_pid: int,
                    exclude_pids: frozenset[int] = frozenset()) -> float:
    """Sum of %cpu (ps's own per-core-normalized figure) over every process NOT in an allowed
    process group -- can exceed 100% on a multi-core machine when several foreign processes
    are each busy; that is expected and more informative than a single normalized idle%, which
    cannot be used here at all (see module docstring: the experiment's OWN cpu usage is
    supposed to be high and must not count as "foreign"). own_pgids is the allowed set (the
    driver's group plus any per-trial experiment groups currently named in the allowlist);
    own_pid (the watchdog itself) and exclude_pids (driver, telemetry loop) are also
    legitimate and never counted as foreign."""
    return sum(r.pcpu for r in rows
               if r.pgid not in own_pgids and r.pid != own_pid and r.pid not in exclude_pids)


def unexpected_processes(rows: list[ProcRow], own_pgids: frozenset[int], own_pid: int,
                          exclude_pids: frozenset[int],
                          glm_re: re.Pattern, compiler_re: re.Pattern,
                          py_analysis_re: re.Pattern) -> list[str]:
    out = []
    for r in rows:
        if r.pgid in own_pgids:
            continue
        # Skip the watchdog itself (own_pid) and declared-legit pids (driver, telemetry
        # loop). Skipping own_pid is ALSO the self-kill guard: the watchdog's own argv
        # carries the analysis pattern strings (e.g. via --python-analysis-pattern), so
        # without this it matches itself and force-kills its own group on sample 1.
        if r.pid == own_pid or r.pid in exclude_pids:
            continue
        argv0 = r.command.split(None, 1)[0] if r.command.split() else ""
        exe = argv0.rsplit("/", 1)[-1]
        if glm_re.search(exe):
            out.append(f"glm pid={r.pid} pgid={r.pgid} cmd={r.command!r}")
        elif compiler_re.search(exe):
            out.append(f"compiler pid={r.pid} pgid={r.pgid} cmd={r.command!r}")
        elif py_analysis_re.search(r.command):
            out.append(f"python-analysis pid={r.pid} pgid={r.pgid} cmd={r.command!r}")
    return out


def foreign_user_processes(rows: list[ProcRow], own_pgids: frozenset[int], own_pid: int,
                           exclude_pids: frozenset[int], my_uid: int, min_pcpu: float,
                           benign_re: "re.Pattern | None") -> list[ProcRow]:
    """User-owned processes outside the allowed groups, above a small per-process CPU floor --
    the GENERIC contamination net that unexpected_processes' name patterns cannot provide: an
    agent, git, a browser worker, a compressor, a shell command, a tool whose name was never
    anticipated -- caught by BEHAVIOR (user-owned + foreign + consuming CPU), not by a hard-coded
    name. Root/system processes (kernel_task doing the engine's OWN streaming I/O, WindowServer,
    launchd daemons) have a different uid and are excluded -- which is exactly why the aggregate
    foreign_cpu metric is unenforceable for a streaming workload (kernel_task dominates it) while
    THIS per-user-process net is not. benign_re (e.g. your terminal/editor processes) is skipped so the
    operator's own idle shell does not itself count as contamination."""
    out = []
    for r in rows:
        if r.pgid in own_pgids:
            continue
        if r.pid == own_pid or r.pid in exclude_pids:
            continue
        if r.uid != my_uid:
            continue
        if r.pcpu < min_pcpu:
            continue
        if benign_re is not None and benign_re.search(r.command):
            continue
        out.append(r)
    return out


class ForeignProcTracker:
    """Confirms a foreign user process as contamination when it either exceeds an INSTANT
    per-process CPU threshold on a single sample, or PERSISTS above the small floor for
    `required` consecutive samples. Per-pid streaks mean a one-sample spawn (a brief `ls` or
    `pgrep`) is ignored, but a sustained agent/browser/build is caught even at modest CPU.
    Implements the durable rule: a new user-owned process outside the experiment's allowed
    groups invalidates timing when it PERSISTS or EXCEEDS a small resource threshold."""

    def __init__(self, required: int = 2, instant_pcpu: float = 25.0):
        self.required = max(1, required)
        self.instant = instant_pcpu
        self.streaks: dict[int, int] = {}

    def update(self, foreign: list[ProcRow]) -> list[ProcRow]:
        present = {r.pid for r in foreign}
        for pid in list(self.streaks):
            if pid not in present:
                del self.streaks[pid]
        confirmed = []
        for r in foreign:
            self.streaks[r.pid] = self.streaks.get(r.pid, 0) + 1
            if r.pcpu >= self.instant or self.streaks[r.pid] >= self.required:
                confirmed.append(r)
        return confirmed


def read_disk_mbs() -> float | None:
    """Instantaneous disk0 throughput (MB/s), NOT the since-boot average.

    `iostat ... -c 1` returns a SINGLE report, which on macOS is the since-boot average
    (a near-constant multi-GB/s figure) -- useless as a live-contamination signal and a
    guaranteed false breach. `-c 2` emits the since-boot line THEN a true 1-second delta;
    we parse the LAST data line (the delta), matching scripts/quiesce_check.sh's own
    `-c 2 | tail -1` convention. NOTE: this is TOTAL disk, not foreign-only, so it is a
    SUSTAINED (debounced) metric, never a hard blocker -- a legitimate cold-phase model
    load would breach any sane threshold on a single sample. Returns None only when iostat
    produced no parseable data at all (a telemetry gap the caller fails closed on)."""
    out = _run(["iostat", "-d", "-w", "1", "-c", "2", "disk0"])
    lines = out.strip().splitlines()
    if len(lines) < 2:
        return None
    last = lines[-1].split()
    if len(last) < 3:
        return None
    try:
        return float(last[2])
    except ValueError:
        return None


def read_thermal_speed_limit_pct() -> int | None:
    """CPU speed-limit percentage (100 = not throttled).

    macOS `pmset -g therm` on an UNTHROTTLED Apple Silicon machine prints only a
    "No thermal warning level has been recorded" note, with NO CPU_Speed_Limit line --
    that is the NOMINAL case and MUST read as 100 (not throttled), not as a telemetry gap.
    The earlier version returned None here, which the caller treated as "thermal telemetry
    unavailable" -> a fail-closed hard kill on sample 1 of every run on this machine.
    quiesce_check.sh gate 5 has the correct posture: fail only on EMPTY output. Returns
    None ONLY when pmset produced no output at all (a genuine gap the caller fails closed on)."""
    out = _run(["pmset", "-g", "therm"])
    if not out.strip():
        return None
    m = re.search(r"CPU_Speed_Limit\D*(\d+)", out, re.IGNORECASE)
    return int(m.group(1)) if m else 100


def read_pageouts_cum() -> int | None:
    """Cumulative pageout counter, same field/format-quirk scripts/quiesce_check.sh gate 6
    already parses (vm_stat prints it with a trailing '.')."""
    out = _run(["vm_stat"])
    m = re.search(r"Pageouts:\s*([\d.]+)", out)
    if not m:
        return None
    return int(m.group(1).replace(".", ""))


def read_pgid_file(path: str | None) -> frozenset[int]:
    """Set of pgids from a whitespace/newline-separated file, RE-READ every sample so a driver
    can add each per-trial experiment's pgid while it runs and drop it afterwards. A missing or
    unreadable file, or a non-integer token, yields the empty set (NOT a telemetry error -- the
    file legitimately may not exist yet, or be empty between trials); the --pgid base still
    applies via the union in run_watchdog."""
    if not path:
        return frozenset()
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return frozenset()
    out: set[int] = set()
    for tok in text.split():
        try:
            out.add(int(tok))
        except ValueError:
            continue
    return frozenset(out)


def read_on_ac_power() -> bool | None:
    """True on AC, False on battery, None if `pmset -g batt` produced no output at all (a
    genuine telemetry gap the caller fails closed on). Mirrors quiesce_check.sh's AC gate
    (`pmset -g batt` then look for 'AC Power'). Losing AC mid-run silently changes the
    power/thermal envelope, so on-battery is a HARD blocker in classify_sample."""
    out = _run(["pmset", "-g", "batt"])
    if not out.strip():
        return None
    return "AC Power" in out


# ---- pure data model + classification -------------------------------------------------

@dataclass
class Thresholds:
    # foreign_cpu_pct and foreign_disk_mbs are recorded as INFORMATIONAL telemetry only and are
    # NOT enforced: on a streaming workload both are dominated by the engine's own kernel_task
    # I/O (see foreign_user_processes). Generic contamination is enforced by the per-user-process
    # net instead. A calibrated total-load envelope (observed > expected_phase_load + tolerance)
    # is the intended future upgrade to also catch root/system load (backups, indexing).
    foreign_cpu_pct: float = 20.0            # informational only (aggregate, kernel_task-coupled)
    foreign_disk_mbs: float = 50.0           # informational only (total disk, streaming-coupled)
    pageout_rate: float = 20.0               # pageouts/sec -- HARD (swap during timing = invalid)
    consecutive_required: int = 2            # persistence samples for the foreign-user-process net
    foreign_proc_min_cpu: float = 3.0        # a foreign user proc below this %cpu is ignored (idle noise)
    foreign_proc_instant_cpu: float = 25.0   # a foreign user proc at/above this %cpu invalidates on ONE sample


@dataclass
class Sample:
    timestamp: str
    foreign_cpu_pct: float | None
    disk_mbs: float | None
    pageout_rate: float | None
    thermal_speed_limit_pct: int | None
    on_ac_power: bool | None = None
    unexpected_processes: list[str] = field(default_factory=list)
    foreign_user_processes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)   # telemetry that could not be read at all


def classify_sample(sample: Sample, thresholds: Thresholds) -> list[str]:
    """Pure, single-sample HARD reasons (no debounce): named unexpected processes, telemetry
    gaps, AC loss, thermal throttle, swap/pageout -- each a deliberate state event that
    invalidates the instant it is seen. foreign_cpu_pct and disk_mbs are recorded on the Sample
    as INFORMATIONAL telemetry only and are NOT enforced here: on a streaming workload both are
    dominated by the engine's own kernel_task I/O. Generic contamination is enforced separately
    by the foreign-user-process net (foreign_user_processes + ForeignProcTracker) in the run
    loop, which is threshold-robust because it is per-user-process rather than an aggregate."""
    hard: list[str] = []
    for proc in sample.unexpected_processes:
        hard.append(f"unexpected process: {proc}")
    for err in sample.errors:
        hard.append(f"telemetry unavailable: {err}")
    if sample.on_ac_power is False:
        hard.append("power: on BATTERY (AC lost) -- timing/thermal envelope changed")
    if sample.thermal_speed_limit_pct is not None and sample.thermal_speed_limit_pct < 100:
        hard.append(f"thermal throttling active (CPU_Speed_Limit={sample.thermal_speed_limit_pct})")
    if sample.pageout_rate is not None and sample.pageout_rate > thresholds.pageout_rate:
        hard.append(f"swap/pageout activity {sample.pageout_rate:.1f}/s > {thresholds.pageout_rate:.1f}/s")
    return hard


# ---- process-group termination (native reimplementation of exp_launch.sh's protocol --
#      see module docstring for why) -----------------------------------------------------

def _group_alive(pgid: int) -> bool:
    return bool(_run(["pgrep", "-g", str(pgid)]).strip())


def terminate_process_group(pgid: int, timeout_s: float = 10.0, log=lambda *a: None) -> dict:
    """mark-cancelling -> SIGTERM group -> wait -> SIGKILL group -> verify -> mark-cancelled.
    Returns {"outcome": "cancelled"|"already-gone"|"cancel-failed", "waited_s": float}."""
    start = time.monotonic()
    if not _group_alive(pgid):
        log(f"terminate_process_group({pgid}): already gone")
        return {"outcome": "already-gone", "waited_s": 0.0}

    log(f"terminate_process_group({pgid}): SIGTERM to process group")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as e:
        log(f"terminate_process_group({pgid}): SIGTERM raised {e!r} (continuing to verify)")

    while time.monotonic() - start < timeout_s:
        if not _group_alive(pgid):
            break
        time.sleep(0.5)

    if _group_alive(pgid):
        log(f"terminate_process_group({pgid}): survived SIGTERM, sending SIGKILL")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError) as e:
            log(f"terminate_process_group({pgid}): SIGKILL raised {e!r} (continuing to verify)")
        time.sleep(1.0)

    waited = time.monotonic() - start
    if _group_alive(pgid):
        log(f"terminate_process_group({pgid}): FAILED -- descendants survived SIGKILL")
        return {"outcome": "cancel-failed", "waited_s": waited}
    log(f"terminate_process_group({pgid}): verified clean, no descendants remain")
    return {"outcome": "cancelled", "waited_s": waited}


def release_lock_for_attempt(attempt_id: str, timing_lock_sh: Path, log=lambda *a: None) -> bool:
    if not timing_lock_sh.is_file():
        log(f"release_lock_for_attempt: {timing_lock_sh} not found -- cannot release, driver's own trap must do it")
        return False
    proc = subprocess.run(["bash", str(timing_lock_sh), "release-attempt", "--attempt-id", attempt_id],
                          capture_output=True, text=True, timeout=15)
    log(f"release_lock_for_attempt({attempt_id}): rc={proc.returncode} {proc.stderr.strip()}")
    return proc.returncode == 0


# ---- main sampling loop -----------------------------------------------------------------

def collect_sample(own_pgids: frozenset[int], own_pid: int, prev_pageouts: int | None,
                   prev_tick: float, glm_re: re.Pattern, compiler_re: re.Pattern,
                   py_analysis_re: re.Pattern, my_uid: int = -1,
                   foreign_proc_min_cpu: float = 1e9, benign_re: "re.Pattern | None" = None,
                   exclude_pids: frozenset[int] = frozenset()
                   ) -> tuple[Sample, list[ProcRow], int | None, float]:
    errors: list[str] = []
    rows = list_processes()
    if rows is None:
        cpu, unexpected, foreign_procs = None, [], []
        errors.append("process list unavailable (ps failed)")
    else:
        cpu = foreign_cpu_pct(rows, own_pgids, own_pid, exclude_pids)      # informational aggregate
        unexpected = unexpected_processes(rows, own_pgids, own_pid, exclude_pids,
                                          glm_re, compiler_re, py_analysis_re)
        foreign_procs = foreign_user_processes(rows, own_pgids, own_pid, exclude_pids,
                                               my_uid, foreign_proc_min_cpu, benign_re)

    disk_mbs = read_disk_mbs()
    if disk_mbs is None:
        errors.append("disk telemetry unavailable (iostat failed)")

    therm = read_thermal_speed_limit_pct()
    if therm is None:
        errors.append("thermal telemetry unavailable (pmset -g therm produced no output)")

    ac = read_on_ac_power()
    if ac is None:
        errors.append("power telemetry unavailable (pmset -g batt produced no output)")

    pageouts_cum = read_pageouts_cum()
    pageout_rate = None
    now_mono = time.monotonic()
    if pageouts_cum is None:
        errors.append("pageout telemetry unavailable (vm_stat failed)")
        new_prev, new_tick = prev_pageouts, prev_tick
    else:
        if prev_pageouts is not None:
            elapsed = max(now_mono - prev_tick, 0.001)
            pageout_rate = max(0.0, (pageouts_cum - prev_pageouts) / elapsed)
        new_prev, new_tick = pageouts_cum, now_mono

    fp_strs = [f"pid={r.pid} pgid={r.pgid} uid={r.uid} cpu={r.pcpu:.1f}% cmd={r.command!r}"
               for r in foreign_procs]
    sample = Sample(now_iso(), cpu, disk_mbs, pageout_rate, therm, ac, unexpected, fp_strs, errors)
    return sample, foreign_procs, new_prev, new_tick


def run_watchdog(args) -> int:
    own_pid = os.getpid()
    own_pgid = args.pgid
    my_uid = os.getuid()
    thresholds = Thresholds(args.foreign_cpu_pct, args.foreign_disk_mbs, args.pageout_rate,
                            args.consecutive_required, args.foreign_proc_min_cpu,
                            args.foreign_proc_instant_cpu)
    glm_re = re.compile(args.glm_pattern, re.IGNORECASE)
    compiler_re = re.compile(args.compiler_pattern, re.IGNORECASE)
    py_analysis_re = re.compile(args.python_analysis_pattern)
    benign_re = re.compile(args.benign_pattern, re.IGNORECASE) if args.benign_pattern else None
    foreign_tracker = ForeignProcTracker(required=thresholds.consecutive_required,
                                         instant_pcpu=thresholds.foreign_proc_instant_cpu)
    exclude_pids = frozenset(
        int(x) for x in re.split(r"[,\s]+", (args.exclude_pids or "").strip()) if x)

    def log(msg: str) -> None:
        print(f"[timing-watchdog] {msg}", file=sys.stderr, flush=True)

    stop = {"flag": False}

    def handle_signal(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    state = {
        "pgid": own_pgid,
        "attempt_id": args.attempt_id,
        "interval_s": args.interval_s,
        "thresholds": asdict(thresholds),
        "started_at": now_iso(),
        "updated_at": None,
        "timing_environment_valid": True,
        "reasons": [],
        "sample_count": 0,
        "invalidated_at": None,
        "termination": None,
        "enforcement": {
            "unexpected_process": "enforced (named glm/compiler/python-analysis -> HARD)",
            "foreign_user_process": "enforced (any user-owned foreign proc that persists or spikes)",
            "thermal": "enforced HARD", "ac_power": "enforced HARD", "pageout": "enforced HARD",
            "telemetry_gap": "enforced (fail-closed HARD)",
            "foreign_cpu_pct": "informational_only (streaming-coupled via kernel_task)",
            "foreign_disk_mbs": "informational_only (total disk, streaming-coupled)",
        },
        "foreign_proc_streaks": {},
        "samples": [],
    }
    write_json_atomic(args.out_json, state)

    prev_pageouts: int | None = None
    prev_tick = time.monotonic()
    start = time.monotonic()
    exit_code = 0
    try:
        while not stop["flag"]:
            tick_start = time.monotonic()
            # If the driver that launched us is gone (a SIGKILL leaves us reparented to init/
            # launchd), stop -- otherwise we sample forever and could drop a spurious
            # TIMING_INVALID.json into this (old) result dir when a LATER run's glm appears under a
            # pgid not in our allowlist.
            if os.getppid() == 1:
                log("parent driver gone (reparented to init) -- exiting to avoid orphaned sampling")
                break
            own_pgids = frozenset({own_pgid}) | read_pgid_file(args.allowlist_file)
            sample, foreign_procs, prev_pageouts, prev_tick = collect_sample(
                own_pgids, own_pid, prev_pageouts, prev_tick, glm_re, compiler_re, py_analysis_re,
                my_uid, thresholds.foreign_proc_min_cpu, benign_re, exclude_pids)

            hard = classify_sample(sample, thresholds)
            confirmed_procs = foreign_tracker.update(foreign_procs)
            proc_reasons = [
                f"foreign user process (persisted>={thresholds.consecutive_required} samples or "
                f">={thresholds.foreign_proc_instant_cpu:.0f}% cpu) pid={r.pid} pgid={r.pgid} "
                f"cpu={r.pcpu:.1f}% cmd={r.command!r}" for r in confirmed_procs]
            state["foreign_proc_streaks"] = dict(foreign_tracker.streaks)
            reasons_this_sample = hard + proc_reasons

            state["sample_count"] += 1
            state["updated_at"] = sample.timestamp
            sample_dict = asdict(sample)
            sample_dict["reasons"] = reasons_this_sample
            state["samples"].append(sample_dict)
            if len(state["samples"]) > SAMPLES_RING_LIMIT:
                state["samples"] = state["samples"][-SAMPLES_RING_LIMIT:]

            if reasons_this_sample and state["timing_environment_valid"]:
                state["timing_environment_valid"] = False
                state["invalidated_at"] = sample.timestamp
                state["reasons"] = reasons_this_sample
                log(f"CONTAMINATION at {sample.timestamp}: {'; '.join(reasons_this_sample)}")
                write_json_atomic(args.out_json, state)   # exact offending sample, before acting

                if args.result_dir:
                    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
                    write_json_atomic(Path(args.result_dir) / "TIMING_INVALID.json", {
                        "timing_environment_valid": False,
                        "invalidated_at": sample.timestamp,
                        "reasons": reasons_this_sample,
                        "offending_sample": sample_dict,
                        "note": "partial outputs in this directory are INVALID; not deleted, only labeled",
                    })

                if args.no_terminate:
                    log("--no-terminate: INVALID recorded; leaving process teardown and lock "
                        "release to the driver (avoids killing a group the watchdog belongs to)")
                    state["termination"] = {"outcome": "not-terminated (driver owns teardown)"}
                    write_json_atomic(args.out_json, state)
                else:
                    term = terminate_process_group(own_pgid, args.terminate_timeout_s, log=log)
                    state["termination"] = term
                    write_json_atomic(args.out_json, state)
                    if term["outcome"] in ("cancelled", "already-gone") and args.attempt_id:
                        release_lock_for_attempt(args.attempt_id, Path(args.timing_lock_sh), log=log)

                exit_code = 1
                break

            write_json_atomic(args.out_json, state)

            if args.duration_s is not None and (time.monotonic() - start) >= args.duration_s:
                break
            remaining = args.interval_s - (time.monotonic() - tick_start)
            slept = 0.0
            while remaining - slept > 0 and not stop["flag"]:
                chunk = min(0.5, remaining - slept)
                time.sleep(chunk)
                slept += chunk
    finally:
        state["stopped_at"] = now_iso()
        write_json_atomic(args.out_json, state)
    return exit_code


# ---- calibration mode -------------------------------------------------------------------

def _phase_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    sorted_v = sorted(values)
    p95_idx = min(len(sorted_v) - 1, int(round(0.95 * (len(sorted_v) - 1))))
    return {
        "n": len(values), "min": min(values), "max": max(values),
        "mean": statistics.fmean(values), "p95": sorted_v[p95_idx],
    }


def _calibrate_phase(label: str, seconds: float, interval_s: float, pgid: int,
                     glm_re, compiler_re, py_analysis_re, log) -> dict:
    own_pid = os.getpid()
    cpu_vals, disk_vals, pageout_vals, unexpected_hits = [], [], [], 0
    prev_pageouts, prev_tick = None, time.monotonic()
    start = time.monotonic()
    n = 0
    while time.monotonic() - start < seconds:
        sample, _foreign, prev_pageouts, prev_tick = collect_sample(
            frozenset({pgid}), own_pid, prev_pageouts, prev_tick, glm_re, compiler_re, py_analysis_re)
        n += 1
        if sample.foreign_cpu_pct is not None:
            cpu_vals.append(sample.foreign_cpu_pct)
        if sample.disk_mbs is not None:
            disk_vals.append(sample.disk_mbs)
        if sample.pageout_rate is not None:
            pageout_vals.append(sample.pageout_rate)
        unexpected_hits += len(sample.unexpected_processes)
        time.sleep(max(0.0, interval_s - (time.monotonic() - start - (n - 1) * interval_s)))
    log(f"calibration phase '{label}': {n} samples over {seconds}s")
    return {
        "label": label, "samples": n,
        "foreign_cpu_pct": _phase_stats(cpu_vals),
        "disk_mbs": _phase_stats(disk_vals),
        "pageout_rate": _phase_stats(pageout_vals),
        "unexpected_process_hits": unexpected_hits,
    }


def _spawn_calibration_benchmark(seconds: float) -> subprocess.Popen:
    """A synthetic, self-contained CPU-burning child in its OWN process group -- stands in
    for "a valid, legitimately busy experiment" (never a real engine) so phase (c) can prove
    the experiment's own load is excluded from "foreign" rather than tripping the thresholds
    it is itself supposed to be exempt from."""
    code = (
        "import time\n"
        f"end = time.monotonic() + {seconds}\n"
        "x = 0\n"
        "while time.monotonic() < end:\n"
        "    x = (x * 1103515245 + 12345) % (2**31)\n"
    )
    return subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_calibration(args) -> int:
    def log(msg: str) -> None:
        print(f"[timing-watchdog:calibrate] {msg}", file=sys.stderr, flush=True)

    glm_re = re.compile(args.glm_pattern, re.IGNORECASE)
    compiler_re = re.compile(args.compiler_pattern, re.IGNORECASE)
    py_analysis_re = re.compile(args.python_analysis_pattern)
    seconds = args.calibrate_seconds

    # Phase (a) idle: an arbitrary, essentially-unusable pgid so nothing on the machine is
    # excluded as "own" -- this is the true ambient OS noise floor.
    idle = _calibrate_phase("idle", seconds, args.interval_s, pgid=1,
                           glm_re=glm_re, compiler_re=compiler_re, py_analysis_re=py_analysis_re, log=log)

    # Phase (b) watchdog-only: excludes THIS process's own pgid, isolating whether the
    # watchdog's own presence measurably adds noise versus phase (a).
    watchdog_only = _calibrate_phase("watchdog_only", seconds, args.interval_s, pgid=os.getpgrp(),
                                    glm_re=glm_re, compiler_re=compiler_re, py_analysis_re=py_analysis_re, log=log)

    # Phase (c) with a synthetic valid benchmark in its own group: proves a legitimately
    # busy experiment's OWN cpu load is excluded from "foreign" and does not itself trip
    # the foreign-cpu threshold.
    bench = _spawn_calibration_benchmark(seconds + 5)
    try:
        bench_pgid = int(_run(["ps", "-o", "pgid=", "-p", str(bench.pid)]).strip() or bench.pid)
        with_benchmark = _calibrate_phase("with_benchmark", seconds, args.interval_s, pgid=bench_pgid,
                                          glm_re=glm_re, compiler_re=compiler_re,
                                          py_analysis_re=py_analysis_re, log=log)
    finally:
        if bench.poll() is None:
            bench.terminate()
            try:
                bench.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bench.kill()

    report = {
        "generated_at": now_iso(),
        "calibrate_seconds_per_phase": seconds,
        "interval_s": args.interval_s,
        "phases": {"idle": idle, "watchdog_only": watchdog_only, "with_benchmark": with_benchmark},
        "note": "principled starting defaults only become validated thresholds once this has "
                "been run for real, on the actual timing machine, during a genuinely idle window.",
    }
    if args.out_json:
        write_json_atomic(args.out_json, report)
    print(json.dumps(report, indent=2))
    return 0


# ---- CLI ----------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--interval-s", type=float, default=7.0,
                        help="sample cadence in seconds (documented range: 5-10s; default 7.0)")
        sp.add_argument("--foreign-cpu-pct", type=float, default=20.0,
                        help="INFORMATIONAL only (recorded, not enforced): aggregate foreign CPU%% "
                             "is kernel_task-coupled on a streaming workload")
        sp.add_argument("--foreign-disk-mbs", type=float, default=50.0,
                        help="INFORMATIONAL only (recorded, not enforced): total disk is streaming-coupled")
        sp.add_argument("--pageout-rate", type=float, default=20.0, help="pageouts/sec (HARD)")
        sp.add_argument("--consecutive-required", type=int, default=2,
                        help="persistence samples for the foreign-user-process net")
        sp.add_argument("--foreign-proc-min-cpu", type=float, default=3.0,
                        help="a foreign USER process below this %%cpu is ignored (idle noise)")
        sp.add_argument("--foreign-proc-instant-cpu", type=float, default=25.0,
                        help="a foreign USER process at/above this %%cpu invalidates on ONE sample")
        sp.add_argument("--benign-pattern", default="",
                        help="regex of commands to treat as legitimate operator processes and skip "
                             "(e.g. your terminal and editor: 'ghostty|node')")
        sp.add_argument("--glm-pattern", default=GLM_PATTERN_DEFAULT)
        sp.add_argument("--compiler-pattern", default=COMPILER_PATTERN_DEFAULT)
        sp.add_argument("--python-analysis-pattern", default=PY_ANALYSIS_PATTERN_DEFAULT)
        sp.add_argument("--exclude-pids", default="",
                        help="comma/space-separated pids that are legitimate (driver, telemetry "
                             "loop) and must never count as foreign or trip the process check")

    run_p = sub.add_parser("run", help="watch a running experiment's process group")
    run_p.add_argument("--pgid", type=int, required=True,
                       help="the experiment's own process group id -- excluded from 'foreign'")
    run_p.add_argument("--out-json", required=True)
    run_p.add_argument("--duration-s", type=float, default=None,
                       help="stop automatically after this long (mainly for tests); default: run until SIGTERM/SIGINT")
    run_p.add_argument("--terminate-timeout-s", type=float, default=10.0)
    run_p.add_argument("--attempt-id", default=None,
                       help="if given, release scripts/timing_lock.sh's lease for this attempt_id once a "
                            "contamination response verifies the watched group has no descendants left")
    run_p.add_argument("--timing-lock-sh", default=str(DEFAULT_TIMING_LOCK_SH))
    run_p.add_argument("--result-dir", default=None,
                       help="if given, drop TIMING_INVALID.json here on invalidation (partial outputs "
                            "are labeled, never deleted)")
    run_p.add_argument("--allowlist-file", default=None,
                       help="path to a file of pgids (whitespace/newline separated), RE-READ every "
                            "sample and unioned with --pgid -- lets a driver add each per-trial "
                            "experiment's pgid while it runs and drop it afterwards")
    run_p.add_argument("--no-terminate", action="store_true",
                       help="on contamination, write the INVALID verdict + TIMING_INVALID.json and "
                            "exit 1, but do NOT kill any process group or release the lock -- for a "
                            "campaign driver that owns orderly teardown itself (and avoids the "
                            "watchdog killing a group it may itself belong to)")
    add_common(run_p)

    cal_p = sub.add_parser("calibrate", help="measure this machine's noise floor per counter")
    cal_p.add_argument("--calibrate-seconds", type=float, default=20.0,
                       help="duration of EACH of the 3 phases (default 20s; total runtime ~= 3x this)")
    cal_p.add_argument("--out-json", default=None)
    add_common(cal_p)

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        return run_watchdog(args)
    if args.command == "calibrate":
        return run_calibration(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
