#!/usr/bin/env python3
"""Same-prompt cold-vs-resume ABBA certification for ILIRIA'S OWN cross-restart
KV-resume (`.ili_kv`, c/glm.c kv_disk_load/kv_disk_append) -- the mechanism, not
the sibling engine's same-process prompt cache.

WHY THIS EXISTS (retirement context, do not re-litigate without reading this
first): the certified 147.8x [137.1, 157.8] TTFT number in
the sibling engine's kv-resume results is real
and rigorous, but it measures the sibling engine (Qwen3-32B-4bit, one MLX server process held
constant across all trials, "cold" = "not in this process's in-RAM prompt-cache
yet"). It is NOT a measurement of iliria's own claim, which is specifically
CROSS-RESTART: the engine process exits entirely between a save and a resume,
and `.ili_kv` on disk is what survives that exit.
iliria's own prior number (3.07s vs 436.65s, "142x", docs/PERFORMANCE_THEORY.md
serve_kv_persistence_ttft_speedup_x) was N=1, used mismatched prompts between
the cold and resume observations, and never stated whether TTFT included
process spawn or model load -- retired as uncitable for exactly those reasons.

This harness ports the sibling engine's kv-resume harness DESIGN
to iliria's actual mechanism, fixing the same three flaws:
  1. SAME prompt, byte-identical, both arms, every trial (one pinned sha256).
  2. TTFT defined ONE way, stated explicitly (see TTFT_DEFINITION below), used
     identically for both arms -- and unlike the sibling engine's definition, this one
     INCLUDES process spawn and model load, because for a cross-restart claim
     those costs are symmetric and paid-in-both-arms, not a confound to
     exclude (the sibling engine excludes them because the sibling engine never restarts its process).
  3. N>=1 per arm with a real bootstrap CI (build_abba_schedule / stats.py
     below are near-verbatim ports of the sibling engine's own, already-reviewed
     implementations -- same algorithm, not a new one invented for this repo).

MECHANISM THIS HARNESS ACTUALLY EXERCISES (read before changing the prompt
protocol): iliria's raw `\x02PROMPT` API frame (glm.c run_serve's raw_mode
branch, reached via openai_server.py's `Engine.generate`) tokenizes the
incoming prompt and walks it against the resident KV history array `hist[]`
token-for-token from position 0 (glm.c: `while(prefix<old_len &&
prefix<prompt_tokens && hist[prefix]==tmp[prefix]) prefix++`). At a fresh
process with `.ili_kv` ABSENT, `hist[]` starts empty, so this walk stops
immediately at prefix=0 -- a full prefill of every prompt token (COLD). At a
fresh process with `.ili_kv` PRESENT from an earlier identical-prompt run,
`kv_disk_load` (called before the engine ever prints READY, so this cost is
inside every trial's timed window, not excluded from it) repopulates `hist[]`
from disk with NO recompute ("no re-prefill", read directly off the engine's
own stderr); the SAME target prompt sent again then matches that resumed
prefix exactly, so the walk reaches prefix==prompt_tokens, `k=0` new tokens to
prefill, and decode starts immediately from the one cached logit
regeneration step (RESUME). This is why priming and the resume arm submit the
prompt as a genuinely fresh generation request rather than an engine-internal
":MORE" continuation: both arms are really asking the engine "generate a
response to prompt P", and the resume-hit proof below is exactly the
evidence that the resume arm's answer to that question skipped recomputing P.

Both `TimedEngine` (this file) and `openai_server.Engine` speak the identical
wire protocol; TimedEngine exists only to add per-trial stderr capture (the
resume-hit proof lives in stderr: "[KV] resumed conversation from disk: N
tokens", "[API] KV slot S prefix P/T token, prefill K") and precise
before-spawn / first-token timestamps that openai_server.Engine has no reason
to expose for its actual job (serving).

CONTENTION DISCLOSURE (read before citing an absolute TTFT from this run):
this harness was run while two sibling engine Qwen3-32B servers (ports 8080/8081)
and their router (port 8100) were live on the same machine -- expected,
user-owned, EXPLICITLY NOT to be waited out or touched (see `preflight`).
Every trial record's `contention` field states what else was running at that
trial's start. Absolute TTFTs in this run are not contention-free; the
cold/resume RATIO (both arms drawn from the same ABBA-interleaved contention
regime) is the headline number for exactly that reason.

Modes:
  --self-test   no model, no engine, no subprocess: validates ABBA scheduling,
                bootstrap stats, and this file's stderr-proof/`.ili_kv`-header
                parsers against synthetic data. Safe to run any time.
  --dry-run     print the planned schedule/config and exit; spawns nothing.
  --pilot       prime once, run ONE cold trial and ONE resume trial, report
                timing. Use this FIRST on a real machine to size --n-per-arm
                to the wall-clock budget before committing to a full run.
  (default)     prime, then execute the full ABBA schedule.
"""
from __future__ import annotations

import argparse
import codecs
import dataclasses
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent          # c/bench-m5max/ili-kv-resume-abba
C_DIR = HERE.parent.parent                       # c/
RESULTS_DIR = HERE / "results"
LOGS_DIR = RESULTS_DIR / "logs"

sys.path.insert(0, str(C_DIR))
from openai_server import render_chat, read_engine_turn, READY, END  # noqa: E402

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path(os.environ.get("ILI_MODEL_DIR", "GLM-5.2-int4-with-int8-mtp"))
DEFAULT_GLM_PATH = C_DIR / "glm"
DEFAULT_SEED = 20260720
DEFAULT_RAM_GB = 40.0
DEFAULT_CAP = 8
DEFAULT_TIMED_MAX_TOKENS = 12
DEFAULT_PROMPT_CHARS = 2000
DEFAULT_PREFLIGHT_HEADROOM_GB = 8.0
DEFAULT_PREFLIGHT_MAX_WAIT_S = 300
DEFAULT_PREFLIGHT_POLL_S = 20

COLD = "cold"
RESUME = "resume"
ARMS = (COLD, RESUME)

# Known-expected services as of 2026-07-20: two sibling engine dogfood servers
# + their router. These are expected to stay up for the duration of this
# run -- never a block condition, only a contention annotation.
KNOWN_EXPECTED_PORTS = {8080: "sibling engine server", 8081: "sibling engine server",
                        8100: "a two-tier router (not yet public)"}

TTFT_DEFINITION = (
    "TTFT = wall-clock seconds from immediately-before-subprocess.Popen of the "
    "iliria engine (c/glm, invoked with SERVE=1 exactly as openai_server.Engine "
    "does) to the harness's own on_text callback's FIRST invocation with "
    "non-empty decoded text -- i.e. the moment the first byte of the model's "
    "actual generated response is available on the engine's stdout stream. "
    "IDENTICAL measurement code (TimedEngine.generate) and definition for both "
    "arms; no per-arm special-casing anywhere in this harness.\n"
    "INCLUDED inside every trial's timed interval, for both arms alike: process "
    "spawn; one-time model-weight load (dense weights + tokenizer + runtime "
    "structures); the per-slot .ili_kv disk load performed at startup, BEFORE "
    "the engine ever emits its READY sentinel (glm.c serve_ctx_init -> "
    "kv_disk_load -- on the resume arm this is where a prior identical-prompt "
    "run's KV rows are loaded with 'no re-prefill', read directly off the "
    "engine's own stderr; on the cold arm .ili_kv is absent so this step is a "
    "silent no-op); receipt of READY; transmission of the \\x02PROMPT frame; "
    "the engine's own prefix-match of the incoming prompt against its resident "
    "KV history (glm.c run_serve raw_mode: a cache MISS/cold arm means the full "
    "chunked prefill over every prompt token, a cache HIT/resume arm means the "
    "prefix walk reaches the full prompt length so k=0 new tokens to prefill); "
    "and the first decode/sampling step.\n"
    "EXCLUDED from every trial in both arms, identically: nothing process-level "
    "is excluded -- see the next paragraph for why this is a deliberate, "
    "documented departure from the sibling engine's own TTFT_DEFINITION.\n"
    "This is DELIBERATELY DIFFERENT from the sibling engine's own "
    "TTFT_DEFINITION, which explicitly EXCLUDES process spawn and "
    "model load (the sibling engine holds one process constant across all ABBA trials, never "
    "restarting, so 'cold' there means only 'not in this process's cache yet'). "
    "iliria's own KV-resume claim is specifically about surviving a full "
    "process restart -- excluding process/model-load time here would silently "
    "re-introduce the exact ambiguity ('did TTFT include process start? model "
    "load? KV load?') that retired the old 142x number in the first place. "
    "Both arms restart the engine process fresh for every trial; process spawn "
    "and model load are therefore symmetric, paid-in-both-arms costs, "
    "deliberately left inside the timed window rather than factored out."
)

# ---------------------------------------------------------------------------
# ABBA schedule -- near-verbatim port of the sibling engine's
# build_abba_schedule / arm_counts (same algorithm,
# same counterbalancing argument: every 4-trial block is either
# [cold,resume,resume,cold] or [resume,cold,cold,resume], chosen by an
# independent fair coin per block via a seeded RNG, so a linear drift within
# a block -- thermal ramp, background load ramping up -- contributes equally
# to both arms' means regardless of which arm leads; see
# the sibling engine's prune-certify harness's docstring for the algebra).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScheduledTrial:
    index: int
    block_index: int
    position_in_block: int
    arm: str


def build_abba_schedule(n_per_arm: int, *, seed: int) -> list[ScheduledTrial]:
    if n_per_arm < 1:
        raise ValueError(f"n_per_arm must be >= 1, got {n_per_arm}")
    num_blocks = -(-n_per_arm // 2)  # ceil division
    rng = random.Random(seed)
    schedule: list[ScheduledTrial] = []
    for block_index in range(num_blocks):
        lead = rng.choice(ARMS)
        other = RESUME if lead == COLD else COLD
        for position_in_block, arm in enumerate((lead, other, other, lead)):
            schedule.append(ScheduledTrial(len(schedule), block_index, position_in_block, arm))
    return schedule


def arm_counts(schedule: list[ScheduledTrial]) -> dict[str, int]:
    counts = {arm: 0 for arm in ARMS}
    for trial in schedule:
        counts[trial.arm] += 1
    return counts


# ---------------------------------------------------------------------------
# Bootstrap stats -- near-verbatim port of the sibling engine's
# bootstrap-stats module (same percentile-bootstrap algorithm: seeded
# random.Random, resample with replacement, sort, slice at the tail
# percentiles; stdlib-only, no numpy/scipy).
# ---------------------------------------------------------------------------

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_CI = 0.95


@dataclass(frozen=True, slots=True)
class MedianCIResult:
    n: int
    median: float
    ci_low: float
    ci_high: float
    ci_level: float
    resamples: int
    seed: int


def bootstrap_median_ci(values, *, resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                         ci_level=DEFAULT_CI, seed=0) -> MedianCIResult:
    clean = [v for v in values if v == v]
    n = len(clean)
    if n == 0:
        return MedianCIResult(0, float("nan"), float("nan"), float("nan"), ci_level, resamples, seed)
    if n == 1:
        return MedianCIResult(1, clean[0], clean[0], clean[0], ci_level, resamples, seed)
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    tail = (1.0 - ci_level) / 2.0
    lo_index = max(0, int(tail * resamples))
    hi_index = min(resamples - 1, int((1.0 - tail) * resamples))
    return MedianCIResult(n=n, median=statistics.median(clean), ci_low=medians[lo_index],
                           ci_high=medians[hi_index], ci_level=ci_level, resamples=resamples, seed=seed)


@dataclass(frozen=True, slots=True)
class SpeedupCIResult:
    n_cold: int
    n_resume: int
    point_estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    resamples: int
    seed: int


def bootstrap_speedup_ci(cold_values, resume_values, *, resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                          ci_level=DEFAULT_CI, seed=0) -> SpeedupCIResult:
    cold_clean = [v for v in cold_values if v == v and v > 0]
    resume_clean = [v for v in resume_values if v == v and v > 0]
    n_cold, n_resume = len(cold_clean), len(resume_clean)
    if n_cold == 0 or n_resume == 0:
        return SpeedupCIResult(n_cold, n_resume, float("nan"), float("nan"), float("nan"), ci_level, resamples, seed)
    point_estimate = statistics.median(cold_clean) / statistics.median(resume_clean)
    if n_cold == 1 and n_resume == 1:
        return SpeedupCIResult(n_cold, n_resume, point_estimate, point_estimate, point_estimate, ci_level, resamples, seed)
    rng = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        cold_sample = [cold_clean[rng.randrange(n_cold)] for _ in range(n_cold)]
        resume_sample = [resume_clean[rng.randrange(n_resume)] for _ in range(n_resume)]
        ratios.append(statistics.median(cold_sample) / statistics.median(resume_sample))
    ratios.sort()
    tail = (1.0 - ci_level) / 2.0
    lo_index = max(0, int(tail * resamples))
    hi_index = min(resamples - 1, int((1.0 - tail) * resamples))
    return SpeedupCIResult(n_cold=n_cold, n_resume=n_resume, point_estimate=point_estimate,
                            ci_low=ratios[lo_index], ci_high=ratios[hi_index],
                            ci_level=ci_level, resamples=resamples, seed=seed)


# ---------------------------------------------------------------------------
# Target prompt -- fixed, deterministic, sha256-pinned. Tiled synthetic
# filler (own text, not copied from the sibling engine's bench/prompts fixtures) so the
# actual token count is whatever iliria's own GLM-5.2 tokenizer produces --
# discovered empirically from the engine's own STAT `prompt_tokens` field
# rather than assumed, exactly because this harness does not depend on a
# separate local tokenizer.
# ---------------------------------------------------------------------------

_SEED_PARAGRAPH = (
    "Iliria streams mixture-of-experts weight shards from local NVMe storage "
    "on demand, keeping only a bounded working set resident in RAM while the "
    "great majority of the parameter count stays on disk until a routed token "
    "actually needs it. This paragraph is deliberately generic, synthetic "
    "filler used only to build a prompt of a controlled, reproducible length "
    "for a cold-versus-resume timing comparison; its content is never scored "
    "and never inspected beyond the token count and byte-identity checks this "
    "harness performs. "
)


def build_target_prompt(min_chars: int) -> str:
    paras = []
    total = 0
    i = 0
    while total < min_chars:
        para = f"[{i}] {_SEED_PARAGRAPH}"
        paras.append(para)
        total += len(para) + 1
        i += 1
    return "\n".join(paras)


# ---------------------------------------------------------------------------
# .ili_kv on-disk header -- direct parse of glm.c's kv_hdr/kv_disk_append
# format (KV_MAGIC "COLIKV1\0" + 8 int32 fields; nrec is h[6]). Used both to
# VERIFY priming actually landed on disk (rather than trusting a sleep/race)
# and as an independent (non-stderr) cross-check on the resume arm's
# pre-trial state.
# ---------------------------------------------------------------------------

KV_MAGIC = b"COLIKV1\x00"


def read_ili_kv_header(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != KV_MAGIC:
            return None
        raw = f.read(32)
        if len(raw) != 32:
            return None
        h = struct.unpack("<8i", raw)
    return {"n_layers": h[0], "kv_lora": h[1], "qk_rope": h[2], "index_hd": h[3],
            "nic": h[4], "vocab": h[5], "nrec": h[6]}


def kv_path(model_dir: Path, slot: int = 0) -> Path:
    return model_dir / (".ili_kv" if slot == 0 else f".ili_kv.{slot}")


def clear_kv(model_dir: Path) -> None:
    p = kv_path(model_dir, 0)
    if p.exists():
        p.unlink()


def install_primed_kv(model_dir: Path, primed_path: Path) -> None:
    p = kv_path(model_dir, 0)
    if p.exists():
        p.unlink()
    shutil.copyfile(primed_path, p)


# ---------------------------------------------------------------------------
# stderr resume-hit / cold-hit proof parsing -- the ground truth this
# harness's citation rests on, read directly off the engine's own diagnostic
# output rather than assumed from which arm the schedule says a trial is:
#   "[KV] resumed conversation from disk: N tokens in T s (no re-prefill)"
#     -- present iff kv_disk_load found a valid, non-empty .ili_kv at
#        startup (glm.c serve_ctx_init, BEFORE the engine ever prints READY).
#   "[API] KV slot S prefix P/T token, prefill K"
#     -- printed for every raw-mode request: P = the length of the resident
#        KV prefix that matched the incoming prompt's own tokenization, T =
#        the incoming prompt's total token count, K = T-P = new tokens that
#        actually needed prefilling. A perfect resume hit is P==T, K==0; a
#        genuinely cold request is P==0, K==T.
# ---------------------------------------------------------------------------

_RESUMED_RE = re.compile(r"\[KV\] resumed conversation from disk: (\d+) tokens in ([\d.]+)s")
_API_PREFIX_RE = re.compile(r"\[API\] KV slot (\d+) prefix (\d+)/(\d+) token, prefill (\d+)")
_VERSION_MISMATCH_RE = re.compile(r"\[KV\] ignoring \.ili_kv from a different model or version")


def parse_resume_proof(stderr_text: str) -> dict:
    resumed = _RESUMED_RE.search(stderr_text)
    api = _API_PREFIX_RE.search(stderr_text)
    mismatch = _VERSION_MISMATCH_RE.search(stderr_text)
    return {
        "kv_resumed_from_disk": ({"tokens": int(resumed.group(1)), "load_seconds": float(resumed.group(2))}
                                  if resumed else None),
        "api_prefix_line": ({"slot": int(api.group(1)), "prefix": int(api.group(2)),
                              "prompt_tokens": int(api.group(3)), "prefill": int(api.group(4))}
                             if api else None),
        "kv_version_mismatch_warning": bool(mismatch),
    }


def proof_matches_arm(arm: str, proof: dict) -> bool | None:
    """None when the engine's own telemetry doesn't say (never silently False)."""
    api = proof["api_prefix_line"]
    if api is None:
        return None
    actual_full_hit = api["prefill"] == 0 and api["prefix"] == api["prompt_tokens"] and api["prompt_tokens"] > 0
    return actual_full_hit == (arm == RESUME)


# ---------------------------------------------------------------------------
# System/contention snapshot -- read-only, never blocks by itself, never
# kills/signals anything. Distinguishes KNOWN_EXPECTED_PORTS (the sibling engine + router,
# per an operator's live mid-task update: these stay up, are NOT a block
# condition, only a contention annotation) from a genuinely unexpected
# competing ili/glm process (which IS a hard preflight failure -- see
# `preflight_with_retry`).
# ---------------------------------------------------------------------------


def _pid_of(ps_line: str) -> int | None:
    parts = ps_line.split()
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return None


def read_vm_stat_gb() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10, check=True).stdout
    m = re.search(r"page size of (\d+) bytes", out)
    page_size = int(m.group(1)) if m else 16384

    def pages(label):
        mm = re.search(rf"{re.escape(label)}:\s+(\d+)\.", out)
        return int(mm.group(1)) if mm else 0

    free = pages("Pages free")
    inactive = pages("Pages inactive")
    spec = pages("Pages speculative")
    active = pages("Pages active")
    wired = pages("Pages wired down")
    to_gb = lambda p: p * page_size / 1e9
    return {
        "free_gb": to_gb(free), "inactive_gb": to_gb(inactive), "speculative_gb": to_gb(spec),
        "active_gb": to_gb(active), "wired_gb": to_gb(wired),
        # macOS reclaims inactive/speculative pages on demand; this is the
        # conventional "loosely available" figure (matches Activity Monitor's
        # App Memory + Cached Files framing), not just the strict free count.
        "available_gb": to_gb(free + inactive + spec),
    }


def snapshot_environment(glm_path: Path) -> dict:
    ps_out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10).stdout
    try:
        lsof_out = subprocess.run(["lsof", "-iTCP", "-sTCP:LISTEN", "-P"],
                                   capture_output=True, text=True, timeout=10).stdout
    except Exception:
        lsof_out = ""
    listening_ports: dict[int, str] = {}
    for line in lsof_out.splitlines()[1:]:
        m = re.search(r":(\d+) \(LISTEN\)", line)
        if m:
            listening_ports[int(m.group(1))] = line.split()[0]
    known_running = {str(p): {"process": listening_ports[p], "role": role}
                      for p, role in KNOWN_EXPECTED_PORTS.items() if p in listening_ports}
    glm_lines = [l for l in ps_out.splitlines()
                 if re.search(re.escape(str(glm_path)) + r"|/glm\b", l) and "grep" not in l]
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ram": read_vm_stat_gb(),
        "known_expected_services_running": known_running,
        "glm_process_lines": glm_lines,
    }


def preflight_with_retry(glm_path: Path, ram_budget_gb: float, tracked_pids: set[int],
                          headroom_gb: float = DEFAULT_PREFLIGHT_HEADROOM_GB,
                          max_wait_s: float = DEFAULT_PREFLIGHT_MAX_WAIT_S,
                          poll_interval_s: float = DEFAULT_PREFLIGHT_POLL_S):
    """Read-only safety gate before every engine spawn. Never kills/signals
    anything (hard rail). Pauses and retries on (a) insufficient RAM headroom
    or (b) a foreign ili/glm process this harness did not itself start;
    treats the known the sibling engine/router services as ambient, non-blocking
    contention to ANNOTATE, not wait out (operator update, 2026-07-20).
    Returns (snapshot, waited_seconds) on success, or (None, waited_seconds)
    if still unsafe after max_wait_s."""
    waited = 0.0
    while True:
        snap = snapshot_environment(glm_path)
        foreign = [l for l in snap["glm_process_lines"] if _pid_of(l) not in tracked_pids]
        ram_ok = snap["ram"]["available_gb"] >= ram_budget_gb + headroom_gb
        if not foreign and ram_ok:
            snap["contended_by_known_services"] = bool(snap["known_expected_services_running"])
            return snap, waited
        if waited >= max_wait_s:
            return None, waited
        reason = ("foreign ili/glm process running: " + "; ".join(foreign) if foreign else
                  f"insufficient RAM headroom (available={snap['ram']['available_gb']:.1f}GB, "
                  f"need>={ram_budget_gb + headroom_gb:.1f}GB)")
        print(f"  [preflight] PAUSED ({reason}); re-checking in {poll_interval_s:.0f}s "
              f"({waited:.0f}/{max_wait_s:.0f}s elapsed) -- never evicting.", file=sys.stderr)
        time.sleep(poll_interval_s)
        waited += poll_interval_s


# ---------------------------------------------------------------------------
# TimedEngine -- adapted from openai_server.Engine (same wire protocol,
# same subprocess.Popen shape) with two additions the production server has
# no reason to need: (1) per-trial stderr captured to its own log file
# (the resume-hit proof lives there), (2) precise before-spawn and
# first-token timestamps for TTFT. `close_gracefully` closes stdin and waits
# for the engine's own natural exit (its getline loop hits EOF and falls
# through to a clean shutdown) rather than SIGTERM, specifically so that a
# turn's kv_disk_append (which the engine runs AFTER printing its STAT line,
# see glm.c run_serve) is guaranteed to have completed before this harness
# ever reads or copies the resulting .ili_kv file -- a signal could arrive
# mid-write; EOF-triggered shutdown cannot, because the engine cannot reach
# its next getline() (and therefore cannot exit on EOF) until the current
# line's full processing, including kv_disk_append, returns.
# ---------------------------------------------------------------------------


class TimedEngine:
    def __init__(self, executable: Path, model: Path, cap: int, ngen_ceiling: int,
                 env: dict, kv_slots: int, stderr_path: Path):
        self.t_spawn = time.time()
        self._stderr_f = open(stderr_path, "wb")
        child_env = dict(env, SNAP=str(model), SERVE="1", NGEN=str(ngen_ceiling), KV_SLOTS=str(kv_slots))
        self.process = subprocess.Popen(
            [str(executable), str(cap)], env=child_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._stderr_f, bufsize=0,
        )
        read_engine_turn(self.process.stdout, READY, lambda _b: None)
        self.t_ready = time.time()

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, cache_slot: int = 0):
        payload = prompt.encode("utf-8")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        chunks: list[str] = []
        first_token_time: list[float | None] = [None]

        def on_text(text: str) -> None:
            if first_token_time[0] is None:
                first_token_time[0] = time.time()
            chunks.append(text)

        def decode(data: bytes) -> None:
            text = decoder.decode(data)
            if text:
                on_text(text)

        header = (f"\x02PROMPT {len(payload)} {max_tokens} {temperature:.8g} "
                  f"{top_p:.8g} {cache_slot}\n").encode()
        view = memoryview(header + payload + b"\n")
        while view:
            written = self.process.stdin.write(view)
            if written is None:
                break
            view = view[written:]
        self.process.stdin.flush()
        stats = read_engine_turn(self.process.stdout, END, decode)
        tail = decoder.decode(b"", final=True)
        if tail:
            on_text(tail)
        return stats, "".join(chunks), first_token_time[0]

    def close_gracefully(self, timeout: float = 60.0) -> bool:
        clean = True
        try:
            self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            clean = False
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        try:
            self._stderr_f.close()
        except Exception:
            pass
        return clean


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------


@dataclass
class Config:
    model_dir: Path
    glm_path: Path
    ram_gb: float
    cap: int
    timed_max_tokens: int
    target_prompt: str
    seed: int
    primed_kv_path: Path | None = None


def build_env(cfg: Config) -> dict:
    env = dict(os.environ)
    env["RAM_GB"] = str(cfg.ram_gb)
    return env


def run_priming(cfg: Config) -> dict:
    """One-time setup: fresh engine, .ili_kv absent, submit the target prompt
    once, verify (by re-reading the .ili_kv header directly, not by trusting
    a sleep) that the resulting KV rows really landed on disk, then snapshot
    that file as results/primed.ili_kv -- the canonical resume-arm seed every
    RESUME trial below reuses. Unlike the sibling engine's in-RAM prompt cache, iliria's
    disk-persisted KV does not need re-priming per trial: the same bytes on
    disk reproduce the same resumed state every time (see install_primed_kv)."""
    clear_kv(cfg.model_dir)
    stderr_path = LOGS_DIR / "priming-server.log"
    engine = TimedEngine(cfg.glm_path, cfg.model_dir, cfg.cap, cfg.timed_max_tokens,
                          build_env(cfg), 1, stderr_path)
    t_spawn = engine.t_spawn
    rendered = render_chat([{"role": "user", "content": cfg.target_prompt}])
    stats, text, t_first = engine.generate(rendered, cfg.timed_max_tokens, 0.0, 1.0, cache_slot=0)
    clean_exit = engine.close_gracefully()

    header = read_ili_kv_header(kv_path(cfg.model_dir, 0))
    # Only the PROMPT portion needs to be durably on disk for a resume trial's
    # prefix-match to land a full hit (glm.c's prefix walk only needs
    # hist[0:prompt_tokens] to match) -- the response tail is irrelevant to
    # that mechanism, and in practice is not always saved token-for-token
    # (e.g. a trailing stop token can be excluded from what's persisted), so
    # requiring nrec >= prompt_tokens + completion_tokens is stricter than
    # what correctness actually requires and was observed to fail by exactly
    # one token on a real run. Gate on the requirement that matters.
    expected_min = stats["prompt_tokens"]
    verified = header is not None and header["nrec"] >= expected_min
    if not verified:
        raise RuntimeError(
            f"priming verification FAILED: on-disk nrec={header and header['nrec']}, "
            f"expected >= {expected_min} (prompt_tokens). See {stderr_path}."
        )

    primed_copy = RESULTS_DIR / "primed.ili_kv"
    shutil.copyfile(kv_path(cfg.model_dir, 0), primed_copy)
    cfg.primed_kv_path = primed_copy
    clear_kv(cfg.model_dir)

    return {
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "ttft_seconds": t_first - t_spawn if t_first else None,
        "generated_text": text,
        "generated_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "clean_exit": clean_exit,
        "primed_kv_path": str(primed_copy),
        "primed_kv_bytes": os.path.getsize(primed_copy),
        "disk_nrec_verified": header["nrec"],
        "stderr_log": str(stderr_path),
    }


def run_one_trial(cfg: Config, scheduled: ScheduledTrial, tracked_pids: set[int]) -> dict:
    arm = scheduled.arm
    snap, waited = preflight_with_retry(cfg.glm_path, cfg.ram_gb, tracked_pids)
    if snap is None:
        return {"ok": False, "phase": "preflight", "arm": arm, "index": scheduled.index,
                "block_index": scheduled.block_index,
                "error": f"unsafe to start engine after {waited:.0f}s of pausing (RAM or foreign process)"}

    clear_kv(cfg.model_dir)
    pre_header = None
    if arm == RESUME:
        assert cfg.primed_kv_path is not None, "resume arm scheduled before priming completed"
        install_primed_kv(cfg.model_dir, cfg.primed_kv_path)
        pre_header = read_ili_kv_header(kv_path(cfg.model_dir, 0))

    stderr_path = LOGS_DIR / f"trial-{scheduled.index:03d}-{arm}-server.log"
    try:
        engine = TimedEngine(cfg.glm_path, cfg.model_dir, cfg.cap, cfg.timed_max_tokens,
                              build_env(cfg), 1, stderr_path)
    except Exception as exc:
        clear_kv(cfg.model_dir)
        return {"ok": False, "phase": "spawn", "arm": arm, "index": scheduled.index,
                "block_index": scheduled.block_index, "error": repr(exc)}
    tracked_pids.add(engine.process.pid)
    t_spawn = engine.t_spawn

    rendered = render_chat([{"role": "user", "content": cfg.target_prompt}])
    try:
        stats, text, t_first = engine.generate(rendered, cfg.timed_max_tokens, 0.0, 1.0, cache_slot=0)
    except Exception as exc:
        engine.close_gracefully()
        tracked_pids.discard(engine.process.pid)
        clear_kv(cfg.model_dir)
        return {"ok": False, "phase": "generate", "arm": arm, "index": scheduled.index,
                "block_index": scheduled.block_index, "error": repr(exc)}

    clean_exit = engine.close_gracefully()
    tracked_pids.discard(engine.process.pid)

    if t_first is None:
        clear_kv(cfg.model_dir)
        return {"ok": False, "phase": "generate", "arm": arm, "index": scheduled.index,
                "block_index": scheduled.block_index,
                "error": "zero content tokens streamed -- ttft unavailable", "clean_exit": clean_exit}

    ttft = t_first - t_spawn
    stderr_text = stderr_path.read_text(errors="replace")
    proof = parse_resume_proof(stderr_text)
    matches = proof_matches_arm(arm, proof)
    clear_kv(cfg.model_dir)

    return {
        "ok": True, "index": scheduled.index, "block_index": scheduled.block_index,
        "position_in_block": scheduled.position_in_block, "arm": arm,
        "ttft_seconds": ttft, "t_spawn": t_spawn, "t_ready": engine.t_ready, "t_first_token": t_first,
        "load_seconds": engine.t_ready - t_spawn,
        "prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["completion_tokens"],
        "tokens_per_second": stats["tokens_per_second"], "rss_gb": stats["rss_gb"],
        "generated_text": text, "generated_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "clean_exit": clean_exit,
        "resume_hit_proof": {**proof, "matches_arm_expectation": matches, "pre_trial_disk_header": pre_header},
        "contention": {
            "known_expected_services_running": snap["known_expected_services_running"],
            "contended": snap["contended_by_known_services"],
            "ram_available_gb_before_start": snap["ram"]["available_gb"],
            "paused_seconds_before_start": waited,
        },
        "stderr_log": str(stderr_path),
    }


# ---------------------------------------------------------------------------
# Self-test -- no model, no engine, no subprocess. Validates ABBA scheduling,
# bootstrap stats, and this file's own stderr/.ili_kv-header parsers against
# synthetic data. Mirrors the sibling engine's own --self-test convention.
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    print("=== run_kv_resume_abba --self-test (no model, no engine, no subprocess) ===", file=sys.stderr)
    failures = []

    n_per_arm = 7
    seed = 12345
    schedule = build_abba_schedule(n_per_arm, seed=seed)
    counts = arm_counts(schedule)
    if counts[COLD] != counts[RESUME]:
        failures.append(f"unbalanced arms: {counts}")
    if counts[COLD] != 8:
        failures.append(f"expected ceil(7/2)*2==8 per arm, got {counts[COLD]}")
    valid = ((COLD, RESUME, RESUME, COLD), (RESUME, COLD, COLD, RESUME))
    blocks: dict[int, list[ScheduledTrial]] = {}
    for t in schedule:
        blocks.setdefault(t.block_index, []).append(t)
    for idx, trials in sorted(blocks.items()):
        pattern = tuple(t.arm for t in trials)
        if pattern not in valid:
            failures.append(f"block {idx} invalid ABBA pattern: {pattern}")
    repeat = build_abba_schedule(n_per_arm, seed=seed)
    if [t.arm for t in repeat] != [t.arm for t in schedule]:
        failures.append("same seed produced a different schedule")
    many = build_abba_schedule(100, seed=seed)
    many_blocks: dict[int, list[ScheduledTrial]] = {}
    for t in many:
        many_blocks.setdefault(t.block_index, []).append(t)
    leads = {trials[0].arm for _, trials in sorted(many_blocks.items())}
    if leads != set(ARMS):
        failures.append(f"block leads never varied across 50 blocks: {leads}")

    rng = random.Random(seed + 1)
    cold_samples = [max(0.01, rng.gauss(30.0, 3.0)) for _ in range(counts[COLD])]
    resume_samples = [max(0.005, rng.gauss(2.0, 0.2)) for _ in range(counts[RESUME])]
    cold_ci = bootstrap_median_ci(cold_samples, resamples=2000, seed=seed)
    resume_ci = bootstrap_median_ci(resume_samples, resamples=2000, seed=seed)
    speedup = bootstrap_speedup_ci(cold_samples, resume_samples, resamples=2000, seed=seed)
    if not (cold_ci.ci_low > resume_ci.ci_high):
        failures.append(f"cold/resume CIs overlap on well-separated mock data: cold={cold_ci} resume={resume_ci}")
    if not (speedup.ci_low > 1.0):
        failures.append(f"speedup CI does not clear 1x on well-separated mock data: {speedup}")

    resume_stderr = (
        "[KV] context slots: 1 x 4096 tokens, projected pool 1.23 GB\n"
        "[KV] resumed conversation from disk: 512 tokens in 0.1s (no re-prefill)\n"
        "[API] KV slot 0 prefix 500/500 token, prefill 0\n"
    )
    cold_stderr = (
        "[KV] context slots: 1 x 4096 tokens, projected pool 1.23 GB\n"
        "[API] prefill 512/500 tok\n"
        "[API] KV slot 0 prefix 0/500 token, prefill 500\n"
    )
    rp = parse_resume_proof(resume_stderr)
    cp = parse_resume_proof(cold_stderr)
    if rp["kv_resumed_from_disk"] is None or rp["kv_resumed_from_disk"]["tokens"] != 512:
        failures.append(f"resume stderr parse failed: {rp}")
    if proof_matches_arm(RESUME, rp) is not True:
        failures.append(f"resume proof should match RESUME arm expectation: {rp}")
    if cp["kv_resumed_from_disk"] is not None:
        failures.append(f"cold stderr should show no resumed-from-disk line: {cp}")
    if proof_matches_arm(COLD, cp) is not True:
        failures.append(f"cold proof should match COLD arm expectation: {cp}")
    if proof_matches_arm(COLD, rp) is not False:
        failures.append("resume proof evaluated against COLD expectation should be False (mismatch), got otherwise")

    scratch = RESULTS_DIR / "_self_test_kv_header.bin"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(scratch, "wb") as f:
        f.write(KV_MAGIC)
        f.write(struct.pack("<8i", 78, 128, 64, 0, 0, 151552, 999, 0))
    hdr = read_ili_kv_header(scratch)
    if hdr is None or hdr["nrec"] != 999 or hdr["n_layers"] != 78:
        failures.append(f".ili_kv header round-trip failed: {hdr}")
    scratch.unlink()
    if read_ili_kv_header(RESULTS_DIR / "_does_not_exist.bin") is not None:
        failures.append("read_ili_kv_header should return None for a missing file")

    if failures:
        print("\nSELF-TEST FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"schedule: n_per_arm={counts[COLD]} (requested {n_per_arm}), blocks={len(schedule)//4}\n"
        f"cold(mock)   median={cold_ci.median:.3f}  CI=[{cold_ci.ci_low:.3f}, {cold_ci.ci_high:.3f}]\n"
        f"resume(mock) median={resume_ci.median:.3f}  CI=[{resume_ci.ci_low:.3f}, {resume_ci.ci_high:.3f}]\n"
        f"speedup(mock) = {speedup.point_estimate:.1f}x  CI=[{speedup.ci_low:.1f}x, {speedup.ci_high:.1f}x]\n"
        "\nSELF-TEST PASSED -- ABBA scheduling, bootstrap stats, stderr resume-hit "
        "proof parsing, and .ili_kv header parsing all verified end-to-end.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                         help="prime once, run ONE cold + ONE resume trial, report timing, exit")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--glm", type=Path, default=DEFAULT_GLM_PATH)
    parser.add_argument("--ram-gb", type=float, default=DEFAULT_RAM_GB)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--timed-max-tokens", type=int, default=DEFAULT_TIMED_MAX_TOKENS)
    parser.add_argument("--prompt-chars", type=int, default=DEFAULT_PROMPT_CHARS)
    parser.add_argument("--n-per-arm", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--ci-level", type=float, default=DEFAULT_CI)
    parser.add_argument("--max-wall-seconds", type=float, default=5400.0,
                         help="stop scheduling NEW trials once this much wall-clock has elapsed "
                              "(default 90 min); already-started trials always finish")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "kv_resume_abba_results.json")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    schedule = build_abba_schedule(args.n_per_arm, seed=args.seed)
    n_actual = arm_counts(schedule)[COLD]

    if args.dry_run:
        print(json.dumps({
            "model": str(args.model), "glm": str(args.glm), "ram_gb": args.ram_gb, "cap": args.cap,
            "timed_max_tokens": args.timed_max_tokens, "prompt_chars": args.prompt_chars,
            "n_per_arm_requested": args.n_per_arm, "n_per_arm_actual": n_actual, "seed": args.seed,
            "abba_order": [t.arm for t in schedule],
            "ttft_definition": TTFT_DEFINITION,
        }, indent=2))
        return

    target_prompt = build_target_prompt(args.prompt_chars)
    cfg = Config(model_dir=args.model, glm_path=args.glm, ram_gb=args.ram_gb, cap=args.cap,
                 timed_max_tokens=args.timed_max_tokens, target_prompt=target_prompt, seed=args.seed)

    print(f"priming: prompt_chars={len(target_prompt)} sha256={hashlib.sha256(target_prompt.encode()).hexdigest()[:16]}...",
          file=sys.stderr)
    t0 = time.time()
    priming = run_priming(cfg)
    _primed_ttft = f"{priming['ttft_seconds']:.2f}s" if priming['ttft_seconds'] is not None else "N/A"
    print(f"  primed in {time.time()-t0:.1f}s wall -- prompt_tokens={priming['prompt_tokens']} "
          f"completion_tokens={priming['completion_tokens']} ttft={_primed_ttft} "
          f"disk_nrec_verified={priming['disk_nrec_verified']}", file=sys.stderr)

    tracked_pids: set[int] = set()

    if args.pilot:
        print("\n--pilot: one COLD trial, one RESUME trial (schedule/stats/ABBA not exercised)", file=sys.stderr)
        cold_trial = ScheduledTrial(0, 0, 0, COLD)
        resume_trial = ScheduledTrial(1, 0, 1, RESUME)
        results = []
        for st in (cold_trial, resume_trial):
            t_trial0 = time.time()
            r = run_one_trial(cfg, st, tracked_pids)
            wall = time.time() - t_trial0
            results.append(r)
            if r["ok"]:
                print(f"  [{st.arm:>6}] ttft={r['ttft_seconds']:.2f}s load={r['load_seconds']:.2f}s "
                      f"prompt_tokens={r['prompt_tokens']} completion_tokens={r['completion_tokens']} "
                      f"proof={r['resume_hit_proof']['api_prefix_line']} "
                      f"matches_expectation={r['resume_hit_proof']['matches_arm_expectation']} "
                      f"wall={wall:.1f}s", file=sys.stderr)
            else:
                print(f"  [{st.arm:>6}] ERROR phase={r['phase']} error={r['error']} wall={wall:.1f}s", file=sys.stderr)
        pilot_out = {"priming": priming, "trials": results, "ttft_definition": TTFT_DEFINITION,
                     "config": dataclasses.asdict(cfg) if False else {
                         "model_dir": str(cfg.model_dir), "ram_gb": cfg.ram_gb, "cap": cfg.cap,
                         "timed_max_tokens": cfg.timed_max_tokens, "prompt_chars": len(cfg.target_prompt),
                     }}
        pilot_path = RESULTS_DIR / "pilot.json"
        pilot_path.write_text(json.dumps(pilot_out, indent=2, default=str))
        print(f"\npilot results written to {pilot_path}", file=sys.stderr)
        clear_kv(cfg.model_dir)
        return

    # Full ABBA run
    print(f"\nfull ABBA run: n_per_arm={n_actual} (requested {args.n_per_arm}), "
          f"blocks={len(schedule)//4}, order={''.join(t.arm[0].upper() for t in schedule)}, "
          f"max_wall_seconds={args.max_wall_seconds:.0f}", file=sys.stderr)

    trials_path = RESULTS_DIR / "trials.jsonl"
    trial_records: list[dict] = []
    run_start = time.time()
    consecutive_errors = 0
    stopped_early_reason = None
    with open(trials_path, "w") as jf:
        for scheduled in schedule:
            elapsed = time.time() - run_start
            if elapsed >= args.max_wall_seconds:
                stopped_early_reason = f"max_wall_seconds budget ({args.max_wall_seconds:.0f}s) reached at trial index {scheduled.index}"
                print(f"  STOPPING: {stopped_early_reason}", file=sys.stderr)
                break
            t_trial0 = time.time()
            record = run_one_trial(cfg, scheduled, tracked_pids)
            wall = time.time() - t_trial0
            record["wall_seconds"] = wall
            trial_records.append(record)
            jf.write(json.dumps(record, default=str) + "\n")
            jf.flush()
            if record["ok"]:
                consecutive_errors = 0
                print(f"  [{scheduled.index:>3}] block={scheduled.block_index:>2} arm={scheduled.arm:<6} "
                      f"ok   ttft={record['ttft_seconds']:7.2f}s  load={record['load_seconds']:6.2f}s  "
                      f"matches_expectation={record['resume_hit_proof']['matches_arm_expectation']}  "
                      f"contended={record['contention']['contended']}  wall={wall:.1f}s  "
                      f"total_elapsed={time.time()-run_start:.0f}s", file=sys.stderr)
            else:
                consecutive_errors += 1
                print(f"  [{scheduled.index:>3}] block={scheduled.block_index:>2} arm={scheduled.arm:<6} "
                      f"ERROR phase={record['phase']} error={record['error']} wall={wall:.1f}s", file=sys.stderr)
                if consecutive_errors >= 3:
                    stopped_early_reason = f"{consecutive_errors} consecutive trial errors -- aborting rather than burning the time budget"
                    print(f"  STOPPING: {stopped_early_reason}", file=sys.stderr)
                    break

    clear_kv(cfg.model_dir)

    errors = [r for r in trial_records if not r["ok"]]
    cold_samples = [r["ttft_seconds"] for r in trial_records if r["ok"] and r["arm"] == COLD]
    resume_samples = [r["ttft_seconds"] for r in trial_records if r["ok"] and r["arm"] == RESUME]
    mismatches = [r for r in trial_records if r["ok"] and r["resume_hit_proof"]["matches_arm_expectation"] is False]

    cold_ci = bootstrap_median_ci(cold_samples, resamples=args.resamples, ci_level=args.ci_level, seed=args.seed)
    resume_ci = bootstrap_median_ci(resume_samples, resamples=args.resamples, ci_level=args.ci_level, seed=args.seed)
    speedup = bootstrap_speedup_ci(cold_samples, resume_samples, resamples=args.resamples,
                                    ci_level=args.ci_level, seed=args.seed)

    cold_texts = {r["generated_text_sha256"] for r in trial_records if r["ok"] and r["arm"] == COLD}
    resume_texts = {r["generated_text_sha256"] for r in trial_records if r["ok"] and r["arm"] == RESUME}
    token_identical = bool(cold_texts) and bool(resume_texts) and cold_texts == resume_texts

    results = {
        "harness": "c/bench-m5max/ili-kv-resume-abba/run_kv_resume_abba.py",
        "purpose": ("Same-prompt cold-vs-resume ABBA for ILIRIA's OWN cross-restart .ili_kv "
                    "KV-resume, replacing the retired, uncitable N=1 142x claim "
                    "(docs/PERFORMANCE_THEORY.md serve_kv_persistence_ttft_speedup_x) with a "
                    "bootstrap-CI'd number measured on iliria's own mechanism, not the sibling engine's "
                    "same-process prompt cache (that is a separate, already-certified 147.8x "
                    "result on a different engine -- see this repo's REPORT.md)."),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(run_start)),
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_dir": str(cfg.model_dir), "glm_path": str(cfg.glm_path),
        "ram_gb_budget": cfg.ram_gb, "cap": cfg.cap,
        "prompt_chars": len(cfg.target_prompt), "target_prompt_sha256": hashlib.sha256(cfg.target_prompt.encode()).hexdigest(),
        "timed_max_tokens": cfg.timed_max_tokens,
        "n_per_arm_requested": args.n_per_arm, "n_per_arm_actual": n_actual, "seed": args.seed,
        "abba_order": [t.arm for t in schedule],
        "ttft_definition": TTFT_DEFINITION,
        "priming": priming,
        "n_trials_scheduled": len(schedule), "n_trials_run": len(trial_records),
        "stopped_early_reason": stopped_early_reason,
        "n_errors": len(errors), "resume_hit_proof_mismatches": len(mismatches),
        "token_identity": {"cold_unique_sha256": sorted(cold_texts), "resume_unique_sha256": sorted(resume_texts),
                            "token_identical_across_arms": token_identical},
        "cold_ttft_seconds": dataclasses.asdict(cold_ci),
        "resume_ttft_seconds": dataclasses.asdict(resume_ci),
        "speedup_cold_over_resume": dataclasses.asdict(speedup),
        "trials_jsonl": str(trials_path),
    }
    args.out.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 78, file=sys.stderr)
    print(f"cold   median={cold_ci.median:7.2f}s  CI=[{cold_ci.ci_low:7.2f}, {cold_ci.ci_high:7.2f}]s  n={cold_ci.n}",
          file=sys.stderr)
    print(f"resume median={resume_ci.median:7.2f}s  CI=[{resume_ci.ci_low:7.2f}, {resume_ci.ci_high:7.2f}]s  n={resume_ci.n}",
          file=sys.stderr)
    print(f"speedup (cold/resume) = {speedup.point_estimate:.1f}x  CI=[{speedup.ci_low:.1f}x, {speedup.ci_high:.1f}x]",
          file=sys.stderr)
    print(f"token-identical across arms: {token_identical}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} trial(s) ERRORED -- see {trials_path}", file=sys.stderr)
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} trial(s) show resume-hit telemetry inconsistent with their "
              "arm's expectation -- investigate before citing these numbers.", file=sys.stderr)
    print(f"\nfull results at {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
