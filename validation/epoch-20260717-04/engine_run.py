#!/usr/bin/env python3
"""Shared engine-invocation + output-parsing + calibration library for the overhead
gate's two experiments (fixture, real-model). Used by run_overhead_gate.sh via `python3
engine_run.py <subcommand> ...` -- see each subcommand's --help.

Every subprocess launch here runs glm.c's REPLAY mode (`REPLAY=1`, teacher-forced
decode against a fixed `full_ids` token sequence -- see glm.c's run_replay(), and this
directory's README.md for why REPLAY, never free-running generate). Nothing here
samples, generates, or depends on model output correctness -- REPLAY feeds a FIXED
sequence regardless of what the model would have predicted, so timing is all this cares
about.

Subcommands:
  slice-ref     write a REF json that is prompt_ids + full_ids[:n] from a master REF file
  verify-hash   sha256-verify a binary against binary_hashes.json, exit nonzero on mismatch
  run-one       launch ONE REPLAY invocation, parse its output, print one JSON record
  calibrate     auto-tune the token count N so REPLAY's OWN reported decode wall time
                (glm.c's "REPLAY decode: N tokens in %.3fs" line -- NOT process wall
                time, which includes model load/PIN overhead this run's 10-20s target
                does not care about) lands in [target_lo, target_hi] seconds
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- output parsing ---------------------------------------------------------------

RE_REPLAY = re.compile(
    r"REPLAY decode: (?P<tokens>\d+) tokens in (?P<decode_s>[\d.]+)s \| "
    r"(?P<tok_s>[\d.]+) tok/s \| expert hit (?P<hit_pct>[\d.]+)%")
RE_STALL_EXPOSED = re.compile(
    r"STALL-EXPOSED: (?P<stall_s>[\d.]+)s .* \| pipe-waits (?P<pipe_waits>\d+) "
    r"blocked (?P<pipe_blocked>\d+) \(occupancy (?P<pipe_occ_pct>[\d.]+)%\)")
RE_IO_BYTES = re.compile(
    r"IO-BYTES: requested (?P<req_bytes>\d+) \| read (?P<read_bytes>\d+) \| "
    r"reads attempted (?P<reads_attempted>\d+) completed (?P<reads_completed>\d+) \| "
    r"hits (?P<hits>\d+) misses (?P<misses>\d+) \((?P<hit_pct>[\d.]+)% hit\)")
RE_IO_LATENCY = re.compile(
    r"IO-LATENCY: read-completion p50 (?P<p50_ms>[\d.]+)ms p95 (?P<p95_ms>[\d.]+)ms "
    r"p99 (?P<p99_ms>[\d.]+)ms \(n=(?P<n_samples>\d+) samples\)")
RE_WALL_SUM = re.compile(
    r"WALL-SUM: compute (?P<compute_s>[\d.]+)s \+ exposed-stall (?P<stall_s>[\d.]+)s \+ "
    r"other (?P<other_s>[\d.]+)s = wall (?P<wall_s>[\d.]+)s \| residual\(other\) "
    r"(?P<residual_pct>[\d.]+)% of wall")
RE_EFFECTIVE_FLAG = re.compile(
    r"EFFECTIVE-FLAGS: (?P<name>\w+) requested=(?P<requested>\S+) effective=(?P<effective>\d+)")
RE_RAM_CAP = re.compile(r"\[RAM_GB=(?P<ram_gb>[\d.]+)(?P<auto> auto)?\] .* cap (?:lowered|raised) (?P<from>\d+)->(?P<to>\d+)")


def parse_output(text: str) -> dict:
    out: dict = {"effective_flags": {}}
    for line in text.splitlines():
        m = RE_REPLAY.search(line)
        if m:
            out["replay"] = {"tokens": int(m["tokens"]), "decode_s": float(m["decode_s"]),
                              "tok_s": float(m["tok_s"]), "hit_pct": float(m["hit_pct"])}
            continue
        m = RE_STALL_EXPOSED.search(line)
        if m:
            out["stall_exposed"] = {"stall_s": float(m["stall_s"]),
                                     "pipe_waits": int(m["pipe_waits"]),
                                     "pipe_blocked": int(m["pipe_blocked"]),
                                     "pipe_occ_pct": float(m["pipe_occ_pct"])}
            continue
        m = RE_IO_BYTES.search(line)
        if m:
            out["io_bytes"] = {"req_bytes": int(m["req_bytes"]), "read_bytes": int(m["read_bytes"]),
                                "reads_attempted": int(m["reads_attempted"]),
                                "reads_completed": int(m["reads_completed"]),
                                "hits": int(m["hits"]), "misses": int(m["misses"]),
                                "hit_pct": float(m["hit_pct"])}
            continue
        m = RE_IO_LATENCY.search(line)
        if m:
            out["io_latency"] = {"p50_ms": float(m["p50_ms"]), "p95_ms": float(m["p95_ms"]),
                                  "p99_ms": float(m["p99_ms"]), "n_samples": int(m["n_samples"])}
            continue
        m = RE_WALL_SUM.search(line)
        if m:
            out["wall_sum"] = {"compute_s": float(m["compute_s"]), "stall_s": float(m["stall_s"]),
                                "other_s": float(m["other_s"]), "wall_s": float(m["wall_s"]),
                                "residual_pct": float(m["residual_pct"])}
            continue
        m = RE_EFFECTIVE_FLAG.search(line)
        if m:
            out["effective_flags"][m["name"]] = {"requested": m["requested"], "effective": int(m["effective"])}
            continue
        m = RE_RAM_CAP.search(line)
        if m:
            out["ram_cap"] = {"ram_gb": float(m["ram_gb"]), "auto": bool(m["auto"]),
                               "cap_from": int(m["from"]), "cap_to": int(m["to"])}
            continue
    return out


# ---- REF slicing -------------------------------------------------------------------

def slice_ref(master_ref_path: Path, n_tokens: int, out_path: Path) -> dict:
    """Writes prompt_ids + full_ids[:prompt_len + n_tokens] from the master REF file.
    n_tokens is the number of REPLAY DECODE steps (glm.c's run_replay(): decode covers
    positions [np-1, nfull-1), i.e. nfull-np decode steps) -- so full_ids length is
    prompt_len + n_tokens."""
    master = json.loads(master_ref_path.read_text())
    prompt_ids = master["prompt_ids"]
    full_ids_master = master["full_ids"]
    prompt_len = len(prompt_ids)
    need = prompt_len + n_tokens
    if need > len(full_ids_master):
        raise ValueError(f"requested {n_tokens} decode tokens needs {need} full_ids, "
                          f"master only has {len(full_ids_master)} -- regenerate the "
                          f"master REF with a longer master_len/decode_len")
    sliced = {"prompt_ids": prompt_ids, "full_ids": full_ids_master[:need],
              "_sliced_from": str(master_ref_path), "_n_decode_tokens": n_tokens}
    out_path.write_text(json.dumps(sliced))
    return sliced


# ---- hash verification --------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(binary_path: Path, expected_sha256: str) -> None:
    if not binary_path.exists():
        raise SystemExit(f"REFUSING TO RUN: binary not found at {binary_path}")
    actual = sha256_file(binary_path)
    if actual != expected_sha256:
        raise SystemExit(
            f"REFUSING TO RUN: sha256 mismatch for {binary_path}\n"
            f"  expected: {expected_sha256}\n  actual:   {actual}\n"
            f"  (binary changed since build_binaries.sh ran -- rebuild and re-verify "
            f"before trusting any timing from this binary)")


# ---- one REPLAY invocation -----------------------------------------------------------

def run_one(binary: Path, snap: Path, ref_path: Path, extra_env: dict, args: list[str],
            timeout_s: float) -> dict:
    env = dict(os.environ)
    env.update({
        "SNAP": str(snap),
        "REPLAY": "1",
        "REF": str(ref_path),
    })
    env.update(extra_env)
    cmd = [str(binary)] + [str(a) for a in args]
    t_wall0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
        exit_code = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        exit_code = 124
        stdout = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        timed_out = True
    t_wall1 = time.time()

    combined = stdout + "\n" + stderr
    parsed = parse_output(combined)
    record = {
        "command": " ".join(shlex.quote(c) for c in cmd),
        "env_overrides": {k: v for k, v in extra_env.items()},
        "process_wall_s": t_wall1 - t_wall0,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_wall0)),
        **parsed,
    }
    # The metric the overhead comparison actually uses: REPLAY's OWN internal decode
    # timer (excludes model load/PIN/mmap warmup, which is identical fixed cost on both
    # arms and would only dilute -- not bias, since both binaries pay it equally, but
    # dilute -- the per-read overhead signal this experiment exists to detect).
    record["wall_s"] = parsed.get("replay", {}).get("decode_s")
    return record


# ---- calibration ---------------------------------------------------------------------

def calibrate_token_count(binary: Path, snap: Path, master_ref: Path, extra_env: dict,
                           args: list[str], timeout_s: float, target_lo_s: float,
                           target_hi_s: float, work_dir: Path, start_n: int,
                           max_attempts: int = 6) -> tuple[int, list[dict]]:
    """Runs successive REPLAY calibration invocations (of the GIVEN binary -- caller's
    choice, numerics are irrelevant here, only decode_s matters) doubling/scaling N
    until decode_s lands in [target_lo_s, target_hi_s]. Returns (chosen_n, attempts).
    These calibration runs are real engine invocations and therefore must happen under
    the SAME preconditions (timing lock, quiesce, hash-verified binary) as any other
    timed run -- run_overhead_gate.sh calls this only after acquiring the lock, never
    before."""
    target_mid = (target_lo_s + target_hi_s) / 2.0
    n = start_n
    attempts = []
    for i in range(max_attempts):
        ref_path = work_dir / f"calib_{i}.json"
        slice_ref(master_ref, n, ref_path)
        rec = run_one(binary, snap, ref_path, extra_env, args, timeout_s)
        attempts.append({"attempt": i, "n": n, **rec})
        decode_s = rec.get("wall_s")
        if rec["exit_code"] != 0 or decode_s is None:
            raise SystemExit(f"calibration run {i} failed (exit={rec['exit_code']}, "
                              f"decode_s={decode_s}) -- cannot calibrate; see attempt log")
        if target_lo_s <= decode_s <= target_hi_s:
            return n, attempts
        rate = n / decode_s if decode_s > 0 else n  # tokens/sec implied by this attempt
        n_next = max(int(rate * target_mid), n + 1)
        # Guard against runaway growth from a near-zero decode_s on the first attempt.
        n_next = min(n_next, n * 20)
        n = n_next
    # Ran out of attempts: return the closest attempt's N as a best-effort choice,
    # loudly flagged in the returned attempts list (run_overhead_gate.sh decides
    # whether that's good enough or an INCONCLUSIVE verdict).
    best = min(attempts, key=lambda a: abs((a.get("wall_s") or 0) - target_mid))
    return best["n"], attempts


# ---- CLI -------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("slice-ref")
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--n-tokens", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("verify-hash")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)

    p = sub.add_parser("run-one")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--snap", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable")
    p.add_argument("--arg", action="append", default=[], help="positional engine arg, repeatable, in order")
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("calibrate")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--snap", type=Path, required=True)
    p.add_argument("--master-ref", type=Path, required=True)
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--arg", action="append", default=[])
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--target-lo-s", type=float, default=10.0)
    p.add_argument("--target-hi-s", type=float, default=20.0)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--start-n", type=int, default=200)
    p.add_argument("--out", type=Path)

    args = ap.parse_args(argv)

    def parse_env_list(items: list[str]) -> dict:
        d = {}
        for item in items:
            k, _, v = item.partition("=")
            d[k] = v
        return d

    if args.cmd == "slice-ref":
        sliced = slice_ref(args.master, args.n_tokens, args.out)
        print(f"wrote {args.out}: {len(sliced['full_ids'])} full_ids ({args.n_tokens} decode tokens)")
        return 0

    if args.cmd == "verify-hash":
        verify_hash(args.binary, args.expected_sha256)
        print(f"OK: {args.binary} matches expected sha256")
        return 0

    if args.cmd == "run-one":
        rec = run_one(args.binary, args.snap, args.ref, parse_env_list(args.env), args.arg or [], args.timeout_s)
        text = json.dumps(rec, indent=2)
        if args.out:
            args.out.write_text(text)
        print(text)
        return 0 if rec["exit_code"] == 0 else 1

    if args.cmd == "calibrate":
        args.work_dir.mkdir(parents=True, exist_ok=True)
        n, attempts = calibrate_token_count(
            args.binary, args.snap, args.master_ref, parse_env_list(args.env), args.arg or [],
            args.timeout_s, args.target_lo_s, args.target_hi_s, args.work_dir, args.start_n)
        result = {"chosen_n": n, "attempts": attempts}
        text = json.dumps(result, indent=2)
        if args.out:
            args.out.write_text(text)
        last = attempts[-1]
        print(f"calibrated N={n} (last attempt: n={last['n']} decode_s={last.get('wall_s')})")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
