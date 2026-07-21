#!/usr/bin/env python3
"""Shared engine-log parsing for the M5 Max frozen-state matrices.

Extracted from summarize_m5max_k6_matrix.py (which imports this module rather
than keeping its own copy) so that fight_report.py can parse the exact same
`run-m5max-fast.sh` log format without a second, independently-drifting
regex set. Both consumers care about staying byte-identical with the real
engine's stdout ("output token hash:", "PROFILE: ...", "(N tok/s) | expert hit
rate N%", the optional PILOT-* lines) -- one module, one contract.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Run:
    prompt: str
    cache_state: str
    trial: int
    mode: str
    tok_s: float
    hit_pct: float
    disk_s: float
    matmul_s: float
    attn_s: float
    output_hash: str
    predicted: int = 0
    enqueued: int = 0
    useful: int = 0
    wasted: int = 0
    late: int = 0
    evictions: int = 0
    pilot_load_s: float = 0.0
    barrier_s: float = 0.0
    blocked_pipe_s: float = 0.0
    swap_before_mb: float | None = None
    swap_after_mb: float | None = None


def last(text: str, pattern: str, default=None):
    values = re.findall(pattern, text, re.M)
    return values[-1] if values else default


def parse_log(path: Path, meta: dict) -> Run:
    """Parse one `run-m5max-fast.sh` log into a Run. `meta` supplies the
    caller's own bookkeeping fields (prompt/cache_state/trial/mode) since
    those describe how the run was invoked, not something printed in the log."""
    text = path.read_text(errors="replace")
    rate = last(text, r"\((\d+(?:\.\d+)?) tok/s\) \| expert hit rate")
    hit = last(text, r"expert hit rate (\d+(?:\.\d+)?)%")
    profile = last(
        text,
        r"PROFILE: expert-disk (\d+(?:\.\d+)?)s \| expert-matmul (\d+(?:\.\d+)?)s \| attention (\d+(?:\.\d+)?)s",
    )
    output_hash = last(text, r"output token hash: ([0-9a-fA-F]+)", "")
    if rate is None or hit is None or profile is None or not output_hash:
        raise ValueError(f"{path}: missing benchmark summary")

    pm1 = last(
        text,
        r"PILOT-METRICS: predicted (\d+) \| enqueued (\d+) \| resident-skip (\d+) \| race-skip (\d+) \| queue-full (\d+)",
    )
    pm2 = last(
        text,
        r"PILOT-OUTCOME: loads (\d+) \| useful (\d+) \| wasted (\d+) \| late (\d+) \| evictions (\d+) \| precision [\d.]+%",
    )
    pm3 = last(
        text,
        r"PILOT-TIME: load (\d+(?:\.\d+)?)s \| layer-barrier (\d+(?:\.\d+)?)s \| blocked-pipe (\d+(?:\.\d+)?)s \((\d+) waits\)",
    )
    swap_before = last(text, r"MATRIX-SWAP-BEFORE-MB: ([\d.]+)")
    swap_after = last(text, r"MATRIX-SWAP-AFTER-MB: ([\d.]+)")

    return Run(
        prompt=str(meta["prompt"]),
        cache_state=str(meta["cache_state"]),
        trial=int(meta["trial"]),
        mode=str(meta["mode"]),
        tok_s=float(rate),
        hit_pct=float(hit),
        disk_s=float(profile[0]),
        matmul_s=float(profile[1]),
        attn_s=float(profile[2]),
        output_hash=output_hash,
        predicted=int(pm1[0]) if pm1 else 0,
        enqueued=int(pm1[1]) if pm1 else 0,
        useful=int(pm2[1]) if pm2 else 0,
        wasted=int(pm2[2]) if pm2 else 0,
        late=int(pm2[3]) if pm2 else 0,
        evictions=int(pm2[4]) if pm2 else 0,
        pilot_load_s=float(pm3[0]) if pm3 else 0.0,
        barrier_s=float(pm3[1]) if pm3 else 0.0,
        blocked_pipe_s=float(pm3[2]) if pm3 else 0.0,
        swap_before_mb=float(swap_before) if swap_before else None,
        swap_after_mb=float(swap_after) if swap_after else None,
    )


def med(values):
    values = list(values)
    return statistics.median(values) if values else float("nan")
