#!/usr/bin/env python3
"""Dual-roofline report builder (docs/performance-theory.json p2-m5-gpu-tensor-path-probe).

Parses the engine's own stderr/stdout counters (the `PROFILE:`/`METAL-ATTN:`/`METAL:`/
`METAL-GEMM:` lines glm.c and backend_metal.mm already print, per run-mode invocation) plus
the telemetry CSVs scripts/roofline_run.sh samples every N seconds, and turns them into:

  1. a per-kernel-class table (attention score/latent, MoE GEMV, router, projections):
     effective MAC/s, bytes/s, arithmetic intensity, an occupancy proxy (GPU-kernel-time /
     GPU-wall-time) and command-gap time (GPU-wall-time - kernel-time, i.e. dispatch/
     scheduling overhead not spent executing).
  2. THREE distinct bandwidth definitions (device throughput, read-path throughput, and the
     EXPOSED critical-path bandwidth = compulsory miss bytes / main-thread storage stall --
     the only one of the three that belongs in a tok/s model, per this repo's own
     topp-adaptive-expert-routing / p2-m5-gpu-tensor-path-probe notes).
  3. a cold-vs-steady-state comparison and a thermal-derate estimate (steady median tok/s /
     cold median tok/s). ENDPOINT CLAIMS MUST USE THE STEADY-STATE MEDIAN, never the cold
     number -- this report always labels which is which.
  4. a roofline classification per kernel class (compute-ish / bandwidth-ish / sync-ish).
     This is a *relative*, self-consistent heuristic over this run's own measured numbers --
     there is no independently measured chip-peak GFLOP/s or GB/s anywhere in this tool, by
     design (method limits are stated explicitly in the report's own "Method & limits"
     section, not just in this docstring).

Two entry points:
  report    parse a completed scripts/roofline_run.sh result directory into a markdown report.
  mock-log  emit one synthetic engine log with the same line formats a real run produces, so
            the harness and this tool can be proven end to end without real hardware/model
            (see scripts/roofline_run.sh --mock, and tests/test_roofline_report.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---- GLM-5.2 shapes compiled into backend_metal.mm / tests/test_backend_metal.mm ----------
# Used ONLY to turn wall-clock GPU-time counters into approximate effective-MAC/s and
# bytes/s figures. The engine's own PROFILE/METAL-* counters are cumulative sums over a
# whole run, not per-op hardware counters, and context length T is a single representative
# value for the whole run (not tracked per-token by this harness) -- so every MAC/s,
# bytes/s, and arithmetic-intensity figure below is an order-of-magnitude modeled estimate,
# not a hardware-measured FLOP count. See "Method & limits" in the generated report.
HIDDEN = 6144
HEADS = 64
KV_LORA = 512
ROPE_DIM = 64
MOE_D = 6144
MOE_I = 2048

DEFAULT_AVG_CONTEXT_TOKENS = 2048
DEFAULT_DECODE_BATCH = 1
DEFAULT_PROJ_I = 6144
DEFAULT_PROJ_O = 2048
DEFAULT_PROJ_BITS = 4

DEFAULT_BYTES_PER_EXPERT_MB = 18.9153     # int4 tier -- docs/performance-theory.json, the container-design plan
DEFAULT_DEVICE_BW_GBS = 13.3              # the prefill-I/O study raw-SSD sweep: a PRIOR measurement,
                                           # not re-measured by this harness (stated below as a method limit)
DEFAULT_OVERLAP_FRACTION = 0.0            # conservative: assume the storage stall is fully exposed
AI_THRESHOLD_DEFAULT = 8.0                # MAC/byte; classification pivot -- NOT a chip-peak reference
OCCUPANCY_SYNC_THRESHOLD = 0.5            # below this, command-gap dominates -> "sync-ish"


# ---- parsing --------------------------------------------------------------------------

def _last_groups(text: str, pattern: str):
    """Return the LAST match's groups for `pattern` in `text`, or None. Engine counters are
    cumulative sums printed once at the end of a `run`-mode invocation, but taking the last
    match (not the first) matches this repo's own convention (tools/summarize_m5max_k6_matrix.py's
    `last()`) and is robust if a log ever contains more than one summary block."""
    matches = list(re.finditer(pattern, text, re.M))
    return matches[-1].groups() if matches else None


@dataclass
class EngineRunStats:
    log_path: str
    tokens: Optional[int] = None
    wall_s: Optional[float] = None
    tok_s: Optional[float] = None
    hit_pct: Optional[float] = None
    experts_per_token: Optional[float] = None
    t_edisk: Optional[float] = None
    t_ematmul: Optional[float] = None
    t_attn: Optional[float] = None
    t_kvb: Optional[float] = None
    t_head: Optional[float] = None
    t_other: Optional[float] = None
    metal_attn: Optional[dict] = None   # {ok, gpu_wall, kernel, cpu_sched, gpu_sched}
    metal_moe: Optional[dict] = None    # {ok, fb, experts, setup, gpu_wall, kernel, scatter}
    metal_gemm: Optional[dict] = None   # {ok, gpu_wall, kernel}

    @property
    def miss_experts(self) -> Optional[float]:
        if self.tokens is None or self.experts_per_token is None or self.hit_pct is None:
            return None
        return self.tokens * self.experts_per_token * (1.0 - self.hit_pct / 100.0)

    @property
    def hit_experts(self) -> Optional[float]:
        if self.tokens is None or self.experts_per_token is None or self.hit_pct is None:
            return None
        return self.tokens * self.experts_per_token * (self.hit_pct / 100.0)


def parse_engine_log(path: Path) -> EngineRunStats:
    text = path.read_text(errors="replace")
    stats = EngineRunStats(log_path=str(path))

    g = _last_groups(
        text,
        r"(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\) \| expert hit rate ([\d.]+)% \| RSS ([\d.]+) GB",
    )
    if g:
        stats.tokens = int(g[0])
        stats.wall_s = float(g[1])
        stats.tok_s = float(g[2])
        stats.hit_pct = float(g[3])

    g = _last_groups(text, r"experts loaded/token: ([\d.]+) \(per-layer")
    if g:
        stats.experts_per_token = float(g[0])

    g = _last_groups(
        text,
        r"PROFILE: expert-disk ([\d.]+)s \| expert-matmul ([\d.]+)s \| attention ([\d.]+)s "
        r"\(including kvb ([\d.]+)s\) \| lm_head ([\d.]+)s \| other (-?[\d.]+)s",
    )
    if g:
        stats.t_edisk, stats.t_ematmul, stats.t_attn, stats.t_kvb, stats.t_head, stats.t_other = (
            float(v) for v in g
        )

    g = _last_groups(
        text,
        r"METAL-ATTN: layer GPU (\d+) \| gpu-wall ([\d.]+)s "
        r"\(kernel ([\d.]+)s \| cpu-sched ([\d.]+)s gpu-sched ([\d.]+)s\)",
    )
    if g:
        stats.metal_attn = {
            "ok": int(g[0]), "gpu_wall": float(g[1]), "kernel": float(g[2]),
            "cpu_sched": float(g[3]), "gpu_sched": float(g[4]),
        }

    g = _last_groups(
        text,
        r"METAL: blocchi GPU (\d+) \| fallback CPU (\d+) \| expert su GPU (\d+) \| "
        r"setup ([\d.]+)s gpu-wall ([\d.]+)s \(kernel ([\d.]+)s\) scatter ([\d.]+)s",
    )
    if g:
        stats.metal_moe = {
            "ok": int(g[0]), "fb": int(g[1]), "experts": int(g[2]),
            "setup": float(g[3]), "gpu_wall": float(g[4]), "kernel": float(g[5]), "scatter": float(g[6]),
        }

    g = _last_groups(text, r"METAL-GEMM: calls (\d+) \| gpu-wall ([\d.]+)s \(kernel ([\d.]+)s\)")
    if g:
        stats.metal_gemm = {"ok": int(g[0]), "gpu_wall": float(g[1]), "kernel": float(g[2])}

    return stats


TELEMETRY_FIELDS = [
    "timestamp", "phase", "elapsed_s", "thermal_speed_limit_pct", "cpu_idle_pct",
    "disk_mb_s", "pageouts_cum",
]


def parse_telemetry_csv(path: Path) -> list:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _floats(rows: list, key: str) -> list:
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            if v not in (None, ""):
                out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def telemetry_phase_summary(rows: list, phase: str) -> dict:
    sub = [r for r in rows if r.get("phase") == phase]
    out = {"phase": phase, "samples": len(sub)}
    for name in ("thermal_speed_limit_pct", "cpu_idle_pct", "disk_mb_s"):
        vals = _floats(sub, name)
        out[name] = {
            "median": statistics.median(vals) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }
    return out


def med(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


# ---- bandwidth math (three distinct definitions -- see module docstring) --------------

@dataclass
class BandwidthEstimate:
    compulsory_miss_bytes: Optional[float]
    device_throughput_gbs: float
    read_path_gbs: Optional[float]
    exposed_gbs: Optional[float]
    miss_experts: Optional[float]
    notes: list = field(default_factory=list)


def estimate_bandwidths(
    stats_list,
    bytes_per_expert_mb: float = DEFAULT_BYTES_PER_EXPERT_MB,
    device_bw_gbs: float = DEFAULT_DEVICE_BW_GBS,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
) -> BandwidthEstimate:
    """Aggregate compulsory-miss bytes and expert-disk wall time across every run in a
    phase, then derive read-path and exposed bandwidth from the totals (a byte-weighted
    average is more honest than averaging per-run ratios when run lengths differ)."""
    notes = []
    total_miss_experts = 0.0
    total_bytes = 0.0
    total_edisk = 0.0
    have_bytes = have_disk = False
    for s in stats_list:
        me = s.miss_experts
        if me is not None:
            total_miss_experts += me
            total_bytes += me * bytes_per_expert_mb * 1e6
            have_bytes = True
        if s.t_edisk is not None:
            total_edisk += s.t_edisk
            have_disk = True

    compulsory_miss_bytes = total_bytes if have_bytes else None
    if not have_bytes:
        notes.append(
            "compulsory-miss bytes unavailable: no run in this phase carried the token-count/"
            "experts-per-token/hit-rate summary line"
        )
    read_path = None
    exposed = None
    if have_bytes and have_disk and total_edisk > 0:
        read_path = compulsory_miss_bytes / 1e9 / total_edisk
        stall = total_edisk * (1.0 - overlap_fraction)
        if stall > 0:
            exposed = compulsory_miss_bytes / 1e9 / stall
    else:
        notes.append("read-path/exposed bandwidth unavailable: no expert-disk PROFILE time in this phase")
    if overlap_fraction == 0.0:
        notes.append(
            "exposed bandwidth assumes overlap_fraction=0 (fully-exposed, conservative default): "
            "this harness has no direct compute/IO-overlap timestamp instrumentation yet "
            "(that is a separate, not-yet-built probe -- see docs/performance-theory.json "
            "p3-unified-memory-overlap-ring); pass --overlap-fraction to override with a measured value"
        )
    notes.append(
        f"device throughput is a CONFIGURED REFERENCE CONSTANT ({device_bw_gbs:.1f} GB/s, "
        "the prefill-I/O study's raw-SSD sweep) -- this harness does not independently "
        "re-measure raw device bandwidth; pass --device-bw-gbs to override"
    )
    return BandwidthEstimate(
        compulsory_miss_bytes, device_bw_gbs, read_path, exposed,
        total_miss_experts if have_bytes else None, notes,
    )


# ---- per-kernel-class roofline table ----------------------------------------------------

@dataclass
class KernelClassRow:
    name: str
    calls: Optional[float] = None
    gpu_wall_s: Optional[float] = None
    kernel_s: Optional[float] = None
    command_gap_s: Optional[float] = None
    occupancy: Optional[float] = None
    bytes_moved: Optional[float] = None
    bytes_per_s: Optional[float] = None
    macs: Optional[float] = None
    mac_per_s: Optional[float] = None
    arithmetic_intensity: Optional[float] = None
    classification: str = "unknown"
    note: str = ""


def classify(occupancy: Optional[float], ai: Optional[float], ai_threshold: float = AI_THRESHOLD_DEFAULT):
    if occupancy is None:
        return "unknown", "no GPU-wall/kernel timing observed for this class in this phase"
    if occupancy < OCCUPANCY_SYNC_THRESHOLD:
        return (
            "sync-ish",
            f"occupancy proxy {occupancy:.2f} < {OCCUPANCY_SYNC_THRESHOLD:g}: command-gap "
            "(dispatch/scheduling) time exceeds actual kernel-execution time",
        )
    if ai is None:
        return "unknown", "occupancy is healthy but no arithmetic-intensity estimate is available for this class"
    if ai >= ai_threshold:
        return "compute-ish", f"arithmetic intensity {ai:.2f} MAC/B >= threshold {ai_threshold:g} MAC/B"
    return "bandwidth-ish", f"arithmetic intensity {ai:.2f} MAC/B < threshold {ai_threshold:g} MAC/B"


def _gpu_class_row(name: str, dicts: list, kernel_key: str = "kernel", gpu_wall_key: str = "gpu_wall"):
    calls = med([d.get("ok") for d in dicts]) if dicts else None
    gpu_wall = med([d.get(gpu_wall_key) for d in dicts]) if dicts else None
    kernel = med([d.get(kernel_key) for d in dicts]) if dicts else None
    occ = None
    gap = None
    if gpu_wall is not None and kernel is not None and gpu_wall > 0:
        occ = kernel / gpu_wall
        gap = gpu_wall - kernel
    return calls, gpu_wall, kernel, gap, occ


def build_kernel_class_table(
    stats_list,
    avg_context_tokens: int = DEFAULT_AVG_CONTEXT_TOKENS,
    decode_batch: int = DEFAULT_DECODE_BATCH,
    bytes_per_expert_mb: float = DEFAULT_BYTES_PER_EXPERT_MB,
    proj_i: int = DEFAULT_PROJ_I,
    proj_o: int = DEFAULT_PROJ_O,
    proj_bits: int = DEFAULT_PROJ_BITS,
    ai_threshold: float = AI_THRESHOLD_DEFAULT,
) -> list:
    rows = []

    # -- attention score/latent -------------------------------------------------------
    attn_dicts = [s.metal_attn for s in stats_list if s.metal_attn]
    calls, gpu_wall, kernel, gap, occ = _gpu_class_row("attn", attn_dicts)
    macs = bytes_moved = mac_s = bytes_s = ai = None
    note = (
        "GPU path (METAL-ATTN). CAVEAT: ili_metal_layer_decode fuses attention-score/latent "
        "with the router AND shared-expert projections into ONE command buffer during decode "
        "-- this run's GPU-wall/kernel figures are that FUSED total whenever any decode steps "
        "ran, not pure attention time; ili_metal_attn_prefill (prefill only) is pure attention "
        "but the two accumulate into the same counters, so they cannot be separated post hoc "
        "with today's instrumentation. Treat this row as 'attention (+ router + shared-expert "
        "projections during decode)', not isolated attention."
    )
    if kernel and kernel > 0:
        macs = decode_batch * HEADS * avg_context_tokens * (KV_LORA + ROPE_DIM + KV_LORA)
        bytes_moved = decode_batch * avg_context_tokens * (KV_LORA + ROPE_DIM) * 4
        mac_s = macs / kernel
        bytes_s = bytes_moved / kernel
        ai = macs / bytes_moved if bytes_moved else None
    cls, cls_note = classify(occ, ai, ai_threshold)
    rows.append(KernelClassRow(
        "attention score/latent", calls, gpu_wall, kernel, gap, occ,
        bytes_moved, bytes_s, macs, mac_s, ai, cls, note + " | " + cls_note,
    ))
    if not attn_dicts:
        rows[-1].note = (
            "no METAL-ATTN line observed (Metal disabled or CPU fallback) -- falling back to the "
            "CPU-side PROFILE 'attention' field is possible via --cpu-fallback but carries no "
            "GPU occupancy/command-gap breakdown"
        )

    # -- MoE GEMV ----------------------------------------------------------------------
    moe_dicts = [s.metal_moe for s in stats_list if s.metal_moe]
    calls, gpu_wall, kernel, gap, occ = _gpu_class_row("moe", moe_dicts)
    macs = bytes_moved = mac_s = bytes_s = ai = None
    total_experts = med([d.get("experts") for d in moe_dicts]) if moe_dicts else None
    if kernel and kernel > 0 and total_experts:
        macs = total_experts * 3 * MOE_D * MOE_I    # gate + up + down GEMV per expert
        bytes_moved = total_experts * bytes_per_expert_mb * 1e6
        mac_s = macs / kernel
        bytes_s = bytes_moved / kernel
        ai = macs / bytes_moved if bytes_moved else None
    cls, cls_note = classify(occ, ai, ai_threshold)
    setup = med([d.get("setup") for d in moe_dicts]) if moe_dicts else None
    scatter = med([d.get("scatter") for d in moe_dicts]) if moe_dicts else None
    note = (
        f"GPU path (METAL:); setup(CPU encode) median {setup:.3f}s, scatter(CPU accumulate) "
        f"median {scatter:.3f}s are OUTSIDE gpu-wall (paid before commit / after completion) "
        "-- a class dominated by setup+scatter rather than kernel time is CPU-bound, not "
        "captured by the occupancy proxy below, which only covers time inside gpu-wall."
        if setup is not None and scatter is not None else
        "GPU path (METAL:), or no MoE block observed in this phase."
    )
    rows.append(KernelClassRow(
        "MoE GEMV", calls, gpu_wall, kernel, gap, occ,
        bytes_moved, bytes_s, macs, mac_s, ai, cls, note + " | " + cls_note,
    ))

    # -- router --------------------------------------------------------------------
    rows.append(KernelClassRow(
        "router", None, None, None, None, None, None, None, None, None, None,
        "unknown (not independently observable)",
        "r_router/r_top8 dispatch only inside ili_metal_layer_decode's single fused decode-"
        "layer command buffer (see attention score/latent's caveat above) -- there is no "
        "separate command buffer or GPUStartTime/GPUEndTime pair for the router alone with "
        "today's instrumentation. Splitting it out would need per-encoder counter sample "
        "buffers (a materially bigger backend_metal.mm change than this harness's counter "
        "plumbing scope), so this row is reported as unmeasured rather than fabricated.",
    ))

    # -- projections -----------------------------------------------------------------
    gemm_dicts = [s.metal_gemm for s in stats_list if s.metal_gemm]
    calls, gpu_wall, kernel, gap, occ = _gpu_class_row("gemm", gemm_dicts)
    macs = bytes_moved = mac_s = bytes_s = ai = None
    if kernel and kernel > 0 and calls:
        macs = calls * decode_batch * proj_i * proj_o
        bytes_moved = calls * proj_o * proj_i * proj_bits / 8.0
        mac_s = macs / kernel
        bytes_s = bytes_moved / kernel
        ai = macs / bytes_moved if bytes_moved else None
    cls, cls_note = classify(occ, ai, ai_threshold)
    note = (
        "GPU path (METAL-GEMM, added by this harness -- see report header). Only fires for "
        "matmul_qt's large-row-batch path (S >= ILI_METAL_GEMM_MIN, default 16: prefill "
        "o_proj/dense-MLP/kv_b-reconstruction/logits) -- short decode-only runs (S=1 per "
        "step) legitimately show zero calls here, since small-S linear layers stay on CPU/"
        "NEON by design (matmul_qt's own comment: 'small-S decode matmuls stay on CPU "
        "(NEON wins)'). Shape (I,O) is a REPRESENTATIVE default, not read per-call from the "
        "log; override with --proj-i/--proj-o if the run's actual dominant projection shape "
        "is known."
    )
    if not gemm_dicts:
        note = (
            "no METAL-GEMM calls observed -- either Metal was disabled, every call in this "
            "phase had S < ILI_METAL_GEMM_MIN (decode-only runs, expected), or this engine "
            "build predates this harness's counter addition."
        )
    rows.append(KernelClassRow(
        "projections", calls, gpu_wall, kernel, gap, occ,
        bytes_moved, bytes_s, macs, mac_s, ai, cls, note + " | " + cls_note,
    ))

    return rows


# ---- markdown report -------------------------------------------------------------------

def _fmt(v, spec="{:.3g}"):
    return "n/a" if v is None else spec.format(v)


def build_report(
    manifest: dict,
    result_dir: Path,
    bytes_per_expert_mb: float = DEFAULT_BYTES_PER_EXPERT_MB,
    device_bw_gbs: float = DEFAULT_DEVICE_BW_GBS,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
    ai_threshold: float = AI_THRESHOLD_DEFAULT,
    avg_context_tokens: int = DEFAULT_AVG_CONTEXT_TOKENS,
    decode_batch: int = DEFAULT_DECODE_BATCH,
    proj_i: int = DEFAULT_PROJ_I,
    proj_o: int = DEFAULT_PROJ_O,
) -> str:
    runs = manifest["runs"]
    for r in runs:
        r["stats"] = parse_engine_log(result_dir / r["log"])

    by_phase = {"cold": [], "steady": []}
    for r in runs:
        by_phase.setdefault(r["phase"], []).append(r)

    telemetry_rows = {}
    for phase, relpath in (manifest.get("telemetry") or {}).items():
        p = result_dir / relpath
        if p.exists():
            telemetry_rows[phase] = parse_telemetry_csv(p)

    lines = [
        "# Dual roofline report",
        "",
        f"attempt_id: `{manifest.get('attempt_id', 'n/a')}` | mock: {manifest.get('mock', False)} | "
        f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Method: docs/performance-theory.json `p2-m5-gpu-tensor-path-probe` (two rooflines: "
        "cold first-5-min burst vs power-constrained 30-60min steady-state; three bandwidth "
        "definitions: device throughput, read-path throughput, EXPOSED critical-path "
        "bandwidth). **Endpoint/promotion claims must use the STEADY-STATE median below, "
        "never the cold number** -- cold is reported for context and thermal-derate only.",
        "",
    ]

    # ---- phase throughput summary + thermal derate ----
    lines += ["## Phase summary (tok/s, hit rate)", "", "| phase | runs | median tok/s | min | max | median hit% |", "|---|---|---|---|---|---|"]
    phase_tok_s = {}
    for phase in ("cold", "steady"):
        rs = by_phase.get(phase, [])
        tok_s_vals = [r["stats"].tok_s for r in rs if r["stats"].tok_s is not None]
        hit_vals = [r["stats"].hit_pct for r in rs if r["stats"].hit_pct is not None]
        phase_tok_s[phase] = tok_s_vals
        lines.append(
            f"| {phase} | {len(rs)} | {_fmt(med(tok_s_vals))} | "
            f"{_fmt(min(tok_s_vals)) if tok_s_vals else 'n/a'} | "
            f"{_fmt(max(tok_s_vals)) if tok_s_vals else 'n/a'} | {_fmt(med(hit_vals))} |"
        )
    cold_med, steady_med = med(phase_tok_s.get("cold", [])), med(phase_tok_s.get("steady", []))
    lines += ["", "## Thermal derate estimate", ""]
    if cold_med and steady_med:
        derate = steady_med / cold_med
        lines.append(
            f"steady median tok/s / cold median tok/s = {steady_med:.4g} / {cold_med:.4g} = "
            f"**{derate:.3f}** (1.0 = no derate; <1.0 = steady-state throughput lower than the "
            "cold/burst window, consistent with thermal/power throttling; this is `theta_thermal`'s "
            "reciprocal in the t_token model, docs/performance-theory.json "
            "topp-adaptive-expert-routing notes -- reported here exactly as this harness's own "
            "spec defines it: steady/cold, not cold/steady)."
        )
    else:
        lines.append("not computable: need at least one tok/s reading in both the cold and steady phases.")

    # ---- telemetry ----
    lines += ["", "## Telemetry (accessible proxies -- see Method & limits)", "",
              "| phase | samples | thermal speed-limit % (median) | cpu idle % (median) | disk MB/s (median) |",
              "|---|---|---|---|---|"]
    for phase in ("cold", "steady"):
        rows = telemetry_rows.get(phase, [])
        summ = telemetry_phase_summary(rows, phase) if rows else None
        if summ:
            lines.append(
                f"| {phase} | {summ['samples']} | {_fmt(summ['thermal_speed_limit_pct']['median'])} | "
                f"{_fmt(summ['cpu_idle_pct']['median'])} | {_fmt(summ['disk_mb_s']['median'])} |"
            )
        else:
            lines.append(f"| {phase} | 0 | n/a | n/a | n/a |")

    # ---- bandwidths (three definitions), per phase ----
    lines += ["", "## Bandwidth (three definitions)", ""]
    for phase in ("cold", "steady"):
        rs = [r["stats"] for r in by_phase.get(phase, [])]
        if not rs:
            continue
        bw = estimate_bandwidths(rs, bytes_per_expert_mb, device_bw_gbs, overlap_fraction)
        lines += [f"### {phase}", ""]
        lines.append(f"- device throughput (reference constant): **{bw.device_throughput_gbs:.2f} GB/s**")
        lines.append(f"- read-path throughput (compulsory-miss bytes / expert-disk wall time): "
                     f"**{_fmt(bw.read_path_gbs, '{:.3f}')} GB/s**")
        lines.append(f"- EXPOSED critical-path bandwidth (compulsory-miss bytes / main-thread "
                     f"storage stall; the only one that belongs in a tok/s model): "
                     f"**{_fmt(bw.exposed_gbs, '{:.3f}')} GB/s**")
        lines.append(f"- compulsory miss bytes (modeled): {_fmt(bw.compulsory_miss_bytes, '{:.4g}')} "
                     f"(~{_fmt(bw.miss_experts, '{:.1f}')} missed expert loads)")
        for n in bw.notes:
            lines.append(f"  - *{n}*")
        lines.append("")

    # ---- per-kernel-class table, cold vs steady ----
    lines += ["## Per-kernel-class roofline table", ""]
    for phase in ("cold", "steady"):
        rs = [r["stats"] for r in by_phase.get(phase, [])]
        if not rs:
            continue
        rows = build_kernel_class_table(
            rs, avg_context_tokens, decode_batch, bytes_per_expert_mb, proj_i, proj_o,
            DEFAULT_PROJ_BITS, ai_threshold,
        )
        lines += [f"### {phase}", "",
                  "| kernel class | calls | gpu-wall (s) | kernel (s) | command-gap (s) | "
                  "occupancy | MAC/s | bytes/s | AI (MAC/B) | classification |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(
                f"| {row.name} | {_fmt(row.calls, '{:.0f}')} | {_fmt(row.gpu_wall_s)} | "
                f"{_fmt(row.kernel_s)} | {_fmt(row.command_gap_s)} | {_fmt(row.occupancy, '{:.2f}')} | "
                f"{_fmt(row.mac_per_s, '{:.3g}')} | {_fmt(row.bytes_per_s, '{:.3g}')} | "
                f"{_fmt(row.arithmetic_intensity, '{:.2f}')} | {row.classification} |"
            )
        lines.append("")
        lines.append("Notes:")
        for row in rows:
            lines.append(f"- **{row.name}**: {row.note}")
        lines.append("")

    # ---- method & limits ----
    lines += [
        "## Method & limits",
        "",
        "- No chip-peak GFLOP/s or GB/s figure is used anywhere in this report. The "
        "compute-ish/bandwidth-ish/sync-ish classification is a *relative* heuristic: "
        f"occupancy = kernel-time/gpu-wall-time < {OCCUPANCY_SYNC_THRESHOLD:g} => sync-ish; "
        f"else arithmetic intensity vs a configurable threshold (default {AI_THRESHOLD_DEFAULT:g} "
        "MAC/byte) => compute-ish or bandwidth-ish. This prioritizes where to look next; it "
        "does not certify hardware utilization against any measured hardware ceiling.",
        "- MAC/s, bytes/s, and arithmetic-intensity figures use GLM-5.2's compiled-in shape "
        "constants (hidden=6144, heads=64, kv_lora=512, expert D=6144/I=2048) at a single "
        "REPRESENTATIVE context length and batch size (`--avg-context-tokens`, "
        "`--decode-batch`), not shapes read per-call from the log -- treat them as "
        "order-of-magnitude, not hardware-counter-precise.",
        "- 'attention score/latent' and 'router' overlap: ili_metal_layer_decode fuses "
        "attention + router + shared-expert projections into ONE command buffer during "
        "decode, so the attention row's GPU time is that fused total whenever decode ran, "
        "and 'router' has no independent timer at all (reported as unmeasured, not "
        "estimated).",
        "- 'projections' only observes matmul_qt's large-row-batch path (S >= "
        "ILI_METAL_GEMM_MIN); short decode-only runs legitimately show zero calls.",
        "- Telemetry is limited to accessible tools on this machine: `pmset -g therm` "
        "(thermal/throttle proxy), `iostat` (disk MB/s, CPU idle %), `vm_stat` (paging). "
        "`powermetrics` is NOT available (no passwordless sudo beyond `/usr/sbin/purge`), "
        "so GPU frequency/power, memory-controller bandwidth, SSD temperature, and fan "
        "state are NOT measured by this harness -- thermal state is inferred only from "
        "`pmset -g therm`'s CPU_Speed_Limit and the steady-vs-cold tok/s ratio above.",
        "- device throughput is a configured reference constant (prior measurement, "
        "the prefill-I/O study), not re-measured by this harness run.",
        "- exposed bandwidth assumes zero compute/IO overlap by default (`--overlap-fraction 0`); "
        "override if a measured overlap fraction is available.",
        "",
    ]

    return "\n".join(lines) + "\n"


def run_report(args) -> None:
    result_dir = Path(args.result_dir)
    manifest = json.loads((result_dir / "manifest.json").read_text())
    report = build_report(
        manifest, result_dir,
        bytes_per_expert_mb=args.bytes_per_expert_mb,
        device_bw_gbs=args.device_bw_gbs,
        overlap_fraction=args.overlap_fraction,
        ai_threshold=args.ai_threshold,
        avg_context_tokens=args.avg_context_tokens,
        decode_batch=args.decode_batch,
        proj_i=args.proj_i,
        proj_o=args.proj_o,
    )
    Path(args.out).write_text(report)
    print(f"wrote {args.out}")


# ---- mock-log: synthetic engine output for --mock end-to-end proof --------------------

def gen_mock_log(
    phase: str,
    seed: int = 0,
    tokens: int = 128,
    tok_s_base: float = 1.45,
    hit_pct_base: float = 68.0,
    metal: bool = True,
    include_gemm: bool = True,
) -> str:
    """Emit one synthetic engine log with the EXACT line formats glm.c/backend_metal.mm
    print in real runs (see parse_engine_log's regexes), so --mock proves the harness+report
    pipeline end to end without a real model or GPU. `steady` is modeled with a mild thermal
    derate (slightly lower tok/s, slightly worse occupancy) relative to `cold`, so the
    cold-vs-steady comparison and thermal-derate math have something real to show."""
    rng = random.Random(seed)
    derate = 0.90 if phase == "steady" else 1.0
    tok_s = tok_s_base * derate * rng.uniform(0.97, 1.03)
    wall_s = tokens / tok_s
    hit_pct = hit_pct_base * (0.98 if phase == "steady" else 1.0) * rng.uniform(0.99, 1.01)
    experts_per_token = 8.0 * rng.uniform(0.95, 1.05)
    rss_gb = 110.0 + rng.uniform(-1, 1)

    t_edisk = wall_s * 0.42 * rng.uniform(0.9, 1.1)
    t_ematmul = wall_s * 0.20 * rng.uniform(0.9, 1.1)
    t_attn = wall_s * 0.18 * rng.uniform(0.9, 1.1)
    t_kvb = t_attn * 0.1
    t_head = wall_s * 0.03 * rng.uniform(0.9, 1.1)
    accounted = t_edisk + t_ematmul + t_attn + t_head
    t_other = max(0.001, wall_s - accounted)

    lines = [
        f"[m5max] RAM=114GB DRAFT=0 PIPE=1/8 mock-log phase={phase}",
        f"\n---\n{tokens} tokens in {wall_s:.2f}s ({tok_s:.2f} tok/s) | expert hit rate {hit_pct:.1f}% | RSS {rss_gb:.2f} GB",
        f"experts loaded/token: {experts_per_token:.1f} (per-layer 1.07 across 75; baseline topk=8) | TOPK=8 TOPP=0.00",
        f"PROFILE: expert-disk {t_edisk:.3f}s | expert-matmul {t_ematmul:.3f}s | attention {t_attn:.3f}s "
        f"(including kvb {t_kvb:.3f}s) | lm_head {t_head:.3f}s | other {t_other:.3f}s",
    ]
    if metal:
        occ_derate = 0.93 if phase == "steady" else 1.0
        attn_gpu_wall = t_attn * 1.05
        attn_kernel = attn_gpu_wall * 0.7 * occ_derate * rng.uniform(0.95, 1.05)
        cpu_sched = (attn_gpu_wall - attn_kernel) * 0.4
        gpu_sched = (attn_gpu_wall - attn_kernel) * 0.6
        lines.append(
            f"METAL-ATTN: layer GPU {tokens} | gpu-wall {attn_gpu_wall:.2f}s "
            f"(kernel {attn_kernel:.2f}s | cpu-sched {cpu_sched:.2f}s gpu-sched {gpu_sched:.2f}s)"
        )
        moe_experts = int(tokens * experts_per_token * (hit_pct / 100.0))
        moe_setup = t_ematmul * 0.15
        moe_gpu_wall = t_ematmul * 0.75
        moe_kernel = moe_gpu_wall * 0.8 * occ_derate * rng.uniform(0.95, 1.05)
        moe_scatter = t_ematmul * 0.1
        lines.append(
            f"METAL: blocchi GPU {tokens} | fallback CPU 0 | expert su GPU {moe_experts} | "
            f"setup {moe_setup:.2f}s gpu-wall {moe_gpu_wall:.2f}s (kernel {moe_kernel:.2f}s) scatter {moe_scatter:.2f}s"
        )
        if include_gemm:
            gemm_calls = max(1, tokens // 32)
            gemm_gpu_wall = t_head * 2.0 * rng.uniform(0.9, 1.1)
            gemm_kernel = gemm_gpu_wall * 0.6 * occ_derate * rng.uniform(0.95, 1.05)
            lines.append(
                f"METAL-GEMM: calls {gemm_calls} | gpu-wall {gemm_gpu_wall:.2f}s (kernel {gemm_kernel:.2f}s)"
            )
    return "\n".join(lines) + "\n"


def run_mock_log(args) -> None:
    text = gen_mock_log(
        phase=args.phase, seed=args.seed, tokens=args.tokens, tok_s_base=args.tok_s_base,
        hit_pct_base=args.hit_pct_base, metal=bool(args.metal), include_gemm=bool(args.include_gemm),
    )
    Path(args.out).write_text(text)
    print(f"wrote {args.out}")


# ---- CLI --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("report", help="parse a roofline_run.sh result dir into a markdown report")
    rep.add_argument("--result-dir", required=True)
    rep.add_argument("--out", required=True)
    rep.add_argument("--bytes-per-expert-mb", type=float, default=DEFAULT_BYTES_PER_EXPERT_MB)
    rep.add_argument("--device-bw-gbs", type=float, default=DEFAULT_DEVICE_BW_GBS)
    rep.add_argument("--overlap-fraction", type=float, default=DEFAULT_OVERLAP_FRACTION)
    rep.add_argument("--ai-threshold", type=float, default=AI_THRESHOLD_DEFAULT)
    rep.add_argument("--avg-context-tokens", type=int, default=DEFAULT_AVG_CONTEXT_TOKENS)
    rep.add_argument("--decode-batch", type=int, default=DEFAULT_DECODE_BATCH)
    rep.add_argument("--proj-i", type=int, default=DEFAULT_PROJ_I)
    rep.add_argument("--proj-o", type=int, default=DEFAULT_PROJ_O)

    mock = sub.add_parser("mock-log", help="emit one synthetic engine log for --mock dry runs")
    mock.add_argument("--out", required=True)
    mock.add_argument("--phase", choices=("cold", "steady"), required=True)
    mock.add_argument("--seed", type=int, default=0)
    mock.add_argument("--tokens", type=int, default=128)
    mock.add_argument("--tok-s-base", type=float, default=1.45)
    mock.add_argument("--hit-pct-base", type=float, default=68.0)
    mock.add_argument("--metal", type=int, choices=(0, 1), default=1)
    mock.add_argument("--include-gemm", type=int, choices=(0, 1), default=1)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "report":
        run_report(args)
    elif args.command == "mock-log":
        run_mock_log(args)


if __name__ == "__main__":
    main()
