#!/usr/bin/env python3
"""Step-0 diagnostic analyzer (#1438 deep-offload reconciliation).

Parses one `ili run` / `./glm` decode log that was produced by a build
containing the [IOKIND] instrumentation (glm.c: io_kind_done/io_kind_latency_totals,
profile_print's "[IOKIND] weight: ... | scale: ..." line) and computes, HONESTLY
and with overlap caveats stated inline (never silently dropped):

  s            = disk-I/O share of the run's wall-clock (TWO variants -- see below)
  g_residual   = coalescing gain already captured (weight-only vs weight+scale)
  avg op cost  = per-pread average latency, weight vs scale
  upside       = s * (1 - 1/g_residual): implied wall-clock win from merging the
                 3 scale preads into the weight pread (i.e. eliminating the
                 scale-pread latency-sum entirely), IF that time were fully
                 exposed (see caveats -- this is explicitly an upper bound).

Does not attempt statistical inference (n=1 run) -- this is a Step-0 measured-vs-
estimated reconciliation, not a certification. Every number below is raw or a
simple ratio of raw counters; nothing is smoothed, bootstrapped, or fit.
"""
import re
import sys
import json
from pathlib import Path

IOKIND_RE = re.compile(
    r"\[IOKIND\] weight: n=(\d+) bytes=(\d+) lat_sum_s=([\d.]+) \| "
    r"scale: n=(\d+) bytes=(\d+) lat_sum_s=([\d.]+)"
)
PROMPT_RE = re.compile(r"prompt: (\d+) tokens \| generating up to (\d+)")
DECODE_WALL_RE = re.compile(r"(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)")
PROFILE_RE = re.compile(
    r"PROFILE: expert-disk ([\d.]+)s \| expert-matmul ([\d.]+)s \| attention ([\d.]+)s "
    r"\(including kvb ([\d.]+)s\) \| lm_head ([\d.]+)s \| other ([\d.]+)s"
)
STALL_RE = re.compile(
    r"STALL-EXPOSED: ([\d.]+)s \(consumer-blocked critical-path only, excludes overlapped "
    r"service\) \| pipe-waits (\d+) blocked (\d+) \(occupancy ([\d.]+)%\)"
)
STALL_TOKEN_RE = re.compile(r"STALL-EXPOSED/TOKEN: ([\d.]+) ms/token \(n=(\d+) decode tokens\)")
IOBYTES_RE = re.compile(
    r"IO-BYTES: requested (\d+) \| read (\d+) \| reads attempted (\d+) completed (\d+) \| "
    r"hits (\d+) misses (\d+) \(([\d.]+)% hit\)"
)
IOLAT_RE = re.compile(
    r"IO-LATENCY: read-completion p50 ([\d.]+)ms p95 ([\d.]+)ms p99 ([\d.]+)ms \(n=(\d+) samples\)"
)
WALLSUM_RE = re.compile(
    r"WALL-SUM: compute ([\d.]+)s \+ exposed-stall ([\d.]+)s \+ other ([\d.]+)s = wall ([\d.]+)s "
    r"\| residual\(other\) ([\d.]+)% of wall"
)
EXPERTS_PER_TOK_RE = re.compile(r"experts loaded/token: ([\d.]+)")


def parse_log(text: str) -> dict:
    out = {}
    if (m := PROMPT_RE.search(text)):
        out["prompt_tokens"] = int(m.group(1))
        out["ngen_requested"] = int(m.group(2))
    if (m := DECODE_WALL_RE.search(text)):
        out["tokens_produced"] = int(m.group(1))
        out["wall_s"] = float(m.group(2))
        out["tok_per_s"] = float(m.group(3))
    if (m := PROFILE_RE.search(text)):
        out["t_edisk"] = float(m.group(1))
        out["t_emm"] = float(m.group(2))
        out["t_attn"] = float(m.group(3))
        out["t_kvb"] = float(m.group(4))
        out["t_head"] = float(m.group(5))
        out["t_other"] = float(m.group(6))
    if (m := STALL_RE.search(text)):
        out["t_stall_exposed"] = float(m.group(1))
        out["pipe_waits"] = int(m.group(2))
        out["pipe_waits_blocked"] = int(m.group(3))
        out["pipe_occupancy_pct"] = float(m.group(4))
    if (m := STALL_TOKEN_RE.search(text)):
        out["stall_ms_per_token"] = float(m.group(1))
        out["stall_token_n"] = int(m.group(2))
    if (m := IOBYTES_RE.search(text)):
        out["io_bytes_requested"] = int(m.group(1))
        out["io_bytes_read"] = int(m.group(2))
        out["io_reads_attempted"] = int(m.group(3))
        out["io_reads_completed"] = int(m.group(4))
        out["expert_hits"] = int(m.group(5))
        out["expert_misses"] = int(m.group(6))
        out["expert_hitpct"] = float(m.group(7))
    if (m := IOLAT_RE.search(text)):
        out["iolat_p50_ms"] = float(m.group(1))
        out["iolat_p95_ms"] = float(m.group(2))
        out["iolat_p99_ms"] = float(m.group(3))
        out["iolat_n"] = int(m.group(4))
    if (m := IOKIND_RE.search(text)):
        out["weight_n"] = int(m.group(1))
        out["weight_bytes"] = int(m.group(2))
        out["weight_lat_sum_s"] = float(m.group(3))
        out["scale_n"] = int(m.group(4))
        out["scale_bytes"] = int(m.group(5))
        out["scale_lat_sum_s"] = float(m.group(6))
    if (m := WALLSUM_RE.search(text)):
        out["wallsum_compute_s"] = float(m.group(1))
        out["wallsum_exposed_stall_s"] = float(m.group(2))
        out["wallsum_other_s"] = float(m.group(3))
        out["wallsum_wall_s"] = float(m.group(4))
        out["wallsum_other_pct"] = float(m.group(5))
    if (m := EXPERTS_PER_TOK_RE.search(text)):
        out["experts_per_token"] = float(m.group(1))
    return out


def compute(d: dict) -> dict:
    r = {}
    if "weight_n" not in d:
        r["error"] = "no [IOKIND] line found in this log -- was it built with the instrumentation?"
        return r

    w_n, w_bytes, w_lat = d["weight_n"], d["weight_bytes"], d["weight_lat_sum_s"]
    s_n, s_bytes, s_lat = d["scale_n"], d["scale_bytes"], d["scale_lat_sum_s"]
    total_lat = w_lat + s_lat

    r["weight_n"], r["weight_bytes"], r["weight_lat_sum_s"] = w_n, w_bytes, w_lat
    r["scale_n"], r["scale_bytes"], r["scale_lat_sum_s"] = s_n, s_bytes, s_lat
    r["total_iokind_lat_sum_s"] = total_lat
    r["scale_bytes_pct_of_total"] = 100.0 * s_bytes / (w_bytes + s_bytes) if (w_bytes + s_bytes) else None
    r["scale_lat_pct_of_total_iokind_lat"] = 100.0 * s_lat / total_lat if total_lat else None

    # sanity cross-check against the pre-existing blended counters
    if "io_bytes_read" in d:
        r["sanity_bytes_match"] = (w_bytes + s_bytes) == d["io_bytes_read"]
        r["sanity_bytes_delta"] = (w_bytes + s_bytes) - d["io_bytes_read"]
    if "io_reads_completed" in d:
        # 1 weight-event per completed expert-load when contig (expected for this
        # container); 3 scale-events per completed expert-load always.
        r["sanity_weight_n_vs_reads_completed"] = (w_n, d["io_reads_completed"])
        r["sanity_scale_n_vs_3x_reads_completed"] = (s_n, 3 * d["io_reads_completed"])

    # avg per-op latency -- THE number that answers "is the scale pread ~3-10us
    # fd-cached, or 40-100us thread-pool overhead" from the scoping pass.
    r["avg_weight_pread_us"] = 1e6 * w_lat / w_n if w_n else None
    r["avg_scale_pread_us"] = 1e6 * s_lat / s_n if s_n else None

    # g_residual: coalescing gain ALREADY captured by merging gate+up+down into one
    # weight pread, expressed as (what total I/O latency-sum would be with only the
    # scale-preads left unmerged) / (what it would be if scale were ALSO merged into
    # the single weight pread, i.e. weight-only). This is the SAME quantity the task
    # spec defines: g_residual = (weight_lat + scale_lat) / weight_lat.
    r["g_residual"] = total_lat / w_lat if w_lat else None

    # s: disk-I/O share of decode wall-clock. TWO variants, both reported --
    # the raw latency-sum share is NOT the same claim as "s fraction of wall-clock
    # was spent stalled on disk": expert_load is OMP-parallel (PIPE_WORKERS threads
    # can each be blocked in a pread concurrently), so summed pread latency can
    # legitimately exceed wall-clock time with no contradiction (that IS overlap,
    # not double-counting a bug). See STALL-EXPOSED (consumer-blocked critical-path
    # time only) for the non-overlapping variant.
    wall = d.get("wall_s")
    if wall:
        r["s_raw_latsum_over_wall"] = total_lat / wall
    if "t_stall_exposed" in d:
        r["s_stall_exposed_over_wall"] = d["t_stall_exposed"] / wall if wall else None
        # what FRACTION of the (non-overlapping) exposed stall is scale-attributable,
        # if we (conservatively, an approximation -- see caveat below) assume exposed
        # stall splits between weight/scale in the same proportion as raw latency-sum:
        if total_lat:
            r["scale_share_of_exposed_stall_s_APPROX"] = d["t_stall_exposed"] * (s_lat / total_lat)

    # implied upside from eliminating the scale-preads entirely (merging into the
    # coalesced weight pread), stated as a fraction of decode wall-clock, using
    # EACH s variant so the reader sees both the optimistic (raw latsum) and the
    # conservative (stall-exposed-only) framing.
    if r.get("g_residual"):
        upside_factor = 1.0 - 1.0 / r["g_residual"]
        r["upside_factor_1_minus_1_over_g"] = upside_factor
        if "s_raw_latsum_over_wall" in r:
            r["implied_upside_pct_of_wall_OPTIMISTIC"] = 100.0 * r["s_raw_latsum_over_wall"] * upside_factor
        if r.get("s_stall_exposed_over_wall") is not None:
            r["implied_upside_pct_of_wall_CONSERVATIVE"] = 100.0 * r["s_stall_exposed_over_wall"] * upside_factor

    # Promotion-bar verdict is deliberately based on the CONSERVATIVE (STALL-EXPOSED)
    # framing, NOT the optimistic raw-latency-sum one: the optimistic variant's own
    # `s` input (s_raw_latsum_over_wall) can and does exceed 1.0 under OMP overlap
    # (proof of overlap, not a literal wall-clock fraction), so treating it as "% of
    # wall-clock saved" overstates the case. The conservative variant uses
    # STALL-EXPOSED, the engine's own non-overlapping consumer-blocked-critical-path
    # counter -- the defensible number for a go/no-go call.
    r["clears_5pct_promotion_bar"] = None
    conservative = r.get("implied_upside_pct_of_wall_CONSERVATIVE")
    if conservative is not None:
        r["clears_5pct_promotion_bar"] = conservative >= 5.0
    r["clears_5pct_promotion_bar_basis"] = "implied_upside_pct_of_wall_CONSERVATIVE (STALL-EXPOSED-based s)"

    return r


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_iokind.py LOGFILE [LOGFILE...]", file=sys.stderr)
        sys.exit(2)
    for path in sys.argv[1:]:
        text = Path(path).read_text(errors="replace")
        parsed = parse_log(text)
        result = compute(parsed)
        print(f"=== {path} ===")
        print(json.dumps({"parsed": parsed, "computed": result}, indent=2))
        print()


if __name__ == "__main__":
    main()
