#!/usr/bin/env python3
"""Entropy-aware trace replay -- revised-sequence step 1.

OFFLINE ONLY: no engine runs, no shard reads. Every input is a committed
trace/usage/manifest/census file already on disk.

Extends the byte-capacity-aware replay
(`c/bench-m5max/offline-replay-20260715/`, tool `simulate_bytes_cache.py` --
imported here, never modified) with lossless entropy-coded int4 sizing,
using the per-(layer-band, projection) stored-ratio distributions measured
by the INT4 entropy census (`c/bench-m5max/new-math-20260715/`, tool
`measure_expert_entropy.py`, 256 of the 19,200 routed experts sampled).

SIZE ASSIGNMENT (the part this file unit-tests): the census sampled 256 of
19,200 routed experts. Every expert priced here gets a PREDICTED stored
size, never a measured one: raw per-projection packed bytes x the sampled
band x projection stored-ratio (median, or p90 as a pessimistic sensitivity)
+ raw (uncompressible) per-row F32 scale bytes, summed over gate/up/down --
mirroring the census's own per-expert formula (see measure_expert_entropy.py
docstring: per-EXPERT ratio = (3 x compressed payload + raw scales) /
(3 x raw payload + raw scales)). All experts in the same layer band get the
SAME predicted size (the census conditions on band x projection only, not
per-expert identity) -- a 19,200-entry table collapses to 3 distinct
predicted values (early/mid/late) per sizing variant.

SIX CASES (same traces, pins, budgets, conventions as the A/B/C/D/E replay;
"A" = case 1, "D" = case 4, re-reported here for alignment):

  1  raw uniform int4                                     (= layout A)
  2  entropy-coded int4, ALL experts, CURRENT slot accounting
  3  entropy-coded int4, ALL experts, byte-budget accounting
  4  repaired int2 cold tier + RAW int4 protected tier     (= layout D)
  5  repaired int2 cold tier + ENTROPY-CODED int4 protected tier (hybrid)
  6  repaired int2 cold tier ALSO entropy-coded -- SENSITIVITY ROWS ONLY:
     6a conservative (int2 codes assumed to barely compress: ratio 0.95)
     6b optimistic   (ratio 0.85)
     (protected tier stays at case 5's median entropy-int4 size; int2's own
     entropy is an ASSUMPTION, never measured -- the census sampled int4)

Plus two more sensitivity axes, each reusing the same six-case machinery:

  - p90-pessimistic INT4 sizing for cases 2/3/5 (columns "-p90"): same
    band x projection conditioning, using the p90 ratio instead of the
    median.
  - mode-1.5 disk-only sizing for cases 2/5 (columns "-mode1.5"): compressed
    bytes are read from disk and expanded before residency, so pin/LRU
    capacity accounting uses RAW sizes (hit/miss classification is then
    IDENTICAL to the uncompressed baseline -- case 1 for case 2, case 4 for
    case 5) and only the bytes charged to a miss shrink.

CACHE-CAPACITY MODE ASSUMPTION (stated, not derived): the primary row of
every entropy-coded case (2, 3, 5, 6a, 6b) assumes MODE-2/3 -- compressed
bytes are what sit resident in the pin and per-layer LRU tiers, so entropy
coding buys extra effective cache capacity (matches the census's mode-2
"RAM-compressed" / mode-3 "GEMV consumes compressed tiles" framing). The
"-mode1.5" rows are the bounding disk-only sensitivity described above.

Reproduction (run from the repo root; cheap CPU-only trace arithmetic --
nice -n 19 (lowest priority), no I/O throttling needed):

  nice -n 19 python3 c/tools/entropy_aware_replay.py \\
      --outdir c/bench-m5max/entropy-replay-20260715

  # unit tests
  cd c && python3 -m unittest discover -s tests
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

GB = 1e9
_TOOLS = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reused machinery -- imported, never modified (ownership rule for this
# task). Loading bsim also loads simulate_m5max_cache as bsim.slotsim.
bsim = _load_module("simulate_bytes_cache", _TOOLS / "simulate_bytes_cache.py")
census_tool = _load_module("measure_expert_entropy", _TOOLS / "measure_expert_entropy.py")
qc = _load_module("quant_container", _TOOLS / "quant_container.py")

BANDS = census_tool.BANDS  # (("early",3,27), ("mid",28,52), ("late",53,77))
band_of = census_tool.band_of
BAND_NAMES = tuple(b[0] for b in BANDS)
PROJ_SHORT = tuple(p.split("_")[0] for p in qc.PROJS)  # ("gate", "up", "down")

# Plain repo-relative paths (matching simulate_bytes_cache.py /
# variable_size_cache_oracle.py convention) -- run this tool from the repo
# root, as documented above. Kept relative (not ROOT-anchored) so the
# committed config.json doesn't embed a machine-local absolute path.
DEFAULT_TRACES = [
    Path("c/bench-m5max/overnight-20260714-001239/routes-baseline.bin"),
    Path("c/bench-m5max/overnight-20260714-001239/routes-coding.bin"),
    Path("c/bench-m5max/overnight3-20260714-003224/routes-shotgun.bin"),
]
DEFAULT_USAGE = Path("c/bench-m5max/k6-matrix-20260714-090059/system/usage-snapshot/.fa_usage")
DEFAULT_MANIFEST = Path("c/bench-m5max/container-20260715/allocation-280gb.json")
DEFAULT_CENSUS = Path("c/bench-m5max/new-math-20260715/census-summary.json")
DEFAULT_OUTDIR = Path("c/bench-m5max/entropy-replay-20260715")


# ------------------------------------------------------- size assignment ----
def load_ratio_table(census_summary: dict, coder: str | None = None,
                      block: int | None = None):
    """-> (table, coder, block).

    table[(band, proj)] = {"p50", "p90", "p99", "n"} read from the census's
    percentiles map for the 3 layer bands x 3 projections (gate/up/down).
    coder/block default to the census's own reported best
    (headline.best_coder / headline.best_block_size).
    """
    headline = census_summary["headline"]
    coder = coder or headline["best_coder"]
    block = int(block if block is not None else headline["best_block_size"])
    pct = census_summary["percentiles"]
    table = {}
    for band in BAND_NAMES:
        for proj in PROJ_SHORT:
            key = f"{band}/{proj}/{coder}/{block}"
            if key not in pct:
                raise KeyError(f"census summary missing percentile key {key!r}")
            entry = pct[key]
            table[(band, proj)] = {
                "p50": entry["p50"],
                "p90": entry["p90"],
                "p99": entry.get("p99"),
                "n": entry.get("n"),
            }
    return table, coder, block


def routed_pairs(cfg: dict) -> list[tuple[int, int]]:
    return [(l, e) for l in range(cfg["first_dense"], cfg["n_layers"])
            for e in range(cfg["n_experts"])]


def predicted_band_bytes(cfg: dict, ratio_table: dict, stat: str = "p50") -> dict:
    """-> {band: predicted per-expert stored bytes} at bits=4.

    Combines the 3 projections' own raw packed bytes (shape-determined,
    identical for every expert regardless of band) with the (band,
    proj)-conditional stored ratio, plus raw (uncompressible) per-row F32
    scale bytes -- the census's own per-expert formula, applied predictively
    (from the sampled ratio distribution) instead of measured.
    """
    out = {}
    for band in BAND_NAMES:
        total = 0.0
        for proj_long in qc.PROJS:
            proj = proj_long.split("_")[0]
            O, I = qc.expert_shape(cfg, proj_long)
            nbytes = qc.packed_nbytes(O, I, 4)
            ratio = ratio_table[(band, proj)][stat]
            total += nbytes * ratio + O * 4
        out[band] = total
    return out


def assign_predicted_sizes(cfg: dict, band_bytes: dict) -> dict:
    """-> {(layer, expert): predicted stored bytes}, every routed expert
    implied by cfg (19,200 for the current model_config) priced from its
    layer band's predicted bytes, rounded to whole bytes.

    PREDICTED FROM SAMPLE: the census measured 256 of these routed experts;
    every other expert here inherits its band's distribution, not its own
    measurement.
    """
    rounded = {band: round(b) for band, b in band_bytes.items()}
    return {(l, e): rounded[band_of(l)] for (l, e) in routed_pairs(cfg)}


# ------------------------------------------------------------ case sizes ----
def uniform_sizes(cfg: dict, size) -> dict:
    return {p: size for p in routed_pairs(cfg)}


def hybrid_sizes(cfg: dict, forced_layers, cold_size, protected_sizes) -> dict:
    """-> {(layer, expert): bytes} over every routed pair.

    protected_sizes: a dict {(layer,expert): bytes} or a flat scalar, applied
    to pairs whose layer is in forced_layers (the protected/early tier).
    cold_size: a scalar applied to every other routed pair (the cold tier).
    """
    forced = set(forced_layers)
    out = {}
    is_dict = isinstance(protected_sizes, dict)
    for p in routed_pairs(cfg):
        if p[0] in forced:
            out[p] = protected_sizes[p] if is_dict else protected_sizes
        else:
            out[p] = cold_size
    return out


# --------------------------------------------------------- dual-size replay ----
def replay_dual(calls, token_of, resident_sizes, transfer_sizes, pinned,
                 lru_mode, lru_budget, page_bytes=0, layout="?", trace="?",
                 collect_m=False):
    """Generalizes simulate_bytes_cache.replay() by decoupling the byte value
    that governs RAM residency/eviction (resident_sizes) from the byte value
    charged to a disk miss (transfer_sizes).

    Passing the same dict for both reproduces bsim.replay() exactly
    (unit-tested below: test_replay_dual_matches_bsim_replay_when_dicts_equal).
    The mode-1.5 disk-only sensitivity passes RAW sizes as resident_sizes (no
    extra effective capacity from compression -- hit/miss classification
    matches the raw baseline byte-for-byte) and compressed sizes as
    transfer_sizes (the disk read itself is smaller). `pinned` must already
    be filled consistently with resident_sizes (pin_load's byte accounting
    is inherently a residency decision; a pinned expert is never re-read
    from disk, so it needs no separate transfer size).
    """
    lrus = defaultdict(lambda: bsim.LayerLRU(lru_mode, lru_budget))
    page = bsim.PageCache(page_bytes) if page_bytes > 0 else None
    res = bsim.ReplayResult(layout, trace, lru_mode, lru_budget,
                             len(pinned), sum(pinned.values()))
    n_tokens = max((t for t in token_of.values() if t is not None), default=-1) + 1
    res.n_tokens = n_tokens
    res.per_token_bytes = [0] * n_tokens
    res.per_token_misses = [0] * n_tokens

    for idx, call in enumerate(calls):
        layer = call.layer
        lru = lrus[layer]
        token = token_of[idx]
        decode = token is not None

        # phase 1: classify the whole batch union against call-entry state
        lru_hit_experts, missing, flags = [], [], []
        pin_h = lru_h = 0
        for expert in call.experts:
            if (layer, expert) in pinned:
                pin_h += 1
                flags.append(1)
            elif expert in lru:
                lru_h += 1
                lru_hit_experts.append(expert)
                flags.append(1)
            else:
                missing.append(expert)
                flags.append(0)

        # phase 2: apply recency updates and load the misses (router order)
        for expert in lru_hit_experts:
            lru.touch(expert)
        page_h = 0
        disk_b = 0
        for expert in missing:
            rsize = resident_sizes[(layer, expert)]
            tsize = transfer_sizes[(layer, expert)]
            if page is not None and page.hit((layer, expert)):
                page_h += 1
            else:
                disk_b += tsize
                if page is not None:
                    page.fill((layer, expert), rsize)
            lru.insert(expert, rsize)

        n = len(call.experts)
        res.requests += n
        res.pin_hits += pin_h
        res.lru_hits += lru_h
        res.page_hits += page_h
        res.disk_misses += len(missing) - page_h
        res.disk_bytes += disk_b
        if decode:
            res.dec_requests += n
            res.dec_pin_hits += pin_h
            res.dec_lru_hits += lru_h
            res.dec_page_hits += page_h
            res.dec_disk_misses += len(missing) - page_h
            res.dec_disk_bytes += disk_b
            res.per_token_bytes[token] += disk_b
            res.per_token_misses[token] += len(missing) - page_h
            if collect_m:
                res.call_m.append((token, layer, len(missing), disk_b, tuple(flags)))
    return res


# --------------------------------------------------------------- cases ----
@dataclass
class CaseSpec:
    name: str
    resident_sizes: dict
    transfer_sizes: dict
    lru_mode: str       # "slots" | "bytes"
    sizing_variant: str  # "raw" | "median" | "p90" | "mode1.5" | "int2-cons-0.95" | "int2-opt-0.85"
    desc: str
    collect_m: bool = False


def build_cases(cfg, forced_layers, int4_b, int2_b, entropy_median, entropy_p90,
                 int2_cons_ratio, int2_opt_ratio) -> list[CaseSpec]:
    case1 = uniform_sizes(cfg, int4_b)
    case2 = assign_predicted_sizes(cfg, entropy_median)
    case2_p90 = assign_predicted_sizes(cfg, entropy_p90)
    case4 = hybrid_sizes(cfg, forced_layers, int2_b, int4_b)
    case5 = hybrid_sizes(cfg, forced_layers, int2_b, case2)
    case5_p90 = hybrid_sizes(cfg, forced_layers, int2_b, case2_p90)
    int2_cons_bytes = round(int2_b * int2_cons_ratio)
    int2_opt_bytes = round(int2_b * int2_opt_ratio)
    case6a = hybrid_sizes(cfg, forced_layers, int2_cons_bytes, case2)
    case6b = hybrid_sizes(cfg, forced_layers, int2_opt_bytes, case2)

    return [
        CaseSpec("1", case1, case1, "slots", "raw",
                 "raw uniform int4 (= layout A)", collect_m=True),
        CaseSpec("2", case2, case2, "slots", "median",
                 "entropy-coded int4, ALL experts, CURRENT slot accounting "
                 "(mode-2/3: compressed-resident)", collect_m=True),
        CaseSpec("2-p90", case2_p90, case2_p90, "slots", "p90",
                 "case 2, p90-pessimistic sizing sensitivity"),
        CaseSpec("2-mode1.5", case1, case2, "slots", "mode1.5",
                 "case 2, disk-only variant: RAW resident sizes (= case 1's "
                 "classification), compressed transfer bytes"),
        CaseSpec("3", case2, case2, "bytes", "median",
                 "entropy-coded int4, ALL experts, byte-budget accounting "
                 "(mode-2/3: compressed-resident)", collect_m=True),
        CaseSpec("3-p90", case2_p90, case2_p90, "bytes", "p90",
                 "case 3, p90-pessimistic sizing sensitivity"),
        CaseSpec("4", case4, case4, "bytes", "raw",
                 "repaired int2 cold tier + RAW int4 protected tier "
                 "(= layout D)", collect_m=True),
        CaseSpec("5", case5, case5, "bytes", "median",
                 "repaired int2 cold tier + ENTROPY-CODED int4 protected "
                 "tier (hybrid; mode-2/3: compressed-resident)", collect_m=True),
        CaseSpec("5-p90", case5_p90, case5_p90, "bytes", "p90",
                 "case 5, p90-pessimistic sizing sensitivity"),
        CaseSpec("5-mode1.5", case4, case5, "bytes", "mode1.5",
                 "case 5, disk-only variant: RAW resident sizes (= case 4's "
                 "classification), compressed transfer bytes"),
        CaseSpec("6a-cons", case6a, case6a, "bytes", "int2-cons-0.95",
                 f"repaired int2 cold tier ALSO entropy-coded, CONSERVATIVE "
                 f"assumed ratio {int2_cons_ratio} (int2 codes assumed to "
                 "barely compress); protected tier as case 5 -- SENSITIVITY "
                 "ROW ONLY"),
        CaseSpec("6b-opt", case6b, case6b, "bytes", "int2-opt-0.85",
                 f"repaired int2 cold tier ALSO entropy-coded, OPTIMISTIC "
                 f"assumed ratio {int2_opt_ratio}; protected tier as case 5 "
                 "-- SENSITIVITY ROW ONLY"),
    ]


# ----------------------------------------------------------- miss structure ----
def _write_miss_structure_csvs(outdir: Path, blobs: dict) -> None:
    if not blobs:
        return
    with (outdir / "missstruct-pm.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "trace", "split"] + [f"p_m{m}" for m in range(9)])
        for (name, tname), s in sorted(blobs.items()):
            w.writerow([name, tname, "all"]
                        + [round(s["pm_all"].get(m, 0.0), 6) for m in range(9)])
            w.writerow([name, tname, "steady"]
                        + [round(s["pm_steady"].get(m, 0.0), 6) for m in range(9)])
    with (outdir / "missstruct-summary.csv").open("w", newline="") as f:
        cols = ["case", "trace", "h_decode", "p_m0", "p_le1", "p_le2",
                "p_m0_steady", "p_le1_steady", "p_le2_steady", "n_decode_calls"]
        w = csv.writer(f)
        w.writerow(cols)
        for (name, tname), s in sorted(blobs.items()):
            w.writerow([name, tname, round(s["h"], 6), round(s["p_m0"], 6),
                        round(s["p_le1"], 6), round(s["p_le2"], 6),
                        round(s["pm_steady"].get(0, 0.0), 6),
                        round(s["p_le1_steady"], 6), round(s["p_le2_steady"], 6),
                        s["n_decode_calls"]])


# -------------------------------------------------------------------- main ----
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--traces", type=Path, nargs="+", default=DEFAULT_TRACES)
    p.add_argument("--usage", type=Path, default=DEFAULT_USAGE)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--census-summary", type=Path, default=DEFAULT_CENSUS)
    p.add_argument("--coder", default=None,
                   help="override census coder (default: headline.best_coder)")
    p.add_argument("--block", type=int, default=None,
                   help="override census block size (default: headline.best_block_size)")
    p.add_argument("--int2-entropy-conservative", type=float, default=0.95,
                   help="case 6a assumed int2-code stored ratio")
    p.add_argument("--int2-entropy-optimistic", type=float, default=0.85,
                   help="case 6b assumed int2-code stored ratio")
    p.add_argument("--warmup-tokens", type=int, default=32)
    p.add_argument("--c-seconds", type=float, default=0.36)
    p.add_argument("--bw-gbs", default="5,8,10,13.3")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--per-token-csv", action="store_true",
                   help="also write per-token disk-bytes/misses CSVs (off by default)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    manifest = json.loads(a.manifest.read_text())
    cfg = manifest["model_config"]
    forced_layers = set(manifest.get("force_int4_layers", []))

    int4_b = bsim.expert_bytes_per_row_scales(cfg, 4)
    int2_b = bsim.expert_bytes_per_row_scales(cfg, 2)

    census_summary = json.loads(a.census_summary.read_text())
    ratio_table, coder, block = load_ratio_table(census_summary, a.coder, a.block)
    band_bytes_median = predicted_band_bytes(cfg, ratio_table, "p50")
    band_bytes_p90 = predicted_band_bytes(cfg, ratio_table, "p90")

    metas, calls_by_trace = {}, {}
    for tpath in a.traces:
        meta, calls = bsim.load_calls(tpath)
        if meta.mtp_present:
            raise SystemExit(f"{tpath}: MTP tier present; unsupported")
        if meta.expert_bytes and meta.expert_bytes != int4_b:
            raise SystemExit(
                f"{tpath}: header expert_bytes {meta.expert_bytes} != "
                f"computed int4 size {int4_b}")
        metas[tpath] = meta
        calls_by_trace[tpath] = calls

    meta0 = metas[a.traces[0]]
    pin_bytes = meta0.pinned_units * int4_b
    lru_slots = meta0.lru_per_layer
    lru_bytes = lru_slots * int4_b
    bws = [float(x) for x in a.bw_gbs.split(",") if x]

    usage = bsim.slotsim.load_usage(a.usage)

    cases = build_cases(cfg, forced_layers, int4_b, int2_b, band_bytes_median,
                         band_bytes_p90, a.int2_entropy_conservative,
                         a.int2_entropy_optimistic)

    a.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[entropy-replay] int4 {int4_b:,} B | int2 {int2_b:,} B | "
          f"census coder={coder} block={block}")
    for band in BAND_NAMES:
        print(f"[entropy-replay] predicted {band}: median "
              f"{band_bytes_median[band]:,.0f} B "
              f"({band_bytes_median[band] / int4_b:.4f}x int4) | p90 "
              f"{band_bytes_p90[band]:,.0f} B "
              f"({band_bytes_p90[band] / int4_b:.4f}x int4)")
    print(f"[entropy-replay] pin budget {pin_bytes:,} B | LRU {lru_slots} "
          f"slots | byte-mode {lru_bytes:,} B/layer")

    grid_rows, tok_rows, struct_blobs = [], [], {}
    for case in cases:
        pinned, pin_used = bsim.fill_pins(usage, case.resident_sizes, pin_bytes)
        budget = lru_slots if case.lru_mode == "slots" else lru_bytes
        for tpath in a.traces:
            tname = tpath.stem
            token_of, n_tokens = bsim.segment_tokens(calls_by_trace[tpath])
            res = replay_dual(calls_by_trace[tpath], token_of,
                               case.resident_sizes, case.transfer_sizes, pinned,
                               case.lru_mode, budget, layout=case.name,
                               trace=tname, collect_m=case.collect_m)
            sb, sm, sn = res.steady(a.warmup_tokens)
            wb, wm, wn = res.warm(a.warmup_tokens)
            row = {
                "case": case.name, "trace": tname,
                "sizing_variant": case.sizing_variant, "lru_mode": case.lru_mode,
                "lru_budget": budget, "pinned_pairs": len(pinned),
                "pinned_bytes": pin_used,
                "hit_rate": round(res.hit_rate, 6),
                "dec_hit_rate": round(res.dec_hit_rate, 6),
                "warm_tokens": wn, "warm_bytes_per_tok": round(wb, 1),
                "warm_miss_per_tok": round(wm, 3),
                "steady_tokens": sn, "steady_bytes_per_tok": round(sb, 1),
                "steady_miss_per_tok": round(sm, 3),
                "desc": case.desc,
            }
            grid_rows.append(row)
            print(f"  case {case.name:11s} {tname:16s} {case.lru_mode:5s} "
                  f"hit {100 * res.hit_rate:6.2f}% "
                  f"(decode {100 * res.dec_hit_rate:6.2f}%) steady "
                  f"{sb / 1e6:9.1f} MB/tok {sm:6.2f} miss/tok")
            for bw in bws:
                t = a.c_seconds + sb / (bw * GB)
                tok_rows.append({
                    "case": case.name, "trace": tname, "bw_gbs": bw,
                    "t_s_per_tok": round(t, 4), "tok_s": round(1.0 / t, 4),
                })
            if case.collect_m:
                struct_blobs[(case.name, tname)] = bsim.miss_structure(res, a.warmup_tokens)
            if a.per_token_csv:
                per_tok = a.outdir / f"per-token-{case.name}-{tname}.csv"
                with per_tok.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["token", "disk_bytes", "disk_misses"])
                    for i, (b, m) in enumerate(
                            zip(res.per_token_bytes, res.per_token_misses)):
                        w.writerow([i, b, m])
        with (a.outdir / f"pinset-{case.name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "expert", "bytes"])
            for (l, e), sz in sorted(pinned.items()):
                w.writerow([l, e, sz])

    # delta vs case 1 at the same trace/BW
    base = {(r["trace"], r["bw_gbs"]): r["tok_s"] for r in tok_rows
            if r["case"] == "1"}
    for r in tok_rows:
        b = base.get((r["trace"], r["bw_gbs"]))
        r["delta_vs_case1_pct"] = round(100 * (r["tok_s"] / b - 1), 2) if b else ""

    with (a.outdir / "case-grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0]))
        w.writeheader()
        w.writerows(grid_rows)
    with (a.outdir / "tok-s-grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tok_rows[0]))
        w.writeheader()
        w.writerows(tok_rows)
    with (a.outdir / "predicted-sizes-by-band.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band", "stat", "bytes", "ratio_vs_int4_raw"])
        for band in BAND_NAMES:
            w.writerow([band, "median", round(band_bytes_median[band], 1),
                        round(band_bytes_median[band] / int4_b, 6)])
            w.writerow([band, "p90", round(band_bytes_p90[band], 1),
                        round(band_bytes_p90[band] / int4_b, 6)])

    _write_miss_structure_csvs(a.outdir, struct_blobs)

    n_routed = len(routed_pairs(cfg))
    config = {
        "traces": {str(t): {"md5": bsim.md5_of(t)} for t in a.traces},
        "usage": {"path": str(a.usage), "md5": bsim.md5_of(a.usage)},
        "manifest": {"path": str(a.manifest), "md5": bsim.md5_of(a.manifest),
                     "force_int4_layers": sorted(forced_layers)},
        "census_summary": {"path": str(a.census_summary),
                            "md5": bsim.md5_of(a.census_summary),
                            "coder": coder, "block": block,
                            "n_sampled_experts": census_summary.get("n_experts"),
                            "n_total_routed_experts": n_routed},
        "int4_bytes": int4_b, "int2_bytes": int2_b,
        "predicted_band_bytes_median": band_bytes_median,
        "predicted_band_bytes_p90": band_bytes_p90,
        "int2_entropy_conservative_ratio": a.int2_entropy_conservative,
        "int2_entropy_optimistic_ratio": a.int2_entropy_optimistic,
        "pin_bytes": pin_bytes, "lru_slots": lru_slots, "lru_bytes": lru_bytes,
        "c_seconds": a.c_seconds, "bw_gbs": bws, "warmup_tokens": a.warmup_tokens,
        "cache_capacity_assumption": (
            "mode-2/3 for the primary row of every entropy-coded case "
            "(2, 3, 5, 6a, 6b): compressed bytes are resident in the pin and "
            "per-layer LRU tiers. The mode-1.5 disk-only sensitivity rows "
            "(2-mode1.5, 5-mode1.5) instead use RAW resident sizes matching "
            "the uncompressed baseline (case 1 / case 4), so hit/miss "
            "classification is unchanged there and only the transferred "
            "bytes shrink."
        ),
        "sizing_method": (
            f"PREDICTED FROM SAMPLE: the census measured 256 of {n_routed} "
            "routed experts; every expert here is priced by its layer "
            "band's median (or p90-pessimistic) band x projection stored "
            "ratio, not its own measurement."
        ),
        "cases": {c.name: c.desc for c in cases},
        "note": "tok/s grid is sensitivity only (t = C + B_miss/BW); "
                "B_miss bytes/token is the authoritative output",
    }
    (a.outdir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[entropy-replay] outputs in {a.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
