#!/usr/bin/env python3
"""Per-expert quantization-error measurement for the mixed-precision container.

For each routed (layer, expert) in an existing pre-quantized container:
dequantize the stored weights (int4 today), requantize to each candidate bit
width (int2, optionally int3), and record the reconstruction error per
projection plus combined. This is the GEMQ-style importance signal the bit
allocator consumes: error, NOT routing frequency (MoPEQ + our #47 both show
frequency alone misleads). Routing mass from .fa_usage is recorded as
SEPARATE columns for reporting only - never folded into the error.

Error metric: relative Frobenius norm per tensor
    err = ||dequant(quant_b(W)) - W||_F / ||W||_F
and combined over gate/up/down as sqrt(sum ||d||^2 / sum ||W||^2).

Resumable (appends to the CSV, skips pairs already measured), shard-friendly
(experts are processed grouped by shard file), rate-limited reads
(--max-mb-s), and refuses to run while the engine is alive unless
--allow-busy (smoke tests only).

Usage (full sweep - ONLY when the engine is idle):
  pgrep -x glm >/dev/null && echo "engine busy" || \
    nice -n 19 python3 c/tools/measure_expert_quant_error.py \
      --model /path/to/models/GLM-5.2-int4-with-int8-mtp \
      --out c/bench-m5max/container-20260715/expert-quant-error.csv

Tiny smoke (<=1 GB reads, throttled, engine may be in a decode phase):
  nice -n 19 python3 c/tools/measure_expert_quant_error.py \
      --model ... --out smoke.csv --layers 3 --limit 48 --max-mb-s 60 --allow-busy
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_container as qc


def csv_header(candidates):
    cols = ["layer", "expert", "src_bits", "bytes_src", "bytes_int2",
            "usage_general", "share_general", "usage_coding", "share_coding"]
    for b in candidates:
        cols += [f"err_int{b}_gate", f"err_int{b}_up", f"err_int{b}_down",
                 f"err_int{b}"]
    return ",".join(cols)


def load_done(path, expected_header):
    """Already-measured (layer, expert) pairs from an existing CSV."""
    done = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return done
    with open(path) as f:
        header = f.readline().rstrip("\n")
        if header != expected_header:
            sys.stderr.write(
                f"ERROR: existing {path} has a different schema.\n"
                f"  found:    {header}\n  expected: {expected_header}\n"
                "Use a fresh --out (or the same --candidates as the first run).\n")
            sys.exit(1)
        for line in f:
            parts = line.split(",")
            if len(parts) >= 2:
                done.add((int(parts[0]), int(parts[1])))
    return done


def layer_totals(usage):
    tot = {}
    for (l, _e), c in usage.items():
        tot[l] = tot.get(l, 0) + c
    return tot


def index_experts(index, cfg):
    """{(layer, expert): {proj: (w_entry, qs_entry)}} for routed layers."""
    experts = {}
    for name, entry in index.items():
        m = qc.EXPERT_RE.match(name)
        if not m:
            continue
        layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (cfg["first_dense"] <= layer < cfg["n_layers"]):
            continue  # MTP head experts stay untouched
        qs = index.get(name + ".qs")
        if qs is None:
            raise SystemExit(f"missing scales for {name} - not a pre-quantized container")
        experts.setdefault((layer, expert), {})[proj] = (entry, qs)
    for key, projs in experts.items():
        if set(projs) != set(qc.PROJS):
            raise SystemExit(f"expert {key} incomplete: has {sorted(projs)}")
    return experts


def measure_expert(cfg, projs, candidates, limiter):
    """-> (src_bits, {cand: (err_gate, err_up, err_down, err_all)})."""
    num = {b: 0.0 for b in candidates}
    den = 0.0
    per_proj = {b: [] for b in candidates}
    src_bits = None
    for proj in qc.PROJS:
        w_entry, qs_entry = projs[proj]
        O, I = qc.expert_shape(cfg, proj)
        bits = qc.infer_bits(w_entry["nbytes"], O, I)
        src_bits = bits if src_bits is None else src_bits
        raw = qc.st_read_tensor(w_entry, limiter)
        qs = qc.st_read_tensor(qs_entry, limiter)
        w = qc.dequant(raw, qs, O, I, bits)
        d = float(np.sum(w.astype(np.float64) ** 2))
        den += d
        for b in candidates:
            packed, scales = qc.quantize(w, b)
            wc = qc.dequant(packed, scales, O, I, b)
            n = float(np.sum((wc.astype(np.float64) - w) ** 2))
            num[b] += n
            per_proj[b].append(math.sqrt(n / d) if d > 0 else 0.0)
    out = {}
    for b in candidates:
        err_all = math.sqrt(num[b] / den) if den > 0 else 0.0
        out[b] = (*per_proj[b], err_all)
    return src_bits, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="pre-quantized container dir")
    ap.add_argument("--out", required=True, help="output CSV (appended, resumable)")
    ap.add_argument("--layers", default=None,
                    help="only these layers, e.g. '3' or '3-27,40' (default: all routed)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N newly measured experts (0 = no limit)")
    ap.add_argument("--candidates", default="2",
                    help="candidate bit widths, comma-separated (default '2'; '2,3' adds int3)")
    ap.add_argument("--max-mb-s", type=float, default=200.0,
                    help="read throttle in MB/s (0 = unlimited; default 200)")
    ap.add_argument("--usage", default=None, help=".fa_usage path (default: <model>/.fa_usage)")
    ap.add_argument("--usage-coding", default=None,
                    help=".fa_usage.coding path (default: <model>/.fa_usage.coding)")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if the engine is alive (tiny throttled smokes only)")
    a = ap.parse_args(argv)

    qc.require_idle(a.allow_busy, "expert quant-error measurement")
    candidates = sorted({int(x) for x in a.candidates.split(",")})
    for b in candidates:
        if b not in (2, 3):
            ap.error(f"candidate bits {b} unsupported (2 or 3)")

    cfg = qc.load_config(a.model)
    index = qc.st_scan(a.model)
    experts = index_experts(index, cfg)

    usage_path = a.usage or os.path.join(a.model, ".fa_usage")
    coding_path = a.usage_coding or os.path.join(a.model, ".fa_usage.coding")
    usage = qc.load_usage(usage_path) if os.path.exists(usage_path) else {}
    coding = qc.load_usage(coding_path) if os.path.exists(coding_path) else {}
    usage_tot, coding_tot = layer_totals(usage), layer_totals(coding)

    header = csv_header(candidates)
    done = load_done(a.out, header)
    layer_filter = qc.parse_layers(a.layers)
    pending = [k for k in sorted(experts)
               if k not in done and (not layer_filter or k[0] in layer_filter)]
    # shard-at-a-time: group by the file holding the gate tensor
    pending.sort(key=lambda k: (experts[k]["gate_proj"][0]["path"], k))
    if a.limit > 0:
        pending = pending[:a.limit]
    print(f"[measure] {len(pending)} experts to do ({len(done)} already in CSV), "
          f"candidates int{'/int'.join(map(str, candidates))}, "
          f"throttle {a.max_mb_s or 'unlimited'} MB/s", flush=True)
    if not pending:
        return 0

    limiter = qc.RateLimiter(a.max_mb_s)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    t0 = time.monotonic()
    with open(a.out, "a") as out:
        if new_file:
            out.write(header + "\n")
        for i, key in enumerate(pending):
            layer, expert = key
            src_bits, errs = measure_expert(cfg, experts[key], candidates, limiter)
            ug = usage.get(key, 0)
            uc = coding.get(key, 0)
            sg = ug / usage_tot[layer] if usage_tot.get(layer) else 0.0
            sc = uc / coding_tot[layer] if coding_tot.get(layer) else 0.0
            row = [str(layer), str(expert), str(src_bits),
                   str(qc.expert_total_bytes(cfg, src_bits)),
                   str(qc.expert_total_bytes(cfg, 2)),
                   str(ug), f"{sg:.6g}", str(uc), f"{sc:.6g}"]
            for b in candidates:
                row += [f"{v:.6g}" for v in errs[b]]
            out.write(",".join(row) + "\n")
            out.flush()
            if (i + 1) % 32 == 0 or i + 1 == len(pending):
                dt = time.monotonic() - t0
                eta = dt / (i + 1) * (len(pending) - i - 1)
                print(f"[measure] {i + 1}/{len(pending)} experts, "
                      f"{limiter.bytes / 1e9:.2f} GB read, {dt:.0f}s elapsed, "
                      f"ETA {eta / 60:.1f} min", flush=True)
    print(f"[measure] DONE: {len(pending)} experts appended to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
