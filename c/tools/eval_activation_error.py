#!/usr/bin/env python3
"""Activation-level quantizer verdict: the SECOND falsifier named in
docs/performance-theory.json's expert-quant-error-saliency notes ("the real
test is activation-output error (||W4x - W2x|| / ||W4x|| on representative
activations x, not the weight matrices in isolation)"), narrowed to per-
expert + layer-output error per the 2026-07-15 build task.

WHY THIS TOOL EXISTS: every existing error measurement in this pipeline
(tests/test_quant_noise_floor.py, tools/measure_expert_quant_error.py) is
WEIGHT-Frobenius error -- ||dequant(quant(W)) - W|| / ||W||, computed on the
weight matrix alone, never multiplied through a real activation. That proxy
is cheap and already showed the shipped quant_int2 is defective (0.868 vs a
0.343 Lloyd-Max optimum), but weight Frobenius alone is never sufficient
evidence of real model-quality impact (same source). This tool computes the
next tier: given REAL captured expert-input activations, how much does each
quantizer's error actually perturb this expert's own output.

============================== THE .npy CONTRACT ==============================
(produced by tomorrow's engine runs -- see docs/roadmap-daily-driver.md; this
tool only reads whatever matches this convention, it does not care how the
files were produced. A synthetic fixture exercising this exact contract is
in tests/test_eval_activation_error.py, so this tool is validated without
the engine capture hook existing yet.)

Directory of one .npy file per captured (layer, expert) pair:

    <activations-dir>/L<layer>_E<expert>.npy

Contents: a plain numpy .npy file (np.save/np.load), float32 ndarray, shape
[N, hidden]. N is however many times this expert was observed as a routed
choice during the capture run (N>=1; a single-sample capture, N=1, is valid,
just noisier). hidden is the model's hidden_size -- the expert's INPUT
dimension (the post-attention, post-norm residual-stream row that feeds
gate_proj/up_proj, i.e. glm.c moe()'s per-expert `xg` rows -- NOT the
expert's output, and NOT down_proj's input, which is a derived
[[N, moe_inter]] tensor this tool reconstructs itself, see "Error model"
below). Row order does not matter -- this tool treats N as an unordered
sample and reports an aggregate (sum-of-squares) relative error, the same
convention tools/measure_expert_quant_error.py uses for its own err_all.

Filenames are matched by the regex ACTIVATION_RE below (`layer`/`expert` are
plain decimal, no zero-padding required, e.g. L7_E212.npy or L07_E0212.npy
both parse). Any other file in the directory is ignored.

Optional companion file, once per capture run (not per expert):
    <activations-dir>/manifest.json: {"hidden": <int>, ...}
  Used only to CROSS-CHECK hidden against the container's own config.json --
  a mismatch is a hard error, never a silent skip. Safe to omit entirely.

============================== Error model ==============================
For a captured expert with input activations X [N, hidden] and its INT4
(reference) and INT2-candidate (both quantizers) weights:

    g4 = X @ Wg4.T ; u4 = X @ Wu4.T ; h4 = silu(g4) * u4 ; y4 = h4 @ Wd4.T

is the reference (int4) forward. For each int2 VARIANT v (defective,
fitted, see tools/quant_container.py INT2_QUANTIZERS):

    g2 = X @ Wg2v.T                         err_gate = ||g2-g4|| / ||g4||
    u2 = X @ Wu2v.T                         err_up   = ||u2-u4|| / ||u4||
    y2d_isolated = h4 @ Wd2v.T              err_down = ||y2d_isolated-y4|| / ||y4||
    h2 = silu(g2) * u2 ; y2 = h2 @ Wd2v.T   err_layer = ||y2-y4|| / ||y4||

err_gate/err_up/err_down ISOLATE each projection's own error (down is fed
the REFERENCE intermediate h4, not the int2 path's own h2, so its number is
not contaminated by upstream gate/up error -- otherwise every projection's
number would just restate the layer's total error). err_layer is the
COMPOUNDING, end-to-end error (all three projections quantized together) --
this is "the quantizer verdict" for the expert as actually used at decode
time, and is the number that should gate a codebook choice, not any single
isolated projection error.

Usage (needs a captured activations dir -- none exists yet; a fixture-built
one is what tests/test_eval_activation_error.py exercises):
  python3 c/tools/eval_activation_error.py \
    --container /path/to/models/GLM-5.2-int4-with-int8-mtp \
    --activations /path/to/captured/activations \
    --out c/bench-m5max/container-<date>/activation-error.csv
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_container as qc

ACTIVATION_RE = re.compile(r"^L(\d+)_E(\d+)\.npy$")
DEFAULT_VARIANTS = ("defective", "fitted")


def discover_activation_files(activations_dir):
    """{(layer, expert): path} for every file matching the .npy contract."""
    out = {}
    for fn in sorted(os.listdir(activations_dir)):
        m = ACTIVATION_RE.match(fn)
        if not m:
            continue
        out[(int(m.group(1)), int(m.group(2)))] = os.path.join(activations_dir, fn)
    return out


def check_manifest(activations_dir, cfg):
    """Optional manifest.json cross-check: hidden must match config.json."""
    path = os.path.join(activations_dir, "manifest.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        manifest = json.load(f)
    hidden = manifest.get("hidden")
    if hidden is not None and int(hidden) != cfg["hidden"]:
        raise SystemExit(
            f"{path}: manifest hidden={hidden} does not match "
            f"--container config.json hidden_size={cfg['hidden']}")


def load_activations(path, hidden):
    """.npy -> float32 [N, hidden]; rejects anything not matching the contract."""
    x = np.load(path)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != hidden:
        raise SystemExit(
            f"{path}: expected a 2-D [N, {hidden}] array per the .npy contract "
            f"(module docstring), got shape {x.shape}")
    if x.shape[0] == 0:
        raise SystemExit(f"{path}: zero captured rows (N=0) -- nothing to measure")
    return x.astype(np.float32)


def relerr(a, b):
    """||a-b|| / ||b|| over the whole array (sum-of-squares, den from b)."""
    num = float(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    den = float(np.sum(b.astype(np.float64) ** 2))
    return math.sqrt(num / den) if den > 0 else 0.0


def silu(x):
    return x / (1.0 + np.exp(-x))


def load_expert_int4(cfg, index, layer, expert):
    """-> {proj: (weight[O,I] f32, src_bits)} dequantized from the container."""
    out = {}
    for proj in qc.PROJS:
        name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
        w_entry, qs_entry = index.get(name), index.get(name + ".qs")
        if w_entry is None or qs_entry is None:
            raise SystemExit(f"container is missing {name} (or its .qs) for L{layer}E{expert}")
        O, I = qc.expert_shape(cfg, proj)
        bits = qc.infer_bits(w_entry["nbytes"], O, I)
        raw = qc.st_read_tensor(w_entry)
        qs = qc.st_read_tensor(qs_entry)
        out[proj] = qc.dequant(raw, qs, O, I, bits).astype(np.float32)
    return out


def dequant_int2(w4, variant):
    """Quantize w4 [O,I] f32 to int2 (given variant) and immediately dequantize
    -- the roundtripped f32 reconstruction is what a real forward pass sees."""
    O, I = w4.shape
    packed, scale = qc.quantize(w4, 2, int2_variant=variant)
    return qc.dequant(packed, scale, O, I, 2)


def evaluate_expert(cfg, w4, X, variants):
    """-> {variant: (err_gate, err_up, err_down, err_layer)}."""
    Wg4, Wu4, Wd4 = w4["gate_proj"], w4["up_proj"], w4["down_proj"]
    g4 = X @ Wg4.T
    u4 = X @ Wu4.T
    h4 = silu(g4) * u4
    y4 = h4 @ Wd4.T

    out = {}
    for variant in variants:
        Wg2 = dequant_int2(Wg4, variant)
        Wu2 = dequant_int2(Wu4, variant)
        Wd2 = dequant_int2(Wd4, variant)

        g2 = X @ Wg2.T
        u2 = X @ Wu2.T
        err_gate = relerr(g2, g4)
        err_up = relerr(u2, u4)

        y2d_isolated = h4 @ Wd2.T           # down isolated: fed the REFERENCE h4
        err_down = relerr(y2d_isolated, y4)

        h2 = silu(g2) * u2                  # fully compounded (all 3 quantized)
        y2 = h2 @ Wd2.T
        err_layer = relerr(y2, y4)

        out[variant] = (err_gate, err_up, err_down, err_layer)
    return out


def csv_header(variants):
    cols = ["layer", "expert", "n_samples", "hidden", "moe_inter"]
    for v in variants:
        cols += [f"err_gate_{v}", f"err_up_{v}", f"err_down_{v}", f"err_layer_{v}"]
    return ",".join(cols)


def load_done(path, expected_header):
    done = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return done
    with open(path) as f:
        header = f.readline().rstrip("\n")
        if header != expected_header:
            sys.stderr.write(
                f"ERROR: existing {path} has a different schema.\n"
                f"  found:    {header}\n  expected: {expected_header}\n"
                "Use a fresh --out (or the same --variants as the first run).\n")
            sys.exit(1)
        for line in f:
            parts = line.split(",")
            if len(parts) >= 2:
                done.add((int(parts[0]), int(parts[1])))
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--container", required=True, help="pre-quantized int4 container dir")
    ap.add_argument("--activations", required=True,
                    help="directory of captured L<layer>_E<expert>.npy activations")
    ap.add_argument("--out", required=True, help="output CSV (appended, resumable)")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                    help=f"comma-separated int2 quantizer variants to score "
                         f"(default '{','.join(DEFAULT_VARIANTS)}'; must be keys of "
                         "tools/quant_container.py INT2_QUANTIZERS)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N newly measured experts (0 = no limit)")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if the engine is alive (tiny smokes only)")
    a = ap.parse_args(argv)

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    for v in variants:
        if v not in qc.INT2_QUANTIZERS:
            ap.error(f"--variants: {v!r} is not a known quantizer "
                     f"({sorted(qc.INT2_QUANTIZERS)})")

    qc.require_idle(a.allow_busy, "activation-error evaluation")

    cfg = qc.load_config(a.container)
    check_manifest(a.activations, cfg)
    captured = discover_activation_files(a.activations)
    if not captured:
        print(f"[eval] no L<layer>_E<expert>.npy files found in {a.activations}")
        return 0
    index = qc.st_scan(a.container)

    header = csv_header(variants)
    done = load_done(a.out, header)
    pending = [k for k in sorted(captured) if k not in done]
    if a.limit > 0:
        pending = pending[:a.limit]
    print(f"[eval] {len(pending)} captured experts to score "
          f"({len(done)} already in CSV), variants={variants}", flush=True)
    if not pending:
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    with open(a.out, "a") as out:
        if new_file:
            out.write(header + "\n")
        for i, (layer, expert) in enumerate(pending):
            X = load_activations(captured[(layer, expert)], cfg["hidden"])
            w4 = load_expert_int4(cfg, index, layer, expert)
            errs = evaluate_expert(cfg, w4, X, variants)
            row = [str(layer), str(expert), str(X.shape[0]), str(cfg["hidden"]),
                   str(cfg["moe_inter"])]
            for v in variants:
                row += [f"{e:.6g}" for e in errs[v]]
            out.write(",".join(row) + "\n")
            out.flush()
            if (i + 1) % 32 == 0 or i + 1 == len(pending):
                print(f"[eval] {i + 1}/{len(pending)} experts scored", flush=True)
    print(f"[eval] DONE: {len(pending)} experts appended to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
