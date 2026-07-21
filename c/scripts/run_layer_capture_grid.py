#!/usr/bin/env python3
"""FAST numerical-capture grid runner for the L2 numerical-validation plan
(scripts/capture_layer_outputs.md/.patch): drives the S x T grid for ONE
ILI_METAL_PREFILL arm against an already-running `ili serve`, landing the engine's
opt-in per-layer capture hook (ILI_LAYER_CAPTURE_DIR/ILI_LAYER_CAPTURE_LAYERS, NOT
applied to glm.c on disk -- see capture_layer_outputs.md) on each (S, T) cell.

Scope: the FAST subset of the full grid in capture_layer_outputs.md -- T in
{128, 1024, 4096} by default (T=16384 is out of scope here: multi-hour prefill
territory per long_ctx_profile.py's own 32K/64K/128K calibration, the wrong shape for
a same-night recovery step). S defaults to the full {1, 4, 16, 64, 256}.

How a cell is hit: reuses scripts/long_ctx_profile.py's `build_transcript()` and
`_post_json()` directly (imported, not copied -- same dependency-free HTTP approach
serve_gate.py/abba_transcript_driver.py already use) for the exact two-call pattern
that script's own `run_one()` already relies on: a `max_tokens=1` "load" call
establishes ~T tokens of monotonic history, then a second `max_tokens=1` "measure"
call appends exactly S new tokens -- landing the capture hook on that (S, T) cell.
The capture .bin files themselves are written by the ENGINE (glm.c), not this script;
--capture-dir here must be the SAME directory the caller passed as
ILI_LAYER_CAPTURE_DIR when it started `ili serve` for this arm.

Honesty note (T drift within one checkpoint): the S-turns at one T checkpoint are sent
back to back on the SAME monotonic slot (append-only, matching run-m5max-serve.sh's
agent-harness contract), so T drifts upward by each turn's own S tokens across the
batch (worst case ~+341 tokens by the last S=256 turn at a given checkpoint). This is
the same trade-off capture_layer_outputs.md itself accepts ("nothing here claims S/T
are independent"); every capture file's own pos_base field records the ACTUAL T for
that cell, so nothing is mis-labeled -- cells are only nominally off the grid's round
numbers, never wrong about what they measured.

--dry-run: no serve, no HTTP, no glm whatsoever. Synthesizes the same S x T x layer
capture files this grid would produce against a real engine, via
tools/compare_layer_captures.py's own write_capture() (imported directly), so
evening_orchestrator.sh's dry run can exercise its downstream compare_layer_captures.py
step end to end without an engine -- same no-engine-but-still-exercise-the-plumbing
convention as scripts/run_abba_matrix.sh's own --dry-run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS_DIR = HERE.parent / "tools"

DEFAULT_LAYERS = "5,39,74"
DEFAULT_S_VALUES = "1,4,16,64,256"
DEFAULT_T_VALUES = "128,1024,4096"     # the FAST subset; T=16384 excluded on purpose


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def build_filler_message(n_tokens, tag):
    """A short synthetic user turn sized to ~n_tokens tokens, using
    long_ctx_profile.py's own 4-chars/token heuristic (CHARS_PER_TOKEN) -- the same
    corpus-independent measure it uses for its own target_chars budget."""
    chars_per_token = 4
    prefix = f"[capture-grid {tag}] "
    pad = max(0, n_tokens * chars_per_token - len(prefix))
    return {"role": "user", "content": prefix + ("x" * pad)}


def run_drive(args) -> int:
    lcp = _load_module("long_ctx_profile", HERE / "long_ctx_profile.py")
    if not args.model_id:
        # Query the already-running ili serve for its advertised model id rather than
        # trust a hardcoded/defaulted string -- see lcp.resolve_model_id's docstring for
        # the 2026-07-14 incident (this exact script's stale `glm-5.2` default 404'd
        # against a server that had been renamed to advertise `glm-5.2-iliria`).
        args.model_id = lcp.resolve_model_id(args.host, args.port)
    print(f"[model-id] using {args.model_id!r}", file=sys.stderr)
    s_values = _parse_int_list(args.s_values)
    t_values = sorted(_parse_int_list(args.t_values))

    capture_dir = Path(args.capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dry_run": False, "metal_prefill": args.metal_prefill, "host": args.host,
                "port": args.port, "s_values": s_values, "t_values": t_values, "cells": []}

    for t in t_values:
        transcript = lcp.build_transcript(t, str(args.repo_root))
        load_payload = {"model": args.model_id, "messages": transcript, "max_tokens": 1,
                        "temperature": 0, "cache_slot": args.slot, "stream": False}
        lcp._post_json(args.host, args.port, "/v1/chat/completions", load_payload, args.timeout)
        for s in s_values:
            filler = build_filler_message(s, f"T{t}-S{s}")
            measure_payload = {"model": args.model_id, "messages": transcript + [filler],
                               "max_tokens": 1, "temperature": 0, "cache_slot": args.slot,
                               "stream": False}
            t0 = time.monotonic()
            lcp._post_json(args.host, args.port, "/v1/chat/completions", measure_payload, args.timeout)
            manifest["cells"].append({"t_nominal": t, "s": s, "wall_s": round(time.monotonic() - t0, 3)})
            # Monotonic append-only history: fold the filler turn in so the NEXT S value at
            # this same T checkpoint starts from here (see the module docstring's T-drift note).
            transcript = transcript + [filler]

    (capture_dir / "grid-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[capture-grid] drove {len(t_values)}x{len(s_values)} cells for metal_prefill="
          f"{args.metal_prefill}; manifest -> {capture_dir / 'grid-manifest.json'}")
    return 0


def run_dry_run(args) -> int:
    clc = _load_module("compare_layer_captures", TOOLS_DIR / "compare_layer_captures.py")
    layers = _parse_int_list(args.layers)
    s_values = _parse_int_list(args.s_values)
    t_values = sorted(_parse_int_list(args.t_values))
    capture_dir = Path(args.capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    D = args.hidden_dim

    # Deterministic per (capture_dir, arm) so re-running a dry run is reproducible, and so
    # the two arms' otherwise-identical cells differ only by the small perturbation below
    # (standing in for real kernel-family rounding variance) rather than by unrelated noise.
    rng = np.random.default_rng(abs(hash((str(capture_dir), args.metal_prefill))) % (2**32))

    written = []
    for t in t_values:
        for s in s_values:
            for layer in layers:
                data = rng.standard_normal((s, D)).astype(np.float32)
                if args.metal_prefill:
                    data = data + rng.normal(scale=1e-3, size=data.shape).astype(np.float32)
                path = capture_dir / f"L{layer:03d}_S{s:04d}_T{t:06d}.bin"
                clc.write_capture(path, layer=layer, S=s, pos_base=t, D=D,
                                  metal_prefill=int(args.metal_prefill), data=data)
                written.append(path.name)

    manifest = {"dry_run": True, "metal_prefill": args.metal_prefill, "layers": layers,
                "s_values": s_values, "t_values": t_values, "files": written}
    (capture_dir / "grid-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[capture-grid] --dry-run synthesized {len(written)} capture files -> {capture_dir}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", required=True,
                        help="ILI_LAYER_CAPTURE_DIR for this arm (real mode: must match what "
                             "the caller passed to ili serve; --dry-run: where synthetic "
                             ".bin files are written)")
    parser.add_argument("--metal-prefill", type=int, choices=(0, 1), required=True,
                        help="which arm this invocation is for. Real mode: labels the "
                             "manifest only (the engine's actual ILI_METAL_PREFILL is set "
                             "by whoever started ili serve). --dry-run: also determines the "
                             "synthetic data's small perturbation, standing in for real "
                             "kernel-family rounding variance.")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--s-values", default=DEFAULT_S_VALUES)
    parser.add_argument("--t-values", default=DEFAULT_T_VALUES)
    parser.add_argument("--dry-run", action="store_true")
    # real-mode (drive) options:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=None,
                        help="chat-completions `model` field; default: query the "
                             "running server's GET /v1/models and use its first "
                             "advertised id (falls back to long_ctx_profile.py's "
                             "DEFAULT_MODEL_ID if that query fails)")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--repo-root", default=str(HERE.parent))
    # --dry-run-only options:
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="dry-run synthetic D (default small and fast; real captures use "
                             "D=6144, GLM-5.2's real hidden size)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        return run_dry_run(args)
    return run_drive(args)


if __name__ == "__main__":
    sys.exit(main())
