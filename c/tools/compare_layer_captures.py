#!/usr/bin/env python3
"""L2 numerical-validation comparison tool: loads two per-layer attention-output
capture files (or two directories of them) written by the env-gated capture hook in
scripts/capture_layer_outputs.patch (NOT applied to glm.c -- see that file and
scripts/capture_layer_outputs.md for the capture plan) and computes, per layer:
max-abs difference, RMS difference, normalized RMS, cosine similarity, and a
top-margin proxy -- the review's metric set for judging how much the CPU
(ILI_METAL_PREFILL=0) and Metal-prefill (=1) attention paths actually diverge.

CAPTURE FILE FORMAT (little-endian, matches the patch):
    offset  0   8 bytes   magic "FACAPT1\0"
    offset  8   uint32    format version (currently 1)
    offset 12   uint32    layer index
    offset 16   uint32    S (sequence length / tokens in this call)
    offset 20   uint32    pos_base (T: context depth at capture time)
    offset 24   uint32    D (hidden dim, e.g. 6144 for GLM-5.2 int4)
    offset 28   uint32    metal_prefill (0=CPU path, 1=Metal-prefill path)
    offset 32   S*D x float32   the pre-residual attention output, row-major [S, D]

METRICS (per position row, i.e. per one of the S tokens' D-dim vector, THEN
aggregated across positions -- both levels are reported, since a divergence that
concentrates in one position is a different finding than one spread evenly):
    max_abs   max(|a - b|)
    rms       sqrt(mean((a - b)^2))
    nrms      rms / rms(a)                      -- normalized by the REFERENCE
                                                    (first file, conventionally CPU
                                                    -- run-m5max-serve.sh documents
                                                    ILI_METAL_PREFILL=0 as the
                                                    byte-exact default)
    cosine    dot(a, b) / (||a|| * ||b||)
    top_margin  proxy metric, NOT a real output-token logit margin (the capture is
                an intermediate activation, not lm_head output): sort the D-dim
                vector descending and take value[0] - value[1]. A SHRINKING gap
                between reference and comparison top_margin is the interesting
                signal -- it means a downstream argmax-like decision (routing,
                greedy decode) got measurably closer to flipping. Reported for
                the reference vector, the comparison vector, and their difference.

Aggregation across the S positions: max_abs and top_margin use the WORST case
(max over positions / min margin over positions, respectively -- both are
"how bad can it get", not "how bad on average"); rms/nrms/cosine use the mean
across positions (a single flattened-vector number is also reported for cosine,
since that is the more common convention and is what most reviewers expect first).

Handles the full S={1,4,16,64,256} x T={128,1K,4K,16K} x {early,mid,late} grid in
--batch mode by pairing files across two directories purely by filename (the
L<layer>_S<S>_T<pos_base>.bin naming the patch writes) -- nothing here is hardcoded
to a specific grid; it processes whatever capture files actually exist.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"FACAPT1\x00"
HEADER_SIZE = 32
# magic(8s) + version,layer,S,pos_base,D,metal_prefill (6 x uint32) = 8 + 24 = 32 bytes,
# matching the patch's header exactly field-for-field.
_HEADER = struct.Struct("<8sIIIIII")
assert _HEADER.size == HEADER_SIZE, f"header struct size {_HEADER.size} != {HEADER_SIZE}"


class CaptureFormatError(ValueError):
    pass


class Capture:
    __slots__ = ("path", "version", "layer", "S", "pos_base", "D", "metal_prefill", "data")

    def __init__(self, path, version, layer, S, pos_base, D, metal_prefill, data):
        self.path = path
        self.version = version
        self.layer = layer
        self.S = S
        self.pos_base = pos_base
        self.D = D
        self.metal_prefill = metal_prefill
        self.data = data   # np.ndarray, shape [S, D], dtype float64 (promoted from f32 on load)

    def __repr__(self):
        return (f"Capture(layer={self.layer}, S={self.S}, T={self.pos_base}, D={self.D}, "
                f"metal_prefill={self.metal_prefill})")


def load_capture(path) -> Capture:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise CaptureFormatError(f"{path}: file shorter than the {HEADER_SIZE}-byte header")
    magic, version, layer, S, pos_base, D, metal_prefill = _HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise CaptureFormatError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
    if version != 1:
        raise CaptureFormatError(f"{path}: unsupported format version {version}")
    expected_bytes = HEADER_SIZE + S * D * 4
    if len(raw) != expected_bytes:
        raise CaptureFormatError(
            f"{path}: expected {expected_bytes} bytes for S={S} D={D} (header {HEADER_SIZE} + "
            f"payload {S * D * 4}), file is {len(raw)} bytes")
    data = np.frombuffer(raw, dtype="<f4", count=S * D, offset=HEADER_SIZE).reshape(S, D).astype(np.float64)
    return Capture(path, version, layer, S, pos_base, D, metal_prefill, data)


def write_capture(path, layer, S, pos_base, D, metal_prefill, data) -> None:
    """Inverse of load_capture(): used by tests to build synthetic fixtures without
    needing the real engine or the patch applied."""
    data = np.asarray(data, dtype="<f4")
    if data.shape != (S, D):
        raise ValueError(f"data shape {data.shape} != (S={S}, D={D})")
    header = _HEADER.pack(MAGIC, 1, layer, S, pos_base, D, metal_prefill)
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(data.tobytes())


# ---- metrics --------------------------------------------------------------------------

def _row_metrics(a_row, b_row):
    diff = a_row - b_row
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    rms = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
    ref_rms = float(np.sqrt(np.mean(a_row ** 2))) if a_row.size else 0.0
    nrms = (rms / ref_rms) if ref_rms > 0 else (0.0 if rms == 0 else float("inf"))
    na, nb = np.linalg.norm(a_row), np.linalg.norm(b_row)
    cosine = float(np.dot(a_row, b_row) / (na * nb)) if na > 0 and nb > 0 else float("nan")

    def top_margin(v):
        if v.size < 2:
            return float("nan")
        top2 = np.partition(v, -2)[-2:]
        return float(max(top2) - min(top2))

    margin_a, margin_b = top_margin(a_row), top_margin(b_row)
    return {
        "max_abs": max_abs, "rms": rms, "nrms": nrms, "cosine": cosine,
        "top_margin_ref": margin_a, "top_margin_cmp": margin_b,
        "top_margin_delta": margin_b - margin_a,
    }


def compare_captures(ref: Capture, cmp_: Capture) -> dict:
    """`ref` is conventionally the CPU (byte-exact, ILI_METAL_PREFILL=0) capture and
    `cmp_` the Metal-prefill one, but the function is symmetric except for which
    side normalizes nrms and which side is `top_margin_ref` vs `_cmp`."""
    if ref.S != cmp_.S or ref.D != cmp_.D:
        raise ValueError(f"shape mismatch: ref S={ref.S} D={ref.D} vs cmp S={cmp_.S} D={cmp_.D}")
    if ref.layer != cmp_.layer or ref.pos_base != cmp_.pos_base:
        print(f"warning: comparing different (layer, T) captures: "
              f"ref layer={ref.layer} T={ref.pos_base} vs cmp layer={cmp_.layer} T={cmp_.pos_base}",
              file=sys.stderr)

    per_position = [_row_metrics(ref.data[i], cmp_.data[i]) for i in range(ref.S)]

    flat_a, flat_b = ref.data.reshape(-1), cmp_.data.reshape(-1)
    flat = _row_metrics(flat_a, flat_b)

    def agg(key, fn):
        return fn(m[key] for m in per_position)

    mean_of = lambda key: agg(key, lambda xs: float(np.mean(list(xs))))
    return {
        "layer": ref.layer, "S": ref.S, "T": ref.pos_base, "D": ref.D,
        "ref_metal_prefill": ref.metal_prefill, "cmp_metal_prefill": cmp_.metal_prefill,
        "max_abs": agg("max_abs", max),
        "rms_mean": mean_of("rms"),
        "nrms_mean": mean_of("nrms"),
        "cosine_mean": mean_of("cosine"),
        "cosine_flat": flat["cosine"],
        # Most negative = the position where the comparison path's top-margin shrank
        # the most relative to the reference -- the worst case for "how close did this
        # get to flipping a downstream argmax-like decision" (a positive delta just
        # means the margin grew, which is not the risk this metric is watching for).
        "top_margin_delta_worst": agg("top_margin_delta", min) if ref.S else 0.0,
        "per_position": per_position,
    }


# ---- batch / directory pairing ---------------------------------------------------------

def discover_pairs(dir_a: Path, dir_b: Path):
    """Pair files present in BOTH directories by filename (the L<layer>_S<S>_T<T>.bin
    convention the patch writes). Files present in only one directory are reported,
    not silently dropped."""
    files_a = {p.name: p for p in dir_a.glob("*.bin")}
    files_b = {p.name: p for p in dir_b.glob("*.bin")}
    common = sorted(set(files_a) & set(files_b))
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    return [(files_a[n], files_b[n]) for n in common], only_a, only_b


def run_batch(dir_a: Path, dir_b: Path):
    pairs, only_a, only_b = discover_pairs(dir_a, dir_b)
    if only_a:
        print(f"warning: {len(only_a)} file(s) only in {dir_a}: {', '.join(only_a[:5])}"
              + (" ..." if len(only_a) > 5 else ""), file=sys.stderr)
    if only_b:
        print(f"warning: {len(only_b)} file(s) only in {dir_b}: {', '.join(only_b[:5])}"
              + (" ..." if len(only_b) > 5 else ""), file=sys.stderr)
    rows = []
    for path_a, path_b in pairs:
        ref = load_capture(path_a)
        cmp_ = load_capture(path_b)
        result = compare_captures(ref, cmp_)
        result["file"] = path_a.name
        rows.append(result)
    return rows


SUMMARY_COLUMNS = ["file", "layer", "S", "T", "max_abs", "rms_mean", "nrms_mean",
                   "cosine_mean", "cosine_flat", "top_margin_delta_worst"]


def write_summary_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in SUMMARY_COLUMNS})


def print_summary(rows):
    header = SUMMARY_COLUMNS
    widths = [max(len(h), *(len(f"{r[h]:.4g}" if isinstance(r[h], float) else str(r[h]))
                           for r in rows)) if rows else len(h) for h in header]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        cells = [f"{row[h]:.4g}" if isinstance(row[h], float) else str(row[h]) for h in header]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))


# ---- CLI -----------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, help="reference capture file (conventionally CPU, MP=0)")
    parser.add_argument("--b", type=Path, help="comparison capture file (conventionally Metal, MP=1)")
    parser.add_argument("--dir-a", type=Path, help="batch mode: directory of reference captures")
    parser.add_argument("--dir-b", type=Path, help="batch mode: directory of comparison captures")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="batch mode: write the summary table here")
    return parser


def main():
    args = build_parser().parse_args()
    if args.a and args.b:
        ref, cmp_ = load_capture(args.a), load_capture(args.b)
        result = compare_captures(ref, cmp_)
        result["file"] = args.a.name
        print_summary([result])
        return
    if args.dir_a and args.dir_b:
        rows = run_batch(args.dir_a, args.dir_b)
        if not rows:
            print("no matching capture file pairs found", file=sys.stderr)
            sys.exit(1)
        print_summary(rows)
        if args.out_csv:
            write_summary_csv(rows, args.out_csv)
            print(f"\nwrote {args.out_csv}")
        return
    print("error: pass either --a/--b (single pair) or --dir-a/--dir-b (batch)", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
