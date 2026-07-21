"""Shared helpers for the mixed-precision (int4/int2) container pipeline.

Used by measure_expert_quant_error.py, allocate_bit_budget.py and
build_mixed_container.py. numpy-only: safetensors files are parsed/written
directly (same convention as tests/test_resource_plan.py) so the pipeline
runs on the stock python3 of this machine and reads can be rate-limited at
the byte level (the quality bench shares the SSD).

Container format facts this module encodes (verified against c/glm.c):
  - every quantized tensor is `name` (U8, packed 1-D) + `name.qs` (F32 [O]
    per-row scales); the engine infers the bit width PER TENSOR from the
    byte count (glm.c expert_load / qt_from_disk):
        nbytes == O*I            -> int8  (fmt 1)
        nbytes == O*((I+1)//2)   -> int4  (fmt 2, 2 values/byte, v+8 nibbles)
        else                     -> int2  (fmt 3, 4 values/byte, v+2 2-bit)
    so a mixed container needs NO extra metadata - just per-tensor sizes.
  - quantization math must match glm.c pack_int4/pack_int2/quantize_rows
    (np.rint == lrintf round-half-even, per-row abs-max scale, s>=1e-8);
    convert_fp8_to_int4.py already implements it and is reused here.
  - safetensors serialization orders F32 tensors before U8 so every offset
    stays 4-byte aligned without padding (the engine's mmap fast path
    requires `(off & 3) == 0`); within a dtype, names are sorted, which
    keeps each expert's down/gate/up weights contiguous for the engine's
    single coalesced pread.
"""

import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the exact shipping quantizers (identical math to the C engine).
_conv = _load_module("convert_fp8_to_int4", os.path.join(TOOLS_DIR, "convert_fp8_to_int4.py"))
quant_int8 = _conv.quant_int8
quant_int4 = _conv.quant_int4
quant_int2 = _conv.quant_int2
# Repaired (fitted-scale, format-compatible) int2 path -- see convert_fp8_to_int4.py's
# module comment above quant_int2_fitted and docs/performance-theory.json
# (expert-quant-error-saliency) for the three-stage repair plan this is stage 1 of.
# quant_int2 (the defective quantizer) stays available unchanged for comparison.
quant_int2_fitted = _conv.quant_int2_fitted
INT2_QUANTIZERS = _conv.INT2_QUANTIZERS

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
PROJS = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- config ----
def load_config(model_dir):
    """The subset of config.json the pipeline needs (same keys glm.c reads)."""
    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    cfg = {
        "hidden": int(raw["hidden_size"]),
        "moe_inter": int(raw["moe_intermediate_size"]),
        "n_layers": int(raw["num_hidden_layers"]),
        "first_dense": int(raw["first_k_dense_replace"]),
        "n_experts": int(raw["n_routed_experts"]),
    }
    for k, v in cfg.items():
        if v <= 0:
            raise ValueError(f"config.json: bad {k}={v}")
    return cfg


def expert_shape(cfg, proj):
    """[O, I] of one expert projection (gate/up: [moe_inter, hidden])."""
    if proj == "down_proj":
        return cfg["hidden"], cfg["moe_inter"]
    return cfg["moe_inter"], cfg["hidden"]


def packed_nbytes(O, I, bits):
    if bits >= 5:
        return O * I
    if bits >= 3:
        return O * ((I + 1) // 2)
    return O * ((I + 3) // 4)


def infer_bits(nbytes, O, I):
    """Mirror of glm.c: bit width from the packed byte count."""
    for bits in (8, 4, 2):
        if nbytes == packed_nbytes(O, I, bits):
            return bits
    raise ValueError(f"cannot infer bits: nbytes={nbytes} for [{O},{I}]")


def expert_total_bytes(cfg, bits):
    """Disk bytes of one full expert (3 packed tensors + their F32 scales)."""
    total = 0
    for proj in PROJS:
        O, I = expert_shape(cfg, proj)
        total += packed_nbytes(O, I, bits) + O * 4
    return total


def routed_experts(cfg):
    """All (layer, expert) pairs the manifest governs (main MoE layers only:
    the MTP head at layer n_layers keeps its int8 experts untouched)."""
    return [(l, e) for l in range(cfg["first_dense"], cfg["n_layers"])
            for e in range(cfg["n_experts"])]


# ------------------------------------------------------------ safetensors ----
def st_read_header(path):
    """-> (header dict name->{dtype,shape,data_offsets}, data_start)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return header, 8 + hlen


def st_scan(model_dir, pattern=re.compile(r".*\.safetensors$")):
    """Scan every shard header -> {tensor_name: entry}. Header-only reads."""
    index = {}
    for fn in sorted(os.listdir(model_dir)):
        if not pattern.match(fn):
            continue
        path = os.path.join(model_dir, fn)
        header, data_start = st_read_header(path)
        for name, meta in header.items():
            o0, o1 = meta["data_offsets"]
            index[name] = {
                "path": path, "dtype": meta["dtype"], "shape": meta["shape"],
                "offset": data_start + o0, "nbytes": o1 - o0,
            }
    return index


def st_read_tensor(entry, limiter=None, chunk=32 << 20):
    """Read one tensor by absolute offset, rate-limited, -> 1-D array."""
    parts = []
    with open(entry["path"], "rb") as f:
        f.seek(entry["offset"])
        remaining = entry["nbytes"]
        while remaining > 0:
            n = min(chunk, remaining)
            buf = f.read(n)
            if len(buf) != n:
                raise IOError(f"short read on {entry['path']}")
            if limiter:
                limiter.account(n)
            parts.append(buf)
            remaining -= n
    raw = b"".join(parts)
    if entry["dtype"] == "F32":
        return np.frombuffer(raw, np.float32).copy()
    if entry["dtype"] == "U8":
        return np.frombuffer(raw, np.uint8).copy()
    raise ValueError(f"unsupported dtype {entry['dtype']} in {entry['path']}")


def st_save(path, tensors):
    """Write a safetensors file: F32 tensors first (keeps every offset 4-byte
    aligned with zero padding, like the Rust serializer), names sorted within
    each dtype group (keeps expert down/gate/up weights contiguous)."""
    order = sorted(tensors, key=lambda n: (tensors[n].dtype != np.float32, n))
    header, offset = {}, 0
    for name in order:
        a = tensors[name]
        if a.dtype == np.float32:
            dt = "F32"
        elif a.dtype == np.uint8:
            dt = "U8"
        else:
            raise ValueError(f"unsupported dtype {a.dtype} for {name}")
        nb = a.nbytes
        header[name] = {"dtype": dt, "shape": list(a.shape),
                        "data_offsets": [offset, offset + nb]}
        offset += nb
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for name in order:
            f.write(tensors[name].tobytes())


# --------------------------------------------------------------- dequant ----
def dequant(raw, qs, O, I, bits):
    """Packed U8 + per-row scales -> f32 [O,I]. Mirrors glm.c matmul decode."""
    if bits >= 5:
        q = raw.view(np.int8).reshape(O, I).astype(np.float32)
        return q * qs[:, None]
    if bits >= 3:
        b = raw.reshape(O, (I + 1) // 2)
        q = np.empty((O, b.shape[1] * 2), np.int16)
        q[:, 0::2] = (b & 15).astype(np.int16) - 8
        q[:, 1::2] = (b >> 4).astype(np.int16) - 8
        return q[:, :I].astype(np.float32) * qs[:, None]
    b = raw.reshape(O, (I + 3) // 4)
    q = np.empty((O, b.shape[1] * 4), np.int16)
    for k in range(4):
        q[:, k::4] = ((b >> (2 * k)) & 3).astype(np.int16) - 2
    return q[:, :I].astype(np.float32) * qs[:, None]


def quantize(w, bits, int2_variant="defective"):
    """f32 [O,I] -> (packed U8 1-D, F32 scales). Same dispatch as the
    converter: <=2 int2, <=4 int4-nibble (bits=3 uses the int4 packing with
    qmax=3 - same disk bytes as int4, engine-compatible), else int8.
    int2_variant selects the bits<=2 path: 'defective' (default, byte-for-byte
    unchanged from every existing caller) is the shipped quant_int2; 'fitted'
    opts into the repaired quant_int2_fitted (see convert_fp8_to_int4.py)."""
    if bits <= 2:
        return INT2_QUANTIZERS[int2_variant](w, bits)
    if bits <= 4:
        return quant_int4(w, bits)
    return quant_int8(w, bits)


# ----------------------------------------------------------------- usage ----
def load_usage(path):
    """.fa_usage histogram -> {(layer, expert): count}. '#' lines skipped."""
    usage = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            l, e, c = line.split()
            usage[(int(l), int(e))] = usage.get((int(l), int(e)), 0) + int(c)
    return usage


# ------------------------------------------------------------------ misc ----
def parse_layers(spec):
    """'3-27,40' -> {3,...,27,40}. Empty/None -> empty set."""
    out = set()
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


ENGINE_NAMES = ("glm", "glm_m5max", "ili")


def engine_busy():
    """True if the inference engine is running (never disturb a live bench)."""
    for name in ENGINE_NAMES:
        try:
            if subprocess.run(["pgrep", "-x", name], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0:
                return True
        except FileNotFoundError:  # no pgrep: fail safe (assume busy)
            return True
    return False


def require_idle(allow_busy, what):
    if allow_busy:
        return
    if engine_busy():
        sys.stderr.write(
            f"REFUSING to start {what}: an engine process ({'/'.join(ENGINE_NAMES)}) "
            "is running and this task does heavy disk I/O on the same SSD.\n"
            "Wait for the bench to finish, or pass --allow-busy for a tiny "
            "throttled smoke only.\n")
        sys.exit(2)


class RateLimiter:
    """Byte-rate limiter: sleep so cumulative reads never exceed mb_s."""

    def __init__(self, mb_s):
        self.rate = float(mb_s) * 1e6
        self.t0 = time.monotonic()
        self.bytes = 0

    def account(self, n):
        self.bytes += n
        if self.rate <= 0:
            return
        ahead = self.bytes / self.rate - (time.monotonic() - self.t0)
        if ahead > 0:
            time.sleep(ahead)
