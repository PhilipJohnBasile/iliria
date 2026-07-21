#!/usr/bin/env python3
"""Build the mixed-precision (int4/int2) container from an allocation manifest.

Reads the existing pre-quantized container shard by shard and writes a NEW
container directory (the source is never touched):
  - expert tensors listed in the manifest's `int2` set: dequantize the stored
    int4 (exact per-row scales), requantize to int2 with the shipping
    converter math (identical to glm.c pack_int2), write new data + .qs;
  - everything else (kept experts, dense/attention, router, norms, MTP head,
    DSA indexer shards, embed/lm_head): byte-identical pass-through - shards
    containing no demoted tensor are copied verbatim.

The engine needs no new metadata: it infers int8/int4/int2 per tensor from
the packed byte count at load time (glm.c expert_load) and reallocates its
expert slots per size, so the mixed container is drop-in.

Resumable (complete output shards are skipped; writes are tmp+rename),
disk-space-checked upfront and per shard, rate-limited reads (--max-mb-s),
and refuses to run while the engine is alive unless --allow-busy.

Usage (ONLY when the engine is idle):
  pgrep -x glm >/dev/null && echo "engine busy" || \
    nice -n 19 python3 c/tools/build_mixed_container.py \
      --indir  /path/to/models/GLM-5.2-int4-with-int8-mtp \
      --outdir /path/to/models/GLM-5.2-mixed-int4-int2 \
      --manifest c/bench-m5max/container-20260715/manifest-270.json

Verify an existing build without writing:
  python3 c/tools/build_mixed_container.py --indir ... --outdir ... \
      --manifest ... --verify-only
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_container as qc

GB = 1e9
COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
              "generation_config.json", ".fa_usage", ".fa_usage.coding")


def load_manifest(path):
    with open(path) as f:
        m = json.load(f)
    if m.get("version") != 1 or m.get("default_bits") != 4:
        raise SystemExit(f"unsupported manifest {path}")
    m["int2_set"] = {tuple(k) for k in m["int2"]}
    return m


def demoted_in_shard(header, int2_set, cfg):
    """Names of weight tensors in this shard that the manifest demotes."""
    out = []
    for name in header:
        match = qc.EXPERT_RE.match(name)
        if not match:
            continue
        layer, expert = int(match.group(1)), int(match.group(2))
        if not (cfg["first_dense"] <= layer < cfg["n_layers"]):
            continue
        if (layer, expert) in int2_set:
            out.append(name)
    return out


def shard_out_bytes(path, header, demoted, cfg):
    """ESTIMATED output size of a shard (input minus int4->int2 weight
    savings; the re-serialized header length can differ by a few bytes)."""
    size = os.path.getsize(path)
    for name in demoted:
        proj = qc.EXPERT_RE.match(name).group(3)
        O, I = qc.expert_shape(cfg, proj)
        size -= qc.packed_nbytes(O, I, 4) - qc.packed_nbytes(O, I, 2)
    return size


def expected_tensor_sizes(header, demoted, cfg):
    """{name: nbytes} the output shard must contain (exact, per tensor)."""
    expected = {}
    demoted_set = set(demoted)
    for name, meta in header.items():
        nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
        if name in demoted_set:
            proj = qc.EXPERT_RE.match(name).group(3)
            O, I = qc.expert_shape(cfg, proj)
            nbytes = qc.packed_nbytes(O, I, 2)
        expected[name] = nbytes  # .qs sizes are unchanged (per-row F32)
    return expected


def shard_complete(path, expected):
    """True if `path` parses and every tensor has the expected byte count."""
    if not os.path.exists(path):
        return False
    try:
        header, _ = qc.st_read_header(path)
    except Exception:
        return False
    got = {n: m["data_offsets"][1] - m["data_offsets"][0] for n, m in header.items()}
    return got == expected


def copy_throttled(src, dst, limiter, chunk=32 << 20):
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            buf = fi.read(chunk)
            if not buf:
                break
            if limiter:
                limiter.account(len(buf))
            fo.write(buf)


def rebuild_shard(path, out_tmp, header, data_start, demoted, cfg, limiter, int2_variant="defective"):
    """Rewrite one shard, requantizing the demoted expert weights to int2."""
    demoted_set = set(demoted)
    tensors = {}
    for name, meta in header.items():
        o0, o1 = meta["data_offsets"]
        entry = {"path": path, "dtype": meta["dtype"], "shape": meta["shape"],
                 "offset": data_start + o0, "nbytes": o1 - o0}
        base = name[:-3] if name.endswith(".qs") else None
        if base in demoted_set:
            continue  # scales are regenerated with the requantized weights
        data = qc.st_read_tensor(entry, limiter)
        if name in demoted_set:
            proj = qc.EXPERT_RE.match(name).group(3)
            O, I = qc.expert_shape(cfg, proj)
            qs_meta = header[name + ".qs"]
            qs_entry = {"path": path, "dtype": qs_meta["dtype"],
                        "shape": qs_meta["shape"],
                        "offset": data_start + qs_meta["data_offsets"][0],
                        "nbytes": qs_meta["data_offsets"][1] - qs_meta["data_offsets"][0]}
            qs = qc.st_read_tensor(qs_entry, limiter)
            bits = qc.infer_bits(entry["nbytes"], O, I)
            if bits == 2:
                tensors[name] = data          # already int2: pass through
                tensors[name + ".qs"] = qs
                continue
            w = qc.dequant(data, qs, O, I, bits)
            packed, scales = qc.quantize(w, 2, int2_variant=int2_variant)
            tensors[name] = packed
            tensors[name + ".qs"] = scales.astype(np.float32)
        else:
            tensors[name] = data
    qc.st_save(out_tmp, tensors)


def verify_container(outdir, manifest, cfg):
    """Header-only scan: every routed expert weight has the manifest's size."""
    index = qc.st_scan(outdir)
    int2_set = manifest["int2_set"]
    seen, bad = set(), []
    for name, entry in index.items():
        match = qc.EXPERT_RE.match(name)
        if not match:
            continue
        layer, expert, proj = int(match.group(1)), int(match.group(2)), match.group(3)
        if not (cfg["first_dense"] <= layer < cfg["n_layers"]):
            continue
        seen.add((layer, expert))
        O, I = qc.expert_shape(cfg, proj)
        bits = 2 if (layer, expert) in int2_set else 4
        if entry["nbytes"] != qc.packed_nbytes(O, I, bits):
            bad.append((name, entry["nbytes"], qc.packed_nbytes(O, I, bits)))
        if name + ".qs" not in index or index[name + ".qs"]["nbytes"] != O * 4:
            bad.append((name + ".qs", "missing-or-wrong-size", O * 4))
    expected = set(qc.routed_experts(cfg))
    missing = expected - seen
    for name, got, want in bad[:10]:
        print(f"    BAD {name}: {got} bytes, expected {want}", file=sys.stderr)
    if missing:
        print(f"    MISSING {len(missing)} experts (first: {sorted(missing)[:3]})",
              file=sys.stderr)
    return not bad and not missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--indir", required=True, help="existing pre-quantized container")
    ap.add_argument("--outdir", required=True, help="NEW container dir (created)")
    ap.add_argument("--manifest", required=True, help="allocation manifest JSON")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N shards this run (testing; resumable)")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="stop when free space drops below this (default 20)")
    ap.add_argument("--max-mb-s", type=float, default=0.0,
                    help="read throttle in MB/s (0 = unlimited)")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if the engine is alive (tiny tests only)")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify an existing outdir against the manifest; no writes")
    ap.add_argument("--int2-quantizer", choices=sorted(qc.INT2_QUANTIZERS),
                    default=os.environ.get("ILI_INT2_QUANTIZER", "defective"),
                    help="which int2 path requantizes demoted experts (default "
                         "'defective', the shipped quant_int2; 'fitted' opts into "
                         "the repaired quant_int2_fitted). Falls back to "
                         "ILI_INT2_QUANTIZER when unset.")
    a = ap.parse_args(argv)
    if a.int2_quantizer not in qc.INT2_QUANTIZERS:
        ap.error(f"--int2-quantizer must be one of {sorted(qc.INT2_QUANTIZERS)}")

    indir = os.path.realpath(a.indir)
    outdir = os.path.realpath(a.outdir)
    if indir == outdir:
        raise SystemExit("outdir must differ from indir (source is never touched)")
    cfg = qc.load_config(indir)
    manifest = load_manifest(a.manifest)
    if manifest.get("model_config") and manifest["model_config"] != cfg:
        raise SystemExit("manifest model_config does not match --indir config.json")

    if a.verify_only:
        ok = verify_container(outdir, manifest, cfg)
        print(f"[build] verify: {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1

    qc.require_idle(a.allow_busy, "mixed-container build")
    os.makedirs(outdir, exist_ok=True)

    # anti-duplicate lock (two builders on one outdir corrupt each other)
    import fcntl
    lock = open(os.path.join(outdir, ".build.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another builder is already using this outdir")

    shards = sorted(f for f in os.listdir(indir) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"no shards in {indir}")

    # upfront disk check: estimated remaining output bytes vs free space
    plan = []  # (fn, header, data_start, demoted_names, expected_sizes, est_bytes)
    remaining = 0
    for fn in shards:
        path = os.path.join(indir, fn)
        header, data_start = qc.st_read_header(path)
        demoted = demoted_in_shard(header, manifest["int2_set"], cfg)
        expected = expected_tensor_sizes(header, demoted, cfg)
        est = shard_out_bytes(path, header, demoted, cfg)
        plan.append((fn, header, data_start, demoted, expected, est))
        if not shard_complete(os.path.join(outdir, fn), expected):
            remaining += est
    free = shutil.disk_usage(outdir).free
    print(f"[build] {len(shards)} shards, estimated output "
          f"{sum(p[5] for p in plan) / GB:.1f} GB, remaining to write "
          f"{remaining / GB:.1f} GB, free {free / GB:.1f} GB")
    if free < remaining + a.min_free_gb * GB:
        raise SystemExit(
            f"not enough disk: need {remaining / GB:.1f} GB + "
            f"{a.min_free_gb:.0f} GB margin, have {free / GB:.1f} GB free")

    limiter = qc.RateLimiter(a.max_mb_s) if a.max_mb_s > 0 else None
    done = 0
    for i, (fn, header, data_start, demoted, expected, est) in enumerate(plan):
        if a.limit and done >= a.limit:
            print(f"[build] --limit {a.limit} reached (rerun to resume)")
            break
        if shutil.disk_usage(outdir).free < a.min_free_gb * GB:
            raise SystemExit(f"free space below {a.min_free_gb} GB - rerun to resume")
        src = os.path.join(indir, fn)
        outp = os.path.join(outdir, fn)
        if shard_complete(outp, expected):
            continue  # complete from a previous run (writes are tmp+rename)
        tmp = outp + ".tmp"
        if demoted:
            rebuild_shard(src, tmp, header, data_start, demoted, cfg, limiter,
                          int2_variant=a.int2_quantizer)
            action = f"requantized {len(demoted)} tensors -> int2 ({a.int2_quantizer})"
        else:
            copy_throttled(src, tmp, limiter)
            action = "verbatim copy"
        if not shard_complete(tmp, expected):
            os.remove(tmp)
            raise SystemExit(f"{fn}: written shard failed the per-tensor "
                             "size check - aborting")
        got = os.path.getsize(tmp)
        os.replace(tmp, outp)
        done += 1
        print(f"[build {i + 1}/{len(plan)}] {fn}: {action} "
              f"({got / GB:.2f} GB)", flush=True)

    for fn in COPY_FILES:
        src = os.path.join(indir, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(outdir, fn)):
            shutil.copy2(src, os.path.join(outdir, fn))
    with open(os.path.join(outdir, "mixed-container-manifest.json"), "w") as f:
        json.dump({**{k: v for k, v in manifest.items() if k != "int2_set"},
                   "int2_quantizer": a.int2_quantizer}, f)

    built = sorted(f for f in os.listdir(outdir) if f.endswith(".safetensors"))
    if len(built) == len(shards):
        ok = verify_container(outdir, manifest, cfg)
        print(f"[build] DONE: {len(built)} shards, verify {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    print(f"[build] INTERRUPTED at {len(built)}/{len(shards)} shards (rerun to resume)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
