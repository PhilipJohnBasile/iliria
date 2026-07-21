#!/usr/bin/env python3
"""Build the mode-1.5 OFFLINE ENTROPY-ENCODED container from the shipped
int4 container (registration: c/bench-m5max/mode15-lossless-pipeline-
the pipeline registration -- bit-exact lossless, canonical Huffman per the codec
race, kill-shots KS1+KS2 both PASSED -> build authorized).

Reads the existing int4 container shard by shard and writes a NEW
container directory (the source is never touched):
  - every ROUTED EXPERT weight tensor (model.layers.{L}.mlp.experts.{E}.
    {gate,up,down}_proj.weight, first_k_dense_replace <= L < n_layers,
    matching quant_container.EXPERT_RE/routed_experts exactly, same gate
    build_mixed_container.py uses) is entropy-encoded: canonical Huffman,
    ONE codebook per tensor (built from that tensor's own O*I symbol
    histogram -- codec_row.h's own definition of "projection": "a tensor
    (one gate/up/down projection of one expert) is O rows ... each row
    encoded independently against a codebook shared across the whole
    projection"), every row independently byte-aligned so a decoder gets
    O(1) random-access via the stored row_offsets index. Bitstream format
    (bit-reversed canonical codewords, LSB-first packing, wide-LUT
    decodability) matches c/codec_row_huff.h EXACTLY -- see
    tools/mode15_container.py's module docstring and
    tools/mode15_cross_check.c for the independent proof this needs NO
    deviation from the actual (future) Metal/C decode path.
  - everything else -- non-expert weights (dense/attention/router/norms/
    embed/lm_head/MTP head/DSA indexer), and EVERY tensor's own per-row
    F32 `.qs` scales (expert or not) -- is copied byte-identical
    (codec_row.h: "Per-row F32 quantization scales are UNCHANGED, stay
    raw"). A shard containing no routed-expert weight is copied verbatim
    in full, same as build_mixed_container.py's own convention.

The engine needs NO extra metadata to tell int4 from mode-1.5 bytes at the
BYTE-COUNT level the way it does for int8/int4/int2 (glm.c's expert_load):
an entropy-coded blob's size is DATA DEPENDENT, so (deliberately) it will
NOT satisfy quant_container.packed_nbytes() for any bit width -- a legacy/
non-mode15-aware reader fails loudly (cannot infer bits) rather than
silently misinterpreting the bytes. A mode-1.5-aware reader (future,
separately-reviewed engine work -- explicitly OUT OF SCOPE here) is
expected to consult this outdir's mode15-container-manifest.json instead.

HARDENING (registration amendment, reproduced literally): per-block
checksums, a container version header, and per-shard provenance (source
shard sha, tool commit) are ALL implemented, as three distinct mechanisms
(see tools/mode15_container.py's module docstring for why they are kept
separate, not one field doing triple duty):
  - per-block CRC32 (+ a whole-tensor CRC32) inside every encoded tensor
    blob -- tools/mode15_container.py's parse_tensor_blob;
  - "format"/"schema_version" in mode15-container-manifest.json (container
    level) + a 4-byte magic/version tag on every tensor blob (tensor
    level);
  - per-shard ledger entries recording the SOURCE shard's sha256 and this
    run's git commit (+dirty flag) -- see the manifest["shards"][fn] dict
    built in main()'s per-shard loop below.

Resumable (complete output shards are skipped after a STRUCTURAL +
CHECKSUM re-verification, not just a byte-count/existence check -- Huffman
output size is data-dependent so byte-count alone cannot prove
completeness the way build_mixed_container.py's int2/int4 check can),
writes are tmp+rename, disk-space-checked upfront (conservatively: assumes
NO savings for planning, since actual per-tensor size is unknown until
encoded), rate-limited reads (--max-mb-s), and refuses to run while the
engine is alive unless --allow-busy -- same operational discipline as
build_mixed_container.py (read that file for the pattern this follows).

Usage (ONLY when the engine is idle):
  nice -n 19 python3 c/tools/encode_mode15_container.py \
      --indir  /path/to/models/GLM-5.2-int4-with-int8-mtp \
      --outdir /path/to/models/GLM-5.2-mode15-huffman \
      --limit 3   # smoke: only the first 3 (sorted) shards this run

Verify an existing build without writing (structural + checksum re-scan,
no --indir reads):
  python3 c/tools/encode_mode15_container.py --indir ... --outdir ... \
      --verify-only
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
import quant_container as qc          # noqa: E402
import mode15_container as m15        # noqa: E402

GB = 1e9
MANIFEST_NAME = "mode15-container-manifest.json"
COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
              "generation_config.json", ".fa_usage", ".fa_usage.coding")
_HASH_CHUNK = 1 << 20


# ------------------------------------------------------------- provenance ----
def sha256_file(path, limiter=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(_HASH_CHUNK)
            if not buf:
                break
            if limiter:
                limiter.account(len(buf))
            h.update(buf)
    return h.hexdigest()


def git_provenance(repo_dir):
    try:
        commit = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"],
                                     capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except Exception as e:  # pragma: no cover - defensive, not expected on this machine
        return f"unknown ({e})", True


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------- manifest ----
def load_or_init_manifest(outdir, indir, cfg, rows_per_block):
    path = os.path.join(outdir, MANIFEST_NAME)
    commit, dirty = git_provenance(os.path.dirname(TOOLS_DIR))
    if os.path.exists(path):
        with open(path) as f:
            manifest = json.load(f)
        if manifest.get("format") != m15.CONTAINER_FORMAT:
            raise SystemExit(f"{path}: unrecognized format {manifest.get('format')!r}, "
                              f"expected {m15.CONTAINER_FORMAT!r}")
        if manifest.get("model_config") != cfg:
            raise SystemExit(f"{path}: model_config does not match --indir config.json "
                              "(mixing two different source models in one outdir)")
        if manifest.get("rows_per_block") != rows_per_block:
            raise SystemExit(f"{path}: rows_per_block={manifest.get('rows_per_block')} on disk, "
                              f"but --rows-per-block={rows_per_block} requested -- would produce "
                              "an inconsistent checksum-block layout across shards")
        manifest["tool_commit"] = commit
        manifest["tool_commit_dirty"] = dirty
        manifest["updated_at"] = now_iso()
        return manifest
    return {
        "schema_version": m15.CONTAINER_SCHEMA_VERSION,
        "format": m15.CONTAINER_FORMAT,
        "codec": "canonical-huffman",
        "rows_per_block": rows_per_block,
        "source_model_dir": os.path.realpath(indir),
        "model_config": cfg,
        "tool_commit": commit,
        "tool_commit_dirty": dirty,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "shards": {},
    }


def save_manifest(outdir, manifest):
    path = os.path.join(outdir, MANIFEST_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ------------------------------------------------------------ shard plan ----
def is_routed_expert_weight(name, cfg):
    match = qc.EXPERT_RE.match(name)
    if not match:
        return False
    layer = int(match.group(1))
    return cfg["first_dense"] <= layer < cfg["n_layers"]


def shard_tensor_plan(header, cfg):
    """{name: "encode"|"verbatim"} for every tensor in a SOURCE (raw int4)
    shard's header. A routed-expert weight tensor that ISN'T actually int4
    (should never happen in this container -- the MTP head's experts are
    int8 and live at a layer index >= n_layers, already excluded by the
    layer-range gate -- but checked defensively) falls back to verbatim +
    a loud warning rather than mis-encoding it.

    MUST be called with the SOURCE shard's header, never an already-
    encoded mode-1.5 OUTPUT header: an encoded tensor's byte count is, by
    design, data-dependent and will not satisfy infer_bits' int4 formula
    (that is the whole point -- see the module docstring's "no extra
    metadata" paragraph), so running this on an output header would
    misclassify every already-encoded tensor as "not int4". verify_outdir
    below deliberately does NOT use this function for that reason -- it
    uses is_routed_expert_weight() directly, which only needs the tensor
    NAME (invariant whether the bytes are still raw or already encoded)."""
    plan = {}
    for name, meta in header.items():
        if name.endswith(".qs") or not is_routed_expert_weight(name, cfg):
            plan[name] = "verbatim"
            continue
        proj = qc.EXPERT_RE.match(name).group(3)
        O, I = qc.expert_shape(cfg, proj)
        o0, o1 = meta["data_offsets"]
        nbytes = o1 - o0
        try:
            bits = qc.infer_bits(nbytes, O, I)
        except ValueError:
            bits = None
        if bits != 4:
            print(f"[encode_mode15] WARNING: {name} is not int4 (bits={bits}) -- "
                  "copying verbatim instead of entropy-encoding it (unexpected for "
                  "this container; investigate before trusting the ratio numbers)",
                  file=sys.stderr)
            plan[name] = "verbatim"
            continue
        plan[name] = "encode"
    return plan


# --------------------------------------------------------- completeness ----
def shard_complete_mode15(outp, source_header, plan, cfg):
    """True if `outp` exists, parses, has EXACTLY the source's tensor name
    set, every verbatim tensor matches the source's byte count, and every
    encoded tensor's blob is structurally valid with PASSING checksums and
    the expected (O,I) shape. Unlike build_mixed_container.py's byte-count
    check, an encoded tensor's size is data-dependent, so completeness
    here is proven via the blob's own whole-tensor CRC32, not a size
    match -- a stronger check than "the file exists and looks the right
    size", by design."""
    if not os.path.exists(outp):
        return False, "missing"
    try:
        out_header, out_data_start = qc.st_read_header(outp)
    except Exception as e:
        return False, f"header parse failed: {e}"
    if set(out_header) != set(source_header):
        return False, "tensor name set mismatch"
    for name, meta in out_header.items():
        o0, o1 = meta["data_offsets"]
        nbytes = o1 - o0
        if plan[name] == "verbatim":
            expect = source_header[name]["data_offsets"]
            expect_n = expect[1] - expect[0]
            if nbytes != expect_n:
                return False, f"{name}: verbatim size {nbytes} != expected {expect_n}"
            continue
        # encoded: structural + checksum parse
        proj = qc.EXPERT_RE.match(name).group(3)
        O, I = qc.expert_shape(cfg, proj)
        try:
            with open(outp, "rb") as f:
                f.seek(out_data_start + o0)
                blob = f.read(nbytes)
            m15.parse_tensor_blob(blob, expect_O=O, expect_I=I, verify_checksums=True)
        except (m15.Mode15FormatError, m15.Mode15ChecksumError) as e:
            return False, f"{name}: {e}"
    return True, "ok"


# --------------------------------------------------------------- build one --
def build_shard(src_path, out_tmp, header, data_start, plan, cfg, limiter, rows_per_block):
    """Rewrite one shard: encode routed-expert weights, copy everything
    else verbatim. Returns (tensors_dict for a final paranoia re-check is
    done by the caller via shard_complete_mode15, raw_expert_bytes,
    encoded_expert_bytes)."""
    tensors = {}
    raw_expert_bytes = 0
    encoded_expert_bytes = 0
    for name, meta in header.items():
        o0, o1 = meta["data_offsets"]
        entry = {"path": src_path, "dtype": meta["dtype"], "shape": meta["shape"],
                 "offset": data_start + o0, "nbytes": o1 - o0}
        if plan[name] == "verbatim":
            tensors[name] = qc.st_read_tensor(entry, limiter)
            continue
        proj = qc.EXPERT_RE.match(name).group(3)
        O, I = qc.expert_shape(cfg, proj)
        raw = qc.st_read_tensor(entry, limiter)
        nibbles = m15.unpack_nibbles(raw, O, I)
        blob = m15.make_tensor_blob(nibbles, rows_per_block=rows_per_block)
        tensors[name] = np.frombuffer(blob, dtype=np.uint8).copy()
        raw_expert_bytes += entry["nbytes"]
        encoded_expert_bytes += len(blob)
    qc.st_save(out_tmp, tensors)
    return raw_expert_bytes, encoded_expert_bytes


# -------------------------------------------------------------- verify-only --
def verify_outdir(outdir, cfg, rows_per_block, limit_shards=0):
    """Structural + checksum re-scan of an EXISTING mode-1.5 outdir. Reads
    only --outdir (never --indir, matching build_mixed_container.py's own
    --verify-only convention), so tensors are classified as encode-vs-
    verbatim purely by NAME (is_routed_expert_weight), NOT by re-running
    shard_tensor_plan's byte-size inference against the output header --
    an already-encoded tensor's size is data-dependent BY DESIGN and will
    never satisfy infer_bits' int4 formula, so doing that would
    misclassify every real encoded tensor as "not int4" (a real bug this
    function used to have -- see git history / tests/
    test_encode_mode15_container.py's test_rebuilds_a_shard_whose_output_
    was_deleted, which caught it)."""
    shards = sorted(f for f in os.listdir(outdir) if f.endswith(".safetensors"))
    if limit_shards:
        shards = shards[:limit_shards]
    n_ok, n_bad = 0, 0
    for fn in shards:
        path = os.path.join(outdir, fn)
        try:
            header, data_start = qc.st_read_header(path)
        except Exception as e:
            print(f"    BAD {fn}: header parse failed: {e}", file=sys.stderr)
            n_bad += 1
            continue
        bad_here = []
        for name, meta in header.items():
            if name.endswith(".qs") or not is_routed_expert_weight(name, cfg):
                continue
            proj = qc.EXPERT_RE.match(name).group(3)
            O, I = qc.expert_shape(cfg, proj)
            o0, o1 = meta["data_offsets"]
            with open(path, "rb") as f:
                f.seek(data_start + o0)
                blob = f.read(o1 - o0)
            try:
                m15.parse_tensor_blob(blob, expect_O=O, expect_I=I, verify_checksums=True)
            except (m15.Mode15FormatError, m15.Mode15ChecksumError) as e:
                bad_here.append((name, str(e)))
        if bad_here:
            n_bad += 1
            print(f"    BAD {fn}: {len(bad_here)} tensor(s) failed", file=sys.stderr)
            for name, err in bad_here[:5]:
                print(f"      {name}: {err}", file=sys.stderr)
        else:
            n_ok += 1
    print(f"[encode_mode15] verify: {n_ok}/{len(shards)} shards OK, {n_bad} bad")
    return n_bad == 0


# ----------------------------------------------------------------- main -----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--indir", required=True, help="existing pre-quantized int4 container")
    ap.add_argument("--outdir", required=True, help="NEW mode-1.5 container dir (created)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N shards this run (testing; resumable)")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="stop when free space drops below this (default 20)")
    ap.add_argument("--max-mb-s", type=float, default=0.0,
                    help="read throttle in MB/s (0 = unlimited)")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if the engine is alive (tiny tests only)")
    ap.add_argument("--rows-per-block", type=int, default=m15.DEFAULT_ROWS_PER_BLOCK,
                    help=f"checksum block granularity in rows (default "
                         f"{m15.DEFAULT_ROWS_PER_BLOCK}, matching the Metal decoder's "
                         "threadgroup row-batch size)")
    ap.add_argument("--verify-only", action="store_true",
                    help="structurally re-verify an existing outdir (checksums incl.); no writes")
    a = ap.parse_args(argv)

    indir = os.path.realpath(a.indir)
    outdir = os.path.realpath(a.outdir)
    if indir == outdir:
        raise SystemExit("outdir must differ from indir (source is never touched)")
    cfg = qc.load_config(indir)

    if a.verify_only:
        ok = verify_outdir(outdir, cfg, a.rows_per_block, limit_shards=a.limit)
        return 0 if ok else 1

    qc.require_idle(a.allow_busy, "mode-1.5 container encode")
    os.makedirs(outdir, exist_ok=True)

    lock = open(os.path.join(outdir, ".build.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another builder is already using this outdir")

    manifest = load_or_init_manifest(outdir, indir, cfg, a.rows_per_block)
    save_manifest(outdir, manifest)

    shards = sorted(f for f in os.listdir(indir) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"no shards in {indir}")

    # upfront disk check: CONSERVATIVE (assumes zero savings -- actual
    # encoded expert bytes are data-dependent and unknown until encoded;
    # real experts compress to ~0.74x per the census, so this over-
    # reserves, which is the safe direction for a free-space check).
    remaining = 0
    for fn in shards:
        outp = os.path.join(outdir, fn)
        if not os.path.exists(outp):
            remaining += os.path.getsize(os.path.join(indir, fn))
    free = shutil.disk_usage(outdir).free
    print(f"[encode_mode15] {len(shards)} shards, conservative remaining-to-write estimate "
          f"{remaining / GB:.1f} GB (assumes no savings), free {free / GB:.1f} GB")
    if free < remaining + a.min_free_gb * GB:
        raise SystemExit(f"not enough disk: need up to {remaining / GB:.1f} GB + "
                          f"{a.min_free_gb:.0f} GB margin, have {free / GB:.1f} GB free")

    limiter = qc.RateLimiter(a.max_mb_s) if a.max_mb_s > 0 else None
    done_this_run = 0
    t0 = time.monotonic()
    for i, fn in enumerate(shards):
        if a.limit and done_this_run >= a.limit:
            print(f"[encode_mode15] --limit {a.limit} reached (rerun to resume)")
            break
        if shutil.disk_usage(outdir).free < a.min_free_gb * GB:
            raise SystemExit(f"free space below {a.min_free_gb} GB - rerun to resume")

        src = os.path.join(indir, fn)
        outp = os.path.join(outdir, fn)
        header, data_start = qc.st_read_header(src)
        plan = shard_tensor_plan(header, cfg)

        complete, reason = shard_complete_mode15(outp, header, plan, cfg)
        if complete:
            continue  # already done (verified structurally, not just present)
        if os.path.exists(outp):
            print(f"[encode_mode15] {fn}: existing output incomplete/invalid "
                  f"({reason}) -- rebuilding", file=sys.stderr)

        shard_t0 = time.monotonic()
        source_sha = sha256_file(src, limiter)
        tmp = outp + ".tmp"
        raw_bytes, enc_bytes = build_shard(src, tmp, header, data_start, plan, cfg,
                                           limiter, a.rows_per_block)
        ok, why = shard_complete_mode15(tmp, header, plan, cfg)
        if not ok:
            os.remove(tmp)
            raise SystemExit(f"{fn}: just-written shard failed self-verification "
                              f"({why}) -- aborting")
        got_size = os.path.getsize(tmp)
        os.replace(tmp, outp)
        done_this_run += 1
        dt = time.monotonic() - shard_t0

        n_encoded = sum(1 for v in plan.values() if v == "encode")
        manifest["shards"][fn] = {
            "status": "done",
            "source_sha256": source_sha,
            "tool_commit": manifest["tool_commit"],
            "tool_commit_dirty": manifest["tool_commit_dirty"],
            "n_tensors": len(header),
            "n_encoded_tensors": n_encoded,
            "raw_expert_bytes": raw_bytes,
            "encoded_expert_bytes": enc_bytes,
            "shard_bytes": got_size,
            "build_seconds": dt,
            "built_at": now_iso(),
        }
        save_manifest(outdir, manifest)
        ratio = (enc_bytes / raw_bytes) if raw_bytes else float("nan")
        print(f"[encode_mode15 {i + 1}/{len(shards)}] {fn}: {n_encoded} expert tensors "
              f"encoded (ratio {ratio:.4f}), {got_size / GB:.2f} GB, {dt:.1f}s", flush=True)

    for fn in COPY_FILES:
        src = os.path.join(indir, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(outdir, fn)):
            shutil.copy2(src, os.path.join(outdir, fn))

    total_raw = sum(s.get("raw_expert_bytes", 0) for s in manifest["shards"].values())
    total_enc = sum(s.get("encoded_expert_bytes", 0) for s in manifest["shards"].values())
    overall_ratio = (total_enc / total_raw) if total_raw else float("nan")
    n_done = len(manifest["shards"])
    elapsed = time.monotonic() - t0
    print(f"[encode_mode15] {n_done}/{len(shards)} shards recorded done; "
          f"aggregate expert-tensor ratio so far: {overall_ratio:.4f} "
          f"({total_raw / GB:.2f} GB -> {total_enc / GB:.2f} GB); "
          f"{elapsed:.1f}s this invocation")
    if n_done < len(shards):
        print(f"[encode_mode15] INTERRUPTED/PARTIAL at {n_done}/{len(shards)} shards "
              "(rerun to resume)")
        return 0
    print("[encode_mode15] DONE: all shards present; run --verify-only for a full re-check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
