#!/usr/bin/env python3
"""Mode-1.5 pipeline Gate G1: BIT-EXACTNESS.

Registered (the Mode-15 lossless-pipeline registration):
    "G1 bit-exactness (encoded->decoded == original bytes, EVERY tensor,
     hash-verified)"
Hardening amendment (same file, "external portfolio-review amendments"):
    "Production hardening added to G1: block checksums, container
     versioning, corruption/truncation recovery, deterministic fallback --
     not just happy-path bit-exactness."

This gate certifies a mode-1.5 OUTDIR (built by tools/encode_mode15_
container.py) two ways, both always run:

  1. BIT-EXACTNESS SCAN (needs --indir, the source int4 container): for
     every routed-expert weight tensor in every gated shard, an
     INDEPENDENT decode-side reference (tools/mode15_container.py's
     decode_blob_to_packed_bytes -- parses the tensor blob, verifies its
     whole-tensor + every per-block CRC32, canonically rebuilds the Huffman
     LUT from the stored length table, decodes EVERY row) reconstructs the
     packed int4 bytes and compares them BYTE-FOR-BYTE against the actual
     source tensor. This does not trust encode_mode15_container.py's own
     bookkeeping (manifest "status":"done" entries) -- it independently
     re-scans whatever shards are physically present in --outdir, exactly
     the "container vs runtime are separate certifications" discipline
     the compression campaign's registration itself calls for (case-D's
     gate1_integrity.sh's own header makes the same point about not
     rubber-stamping the builder's self-check).
  2. CORRUPTION/TRUNCATION RECOVERY (synthetic, no --indir/--outdir I/O):
     proves the format's hardening actually works -- flip one payload
     byte -> the (block or whole-tensor) CRC32 catches it; flip one
     row_offsets/length-table byte -> the whole-tensor CRC32 catches it;
     truncate at several cut points -> fails closed with a clear error,
     never a partial/garbage decode. Always run, regardless of which real
     container --outdir points at, so a G1 report always demonstrates the
     hardening properties it certifies.

Usage (the real gate, on a built mode-1.5 outdir):
  python3 gate_m15_g1.py --outdir /path/to/GLM-5.2-mode15-huffman \
      [--indir /path/to/GLM-5.2-int4-with-int8-mtp]  # or set $ILI_MODEL_DIR \
      [--limit N]      # gate only the first N (sorted) shards present in outdir

Selftest (synthetic container + corruption tests only; no real paths):
  python3 gate_m15_g1.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(SCRIPT_DIR)          # c/bench-m5max
C_DIR = os.path.dirname(BENCH_DIR)               # c
TOOLS_DIR = os.path.join(C_DIR, "tools")
sys.path.insert(0, TOOLS_DIR)
import quant_container as qc                      # noqa: E402
import mode15_container as m15                     # noqa: E402
import encode_mode15_container as enc               # noqa: E402

DEFAULT_INDIR = os.environ.get("ILI_MODEL_DIR", "GLM-5.2-int4-with-int8-mtp")
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ---------------------------------------------------------- bit-exactness ----
def gate_shard_bit_exact(src_path, out_path, cfg):
    """Every routed-expert tensor in `out_path`: decode (verifying every
    checksum along the way) and compare byte-for-byte vs `src_path`.
    Returns (n_checked, n_rows_checked, n_blocks_checked, raw_bytes,
    encoded_bytes, failures[list of dict])."""
    src_header, src_data_start = qc.st_read_header(src_path)
    out_header, out_data_start = qc.st_read_header(out_path)

    failures = []
    n_checked = 0
    n_rows = 0
    n_blocks = 0
    raw_bytes = 0
    encoded_bytes = 0

    for name, out_meta in out_header.items():
        if name.endswith(".qs") or not enc.is_routed_expert_weight(name, cfg):
            continue
        if name not in src_header:
            failures.append({"tensor": name, "error": "present in outdir, missing in indir"})
            continue
        proj = qc.EXPERT_RE.match(name).group(3)
        O, I = qc.expert_shape(cfg, proj)

        so0, so1 = src_header[name]["data_offsets"]
        with open(src_path, "rb") as f:
            f.seek(src_data_start + so0)
            src_bytes = f.read(so1 - so0)

        oo0, oo1 = out_meta["data_offsets"]
        with open(out_path, "rb") as f:
            f.seek(out_data_start + oo0)
            blob = f.read(oo1 - oo0)

        n_checked += 1
        raw_bytes += len(src_bytes)
        encoded_bytes += len(blob)
        try:
            parsed = m15.parse_tensor_blob(blob, expect_O=O, expect_I=I, verify_checksums=True)
            got_packed = m15.decode_blob_to_packed_bytes(blob, expect_O=O, expect_I=I)
        except (m15.Mode15FormatError, m15.Mode15ChecksumError) as e:
            failures.append({"tensor": name, "error": f"blob parse/checksum failed: {e}"})
            continue
        n_rows += O
        n_blocks += parsed["n_blocks"]
        if got_packed != src_bytes:
            # localize: which rows differ, for a useful failure record
            src_nib = m15.unpack_nibbles(np.frombuffer(src_bytes, dtype=np.uint8), O, I)
            got_nib = m15.unpack_nibbles(np.frombuffer(got_packed, dtype=np.uint8), O, I)
            bad_rows = np.nonzero((src_nib != got_nib).any(axis=1))[0]
            failures.append({
                "tensor": name, "error": "decoded bytes != source bytes",
                "n_bad_rows": int(len(bad_rows)),
                "first_bad_rows": [int(r) for r in bad_rows[:10]],
            })
    return n_checked, n_rows, n_blocks, raw_bytes, encoded_bytes, failures


def gate_bit_exactness(indir, outdir, limit_shards=0):
    cfg = qc.load_config(indir)
    shards = sorted(f for f in os.listdir(outdir) if f.endswith(".safetensors"))
    if limit_shards:
        shards = shards[:limit_shards]
    if not shards:
        return {"status": "FAIL_NO_SHARDS", "shards_checked": [], "n_tensors_checked": 0,
                "n_rows_checked": 0, "n_blocks_checked": 0, "raw_bytes": 0, "encoded_bytes": 0,
                "failures": [{"error": f"no *.safetensors present in {outdir}"}]}

    per_shard = []
    all_failures = []
    total_tensors = total_rows = total_blocks = total_raw = total_enc = 0
    for fn in shards:
        src_path = os.path.join(indir, fn)
        out_path = os.path.join(outdir, fn)
        if not os.path.exists(src_path):
            all_failures.append({"shard": fn, "error": f"source shard missing: {src_path}"})
            continue
        n_t, n_r, n_b, raw_b, enc_b, fails = gate_shard_bit_exact(src_path, out_path, cfg)
        total_tensors += n_t
        total_rows += n_r
        total_blocks += n_b
        total_raw += raw_b
        total_enc += enc_b
        for f in fails:
            f["shard"] = fn
        all_failures.extend(fails)
        per_shard.append({
            "shard": fn, "n_tensors_checked": n_t, "n_rows_checked": n_r,
            "n_blocks_checked": n_b, "raw_bytes": raw_b, "encoded_bytes": enc_b,
            "ratio": (enc_b / raw_b) if raw_b else None, "n_failures": len(fails),
        })
        print(f"[gate_m15_g1] {fn}: {n_t} tensors, {n_r} rows, {n_b} blocks checked "
              f"bit-exact vs source, ratio={(enc_b / raw_b) if raw_b else float('nan'):.4f}, "
              f"failures={len(fails)}", flush=True)

    status = "PASS" if not all_failures and total_tensors > 0 else "FAIL"
    return {
        "status": status,
        "shards_checked": shards,
        "per_shard": per_shard,
        "n_tensors_checked": total_tensors,
        "n_rows_checked": total_rows,
        "n_blocks_checked": total_blocks,
        "raw_bytes": total_raw,
        "encoded_bytes": total_enc,
        "ratio": (total_enc / total_raw) if total_raw else None,
        "failures": all_failures[:50],
        "n_failures": len(all_failures),
    }


# ---------------------------------------------------- corruption/truncation --
def _synthetic_tensor(rng, O=512, I=2048, target_h=2.9):
    """A real-shape-like (down_proj: O=6144/I=2048 scaled down for a fast
    gate run) synthetic tensor at the census's own target entropy."""
    def gauss(sigma):
        x = np.arange(16) - 7.5
        p = np.exp(-(x * x) / (2 * sigma * sigma))
        return p / p.sum()

    def H(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    lo, hi = 0.06, 200.0
    mid = hi
    for _ in range(60):
        mid = (lo + hi) / 2
        p = gauss(mid)
        h = H(p)
        if abs(h - target_h) < 1e-4:
            break
        if h < target_h:
            lo = mid
        else:
            hi = mid
    p = gauss(mid)
    return rng.choice(16, size=(O, I), p=p).astype(np.uint8)


def run_corruption_tests(rows_per_block=m15.DEFAULT_ROWS_PER_BLOCK, seed=20260718):
    """Synthetic (no real-container I/O) proof that the hardening actually
    catches corruption and fails closed on truncation. Returns a dict of
    named sub-tests -> {"pass": bool, "detail": str}."""
    rng = np.random.default_rng(seed)
    nib = _synthetic_tensor(rng)
    O, I = nib.shape
    blob = bytearray(m15.make_tensor_blob(nib, rows_per_block=rows_per_block))
    results = {}

    # 1. flip a payload byte -> caught (whole-tensor, and independently the
    #    specific block via verify_block())
    b1 = bytearray(blob)
    b1[-1] ^= 0xFF
    try:
        m15.parse_tensor_blob(bytes(b1))
        results["flip_payload_byte"] = {"pass": False, "detail": "NOT CAUGHT (bug)"}
    except m15.Mode15ChecksumError as e:
        results["flip_payload_byte"] = {"pass": True, "detail": f"whole-tensor CRC32 caught it: {e}"}

    # 2. flip a byte inside row_offsets -> whole-tensor CRC32 catches it
    b2 = bytearray(blob)
    b2[m15.TENSOR_HEADER_LEN + m15.LENGTHS_BYTES + 4] ^= 0x01
    try:
        m15.parse_tensor_blob(bytes(b2))
        results["flip_row_offsets_byte"] = {"pass": False, "detail": "NOT CAUGHT (bug)"}
    except (m15.Mode15ChecksumError, m15.Mode15FormatError) as e:
        results["flip_row_offsets_byte"] = {"pass": True, "detail": f"caught: {type(e).__name__}: {e}"}

    # 3. truncate at several cut points -> fails closed (never a partial decode)
    cuts = [0, 1, m15.TENSOR_HEADER_LEN, len(blob) // 2, len(blob) - 1]
    trunc_results = []
    all_caught = True
    for cut in cuts:
        try:
            m15.parse_tensor_blob(bytes(blob[:cut]))
            trunc_results.append({"cut_at": cut, "caught": False})
            all_caught = False
        except m15.Mode15FormatError as e:
            trunc_results.append({"cut_at": cut, "caught": True, "detail": str(e)})
    results["truncation_fails_closed"] = {"pass": all_caught, "detail": trunc_results}

    # 4. block-level localization: corrupt ONE block's payload, confirm
    #    verify_block() flags exactly that block (proves per-block
    #    checksums add real localization, not just piggybacking on the
    #    whole-tensor check)
    b4 = bytearray(blob)
    parsed = m15.parse_tensor_blob(bytes(b4))
    n_blocks = parsed["n_blocks"]
    target_block = n_blocks // 2
    r0 = target_block * rows_per_block
    payload_off = (m15.TENSOR_HEADER_LEN + m15.LENGTHS_BYTES
                   + (parsed["O"] + 1) * 4 + n_blocks * 4)
    byte_idx = payload_off + int(parsed["row_offsets"][r0])
    b4[byte_idx] ^= 0xFF
    flagged = [b for b in range(n_blocks) if not m15.verify_block(bytes(b4), b)]
    results["block_checksum_localizes_corruption"] = {
        "pass": flagged == [target_block],
        "detail": f"corrupted block {target_block}, flagged {flagged}",
    }

    # 5. bad magic -> rejected
    b5 = bytearray(blob)
    b5[0] ^= 0xFF
    try:
        m15.parse_tensor_blob(bytes(b5))
        results["bad_magic_rejected"] = {"pass": False, "detail": "NOT CAUGHT (bug)"}
    except m15.Mode15FormatError as e:
        results["bad_magic_rejected"] = {"pass": True, "detail": str(e)}

    overall = all(r["pass"] for r in results.values())
    return {"status": "PASS" if overall else "FAIL", "tests": results}


# ------------------------------------------------------------- selftest ------
def selftest():
    """Synthetic bit-exactness (a tiny in-memory int4 container, encode
    then decode-vs-source) + the corruption/truncation battery -- no real
    --indir/--outdir reads. Mirrors tests/test_encode_mode15_container.py's
    fixture but inline here so the gate is self-verifying without needing
    the test suite."""
    rng = np.random.default_rng(20260718)
    O, I = 64, 6144
    nib = _synthetic_tensor(rng, O=O, I=I)
    src_packed = m15.pack_nibbles(nib, I).tobytes()
    blob = m15.make_tensor_blob(nib)
    got_packed = m15.decode_blob_to_packed_bytes(blob, expect_O=O, expect_I=I)
    bitexact = {
        "status": "PASS" if got_packed == src_packed else "FAIL",
        "n_tensors_checked": 1, "n_rows_checked": O,
        "detail": "synthetic single-tensor decode-vs-source",
    }
    corruption = run_corruption_tests()
    overall = bitexact["status"] == "PASS" and corruption["status"] == "PASS"
    return {
        "gate": "gate_m15_g1", "mode": "selftest",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bit_exactness": bitexact, "corruption_truncation": corruption,
        "overall": "PASS" if overall else "FAIL",
    }


# ------------------------------------------------------------------ main -----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", help="mode-1.5 encoded container to gate")
    ap.add_argument("--indir", default=DEFAULT_INDIR,
                    help="source int4 container (bit-exactness reference)")
    ap.add_argument("--limit", type=int, default=0,
                    help="gate only the first N (sorted) shards present in --outdir; "
                         "0 = every shard present")
    ap.add_argument("--out", default=None, help="JSON report path (default: timestamped, results/)")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic bit-exactness + corruption/truncation tests only; "
                         "no --indir/--outdir reads")
    a = ap.parse_args(argv)

    if a.selftest:
        report = selftest()
        out_path = a.out or os.path.join(
            DEFAULT_RESULTS_DIR,
            f"gate_m15_g1-selftest-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[gate_m15_g1] selftest bit-exactness: {report['bit_exactness']['status']}")
        print(f"[gate_m15_g1] selftest corruption/truncation: {report['corruption_truncation']['status']}")
        print(f"[gate_m15_g1] report written -> {out_path}")
        print(f"GATE G1 (selftest): {report['overall']}")
        return 0 if report["overall"] == "PASS" else 1

    if not a.outdir:
        ap.error("--outdir is required unless --selftest")

    indir = os.path.realpath(a.indir)
    outdir = os.path.realpath(a.outdir)

    report = {
        "gate": "gate_m15_g1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "indir": indir,
        "outdir": outdir,
        "limit_shards": a.limit,
    }

    bitexact = gate_bit_exactness(indir, outdir, limit_shards=a.limit)
    report["bit_exactness"] = bitexact
    print(f"[gate_m15_g1] bit-exactness: {bitexact['n_tensors_checked']} tensors / "
          f"{bitexact['n_rows_checked']} rows / {bitexact['n_blocks_checked']} blocks "
          f"across {len(bitexact['shards_checked'])} shard(s) -> {bitexact['status']} "
          f"(ratio {bitexact['ratio']:.4f})" if bitexact["ratio"] else
          f"[gate_m15_g1] bit-exactness: {bitexact['status']}")

    corruption = run_corruption_tests()
    report["corruption_truncation"] = corruption
    print(f"[gate_m15_g1] corruption/truncation battery: {corruption['status']}")
    for name, r in corruption["tests"].items():
        print(f"    {name}: {'PASS' if r['pass'] else 'FAIL'}")

    overall = bitexact["status"] == "PASS" and corruption["status"] == "PASS"
    report["overall"] = "PASS" if overall else "FAIL"

    out_path = a.out or os.path.join(
        DEFAULT_RESULTS_DIR,
        f"gate_m15_g1-{os.path.basename(outdir.rstrip('/'))}-"
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[gate_m15_g1] report written -> {out_path}")
    print(f"GATE G1: {report['overall']}" + ("  (fail-closed)" if not overall else ""))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
