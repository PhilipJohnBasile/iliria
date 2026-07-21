#!/usr/bin/env python3
"""INT4 expert entropy census: is a lossless-compressed cold tier worth building?

Samples routed experts stratified across early/mid/late layer bands (every
sampled expert contributes all three projections, so gate/up/down are balanced
by construction) from the shipped int4 container and measures, per tensor:

  (a) global symbol entropy H(Q) of the 4-bit codes, bits/weight;
  (b) tile entropy at 4KB/16KB/64KB packed-byte blocks (weighted mean of
      per-block entropies -- what a per-block static model could achieve);
  (c) conditional entropy given the row's scale bin (8 within-tensor quantile
      bins of the per-row F32 scale) and, via the tensor = projection
      identity, conditional entropy by projection;

and REAL stored lossless ratios per independently-decodable block:

  - canonical Huffman over the 16 nibble symbols (implemented here; stored
    size = exact codeword bits + 8 B canonical length table + 4 B index);
  - zlib level 6 as raw DEFLATE (wbits=-15; stored = stream + 4 B index);
  - LZMA2 preset 6 raw (FORMAT_RAW) as the zstd-like ceiling reference,
    measured on a deterministic 1-in-N block subsample (stored = stream
    + 4 B index; ratio over sampled blocks only, flagged in the output);
  - static rANS with a per-block 12-bit quantized frequency table. The
    per-block stored size is the QUANTIZED-TABLE CROSS-ENTROPY ESTIMATE
    (ceil(sum n_s*log2(4096/f_s) / 8) + 24 B table + 4 B state + 4 B index),
    VALIDATED against the real rANS codec implemented in this file
    (encode+decode roundtrip on a deterministic block subsample; the
    summary reports the max estimate-vs-real deviation). This is the
    'arithmetic-coding estimate + overhead, validated' option the study
    allows, and which of the two was used is recorded here and in the
    summary JSON.

All stored ratios include the random-access index and frequency-table
metadata. Per-row F32 scales stay raw (they are 0.22% of an expert); the
per-EXPERT stored ratio therefore is (3*compressed payload + raw scales) /
(3*raw payload + raw scales).

Reads go through quant_container.RateLimiter and are hard-clamped to
<= 100 MB/s (documented bench-coexistence exception: ~2% of the quality
bench's I/O, loglik-quality-insensitive). Run under nice -n 19. Refuses to
start while the engine is alive unless --allow-busy (the documented
exception for THIS census).

Usage:
  nice -n 19 python3 c/tools/measure_expert_entropy.py \
      --model /path/to/models/GLM-5.2-int4-with-int8-mtp \
      --outdir c/bench-m5max/new-math-20260715 --allow-busy
"""

from __future__ import annotations

import argparse
import heapq
import json
import lzma
import math
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_container as qc

M_BITS = 12                 # rANS quantized frequency precision
M_TOTAL = 1 << M_BITS
RANS_L = 1 << 16            # rANS lower renormalization bound (16-bit words)
HUFF_TABLE_BYTES = 8        # 16 canonical code lengths x 4 bits
RANS_TABLE_BYTES = 24       # 16 frequencies x 12 bits
RANS_STATE_BYTES = 4        # final encoder state
INDEX_BYTES = 4             # u32 per block in the random-access index
BLOCK_SIZES = (4096, 16384, 65536)
BANDS = (("early", 3, 27), ("mid", 28, 52), ("late", 53, 77))
N_SCALE_BINS = 8


# ------------------------------------------------------------- entropy ----
def entropy_bits(counts) -> float:
    """Shannon entropy in bits/symbol of a count vector."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def block_counts(nibbles: np.ndarray, block_nibbles: int) -> np.ndarray:
    """(n_blocks, 16) symbol counts over contiguous blocks (tail included)."""
    n = len(nibbles)
    n_full = n // block_nibbles
    rows = []
    if n_full:
        body = nibbles[: n_full * block_nibbles]
        idx = (
            np.arange(n_full, dtype=np.int64).repeat(block_nibbles) * 16
            + body.astype(np.int64)
        )
        rows.append(np.bincount(idx, minlength=n_full * 16).reshape(n_full, 16))
    tail = nibbles[n_full * block_nibbles:]
    if len(tail):
        rows.append(np.bincount(tail, minlength=16).reshape(1, 16))
    return np.vstack(rows) if rows else np.zeros((0, 16), dtype=np.int64)


def tile_entropy(counts: np.ndarray) -> float:
    """Size-weighted mean of per-block entropies, bits/symbol."""
    n_per = counts.sum(axis=1).astype(np.float64)
    total = n_per.sum()
    if total <= 0:
        return 0.0
    h = np.array([entropy_bits(c) for c in counts])
    return float((h * n_per).sum() / total)


# ------------------------------------------------------------- huffman ----
def huffman_code_lengths(counts) -> np.ndarray:
    """Huffman code lengths (int array of 16; 0 = symbol absent).

    Single-symbol blocks get a 1-bit code (a real stored bit per symbol,
    decodable). Max depth for 16 symbols is 15, so lengths fit 4 bits.
    """
    lens = np.zeros(16, dtype=np.int64)
    heap = [(int(c), s, None) for s, c in enumerate(counts) if c > 0]
    if not heap:
        return lens
    if len(heap) == 1:
        lens[heap[0][1]] = 1
        return lens
    heapq.heapify(heap)
    next_id = 16
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, (a[0] + b[0], next_id, (a, b)))
        next_id += 1
    stack = [(heap[0], 0)]
    while stack:
        (c, s, kids), depth = stack.pop()
        if kids is None:
            lens[s] = depth
        else:
            stack.append((kids[0], depth + 1))
            stack.append((kids[1], depth + 1))
    return lens


def canonical_codes(lens: np.ndarray):
    """Canonical Huffman codebook {symbol: (code, length)} from lengths."""
    order = sorted((int(l), s) for s, l in enumerate(lens) if l > 0)
    codes = {}
    code = 0
    prev_len = 0
    for length, sym in order:
        code <<= length - prev_len
        codes[sym] = (code, length)
        code += 1
        prev_len = length
    return codes


def huffman_encode(symbols: np.ndarray, lens: np.ndarray) -> bytes:
    """Canonical-Huffman bitstream (MSB first), zero-padded to a byte."""
    codes = canonical_codes(lens)
    acc = 0
    nbits = 0
    out = bytearray()
    for s in symbols:
        code, length = codes[int(s)]
        acc = (acc << length) | code
        nbits += length
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def huffman_decode(data: bytes, lens: np.ndarray, n: int) -> np.ndarray:
    """Decode n symbols from a canonical-Huffman bitstream."""
    decode = {v: s for s, v in canonical_codes(lens).items()}
    out = np.empty(n, dtype=np.uint8)
    code = 0
    length = 0
    k = 0
    for byte in data:
        for bit in range(7, -1, -1):
            code = (code << 1) | ((byte >> bit) & 1)
            length += 1
            sym = decode.get((code, length))
            if sym is not None:
                out[k] = sym
                k += 1
                if k == n:
                    return out
                code = 0
                length = 0
    raise ValueError("truncated Huffman stream")


def huffman_stored_bytes(counts) -> int:
    """Exact stored bytes for one block: codeword bits + table + index."""
    lens = huffman_code_lengths(counts)
    bits = int((np.asarray(counts, dtype=np.int64) * lens).sum())
    return (bits + 7) // 8 + HUFF_TABLE_BYTES + INDEX_BYTES


def huffman_payload_bits(counts_matrix: np.ndarray) -> np.ndarray:
    """Vector of codeword-bit totals for an (n_blocks, 16) count matrix."""
    return np.array(
        [int((c * huffman_code_lengths(c)).sum()) for c in counts_matrix],
        dtype=np.int64,
    )


# ---------------------------------------------------------------- rANS ----
def quantize_freqs(counts) -> np.ndarray:
    """Quantize counts to a frequency table summing to M_TOTAL (present
    symbols get >= 1)."""
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    if total <= 0:
        raise ValueError("empty block")
    f = np.zeros(16, dtype=np.int64)
    present = c > 0
    f[present] = np.maximum(1, np.rint(c[present] / total * M_TOTAL).astype(np.int64))
    diff = M_TOTAL - int(f.sum())
    # repair the sum on the largest buckets, never dropping below 1
    order = np.argsort(-f)
    i = 0
    while diff != 0:
        s = order[i % len(order)]
        if f[s] > 0:
            step = 1 if diff > 0 else -1
            if f[s] + step >= 1:
                f[s] += step
                diff -= step
        i += 1
        if i > 100000:
            raise RuntimeError("frequency quantization failed to converge")
    return f


def rans_encode(symbols: np.ndarray, freqs: np.ndarray) -> bytes:
    """Static rANS encode (32-bit state, 16-bit renorm words)."""
    cum = np.zeros(17, dtype=np.int64)
    cum[1:] = np.cumsum(freqs)
    if cum[16] != M_TOTAL:
        raise ValueError("frequency table must sum to M_TOTAL")
    words = []
    x = RANS_L
    x_max_base = (RANS_L >> M_BITS) << 16
    for s in symbols[::-1]:
        s = int(s)
        f = int(freqs[s])
        x_max = x_max_base * f
        while x >= x_max:
            words.append(x & 0xFFFF)
            x >>= 16
        x = ((x // f) << M_BITS) | ((x % f) + int(cum[s]))
    out = bytearray()
    out += int(x).to_bytes(4, "little")
    for w in reversed(words):
        out += int(w).to_bytes(2, "little")
    return bytes(out)


def rans_decode(data: bytes, freqs: np.ndarray, n: int) -> np.ndarray:
    """Static rANS decode of n symbols."""
    cum = np.zeros(17, dtype=np.int64)
    cum[1:] = np.cumsum(freqs)
    slot2sym = np.zeros(M_TOTAL, dtype=np.uint8)
    for s in range(16):
        slot2sym[cum[s]:cum[s + 1]] = s
    x = int.from_bytes(data[:4], "little")
    pos = 4
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        slot = x & (M_TOTAL - 1)
        s = int(slot2sym[slot])
        out[i] = s
        x = int(freqs[s]) * (x >> M_BITS) + slot - int(cum[s])
        while x < RANS_L and pos + 2 <= len(data):
            x = (x << 16) | int.from_bytes(data[pos:pos + 2], "little")
            pos += 2
    return out


def rans_estimate_bytes(counts) -> int:
    """Cross-entropy of the block under its own quantized table + overhead."""
    c = np.asarray(counts, dtype=np.float64)
    f = quantize_freqs(counts)
    present = c > 0
    bits = float((c[present] * np.log2(M_TOTAL / f[present])).sum())
    return math.ceil(bits / 8) + RANS_TABLE_BYTES + RANS_STATE_BYTES + INDEX_BYTES


def rans_real_bytes(symbols: np.ndarray, counts) -> int:
    """Real encoded bytes (same overhead accounting as the estimate)."""
    f = quantize_freqs(counts)
    payload = rans_encode(symbols, f)
    return len(payload) + RANS_TABLE_BYTES + INDEX_BYTES


# ---------------------------------------------------------------- zlib ----
def zlib_stored_bytes(block: bytes) -> int:
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    return len(c.compress(block) + c.flush()) + INDEX_BYTES


_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 6}]


def lzma_stored_bytes(block: bytes) -> int:
    return len(lzma.compress(block, format=lzma.FORMAT_RAW,
                             filters=_LZMA_FILTERS)) + INDEX_BYTES


# ------------------------------------------------------------- census ----
def band_of(layer: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= layer <= hi:
            return name
    raise ValueError(f"layer {layer} outside routed bands")


def sample_experts(cfg, n_total: int, seed: int):
    """Stratified sample: n_total split evenly across the three bands."""
    rng = np.random.default_rng(seed)
    per_band = [n_total // 3] * 3
    for i in range(n_total - sum(per_band)):
        per_band[2 - i] += 1  # remainder goes to late, then mid
    chosen = []
    for (name, lo, hi), n_band in zip(BANDS, per_band):
        pool = [(l, e) for l in range(lo, hi + 1) for e in range(cfg["n_experts"])]
        idx = rng.choice(len(pool), size=min(n_band, len(pool)), replace=False)
        chosen.extend(pool[i] for i in sorted(idx))
    return chosen


def unpack_nibbles(raw: np.ndarray) -> np.ndarray:
    """Packed int4 bytes -> nibble symbols 0..15 in storage order
    (low nibble = even element, matching glm.c/quant_container dequant)."""
    out = np.empty(len(raw) * 2, dtype=np.uint8)
    out[0::2] = raw & 15
    out[1::2] = raw >> 4
    return out


def scale_bin_conditional_entropy(nibbles: np.ndarray, qs: np.ndarray,
                                  O: int) -> float:
    """H(Q | row-scale quantile bin), bits/symbol, over N_SCALE_BINS bins."""
    per_row = len(nibbles) // O
    edges = np.quantile(qs, np.linspace(0, 1, N_SCALE_BINS + 1)[1:-1])
    row_bin = np.searchsorted(edges, qs, side="right").astype(np.int64)
    sym_bin = np.repeat(row_bin, per_row) * 16 + nibbles.astype(np.int64)
    counts = np.bincount(sym_bin, minlength=N_SCALE_BINS * 16).reshape(-1, 16)
    return tile_entropy(counts)


def measure_tensor(raw: np.ndarray, qs: np.ndarray, O: int, I: int,
                   lzma_every: int, rans_checks: list, want_rans_check: bool,
                   rng: np.random.Generator) -> dict:
    nibbles = unpack_nibbles(raw)
    packed = raw.tobytes()
    global_counts = np.bincount(nibbles, minlength=16)
    out = {
        "nbytes": len(packed),
        "h_global": entropy_bits(global_counts),
        "h_cond_scalebin": scale_bin_conditional_entropy(nibbles, qs, O),
    }
    for bs in BLOCK_SIZES:
        counts = block_counts(nibbles, bs * 2)
        out[f"h_tile_{bs}"] = tile_entropy(counts)

        huff_bits = huffman_payload_bits(counts)
        huff_total = int(((huff_bits + 7) // 8).sum()
                         + len(counts) * (HUFF_TABLE_BYTES + INDEX_BYTES))
        out[f"huff_bytes_{bs}"] = huff_total

        rans_total = 0
        for c in counts:
            rans_total += rans_estimate_bytes(c)
        out[f"rans_bytes_{bs}"] = rans_total

        zlib_total = 0
        for off in range(0, len(packed), bs):
            zlib_total += zlib_stored_bytes(packed[off:off + bs])
        out[f"zlib_bytes_{bs}"] = zlib_total

        lz_bytes = lz_raw = 0
        block_starts = list(range(0, len(packed), bs))
        for bi in range(0, len(block_starts), lzma_every):
            blk = packed[block_starts[bi]:block_starts[bi] + bs]
            lz_bytes += lzma_stored_bytes(blk)
            lz_raw += len(blk)
        out[f"lzma_bytes_{bs}"] = lz_bytes
        out[f"lzma_raw_{bs}"] = lz_raw

        if want_rans_check and len(counts) > 0:
            bi = int(rng.integers(len(counts) - 1)) if len(counts) > 1 else 0
            blk_syms = nibbles[bi * bs * 2:(bi + 1) * bs * 2]
            c = np.bincount(blk_syms, minlength=16)
            est = rans_estimate_bytes(c)
            # real stream already carries the 4-byte final state; add the same
            # table + index overhead the estimate charges
            real = rans_real_bytes(blk_syms, c)
            f = quantize_freqs(c)
            round_trip = rans_decode(rans_encode(blk_syms, f), f, len(blk_syms))
            if not np.array_equal(round_trip, blk_syms):
                raise AssertionError("rANS roundtrip failed on a census block")
            rans_checks.append((bs, est, real, len(blk_syms)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-experts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--max-mb-s", type=float, default=100.0,
                    help="read throttle; hard-clamped to <= 100 MB/s")
    ap.add_argument("--lzma-every", type=int, default=16,
                    help="measure LZMA on every Nth block (deterministic)")
    ap.add_argument("--rans-check-every", type=int, default=24,
                    help="roundtrip-verify real rANS on 1 block/size every Nth expert")
    ap.add_argument("--limit", type=int, default=0, help="smoke: stop after N experts")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run alongside the bench (documented census exception; "
                         "throttled + nice only)")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild census-summary.json from the CSVs already "
                         "in --outdir; zero shard reads")
    a = ap.parse_args(argv)

    if a.report_only:
        return report_only(a)

    qc.require_idle(a.allow_busy, "int4 entropy census")
    if a.allow_busy and qc.engine_busy():
        print("[census] engine alive: proceeding under the documented exception "
              "(<=100 MB/s throttle, ~2% of bench I/O, nice -n 19 expected)",
              flush=True)
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass

    mb_s = min(a.max_mb_s, 100.0) if a.max_mb_s > 0 else 100.0
    if mb_s != a.max_mb_s:
        print(f"[census] clamped throttle to {mb_s} MB/s", flush=True)

    cfg = qc.load_config(a.model)
    index = qc.st_scan(a.model)
    chosen = sample_experts(cfg, a.n_experts, a.seed)
    if a.limit > 0:
        chosen = chosen[:a.limit]

    # group reads by shard file (gate tensor's shard) to reduce seek churn
    def gate_path(key):
        name = f"model.layers.{key[0]}.mlp.experts.{key[1]}.gate_proj.weight"
        return index[name]["path"]
    chosen.sort(key=lambda k: (gate_path(k), k))

    est_bytes = len(chosen) * qc.expert_total_bytes(cfg, 4)
    print(f"[census] {len(chosen)} experts sampled (seed {a.seed}), "
          f"~{est_bytes/1e9:.2f} GB to read at <={mb_s:.0f} MB/s", flush=True)

    limiter = qc.RateLimiter(mb_s)
    rng = np.random.default_rng(a.seed + 1)
    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    rans_checks: list = []
    t0 = time.monotonic()
    for i, (layer, expert) in enumerate(chosen):
        for proj in qc.PROJS:
            name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
            w_entry = index.get(name)
            qs_entry = index.get(name + ".qs")
            if w_entry is None or qs_entry is None:
                raise SystemExit(f"missing tensor {name}")
            O, I = qc.expert_shape(cfg, proj)
            bits = qc.infer_bits(w_entry["nbytes"], O, I)
            if bits != 4:
                raise SystemExit(f"{name}: expected int4, found int{bits}")
            raw = qc.st_read_tensor(w_entry, limiter)
            qs = qc.st_read_tensor(qs_entry, limiter)
            m = measure_tensor(raw, qs, O, I, a.lzma_every, rans_checks,
                               want_rans_check=(i % a.rans_check_every == 0),
                               rng=rng)
            m.update({"layer": layer, "expert": expert, "band": band_of(layer),
                      "proj": proj.split("_")[0]})
            rows.append(m)
        if (i + 1) % 16 == 0 or i + 1 == len(chosen):
            dt = time.monotonic() - t0
            eta = dt / (i + 1) * (len(chosen) - i - 1)
            print(f"[census] {i+1}/{len(chosen)} experts, "
                  f"{limiter.bytes/1e9:.2f} GB read, {dt:.0f}s, ETA {eta/60:.1f} min",
                  flush=True)

    write_outputs(a, cfg, rows, rans_checks, limiter.bytes)
    return 0


# ------------------------------------------------------------- reports ----
def stored_ratio(row: dict, coder: str, bs: int) -> float:
    key = f"ratio_{coder}_{bs}"
    if key in row:  # report-only rows carry precomputed ratios
        return row[key]
    if coder == "lzma":
        raw = row[f"lzma_raw_{bs}"]
        return row[f"lzma_bytes_{bs}"] / raw if raw else float("nan")
    return row[f"{coder}_bytes_{bs}"] / row["nbytes"]


def pctile_table(rows, coders):
    """{(band, proj, coder, bs): (p50, p90, p99, n)} over per-tensor ratios."""
    out = {}
    bands = [b[0] for b in BANDS] + ["all"]
    projs = ["gate", "up", "down", "all"]
    for band in bands:
        for proj in projs:
            sel = [r for r in rows
                   if (band == "all" or r["band"] == band)
                   and (proj == "all" or r["proj"] == proj)]
            if not sel:
                continue
            for coder in coders:
                for bs in BLOCK_SIZES:
                    vals = np.array([stored_ratio(r, coder, bs) for r in sel])
                    vals = vals[~np.isnan(vals)]
                    if not len(vals):
                        continue
                    out[(band, proj, coder, bs)] = (
                        float(np.percentile(vals, 50)),
                        float(np.percentile(vals, 90)),
                        float(np.percentile(vals, 99)),
                        len(vals),
                    )
    return out


def decode_breakeven(median_expert_ratio: float) -> dict:
    """Strict break-even framing (external review): with storage read rate
    B_s and stored ratio rho, a SEPARATE decompression pass must sustain
    B_d > B_s/(1-rho) (raw output bytes/s) just to not lose time; a fully
    pipelined read||decompress still needs B_d > B_s. Three consumption
    modes decide where that requirement lands:

      mode-1 disk-compressed / RAM-expanded: I/O gain only; B_d applies on
        the MISS path (volume = B_miss raw bytes/token, SSD-relative).
      mode-2 RAM-compressed, expanded before each GEMV: adds the capacity
        gain (1/rho more experts resident); the expansion pass now runs on
        every CONSUMED expert: 600 expert-uses/token x 18.9153 MB = 11.35
        GB raw per token -> required B_d ~ 11.35 x tok/s GB/s on top of
        the same strict formula at the consumption boundary.
      mode-3 GEMV consumes compressed tiles directly: both gains, no
        separate expansion pass; the cost becomes added kernel time per
        tile. Per-raw-byte decode budget before the win evaporates:
        (1-rho)/BW, i.e. ~(1-rho)/13.3e9 s on the SSD-miss path and
        ~(1-rho)/BW_dram on the RAM path. Only mode-3 escapes the strict
        B_d requirement.
    """
    ratios = sorted({0.50, 0.625, 0.75, 0.85, round(median_expert_ratio, 4)})
    bws = (5.0, 10.0, 13.3)
    strict = []
    for r in ratios:
        for bw in bws:
            strict.append({
                "stored_ratio": r, "read_gbs": bw,
                "mode1_2_strict_min_decompress_gbs":
                    bw / (1 - r) if r < 1 else float("inf"),
                "pipelined_floor_gbs": bw,
            })
    tok_rates = (1.5, 2.0, 3.0)
    mode2 = [{"tok_s": ts,
              "required_expand_gbs_raw": 600 * 18.915328e6 * ts / 1e9}
             for ts in tok_rates]
    r = median_expert_ratio
    mode3 = {
        "per_raw_byte_decode_budget_s_at_13.3GBs_miss_path": (1 - r) / 13.3e9,
        "per_4KB_block_decode_budget_ns_at_13.3GBs_miss_path":
            (1 - r) / 13.3e9 * 4096 * 1e9,
        "per_raw_byte_decode_budget_s_at_400GBs_dram_path": (1 - r) / 400e9,
        "per_4KB_block_decode_budget_ns_at_400GBs_dram_path":
            (1 - r) / 400e9 * 4096 * 1e9,
    }
    return {
        "model": "strict serial break-even B_d > B_s/(1-rho); pipelined "
                 "floor B_d > B_s; see docstring for the three modes",
        "strict_table": strict,
        "mode2_expansion_volume": {
            "raw_gb_per_token": 600 * 18.915328e6 / 1e9,
            "required_at_tok_s": mode2,
        },
        "mode3_tile_decode_budget": mode3,
        "median_expert_stored_ratio": r,
    }


def expert_stored_bytes(rows_of_expert, coder: str, bs: int, cfg) -> float:
    """Compressed payloads + raw F32 scales for one expert."""
    scales = sum(qc.expert_shape(cfg, p)[0] * 4 for p in qc.PROJS)
    if coder == "lzma":
        payload_ratio = (sum(r[f"lzma_bytes_{bs}"] for r in rows_of_expert)
                         / max(1, sum(r[f"lzma_raw_{bs}"] for r in rows_of_expert)))
        payload = payload_ratio * sum(r["nbytes"] for r in rows_of_expert)
    else:
        payload = sum(r[f"{coder}_bytes_{bs}"] for r in rows_of_expert)
    return payload + scales


def write_outputs(a, cfg, rows, rans_checks, bytes_read=0):
    coders = ("huff", "zlib", "lzma", "rans")
    int4_expert = qc.expert_total_bytes(cfg, 4)

    csv_path = os.path.join(a.outdir, "census-tensors.csv")
    cols = ["layer", "expert", "band", "proj", "nbytes", "h_global",
            "h_cond_scalebin"]
    for bs in BLOCK_SIZES:
        cols.append(f"h_tile_{bs}")
    for coder in coders:
        for bs in BLOCK_SIZES:
            cols.append(f"ratio_{coder}_{bs}")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = [str(r["layer"]), str(r["expert"]), r["band"], r["proj"],
                    str(r["nbytes"]), f"{r['h_global']:.6f}",
                    f"{r['h_cond_scalebin']:.6f}"]
            vals += [f"{r[f'h_tile_{bs}']:.6f}" for bs in BLOCK_SIZES]
            for coder in coders:
                for bs in BLOCK_SIZES:
                    vals.append(f"{stored_ratio(r, coder, bs):.6f}")
            f.write(",".join(vals) + "\n")

    # per-expert stored bytes (payload compressed, scales raw)
    by_expert = {}
    for r in rows:
        by_expert.setdefault((r["layer"], r["expert"], r["band"]), []).append(r)
    expert_rows = []
    for (layer, expert, band), rr in sorted(by_expert.items()):
        row = {"layer": layer, "expert": expert, "band": band}
        for coder in coders:
            for bs in BLOCK_SIZES:
                row[f"bytes_{coder}_{bs}"] = expert_stored_bytes(rr, coder, bs, cfg)
        expert_rows.append(row)
    with open(os.path.join(a.outdir, "census-experts.csv"), "w") as f:
        hdr = ["layer", "expert", "band"]
        for coder in coders:
            for bs in BLOCK_SIZES:
                hdr += [f"bytes_{coder}_{bs}", f"ratio_{coder}_{bs}"]
        f.write(",".join(hdr) + "\n")
        for row in expert_rows:
            vals = [str(row["layer"]), str(row["expert"]), row["band"]]
            for coder in coders:
                for bs in BLOCK_SIZES:
                    b = row[f"bytes_{coder}_{bs}"]
                    vals += [f"{b:.0f}", f"{b/int4_expert:.6f}"]
            f.write(",".join(vals) + "\n")

    checks = [{"block_size": bs, "est_bytes": est, "real_bytes": real,
               "n_symbols": n, "rel_err": (est - real) / real}
              for bs, est, real, n in rans_checks]
    summarize(a, cfg, rows, expert_rows, checks, bytes_read)


def summarize(a, cfg, rows, expert_rows, checks, bytes_read, prior=None):
    """Build and write census-summary.json + the printed report. Pure
    post-processing: no shard reads."""
    coders = ("huff", "zlib", "lzma", "rans")
    int4_expert = qc.expert_total_bytes(cfg, 4)
    pct = pctile_table(rows, coders)
    max_dev = max((abs(c["rel_err"]) for c in checks), default=float("nan"))

    # headline: best real coder at the most favorable block size
    def med(coder, bs):
        return pct[("all", "all", coder, bs)][0]
    best = min(((med(c, bs), c, bs) for c in coders for bs in BLOCK_SIZES))
    best_med, best_coder, best_bs = best
    expert_ratios = np.array(
        [r[f"bytes_{best_coder}_{best_bs}"] / int4_expert for r in expert_rows])

    prior = prior or {}
    summary = {
        "n_experts": len(expert_rows),
        "n_tensors": len(rows),
        "seed": prior.get("seed", a.seed),
        "bytes_read_gb": bytes_read / 1e9,
        "block_sizes": list(BLOCK_SIZES),
        "coders": list(coders),
        "rans_sizing": prior.get(
            "rans_sizing",
            "quantized-table cross-entropy estimate + 24B table + 4B "
            "state + 4B index, validated by the real rANS codec on "
            f"{len(checks)} roundtrip-checked blocks "
            f"(max |est-real|/real = {max_dev:.4%})"),
        "lzma_sampling": prior.get("lzma_sampling",
                                   f"every {a.lzma_every}th block"),
        "percentiles": {
            f"{band}/{proj}/{coder}/{bs}": {
                "p50": v[0], "p90": v[1], "p99": v[2], "n": v[3]}
            for (band, proj, coder, bs), v in pct.items()},
        "headline": {
            "best_coder": best_coder,
            "best_block_size": best_bs,
            "median_tensor_stored_ratio": best_med,
            "median_expert_stored_ratio": float(np.median(expert_ratios)),
            "kill_line": "median stored ratio > 0.85 = dead",
            "verdict": "DEAD" if best_med > 0.85 else "ALIVE",
        },
        "rans_checks": checks,
    }

    summary["decode_breakeven"] = decode_breakeven(
        summary["headline"]["median_expert_stored_ratio"])
    with open(os.path.join(a.outdir, "census-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[census] stored-ratio percentiles (band/proj/coder/blocksize):")
    print("  band   proj   coder  block   p50     p90     p99     n")
    for (band, proj, coder, bs), (p50, p90, p99, n) in sorted(pct.items()):
        print(f"  {band:6s} {proj:6s} {coder:5s} {bs:6d}  {p50:.4f}  {p90:.4f}"
              f"  {p99:.4f}  {n}")
    hl = summary["headline"]
    print(f"\n[census] best real coder: {hl['best_coder']} @ {hl['best_block_size']} B"
          f" blocks; median tensor ratio {hl['median_tensor_stored_ratio']:.4f}, "
          f"median expert ratio {hl['median_expert_stored_ratio']:.4f}")
    print(f"[census] KILL LINE (>0.85 dead): {hl['verdict']}")
    print(f"[census] rANS estimate validation: {summary['rans_sizing']}")
    print(f"[census] outputs in {a.outdir}: census-tensors.csv, "
          f"census-experts.csv, census-summary.json")


def report_only(a) -> int:
    """Regenerate census-summary.json + the printed report from the CSVs
    already on disk. ZERO shard reads (only <model>/config.json)."""
    import csv as csvmod

    cfg = qc.load_config(a.model)
    rows = []
    with open(os.path.join(a.outdir, "census-tensors.csv")) as f:
        for rec in csvmod.DictReader(f):
            row = {"layer": int(rec["layer"]), "expert": int(rec["expert"]),
                   "band": rec["band"], "proj": rec["proj"],
                   "nbytes": int(rec["nbytes"])}
            for k, v in rec.items():
                if k.startswith(("h_", "ratio_")):
                    row[k] = float(v)
            rows.append(row)
    expert_rows = []
    with open(os.path.join(a.outdir, "census-experts.csv")) as f:
        for rec in csvmod.DictReader(f):
            row = {"layer": int(rec["layer"]), "expert": int(rec["expert"]),
                   "band": rec["band"]}
            for k, v in rec.items():
                if k.startswith("bytes_"):
                    row[k] = float(v)
            expert_rows.append(row)
    prior, checks, bytes_read = {}, [], 0.0
    spath = os.path.join(a.outdir, "census-summary.json")
    if os.path.exists(spath):
        with open(spath) as f:
            prior = json.load(f)
        checks = prior.get("rans_checks", [])
        bytes_read = prior.get("bytes_read_gb", 0.0) * 1e9
    print(f"[census] report-only: {len(rows)} tensors / {len(expert_rows)} "
          f"experts from {a.outdir} (no shard reads)")
    summarize(a, cfg, rows, expert_rows, checks, bytes_read, prior)
    return 0


if __name__ == "__main__":
    sys.exit(main())
