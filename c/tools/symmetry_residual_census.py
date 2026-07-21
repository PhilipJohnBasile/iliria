#!/usr/bin/env python3
"""n3 falsifier: symmetry-aligned residual coding for SwiGLU MoE experts.

docs/performance-theory.json entry n3-symmetry-aligned-residual-coding:
a SwiGLU expert's function is EXACTLY invariant under any joint permutation
P_e of its intermediate-neuron axis (the same P_e applied to gate's and up's
output ROWS and down's input COLUMNS leaves the expert's output identical).
This is a structural fact, not measured here. What IS measured here is
EXPLOITABILITY: whether real trained experts, once aligned to a common
per-layer prototype under some permutation, look enough alike that a
prototype + entropy-coded-residual representation stores smaller than
independently entropy-coding each expert (n1's method, 0.7379 median expert
stored ratio, docs/performance-theory.json n1 / c/bench-m5max/new-math-20260715).

Pipeline, per sampled layer (one early/mid/late band, BANDS below):
  1. Per-neuron signatures: for each of an expert's moe_inter intermediate
     neurons, concat(gate_proj row, up_proj row, down_proj column, and the
     neuron's own gate/up quantization scales) random-projected to ~64 dims
     (build_projection / neuron_signatures) -- cheap because it never
     materializes the O(hidden) concatenation, just sums three block-matmuls.
  2. A per-layer medoid/prototype expert is picked via a permutation-INVARIANT
     proxy (the mean signature over neurons -- order-independent by
     construction; pick_medoid), avoiding an O(N^2) all-pairs bipartite-match
     search for a true medoid (documented approximation).
  3. Every other sampled expert is aligned to the prototype via a scipy-free
     bipartite match on the (moe_inter x moe_inter) signature-distance cost
     matrix: greedy-edge construction + vectorized random-pair 2-opt
     refinement (align_bipartite) -- NOT the optimal (Jonker-Volgenant/
     Hungarian) assignment; the optimality gap against the row-min
     relaxation lower bound is measured and reported per expert, per the
     n3 entry's explicit "greedy+refine ... document optimality gap" option.
  4. Residuals are coded LOSSLESSLY in the mod-16 nibble domain
     (residual = (aligned_expert_nibble - prototype_nibble) % 16) so the
     EXACT Huffman/rANS/zlib estimators from measure_expert_entropy.py apply
     unmodified (they all just want 16-symbol nibble counts or packed
     nibble-bytes). Gate/up per-row quantization scales are neuron-indexed
     (permuted along with the rows) so their deltas are coded too; down_proj's
     per-row scale is indexed by the OUTPUT axis (not the neuron axis) so it
     is invariant to P_e and is kept raw, exactly like n1's treatment of all
     scales.
  5. Stored-ratio accounting per follower expert: residual payload +
     scale-delta payload + down's raw scale + this expert's OWN permutation
     table (fixed ceil(log2(moe_inter)) bits/entry -- a conservative,
     non-entropy-coded accounting) + the prototype's own independently-coded
     bytes amortized over all N sampled experts in the layer. Divided by the
     same raw int4 expert size n1 used (qc.expert_total_bytes(cfg, 4)).
  6. Two KILL LINES, evaluated on the pooled per-expert medians across all
     sampled layers/experts (see compute_verdict):
       KILL if aligned stored size <10% better than independent entropy
         coding AND residual density >70%.
       PROCEED if aligned exact stored size <0.70x raw int4 OR modeled
         shared-base compute saving (work fraction 0.125+r on gate/up,
         r = measured gate/up residual density) >15%.

SHARD READS ARE GATED (evening-orchestrator convention): permitted only when
MARKER_PATH exists (the evening orchestrator writes it after ABBA) or
`pgrep -x glm` is empty. Reads are additionally rate-limited (<=100 MB/s,
quant_container.RateLimiter) and run under nice -n 19, mirroring
measure_expert_entropy.py's conventions.

Usage:
  nice -n 19 python3 c/tools/symmetry_residual_census.py \
      --model /path/to/models/GLM-5.2-int4-with-int8-mtp \
      --outdir c/bench-m5max/n3-symmetry-residual-20260715

  # poll every 10 min (nice -19) until the gate opens, instead of exiting:
  python3 c/tools/symmetry_residual_census.py --model ... --outdir ... \
      --wait-for-gate

  # synthetic/local test data that isn't on the shared SSD (conscious
  # override of the gate, same convention as measure_expert_entropy.py):
  python3 c/tools/symmetry_residual_census.py --model /tmp/tiny --outdir ... \
      --allow-busy
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_container as qc
import measure_expert_entropy as ent

# ------------------------------------------------------------- constants ----
BANDS = ent.BANDS                       # reuse the same early/mid/late split as n1
MARKER_PATH = "/tmp/iliria-evening-marker-shard-reads-ok"
DEFAULT_N_EXPERTS_PER_LAYER = 64        # n3 spec: 32-64 experts/layer
DEFAULT_PROJ_DIM = 64                   # n3 spec: ~64-dim random-projected signatures
DEFAULT_REFINE_ROUNDS = 40
DEFAULT_SEED = 20260715                 # matches measure_expert_entropy.py's census seed
DEFAULT_POLL_SECONDS = 600              # "poll every 10 min"

# residual nibble codes within +-1 (mod-16 wraparound: 15 == -1) of an exact
# match; "near-zero" per the n3 falsifier spec's residual-density measurement
NEAR_ZERO_RESIDUALS = frozenset({0, 1, 15})

# scale-delta quantization step in NATURAL LOG space (~5%/code), fixed and
# data-independent so entropy across experts is comparable (see
# scale_delta_codes docstring for why data-dependent/quantile bins would be
# circular here)
SCALE_DELTA_STEP = 0.05
SCALE_DELTA_CODES = 256                 # int8-range alphabet (0..255, center=128)

# n1-lossless-entropy-coded-experts headline figures (docs/performance-theory.json
# n1 entry; c/bench-m5max/new-math-20260715/census-summary.json, commit 0af7371):
# best real coder (static rANS, 64KB blocks) median EXPERT stored ratio and
# global mean symbol entropy, over the SAME model's routed int4 experts.
N1_MEDIAN_EXPERT_STORED_RATIO = 0.7379
N1_MEAN_BITS_PER_WEIGHT = 2.9401

KILL_RELATIVE_IMPROVEMENT = 0.10        # "<10% better than independent"
KILL_RESIDUAL_DENSITY = 0.70            # "residual density >70%"
PROCEED_STORED_RATIO = 0.70             # "<0.70x raw int4"
PROCEED_COMPUTE_SAVING = 0.15           # "compute saving >15%"
COMPUTE_BASE_WORK_FRACTION = 0.125      # 1 shared prototype GEMV / 8 routed experts


# ---------------------------------------------------------- shard-read gate
def marker_present(path: str = MARKER_PATH) -> bool:
    return os.path.exists(path)


def glm_running(name: str = "glm") -> bool:
    """True if a process named `name` is alive. Fails CLOSED (assumes busy)
    if pgrep itself is missing, mirroring quant_container.engine_busy."""
    try:
        return subprocess.run(["pgrep", "-x", name], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        return True


def shard_reads_allowed(marker_path: str = MARKER_PATH, proc_name: str = "glm") -> bool:
    """Evening-orchestrator gate: shard reads of the real model are permitted
    ONLY when the marker file exists (written after ABBA) or the engine
    process isn't running. No implicit override -- see require_shard_reads_allowed
    for the explicit --allow-busy escape hatch (synthetic/test data only)."""
    return marker_present(marker_path) or not glm_running(proc_name)


def require_shard_reads_allowed(allow_busy: bool, wait: bool, poll_s: int) -> None:
    if allow_busy:
        return
    if shard_reads_allowed():
        return
    if wait:
        wait_for_gate(poll_s)
        return
    sys.stderr.write(
        f"REFUSING shard reads: no marker at {MARKER_PATH} and `pgrep -x glm` "
        "found a running engine. Per the n3 falsifier's evening-orchestrator "
        "gate, reads are permitted only once the marker appears or glm exits. "
        "Pass --wait-for-gate to poll every 10 min (nice -19) until then, or "
        "--allow-busy for a conscious override (e.g. synthetic/test data that "
        "isn't on the shared SSD).\n")
    sys.exit(2)


def wait_for_gate(poll_s: int) -> None:
    """Block (nice -19; sleeps poll_s between checks) until shard reads are
    allowed. Emits a heartbeat every poll so a long wait doesn't look
    identical to a hang."""
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    waited = 0
    while not shard_reads_allowed():
        print(f"[n3-census] gate closed (no marker, glm running); waited "
              f"{waited // 60} min, next check in {poll_s}s", flush=True)
        time.sleep(poll_s)
        waited += poll_s
    print(f"[n3-census] gate open after {waited // 60} min", flush=True)


# ------------------------------------------------------- nibble utilities ----
def unpack_matrix(raw: np.ndarray, O: int, I: int) -> np.ndarray:
    """Packed int4 bytes -> (O, I) uint8 nibble matrix (0..15), row-major.
    Assumes I even (true for this model's gate/up/down_proj shapes, all
    multiples of the hidden/moe_inter dims, both even) so every row packs to
    an exact I/2 bytes with no per-row tail padding."""
    assert I % 2 == 0, f"unpack_matrix assumes even I (got {I})"
    nib = ent.unpack_nibbles(raw)
    return nib.reshape(O, I)


def pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    """Inverse of ent.unpack_nibbles for an even-length 1-D nibble array:
    packs 2 nibbles/byte, low nibble = even index (mirrors glm.c pack_int4 /
    quant_container.quant_int4)."""
    flat = nibbles.reshape(-1)
    assert flat.size % 2 == 0
    lo = flat[0::2].astype(np.uint8)
    hi = flat[1::2].astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8)


def nibble_residual(prototype_nibbles: np.ndarray, aligned_nibbles: np.ndarray) -> np.ndarray:
    """Lossless mod-16 residual in the SAME 0..15 nibble alphabet
    ent.unpack_nibbles produces, so ent's Huffman/rANS/zlib estimators apply
    unmodified. Invertible: aligned = (residual + prototype) % 16."""
    return ((aligned_nibbles.astype(np.int16) - prototype_nibbles.astype(np.int16))
             % 16).astype(np.uint8)


def scale_delta_codes(proto_scale: np.ndarray, aligned_scale: np.ndarray) -> np.ndarray:
    """Quantize log(aligned/proto) to a FIXED, data-independent step
    (SCALE_DELTA_STEP in natural-log space, ~5%/code) into an int8-range
    alphabet (0..255, center=128=no change). Fixed bins (not per-sample
    quantile bins) are essential here: quantile-binning a distribution using
    the SAME samples that define the bins trivially yields near-uniform
    occupancy (entropy ~= log2(n_bins)) regardless of true structure -- it
    would make the scale-delta entropy measurement circular."""
    delta = (np.log(np.maximum(aligned_scale, 1e-8))
             - np.log(np.maximum(proto_scale, 1e-8)))
    code = np.clip(np.rint(delta / SCALE_DELTA_STEP), -128, 127).astype(np.int16) + 128
    return code.astype(np.uint8)


def scale_delta_stats(proto_scale: np.ndarray, aligned_scale: np.ndarray) -> dict:
    codes = scale_delta_codes(proto_scale, aligned_scale)
    counts = np.bincount(codes, minlength=SCALE_DELTA_CODES)
    h = ent.entropy_bits(counts)
    # zlib is domain-agnostic (works on any byte sequence), so it is reused
    # directly here; ent's Huffman/rANS are hardcoded to a 16-symbol nibble
    # alphabet and are reserved for the weight residuals above, where that
    # alphabet is the natural (and exactly reused) fit.
    zbytes = ent.zlib_stored_bytes(codes.tobytes())
    return {"h_bits_per_code": h, "stored_bytes": zbytes, "n": int(len(codes))}


# --------------------------------------------------------------- signatures
def build_projection(hidden: int, n_scale_feats: int, proj_dim: int, rng) -> dict:
    """One shared (3*hidden + n_scale_feats, proj_dim) Gaussian random
    projection, sliced into per-block matrices. Using the SAME blocks for
    the prototype and every candidate expert in a layer keeps signatures
    comparable. 1/sqrt(proj_dim) scaling keeps expected squared length
    invariant (standard JL-projection normalization)."""
    total_in = 3 * hidden + n_scale_feats
    R = (rng.standard_normal((total_in, proj_dim)) / math.sqrt(proj_dim)).astype(np.float32)
    return {"gate": R[:hidden], "up": R[hidden:2 * hidden],
            "down": R[2 * hidden:3 * hidden], "scale": R[3 * hidden:]}


def neuron_signatures(gate_codes: np.ndarray, up_codes: np.ndarray,
                      down_codes: np.ndarray, gate_scale: np.ndarray,
                      up_scale: np.ndarray, proj: dict) -> np.ndarray:
    """Per-neuron signature, ~proj_dim wide.

    gate_codes, up_codes: (moe_inter, hidden) CENTERED int4 codes (-8..7,
      float); one row per intermediate neuron.
    down_codes: (hidden, moe_inter) centered int4 codes; one COLUMN per
      intermediate neuron (down_proj's input axis is the neuron axis).
    gate_scale, up_scale: (moe_inter,) F32 per-neuron row scales.

    Mathematically equivalent to concatenating
    [gate_row_i, up_row_i, down_col_i, scale_feats_i] per neuron i and
    right-multiplying by one shared (3*hidden+2, D) matrix, but implemented
    as a sum of block matmuls so the (moe_inter, 3*hidden+2) concatenation is
    never materialized.

    Scale features are this expert's OWN log(gate_scale)/log(up_scale),
    z-scored over its moe_inter neurons (mean/std are permutation-invariant,
    so this needs no cross-expert pass). down_proj's per-row scale is
    indexed by the OUTPUT (hidden) axis, not the neuron axis, so it has no
    neuron-permutation-equivariant scale to contribute here (see module
    docstring) -- it is handled separately, kept raw, in the stored-ratio
    accounting.

    EQUIVARIANCE (load-bearing correctness property, see
    test_symmetry_residual_census.py): permuting the neuron axis of every
    input by pi produces exactly neuron_signatures(original)[pi] -- each
    term is independently row/column-selected by pi before being summed, so
    this holds bit-exactly (no floating-point reassociation), not just
    approximately.
    """
    def z(v):
        mu, sd = float(v.mean()), float(v.std())
        return (v - mu) / sd if sd > 1e-12 else np.zeros_like(v)

    log_gs = z(np.log(np.maximum(gate_scale, 1e-8)).astype(np.float32))
    log_us = z(np.log(np.maximum(up_scale, 1e-8)).astype(np.float32))
    scale_feats = np.stack([log_gs, log_us], axis=1).astype(np.float32)

    sig = (gate_codes @ proj["gate"] + up_codes @ proj["up"]
           + down_codes.T @ proj["down"] + scale_feats @ proj["scale"])
    return sig.astype(np.float32)


def expert_signature(tensors: dict, proj: dict) -> np.ndarray:
    """tensors: {"gate_proj"/"up_proj"/"down_proj": {"nibbles": ..., "scale": ...}}
    (raw 0..15 nibbles, as produced by unpack_matrix/load_expert_tensors)."""
    gate = tensors["gate_proj"]["nibbles"].astype(np.float32) - 8.0
    up = tensors["up_proj"]["nibbles"].astype(np.float32) - 8.0
    down = tensors["down_proj"]["nibbles"].astype(np.float32) - 8.0
    return neuron_signatures(gate, up, down, tensors["gate_proj"]["scale"],
                             tensors["up_proj"]["scale"], proj)


def pick_medoid(signatures: dict) -> int:
    """signatures: {expert_id: (moe_inter, D) array}. Returns the expert
    whose PERMUTATION-INVARIANT centroid (mean signature over neurons -- the
    mean doesn't depend on neuron order) is closest to the population's mean
    centroid. A cheap O(N*D) medoid proxy: a true medoid would need O(N^2)
    pairwise bipartite-matching costs (documented approximation, matching
    the n3 entry's "medoid-based alignment as the bipartite-matching/OT
    stand-in" framing)."""
    ids = list(signatures)
    centroids = np.stack([signatures[i].mean(axis=0) for i in ids])
    mean_centroid = centroids.mean(axis=0)
    d = ((centroids - mean_centroid) ** 2).sum(axis=1)
    return ids[int(np.argmin(d))]


# ------------------------------------------------------- bipartite matching
def pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n,D),(m,D) -> (n,m) squared Euclidean distances via
    |a-b|^2 = |a|^2+|b|^2-2a.b (one matmul, no O(n*m*D) python loop)."""
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)


def greedy_assignment(cost: np.ndarray) -> np.ndarray:
    """Greedy-edge bipartite matching: visit (row, col) pairs in ascending
    cost order, accept a pair if neither index is used yet. O(n^2 log n) for
    the sort; the acceptance loop stops as soon as all n rows are matched, so
    it does not actually walk all n^2 pairs except in an adversarial
    worst case (see test_greedy_assignment_completes_quickly_at_real_scale).
    Returns pi with pi[i] = j meaning prototype-row i is matched to
    candidate-row j (aligned_row[i] := candidate_row[pi[i]])."""
    n, m = cost.shape
    assert n == m, "expects a square (prototype x candidate) cost matrix"
    order = np.argsort(cost, axis=None, kind="stable")  # C-order flatten: idx = r*m+c
    rows = order // m
    cols = order % m
    pi = np.full(n, -1, dtype=np.int64)
    used_rows = np.zeros(n, dtype=bool)
    used_cols = np.zeros(m, dtype=bool)
    filled = 0
    for r, c in zip(rows.tolist(), cols.tolist()):
        if used_rows[r] or used_cols[c]:
            continue
        pi[r] = c
        used_rows[r] = True
        used_cols[c] = True
        filled += 1
        if filled == n:
            break
    return pi


def refine_2opt(cost: np.ndarray, pi: np.ndarray, rounds: int, rng) -> np.ndarray:
    """Vectorized local search: each round draws a random pairing of row
    indices (a random permutation split into disjoint halves) and swaps each
    pair's assigned columns wherever doing so strictly reduces summed cost.
    Pairs within a round are disjoint by construction, so every beneficial
    swap in a round can be applied at once (no read/write hazards)."""
    n = len(pi)
    pi = pi.copy()
    for _ in range(rounds):
        order = rng.permutation(n)
        if n % 2:
            order = order[:-1]
        a = order[0::2]
        b = order[1::2]
        cur = cost[a, pi[a]] + cost[b, pi[b]]
        swapped = cost[a, pi[b]] + cost[b, pi[a]]
        do_swap = swapped < cur
        ai, bi = a[do_swap], b[do_swap]
        pa, pb = pi[ai].copy(), pi[bi].copy()
        pi[ai], pi[bi] = pb, pa
    return pi


def align_bipartite(cost: np.ndarray, refine_rounds: int, rng) -> dict:
    """scipy-free assignment heuristic (greedy-edge construction + vectorized
    random-pair 2-opt refinement), NOT the globally optimal (Jonker-Volgenant
    / Hungarian) assignment -- the n3 entry explicitly allows either; this
    file implements greedy+refine and reports the OPTIMALITY GAP against the
    row-min relaxation lower bound (sum of each prototype neuron's distance
    to its nearest candidate neuron, ignoring the bijection constraint -- a
    valid lower bound on the true optimal assignment cost, computable without
    an exact solver)."""
    pi0 = greedy_assignment(cost)
    pi = refine_2opt(cost, pi0, refine_rounds, rng)
    n = len(pi)
    idx = np.arange(n)
    assigned_cost = float(cost[idx, pi].sum())
    greedy_cost = float(cost[idx, pi0].sum())
    lower_bound = float(cost.min(axis=1).sum())
    gap = (assigned_cost - lower_bound) / lower_bound if lower_bound > 0 else 0.0
    return {"perm": pi, "cost": assigned_cost, "greedy_cost": greedy_cost,
            "lower_bound": lower_bound, "optimality_gap": gap}


# ---------------------------------------------------- residual/entropy math
def _best_nibble_coding_bytes(nibbles_flat: np.ndarray) -> dict:
    counts = np.bincount(nibbles_flat, minlength=16)
    packed = pack_nibbles(nibbles_flat)
    huff = ent.huffman_stored_bytes(counts)
    zlibb = ent.zlib_stored_bytes(packed.tobytes())
    rans = ent.rans_estimate_bytes(counts)
    return {"huff": huff, "zlib": zlibb, "rans": rans,
            "best": min(huff, zlibb, rans), "h_bits": ent.entropy_bits(counts)}


def compute_expert_residual_stats(proto: dict, expert_t: dict, pi: np.ndarray) -> dict:
    """Per-follower-expert residual/entropy/stored-size stats against the
    layer's prototype, given an alignment permutation pi (pi[i] = which of
    the expert's neurons is matched to the prototype's neuron i)."""
    out: dict = {}
    total_w = total_exact_zero = total_near_zero = 0
    gu_w = gu_near_zero = 0
    for proj_name in qc.PROJS:
        proto_nib = proto[proj_name]["nibbles"]
        exp_nib = expert_t[proj_name]["nibbles"]
        aligned = exp_nib[:, pi] if proj_name == "down_proj" else exp_nib[pi, :]
        residual = nibble_residual(proto_nib, aligned)
        flat = residual.reshape(-1)
        coding = _best_nibble_coding_bytes(flat)
        n = int(flat.size)
        exact_zero = int((flat == 0).sum())
        near_zero = int(np.isin(flat, list(NEAR_ZERO_RESIDUALS)).sum())

        out[f"{proj_name}_h_residual_bits"] = coding["h_bits"]
        out[f"{proj_name}_best_residual_bytes"] = coding["best"]
        out[f"{proj_name}_n_weights"] = n
        out[f"{proj_name}_exact_zero_rate"] = exact_zero / n
        out[f"{proj_name}_near_zero_rate"] = near_zero / n

        total_w += n
        total_exact_zero += exact_zero
        total_near_zero += near_zero
        if proj_name != "down_proj":
            gu_w += n
            gu_near_zero += near_zero

    for proj_name in ("gate_proj", "up_proj"):
        proto_scale = proto[proj_name]["scale"]
        aligned_scale = expert_t[proj_name]["scale"][pi]
        sd = scale_delta_stats(proto_scale, aligned_scale)
        out[f"{proj_name}_scale_delta_h_bits"] = sd["h_bits_per_code"]
        out[f"{proj_name}_scale_delta_bytes"] = sd["stored_bytes"]

    out["residual_density_all"] = 1.0 - total_near_zero / total_w
    out["residual_density_gate_up"] = 1.0 - gu_near_zero / gu_w
    out["exact_zero_rate_all"] = total_exact_zero / total_w
    return out


def independent_coding_stats(tensors: dict) -> dict:
    """Same-sample, freshly-measured n1-style independent entropy coding
    (best of Huffman/zlib/rANS-estimate over the WHOLE tensor as one block --
    a scope simplification vs n1's multi-block-size sweep, reasonable at
    this falsifier's smaller sample) for ONE expert, no alignment. Used both
    for the prototype's own stored size (it isn't aligned to anything) and
    as the freshly-measured 'independent' comparator alongside the
    historical n1 census median (N1_MEDIAN_EXPERT_STORED_RATIO)."""
    total = 0
    per_proj = {}
    for proj_name in qc.PROJS:
        nib = tensors[proj_name]["nibbles"].reshape(-1)
        best = _best_nibble_coding_bytes(nib)["best"]
        per_proj[proj_name] = best
        total += best
    scale_bytes = sum(int(tensors[p]["scale"].nbytes) for p in qc.PROJS)
    return {"payload_bytes": total, "scale_bytes": scale_bytes,
            "independent_bytes": total + scale_bytes, "per_proj": per_proj}


def permutation_metadata_bits(moe_inter: int) -> int:
    """Practical FIXED-WIDTH storage: each of moe_inter permutation entries
    stored in ceil(log2(moe_inter)) bits. Conservative/pessimistic on
    purpose: an entropy coder exploiting the near-uniform distribution of a
    random permutation could approach log2(moe_inter!) bits total (a bit
    less per entry), but fixed-width indices are what a first real
    implementation would plausibly ship, and a falsifier should not flatter
    its own win with an uncosted, unbuilt metadata coder."""
    return math.ceil(math.log2(moe_inter)) * moe_inter


def expert_aligned_stored_bytes(res: dict, proto_stats: dict,
                                n_experts_in_layer: int, cfg: dict) -> float:
    payload = (res["gate_proj_best_residual_bytes"] + res["up_proj_best_residual_bytes"]
              + res["down_proj_best_residual_bytes"])
    scale_delta = res["gate_proj_scale_delta_bytes"] + res["up_proj_scale_delta_bytes"]
    down_scale_raw = cfg["hidden"] * 4          # down's scale isn't neuron-permuted; kept raw (n1-style)
    perm_bytes = math.ceil(permutation_metadata_bits(cfg["moe_inter"]) / 8)
    proto_amortized = proto_stats["independent_bytes"] / n_experts_in_layer
    return payload + scale_delta + down_scale_raw + perm_bytes + proto_amortized


# ------------------------------------------------------------ layer pipeline
def pick_band_layers(rng) -> dict:
    """One seeded-random layer per band (early/mid/late), matching
    measure_expert_entropy's stratified-random-sampling convention."""
    return {name: int(rng.integers(lo, hi + 1)) for name, lo, hi in BANDS}


def load_expert_tensors(index: dict, cfg: dict, layer: int, expert: int,
                        limiter) -> dict:
    """Read + unpack one expert's 3 projections. One rate-limited shard read
    per tensor (weight + .qs), same convention as
    measure_expert_entropy.measure_tensor's reads."""
    out = {}
    for proj_name in qc.PROJS:
        name = f"model.layers.{layer}.mlp.experts.{expert}.{proj_name}.weight"
        w_entry = index.get(name)
        qs_entry = index.get(name + ".qs")
        if w_entry is None or qs_entry is None:
            raise SystemExit(f"missing tensor {name}")
        O, I = qc.expert_shape(cfg, proj_name)
        bits = qc.infer_bits(w_entry["nbytes"], O, I)
        if bits != 4:
            raise SystemExit(f"{name}: expected int4, found int{bits}")
        raw = qc.st_read_tensor(w_entry, limiter)
        qs = qc.st_read_tensor(qs_entry, limiter)
        out[proj_name] = {"nibbles": unpack_matrix(raw, O, I), "scale": qs, "O": O, "I": I}
    return out


def analyze_layer(index: dict, cfg: dict, layer: int, expert_ids: list,
                  proj_dim: int, refine_rounds: int, rng, limiter) -> dict:
    n_scale_feats = 2
    proj = build_projection(cfg["hidden"], n_scale_feats, proj_dim, rng)
    tensors_by_expert = {}
    sig_by_expert = {}
    for e in expert_ids:
        t = load_expert_tensors(index, cfg, layer, e, limiter)
        tensors_by_expert[e] = t
        sig_by_expert[e] = expert_signature(t, proj)

    proto_id = pick_medoid(sig_by_expert)
    proto = tensors_by_expert[proto_id]
    proto_sig = sig_by_expert[proto_id]
    proto_stats = independent_coding_stats(proto)

    expert_results = []
    for e in expert_ids:
        if e == proto_id:
            continue
        cost = pairwise_sqdist(proto_sig, sig_by_expert[e])
        align = align_bipartite(cost, refine_rounds, rng)
        pi = align["perm"]
        res = compute_expert_residual_stats(proto, tensors_by_expert[e], pi)
        res["independent_bytes"] = independent_coding_stats(
            tensors_by_expert[e])["independent_bytes"]
        res.update({"expert": e, "prototype": proto_id,
                    "optimality_gap": align["optimality_gap"],
                    "assignment_cost": align["cost"],
                    "greedy_cost": align["greedy_cost"]})
        expert_results.append(res)

    return {"layer": layer, "prototype": proto_id, "n_experts": len(expert_ids),
            "prototype_stats": proto_stats, "expert_results": expert_results}


# --------------------------------------------------------- verdict/reports
def aggregate_rows(layer_results: list, cfg: dict) -> list:
    raw_bytes = qc.expert_total_bytes(cfg, 4)
    rows = []
    for lr in layer_results:
        for res in lr["expert_results"]:
            aligned_bytes = expert_aligned_stored_bytes(
                res, lr["prototype_stats"], lr["n_experts"], cfg)
            row = dict(res)
            row.update({
                "band": lr["band"], "layer": lr["layer"],
                "aligned_stored_bytes": aligned_bytes,
                "aligned_stored_ratio": aligned_bytes / raw_bytes,
                "independent_stored_ratio": res["independent_bytes"] / raw_bytes,
            })
            rows.append(row)
    return rows


def compute_verdict(rows: list) -> dict:
    """Pure function over aggregated per-expert rows (no I/O): the two n3
    KILL LINES, evaluated on pooled medians across all sampled layers/experts."""
    aligned = np.array([r["aligned_stored_ratio"] for r in rows])
    indep_fresh = np.array([r["independent_stored_ratio"] for r in rows])
    density_all = np.array([r["residual_density_all"] for r in rows])
    density_gu = np.array([r["residual_density_gate_up"] for r in rows])
    gap = np.array([r["optimality_gap"] for r in rows])

    med_aligned = float(np.median(aligned))
    med_indep_fresh = float(np.median(indep_fresh))
    med_density_all = float(np.median(density_all))
    med_density_gu = float(np.median(density_gu))
    med_gap = float(np.median(gap))

    rel_improve_fresh = 1.0 - med_aligned / med_indep_fresh if med_indep_fresh > 0 else float("nan")
    rel_improve_vs_n1 = 1.0 - med_aligned / N1_MEDIAN_EXPERT_STORED_RATIO

    r_modeled = med_density_gu
    work_fraction = COMPUTE_BASE_WORK_FRACTION + r_modeled
    compute_saving = 1.0 - work_fraction

    kill = bool(rel_improve_fresh < KILL_RELATIVE_IMPROVEMENT
               and med_density_all > KILL_RESIDUAL_DENSITY)
    proceed_storage = bool(med_aligned < PROCEED_STORED_RATIO)
    proceed_compute = bool(compute_saving > PROCEED_COMPUTE_SAVING)
    proceed = bool(proceed_storage or proceed_compute)

    kill_line = (
        f"KILL if aligned stored size <10% better than independent entropy "
        f"coding AND residual density >70%: aligned is "
        f"{rel_improve_fresh * 100:.2f}% better than fresh same-sample "
        f"independent coding (median {med_indep_fresh:.4f}) and "
        f"{rel_improve_vs_n1 * 100:.2f}% better than n1's historical 0.7379 "
        f"median; residual density (all projections) {med_density_all * 100:.2f}% "
        f"-> {'KILL' if kill else 'NOT-KILLED'}")
    proceed_line = (
        f"PROCEED if aligned exact stored size <0.70x raw int4 OR modeled "
        f"shared-base compute saving >15%: aligned median stored ratio "
        f"{med_aligned:.4f} ({'<0.70 -> yes' if proceed_storage else '>=0.70 -> no'}); "
        f"modeled compute saving {compute_saving * 100:.2f}% at r={r_modeled:.4f} "
        f"(gate/up residual density) -> work fraction "
        f"{work_fraction:.4f} ({'>15% saving -> yes' if proceed_compute else '<=15% saving -> no'}) "
        f"-> {'PROCEED' if proceed else 'NOT-PROCEED'}")

    return {
        "median_aligned_stored_ratio": med_aligned,
        "median_independent_stored_ratio_fresh": med_indep_fresh,
        "n1_reference_median_stored_ratio": N1_MEDIAN_EXPERT_STORED_RATIO,
        "relative_improvement_vs_fresh_independent": rel_improve_fresh,
        "relative_improvement_vs_n1_reference": rel_improve_vs_n1,
        "median_residual_density_all": med_density_all,
        "median_residual_density_gate_up": med_density_gu,
        "median_optimality_gap": med_gap,
        "modeled_r_gate_up_density": r_modeled,
        "modeled_work_fraction": work_fraction,
        "modeled_compute_saving": compute_saving,
        "kill_condition": kill,
        "proceed_condition_storage": proceed_storage,
        "proceed_condition_compute": proceed_compute,
        "proceed_condition": proceed,
        "kill_line": kill_line,
        "proceed_line": proceed_line,
        "n_experts_total": len(rows),
    }


def write_outputs(a, cfg: dict, layer_results: list, bytes_read: float) -> dict:
    rows = aggregate_rows(layer_results, cfg)
    v = compute_verdict(rows)

    csv_path = os.path.join(a.outdir, "n3-residual-experts.csv")
    cols = ["band", "layer", "prototype", "expert", "aligned_stored_ratio",
            "independent_stored_ratio", "residual_density_all",
            "residual_density_gate_up", "exact_zero_rate_all",
            "optimality_gap", "gate_proj_h_residual_bits",
            "up_proj_h_residual_bits", "down_proj_h_residual_bits"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    band_summary = []
    for band, _, _ in BANDS:
        sel = [r for r in rows if r["band"] == band]
        if not sel:
            continue
        band_summary.append({
            "band": band, "layer": sel[0]["layer"], "n_experts": len(sel),
            "median_aligned_stored_ratio": float(np.median([r["aligned_stored_ratio"] for r in sel])),
            "median_independent_stored_ratio": float(np.median([r["independent_stored_ratio"] for r in sel])),
            "median_residual_density_all": float(np.median([r["residual_density_all"] for r in sel])),
            "median_optimality_gap": float(np.median([r["optimality_gap"] for r in sel])),
        })

    summary = {
        "model": a.model,
        "seed": a.seed,
        "n_experts_per_layer": a.n_experts_per_layer,
        "proj_dim": a.proj_dim,
        "refine_rounds": a.refine_rounds,
        "bytes_read_gb": bytes_read / 1e9,
        "bands": band_summary,
        "headline": v,
    }
    with open(os.path.join(a.outdir, "n3-residual-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[n3-census] per-band stored ratios (aligned / independent-fresh / density / opt-gap):")
    for b in band_summary:
        print(f"  {b['band']:6s} layer={b['layer']:3d} n={b['n_experts']:3d}  "
              f"aligned={b['median_aligned_stored_ratio']:.4f}  "
              f"indep={b['median_independent_stored_ratio']:.4f}  "
              f"density={b['median_residual_density_all']:.4f}  "
              f"gap={b['median_optimality_gap']:.4f}")
    print(f"\n[n3-census] {v['kill_line']}")
    print(f"[n3-census] {v['proceed_line']}")
    print(f"[n3-census] outputs in {a.outdir}: n3-residual-experts.csv, "
          f"n3-residual-summary.json")
    return summary


def report_only(a) -> int:
    """Regenerate the verdict from n3-residual-experts.csv already on disk.
    ZERO shard reads (only reads the CSV written by a prior real run)."""
    path = os.path.join(a.outdir, "n3-residual-experts.csv")
    rows = []
    with open(path) as f:
        for rec in csvmod.DictReader(f):
            row = {"band": rec["band"], "layer": int(rec["layer"]),
                   "prototype": int(rec["prototype"]), "expert": int(rec["expert"])}
            for k, v in rec.items():
                if k not in row:
                    row[k] = float(v)
            rows.append(row)
    verdict = compute_verdict(rows)
    print(f"[n3-census] report-only: {len(rows)} expert rows from {a.outdir} "
          "(no shard reads)")
    print(f"[n3-census] {verdict['kill_line']}")
    print(f"[n3-census] {verdict['proceed_line']}")
    with open(os.path.join(a.outdir, "n3-residual-summary.json"), "w") as f:
        json.dump({"headline": verdict, "report_only": True}, f, indent=2)
    return 0


# ------------------------------------------------------------------- main --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-experts-per-layer", type=int, default=DEFAULT_N_EXPERTS_PER_LAYER,
                    help="32-64 per the n3 falsifier spec")
    ap.add_argument("--proj-dim", type=int, default=DEFAULT_PROJ_DIM)
    ap.add_argument("--refine-rounds", type=int, default=DEFAULT_REFINE_ROUNDS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-mb-s", type=float, default=100.0,
                    help="read throttle; hard-clamped to <= 100 MB/s")
    ap.add_argument("--limit-layers", type=int, default=0,
                    help="smoke: analyze only the first N of the 3 bands")
    ap.add_argument("--allow-busy", action="store_true",
                    help="conscious override of the shard-read gate (synthetic/"
                         "test data only -- NOT for the real model)")
    ap.add_argument("--wait-for-gate", action="store_true",
                    help="poll every --poll-seconds (nice -19) until the gate "
                         "opens, instead of refusing immediately")
    ap.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the verdict from a prior run's CSV; zero shard reads")
    a = ap.parse_args(argv)

    if a.report_only:
        return report_only(a)

    require_shard_reads_allowed(a.allow_busy, a.wait_for_gate, a.poll_seconds)
    if a.allow_busy and glm_running():
        print("[n3-census] engine alive: proceeding under --allow-busy "
              "(<=100 MB/s throttle, nice -19 expected)", flush=True)
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass

    mb_s = min(a.max_mb_s, 100.0) if a.max_mb_s > 0 else 100.0
    if mb_s != a.max_mb_s:
        print(f"[n3-census] clamped throttle to {mb_s} MB/s", flush=True)

    cfg = qc.load_config(a.model)
    index = qc.st_scan(a.model)
    rng = np.random.default_rng(a.seed)
    band_layers = pick_band_layers(rng)
    if a.limit_layers:
        band_layers = dict(list(band_layers.items())[:a.limit_layers])

    n_per_layer = min(a.n_experts_per_layer, cfg["n_experts"])
    est_bytes = len(band_layers) * n_per_layer * qc.expert_total_bytes(cfg, 4)
    print(f"[n3-census] {len(band_layers)} layers x {n_per_layer} experts, "
          f"~{est_bytes / 1e9:.2f} GB to read at <={mb_s:.0f} MB/s", flush=True)

    limiter = qc.RateLimiter(mb_s)
    os.makedirs(a.outdir, exist_ok=True)
    layer_results = []
    for band, layer in band_layers.items():
        expert_ids = sorted(rng.choice(cfg["n_experts"], size=n_per_layer,
                                       replace=False).tolist())
        t0 = time.monotonic()
        lr = analyze_layer(index, cfg, layer, expert_ids, a.proj_dim,
                           a.refine_rounds, rng, limiter)
        lr["band"] = band
        layer_results.append(lr)
        print(f"[n3-census] band={band} layer={layer} done in "
              f"{time.monotonic() - t0:.1f}s, prototype=expert {lr['prototype']}, "
              f"{limiter.bytes / 1e9:.2f} GB read so far", flush=True)

    write_outputs(a, cfg, layer_results, limiter.bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
