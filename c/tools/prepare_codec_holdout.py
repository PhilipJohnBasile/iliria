#!/usr/bin/env python3
"""FROZEN HOLDOUT PROTOCOL for the row-independent int4 codec race
(docs/PERFORMANCE_THEORY.md n1; the codec-race results).

PREPARED 2026-07-15, NOT YET RUN. This script is committed today so it is
ready to execute "tomorrow" once shard reads are permitted and the engine-
idle / SSD-quiesce gate clears (the container-design plan's discipline: no
`/path/to/models` reads today, synthetic + census CSVs only). Running it
for real requires real shard reads this harness's own constraints forbid; its
`--self-check` mode (see below) validates the statistics/acceptance logic
against SYNTHETIC data only and touches no model directory, so it is safe to
run today and IS run as part of this harness's own verification.

WHY A HOLDOUT, AND WHY DIFFERENT FROM THE CENSUS
-------------------------------------------------
The census (tools/measure_expert_entropy.py) and this codec race
(codec_race.c / gen_codec_race_synthetic.py) both measured entropy/ratio
figures either on the census's OWN 256-expert sample or on SYNTHETIC data
shaped to match it. Neither checks whether a codec configuration frozen
from that work still performs as predicted on a genuinely independent
sample of REAL experts. This script is that check, deliberately built to
differ from the census in the two ways that matter for an honest holdout:

  - SEED: 20270115, not the census's 20260715 -- a different draw, not a
    relabeled rerun of the same one.
  - STRATIFICATION AXIS: band x hot/cold (not band alone). "Hot"/"cold"
    splits each band's experts at the MEDIAN touch count from the three
    committed route traces (routes-baseline/coding/shotgun -- already-
    committed trace files, not model shards, so touch-count computation
    itself is exercised and tested today). This guarantees both frequently-
    and rarely-routed experts are represented, which plain band-only
    stratification does not guarantee.

FROZEN, NOT RE-FIT
-------------------
The codec configuration (which coder -- rANS, 4-way interleaved, the codec
race's own winner among codecs that actually compress; see
c/bench-m5max/codec-race-20260715/frozen-codec-config.json) and its actual
per-(band,proj) quantized frequency tables are loaded from that frozen
config file and applied UNCHANGED to the holdout sample. Nothing is re-fit
here -- the whole point is to test whether a codebook trained on this
study's synthetic, census-entropy-matched data generalizes to real weights,
not to re-optimize it on the holdout sample (which would make the
comparison meaningless).

PREDICTED bytes for a held-out tensor = a cross-entropy SIZE ESTIMATE of
that tensor's own symbol counts against the FROZEN (foreign) quantized
frequency table -- the same "quantized-table cross-entropy estimate"
technique tools/measure_expert_entropy.py already uses and validates
against its own real rANS codec (see that file's `rans_estimate_bytes`);
here evaluated against a FROZEN table instead of the tensor's own best-fit
table, which is exactly what "does the frozen codebook generalize" means.
OBSERVED bytes = the same cross-entropy estimate but against the tensor's
OWN (re-fit) quantized table -- the best any per-tensor codebook could do.
The gap between them is the generalization cost this holdout measures.

ACCEPTANCE (PREREGISTERED before any holdout read; do not move these after
seeing results):
  - median relative error <= 2%
  - p95 relative error <= 5%
  reported overall AND by projection AND by band.

GATE: refuses to touch real shards unless the engine is idle
(quant_container.require_idle) or --allow-throttled is passed together with
--max-mb-s (a narrower, more explicit opt-in than the census's --allow-busy:
this script does no silent bench-coexistence reads by default).

Usage (TOMORROW, once shard reads are permitted):
  nice -n 19 python3 c/tools/prepare_codec_holdout.py \\
      --model /path/to/model --outdir c/bench-m5max/codec-holdout-YYYYMMDD

Usage (TODAY, safe -- synthetic data only, never touches /path/to/models):
  python3 c/tools/prepare_codec_holdout.py --self-check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))
import quant_container as qc  # noqa: E402
import measure_expert_entropy as census  # noqa: E402  (reuse M_BITS/M_TOTAL/quantize_freqs/unpack_nibbles)

BANDS = (("early", 3, 27), ("mid", 28, 52), ("late", 53, 77))
N_HOLDOUT_DEFAULT = 256
HOLDOUT_SEED_DEFAULT = 20270115  # DIFFERENT from the census's 20260715 -- see module docstring
MEDIAN_ERR_BOUND_PCT = 2.0        # PREREGISTERED acceptance bound
P95_ERR_BOUND_PCT = 5.0           # PREREGISTERED acceptance bound

_REPO_ROOT = _TOOLS.parent  # c/
DEFAULT_TRACES = [
    _REPO_ROOT / "bench-m5max" / "overnight-20260714-001239" / "routes-baseline.bin",
    _REPO_ROOT / "bench-m5max" / "overnight-20260714-001239" / "routes-coding.bin",
    _REPO_ROOT / "bench-m5max" / "overnight3-20260714-003224" / "routes-shotgun.bin",
]
DEFAULT_FROZEN_CONFIG = (_REPO_ROOT / "bench-m5max" / "codec-race-20260715"
                          / "frozen-codec-config.json")


def band_of(layer: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= layer <= hi:
            return name
    raise ValueError(f"layer {layer} outside routed bands")


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # must be registered BEFORE exec_module: the target
    spec.loader.exec_module(mod)  # module uses @dataclass, which resolves annotations
    return mod                    # via sys.modules[cls.__module__] and fails otherwise


def touch_counts(trace_paths) -> Counter:
    """{(layer, expert): touch_count} from the committed route traces (NOT
    /path/to/models -- these are already-committed trace files; reading
    them is exercised and tested today, unlike the shard-reading path
    below)."""
    vsco = _load_sibling("variable_size_cache_oracle", "variable_size_cache_oracle.py")
    counts: Counter = Counter()
    for p in trace_paths:
        _, calls = vsco.load_calls(Path(p))
        for call in calls:
            for e in call.experts:
                counts[(call.layer, e)] += 1
    return counts


def stratified_sample(cfg, counts, n_total: int, seed: int):
    """band x hot/cold stratified sample given a PRE-COMPUTED {(layer,expert):
    touch_count} mapping -- deliberately a DIFFERENT stratification axis than
    the census's band-only draw (see module docstring). Experts absent from
    `counts` are, by construction, cold (touch count 0). Split out from
    sample_holdout_experts() so the stratification algorithm itself is
    testable with injected/mock touch data, independent of whether the real
    route-trace files are present in a given checkout (they are large,
    environment-local artifacts -- see module docstring on the holdout's own
    trace dependency)."""
    rng = np.random.default_rng(seed)
    per_band = [n_total // 3] * 3
    for i in range(n_total - sum(per_band)):
        per_band[2 - i] += 1
    chosen = []
    for (name, lo, hi), n_band in zip(BANDS, per_band):
        pool = [(l, e) for l in range(lo, hi + 1) for e in range(cfg["n_experts"])]
        band_touch = [counts.get(p, 0) for p in pool]
        median = statistics.median(band_touch) if band_touch else 0
        hot = [p for p, c in zip(pool, band_touch) if c > median]
        cold = [p for p, c in zip(pool, band_touch) if c <= median]
        n_hot = n_band // 2
        n_cold = n_band - n_hot
        # cap each bucket's draw to what's available, then REDISTRIBUTE any
        # shortfall to the other bucket -- an empty/tiny hot pool (e.g. an
        # all-zero touch-count input, or a band the traces barely covered)
        # must not silently shrink the total sample below n_total; it must
        # fall back to drawing more cold experts instead (and vice versa).
        take_hot = min(n_hot, len(hot))
        take_cold = min(n_cold, len(cold))
        shortfall = (n_hot - take_hot) + (n_cold - take_cold)
        if shortfall > 0:
            extra_cold = min(shortfall, len(cold) - take_cold)
            take_cold += extra_cold
            shortfall -= extra_cold
            extra_hot = min(shortfall, len(hot) - take_hot)
            take_hot += extra_hot
            shortfall -= extra_hot
        for sub, n_sub in ((hot, take_hot), (cold, take_cold)):
            if not sub or n_sub <= 0:
                continue
            idx = rng.choice(len(sub), size=n_sub, replace=False)
            chosen.extend(sub[i] for i in sorted(idx))
    return chosen


def sample_holdout_experts(cfg, trace_paths, n_total: int, seed: int):
    """Real entry point: reads the committed route traces, then delegates
    to stratified_sample(). See that function for the algorithm itself."""
    counts = touch_counts(trace_paths)
    return stratified_sample(cfg, counts, n_total, seed)


def load_frozen_config(path) -> dict:
    with open(path) as f:
        return json.load(f)


def cross_entropy_bytes(counts16, freq16) -> float:
    """Quantized-table cross-entropy size estimate of `counts16` against a
    (possibly FOREIGN) frequency table `freq16` summing to census.M_TOTAL --
    same formula as measure_expert_entropy.rans_estimate_bytes's bit count,
    generalized to an externally-supplied table (predicted, using the
    FROZEN table) vs. the tensor's own best-fit table (observed)."""
    total = int(sum(counts16))
    if total <= 0:
        return 0.0
    bits = 0.0
    for s in range(16):
        c = counts16[s]
        if c > 0:
            f = max(1, int(freq16[s]))
            bits += c * math.log2(census.M_TOTAL / f)
    return bits / 8.0


def band_proj_key(band: str, proj_short: str) -> str:
    return f"{band}/{proj_short}"


def relative_error_pct(observed: float, predicted: float) -> float:
    if observed <= 0:
        return 0.0
    return abs(predicted - observed) / observed * 100.0


def acceptance_report(rows: list[dict]) -> dict:
    """rows: [{"band":..,"proj":..,"observed":..,"predicted":..}, ...]
    -> {"median_pct":, "p95_pct":, "accept":, per-band/proj breakdowns}."""
    errs = [relative_error_pct(r["observed"], r["predicted"]) for r in rows]
    if not errs:
        return {"median_pct": None, "p95_pct": None, "accept": False, "n": 0}
    median = float(np.median(errs))
    p95 = float(np.percentile(errs, 95))
    accept = median <= MEDIAN_ERR_BOUND_PCT and p95 <= P95_ERR_BOUND_PCT

    def _slice(key_fn, keys):
        out = {}
        for k in keys:
            sub = [relative_error_pct(r["observed"], r["predicted"]) for r in rows if key_fn(r) == k]
            if sub:
                out[k] = {"median_pct": float(np.median(sub)), "p95_pct": float(np.percentile(sub, 95)), "n": len(sub)}
        return out

    return {
        "n": len(rows),
        "median_pct": median,
        "p95_pct": p95,
        "median_bound_pct": MEDIAN_ERR_BOUND_PCT,
        "p95_bound_pct": P95_ERR_BOUND_PCT,
        "accept": accept,
        "by_band": _slice(lambda r: r["band"], [b[0] for b in BANDS]),
        "by_proj": _slice(lambda r: r["proj"], ("gate", "up", "down")),
    }


# ------------------------------------------------------- self-check mode --
def self_check() -> int:
    """Validates the acceptance/statistics logic against SYNTHETIC
    observed/predicted pairs -- NEVER touches /path/to/models. Two cases:
    a "should pass" case (small, bounded synthetic error) and a "should
    fail" case (deliberately large error), confirming acceptance_report()
    and relative_error_pct() actually gate correctly rather than always
    reporting green."""
    rng = np.random.default_rng(1234)
    ok = True

    # case 1: errors drawn small (~1% median, ~3% p95) -> must ACCEPT
    rows_pass = []
    for band, _, _ in BANDS:
        for proj in ("gate", "up", "down"):
            for _ in range(20):
                observed = float(rng.uniform(1e6, 2e6))
                err_frac = float(np.clip(rng.normal(0.01, 0.01), -0.03, 0.03))
                predicted = observed * (1 + err_frac)
                rows_pass.append({"band": band, "proj": proj, "observed": observed, "predicted": predicted})
    rep_pass = acceptance_report(rows_pass)
    if not rep_pass["accept"]:
        print(f"[self-check] FAIL: expected ACCEPT on small-error synthetic rows, got reject "
              f"(median={rep_pass['median_pct']:.3f}%, p95={rep_pass['p95_pct']:.3f}%)")
        ok = False
    else:
        print(f"[self-check] small-error case correctly ACCEPTED "
              f"(median={rep_pass['median_pct']:.3f}%, p95={rep_pass['p95_pct']:.3f}%)")

    # case 2: errors drawn large (~10% median) -> must REJECT
    rows_fail = []
    for band, _, _ in BANDS:
        for proj in ("gate", "up", "down"):
            for _ in range(20):
                observed = float(rng.uniform(1e6, 2e6))
                err_frac = float(rng.uniform(0.08, 0.15))
                predicted = observed * (1 + err_frac)
                rows_fail.append({"band": band, "proj": proj, "observed": observed, "predicted": predicted})
    rep_fail = acceptance_report(rows_fail)
    if rep_fail["accept"]:
        print(f"[self-check] FAIL: expected REJECT on large-error synthetic rows, got accept "
              f"(median={rep_fail['median_pct']:.3f}%, p95={rep_fail['p95_pct']:.3f}%)")
        ok = False
    else:
        print(f"[self-check] large-error case correctly REJECTED "
              f"(median={rep_fail['median_pct']:.3f}%, p95={rep_fail['p95_pct']:.3f}%)")

    # cross_entropy_bytes: a tensor scored against ITS OWN quantized table
    # must estimate close to its own entropy (sanity: frozen-vs-self should
    # be the FLOOR of the generalization-cost comparison, not an outlier).
    counts = np.array([2, 5, 40, 200, 900, 2500, 4800, 6800, 7200, 6800, 4800,
                        2500, 900, 200, 40, 5], dtype=np.int64)
    own_freq = census.quantize_freqs(counts)
    self_bytes = cross_entropy_bytes(counts, own_freq)
    h_bits = census.entropy_bits(counts)
    n = int(counts.sum())
    ideal_bytes = h_bits * n / 8.0
    if self_bytes <= 0 or abs(self_bytes - ideal_bytes) / ideal_bytes > 0.02:
        print(f"[self-check] FAIL: cross_entropy_bytes against own table "
              f"({self_bytes:.1f} B) should track the raw entropy estimate "
              f"({ideal_bytes:.1f} B) within 2%")
        ok = False
    else:
        print(f"[self-check] cross_entropy_bytes against own table OK "
              f"({self_bytes:.1f} B vs entropy estimate {ideal_bytes:.1f} B)")

    # a genuinely FOREIGN (mismatched) table must estimate MORE bytes than
    # the tensor's own best-fit table (cross-entropy >= entropy, Gibbs'
    # inequality) -- catches a sign error in cross_entropy_bytes.
    foreign_freq = np.full(16, census.M_TOTAL // 16, dtype=np.int64)  # uniform, deliberately mismatched
    foreign_bytes = cross_entropy_bytes(counts, foreign_freq)
    if foreign_bytes < self_bytes:
        print(f"[self-check] FAIL: a mismatched frozen table ({foreign_bytes:.1f} B) "
              f"scored CHEAPER than the tensor's own best-fit table ({self_bytes:.1f} B) "
              f"-- violates Gibbs' inequality, a sign error somewhere")
        ok = False
    else:
        print(f"[self-check] mismatched-table cross-entropy correctly >= own-table "
              f"({foreign_bytes:.1f} B >= {self_bytes:.1f} B)")

    # stratified_sample: mock touch counts (no route-trace files needed --
    # those are large, environment-local artifacts not necessarily present
    # in every checkout; the ALGORITHM is what's under test here, not
    # whether this checkout happens to have the trace .bin files).
    cfg = {"n_experts": 128}
    rng2 = np.random.default_rng(7)
    mock_counts = Counter()
    for name, lo, hi in BANDS:
        for l in range(lo, hi + 1):
            for e in range(cfg["n_experts"]):
                if rng2.random() < 0.3:  # ~30% of (layer,expert) pairs ever touched
                    mock_counts[(l, e)] = int(rng2.integers(1, 50))
    chosen = stratified_sample(cfg, mock_counts, N_HOLDOUT_DEFAULT, HOLDOUT_SEED_DEFAULT)
    band_n = Counter(band_of(l) for (l, e) in chosen)
    n_unique = len(set(chosen))
    hot_n = sum(1 for p in chosen if mock_counts.get(p, 0) > 0)
    if len(chosen) != N_HOLDOUT_DEFAULT:
        print(f"[self-check] FAIL: stratified_sample returned {len(chosen)} experts, "
              f"expected {N_HOLDOUT_DEFAULT}")
        ok = False
    elif n_unique != len(chosen):
        print(f"[self-check] FAIL: stratified_sample returned duplicate experts "
              f"({n_unique} unique of {len(chosen)})")
        ok = False
    elif not band_n or min(band_n.values()) < N_HOLDOUT_DEFAULT // 3 - 1:
        print(f"[self-check] FAIL: stratified_sample band split is not roughly even: {dict(band_n)}")
        ok = False
    else:
        print(f"[self-check] stratified_sample OK: {len(chosen)} unique experts, "
              f"band split {dict(band_n)}, {hot_n} with nonzero mock touch count "
              f"(hot/cold split exercised)")

    print(f"[self-check] {'ALL OK' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


# --------------------------------------------------------------- real run --
def run_real(a) -> int:
    qc.require_idle(a.allow_throttled, "codec-race frozen holdout")
    if a.allow_throttled and not a.max_mb_s:
        sys.stderr.write("--allow-throttled requires --max-mb-s (explicit, "
                          "supervised throttle -- not silent bench-coexistence)\n")
        return 2
    mb_s = min(a.max_mb_s, 100.0) if a.max_mb_s else 100.0

    cfg = qc.load_config(a.model)
    frozen = load_frozen_config(a.frozen_config)
    freq_tables = frozen["frozen_quantized_freq16_by_band_proj"]

    trace_paths = [Path(p) for p in (a.traces or DEFAULT_TRACES)]
    chosen = sample_holdout_experts(cfg, trace_paths, a.n_experts, a.seed)
    print(f"[holdout] {len(chosen)} experts sampled (seed {a.seed}, band x hot/cold "
          f"stratified, DIFFERENT axis than the census's band-only draw)")

    index = qc.st_scan(a.model)
    limiter = qc.RateLimiter(mb_s)
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for layer, expert in chosen:
        band = band_of(layer)
        for proj in qc.PROJS:
            proj_short = proj.split("_")[0]
            key = band_proj_key(band, proj_short)
            name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
            w_entry = index.get(name)
            if w_entry is None:
                raise SystemExit(f"missing tensor {name}")
            raw = qc.st_read_tensor(w_entry, limiter)
            nibbles = census.unpack_nibbles(raw)
            counts = np.bincount(nibbles, minlength=16)
            own_freq = census.quantize_freqs(counts)
            observed = cross_entropy_bytes(counts, own_freq)
            predicted = cross_entropy_bytes(counts, np.array(freq_tables[key]))
            rows.append({"layer": layer, "expert": expert, "band": band,
                         "proj": proj_short, "observed": observed, "predicted": predicted})

    report = acceptance_report(rows)
    with open(os.path.join(a.outdir, "holdout-report.json"), "w") as f:
        json.dump({"seed": a.seed, "n_experts": len(chosen), "frozen_config": str(a.frozen_config),
                   "report": report}, f, indent=2)
    with open(os.path.join(a.outdir, "holdout-rows.csv"), "w") as f:
        f.write("layer,expert,band,proj,observed_bytes,predicted_bytes,rel_err_pct\n")
        for r in rows:
            f.write(f"{r['layer']},{r['expert']},{r['band']},{r['proj']},"
                     f"{r['observed']:.2f},{r['predicted']:.2f},"
                     f"{relative_error_pct(r['observed'], r['predicted']):.4f}\n")

    verdict = "ACCEPT" if report["accept"] else "REJECT"
    print(f"[holdout] median rel err {report['median_pct']:.3f}% (bound <={MEDIAN_ERR_BOUND_PCT}%), "
          f"p95 {report['p95_pct']:.3f}% (bound <={P95_ERR_BOUND_PCT}%) -> {verdict}")
    return 0 if report["accept"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model")
    ap.add_argument("--outdir")
    ap.add_argument("--n-experts", type=int, default=N_HOLDOUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=HOLDOUT_SEED_DEFAULT)
    ap.add_argument("--traces", nargs="*", default=None)
    ap.add_argument("--frozen-config", default=str(DEFAULT_FROZEN_CONFIG))
    ap.add_argument("--max-mb-s", type=float, default=0.0)
    ap.add_argument("--allow-throttled", action="store_true",
                     help="explicit, supervised throttled run against real shards "
                          "(requires --max-mb-s); default is to refuse unless idle")
    ap.add_argument("--self-check", action="store_true",
                     help="validate acceptance/statistics logic on SYNTHETIC data only; "
                          "never touches /path/to/models")
    a = ap.parse_args(argv)

    if a.self_check:
        return self_check()

    if not a.model or not a.outdir:
        ap.error("--model and --outdir are required unless --self-check is given")
    return run_real(a)


if __name__ == "__main__":
    sys.exit(main())
