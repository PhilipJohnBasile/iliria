#!/usr/bin/env python3
"""Paired B0/B1 quality gate: exact McNemar + paired CIs + paired bootstrap
on normalized-loglik margins, computed from two --dump-per-item JSONL files
produced by tools/eval_glm.py (see c/bench-m5max/campaign-log.md 2026-07-15
for the preregistered protocol this implements).

WHY PAIRED: the two arms (a BASELINE, e.g. B0, and a CANDIDATE, e.g. B1) are
scored on the IDENTICAL question set, so per-item outcomes are correlated.
Comparing aggregate accuracy alone throws that correlation away and is
under-powered; this tool works item-by-item instead:

  - exact McNemar test (binomial on discordant pairs, no chi-square/
    continuity-correction approximation) for both acc and acc_norm;
  - a paired bootstrap CI for the accuracy delta (candidate - baseline),
    for both acc and acc_norm;
  - a paired bootstrap (default 10k resamples) on the MEAN normalized
    log-likelihood margin of the gold answer (candidate - baseline) --
    a continuous, more sample-efficient companion to the binary accuracy
    signal (see norm_loglik_gold below);
  - the observed discordance rate, and the number of paired items a future
    run would need for 80%/90% power at the SAME preregistered margin
    (Connor RJ, "Sample size for testing differences in proportions for
    the paired-sample design", Statistics in Medicine 6:619-625, 1987;
    normal approximation to McNemar's test -- documented, not the exact
    test, and reported purely as planning context alongside the real
    verdict below, not as part of it);
  - a VERDICT: the candidate is declared noninferior on a metric when the
    one-sided (1-alpha) bootstrap lower confidence bound on
    (candidate - baseline) clears -margin. --noninferiority-margin has NO
    DEFAULT: the margin must be chosen BEFORE looking at the results (see
    task notes / campaign-state.json), not fitted to them afterwards.

Per-task breakdown plus a pooled (all tasks) summary; Markdown + JSON output.

Dependencies: Python standard library only (json, argparse, math, random) --
no numpy/scipy, no engine, no network. Safe to run on any dev box tonight.

USO:
  python3 tools/paired_quality_gate.py b0_per_item.jsonl b1_per_item.jsonl \\
      --noninferiority-margin 0.02 --out-json gate.json --out-md gate.md
  # plumbing self-test (no files needed):
  python3 tools/paired_quality_gate.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 1234
DEFAULT_ALPHA = 0.05          # one-sided test level (noninferiority convention)


# ---------------------------------------------------------------------------
# stats primitives (stdlib only)
# ---------------------------------------------------------------------------

def norm_ppf(p: float) -> float:
    """Inverse CDF (quantile function) of the standard normal distribution.

    Peter Acklam's rational approximation (|error| < 1.15e-9 on its own),
    refined here with one Halley step against `math.erfc` (stdlib, exact)
    so the result is anchored to ground truth rather than trusted
    coefficients alone -- converges to within ~1e-15 of the true quantile.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    # one Halley refinement step using the stdlib's exact erfc
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    x = x - u / (1 + x * u / 2)
    return x


def mcnemar_exact_p(n_candidate_only: int, n_baseline_only: int) -> float:
    """Exact two-sided McNemar test p-value: a binomial test of the
    discordant pairs against p=0.5, i.e. P(X<=k)+P(X>=n-k) for
    X~Binomial(n, 0.5), k=min(n_candidate_only, n_baseline_only),
    n=n_candidate_only+n_baseline_only. Symmetric in its two arguments.
    No continuity correction and no chi-square approximation.
    """
    n = n_candidate_only + n_baseline_only
    if n == 0:
        return 1.0
    k = min(n_candidate_only, n_baseline_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def required_pairs(psi: float, margin: float, alpha: float, beta: float):
    """Approximate number of PAIRED items needed for power (1-beta) at
    one-sided level alpha to detect a true paired-proportion difference of
    `margin`, given an anticipated discordance rate `psi` = (b+c)/n.
    Connor RJ (1987), Statistics in Medicine 6:619-625 -- a normal
    approximation to McNemar's test. Returns None when psi<=0 (no
    discordance observed -- the approximation has no meaningful answer)."""
    if margin <= 0:
        raise ValueError("noninferiority margin must be positive")
    if psi <= 0:
        return None
    z_a, z_b = norm_ppf(1 - alpha), norm_ppf(1 - beta)
    inner = max(0.0, psi - margin * margin)
    n = (z_a * math.sqrt(psi) + z_b * math.sqrt(inner)) ** 2 / (margin * margin)
    return math.ceil(n)


def percentile(sorted_values, pct: float) -> float:
    """Linear-interpolation percentile (pct in [0,100]); `sorted_values`
    must already be sorted ascending."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


def bootstrap_mean(values, n_resamples: int, seed: int):
    """Paired bootstrap over per-item scalar differences. `values[i]` must
    already BE the per-item (candidate - baseline) difference, so
    resampling item indices with replacement and averaging each resample
    is exactly the paired bootstrap (the pairing is preserved because both
    arms' contributions to item i are collapsed into the one number
    values[i] before resampling -- there is nothing left to decorrelate).
    Returns the resampled means, sorted ascending. Deterministic for a
    given (values, n_resamples, seed).
    """
    n = len(values)
    if n == 0:
        return []
    rng = random.Random(seed)
    dist = [sum(values[rng.randrange(n)] for _ in range(n)) / n
            for _ in range(n_resamples)]
    dist.sort()
    return dist


def norm_loglik_gold(rec) -> float:
    """Per-character-normalized log-likelihood the model assigned to the
    GOLD (correct) option -- the same quantity acc_norm's argmax uses, but
    kept continuous instead of collapsed to a win/lose bit. A model that
    is drifting can lose confidence on the correct answer for many items
    before any of them actually flip the argmax, so this is a more
    sample-efficient quality signal than accuracy alone."""
    g = rec["gold"]
    return rec["lp_per_option"][g] / rec["option_lengths"][g]


# ---------------------------------------------------------------------------
# loading + alignment
# ---------------------------------------------------------------------------

def load_per_item(path: str):
    """Load a --dump-per-item JSONL file into a dict keyed by (task, qid)."""
    items = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{lineno}: invalid JSON ({e})")
            for field in ("task", "qid", "gold", "lp_per_option", "option_lengths",
                          "correct_acc", "correct_accnorm", "chosen_acc", "chosen_accnorm"):
                if field not in rec:
                    sys.exit(f"{path}:{lineno}: missing required field {field!r}")
            key = (rec["task"], rec["qid"])
            if key in items:
                sys.exit(f"{path}:{lineno}: duplicate (task,qid) {key}")
            items[key] = rec
    if not items:
        sys.exit(f"{path}: no per-item records found (empty file?)")
    return items


def align(a_items, b_items, a_path: str, b_path: str):
    """Match the two arms by (task, qid). HARD-ERRORS (sys.exit, not a
    warning) on any mismatch: missing/extra keys, disagreeing gold, or a
    disagreeing option count -- all mean the two arms were NOT run on the
    identical question set, which invalidates every paired statistic
    downstream. Returns the sorted list of common (task, qid) keys."""
    a_keys, b_keys = set(a_items), set(b_items)
    only_a, only_b = a_keys - b_keys, b_keys - a_keys
    if only_a or only_b:
        msg = [f"qid mismatch between {a_path} and {b_path}: the two arms "
               f"are not on the identical question set."]
        if only_a:
            msg.append(f"  {len(only_a)} item(s) only in {a_path}: {sorted(only_a)[:5]}"
                       f"{' ...' if len(only_a) > 5 else ''}")
        if only_b:
            msg.append(f"  {len(only_b)} item(s) only in {b_path}: {sorted(only_b)[:5]}"
                       f"{' ...' if len(only_b) > 5 else ''}")
        sys.exit("\n".join(msg))
    mismatches = []
    for key in sorted(a_keys):
        ra, rb = a_items[key], b_items[key]
        if ra["gold"] != rb["gold"]:
            mismatches.append(f"  {key}: gold differs ({ra['gold']!r} vs {rb['gold']!r})")
        elif len(ra["lp_per_option"]) != len(rb["lp_per_option"]):
            mismatches.append(f"  {key}: option count differs "
                              f"({len(ra['lp_per_option'])} vs {len(rb['lp_per_option'])})")
    if mismatches:
        sys.exit(f"qid alignment error between {a_path} and {b_path}: same "
                 f"(task,qid) keys but disagreeing content (not the identical "
                 f"question set):\n" + "\n".join(mismatches[:20]))
    return sorted(a_keys)


# ---------------------------------------------------------------------------
# per-group statistics (called once pooled, once per task)
# ---------------------------------------------------------------------------

def metric_block(a_vals, b_vals, resamples, seed, alpha, margin):
    n = len(a_vals)
    acc_a, acc_b = sum(a_vals) / n, sum(b_vals) / n
    n_b_only = sum(1 for x, y in zip(a_vals, b_vals) if x == 0.0 and y == 1.0)
    n_a_only = sum(1 for x, y in zip(a_vals, b_vals) if x == 1.0 and y == 0.0)
    psi = (n_a_only + n_b_only) / n
    diffs = [y - x for x, y in zip(a_vals, b_vals)]
    dist = bootstrap_mean(diffs, resamples, seed)
    ci_lo, ci_hi = percentile(dist, 2.5), percentile(dist, 97.5)
    lower_one_sided = percentile(dist, 100 * alpha)
    return {
        "baseline": acc_a, "candidate": acc_b, "delta": acc_b - acc_a,
        "discordant_baseline_only": n_a_only, "discordant_candidate_only": n_b_only,
        "discordance_rate": psi,
        "mcnemar_p": mcnemar_exact_p(n_b_only, n_a_only),
        "bootstrap_ci95": [ci_lo, ci_hi],
        "bootstrap_lower_one_sided": lower_one_sided,
        "verdict": "PASS" if lower_one_sided > -margin else "FAIL",
        "required_pairs": {"power80": required_pairs(psi, margin, alpha, 0.20),
                           "power90": required_pairs(psi, margin, alpha, 0.10)},
    }


def margin_block(values, resamples, seed, alpha):
    n = len(values)
    mean_margin = sum(values) / n
    dist = bootstrap_mean(values, resamples, seed)
    return {
        "mean": mean_margin,
        "bootstrap_ci95": [percentile(dist, 2.5), percentile(dist, 97.5)],
        "bootstrap_lower_one_sided": percentile(dist, 100 * alpha),
    }


def compute_group(keys, a_items, b_items, resamples, seed, alpha, margin):
    n = len(keys)
    a_acc  = [1.0 if a_items[k]["correct_acc"]     else 0.0 for k in keys]
    b_acc  = [1.0 if b_items[k]["correct_acc"]     else 0.0 for k in keys]
    a_accn = [1.0 if a_items[k]["correct_accnorm"] else 0.0 for k in keys]
    b_accn = [1.0 if b_items[k]["correct_accnorm"] else 0.0 for k in keys]
    margins = [norm_loglik_gold(b_items[k]) - norm_loglik_gold(a_items[k]) for k in keys]
    return {
        "n": n,
        "acc": metric_block(a_acc, b_acc, resamples, seed, alpha, margin),
        # distinct (but still fully deterministic) sub-seeds per metric so the
        # three bootstraps don't reuse the identical resample sequence
        "acc_norm": metric_block(a_accn, b_accn, resamples, seed + 1, alpha, margin),
        "loglik_margin": margin_block(margins, resamples, seed + 2, alpha),
        "churn_acc": sum(1 for k in keys if a_items[k]["chosen_acc"] != b_items[k]["chosen_acc"]),
        "churn_accnorm": sum(1 for k in keys if a_items[k]["chosen_accnorm"] != b_items[k]["chosen_accnorm"]),
    }


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------

def _fmt_req(x):
    return "n/a" if x is None else str(x)


def render_markdown(report) -> str:
    lines = [
        f"# Paired quality gate: {report['candidate']} vs {report['baseline']}",
        "",
        f"- items compared (paired): **{report['n_items']}**",
        f"- preregistered noninferiority margin: **{report['noninferiority_margin']}**",
        f"- alpha (one-sided): {report['alpha']}  ·  bootstrap resamples: "
        f"{report['bootstrap_resamples']}  ·  seed: {report['seed']}",
        f"- **VERDICT: {report['verdict']}**",
        "",
    ]

    def block(name, g):
        lines.append(f"## {name} (n={g['n']})")
        lines.append("")
        lines.append("| metric | baseline | candidate | delta | discordant (base-only/cand-only) "
                     "| McNemar p | 95% CI | one-sided lower | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for label in ("acc", "acc_norm"):
            m = g[label]
            lines.append(
                f"| {label} | {m['baseline']:.4f} | {m['candidate']:.4f} | {m['delta']:+.4f} "
                f"| {m['discordant_baseline_only']}/{m['discordant_candidate_only']} "
                f"| {m['mcnemar_p']:.4g} | [{m['bootstrap_ci95'][0]:+.4f}, {m['bootstrap_ci95'][1]:+.4f}] "
                f"| {m['bootstrap_lower_one_sided']:+.4f} | {m['verdict']} |")
        lm = g["loglik_margin"]
        lines.append("")
        lines.append(
            f"normalized-loglik margin of the gold answer (candidate - baseline): "
            f"mean {lm['mean']:+.4f}, 95% CI [{lm['bootstrap_ci95'][0]:+.4f}, "
            f"{lm['bootstrap_ci95'][1]:+.4f}], one-sided lower {lm['bootstrap_lower_one_sided']:+.4f} "
            f"(diagnostic signal -- not gated against the noninferiority margin, "
            f"which is on the accuracy proportion scale)")
        lines.append("")
        lines.append(f"answer churn (top choice changed): "
                     f"acc {g['churn_acc']}/{g['n']}, acc_norm {g['churn_accnorm']}/{g['n']}")
        ra, rn = g["acc"]["required_pairs"], g["acc_norm"]["required_pairs"]
        lines.append("")
        lines.append(
            f"pairs needed for the preregistered margin at 80%/90% power "
            f"(Connor 1987 normal approximation): "
            f"acc {_fmt_req(ra['power80'])}/{_fmt_req(ra['power90'])}, "
            f"acc_norm {_fmt_req(rn['power80'])}/{_fmt_req(rn['power90'])}")
        lines.append("")

    block("Pooled", report["pooled"])
    for t in sorted(report["tasks"]):
        block(f"Task: {t}", report["tasks"][t])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_report(a_items, b_items, common_keys, baseline_label, candidate_label,
                  margin, alpha, resamples, seed):
    tasks = sorted({k[0] for k in common_keys})
    report = {
        "schema_version": 1,
        "baseline": baseline_label, "candidate": candidate_label,
        "noninferiority_margin": margin, "alpha": alpha,
        "bootstrap_resamples": resamples, "seed": seed,
        "n_items": len(common_keys),
        "tasks": {}, "pooled": None, "verdict": None,
    }
    for t in tasks:
        keys = [k for k in common_keys if k[0] == t]
        report["tasks"][t] = compute_group(keys, a_items, b_items, resamples, seed, alpha, margin)
    report["pooled"] = compute_group(common_keys, a_items, b_items, resamples, seed, alpha, margin)
    report["verdict"] = ("PASS" if report["pooled"]["acc"]["verdict"] == "PASS"
                         and report["pooled"]["acc_norm"]["verdict"] == "PASS" else "FAIL")
    return report


def _selftest():
    """Plumbing self-test with hand-computed synthetic fixtures -- no files,
    no engine. Mirrors tools/eval_glm.py's own --selftest convention."""
    # 4 items: 1 discordant baseline-only, 1 discordant candidate-only,
    # 2 concordant (both correct). Every field the real dumper emits.
    def rec(gold, lp, lens):
        chosen = max(range(len(lp)), key=lambda i: lp[i])
        return {"task": "t", "qid": 0, "gold": gold, "chosen_acc": chosen,
                "chosen_accnorm": chosen, "correct_acc": chosen == gold,
                "correct_accnorm": chosen == gold, "lp_per_option": lp, "option_lengths": lens}
    a_items = {("t", 0): rec(0, [-1.0, -2.0], [1, 1]),   # baseline right
               ("t", 1): rec(0, [-2.0, -1.0], [1, 1]),   # baseline wrong
               ("t", 2): rec(0, [-1.0, -2.0], [1, 1]),   # both right
               ("t", 3): rec(0, [-2.0, -1.0], [1, 1])}   # both wrong
    b_items = {("t", 0): rec(0, [-2.0, -1.0], [1, 1]),   # candidate wrong -> discordant (baseline-only)
               ("t", 1): rec(0, [-1.0, -2.0], [1, 1]),   # candidate right -> discordant (candidate-only)
               ("t", 2): rec(0, [-1.0, -2.0], [1, 1]),   # both right
               ("t", 3): rec(0, [-2.0, -1.0], [1, 1])}   # both wrong
    common = align(a_items, b_items, "A", "B")
    assert common == [("t", 0), ("t", 1), ("t", 2), ("t", 3)]
    report = build_report(a_items, b_items, common, "A", "B", margin=0.5,
                          alpha=0.05, resamples=200, seed=1)
    pooled_acc = report["pooled"]["acc"]
    assert pooled_acc["discordant_baseline_only"] == 1
    assert pooled_acc["discordant_candidate_only"] == 1
    assert pooled_acc["mcnemar_p"] == 1.0     # 1 vs 1 discordant: p=1 exactly
    assert pooled_acc["delta"] == 0.0
    print(render_markdown(report))
    print("selftest OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", nargs="?", help="--dump-per-item JSONL for the baseline arm (e.g. B0)")
    ap.add_argument("candidate", nargs="?", help="--dump-per-item JSONL for the candidate arm (e.g. B1)")
    ap.add_argument("--noninferiority-margin", type=float, default=None,
                    help="REQUIRED (no default): preregistered noninferiority margin as a "
                         "proportion, e.g. 0.02 = 2 accuracy points. Choose it before looking "
                         "at results -- see campaign-state.json / campaign-log.md.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help=f"one-sided test level (default {DEFAULT_ALPHA})")
    ap.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_RESAMPLES,
                    help=f"default {DEFAULT_RESAMPLES}")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"default {DEFAULT_SEED}")
    ap.add_argument("--baseline-label", default=None)
    ap.add_argument("--candidate-label", default=None)
    ap.add_argument("--out-json", default=None, help="write the full JSON report to PATH")
    ap.add_argument("--out-md", default=None, help="write the Markdown report to PATH")
    ap.add_argument("--quiet", action="store_true", help="don't echo the Markdown report to stdout")
    ap.add_argument("--selftest", action="store_true", help="run the built-in synthetic self-test and exit")
    a = ap.parse_args(argv)

    if a.selftest:
        _selftest()
        return 0

    if a.baseline is None or a.candidate is None:
        ap.error("baseline and candidate JSONL files are required (unless --selftest)")
    if a.noninferiority_margin is None:
        ap.error("--noninferiority-margin is required and has no default -- "
                 "preregister it before running the gate")
    if a.noninferiority_margin <= 0:
        ap.error("--noninferiority-margin must be > 0")
    if a.bootstrap_resamples < 1:
        ap.error("--bootstrap-resamples must be >= 1")

    a_items = load_per_item(a.baseline)
    b_items = load_per_item(a.candidate)
    common = align(a_items, b_items, a.baseline, a.candidate)

    report = build_report(
        a_items, b_items, common,
        baseline_label=a.baseline_label or a.baseline,
        candidate_label=a.candidate_label or a.candidate,
        margin=a.noninferiority_margin, alpha=a.alpha,
        resamples=a.bootstrap_resamples, seed=a.seed)

    if a.out_json:
        Path(a.out_json).write_text(json.dumps(report, indent=2) + "\n")
    if a.out_md:
        Path(a.out_md).write_text(render_markdown(report))
    if not a.quiet:
        print(render_markdown(report))

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
