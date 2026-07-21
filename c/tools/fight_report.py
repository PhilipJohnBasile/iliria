#!/usr/bin/env python3
"""Aggregate a fight_card.sh result directory into the Saturday FIGHT card report.

Reads manifest.json (the run list fight_card.sh produced) and plan.json (the
resolved arm definitions -- kind, container, resolved env, kernel-family
fingerprint, omitted lever for ablations, trigger reason for revival rows) from
the SAME result directory, parses every run's log via the shared
tools/m5max_log_parse.py (the exact parser summarize_m5max_k6_matrix.py uses,
so this report and the older K6-only one never drift on log format), and
produces:

  1. Median paired throughput per row vs baseline (Medians + Paired-vs-baseline
     tables).
  2. The stack-vs-sum-of-parts interaction table (each stack lever's marginal
     contribution, from its ablation vs the full stack; their sum vs the
     stack's own measured delta vs baseline; the residual is the interaction
     effect).
  3. Promotion verdicts per the STACK-LEVEL +-5% bar: the STACK's own verdict
     is vs baseline (PROMOTE/REVERT/KEEP PROVISIONAL, the same bar
     summarize_m5max_k6_matrix.py uses); each ABLATION's verdict is a NO-HARM
     check vs the stack itself (an existing stack member does not need its own
     +5% -- only "removing it doesn't help a lot"); each RAN revival row is
     held to the promotion bar vs the stack (adding a currently-dead/opt-in
     lever back in is a new candidate and must earn its own win, same as any
     lever that was originally promoted); each SKIPPED revival row is reported
     with its trigger's own reason.
  4. FIGHT_CARD.md.

Hash gating: DELIBERATELY different from summarize_m5max_k6_matrix.py's rule
(exactly one hash per cache/prompt across every mode). This card's STACK
includes ILI_METAL_PREFILL, which this project's own kernel-family-rounding
policy allows to fork the output hash vs a CPU baseline (README.md; K6/Metal4/
persistent-state's own axes never touched that policy, which is why the older,
stricter rule was safe there). Here, gating is SAME-CONFIG self-consistency
(every trial of one arm at one cache/prompt must hash-match -- a divergence
there is a real non-determinism bug) plus a RECORDED (never failing)
cross-arm fork table, annotated against each pair's kernel-family fingerprint
so an expected fork (kernel family differs) reads differently from an
unexplained one (kernel family identical but hashes still differ).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m5max_log_parse import Run, med, parse_log  # noqa: E402

PROMOTE_BAR = 5.0
REVERT_BAR = -5.0
NO_HARM_BAR = 5.0
INTERACTION_NOISE_BAND = 2.0


def load_runs(root: Path, manifest: dict) -> list[Run]:
    runs = []
    for item in manifest["runs"]:
        meta = {"prompt": item["prompt"], "cache_state": item["cache_state"],
                "trial": item["trial"], "mode": item["arm"]}
        runs.append(parse_log(root / item["log"], meta))
    return runs


def group_by(runs: list[Run], *keys):
    out: dict[tuple, list[Run]] = {}
    for r in runs:
        out.setdefault(tuple(getattr(r, k) for k in keys), []).append(r)
    return out


def hash_self_consistency(by_arm_cache_prompt: dict) -> list[tuple]:
    """Returns a list of (arm, cache, prompt, hashes) for any cell whose trials
    did not all agree on one output hash -- a genuine same-config
    non-determinism bug, never expected under the kernel-family policy
    (that policy only excuses CROSS-arm forks, not cross-trial ones)."""
    failures = []
    for key, items in by_arm_cache_prompt.items():
        hashes = {r.output_hash for r in items}
        if len(hashes) != 1:
            failures.append((*key, sorted(hashes)))
    return failures


def cell_medians(by_arm_cache_prompt: dict) -> dict[tuple, float]:
    return {key: med([r.tok_s for r in items]) for key, items in by_arm_cache_prompt.items()}


def paired_deltas_pct(medians: dict[tuple, float], arm_a: str, arm_b: str,
                       cache_prompt_keys: set[tuple]) -> dict[tuple, float]:
    """b vs a, as a percentage delta, over every (cache, prompt) both have."""
    out = {}
    for cache, prompt in cache_prompt_keys:
        a = medians.get((arm_a, cache, prompt))
        b = medians.get((arm_b, cache, prompt))
        if a is None or b is None:
            continue
        out[(cache, prompt)] = 100.0 * (b / a - 1.0)
    return out


def verdict_promote_bar(deltas: list[float]) -> str:
    if not deltas:
        return "NO DATA"
    m = med(deltas)
    if m >= PROMOTE_BAR and all(d > 0 for d in deltas):
        return f"PROMOTE ({m:+.2f}%, all cells positive)"
    if m <= REVERT_BAR:
        return f"REVERT/REJECT ({m:+.2f}%)"
    return f"KEEP PROVISIONAL ({m:+.2f}%; not every cell agrees or below +{PROMOTE_BAR:.0f}%)"


def verdict_no_harm(ablation_vs_stack_deltas: list[float]) -> str:
    if not ablation_vs_stack_deltas:
        return "NO DATA"
    m = med(ablation_vs_stack_deltas)
    if m >= NO_HARM_BAR:
        return f"LEVER APPEARS HARMFUL WITHIN STACK ({m:+.2f}% without it) -- consider dropping"
    return f"NO-HARM ({m:+.2f}% without it; lever earns its keep or is neutral)"


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def build_report(root: Path) -> str:
    manifest = json.loads((root / "manifest.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    runs = load_runs(root, manifest)

    arms = plan["arms"]
    arm_order = plan["arm_order"]
    stack_levers = plan["stack_levers"]
    skipped = plan["skipped"]
    ablation_arms = {a: arms[a]["omits_lever"] for a in arm_order if arms[a]["kind"] == "ablation"}
    revival_arms = [a for a in arm_order if arms[a]["kind"] == "revival"]

    by_acp = group_by(runs, "mode", "cache_state", "prompt")  # arm, cache, prompt
    hash_failures = hash_self_consistency(by_acp)
    medians = cell_medians(by_acp)
    cache_prompt_keys = {(c, p) for (_, c, p) in by_acp}

    lines: list[str] = []
    lines.append("# FIGHT Card Report")
    lines.append("")
    lines.append(f"attempt_id: `{manifest['attempt_id']}`  ")
    lines.append(f"dry_run: `{manifest['dry_run']}`  ")
    lines.append(f"passes/prompt/cache: {manifest['passes']} x 3 prompts x cache states  ")
    lines.append(f"ngen: {manifest['ngen']}  ")
    lines.append(f"quiesce_granularity: `{manifest.get('quiesce_granularity', 'unknown')}`")
    lines.append("")

    # ---- hash self-consistency (same-config; a real failure if non-empty) --------------
    lines.append("## Hash self-consistency (same-config; must be clean)")
    lines.append("")
    if hash_failures:
        lines.append("**FAIL:** output hashes diverged across trials of an identical (arm, cache, prompt):")
        lines.append("")
        for arm, cache, prompt, hashes in hash_failures:
            lines.append(f"- {arm} / {cache} / {prompt}: {hashes}")
    else:
        lines.append("PASS -- every arm hash-matched itself across all repeated trials.")
    lines.append("")

    # ---- medians table -------------------------------------------------------------------
    lines.append("## Medians")
    lines.append("")
    lines.append("| cache | prompt | arm | tok/s | hit % | disk s | matmul s | attn s | hash |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for key in sorted(by_acp):
        arm, cache, prompt = key
        items = by_acp[key]
        hashes = sorted({r.output_hash for r in items})
        hash_disp = hashes[0] if len(hashes) == 1 else f"INCONSISTENT{hashes}"
        lines.append(
            f"| {cache} | {prompt} | {arm} | {med([r.tok_s for r in items]):.3f} | "
            f"{med([r.hit_pct for r in items]):.1f} | {med([r.disk_s for r in items]):.2f} | "
            f"{med([r.matmul_s for r in items]):.2f} | {med([r.attn_s for r in items]):.2f} | {hash_disp} |"
        )
    lines.append("")

    # ---- 1. median paired throughput per row vs baseline --------------------------------
    lines.append("## Paired throughput vs baseline")
    lines.append("")
    lines.append("| cache | prompt | arm | tok/s delta vs baseline |")
    lines.append("|---|---|---|---:|")
    per_arm_baseline_deltas: dict[str, list[float]] = {}
    for arm in arm_order:
        if arm == "baseline":
            continue
        deltas = paired_deltas_pct(medians, "baseline", arm, cache_prompt_keys)
        per_arm_baseline_deltas[arm] = list(deltas.values())
        for (cache, prompt), d in sorted(deltas.items()):
            lines.append(f"| {cache} | {prompt} | {arm} | {fmt_pct(d)} |")
    lines.append("")

    lines.append("## Row summary vs baseline")
    lines.append("")
    lines.append("| arm | kind | median tok/s delta vs baseline |")
    lines.append("|---|---|---:|")
    for arm in arm_order:
        if arm == "baseline":
            continue
        ds = per_arm_baseline_deltas.get(arm, [])
        lines.append(f"| {arm} | {arms[arm]['kind']} | {fmt_pct(med(ds)) if ds else 'NO DATA'} |")
    lines.append("")

    # ---- 3. promotion verdicts -----------------------------------------------------------
    lines.append("## Promotion verdicts")
    lines.append("")
    lines.append(f"STACK-LEVEL bar: promote at >= {PROMOTE_BAR:.0f}% median paired (every cell positive); "
                  f"revert at <= {REVERT_BAR:.0f}%; existing stack members only need NO-HARM "
                  f"(ablating them must not itself clear +{NO_HARM_BAR:.0f}%).")
    lines.append("")
    lines.append("### STACK vs baseline")
    lines.append("")
    stack_deltas = per_arm_baseline_deltas.get("stack", [])
    lines.append(f"- {verdict_promote_bar(stack_deltas)}")
    lines.append("")

    lines.append("### Ablations vs STACK (no-harm bar)")
    lines.append("")
    lines.append("| ablation (lever removed) | median tok/s delta vs stack | verdict |")
    lines.append("|---|---:|---|")
    marginal_pct: dict[str, float] = {}
    for arm, lever in ablation_arms.items():
        deltas = list(paired_deltas_pct(medians, "stack", arm, cache_prompt_keys).values())
        m = med(deltas) if deltas else float("nan")
        marginal_pct[lever] = -m if deltas else float("nan")
        lines.append(f"| {lever} | {fmt_pct(m) if deltas else 'NO DATA'} | {verdict_no_harm(deltas)} |")
    lines.append("")

    lines.append("### Revival rows")
    lines.append("")
    lines.append("| name | status | detail |")
    lines.append("|---|---|---|")
    for arm in revival_arms:
        deltas = list(paired_deltas_pct(medians, "stack", arm, cache_prompt_keys).values())
        lines.append(f"| {arms[arm].get('display_name', arm)} | RAN | {verdict_promote_bar(deltas)} (vs stack) |")
    for s in skipped:
        lines.append(f"| {s['display_name']} | SKIPPED | {s['reason']} |")
    lines.append("")

    # ---- 2. stack-vs-sum-of-parts interaction table --------------------------------------
    lines.append("## Interaction: stack vs sum-of-parts")
    lines.append("")
    lines.append("Each lever's marginal contribution is the negative of its ablation-vs-stack delta "
                  "(how much WORSE the stack gets without it). The residual (actual stack delta minus "
                  "the sum of marginals) is the interaction effect: near zero means the levers are "
                  "roughly additive/independent; negative means they partially compete (e.g. for "
                  "memory bandwidth or dispatch overhead -- sub-additive); positive means synergy.")
    lines.append("")
    lines.append("| lever | marginal contribution (vs stack, from its ablation) |")
    lines.append("|---|---:|")
    for lever in stack_levers:
        m = marginal_pct.get(lever, float("nan"))
        lines.append(f"| {lever} | {fmt_pct(m) if m == m else 'NO DATA'} |")
    valid_marginals = [v for v in marginal_pct.values() if v == v]
    sum_marginals = sum(valid_marginals) if valid_marginals else float("nan")
    actual_stack_delta = med(stack_deltas) if stack_deltas else float("nan")
    lines.append("")
    lines.append(f"- Sum of marginal contributions: {fmt_pct(sum_marginals) if sum_marginals == sum_marginals else 'NO DATA'}")
    lines.append(f"- Actual STACK delta vs baseline: {fmt_pct(actual_stack_delta) if actual_stack_delta == actual_stack_delta else 'NO DATA'}")
    if sum_marginals == sum_marginals and actual_stack_delta == actual_stack_delta:
        residual = actual_stack_delta - sum_marginals
        if abs(residual) <= INTERACTION_NOISE_BAND:
            interp = "additive / independent (within noise band)"
        elif residual < 0:
            interp = "NEGATIVE interaction (sub-additive -- levers partially compete)"
        else:
            interp = "POSITIVE interaction (super-additive synergy)"
        lines.append(f"- Residual (actual - sum of parts): {fmt_pct(residual)} -- {interp}")
    else:
        lines.append("- Residual: NO DATA (missing a stack or ablation cell)")
    lines.append("")

    # ---- cross-arm hash-fork table (recorded, not gated; kernel-family-policy-aware) -----
    lines.append("## Cross-arm hash forks (recorded, not gated -- kernel-family policy)")
    lines.append("")
    lines.append("Cross-arm hash differences are EXPECTED and acceptable when the two arms differ on "
                  "any kernel-family-relevant lever (ILI_METAL_PREFILL / ILI_METAL4_MOE / "
                  "ILI_METAL_PERSISTENT_STATE / ILI_DSA) -- a rounding-level fork from dispatching "
                  "the same arithmetic to a different kernel family, ruled an acceptable faithful "
                  "forward (README.md; user ruling 2026-07-14). A fork between two arms with an "
                  "IDENTICAL kernel-family fingerprint would NOT be expected and is flagged.")
    lines.append("")
    lines.append("| cache | prompt | arm A | arm B | same hash? | kernel family differs? | flag |")
    lines.append("|---|---|---|---|---|---|---|")
    arms_by_cp: dict[tuple, list[str]] = {}
    for (arm, cache, prompt) in by_acp:
        arms_by_cp.setdefault((cache, prompt), []).append(arm)
    for (cache, prompt), cell_arms in sorted(arms_by_cp.items()):
        cell_arms = sorted(cell_arms)
        for i, a in enumerate(cell_arms):
            for b in cell_arms[i + 1:]:
                hash_a = sorted({r.output_hash for r in by_acp[(a, cache, prompt)]})
                hash_b = sorted({r.output_hash for r in by_acp[(b, cache, prompt)]})
                same_hash = hash_a == hash_b
                kf_a, kf_b = arms[a]["kernel_family"], arms[b]["kernel_family"]
                kf_differs = kf_a != kf_b
                if same_hash:
                    flag = "-"
                elif kf_differs:
                    flag = "expected fork (kernel-family policy)"
                else:
                    flag = "**UNEXPECTED FORK -- investigate**"
                lines.append(f"| {cache} | {prompt} | {a} | {b} | {same_hash} | {kf_differs} | {flag} |")
    lines.append("")

    (root / "FIGHT_CARD.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: fight_report.py RESULT_DIR")
    root = Path(sys.argv[1])
    print(build_report(root))


if __name__ == "__main__":
    main()
