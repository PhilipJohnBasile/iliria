#!/usr/bin/env python3
"""Resolve a fight-card JSON (see scripts/fight_card.default.json) into a
concrete, fully-expanded execution plan: the baseline row, the STACK row, one
ablation row per stack lever (mechanically "stack minus one lever" -- never
hand-listed, so it cannot drift out of sync with the stack's own lever list),
and any GRAVEYARD REVIVAL rows whose trigger currently evaluates to met.

This is a pure config/decision step (no engine, no filesystem writes outside
its own stdout) so fight_card.sh's --dry-run and real-mode paths share the
exact same resolution logic -- the plan is computed once, identically, either
way; only what fight_card.sh DOES with each arm (mock engine vs real engine)
differs.

Usage:
  python3 tools/fight_plan.py CARD.json
      [--set-measurement metric=value ...]
      [--force-revival name[,name...]]
      [--perf-theory-json path/to/performance-theory.json]
  Prints the resolved plan as JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def resolve_env(fixed_env: dict, lever_env: dict, add_env: dict | None = None) -> dict:
    merged = dict(fixed_env)
    merged.update(lever_env)
    if add_env:
        merged.update(add_env)
    return {str(k): str(v) for k, v in merged.items()}


def kernel_family_fingerprint(env: dict) -> dict:
    """The subset of resolved env that can legitimately fork the output hash
    (dispatches different arithmetic to a different kernel family), per this
    project's own kernel-family-rounding policy. Used by fight_report.py to
    decide whether a cross-arm hash difference is an EXPECTED fork or a
    genuine same-family non-determinism bug -- not used here, but computed
    once, in one place, so both fight_card.sh's plan and fight_report.py agree
    on exactly which levers count."""
    keys = ("ILI_METAL_PREFILL", "ILI_METAL4_MOE", "ILI_METAL_PERSISTENT_STATE", "ILI_DSA")
    return {k: env.get(k, "0") for k in keys}


def evaluate_trigger(trigger: dict, row_name: str, measured: dict, forced: set[str],
                      perf_theory_path: Path) -> tuple[bool, str]:
    if row_name in forced:
        return True, "forced via --force-revival (test/override), trigger not actually evaluated"

    kind = trigger["kind"]
    if kind == "measured_metric":
        metric = trigger["metric"]
        value = measured.get(metric)
        if value is None:
            return False, (
                f"no measured value on record for metric '{metric}'; supply one with "
                f"--set-measurement {metric}=<value> (fail-closed: unmeasured means do not run)"
            )
        comparator = trigger["comparator"]
        threshold = trigger["threshold"]
        ok = {"lt": value < threshold, "le": value <= threshold,
              "gt": value > threshold, "ge": value >= threshold}[comparator]
        verb = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}[comparator]
        if ok:
            return True, f"measured {metric}={value} {verb} threshold {threshold}: trigger met"
        return False, f"measured {metric}={value} does not satisfy {verb} {threshold}: trigger not met"

    if kind == "performance_theory_status":
        if not perf_theory_path.exists():
            return False, f"performance-theory.json not found at {perf_theory_path}; fail-closed"
        data = json.loads(perf_theory_path.read_text())
        by_id = {e["id"]: e for e in data["entries"]}
        watch_ids = trigger["watch_ids"]
        requires = set(trigger["requires_shipping_in"])
        mode = trigger.get("any_or_all", "any")
        statuses = {}
        for wid in watch_ids:
            entry = by_id.get(wid)
            statuses[wid] = entry["status"]["shipping"]["code"] if entry else "<unknown id>"
        satisfied = {wid: (code in requires) for wid, code in statuses.items()}
        met = any(satisfied.values()) if mode == "any" else all(satisfied.values())
        detail = ", ".join(f"{wid}={code}" for wid, code in statuses.items())
        if met:
            return True, f"{mode}-of watched ids satisfy required shipping status ({detail}): trigger met"
        return False, f"none of the watched ids ({detail}) are in {sorted(requires)} yet: trigger not met"

    return False, f"unknown trigger kind: {kind!r}"


def build_plan(card: dict, measured: dict, forced: set[str], perf_theory_path: Path) -> dict:
    containers = {name: c["path"] for name, c in card["containers"].items()}
    default_container = card["default_container"]
    fixed_env = card["fixed_env"]
    levers = card["levers"]
    stack_levers = card["stack"]["levers"]
    stack_container = card["stack"].get("container", default_container)

    arms: dict[str, dict] = {}
    arm_order: list[str] = []
    skipped: list[dict] = []

    def add_arm(name: str, kind: str, lever_values: dict, container: str,
                add_env: dict | None = None, extra: dict | None = None) -> None:
        env = resolve_env(fixed_env, lever_values, add_env)
        requires_lab_build = any(
            levers[lv]["requires_lab_build"]
            for lv in stack_levers
            if lever_values.get(lv) == levers[lv]["candidate_value"]
        )
        arm = {
            "kind": kind,
            "container": container,
            "container_path": containers[container],
            "env": env,
            "kernel_family": kernel_family_fingerprint(env),
            "requires_lab_build": requires_lab_build,
        }
        if extra:
            arm.update(extra)
        arms[name] = arm
        arm_order.append(name)

    # ---- baseline: every stack lever at its shipped_default_value (today's real launcher
    #      default -- "what a user gets today with no fight-card overrides"). -------------
    baseline_levers = {lv: levers[lv]["shipped_default_value"] for lv in stack_levers}
    add_arm("baseline", "baseline", baseline_levers, default_container)

    # ---- stack: every stack lever at its candidate_value ----------------------------------
    stack_lever_values = {lv: levers[lv]["candidate_value"] for lv in stack_levers}
    add_arm("stack", "stack", stack_lever_values, stack_container)

    # ---- ablations: mechanically "stack minus one lever" (never hand-listed). Each ablated
    #      lever is forced to its off_value (NOT shipped_default_value -- for a lever that is
    #      ALREADY shipped default_on, e.g. ILI_METAL_PREFILL, those two differ: off_value
    #      is what actually removes the lever's effect, so this is the only correct target
    #      for "what if the stack didn't have this lever". Using shipped_default_value here
    #      was tried first and silently degenerated ablate_ILI_METAL_PREFILL into a byte-for
    #      byte copy of the stack row -- caught by the --dry-run proof run precisely because
    #      the two arms' env dicts (and, in a real run, hashes) turned out identical. --------
    for lv in stack_levers:
        ablated = dict(stack_lever_values)
        ablated[lv] = levers[lv]["off_value"]
        add_arm(f"ablate_{lv}", "ablation", ablated, stack_container, extra={"omits_lever": lv})

    # ---- graveyard revival: only added if triggered ---------------------------------------
    for rev in card.get("revival", []):
        name = rev["name"]
        met, reason = evaluate_trigger(rev["trigger"], name, measured, forced, perf_theory_path)
        if met:
            add_arm(name, "revival", stack_lever_values, stack_container,
                    add_env=rev["add_env"],
                    extra={"display_name": rev["display_name"], "status_ref": rev["status_ref"],
                           "trigger_reason": reason})
        else:
            skipped.append({
                "name": name, "display_name": rev["display_name"],
                "status_ref": rev["status_ref"], "reason": reason,
            })

    needs_lab_build = any(a["requires_lab_build"] for a in arms.values())

    return {
        "matrix": card["matrix"],
        "quiesce_granularity": card.get("quiesce_granularity", "per_cell"),
        "containers": containers,
        "stack_levers": stack_levers,
        "arms": arms,
        "arm_order": arm_order,
        "skipped": skipped,
        "needs_lab_build": needs_lab_build,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("card", type=Path)
    parser.add_argument("--set-measurement", action="append", default=[], metavar="metric=value")
    parser.add_argument("--force-revival", default="", help="comma-separated row names (test/override escape hatch)")
    parser.add_argument(
        "--perf-theory-json",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "performance-theory.json",
    )
    args = parser.parse_args()

    measured = {}
    for item in args.set_measurement:
        if "=" not in item:
            raise SystemExit(f"--set-measurement expects metric=value, got: {item!r}")
        k, v = item.split("=", 1)
        measured[k] = float(v)
    forced = {n for n in args.force_revival.split(",") if n}

    card = json.loads(args.card.read_text())
    plan = build_plan(card, measured, forced, args.perf_theory_json)
    json.dump(plan, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
