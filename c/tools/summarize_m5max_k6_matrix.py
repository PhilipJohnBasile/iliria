#!/usr/bin/env python3
"""Summarize the controlled PILOT-off versus K6 benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m5max_log_parse import Run, med, parse_log  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    manifest = json.loads((root / "manifest.json").read_text())
    runs = [parse_log(root / item["log"], item) for item in manifest["runs"]]

    csv_path = root / "runs.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(runs[0]).keys()))
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))

    modes = {item["mode"] for item in manifest["runs"]}
    cand_labels = sorted(modes - {"off"})
    if len(cand_labels) != 1:
        raise SystemExit(f"expected exactly one candidate mode besides 'off', got: {sorted(modes)}")
    cand = cand_labels[0]
    cand_name = {"k6": "PILOT K6", "m4": "Metal 4 MoE", "pst": "persistent Metal state"}.get(cand, cand)

    lines = [f"# M5 Max {cand_name} Confirmation Matrix", ""]
    lines += [
        f"Runs: {len(runs)}",
        f"Trials per prompt/cache/mode: {manifest['passes']}",
        f"Generated tokens per run: {manifest['ngen']}",
        "",
        "## Medians",
        "",
        "| cache | prompt | mode | tok/s | hit % | disk s | matmul s | attn s | blocked pipe s | barrier s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    groups: dict[tuple[str, str, str], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.cache_state, run.prompt, run.mode), []).append(run)
    for key in sorted(groups):
        items = groups[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {med([r.tok_s for r in items]):.3f} | "
            f"{med([r.hit_pct for r in items]):.1f} | {med([r.disk_s for r in items]):.2f} | "
            f"{med([r.matmul_s for r in items]):.2f} | {med([r.attn_s for r in items]):.2f} | "
            f"{med([r.blocked_pipe_s for r in items]):.3f} | {med([r.barrier_s for r in items]):.3f} |"
        )

    lines += ["", f"## Paired {cand_name} deltas", "", "| cache | prompt | tok/s | hit pts | disk s | attn s |", "|---|---|---:|---:|---:|---:|"]
    paired_deltas: list[float] = []
    for cache in sorted({r.cache_state for r in runs}):
        for prompt in sorted({r.prompt for r in runs}):
            off = groups.get((cache, prompt, "off"), [])
            k6 = groups.get((cache, prompt, cand), [])
            if not off or not k6:
                continue
            off_rate, k6_rate = med([r.tok_s for r in off]), med([r.tok_s for r in k6])
            delta = 100.0 * (k6_rate / off_rate - 1.0)
            paired_deltas.append(delta)
            lines.append(
                f"| {cache} | {prompt} | {delta:+.2f}% | "
                f"{med([r.hit_pct for r in k6])-med([r.hit_pct for r in off]):+.1f} | "
                f"{med([r.disk_s for r in k6])-med([r.disk_s for r in off]):+.2f} | "
                f"{med([r.attn_s for r in k6])-med([r.attn_s for r in off]):+.2f} |"
            )

    hash_failures = []
    for cache in sorted({r.cache_state for r in runs}):
        for prompt in sorted({r.prompt for r in runs}):
            hashes = {r.output_hash for r in runs if r.cache_state == cache and r.prompt == prompt}
            if len(hashes) != 1:
                hash_failures.append((cache, prompt, hashes))

    k6_runs = [r for r in runs if r.mode == cand]
    lines += [
        "",
        "## PILOT outcomes",
        "",
        f"Median useful completed loads: {med([r.useful for r in k6_runs]):.0f}",
        f"Median wasted completed loads: {med([r.wasted for r in k6_runs]):.0f}",
        f"Median late predictions: {med([r.late for r in k6_runs]):.0f}",
        f"Median prefetch evictions: {med([r.evictions for r in k6_runs]):.0f}",
        f"Median PILOT load-worker time: {med([r.pilot_load_s for r in k6_runs]):.3f}s",
        "",
        "## Decision",
        "",
    ]
    overall = med(paired_deltas)
    consistent = sum(1 for d in paired_deltas if d > 0)
    if hash_failures:
        lines.append("**FAIL:** output hashes diverged within an identical prompt/cache group.")
    elif paired_deltas and overall >= 5.0 and consistent == len(paired_deltas):
        lines.append(f"**PROMOTE:** {cand_name} median paired improvement is {overall:+.2f}% and every group wins.")
    elif paired_deltas and overall <= -5.0:
        lines.append(f"**REVERT DEFAULT:** {cand_name} median paired result is {overall:+.2f}%.")
    else:
        lines.append(
            f"**KEEP PROVISIONAL:** median paired result is {overall:+.2f}%; "
            f"{cand_name} won {consistent}/{len(paired_deltas)} groups. Do not claim a speedup."
        )
    lines += ["", "Hash failures:"]
    if hash_failures:
        lines.extend(f"- {cache}/{prompt}: {sorted(hashes)}" for cache, prompt, hashes in hash_failures)
    else:
        lines.append("- none")
    lines += ["", f"Raw CSV: `{csv_path.name}`", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
