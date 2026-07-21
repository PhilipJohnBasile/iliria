#!/usr/bin/env python3
"""Provenance-manifest comparator: the gate that catches "you compared two different
binaries." Given two scripts/provenance.sh manifests (see tools/provenance_manifest.py for
the schema), asserts they agree on EVERY field except a small always-different bookkeeping
set (timestamps, attempt_id, ...) and a caller-declared --vary set (the field(s) the A/B
design actually intends to change). Any other difference is a hard FAIL with a diff --
especially binary_sha256 / generated_source_sha256: if the two arms of an A/B ran different
compiled engines, the comparison between them is meaningless regardless of what the results
say, and this tool exists specifically to catch that before anyone reads the results.

Usage:
    python3 tools/provenance_compare.py --a bench-m5max/abba-.../provenance-arm1.json \\
                                         --b bench-m5max/abba-.../provenance-arm2.json \\
                                         --vary ILI_METAL_PREFILL

    python3 tools/provenance_compare.py --a .../provenance-b0.json --b .../provenance-b1.json \\
                                         --vary model_dir
        (model_dir also implies container_manifest_hash and pin_profile_hash, since both are
        derived FROM the model directory and are expected to change whenever it does -- see
        IMPLIED_VARY_DEPENDENCIES.)

Matching a --vary token: a flattened manifest path (see flatten() -- "/"-joined, e.g.
"launcher_env/ILI_METAL_PREFILL" or "generated_source_sha256/glm_m5max.c"; "/" rather than
"." specifically so a dot INSIDE a filename key, e.g. "glm_m5max.c", is never ambiguous with a
path separator) is exempted from the strict-match requirement if the token equals either the
full path or its final segment -- so `--vary ILI_METAL_PREFILL` reaches into launcher_env
without the caller needing to spell out the full path, and `--vary model_dir` matches the
top-level key directly.

Exit codes: 0 = manifests match (modulo ignored/varied fields); 1 = a non-varied field
differs (the hard-fail gate); 2 = usage/IO error (bad args, missing/unparseable file).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Fields that differ between ANY two manifests by construction (per-run bookkeeping, not
# evidence about what ran) -- always exempt, never need to be named via --vary.
ALWAYS_IGNORED = frozenset({
    "attempt_id",
    "generated_at",
    "start_ts",
    "end_ts",
    "manifest_path",
    "quiesce/output",        # live-sampled telemetry text + a timestamp line: always differs
    "launcher_env_digest",   # derived from launcher_env/*, which IS compared field-by-field;
                             # comparing the digest too would make a declared --vary of one env
                             # var (e.g. ILI_METAL_PREFILL) fail anyway, since the digest folds
                             # every var into one hash with no way to exempt a single key.
})

# A --vary key that legitimately implies other fields must also be allowed to differ, because
# those fields are DERIVED from it rather than independent confounds. Keyed by the token as a
# caller would type it (matched the same way as any other --vary entry: full path or final
# segment).
IMPLIED_VARY_DEPENDENCIES = {
    "model_dir": ("container_manifest_hash", "pin_profile_hash"),
}

# Paths whose mismatch gets a loud callout: exactly the two fields the task motivating this
# tool exists to catch ("a binary mismatch between two A/B arms invalidates the comparison").
CRITICAL_PATHS_PREFIXES = ("binary_sha256", "generated_source_sha256")


def flatten(obj, prefix: str = "") -> dict:
    """Recursively flattens a JSON-decoded manifest into {"/"-joined path: leaf value}. Lists
    are kept as atomic leaf values (this schema's only list is quiesce.output, which is
    always-ignored anyway) -- only dicts are descended into."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            child_prefix = f"{prefix}/{k}" if prefix else k
            out.update(flatten(v, child_prefix))
        return out
    return {prefix: obj}


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


def expand_vary(vary):
    expanded = set(vary)
    for token in list(expanded):
        expanded.update(IMPLIED_VARY_DEPENDENCIES.get(token, ()))
    return expanded


def is_exempt(path: str, vary: set) -> bool:
    if path in ALWAYS_IGNORED:
        return True
    final = path.rsplit("/", 1)[-1]
    return path in vary or final in vary


def is_critical(path: str) -> bool:
    final = path.rsplit("/", 1)[-1] if "/" in path else path
    top = path.split("/", 1)[0]
    return top in CRITICAL_PATHS_PREFIXES or final in CRITICAL_PATHS_PREFIXES


_MISSING = object()


def compare_manifests(manifest_a: dict, manifest_b: dict, vary) -> dict:
    """Returns {"mismatches": [...], "exempted": [...]}. Each mismatch/exempted entry is a
    dict with path/value_a/value_b (value is _MISSING's repr "<absent>" when a key exists in
    only one manifest -- itself a real, reportable difference, never silently dropped)."""
    flat_a = flatten(manifest_a)
    flat_b = flatten(manifest_b)
    vary_expanded = expand_vary(vary)

    mismatches = []
    exempted = []
    for path in sorted(set(flat_a) | set(flat_b)):
        va = flat_a.get(path, _MISSING)
        vb = flat_b.get(path, _MISSING)
        if path in ALWAYS_IGNORED:
            continue
        differs = va != vb
        if is_exempt(path, vary_expanded):
            if differs:
                exempted.append({"path": path, "value_a": _fmt(va), "value_b": _fmt(vb)})
            continue
        if differs:
            mismatches.append({
                "path": path, "value_a": _fmt(va), "value_b": _fmt(vb),
                "critical": is_critical(path),
            })
    return {"mismatches": mismatches, "exempted": exempted}


def _fmt(v):
    return "<absent>" if v is _MISSING else v


def format_report(result: dict, path_a: str, path_b: str) -> str:
    lines = []
    if not result["mismatches"]:
        lines.append(f"MATCH: {path_a} and {path_b} agree on every non-varied field.")
        if result["exempted"]:
            lines.append("Declared variance (--vary), differs as expected:")
            for e in result["exempted"]:
                lines.append(f"  {e['path']}: A={e['value_a']!r} B={e['value_b']!r}")
        return "\n".join(lines)

    lines.append(f"MISMATCH: {path_a} and {path_b} differ on {len(result['mismatches'])} "
                 f"non-varied field(s) -- this comparison is NOT valid.")
    critical = [m for m in result["mismatches"] if m["critical"]]
    if critical:
        lines.append("")
        lines.append("*** CRITICAL: the executable itself differs between these two "
                     "manifests -- any result comparison built on them is meaningless. ***")
        for m in critical:
            lines.append(f"  {m['path']}: A={m['value_a']!r} B={m['value_b']!r}")
    other = [m for m in result["mismatches"] if not m["critical"]]
    if other:
        lines.append("")
        lines.append("Other non-varied differences:")
        for m in other:
            lines.append(f"  {m['path']}: A={m['value_a']!r} B={m['value_b']!r}")
    if result["exempted"]:
        lines.append("")
        lines.append("Declared variance (--vary), differs as expected (not counted above):")
        for e in result["exempted"]:
            lines.append(f"  {e['path']}: A={e['value_a']!r} B={e['value_b']!r}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="first provenance manifest JSON file")
    parser.add_argument("--b", required=True, help="second provenance manifest JSON file")
    parser.add_argument("--vary", action="append", default=[],
                        help="a field allowed to differ between --a and --b (repeatable); "
                             "matches either the full flattened path or its final segment, "
                             "e.g. ILI_METAL_PREFILL or model_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest_a = load_manifest(args.a)
        manifest_b = load_manifest(args.b)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not load a manifest: {exc}", file=sys.stderr)
        sys.exit(2)

    result = compare_manifests(manifest_a, manifest_b, args.vary)
    print(format_report(result, args.a, args.b))
    sys.exit(1 if result["mismatches"] else 0)


if __name__ == "__main__":
    main()
