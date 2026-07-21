#!/usr/bin/env python3
"""Regenerate docs/PERFORMANCE_THEORY.md from docs/performance-theory.json.

docs/performance-theory.json is the source of truth for iliria's
performance-theory records. This script only formats it back out to
Markdown -- it does not add or infer any content. Do not hand-edit
docs/PERFORMANCE_THEORY.md; edit the JSON and re-run this script.

Zero dependencies: python3 standard library only (json, argparse, pathlib,
sys). No PyYAML, no third-party packages -- the source format is JSON
specifically so this generator needs nothing beyond the interpreter.

Usage:
    python3 docs/gen_performance_theory.py              # regenerate in place
    python3 docs/gen_performance_theory.py --check       # exit 1 if the
                                                          # committed .md is
                                                          # stale vs. the JSON
    python3 docs/gen_performance_theory.py --source X.json --out Y.md

Determinism: rendering depends only on the JSON file's own content and
array order (the JSON source lists entries in the project's narrative
order, not alphabetically by id -- e.g. the DSA design study immediately
follows the DSA measurement it supersedes). Nothing else -- no wall-clock
timestamps, no filesystem iteration order, no locale-dependent formatting
-- feeds into the output, so re-running this script against an unchanged
JSON file always produces byte-identical Markdown.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "performance-theory.json"
DEFAULT_OUT = HERE / "PERFORMANCE_THEORY.md"

GENERATED_HEADER = (
    "<!--\n"
    "THIS FILE IS GENERATED. DO NOT EDIT DIRECTLY.\n"
    "Source: docs/performance-theory.json\n"
    "Regenerate with: python3 docs/gen_performance_theory.py\n"
    "-->\n"
)

VALID_PHASES = {"prefill", "decode", "serve", "offline"}
VALID_EVIDENCE_CLASSES = {
    "measured/same-commit",
    "measured/cross-session",
    "synthetic-fixture",
    "offline-simulation",
    "calibrated-model",
    "literature-only",
}
# Statuses are {code, display}: `code` is a small closed vocabulary so the
# three status axes stay machine-comparable across entries; `display` is
# free text (the nuanced, entry-specific wording -- "Confirmed-at-3.8K",
# "Serve-ABBA-pending", etc.) that the enum deliberately does not try to
# capture.
VALID_STATUS_CODES = {
    "confirmed", "falsified", "provisional", "pending", "null",
    "opt_in", "retained", "dead", "default_on", "enabled", "not_built",
    "shipped_anyway", "unknown", "invalidated_instrument",
}
REQUIRED_REGIME_KEYS = ("phase", "context", "cache", "batch_S", "container", "backend")
REQUIRED_STATUS_KEYS = ("mechanism", "performance", "shipping")
REQUIRED_ENTRY_KEYS = (
    "id", "name", "regime", "hypothesis", "prediction",
    "evidence_class", "measurements", "status", "notes", "superseded_by",
)


class SchemaError(ValueError):
    """Raised when the source JSON fails validation."""


def format_number(value: float) -> str:
    """Single, stable formatting function for any raw numeric value that
    appears in generated text (e.g. canonical_raw_value). Using one
    function everywhere avoids repr() drift between Python versions/runs.
    """
    if value == int(value):
        return str(int(value))
    # Fixed precision, trailing zeros stripped, always the same for the
    # same input -- no locale, no repr().
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text


def validate(data: dict) -> list[str]:
    """Return a list of human-readable schema violations (empty = valid)."""
    errors: list[str] = []
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        return ["'entries' must be a non-empty list"]

    seen_ids: set[str] = set()
    for i, e in enumerate(entries):
        where = f"entries[{i}]"
        missing = [k for k in REQUIRED_ENTRY_KEYS if k not in e]
        if missing:
            errors.append(f"{where}: missing key(s) {missing}")
            continue
        where = f"entries[{i}] ({e['id']})"

        if e["id"] in seen_ids:
            errors.append(f"{where}: duplicate id")
        seen_ids.add(e["id"])
        if e["id"] != e["id"].lower() or " " in e["id"]:
            errors.append(f"{where}: id must be kebab-case")

        regime = e.get("regime", {})
        missing_regime = [k for k in REQUIRED_REGIME_KEYS if k not in regime]
        if missing_regime:
            errors.append(f"{where}: regime missing key(s) {missing_regime}")
        elif regime["phase"] not in VALID_PHASES:
            errors.append(
                f"{where}: regime.phase {regime['phase']!r} not in {sorted(VALID_PHASES)}"
            )

        if e["evidence_class"] not in VALID_EVIDENCE_CLASSES:
            errors.append(
                f"{where}: evidence_class {e['evidence_class']!r} not in "
                f"{sorted(VALID_EVIDENCE_CLASSES)}"
            )

        measurements = e.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            errors.append(f"{where}: measurements must be a non-empty list")
        else:
            for j, m in enumerate(measurements):
                mwhere = f"{where}: measurements[{j}]"
                if not m.get("metric"):
                    errors.append(f"{mwhere} missing 'metric'")
                if not m.get("display"):
                    errors.append(f"{mwhere} missing 'display'")
                if not m.get("source"):
                    errors.append(f"{mwhere} missing 'source'")
                # 'raw_value' is a number-or-null field (0 is a legitimate
                # measurement, e.g. a confirmed null result) -- check for
                # the KEY, not truthiness. `not m.get("raw_value")` would
                # wrongly reject a real 0.
                if "raw_value" not in m:
                    errors.append(f"{mwhere} missing 'raw_value' key (use null if none)")
                elif m["raw_value"] is not None and not isinstance(m["raw_value"], (int, float)):
                    errors.append(f"{mwhere}: raw_value must be a number or null")
                if "unit" not in m:
                    errors.append(f"{mwhere} missing 'unit' key (use null if none)")
                mec = m.get("evidence_class")
                if mec is not None and mec not in VALID_EVIDENCE_CLASSES:
                    errors.append(
                        f"{mwhere}: evidence_class {mec!r} not in {sorted(VALID_EVIDENCE_CLASSES)}"
                    )

        status = e.get("status", {})
        missing_status = [k for k in REQUIRED_STATUS_KEYS if not status.get(k)]
        if missing_status:
            errors.append(f"{where}: status missing/empty key(s) {missing_status}")
        else:
            for axis in REQUIRED_STATUS_KEYS:
                axis_val = status[axis]
                if not isinstance(axis_val, dict) or not axis_val.get("display"):
                    errors.append(f"{where}: status.{axis} must be {{code, display}} with non-empty display")
                    continue
                if axis_val.get("code") not in VALID_STATUS_CODES:
                    errors.append(
                        f"{where}: status.{axis}.code {axis_val.get('code')!r} not in "
                        f"{sorted(VALID_STATUS_CODES)}"
                    )
        # mechanism/performance/shipping .display are intentionally free-text
        # (the spec's own worked examples -- "Confirmed-at-3.8K",
        # "Real-model-survival-provisional" -- are bespoke, not a closed
        # enum); only .code is validated against a fixed vocabulary.

    superseding_targets = {e["id"] for e in entries}
    for e in entries:
        sb = e.get("superseded_by")
        if sb is not None and sb not in superseding_targets:
            errors.append(
                f"entries ({e['id']}): superseded_by {sb!r} does not match any entry id"
            )

    for i, c in enumerate(data.get("source_conflicts", [])):
        where = f"source_conflicts[{i}]"
        for key in ("id", "description", "values", "resolution"):
            if key not in c:
                errors.append(f"{where}: missing key {key!r}")
        for ref in c.get("entries", []):
            if ref not in superseding_targets:
                errors.append(f"{where}: entries reference {ref!r} does not match any entry id")

    return errors


def render_regime(regime: dict) -> str:
    parts = [f"{key}=`{regime.get(key, '')}`" for key in REQUIRED_REGIME_KEYS]
    return " &middot; ".join(parts).replace("&middot;", "·")


def render_measurement(m: dict, entry_evidence_class: str) -> str:
    text = f"`{m['metric']}` — {m['display']} ({m['source']})"
    if m.get("raw_value") is not None:
        unit = f" {m['unit']}" if m.get("unit") else ""
        text += f" [raw value: {format_number(m['raw_value'])}{unit}]"
    # evidence_class lives per-measurement now, falling back to the entry's;
    # only call it out inline when a measurement overrides that default --
    # otherwise the entry-level "Evidence class" bullet already says it once.
    ec = m.get("evidence_class") or entry_evidence_class
    if ec != entry_evidence_class:
        text += f" [evidence: `{ec}`, overrides entry default `{entry_evidence_class}`]"
    return text


def render_measurements(measurements: list, entry_evidence_class: str) -> str:
    if len(measurements) == 1:
        return render_measurement(measurements[0], entry_evidence_class)
    return "\n" + "\n".join(
        f"  - {render_measurement(m, entry_evidence_class)}" for m in measurements
    )


def render_status_axis(axis: dict) -> str:
    # Bold just the short code, not the (sometimes paragraph-length) display
    # text -- a handful of entries now carry a full sentence in .display, and
    # bolding a whole paragraph reads as shouting.
    return f"**`{axis['code']}`** — {axis['display']}"


def render_status(status: dict) -> str:
    return (
        f"mechanism {render_status_axis(status['mechanism'])} · "
        f"performance {render_status_axis(status['performance'])} · "
        f"shipping {render_status_axis(status['shipping'])}"
    )


def render_entry(e: dict) -> str:
    lines = [f"## {e['name']}", "", f"<!-- id: {e['id']} -->", ""]
    lines.append(f"- **Regime:** {render_regime(e['regime'])}")
    lines.append(f"- **Hypothesis:** {e['hypothesis']}")
    lines.append(f"- **Prediction:** {e['prediction']}")
    lines.append(f"- **Evidence class:** `{e['evidence_class']}` (default; a measurement below may override)")
    lines.append(f"- **Measurement:** {render_measurements(e['measurements'], e['evidence_class'])}")
    lines.append(f"- **Status:** {render_status(e['status'])}")
    lines.append(f"- **Notes:** {e['notes']}")
    if e.get("superseded_by"):
        lines.append(f"- **Superseded by:** `{e['superseded_by']}`")
    return "\n".join(lines) + "\n"


def render_usage_note(usage_note: dict) -> str:
    lines = ["## How to read this", ""]
    for field in usage_note["fields"]:
        lines.append(f"- **{field['name']}** — {field['text']}")
    lines.append("")
    lines.append(usage_note["closing"])
    return "\n".join(lines)


def render_source_conflicts(conflicts: list) -> str:
    if not conflicts:
        return ""
    lines = ["## Cross-source conflicts (flagged, not silently resolved)", ""]
    for c in conflicts:
        lines.append(f"### {c['id']}")
        lines.append("")
        if c.get("entries"):
            refs = ", ".join(f"`{r}`" for r in c["entries"])
            lines.append(f"- **Affected entries:** {refs}")
        lines.append(f"- **What conflicts:** {c['description']}")
        for v in c.get("values", []):
            lines.append(f"  - {v}")
        if c.get("canonical_raw_value") is not None:
            lines.append(
                f"- **Canonical raw value:** {format_number(c['canonical_raw_value'])}"
            )
        lines.append(f"- **Resolution:** {c['resolution']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_sources(sources: dict) -> str:
    lines = ["## Sources", ""]
    lines.append(sources["internal_findings"])
    lines.append("")
    lines.append(sources["iliria"])
    return "\n".join(lines)


def render_markdown(data: dict) -> str:
    parts = [GENERATED_HEADER]
    parts.append(f"# {data['title']}\n")
    parts.append(f"> {data['principle']}\n")
    for para in data["intro"]:
        parts.append(para + "\n")
    parts.append(render_usage_note(data["usage_note"]) + "\n")
    parts.append("---\n")
    for entry in data["entries"]:
        parts.append(render_entry(entry))
    parts.append("---\n")
    conflicts_md = render_source_conflicts(data.get("source_conflicts", []))
    if conflicts_md:
        parts.append(conflicts_md)
        parts.append("---\n")
    parts.append(render_sources(data["sources"]) + "\n")
    return "\n".join(parts)


def load_source(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    errors = validate(data)
    if errors:
        raise SchemaError(
            f"{path}: {len(errors)} schema violation(s):\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                         help="path to performance-theory.json (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                         help="path to write PERFORMANCE_THEORY.md (default: %(default)s)")
    parser.add_argument("--check", action="store_true",
                         help="do not write; exit 1 if --out is missing or stale vs. --source")
    args = parser.parse_args(argv)

    try:
        data = load_source(args.source)
    except SchemaError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rendered = render_markdown(data)

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist", file=sys.stderr)
            return 1
        current = args.out.read_text(encoding="utf-8")
        if current != rendered:
            print(
                f"{args.out} is stale vs. {args.source} -- "
                f"run 'python3 {Path(__file__).name}' to regenerate",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} is up to date with {args.source}")
        return 0

    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out} from {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
