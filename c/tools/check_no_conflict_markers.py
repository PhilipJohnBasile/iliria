"""Grep-guard for unresolved merge-conflict markers (stdlib only).

Scans a set of paths -- default: every git-tracked file in the repo, via
`git ls-files` -- for a line that OPENS with seven '<' characters, a line
that opens with seven '=' characters, or a line that opens with seven '>'
characters: the three marker lines `git merge`/`git rebase` leave behind
when a conflict is left unresolved (git also appends a ref/branch name
after the opening and closing markers, e.g. seven '<' then " HEAD", but
the middle marker is always bare). Binary files -- detected the same way
git/grep do, by sniffing for a NUL byte -- are skipped rather than decoded.

Usage:
    python3 tools/check_no_conflict_markers.py            # scan `git ls-files`
    python3 tools/check_no_conflict_markers.py a.py b/c.c  # scan explicit paths
    python3 tools/check_no_conflict_markers.py some_dir/   # recurse a directory

Exit status: 0 if no marker line was found anywhere scanned, 1 otherwise
(also 1 if `git ls-files`/`git rev-parse` themselves fail with no explicit
paths given, e.g. invoked outside a repository).

This module's own source is not exempt from the scan, so none of the code
below spells a marker as a bare, line-initial literal -- every mention is
built from `"<" * 7` etc. or embedded after other text on the same line.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# git's own marker size (default, unconfigured `core.conflictMarkerSize`)
# is always seven repeats of the marker character. Built here by
# repetition -- not typed as a literal run at the start of a line -- so
# this scanner never flags itself.
MARKER_LEN = 7
MARKER_CHARS = ("<", "=", ">")

BINARY_SNIFF_BYTES = 8192


class Violation:
    __slots__ = ("path", "lineno", "text")

    def __init__(self, path: Path, lineno: int, text: str) -> None:
        self.path = path
        self.lineno = lineno
        self.text = text

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: {self.text}"


def is_probably_binary(data: bytes) -> bool:
    """Same cheap heuristic git/grep use: a NUL byte anywhere in the first
    chunk means treat the file as binary and never scan its text -- a real
    conflict marker is never inserted into binary blobs (git just picks a
    whole-file winner there), so skipping is safe, not just convenient."""
    return b"\0" in data[:BINARY_SNIFF_BYTES]


def _opens_with_exact_marker(line: str, ch: str) -> bool:
    """True if `line` opens with exactly MARKER_LEN copies of `ch` -- not
    merely at least that many.

    Two things can open with seven-or-more of the same character: a real
    git conflict marker (always exactly seven, by default), and a
    decorative ASCII banner comment, e.g. a row of thirty '=' characters
    used as a section divider -- this codebase has exactly that pattern in
    c/tools/eval_activation_error.py. Requiring the run to stop at exactly
    seven (the next character, if any, must differ from `ch`) is what
    tells the two apart: real markers stop there because git hard-codes
    the length, banners keep going because they don't.

    Trade-off accepted on purpose: `git merge -Xconflict-marker-size=N`
    can widen real markers past seven in deeply nested/recursive-merge
    conflicts. That's rare, and out of scope for the fast, everyday
    "did a stray marker slip through" gate this tool exists for; a
    same-size decorative banner colliding with a real conflict is rarer
    still, so exact-seven is the safer default in both directions here.
    """
    run = ch * MARKER_LEN
    if not line.startswith(run):
        return False
    return len(line) == MARKER_LEN or line[MARKER_LEN] != ch


def find_markers_in_text(text: str) -> list[tuple[int, str]]:
    """Return (1-based line number, line text) for every conflict-marker
    line in `text`. A line counts only if the marker OPENS the line (no
    leading whitespace or other text first) -- matching how git itself
    emits these, and avoiding false positives on prose that merely
    mentions a marker mid-sentence -- and is exactly seven characters long
    (see `_opens_with_exact_marker`), to avoid false positives on
    decorative banner comments of the same character."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(_opens_with_exact_marker(line, ch) for ch in MARKER_CHARS):
            hits.append((lineno, line))
    return hits


def scan_file(path: Path) -> list[Violation]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return []
    if is_probably_binary(data):
        return []
    text = data.decode("utf-8", errors="replace")
    return [Violation(path, lineno, line) for lineno, line in find_markers_in_text(text)]


def iter_tracked_files(root: Path) -> list[Path]:
    """Every path `git ls-files` reports under `root`: tracked and not
    ignored, which keeps the scan off `.git/` internals, build output,
    and anything .gitignore already excludes."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True, check=True,
    )
    names = result.stdout.split(b"\0")
    return [root / name.decode("utf-8", errors="surrogateescape") for name in names if name]


def repo_root(start: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(result.stdout.strip())


def _walk_directory(d: Path) -> list[Path]:
    return sorted(
        p for p in d.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(d).parts
    )


def collect_targets(paths: list[str]) -> list[Path]:
    """Explicit files/directories if given (recursing into directories,
    skipping any nested `.git`); otherwise every git-tracked file under
    the current repo."""
    if paths:
        targets: list[Path] = []
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                targets.extend(_walk_directory(p))
            else:
                targets.append(p)
        return targets
    return iter_tracked_files(repo_root(Path.cwd()))


def scan(paths: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for target in collect_targets(paths):
        violations.extend(scan_file(target))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*",
        help="files or directories to scan (default: `git ls-files` from the repo root)",
    )
    args = parser.parse_args(argv)

    try:
        violations = scan(args.paths)
    except subprocess.CalledProcessError as exc:
        print(f"check_no_conflict_markers: {exc}", file=sys.stderr)
        return 1

    for v in violations:
        print(str(v))
    if violations:
        print(
            f"check_no_conflict_markers: {len(violations)} unresolved conflict "
            "marker line(s) found",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
