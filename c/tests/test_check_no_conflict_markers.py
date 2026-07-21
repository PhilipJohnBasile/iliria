"""tools/check_no_conflict_markers.py: planted-marker fixtures are caught,
a clean tree passes, binary files are skipped without crashing, and the
match is anchored to the start of the line (not a mid-line mention).

Every fixture below builds marker text from repeated characters (`"<" * 7`
etc.) rather than typing a bare marker line into this file's own source,
so this test file is never mistaken for the conflicts it is testing for.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import check_no_conflict_markers as cncm  # noqa: E402

TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "check_no_conflict_markers.py"

LEFT = "<" * 7
MID = "=" * 7
RIGHT = ">" * 7


def conflicted_text() -> str:
    return "\n".join([
        "def before():",
        "    return 1",
        f"{LEFT} HEAD",
        "    return 1",
        MID,
        "    return 2",
        f"{RIGHT} feature-branch",
        "def after():",
        "    return 3",
        "",
    ])


def clean_text() -> str:
    return "\n".join([
        "def before():",
        "    return 1",
        "def after():",
        "    return 3",
        "",
    ])


class FindMarkersInTextTest(unittest.TestCase):
    def test_clean_text_has_no_hits(self):
        self.assertEqual(cncm.find_markers_in_text(clean_text()), [])

    def test_conflicted_text_reports_all_three_lines_with_correct_numbers(self):
        hits = cncm.find_markers_in_text(conflicted_text())
        self.assertEqual([lineno for lineno, _ in hits], [3, 5, 7])
        self.assertEqual(hits[0][1], f"{LEFT} HEAD")
        self.assertEqual(hits[1][1], MID)
        self.assertEqual(hits[2][1], f"{RIGHT} feature-branch")

    def test_six_char_run_is_not_a_marker(self):
        # One short of the real marker -- must not false-positive.
        short = ("<" * 6) + " not actually a conflict"
        self.assertEqual(cncm.find_markers_in_text(short), [])

    def test_longer_run_is_a_banner_not_a_marker(self):
        # git's own marker is always exactly seven characters; an eighth
        # copy of the same character means this is a decorative ASCII
        # banner (this codebase has exactly this pattern -- thirty '='
        # characters -- in c/tools/eval_activation_error.py), not a real
        # conflict marker, and must not be flagged.
        eight_chars = ("<" * 8) + " HEAD"
        self.assertEqual(cncm.find_markers_in_text(eight_chars), [])

        banner = "=" * 30 + " THE .npy CONTRACT " + "=" * 30
        self.assertEqual(cncm.find_markers_in_text(banner), [])

    def test_exactly_seven_followed_by_other_text_still_matches(self):
        # The boundary this codebase actually needs: seven, then a space
        # and a ref name (real git output) must still be caught.
        self.assertEqual(len(cncm.find_markers_in_text(f"{LEFT} HEAD")), 1)
        self.assertEqual(len(cncm.find_markers_in_text(MID)), 1)
        self.assertEqual(len(cncm.find_markers_in_text(f"{RIGHT} feature-branch")), 1)

    def test_marker_not_at_start_of_line_is_ignored(self):
        # Anchored to line start: a mid-line mention (e.g. in prose or a
        # code comment quoting the marker) must not trigger.
        text = f"    # example: a line reading {LEFT} HEAD is a conflict marker"
        self.assertEqual(cncm.find_markers_in_text(text), [])


class ScanFileTest(unittest.TestCase):
    def test_clean_file_yields_no_violations(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.txt"
            path.write_text(clean_text())
            self.assertEqual(cncm.scan_file(path), [])

    def test_conflicted_file_yields_three_violations(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.txt"
            path.write_text(conflicted_text())
            violations = cncm.scan_file(path)
            self.assertEqual(len(violations), 3)
            self.assertEqual([v.lineno for v in violations], [3, 5, 7])
            self.assertIn(str(path), str(violations[0]))

    def test_binary_file_is_skipped_without_crashing(self):
        binary_content = (
            b"\x00\x01\x02binary junk \xff\xfe"
            + LEFT.encode() + b" HEAD\n"  # even with marker bytes present
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.bin"
            path.write_bytes(binary_content)
            self.assertEqual(cncm.scan_file(path), [])

    def test_missing_file_warns_and_returns_no_violations(self):
        missing = Path(tempfile.gettempdir()) / "does-not-exist-check-no-conflict-markers.txt"
        self.assertEqual(cncm.scan_file(missing), [])


class CollectTargetsDirectoryTest(unittest.TestCase):
    """Exercises the explicit-directory-argument path (recursion + the
    nested-.git skip), independent of any real git repository."""

    def test_directory_scan_finds_planted_marker_in_a_nested_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "sub").mkdir()
            (root / "sub" / "clean.py").write_text(clean_text())
            (root / "sub" / "conflicted.py").write_text(conflicted_text())
            # A nested .git-like directory must be skipped even when a
            # plain directory (not `git ls-files`) is the scan target.
            (root / ".git").mkdir()
            (root / ".git" / "COMMIT_EDITMSG").write_text(f"{LEFT} should not be scanned")

            violations = cncm.scan([str(root)])
            paths_hit = {str(v.path) for v in violations}
            self.assertEqual(len(violations), 3)
            self.assertTrue(any("conflicted.py" in p for p in paths_hit))
            self.assertFalse(any(".git" in p for p in paths_hit))

    def test_directory_scan_of_all_clean_files_finds_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.py").write_text(clean_text())
            (root / "b.py").write_text(clean_text())
            self.assertEqual(cncm.scan([str(root)]), [])


class CliTest(unittest.TestCase):
    """Black-box: invoke the tool exactly as the pre-push hook / CI would."""

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(TOOL_PATH), *args],
            capture_output=True, text=True,
        )

    def test_clean_tree_exits_zero_with_no_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.py").write_text(clean_text())
            (root / "b.py").write_text(clean_text())
            result = self.run_cli(str(root))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout, "")

    def test_planted_marker_fixture_exits_one_and_reports_the_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            planted = root / "conflicted.py"
            planted.write_text(conflicted_text())
            (root / "clean.py").write_text(clean_text())

            result = self.run_cli(str(root))
            self.assertEqual(result.returncode, 1)
            self.assertIn("conflicted.py", result.stdout)
            self.assertIn("3 unresolved conflict marker line(s) found", result.stderr)

    def test_explicit_single_file_argument(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            planted = root / "conflicted.py"
            planted.write_text(conflicted_text())
            result = self.run_cli(str(planted))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout.count("\n"), 3)


if __name__ == "__main__":
    unittest.main()
