"""scripts/collapse_detector.py: unit tests pinning the 2026-07-15 gate A near-miss.

What actually happened: the defective-int2 mixed container collapsed into
degenerate loops emitted as ONE unbroken newline-less line
(collapse-p1.txt: '</think></think>...' repeated ~270x, then EOS at 276
tokens; collapse-p2.txt: "the C source-compiler's, " repeated for
kilobytes). run_container_gates.sh's then-heuristic -- chars<2000 || any
60-char LINE repeated >5x (awk, per-line) -- saw each giant line exactly
once and logged 'gate A p1: 3449 chars, max-repeated-line x1': a false
PASS that only a human eye caught (c/bench-m5max/container-20260715/
collapse-semantic-verdict.md). These tests rebuild that exact shape
synthetically and assert the OLD heuristic passes it (the bug,
demonstrated verbatim) while the NEW newline-agnostic detector fails it;
plus a healthy-essay fixture BOTH accept, char-floor retention, awk
parity of the legacy CSV column, generated-span extraction (including
the killed-run/no-footer shape), and the CLI contract
run_container_gates.sh relies on (csv row on stdout, exit 3 on collapse).

No engine, no network, no model or bench-output reads: fixtures are
built in tempdirs, mimicking the real engine stdout wrapper (banner +
prompt echo + generation + stats footer).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
DETECTOR = C_DIR / "scripts" / "collapse_detector.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cd = load_module("collapse_detector", DETECTOR)

# The pre-fix heuristic, verbatim from run_container_gates.sh as of commit
# 19a50d0 (the version that was live during the 20260715 gate A run). Kept
# here as a regression witness: these tests PROVE it passes a degenerate
# single-line loop, which is exactly why it was replaced.
OLD_AWK = (
    "{ s=substr($0,1,60); c[s]++ } "
    "END { m=0; for (k in c) if (c[k]>m) m=c[k]; print m }"
)


def old_awk_reps(path: Path) -> int:
    out = subprocess.run(["awk", OLD_AWK, str(path)],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def old_gate_passes(path: Path) -> bool:
    """The old COLLAPSE condition was: chars < 2000 || reps > 5."""
    return path.stat().st_size >= 2000 and old_awk_reps(path) <= 5


# --- fixtures: the real engine stdout wrapper around a synthetic span -------

BANNER = (
    "== GLM C engine (glm_moe_dsa), cache=8 experts/layer | experts@8-bit dense@8-bit | idot: neon ==\n"
    "loaded in 1.19s | resident dense: 9647.10 MB | layers=78 experts=256 | MTP absent (draft=0)\n"
    "prompt: 48 tokens | generating up to 1600 (EOS stop=154820) | n-gram draft=0\n"
)
FOOTER = (
    "\n---\n"
    "276 tokens in 201.37s (1.37 tok/s) | expert hit rate 92.1% | RSS 82.60 GB\n"
    "experts loaded/token: 704.3 (per-layer 9.39 across 75; baseline topk=8) | TOPK=0 TOPP=0.00\n"
    "speculation: 1.00 tokens/forward (276 forwards per 276 tokens) | MTP acceptance 0% (0/0)\n"
    "output token hash: e515fa03deaf6721\n"
    "PROFILE: expert-disk 13.558s | expert-matmul 91.857s | attention 84.258s (including kvb 0.000s) | lm_head 0.007s | other 11.693s\n"
    "METAL-ATTN: layer GPU 21606 | gpu-wall 82.67s (kernel 70.49s | cpu-sched 0.35s gpu-sched 9.87s)\n"
    "METAL: blocchi GPU 0 | fallback CPU 0 | expert su GPU 0 | setup 0.00s gpu-wall 0.00s (kernel 0.00s) scatter 0.00s\n"
    "METAL-GEMM: calls 606 | gpu-wall 1.81s (kernel 1.57s)\n"
)
PROMPT = ("Write a detailed technical essay on how operating systems schedule "
          "processes, covering priorities, preemption, and fairness.")


def engine_file(gen: str, footer: bool = True) -> str:
    """BANNER + prompt echo + generated span [+ stats footer], the exact
    stdout shape run-m5max-fast.sh captures into collapse-pN.txt. With
    footer=False this is the killed-mid-generation shape (real
    collapse-p2.txt ends mid-word with no stats block)."""
    body = BANNER + f"[gMASK]<sop><|user|>{PROMPT}<|assistant|>" + gen
    return body + FOOTER if footer else body

# p1 shape: special-token loop with NO whitespace anywhere in the loop --
# whitespace tokenization sees ONE giant word, so only the 60-char shingle
# signal can catch it. One unbroken line, exactly like the real output.
GEN_NO_WHITESPACE_LOOP = "<think></think>\n" + "</think>" * 400

# p2 shape (the near-miss the task force actually hit): a short pseudo-
# English preamble, then one newline-less repeated phrase for kilobytes.
GEN_PHRASE_LOOP = ("<think></think>Lefort's Castle's and of code-to-Eachly. "
                   + "the C source-compiler's, " * 150)

# Healthy control: varied technical prose, multiple paragraphs (blank
# lines deliberately <= 5: the OLD heuristic counted blank lines as
# repeated ''-lines, one of its latent false-positive modes).
GEN_HEALTHY = """<think></think>Operating systems schedule processes by balancing three pressures: responsiveness, throughput, and fairness. A desktop kernel leans toward responsiveness, waking the compositor within microseconds of an input interrupt, while a batch cluster happily trades latency for cache-warm throughput.

Priorities express which pressure wins. Classic Unix nice values bias a decaying CPU-usage estimate; the more a task runs, the further its dynamic priority sinks, letting starved peers overtake it. Real-time classes sit above this decay entirely, which is why a runaway SCHED_FIFO thread can wedge a core until the watchdog fires.

Preemption is the enforcement arm. On every timer interrupt the kernel compares the running task against the head of the ready queue, and if the incumbent has exhausted its slice it is descheduled at the next safe point. Context switches are not free: saving registers, switching page tables, and refilling the TLB costs microseconds, so schedulers deliberately stretch slices for CPU-bound work.

Fairness is bookkeeping over time. CFS tracks virtual runtime per task and always runs the one furthest behind, approximating an ideal processor that gives every runnable thread an equal share. Multicore adds load balancing on top: idle cores steal work, but only when the migration cost is amortized by the imbalance, since dragging a hot working set across NUMA nodes is often slower than waiting.

Timer interrupts themselves are a tunable cost. A 1000 Hz tick gives crisp slices but burns cycles in the handler; tickless kernels arm the next interrupt only when a deadline actually approaches, which matters on laptops where every wakeup fights the battery. The scheduler, in the end, is a policy engine wearing a mechanism's clothes: the arithmetic is simple, and the art is choosing what to count."""


def write_fixture(tmp: str, name: str, content: str) -> Path:
    p = Path(tmp) / name
    p.write_text(content)
    return p


class NearMissRegressionTests(unittest.TestCase):
    """The heart of the fix: OLD heuristic passes, NEW detector collapses."""

    def test_p2_shape_single_line_phrase_loop_old_passes_new_collapses(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fixture(tmp, "p2.txt", engine_file(GEN_PHRASE_LOOP))
            # The demonstrated bug: a multi-kilobyte degenerate loop on one
            # unbroken line sails through the old line-based check.
            self.assertTrue(old_gate_passes(p),
                            "old heuristic was supposed to (wrongly) pass this")
            r = cd.analyze(p.read_bytes())
            self.assertLessEqual(r["line60_reps"], 5,
                                 "legacy line metric must stay blind here -- "
                                 "that blindness is the documented bug")
            self.assertGreater(r["word8_reps"], 5)
            self.assertGreater(r["shingle60_reps"], 5)
            self.assertTrue(r["collapse"])

    def test_p1_shape_no_whitespace_token_loop_old_passes_new_collapses(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fixture(tmp, "p1.txt", engine_file(GEN_NO_WHITESPACE_LOOP))
            self.assertTrue(old_gate_passes(p),
                            "old heuristic was supposed to (wrongly) pass this")
            r = cd.analyze(p.read_bytes())
            # No whitespace inside the loop => word n-grams cannot exist;
            # the shingle signal is the one that must catch this shape.
            self.assertEqual(r["word8_reps"], 0)
            self.assertGreater(r["shingle60_reps"], 5)
            self.assertTrue(r["collapse"])

    def test_healthy_essay_passes_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fixture(tmp, "healthy.txt", engine_file(GEN_HEALTHY))
            self.assertGreater(p.stat().st_size, 2000, "fixture must clear the floor")
            self.assertTrue(old_gate_passes(p))
            r = cd.analyze(p.read_bytes())
            self.assertFalse(r["collapse"])
            # Healthy prose should sit far below the K=5 threshold, not
            # graze it -- headroom is part of the contract.
            self.assertLessEqual(r["word8_reps"], 2)
            self.assertLessEqual(r["shingle60_reps"], 2)


class CharFloorTests(unittest.TestCase):
    def test_short_output_still_collapses_under_both(self):
        # The 2000-char whole-file floor is retained unchanged: a tiny
        # output fails even with zero repetition.
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fixture(tmp, "short.txt",
                              BANNER + f"[gMASK]<sop><|user|>{PROMPT}<|assistant|>Sure.\n")
            self.assertLess(p.stat().st_size, 2000)
            self.assertFalse(old_gate_passes(p))
            r = cd.analyze(p.read_bytes())
            self.assertTrue(r["collapse"])
            self.assertEqual(r["word8_reps"], 0)


class LegacyColumnParityTests(unittest.TestCase):
    def test_line60_reps_matches_the_original_awk_exactly(self):
        # collapse-summary.csv column 3 keeps its old meaning ONLY if the
        # python reimplementation agrees with the retired awk one-liner.
        cases = {
            "p1.txt": engine_file(GEN_NO_WHITESPACE_LOOP),
            "p2.txt": engine_file(GEN_PHRASE_LOOP),
            "healthy.txt": engine_file(GEN_HEALTHY),
            "killed.txt": engine_file(GEN_PHRASE_LOOP, footer=False),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in cases.items():
                p = write_fixture(tmp, name, content)
                self.assertEqual(cd.line60_reps(content), old_awk_reps(p),
                                 f"awk parity broken for {name}")


class SpanExtractionTests(unittest.TestCase):
    def test_roundtrip_between_prompt_echo_and_footer(self):
        self.assertEqual(cd.extract_generated_span(engine_file(GEN_PHRASE_LOOP)),
                         GEN_PHRASE_LOOP)

    def test_killed_run_without_footer_is_still_analyzable(self):
        # Real collapse-p2.txt: the chain was killed mid-generation, so the
        # file ends mid-word with no stats footer. Extraction must degrade
        # gracefully and the verdict must still be collapse.
        text = engine_file(GEN_PHRASE_LOOP, footer=False)
        self.assertEqual(cd.extract_generated_span(text), GEN_PHRASE_LOOP)
        self.assertTrue(cd.analyze(text.encode())["collapse"])

    def test_plain_text_without_markers_is_the_whole_span(self):
        self.assertEqual(cd.extract_generated_span("no markers at all here"),
                         "no markers at all here")

    def test_generation_containing_dashes_line_is_not_truncated(self):
        # The footer cut keys on the LAST '---' line followed by the token
        # stats -- a '---' inside the generation itself must survive.
        gen = "alpha beta gamma\n---\ndelta epsilon zeta eta theta"
        self.assertEqual(cd.extract_generated_span(engine_file(gen)), gen)


class CliContractTests(unittest.TestCase):
    """run_container_gates.sh consumes: csv row on stdout, exit 3 = collapse,
    exit 0 = ok, any other nonzero = error (treated as collapse upstream)."""

    def run_cli(self, path: Path):
        return subprocess.run([sys.executable, str(DETECTOR), str(path)],
                              capture_output=True, text=True)

    def test_collapse_gives_exit_3_and_a_csv_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fixture(tmp, "p2.txt", engine_file(GEN_PHRASE_LOOP))
            proc = self.run_cli(p)
            self.assertEqual(proc.returncode, 3, proc.stderr)
            row = proc.stdout.strip()
            fields = row.split(",")
            self.assertEqual(len(fields), 5, row)
            self.assertEqual(fields[0], str(p.stat().st_size))
            self.assertEqual(fields[4], "COLLAPSE")
            # Quoted evidence for the human reviewer goes to stderr.
            self.assertIn("most-repeated", proc.stderr)

    def test_healthy_gives_exit_0_and_ok_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fixture(tmp, "healthy.txt", engine_file(GEN_HEALTHY))
            proc = self.run_cli(p)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(proc.stdout.strip().endswith(",OK"), proc.stdout)

    def test_missing_file_gives_exit_2(self):
        proc = self.run_cli(Path("/nonexistent/collapse-p9.txt"))
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
