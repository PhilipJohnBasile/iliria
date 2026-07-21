#!/usr/bin/env python3
"""Newline-agnostic long-generation collapse detector (gate A helper).

Why this exists -- the 2026-07-15 near-miss: gate A's original inline
heuristic in scripts/run_container_gates.sh was

    chars < 2000  OR  any exact 60-char LINE PREFIX repeated > 5 times

i.e. newline-DEPENDENT. The defective-int2 mixed container collapsed into
degenerate loops emitted as ONE unbroken line with no newlines at all
(p1: '</think></think></think>...', p2: "the C source-compiler's, "
repeated for kilobytes), so the awk line counter saw each giant line
exactly once and reported max-repeated-line x1 -- a false PASS on an
unambiguous collapse (collapse-p1.txt: 3449 chars, x1). A human caught it
by eye (the container collapse-semantic verdict).

This detector normalizes ALL whitespace before looking for repetition, so
line structure is irrelevant. Two independent, newline-agnostic signals;
either one trips the gate (same K=5 threshold the old line check used):

  - word 8-gram repetition: tokenize the generated span on whitespace and
    count every consecutive-8-word window. A healthy essay never repeats
    an exact 8-word phrase more than a few times; a phrase loop repeats
    its windows hundreds of times. (Catches p2-style loops. Cannot catch
    p1: '</think></think>...' contains no whitespace at all, so the whole
    loop tokenizes as ONE giant word and no 8-gram ever repeats.)
  - 60-char shingle repetition: collapse all whitespace runs to single
    spaces, then count every 60-char substring at EVERY offset -- the
    newline-agnostic generalization of the old 60-char-line check. A loop
    with period p chars repeats each of its p distinct alignments about
    len/p times, so any degenerate loop lights this up regardless of
    whether it contains spaces or newlines. (Catches BOTH p1 and p2.)

The old 60-char-LINE metric is still computed and reported so column 3 of
collapse-summary.csv keeps its exact old meaning (awk parity is asserted
in tests/test_collapse_detector.py), but it no longer feeds the verdict:
as a decision signal it was both blind to single-line loops (above) and a
latent false-positive generator (>5 blank lines, or >5 identical short
lines like markdown ``` fences, in an otherwise-healthy essay trip it).

The 2000-char floor on the WHOLE FILE (bytes, wc -c parity) is retained
unchanged from the original heuristic.

Only the generated span is analyzed for repetition: text after the first
'<|assistant|>' marker (end of the engine's prompt echo), cut before the
engine stats footer (the last "\\n---\\n<N> tokens in " line, falling back
to the trailing METAL-ATTN block). Files missing either marker degrade
gracefully (whole file / no footer cut), so the partial output of a
killed run is still analyzable.

Zero dependencies: python3 standard library only, same posture as
docs/gen_performance_theory.py.

Usage:
    python3 collapse_detector.py FILE

    stdout: chars,line60_reps,word8_reps,shingle60_reps,verdict
            (exactly the collapse-summary.csv row minus the leading
            prompt number; verdict is OK or COLLAPSE)
    stderr: one metrics line, plus quoted most-repeated evidence when
            the verdict is COLLAPSE (lands in gates.log for the human)
    exit:   0 = OK, 3 = COLLAPSE (mirrors gate A's own exit 3),
            2 = usage/IO error. run_container_gates.sh treats ANY
            nonzero exit as collapse -- fail-closed.
"""
from __future__ import annotations

import re
import sys
from collections import Counter

CHAR_FLOOR = 2000   # unchanged from the original heuristic (whole file, bytes)
MAX_REPS = 5        # unchanged K: anything repeated >5 times = collapse
NGRAM_WORDS = 8     # words per n-gram window
SHINGLE_CHARS = 60  # chars per shingle window (same width as the old line check)

ASSISTANT_MARKER = "<|assistant|>"
FOOTER_RE = re.compile(r"\n---\n(?=\d+ tokens in )")


def extract_generated_span(text: str) -> str:
    """Model generation only: after the first '<|assistant|>' (where the
    prompt echo ends), before the engine stats footer. Both cuts are
    best-effort so partial output from a killed run still analyzes."""
    i = text.find(ASSISTANT_MARKER)
    span = text[i + len(ASSISTANT_MARKER):] if i >= 0 else text
    footer = None
    for footer in FOOTER_RE.finditer(span):
        pass  # keep the LAST match: generation itself may contain '---' lines
    if footer is not None:
        return span[: footer.start()]
    j = span.rfind("\nMETAL-ATTN:")  # fallback if the '---' stats line is absent
    return span[:j] if j >= 0 else span


def line60_reps(text: str) -> int:
    """The ORIGINAL newline-dependent metric, semantics identical to
    awk '{ s=substr($0,1,60); c[s]++ } END { print max }' over the whole
    file. Kept only so collapse-summary.csv column 3 stays comparable
    across gate runs; reported, never used for the verdict."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # a trailing newline does not create an awk record
    if not lines:
        return 0
    return max(Counter(line[:SHINGLE_CHARS] for line in lines).values())


def word_ngram_max_reps(span: str, n: int = NGRAM_WORDS) -> tuple[int, str]:
    """Max repetition count of any consecutive n-word window (whitespace
    tokenization, so newlines are irrelevant), plus the winning phrase."""
    words = span.split()
    if len(words) < n:
        return 0, ""
    counts = Counter(tuple(words[i: i + n]) for i in range(len(words) - n + 1))
    (gram, reps), = counts.most_common(1)
    return reps, " ".join(gram)


def shingle_max_reps(span: str, width: int = SHINGLE_CHARS) -> tuple[int, str]:
    """Max repetition count of any width-char substring of the
    whitespace-normalized span, at every offset, plus the winning
    shingle. This is the newline-agnostic generalization of the old
    repeated-60-char-line check."""
    normalized = " ".join(span.split())  # ALL whitespace runs -> single spaces
    if len(normalized) < width:
        return 0, ""
    counts = Counter(normalized[i: i + width] for i in range(len(normalized) - width + 1))
    (shingle, reps), = counts.most_common(1)
    return reps, shingle


def analyze(raw: bytes) -> dict:
    """All metrics plus the verdict for one engine output file (bytes)."""
    text = raw.decode("utf-8", errors="replace")
    span = extract_generated_span(text)
    word8_reps, word8_evidence = word_ngram_max_reps(span)
    shingle60_reps, shingle60_evidence = shingle_max_reps(span)
    chars = len(raw)  # bytes of the WHOLE file: wc -c parity with the old check
    collapse = (
        chars < CHAR_FLOOR
        or word8_reps > MAX_REPS
        or shingle60_reps > MAX_REPS
    )
    return {
        "chars": chars,
        "line60_reps": line60_reps(text),
        "word8_reps": word8_reps,
        "word8_evidence": word8_evidence,
        "shingle60_reps": shingle60_reps,
        "shingle60_evidence": shingle60_evidence,
        "collapse": collapse,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: collapse_detector.py FILE", file=sys.stderr)
        return 2
    try:
        with open(argv[1], "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        print(f"collapse_detector: cannot read {argv[1]}: {exc}", file=sys.stderr)
        return 2
    r = analyze(raw)
    verdict = "COLLAPSE" if r["collapse"] else "OK"
    print(f"{r['chars']},{r['line60_reps']},{r['word8_reps']},{r['shingle60_reps']},{verdict}")
    print(
        f"collapse_detector: chars={r['chars']} (floor {CHAR_FLOOR})"
        f" line60_reps={r['line60_reps']} word8_reps={r['word8_reps']}"
        f" shingle60_reps={r['shingle60_reps']} (K={MAX_REPS}) verdict={verdict}",
        file=sys.stderr,
    )
    if r["collapse"]:
        if r["shingle60_reps"] > MAX_REPS:
            print(
                f"collapse_detector: most-repeated 60-char shingle"
                f" (x{r['shingle60_reps']}): {r['shingle60_evidence']!r}",
                file=sys.stderr,
            )
        if r["word8_reps"] > MAX_REPS:
            print(
                f"collapse_detector: most-repeated 8-word n-gram"
                f" (x{r['word8_reps']}): {r['word8_evidence']!r}",
                file=sys.stderr,
            )
    return 3 if r["collapse"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
