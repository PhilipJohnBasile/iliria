"""Direct unit tests for tools/m5max_log_parse.py -- the log-parsing contract shared by
summarize_m5max_k6_matrix.py and fight_report.py. Kept separate from both consumers' own
tests so the shared contract has its own, consumer-independent coverage."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from m5max_log_parse import med, parse_log  # noqa: E402

GOOD_LOG = """
112 tokens in 70.00s (1.60 tok/s) | expert hit rate 73.0% | RSS 110.0 GB
output token hash: abcd1234ef567890
PROFILE: expert-disk 30.000s | expert-matmul 12.000s | attention 15.000s (including kvb 0.0s) | lm_head 1.0s | other 1.0s
PILOT-METRICS: predicted 100 | enqueued 80 | resident-skip 20 | race-skip 0 | queue-full 0
PILOT-OUTCOME: loads 70 | useful 55 | wasted 10 | late 5 | evictions 8 | precision 78.6%
PILOT-TIME: load 1.200s | layer-barrier 0.300s | blocked-pipe 0.400s (4 waits)
MATRIX-SWAP-BEFORE-MB: 10.5
MATRIX-SWAP-AFTER-MB: 12.0
"""

MINIMAL_LOG = """
112 tokens in 70.00s (1.60 tok/s) | expert hit rate 73.0% | RSS 110.0 GB
output token hash: abcd1234ef567890
PROFILE: expert-disk 30.000s | expert-matmul 12.000s | attention 15.000s (including kvb 0.0s) | lm_head 1.0s | other 1.0s
"""


class ParseLogTest(unittest.TestCase):
    def test_extracts_all_fields_including_optional_pilot_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(GOOD_LOG)
            run = parse_log(path, {"prompt": "p", "cache_state": "warm", "trial": 1, "mode": "stack"})
        self.assertEqual(run.tok_s, 1.60)
        self.assertEqual(run.hit_pct, 73.0)
        self.assertEqual(run.disk_s, 30.0)
        self.assertEqual(run.matmul_s, 12.0)
        self.assertEqual(run.attn_s, 15.0)
        self.assertEqual(run.output_hash, "abcd1234ef567890")
        self.assertEqual(run.useful, 55)
        self.assertEqual(run.wasted, 10)
        self.assertEqual(run.pilot_load_s, 1.2)
        self.assertEqual(run.swap_before_mb, 10.5)
        self.assertEqual(run.swap_after_mb, 12.0)
        self.assertEqual(run.prompt, "p")
        self.assertEqual(run.cache_state, "warm")
        self.assertEqual(run.trial, 1)
        self.assertEqual(run.mode, "stack")

    def test_optional_pilot_fields_default_to_zero_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(MINIMAL_LOG)
            run = parse_log(path, {"prompt": "p", "cache_state": "cold", "trial": 2, "mode": "baseline"})
        self.assertEqual(run.useful, 0)
        self.assertEqual(run.pilot_load_s, 0.0)
        self.assertIsNone(run.swap_before_mb)
        self.assertIsNone(run.swap_after_mb)

    def test_missing_hash_is_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text("112 tokens in 70.00s (1.60 tok/s) | expert hit rate 73.0% | RSS 1 GB\n")
            with self.assertRaises(ValueError):
                parse_log(path, {"prompt": "p", "cache_state": "warm", "trial": 1, "mode": "x"})

    def test_last_occurrence_wins_when_a_line_repeats(self) -> None:
        # run-m5max-fast.sh only prints these once per run, but parse_log takes the LAST
        # match on purpose (matches the original summarize_m5max_k6_matrix.py behavior) --
        # regression-guard that behavior through the extraction.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.log"
            path.write_text(GOOD_LOG + "\noutput token hash: 1111111111111111\n")
            run = parse_log(path, {"prompt": "p", "cache_state": "warm", "trial": 1, "mode": "x"})
        self.assertEqual(run.output_hash, "1111111111111111")


class MedTest(unittest.TestCase):
    def test_empty_is_nan(self) -> None:
        self.assertNotEqual(med([]), med([]))  # nan != nan

    def test_median_of_odd_and_even_length(self) -> None:
        self.assertEqual(med([1.0, 2.0, 3.0]), 2.0)
        self.assertEqual(med([1.0, 2.0, 3.0, 4.0]), 2.5)


if __name__ == "__main__":
    unittest.main()
