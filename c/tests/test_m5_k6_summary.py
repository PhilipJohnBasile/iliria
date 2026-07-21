from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "tools" / "summarize_m5max_k6_matrix.py"


def log_text(rate: float, hit: float, disk: float, attn: float, output_hash: str, useful: int = 0) -> str:
    return f"""
112 tokens in 70.00s ({rate:.2f} tok/s) | expert hit rate {hit:.1f}% | RSS 110.0 GB
output token hash: {output_hash}
PROFILE: expert-disk {disk:.3f}s | expert-matmul 12.000s | attention {attn:.3f}s (including kvb 0.0s) | lm_head 1.0s | other 1.0s
PILOT-METRICS: predicted 100 | enqueued 80 | resident-skip 20 | race-skip 0 | queue-full 0
PILOT-OUTCOME: loads 70 | useful {useful} | wasted 5 | late 3 | evictions 8 | precision 50.0%
PILOT-TIME: load 10.000s | layer-barrier 0.500s | blocked-pipe 4.000s (10 waits)
MATRIX-SWAP-BEFORE-MB: 0
MATRIX-SWAP-AFTER-MB: 0
"""


class K6SummaryTests(unittest.TestCase):
    def test_summary_reports_paired_result_and_hash_safety(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs = root / "logs"
            logs.mkdir()
            runs = []
            for trial in (1, 2, 3):
                for mode, rate, hit, useful in (("off", 1.00, 70.0, 0), ("k6", 1.10, 80.0, 50)):
                    name = f"warm-p-t{trial}-{mode}.log"
                    (logs / name).write_text(log_text(rate, hit, 30 if mode == "off" else 24, 20, "abcd", useful))
                    runs.append({
                        "prompt": "p",
                        "cache_state": "warm",
                        "trial": trial,
                        "mode": mode,
                        "log": f"logs/{name}",
                    })
            (root / "manifest.json").write_text(json.dumps({"passes": 3, "ngen": 112, "runs": runs}))
            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True, capture_output=True, text=True)
            summary = (root / "SUMMARY.md").read_text()
            self.assertIn("+10.00%", summary)
            self.assertIn("PROMOTE", summary)
            self.assertIn("Hash failures:\n- none", summary)
            self.assertTrue((root / "runs.csv").exists())

    def test_summary_rejects_hash_divergence_for_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs = root / "logs"
            logs.mkdir()
            runs = []
            for mode, output_hash in (("off", "aaaa"), ("k6", "bbbb")):
                name = f"warm-p-t1-{mode}.log"
                (logs / name).write_text(log_text(1.0, 70.0, 30, 20, output_hash))
                runs.append({"prompt": "p", "cache_state": "warm", "trial": 1, "mode": mode, "log": f"logs/{name}"})
            (root / "manifest.json").write_text(json.dumps({"passes": 1, "ngen": 112, "runs": runs}))
            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True, capture_output=True, text=True)
            summary = (root / "SUMMARY.md").read_text()
            self.assertIn("FAIL", summary)
            self.assertIn("aaaa", summary)
            self.assertIn("bbbb", summary)


if __name__ == "__main__":
    unittest.main()
