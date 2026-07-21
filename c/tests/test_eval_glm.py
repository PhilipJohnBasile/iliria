"""tools/eval_glm.py --dump-per-item: unit tests.

No engine, no network, no /path/to/models reads. Exercises score_accuracy()
directly (in-process) for the JSONL contents, and the CLI's --selftest path
(subprocess) for the byte-identical-aggregate-output guarantee and the
--dump-per-item wiring end to end.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "eval_glm.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eg = load_module("eval_glm", SCRIPT)


# Same synthetic fixture as eval_glm.py's own `--selftest` branch: 1 question,
# 3 options, gold=1. lp favors option 1 for raw acc; length-normalizing by
# option_lengths [4, 2, 8] favors option 2 for acc_norm instead.
SELFTEST_META = [("t", 0, 0, 1, 4, 1), ("t", 0, 1, 1, 2, 1), ("t", 0, 2, 1, 8, 1)]
SELFTEST_PERQ = {("t", 0): [0, 1, 2]}
SELFTEST_LP = [-3.0, -2.0, -5.0]


class ScoreAccuracyDumpPerItemTests(unittest.TestCase):
    def test_dump_matches_hand_computed_record(self):
        # by hand: best = argmax(lp) = index 1 (lp=-2.0) -> option oi=1 -> not gold(1==1) -> True
        #          bestn = argmax(lp/len) = argmax(-3/4=-0.75, -2/2=-1.0, -5/8=-0.625) = index 2
        #                  -> option oi=2 -> gold=1 -> correct_accnorm False
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / "per_item.jsonl"
            eg.score_accuracy(["t"], SELFTEST_META, SELFTEST_PERQ, SELFTEST_LP,
                              dump_per_item=str(dump_path))
            lines = dump_path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec, {
            "task": "t", "qid": 0, "gold": 1,
            "chosen_acc": 1, "chosen_accnorm": 2,
            "correct_acc": True, "correct_accnorm": False,
            "lp_per_option": [-3.0, -2.0, -5.0],
            "option_lengths": [4, 2, 8],
        })

    def test_dump_option_ordering_survives_shuffled_meta_indices(self):
        # perq lists row indices out of option-index order; the dump must
        # still emit lp_per_option / option_lengths sorted by option index.
        meta = [("t", 0, 2, 1, 8, 0), ("t", 0, 0, 1, 4, 0), ("t", 0, 1, 1, 2, 0)]
        perq = {("t", 0): [0, 1, 2]}   # row0=oi2, row1=oi0, row2=oi1
        lp = [-5.0, -1.0, -2.0]
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / "x.jsonl"
            eg.score_accuracy(["t"], meta, perq, lp, dump_per_item=str(dump_path))
            rec = json.loads(dump_path.read_text().strip())
        self.assertEqual(rec["lp_per_option"], [-1.0, -2.0, -5.0])       # oi 0,1,2 in order
        self.assertEqual(rec["option_lengths"], [4, 2, 8])
        self.assertEqual(rec["chosen_acc"], 0)                            # best lp is row1 -> oi 0 -> gold
        self.assertTrue(rec["correct_acc"])

    def test_one_line_per_question_multi_task(self):
        meta = [("A", 0, 0, 1, 1, 0), ("A", 0, 1, 1, 1, 0),
                ("A", 1, 0, 1, 1, 1), ("A", 1, 1, 1, 1, 1),
                ("B", 0, 0, 1, 1, 0), ("B", 0, 1, 1, 1, 0)]
        perq = {("A", 0): [0, 1], ("A", 1): [2, 3], ("B", 0): [4, 5]}
        lp = [-1.0, -2.0, -2.0, -1.0, -1.0, -2.0]
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / "multi.jsonl"
            eg.score_accuracy(["A", "B"], meta, perq, lp, dump_per_item=str(dump_path))
            lines = [json.loads(l) for l in dump_path.read_text().strip().splitlines()]
        self.assertEqual(len(lines), 3)
        self.assertEqual([(r["task"], r["qid"]) for r in lines],
                         [("A", 0), ("A", 1), ("B", 0)])

    def test_no_dump_file_written_when_flag_absent(self):
        with tempfile.TemporaryDirectory() as td:
            maybe_path = Path(td) / "should_not_exist.jsonl"
            eg.score_accuracy(["t"], SELFTEST_META, SELFTEST_PERQ, SELFTEST_LP)
            self.assertFalse(maybe_path.exists())


class AggregateOutputByteIdenticalTests(unittest.TestCase):
    """The task requirement: turning --dump-per-item on must not change a
    single byte of the aggregate stdout output."""

    def _run(self, extra_args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--snap", "/nonexistent", "--selftest", *extra_args],
            capture_output=True, text=True, check=False, timeout=30,
        )

    def test_stdout_identical_with_and_without_flag(self):
        without = self._run([])
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / "dump.jsonl"
            with_flag = self._run(["--dump-per-item", str(dump_path)])
            self.assertTrue(dump_path.exists())
        self.assertEqual(without.returncode, 0, without.stderr)
        self.assertEqual(with_flag.returncode, 0, with_flag.stderr)
        self.assertEqual(without.stdout, with_flag.stdout)

    def test_cli_selftest_dump_writes_expected_record(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / "dump.jsonl"
            result = self._run(["--dump-per-item", str(dump_path)])
            self.assertEqual(result.returncode, 0, result.stderr)
            rec = json.loads(dump_path.read_text().strip())
        self.assertEqual(rec["task"], "t")
        self.assertEqual(rec["gold"], 1)
        self.assertIn("per-item dump: 1 questions", result.stderr)


class IliBenchWiringTests(unittest.TestCase):
    """The `ili bench` subcommand must forward --dump-per-item to
    eval_glm.py unchanged. Loaded as a module (ili has no .py suffix) so
    subprocess.call can be monkeypatched instead of actually invoking the
    engine/tokenizer."""

    def setUp(self):
        import importlib.machinery
        ili_path = ROOT / "ili"
        loader = importlib.machinery.SourceFileLoader("ili_cli_under_test", str(ili_path))
        spec = importlib.util.spec_from_file_location(
            "ili_cli_under_test", ili_path, loader=loader)
        self.ili = importlib.util.module_from_spec(spec)
        sys.modules["ili_cli_under_test"] = self.ili
        loader.exec_module(self.ili)

    def test_bench_forwards_dump_per_item_flag(self):
        calls = []

        def fake_call(cmd, env=None):
            calls.append(cmd)
            return 0

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model"
            model_dir.mkdir()
            (model_dir / "tokenizer.json").write_text("{}")
            fake_glm = Path(td) / "glm"
            fake_glm.write_text("")
            data_dir = Path(td) / "bench"
            data_dir.mkdir()
            (data_dir / "smoke.jsonl").write_text("")   # so cmd_bench sees no missing datasets
            dump_path = Path(td) / "per_item.jsonl"

            self.ili.GLM = str(fake_glm)   # bypass "engine is not built"
            self.ili.subprocess.call = fake_call
            argv = ["ili", "bench", "smoke", "--model", str(model_dir),
                    "--data", str(data_dir), "--dump-per-item", str(dump_path)]
            old_argv = sys.argv
            sys.argv = argv
            try:
                with self.assertRaises(SystemExit) as cm:
                    self.ili.main()
                self.assertEqual(cm.exception.code, 0)
            finally:
                sys.argv = old_argv

        self.assertEqual(len(calls), 1)
        cmd = calls[0]
        self.assertIn("--dump-per-item", cmd)
        self.assertEqual(cmd[cmd.index("--dump-per-item") + 1], str(dump_path))

    def test_bench_help_documents_flag(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "ili"), "bench", "--help"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dump-per-item", result.stdout)


if __name__ == "__main__":
    unittest.main()
