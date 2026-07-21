"""tools/patch_m5max_staged_expert_read.py: structural checks plus a real
compile+run fixture validation proving the staged (gate+up / down
concurrent) miss read produces IDENTICAL output to today's single coalesced
read -- the explicit correctness bar from the 2026-07-15 build task ("staged
path produces identical output to single-read on the fixture"). Also
compiles and dry-runs tests/staged_expert_read_microbench.c end to end.

Same conventions as tests/test_patch_m5max_stall_trace.py: structural tests
run the real generator/patcher chain into a tempdir and assert on text
(no build needed); compile+run tests build a real, tiny, throwaway binary
and drive it against tests/tiny_serve_fixture.py's numpy-only fixture.
Skips cleanly (not a failure) when no C compiler is available.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
TOOLS = C_DIR / "tools"
PATCHER_PATH = TOOLS / "patch_m5max_staged_expert_read.py"
MICROBENCH_SRC = C_DIR / "tests" / "staged_expert_read_microbench.c"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


patcher = load_module("patch_m5max_staged_expert_read", PATCHER_PATH)

TIMING_RE = re.compile(
    r"\d+\.\d+\s*tok/s|\d+\.\d+s\b|\d+\.\d+\s*tok/fw|RSS \d+\.\d+ GB|STALL-EXPOSED[^\n]*|IO-BYTES[^\n]*|IO-LATENCY[^\n]*|WALL-SUM[^\n]*|\[IOKIND\][^\n]*")


def normalize_timing(text: str) -> str:
    """Mask wall-clock timing and RSS substrings that vary run to run for
    reasons unrelated to correctness -- staged mode's own background
    pthreads legitimately shift RSS by a small, expected amount (thread
    stacks), same category as scheduler-noise timing jitter. Everything
    else (generated tokens, hit rate, byte hashes) must still match
    exactly."""
    return TIMING_RE.sub("<T>", text)


class StagedReadPatcherStructureTest(unittest.TestCase):
    def test_patches_a_pristine_glm_c(self):
        original = (C_DIR / "glm.c").read_text()
        patched = patcher.patch_text(original)
        self.assertIn("expert_load_staged_begin", patched)
        self.assertIn("expert_load_staged_join", patched)
        self.assertIn("m5_staged_down_reader", patched)
        self.assertIn('ili_env("STAGED_EXPERT_READ")', patched)
        self.assertIn("static int g_staged_expert_read=0;", patched)
        # down's read is spawned BEFORE gate+up's synchronous read
        begin_body = patched[patched.index("static int expert_load_staged_begin("):
                             patched.index("static int expert_load_staged_join(")]
        self.assertLess(begin_body.index("pthread_create(&sl->th"),
                        begin_body.index("gu_contig"))
        # wired into the blocking (non-PIPE) miss-load loop only
        self.assertIn("if(g_staged_expert_read){", patched)
        self.assertIn("StagedLoad m5_sl;", patched)

    def test_patches_the_gen_m5max_engine_output(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "glm_m5max.c"
            subprocess.run([sys.executable, str(TOOLS / "gen_m5max_engine.py"),
                           str(C_DIR / "glm.c"), str(output)], check=True)
            original = output.read_text()
        patched = patcher.patch_text(original)
        self.assertIn("expert_load_staged_begin", patched)
        self.assertIn("expert_load_staged_join", patched)

    def test_double_application_is_rejected(self):
        original = (C_DIR / "glm.c").read_text()
        patched = patcher.patch_text(original)
        with self.assertRaises(RuntimeError):
            patcher.patch_text(patched)

    def test_unrecognized_source_raises_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            patcher.patch_text("int main(void){ return 0; }\n")


def find_compiler():
    for cc in (os.environ.get("CC"), "clang", "cc", "gcc"):
        if cc and shutil.which(cc):
            return cc
    return None


def compile_flags():
    flags = ["-O1", "-Wall", "-Wextra", "-Wno-unused-parameter",
             "-Wno-misleading-indentation", "-Wno-unused-function"]
    ldflags = ["-lm"]
    try:
        omp = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True,
                             text=True, timeout=5)
        if omp.returncode == 0:
            prefix = omp.stdout.strip()
            flags += ["-Xclang", "-fopenmp", f"-I{prefix}/include"]
            ldflags += [f"-L{prefix}/lib", "-lomp"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return flags, ldflags


def compile_source_text(source_text: str, binary_path: Path, extra_cflags=()):
    """Write source_text as a sibling of glm.c (so quoted #includes resolve)
    and compile it standalone to binary_path."""
    cc = find_compiler()
    if not cc:
        return None
    src = C_DIR / f".tmp_staged_read_test_{os.getpid()}.c"
    src.write_text(source_text)
    try:
        flags, ldflags = compile_flags()
        return subprocess.run(
            [cc, *flags, *extra_cflags, str(src), "-o", str(binary_path), *ldflags],
            capture_output=True, text=True, timeout=120)
    finally:
        src.unlink(missing_ok=True)


@unittest.skipUnless(find_compiler(), "no C compiler available")
class StagedReadFixtureCompileRunTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(C_DIR / "tests"))
        import tiny_serve_fixture
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = tiny_serve_fixture.build(Path(cls._tmp.name) / "glm_tiny")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def build(self, source_text: str, name: str, extra_cflags=()) -> Path:
        binary = C_DIR / f".tmp_staged_read_{name}_{os.getpid()}"
        result = compile_source_text(source_text, binary, extra_cflags)
        self.addCleanup(lambda: binary.unlink(missing_ok=True))
        self.assertEqual(result.returncode, 0,
                         f"compiling ({name}) failed:\n{result.stderr}")
        self.assertEqual(result.stderr.strip(), "",
                         f"compiler warnings on ({name}):\n{result.stderr}")
        return binary

    def run_engine(self, binary: Path, **env_extra):
        env = dict(os.environ, SNAP=str(self.model),
                   PROMPT="the quick brown fox jumps over a lazy staged read",
                   NGEN="15", OMP_NUM_THREADS="2", RAM_GB="1", TEMP="0")
        env.update({k: str(v) for k, v in env_extra.items()})
        return subprocess.run([str(binary), "8"], env=env, capture_output=True,
                              text=True, errors="replace", timeout=60)

    def test_staged_reads_produce_byte_identical_generation_output(self):
        """The design's explicit correctness bar: staged path == single-read
        path, byte for byte, on the fixture."""
        patched = patcher.patch_text((C_DIR / "glm.c").read_text())
        binary = self.build(patched, "plain")

        off = self.run_engine(binary)  # ILI_STAGED_EXPERT_READ unset
        on = self.run_engine(binary, ILI_STAGED_EXPERT_READ="1")
        self.assertEqual(off.returncode, 0, off.stderr)
        self.assertEqual(on.returncode, 0, on.stderr)
        self.assertEqual(normalize_timing(off.stdout), normalize_timing(on.stdout),
                         "staged-read output differs from single-read output "
                         "with the SAME prompt/seed -- must be byte-identical")

        off_hash = re.search(r"output token hash: (\w+)", off.stdout)
        on_hash = re.search(r"output token hash: (\w+)", on.stdout)
        self.assertIsNotNone(off_hash)
        self.assertIsNotNone(on_hash)
        self.assertEqual(off_hash.group(1), on_hash.group(1))

    def test_tracing_off_matches_the_unpatched_engine(self):
        """Opt-in only: zero output change when ILI_STAGED_EXPERT_READ is unset."""
        baseline = self.build((C_DIR / "glm.c").read_text(), "baseline")
        patched_binary = self.build(patcher.patch_text((C_DIR / "glm.c").read_text()), "patched_off")

        base_result = self.run_engine(baseline)
        patched_result = self.run_engine(patched_binary)
        self.assertEqual(base_result.returncode, 0, base_result.stderr)
        self.assertEqual(patched_result.returncode, 0, patched_result.stderr)
        self.assertEqual(normalize_timing(base_result.stdout), normalize_timing(patched_result.stdout))

    def test_staged_read_still_produces_correct_output_at_higher_concurrency(self):
        """Same correctness bar, but with OMP_NUM_THREADS raised so several
        experts load concurrently at once (closer to the microbench's own
        depth>1 cells) -- staged mode's background pthreads must not corrupt
        anything under real cross-expert concurrency."""
        patched = patcher.patch_text((C_DIR / "glm.c").read_text())
        binary = self.build(patched, "concurrent")
        off = self.run_engine(binary, OMP_NUM_THREADS="4")
        on = self.run_engine(binary, OMP_NUM_THREADS="4", ILI_STAGED_EXPERT_READ="1")
        self.assertEqual(off.returncode, 0, off.stderr)
        self.assertEqual(on.returncode, 0, on.stderr)
        self.assertEqual(normalize_timing(off.stdout), normalize_timing(on.stdout))

    def test_microbench_compiles_and_dry_runs_against_the_fixture(self):
        patched_text = patcher.patch_text((C_DIR / "glm.c").read_text())
        engine_src = C_DIR / f".tmp_staged_read_engine_{os.getpid()}.c"
        engine_src.write_text(patched_text)
        self.addCleanup(lambda: engine_src.unlink(missing_ok=True))

        binary = C_DIR / f".tmp_staged_read_microbench_{os.getpid()}"
        self.addCleanup(lambda: binary.unlink(missing_ok=True))
        cc = find_compiler()
        flags, ldflags = compile_flags()
        result = subprocess.run(
            [cc, *flags, f'-DGLM_STAGED_INCLUDE="{engine_src}"',
             str(MICROBENCH_SRC), "-o", str(binary), *ldflags],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, f"microbench compile failed:\n{result.stderr}")
        self.assertEqual(result.stderr.strip(), "", f"microbench compiler warnings:\n{result.stderr}")

        run = subprocess.run([str(binary), str(self.model)], capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("staged-read-microbench", run.stdout)
        self.assertIn("DISCLAIMER", run.stdout)
        depth_lines = [l for l in run.stdout.splitlines() if l.startswith("depth=")]
        self.assertGreaterEqual(len(depth_lines), 1)
        for line in depth_lines:
            self.assertIn("single-read median", line)
            self.assertIn("staged median", line)


if __name__ == "__main__":
    unittest.main()
