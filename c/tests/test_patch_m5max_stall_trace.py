"""tools/patch_m5max_stall_trace.py: structural checks against both moe()
shapes the m5max chain can produce, plus a real compile+run fixture
validation (the "trace lines parse" bar from the 2026-07-15 build task).

Structural tests mirror the established convention (tests/test_gen_m5max_engine.py,
tests/test_patch_m5max_persistent_state.py): run the REAL generator/patcher
chain into a tempdir and assert on the text -- no engine build needed for
those. The compile+run tests go one step further (this patch's own
docstring promises fixture-validated behavior, not just text shape): build a
real, tiny, throwaway binary from the patched source and drive it against
the numpy-only fixture (tests/tiny_serve_fixture.py), the same fixture
tests/test_serve_large_prompt.py already uses for the real `glm` serve loop.
Skips cleanly (not a failure) when no C compiler is available; a genuine
compile error in the generated/patched source FAILS loudly, same
convention as tests/test_quant_noise_floor.py's numpy-import philosophy.
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
PATCHER_PATH = TOOLS / "patch_m5max_stall_trace.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


patcher = load_module("patch_m5max_stall_trace", PATCHER_PATH)
pst = load_module("parse_stall_trace", TOOLS / "parse_stall_trace.py")


def generated_m5max_chain(tmpdir: Path) -> str:
    """Run the REAL lab chain (gen_m5max_engine -> route_trace -> pilot_metrics)
    into tmpdir, mirroring build-m5max-lab.sh's own order, WITHOUT this
    patch. Returns the resulting text (grouped-CPU-MoE form, since
    gen_m5max_engine.py's own main() unconditionally chains
    patch_m5max_grouped_cpu_moe.py + fix_m5max_grouped_build.py)."""
    output = tmpdir / "glm_m5max.c"
    for script, args in (
        ("gen_m5max_engine.py", [str(C_DIR / "glm.c"), str(output)]),
        ("patch_m5max_route_trace.py", [str(output), str(output)]),
        ("patch_m5max_pilot_metrics.py", [str(output), str(output)]),
    ):
        subprocess.run([sys.executable, str(TOOLS / script), *args], check=True)
    return output.read_text()


class StallTracePatcherStructureTest(unittest.TestCase):
    def test_patches_a_pristine_glm_c_with_full_per_expert_granularity(self):
        original = (C_DIR / "glm.c").read_text()
        patched = patcher.patch_text(original)

        self.assertIn("static int g_stall_trace=0, g_stall_trace_init=0;", patched)
        self.assertIn('ili_env("STALL_TRACE")', patched)
        self.assertIn("m5_stall_trace_emit", patched)
        self.assertEqual(patched.count("STALL_TRACE fwd="), 1)  # one format string

        # route-ready capture landed right after the union loop
        self.assertIn("double m5_route_ready = m5_trace_on ? now_s() : 0;", patched)
        # per-block scratch declared
        self.assertIn("double m5_miss_issue_ts=0, m5_miss_complete_ts[64]={0};", patched)
        # per-expert hit/miss differentiation (plain form only)
        self.assertIn("if(m5_trace_on && qof[j]<0){ if(!m5_resident_start_ts)", patched)
        self.assertIn("if(m5_trace_on && !m5_reduction_start_ts) m5_reduction_start_ts=now_s();",
                     patched)
        # per-miss completion captured in BOTH the blocking loop and pipe_wait
        self.assertIn("if(m5_trace_on) m5_miss_complete_ts[q]=now_s();", patched)
        self.assertIn("if(m5_trace_on) m5_miss_complete_ts[qof[j]]=tc;", patched)
        # emit call is gated on m5_any_cpu, the handled[]-mask-based stand-in
        # for the removed block-level `metal_done` (Metal-resolved blocks,
        # i.e. every expert handled, still emit nothing)
        self.assertIn("if(m5_trace_on && m5_any_cpu) m5_stall_trace_emit(", patched)

    def test_patches_the_full_m5max_lab_chain_grouped_cpu_moe_form(self):
        with tempfile.TemporaryDirectory() as td:
            original = generated_m5max_chain(Path(td))
        patched = patcher.patch_text(original)

        self.assertIn("m5_stall_trace_emit", patched)
        # m5_cpu_moe_subset takes the mixed-format guard's handled[] mask
        # directly (as m5_mask) -- no more per-category take_res/take_miss
        self.assertIn("m5_moe_rows=m5_cpu_moe_subset(use,uniq+base,nb,m5_mask,\n"
                     "                              x,S,D,I,K,idxs,ws,keff,out);", patched)
        # resident_start/finish now bracket the WHOLE grouped compute call
        self.assertIn("if(m5_trace_on) m5_resident_start_ts=t0;", patched)
        self.assertIn("if(m5_trace_on) m5_resident_finish_ts=now_s();", patched)
        # miss completion captured in the hoisted upfront drain loop
        self.assertIn("if(m5_trace_on) m5_miss_complete_ts[qof[j]]=tc;", patched)
        # emit call is gated on m5_moe_rows>0 (the call's own routed-row
        # return), the grouped-form stand-in for the removed `metal_done`
        self.assertIn("if(m5_trace_on && m5_moe_rows>0) m5_stall_trace_emit(", patched)
        # reduction_start is NEVER set in this form -- stays at its 0 init,
        # which the emit function reports as -1 (no per-expert hook exists here)
        self.assertNotIn("m5_reduction_start_ts=now_s()", patched)

    def test_double_application_is_rejected_not_silently_double_patched(self):
        original = (C_DIR / "glm.c").read_text()
        patched = patcher.patch_text(original)
        with self.assertRaises(RuntimeError):
            patcher.patch_text(patched)

    def test_ambiguous_or_unrecognized_source_raises_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            patcher.patch_text("int main(void){ return 0; }\n")


TIMING_RE = re.compile(
    r"\d+\.\d+\s*tok/s|\d+\.\d+s\b|\d+\.\d+\s*tok/fw|RSS \d+\.\d+ GB|STALL-EXPOSED[^\n]*|IO-BYTES[^\n]*|IO-LATENCY[^\n]*|WALL-SUM[^\n]*|\[IOKIND\][^\n]*")


def normalize_timing(text: str) -> str:
    """Mask wall-clock timing and RSS substrings ("in 0.00s", "(3046.92
    tok/s)", PROFILE's per-phase "0.001s", "RSS 0.00 GB") that vary run to
    run for reasons unrelated to correctness (scheduler noise, cache
    warmth, allocator/page-rounding jitter) -- everything else (generated
    tokens, hit rate, byte hashes) must still match exactly for a genuine
    no-op comparison."""
    return TIMING_RE.sub("<T>", text)


def find_compiler():
    for cc in (os.environ.get("CC"), "clang", "cc", "gcc"):
        if cc and shutil.which(cc):
            return cc
    return None


def compile_flags():
    """Mirrors c/Makefile's Darwin CFLAGS (minus LTO/native tuning, to keep
    test builds fast); degrades to single-threaded if libomp is missing,
    same as the Makefile's own $(warning libomp not found...) fallback."""
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


def compile_engine(source_text: str, binary_path: Path):
    """Write source_text as a sibling of glm.c (so quoted #includes like
    "st.h" resolve) and compile it to binary_path. Returns the compiler's
    CompletedProcess; caller decides skip-vs-fail based on availability."""
    cc = find_compiler()
    if not cc:
        return None
    src = C_DIR / f".tmp_stall_trace_test_{os.getpid()}.c"
    src.write_text(source_text)
    try:
        flags, ldflags = compile_flags()
        return subprocess.run(
            [cc, *flags, str(src), "-o", str(binary_path), *ldflags],
            capture_output=True, text=True, timeout=120)
    finally:
        src.unlink(missing_ok=True)


@unittest.skipUnless(find_compiler(), "no C compiler available")
class StallTraceFixtureCompileRunTest(unittest.TestCase):
    """Real compile + run against the numpy-only tiny fixture: the "trace
    lines parse" bar. One shared fixture model across the class (built
    once); each test compiles its own throwaway binary and cleans up."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(C_DIR / "tests"))
        import tiny_serve_fixture
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = tiny_serve_fixture.build(Path(cls._tmp.name) / "glm_tiny")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # Pre-existing (not this patch's doing -- present even with NO stall-trace
    # patch applied, see the module docstring's chain investigation) unused-
    # variable warnings from patch_m5max_grouped_cpu_moe.py + fix_m5max_grouped_build.py's
    # own interaction: the compact xg/gg/uu/rows/rw scratch fix_m5max_grouped_build.py
    # restores at moe()'s top is no longer read once the grouped call stops
    # taking them as arguments. Allowed ONLY for chain-derived sources; a
    # pristine glm.c patch must still compile with zero warnings.
    KNOWN_GROUPED_CHAIN_WARNINGS = {"xg", "gg", "uu", "rows", "rw"}

    def build(self, source_text: str, name: str, allow_known_chain_warnings: bool = False) -> Path:
        binary = C_DIR / f".tmp_stall_trace_{name}_{os.getpid()}"
        result = compile_engine(source_text, binary)
        self.addCleanup(lambda: binary.unlink(missing_ok=True))
        self.assertEqual(
            result.returncode, 0,
            f"compiling patched engine ({name}) failed:\n{result.stderr}")
        if allow_known_chain_warnings:
            unexpected = [
                line for line in result.stderr.splitlines()
                if "warning:" in line and not any(
                    f"unused variable '{v}'" in line for v in self.KNOWN_GROUPED_CHAIN_WARNINGS)
            ]
            self.assertEqual(unexpected, [],
                             f"unexpected compiler warnings on ({name}):\n{unexpected}")
        else:
            self.assertEqual(result.stderr.strip(), "",
                             f"compiler warnings on patched engine ({name}):\n{result.stderr}")
        return binary

    def run_engine(self, binary: Path, **env_extra):
        env = dict(os.environ, SNAP=str(self.model),
                   PROMPT="the quick brown fox jumps over a lazy stall trace",
                   NGEN="10", OMP_NUM_THREADS="2", RAM_GB="1",
                   TEMP="0")  # greedy: reproducible tokens across separate runs
        env.update({k: str(v) for k, v in env_extra.items()})
        # errors="replace": the tiny fixture's random weights make the model
        # babble (tests/tiny_serve_fixture.py's own docstring says so) --
        # byte-level BPE output can include invalid UTF-8, which is fine here
        # since these tests only inspect stderr's STALL_TRACE lines (pure
        # ASCII) and compare stdout for byte-level equality, never decode it.
        return subprocess.run([str(binary), "8"], env=env, capture_output=True,
                              text=True, errors="replace", timeout=60)

    def test_plain_glm_c_patch_emits_parseable_stall_trace_lines(self):
        patched = patcher.patch_text((C_DIR / "glm.c").read_text())
        binary = self.build(patched, "plain")
        result = self.run_engine(binary, ILI_STALL_TRACE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [l for l in result.stderr.splitlines() if l.startswith("STALL_TRACE")]
        self.assertGreater(len(lines), 0, "no STALL_TRACE lines in stderr:\n" + result.stderr)
        for line in lines:
            record = pst.parse_line(line)
            self.assertIsNotNone(record)
            self.assertIn(record["layer"], (3, 4))  # the tiny fixture's 2 MoE layers
            self.assertGreaterEqual(record["exposed_stall_ms"], 0.0)

        rows = pst.build_rows([pst.parse_line(l) for l in lines])
        self.assertTrue(rows)
        for row in rows:
            self.assertIn(row["fast_path_eligible"], ("true", "false"))

    def test_pipe_mode_also_produces_parseable_lines_with_real_miss_completions(self):
        patched = patcher.patch_text((C_DIR / "glm.c").read_text())
        binary = self.build(patched, "plain_pipe")
        result = self.run_engine(binary, ILI_STALL_TRACE="1", PIPE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [l for l in result.stderr.splitlines() if l.startswith("STALL_TRACE")]
        self.assertGreater(len(lines), 0)
        for line in lines:
            self.assertIsNotNone(pst.parse_line(line))

    def test_tracing_off_is_byte_identical_to_the_unpatched_engine(self):
        """The whole point of an opt-in instrument: zero output change when
        ILI_STALL_TRACE is unset."""
        baseline_binary = self.build((C_DIR / "glm.c").read_text(), "baseline")
        patched = patcher.patch_text((C_DIR / "glm.c").read_text())
        patched_binary = self.build(patched, "patched_off")

        baseline = self.run_engine(baseline_binary)
        traced_off = self.run_engine(patched_binary)  # ILI_STALL_TRACE unset
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(traced_off.returncode, 0, traced_off.stderr)
        self.assertNotIn("STALL_TRACE", traced_off.stderr)
        self.assertEqual(normalize_timing(baseline.stdout), normalize_timing(traced_off.stdout),
                         "output changed with ILI_STALL_TRACE unset -- the "
                         "instrumentation must be a true no-op when off")

    def test_full_m5max_chain_compiles_and_traces_with_documented_grouped_scope(self):
        """'m5max chain reproduces': gen_m5max_engine -> route_trace ->
        pilot_metrics -> stall_trace, compiled CPU-only (no Metal framework
        needed -- ILI_METAL is simply undefined, exactly like every other
        #ifdef ILI_METAL block in this codebase), run against the fixture.
        The grouped-CPU-MoE structure has no per-expert reduction hook (see
        the patch's module docstring), so reduction_start_ms must read -1
        for every line here -- asserted, not just hoped for."""
        with tempfile.TemporaryDirectory() as td:
            chain_text = generated_m5max_chain(Path(td))
        patched = patcher.patch_text(chain_text)
        binary = self.build(patched, "m5chain", allow_known_chain_warnings=True)
        result = self.run_engine(binary, ILI_STALL_TRACE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [l for l in result.stderr.splitlines() if l.startswith("STALL_TRACE")]
        self.assertGreater(len(lines), 0, result.stderr)
        for line in lines:
            record = pst.parse_line(line)
            self.assertIsNotNone(record)
            self.assertEqual(record["reduction_start_ms"], -1.0,
                             "grouped-CPU-MoE form has no per-expert reduction "
                             "hook; reduction_start must be honestly unavailable")

    def test_full_m5max_chain_tracing_off_matches_chain_without_the_patch(self):
        """Isolates THIS patch's effect on the chain: with tracing off, output
        must match the same chain built WITHOUT the stall-trace patch at all
        (not compared against plain glm.c, since grouped-CPU-MoE may
        legitimately reorder floating-point accumulation)."""
        with tempfile.TemporaryDirectory() as td:
            chain_text = generated_m5max_chain(Path(td))
        chain_binary = self.build(chain_text, "chain_unpatched", allow_known_chain_warnings=True)
        patched_binary = self.build(patcher.patch_text(chain_text), "chain_patched_off",
                                    allow_known_chain_warnings=True)

        unpatched = self.run_engine(chain_binary)
        patched_off = self.run_engine(patched_binary)
        self.assertEqual(unpatched.returncode, 0, unpatched.stderr)
        self.assertEqual(patched_off.returncode, 0, patched_off.stderr)
        self.assertEqual(normalize_timing(unpatched.stdout), normalize_timing(patched_off.stdout))


if __name__ == "__main__":
    unittest.main()
