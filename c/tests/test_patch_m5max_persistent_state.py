import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
TOOLS = C_DIR / "tools"
PATCHER_PATH = TOOLS / "patch_m5max_persistent_state.py"


def load_patcher():
    spec = importlib.util.spec_from_file_location(
        "patch_m5max_persistent_state", PATCHER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PersistentStatePatcherTest(unittest.TestCase):
    def generated_m5max_backend(self) -> str:
        """Build the real metal4-patched fixture without touching the worktree."""
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "backend_metal_m5max.mm"
            for script, args in (
                ("gen_m5max_backend.py", [str(C_DIR / "backend_metal.mm"), str(output)]),
                ("patch_m5max_command_buffers.py", [str(output), str(output)]),
                ("patch_m5max_metal4.py", [str(output), str(output)]),
            ):
                subprocess.run(
                    [sys.executable, str(TOOLS / script), *args], check=True
                )
            return output.read_text()

    def test_lifecycle_transform_on_real_generator_output(self):
        patcher = load_patcher()
        original = self.generated_m5max_backend()
        patched = patcher.patch_text(original)

        # Requirement 1: per-registration generation ids on lifecycle nodes.
        self.assertIn("struct SlabLife {", patched)
        self.assertIn(
            "SlabLife *life; uint64_t gen;", patched
        )
        self.assertIn(
            "uint64_t gen=g_slab_gen.fetch_add(1,std::memory_order_release)+1;",
            patched,
        )

        # Requirement 2: refs acquired before commit, released on completion,
        # on both the legacy and Metal 4 submission paths.
        self.assertEqual(patched.count("slab_refs_acquire(use,nuse,&held_refs,&nheld)"), 2)
        self.assertEqual(patched.count("SlabRefGuard ref_guard{held_refs,nheld};"), 2)
        self.assertIn("addCompletedHandler:^(id<MTLCommandBuffer> done)", patched)
        self.assertIn("addFeedbackHandler:^(id<MTL4CommitFeedback> fb4)", patched)
        # Acquire happens before the command buffer is created/committed.
        legacy = patched[
            patched.index("static id<MTLCommandBuffer> moe_submit(") :
            patched.index('extern "C" int ili_metal_moe_block(')
        ]
        self.assertLess(
            legacy.index("slab_refs_acquire"), legacy.index("[cb commit]")
        )
        self.assertLess(
            legacy.index("addCompletedHandler"), legacy.index("[cb commit]")
        )

        # Requirement 3: deferred unregister with a bounded queue drained on
        # completions, blocking so the caller can free the backing safely.
        self.assertIn("g_retired_slab_lives", patched)
        self.assertIn("ILI_SLAB_RETIRE_MAX", patched)
        self.assertIn("slab_life_retire_locked(lk,life,true);", patched)
        self.assertIn("g_slab_cv.wait_for", patched)

        # Requirement 4: explicit pinned fast path.
        self.assertIn(
            'extern "C" void ili_metal_register_pinned(void *base, size_t len)',
            patched,
        )
        self.assertIn("if(hit->life->pinned) continue;", patched)

        # Requirement 5: bounded retirement of residency snapshots.
        self.assertIn("ILI_SNAP_RETIRE_MAX", patched)
        self.assertIn("metal4_snapshot_sweep_locked", patched)
        self.assertIn("snapshot_inflight4", patched)

        # Requirement 6: a nil command buffer must bail out while the RAII ref
        # guard is still armed (before addCompletedHandler could silently no-op),
        # and a failed lifecycle-node allocation refuses the registration rather
        # than degrading to an untracked (unsafe) slab.
        legacy_body = legacy
        self.assertIn("if(!cb || !e){ g_moe_fb++; return nil; }", legacy_body)
        self.assertLess(
            legacy_body.index("if(!cb || !e){ g_moe_fb++; return nil; }"),
            legacy_body.index("[cb addCompletedHandler"),
        )
        self.assertIn("slab lifecycle node alloc failed; slab not registered", patched)
        self.assertIn("ili_metal_lab_fail_next_command_buffer", patched)

        # Requirement 6/8: opt-in flag gates the runtime, defaults preserved.
        self.assertIn('ili_env("METAL_PERSISTENT_STATE")', patched)
        self.assertIn("if(!g_metal_persistent_state || nuse<=0) return 1;", patched)
        # Opt-in flag is readable outside the Metal 4 compile guard so legacy
        # Metal lab builds exercise the slab lifecycle too.
        flag_decl = patched.index("static int g_metal_persistent_state=0;")
        guard = patched.index("#if ILI_METAL4_COMPILED", patched.index(
            "static int g_metal4_requested=0;"))
        self.assertLess(flag_decl, guard)

        # Shutdown drains the retirement queue (bounded wait, no deadlock).
        self.assertIn("cannot deadlock at shutdown", patched)

        # The transform is not idempotent: a second application must fail loudly
        # instead of silently double-patching.
        with self.assertRaises(RuntimeError):
            patcher.patch_text(patched)


if __name__ == "__main__":
    unittest.main()
