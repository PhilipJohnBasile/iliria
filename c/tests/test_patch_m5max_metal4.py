import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
TOOLS = C_DIR / "tools"
PATCHER_PATH = TOOLS / "patch_m5max_metal4.py"


def load_patcher():
    spec = importlib.util.spec_from_file_location("patch_m5max_metal4", PATCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Metal4PatcherTest(unittest.TestCase):
    def generated_m5max_backend(self) -> str:
        """Build the real generated-text fixture without touching the worktree."""
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "backend_metal_m5max.mm"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "gen_m5max_backend.py"),
                    str(C_DIR / "backend_metal.mm"),
                    str(output),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "patch_m5max_command_buffers.py"),
                    str(output),
                    str(output),
                ],
                check=True,
            )
            return output.read_text()

    def test_real_generator_output_gets_guarded_metal4_path(self):
        patcher = load_patcher()
        original = self.generated_m5max_backend()
        patched = patcher.patch_text(original)

        self.assertEqual(patched.count("#define ILI_METAL4_COMPILED 1"), 1)
        self.assertIn("defined(ILI_METAL4) && ILI_METAL4", patched)
        self.assertEqual(patched.count("static int metal4_moe_submit("), 1)
        self.assertEqual(patched.count("static int metal4_moe_finish("), 1)
        self.assertIn("ili_env(\"METAL4_MOE\")", patched)
        self.assertIn("newMTL4CommandQueue", patched)
        self.assertIn("newCommandAllocator", patched)
        self.assertIn("newArgumentTableWithDescriptor", patched)
        self.assertIn("barrierAfterEncoderStages:MTLStageDispatch", patched)
        self.assertIn("useResidencySet:slot->residency4", patched)
        self.assertIn("addFeedbackHandler:slot->feedback_handler4", patched)
        self.assertIn(
            "__unsafe_unretained id<MTLAllocation> allocations[395]", patched
        )
        self.assertIn("if(submit_result4<0)", patched)
        self.assertIn("clear4(&g_moe_sync); clear4(&g_moe_async);", patched)

        helper_start = patched.index("static int metal4_moe_submit(")
        helper_end = patched.index("#endif", helper_start)
        helper = patched[helper_start:helper_end]
        self.assertNotIn("g_metal4_moe=0;", helper)
        gate = helper.index(
            "gemv(oag,osg,slot->xg,slot->gg,oc_gate,R*Iinter);"
        )
        up = helper.index(
            "gemv(oau,osu,slot->xg,slot->uu,oc_up,R*Iinter);"
        )
        silu = helper.index("[e setComputePipelineState:g_moe_silu];")
        down = helper.index(
            "gemv(oad,osd,slot->gg,slot->hh,oc_down,R*D);"
        )
        barriers = [
            index
            for index in range(len(helper))
            if helper.startswith(
                "[e barrierAfterEncoderStages:MTLStageDispatch", index
            )
        ]
        self.assertEqual(len(barriers), 2)
        self.assertLess(gate, up)
        self.assertLess(up, barriers[0])
        self.assertLess(barriers[0], silu)
        self.assertLess(silu, barriers[1])
        self.assertLess(barriers[1], down)

        # The legacy implementation remains present and is still the fallback
        # for both the synchronous and asynchronous entry points.
        self.assertEqual(
            patched.count("static id<MTLCommandBuffer> moe_submit("), 1
        )
        self.assertEqual(patched.count("static int moe_finish("), 1)
        self.assertIn(
            "id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt",
            patched,
        )
        self.assertIn(
            "cb=moe_submit(nb,D,Iinter,fmt",
            patched,
        )
        legacy_start = original.index("static id<MTLCommandBuffer> moe_submit(")
        legacy_end = original.index(
            'extern "C" int ili_metal_moe_block(', legacy_start
        )
        self.assertIn(original[legacy_start:legacy_end], patched)
        self.assertEqual(
            patched.count("memoryBarrierWithScope:MTLBarrierScopeBuffers"),
            original.count("memoryBarrierWithScope:MTLBarrierScopeBuffers"),
        )
        scatter = (
            "for(int gr=0;gr<R;gr++){ float *os=out+(size_t)rows[gr]*D, "
            "w=rw[gr]; const float *hr=hh+(size_t)gr*D;"
        )
        self.assertEqual(patched.count(scatter), original.count(scatter) + 1)

        # Strict one-match replacements make stale or double application fail
        # loudly instead of silently emitting a half-patched backend.
        with self.assertRaises(RuntimeError):
            patcher.patch_text(patched)


if __name__ == "__main__":
    unittest.main()
