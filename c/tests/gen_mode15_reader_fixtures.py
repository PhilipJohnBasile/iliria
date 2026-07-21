"""Generates tiny SYNTHETIC mode-1.5 tensor-blob fixtures for
tests/test_mode15_reader.c, via tools/mode15_container.py's own encoder
(make_tensor_blob) -- reused unmodified, per that module's own format-
compatibility contract (see its docstring): "a future C/Metal reader that
rebuilds its own LUT from this module's stored 8-byte length table via the
SAME canonical-code algorithm will ... reconstruct byte-identical
codewords." This script is that future C reader's fixture supplier.

Two fixtures, written as plain files (tiny, synthetic, disposable --
nothing here is a model weight or lives under a models directory):

  happy_O37_I64_rpb8.bin  (+ .orig sidecar) -- O=37 (NOT a multiple of
    rows_per_block=8: exercises a partial final block, rows [32,37)), I=64,
    5 blocks. The C test's main happy-path fixture: parsed via
    m15_open_structural(), every row decoded via m15_get_row_span() + the
    REAL codec_row_huff.h functions, and compared byte-for-byte against
    `.orig` (O*I raw bytes, one nibble/byte, row-major -- same convention
    as tools/mode15_cross_check.c's "MHFX" fixture nibble section). Also
    the base buffer the C test mutates in memory for its truncation/
    corrupted-CRC/version-mismatch/row_offsets-hardening experiments
    (mirrors tests/test_mode15_container.py's own bytearray-mutation
    pattern: corrupt ONE clean fixture per test case in memory, rather
    than shipping a separate pre-corrupted file per failure mode).

  zero_row_I64.bin -- O=0, I=64: the zero-row edge case (n_blocks=0,
    payload_len=0, row_offsets is the single entry [0]).

Usage: python3 tests/gen_mode15_reader_fixtures.py [OUT_DIR]
  (OUT_DIR defaults to "mode15_reader_fixtures" in the current directory;
  build_mode15_sidecar.sh passes an explicit mktemp'd scratch dir.)
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(TESTS_DIR, "..", "tools")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m15 = _load_module("mode15_container", os.path.join(TOOLS_DIR, "mode15_container.py"))

# HAPPY_ROWS_PER_BLOCK must match ROWS_PER_BLOCK in tests/test_mode15_reader.c
HAPPY_O = 37
HAPPY_I = 64
HAPPY_ROWS_PER_BLOCK = 8
ZERO_ROW_I = 64
SEED = 20260718   # deterministic; arbitrary (the date this sidecar was authored)


def sample_nibbles(rng: np.random.Generator, O: int, I: int) -> np.ndarray:
    """A mildly-skewed 16-symbol distribution (not uniform, not
    degenerate) -- real enough to produce a non-trivial canonical-Huffman
    codebook (varied code lengths across multiple present symbols) without
    needing the full census machinery."""
    x = np.arange(16) - 7.5
    p = np.exp(-(x * x) / (2 * 3.0 * 3.0))
    p = p / p.sum()
    return rng.choice(16, size=(O, I), p=p).astype(np.uint8)


def write_fixture(out_dir: str, name: str, nibbles: np.ndarray, rows_per_block: int) -> None:
    blob = m15.make_tensor_blob(nibbles, rows_per_block=rows_per_block)
    blob_path = os.path.join(out_dir, f"{name}.bin")
    with open(blob_path, "wb") as f:
        f.write(blob)
    orig_path = os.path.join(out_dir, f"{name}.orig")
    with open(orig_path, "wb") as f:
        f.write(np.ascontiguousarray(nibbles, dtype=np.uint8).tobytes())
    print(f"wrote {blob_path} ({len(blob)} B) + {orig_path} ({nibbles.size} B) "
          f"O={nibbles.shape[0]} I={nibbles.shape[1]} rows_per_block={rows_per_block}")


def build(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    happy = sample_nibbles(rng, HAPPY_O, HAPPY_I)
    write_fixture(out_dir, "happy_O37_I64_rpb8", happy, HAPPY_ROWS_PER_BLOCK)

    zero_row = np.zeros((0, ZERO_ROW_I), dtype=np.uint8)
    write_fixture(out_dir, "zero_row_I64", zero_row, HAPPY_ROWS_PER_BLOCK)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "mode15_reader_fixtures"
    build(out)
