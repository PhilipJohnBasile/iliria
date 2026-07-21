#!/usr/bin/env python3
"""Add optional unretained command buffers, then remove C++ block allocations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.source.read_text()

    original = '[g_queue commandBuffer]'
    count = text.count(original)
    if count < 3:
        raise RuntimeError(f"command buffers: expected at least three main-queue submissions, found {count}")
    text = text.replace(original, 'm5_command_buffer()')

    marker = 'static MTLResourceOptions g_res_opts = MTLResourceStorageModeShared;   // ILI_METAL_UNTRACKED=1 adds HazardTrackingModeUntracked\n'
    helper = marker + '''static int g_unretained_cb = 0;
static id<MTLCommandBuffer> m5_command_buffer(){
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  if(g_unretained_cb && [g_queue respondsToSelector:@selector(commandBufferWithUnretainedReferences)])
    return [g_queue commandBufferWithUnretainedReferences];
#pragma clang diagnostic pop
  return [g_queue commandBuffer];
}
'''
    text = replace_once(text, marker, helper, 'command-buffer helper')

    text = replace_once(
        text,
        'extern "C" int ili_metal_init(void) {\n  if (g_dev) return 1;\n',
        '''extern "C" int ili_metal_init(void) {
  if (g_dev) return 1;
  g_unretained_cb = ili_env("METAL_UNRETAINED") && atoi(ili_env("METAL_UNRETAINED"));
''',
        'command-buffer environment control',
    )

    args.output.write_text(text)
    allocation_patcher = Path(__file__).with_name('patch_m5max_cpp_allocations.py')
    subprocess.run(
        [sys.executable, str(allocation_patcher), str(args.output), str(args.output)],
        check=True,
    )


if __name__ == '__main__':
    main()
