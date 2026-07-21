#!/usr/bin/env bash
# Synthetic stand-in for run-m5max-fast.sh, used ONLY by `fight_card.sh --dry-run`
# (ILI_FIGHT_RUNNER_BIN defaults to this file under --dry-run; real mode always
# uses run-m5max-fast.sh). Mirrors run-m5max-fast.sh's own CLI (MODEL_DIR run
# PROMPT) and prints stdout in the EXACT format tools/m5max_log_parse.py expects
# (output token hash / PROFILE / tok-s / optional PILOT-* lines), so the SAME
# downstream parser and report logic that runs against real engine output in
# production is exercised end to end by --dry-run -- never a separate,
# test-only parsing path. Same principle as scripts/abba_transcript_driver.py's
# mock-serve subcommand.
#
# Hash model: deterministic sha256 of (container basename, prompt text, and
# ONLY the kernel-family-relevant subset of env: ILI_METAL_PREFILL,
# ILI_METAL4_MOE, ILI_METAL_PERSISTENT_STATE, ILI_DSA). Levers that do not
# change the arithmetic path (ILI_PILOT*, ILI_DRAFT, ILI_CPU_GROUPED_MOE,
# ...) are deliberately EXCLUDED from the hash input -- matching this project's
# own kernel-family-rounding policy (README.md: "kernel-family rounding can
# flip greedy tokens"; docs/PERFORMANCE_THEORY.json's
# s-row-projection-gemms-prefill-gate-1 / dsa-sparse-indexer entries). Repeated
# trials of the SAME config, and configs that only touch prefetch/scheduling,
# hash-match; configs that flip a GPU-vs-CPU kernel family are free to fork --
# on purpose, so the dry run proves fight_report.py RECORDS (never fails on)
# exactly that case.
set -euo pipefail

MODEL_DIR="${1:?usage: fight_mock_engine.sh MODEL_DIR run PROMPT}"
MODE="${2:?usage: fight_mock_engine.sh MODEL_DIR run PROMPT}"
PROMPT="${3:-}"
[[ "$MODE" == run ]] || { echo "fight_mock_engine.sh only supports the 'run' mode (got: $MODE)" >&2; exit 2; }

NGEN="${ILI_NGEN:-112}"

python3 - "$MODEL_DIR" "$PROMPT" "$NGEN" <<'PY'
import hashlib
import os
import random
import sys

model_dir, prompt, ngen = sys.argv[1], sys.argv[2], int(sys.argv[3])

kernel_family_bits = "|".join(
    f"{k}={os.environ.get(k, '0')}"
    for k in ("ILI_METAL_PREFILL", "ILI_METAL4_MOE", "ILI_METAL_PERSISTENT_STATE", "ILI_DSA")
)
basis = f"{os.path.basename(model_dir.rstrip('/'))}|{prompt}|{kernel_family_bits}"
output_hash = hashlib.sha256(basis.encode()).hexdigest()[:16]

# tok/s model: deterministic per (config, prompt), with a small "lever bonus" so the
# dry-run report exercises real promote / no-harm / interaction-table code paths (not just
# the log parser). Trial-to-trial jitter is seeded by FIGHT_MOCK_TRIAL_SALT (set by
# fight_card.sh per trial) so runs stay fully deterministic -- no real randomness -- while
# still varying trial to trial like a genuine noisy measurement.
salt = os.environ.get("FIGHT_MOCK_TRIAL_SALT", "0")
rng = random.Random(f"{basis}|{salt}")

base_rate = 1.40
bonus = 0.0
bonus += 0.05 if os.environ.get("ILI_METAL_PREFILL", "0") == "1" else 0.0
bonus += 0.04 if os.environ.get("ILI_METAL4_MOE", "0") == "1" else 0.0
bonus += 0.01 if os.environ.get("ILI_METAL_PERSISTENT_STATE", "0") == "1" else 0.0
# DSA is regime-harmful at this harness's short-prompt/decode regime (per
# docs/performance-theory.json dsa-sparse-indexer: "-10% decode below 2048 ctx") --
# modeled as a penalty so a card that (wrongly) included it fails its own no-harm bar
# visibly in the dry-run proof instead of silently looking fine.
bonus -= 0.15 if os.environ.get("ILI_DSA", "0") == "1" else 0.0
jitter = rng.uniform(-0.015, 0.015)
tok_s = max(0.05, base_rate + bonus + jitter)

hit_pct = 65.0 + (8.0 if os.environ.get("ILI_METAL4_MOE", "0") == "1" else 0.0)
disk_s = max(0.1, 35.0 - 10.0 * bonus)
matmul_s = 12.0
attn_s = max(0.1, 20.0 - (5.0 if os.environ.get("ILI_METAL_PREFILL", "0") == "1" else 0.0))
wall_s = ngen / tok_s

print("[mock-engine] fight_mock_engine.sh standing in for run-m5max-fast.sh (--dry-run only)")
print(f"{ngen} tokens in {wall_s:.2f}s ({tok_s:.2f} tok/s) | expert hit rate {hit_pct:.1f}% | RSS 1.0 GB")
print(f"output token hash: {output_hash}")
print(
    f"PROFILE: expert-disk {disk_s:.3f}s | expert-matmul {matmul_s:.3f}s | "
    f"attention {attn_s:.3f}s (including kvb 0.0s) | lm_head 0.100s | other 0.100s"
)
if os.environ.get("ILI_PILOT", "0") == "1":
    print("PILOT-METRICS: predicted 100 | enqueued 80 | resident-skip 20 | race-skip 0 | queue-full 0")
    print("PILOT-OUTCOME: loads 70 | useful 55 | wasted 10 | late 5 | evictions 8 | precision 78.6%")
    print("PILOT-TIME: load 1.200s | layer-barrier 0.300s | blocked-pipe 0.400s (4 waits)")
PY
