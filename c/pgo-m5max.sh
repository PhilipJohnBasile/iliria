#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || { echo "Usage: bash pgo-m5max.sh MODEL_DIR [PROMPT]" >&2; exit 2; }
MODEL_DIR="${1%/}"
PROMPT="${2:-Explain the hottest execution paths in a mixture-of-experts inference engine.}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo "error: M5 Max PGO requires Apple Silicon macOS" >&2
  exit 1
}
[[ -d "$MODEL_DIR" ]] || { echo "error: model directory not found: $MODEL_DIR" >&2; exit 1; }

LLVM_PROFDATA="$(xcrun --find llvm-profdata 2>/dev/null || true)"
[[ -x "$LLVM_PROFDATA" ]] || {
  echo "error: llvm-profdata is required; install the Xcode command-line tools" >&2
  exit 1
}

PGO_DIR="${ILI_PGO_DIR:-$SCRIPT_DIR/.pgo-m5max}"
PGO_NGEN="${ILI_PGO_NGEN:-128}"
rm -rf "$PGO_DIR"
mkdir -p "$PGO_DIR"

printf '%s\n' "[1/4] Building instrumented native M5 Max engine and Metal backend..."
make clean
make glm METAL=1 M5MAX_BACKEND=1 M5MAX_ENGINE=1 APPLE_CPU=native APPLE_LTO=0 \
  EXTRA_CFLAGS="-fprofile-instr-generate" \
  EXTRA_LDFLAGS="-fprofile-instr-generate"

printf '%s\n' "[2/4] Collecting a representative decode profile with the saved M5 Max runtime profile..."
LLVM_PROFILE_FILE="$PGO_DIR/iliria-%p.profraw" \
ILI_IGNORE_PROFILE="${ILI_PGO_IGNORE_PROFILE:-0}" ILI_NGEN="$PGO_NGEN" \
  bash ./run-m5max-fast.sh "$MODEL_DIR" run "$PROMPT"

shopt -s nullglob
profiles=("$PGO_DIR"/*.profraw)
(( ${#profiles[@]} > 0 )) || { echo "error: no PGO profile was produced" >&2; exit 1; }

printf '%s\n' "[3/4] Merging profile data..."
"$LLVM_PROFDATA" merge -output="$PGO_DIR/m5max.profdata" "${profiles[@]}"

printf '%s\n' "[4/4] Rebuilding generated engine/backend with ThinLTO and profile feedback..."
make clean
make glm METAL=1 M5MAX_BACKEND=1 M5MAX_ENGINE=1 APPLE_CPU=native APPLE_LTO=1 \
  EXTRA_CFLAGS="-fprofile-instr-use=$PGO_DIR/m5max.profdata -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date" \
  EXTRA_LDFLAGS="-fprofile-instr-use=$PGO_DIR/m5max.profdata"

cp -f ./glm ./glm-m5max-pgo

cat <<EOF
PGO build complete.

Active binary:   $SCRIPT_DIR/glm
Saved copy:      $SCRIPT_DIR/glm-m5max-pgo
Profile data:    $PGO_DIR/m5max.profdata
Engine:          generated allocation-free M5 Max variant
Backend:         generated M5 Max Metal variant

Use the normal launcher; it will run the newly optimized ./glm binary through ili.
EOF
