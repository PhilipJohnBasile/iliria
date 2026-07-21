#!/usr/bin/env bash
set -euo pipefail

# Native Apple Silicon build for iliria's Metal backend.
# Builds the CPU fallback for this Mac (-mcpu=native) while enabling the
# Objective-C++ Metal backend from PR #72.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
  echo "error: this build profile requires Apple Silicon macOS" >&2
  exit 1
fi

command -v clang >/dev/null || { echo "error: clang is required" >&2; exit 1; }
command -v clang++ >/dev/null || { echo "error: clang++ is required" >&2; exit 1; }

omp_cflags=()
omp_ldflags=()
if command -v brew >/dev/null 2>&1 && omp_prefix="$(brew --prefix libomp 2>/dev/null)"; then
  omp_cflags=(-Xclang -fopenmp -I"$omp_prefix/include")
  omp_ldflags=(-L"$omp_prefix/lib" -lomp)
else
  echo "warning: Homebrew libomp not found; CPU fallback will be single-threaded" >&2
  echo "         install it with: brew install libomp" >&2
fi

rm -f backend_metal.o glm

clang++ -x objective-c++ -fobjc-arc -O3 -mcpu=native \
  -c backend_metal.mm -o backend_metal.o

clang -O3 -mcpu=native -DILI_METAL \
  "${omp_cflags[@]}" \
  -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function \
  glm.c backend_metal.o -o glm \
  -lm "${omp_ldflags[@]}" \
  -framework Metal -framework Foundation -lc++

./glm --help >/dev/null 2>&1 || true

echo "Built: $SCRIPT_DIR/glm"
echo "CPU fallback: -mcpu=native"
echo "GPU backend: Metal enabled"
echo "Run tests with: make metal-test"