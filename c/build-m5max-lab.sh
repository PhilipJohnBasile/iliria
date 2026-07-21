#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

METAL4_BUILD="${ILI_M5_LAB_METAL4:-0}"
case "$METAL4_BUILD" in 0|1) ;; *) echo "ILI_M5_LAB_METAL4 must be 0 or 1" >&2; exit 2 ;; esac

# Apple Clang's Objective-C++ default can still be gnu++98.  The backend has used
# C++ raw strings and modern Objective-C++ constructs from the beginning, so make
# the required language level explicit for reproducible hosted and local builds.
METALXX_CMD="${ILI_M5_LAB_METALXX:-clang++ -x objective-c++ -std=c++17 -fobjc-arc -O3 -mcpu=native -flto=thin -fno-math-errno}"

make clean

python3 tools/gen_m5max_engine.py glm.c glm_m5max.c
python3 tools/patch_m5max_route_trace.py glm_m5max.c glm_m5max.c
python3 tools/patch_m5max_pilot_metrics.py glm_m5max.c glm_m5max.c
python3 tools/patch_m5max_stall_trace.py glm_m5max.c glm_m5max.c

python3 tools/gen_m5max_backend.py backend_metal.mm backend_metal_m5max.mm
python3 tools/patch_m5max_command_buffers.py backend_metal_m5max.mm backend_metal_m5max.mm
python3 tools/patch_m5max_metal4.py backend_metal_m5max.mm backend_metal_m5max.mm
python3 tools/patch_m5max_persistent_state.py backend_metal_m5max.mm backend_metal_m5max.mm

# The generated files are newer than every generator dependency, so make compiles
# them without regenerating and discarding the lab transforms.
make glm \
  METAL=1 M5MAX_BACKEND=1 M5MAX_ENGINE=1 METAL4="$METAL4_BUILD" \
  APPLE_CPU=native APPLE_LTO=1 METALXX="$METALXX_CMD"

cat <<EOF
M5 Max lab build complete.
  route trace:       ILI_ROUTE_TRACE=/path/routes.bin
  PILOT metrics:     ILI_PILOT_METRICS=1
  stall trace:       ILI_STALL_TRACE=1 (a2 arrival-order-overlap falsifier;
                     parse with tools/parse_stall_trace.py)
  Metal 4 build:     $METAL4_BUILD
  Metal 4 runtime:   ILI_METAL4_MOE=1
  persistent state:  ILI_METAL_PERSISTENT_STATE=1

All experimental features are opt-in. A normal launch remains unchanged.
EOF
