# Derived from colibri (https://github.com/JustVugg/colibri), Apache-2.0. Modified 2026 by Philip John Basile. See NOTICE.
UNAME_S := $(shell uname -s)
MINGW    := $(findstring MINGW,$(UNAME_S))
MSYS     := $(findstring MSYS,$(UNAME_S))
IS_WIN   := $(MINGW)$(MSYS)

EXTRA_CFLAGS  ?=
EXTRA_LDFLAGS ?=

ifeq ($(UNAME_S),Darwin)
# --- macOS / Apple Silicon ---
CC      = clang
APPLE_CPU ?= native
APPLE_TUNE = -mcpu=$(APPLE_CPU)
APPLE_LTO ?= 1
ifeq ($(APPLE_LTO),1)
APPLE_OPT   = -flto=thin -fno-math-errno
APPLE_LDOPT = -flto=thin -Wl,-dead_strip
else
APPLE_OPT   = -fno-math-errno
APPLE_LDOPT =
endif
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp not found: building single-threaded. For multithreading: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(APPLE_TUNE) $(APPLE_OPT) $(OMPC) $(EXTRA_CFLAGS) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm $(OMPL) $(APPLE_LDOPT) $(EXTRA_LDFLAGS)
EXE     =
else ifneq ($(IS_WIN),)
# --- Windows 11 x86-64 (MinGW-w64 / MSYS2) ---
CC      = gcc
ARCH   ?= x86-64-v3
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp $(EXTRA_CFLAGS) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -static $(EXTRA_LDFLAGS)
EXE     = .exe
else
UNAME_M := $(shell uname -m)
ifneq (,$(filter ppc64le ppc64,$(UNAME_M)))
# --- Linux PowerPC ---
CC      = gcc
ARCH   ?= native
CFLAGS  = -O3 -mcpu=$(ARCH) -fopenmp $(EXTRA_CFLAGS) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp $(EXTRA_LDFLAGS)
EXE     =
else
# --- Linux x86-64 ---
CC      = gcc
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp $(EXTRA_CFLAGS) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp $(EXTRA_LDFLAGS)
EXE     =
endif
endif

CUDA      ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
PYTHON    ?= python3
CUDA_OBJ  =
TEST_BINS = tests/test_json$(EXE) tests/test_st$(EXE) tests/test_st_header_guard$(EXE) tests/test_tier$(EXE) tests/test_grammar$(EXE) tests/test_idot$(EXE) \
            tests/test_codec_row_huff$(EXE) tests/test_codec_row_rans$(EXE) tests/test_codec_row_fse$(EXE) tests/test_codec_row_bitplane$(EXE) \
            tests/test_codec_row_gemv$(EXE) tests/test_dsa_gate_regression$(EXE) tests/test_mode15_engine_guard$(EXE) \
            tests/test_mode15_decode$(EXE) tests/test_legacy_fmt_guard$(EXE) tests/test_mode15_slab_reuse$(EXE)
ifeq ($(CUDA),1)
ifeq ($(UNAME_S),Darwin)
$(error CUDA=1 is supported only on Linux)
endif
ifneq ($(IS_WIN),)
$(error CUDA=1 on Windows requires G1: build ili_cuda.dll with MSVC+nvcc (see the Windows-port plan §8))
endif
CFLAGS   += -DILI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
endif

METAL          ?= 0
M5MAX_BACKEND  ?= 0
METAL4         ?= 0
METAL_OBJ       =
METAL_SRC       = backend_metal.mm
METAL4_FLAGS    =
ifeq ($(METAL4),1)
ifneq ($(UNAME_S),Darwin)
$(error METAL4=1 is supported only on macOS)
endif
METAL4_FLAGS    = -DILI_METAL4=1
endif
METALXX         = clang++ -x objective-c++ -fobjc-arc -O3 $(APPLE_TUNE) $(APPLE_OPT) $(METAL4_FLAGS) $(EXTRA_CFLAGS)
ifeq ($(M5MAX_BACKEND),1)
METAL_SRC       = backend_metal_m5max.mm
endif
ifeq ($(METAL),1)
ifneq ($(UNAME_S),Darwin)
$(error METAL=1 is supported only on macOS)
endif
CFLAGS   += -DILI_METAL
LDFLAGS  += -framework Metal -framework Foundation -lc++
METAL_OBJ = backend_metal.o
endif

M5MAX_ENGINE ?= 0
ENGINE_SRC    = glm.c
ifeq ($(M5MAX_ENGINE),1)
ENGINE_SRC    = glm_m5max.c
endif

# Mode-1.5 (Huffman-compressed expert-tensor container) CPU decode support --
# OPT-IN, mirrors CUDA/METAL above (see the Mode-15 integration design notes
# step 4). MODE15=0 (default): the shipped engine keeps the fail-loud guard as
# its ONLY behavior on an "MH01" tensor, byte-for-byte unchanged from before this
# lever existed. MODE15=1: glm.c decodes mode-1.5 tensors transparently via
# mode15_reader.c/codec_row_huff.h (pulled in as a single translation unit, no
# new link step -- see glm.c's own ILI_MODE15 include block). Independent of
# this flag, tests/test_mode15_decode's own build rule below always compiles
# with -DILI_MODE15 so `make test-c` continuously proves the decode path
# regardless of what the shipped-binary flag is set to.
MODE15 ?= 0
ifeq ($(MODE15),1)
CFLAGS += -DILI_MODE15
endif

all: glm$(EXE)

glm: glm$(EXE)

glm$(EXE): $(ENGINE_SRC) st.h json.h tok.h tok_unicode.h compat.h grammar.h $(CUDA_OBJ) $(METAL_OBJ)
	$(CC) $(CFLAGS) $(ENGINE_SRC) $(CUDA_OBJ) $(METAL_OBJ) -o glm$(EXE) $(LDFLAGS)

glm_m5max.c: glm.c tools/gen_m5max_engine.py tools/patch_m5max_grouped_cpu_moe.py tools/fix_m5max_grouped_build.py
	$(PYTHON) tools/gen_m5max_engine.py glm.c glm_m5max.c

# Staged intra-expert read variant (n2 falsifier hardware-prep). Kept under a
# distinct name so it never clobbers the production glm_m5max.c -- same
# convention as backend_metal_m5max_lab.mm below. Not composed with the other
# lab instrumentation (route trace / PILOT metrics / stall trace): each is
# independently applied straight after gen_m5max_engine.py, see
# tools/patch_m5max_staged_expert_read.py's module docstring.
glm_m5max_staged.c: glm.c tools/gen_m5max_engine.py tools/patch_m5max_grouped_cpu_moe.py tools/fix_m5max_grouped_build.py tools/patch_m5max_staged_expert_read.py
	$(PYTHON) tools/gen_m5max_engine.py glm.c glm_m5max_staged.c
	$(PYTHON) tools/patch_m5max_staged_expert_read.py glm_m5max_staged.c glm_m5max_staged.c

# Forced-miss microbench: staged (gate+up / down concurrent) vs single-read,
# swept over concurrency depth 1/2/4. Dry-run on the tiny numpy fixture:
#   python3 tools/make_glm_bench_model.py /tmp/glm_tiny_bench   (or
#   tests/tiny_serve_fixture.py's build() for a torch-free fixture)
#   ./tests/staged_expert_read_microbench /tmp/glm_tiny_bench
staged-read-microbench: glm_m5max_staged.c tests/staged_expert_read_microbench.c
	$(CC) $(CFLAGS) tests/staged_expert_read_microbench.c -o tests/staged_expert_read_microbench$(EXE) $(LDFLAGS)

backend_cuda.o: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -c backend_cuda.cu -o $@

backend_metal_m5max.mm: backend_metal.mm tools/gen_m5max_backend.py tools/patch_m5max_moe_pool.py tools/patch_m5max_command_buffers.py tools/patch_m5max_metal4.py
	$(PYTHON) tools/gen_m5max_backend.py backend_metal.mm backend_metal_m5max.mm
	$(PYTHON) tools/patch_m5max_command_buffers.py backend_metal_m5max.mm backend_metal_m5max.mm
	$(PYTHON) tools/patch_m5max_metal4.py backend_metal_m5max.mm backend_metal_m5max.mm

backend_metal.o: $(METAL_SRC) backend_metal.h
	$(METALXX) -c $(METAL_SRC) -o $@

metal-test: tests/test_backend_metal.mm backend_metal.mm backend_metal.h
	$(METALXX) tests/test_backend_metal.mm backend_metal.mm -framework Metal -framework Foundation -o backend_metal_test
	./backend_metal_test

metal-test-m5max: backend_metal_m5max.mm tests/test_backend_metal.mm backend_metal.h
	$(METALXX) tests/test_backend_metal.mm backend_metal_m5max.mm -framework Metal -framework Foundation -o backend_metal_test
	./backend_metal_test

metal4-test-m5max:
	ILI_METAL4_MOE=1 $(MAKE) metal-test-m5max M5MAX_BACKEND=1 METAL4=1

# Lab backend (persistent-state transform applied) for the lifecycle stress test.
# Kept under a distinct name so it never clobbers the production backend_metal_m5max.mm.
backend_metal_m5max_lab.mm: backend_metal.mm tools/gen_m5max_backend.py tools/patch_m5max_moe_pool.py tools/patch_m5max_cpp_allocations.py tools/patch_m5max_command_buffers.py tools/patch_m5max_metal4.py tools/patch_m5max_persistent_state.py
	$(PYTHON) tools/gen_m5max_backend.py backend_metal.mm $@
	$(PYTHON) tools/patch_m5max_command_buffers.py $@ $@
	$(PYTHON) tools/patch_m5max_metal4.py $@ $@
	$(PYTHON) tools/patch_m5max_persistent_state.py $@ $@

# Slab-lifecycle stress test: register/unregister + realloc churn against in-flight
# async MoE command buffers.  Runs with the lifecycle enabled and disabled, under
# MallocScribble so a use-after-free of a no-copy slab corrupts results loudly.
metal-test-lifecycle: backend_metal_m5max_lab.mm tests/test_backend_metal_lifecycle.mm backend_metal.h
	$(METALXX) tests/test_backend_metal_lifecycle.mm backend_metal_m5max_lab.mm -framework Metal -framework Foundation -o backend_metal_lifecycle_test
	MallocScribble=1 ILI_METAL_PERSISTENT_STATE=1 ./backend_metal_lifecycle_test
	MallocScribble=1 ./backend_metal_lifecycle_test

# Same stress test under AddressSanitizer (no LTO/native tuning: keep ASan happy).
metal-test-lifecycle-asan: backend_metal_m5max_lab.mm tests/test_backend_metal_lifecycle.mm backend_metal.h
	clang++ -x objective-c++ -std=c++17 -fobjc-arc -O1 -g -fsanitize=address -fno-omit-frame-pointer tests/test_backend_metal_lifecycle.mm backend_metal_m5max_lab.mm -framework Metal -framework Foundation -o backend_metal_lifecycle_asan
	ILI_METAL_PERSISTENT_STATE=1 ./backend_metal_lifecycle_asan
	./backend_metal_lifecycle_asan

online-attn-test: tests/test_online_attn.mm backend_metal.mm backend_metal.h
	$(METALXX) tests/test_online_attn.mm backend_metal.mm -framework Metal -framework Foundation -o tests/test_online_attn
	./tests/test_online_attn

score-stage-prefalsifier: tests/test_score_stage_prefalsifier.mm
	$(METALXX) tests/test_score_stage_prefalsifier.mm -framework Metal -framework Foundation -o tests/test_score_stage_prefalsifier
	./tests/test_score_stage_prefalsifier

metal4-probe: tests/test_metal4_probe.mm
	$(METALXX) tests/test_metal4_probe.mm -framework Metal -framework Foundation -o tests/test_metal4_probe
	./tests/test_metal4_probe

# Standalone GPU probe for the mode-3/mode-1.5 codec-race follow-up (see
# the codec-race results and
# the Metal row-decode results). Self-contained: does
# NOT link backend_metal.mm, only depends on the row-codec headers for its
# own synthetic-data generation and CPU-reference correctness checks.
metal-row-decode-test: tests/test_metal_row_decode.mm codec_row.h codec_row_huff.h codec_row_bitplane.h
	$(METALXX) tests/test_metal_row_decode.mm -framework Metal -framework Foundation -o tests/test_metal_row_decode
	./tests/test_metal_row_decode

cuda-test: backend_cuda.cu backend_cuda.h tests/test_backend_cuda.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE)
	./backend_cuda_test$(EXE)

olmoe$(EXE): olmoe.c st.h json.h compat.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe$(EXE) $(LDFLAGS)

portable:
	$(MAKE) glm$(EXE) ARCH=x86-64-v3

mac-fast:
	$(MAKE) clean
	$(MAKE) glm METAL=1 M5MAX_BACKEND=1 M5MAX_ENGINE=1 APPLE_CPU=native APPLE_LTO=1

mac-fast-metal4:
	$(MAKE) clean
	$(MAKE) glm METAL=1 M5MAX_BACKEND=1 M5MAX_ENGINE=1 METAL4=1 APPLE_CPU=native APPLE_LTO=1

mac-upstream:
	$(MAKE) clean
	$(MAKE) glm METAL=1 M5MAX_BACKEND=0 M5MAX_ENGINE=0 APPLE_CPU=native APPLE_LTO=1

mac-upstream-m5engine:
	$(MAKE) clean
	$(MAKE) glm METAL=1 M5MAX_BACKEND=0 M5MAX_ENGINE=1 APPLE_CPU=native APPLE_LTO=1

mac-portable:
	$(MAKE) clean
	$(MAKE) glm METAL=1 M5MAX_BACKEND=0 M5MAX_ENGINE=0 APPLE_CPU=generic APPLE_LTO=0

mac-run:
	@test -n "$(MODEL)" || { echo "Usage: make mac-run MODEL=/path/to/model" >&2; exit 2; }
	bash ./run-m5max-fast.sh "$(MODEL)" chat

mac-bench:
	@test -n "$(MODEL)" || { echo "Usage: make mac-bench MODEL=/path/to/model" >&2; exit 2; }
	bash ./benchmark-m5max.sh "$(MODEL)" "$(PROMPT)"

mac-autotune:
	@test -n "$(MODEL)" || { echo "Usage: make mac-autotune MODEL=/path/to/model" >&2; exit 2; }
	bash ./autotune-m5max.sh "$(MODEL)" "$(PROMPT)"

mac-pgo:
	@test -n "$(MODEL)" || { echo "Usage: make mac-pgo MODEL=/path/to/model" >&2; exit 2; }
	bash ./pgo-m5max.sh "$(MODEL)" "$(PROMPT)"








mac-train-hotset:
	@test -n "$(MODEL)" || { echo "Usage: make mac-train-hotset MODEL=/path/to/model" >&2; exit 2; }
	bash ./train-m5max-hotset.sh "$(MODEL)" "$(PASSES)"

iobench$(EXE): iobench.c compat.h
	$(CC) $(CFLAGS) iobench.c -o iobench$(EXE) $(LDFLAGS)

# A/B harness: DIRECT/F_NOCACHE vs buffered + single- vs dual-stream aggregate
# NVMe bandwidth, at the engine's real expert-read size. Standalone dev tool,
# same convention as iobench$(EXE) above: not wired into all/check/test, so it
# never affects CI (see iobench_ab.c's header comment for the NVMe-quiet guard
# and the depth-prefetch trace verdict for what it measures).
iobench_ab$(EXE): iobench_ab.c compat.h
	$(CC) $(CFLAGS) iobench_ab.c -o iobench_ab$(EXE) $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st$(EXE): tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st_header_guard$(EXE): tests/test_st_header_guard.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier$(EXE): tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_grammar$(EXE): tests/test_grammar.c grammar.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_idot$(EXE): tests/test_idot.c glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_dsa_gate_regression$(EXE): tests/test_dsa_gate_regression.c glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_mode15_engine_guard$(EXE): tests/test_mode15_engine_guard.c glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Always built with decode support compiled in (regardless of the top-level MODE15
# variable, which only gates the SHIPPED glm binary) -- this is the round-trip
# correctness proof for the Mode-15 integration design notes step 4's CPU decode
# path, and must run as part of `make test-c` unconditionally.
tests/test_mode15_decode$(EXE): tests/test_mode15_decode.c glm.c mode15_reader.h mode15_reader.c codec_row.h codec_row_huff.h st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) -DILI_MODE15 $< -o $@ $(LDFLAGS)

# Regression coverage for the mode15_e2e_certify.sh-discovered "silent death on a
# later turn" robustness gap: repeated expert_load_mode15() calls (including the
# real ws[]<->ecache[layer] struct-swap promotion idiom) reusing the SAME
# long-lived ESlot across many cycles, the exact shape a persistent `ili chat`/
# `ili serve` process exercises and a single `ili run` never does. Also always
# built with decode support compiled in, same reasoning as test_mode15_decode above.
tests/test_mode15_slab_reuse$(EXE): tests/test_mode15_slab_reuse.c glm.c mode15_reader.h mode15_reader.c codec_row.h codec_row_huff.h st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) -DILI_MODE15 $< -o $@ $(LDFLAGS)

# Legacy (non-mode-1.5) int8/int4/int2 format-determination fix: reject-unknown-size
# + legacy-still-loads coverage, including the g_mmap zero-copy fast path.
tests/test_legacy_fmt_guard$(EXE): tests/test_legacy_fmt_guard.c glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_codec_row_huff$(EXE): tests/test_codec_row_huff.c codec_row.h codec_row_huff.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_codec_row_rans$(EXE): tests/test_codec_row_rans.c codec_row.h codec_row_rans.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_codec_row_fse$(EXE): tests/test_codec_row_fse.c codec_row.h codec_row_fse.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_codec_row_bitplane$(EXE): tests/test_codec_row_bitplane.c codec_row.h codec_row_bitplane.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

codec_race$(EXE): codec_race.c codec_row.h codec_row_huff.h codec_row_rans.h codec_row_fse.h codec_row_bitplane.h
	$(CC) $(CFLAGS) codec_race.c -o codec_race$(EXE) $(LDFLAGS)

tests/test_codec_row_gemv$(EXE): tests/test_codec_row_gemv.c codec_row.h codec_row_rans.h codec_row_gemv.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

m5-cpu-moe-test: glm_m5max.c tests/test_m5_grouped_cpu_moe.c
	$(CC) $(CFLAGS) tests/test_m5_grouped_cpu_moe.c -o tests/test_m5_grouped_cpu_moe$(EXE) $(LDFLAGS)
	./tests/test_m5_grouped_cpu_moe$(EXE)

test-c: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test || exit 1; done

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' $(PYFLAGS)

test: test-c test-python

check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

clean:
	rm -f olmoe$(EXE) glm$(EXE) glm-m5max-pgo iobench$(EXE) iobench_ab$(EXE) codec_race$(EXE) backend_cuda.o backend_cuda_test$(EXE) backend_metal.o backend_metal_test $(TEST_BINS)
	rm -f tests/test_m5_grouped_cpu_moe$(EXE)
	rm -f backend_metal_m5max.mm glm_m5max.c glm_m5max_staged.c
	rm -f backend_metal_m5max_lab.mm backend_metal_lifecycle_test backend_metal_lifecycle_asan
	rm -f tests/test_online_attn tests/test_score_stage_prefalsifier
	rm -f tests/staged_expert_read_microbench$(EXE)
	rm -f tests/test_metal_row_decode
	rm -rf tests/__pycache__

.PHONY: all glm cuda-test metal-test metal-test-m5max metal4-test-m5max metal4-probe online-attn-test score-stage-prefalsifier metal-row-decode-test portable mac-fast mac-fast-metal4 mac-upstream mac-upstream-m5engine mac-portable mac-run mac-bench mac-autotune mac-pgo mac-train-hotset m5-cpu-moe-test staged-read-microbench test-c test-python test check clean
