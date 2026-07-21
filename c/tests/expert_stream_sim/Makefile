# Standalone build for the expert-stream/cache simulator + its test suite.
#
# Deliberately NOT wired into the root c/Makefile: this whole directory has
# zero #include edges into c/*.h (see sim_nvme.h's file header), and another
# agent is concurrently editing the real loader/matmul path (glm.c and
# friends) -- keeping this build self-contained means it can never be broken
# by, or accused of touching, that work. `make -C c/tests/expert_stream_sim
# test` builds and runs this suite on its own; folding it into the root
# Makefile's TEST_BINS/test-c target is a trivial follow-up once that other
# work has landed.
#
# CC/CFLAGS mirror the root c/Makefile's per-OS conventions (same compiler
# choice, same warning flags); OpenMP/Metal/CUDA are irrelevant here (no
# threads, no GPU -- this is a single-threaded, CPU-only, discrete-event
# simulator) so those knobs are simply omitted rather than copied unused.

UNAME_S := $(shell uname -s)
EXTRA_CFLAGS  ?=
EXTRA_LDFLAGS ?=

ifeq ($(UNAME_S),Darwin)
CC     = clang
APPLE_CPU ?= native
CFLAGS = -O2 -g -mcpu=$(APPLE_CPU) -std=c11 $(EXTRA_CFLAGS) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
else
CC     = gcc
CFLAGS = -O2 -g -std=c11 $(EXTRA_CFLAGS) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
endif
LDFLAGS = -lm $(EXTRA_LDFLAGS)
EXE     =

HDRS = sim_nvme.h sim_cache.h sim_expert_io.h sim_router.h

.PHONY: all test cross-check clean

all: test

test_expert_stream_sim$(EXE): test_expert_stream_sim.c $(HDRS)
	$(CC) $(CFLAGS) test_expert_stream_sim.c -o test_expert_stream_sim$(EXE) $(LDFLAGS)

test: test_expert_stream_sim$(EXE)
	./test_expert_stream_sim$(EXE)

# Optional, separate fidelity check: cross-validates sim_cache.h's ported
# REPIN swap policy against the REAL c/tier.h (read-only #include -- this is
# the one file in this directory that reaches outside it, purely to prove
# the port isn't subtly wrong; it is not part of `make test` and never
# writes to anything under c/). Requires c/tier.h to exist two directories
# up, i.e. this file's normal location (c/tests/expert_stream_sim/).
test_tier_cross_check$(EXE): test_tier_cross_check.c sim_cache.h ../../tier.h
	$(CC) $(CFLAGS) test_tier_cross_check.c -o test_tier_cross_check$(EXE) $(LDFLAGS)

cross-check: test_tier_cross_check$(EXE)
	./test_tier_cross_check$(EXE)

clean:
	rm -f test_expert_stream_sim$(EXE) test_tier_cross_check$(EXE)
	rm -rf test_expert_stream_sim.dSYM test_tier_cross_check.dSYM
