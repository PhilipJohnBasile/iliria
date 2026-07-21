# iliria — root convenience entry point.
# The engine and all real build/test targets live in c/ (see c/Makefile).
# On an M5 Max, prefer `make mac-fast` (measured defaults); `make` builds the
# portable engine.
.PHONY: all mac-fast check clean

all:
	$(MAKE) -C c

mac-fast:
	$(MAKE) -C c mac-fast

check:
	$(MAKE) -C c check

clean:
	$(MAKE) -C c clean
