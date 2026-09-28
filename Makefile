# CIPHER C++ Test Suite Makefile
# Works on: Mac M4 (Apple Silicon) + Linux x86_64
# ================================================
#
# Quick start — Mac M4:
#   brew install openssl
#   make
#   make test
#
# Quick start — Linux:
#   sudo apt-get install libssl-dev
#   make
#   make test
#
# Without OpenSSL (mock HMAC):
#   make MOCK_HMAC=1
#
# NOTE ON TEST SIZES (current = container-safe):
#   For full Nebius benchmark, edit .cpp files:
#     RING_SIZE = 65536  (cipher_test_01_ring.cpp)
#     N = 100000         (cipher_test_03_shadow.cpp)
#     PARAM_COUNT = 50000
# ================================================

CXX      := g++
CXXFLAGS := -O3 -std=c++17 -march=native -pthread -Wall -Wextra

# ── Platform-specific flags ────────────────────────────────────────────────
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  # Mac — detect OpenSSL from Homebrew
  BREW_PREFIX := $(shell brew --prefix openssl 2>/dev/null)
  ifdef BREW_PREFIX
    CXXFLAGS += -I$(BREW_PREFIX)/include
    LDFLAGS  := -L$(BREW_PREFIX)/lib -lssl -lcrypto
    SSL_MSG  := Homebrew OpenSSL at $(BREW_PREFIX)
  else
    # Try system OpenSSL
    CXXFLAGS += -I/usr/local/include
    LDFLAGS  := -L/usr/local/lib -lssl -lcrypto
    SSL_MSG  := system OpenSSL
  endif
  # Mac-specific: link CoreFoundation for mach_timebase_info
  LDFLAGS += -framework CoreFoundation
  CXXFLAGS += -Wno-deprecated-declarations
else
  # Linux
  LDFLAGS  := -lssl -lcrypto
  SSL_MSG  := system OpenSSL
endif

ifdef MOCK_HMAC
  CXXFLAGS += -DMOCK_HMAC
  LDFLAGS  :=
  # On Mac still need CoreFoundation for timing
  ifeq ($(UNAME_S),Darwin)
    LDFLAGS := -framework CoreFoundation
  endif
  SSL_MSG := MOCK (rebuild without MOCK_HMAC=1 for real SHA)
endif

BINS := test_01_ring test_03_shadow

# ── Targets ────────────────────────────────────────────────────────────────

.PHONY: all test test_ring test_shadow clean info

all: info $(BINS)

test_01_ring: cipher_test_01_ring.cpp cipher_test_core.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "  Built: $@"

test_03_shadow: cipher_test_03_shadow.cpp cipher_test_core.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "  Built: $@"

test: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║   CIPHER C++ TEST SUITE                                  ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@./test_01_ring
	@echo ""
	@./test_03_shadow
	@echo ""
	@echo "══════════════════════════════════════════════════════════"
	@echo "Done. Bring these numbers to the Nebius deployment brief."
	@echo "══════════════════════════════════════════════════════════"

test_ring: test_01_ring
	@./test_01_ring

test_shadow: test_03_shadow
	@./test_03_shadow

bench: all
	@echo "=== CIPHER Quick Benchmark ==="
	@./test_01_ring 2>&1  | grep -E "p50|p99|PASS|FAIL|Throughput"
	@./test_03_shadow 2>&1 | grep -E "p50|p99|PASS|FAIL|Accuracy"

info:
	@echo "Building CIPHER C++ tests..."
	@echo "  Platform: $(UNAME_S) $(UNAME_M)"
	@echo "  Compiler: $(CXX)"
	@echo "  SSL:      $(SSL_MSG)"
	@echo ""

clean:
	rm -f $(BINS)
