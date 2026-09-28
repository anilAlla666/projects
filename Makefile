# =============================================================================
# CIPHER v2 Build System
# =============================================================================
# Two DSOs:
#   libcipher_hook.so  — LD_PRELOAD hook (pure C++17, no CUDA SDK)
#   libcipher_rt.so    — Runtime (C++17 + CUDA .cu files)
# =============================================================================

CXX       := g++
NVCC      := nvcc
CUDA_PATH ?= /usr/local/cuda

# Flags
CXXFLAGS  := -std=c++17 -O2 -fPIC -Wall -Wextra -Wno-unused-parameter
NVCCFLAGS := -std=c++17 -O2 -arch=sm_90 --compiler-options -fPIC,-Wall,-Wextra,-Wno-unused-parameter
INCLUDES  := -I./include -I$(CUDA_PATH)/include

BUILDDIR  := build

# =============================================================================
# Source file lists
# =============================================================================

# Hook DSO: the intercept shim + persistent-mode detector
HOOK_SRC := src/cipher_intercept_cudart.cpp src/cipher_persist.cpp src/cipher_graph_inspect.cpp src/cipher_kernel_table.cpp

# RT DSO: all src/*.cpp EXCEPT the hook sources and the standalone tuner DSO,
# PLUS top-level dispatch and oracle
RT_CPP_SRC := $(filter-out src/cipher_intercept_cudart.cpp src/cipher_persist.cpp src/cipher_nccl_tuner.cpp src/cipher_dispatch.cpp src/cipher_oracle.cpp, \
                $(wildcard src/*.cpp))
RT_CPP_SRC += cipher_dispatch.cpp cipher_oracle.cpp

RT_CU_SRC  := $(wildcard src/*.cu)

# =============================================================================
# Object file lists (all go into build/)
# =============================================================================

HOOK_OBJ := $(BUILDDIR)/cipher_intercept_cudart.o $(BUILDDIR)/cipher_persist.o $(BUILDDIR)/cipher_graph_inspect.o $(BUILDDIR)/cipher_kernel_table.o

# For src/*.cpp -> build/src__<name>.o  (flatten into build/)
# For top-level *.cpp -> build/<name>.o
RT_CPP_OBJ := $(patsubst src/%.cpp,$(BUILDDIR)/src__%.o,$(filter src/%,$(RT_CPP_SRC))) \
              $(patsubst %.cpp,$(BUILDDIR)/%.o,$(filter-out src/%,$(RT_CPP_SRC)))

RT_CU_OBJ  := $(patsubst src/%.cu,$(BUILDDIR)/src__%.cu.o,$(RT_CU_SRC))

RT_OBJ     := $(RT_CPP_OBJ) $(RT_CU_OBJ)

# =============================================================================
# Targets
# =============================================================================

.PHONY: all hook rt tuner clean

all: hook rt tuner

hook: libcipher_hook.so

rt: libcipher_rt.so

tuner: libcipher_nccl_tuner.so libnccl-tuner-cipher.so

# --- Hook DSO ----------------------------------------------------------------

libcipher_hook.so: $(HOOK_OBJ) exports.map
	$(CXX) -shared -o $@ $(HOOK_OBJ) -Wl,--version-script=exports.map -ldl

$(BUILDDIR)/cipher_intercept_cudart.o: src/cipher_intercept_cudart.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/cipher_persist.o: src/cipher_persist.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/cipher_graph_inspect.o: src/cipher_graph_inspect.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/cipher_kernel_table.o: src/cipher_kernel_table.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# --- Tuner plugin DSO --------------------------------------------------------
# Standalone .so loaded by NCCL via NCCL_TUNER_PLUGIN. No link-time deps on
# libcipher_rt.so — the plugin resolves cipher_nccl_record_decide via
# dlsym(RTLD_DEFAULT) at init time so it can load cleanly even when CIPHER
# is not LD_PRELOADed (in which case the plugin is a silent passthrough).
libcipher_nccl_tuner.so: $(BUILDDIR)/cipher_nccl_tuner.o
	$(CXX) -shared -o $@ $< -ldl

# NCCL 2.21+ searches for plugins named libnccl-tuner-<NAME>.so where
# NCCL_TUNER_PLUGIN=<NAME>. We ship the .so under BOTH names: the
# descriptive name for ctypes tests, and NCCL's mandatory naming scheme
# for runtime load via NCCL_TUNER_PLUGIN=cipher.
libnccl-tuner-cipher.so: libcipher_nccl_tuner.so
	cp $< $@

$(BUILDDIR)/cipher_nccl_tuner.o: src/cipher_nccl_tuner.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# --- RT DSO ------------------------------------------------------------------

libcipher_rt.so: $(RT_OBJ)
	$(CXX) -shared -o $@ $(RT_OBJ) -L$(CUDA_PATH)/lib64 -lcudart -lssl -lcrypto -lpthread -ldl

# Pattern rule: src/*.cpp -> build/src__*.o
$(BUILDDIR)/src__%.o: src/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Pattern rule: top-level *.cpp -> build/*.o
$(BUILDDIR)/%.o: %.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Pattern rule: src/*.cu -> build/src__*.cu.o
$(BUILDDIR)/src__%.cu.o: src/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

# --- Build directory ---------------------------------------------------------

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# --- Clean -------------------------------------------------------------------

clean:
	rm -rf $(BUILDDIR) libcipher_hook.so libcipher_rt.so
