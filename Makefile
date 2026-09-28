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
# --cudart=shared: nvcc emits the device fatbin and the
# __cudaRegisterFatBinary calls that wire it up, but does NOT statically
# link any cudart symbol. The references resolve LAZILY against whatever
# libcudart soname the host process has loaded at runtime (.so.12 for
# transformers/cu128, .so.13 for vllm 0.20+torch 2.11+cu130). This
# matters because:
#   1. With --cudart=static (default for some toolchains), the .cu fatbin
#      ends up registered into nvcc's bundled cudart instance, splitting
#      the kernel registry from torch's cudart.
#   2. Without --cudart=shared, the resulting .o pulls in libcudart_static
#      forcing a NEEDED libcudart.so.12 dependency on libcipher_rt.so.
#   3. The compat shim (cipher_cudart_compat.cpp) handles cudaXxx() API
#      calls; this flag handles the fatbin-registration glue.
NVCCFLAGS := -std=c++17 -O2 -arch=sm_90 --cudart=shared --compiler-options -fPIC,-Wall,-Wextra,-Wno-unused-parameter
INCLUDES  := -I./include -I$(CUDA_PATH)/include

BUILDDIR  := build

# =============================================================================
# Source file lists
# =============================================================================

# Hook DSO: the intercept shim + persistent-mode detector
HOOK_SRC := src/cipher_intercept_cudart.cpp src/cipher_persist.cpp src/cipher_graph_inspect.cpp src/cipher_kernel_table.cpp src/cipher_flow_recorder.cpp src/cipher_flow_patterns.cpp src/cipher_flow_substitute.cpp

# RT DSO: all src/*.cpp EXCEPT the hook sources and the standalone tuner DSO,
# PLUS top-level dispatch and oracle
RT_CPP_SRC := $(filter-out src/cipher_intercept_cudart.cpp src/cipher_persist.cpp src/cipher_nccl_tuner.cpp src/cipher_dispatch.cpp src/cipher_oracle.cpp src/cipher_receipt_signing.cpp, \
                $(wildcard src/*.cpp))
RT_CPP_SRC += cipher_dispatch.cpp cipher_oracle.cpp

RT_CU_SRC  := $(wildcard src/*.cu)

# =============================================================================
# Object file lists (all go into build/)
# =============================================================================

HOOK_OBJ := $(BUILDDIR)/cipher_intercept_cudart.o $(BUILDDIR)/cipher_persist.o $(BUILDDIR)/cipher_graph_inspect.o $(BUILDDIR)/cipher_kernel_table.o $(BUILDDIR)/cipher_flow_recorder.o $(BUILDDIR)/cipher_flow_patterns.o $(BUILDDIR)/cipher_flow_substitute.o

# For src/*.cpp -> build/src__<name>.o  (flatten into build/)
# For top-level *.cpp -> build/<name>.o
RT_CPP_OBJ := $(patsubst src/%.cpp,$(BUILDDIR)/src__%.o,$(filter src/%,$(RT_CPP_SRC))) \
              $(patsubst %.cpp,$(BUILDDIR)/%.o,$(filter-out src/%,$(RT_CPP_SRC)))

RT_CU_OBJ  := $(patsubst src/%.cu,$(BUILDDIR)/src__%.cu.o,$(RT_CU_SRC))

RT_OBJ     := $(RT_CPP_OBJ) $(RT_CU_OBJ)

# =============================================================================
# Targets
# =============================================================================

.PHONY: all hook rt tuner receipt clean

all: hook rt tuner receipt

hook: libcipher_hook.so

rt: libcipher_rt.so

tuner: libcipher_nccl_tuner.so libnccl-tuner-cipher.so

receipt: libcipher_receipt.so

# --- Receipt-signing DSO (Tier B isolation per OP_CONTRACT.md) -----
# Standalone HMAC signing of billing receipts. Loaded via dlopen by the
# orchestrator when CIPHER_RECEIPT_SIGN=on. NO link-time deps on
# libcipher_rt.so / libcipher_hook.so — keeps key material out of the
# hot DSO and lets crashes in signing not crash CIPHER core.
libcipher_receipt.so: $(BUILDDIR)/cipher_receipt_signing.o
	$(CXX) -shared -o $@ $< -lssl -lcrypto

$(BUILDDIR)/cipher_receipt_signing.o: src/cipher_receipt_signing.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

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

$(BUILDDIR)/cipher_flow_recorder.o: src/cipher_flow_recorder.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/cipher_flow_patterns.o: src/cipher_flow_patterns.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/cipher_flow_substitute.o: src/cipher_flow_substitute.cpp | $(BUILDDIR)
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

# Note: deliberately do NOT link -lcudart. We want unversioned cudaXxx
# symbols that resolve LAZILY at runtime against whatever cudart soname
# the host process loaded (libcudart.so.12 for transformers/cu128,
# libcudart.so.13 for vllm 0.20 + torch 2.11+cu130, etc.). Linking to a
# specific .so.12 forces a NEEDED dependency that loads .12 alongside
# whatever the application uses, splitting cuda registration state and
# producing "invalid device function" errors on multi-cudart processes.
#
# --allow-shlib-undefined tells the linker to ship unresolved cudaXxx;
# they bind at first call via the global symbol table.
libcipher_rt.so: $(RT_OBJ)
	$(CXX) -shared -o $@ $(RT_OBJ) \
	  -Wl,--allow-shlib-undefined \
	  -Wl,--unresolved-symbols=ignore-all \
	  -lssl -lcrypto -lpthread -ldl

# Lite build retired — the sentinel-based deferred fatbin registration
# in cipher_intercept_cudart.cpp's __cudaRegister* shims handles the
# vLLM/cu13 race correctly. One rt build works on transformers/cu12 AND
# vllm/cu13. The lite target is kept around as a phony alias only for
# any deployment scripts that still reference it.
libcipher_rt_lite.so: libcipher_rt.so
	cp $< $@

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
