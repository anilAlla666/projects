# libcipher_rt — Phase 4 T4.2.2 build.
# Extends libcipher_v2 Phase 3 substrate (REGISTER_TENANT + CUPTI launch
# counter) with PARTITION_ROUTER actuator (depth win #1: stream priority
# + mem_sync_domain_map).

CC      ?= gcc
CXX     ?= g++
CFLAGS  ?= -O2 -Wall -Wextra -fPIC
CXXFLAGS ?= -O2 -Wall -Wextra -fPIC
LDFLAGS ?= -shared -fPIC
LIBS    := -lpthread -lcupti -lcuda -ldl

CIPHER_KMOD_DIR ?= /home/ubuntu/cipher_kmod
CUDA_INCLUDE    ?= /usr/include
CUPTI_INCLUDE   ?= /usr/include

# T4.6.1: ATen headers + libtorch_cpu for SDPA dispatcher symbol
# interposition. Discovered at config time.
TORCH_DIR     := $(shell python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
TORCH_INCLUDE := $(TORCH_DIR)/include
TORCH_LIB     := $(TORCH_DIR)/lib

# CP 2.5 D2(iii) addendum — link the small ABI-stable c10 core only.
# Bare-drop (sever ALL torch libs) was tested and failed: the attn TU's
# inline ATen header code pulls 8 out-of-line c10:: symbols, one a *data*
# symbol (c10::UndefinedTensorImpl::_singleton) that cannot be lazy-deferred.
# libtorch_cpu (451 MB, version-fragile) IS severed; libc10 (1.5 MB,
# unversioned soname, ABI-stable) is retained via the linker and bundled in
# cipher-platform.deb. -rpath points the loader at libc10.so at runtime.
C10_LIBS := -Wl,-rpath,$(TORCH_LIB) -L$(TORCH_LIB) -lc10

INCLUDES := -I$(CIPHER_KMOD_DIR) -I$(CUDA_INCLUDE) -I$(CUPTI_INCLUDE)
TORCH_INCLUDES := -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include
# CP 2.5 — libtorch_cpu is NOT linked; libc10 IS (see C10_LIBS / D2(iii)
# addendum above). The attn substrate needs ATen headers at compile time
# (TORCH_INCLUDES); its inline ATen header code resolves to 8 out-of-line
# c10:: core symbols, supplied by libc10.so (DT_NEEDED libc10.so only — no
# libtorch_cpu). libcipher_rt.so thus loads in any CUDA process whose
# c10 core is reachable; the .deb bundles libc10.so to guarantee that.
# Real SDPA functions are still resolved at runtime via
# dlopen(libtorch_cpu.so)+dlsym (resolve_lazy). T4.6.1 linked -ltorch_cpu
# (forced via --no-as-needed) only so dlsym(RTLD_NEXT) had a fall-through
# target; CP 2.5 retired that.

# Symbols we DO want exported (InitializeInjection variants need to be
# dlsym-able by the CUDA driver).
TARGET := libcipher_rt.so
OBJS   := cipher_inject.o cipher_tenant.o cipher_cupti.o \
          cipher_rt_partition_router.o cipher_rt_tenant.o \
          cipher_rt_arbitrate.o cipher_rt_sm_packer.o \
          cipher_rt_green_ctx.o cipher_rt_volt.o \
          cipher_rt_matmul_dispatch.o cipher_rt_cublas_shim.o \
          cipher_rt_got_patch.o \
          cipher_rt_marlin_kernel_src.o cipher_rt_marlin_engine.o \
          cipher_rt_marlin_actuator.o \
          cipher_rt_attn_dispatch.o cipher_rt_attn_test_actuator.o \
          cipher_rt_audit.o

# CP 2.5 — no version script. T4.5's cublas_version.map version-tagged the
# shim's cublasGemmEx@libcublas.so.13 for LD_PRELOAD link-order resolution;
# GOT patching (cipher_rt_got_patch.c) replaces that. cublas_version.map is
# retained on disk for history but is no longer part of the build.

all: $(TARGET)

cipher_inject.o: cipher_inject.c cipher_v2_internal.h
	$(CC) $(CFLAGS) -fvisibility=default $(INCLUDES) -c -o $@ $<

cipher_tenant.o: cipher_tenant.c cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_cupti.o: cipher_cupti.c cipher_v2_internal.h cipher_rt_partition_router.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_partition_router.o: cipher_rt_partition_router.c cipher_rt_partition_router.h cipher_v2_internal.h cipher_rt_tenant.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_tenant.o: cipher_rt_tenant.cpp cipher_rt_tenant.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_arbitrate.o: cipher_rt_arbitrate.c cipher_rt_arbitrate.h cipher_v2_internal.h cipher_rt_tenant.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_sm_packer.o: cipher_rt_sm_packer.c cipher_rt_sm_packer.h cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_green_ctx.o: cipher_rt_green_ctx.c cipher_rt_green_ctx.h cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_volt.o: cipher_rt_volt.c cipher_rt_volt.h cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_matmul_dispatch.o: cipher_rt_matmul_dispatch.c cipher_rt_matmul_dispatch.h cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_cublas_shim.o: cipher_rt_cublas_shim.c cipher_rt_matmul_dispatch.h cipher_rt_got_patch.h cipher_v2_internal.h
	$(CC) $(CFLAGS) -fvisibility=default $(INCLUDES) -c -o $@ $<

# CP 2.5 — GOT/PLT patcher. Replaces LD_PRELOAD link-order interposition.
cipher_rt_got_patch.o: cipher_rt_got_patch.c cipher_rt_got_patch.h cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_marlin_kernel_src.o: cipher_rt_marlin_kernel_src.cpp cipher_rt_marlin_kernel_src.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_marlin_engine.o: cipher_rt_marlin_engine.cpp cipher_rt_marlin.h cipher_rt_marlin_kernel_src.h cipher_rt_marlin_perms.h cipher_v2_internal.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_marlin_actuator.o: cipher_rt_marlin_actuator.c cipher_rt_marlin.h cipher_rt_matmul_dispatch.h cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Attention substrate. C++ for the ATen SDPA dispatcher signatures
# (headers only — libtorch is not linked, see the TORCH note above).
# CP 2.5: the 3 SDPA trampolines are hidden (pragma in the .cpp) and
# installed by the GOT patcher; -fvisibility=default keeps the substrate's
# own API (init / register / diagnostics) visible.
cipher_rt_attn_dispatch.o: cipher_rt_attn_dispatch.cpp cipher_rt_attn_dispatch.h cipher_rt_got_patch.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 \
	       -fvisibility=default \
	       $(INCLUDES) $(TORCH_INCLUDES) -c -o $@ $<

cipher_rt_attn_test_actuator.o: cipher_rt_attn_test_actuator.c cipher_rt_attn_dispatch.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Fusion Op 9 AUDIT — HMAC-SHA256 tamper-evident chain (links libcrypto).
cipher_rt_audit.o: cipher_rt_audit.c cipher_rt_audit.h cipher_rt_matmul_dispatch.h cipher_rt_attn_dispatch.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LIBS) $(C10_LIBS) -lcrypto

clean:
	rm -f $(TARGET) $(OBJS)
