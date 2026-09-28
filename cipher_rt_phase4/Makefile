# libcipher_rt — Phase 4 T4.2.2 build.
# Extends libcipher_v2 Phase 3 substrate (REGISTER_TENANT + CUPTI launch
# counter) with PARTITION_ROUTER actuator (depth win #1: stream priority
# + mem_sync_domain_map).

CC      ?= gcc
CXX     ?= g++
NVCC    ?= nvcc
CFLAGS  ?= -O2 -Wall -Wextra -fPIC
CXXFLAGS ?= -O2 -Wall -Wextra -fPIC
# Week 2 Step 2 v3 D1 — mirrored from cipher-may13-evidence/Makefile for the
# 3 may13 .cu ports (cipher_green_ctx.cu, cipher_l2_persist.cu, cipher_liquid_state.cu).
NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_90 --compiler-options -fPIC,-Wall,-Wextra,-Wno-unused-parameter
LDFLAGS ?= -shared -fPIC
LIBS    := -lpthread -lcupti -lcuda -ldl -lcudart -lcusolver

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

INCLUDES := -I$(CIPHER_KMOD_DIR) -I$(CUDA_INCLUDE) -I$(CUPTI_INCLUDE) -Iinclude -I.
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
# CP 5.4 Step 1.3: cipher_rt_arbitrate.o retired (ARB / ioctl nr 9 — nr 9
# returns -ENOSYS on the CP 5.4 kmod; allocation moved to the CP 5.4 group
# ledger via CIPHER_CP54_ALLOCATE). The .c is kept in-tree, unbuilt.
OBJS   := cipher_inject.o cipher_tenant.o cipher_cupti.o \
          cipher_rt_geom_capture.o \
          cipher_rt_partition_router.o cipher_rt_tenant.o \
          cipher_rt_sm_packer.o \
          cipher_rt_green_ctx.o cipher_rt_volt.o \
          cipher_rt_matmul_dispatch.o cipher_rt_cublas_shim.o \
          cipher_rt_cublaslt_variants.o cipher_rt_cublaslt_layout.o \
          cipher_rt_dlsym_hook.o \
          cipher_rt_got_patch.o \
          cipher_rt_marlin_kernel_src.o cipher_rt_marlin_engine.o \
          cipher_rt_marlin_actuator.o \
          cipher_rt_fp8_engine.o cipher_rt_fp8_actuator.o \
          cipher_rt_koopman_engine.o \
          cipher_rt_attn_dispatch.o cipher_rt_attn_test_actuator.o \
          cipher_rt_audit.o \
          cipher_rt_classify_substrate.o cipher_rt_classify_observer.o \
          cipher_rt_dispatch.o cipher_rt_oracle_bridge.o \
          cipher_rt_sense_transition.o \
          cipher_rt_commit.o \
          cipher_rt_ring_write.o \
          cipher_rt_pager.o \
          cipher_rt_remember_consumer.o \
          cipher_rt_counter_dump.o \
          cipher_rt_cuinit_hook.o \
          cipher_rt_tc_probe.o \
          cipher_rt_machete_intercept.o \
          cipher_rt_attn_6pattern.o \
          cipher_rt_coresidence.o \
          cipher_rt_pool.o \
          cipher_stream_resolver.o \
          cipher_rt_fairness.o \
          cipher_workload_detect.o \
          cipher_may13_harness.o \
          may13_cipher_dispatch.o may13_cipher_oracle.o \
          may13_cipher_recipes.o may13_cipher_sense.o \
          may13_cipher_structural_lookup.o may13_cipher_kernel_table.o \
          may13_cipher_runtime.o may13_cipher_telemetry.o \
          may13_cipher_continuity.o may13_cipher_loop.o \
          may13_cipher_pipeline.o may13_cipher_pulse.o \
          may13_cipher_trace.o may13_cipher_receipt.o \
          may13_cipher_carbon.o may13_cipher_fairness.o \
          may13_cipher_fairness_shm.o may13_cipher_guard.o \
          may13_cipher_determinism.o may13_cipher_topology.o \
          may13_cipher_comply.o \
          may13_cipher_green_ctx.o may13_cipher_l2_persist.o \
          may13_cipher_liquid_state.o \
          may13_cipher_edmd.o may13_cipher_edmd_live.o \
          may13_cipher_koopman_runtime.o may13_cipher_lnn.o \
          may13_cipher_block_sub_kernel.o \
          cipher_may13_stubs.o \
          cipher_rt_intercept_cudart.o \
          cipher_rt_graph_inspect.o \
          cipher_rt_intercept_stubs.o

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

# F-B.3.6.1 dlsym hook + cuBLAS-Lt variant trampolines (Path B').
cipher_rt_dlsym_hook.o: cipher_rt_dlsym_hook.c cipher_rt_dlsym_hook.h cipher_rt_got_patch.h cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_cublaslt_layout.o: cipher_rt_cublaslt_layout.c cipher_rt_cublaslt_layout.h cipher_rt_matmul_dispatch.h cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_cublaslt_variants.o: cipher_rt_cublaslt_variants.c cipher_rt_dlsym_hook.h cipher_v2_internal.h cipher_rt_matmul_dispatch.h cipher_rt_cublaslt_layout.h
	$(CC) $(CFLAGS) -fvisibility=default $(INCLUDES) -c -o $@ $<

cipher_rt_marlin_kernel_src.o: cipher_rt_marlin_kernel_src.cpp cipher_rt_marlin_kernel_src.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_marlin_engine.o: cipher_rt_marlin_engine.cpp cipher_rt_marlin.h cipher_rt_marlin_kernel_src.h cipher_rt_marlin_perms.h cipher_v2_internal.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_marlin_actuator.o: cipher_rt_marlin_actuator.c cipher_rt_marlin.h cipher_rt_matmul_dispatch.h cipher_v2_internal.h \
                              cipher_rt_dispatch.h cipher_rt_classify_observer.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# D.9 FP8 actuator (per-tensor scalar fused FP8 E4M3). Net-new, default-OFF.
# Quant kernels are NVRTC-compiled at runtime + launched via driver-API
# cuLaunchKernel (NOT statically compiled) — static .cu kernels would register
# through CIPHER's own __cudaRegisterFunction intercept at LD_PRELOAD time
# (before substrate init) and SIGFPE. Driver-API module load sidesteps that.
cipher_rt_fp8_engine.o: cipher_rt_fp8_engine.cpp cipher_rt_fp8.h cipher_v2_internal.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_fp8_actuator.o: cipher_rt_fp8_actuator.c cipher_rt_fp8.h cipher_rt_matmul_dispatch.h cipher_v2_internal.h
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
cipher_stream_resolver.o: cipher_stream_resolver.c cipher_stream_resolver.h
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) -c -o $@ $<

cipher_rt_fairness.o: cipher_rt_fairness.c cipher_rt_fairness.h
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) -c -o $@ $<

cipher_rt_pager.o: cipher_rt_pager.c cipher_rt_pager.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<
cipher_rt_ring_write.o: cipher_rt_ring_write.c cipher_rt_ring_write.h
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) -c -o $@ $<

cipher_rt_remember_consumer.o: cipher_rt_remember_consumer.cpp cipher_rt_remember_consumer.h cipher_rt_ring_write.h include/may13/cipher_lnn.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_counter_dump.o: cipher_rt_counter_dump.c cipher_rt_counter_dump.h cipher_v2_internal.h
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) -c -o $@ $<

cipher_rt_cuinit_hook.o: cipher_rt_cuinit_hook.c cipher_v2_internal.h
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) -c -o $@ $<

cipher_rt_tc_probe.o: cipher_rt_tc_probe.c cipher_rt_tc_probe.h
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) -c -o $@ $<

cipher_rt_machete_intercept.o: cipher_rt_machete_intercept.c cipher_v2_internal.h cipher_rt_got_patch.h
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) -c -o $@ $<

cipher_rt_attn_6pattern.o: cipher_rt_attn_6pattern.c cipher_v2_internal.h cipher_rt_got_patch.h
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) -c -o $@ $<

cipher_rt_commit.o: cipher_rt_commit.c cipher_rt_commit.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# W.6 sub-C — substrate co-residence client (kmod cohort registry nr 30/31).
cipher_rt_coresidence.o: cipher_rt_coresidence.c cipher_rt_coresidence.h cipher_v2_internal.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# W.4a — cross-tenant POOL eligibility + correctness substrate (decision-only).
cipher_rt_pool.o: cipher_rt_pool.c cipher_rt_pool.h cipher_rt_coresidence.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

cipher_rt_audit.o: cipher_rt_audit.c cipher_rt_audit.h cipher_rt_matmul_dispatch.h cipher_rt_attn_dispatch.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Week 2 Step 3 — classify-routing substrate (3rd in matmul/attn family).
# Priority-ordered 16-slot registry mirroring cipher_rt_matmul_dispatch.c.
# may13 default classifier auto-registers at static-init time, wrapping
# cipher::classify_launch (pure geometry-based, no CUDA/NVML at .so load).
# Not yet wired to any hot path — Step 6 wires it into F1 intercept.
cipher_rt_classify_substrate.o: cipher_rt_classify_substrate.cpp \
                                 cipher_rt_classify_substrate.h \
                                 include/may13/cipher_classify.hpp
	$(CXX) $(CXXFLAGS) -std=c++17 $(INCLUDES) -c -o $@ $<

# Week 2 Step 4 — classify observer stub. Atomic-counter sink for the
# classify substrate. Step 6 wires observe() into the route loop;
# Step 5 /proc/cipher/classify_stats pulls from snapshot() via ioctl.
cipher_rt_classify_observer.o: cipher_rt_classify_observer.c \
                                cipher_rt_classify_observer.h \
                                cipher_rt_classify_substrate.h \
                                cipher_rt_dispatch.h \
                                cipher_rt_oracle_bridge.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Week 4 Step 1 — C-callable bridge over cipher_oracle (C++ TU).
# cipher_oracle.h transitively includes cipher_classify.hpp (namespace +
# enum class + <atomic>), so direct inclusion from a C TU fails. The
# bridge owns the BSS-resident CipherOracleState + CAS lazy-init flag
# and exposes oracle_decide() through a plain-C surface.
cipher_rt_oracle_bridge.o: cipher_rt_oracle_bridge.cpp cipher_rt_oracle_bridge.h \
                            include/may13/cipher_oracle.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

# Week 3 Step 1 — GEMM dispatch routing scaffolding (v1: GEMM->Marlin,
# else PASS_THROUGH). CIPHER_DISPATCH_LIVE env-gated; default 0 keeps
# the substrate observe-only through Steps 1-3. Step 4 flips to 1.
cipher_rt_dispatch.o: cipher_rt_dispatch.cpp cipher_rt_dispatch.h \
                       cipher_rt_classify_substrate.h
	$(CXX) $(CXXFLAGS) -std=c++17 $(INCLUDES) -c -o $@ $<

# Week 3 Step 4 Option II-a — invented transition-detection wrapper
# above SENSE. Per-tenant fingerprint table + hysteresis state machine
# + tool-idle heuristic + bounded proposals queue + ioctl push.
# Research-grade v1; thresholds need measurement-driven tuning.
cipher_rt_sense_transition.o: cipher_rt_sense_transition.c \
                               cipher_rt_sense_transition.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Week 1 Step 3 — cross-tree compile harness. Includes one may13 header
# (cipher_classify.hpp via may13/ prefix) and declares one static marker
# string plus two static may13-typed values. No runtime effect; verified
# by nm-diff that only the harness marker is added to libcipher_rt.so.
cipher_may13_harness.o: src/cipher_may13_harness.cpp include/may13/cipher_classify.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Week 2 Step 2 v3 D1 — A4+H1 may13 ports (ROOT variants for cipher_dispatch.cpp
# and cipher_oracle.cpp per D3 ROOT-CANONICAL adjudication; src/ for the other
# 9 sources which have no variant pair). 8 .cpp + 3 .cu inert TUs.
# A4 adjudication: cipher_intercept.cpp + cipher_intercept_cudart.cpp excluded
# (static-init pre-flight FLAG; conflicts with cipher_rt_phase4's GOT patcher
# per CP 2.5 D2(iii)). H1 stub provides cipher_intercept_stats() locally.
may13_cipher_dispatch.o: src/may13/cipher_dispatch.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_oracle.o: src/may13/cipher_oracle.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_recipes.o: src/may13/cipher_recipes.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_sense.o: src/may13/cipher_sense.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_structural_lookup.o: src/may13/cipher_structural_lookup.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_kernel_table.o: src/may13/cipher_kernel_table.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_runtime.o: src/may13/cipher_runtime.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_telemetry.o: src/may13/cipher_telemetry.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

# Week 4 Step 2 — Tier A observability ports (LOOP/PIPELINE/PULSE/CONTINUITY).
# Closure-clean per WEEK_4_STEP_2_ENTRY_VERIFICATION.md §B.4: only fortified-libc
# undefs + cipher_sense_get_type (already exported by libcipher_rt).
may13_cipher_loop.o: src/may13/cipher_loop.cpp include/may13/cipher_loop.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_pipeline.o: src/may13/cipher_pipeline.cpp include/may13/cipher_pipeline.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_pulse.o: src/may13/cipher_pulse.cpp include/may13/cipher_pulse.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_continuity.o: src/may13/cipher_continuity.cpp include/may13/cipher_continuity.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

# Week 4 Step 3 — Tier B observability ports.
may13_cipher_trace.o: src/may13/cipher_trace.cpp include/may13/cipher_trace.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_receipt.o: src/may13/cipher_receipt.cpp include/may13/cipher_receipt.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_carbon.o: src/may13/cipher_carbon.cpp include/may13/cipher_carbon.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_fairness.o: src/may13/cipher_fairness.cpp include/may13/cipher_fairness.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_fairness_shm.o: src/may13/cipher_fairness_shm.cpp include/may13/cipher_fairness_shm.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_guard.o: src/may13/cipher_guard.cpp include/may13/cipher_guard.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_determinism.o: src/may13/cipher_determinism.cpp include/may13/cipher_determinism.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

# cipher_topology.cpp has CUDA driver calls inside cipher_topology_init() —
# gated on CIPHER_TOPOLOGY=on AND explicit caller invocation. Static-init
# is empty so .so load does NOT touch CUDA. See WEEK_4_STEP_3_RESULT.md §I.
may13_cipher_topology.o: src/may13/cipher_topology.cpp include/may13/cipher_topology.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

# cipher_comply.cpp aggregates the prior 7 Tier B exports (lands LAST in
# OBJS so 8 of its 9 TREE-PULLs are satisfied by sibling .o's).
may13_cipher_comply.o: src/may13/cipher_comply.cpp include/may13/cipher_comply.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_green_ctx.o: src/may13/cipher_green_ctx.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

may13_cipher_l2_persist.o: src/may13/cipher_l2_persist.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

may13_cipher_liquid_state.o: src/may13/cipher_liquid_state.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

# W14 Step 2 — Koopman tier port (5 sources from cipher-may13-evidence).
# Port covers the full narrow-domain Koopman compute substrate: EDMD pipeline,
# live calibrator, runtime derivation, CfC LNN, and GPU kernel surface.
# Adjudication 2026-05-23 path (β): goal 4 v1 narrow-domain ships via new
# cipher_rt_koopman_engine wire-up (W14 Step 2 B.2), not apply_recipe path.
may13_cipher_edmd.o: src/may13/cipher_edmd.cpp include/may13/cipher_edmd.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_edmd_live.o: src/may13/cipher_edmd_live.cpp include/may13/cipher_edmd_live.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_koopman_runtime.o: src/may13/cipher_koopman_runtime.cpp include/may13/cipher_koopman_runtime.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_lnn.o: src/may13/cipher_lnn.cpp include/may13/cipher_lnn.h
	$(CXX) $(CXXFLAGS) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 $(INCLUDES) -c -o $@ $<

may13_cipher_block_sub_kernel.o: src/may13/cipher_block_sub_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

# W14 Step 2 B.2 — Koopman engine (second actuator on the matmul-dispatch
# substrate; pattern parallel to cipher_rt_marlin_actuator). Dispatches
# cipher_koopman_fp16_launch_shape() from the may13_cipher_block_sub_kernel
# .cu for shapes calibrated into the .cu shape registry by
# cipher_edmd_live_collect's background SVD pipeline.
cipher_rt_koopman_engine.o: cipher_rt_koopman_engine.cpp cipher_rt_koopman.h \
                            cipher_rt_matmul_dispatch.h
	$(CXX) $(CXXFLAGS) -std=c++17 $(INCLUDES) -c -o $@ $<

# K.1 — Workload Classifier (cipher_workload_detect.cpp). Builds the
# classifier substrate that emits CipherProfile from driver-level signals;
# observe_* calls wired at cipher_cupti.c, cipher_rt_cublas_shim.c, and
# the cudaMalloc + cuStreamBeginCapture intercept hooks in
# src/may13_intercept/cipher_intercept_cudart.cpp.
cipher_workload_detect.o: src/cipher_workload_detect.cpp \
                          include/cipher_workload_detect.h \
                          cipher_rt_coresidence.h cipher_rt_pool.h
	$(CXX) $(CXXFLAGS) -std=c++17 $(INCLUDES) -c -o $@ $<

# H1 stub — provides cipher_intercept_stats() returning zero counters.
# See WEEK_2_STEP_2_A4_ORPHAN_CONSUMER.md for rationale.
cipher_may13_stubs.o: src/cipher_may13_stubs.cpp include/may13/cipher_intercept.h
	$(CXX) $(CXXFLAGS) -std=c++17 $(INCLUDES) -c -o $@ $<

# V1 Phase B B.6''.9.8.1 — may13 driver-API intercept port. Adapted from
# cipher-may13-evidence/src/cipher_intercept_cudart.cpp (md5
# 75c555287eac2f90ca0f17c4fe669664) and src/cipher_graph_inspect.cpp (md5
# 05b9ac7171baabfcd0e872a76108b105). cuBLAS + NCCL entries dropped from
# may13 g_patches[] (cipher_rt_phase4 canonical). Constructor adapted to
# explicit cipher_rt_intercept_init() called from cipher_inject.c.
cipher_rt_intercept_cudart.o: src/may13_intercept/cipher_intercept_cudart.cpp \
                              cipher_rt_intercept.h
	$(CXX) $(CXXFLAGS) -std=c++17 -fvisibility=default $(INCLUDES) -c -o $@ $<

cipher_rt_graph_inspect.o: src/may13_intercept/cipher_graph_inspect.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -fvisibility=default $(INCLUDES) -c -o $@ $<

# B.6''.9.8.2: inert stubs for 6 helpers referenced by may13 cipher_intercept_cudart
# that come from cipher_persist + cipher_flow_* substrate (NOT ported in B.6''.9.8;
# v1.x scope). Make the may13 hot path fall through to normal launch behavior.
cipher_rt_intercept_stubs.o: src/may13_intercept/cipher_intercept_stubs.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 $(INCLUDES) -c -o $@ $<

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LIBS) $(C10_LIBS) -lcrypto

clean:
	rm -f $(TARGET) $(OBJS)
