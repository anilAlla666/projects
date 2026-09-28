// =============================================================================
// CIPHER — driver-API + cudart launch interception via cuGetProcAddress hook
//          + ELF GOT patching (PORTED FROM may13)
//
// V1 Phase B B.6''.9.8.1 (2026-05-27): ported from cipher-may13-evidence
//   src/cipher_intercept_cudart.cpp (md5 75c555287eac2f90ca0f17c4fe669664).
//
// Adaptations vs may13 (additive port; existing cipher_rt_phase4 substrate
// PRESERVED as defense-in-depth):
//   - cuBLAS shim entries DROPPED from g_patches[] (cublasGemmEx,
//     cublasLtMatmul, ncclAllReduce) — cipher_rt_phase4's CUDA-13 + Path B'
//     (cipher_rt_cublas_shim.c + cipher_rt_cublaslt_variants.c +
//     cipher_rt_dlsym_hook.c) is canonical. The impl functions themselves
//     are retained but unreachable.
//   - __attribute__((constructor(101))) cipher_hook_init() REPLACED with
//     extern "C" cipher_rt_intercept_init() called from
//     cipher_inject.c:cipher_v2_init_body() AFTER cipher_rt_got_patch_init.
//   - pthread_once gates idempotency in the calling context.
//
// Two-layer interception (preserved from may13):
//   Layer 1: Exported symbols (cuLaunchKernel, cudaLaunchKernel, etc.)
//            Picked up by the dynamic linker for any DSO loaded after us.
//   Layer 2: GOT patching via dl_iterate_phdr.  Re-run on init.
//
// No dlsym hooking.  No CUDA SDK linkage at compile time.
// =============================================================================

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <sys/mman.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <x86intrin.h>

#include "may13/cipher_persist.h"
#include "may13/cipher_kernel_table.h"
#include "may13/cipher_flow_recorder.h"
#include "may13/cipher_flow_patterns.h"
#include "may13/cipher_flow_substitute.h"
#include "may13/cipher_attn_koopman.h"
#include "cipher_workload_detect.h"   /* K.1: observe alloc + capture/graph + nccl */

// Weak-link the attention FSM entry points — they live in libcipher_rt.so,
// not the hook DSO. When CIPHER_ATTN_KOOPMAN=0 (default) the FSM returns
// PASS_THROUGH for every call so this adds one branch on the hot path.
extern "C" __attribute__((weak)) CipherAttnAction cipher_attn_fsm_on_gemm(
    void*, int, int, int, int, int,
    const void*, const void*, int, int,
    const void*, int, int,
    const void*, void*, int, int,
    int, int, void*);
extern "C" __attribute__((weak)) CipherAttnAction cipher_attn_fsm_on_kernel(
    const void*, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned,
    void**, void**, void*, void*);
extern "C" __attribute__((weak)) bool cipher_attn_koopman_enabled(void);

// Version-agnostic cudart resolver. torch 2.7 ships cudart 12, torch 2.11
// ships cudart 13 at a bundled path (nvidia/cu13/lib/) that's not in the
// standard linker search path. If we proactively dlopen "libcudart.so.13",
// the linker can't find it; we'd silently fall through to the system
// libcudart.so.12, end up with TWO cudarts in-process, and forward torch's
// __cudaRegister* into the wrong one → cudaErrorInvalidResourceHandle.
//
// Strategy: ONLY adopt a cudart that is already mapped into the process.
// If torch has loaded its bundled cudart, RTLD_NOLOAD finds it by SONAME.
// If nothing is loaded yet (constructor-time), return nullptr; callers
// retry on next invocation, by which time torch will have run its own
// DT_NEEDED chain.
static void* resolve_cudart() {
    static std::atomic<void*> cached{nullptr};
    void* h = cached.load(std::memory_order_acquire);
    if (h) return h;
    static const char* const kVersions[] = {
        "libcudart.so.13", "libcudart.so.12", "libcudart.so.11",
        "libcudart.so",    nullptr,
    };
    for (const char* const* v = kVersions; *v; ++v) {
        h = dlopen(*v, RTLD_NOW | RTLD_NOLOAD);
        if (h) { cached.store(h, std::memory_order_release); return h; }
    }
    return nullptr;
}

// Shim-overhead instrumentation (Change 2 gate measurement).
// Tracks TSC cycles spent inside the shim, excluding the real CUDA launch.
// Exported accessors let the gate test compute per-launch CIPHER overhead
// independently of driver noise.
static std::atomic<uint64_t> g_shim_tsc_total{0};
static std::atomic<uint64_t> g_shim_tsc_calls{0};
extern "C" __attribute__((visibility("default")))
uint64_t cipher_shim_tsc_total(void) {
    return g_shim_tsc_total.load(std::memory_order_relaxed);
}
extern "C" __attribute__((visibility("default")))
uint64_t cipher_shim_tsc_calls(void) {
    return g_shim_tsc_calls.load(std::memory_order_relaxed);
}
extern "C" __attribute__((visibility("default")))
void cipher_shim_tsc_reset(void) {
    g_shim_tsc_total.store(0, std::memory_order_relaxed);
    g_shim_tsc_calls.store(0, std::memory_order_relaxed);
}

// ── Minimal CUDA types ──────────────────────────────────────────────────────
typedef int      CUresult;
typedef void*    CUfunction;
typedef void*    CUstream;
typedef void*    cudaStream_t;
typedef int      cudaError_t;
typedef int      CUdriverProcAddressQueryResult;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_FOUND 500

struct dim3 { unsigned x, y, z; };

// cudaLaunchConfig_t — ABI-compatible minimal definition
struct cudaLaunchConfig_t {
    dim3     gridDim;
    dim3     blockDim;
    size_t   dynamicSmemBytes;
    void*    stream;
    void*    attrs;
    unsigned numAttrs;
};

// ── Function pointer types ──────────────────────────────────────────────────
// ── B.6''.9.8.4 forward typedefs + state ────────────────────────────────────
typedef CUresult     (*cuGraphLaunch_fn)(void* /*CUgraphExec*/, void* /*CUstream*/);
typedef CUresult     (*cuGraphLaunchPipelined_fn)(void* /*CUgraphExec*/, void* /*hStream array*/, void* /*CUstream*/);
typedef cudaError_t  (*cudaGraphLaunch_fn)(void* /*cudaGraphExec_t*/, void* /*cudaStream_t*/);

extern "C" __attribute__((visibility("default"))) CUresult    cuGraphLaunch(void* exec, void* stream);
extern "C" __attribute__((visibility("default"))) CUresult    cuGraphLaunch_ptsz(void* exec, void* stream);
extern "C" __attribute__((visibility("default"))) CUresult    cuGraphLaunchPipelined(void* exec, void* phStreamArray, void* stream);
extern "C" __attribute__((visibility("default"))) cudaError_t cudaGraphLaunch(void* exec, void* stream);
extern "C" __attribute__((visibility("default"))) cudaError_t cudaGraphLaunch_ptsz(void* exec, void* stream);
extern "C" __attribute__((visibility("default"))) uint64_t    cipher_rt_graph_launch_calls(void);

static cuGraphLaunch_fn           g_real_cuGraphLaunch           = nullptr;
static cuGraphLaunch_fn           g_real_cuGraphLaunch_ptsz      = nullptr;
static cuGraphLaunchPipelined_fn  g_real_cuGraphLaunchPipelined  = nullptr;
static cudaGraphLaunch_fn         g_real_cudaGraphLaunch         = nullptr;
static cudaGraphLaunch_fn         g_real_cudaGraphLaunch_ptsz    = nullptr;
static std::atomic<uint64_t>      g_graph_launch_calls{0};

// ── B.6''.9.8.5 forward typedefs + state ────────────────────────────────────
// CUDA_KERNEL_NODE_PARAMS layout mirror — same v1 layout used in
// cipher_graph_inspect.cpp. Forward-compatible (leading fields stable
// across cuGraphAddKernelNode and cuGraphAddKernelNode_v2 versions).
struct CipherKernelNodeParamsShim {
    void*     func;
    unsigned  gridDimX, gridDimY, gridDimZ;
    unsigned  blockDimX, blockDimY, blockDimZ;
    unsigned  sharedMemBytes;
    void**    kernelParams;
    void**    extra;
    // v2 extension (CUDA 12+ may add kern handle field; v1 4-arg path
    // is what we mutate). Substitution is gated to func swap only —
    // grid/block/params untouched per ABI-compat discipline.
};

typedef CUresult (*cuGraphAddKernelNode_fn)(
    void* /*CUgraphNode*/, void* /*CUgraph*/, const void* /*deps*/, size_t /*depCount*/,
    CipherKernelNodeParamsShim* /*params*/);
typedef CUresult (*cuGraphExecKernelNodeSetParams_fn)(
    void* /*CUgraphExec*/, void* /*CUgraphNode*/,
    const CipherKernelNodeParamsShim* /*params*/);
typedef cudaError_t (*cudaGraphAddKernelNode_fn)(
    void* /*node*/, void* /*graph*/, const void* /*deps*/, size_t /*depCount*/,
    const CipherKernelNodeParamsShim* /*params*/);

extern "C" __attribute__((visibility("default"))) CUresult cuGraphAddKernelNode(
    void* node, void* graph, const void* deps, size_t depCount,
    CipherKernelNodeParamsShim* params);
extern "C" __attribute__((visibility("default"))) CUresult cuGraphAddKernelNode_v2(
    void* node, void* graph, const void* deps, size_t depCount,
    CipherKernelNodeParamsShim* params);
extern "C" __attribute__((visibility("default"))) CUresult cuGraphExecKernelNodeSetParams(
    void* exec, void* node, const CipherKernelNodeParamsShim* params);
extern "C" __attribute__((visibility("default"))) CUresult cuGraphExecKernelNodeSetParams_v2(
    void* exec, void* node, const CipherKernelNodeParamsShim* params);
extern "C" __attribute__((visibility("default"))) cudaError_t cudaGraphAddKernelNode(
    void* node, void* graph, const void* deps, size_t depCount,
    const CipherKernelNodeParamsShim* params);
extern "C" __attribute__((visibility("default"))) uint64_t cipher_rt_graph_add_kernel_calls(void);
extern "C" __attribute__((visibility("default"))) uint64_t cipher_rt_graph_node_substitutions(void);

static cuGraphAddKernelNode_fn             g_real_cuGraphAddKernelNode       = nullptr;
static cuGraphAddKernelNode_fn             g_real_cuGraphAddKernelNode_v2    = nullptr;
static cuGraphExecKernelNodeSetParams_fn   g_real_cuGraphExecKernelNodeSetParams    = nullptr;
static cuGraphExecKernelNodeSetParams_fn   g_real_cuGraphExecKernelNodeSetParams_v2 = nullptr;
static cudaGraphAddKernelNode_fn           g_real_cudaGraphAddKernelNode     = nullptr;
static std::atomic<uint64_t>               g_graph_add_kernel_calls{0};
static std::atomic<uint64_t>               g_graph_node_substitutions{0};

// Weak hook into actuator substitution decision. May13 actuators (Marlin,
// Koopman) wire this if they want to participate in graph-capture-time
// substitution. Returns substitute CUfunction or NULL for no substitution.
// Signature is intentionally minimal — substitute MUST be ABI-compatible
// (same grid/block dims accepted, same kernelParams layout).
extern "C" __attribute__((weak))
void* cipher_rt_graph_substitute_decide(void* original_func,
                                        unsigned gx, unsigned gy, unsigned gz,
                                        unsigned bx, unsigned by, unsigned bz,
                                        unsigned shared);

// ── B.6''.9.8.5b cuStreamGetCaptureInfo capture-detection ──────────────────
//
// PyTorch CUDAGraph uses cuStreamBeginCapture + cuLaunchKernel-during-capture
// rather than explicit cuGraphAddKernelNode. To intercept substitution at
// the right boundary on the PyTorch path, the cuLaunchKernel shim must
// detect capture mode via cuStreamGetCaptureInfo and route substitution
// through cipher_rt_graph_substitute_decide BEFORE the real launch call.
//
// CUDA driver value: CU_STREAM_CAPTURE_STATUS_ACTIVE = 1.
typedef CUresult (*cuStreamGetCaptureInfo_fn)(void* /*CUstream*/,
                                              int* /*status*/,
                                              unsigned long long* /*id*/);
static cuStreamGetCaptureInfo_fn g_real_cuStreamGetCaptureInfo = nullptr;

// cipher_rt_capture_check_and_substitute is defined AFTER tls_shape_valid /
// tls_launch_valid file-statics (search "B.6''.9.8.5b helper definition").

// Weak refs to existing cipher_rt_phase4 substrate APIs. If those translation
// units are linked into libcipher_rt.so, the references resolve; otherwise
// the shim degrades to passthrough + counter only. Framework-agnostic.
extern "C" __attribute__((weak)) uint32_t cipher_v2_current_tenant_id_from_stream(uintptr_t);
extern "C" __attribute__((weak)) uint64_t cipher_rt_commit_observe_and_publish(uint32_t);

typedef CUresult    (*cuLaunchKernel_fn)(CUfunction, unsigned, unsigned, unsigned,
                     unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
typedef CUresult    (*cuLaunchKernelEx_fn)(void*, CUfunction, void**, void**);
typedef cudaError_t (*cudaLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*cudaLaunchKernelExC_fn)(const cudaLaunchConfig_t*, const void*, void**);
typedef CUresult    (*cuGetProcAddress_fn)(const char*, void**, int, uint64_t, CUdriverProcAddressQueryResult*);

// ── Real function pointers ──────────────────────────────────────────────────
static cuLaunchKernel_fn      g_real_cuLaunch           = nullptr;
static cuLaunchKernel_fn      g_real_cuLaunch_ptsz      = nullptr;
static cuLaunchKernelEx_fn    g_real_cuLaunchEx         = nullptr;
static cuLaunchKernelEx_fn    g_real_cuLaunchEx_ptsz    = nullptr;
static cudaLaunchKernel_fn    g_real_cudaLaunch         = nullptr;
static cudaLaunchKernel_fn    g_real_cudaLaunch_ptsz    = nullptr;
// torch 2.11 / cudart 13: nvcc-generated host stubs route through these
// double-underscore entry points instead of the public `cudaLaunchKernel`.
// Same signature; we shim them with bodies identical to cudaLaunchKernel*.
static cudaLaunchKernel_fn    g_real___cudaLaunch       = nullptr;
static cudaLaunchKernel_fn    g_real___cudaLaunch_ptsz  = nullptr;
static cudaLaunchKernelExC_fn g_real_cudaLaunchExC      = nullptr;
static cudaLaunchKernelExC_fn g_real_cudaLaunchExC_ptsz = nullptr;
static cuGetProcAddress_fn    g_real_cuGetProc          = nullptr;
static cuGetProcAddress_fn    g_real_cuGetProc_v2       = nullptr;

typedef void** (*cudaRegisterFatBinary_fn)(void*);
typedef void   (*cudaRegisterFatBinaryEnd_fn)(void**);
typedef void   (*cudaRegisterFunction_fn)(void**, const char*, char*, const char*,
                                            int, void*, void*, void*, void*, int*);
static cudaRegisterFatBinary_fn    g_real_RegisterFatBinary    = nullptr;
static cudaRegisterFatBinaryEnd_fn g_real_RegisterFatBinaryEnd = nullptr;
static cudaRegisterFunction_fn     g_real_RegisterFunction     = nullptr;

// Driver-level cubin loaders (libcuda.so.1) — these receive the
// DECOMPRESSED cubin blob from cudart_static, so we can parse cubin
// metadata directly without intercepting __cudaRegister*.
typedef CUresult (*cuModuleLoadData_fn)(void**, const void*);
typedef CUresult (*cuModuleLoadDataEx_fn)(void**, const void*,
                                            unsigned, int*, void**);
typedef CUresult (*cuModuleGetFunction_fn)(void**, void*, const char*);
static cuModuleLoadData_fn     g_real_cuModuleLoadData    = nullptr;
static cuModuleLoadDataEx_fn   g_real_cuModuleLoadDataEx  = nullptr;
static cuModuleGetFunction_fn  g_real_cuModuleGetFunction = nullptr;

// CUDA 12 'Library' API — what static cudart 12 actually uses.
typedef CUresult (*cuLibraryLoadData_fn)(void**, const void*,
                                            int*, void**, unsigned int,
                                            int*, void**, unsigned int);
typedef CUresult (*cuLibraryGetKernel_fn)(void**, void*, const char*);
typedef CUresult (*cuKernelGetFunction_fn)(void**, void*);
typedef CUresult (*cuKernelGetName_fn)(const char**, void*);
typedef CUresult (*cuKernelGetParamInfo_fn)(void*, size_t, size_t*, size_t*);
static cuLibraryLoadData_fn    g_real_cuLibraryLoadData    = nullptr;
static cuLibraryGetKernel_fn   g_real_cuLibraryGetKernel   = nullptr;
static cuKernelGetFunction_fn  g_real_cuKernelGetFunction  = nullptr;
static cuKernelGetName_fn      g_real_cuKernelGetName      = nullptr;
static cuKernelGetParamInfo_fn g_real_cuKernelGetParamInfo = nullptr;

static std::atomic<uint64_t> g_intercept_count{0};
std::atomic<int> g_fa_call_count{0};   // global, exposed to KV-compress probe
static std::atomic<uint64_t> g_proc_addr_count{0};
static std::atomic<uint64_t> g_got_patches{0};
static std::atomic<bool>     g_resolved{false};
static std::atomic<bool>     g_auto_patched{false};
static long                  g_page_size = 4096;

// Real glibc dlsym — saved from GOT before patching. Used by ensure_cublasLt
// and cipher_substitute_sgemm to avoid recursion through our dlsym wrapper.
static void* (*g_real_dlsym)(void*, const char*) = nullptr;

// ── Thread-local GEMM shape + pointer storage ───────────────────────────────
// Written by cublasGemmEx / cublasLtMatmul shims, read by cipher_dispatch.
static __thread int   tls_M = 0, tls_N = 0, tls_K = 0;
static __thread int   tls_shape_valid = 0;
static __thread const void* tls_A = nullptr;
static __thread const void* tls_B = nullptr;
static __thread void*       tls_C = nullptr;
static __thread const void* tls_alpha = nullptr;
static __thread const void* tls_beta  = nullptr;
static __thread void*       tls_user_stream = nullptr;
static __thread int   tls_Atype = 0, tls_Btype = 0, tls_Ctype = 0;
static __thread int   tls_lda = 0, tls_ldb = 0, tls_ldc = 0;
static __thread int   tls_computeType = 0;

// Export TLS accessors for libcipher_rt.so (called from cipher_dispatch)
extern "C" __attribute__((visibility("default")))
void cipher_tls_get_gemm_shape(int* m, int* n, int* k, int* valid) {
    *m = tls_M; *n = tls_N; *k = tls_K; *valid = tls_shape_valid;
}

extern "C" __attribute__((visibility("default")))
void cipher_tls_get_gemm_ptrs(const void** A, const void** B, void** C,
                               const void** alpha, const void** beta,
                               void** stream, int* valid) {
    *A = tls_A; *B = tls_B; *C = tls_C;
    *alpha = tls_alpha; *beta = tls_beta;
    *stream = tls_user_stream;
    *valid = tls_shape_valid;
}

extern "C" __attribute__((visibility("default")))
void cipher_tls_get_gemm_types(int* Atype, int* Btype, int* Ctype) {
    *Atype = tls_Atype; *Btype = tls_Btype; *Ctype = tls_Ctype;
}

// cuLaunchKernelEx args — stored so apply_recipe can re-launch the kernel
static __thread void*  tls_launch_config = nullptr;
static __thread void*  tls_launch_fn     = nullptr;
static __thread void** tls_launch_params = nullptr;
static __thread void** tls_launch_extra  = nullptr;
static __thread int    tls_launch_valid  = 0;

// Recursion guard — prevent dispatch loop when apply_recipe calls cuLaunchKernelEx
static __thread int tls_in_recipe = 0;

// ── B.6''.9.8.5b helper definition ──────────────────────────────────────────
// Defined AFTER tls_shape_valid + tls_launch_valid above.
//
// Gated by env CIPHER_GRAPH_CAPTURE_DETECT=1 (default OFF). Honest residue:
// cuStreamGetCaptureInfo SEGVs when invoked from inside cuBLAS-internal
// cuLaunchKernelEx during GEMM dispatch (stack: cuStreamGetCaptureInfo_v2 <
// cuLaunchKernelEx < cublasLtHSHMatmul < cublasGemmEx). The tls_shape_valid
// /tls_launch_valid markers are set by may13's own cuBLAS shims which we
// dropped from g_patches[] per .8.1 conflict avoidance; cipher_rt_phase4's
// canonical cuBLAS shim lives in a different TU and cannot set may13's
// file-static TLS. v1.x carry: implement per-stream capture-state cache
// populated by cuStreamBeginCapture hook (no per-launch driver call needed)
// OR add cross-TU shared TLS marker via cipher_v2_internal.h.
//
// Until v1.x fix, gate is OFF by default for safety. Actuators that want
// capture-time substitution set CIPHER_GRAPH_CAPTURE_DETECT=1 in their
// startup; users opt in at their own discretion.
static inline void* cipher_rt_capture_check_and_substitute(
    void* f, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, void* stream)
{
    // Fast path: opt-in env not set → no-op (default for v1; SEGV-safe)
    static int s_enabled = -1;
    if (s_enabled == -1) {
        const char* v = getenv("CIPHER_GRAPH_CAPTURE_DETECT");
        s_enabled = (v && *v && v[0] != '0') ? 1 : 0;
    }
    if (__builtin_expect(!s_enabled, 1)) return f;
    if (__builtin_expect(!cipher_rt_graph_substitute_decide, 1)) return f;
    if (__builtin_expect(!g_real_cuStreamGetCaptureInfo, 0)) return f;
    // Defense in depth: bail in cuBLAS-internal flow (only catches calls
    // routed through may13's own dropped cuBLAS shims; cipher_rt_phase4's
    // canonical cuBLAS shim path is NOT caught here — see v1.x carry note).
    if (tls_shape_valid || tls_launch_valid) return f;
    int status = 0;
    unsigned long long unused_id = 0;
    CUresult r = g_real_cuStreamGetCaptureInfo(stream, &status, &unused_id);
    if (r != CUDA_SUCCESS || status != 1 /* CU_STREAM_CAPTURE_STATUS_ACTIVE */) return f;
    g_graph_add_kernel_calls.fetch_add(1, std::memory_order_relaxed);
    void* sub = cipher_rt_graph_substitute_decide(f, gx, gy, gz, bx, by, bz, shared);
    if (sub && sub != f) {
        g_graph_node_substitutions.fetch_add(1, std::memory_order_relaxed);
        return sub;
    }
    return f;
}

// Stage 3: lazy resolution of cross-DSO calls into libcipher_rt.so.
typedef void (*predict_observe_ptr_fn)(void*, unsigned long);
typedef int  (*persist_enabled_fn)(void);
typedef int  (*persist_apply_fn)(void*);
typedef int  (*persist_top_window_fn)(void*);    // raw 32-byte buf
typedef int  (*wcomp_observe_fn)(void*, unsigned long);
static std::atomic<predict_observe_ptr_fn> g_predict_observe_ptr_fn{nullptr};
static std::atomic<persist_enabled_fn>     g_persist_enabled_fn{nullptr};
static std::atomic<persist_apply_fn>       g_persist_apply_fn{nullptr};
static std::atomic<persist_top_window_fn>  g_persist_top_window_fn{nullptr};
static std::atomic<wcomp_observe_fn>       g_wcomp_observe_fn{nullptr};
static std::atomic<int>                    g_stage3_resolved{0};

// Track 2 — kernel name recognition table. On first observation of each
// unique CUfunction, call cuFuncGetName and log it. Bounded LRU.
struct KnownKernel {
    CUfunction              fn;
    char                    name[128];
    uint8_t                 op_class;     // mapped from name pattern
    std::atomic<uint64_t>   count;
};
constexpr unsigned KK_SIZE = 1024;
static KnownKernel g_kk_table[KK_SIZE]{};
static std::atomic<unsigned> g_kk_used{0};
static std::atomic<int>      g_kk_log_enabled{-1};

typedef int (*pf_cuFuncGetName)(const char**, CUfunction);
static pf_cuFuncGetName g_cuFuncGetName2 = nullptr;
static std::atomic<int> g_cuFuncGetName2_resolved{0};

static const char* resolve_func_name(CUfunction f) {
    if (!g_cuFuncGetName2_resolved.load(std::memory_order_acquire)) {
        void* lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!lc) lc = dlopen("libcuda.so.1", RTLD_LAZY);
        if (lc) g_cuFuncGetName2 = (pf_cuFuncGetName)dlsym(lc, "cuFuncGetName");
        g_cuFuncGetName2_resolved.store(1, std::memory_order_release);
    }
    if (g_cuFuncGetName2) {
        const char* nm = nullptr;
        if (g_cuFuncGetName2(&nm, f) == 0 && nm && nm[0]) return nm;
    }
    // Fallback: dladdr the function pointer. PyTorch's runtime-API kernels
    // are registered with hostFun pointing into libtorch_cuda.so's symbol
    // table; dladdr returns that symbol's mangled name.
    Dl_info info{};
    if (dladdr((const void*)f, &info) && info.dli_sname) return info.dli_sname;
    return nullptr;
}

// Map mangled C++ name → coarse op class. Searches for stable identifier
// substrings produced by Itanium-ABI mangling of PyTorch / cuBLAS / FA names.
// Ordered most specific first.
static uint8_t classify_kernel_name(const char* name) {
    if (!name) return 0xFF;
    // FlashAttention (PyTorch's bundled FlashAttention2 / 3)
    if (strstr(name, "flash_fwd") || strstr(name, "flash_bwd")
        || strstr(name, "flashattn") || strstr(name, "fmha")
        || strstr(name, "flash_attention")) return 7;
    // GEMM-class
    if (strstr(name, "splitKreduce") || strstr(name, "splitkreduce")) return 0;   // cuBLAS GEMM output reduction
    if (strstr(name, "wgmma") || strstr(name, "gemm") || strstr(name, "GEMM")
        || strstr(name, "matmul") || strstr(name, "cutlass")) return 0;
    if (strstr(name, "_sgemm") || strstr(name, "_hgemm") || strstr(name, "_dgemm")) return 0;
    // RMSNorm / LayerNorm
    if (strstr(name, "rms_norm") || strstr(name, "RMSNorm") || strstr(name, "rmsnorm")) return 1;
    if (strstr(name, "layer_norm") || strstr(name, "LayerNorm") || strstr(name, "layernorm")) return 2;
    // SiLU / GeLU activations
    if (strstr(name, "SiLUFunctor") || strstr(name, "silu") || strstr(name, "swish")
        || strstr(name, "Silu")) return 3;
    if (strstr(name, "GeLUFunctor") || strstr(name, "gelu") || strstr(name, "GELU")) return 4;
    // Softmax
    if (strstr(name, "softmax") || strstr(name, "Softmax") || strstr(name, "SoftMax")) return 5;
    // RoPE
    if (strstr(name, "rotary") || strstr(name, "rope") || strstr(name, "RoPE")) return 6;
    // Attention catch-all
    if (strstr(name, "attn") || strstr(name, "Attention")) return 8;
    // Embedding
    if (strstr(name, "embedding") || strstr(name, "Embedding") || strstr(name, "index_select")) return 9;
    // PyTorch elementwise kernel families — mangled names contain "vectorized_elementwise"
    // for binary, "unrolled_elementwise" for fallback, plain "elementwise_kernel" for 2-arg.
    if (strstr(name, "CUDAFunctor_add") || strstr(name, "AddFunctor")
        || strstr(name, "add_kernel") || strstr(name, "elementwise_add")
        || strstr(name, "residual")) return 11;
    if (strstr(name, "BinaryFunctor") || strstr(name, "MulFunctor")
        || strstr(name, "vectorized_elementwise")) return 10;
    // Reductions (mean/sum/max/etc.) used by RMSNorm and softmax
    if (strstr(name, "MeanOps") || strstr(name, "SumOps") || strstr(name, "MaxOps")
        || strstr(name, "ReduceOp") || strstr(name, "reduce_kernel")) return 14;
    if (strstr(name, "elementwise_kernel") || strstr(name, "unrolled_elementwise")) return 16;
    // Memcpy / copy / cast / transpose
    if (strstr(name, "copy_kernel") || strstr(name, "Copy") || strstr(name, "Memcpy")) return 12;
    if (strstr(name, "transpose") || strstr(name, "permute") || strstr(name, "Transpose")) return 13;
    if (strstr(name, "Cast") || strstr(name, "Convert") || strstr(name, "cast_kernel")
        || strstr(name, "FillFunctor")) return 15;
    return 0xFF;     // truly unknown
}

static const char* op_class_label(uint8_t c) {
    static const char* lbl[] = {
        "GEMM","RMSNorm","LayerNorm","SiLU","GeLU","Softmax","RoPE",
        "FlashAttn","AttnOther","Embedding","ElementMul","ResidualAdd",
        "Copy","Transpose","Reduce","Cast","ElemwiseGen"};
    if (c <= 16) return lbl[c];
    return "Unknown";
}

// Look up or insert a kernel. Returns pointer to entry (always valid until
// table fills); on table-full, returns nullptr. Thread-safe (atomic CAS).
static KnownKernel* track_kernel(CUfunction f) {
    // Linear hash with FNV mix on the function pointer.
    uint64_t h = (uint64_t)(uintptr_t)f;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    for (unsigned probe = 0; probe < 16; ++probe) {
        unsigned i = (unsigned)((h + probe) & (KK_SIZE - 1));
        KnownKernel& k = g_kk_table[i];
        CUfunction cur = k.fn;
        if (cur == f) {
            k.count.fetch_add(1, std::memory_order_relaxed);
            return &k;
        }
        if (cur == nullptr) {
            // Try to claim
            CUfunction expected = nullptr;
            if (__atomic_compare_exchange_n(&k.fn, &expected, f, false,
                                            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                const char* nm = resolve_func_name(f);
                if (nm) {
                    strncpy(k.name, nm, sizeof(k.name) - 1);
                    k.name[sizeof(k.name) - 1] = 0;
                } else {
                    snprintf(k.name, sizeof(k.name), "<%p>", (void*)f);
                }
                k.op_class = classify_kernel_name(k.name);
                k.count.store(1, std::memory_order_relaxed);
                g_kk_used.fetch_add(1, std::memory_order_relaxed);

                if (g_kk_log_enabled.load(std::memory_order_relaxed) < 0) {
                    g_kk_log_enabled.store(getenv("CIPHER_KERNEL_LOG") ? 1 : 0,
                                            std::memory_order_relaxed);
                }
                if (g_kk_log_enabled.load(std::memory_order_relaxed) == 1) {
                    fprintf(stderr,
                        "[CIPHER KERNEL] new fn=%p class=%s name=%s\n",
                        (void*)f, op_class_label(k.op_class), k.name);
                }
                return &k;
            }
            // CAS lost, recheck
            if (k.fn == f) {
                k.count.fetch_add(1, std::memory_order_relaxed);
                return &k;
            }
        }
    }
    return nullptr;
}

// Public dump for diagnostics.
extern "C" __attribute__((visibility("default")))
void cipher_kernel_table_report(void) {
    FILE* f = fopen("/tmp/cipher_kernel_table.json", "w");
    if (!f) return;
    fprintf(f, "{\n  \"kernels\": [\n");
    bool first = true;
    for (unsigned i = 0; i < KK_SIZE; ++i) {
        const KnownKernel& k = g_kk_table[i];
        if (k.fn == nullptr) continue;
        fprintf(f, "%s    {\"fn\":\"%p\",\"name\":\"%s\",\"class\":\"%s\",\"count\":%llu}",
                first ? "" : ",\n", (void*)k.fn, k.name,
                op_class_label(k.op_class),
                (unsigned long long)k.count.load(std::memory_order_relaxed));
        first = false;
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
}

static inline void stage3_resolve_once() {
    if (g_stage3_resolved.load(std::memory_order_acquire)) return;
    g_predict_observe_ptr_fn.store(
        (predict_observe_ptr_fn)dlsym(RTLD_DEFAULT, "cipher_predict_observe_ptr"),
        std::memory_order_release);
    g_persist_enabled_fn.store(
        (persist_enabled_fn)dlsym(RTLD_DEFAULT, "cipher_persist_engine_enabled"),
        std::memory_order_release);
    g_persist_apply_fn.store(
        (persist_apply_fn)dlsym(RTLD_DEFAULT, "cipher_persist_engine_apply_to_stream"),
        std::memory_order_release);
    g_persist_top_window_fn.store(
        (persist_top_window_fn)dlsym(RTLD_DEFAULT, "cipher_persist_engine_get_top_window_raw"),
        std::memory_order_release);
    g_wcomp_observe_fn.store(
        (wcomp_observe_fn)dlsym(RTLD_DEFAULT, "cipher_weight_compress_observe"),
        std::memory_order_release);
    g_stage3_resolved.store(1, std::memory_order_release);
}

// Stage 3 actuation: layout-stable shadow types for cuLaunchKernelEx
// attribute augmentation. ABI matches cuda.h CUlaunchAttribute /
// CUlaunchConfig (CUDA 12+).
struct CipherLaunchAttrShadow {
    int  id;        // CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
    char pad[4];
    char value[64]; // CUlaunchAttributeValue union — 32 bytes window into [0..31]
};
static constexpr int CIPHER_CU_LAUNCH_ATTR_ACCESS_POLICY_WINDOW = 1;

// Stream-level L2 persistence applier. Cheap when engine is OFF (one fn-ptr
// load + one indirect call returning 0). Dedupes per (thread, stream) to
// avoid hammering cudaStreamSetAttribute on every launch.
static inline void cipher_persist_maybe_apply(void* stream) {
    if (__builtin_expect(stream == nullptr, 0)) return;
    stage3_resolve_once();
    auto en_fn = g_persist_enabled_fn.load(std::memory_order_relaxed);
    if (__builtin_expect(!en_fn || !en_fn(), 1)) return;
    static thread_local void* last_applied[8] = {0};
    static thread_local int   last_idx = 0;
    for (int i = 0; i < 8; ++i) if (last_applied[i] == stream) return;
    auto ap_fn = g_persist_apply_fn.load(std::memory_order_relaxed);
    if (ap_fn && ap_fn(stream)) {
        last_applied[last_idx & 7] = stream;
        last_idx++;
    }
}

// Export: Python/C caller sets A,B,C,M,N,K before torch.mm
extern "C" __attribute__((visibility("default")))
void cipher_set_gemm_ptrs(int m, int n, int k,
                          const void* A, const void* B, void* C) {
    tls_M = m; tls_N = n; tls_K = k;
    tls_A = A; tls_B = B; tls_C = C;
    tls_shape_valid = 1;

    // Stage 3: feed pointer reuse data to PREDICT (no-op when PREDICT is OFF).
    stage3_resolve_once();
    auto fn = g_predict_observe_ptr_fn.load(std::memory_order_relaxed);
    if (fn) {
        unsigned long bytes_a = (unsigned long)m * (unsigned long)k * 2; // fp16 default
        unsigned long bytes_b = (unsigned long)k * (unsigned long)n * 2;
        unsigned long bytes_c = (unsigned long)m * (unsigned long)n * 2;
        if (A) fn((void*)A, bytes_a);
        if (B) fn((void*)B, bytes_b);
        if (C) fn((void*)C, bytes_c);
    }
}

extern "C" __attribute__((visibility("default")))
void cipher_clear_gemm_ptrs(void) {
    tls_shape_valid = 0;
}

// Export: let libcipher_rt.so re-launch the intercepted kernel
extern "C" __attribute__((visibility("default")))
int cipher_tls_relaunch(void) {
    if (!tls_launch_valid || !g_real_cuLaunchEx) return -1;
    tls_in_recipe = 1;
    CUresult r = g_real_cuLaunchEx(tls_launch_config,
                                    (CUfunction)tls_launch_fn,
                                    tls_launch_params,
                                    tls_launch_extra);
    tls_in_recipe = 0;
    return (int)r;
}

// ── cipher_substitute_sgemm — Op 3: direct cublasSgemm with TLS pointers ────
// Called from apply_recipe() in libcipher_rt.so.  Resolves cublasSgemm and
// cublasCreate via g_real_dlsym so we never recurse through our own wrappers.

typedef int (*cublasSgemm_fn)(void*, int, int, int, int, int,
    const float*, const float*, int, const float*, int,
    const float*, float*, int);
typedef int (*cublasCreate_fn)(void**);
typedef int (*cublasSetStream_fn)(void*, void*);
typedef int (*cudaGetStream_fn)(void**);

static void*             g_cublas_handle   = nullptr;
static cublasSgemm_fn    g_real_sgemm      = nullptr;
static cublasSetStream_fn g_cublasSetStream = nullptr;

// __libc_dlsym bypasses our dlsym wrapper entirely — guaranteed real symbols.
extern "C" void* __libc_dlsym(void*, const char*);

static void resolve_cublas() {
    if (g_real_sgemm) return;
    void* h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcublas.so.12", RTLD_NOW);
    if (!h) return;
    // dlvsym bypasses our GOT-patched dlsym — gets the real versioned symbol
    g_real_sgemm      = (cublasSgemm_fn)    dlvsym(h, "cublasSgemm_v2", "libcublas.so.12");
    g_cublasSetStream = (cublasSetStream_fn) dlvsym(h, "cublasSetStream_v2", "libcublas.so.12");
    auto cCreate      = (cublasCreate_fn)    dlvsym(h, "cublasCreate_v2", "libcublas.so.12");
    if (!g_real_sgemm)
        g_real_sgemm  = (cublasSgemm_fn)    dlsym(h, "cublasSgemm_v2");
    if (!g_cublasSetStream)
        g_cublasSetStream = (cublasSetStream_fn) dlsym(h, "cublasSetStream_v2");
    if (!cCreate)
        cCreate       = (cublasCreate_fn)    dlsym(h, "cublasCreate_v2");
    if (cCreate) cCreate(&g_cublas_handle);
}

extern "C" __attribute__((visibility("default")))
int cipher_substitute_sgemm(int M, int N, int K,
                            const void* A, const void* B, void* C)
{
    resolve_cublas();
    if (!g_real_sgemm || !g_cublas_handle) return -1;

    if (g_cublasSetStream)
        g_cublasSetStream(g_cublas_handle, nullptr);

    // cublasSgemm_v2: Row-major torch C = A*B → col-major C^T = B^T * A^T
    static const float alpha = 1.0f, beta = 0.0f;
    if (!g_real_sgemm) return -5;

    tls_in_recipe = 1;
    int rc = g_real_sgemm(g_cublas_handle,
        0 /*CUBLAS_OP_N*/, 0 /*CUBLAS_OP_N*/,
        N, M, K,
        &alpha,
        (const float*)B, N,
        (const float*)A, K,
        &beta,
        (float*)C, N);
    tls_in_recipe = 0;
    return rc;
}

// ── cublasGemmEx shim — captures M,N,K before cuBLAS launches the kernel ────

typedef int cublasStatus_t;
typedef void* cublasHandle_t;

typedef cublasStatus_t (*cublasGemmEx_fn)(
    cublasHandle_t handle, int transa, int transb,
    int m, int n, int k,
    const void* alpha, const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta, void* C, int Ctype, int ldc,
    int computeType, int algo);

static cublasGemmEx_fn g_real_cublasGemmEx = nullptr;

// Base implementation — versioned alias created below
extern "C" __attribute__((visibility("default")))
cublasStatus_t cipher_cublasGemmEx_impl(
    cublasHandle_t handle, int transa, int transb,
    int m, int n, int k,
    const void* alpha, const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta, void* C, int Ctype, int ldc,
    int computeType, int algo)
{
    if (__builtin_expect(!g_real_cublasGemmEx, 0)) {
        void* h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen("libcublas.so.12", RTLD_NOW);
        if (h) g_real_cublasGemmEx = (cublasGemmEx_fn)dlsym(h, "cublasGemmEx");
        if (!g_real_cublasGemmEx) return 15; // CUBLAS_STATUS_NOT_SUPPORTED
    }

    // Bug-3 hook-level fast-out (option A: process-level passive opt-in).
    // CIPHER's per-call shim bookkeeping (PREDICT observers, weight-compress
    // observe, attn-FSM, Koopman substitute lookup, FP8 gate, fairness, weight
    // share, ring emit) costs 2–5 µs per cublasGemmEx. For ~10 µs GEMMs in
    // small encoders (MiniLM, all-mpnet, etc.) that's a 20–50 % overhead.
    // Set CIPHER_HOOK_PASSIVE=1 in the process env to call the real cublas
    // path with zero CIPHER bookkeeping — the LD_PRELOAD hook stays loaded
    // (you can still toggle FP8/fusion via the actuator-level env vars on
    // other shim entry points), but this hot path is bypassed.
    static const bool s_passive = []() {
        const char* env = getenv("CIPHER_HOOK_PASSIVE");
        return env && *env && env[0] != '0';
    }();
    if (__builtin_expect(s_passive, 0)) {
        return g_real_cublasGemmEx(handle, transa, transb, m, n, k,
                                    alpha, A, Atype, lda,
                                    B, Btype, ldb,
                                    beta, C, Ctype, ldc,
                                    computeType, algo);
    }

    // Bug-3 hook-level fast-out (option B: shape-based auto-bypass).
    // For GEMMs where the shorter weight dimension is small enough that the
    // GEMM itself runs in <10 µs, the shim bookkeeping eats the kernel.
    // min(m,k) is the weight matrix's short axis (transA=T means weight is
    // m×k stored k×m). MiniLM: 384. all-mpnet-base: 768. Llama-3.x: 4096+.
    // Default 768 catches sentence-transformers without touching Llama.
    // Tunable via CIPHER_GEMM_BYPASS_MIN_DIM; set to 0 to disable.
    static const int s_bypass_dim = []() {
        const char* env = getenv("CIPHER_GEMM_BYPASS_MIN_DIM");
        if (!env || !*env) return 768;
        char* end = nullptr;
        long v = strtol(env, &end, 10);
        return (v >= 0) ? (int)v : 768;
    }();
    if (s_bypass_dim > 0 && m > 0 && k > 0
        && (m < s_bypass_dim || k < s_bypass_dim)) {
        return g_real_cublasGemmEx(handle, transa, transb, m, n, k,
                                    alpha, A, Atype, lda,
                                    B, Btype, ldb,
                                    beta, C, Ctype, ldc,
                                    computeType, algo);
    }
    // Store M,N,K in TLS — the upcoming cuLaunchKernelEx will read them
    tls_M = m; tls_N = n; tls_K = k;
    tls_A = A; tls_B = B; tls_C = C;
    tls_alpha = alpha; tls_beta = beta;
    tls_Atype = Atype; tls_Btype = Btype; tls_Ctype = Ctype;
    tls_lda = lda; tls_ldb = ldb; tls_ldc = ldc;
    tls_computeType = computeType;
    tls_shape_valid = 1;

    // Stage 3: feed pointer reuse data to PREDICT — the real workload path.
    // dtype-aware byte sizing: Atype/Btype/Ctype = 2 (CUDA_R_16F) → 2 bytes,
    // = 0 (CUDA_R_32F) → 4 bytes; default to 2 for fp16 since PyTorch fp16
    // is the dominant case here.
    stage3_resolve_once();
    auto pobs = g_predict_observe_ptr_fn.load(std::memory_order_relaxed);
    auto wobs = g_wcomp_observe_fn.load(std::memory_order_relaxed);
    if (pobs || wobs) {
        unsigned long es_a = (Atype == 0) ? 4UL : 2UL;
        unsigned long es_b = (Btype == 0) ? 4UL : 2UL;
        unsigned long es_c = (Ctype == 0) ? 4UL : 2UL;
        unsigned long bytes_a = (unsigned long)m * (unsigned long)k * es_a;
        unsigned long bytes_b = (unsigned long)k * (unsigned long)n * es_b;
        unsigned long bytes_c = (unsigned long)m * (unsigned long)n * es_c;
        if (pobs) {
            if (A) pobs((void*)A, bytes_a);
            if (B) pobs((void*)B, bytes_b);
            if (C) pobs((void*)C, bytes_c);
        }
        // Stage 7 wire: A is the weight (constant across batches), B is the
        // activation. Feed both A and B to the weight-compress observer so
        // it can promote whichever turns out to be stable.
        if (wobs) {
            if (A) wobs((void*)A, bytes_a);
            if (B) wobs((void*)B, bytes_b);
        }
    }

    // -- Change 3: Attention Koopman FSM hook (default OFF via env flag) ---
    // The FSM may SUPPRESS this GEMM (state enters SAW_QK) or may have just
    // replayed buffered calls and returned REVERT_AND_PASS (we run the real
    // GEMM normally). On PASS_THROUGH the shim continues unchanged. The FSM
    // never returns a result — it only decides whether we run the real call.
    //
    // We extract the stream bound to this cuBLAS handle via cublasGetStream
    // and pass it to the FSM so its cross-stream-abort rule has something
    // meaningful to compare. Resolved lazily via dlsym; cached across calls.
    if (cipher_attn_fsm_on_gemm) {
        static cublasSetStream_fn g_cublasGetStream = nullptr;
        static int                g_get_stream_resolved = 0;
        if (!g_get_stream_resolved) {
            void* h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
            if (!h) h = dlopen("libcublas.so.12", RTLD_NOW);
            if (h) {
                g_cublasGetStream = (cublasSetStream_fn)dlvsym(
                    h, "cublasGetStream_v2", "libcublas.so.12");
                if (!g_cublasGetStream)
                    g_cublasGetStream = (cublasSetStream_fn)dlsym(
                        h, "cublasGetStream_v2");
            }
            g_get_stream_resolved = 1;
        }
        void* handle_stream = nullptr;
        if (g_cublasGetStream) {
            g_cublasGetStream(handle, &handle_stream);
        }
        CipherAttnAction act = cipher_attn_fsm_on_gemm(
            (void*)handle, transa, transb, m, n, k,
            alpha, A, Atype, lda, B, Btype, ldb,
            beta, C, Ctype, ldc,
            computeType, algo, handle_stream);
        if (act == CIPHER_ATTN_SUPPRESS) {
            tls_shape_valid = 0;
            return (cublasStatus_t)0;   // CUBLAS_STATUS_SUCCESS; C is left
                                         // untouched per scratch-buffer model
        }
        // REVERT_AND_PASS or PASS_THROUGH: fall through and run the real GEMM
        (void)act;
    }

    // Driver-level Koopman substitute — fires in cublasGemmEx shim
    // Geometry-only: attempts substitute for any fp16 shape.
    // Falls through to real cuBLAS if Koopman kernel returns non-zero
    // (no calibrated matrices for this shape yet).
    {
        static int shape_log = 0;
        int log_cap = getenv("CIPHER_KERNEL_LOG") ? 800 : 20;
        if (Ctype == 2 && shape_log < log_cap) {
            fprintf(stderr, "[CIPHER GEMM-EX] #%d tA=%d tB=%d m=%d n=%d k=%d "
                "Atype=%d Btype=%d Ctype=%d A=%p B=%p C=%p\n",
                shape_log, transa, transb, m, n, k,
                Atype, Btype, Ctype, A, B, C);
            shape_log++;
        }
    }
    if (Ctype == 2 /* fp16 */ && m > 0 && k > 0 && n > 0) {
        typedef int (*fp16_launch_shape_fn)(const void*, void*, int, int, int);
        static fp16_launch_shape_fn s_launch = nullptr;
        if (!s_launch)
            s_launch = (fp16_launch_shape_fn)dlsym(RTLD_DEFAULT,
                                             "cipher_koopman_fp16_launch_shape");
        // cublasGemmEx: C(m×n) = A(m×k) @ B(k×n). B=activation, C=output.
        // Koopman sees: M=n rows, input_dim=k, output_dim=m.
        if (s_launch && s_launch(B, C, n, k, m) == 0) {
            tls_shape_valid = 0;
            static uint64_t s_sub_count = 0;
            s_sub_count++;
            if (s_sub_count <= 5 || (s_sub_count % 500) == 0)
                fprintf(stderr, "[O(1)-driver] M=%d K=%d N=%d count=%llu\n",
                    m, k, n, (unsigned long long)s_sub_count);
            return (cublasStatus_t)0;  // CUBLAS_STATUS_SUCCESS
        }
    }

    // ── Stage 13: FP8 compute substitution ─────────────────────────────
    // Dispatch rule for the full M=1 INT4 + M=2..512 FP8 + M>512 fp16 stack:
    //   * M=1 (n_cublas=1) is RESERVED for the INT4 GEMV path (which is
    //     wired in via the Python-level Int4Linear monkey-patch — it
    //     never reaches cublasGemmEx).  We gate FP8 to n_cublas >= 2 so
    //     the two paths compose cleanly when both are enabled.
    //   * M in [2, 512] -> FP8 cublasLtMatmul (this block).  Upper bound
    //     was 64 originally; raised to 512 to capture B=32/B=64 prefill
    //     where the GEMMs are compute-bound (FP8 wgmma = 1979 TOPS vs
    //     fp16 = 989 TFLOPS so the 2× TOPS advantage matters most here).
    //     Activation workspace is 8 MB which covers n_cublas*k_cublas up
    //     to 512*14336 ≈ 7.3 MB on the largest Mistral shape.
    //   * M > 512      -> fall through to fp16 cuBLAS below.
    // Other gates:
    //   * env CIPHER_FP8_COMPUTE=on (the engine self-disables otherwise)
    //   * fp16 inputs / fp16 output                  (Atype=Btype=Ctype=2)
    //   * standard PyTorch nn.Linear convention      (transa=T, transb=N)
    // First N hits at the same (A, m, k) reserve a slot via _observe; on
    // crossing the stability threshold we quantize once and route all
    // subsequent calls to cublasLtMatmul on the cuBLAS-bound stream.
    if (Atype == 2 && Btype == 2 && Ctype == 2
        && transa == /* CUBLAS_OP_T */ 1 && transb == /* CUBLAS_OP_N */ 0
        && m > 0 && n >= 2 && n <= 512 && k > 0 && A && B && C)
    {
        typedef int (*fp8_enabled_fn)(void);
        typedef int (*fp8_observe_fn)(void*, int, int);
        typedef int (*fp8_quant_fn)(void*, int, int, void*);
        typedef int (*fp8_matmul_fn)(void*, const void*, void*,
                                      int, int, int, void*);
        static fp8_enabled_fn s_enabled = nullptr;
        static fp8_observe_fn s_observe = nullptr;
        static fp8_quant_fn   s_quant   = nullptr;
        static fp8_matmul_fn  s_matmul  = nullptr;
        static int            s_resolved = 0;
        if (!s_resolved) {
            s_enabled = (fp8_enabled_fn)dlsym(RTLD_DEFAULT, "cipher_fp8_compute_enabled");
            s_observe = (fp8_observe_fn)dlsym(RTLD_DEFAULT, "cipher_fp8_compute_observe");
            s_quant   = (fp8_quant_fn)  dlsym(RTLD_DEFAULT, "cipher_fp8_compute_quantize_weight");
            s_matmul  = (fp8_matmul_fn) dlsym(RTLD_DEFAULT, "cipher_fp8_compute_matmul");
            s_resolved = 1;
        }
        if (s_enabled && s_observe && s_quant && s_matmul && s_enabled()) {
            static std::atomic<uint64_t> s_ge_attempts{0};
            uint64_t att = s_ge_attempts.fetch_add(1, std::memory_order_relaxed);
            // Resolve cuBLAS-bound stream once.
            static cublasSetStream_fn s_get_stream = nullptr;
            static int                s_get_stream_resolved = 0;
            if (!s_get_stream_resolved) {
                void* h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
                if (!h) h = dlopen("libcublas.so.12", RTLD_NOW);
                if (h) {
                    s_get_stream = (cublasSetStream_fn)dlvsym(
                        h, "cublasGetStream_v2", "libcublas.so.12");
                    if (!s_get_stream)
                        s_get_stream = (cublasSetStream_fn)dlsym(
                            h, "cublasGetStream_v2");
                }
                s_get_stream_resolved = 1;
            }
            void* gs = nullptr;
            if (s_get_stream) s_get_stream(handle, &gs);

            int hits = s_observe((void*)A, m, k);
            const int kStability = 2;
            if (att < 8 || (att % 200 == 0) || hits >= 2) {
                fprintf(stderr,
                    "[CIPHER FP8-GE-DBG] att#%llu hits=%d M=%d N=%d K=%d A=%p\n",
                    (unsigned long long)att, hits, m, n, k, A);
            }
            if (hits >= kStability) {
                // Quantize on first crossing (idempotent thereafter).
                int qrc = s_quant((void*)A, m, k, gs);
                int mrc = s_matmul((void*)A, B, C, m, n, k, gs);
                if (att < 8 || (att % 200 == 0) || hits >= 2) {
                    fprintf(stderr, "[CIPHER FP8-GE-DBG] att#%llu qrc=%d mrc=%d\n",
                        (unsigned long long)att, qrc, mrc);
                }
                // Run FP8 matmul.  Returns 1 on success.
                if (mrc == 1) {
                    tls_shape_valid = 0;
                    static uint64_t s_fp8_count = 0;
                    s_fp8_count++;
                    if (getenv("CIPHER_FP8_VERBOSE")
                        && (s_fp8_count <= 5 || (s_fp8_count % 1000) == 0))
                        fprintf(stderr,
                                "[CIPHER FP8] subst #%llu m=%d n=%d k=%d "
                                "A=%p B=%p C=%p\n",
                                (unsigned long long)s_fp8_count,
                                m, n, k, A, B, C);
                    return (cublasStatus_t)0;   // CUBLAS_STATUS_SUCCESS
                }
            }
        }
    }

    uint64_t t0_gemm = 0;
    {
        struct timespec _ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &_ts);
        t0_gemm = (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;
    }

    // ── Multi-tenant FAIRNESS yield (cross-process via /cipher_fairness shm).
    // Only the heaviest GEMM hitter gets throttled; light tenants run free.
    {
        typedef int  (*fair_yield_fn)(void);
        typedef void (*fair_record_fn)(void);
        static fair_yield_fn   s_yield  = nullptr;
        static fair_record_fn  s_record = nullptr;
        static int             s_resolved = 0;
        if (!s_resolved) {
            s_yield  = (fair_yield_fn)
                dlsym(RTLD_DEFAULT, "cipher_fairness_shm_should_yield");
            s_record = (fair_record_fn)
                dlsym(RTLD_DEFAULT, "cipher_fairness_shm_record_gemm");
            s_resolved = 1;
        }
        if (s_yield && s_yield()) {
            usleep(20);    // 20 us backpressure for the heavy tenant
            static uint64_t s_yielded = 0;
            if ((++s_yielded & 0x3FF) == 1) {
                fprintf(stderr,
                    "[CIPHER FAIR] yielded #%llu (heavy tenant)\n",
                    (unsigned long long)s_yielded);
            }
        }
        if (s_record) s_record();
    }

    // ── Multi-process WEIGHT SHARE: observe + substitute A with a shared
    // dev pointer when CIPHER_WEIGHT_SHARE=on and content_hash stable.
    const void* A_eff = A;
    if (Atype == 2 /* fp16 */ && A && m > 0 && k > 0) {
        typedef void  (*ws_observe_fn)(void*, size_t);
        typedef void* (*ws_lookup_fn)(void*);
        static ws_observe_fn s_ws_observe = nullptr;
        static ws_lookup_fn  s_ws_lookup  = nullptr;
        static int           s_ws_resolved = 0;
        if (!s_ws_resolved) {
            s_ws_observe = (ws_observe_fn)
                dlsym(RTLD_DEFAULT, "cipher_weight_share_observe");
            s_ws_lookup  = (ws_lookup_fn)
                dlsym(RTLD_DEFAULT, "cipher_weight_share_lookup");
            s_ws_resolved = 1;
        }
        size_t a_bytes = (size_t)m * (size_t)k * 2;
        if (s_ws_observe) s_ws_observe((void*)A, a_bytes);
        if (s_ws_lookup) {
            void* shared = s_ws_lookup((void*)A);
            if (shared) {
                A_eff = shared;
                static uint64_t s_subs = 0;
                if ((++s_subs & 0xFFFF) == 1) {
                    fprintf(stderr,
                        "[CIPHER WS] subst #%llu A=%p -> shared=%p (m=%d k=%d)\n",
                        (unsigned long long)s_subs, A, shared, m, k);
                }
            }
        }
    }

    cublasStatus_t ret = g_real_cublasGemmEx(handle, transa, transb,
        m, n, k, alpha, A_eff, Atype, lda, B, Btype, ldb,
        beta, C, Ctype, ldc, computeType, algo);

    // V3 KV redirect: after K_proj/V_proj GEMM completes, quant the output C
    // into the per-layer compressed cache. Detection by m == 1024 (Mistral's
    // num_kv_heads × head_dim). The redirect module ignores other shapes and
    // is a no-op when CIPHER_KV_RDR_V3 isn't set.
    if (ret == 0 && Ctype == 2 /* fp16 */ && m == 1024 && C) {
        typedef int (*kv_quant_fn)(void*, int, int, void*);
        static kv_quant_fn s_kv_q = nullptr;
        static int         s_kv_q_resolved = 0;
        if (!s_kv_q_resolved) {
            s_kv_q = (kv_quant_fn)dlsym(RTLD_DEFAULT, "cipher_kv_redirect_quant_kv");
            s_kv_q_resolved = 1;
        }
        if (s_kv_q) {
            // Stream — reuse the cached cublasGetStream resolver if available.
            void* gs = nullptr;
            static cublasSetStream_fn g_get_stream = nullptr;
            static int                g_get_stream_resolved = 0;
            if (!g_get_stream_resolved) {
                void* h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
                if (!h) h = dlopen("libcublas.so.12", RTLD_NOW);
                if (h) {
                    g_get_stream = (cublasSetStream_fn)dlvsym(
                        h, "cublasGetStream_v2", "libcublas.so.12");
                    if (!g_get_stream)
                        g_get_stream = (cublasSetStream_fn)dlsym(
                            h, "cublasGetStream_v2");
                }
                g_get_stream_resolved = 1;
            }
            if (g_get_stream) g_get_stream(handle, &gs);
            s_kv_q(C, m, n, gs);
        }
    }

    // Emit a single GEMM-class ring entry per cublasGemmEx call so observer
    // ops (SENSE/GUARD/PREDICT/RECEIPT/...) see one event per torch.mm.
    // Backing cuLaunchKernelEx calls inside cuBLAS are not ring-written
    // anymore (tls_shape_valid==1 short-circuits the per-launch path).
    {
        typedef void (*ring_write_fn)(uint8_t, float, uint64_t,
                                      uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t,
                                      uint64_t, uint64_t);
        static ring_write_fn s_ring_write = nullptr;
        static int s_ring_resolved = 0;
        if (!s_ring_resolved) {
            s_ring_write = (ring_write_fn)dlsym(RTLD_DEFAULT, "cipher_f1_ring_write");
            s_ring_resolved = 1;
        }
        if (s_ring_write && ret == 0) {
            uint64_t seq_now = g_intercept_count.load(std::memory_order_relaxed);
            // Synthetic grid/block dims so observers that compute work units
            // from grid*block volume (e.g. FAIRNESS, CARBON) get a
            // meaningful FLOP-volume proxy = M*K*N for this GEMM.
            uint32_t gx = (uint32_t)(m > 0 ? m : 1);
            uint32_t gy = (uint32_t)(n > 0 ? n : 1);
            uint32_t bx = (uint32_t)(k > 0 ? k : 1);
            s_ring_write(
                /* op_class */ 0 /* GEMM */,
                /* confidence */ 1.0f,
                seq_now,
                /* grid */ gx, gy, 1,
                /* block */ bx, 1, 1,
                (uint32_t)m, (uint32_t)k, (uint32_t)n,
                (uint64_t)(uintptr_t)handle,
                t0_gemm);
        }
    }

    // MFU telemetry — geometry only, O(1)
    if (ret == 0 && m > 0 && n > 0 && k > 0) {
        typedef void (*record_gemm_fn)(uint32_t, uint32_t, uint32_t, uint64_t);
        static record_gemm_fn s_record = nullptr;
        static int s_resolved = 0;
        if (!s_resolved) {
            s_record = (record_gemm_fn)dlsym(RTLD_DEFAULT, "cipher_telemetry_record_gemm");
            s_resolved = 1;
        }
        if (s_record) s_record((uint32_t)m, (uint32_t)n, (uint32_t)k, t0_gemm);
    }

    tls_shape_valid = 0;
    return ret;
}

// ── cublasLtMatmul shim — extracts M,N,K from matrix layout descriptors ─────

typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulAlgo_t;

typedef cublasStatus_t (*cublasLtMatmul_fn)(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    const void*, cublasLtMatrixLayout_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    void*, cublasLtMatrixLayout_t,
    const void*, void*, size_t, cudaStream_t);

typedef cublasStatus_t (*cublasLtLayoutGetAttr_fn)(
    cublasLtMatrixLayout_t, int, void*, size_t, size_t*);
typedef cublasStatus_t (*cublasLtDescGetAttr_fn)(
    cublasLtMatmulDesc_t, int, void*, size_t, size_t*);

static cublasLtMatmul_fn       g_real_cublasLtMatmul = nullptr;
static cublasLtLayoutGetAttr_fn g_cublasLtLayoutGetAttr = nullptr;
static cublasLtDescGetAttr_fn   g_cublasLtDescGetAttr = nullptr;

// Version-agnostic libcublasLt resolver. torch 2.7 bundles cublasLt.so.12 at
// site-packages/nvidia/cublas/lib/, torch 2.11 bundles cublasLt.so.13. Try
// already-loaded SONAMEs first (RTLD_NOLOAD), fall back to load attempts.
static void* resolve_cublasLt() {
    static std::atomic<void*> cached{nullptr};
    void* h = cached.load(std::memory_order_acquire);
    if (h) return h;
    static const char* const kVersions[] = {
        "libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so.11",
        "libcublasLt.so",    nullptr,
    };
    for (const char* const* v = kVersions; *v; ++v) {
        h = dlopen(*v, RTLD_NOW | RTLD_NOLOAD);
        if (h) { cached.store(h, std::memory_order_release); return h; }
    }
    return nullptr;
}

static void ensure_cublasLt() {
    if (g_real_cublasLtMatmul && g_cublasLtLayoutGetAttr && g_cublasLtDescGetAttr) return;
    void* h = resolve_cublasLt();
    if (h) {
        auto rsym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
        if (!g_real_cublasLtMatmul)
            g_real_cublasLtMatmul = (cublasLtMatmul_fn)rsym(h, "cublasLtMatmul");
        if (!g_cublasLtLayoutGetAttr)
            g_cublasLtLayoutGetAttr = (cublasLtLayoutGetAttr_fn)rsym(h, "cublasLtMatrixLayoutGetAttribute");
        if (!g_cublasLtDescGetAttr)
            g_cublasLtDescGetAttr = (cublasLtDescGetAttr_fn)rsym(h, "cublasLtMatmulDescGetAttribute");
    }
}

// B.6''.9.8.1: PLT-exported cublasLtMatmul alias DROPPED. cipher_rt_phase4's
// cipher_rt_cublaslt_variants.c + dlsym_hook.c is canonical for cuBLAS-Lt.
// The cipher_cublasLtMatmul_impl function above is retained as dead code
// (not in g_patches[]; not aliased; unreachable). Originally:
//   extern "C" __attribute__((visibility("default"),
//     alias("cipher_cublasLtMatmul_impl"))) cublasStatus_t cublasLtMatmul(...)

extern "C" __attribute__((visibility("default")))
cublasStatus_t cipher_cublasLtMatmul_impl(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc,
    const void* B, cublasLtMatrixLayout_t Bdesc,
    const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc,
    void* D, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t algo, void* workspace,
    size_t workspaceSizeInBytes, cudaStream_t stream)
{
    // Bug-3 hook-level fast-out (passive opt-in). cublasLtMatmul is the
    // dominant matmul path for many encoder workloads (BERT-style models
    // including sentence-transformers' MiniLM use cublasLt directly via
    // PyTorch). Skip all CIPHER bookkeeping when CIPHER_HOOK_PASSIVE=1.
    static const bool s_lt_passive = []() {
        const char* env = getenv("CIPHER_HOOK_PASSIVE");
        return env && *env && env[0] != '0';
    }();
    if (__builtin_expect(s_lt_passive, 0)) {
        ensure_cublasLt();
        if (!g_real_cublasLtMatmul) return 15;
        return g_real_cublasLtMatmul(lightHandle, computeDesc,
            alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
            algo, workspace, workspaceSizeInBytes, stream);
    }

    static std::atomic<uint64_t> s_lt_call_count{0};
    uint64_t call_n = s_lt_call_count.fetch_add(1, std::memory_order_relaxed);
    if (call_n < 5) {
        fprintf(stderr,
            "[CIPHER LT-CALL] #%llu cipher_cublasLtMatmul_impl ENTERED\n",
            (unsigned long long)call_n);
    }
    ensure_cublasLt();
    if (!g_real_cublasLtMatmul) return 15;
    // Recursion guard: skip if called from within a recipe/relaunch
    static thread_local int in_lt = 0;
    if (in_lt || tls_in_recipe) return g_real_cublasLtMatmul(lightHandle, computeDesc,
        alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
        algo, workspace, workspaceSizeInBytes, stream);

    in_lt = 1;
    // Extract storage shape, transA/transB, and dtypes from descriptors.
    // Logical matmul dims depend on transA: with transA=T (PyTorch nn.Linear
    // convention) the matmul output rows = colsA, inner dim = rowsA.
    // CUBLASLT_MATRIX_LAYOUT_ROWS=2, COLS=3, TYPE=0
    // CUBLASLT_MATMUL_DESC_TRANSA=3, TRANSB=4
    int lt_transA = 0, lt_transB = 0;
    int lt_Atype = -1, lt_Btype = -1;
    if (g_cublasLtLayoutGetAttr) {
        uint64_t rowsA=0, colsA=0, colsB=0;
        size_t written = 0;
        g_cublasLtLayoutGetAttr(Adesc, 2, &rowsA, sizeof(rowsA), &written);
        g_cublasLtLayoutGetAttr(Adesc, 3, &colsA, sizeof(colsA), &written);
        g_cublasLtLayoutGetAttr(Bdesc, 3, &colsB, sizeof(colsB), &written);
        // Read transA / transB from the matmul descriptor when available.
        if (g_cublasLtDescGetAttr) {
            int32_t ta = 0, tb = 0;
            g_cublasLtDescGetAttr(computeDesc, 3, &ta, sizeof(ta), &written);
            g_cublasLtDescGetAttr(computeDesc, 4, &tb, sizeof(tb), &written);
            lt_transA = ta; lt_transB = tb;
        }
        // Query input dtypes from layouts (TYPE attr = 0).
        uint32_t aty = 0, bty = 0;
        g_cublasLtLayoutGetAttr(Adesc, 0, &aty, sizeof(aty), &written);
        g_cublasLtLayoutGetAttr(Bdesc, 0, &bty, sizeof(bty), &written);
        lt_Atype = (int)aty; lt_Btype = (int)bty;
        // Logical matmul dims (account for transA).
        if (lt_transA == /*CUBLAS_OP_T*/1) {
            tls_M = (int)colsA;
            tls_K = (int)rowsA;
        } else {
            tls_M = (int)rowsA;
            tls_K = (int)colsA;
        }
        tls_N = (int)colsB;
        tls_A = A; tls_B = B; tls_C = D;
        tls_alpha = alpha; tls_beta = beta;
        tls_user_stream = (void*)stream;
        tls_shape_valid = 1;
        // Query output dtype.
        uint32_t dtype_val = 0;
        g_cublasLtLayoutGetAttr(Ddesc, 0, &dtype_val, sizeof(dtype_val), &written);
        tls_Ctype = (int)dtype_val;
        tls_Atype = lt_Atype; tls_Btype = lt_Btype;
        static int lt_log_count = 0;
        if (lt_log_count < 3) {
            fprintf(stderr,
                "[CIPHER HOOK] cublasLtMatmul intercepted: M=%d N=%d K=%d "
                "transA=%d transB=%d Atype=%d Btype=%d Ctype=%d D=%p\n",
                tls_M, tls_N, tls_K, lt_transA, lt_transB,
                lt_Atype, lt_Btype, tls_Ctype, D);
            lt_log_count++;
        }
    }

    // ── Stage 13 (Lt path): FP8 compute substitution ─────────────────────
    // Mirrors the cublasGemmEx FP8 block. Same gates: fp16 in/out, transA=T
    // transB=N (nn.Linear convention), N in [2, 512]. cublasLt provides
    // A, B, C, D pointers and stream directly in the API — no struct walking.
    if (lt_Atype == 2 && lt_Btype == 2 && tls_Ctype == 2
        && lt_transA == /*CUBLAS_OP_T*/1 && lt_transB == /*CUBLAS_OP_N*/0
        && tls_M > 0 && tls_N >= 2 && tls_N <= 512 && tls_K > 0
        && A && B && D)
    {
        typedef int (*fp8_enabled_fn)(void);
        typedef int (*fp8_observe_fn)(void*, int, int);
        typedef int (*fp8_quant_fn)(void*, int, int, void*);
        typedef int (*fp8_matmul_fn)(void*, const void*, void*,
                                      int, int, int, void*);
        static fp8_enabled_fn s_enabled = nullptr;
        static fp8_observe_fn s_observe = nullptr;
        static fp8_quant_fn   s_quant   = nullptr;
        static fp8_matmul_fn  s_matmul  = nullptr;
        static int            s_resolved = 0;
        if (!s_resolved) {
            s_enabled = (fp8_enabled_fn)dlsym(RTLD_DEFAULT, "cipher_fp8_compute_enabled");
            s_observe = (fp8_observe_fn)dlsym(RTLD_DEFAULT, "cipher_fp8_compute_observe");
            s_quant   = (fp8_quant_fn)  dlsym(RTLD_DEFAULT, "cipher_fp8_compute_quantize_weight");
            s_matmul  = (fp8_matmul_fn) dlsym(RTLD_DEFAULT, "cipher_fp8_compute_matmul");
            s_resolved = 1;
        }
        if (s_enabled && s_observe && s_quant && s_matmul && s_enabled()) {
            int hits = s_observe((void*)A, tls_M, tls_K);
            const int kStability = 2;
            static std::atomic<uint64_t> s_lt_attempts{0};
            uint64_t att = s_lt_attempts.fetch_add(1, std::memory_order_relaxed);
            if (att < 8 || (att % 200 == 0) || hits >= 2) {
                fprintf(stderr,
                    "[CIPHER FP8-LT-DBG] att#%llu hits=%d M=%d N=%d K=%d A=%p\n",
                    (unsigned long long)att, hits, tls_M, tls_N, tls_K, A);
            }
            if (hits >= kStability) {
                int qrc = s_quant((void*)A, tls_M, tls_K, (void*)stream);
                int mrc = s_matmul((void*)A, B, D, tls_M, tls_N, tls_K, (void*)stream);
                if (att < 8 || (att % 200 == 0) || hits >= 2) {
                    fprintf(stderr,
                        "[CIPHER FP8-LT-DBG] att#%llu qrc=%d mrc=%d\n",
                        (unsigned long long)att, qrc, mrc);
                }
                if (mrc == 1) {
                    in_lt = 0;
                    tls_shape_valid = 0;
                    static uint64_t s_fp8_count = 0;
                    s_fp8_count++;
                    if (getenv("CIPHER_FP8_VERBOSE")
                        && (s_fp8_count <= 5 || (s_fp8_count % 1000) == 0))
                        fprintf(stderr,
                                "[CIPHER FP8-LT] subst #%llu m=%d n=%d k=%d "
                                "A=%p B=%p D=%p\n",
                                (unsigned long long)s_fp8_count,
                                tls_M, tls_N, tls_K, A, B, D);
                    return (cublasStatus_t)0;
                }
            }
        }
    }

    // Driver-level Koopman substitute — no Python, no hooks
    {
        static int dtype_log = 0;
        if (dtype_log < 3) {
            fprintf(stderr, "[CIPHER LT-SUB] M=%d K=%d N=%d Ctype=%d A=%p D=%p\n",
                tls_M, tls_K, tls_N, tls_Ctype, A, D);
            dtype_log++;
        }
    }
    if (tls_Ctype == 2 /* fp16 */ && tls_M > 0 && tls_K > 0 && tls_N > 0) {
        typedef int (*fp16_launch_shape_fn)(const void*, void*, int, int, int);
        static fp16_launch_shape_fn s_launch = nullptr;
        if (!s_launch)
            s_launch = (fp16_launch_shape_fn)dlsym(RTLD_DEFAULT,
                                             "cipher_koopman_fp16_launch_shape");
        // cublasLtMatmul: D(M×N) = A(M×K) @ B(K×N).
        // Koopman sees: M=tls_M rows, input_dim=tls_K, output_dim=tls_N.
        if (s_launch && s_launch(tls_A, tls_C, tls_M, tls_K, tls_N) == 0) {
            in_lt = 0;
            tls_shape_valid = 0;
            static uint64_t s_sub_count = 0;
            s_sub_count++;
            if (s_sub_count <= 5 || (s_sub_count % 500) == 0)
                fprintf(stderr, "[O(1)-driver] M=%d K=%d N=%d count=%llu\n",
                    tls_M, tls_K, tls_N, (unsigned long long)s_sub_count);
            return (cublasStatus_t)0;  // CUBLAS_STATUS_SUCCESS
        }
    }

    cublasStatus_t ret = g_real_cublasLtMatmul(lightHandle, computeDesc,
        alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
        algo, workspace, workspaceSizeInBytes, stream);

    in_lt = 0;
    // Block-level calibration: disabled in cublasLtMatmul path.
    // Collection handled by cipher_wrapper.py (dtype-safe).
    // cublasLtMatmul carries fp16/bf16 tensors that cannot be read as fp32.

    tls_shape_valid = 0;
    return ret;
}

// cublasLtMatmul versioned PLT interception deferred to a later session.
// For now, M/N/K come from geometry inference in gemm_shape_hash().

// ── ncclAllReduce shim — intercepts NCCL collectives for timing + overlap ───
// Geometry only: sees count, datatype, message size. No gradient semantics.

typedef int ncclResult_t_hook;
typedef void* ncclComm_t_hook;

typedef ncclResult_t_hook (*ncclAllReduce_fn)(
    const void*, void*, size_t, int, int, ncclComm_t_hook, void*);

static ncclAllReduce_fn g_real_ncclAllReduce = nullptr;

static void ensure_nccl() {
    if (g_real_ncclAllReduce) return;
    void* h = dlopen("libnccl.so.2", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libnccl.so.2", RTLD_NOW);
    if (h) {
        auto rsym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
        g_real_ncclAllReduce = (ncclAllReduce_fn)dlvsym(h, "ncclAllReduce", "libnccl.so.2");
        if (!g_real_ncclAllReduce)
            g_real_ncclAllReduce = (ncclAllReduce_fn)rsym(h, "ncclAllReduce");
    }
}

extern "C" __attribute__((visibility("default")))
ncclResult_t_hook cipher_ncclAllReduce_impl(
    const void* sendbuff, void* recvbuff, size_t count,
    int datatype, int op, ncclComm_t_hook comm, void* stream)
{
    ensure_nccl();
    if (!g_real_ncclAllReduce) return 1;  // ncclInternalError

    // Estimate bytes from count + datatype (geometry only, no gradient knowledge)
    // ncclDataType: 0=int8, 1=uint8, 2=int32, 3=uint32, 4=int64, 5=uint64,
    //               6=fp16, 7=fp32, 8=fp64, 9=bf16
    static const size_t dtype_size[] = {1,1,4,4,8,8,2,4,8,2};
    size_t elem_size = (datatype >= 0 && datatype <= 9) ? dtype_size[datatype] : 2;
    size_t bytes = count * elem_size;

    static int nccl_log = 0;
    if (nccl_log < 10) {
        fprintf(stderr, "[CIPHER HOOK] ncclAllReduce: count=%zu dtype=%d bytes=%zu\n",
                count, datatype, bytes);
        nccl_log++;
    }

    // Change 4 Part A — CfC policy DECIDE before the real collective.
    // Resolve both RT bridges lazily via dlsym; cached in function-static
    // variables so the lookup happens once per process.
    typedef int  (*nccl_decide_fn)(uint64_t, uint32_t);
    typedef void (*nccl_feedback_fn)(uint64_t, uint64_t, int);
    typedef void (*nccl_record_fn)(uint64_t, uint64_t);
    static nccl_decide_fn   s_decide   = nullptr;
    static nccl_feedback_fn s_feedback = nullptr;
    static nccl_record_fn   s_record   = nullptr;
    static int              s_resolved = 0;
    if (!s_resolved) {
        s_decide   = (nccl_decide_fn)  dlsym(RTLD_DEFAULT, "cipher_nccl_record_decide");
        s_feedback = (nccl_feedback_fn)dlsym(RTLD_DEFAULT, "cipher_nccl_record_feedback");
        s_record   = (nccl_record_fn)  dlsym(RTLD_DEFAULT, "cipher_nccl_record");
        s_resolved = 1;
    }

    // num_ranks is not exposed by the ncclAllReduce signature; a proper
    // query would call ncclCommCount(comm, &ranks), but we avoid taking
    // a hard dep on libnccl inside the hook DSO. 0 means "unknown" and
    // the CfC uses its default of 8.
    int chosen_algo = 4 /* NCCL_ALGO_AUTO */;
    if (s_decide) {
        chosen_algo = s_decide((uint64_t)bytes, 0u /* num_ranks unknown */);
    }

    // Time the real AllReduce call
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    ncclResult_t_hook ret = g_real_ncclAllReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream);

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    uint64_t dur_ns = (t1.tv_sec - t0.tv_sec) * 1000000000ULL
                    + (t1.tv_nsec - t0.tv_nsec);

    // Change 4 Part A — FEEDBACK: loop closed.
    if (s_feedback) s_feedback((uint64_t)bytes, dur_ns, chosen_algo);

    // Legacy liquid-state EMA update (kept for telemetry continuity).
    if (s_record) s_record(dur_ns, bytes);

    return ret;
}

// B.6''.9.8.1: PLT-exported ncclAllReduce alias DROPPED. NCCL is out of
// scope per V1 Phase B B.6''.9.8 scope. Originally:
//   extern "C" __attribute__((visibility("default"),
//     alias("cipher_ncclAllReduce_impl"))) ncclResult_t_hook ncclAllReduce(...)

// ── Runtime bridge to cipher_dispatch() in libcipher_rt.so ──────────────────
// Resolved lazily via dlsym — no link-time dependency.

// Mirror of CipherKernelDesc from cipher_intercept.h
struct HookKernelDesc {
    void*    fn;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t shared_bytes;
    void*    stream;
    void**   params;
    void**   extra;
    uint8_t  op_class;
    uint8_t  confidence;
    uint64_t intercept_ns;
};

// Mirror of CipherDispatchResult
enum { HOOK_PASS_THROUGH = 0, HOOK_SUBSTITUTED = 1, HOOK_DEFERRED = 2 };

typedef int (*cipher_dispatch_fn)(HookKernelDesc*);
static cipher_dispatch_fn g_cipher_dispatch = nullptr;
static std::atomic<int>   g_dispatch_probe{0};  // 0=untried, 1=found, -1=absent

static const char* op_class_name(uint8_t c) {
    switch (c) {
        case 0: return "GEMM";
        case 1: return "ATTENTION";
        case 2: return "CONVOLUTION";
        case 3: return "ELEMENTWISE";
        case 4: return "REDUCTION";
        case 5: return "MEMCPY_TRANSPOSE";
        case 6: return "ITERATIVE_CUSTOM";
        default: return "UNCLASSIFIED";
    }
}

static void try_resolve_dispatch() {
    if (g_dispatch_probe.load(std::memory_order_acquire) != 0) return;
    // CIPHER_NO_DISPATCH=1 bisect: keep g_cipher_dispatch NULL even if rt is
    // loaded. dispatch_and_log will early-exit to HOOK_PASS_THROUGH.
    const char* nd = getenv("CIPHER_NO_DISPATCH");
    if (nd && (nd[0] == '1' || nd[0] == 'o' || nd[0] == 'O')) {
        g_dispatch_probe.store(-1, std::memory_order_release);
        return;
    }
    // Look in the global symbol namespace (libcipher_rt.so must be loaded)
    void* sym = dlsym(RTLD_DEFAULT, "cipher_dispatch");
    if (sym) {
        g_cipher_dispatch = (cipher_dispatch_fn)sym;
        g_dispatch_probe.store(1, std::memory_order_release);
    } else {
        // Try explicit handle
        void* rt = dlopen("libcipher_rt.so", RTLD_NOW | RTLD_NOLOAD);
        if (rt) sym = dlsym(rt, "cipher_dispatch");
        if (sym) {
            g_cipher_dispatch = (cipher_dispatch_fn)sym;
            g_dispatch_probe.store(1, std::memory_order_release);
        } else {
            g_dispatch_probe.store(-1, std::memory_order_release);
        }
    }
}

// Weak extern: implemented in cipher_intercept.cpp (libcipher_rt.so)
// Writes ring entry for Stage 1 consumption. No kernel launch, no passthrough.
// Geometry args (gx..bz, M, K, N) plumbed so Stage 1 SENSE/SHIELD can perform
// real prefill/decode discrimination instead of class-count heuristics.
// (M, K, N) are 0 for non-GEMM kernels; grid_*/block_* are always populated.
extern "C" __attribute__((weak))
void cipher_f1_ring_write(uint8_t op_class, float confidence, uint64_t seq,
                          uint32_t gx, uint32_t gy, uint32_t gz,
                          uint32_t bx, uint32_t by, uint32_t bz,
                          uint32_t M, uint32_t K, uint32_t N,
                          uint64_t func_ptr_hash, uint64_t timestamp_ns);

static void maybe_collect_sample(uint8_t op_class); // forward decl

// Call cipher_dispatch and print classification. Returns the dispatch result.
static int dispatch_and_log(CUfunction f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void** params, void** extra)
{
    // No early exit by grid size — let CLASSIFY decide.
    // Elementwise ops (SiLU, GeLU, RMSNorm) have small grids but are substitutable.

    if (g_dispatch_probe.load(std::memory_order_acquire) == 0)
        try_resolve_dispatch();
    if (!g_cipher_dispatch) return HOOK_PASS_THROUGH;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    HookKernelDesc desc = {};
    desc.fn          = (void*)f;
    desc.grid_x      = gx; desc.grid_y = gy; desc.grid_z = gz;
    desc.block_x     = bx; desc.block_y = by; desc.block_z = bz;
    desc.shared_bytes = shared;
    desc.stream      = (void*)stream;
    desc.params      = params;
    desc.extra       = extra;
    desc.op_class    = 0xFF;
    desc.confidence  = 0;
    desc.intercept_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;

    // Fast path for cuBLAS-internal launches: when the parent cublasGemmEx
    // shim has already classified this GEMM and emitted its own ring entry
    // (tls_shape_valid==1), skip the entire dispatch path here. Recovers
    // ~11% TFLOPS on compute-bound GEMM.
    if (tls_shape_valid) return HOOK_PASS_THROUGH;
    int result = g_cipher_dispatch(&desc);

    uint64_t seq = g_intercept_count.load(std::memory_order_relaxed);

    // Verbose per-kernel logging — gated to first 30 + every 5000th to avoid
    // ~2us fprintf overhead on every GEMM dispatch (was the primary overhead source)
    if (seq <= 30 || (seq % 5000) == 0) {
        const char* decision = (result == HOOK_SUBSTITUTED) ? "PERMIT" :
                               (result == HOOK_DEFERRED)    ? "DEFER"  : "PASS";
        if (tls_shape_valid && desc.op_class == 0 /* GEMM */) {
            fprintf(stderr,
                "[CIPHER] kernel #%lu fn=%p class=%s conf=%u decision=%s shape=%dx%dx%d\n",
                seq, (void*)f, op_class_name(desc.op_class), desc.confidence, decision,
                tls_M, tls_N, tls_K);
        } else {
            fprintf(stderr,
                "[CIPHER] kernel #%lu fn=%p class=%s conf=%u decision=%s\n",
                seq, (void*)f, op_class_name(desc.op_class), desc.confidence, decision);
        }
    }

    // Collect EDMD sample after dispatch (tensor still alive)
    maybe_collect_sample(desc.op_class);

    // Feed F1 ring for Stage 1 (REMEMBER/VALIDATE/AUDIT/SPECULATE)
    if (cipher_f1_ring_write) {
        uint32_t Mx = tls_shape_valid ? (uint32_t)tls_M : 0u;
        uint32_t Kx = tls_shape_valid ? (uint32_t)tls_K : 0u;
        uint32_t Nx = tls_shape_valid ? (uint32_t)tls_N : 0u;
        cipher_f1_ring_write(
            desc.op_class, (float)desc.confidence / 100.0f, seq,
            gx, gy, gz, bx, by, bz,
            Mx, Kx, Nx,
            (uint64_t)(uintptr_t)f, desc.intercept_ns);
    }

    return result;
}

// ── Forward declarations ────────────────────────────────────────────────────
extern "C" {
    CUresult    cuInit(unsigned);
    CUresult    cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
    CUresult    cuLaunchKernel_ptsz(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
    CUresult    cuLaunchKernelEx(void*, CUfunction, void**, void**);
    CUresult    cuLaunchKernelEx_ptsz(void*, CUfunction, void**, void**);
    cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    cudaError_t cudaLaunchKernel_ptsz(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    cudaError_t __cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    cudaError_t __cudaLaunchKernel_ptsz(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t*, const void*, void**);
    cudaError_t cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t*, const void*, void**);
    CUresult    cuGetProcAddress(const char*, void**, int, uint64_t, CUdriverProcAddressQueryResult*);
    CUresult    cuGetProcAddress_v2(const char*, void**, int, uint64_t, CUdriverProcAddressQueryResult*);
    // Phase 1 — cudart fatbin/function registration
    void**      __cudaRegisterFatBinary(void* fatCubin);
    void        __cudaRegisterFatBinaryEnd(void** fatCubinHandle);
    void        __cudaRegisterFunction(void** fatCubinHandle,
                                       const char* hostFun,
                                       char*       deviceFun,
                                       const char* deviceName,
                                       int thread_limit, void*, void*, void*, void*, int*);
    // Phase 1 fix — driver-level loaders (decompressed cubin path)
    CUresult cuModuleLoadData(void** mod, const void* image);
    CUresult cuModuleLoadDataEx(void** mod, const void* image,
                                  unsigned numOpts, int* opts, void** optVals);
    CUresult cuModuleGetFunction(void** hfunc, void* hmod, const char* name);
    // CUDA 12 'Library' API — the path PyTorch's cudart_static actually uses.
    CUresult cuLibraryLoadData(void** lib, const void* image,
                                  int* jitOptions, void** jitOptValues, unsigned numJitOptions,
                                  int* libOptions, void** libOptValues, unsigned numLibOptions);
    CUresult cuLibraryGetKernel(void** ker, void* lib, const char* name);
    CUresult cuKernelGetFunction(void** hfunc, void* ker);
    void        cipher_repatch(void);
    uint64_t    cipher_intercept_count(void);
    // Provided by cipher_param_recovery (rt-side)
    void  cipher_param_register_fatbin(void* handle, const void* fatbin_blob);
    void  cipher_param_register_function(void* fat_handle, const void* host_fun,
                                          const char* device_fun, const char* device_name);
    int   cipher_param_ingest_image(const void* image);
    void  cipher_param_register_cufunc(const void* cufunc, const char* name);
}

// ── Phase 1 shims: __cudaRegisterFatBinary / __cudaRegisterFunction ─────────
//
// PyTorch (and any CUDA app) calls these at module-load time. We forward to
// the real cudart impl unchanged — only purpose is to record the host-fun /
// device-name / fatbin association into the cipher_param_recovery registry.

// Forward decls for the deferred GOT repatch we trigger from inside
// __cudaRegisterFatBinary once libtorch_cuda + its bundled cudart are loaded.
static void resolve_real();
static void auto_repatch_once();
static void auto_repatch_now();

static void resolve_register_fns_lazy() {
    if (g_real_RegisterFatBinary && g_real_RegisterFunction) return;
    void* lr = resolve_cudart();
    if (!lr) return;
    if (!g_real_RegisterFatBinary)
        g_real_RegisterFatBinary    = (cudaRegisterFatBinary_fn) dlsym(lr, "__cudaRegisterFatBinary");
    if (!g_real_RegisterFatBinaryEnd)
        g_real_RegisterFatBinaryEnd = (cudaRegisterFatBinaryEnd_fn) dlsym(lr, "__cudaRegisterFatBinaryEnd");
    if (!g_real_RegisterFunction)
        g_real_RegisterFunction     = (cudaRegisterFunction_fn) dlsym(lr, "__cudaRegisterFunction");
}

extern "C" __attribute__((visibility("default")))
void** __cudaRegisterFatBinary(void* fatCubin)
{
    resolve_register_fns_lazy();
    if (!g_real_RegisterFatBinary) return nullptr;
    void** handle = g_real_RegisterFatBinary(fatCubin);
    static int dbg = 0;
    if (dbg < 3) {
        fprintf(stderr, "[CIPHER REGFAT] call#%d fat=%p handle=%p\n",
                dbg, fatCubin, (void*)handle);
        dbg++;
    }
    typedef void (*reg_fn)(void*, const void*);
    static reg_fn s = nullptr;
    static int    sr = 0;
    if (!sr) { s = (reg_fn)dlsym(RTLD_DEFAULT, "cipher_param_register_fatbin"); sr = 1; }
    if (s && handle) s((void*)handle, fatCubin);
    // Deferred GOT repatch: every fatbin registration may correspond to a
    // newly-loaded .so (vllm/_C.abi3.so, flashinfer, bitsandbytes, …) with
    // its own __cudaLaunchKernel slot. Walk all DSOs each time —
    // patch_slot's `orig == shim` early-out makes subsequent walks O(N) and
    // cheap.
    resolve_real();
    auto_repatch_now();
    return handle;
}

extern "C" __attribute__((visibility("default")))
void __cudaRegisterFatBinaryEnd(void** fatCubinHandle)
{
    resolve_register_fns_lazy();
    if (g_real_RegisterFatBinaryEnd) g_real_RegisterFatBinaryEnd(fatCubinHandle);
}

extern "C" __attribute__((visibility("default")))
void __cudaRegisterFunction(void** fatCubinHandle,
                             const char* hostFun,
                             char*       deviceFun,
                             const char* deviceName,
                             int thread_limit,
                             void* tid, void* bid, void* bDim, void* gDim, int* wSize)
{
    resolve_register_fns_lazy();
    if (g_real_RegisterFunction) {
        g_real_RegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                 thread_limit, tid, bid, bDim, gDim, wSize);
    }
    static int dbg = 0;
    if (dbg < 3) {
        fprintf(stderr, "[CIPHER REGFN] #%d host=%p name=%s\n",
                dbg, hostFun, deviceName ? deviceName : "(null)");
        dbg++;
    }
    typedef void (*reg_fn)(void*, const void*, const char*, const char*);
    static reg_fn s = nullptr;
    static int    sr = 0;
    if (!sr) { s = (reg_fn)dlsym(RTLD_DEFAULT, "cipher_param_register_function"); sr = 1; }
    if (s) s((void*)fatCubinHandle, (const void*)hostFun, deviceFun, deviceName);

    /* K.1.5 Step 0 Bug #5 fix v4 (advisor 2026-05-27): populate workload
     * classifier's host_ptr -> device_name map. PyTorch + vLLM launch via
     * cudaLaunchKernel (runtime API) which passes host_fn (this hostFun
     * arg). cuFuncGetName cannot resolve host pointers; the registration
     * map closes the gap. */
    cipher_workload_register_kernel((const void *)hostFun, deviceName);
    cipher_workload_observe_register_site((const void *)hostFun,
        CIPHER_WL_REG_SITE_CUDA_REGISTER_FN);
}

// ── Phase 1 driver-level shims: cuModuleLoadData / cuModuleLoadDataEx /
//    cuModuleGetFunction ────────────────────────────────────────────────────
//
// These are the functions cudart_static calls into when registering kernels
// with the driver. The image blob is the DECOMPRESSED cubin — exactly what
// we need to parse `.nv.info.<funcname>` sections.

static void resolve_module_fns_lazy() {
    if (g_real_cuModuleLoadData && g_real_cuModuleGetFunction) return;
    void* lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!lc) lc = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lc) return;
    if (!g_real_cuModuleLoadData)
        g_real_cuModuleLoadData    = (cuModuleLoadData_fn)    dlsym(lc, "cuModuleLoadData");
    if (!g_real_cuModuleLoadDataEx)
        g_real_cuModuleLoadDataEx  = (cuModuleLoadDataEx_fn)  dlsym(lc, "cuModuleLoadDataEx");
    if (!g_real_cuModuleGetFunction)
        g_real_cuModuleGetFunction = (cuModuleGetFunction_fn) dlsym(lc, "cuModuleGetFunction");
}

extern "C" __attribute__((visibility("default")))
CUresult cuModuleLoadData(void** mod, const void* image)
{
    resolve_module_fns_lazy();
    if (!g_real_cuModuleLoadData) return (CUresult)3 /* CUDA_ERROR_NOT_INITIALIZED */;
    static std::atomic<int> entry_dbg{0};
    int n = entry_dbg.fetch_add(1, std::memory_order_relaxed);
    if (n < 5) {
        fprintf(stderr, "[CIPHER MODLOAD-ENTRY] #%d image=%p\n", n, image);
    }
    CUresult r = g_real_cuModuleLoadData(mod, image);
    if (r == CUDA_SUCCESS) {
        typedef int (*ingest_fn)(const void*);
        static ingest_fn s = nullptr;
        static int       sr = 0;
        if (!sr) { s = (ingest_fn)dlsym(RTLD_DEFAULT, "cipher_param_ingest_image"); sr = 1; }
        if (s) {
            int added = s(image);
            static int dbg = 0;
            if (added > 0 && dbg < 10) {
                fprintf(stderr, "[CIPHER MODLOAD] image=%p added=%d kernels\n",
                        image, added);
                dbg++;
            }
        }
        // A new CUDA module load typically means a new .so was just
        // dlopened.  Re-walk all loaded DSOs to patch any newly-arrived
        // GOT slots (vllm/_C.abi3.so, flashinfer, etc.).
        auto_repatch_now();
    }
    return r;
}

extern "C" __attribute__((visibility("default")))
CUresult cuModuleLoadDataEx(void** mod, const void* image,
                              unsigned numOpts, int* opts, void** optVals)
{
    resolve_module_fns_lazy();
    if (!g_real_cuModuleLoadDataEx) return (CUresult)3 /* CUDA_ERROR_NOT_INITIALIZED */;
    CUresult r = g_real_cuModuleLoadDataEx(mod, image, numOpts, opts, optVals);
    if (r == CUDA_SUCCESS) {
        typedef int (*ingest_fn)(const void*);
        static ingest_fn s = nullptr;
        static int       sr = 0;
        if (!sr) { s = (ingest_fn)dlsym(RTLD_DEFAULT, "cipher_param_ingest_image"); sr = 1; }
        if (s) {
            int added = s(image);
            static int dbg = 0;
            if (added > 0 && dbg < 10) {
                fprintf(stderr, "[CIPHER MODLOADEX] image=%p added=%d kernels\n",
                        image, added);
                dbg++;
            }
        }
        auto_repatch_now();
    }
    return r;
}

extern "C" __attribute__((visibility("default")))
CUresult cuModuleGetFunction(void** hfunc, void* hmod, const char* name)
{
    resolve_module_fns_lazy();
    if (!g_real_cuModuleGetFunction) return (CUresult)3 /* CUDA_ERROR_NOT_INITIALIZED */;
    CUresult r = g_real_cuModuleGetFunction(hfunc, hmod, name);
    if (r == CUDA_SUCCESS && hfunc && *hfunc) {
        typedef void (*reg_fn)(const void*, const char*);
        static reg_fn s = nullptr;
        static int    sr = 0;
        if (!sr) { s = (reg_fn)dlsym(RTLD_DEFAULT, "cipher_param_register_cufunc"); sr = 1; }
        if (s) s(*hfunc, name);

        /* K.1.5 Step 0 Sub-step 2 (advisor 2026-05-27): register
         * (CUfunction, name) in workload classifier's map. Driver-API
         * cuLaunchKernel sees this CUfunction; map lookup is now the
         * primary name-resolution path for driver-API-launched kernels
         * (e.g. Triton-loaded). cuFuncGetName remains as fallback. */
        cipher_workload_register_kernel((const void *)*hfunc, name);
        cipher_workload_observe_register_site((const void *)*hfunc,
            CIPHER_WL_REG_SITE_CU_MODULE_GET_FN);
    }
    return r;
}

// ── CUDA 12 Library API shims — the path PyTorch's static cudart actually
//    uses. cuLibraryLoadData receives the decompressed cubin image; we
//    parse it for kernel param info. cuLibraryGetKernel + cuKernelGetFunction
//    together give us the (CUfunction, kernel-name) association.

// Track CUkernel → (name, library) and CUfunction → CUkernel. Lets us
// resolve a launch-time CUfunction back to its name AND ask the driver
// (cuKernelGetParamInfo) for parameter offsets on demand — no cubin
// parsing required.
static std::mutex g_cukernel_mu;
static std::unordered_map<const void*, std::string> g_cukernel_name;
static std::unordered_map<const void*, const void*> g_cufunc_to_kernel;

extern "C" __attribute__((visibility("default")))
CUresult cuLibraryLoadData(void** lib, const void* image,
                              int* jitOptions, void** jitOptValues, unsigned numJitOptions,
                              int* libOptions, void** libOptValues, unsigned numLibOptions)
{
    resolve_module_fns_lazy();
    if (!g_real_cuLibraryLoadData) return (CUresult)3;
    static std::atomic<int> entry_dbg{0};
    int n = entry_dbg.fetch_add(1, std::memory_order_relaxed);
    if (n < 5) {
        fprintf(stderr, "[CIPHER LIBLOAD-ENTRY] #%d image=%p\n", n, image);
    }
    CUresult r = g_real_cuLibraryLoadData(lib, image,
                                            jitOptions, jitOptValues, numJitOptions,
                                            libOptions, libOptValues, numLibOptions);
    if (r == CUDA_SUCCESS) {
        typedef int (*ingest_fn)(const void*);
        static ingest_fn s = nullptr;
        static int       sr = 0;
        if (!sr) { s = (ingest_fn)dlsym(RTLD_DEFAULT, "cipher_param_ingest_image"); sr = 1; }
        if (s) {
            int added = s(image);
            static std::atomic<int> dbg{0};
            int dn = dbg.fetch_add(1, std::memory_order_relaxed);
            if (added > 0 && dn < 10) {
                fprintf(stderr, "[CIPHER LIBLOAD] image=%p added=%d kernels\n",
                        image, added);
            }
        }
        auto_repatch_now();
    }
    return r;
}

extern "C" __attribute__((visibility("default")))
CUresult cuLibraryGetKernel(void** ker, void* lib, const char* name)
{
    resolve_module_fns_lazy();
    if (!g_real_cuLibraryGetKernel) return (CUresult)3;
    CUresult r = g_real_cuLibraryGetKernel(ker, lib, name);
    if (r == CUDA_SUCCESS && ker && *ker && name) {
        std::lock_guard<std::mutex> lk(g_cukernel_mu);
        g_cukernel_name[*ker] = name;
        static std::atomic<int> dbg{0};
        int n = dbg.fetch_add(1, std::memory_order_relaxed);
        if (n < 5) {
            fprintf(stderr, "[CIPHER LIBGETKER] #%d ker=%p name=%.110s\n",
                    n, *ker, name);
        }

        /* K.1.5 Step 0 Sub-step 3: register (CUkernel, name) for the case
         * some launch paths pass CUkernel directly (CUDA 12.4+ extended
         * cuLaunchKernelEx). cuKernelGetFunction below covers the more
         * common CUfunction path. Both registrations are no-ops on
         * duplicate (map dedupes by ptr). */
        cipher_workload_register_kernel((const void *)*ker, name);
        cipher_workload_observe_register_site((const void *)*ker,
            CIPHER_WL_REG_SITE_CU_LIBRARY_GET);
    }
    return r;
}

extern "C" __attribute__((visibility("default")))
CUresult cuKernelGetFunction(void** hfunc, void* ker)
{
    resolve_module_fns_lazy();
    if (!g_real_cuKernelGetFunction) return (CUresult)3;
    CUresult r = g_real_cuKernelGetFunction(hfunc, ker);
    if (r == CUDA_SUCCESS && hfunc && *hfunc && ker) {
        std::string name;
        {
            std::lock_guard<std::mutex> lk(g_cukernel_mu);
            auto it = g_cukernel_name.find(ker);
            if (it != g_cukernel_name.end()) name = it->second;
            g_cufunc_to_kernel[*hfunc] = ker;
        }
        static std::atomic<int> dbg{0};
        int n = dbg.fetch_add(1, std::memory_order_relaxed);
        if (n < 5) {
            fprintf(stderr, "[CIPHER KERGETFN] #%d ker=%p func=%p name=%.100s\n",
                    n, ker, *hfunc, name.empty() ? "(unknown)" : name.c_str());
        }
        if (!name.empty()) {
            typedef void (*reg_fn)(const void*, const char*);
            static reg_fn s = nullptr;
            static int    sr = 0;
            if (!sr) { s = (reg_fn)dlsym(RTLD_DEFAULT, "cipher_param_register_cufunc"); sr = 1; }
            if (s) s(*hfunc, name.c_str());

            /* K.1.5 Step 0 Sub-step 3: cuKernelGetFunction is the bridge
             * from cuLibraryGetKernel's CUkernel to the CUfunction handle
             * that cuLaunchKernel actually sees. This is the primary
             * coverage path for Triton/Inductor-loaded kernels in PyTorch
             * 2.x + vLLM. The name lookup map (g_cukernel_name) was
             * populated by cuLibraryGetKernel above; we promote that
             * (CUkernel,name) edge into (CUfunction,name) here so
             * lookup_fn_bits can resolve names for driver-API launches.
             *
             * Memory caveat: name.c_str() is owned by the local std::string
             * which dies at scope exit. The workload classifier map stores
             * the const char* pointer by reference (no strdup). We dedupe
             * the name into the g_cukernel_name unordered_map (heap-stable
             * std::string storage) and use ITS c_str() which lives for
             * process lifetime. */
            const char *stable_name = nullptr;
            {
                std::lock_guard<std::mutex> lk(g_cukernel_mu);
                auto it = g_cukernel_name.find(ker);
                if (it != g_cukernel_name.end()) {
                    stable_name = it->second.c_str();
                }
            }
            if (stable_name) {
                cipher_workload_register_kernel((const void *)*hfunc, stable_name);
                cipher_workload_observe_register_site((const void *)*hfunc,
                    CIPHER_WL_REG_SITE_CU_KERNEL_GET_FN);
            }
        }
    }
    return r;
}

// Public: at lookup time, retrieve param info via CUDA's own API. Avoids
// cubin parsing entirely. Uses our CUfunction → CUkernel map, then calls
// cuKernelGetParamInfo for each ordinal. Returns N params filled, or 0.
extern "C" __attribute__((visibility("default")))
int cipher_param_query_via_cuda(const void* cufunc,
                                  uint16_t* offsets, uint16_t* sizes, int max_count)
{
    if (!cufunc) return 0;
    resolve_module_fns_lazy();
    if (!g_real_cuKernelGetParamInfo) return 0;
    void* kernel = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_cukernel_mu);
        auto it = g_cufunc_to_kernel.find(cufunc);
        if (it == g_cufunc_to_kernel.end()) return 0;
        kernel = (void*)it->second;
    }
    int n = 0;
    while (n < max_count) {
        size_t off = 0, sz = 0;
        CUresult r = g_real_cuKernelGetParamInfo(kernel, (size_t)n, &off, &sz);
        if (r != CUDA_SUCCESS) break;
        offsets[n] = (uint16_t)off;
        sizes[n]   = (uint16_t)sz;
        n++;
    }
    return n;
}

// Lookup name by CUfunction (does not require cubin parsing).
extern "C" __attribute__((visibility("default")))
const char* cipher_param_query_name(const void* cufunc)
{
    if (!cufunc) return nullptr;
    static thread_local std::string buf;
    void* kernel = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_cukernel_mu);
        auto it = g_cufunc_to_kernel.find(cufunc);
        if (it == g_cufunc_to_kernel.end()) return nullptr;
        kernel = (void*)it->second;
        auto nit = g_cukernel_name.find(kernel);
        if (nit != g_cukernel_name.end()) {
            buf = nit->second;
            return buf.c_str();
        }
    }
    return nullptr;
}

// ── dlsym wrapper (patched into cuBLASLt's GOT only) ────────────────────────
// This is NOT a global dlsym hook. We only replace dlsym's GOT entry inside
// libraries that use dlsym to resolve CUDA functions (e.g. cuBLASLt).

static void* cipher_dlsym_wrapper(void* handle, const char* symbol);

// ── Shim table ──────────────────────────────────────────────────────────────
struct SymPatch { const char* name; void* shim; void** real; };

static SymPatch g_patches[] = {
    { "cuLaunchKernel",            (void*)cuLaunchKernel,            (void**)&g_real_cuLaunch },
    { "cuLaunchKernel_ptsz",       (void*)cuLaunchKernel_ptsz,       (void**)&g_real_cuLaunch_ptsz },
    { "cuLaunchKernelEx",          (void*)cuLaunchKernelEx,          (void**)&g_real_cuLaunchEx },
    { "cuLaunchKernelEx_ptsz",     (void*)cuLaunchKernelEx_ptsz,     (void**)&g_real_cuLaunchEx_ptsz },
    { "cudaLaunchKernel",          (void*)cudaLaunchKernel,          (void**)&g_real_cudaLaunch },
    { "cudaLaunchKernel_ptsz",     (void*)cudaLaunchKernel_ptsz,     (void**)&g_real_cudaLaunch_ptsz },
    { "__cudaLaunchKernel",        (void*)__cudaLaunchKernel,        (void**)&g_real___cudaLaunch },
    { "__cudaLaunchKernel_ptsz",   (void*)__cudaLaunchKernel_ptsz,   (void**)&g_real___cudaLaunch_ptsz },
    { "cudaLaunchKernelExC",       (void*)cudaLaunchKernelExC,       (void**)&g_real_cudaLaunchExC },
    { "cudaLaunchKernelExC_ptsz",  (void*)cudaLaunchKernelExC_ptsz,  (void**)&g_real_cudaLaunchExC_ptsz },
    { "cuGetProcAddress",          (void*)cuGetProcAddress,          (void**)&g_real_cuGetProc },
    { "cuGetProcAddress_v2",       (void*)cuGetProcAddress_v2,       (void**)&g_real_cuGetProc_v2 },
    // B.6''.9.8.1: cuBLAS + NCCL entries DROPPED (cipher_rt_phase4
    // canonical: cipher_rt_cublas_shim.c + cublaslt_variants.c +
    // dlsym_hook.c). Impl functions retained but unreachable.
    // { "cublasLtMatmul",         (void*)cipher_cublasLtMatmul_impl,(void**)&g_real_cublasLtMatmul },
    // { "cublasGemmEx",           (void*)cipher_cublasGemmEx_impl,  (void**)&g_real_cublasGemmEx },
    // { "ncclAllReduce",          (void*)cipher_ncclAllReduce_impl, (void**)&g_real_ncclAllReduce },
    { "__cudaRegisterFatBinary",    (void*)__cudaRegisterFatBinary,    (void**)&g_real_RegisterFatBinary },
    { "__cudaRegisterFatBinaryEnd", (void*)__cudaRegisterFatBinaryEnd, (void**)&g_real_RegisterFatBinaryEnd },
    { "__cudaRegisterFunction",     (void*)__cudaRegisterFunction,     (void**)&g_real_RegisterFunction },
    // B.6''.9.8.4: cuGraphLaunch + variants + cudaGraphLaunch — graph-replay observability
    { "cuGraphLaunch",              (void*)cuGraphLaunch,              (void**)&g_real_cuGraphLaunch },
    { "cuGraphLaunch_ptsz",         (void*)cuGraphLaunch_ptsz,         (void**)&g_real_cuGraphLaunch_ptsz },
    { "cuGraphLaunchPipelined",     (void*)cuGraphLaunchPipelined,     (void**)&g_real_cuGraphLaunchPipelined },
    { "cudaGraphLaunch",            (void*)cudaGraphLaunch,            (void**)&g_real_cudaGraphLaunch },
    { "cudaGraphLaunch_ptsz",       (void*)cudaGraphLaunch_ptsz,       (void**)&g_real_cudaGraphLaunch_ptsz },
    // B.6''.9.8.5: cuGraphAddKernelNode + ExecKernelNodeSetParams + cudart wrapper — capture-time substitution
    { "cuGraphAddKernelNode",       (void*)cuGraphAddKernelNode,       (void**)&g_real_cuGraphAddKernelNode },
    { "cuGraphAddKernelNode_v2",    (void*)cuGraphAddKernelNode_v2,    (void**)&g_real_cuGraphAddKernelNode_v2 },
    { "cuGraphExecKernelNodeSetParams",    (void*)cuGraphExecKernelNodeSetParams,    (void**)&g_real_cuGraphExecKernelNodeSetParams },
    { "cuGraphExecKernelNodeSetParams_v2", (void*)cuGraphExecKernelNodeSetParams_v2, (void**)&g_real_cuGraphExecKernelNodeSetParams_v2 },
    { "cudaGraphAddKernelNode",     (void*)cudaGraphAddKernelNode,     (void**)&g_real_cudaGraphAddKernelNode },
    { "cuModuleLoadData",           (void*)cuModuleLoadData,           (void**)&g_real_cuModuleLoadData },
    { "cuModuleLoadDataEx",         (void*)cuModuleLoadDataEx,         (void**)&g_real_cuModuleLoadDataEx },
    { "cuModuleGetFunction",        (void*)cuModuleGetFunction,        (void**)&g_real_cuModuleGetFunction },
    { "cuLibraryLoadData",          (void*)cuLibraryLoadData,          (void**)&g_real_cuLibraryLoadData },
    { "cuLibraryGetKernel",         (void*)cuLibraryGetKernel,         (void**)&g_real_cuLibraryGetKernel },
    { "cuKernelGetFunction",        (void*)cuKernelGetFunction,        (void**)&g_real_cuKernelGetFunction },
};
static constexpr int N_PATCHES = sizeof(g_patches) / sizeof(g_patches[0]);

static void* get_shim(const char* s) {
    // Allow 'c' (cuda*/cublas*/cudnn*), 'n' (nccl*), and '_' (__cuda*).
    // The '_' branch is mandatory for torch 2.11/cudart 13: nvcc-generated
    // host stubs route launches via __cudaLaunchKernel (double underscore),
    // and __cudaRegister* shims must be findable too.
    if (!s || (s[0] != 'c' && s[0] != 'n' && s[0] != '_')) return nullptr;
    // Never redirect these — CIPHER calls them for substitution / internal use
    if (strstr(s, "cublasSgemm") || strstr(s, "cublasCreate") ||
        strstr(s, "cublasSetStream"))
        return nullptr;
    for (int i = 0; i < N_PATCHES; i++)
        if (strcmp(s, g_patches[i].name) == 0) return g_patches[i].shim;
    return nullptr;
}

// ── Resolve real pointers ───────────────────────────────────────────────────

static void resolve_real() {
    if (g_resolved.load(std::memory_order_acquire)) return;

    void* lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!lc) lc = dlopen("libcuda.so.1", RTLD_NOW);
    void* lr = resolve_cudart();

    if (lc) {
        if (!g_real_cuLaunch)        g_real_cuLaunch        = (cuLaunchKernel_fn)   dlsym(lc, "cuLaunchKernel");
        if (!g_real_cuLaunch_ptsz)   g_real_cuLaunch_ptsz   = (cuLaunchKernel_fn)   dlsym(lc, "cuLaunchKernel_ptsz");
        if (!g_real_cuLaunchEx)      g_real_cuLaunchEx      = (cuLaunchKernelEx_fn)  dlsym(lc, "cuLaunchKernelEx");
        if (!g_real_cuLaunchEx_ptsz) g_real_cuLaunchEx_ptsz = (cuLaunchKernelEx_fn)  dlsym(lc, "cuLaunchKernelEx_ptsz");
        // B.6''.9.8.4: resolve real cuGraphLaunch + variants
        if (!g_real_cuGraphLaunch)          g_real_cuGraphLaunch          = (cuGraphLaunch_fn) dlsym(lc, "cuGraphLaunch");
        if (!g_real_cuGraphLaunch_ptsz)     g_real_cuGraphLaunch_ptsz     = (cuGraphLaunch_fn) dlsym(lc, "cuGraphLaunch_ptsz");
        if (!g_real_cuGraphLaunchPipelined) g_real_cuGraphLaunchPipelined = (cuGraphLaunchPipelined_fn) dlsym(lc, "cuGraphLaunchPipelined");
        // B.6''.9.8.5: resolve real cuGraphAddKernelNode + ExecKernelNodeSetParams
        if (!g_real_cuGraphAddKernelNode)    g_real_cuGraphAddKernelNode    = (cuGraphAddKernelNode_fn) dlsym(lc, "cuGraphAddKernelNode");
        if (!g_real_cuGraphAddKernelNode_v2) g_real_cuGraphAddKernelNode_v2 = (cuGraphAddKernelNode_fn) dlsym(lc, "cuGraphAddKernelNode_v2");
        if (!g_real_cuGraphExecKernelNodeSetParams)
            g_real_cuGraphExecKernelNodeSetParams = (cuGraphExecKernelNodeSetParams_fn) dlsym(lc, "cuGraphExecKernelNodeSetParams");
        if (!g_real_cuGraphExecKernelNodeSetParams_v2)
            g_real_cuGraphExecKernelNodeSetParams_v2 = (cuGraphExecKernelNodeSetParams_fn) dlsym(lc, "cuGraphExecKernelNodeSetParams_v2");
        // B.6''.9.8.5b: cuStreamGetCaptureInfo for PyTorch capture-detection.
        // Prefer v3 if present (CUDA 12.3+), fall back to v2, then v1.
        if (!g_real_cuStreamGetCaptureInfo)
            g_real_cuStreamGetCaptureInfo = (cuStreamGetCaptureInfo_fn) dlsym(lc, "cuStreamGetCaptureInfo_v2");
        if (!g_real_cuStreamGetCaptureInfo)
            g_real_cuStreamGetCaptureInfo = (cuStreamGetCaptureInfo_fn) dlsym(lc, "cuStreamGetCaptureInfo");
        if (!g_real_cuGetProc)       g_real_cuGetProc       = (cuGetProcAddress_fn)  dlsym(lc, "cuGetProcAddress");
        if (!g_real_cuGetProc_v2)    g_real_cuGetProc_v2    = (cuGetProcAddress_fn)  dlsym(lc, "cuGetProcAddress_v2");
        if (!g_real_cuModuleLoadData)
            g_real_cuModuleLoadData    = (cuModuleLoadData_fn)    dlsym(lc, "cuModuleLoadData");
        if (!g_real_cuModuleLoadDataEx)
            g_real_cuModuleLoadDataEx  = (cuModuleLoadDataEx_fn)  dlsym(lc, "cuModuleLoadDataEx");
        if (!g_real_cuModuleGetFunction)
            g_real_cuModuleGetFunction = (cuModuleGetFunction_fn) dlsym(lc, "cuModuleGetFunction");
        if (!g_real_cuLibraryLoadData)
            g_real_cuLibraryLoadData   = (cuLibraryLoadData_fn)   dlsym(lc, "cuLibraryLoadData");
        if (!g_real_cuLibraryGetKernel)
            g_real_cuLibraryGetKernel  = (cuLibraryGetKernel_fn)  dlsym(lc, "cuLibraryGetKernel");
        if (!g_real_cuKernelGetFunction)
            g_real_cuKernelGetFunction = (cuKernelGetFunction_fn) dlsym(lc, "cuKernelGetFunction");
        if (!g_real_cuKernelGetName)
            g_real_cuKernelGetName     = (cuKernelGetName_fn)     dlsym(lc, "cuKernelGetName");
        if (!g_real_cuKernelGetParamInfo)
            g_real_cuKernelGetParamInfo = (cuKernelGetParamInfo_fn) dlsym(lc, "cuKernelGetParamInfo");
    }
    if (lr) {
        if (!g_real_cudaLaunch)         g_real_cudaLaunch         = (cudaLaunchKernel_fn)    dlsym(lr, "cudaLaunchKernel");
        if (!g_real_cudaLaunch_ptsz)    g_real_cudaLaunch_ptsz    = (cudaLaunchKernel_fn)    dlsym(lr, "cudaLaunchKernel_ptsz");
        if (!g_real___cudaLaunch)       g_real___cudaLaunch       = (cudaLaunchKernel_fn)    dlsym(lr, "__cudaLaunchKernel");
        if (!g_real___cudaLaunch_ptsz)  g_real___cudaLaunch_ptsz  = (cudaLaunchKernel_fn)    dlsym(lr, "__cudaLaunchKernel_ptsz");
        if (!g_real_cudaLaunchExC)      g_real_cudaLaunchExC      = (cudaLaunchKernelExC_fn) dlsym(lr, "cudaLaunchKernelExC");
        if (!g_real_cudaLaunchExC_ptsz) g_real_cudaLaunchExC_ptsz = (cudaLaunchKernelExC_fn) dlsym(lr, "cudaLaunchKernelExC_ptsz");
        // B.6''.9.8.4: cudart-level cudaGraphLaunch
        if (!g_real_cudaGraphLaunch)      g_real_cudaGraphLaunch      = (cudaGraphLaunch_fn) dlsym(lr, "cudaGraphLaunch");
        if (!g_real_cudaGraphLaunch_ptsz) g_real_cudaGraphLaunch_ptsz = (cudaGraphLaunch_fn) dlsym(lr, "cudaGraphLaunch_ptsz");
        // B.6''.9.8.5: cudart-level cudaGraphAddKernelNode
        if (!g_real_cudaGraphAddKernelNode)
            g_real_cudaGraphAddKernelNode = (cudaGraphAddKernelNode_fn) dlsym(lr, "cudaGraphAddKernelNode");
        if (!g_real_RegisterFatBinary)
            g_real_RegisterFatBinary = (cudaRegisterFatBinary_fn) dlsym(lr, "__cudaRegisterFatBinary");
        if (!g_real_RegisterFatBinaryEnd)
            g_real_RegisterFatBinaryEnd = (cudaRegisterFatBinaryEnd_fn) dlsym(lr, "__cudaRegisterFatBinaryEnd");
        if (!g_real_RegisterFunction)
            g_real_RegisterFunction = (cudaRegisterFunction_fn) dlsym(lr, "__cudaRegisterFunction");
    }

    // Only mark fully resolved when BOTH driver and cudart symbols landed.
    // Cudart on torch 2.11 lives at a bundled path that isn't reachable by
    // default linker search; resolve_cudart() returns nullptr until torch
    // has loaded it via its own RPATH. Allow re-entry until then so the
    // first __cudaRegister*/cudaLaunchKernel call picks up the real cudart.
    if (lc && lr) g_resolved.store(true, std::memory_order_release);
}

// ── GOT patching ────────────────────────────────────────────────────────────

static bool patch_slot(void** slot, void* shim, const char* name, const char* dso) {
    void* orig = *slot;
    if (orig == shim) return false;   // already ours
    if (orig == nullptr) return false; // not yet resolved

    // Save original (first writer wins)
    for (int i = 0; i < N_PATCHES; i++) {
        if (strcmp(name, g_patches[i].name) == 0) {
            if (*g_patches[i].real == nullptr)
                *g_patches[i].real = orig;
            break;
        }
    }

    uintptr_t page = (uintptr_t)slot & ~(g_page_size - 1);
    mprotect((void*)page, g_page_size * 2, PROT_READ | PROT_WRITE);
    *slot = shim;
    // Leave writable — dynamic linker needs write access during cleanup
    // (alternative: track pages and restore on teardown)

    g_got_patches.fetch_add(1, std::memory_order_relaxed);
    fprintf(stderr, "[CIPHER HOOK]   patched %-28s in %s (was %p)\n", name, dso, orig);
    return true;
}

static void patch_dso(const char* dso, ElfW(Addr) base, const ElfW(Dyn)* dyn) {
    const ElfW(Rela)* jmprel = nullptr;
    size_t jmprel_sz = 0;
    const ElfW(Sym)* symtab = nullptr;
    const char* strtab = nullptr;
    const ElfW(Rela)* rela = nullptr;
    size_t rela_sz = 0;

    for (const ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; d++) {
        switch (d->d_tag) {
            case DT_JMPREL:   jmprel    = (const ElfW(Rela)*)d->d_un.d_ptr; break;
            case DT_PLTRELSZ: jmprel_sz = d->d_un.d_val; break;
            case DT_SYMTAB:   symtab    = (const ElfW(Sym)*)d->d_un.d_ptr;  break;
            case DT_STRTAB:   strtab    = (const char*)d->d_un.d_ptr;       break;
            case DT_RELA:     rela      = (const ElfW(Rela)*)d->d_un.d_ptr; break;
            case DT_RELASZ:   rela_sz   = d->d_un.d_val; break;
        }
    }
    if (!symtab || !strtab) return;


    auto walk = [&](const ElfW(Rela)* tbl, size_t sz, unsigned req_type) {
        if (!tbl || !sz) return;
        size_t n = sz / sizeof(ElfW(Rela));
        for (size_t i = 0; i < n; i++) {
            unsigned rtype = ELF64_R_TYPE(tbl[i].r_info);
            if (req_type && rtype != req_type && rtype != R_X86_64_JUMP_SLOT) continue;
            unsigned si = ELF64_R_SYM(tbl[i].r_info);
            const char* nm = strtab + symtab[si].st_name;
            void* shim = get_shim(nm);
            if (shim) {
                void** slot = (void**)(base + tbl[i].r_offset);
                patch_slot(slot, shim, nm, dso);
            }
            if (getenv("CIPHER_GOT_DEBUG")
                && (strncmp(nm, "cublasLt", 8) == 0
                    || strncmp(nm, "cublasGemm", 10) == 0)) {
                void** slot = (void**)(base + tbl[i].r_offset);
                fprintf(stderr,
                    "[CIPHER GOT-DBG] sym='%s' shim=%p slot=%p dso=%s\n",
                    nm, shim, *slot, dso);
            }
        }
    };

    walk(jmprel, jmprel_sz, R_X86_64_JUMP_SLOT);
    walk(rela, rela_sz, R_X86_64_GLOB_DAT);
}

// ── Targeted dlsym GOT patch for cuBLASLt ───────────────────────────────────

// g_real_dlsym defined above, near TLS storage

static void* cipher_dlsym_wrapper(void* handle, const char* symbol) {
    // Call the REAL glibc dlsym (saved before patching)
    void* result = g_real_dlsym(handle, symbol);
    if (result && symbol) {
        void* shim = get_shim(symbol);
        if (shim) {
            resolve_real();
            return shim;
        }
    }
    return result;
}

// Patch dlsym GOT entry in a DSO (for libraries that use dlsym to resolve CUDA)
static void patch_dlsym_got(const char* dso, ElfW(Addr) base, const ElfW(Dyn)* dyn) {
    const ElfW(Rela)* jmprel = nullptr;
    size_t jmprel_sz = 0;
    const ElfW(Sym)* symtab = nullptr;
    const char* strtab = nullptr;

    for (const ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; d++) {
        switch (d->d_tag) {
            case DT_JMPREL:   jmprel    = (const ElfW(Rela)*)d->d_un.d_ptr; break;
            case DT_PLTRELSZ: jmprel_sz = d->d_un.d_val; break;
            case DT_SYMTAB:   symtab    = (const ElfW(Sym)*)d->d_un.d_ptr;  break;
            case DT_STRTAB:   strtab    = (const char*)d->d_un.d_ptr;       break;
        }
    }
    if (!jmprel || !jmprel_sz || !symtab || !strtab) return;

    size_t n = jmprel_sz / sizeof(ElfW(Rela));
    for (size_t i = 0; i < n; i++) {
        unsigned si = ELF64_R_SYM(jmprel[i].r_info);
        const char* nm = strtab + symtab[si].st_name;
        if (strcmp(nm, "dlsym") == 0) {
            void** slot = (void**)(base + jmprel[i].r_offset);
            void* orig = *slot;
            if (orig == (void*)cipher_dlsym_wrapper) return; // already patched

            // Save real dlsym (first one wins)
            if (!g_real_dlsym) g_real_dlsym = (void*(*)(void*, const char*))orig;

            uintptr_t page = (uintptr_t)slot & ~(g_page_size - 1);
            mprotect((void*)page, g_page_size * 2, PROT_READ | PROT_WRITE);
            *slot = (void*)cipher_dlsym_wrapper;

            g_got_patches.fetch_add(1, std::memory_order_relaxed);
            fprintf(stderr, "[CIPHER HOOK]   patched dlsym GOT in %s (was %p)\n", dso, orig);
            return;
        }
    }
}

static int phdr_cb(struct dl_phdr_info* info, size_t, void*) {
    for (int i = 0; i < info->dlpi_phnum; i++) {
        if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
            const ElfW(Dyn)* dyn = (const ElfW(Dyn)*)(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
            const char* name = info->dlpi_name && info->dlpi_name[0] ? info->dlpi_name : "[main]";

            // Skip CIPHER's own libraries — patching their cudart calls back
            // to our shims creates recursion (rt's dispatcher → cudaLaunchKernel
            // shim → rt's dispatcher → …) and breaks rt's actuator path.
            if (strstr(name, "libcipher_")) { break; }

            // K.1: detect libnccl presence via DSO enumeration. Used by the
            // workload classifier as a B-class (training) signal -- libnccl
            // loaded => collective ops likely in use even before the first
            // nccl* kernel fires.
            if (strstr(name, "libnccl")) {
                cipher_workload_observe_nccl_present();
            }

            // Standard GOT patch for PLT relocations (cuLaunchKernel etc.)
            patch_dso(name, info->dlpi_addr, dyn);

            // Targeted dlsym GOT patch for libraries that use dlsym to resolve
            // CUDA functions (cuBLASLt, cuSPARSELt, cuDNN, etc.)
            if (strstr(name, "cublas") || strstr(name, "cusparse") ||
                strstr(name, "cudnn") || strstr(name, "cufft") ||
                strstr(name, "cusolver") || strstr(name, "curand") ||
                strstr(name, "libtorch_cuda")) {
                patch_dlsym_got(name, info->dlpi_addr, dyn);
            }
            break;
        }
    }
    return 0;
}

extern "C" __attribute__((visibility("default")))
void cipher_repatch(void) {
    resolve_real();
    dl_iterate_phdr(phdr_cb, nullptr);
    fprintf(stderr, "[CIPHER HOOK] repatch done — total GOT patches: %lu\n",
            g_got_patches.load());
}

static void auto_repatch_once();

// ── cudaMalloc shim — triggers auto-repatch on first GPU allocation ─────────
// cudaMalloc is imported via PLT by libc10_cuda and always called before
// any kernel launch (for tensor allocation). We hook it to trigger GOT
// patching at the right moment.

typedef cudaError_t (*cudaMalloc_fn)(void**, size_t);
static cudaMalloc_fn g_real_cudaMalloc = nullptr;

extern "C" __attribute__((visibility("default")))
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!g_real_cudaMalloc) {
        void* lr = resolve_cudart();
        if (lr) g_real_cudaMalloc = (cudaMalloc_fn)dlsym(lr, "cudaMalloc");
        if (!g_real_cudaMalloc) return (cudaError_t)999;
    }
    cudaError_t r = g_real_cudaMalloc(devPtr, size);
    // After first successful malloc, CUDA is fully initialized — patch GOTs
    if (r == 0) {
        resolve_real();
        auto_repatch_once();
        /* K.1 workload classifier alloc observe. Used for weights-vs-KV-cache
         * heuristic (large contiguous = weights; many small = KV cache). */
        cipher_workload_observe_alloc((size_t)size, *devPtr);
    }
    return r;
}

// ── Shim implementations ────────────────────────────────────────────────────

static void auto_repatch_once() {
    if (!g_auto_patched.exchange(true, std::memory_order_acq_rel)) {
        dl_iterate_phdr(phdr_cb, nullptr);
    }
}

// Force a fresh GOT walk over ALL loaded DSOs.  Cheap when nothing is
// new — patch_slot's `orig == shim` early-out skips already-patched
// slots.  Called from cuModuleLoadData / cuLibraryLoadData / fatbin
// registration shims so that custom-extension .so's (vllm/_C.abi3.so,
// flashinfer, bitsandbytes, …) loaded AFTER the initial repatch get
// their __cudaLaunchKernel slots redirected to CIPHER.
static void auto_repatch_now() {
    dl_iterate_phdr(phdr_cb, nullptr);
}

#define ENSURE_REAL(ptr) do { if (__builtin_expect(!(ptr), 0)) { resolve_real(); auto_repatch_once(); if (!(ptr)) return CUDA_ERROR_NOT_FOUND; } } while(0)
#define ENSURE_RT(ptr)   do { if (__builtin_expect(!(ptr), 0)) { resolve_real(); auto_repatch_once(); if (!(ptr)) return (cudaError_t)999; } } while(0)

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernel(CUfunction f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void** params, void** extra)
{
    ENSURE_REAL(g_real_cuLaunch);
    // Bug-3 hook-level fast-out: bypass all CIPHER bookkeeping when
    // CIPHER_HOOK_PASSIVE=1 is set. This complements the cublasGemmEx
    // shim's same-named env check; together they reduce CIPHER's
    // per-launch overhead to "load the LD_PRELOAD'd .so" + "branch
    // on s_passive". Used for small/launch-bound workloads where
    // CIPHER's actuators don't amortize.
    static const bool s_passive = []() {
        const char* env = getenv("CIPHER_HOOK_PASSIVE");
        return env && *env && env[0] != '0';
    }();
    if (__builtin_expect(s_passive, 0)) {
        return g_real_cuLaunch(f, gx, gy, gz, bx, by, bz,
                                shared, stream, params, extra);
    }
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);

    // Stage 3: per-stream L2-persistence application. No-op when engine OFF.
    cipher_persist_maybe_apply((void*)stream);

    // Persistent-mode fast path: bypass classify/dispatch/ring on stable
    // repeating sequences. Only valid when we are not inside a GEMM intercept
    // (tls_shape_valid==0), because GEMM has its own handling path.
    uint64_t persist_fp = 0;
    bool persist_fast = false;
    if (!tls_shape_valid) {
        persist_fp = cipher_persist_fingerprint(
            (const void*)f, gx, gy, gz, bx, by, bz, shared);
        persist_fast = cipher_persist_try_fast_path(persist_fp);
        if (persist_fast) {
            return g_real_cuLaunch(f, gx, gy, gz, bx, by, bz, shared,
                                    stream, params, extra);
        }
    }

    // Track 2: kernel-name recognition (first-time-seen logging + counters).
    KnownKernel* kk = track_kernel(f);

    // KV-compress driver-level pointer-flow probe (cuLaunchKernel path).
    if (kk && kk->op_class == 7 && params
        && g_kk_log_enabled.load(std::memory_order_relaxed) == 1) {
        static int fa_log_lk = 0;
        if (fa_log_lk < 20) {
            fprintf(stderr, "[CIPHER FA-PARAMS-LK] #%d fn=%p name=%s",
                fa_log_lk, (void*)f, kk->name);
            for (int i = 0; i < 6; ++i) {
                void* slot = params[i];
                void* p = slot ? *(void**)slot : (void*)0;
                fprintf(stderr, " p%d=%p", i, p);
            }
            fprintf(stderr, "\n");
            fa_log_lk++;
        }
    }

    // Skip dispatch for GEMM kernels (already handled by cublasGemmEx shim).
    // Only dispatch non-GEMM kernels for classification and Chebyshev opportunity logging.
    if (!tls_shape_valid) {
        int dispatch_result = dispatch_and_log(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
        if (dispatch_result == HOOK_SUBSTITUTED)
            return CUDA_SUCCESS;
    }

    // B.6''.9.8.5b: capture-aware substitution. No-op fast path when no
    // actuator wired or stream not in capture. Memory #20 additive: existing
    // dispatch_and_log path above is untouched.
    f = (CUfunction)cipher_rt_capture_check_and_substitute(
        (void*)f, gx, gy, gz, bx, by, bz, shared, (void*)stream);

    CUresult ret = g_real_cuLaunch(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);

    // Feed persistent-mode detector post-launch (normal path only)
    if (!tls_shape_valid && ret == CUDA_SUCCESS) {
        cipher_persist_observe(persist_fp);
    }

    // Block-level calibration: disabled in cuLaunchKernel path.
    // Collection handled by cipher_wrapper.py (dtype-safe).
    if (0 && tls_shape_valid && tls_M == 4096 && tls_K == 4096 && tls_A && tls_C && ret == 0) {
        typedef bool (*block_collect_fn)(const float*, const float*, uint32_t);
        static block_collect_fn fn = nullptr;
        static int resolved = 0;
        if (!resolved) {
            fn = (block_collect_fn)dlsym(RTLD_DEFAULT, "cipher_block_sub_collect");
            resolved = 1;
        }
        if (fn) {
            // Sync to ensure output is ready (calibration only, not on hot path)
            typedef int (*sync_fn)(void*);
            static sync_fn cu_sync = (sync_fn)dlsym(RTLD_DEFAULT, "cudaStreamSynchronize");
            if (cu_sync) cu_sync(stream);
            fn((const float*)tls_A, (const float*)tls_C, 4096);
        }
    }

    return ret;
}

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernel_ptsz(CUfunction f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void** params, void** extra)
{
    ENSURE_REAL(g_real_cuLaunch_ptsz);
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    f = (CUfunction)cipher_rt_capture_check_and_substitute(
        (void*)f, gx, gy, gz, bx, by, bz, shared, (void*)stream);
    return g_real_cuLaunch_ptsz(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

// ABI-compatible mirror of CUlaunchConfig_st (CUDA 12+)
struct CUlaunchConfigMirror {
    unsigned gridDimX, gridDimY, gridDimZ;
    unsigned blockDimX, blockDimY, blockDimZ;
    unsigned sharedMemBytes;
    void*    hStream;       // CUstream
    void*    attrs;
    unsigned numAttrs;
};

// ── Extract A,B,C device pointers from cuLaunchKernelEx params ───────────────

typedef CUresult (*cuFuncGetName_fn)(const char**, CUfunction);
typedef CUresult (*cuPointerGetAttribute_fn)(void*, int, uintptr_t);

static cuFuncGetName_fn         g_cuFuncGetName = nullptr;
static cuPointerGetAttribute_fn g_cuPtrGetAttr  = nullptr;

static void resolve_introspection() {
    if (g_cuFuncGetName) return;
    void* lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!lc) return;
    auto rsym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
    g_cuFuncGetName = (cuFuncGetName_fn)rsym(lc, "cuFuncGetName");
    g_cuPtrGetAttr  = (cuPointerGetAttribute_fn)rsym(lc, "cuPointerGetAttribute");
}

// ── EDMD sample ring — tensor samples for Koopman derivation ────────────────
// After each GEMM kernel, sample 64 strided floats from C (output tensor)
// into a host ring buffer. ADAPT thread reads this for EDMD snapshots.

#define CIPHER_SAMPLE_DIM    1   // 1D: normalized Frobenius norm
#define CIPHER_SAMPLE_RING  512

struct CipherSampleEntry {
    float x[CIPHER_SAMPLE_DIM];  // input norm (||A||_F / sqrt(M*K))
    float y[CIPHER_SAMPLE_DIM];  // output norm (||C||_F / sqrt(M*N))
    uint8_t op_class;
    uint8_t valid;
};

static CipherSampleEntry g_sample_ring[CIPHER_SAMPLE_RING];
static std::atomic<uint64_t> g_sample_wseq{0};

// Export for ADAPT thread in libcipher_rt.so
extern "C" __attribute__((visibility("default")))
int cipher_read_sample(uint64_t seq, float* x_out, float* y_out, uint8_t* op_class) {
    if (seq >= g_sample_wseq.load(std::memory_order_acquire)) return 0;
    auto& e = g_sample_ring[seq % CIPHER_SAMPLE_RING];
    if (!e.valid) return 0;
    memcpy(x_out, e.x, CIPHER_SAMPLE_DIM * sizeof(float));
    memcpy(y_out, e.y, CIPHER_SAMPLE_DIM * sizeof(float));
    *op_class = e.op_class;
    return 1;
}

extern "C" __attribute__((visibility("default")))
uint64_t cipher_sample_wseq(void) {
    return g_sample_wseq.load(std::memory_order_acquire);
}

// Compute normalized Frobenius norm from a 256-element sample of the tensor.
// Returns ||sample||_F / sqrt(n_elems) — a scalar capturing output magnitude.
static float compute_sample_norm(const void* dev_ptr, int total_elems) {
    if (!dev_ptr || total_elems < 1) return 0.0f;
    typedef int (*cudaMemcpy_fn)(void*, const void*, size_t, int);
    static cudaMemcpy_fn s_memcpy = nullptr;
    if (!s_memcpy) {
        auto rsym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
        void* lr = resolve_cudart();
        if (lr) s_memcpy = (cudaMemcpy_fn)rsym(lr, "cudaMemcpy");
    }
    if (!s_memcpy) return 0.0f;
    float raw[256];
    int ncopy = (total_elems < 256) ? total_elems : 256;
    if (s_memcpy(raw, dev_ptr, ncopy * sizeof(float), 2) != 0) return 0.0f;
    double sq = 0.0;
    for (int i = 0; i < ncopy; i++) sq += (double)raw[i] * raw[i];
    return (float)(sqrt(sq / ncopy));
}

// Called after kernel launch to collect 1D norm samples for EDMD
static void maybe_collect_sample(uint8_t op_class) {
    static uint64_t s_launch_count = 0;
    s_launch_count++;
    if (s_launch_count > 500) return;
    if (!tls_shape_valid || !tls_A || !tls_C) return;

    float norm_a = compute_sample_norm(tls_A, tls_M * tls_K);
    float norm_c = compute_sample_norm(tls_C, tls_M * tls_N);
    if (norm_a < 1e-10f && norm_c < 1e-10f) return;

    uint64_t idx = g_sample_wseq.load(std::memory_order_relaxed);
    auto& e = g_sample_ring[idx % CIPHER_SAMPLE_RING];
    e.x[0] = norm_a;
    e.y[0] = norm_c;
    e.op_class = op_class;
    e.valid = 1;
    g_sample_wseq.fetch_add(1, std::memory_order_release);
}

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernelEx(void* config, CUfunction f, void** params, void** extra)
{
    ENSURE_REAL(g_real_cuLaunchEx);
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    // Skip dispatch if we're inside apply_recipe (recursion guard)
    if (tls_in_recipe)
        return g_real_cuLaunchEx(config, f, params, extra);

    auto* cfg = (CUlaunchConfigMirror*)config;

    // -- Change 3: Attention Koopman FSM observer (default OFF) ------------
    // Only call when the FSM is enabled AND we're not inside a cublas GEMM
    // intercept (the QK/atV GEMMs are handled by cipher_attn_fsm_on_gemm).
    if (cipher_attn_fsm_on_kernel && !tls_shape_valid && !tls_in_recipe
        && cipher_attn_koopman_enabled
        && cipher_attn_koopman_enabled()) {
        CipherAttnAction act = cipher_attn_fsm_on_kernel(
            (const void*)f,
            cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
            cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
            cfg->sharedMemBytes,
            params, extra, config, (void*)cfg->hStream);
        if (act == CIPHER_ATTN_SUPPRESS) {
            return (CUresult)0;
        }
        // REVERT_AND_PASS: FSM already replayed saved calls on this stream;
        // we still need to run the current kernel normally, so fall through.
    }

    // Stage 14 — full kernel table (Task 1).  Hot-path probe BEFORE the
    // persist fast-path short-circuit so every distinct CUfunction gets
    // recorded at least once.  The hash-table lookup is a single cache
    // line on the warm path; first-observation classification + param
    // probe is one-shot per fn handle.
    cipher_kt_observe((void*)f,
        cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
        cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
        cfg->sharedMemBytes);

    // Persistent-mode fast path: skip classify/dispatch/ring on stable
    // repeating sequences. Must not run when the GEMM intercept set the
    // shape TLS, because the cuBLAS-internal launches are the backing
    // kernels for a GEMM we're planning to substitute.
    uint64_t persist_fp = 0;
    bool persist_fast = false;
    if (!tls_shape_valid) {
        persist_fp = cipher_persist_fingerprint(
            (const void*)f,
            cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
            cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
            cfg->sharedMemBytes);
        persist_fast = cipher_persist_try_fast_path(persist_fp);
        if (persist_fast) {
            return g_real_cuLaunchEx(config, f, params, extra);
        }
    }

    // Track 2: kernel-name recognition (legacy KK table).
    KnownKernel* kk = track_kernel(f);

    // KV-compress driver-level pointer-flow probe. Logs FlashAttention's first
    // 6 kernelParams as 8-byte values when CIPHER_KERNEL_LOG=1, so we can
    // confirm K_proj/V_proj GEMM output pointers appear in FA's params.
    if (kk && kk->op_class == 7 && params
        && g_kk_log_enabled.load(std::memory_order_relaxed) == 1) {
        static int fa_log = 0;
        if (fa_log < 20) {
            fprintf(stderr, "[CIPHER FA-PARAMS-EX] #%d fn=%p name=%s",
                fa_log, (void*)f, kk->name);
            for (int i = 0; i < 6; ++i) {
                void* slot = params[i];
                void* p = slot ? *(void**)slot : (void*)0;
                fprintf(stderr, " p%d=%p", i, p);
            }
            fprintf(stderr, "\n");
            fa_log++;
        }
    }

    // Store launch args in TLS so apply_recipe can re-launch
    tls_launch_config = config;
    tls_launch_fn     = (void*)f;
    tls_launch_params = params;
    tls_launch_extra  = extra;
    tls_launch_valid  = 1;
    // dispatch_and_log decides internally whether to skip classify when
    // tls_shape_valid==1 (cuBLAS-internal launch). Ring write still happens.
    int result = dispatch_and_log(f,
        cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
        cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
        cfg->sharedMemBytes, (CUstream)cfg->hStream, params, extra);
    tls_launch_valid = 0;
    if (result == HOOK_SUBSTITUTED)
        return CUDA_SUCCESS; // recipe already computed the result

    // Stage 3 actuation: inject CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW.
    // No-op when persist engine is OFF or has no admitted region. When ON,
    // we build an augmented attr array on the stack and call the real
    // cuLaunchKernelEx with it.
    auto top_window_fn = g_persist_top_window_fn.load(std::memory_order_relaxed);
    auto en_fn         = g_persist_enabled_fn.load(std::memory_order_relaxed);
    if (__builtin_expect(en_fn && top_window_fn && en_fn(), 0)) {
        char winbuf[32];
        if (top_window_fn(winbuf)) {
            const unsigned origN = cfg->numAttrs;
            const unsigned newN  = origN + 1;
            CipherLaunchAttrShadow* newAttrs =
                (CipherLaunchAttrShadow*)alloca(sizeof(CipherLaunchAttrShadow) * newN);
            if (origN > 0 && cfg->attrs) {
                memcpy(newAttrs, cfg->attrs,
                       sizeof(CipherLaunchAttrShadow) * origN);
            }
            memset(&newAttrs[origN], 0, sizeof(CipherLaunchAttrShadow));
            newAttrs[origN].id = CIPHER_CU_LAUNCH_ATTR_ACCESS_POLICY_WINDOW;
            memcpy(newAttrs[origN].value, winbuf, 32);
            CUlaunchConfigMirror newCfg = *cfg;
            newCfg.attrs    = newAttrs;
            newCfg.numAttrs = newN;
            CUresult r = g_real_cuLaunchEx(&newCfg, f, params, extra);
            if (!tls_shape_valid && r == CUDA_SUCCESS) cipher_persist_observe(persist_fp);
            return r;
        }
    }

    // B.6''.9.8.5b: capture-aware substitution on the Ex main path.
    f = (CUfunction)cipher_rt_capture_check_and_substitute(
        (void*)f, cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
        cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
        cfg->sharedMemBytes, (void*)cfg->hStream);

    CUresult ret = g_real_cuLaunchEx(config, f, params, extra);
    if (!tls_shape_valid && ret == CUDA_SUCCESS) {
        cipher_persist_observe(persist_fp);
    }
    return ret;
}

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernelEx_ptsz(void* config, CUfunction f, void** params, void** extra)
{
    ENSURE_REAL(g_real_cuLaunchEx_ptsz);
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    // B.6''.9.8.5b: capture-aware substitution. Read dims from config.
    auto* cfg = (CUlaunchConfigMirror*)config;
    f = (CUfunction)cipher_rt_capture_check_and_substitute(
        (void*)f, cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
        cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
        cfg->sharedMemBytes, (void*)cfg->hStream);
    return g_real_cuLaunchEx_ptsz(config, f, params, extra);
}

// Shared body for cudaLaunchKernel + __cudaLaunchKernel (and _ptsz variants).
// `real_fn` is the cudart entry to forward to after CIPHER's dispatch chain.
// torch 2.7 routes nvcc-host-stub launches through cudaLaunchKernel; torch 2.11
// routes them through __cudaLaunchKernel — same signature, different cudart
// entry point. We need to intercept BOTH so the actuators see every kernel.
static cudaError_t cudaLaunchKernel_dispatch(
    cudaLaunchKernel_fn real_fn,
    const void* func, dim3 gd, dim3 bd,
    void** args, size_t smem, cudaStream_t stream)
{
    // Bug-3 hook-level fast-out: same CIPHER_HOOK_PASSIVE knob as the
    // cuLaunchKernel and cublasGemmEx shims. PyTorch's CUDA runtime
    // path lands here, NOT in cuLaunchKernel — every kernel a PyTorch
    // model launches goes through cudaLaunchKernel_dispatch, so this
    // is the dominant path for production inference on torch.
    static const bool s_passive = []() {
        const char* env = getenv("CIPHER_HOOK_PASSIVE");
        return env && *env && env[0] != '0';
    }();
    if (__builtin_expect(s_passive, 0)) {
        if (__builtin_expect(!real_fn, 0)) {
            resolve_real();
            if (!real_fn) return (cudaError_t)999;
        }
        return real_fn(func, gd, bd, args, smem, stream);
    }

    // Shim-overhead bracket — excludes the real launch.
    uint64_t tsc_entry = __rdtsc();

    if (__builtin_expect(!real_fn, 0)) { resolve_real(); auto_repatch_once();
        if (!real_fn) return (cudaError_t)999; }
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);

    // Track 2: runtime-API kernel name via dladdr on host stub. The host
    // stub is in libtorch_cuda.so's symbol table for PyTorch kernels.
    KnownKernel* kk_rt = track_kernel((CUfunction)func);

    // Stage 14 — full kernel table (Task 1) on the runtime-API path.
    // PyTorch's RMSNorm / SiLU / RoPE / elementwise / etc. all enter
    // here; this is where the dispatcher needs to recognize them.
    cipher_kt_observe((void*)func,
        gd.x, gd.y, gd.z, bd.x, bd.y, bd.z, (uint32_t)smem);

    // Stage 1 — pointer-flow recorder (gated by CIPHER_FLOW_RECORD).
    // Records {func, grid, block, smem, args[0..7]} into a circular ring
    // for downstream pattern matching.  No-op when disabled.
    cipher_flow_recorder_observe((void*)func,
        gd.x, gd.y, gd.z, bd.x, bd.y, bd.z, (uint32_t)smem,
        args, (void*)stream);

    // Stage 2 — pointer-flow pattern matcher (gated by CIPHER_FLOW_MATCH).
    // Anchor-driven state machine.  Detects RMSNorm-shaped sequences and
    // records recipes for Stage 3 substitution.
    cipher_flow_patterns_check((void*)func,
        gd.x, gd.y, gd.z, bd.x, bd.y, bd.z, (uint32_t)smem,
        args, (void*)stream);

    // Stage 3 — flow-substitute decision (gated by CIPHER_FLOW_SUBSTITUTE).
    // Returns 1 if this launch should be SUPPRESSED (we replace it with
    // a single fused kernel and skip the real cuBLAS launch).
    if (cipher_flow_substitute_consider((void*)func,
            gd.x, gd.y, gd.z, bd.x, bd.y, bd.z, (uint32_t)smem,
            args, (void*)stream)) {
        // Suppress this launch.  Bookkeeping only — the fused kernel was
        // already issued inside _consider() when it returned 1.
        g_shim_tsc_total.fetch_add(__rdtsc() - tsc_entry,
                                    std::memory_order_relaxed);
        return (cudaError_t)0;   // cudaSuccess
    }

    // Kernel parameter PROBE — for fusion template discovery.  Logs
    // mangled name + first 16 args[] entries (as raw uint64) for kernels
    // whose name matches a fusion-target pattern.  Each pattern is
    // sampled at most N_SAMPLE_PER_PATTERN times so we don't drown.
    {
        static int probe_on = -1;
        if (probe_on < 0) probe_on = getenv("CIPHER_KERNEL_PROBE") ? 1 : 0;
        if (probe_on == 1 && kk_rt && kk_rt->name && args) {
            const char* nm = kk_rt->name;
            const char* tag = nullptr;
            // Pattern-match against fusion targets.
            if (strstr(nm, "RMSNorm") || strstr(nm, "rms_norm")
                || strstr(nm, "rmsnorm")) tag = "RMSNORM";
            else if (strstr(nm, "rotary") || strstr(nm, "Rotary")
                || strstr(nm, "rope")) tag = "ROPE";
            else if (strstr(nm, "silu") || strstr(nm, "SiLU")
                || strstr(nm, "Silu") || strstr(nm, "swish")) tag = "SILU";
            else if (strstr(nm, "BinaryFunctor") &&
                     (strstr(nm, "Add") || strstr(nm, "add"))) tag = "ADD";
            else if (strstr(nm, "BinaryFunctor") &&
                     (strstr(nm, "Mul") || strstr(nm, "mul"))) tag = "MUL";
            else if (strstr(nm, "vectorized_elementwise")) tag = "VEC_ELEM";
            if (tag) {
                static std::atomic<int> s_count_RMSNORM{0};
                static std::atomic<int> s_count_ROPE{0};
                static std::atomic<int> s_count_SILU{0};
                static std::atomic<int> s_count_ADD{0};
                static std::atomic<int> s_count_MUL{0};
                static std::atomic<int> s_count_VEC{0};
                std::atomic<int>* cnt = nullptr;
                if (!strcmp(tag, "RMSNORM")) cnt = &s_count_RMSNORM;
                else if (!strcmp(tag, "ROPE")) cnt = &s_count_ROPE;
                else if (!strcmp(tag, "SILU")) cnt = &s_count_SILU;
                else if (!strcmp(tag, "ADD"))  cnt = &s_count_ADD;
                else if (!strcmp(tag, "MUL"))  cnt = &s_count_MUL;
                else                            cnt = &s_count_VEC;
                int c = cnt->fetch_add(1, std::memory_order_relaxed);
                if (c < 3) {
                    fprintf(stderr,
                        "[CIPHER PROBE %s #%d] grid=%dx%dx%d block=%dx%dx%d "
                        "smem=%zu fn=%p name=%.140s\n",
                        tag, c,
                        gd.x, gd.y, gd.z, bd.x, bd.y, bd.z,
                        smem, func, nm);
                    // Read first 16 args entries — args[i] points to the
                    // value of the i-th kernel parameter.  We dereference
                    // each as a uint64_t for inspection (most params are
                    // either ptrs or small ints; bigger values are
                    // truncated but the first 8 bytes usually carry the
                    // identifying info).
                    for (int i = 0; i < 16; ++i) {
                        if (!args[i]) {
                            fprintf(stderr, "  args[%2d] = NULL\n", i);
                            break;
                        }
                        // args[i] is a pointer to the parameter value on
                        // the caller's stack.  Read 8 bytes from it.
                        uint64_t v = 0;
                        memcpy(&v, args[i], sizeof(uint64_t));
                        // Heuristic classification.
                        const char* hint = "";
                        if (v >= 0x100000000ULL && v < 0x800000000000ULL) {
                            hint = " (looks-like-dev-ptr)";
                        } else if (v < (1ULL << 32)) {
                            hint = " (small-int)";
                        }
                        fprintf(stderr,
                            "  args[%2d] @ %p = 0x%016lx%s\n",
                            i, args[i], (unsigned long)v, hint);
                    }
                }
            }
        }
    }

    // KV-compress inter-kernel sequence probe. Logs every cudaLaunchKernel
    // call during decode with name + first 2 args interpreted as device ptrs.
    // Heavy output — gated by env CIPHER_KERNEL_TRACE.
    {
        static int trace_on = -1;
        if (trace_on < 0) trace_on = getenv("CIPHER_KERNEL_TRACE") ? 1 : 0;
        static int trace_n = 0;
        // Only trace post-prefill (Mistral has 32 layers → 32 FA calls in prefill).
        extern std::atomic<int> g_fa_call_count;
        if (trace_on == 1 && trace_n < 600 && kk_rt
            && g_fa_call_count.load(std::memory_order_relaxed) >= 32) {
            fprintf(stderr, "[CIPHER TRACE] #%d cls=%s fn=%p name=%.110s\n",
                trace_n, op_class_label(kk_rt->op_class), func, kk_rt->name);
            trace_n++;
        }
    }

    // KV-compress driver-level pointer-flow probe (cudart path).
    if (kk_rt && kk_rt->op_class == 7 && args && args[0]) {
        int fa_pre = g_fa_call_count.fetch_add(1, std::memory_order_relaxed);
        if (g_kk_log_enabled.load(std::memory_order_relaxed) == 1) {
            static int fa_log_rt = 0;
            if (fa_log_rt < 256) {
                uintptr_t* st = (uintptr_t*)args[0];
                fprintf(stderr, "[CIPHER FA-PARAMS-RT] #%d fn=%p struct=%p",
                    fa_log_rt, func, (void*)st);
                for (int i = 0; i < 8; ++i) {
                    fprintf(stderr, " w%d=0x%lx", i, (unsigned long)st[i]);
                }
                fprintf(stderr, "\n");
                fa_log_rt++;
            }
        }
        // Path A V1: rewrite struct.K/V to point at CIPHER side buffers.
        // Layer = (call count) mod num_layers; for Mistral-7B this is 32,
        // and the prefill pass goes through all 32 layers before the first
        // decode token, so (fa_pre % 32) gives the right layer regardless
        // of phase.
        typedef int (*kv_rdr_fn)(void*, int, void*);
        static kv_rdr_fn s_rdr = nullptr;
        static int       s_rdr_resolved = 0;
        if (!s_rdr_resolved) {
            s_rdr = (kv_rdr_fn)dlsym(RTLD_DEFAULT, "cipher_kv_redirect_on_fa_launch");
            s_rdr_resolved = 1;
        }
        if (s_rdr) {
            int layer = fa_pre % 32;
            s_rdr(args[0], layer, (void*)stream);
        }
    }

    // Persistent-mode fast path — bypass classify/dispatch/ring on stable
    // repeating sequences. Only valid outside GEMM intercept windows and
    // outside recursive dispatch.
    uint64_t persist_fp = 0;
    bool persist_fast = false;
    if (!tls_in_recipe && !tls_shape_valid) {
        persist_fp = cipher_persist_fingerprint(
            func, gd.x, gd.y, gd.z, bd.x, bd.y, bd.z, (unsigned)smem);
        persist_fast = cipher_persist_try_fast_path(persist_fp);
        if (persist_fast) {
            uint64_t tsc_before_real = __rdtsc();
            cudaError_t ret = real_fn(func, gd, bd, args, smem, stream);
            uint64_t tsc_after_real = __rdtsc();
            g_shim_tsc_total.fetch_add(
                (tsc_before_real - tsc_entry) + (__rdtsc() - tsc_after_real),
                std::memory_order_relaxed);
            g_shim_tsc_calls.fetch_add(1, std::memory_order_relaxed);
            return ret;
        }
    }

    // Dispatch non-GEMM kernels only. GEMM classification and billing is fully
    // handled by the cublasGemmEx shim — dispatching GEMM kernels again here
    // would be pure overhead (~2us classify + oracle per call).
    // Skip when tls_shape_valid=1 (GEMM shim already processed this call).
    static thread_local int in_dispatch = 0;
    if (!in_dispatch && !tls_in_recipe && !tls_shape_valid) {
        in_dispatch = 1;
        dispatch_and_log(
            (CUfunction)func, gd.x, gd.y, gd.z, bd.x, bd.y, bd.z,
            (unsigned)smem, (CUstream)stream, args, nullptr);
        in_dispatch = 0;
    }

    uint64_t tsc_before_real = __rdtsc();
    cudaError_t ret = real_fn(func, gd, bd, args, smem, stream);
    uint64_t tsc_after_real = __rdtsc();

    if (!tls_in_recipe && !tls_shape_valid && ret == 0) {
        cipher_persist_observe(persist_fp);
    }
    g_shim_tsc_total.fetch_add(
        (tsc_before_real - tsc_entry) + (__rdtsc() - tsc_after_real),
        std::memory_order_relaxed);
    g_shim_tsc_calls.fetch_add(1, std::memory_order_relaxed);
    return ret;
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaLaunchKernel(const void* func, dim3 gd, dim3 bd,
    void** args, size_t smem, cudaStream_t stream)
{
    return cudaLaunchKernel_dispatch(g_real_cudaLaunch,
                                      func, gd, bd, args, smem, stream);
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaLaunchKernel_ptsz(const void* func, dim3 gd, dim3 bd,
    void** args, size_t smem, cudaStream_t stream)
{
    return cudaLaunchKernel_dispatch(g_real_cudaLaunch_ptsz,
                                      func, gd, bd, args, smem, stream);
}

// torch 2.11 / cudart 13: nvcc-generated host stubs route through these.
extern "C" __attribute__((visibility("default")))
cudaError_t __cudaLaunchKernel(const void* func, dim3 gd, dim3 bd,
    void** args, size_t smem, cudaStream_t stream)
{
    return cudaLaunchKernel_dispatch(g_real___cudaLaunch,
                                      func, gd, bd, args, smem, stream);
}

extern "C" __attribute__((visibility("default")))
cudaError_t __cudaLaunchKernel_ptsz(const void* func, dim3 gd, dim3 bd,
    void** args, size_t smem, cudaStream_t stream)
{
    return cudaLaunchKernel_dispatch(g_real___cudaLaunch_ptsz,
                                      func, gd, bd, args, smem, stream);
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args)
{
    ENSURE_RT(g_real_cudaLaunchExC);
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);

    uint64_t persist_fp = 0;
    bool persist_fast = false;
    if (!tls_in_recipe && !tls_shape_valid && config) {
        persist_fp = cipher_persist_fingerprint(
            func,
            config->gridDim.x,  config->gridDim.y,  config->gridDim.z,
            config->blockDim.x, config->blockDim.y, config->blockDim.z,
            (unsigned)config->dynamicSmemBytes);
        persist_fast = cipher_persist_try_fast_path(persist_fp);
        if (persist_fast) {
            return g_real_cudaLaunchExC(config, func, args);
        }
    }

    cudaError_t ret = g_real_cudaLaunchExC(config, func, args);
    if (!tls_in_recipe && !tls_shape_valid && ret == 0) {
        cipher_persist_observe(persist_fp);
    }
    return ret;
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t* config, const void* func, void** args)
{
    ENSURE_RT(g_real_cudaLaunchExC_ptsz);
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    return g_real_cudaLaunchExC_ptsz(config, func, args);
}

extern "C" __attribute__((visibility("default")))
CUresult cuGetProcAddress(const char* symbol, void** pfn, int ver,
    uint64_t flags, CUdriverProcAddressQueryResult* status)
{
    ENSURE_REAL(g_real_cuGetProc);
    g_proc_addr_count.fetch_add(1, std::memory_order_relaxed);
    CUresult r = g_real_cuGetProc(symbol, pfn, ver, flags, status);
    if (r == CUDA_SUCCESS && pfn && symbol) {
        void* s = get_shim(symbol);
        if (s) *pfn = s;
    }
    return r;
}

extern "C" __attribute__((visibility("default")))
CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int ver,
    uint64_t flags, CUdriverProcAddressQueryResult* status)
{
    cuGetProcAddress_fn real = g_real_cuGetProc_v2 ? g_real_cuGetProc_v2 : g_real_cuGetProc;
    if (!real) { resolve_real(); real = g_real_cuGetProc_v2 ? g_real_cuGetProc_v2 : g_real_cuGetProc; }
    if (!real) return CUDA_ERROR_NOT_FOUND;
    g_proc_addr_count.fetch_add(1, std::memory_order_relaxed);
    CUresult r = real(symbol, pfn, ver, flags, status);
    if (r == CUDA_SUCCESS && pfn && symbol) {
        void* s = get_shim(symbol);
        if (s) *pfn = s;
    }
    return r;
}

extern "C" __attribute__((visibility("default")))
uint64_t cipher_intercept_count(void) {
    return g_intercept_count.load(std::memory_order_relaxed);
}

// ── B.6''.9.8.4 cuGraphLaunch + cuGraphLaunchPipelined shim implementations ─
//
// Graph-replay observability. Forward declarations + typedefs hoisted to top
// of file (search "B.6''.9.8.4 forward"). Impl below.

static inline void cipher_rt_graph_launch_observe(void* stream) {
    g_graph_launch_calls.fetch_add(1, std::memory_order_relaxed);
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    if (cipher_v2_current_tenant_id_from_stream && cipher_rt_commit_observe_and_publish) {
        uint32_t tid = cipher_v2_current_tenant_id_from_stream((uintptr_t)stream);
        cipher_rt_commit_observe_and_publish(tid);
    }
}

extern "C" __attribute__((visibility("default")))
CUresult cuGraphLaunch(void* exec, void* stream) {
    if (__builtin_expect(!g_real_cuGraphLaunch, 0)) {
        resolve_real();
        if (!g_real_cuGraphLaunch) return (CUresult)500; // CUDA_ERROR_NOT_FOUND
    }
    cipher_rt_graph_launch_observe(stream);
    /* K.1: graph launch implies capture/replay mode is active in this
     * process. Set flag once; classifier reads at next reclassification. */
    cipher_workload_observe_capture_begin();
    return g_real_cuGraphLaunch(exec, stream);
}

extern "C" __attribute__((visibility("default")))
CUresult cuGraphLaunch_ptsz(void* exec, void* stream) {
    if (__builtin_expect(!g_real_cuGraphLaunch_ptsz, 0)) {
        resolve_real();
        if (!g_real_cuGraphLaunch_ptsz)
            g_real_cuGraphLaunch_ptsz = g_real_cuGraphLaunch;
        if (!g_real_cuGraphLaunch_ptsz) return (CUresult)500;
    }
    cipher_rt_graph_launch_observe(stream);
    return g_real_cuGraphLaunch_ptsz(exec, stream);
}

extern "C" __attribute__((visibility("default")))
CUresult cuGraphLaunchPipelined(void* exec, void* phStreamArray, void* stream) {
    if (__builtin_expect(!g_real_cuGraphLaunchPipelined, 0)) {
        resolve_real();
        if (!g_real_cuGraphLaunchPipelined) return (CUresult)500;
    }
    cipher_rt_graph_launch_observe(stream);
    return g_real_cuGraphLaunchPipelined(exec, phStreamArray, stream);
}

// B.6''.9.8.4 cudart-level wrapper. PyTorch's CUDAGraph.replay() calls
// cudaGraphLaunch (cudart) which internally calls cuGraphLaunch (driver).
// cudart caches the real cuGraphLaunch pointer at libcudart init time —
// BEFORE our cuGetProcAddress hook can redirect — so driver-level
// cuGraphLaunch shim alone is insufficient. We also shim cudaGraphLaunch
// directly via GOT patching of cudart's GOT entry.
extern "C" __attribute__((visibility("default")))
cudaError_t cudaGraphLaunch(void* exec, void* stream) {
    if (__builtin_expect(!g_real_cudaGraphLaunch, 0)) {
        resolve_real();
        if (!g_real_cudaGraphLaunch) return (cudaError_t)999;
    }
    cipher_rt_graph_launch_observe(stream);
    return g_real_cudaGraphLaunch(exec, stream);
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaGraphLaunch_ptsz(void* exec, void* stream) {
    if (__builtin_expect(!g_real_cudaGraphLaunch_ptsz, 0)) {
        resolve_real();
        if (!g_real_cudaGraphLaunch_ptsz)
            g_real_cudaGraphLaunch_ptsz = g_real_cudaGraphLaunch;
        if (!g_real_cudaGraphLaunch_ptsz) return (cudaError_t)999;
    }
    cipher_rt_graph_launch_observe(stream);
    return g_real_cudaGraphLaunch_ptsz(exec, stream);
}

extern "C" __attribute__((visibility("default")))
uint64_t cipher_rt_graph_launch_calls(void) {
    return g_graph_launch_calls.load(std::memory_order_relaxed);
}

// ── B.6''.9.8.5 cuGraphAddKernelNode + cuGraphExecKernelNodeSetParams ──────
//
// Capture-time kernel-node substitution. Marlin/Koopman/VOLT actuators can
// transform CUfunction in the params struct before the graph is sealed at
// instantiate. Replays then execute the substituted topology natively —
// no per-launch hook overhead at replay time.
//
// Substitution decision is delegated to cipher_rt_graph_substitute_decide
// (weak ref). NULL or weak-unresolved → no substitution; func untouched;
// graph correctness preserved bit-for-bit vs no-shim baseline.
//
// ABI-compat discipline: substitute kernel MUST accept the same grid/block
// dims and same kernelParams layout. Decision function inspects those dims
// and returns NULL if no compatible substitute exists. We only mutate
// params->func — never the dims or kernelParams pointers.

static inline void cipher_rt_graph_node_substitute_maybe(CipherKernelNodeParamsShim* params) {
    g_graph_add_kernel_calls.fetch_add(1, std::memory_order_relaxed);
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    if (!params || !params->func) return;
    if (!cipher_rt_graph_substitute_decide) return;
    void* sub = cipher_rt_graph_substitute_decide(
        params->func,
        params->gridDimX, params->gridDimY, params->gridDimZ,
        params->blockDimX, params->blockDimY, params->blockDimZ,
        params->sharedMemBytes);
    if (sub && sub != params->func) {
        params->func = sub;
        g_graph_node_substitutions.fetch_add(1, std::memory_order_relaxed);
    }
}

extern "C" __attribute__((visibility("default")))
CUresult cuGraphAddKernelNode(void* node, void* graph, const void* deps, size_t depCount,
                              CipherKernelNodeParamsShim* params) {
    if (__builtin_expect(!g_real_cuGraphAddKernelNode, 0)) {
        resolve_real();
        if (!g_real_cuGraphAddKernelNode) return (CUresult)500;
    }
    cipher_rt_graph_node_substitute_maybe(params);
    return g_real_cuGraphAddKernelNode(node, graph, deps, depCount, params);
}

extern "C" __attribute__((visibility("default")))
CUresult cuGraphAddKernelNode_v2(void* node, void* graph, const void* deps, size_t depCount,
                                 CipherKernelNodeParamsShim* params) {
    if (__builtin_expect(!g_real_cuGraphAddKernelNode_v2, 0)) {
        resolve_real();
        if (!g_real_cuGraphAddKernelNode_v2)
            g_real_cuGraphAddKernelNode_v2 = g_real_cuGraphAddKernelNode;
        if (!g_real_cuGraphAddKernelNode_v2) return (CUresult)500;
    }
    cipher_rt_graph_node_substitute_maybe(params);
    return g_real_cuGraphAddKernelNode_v2(node, graph, deps, depCount, params);
}

extern "C" __attribute__((visibility("default")))
CUresult cuGraphExecKernelNodeSetParams(void* exec, void* node,
                                        const CipherKernelNodeParamsShim* params) {
    if (__builtin_expect(!g_real_cuGraphExecKernelNodeSetParams, 0)) {
        resolve_real();
        if (!g_real_cuGraphExecKernelNodeSetParams) return (CUresult)500;
    }
    // Post-instantiate path: caller already owns the params memory. We make
    // a local mutable copy so the substitute decision can mutate func
    // without aliasing the caller's const view. ABI-equiv (same byte layout).
    CipherKernelNodeParamsShim local = *params;
    cipher_rt_graph_node_substitute_maybe(&local);
    return g_real_cuGraphExecKernelNodeSetParams(exec, node, &local);
}

extern "C" __attribute__((visibility("default")))
CUresult cuGraphExecKernelNodeSetParams_v2(void* exec, void* node,
                                           const CipherKernelNodeParamsShim* params) {
    if (__builtin_expect(!g_real_cuGraphExecKernelNodeSetParams_v2, 0)) {
        resolve_real();
        if (!g_real_cuGraphExecKernelNodeSetParams_v2)
            g_real_cuGraphExecKernelNodeSetParams_v2 = g_real_cuGraphExecKernelNodeSetParams;
        if (!g_real_cuGraphExecKernelNodeSetParams_v2) return (CUresult)500;
    }
    CipherKernelNodeParamsShim local = *params;
    cipher_rt_graph_node_substitute_maybe(&local);
    return g_real_cuGraphExecKernelNodeSetParams_v2(exec, node, &local);
}

// cudart-level wrapper. PyTorch may call cudaGraphAddKernelNode (cudart)
// which internally calls cuGraphAddKernelNode (driver). Same cache pattern
// as cudaGraphLaunch from B.6''.9.8.4 — must intercept at cudart level too.
extern "C" __attribute__((visibility("default")))
cudaError_t cudaGraphAddKernelNode(void* node, void* graph, const void* deps, size_t depCount,
                                   const CipherKernelNodeParamsShim* params) {
    if (__builtin_expect(!g_real_cudaGraphAddKernelNode, 0)) {
        resolve_real();
        if (!g_real_cudaGraphAddKernelNode) return (cudaError_t)999;
    }
    CipherKernelNodeParamsShim local = *params;
    cipher_rt_graph_node_substitute_maybe(&local);
    return g_real_cudaGraphAddKernelNode(node, graph, deps, depCount, &local);
}

extern "C" __attribute__((visibility("default")))
uint64_t cipher_rt_graph_add_kernel_calls(void) {
    return g_graph_add_kernel_calls.load(std::memory_order_relaxed);
}

extern "C" __attribute__((visibility("default")))
uint64_t cipher_rt_graph_node_substitutions(void) {
    return g_graph_node_substitutions.load(std::memory_order_relaxed);
}


// ── B.6''.9.8.1 init entry ──────────────────────────────────────────────────
//
// may13's __attribute__((constructor(101))) cipher_hook_init() REPLACED with
// an explicit init function called from cipher_inject.c:cipher_v2_init_body()
// AFTER cipher_rt_got_patch_init(). pthread_once in the caller gates
// idempotency. This preserves init order with cipher_rt_phase4's existing
// cuBLAS GOT patcher (which is canonical for cuBLAS).

static std::atomic<bool> g_intercept_initialized{false};

extern "C" __attribute__((visibility("default")))
void cipher_rt_intercept_init(void) {
    bool expected = false;
    if (!g_intercept_initialized.compare_exchange_strong(expected, true)) return;
    g_page_size = sysconf(_SC_PAGESIZE);
    fprintf(stderr, "[CIPHER MAY13-INTERCEPT] Init — cuGetProcAddress + cuLaunch shims via GOT-patch (B.6''.9.8.1)\n");
    // Attempt early patch (resolves real fn pointers + walks GOT)
    resolve_real();
    if (g_real_cuLaunch)
        dl_iterate_phdr(phdr_cb, nullptr);
    // Stage 2: silicon model — fail-soft if libcipher_rt.so isn't preloaded.
    if (auto silicon_init = reinterpret_cast<int (*)(void)>(
            dlsym(RTLD_DEFAULT, "cipher_silicon_init"))) {
        silicon_init();
    }
}

// ── cudaMemcpy{,Async} probe — for weight-load tracking ────────────────────
// Logs H2D copies during model load.  Goal: identify the per-layer RMSNorm
// weight pointers (each ~hidden_dim*2 bytes, copied once at load time).
typedef cudaError_t (*cudaMemcpy_fn_real)(void*, const void*, size_t, int);
typedef cudaError_t (*cudaMemcpyAsync_fn_real)(void*, const void*, size_t, int, void*);
static cudaMemcpy_fn_real      g_real_cudaMemcpy = nullptr;
static cudaMemcpyAsync_fn_real g_real_cudaMemcpyAsync = nullptr;
static std::atomic<uint64_t> g_memcpy_calls{0};
static std::atomic<uint64_t> g_memcpy_h2d_count{0};

static void log_memcpy_if_probing(void* dst, const void* src, size_t size,
                                   int kind, const char* api) {
    static int probe_on = -1;
    if (probe_on < 0) probe_on = getenv("CIPHER_MEMCPY_PROBE") ? 1 : 0;
    if (probe_on != 1) return;
    uint64_t h = g_memcpy_h2d_count.fetch_add(1, std::memory_order_relaxed);
    if (h < 2000) {
        const char* kstr = (kind == 0) ? "H2H" : (kind == 1) ? "H2D" :
                           (kind == 2) ? "D2H" : (kind == 3) ? "D2D" :
                           (kind == 4) ? "DEF" : "?";
        fprintf(stderr,
            "[CIPHER MEMCPY] #%lu  api=%s  kind=%s  dst=%p  src=%p  size=%zu\n",
            (unsigned long)h, api, kstr, dst, src, size);
    }
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, int kind) {
    if (__builtin_expect(!g_real_cudaMemcpy, 0)) {
        void* h = resolve_cudart();
        if (h) g_real_cudaMemcpy = (cudaMemcpy_fn_real)dlsym(h, "cudaMemcpy");
        if (!g_real_cudaMemcpy) return (cudaError_t)999;
    }
    g_memcpy_calls.fetch_add(1, std::memory_order_relaxed);
    log_memcpy_if_probing(dst, src, size, kind, "Memcpy");
    return g_real_cudaMemcpy(dst, src, size, kind);
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t size,
                             int kind, void* stream) {
    if (__builtin_expect(!g_real_cudaMemcpyAsync, 0)) {
        void* h = resolve_cudart();
        if (h) g_real_cudaMemcpyAsync =
            (cudaMemcpyAsync_fn_real)dlsym(h, "cudaMemcpyAsync");
        if (!g_real_cudaMemcpyAsync) return (cudaError_t)999;
    }
    g_memcpy_calls.fetch_add(1, std::memory_order_relaxed);
    log_memcpy_if_probing(dst, src, size, kind, "MemcpyAsync");
    return g_real_cudaMemcpyAsync(dst, src, size, kind, stream);
}

__attribute__((destructor))
static void cipher_hook_fini() {
    uint64_t ic = g_intercept_count.load();
    uint64_t pc = g_proc_addr_count.load();
    uint64_t gp = g_got_patches.load();
    fprintf(stderr,
        "[CIPHER HOOK] Teardown. Intercepts: %lu | ProcAddr: %lu | GOT patches: %lu\n",
        ic, pc, gp);
    if (ic == 0) fprintf(stderr, "[CIPHER HOOK] WARNING: 0 intercepts fired!\n");

    // Print billing report if oracle is available
    typedef void (*billing_fn)(const void*);
    typedef void* (*get_oracle_fn)(void);
    get_oracle_fn go = (get_oracle_fn)dlsym(RTLD_DEFAULT, "cipher_get_oracle");
    billing_fn br = (billing_fn)dlsym(RTLD_DEFAULT, "cipher_oracle_billing_report");
    if (go && br) {
        void* oracle = go();
        if (oracle) br(oracle);
    }
}
