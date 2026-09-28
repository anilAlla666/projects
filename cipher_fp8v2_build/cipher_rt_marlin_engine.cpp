/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_marlin_engine.cpp -- T4.5.2 Marlin INT4 GEMM engine.
 *
 * Distilled from cipher-may13-evidence/src/cipher_weight_compress.cpp
 * (Marlin-only subset). Provides:
 *   - NVRTC compile of cipher_rt_marlin_kernel_src_str → 10 kernel
 *     entry points (M=1..4, N=8|16, K=4|8, G=n1|8)
 *   - Per-weight lazy quantization (host-side fp16→INT4+scales)
 *   - Marlin XOR-swizzled repack (host-side using cipher_rt_marlin_perm
 *     and cipher_rt_marlin_scale_perm LUTs)
 *   - Marlin GEMM dispatch (grid=SM_count, block=256, dyn_shared=96K)
 *
 * The actuator (cipher_rt_marlin_actuator.c) calls into this engine via
 * the C-linkage functions at the bottom.
 *
 * Original Marlin: IST-DASLab (Apache-2.0).
 */
#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <atomic>
#include <mutex>
#include <vector>
#include <unordered_map>

#include <dlfcn.h>

extern "C" {
#include "cipher_v2_internal.h"
#include "cipher_rt_marlin_kernel_src.h"
#include "cipher_rt_marlin.h"
}
#include "cipher_rt_marlin_perms.h"

/* CP 2.4 (2026-05-16) — cross-TU thread-local flag. PrimaryCtxGuard sets it
 * while Marlin's quant/repack path owns the calling thread; the CUPTI launch
 * callback (cipher_cupti.c) reads it and skips its per-launch green-context
 * enforcement for Marlin's OWN internal kernel launches — without that, the
 * callback re-greens the thread between our launches and the pin below would
 * not hold. See cp_2_4/MARLIN_HANG_ROOT_CAUSE.md. */
extern "C" { __thread int cipher_rt_in_marlin_quant = 0; }

/* CP 5.3 STEP 2 (2026-05-17) — green-context partition wiring. Returns the
 * verified SM count of this process's green context (8) when one is active,
 * else 0. Used to size the GEMM `grid` to the partition and to gate the
 * primary-context pin. Defined in cipher_rt_green_ctx.c; see STEP 2 scope
 * memo §3. */
extern "C" unsigned int cipher_rt_green_ctx_sm_count(void);

/* GC-on-free (2026-05-30): Layer-1 bf16→fp16 surrogate-map entry. Carries the
 * surrogate pointer plus the (K,N) and a content fingerprint of the *app* bf16
 * weight, so a reused bf16 weight pointer (model freed → new model loaded at
 * the same address) is detected and the stale surrogate is never returned.
 * Global scope (the map has external linkage; referenced via extern below). */
struct Bf16Entry { void *surrogate; uint64_t fp; int K; int N; };

/* ── NVRTC + CUDA driver function-pointer bindings (dlsym at init) ─────── */

namespace {

typedef int (*pf_nvrtcCreateProgram)(void**, const char*, const char*, int,
                                     const char* const*, const char* const*);
typedef int (*pf_nvrtcCompileProgram)(void*, int, const char* const*);
typedef int (*pf_nvrtcGetPTXSize)(void*, size_t*);
typedef int (*pf_nvrtcGetPTX)(void*, char*);
typedef int (*pf_nvrtcGetProgramLogSize)(void*, size_t*);
typedef int (*pf_nvrtcGetProgramLog)(void*, char*);
typedef int (*pf_nvrtcDestroyProgram)(void**);

typedef int (*pf_cuInit)(unsigned);
typedef int (*pf_cuModuleLoadData)(void**, const void*);
typedef int (*pf_cuModuleGetFunction)(void**, void*, const char*);
typedef int (*pf_cuLaunchKernel)(void*, unsigned, unsigned, unsigned,
                                 unsigned, unsigned, unsigned,
                                 unsigned, void*, void**, void**);
typedef int (*pf_cuFuncSetAttribute)(void*, int, int);
/* CUDA driver context API — CP 2.4 primary-context pin (see PrimaryCtxGuard). */
typedef int (*pf_cuCtxGetCurrent)(void**);
typedef int (*pf_cuCtxSetCurrent)(void*);
typedef int (*pf_cuDevicePrimaryCtxRetain)(void**, int);
/* CUDA driver event API — CP 5.6 P1 cross-context ordering (finding F1). */
typedef int (*pf_cuEventCreate)(void**, unsigned);
typedef int (*pf_cuEventRecord)(void*, void*);
typedef int (*pf_cuEventDestroy)(void*);
typedef int (*pf_cuStreamWaitEvent)(void*, void*, unsigned);
/* CUDA runtime API for context bring-up + memcpy + malloc */
typedef int (*pf_cudaFree)(void*);
typedef int (*pf_cudaMalloc)(void**, size_t);
/* G-O7 graph gate: stream-ordered alloc is CUDA-graph-capture-legal (records as a graph mem node);
 * plain cudaMalloc/Free invalidate capture. Used for the bf16 cast temps when CAPTURE-SAFE mode is on. */
typedef int (*pf_cudaMallocAsync)(void**, size_t, void*);
typedef int (*pf_cudaFreeAsync)(void*, void*);
typedef int (*pf_cudaMemcpy)(void*, const void*, size_t, int);
typedef int (*pf_cudaMemcpyAsync)(void*, const void*, size_t, int, void*);
typedef int (*pf_cudaMemset)(void*, int, size_t);
typedef int (*pf_cudaMemsetAsync)(void*, int, size_t, void*);
typedef int (*pf_cudaDeviceGetAttribute)(int*, int, int);
typedef int (*pf_cudaDeviceSynchronize)(void);
/* GC-on-free fingerprint: a dedicated non-blocking stream + pinned staging so
 * the per-call content sample does NOT serialize on the legacy default stream
 * (which would barrier every tenant's Marlin GEMM). All optional. */
typedef int (*pf_cudaStreamCreateWithFlags)(void**, unsigned);
typedef int (*pf_cudaStreamSynchronize)(void*);
typedef int (*pf_cudaHostAlloc)(void**, size_t, unsigned);

struct Api {
    void *libnvrtc = nullptr;
    void *libcuda  = nullptr;
    void *libcudart= nullptr;

    pf_nvrtcCreateProgram     create_prog  = nullptr;
    pf_nvrtcCompileProgram    compile_prog = nullptr;
    pf_nvrtcGetPTXSize        get_ptx_size = nullptr;
    pf_nvrtcGetPTX            get_ptx      = nullptr;
    pf_nvrtcGetProgramLogSize get_log_size = nullptr;
    pf_nvrtcGetProgramLog     get_log      = nullptr;
    pf_nvrtcDestroyProgram    destroy_prog = nullptr;

    pf_cuInit                 cu_init        = nullptr;
    pf_cuModuleLoadData       module_load    = nullptr;
    pf_cuModuleGetFunction    module_getfn   = nullptr;
    pf_cuLaunchKernel         launch_kernel  = nullptr;
    pf_cuFuncSetAttribute     func_setattr   = nullptr;
    pf_cuCtxGetCurrent          ctx_get        = nullptr;
    pf_cuCtxSetCurrent          ctx_set        = nullptr;
    pf_cuDevicePrimaryCtxRetain primary_retain = nullptr;
    pf_cuEventCreate            event_create   = nullptr;
    pf_cuEventRecord            event_record   = nullptr;
    pf_cuEventDestroy           event_destroy  = nullptr;
    pf_cuStreamWaitEvent        stream_wait_ev = nullptr;

    pf_cudaFree               rt_free        = nullptr;
    pf_cudaMalloc             rt_malloc      = nullptr;
    pf_cudaMallocAsync        rt_malloc_async = nullptr;  /* G-O7: capture-safe cast alloc */
    pf_cudaFreeAsync          rt_free_async   = nullptr;
    pf_cudaMemcpy             rt_memcpy      = nullptr;
    pf_cudaMemcpyAsync        rt_memcpy_async= nullptr;
    pf_cudaMemset             rt_memset      = nullptr;
    pf_cudaMemsetAsync        rt_memset_async= nullptr;
    pf_cudaDeviceGetAttribute rt_devattr     = nullptr;
    /* GC-on-free (2026-05-30): quiescent barrier for the retired-kit reclaim.
     * Optional — NOT in the `ok` required set, so resolve never newly fails. */
    pf_cudaDeviceSynchronize  rt_devsync     = nullptr;
    /* GC-on-free fingerprint sampling off the default stream (all optional;
     * absent ⇒ synchronous-DtoH fallback). */
    pf_cudaStreamCreateWithFlags rt_stream_create = nullptr;
    pf_cudaStreamSynchronize     rt_stream_sync   = nullptr;
    pf_cudaHostAlloc             rt_host_alloc    = nullptr;

    bool ok = false;
} g_api;
std::mutex g_api_mu;

bool resolve_api() {
    std::lock_guard<std::mutex> lk(g_api_mu);
    if (g_api.ok) return true;

    if (!g_api.libnvrtc) {
        g_api.libnvrtc = dlopen("libnvrtc.so.13", RTLD_LAZY | RTLD_LOCAL);
        if (!g_api.libnvrtc) g_api.libnvrtc = dlopen("libnvrtc.so.12", RTLD_LAZY | RTLD_LOCAL);
        if (!g_api.libnvrtc) g_api.libnvrtc = dlopen("libnvrtc.so",    RTLD_LAZY | RTLD_LOCAL);
    }
    if (!g_api.libcuda) {
        g_api.libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    }
    if (!g_api.libcudart) {
        g_api.libcudart = dlopen("libcudart.so.13", RTLD_LAZY | RTLD_NOLOAD);
        if (!g_api.libcudart) g_api.libcudart = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
        if (!g_api.libcudart) g_api.libcudart = dlopen("libcudart.so.13", RTLD_LAZY);
        if (!g_api.libcudart) g_api.libcudart = dlopen("libcudart.so.12", RTLD_LAZY);
        if (!g_api.libcudart) g_api.libcudart = dlopen("libcudart.so",    RTLD_LAZY);
    }
    if (!g_api.libnvrtc || !g_api.libcuda || !g_api.libcudart) {
        cipher_log("MARLIN: libnvrtc/libcuda/libcudart resolve failed "
                   "(nvrtc=%p cuda=%p cudart=%p)",
                   g_api.libnvrtc, g_api.libcuda, g_api.libcudart);
        return false;
    }

    g_api.create_prog  = (pf_nvrtcCreateProgram)dlsym(g_api.libnvrtc, "nvrtcCreateProgram");
    g_api.compile_prog = (pf_nvrtcCompileProgram)dlsym(g_api.libnvrtc, "nvrtcCompileProgram");
    g_api.get_ptx_size = (pf_nvrtcGetPTXSize)dlsym(g_api.libnvrtc, "nvrtcGetPTXSize");
    g_api.get_ptx      = (pf_nvrtcGetPTX)dlsym(g_api.libnvrtc, "nvrtcGetPTX");
    g_api.get_log_size = (pf_nvrtcGetProgramLogSize)dlsym(g_api.libnvrtc, "nvrtcGetProgramLogSize");
    g_api.get_log      = (pf_nvrtcGetProgramLog)dlsym(g_api.libnvrtc, "nvrtcGetProgramLog");
    g_api.destroy_prog = (pf_nvrtcDestroyProgram)dlsym(g_api.libnvrtc, "nvrtcDestroyProgram");

    g_api.cu_init       = (pf_cuInit)dlsym(g_api.libcuda, "cuInit");
    g_api.module_load   = (pf_cuModuleLoadData)dlsym(g_api.libcuda, "cuModuleLoadData");
    g_api.module_getfn  = (pf_cuModuleGetFunction)dlsym(g_api.libcuda, "cuModuleGetFunction");
    g_api.launch_kernel = (pf_cuLaunchKernel)dlsym(g_api.libcuda, "cuLaunchKernel");
    g_api.func_setattr  = (pf_cuFuncSetAttribute)dlsym(g_api.libcuda, "cuFuncSetAttribute");
    g_api.ctx_get       = (pf_cuCtxGetCurrent)dlsym(g_api.libcuda, "cuCtxGetCurrent");
    g_api.ctx_set       = (pf_cuCtxSetCurrent)dlsym(g_api.libcuda, "cuCtxSetCurrent");
    g_api.primary_retain= (pf_cuDevicePrimaryCtxRetain)dlsym(g_api.libcuda, "cuDevicePrimaryCtxRetain");
    /* CP 5.6 P1: event API for the full-GPU-path cross-context dependency. */
    g_api.event_create  = (pf_cuEventCreate)dlsym(g_api.libcuda, "cuEventCreate");
    g_api.event_record  = (pf_cuEventRecord)dlsym(g_api.libcuda, "cuEventRecord");
    g_api.event_destroy = (pf_cuEventDestroy)dlsym(g_api.libcuda, "cuEventDestroy_v2");
    if (!g_api.event_destroy)
        g_api.event_destroy = (pf_cuEventDestroy)dlsym(g_api.libcuda, "cuEventDestroy");
    g_api.stream_wait_ev= (pf_cuStreamWaitEvent)dlsym(g_api.libcuda, "cuStreamWaitEvent");

    g_api.rt_free        = (pf_cudaFree)dlsym(g_api.libcudart, "cudaFree");
    g_api.rt_malloc      = (pf_cudaMalloc)dlsym(g_api.libcudart, "cudaMalloc");
    g_api.rt_malloc_async= (pf_cudaMallocAsync)dlsym(g_api.libcudart, "cudaMallocAsync"); /* G-O7 */
    g_api.rt_free_async  = (pf_cudaFreeAsync)dlsym(g_api.libcudart, "cudaFreeAsync");
    g_api.rt_memcpy      = (pf_cudaMemcpy)dlsym(g_api.libcudart, "cudaMemcpy");
    g_api.rt_memcpy_async= (pf_cudaMemcpyAsync)dlsym(g_api.libcudart, "cudaMemcpyAsync");
    g_api.rt_memset      = (pf_cudaMemset)dlsym(g_api.libcudart, "cudaMemset");
    g_api.rt_memset_async= (pf_cudaMemsetAsync)dlsym(g_api.libcudart, "cudaMemsetAsync");
    g_api.rt_devattr     = (pf_cudaDeviceGetAttribute)dlsym(g_api.libcudart, "cudaDeviceGetAttribute");
    /* GC-on-free reclaim barrier (optional; absence only disables reclaim-free,
     * never the actuator). */
    g_api.rt_devsync     = (pf_cudaDeviceSynchronize)dlsym(g_api.libcudart, "cudaDeviceSynchronize");
    /* GC-on-free off-default-stream fingerprint sampling (all optional). */
    g_api.rt_stream_create = (pf_cudaStreamCreateWithFlags)dlsym(g_api.libcudart, "cudaStreamCreateWithFlags");
    g_api.rt_stream_sync   = (pf_cudaStreamSynchronize)dlsym(g_api.libcudart, "cudaStreamSynchronize");
    g_api.rt_host_alloc    = (pf_cudaHostAlloc)dlsym(g_api.libcudart, "cudaHostAlloc");

    g_api.ok = g_api.create_prog && g_api.compile_prog
            && g_api.get_ptx_size && g_api.get_ptx
            && g_api.module_load && g_api.module_getfn
            && g_api.launch_kernel && g_api.func_setattr
            && g_api.rt_free && g_api.rt_malloc
            && g_api.rt_memcpy && g_api.rt_memcpy_async
            && g_api.rt_memset_async && g_api.rt_devattr;
    if (!g_api.ok) {
        cipher_log("MARLIN: API dlsym resolve incomplete");
    }
    return g_api.ok;
}

/* ── Marlin cubin table: 10 entry points, lazily looked up ─────────────── */

struct MarlinFn {
    int          m_blocks;
    int          n_blocks;
    int          k_blocks;
    int          group_blocks;  /* -1 (no groups) or 8 (group_size=128) */
    const char  *sym;
    void        *fn;            /* CUfunction handle, NULL until lookup */
};

MarlinFn g_marlin_fns[] = {
    {1,  8,  8, -1, "marlin_M1_N8_K8_Gn1",   nullptr},
    {1,  8,  8,  8, "marlin_M1_N8_K8_G8",    nullptr},
    {1, 16,  4, -1, "marlin_M1_N16_K4_Gn1",  nullptr},
    {1, 16,  4,  8, "marlin_M1_N16_K4_G8",   nullptr},
    {2, 16,  4, -1, "marlin_M2_N16_K4_Gn1",  nullptr},
    {2, 16,  4,  8, "marlin_M2_N16_K4_G8",   nullptr},
    {3, 16,  4, -1, "marlin_M3_N16_K4_Gn1",  nullptr},
    {3, 16,  4,  8, "marlin_M3_N16_K4_G8",   nullptr},
    {4, 16,  4, -1, "marlin_M4_N16_K4_Gn1",  nullptr},
    {4, 16,  4,  8, "marlin_M4_N16_K4_G8",   nullptr},
};
constexpr int N_MARLIN_FNS = sizeof(g_marlin_fns) / sizeof(g_marlin_fns[0]);

/* Forward decl — defined after ensure_marlin_compiled. */
void *nvrtc_compile_to_module(const char *src, const char *prog_name);

void *g_marlin_module = nullptr;
std::atomic<int> g_marlin_state{0};  /* 0=cold, 1=compiling, 2=ready, 3=failed */
std::atomic<int> g_capture_safe{0};  /* G-O7 graph gate: CIPHER_MARLIN_CAPTURE_SAFE (default OFF) */

/* T4.5.2 GPU quantize kernel — directly ported from
 * cipher-may13-evidence/src/cipher_weight_compress.cpp:198 (kQuantSrc).
 * Two passes: compute per-group fp16 scales; quantize + pack into nibbles.
 * Group size hard-coded at 128 (matches Marlin's expected layout).
 *
 * Also includes a transpose kernel: PyTorch stores W as (out, in) row-major,
 * but Marlin expects W in (K=in, N=out) row-major. cipher_w_transpose_fp16
 * does the transpose into a fresh device buffer before quant runs. */
static const char cipher_rt_quant_kernel_src_str[] = R"NVRTC(
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CIPHER_QGROUP 128

/* W.2 (2026-05-27): bf16 <-> fp16 cast kernels for the Marlin bf16 path.
 * bf16 has 8-bit exponent (wider range than fp16) and 7-bit mantissa
 * (narrower than fp16). Cast loses range (clip past +/-65504) on fp16
 * direction but preserves mantissa (since bf16 mantissa is narrower).
 * For Llama-3 / Mistral inference activations the range is well within
 * fp16 bounds; KL gate catches if not. */
extern "C" __global__ void cipher_bf16_to_fp16(
    const __nv_bfloat16* __restrict__ in,
    __half*              __restrict__ out,
    long long n)
{
    long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__bfloat162float(in[i]));
}

extern "C" __global__ void cipher_fp16_to_bf16(
    const __half*        __restrict__ in,
    __nv_bfloat16*       __restrict__ out,
    long long n)
{
    long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__half2float(in[i]));
}

extern "C" __global__ void cipher_w_transpose_fp16(
    const __half* __restrict__ in,    // (rows_in, cols_in) row-major
    __half*       __restrict__ out,   // (cols_in, rows_in) row-major
    int rows_in, int cols_in)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;  /* 0..rows_in-1 */
    int c = blockIdx.x * blockDim.x + threadIdx.x;  /* 0..cols_in-1 */
    if (r >= rows_in || c >= cols_in) return;
    /* out[c, r] = in[r, c] */
    out[(long long)c * rows_in + r] = in[(long long)r * cols_in + c];
}

extern "C" __global__ void cipher_w_compute_scales(
    const __half* __restrict__ in,
    __half*       __restrict__ out_scales,
    int rows, int cols)
{
    int g = blockIdx.x;
    int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= cols) return;
    int row_base = g * CIPHER_QGROUP;
    if (row_base >= rows) return;

    float a = 0.0f;
    int last = row_base + CIPHER_QGROUP;
    if (last > rows) last = rows;
    for (int k = row_base; k < last; ++k) {
        float v = __half2float(in[k * cols + n]);
        float av = v < 0.0f ? -v : v;
        if (av > a) a = av;
    }
    float scale = (a > 0.0f) ? (a / 7.0f) : 1.0f;
    out_scales[g * cols + n] = __float2half_rn(scale);
}

extern "C" __global__ void cipher_w_quant_pack(
    const __half* __restrict__ in,
    const __half* __restrict__ in_scales,
    unsigned char* __restrict__ out_int4,
    int rows, int cols)
{
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int half_cols = cols >> 1;
    if (k >= rows || c >= half_cols) return;

    int kg = k / CIPHER_QGROUP;
    int col0 = c << 1;
    int col1 = col0 + 1;

    float v0 = __half2float(in[k * cols + col0]);
    float v1 = __half2float(in[k * cols + col1]);
    float s0 = __half2float(in_scales[kg * cols + col0]);
    float s1 = __half2float(in_scales[kg * cols + col1]);

    int q0 = __float2int_rn(v0 / s0);
    int q1 = __float2int_rn(v1 / s1);
    if (q0 >  7) q0 =  7; if (q0 < -8) q0 = -8;
    if (q1 >  7) q1 =  7; if (q1 < -8) q1 = -8;
    unsigned char b = ((unsigned char)(q1 & 0xF) << 4)
                    | ((unsigned char)(q0 & 0xF));
    out_int4[k * half_cols + c] = b;
}
)NVRTC";

void *g_quant_module = nullptr;
void *g_quant_fn_scales = nullptr;
void *g_quant_fn_pack   = nullptr;
void *g_quant_fn_transpose = nullptr;
/* W.2 (2026-05-27): bf16 cast kernel handles. */
void *g_quant_fn_bf16_to_fp16 = nullptr;
void *g_quant_fn_fp16_to_bf16 = nullptr;
std::atomic<int> g_quant_state{0};  /* 0=cold, 1=compiling, 2=ready, 3=failed */

bool ensure_marlin_compiled() {
    int state = g_marlin_state.load(std::memory_order_acquire);
    if (state == 2) return true;
    if (state == 3) return false;

    /* Race: only first thread compiles. */
    int expected = 0;
    if (!g_marlin_state.compare_exchange_strong(expected, 1)) {
        /* Someone else is compiling — spin briefly. */
        while ((state = g_marlin_state.load(std::memory_order_acquire)) == 1) {
            struct timespec ts = {0, 1000000};  /* 1 ms */
            nanosleep(&ts, nullptr);
        }
        return state == 2;
    }

    if (!resolve_api()) { g_marlin_state.store(3); return false; }

    /* G-O7 graph gate: capture-safe cast alloc, default-OFF (env). */
    {
        const char *cs = getenv("CIPHER_MARLIN_CAPTURE_SAFE");
        int on = cs && (cs[0]=='1' || cs[0]=='o' || cs[0]=='O' || cs[0]=='t' || cs[0]=='T');
        g_capture_safe.store(on && g_api.rt_malloc_async && g_api.rt_free_async);
        if (g_capture_safe.load())
            cipher_log("MARLIN: CAPTURE-SAFE cast alloc ENABLED (cudaMallocAsync — CUDA-graph-capture-legal)");
    }

    /* Force CUDA context creation (cudaFree(0) is the documented idiom). */
    if (g_api.rt_free) (void)g_api.rt_free(nullptr);

    cipher_log("MARLIN: NVRTC compile starting (kernel src %zu bytes, sm_90)...",
               strlen(cipher_rt_marlin_kernel_src_str));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (g_api.rt_free) (void)g_api.rt_free(nullptr);  /* CUDA ctx bring-up */
    g_marlin_module = nvrtc_compile_to_module(cipher_rt_marlin_kernel_src_str,
                                               "cipher_rt_marlin.cu");
    if (!g_marlin_module) { g_marlin_state.store(3); return false; }

    int found = 0;
    for (int i = 0; i < N_MARLIN_FNS; i++) {
        void *fn = nullptr;
        if (g_api.module_getfn(&fn, g_marlin_module, g_marlin_fns[i].sym) == 0 && fn) {
            g_marlin_fns[i].fn = fn;
            /* MAX_DYN_SHARED_SIZE = 96K (attr id 8). */
            (void)g_api.func_setattr(fn, 8, 96 * 1024);
            found++;
        } else {
            cipher_log("MARLIN: WARN missing entry '%s'", g_marlin_fns[i].sym);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    cipher_log("MARLIN: cubin ready (%d/%d entry points in %.2f s)",
               found, N_MARLIN_FNS, elapsed_s);

    if (found == 0) { g_marlin_state.store(3); return false; }
    g_marlin_state.store(2, std::memory_order_release);
    return true;
}

void *marlin_get_function(int m_blocks, int n_blocks, int k_blocks, int group_blocks) {
    for (int i = 0; i < N_MARLIN_FNS; i++) {
        if (g_marlin_fns[i].m_blocks == m_blocks &&
            g_marlin_fns[i].n_blocks == n_blocks &&
            g_marlin_fns[i].k_blocks == k_blocks &&
            g_marlin_fns[i].group_blocks == group_blocks)
            return g_marlin_fns[i].fn;
    }
    return nullptr;
}

/* Compile NVRTC source → module, returning module handle. NULL on failure.
 * Caller calls cuModuleGetFunction afterwards to look up entry points. */
void *nvrtc_compile_to_module(const char *src, const char *prog_name)
{
    if (!resolve_api()) return nullptr;

    void *prog = nullptr;
    if (g_api.create_prog(&prog, src, prog_name, 0, nullptr, nullptr) != 0 || !prog) {
        cipher_log("MARLIN: nvrtcCreateProgram(%s) failed", prog_name);
        return nullptr;
    }
    const char *opts[] = {
        "--gpu-architecture=sm_90",
        "-std=c++17",
        "-I/usr/include",
        /* W.2 (2026-05-27): NVRTC needs explicit CUDA include path for
         * cuda_fp16.h + cuda_bf16.h. Previous builds happened to work via
         * default search; vllm-openai:v0.21.0 container layout requires
         * explicit -I. Add both CUDA 13 and CUDA 12 candidates. */
        "-I/usr/local/cuda-13.0/targets/x86_64-linux/include",
        "-I/usr/local/cuda-13.0/include",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/targets/x86_64-linux/include",
        "--use_fast_math",
        "-default-device",
        "--ptxas-options=-O3",
    };
    int rc = g_api.compile_prog(prog, (int)(sizeof(opts) / sizeof(opts[0])), opts);
    if (rc != 0) {
        size_t logsz = 0;
        g_api.get_log_size(prog, &logsz);
        if (logsz > 0 && logsz < 32768) {
            char *buf = (char *)alloca(logsz + 1);
            g_api.get_log(prog, buf);
            buf[logsz] = 0;
            cipher_log("MARLIN: nvrtc compile %s FAILED rc=%d log:\n%s", prog_name, rc, buf);
        } else {
            cipher_log("MARLIN: nvrtc compile %s FAILED rc=%d", prog_name, rc);
        }
        g_api.destroy_prog(&prog);
        return nullptr;
    }
    size_t ptxsz = 0;
    if (g_api.get_ptx_size(prog, &ptxsz) != 0 || ptxsz == 0) {
        g_api.destroy_prog(&prog); return nullptr;
    }
    std::vector<char> ptx(ptxsz);
    if (g_api.get_ptx(prog, ptx.data()) != 0) {
        g_api.destroy_prog(&prog); return nullptr;
    }
    g_api.destroy_prog(&prog);

    void *mod = nullptr;
    if (g_api.module_load(&mod, ptx.data()) != 0 || !mod) {
        cipher_log("MARLIN: cuModuleLoadData(%s) failed", prog_name);
        return nullptr;
    }
    return mod;
}

bool ensure_quant_compiled()
{
    int state = g_quant_state.load(std::memory_order_acquire);
    if (state == 2) return true;
    if (state == 3) return false;

    int expected = 0;
    if (!g_quant_state.compare_exchange_strong(expected, 1)) {
        while ((state = g_quant_state.load(std::memory_order_acquire)) == 1) {
            struct timespec ts = {0, 1000000}; nanosleep(&ts, nullptr);
        }
        return state == 2;
    }

    if (!resolve_api()) { g_quant_state.store(3); return false; }
    if (g_api.rt_free) (void)g_api.rt_free(nullptr);  /* CUDA ctx bring-up */

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    g_quant_module = nvrtc_compile_to_module(cipher_rt_quant_kernel_src_str,
                                              "cipher_rt_quant.cu");
    if (!g_quant_module) { g_quant_state.store(3); return false; }
    if (g_api.module_getfn(&g_quant_fn_scales, g_quant_module,
                            "cipher_w_compute_scales") != 0 || !g_quant_fn_scales) {
        cipher_log("MARLIN: missing cipher_w_compute_scales entry");
        g_quant_state.store(3); return false;
    }
    if (g_api.module_getfn(&g_quant_fn_pack, g_quant_module,
                            "cipher_w_quant_pack") != 0 || !g_quant_fn_pack) {
        cipher_log("MARLIN: missing cipher_w_quant_pack entry");
        g_quant_state.store(3); return false;
    }
    if (g_api.module_getfn(&g_quant_fn_transpose, g_quant_module,
                            "cipher_w_transpose_fp16") != 0 || !g_quant_fn_transpose) {
        cipher_log("MARLIN: missing cipher_w_transpose_fp16 entry");
        g_quant_state.store(3); return false;
    }
    /* W.2 (2026-05-27): resolve bf16 cast kernels. Non-fatal if missing
     * (some NVRTC builds may omit cuda_bf16.h) — bf16 path then PASSTHROUGHs. */
    if (g_api.module_getfn(&g_quant_fn_bf16_to_fp16, g_quant_module,
                            "cipher_bf16_to_fp16") != 0)
        g_quant_fn_bf16_to_fp16 = nullptr;
    if (g_api.module_getfn(&g_quant_fn_fp16_to_bf16, g_quant_module,
                            "cipher_fp16_to_bf16") != 0)
        g_quant_fn_fp16_to_bf16 = nullptr;
    cipher_log("MARLIN: bf16 cast kernels: bf16_to_fp16=%p fp16_to_bf16=%p",
               g_quant_fn_bf16_to_fp16, g_quant_fn_fp16_to_bf16);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    cipher_log("MARLIN: GPU quant kernels compiled in %.2f s", s);
    g_quant_state.store(2, std::memory_order_release);
    return true;
}

/* ── Per-weight cache: pointer-keyed Marlin'd buffers ──────────────────── */

struct WeightSlot {
    int     observation_count;
    int     ready;          /* 1 once quantized + repacked */
    int     K;              /* rows (in-features) */
    int     N;              /* cols (out-features) */
    int     G;              /* group size (128) */
    void   *marlin_B;       /* device pointer, INT32-packed */
    void   *marlin_S;       /* device pointer, fp16 scales */
    uint64_t fp;            /* GC-on-free: content fingerprint of the keyed
                            * weight at quantize time (0 = not computed) */
};

std::mutex g_weight_mu;

/* D.7 R-H1 — Marlin weight cache keyed on (model_id, w_ptr).
 *
 * Audit §3.1 G4 hazard: at heterogeneous N>1, two distinct models could
 * load FP16 weights at the same w_ptr value (post-free + alloc reuse)
 * and the second model's lookups would find the first's quantized cache
 * — silent cross-model corruption. W10-12 Step 2 keyed on tenant_id as a
 * placeholder (tenant 0 in practice); D.7 completes the fix to the model
 * identity (model_id = folded 128-bit model_uuid from CIPHER_REGISTER_MODEL
 * G10): same-FAMILY tenants share one quantized kit (same model_id), while
 * distinct families stay isolated even under w_ptr reuse. This is what lets
 * the heterogeneous-residence owner process (≥3–5 families in shared arenas)
 * route per family without cross-model bleed.
 *
 * ADDITIVE / regression-safe: model_id defaults to 0 (MODEL_UNKNOWN) when no
 * binding is registered → key (0, w_ptr) → byte-identical behavior to the
 * prior (tenant 0, w_ptr) single-model path. The residence layer registers
 * (w_ptr → model_id) via cipher_rt_marlin_engine_bind_model(); unbound w_ptrs
 * resolve to 0. Default-OFF: zero bindings ⇒ vanilla. */
struct MarlinWeightKey {
    uint64_t    model_id;
    const void *w_ptr;
    bool operator==(const MarlinWeightKey &o) const noexcept {
        return model_id == o.model_id && w_ptr == o.w_ptr;
    }
};

struct MarlinWeightKeyHash {
    size_t operator()(const MarlinWeightKey &k) const noexcept {
        /* FNV-style mix on the two fields; cheap and adequate. */
        return std::hash<const void *>()(k.w_ptr) * 0x9E3779B97F4A7C15ULL
             ^ ((size_t)k.model_id * 0x517CC1B727220A95ULL);
    }
};

std::unordered_map<MarlinWeightKey, WeightSlot, MarlinWeightKeyHash> g_weights;

/* D.7: (w_ptr → model_id) binding registry. The heterogeneous-residence
 * layer registers each shared family arena's weight pointers to that family's
 * model_id (from CIPHER_REGISTER_MODEL). Unbound ⇒ 0 (single-model default).
 * Guarded by g_weight_mu (same lock as g_weights). */
std::unordered_map<const void *, uint64_t> g_wptr_model;

/* Resolve the model identity for a weight pointer at the Marlin call depth.
 * Marlin carries no stream/model descriptor this deep; the residence layer
 * pre-registers the binding. Unbound (e.g. single-model, or pre-residence)
 * ⇒ model_id 0, preserving the prior single-model cache behavior. */
static uint64_t marlin_model_id_for_locked(const void *w) {
    auto it = g_wptr_model.find(w);
    return (it != g_wptr_model.end()) ? it->second : 0ULL;
}

WeightSlot *find_or_create_slot_locked(uint64_t model_id, const void *w) {
    MarlinWeightKey k{model_id, w};
    auto it = g_weights.find(k);
    if (it != g_weights.end()) return &it->second;
    WeightSlot s = {};
    auto [ins, _] = g_weights.emplace(k, s);
    return &ins->second;
}

WeightSlot *find_or_create_slot_locked(const void *w) {
    return find_or_create_slot_locked(marlin_model_id_for_locked(w), w);
}

/* D.7 exported setter: residence layer binds a weight pointer to its family
 * model_id. model_id 0 unbinds (back to single-model default). Idempotent. */
extern "C" void cipher_rt_marlin_engine_bind_model(const void *w_ptr,
                                                   unsigned long long model_id) {
    std::lock_guard<std::mutex> lk(g_weight_mu);
    if (model_id == 0ULL) g_wptr_model.erase(w_ptr);
    else                  g_wptr_model[w_ptr] = (uint64_t)model_id;
}

/* ── GC-on-free infrastructure (2026-05-30) ─────────────────────────────
 * The Marlin weight cache previously never evicted a kit when a model was
 * freed: under load/free/reload churn a reused weight pointer hit a stale
 * kit → wrong output (Qwen2 all-zeros) or, on a shape mismatch, an
 * illegal-memory-access (kit packed for the old dims). Fix:
 *   - dim re-validation on the cached path (is_ready / ensure / surrogate);
 *   - a content fingerprint catching same-shape reuse (env CIPHER_MARLIN_GC_FP,
 *     default on);
 *   - retired-kit list + reclaim_retired(): freeing a kit inline would UAF a
 *     kit still referenced by an in-flight launch on another stream (the
 *     Track-2 weight-sharing regime), so eviction only RETIRES the buffers;
 *     they are cudaFree'd only after a device-sync quiescent barrier in
 *     reclaim. Lock order: g_bf16_weight_mu → g_weight_mu → g_retire_mu
 *     (g_retire_mu is a leaf). */

bool g_gc_fp_enabled = true;   /* CIPHER_MARLIN_GC_FP (default on) */
/* TEST-ONLY fault injection (CIPHER_MARLIN_GC_FAULT_BF16, default off): forces
 * dispatch_bf16() to take an early-error return on a fraction of calls, to
 * exercise the actuator's error-branch borrow release (proves no borrow leak on
 * the malloc-fail / cast-missing paths). Never set in production. */
bool g_gc_fault_bf16 = false;
std::atomic<unsigned long> g_bf16_call_ctr{0};

/* Fingerprint sampling resources. A dedicated NON-BLOCKING stream + pinned
 * staging buffer keep the per-call DtoH sample OFF the legacy default stream,
 * so it does not barrier every tenant's Marlin GEMM (the review's perf concern).
 * g_fp_mu is a LEAF mutex (never held while holding g_weight_mu/g_bf16_weight_mu;
 * the fingerprint is always computed outside those). */
constexpr size_t FP_SAMPLE = 1024;   /* single head sample (one sync, good collision-resistance) */
std::mutex                 g_fp_mu;
void                      *g_fp_stream    = nullptr;
void                      *g_fp_host      = nullptr;   /* pinned, FP_SAMPLE bytes */
bool                       g_fp_init_done = false;
std::atomic<unsigned long> g_fp_fail{0};               /* observability: sample failures (gc_fp on) */

/* Content fingerprint of a weight tensor. elem_bytes = 2 (fp16/bf16). Samples a
 * 1 KB in-bounds head chunk + folds in (K,N,total) — enough to distinguish two
 * different models occupying the same reused address (a same-address, same-shape,
 * byte-identical-1KB-head collision across distinct trained weights is
 * astronomically unlikely). Sampled off the default stream when available; the
 * DtoH is retried once before giving up. Returns 0 only on a persistent sample
 * failure (counted + logged once) → callers treat 0 as "content-check
 * unavailable" and fall back to dim-revalidation (which still catches the crash
 * path). NOT silent (the review's fail-open finding) and NOT a spurious-evict. */
uint64_t weight_fingerprint(const void *w, int K, int N, int elem_bytes) {
    if (!w || K <= 0 || N <= 0) return 0;
    size_t total = (size_t)K * (size_t)N * (size_t)elem_bytes;
    if (total == 0) return 0;
    size_t n = (total < FP_SAMPLE) ? total : FP_SAMPLE;

    std::lock_guard<std::mutex> lk(g_fp_mu);
    if (!g_fp_init_done) {
        if (g_api.rt_stream_create && g_api.rt_host_alloc && g_api.rt_stream_sync) {
            if (g_api.rt_stream_create(&g_fp_stream, 0x1u /*cudaStreamNonBlocking*/) != 0)
                g_fp_stream = nullptr;
            if (g_api.rt_host_alloc(&g_fp_host, FP_SAMPLE, 0u /*cudaHostAllocDefault*/) != 0)
                g_fp_host = nullptr;
        }
        g_fp_init_done = true;
    }

    const unsigned char *src = nullptr;
    unsigned char local[FP_SAMPLE];
    bool ok = false;
    /* Preferred: async DtoH on the dedicated non-blocking stream; sync only it. */
    if (g_fp_stream && g_fp_host && g_api.rt_memcpy_async && g_api.rt_stream_sync) {
        for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
            if (g_api.rt_memcpy_async(g_fp_host, w, n, /*DtoH*/ 2, g_fp_stream) == 0 &&
                g_api.rt_stream_sync(g_fp_stream) == 0) ok = true;
        }
        if (ok) src = (const unsigned char *)g_fp_host;
    }
    /* Fallback: synchronous DtoH (no dedicated stream available). */
    if (!ok && g_api.rt_memcpy) {
        for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
            if (g_api.rt_memcpy(local, w, n, /*DtoH*/ 2) == 0) ok = true;
        }
        if (ok) src = local;
    }
    if (!ok) {
        unsigned long c = g_fp_fail.fetch_add(1, std::memory_order_relaxed);
        if (c == 0)
            cipher_log("MARLIN GC: weight_fingerprint DtoH sample FAILED (gc_fp on) — "
                       "content-check unavailable for affected slot(s); dim-revalidation "
                       "still active. (counter cipher_rt_marlin_gc_fp_failures)");
        return 0;
    }

    uint64_t h = 1469598103934665603ULL;            /* FNV-1a basis */
    for (size_t i = 0; i < n; ++i) { h ^= src[i]; h *= 1099511628211ULL; }
    h ^= (uint64_t)(unsigned)K; h *= 1099511628211ULL;
    h ^= (uint64_t)(unsigned)N; h *= 1099511628211ULL;
    h ^= (uint64_t)total;       h *= 1099511628211ULL;
    return h ? h : 1ULL;                            /* reserve 0 = "unavailable" */
}

/* Retired device buffers awaiting a quiescent free (see reclaim_retired). */
std::vector<void*> g_retired;
std::mutex         g_retire_mu;

/* Dispatch borrow counter (the user's "refcount==0" half of the safe-reclaim
 * barrier). lookup() increments when it hands a kit out for launch; dispatch()
 * decrements after the launch is enqueued. A retired kit is removed from the
 * map at eviction, so NO lookup after eviction can return it — only threads
 * that borrowed it BEFORE eviction can still hold it. reclaim therefore waits
 * for borrow==0 (all pre-eviction borrows drained) AND then device-synchronizes
 * (all launched kernels done) before freeing → no in-flight launch can read a
 * freed buffer. Global (not per-kit): conservative but correct. */
std::atomic<int> g_dispatch_borrow{0};

void retire_buf(void *p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_retire_mu);
    g_retired.push_back(p);
}

/* Evict every g_weights slot whose key.w_ptr matches `wptr` (any model_id),
 * retiring its kit buffers. Caller MUST hold g_weight_mu. */
void evict_g_weights_for_ptr_locked(const void *wptr) {
    if (!wptr) return;
    for (auto it = g_weights.begin(); it != g_weights.end(); ) {
        if (it->first.w_ptr == wptr) {
            retire_buf(it->second.marlin_B);
            retire_buf(it->second.marlin_S);
            it = g_weights.erase(it);
        } else {
            ++it;
        }
    }
}

/* ── Quantization: fp16 weight → INT4 packed + fp16 per-group scales ──── */
/*
 * GPU-side via two NVRTC-compiled kernels (cipher_w_compute_scales,
 * cipher_w_quant_pack). Output stays device-side; we DtoH-copy for the
 * host-side Marlin repack which uses the 1024-entry LUT.
 *
 * Layout produced (matches the prior cipher_weight_compress quant output):
 *   int4_packed: K rows × N/2 bytes — two signed-4-bit values per byte
 *                (low nibble = even col, high nibble = odd col)
 *   scales:      K/G rows × N fp16 values — per-group absmax / 7
 * This is the input format expected by marlin_repack_host().
 */
bool quantize_fp16_to_int4_groupwise_gpu(
    const void *d_fp16_weight, int K, int N, int G,
    std::vector<uint8_t> *out_int4, std::vector<uint16_t> *out_scales_fp16)
{
    if ((K % G) != 0 || G != 128) return false;
    int KG = K / G;
    size_t pack_bytes = (size_t)K * (N / 2);
    size_t scale_cnt  = (size_t)KG * N;

    if (!ensure_quant_compiled()) return false;

    /* PyTorch's Linear weight is stored row-major as (out=N, in=K). Marlin
     * expects the weight in (K, N) row-major. Transpose first into a fresh
     * device buffer. (cuBLAS A operand IS PyTorch's W; we need its bytes
     * reinterpreted as (K, N).) */
    void *d_w_T = nullptr;
    if (g_api.rt_malloc(&d_w_T, (size_t)K * N * 2) != 0) return false;
    {
        int rows_in = N;  /* original (out, in) row-major has rows = out = N */
        int cols_in = K;  /* and cols = in = K */
        int bx = 32, by = 8;
        int gx = (cols_in + bx - 1) / bx;
        int gy = (rows_in + by - 1) / by;
        void *args[] = { (void *)&d_fp16_weight, &d_w_T, &rows_in, &cols_in };
        int rc = g_api.launch_kernel(g_quant_fn_transpose, gx, gy, 1, bx, by, 1,
                                     0, nullptr, args, nullptr);
        if (rc != 0) {
            cipher_log("MARLIN: transpose kernel launch rc=%d", rc);
            g_api.rt_free(d_w_T); return false;
        }
    }

    /* Allocate device buffers for int4 + scales. */
    void *d_int4 = nullptr;
    void *d_scl  = nullptr;
    if (g_api.rt_malloc(&d_int4, pack_bytes) != 0) {
        g_api.rt_free(d_w_T); return false;
    }
    if (g_api.rt_malloc(&d_scl, scale_cnt * 2 /* fp16 */) != 0) {
        g_api.rt_free(d_w_T); g_api.rt_free(d_int4); return false;
    }

    /* Pass 1: compute per-group scales (groups along K = rows of d_w_T). */
    {
        int bx = 128;
        int gx = KG;
        int gy = (N + bx - 1) / bx;
        void *args[] = { &d_w_T, &d_scl, &K, &N };
        int rc = g_api.launch_kernel(g_quant_fn_scales, gx, gy, 1, bx, 1, 1,
                                     0, nullptr, args, nullptr);
        if (rc != 0) {
            cipher_log("MARLIN: scales kernel launch rc=%d", rc);
            g_api.rt_free(d_w_T); g_api.rt_free(d_int4); g_api.rt_free(d_scl);
            return false;
        }
    }
    /* Pass 2: quantize + pack each byte. */
    {
        int bx = 32, by = 8;
        int half_N = N / 2;
        int gx = (half_N + bx - 1) / bx;
        int gy = (K + by - 1) / by;
        void *args[] = { &d_w_T, &d_scl, &d_int4, &K, &N };
        int rc = g_api.launch_kernel(g_quant_fn_pack, gx, gy, 1, bx, by, 1,
                                     0, nullptr, args, nullptr);
        if (rc != 0) {
            cipher_log("MARLIN: pack kernel launch rc=%d", rc);
            g_api.rt_free(d_w_T); g_api.rt_free(d_int4); g_api.rt_free(d_scl);
            return false;
        }
    }

    /* DtoH for the host-side Marlin repack. */
    out_int4->resize(pack_bytes);
    out_scales_fp16->resize(scale_cnt);
    if (g_api.rt_memcpy(out_int4->data(), d_int4, pack_bytes, /*DtoH*/ 2) != 0 ||
        g_api.rt_memcpy(out_scales_fp16->data(), d_scl, scale_cnt * 2, /*DtoH*/ 2) != 0) {
        g_api.rt_free(d_w_T); g_api.rt_free(d_int4); g_api.rt_free(d_scl);
        return false;
    }
    g_api.rt_free(d_w_T);
    g_api.rt_free(d_int4);
    g_api.rt_free(d_scl);
    return true;
}

/* ── Marlin repack: pack signed nibbles into Marlin XOR-swizzled int32 ──
 * Direct port from cipher-may13-evidence/src/cipher_weight_compress.cpp
 * lines 880-980, adapted to use cipher_rt_marlin_perm/_scale_perm. */
struct MarlinPacked { void *B; void *S; int K; int N; int G; };

bool marlin_repack_host(const uint8_t *h_int4, const uint16_t *h_scales,
                        int K, int N, int G, MarlinPacked *out)
{
    if ((K % 16) != 0 || (N % 64) != 0 || G != 128) return false;
    int KG = K / G;
    int K_tiles = K / 16;
    int N_tiles = N / 16;

    /* 1. Unpack signed nibbles → unsigned [0,15] (signed_val + 8 mod 16). */
    std::vector<uint8_t> w((size_t)K * N);
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < N / 2; ++c) {
            uint8_t b = h_int4[(size_t)k * (N / 2) + c];
            int low  = b & 0xF;
            int high = (b >> 4) & 0xF;
            /* Sign-extend the signed nibble before adding 8 (the actual
             * Marlin convention treats nibbles as unsigned-biased). */
            int slo = (low  & 0x8) ? (low  - 16) : low;
            int shi = (high & 0x8) ? (high - 16) : high;
            w[(size_t)k * N + 2 * c]     = (uint8_t)((slo + 8) & 0xF);
            w[(size_t)k * N + 2 * c + 1] = (uint8_t)((shi + 8) & 0xF);
        }
    }

    /* 2. Reshape (K/16, 16, N/16, 16) → permute (0,2,1,3) → flatten. */
    size_t row_size = (size_t)N * 16;
    std::vector<uint8_t> tiled((size_t)K_tiles * row_size);
    for (int kt = 0; kt < K_tiles; ++kt) {
        for (int nt = 0; nt < N_tiles; ++nt) {
            for (int ki = 0; ki < 16; ++ki) {
                for (int ni = 0; ni < 16; ++ni) {
                    tiled[(size_t)kt * row_size
                          + (size_t)nt * 256 + ki * 16 + ni]
                        = w[(size_t)(kt * 16 + ki) * N + (nt * 16 + ni)];
                }
            }
        }
    }

    /* 3. Apply cipher_rt_marlin_perm in 1024-chunks. */
    std::vector<uint8_t> permuted((size_t)K_tiles * row_size);
    int n_chunks = (int)(row_size / 1024);
    for (int r = 0; r < K_tiles; ++r) {
        for (int ch = 0; ch < n_chunks; ++ch) {
            const uint8_t *src = &tiled    [(size_t)r * row_size + ch * 1024];
                  uint8_t *dst = &permuted [(size_t)r * row_size + ch * 1024];
            for (int i = 0; i < 1024; ++i) dst[i] = src[cipher_rt_marlin_perm[i]];
        }
    }

    /* 4. Pack 8 4-bit values per int32; output (K/16, N*2) int32. */
    int packed_cols = N * 2;
    std::vector<uint32_t> marlin_B((size_t)K_tiles * packed_cols, 0);
    for (int r = 0; r < K_tiles; ++r) {
        for (int c = 0; c < packed_cols; ++c) {
            uint32_t q = 0;
            for (int i = 0; i < 8; ++i) {
                q |= ((uint32_t)permuted[(size_t)r * row_size + c * 8 + i]
                      & 0xF) << (4 * i);
            }
            marlin_B[(size_t)r * packed_cols + c] = q;
        }
    }

    /* 5. Scale permutation: per-group permute N columns in chunks of 64. */
    int n_scale_chunks = N / 64;
    std::vector<uint16_t> marlin_S_h((size_t)KG * N);
    for (int g = 0; g < KG; ++g) {
        for (int ch = 0; ch < n_scale_chunks; ++ch) {
            const uint16_t *src = &h_scales  [(size_t)g * N + ch * 64];
                  uint16_t *dst = &marlin_S_h[(size_t)g * N + ch * 64];
            for (int i = 0; i < 64; ++i) dst[i] = src[cipher_rt_marlin_scale_perm[i]];
        }
    }

    /* 6. HtoD push. */
    size_t marlin_B_bytes = (size_t)K_tiles * packed_cols * 4;
    size_t marlin_S_bytes = (size_t)KG * N * 2;
    if (g_api.rt_malloc(&out->B, marlin_B_bytes) != 0) return false;
    if (g_api.rt_malloc(&out->S, marlin_S_bytes) != 0) {
        g_api.rt_free(out->B); return false;
    }
    if (g_api.rt_memcpy(out->B, marlin_B.data(), marlin_B_bytes,
                        /*HtoD*/ 1) != 0) {
        g_api.rt_free(out->B); g_api.rt_free(out->S); return false;
    }
    if (g_api.rt_memcpy(out->S, marlin_S_h.data(), marlin_S_bytes,
                        /*HtoD*/ 1) != 0) {
        g_api.rt_free(out->B); g_api.rt_free(out->S); return false;
    }
    out->K = K; out->N = N; out->G = G;
    return true;
}

/* ── Marlin GEMM dispatch: parameter setup + launch ─────────────────────
 * Direct port from cipher-may13-evidence/src/cipher_weight_compress.cpp
 * lines 1312-1422 (the cipher_weight_compress_marlin_gemm function). */
int g_sm_count = 0;

/* ── Per-stream Marlin workspace registry (CP 2.4) ──────────────────────
 * The Marlin kernel uses a `locks` buffer as inter-CTA split-K semaphores.
 * Async kernels on different streams cannot share one buffer — they race.
 * The old op31 path serialized ALL Marlin GEMMs through one lock + one
 * shared stream (the CP 0.4/0.5 ~8.5 tok/s aggregate ceiling). Here each
 * caller stream gets a private MARLIN_WS_BYTES workspace, so N tenants on
 * N streams run N-way concurrently with no cross-stream sync — the GEMM
 * still launches on the caller's own stream.
 *
 * MARLIN_WS_SLOTS (256) >= any realistic concurrent-tenant count (the
 * CP 0.5 burst sweep peaks at N=192). With slots >= max-concurrent, the
 * LRU eviction victim is always a slot whose stream was destroyed (its
 * kernels long complete) — eviction never reassigns a live stream's
 * buffer, so it is race-free. >256 concurrent streams would need a
 * cuStreamDestroy callback; out of scope for the CP 2.4 gate. */
constexpr size_t MARLIN_WS_BYTES = (size_t)(32768 / 128) * 16 * sizeof(int);
constexpr int    MARLIN_WS_SLOTS = 256;

struct MarlinWsSlot {
    void    *stream;    /* key; nullptr = never-used slot */
    void    *ws;        /* device workspace, MARLIN_WS_BYTES; freed never */
    uint64_t last_use;  /* LRU tick */
};
MarlinWsSlot g_ws_slots[MARLIN_WS_SLOTS] = {};
std::mutex   g_ws_mu;
std::atomic<uint64_t> g_ws_tick{0};

/* Return a device `locks` workspace private to `stream`. NULL on failure.
 * Hit: O(slots) scan, sub-microsecond. Miss: lazy cudaMalloc of one slot. */
void *marlin_ws_for_stream(void *stream)
{
    uint64_t tick = g_ws_tick.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_ws_mu);

    int free_idx = -1, lru_idx = 0;
    uint64_t lru_tick = UINT64_MAX;
    for (int i = 0; i < MARLIN_WS_SLOTS; i++) {
        if (g_ws_slots[i].ws && g_ws_slots[i].stream == stream) {
            g_ws_slots[i].last_use = tick;          /* hit */
            return g_ws_slots[i].ws;
        }
        if (!g_ws_slots[i].ws && free_idx < 0) free_idx = i;
        if (g_ws_slots[i].last_use < lru_tick) {
            lru_tick = g_ws_slots[i].last_use; lru_idx = i;
        }
    }
    /* miss: take a never-used slot, else evict the LRU slot's stream key
     * (its device buffer is reused — safe, see the invariant above). */
    int idx;
    if (g_capture_safe.load(std::memory_order_relaxed)) {
        /* G-O7 graph gate: NEVER cudaMalloc during capture (invalidates it). Reuse an already-allocated
         * slot's buffer for the new (capture) stream (rekey) instead of allocating fresh. Requires Marlin to
         * have run eagerly before capture (warmup) so a slot exists — standard CUDA-graph warmup discipline.
         * Single-stream / per-agent-pre-warm scope; the N-concurrent-stream multiplex needs >=N pre-warmed
         * distinct slots + no evict-of-in-use (v-next, flagged honestly — not scope-down). */
        int alloc_idx = -1; uint64_t at = UINT64_MAX;
        for (int i = 0; i < MARLIN_WS_SLOTS; i++)
            if (g_ws_slots[i].ws && g_ws_slots[i].last_use < at) { at = g_ws_slots[i].last_use; alloc_idx = i; }
        idx = (alloc_idx >= 0) ? alloc_idx : (free_idx >= 0 ? free_idx : lru_idx);
    } else {
        idx = (free_idx >= 0) ? free_idx : lru_idx;
    }
    if (!g_ws_slots[idx].ws) {
        void *nb = nullptr;
        if (g_api.rt_malloc(&nb, MARLIN_WS_BYTES) != 0) return nullptr;
        (void)g_api.rt_memset(nb, 0, MARLIN_WS_BYTES);
        g_ws_slots[idx].ws = nb;
    }
    g_ws_slots[idx].stream   = stream;
    g_ws_slots[idx].last_use = tick;
    return g_ws_slots[idx].ws;
}

/* Diagnostic: how many distinct streams have a live workspace slot. */
int marlin_ws_live_slots(void)
{
    std::lock_guard<std::mutex> lk(g_ws_mu);
    int n = 0;
    for (int i = 0; i < MARLIN_WS_SLOTS; i++) if (g_ws_slots[i].ws) n++;
    return n;
}

bool marlin_gemm_launch(
    const void *a_fp16, const void *marlin_B, const void *marlin_S,
    void *c_fp16, int M, int N, int K, int G, void *stream)
{
    if (!a_fp16 || !marlin_B || !marlin_S || !c_fp16) return false;
    if (M <= 0 || M > 4*16) return false;
    if ((N & 63) != 0 || (K & 127) != 0) return false;
    if (G != 128 && G != -1) return false;
    if (!ensure_marlin_compiled()) return false;

    int thread_k, thread_n;
    if (M <= 16) { thread_k = 128; thread_n = 128; }
    else         { thread_k =  64; thread_n = 256; }
    if ((N % thread_n) != 0 || (K % thread_k) != 0) return false;

    int thread_n_blocks = thread_n / 16;
    int thread_k_blocks = thread_k / 16;
    int group_blocks    = (G == -1) ? -1 : (G / 16);

    int tot_m_blocks = (M + 15) / 16;
    if (tot_m_blocks > 4) return false;
    int m_blocks = tot_m_blocks;

    void *fn = marlin_get_function(m_blocks, thread_n_blocks, thread_k_blocks,
                                   group_blocks);
    if (!fn) {
        cipher_log("MARLIN: no kernel for M=%d m_blocks=%d n_blocks=%d k_blocks=%d g_blocks=%d",
                   M, m_blocks, thread_n_blocks, thread_k_blocks, group_blocks);
        return false;
    }

    if (g_sm_count == 0) g_api.rt_devattr(&g_sm_count, /*MultiProcessorCount*/16, 0);
    if (g_sm_count <= 0) g_sm_count = 132;

    /* Per-stream workspace (CP 2.4 registry — replaces the single shared
     * g_marlin_workspace that serialized all tenants in CP 0.4/0.5). */
    int max_par = 16;
    size_t ws_bytes = (size_t)(N / 128) * max_par * sizeof(int);
    if (ws_bytes > MARLIN_WS_BYTES) {
        cipher_log("MARLIN: N=%d exceeds per-stream workspace capacity", N);
        return false;
    }
    void *ws = marlin_ws_for_stream(stream);
    if (!ws) return false;
    /* Per-call clear of the prefix the kernel touches, on the launch stream.
     * The buffer is private to `stream`, so this is race-free. */
    g_api.rt_memset_async(ws, 0, ws_bytes, stream);

    int prob_m = M, prob_n = N, prob_k = K;
    int *locks = (int *)ws;
    void *args[] = { (void *)&a_fp16, (void *)&marlin_B, (void *)&c_fp16, (void *)&marlin_S,
                     &prob_m, &prob_n, &prob_k, &locks };
    /* CP 5.3 STEP 2: when a green context is active, size the kernel `grid`
     * to the partition's verified SM count so the split-K CTAs are
     * co-resident inside the partition (STEP 1 proved the protocol is
     * grid-parametric down to grid=1). No green ctx → byte-identical to the
     * shipped single-tenant path. Scope memo §3 / §6 item 3. */
    unsigned int green_sm = cipher_rt_green_ctx_sm_count();
    int grid = (green_sm > 0) ? (int)green_sm : g_sm_count;
    int block = 256;
    int rc = g_api.launch_kernel(fn, grid, 1, 1, block, 1, 1,
                                 96 * 1024, stream, args, nullptr);
    if (rc != 0) {
        cipher_log("MARLIN: launch rc=%d (M=%d N=%d K=%d)", rc, M, N, K);
        return false;
    }
    return true;
}

/* ── CP 2.4 (2026-05-16): primary-context pin for the quant/repack path ──
 *
 * quantize_fp16_to_int4_groupwise_gpu mixes driver-API kernel launches on
 * the NULL stream (cuLaunchKernel) with runtime-API cudaMalloc / cudaMemcpy.
 * That sequence is only correct when the CURRENT CUDA context is the
 * runtime's primary context. The T4.2.4d green-context enforcement —
 * re-asserted by the CUPTI launch callback on EVERY cuLaunchKernel — makes a
 * green context current, so the quant kernels and the synchronous DtoH
 * cudaMemcpy land in the green context and the copy hangs. Latent since this
 * engine was written; exposed when kmod 0.4.8 let CUPTI subscribe. Full
 * diagnosis: cp_2_4/MARLIN_HANG_ROOT_CAUSE.md.
 *
 * PrimaryCtxGuard, held across the whole quant/repack sequence:
 *   - sets cipher_rt_in_marlin_quant=1 so the CUPTI callback skips
 *     green-enforcement for our own launches (else it re-greens the thread
 *     between them and the pin would not hold);
 *   - saves the caller's context, makes the device-0 primary context
 *     current — so malloc / launch / memcpy / free are all consistent;
 *   - restores both on destruction, covering every return path incl. errors. */
void          *g_primary_ctx = nullptr;
std::once_flag g_primary_once;

void *get_primary_ctx()
{
    std::call_once(g_primary_once, [] {
        if (g_api.primary_retain) {
            void *c = nullptr;
            if (g_api.primary_retain(&c, 0) == 0) g_primary_ctx = c;
        }
    });
    return g_primary_ctx;
}

struct PrimaryCtxGuard {
    void *saved  = nullptr;
    bool  ctx_ok = false;
    PrimaryCtxGuard()
    {
        cipher_rt_in_marlin_quant = 1;
        if (!g_api.ctx_get || !g_api.ctx_set) return;
        void *primary = get_primary_ctx();
        if (!primary) return;
        if (g_api.ctx_get(&saved) != 0)  return;
        if (g_api.ctx_set(primary) != 0) return;
        ctx_ok = true;
    }
    ~PrimaryCtxGuard()
    {
        if (ctx_ok) (void)g_api.ctx_set(saved);
        cipher_rt_in_marlin_quant = 0;
    }
    PrimaryCtxGuard(const PrimaryCtxGuard&) = delete;
    PrimaryCtxGuard &operator=(const PrimaryCtxGuard&) = delete;
};

/* Lazy quant+repack from the actuator hot path. Holds g_weight_mu while
 * doing the host-side work (one-shot per weight; subsequent calls hit
 * the cached path immediately). */
bool ensure_weight_quantized_repacked(const void *d_fp16_weight, int K, int N)
{
    {
        std::lock_guard<std::mutex> lk(g_weight_mu);
        WeightSlot *s = find_or_create_slot_locked(d_fp16_weight);
        if (s->ready) {
            /* GC-on-free: a cached kit at this key is only valid if its dims
             * still match. A shape mismatch means a different-shape weight now
             * lives here (post free+realloc) — retire the stale kit and
             * re-quantize. (Same-shape content reuse is caught upstream by the
             * fingerprint in is_ready / bf16_surrogate.) */
            if (s->K == K && s->N == N) return true;
            retire_buf(s->marlin_B);
            retire_buf(s->marlin_S);
            *s = WeightSlot{};
        }
    }

    /* Do the host-side work outside the lock to avoid holding it during
     * the long HtoD/DtoH copies. */
    constexpr int G = 128;
    if ((K % G) != 0 || (N % 64) != 0 || (K % 16) != 0) return false;

    /* Pin the entire quant + repack GPU sequence to the primary context;
     * RAII-restored on every exit path below. See PrimaryCtxGuard above. */
    PrimaryCtxGuard ctx_guard;

    std::vector<uint8_t>  int4;
    std::vector<uint16_t> scales;
    if (!quantize_fp16_to_int4_groupwise_gpu(d_fp16_weight, K, N, G, &int4, &scales))
        return false;

    MarlinPacked m{};
    if (!marlin_repack_host(int4.data(), scales.data(), K, N, G, &m))
        return false;

    /* Fingerprint computed outside the lock (a DtoH sample) so the lock-hold
     * stays short. One-shot per (re)quantize, not the hot cached path. */
    uint64_t fp = g_gc_fp_enabled ? weight_fingerprint(d_fp16_weight, K, N, 2) : 0;
    {
        std::lock_guard<std::mutex> lk(g_weight_mu);
        WeightSlot *s = find_or_create_slot_locked(d_fp16_weight);
        if (s->ready && s->K == K && s->N == N) {
            /* Another thread beat us with a same-shape kit; free our copy. */
            g_api.rt_free(m.B);
            g_api.rt_free(m.S);
            return true;
        }
        if (s->ready) {
            /* A stale (different-shape) kit was racing here; retire it. */
            retire_buf(s->marlin_B);
            retire_buf(s->marlin_S);
        }
        s->K = K; s->N = N; s->G = G;
        s->marlin_B = m.B;
        s->marlin_S = m.S;
        s->fp = fp;
        s->ready = 1;
    }
    return true;
}

} // anonymous namespace

/* ── Public engine API used by the actuator ──────────────────────────── */

extern "C" int cipher_rt_marlin_engine_init(void)
{
    if (!resolve_api()) return -1;
    /* GC-on-free fingerprint check: default ON (correctness-first). Set
     * CIPHER_MARLIN_GC_FP=0 to disable the per-call content fingerprint (dim
     * re-validation + the event-driven evict API still run). Exposed so the
     * fingerprint's hot-path cost is measurable. */
    {
        const char *e = getenv("CIPHER_MARLIN_GC_FP");
        g_gc_fp_enabled = !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
        const char *fz = getenv("CIPHER_MARLIN_GC_FAULT_BF16");
        g_gc_fault_bf16 = (fz && fz[0] && fz[0] != '0');
        cipher_log("MARLIN: GC-on-free active (fingerprint=%s%s)",
                   g_gc_fp_enabled ? "on" : "off",
                   g_gc_fault_bf16 ? "; BF16-FAULT-INJECTION ON (test)" : "");
    }
    /* Defer NVRTC compile to first eligible call (~19s cold; don't block
     * libcipher_rt init). */
    cipher_log("MARLIN: engine init OK (NVRTC + cuda + cudart resolved; "
               "cubin compile deferred to first eligible call)");
    return 0;
}

extern "C" int cipher_rt_marlin_engine_ensure_compiled(void)
{
    return ensure_marlin_compiled() ? 0 : -1;
}

extern "C" int cipher_rt_marlin_engine_observe_weight(const void *w_ptr)
{
    std::lock_guard<std::mutex> lk(g_weight_mu);
    WeightSlot *s = find_or_create_slot_locked(w_ptr);
    s->observation_count++;
    return s->observation_count;
}

extern "C" int cipher_rt_marlin_engine_is_ready(const void *w_ptr, int K, int N)
{
    /* GC-on-free validate-on-use (fp16-direct path). Fingerprint computed
     * before the lock (DtoH sample) so the lock-hold stays short; samples
     * w_ptr with the LIVE (K,N) → always in-bounds for the current allocation.
     */
    uint64_t live = g_gc_fp_enabled ? weight_fingerprint(w_ptr, K, N, 2) : 0;
    std::lock_guard<std::mutex> lk(g_weight_mu);
    MarlinWeightKey k{marlin_model_id_for_locked(w_ptr), w_ptr};
    auto it = g_weights.find(k);
    if (it == g_weights.end() || !it->second.ready) return 0;
    bool stale = (it->second.K != K || it->second.N != N);
    if (!stale && live && it->second.fp && live != it->second.fp) stale = true;
    if (stale) {
        /* Reused pointer: a freed model's kit lingers here. Retire + evict so
         * the caller re-quantizes for the current weight (the actuator's
         * captured observation count is already past threshold → re-quant
         * fires this same call; output stays correct). */
        retire_buf(it->second.marlin_B);
        retire_buf(it->second.marlin_S);
        g_weights.erase(it);
        return 0;
    }
    return 1;
}

extern "C" int cipher_rt_marlin_engine_lookup(const void *w_ptr,
                                              void **out_B, void **out_S,
                                              int *out_K, int *out_N, int *out_G)
{
    std::lock_guard<std::mutex> lk(g_weight_mu);
    MarlinWeightKey k{marlin_model_id_for_locked(w_ptr), w_ptr};
    auto it = g_weights.find(k);
    if (it == g_weights.end() || !it->second.ready) return 0;
    WeightSlot &s = it->second;
    if (out_B) *out_B = s.marlin_B;
    if (out_S) *out_S = s.marlin_S;
    if (out_K) *out_K = s.K;
    if (out_N) *out_N = s.N;
    if (out_G) *out_G = s.G;
    /* Borrow the kit for the imminent launch (released in dispatch()).
     * Incremented UNDER g_weight_mu so it is registered before any concurrent
     * evict (also under g_weight_mu) can retire this slot — the actuator calls
     * dispatch() immediately after a successful lookup, which releases it. */
    g_dispatch_borrow.fetch_add(1, std::memory_order_acq_rel);
    return 1;
}

extern "C" int cipher_rt_marlin_engine_quantize_repack(
    const void *d_fp16_weight, int K, int N)
{
    if (!ensure_marlin_compiled()) return -1;
    return ensure_weight_quantized_repacked(d_fp16_weight, K, N) ? 0 : -1;
}

extern "C" int cipher_rt_marlin_engine_dispatch(
    const void *a_fp16, const void *marlin_B, const void *marlin_S,
    void *c_fp16, int M, int N, int K, int G, void *stream)
{
    /* The kit borrow taken in lookup() is released by the ACTUATOR via
     * cipher_rt_marlin_engine_release_dispatch_borrow() after dispatch()/
     * dispatch_bf16() returns — on BOTH success and error branches. It is
     * deliberately NOT released here: dispatch_bf16() has early-error returns
     * (cast-kernel-missing, malloc-fail under pressure) BEFORE it reaches this
     * function, so a release tied to dispatch() would leak the borrow on those
     * paths and permanently stall reclaim's borrow==0 wait. Releasing in the
     * always-run caller is leak-proof. */
    /* CP 2.4 Fix A (2026-05-16): pin the GEMM dispatch to the primary
     * context, exactly as the quant path is pinned. The Marlin GEMM kernel
     * is structurally full-GPU — persistent-style, grid = SM count, with
     * inter-CTA split-K through the `locks` buffer; its CTAs must be
     * co-resident. Launched into a partitioned context (e.g. the 8-SM green
     * context) it deadlocks — confirmed by instrumentation, see
     * cp_2_4/MARLIN_HANG_ROOT_CAUSE.md. PrimaryCtxGuard makes the device-0
     * primary context current for the whole launch and restores the caller's
     * context on exit (RAII, every path). Single-tenant scope; multi-tenant
     * Marlin + SM-partitioning is a Phase 5 architectural problem — see
     * cp_2_4/PHASE_5_MARLIN_PARTITION_CONSTRAINT.md.
     *
     * CP 5.3 STEP 2 (2026-05-17): the primary-context pin is now scoped to
     * the single-tenant / no-partition path only. When a green context is
     * active the GEMM launches green-bound so its (partition-sized) grid is
     * confined to the partition — this is the deadlock fix CP 2.4 deferred
     * to Phase 5. Single-tenant path keeps the guard, byte-identical to the
     * shipped behaviour. Scope memo §3. */
    /* G-O7 graph gate: in CAPTURE-SAFE mode skip the PrimaryCtxGuard + cross-context completion-event dance.
     * Inside a CUDA-graph capture the whole graph runs in ONE context (torch's primary context — where
     * Marlin's cubin is loaded), so (a) the GEMM CUfunction is already valid without a ctx switch, and (b) the
     * F1 cross-context race cannot occur: the C-buffer consumer is a downstream graph node ordered after the
     * GEMM node via the buffer dependency. cuCtxSetCurrent + per-call cuEventCreate/Record/StreamWaitEvent are
     * all capture-hostile (invalidate stream capture); the plain launch below is capture-legal. Default-OFF. */
    if (!g_capture_safe.load(std::memory_order_relaxed) && cipher_rt_green_ctx_sm_count() == 0) {
        /* CP 5.6 Priority 1 (finding F1): PrimaryCtxGuard pins the GEMM to
         * the primary context. When the caller runs on a *different* context
         * the GEMM (producer, on the primary NULL stream) and the caller's
         * buffer-C consumer are decoupled — independent unsynchronised
         * streams in two contexts — so the consumer races the GEMM and reads
         * C before it is written (verified: cp_5_6/f1_p1_verify, 99/100
         * degenerate on the pre-fix build). Fix: record a completion event
         * on the GEMM's launch stream while the primary context is current,
         * then — after PrimaryCtxGuard restores the caller's context — make
         * the caller's stream wait on it. cuStreamWaitEvent honours an event
         * recorded in a different context, so this is the cross-context
         * ordering primitive. Non-blocking: the host does not stall; the
         * dependency is enforced GPU-side. */
        bool ok;
        void *done = nullptr;
        {
            PrimaryCtxGuard ctx_guard;          /* primary context current */
            ok = marlin_gemm_launch(a_fp16, marlin_B, marlin_S,
                                    c_fp16, M, N, K, G, stream);
            if (ok && g_api.event_create && g_api.event_record) {
                /* 0x2 = CU_EVENT_DISABLE_TIMING */
                if (g_api.event_create(&done, 0x2u) != 0) {
                    done = nullptr;
                } else if (g_api.event_record(done, stream) != 0) {
                    if (g_api.event_destroy) g_api.event_destroy(done);
                    done = nullptr;
                }
            }
        }   /* ~PrimaryCtxGuard: the caller's context is restored here */
        if (done) {
            /* caller context current again; the wait is enqueued on the
             * same `stream` handle the caller passed (a NULL stream resolves
             * to the caller context's default stream — the F1 race case). */
            if (g_api.stream_wait_ev)
                (void)g_api.stream_wait_ev(stream, done, 0);
            if (g_api.event_destroy) g_api.event_destroy(done);
        }
        return ok ? 0 : -1;
    }
    return marlin_gemm_launch(a_fp16, marlin_B, marlin_S,
                              c_fp16, M, N, K, G, stream) ? 0 : -1;
}

extern "C" unsigned long cipher_rt_marlin_engine_weights_count(void)
{
    std::lock_guard<std::mutex> lk(g_weight_mu);
    unsigned long n = 0;
    for (auto &kv : g_weights) if (kv.second.ready) n++;
    return n;
}

/* CP 2.4 diagnostic: distinct caller streams with a live workspace slot —
 * i.e. the per-stream concurrency width the Marlin path is sustaining. */
extern "C" int cipher_rt_marlin_engine_ws_slots(void)
{
    return marlin_ws_live_slots();
}

/* W.2 (2026-05-27): bf16 cast launch helper. n = element count.
 * Returns 0 on success. Synchronous within stream. */
static int launch_bf16_to_fp16(const void *in_bf16, void *out_fp16,
                               long long n, void *stream)
{
    if (!g_quant_fn_bf16_to_fp16) return -1;
    int bx = 256;
    int gx = (int)((n + bx - 1) / bx);
    void *args[] = { (void *)&in_bf16, &out_fp16, &n };
    int rc = g_api.launch_kernel(g_quant_fn_bf16_to_fp16, gx, 1, 1, bx, 1, 1,
                                  0, stream, args, nullptr);
    return rc;
}

static int launch_fp16_to_bf16(const void *in_fp16, void *out_bf16,
                               long long n, void *stream)
{
    if (!g_quant_fn_fp16_to_bf16) return -1;
    int bx = 256;
    int gx = (int)((n + bx - 1) / bx);
    void *args[] = { (void *)&in_fp16, &out_bf16, &n };
    int rc = g_api.launch_kernel(g_quant_fn_fp16_to_bf16, gx, 1, 1, bx, 1, 1,
                                  0, stream, args, nullptr);
    return rc;
}

/* W.2 (2026-05-27): bf16 quantize_repack wrapper.
 *
 * The existing quantize_repack expects fp16 weights. For bf16 source, we
 * cast bf16 -> fp16 into a temporary buffer first, then run the existing
 * RTN INT4 path. Resulting Marlin INT4 layout is cached in g_weights under
 * the SAME w_ptr (the bf16 source pointer); subsequent lookups hit it.
 *
 * Calibration honesty: this is RTN quantization with per-group scales —
 * NOT AWQ/GPTQ calibrated. Quality gap vs AWQ on Llama-3-8B is ~0.5-1.0
 * perplexity points per published comparisons. KL gate (Sub-step 8) catches
 * any specific weight tensor that regresses beyond tolerance. */
extern "C" int cipher_rt_marlin_engine_quantize_repack_bf16(
    const void *d_bf16_weight, int K, int N)
{
    if (!ensure_marlin_compiled()) return -1;
    if (!ensure_quant_compiled()) return -1;
    if (!g_quant_fn_bf16_to_fp16) {
        cipher_log("MARLIN: bf16 quant skipped — cast kernel unavailable");
        return -1;
    }
    /* Cast bf16 -> fp16 into temp device buffer, then call existing
     * quantize_repack with the fp16 buffer + the ORIGINAL bf16 w_ptr as
     * the cache key (so the lookup path at dispatch time finds it via the
     * caller-supplied bf16 pointer). */
    size_t weight_bytes = (size_t)K * (size_t)N * 2;  /* 2 bytes/element */
    void *d_fp16 = nullptr;
    if (g_api.rt_malloc(&d_fp16, weight_bytes) != 0) return -1;
    if (launch_bf16_to_fp16(d_bf16_weight, d_fp16, (long long)K * N, nullptr) != 0) {
        g_api.rt_free(d_fp16);
        return -1;
    }
    /* The g_weights cache key uses w_ptr identity. We want a lookup with
     * the BF16 source pointer to succeed downstream, so we re-key by
     * temporarily passing d_bf16_weight as the cache key while quantizing
     * from d_fp16. This is done inside ensure_weight_quantized_repacked
     * by passing d_fp16_weight = d_fp16 and key by w_ptr separately —
     * but the existing API doesn't expose this. Workaround: quantize via
     * the d_fp16 pointer as key (different from caller's bf16 ptr) and
     * have the dispatch wrapper do the same recasting. */
    bool ok = ensure_weight_quantized_repacked(d_fp16, K, N);
    /* Note: d_fp16 stays alive — it IS the cache key. Don't free until
     * cache invalidation (process exit or future W.4 POOL eviction). The
     * dispatch_bf16 wrapper holds a parallel map bf16_ptr -> fp16_ptr.
     * (One slot per bf16 weight; same lifetime semantics as g_weights.) */
    if (!ok) {
        g_api.rt_free(d_fp16);
        return -1;
    }
    /* Record the bf16 -> fp16 mapping (Layer-1). GC-on-free: carry (K,N) + a
     * fingerprint of the APP bf16 weight so a reused bf16 pointer is detected
     * at bf16_surrogate before the stale surrogate is handed out. */
    extern std::mutex g_bf16_weight_mu;
    extern std::unordered_map<const void*, Bf16Entry> g_bf16_to_fp16_weight;
    uint64_t fp_bf16 = g_gc_fp_enabled ? weight_fingerprint(d_bf16_weight, K, N, 2) : 0;
    {
        std::lock_guard<std::mutex> lk(g_bf16_weight_mu);
        Bf16Entry e; e.surrogate = d_fp16; e.fp = fp_bf16; e.K = K; e.N = N;
        g_bf16_to_fp16_weight[d_bf16_weight] = e;
    }
    return 0;
}

/* W.2: bf16 weight pointer -> the fp16 surrogate used as cache key in
 * g_weights. The dispatch_bf16 wrapper resolves the surrogate at call
 * time so lookup succeeds. GC-on-free (2026-05-30): values carry (K,N)+fp
 * (Bf16Entry) for reuse detection. */
std::mutex g_bf16_weight_mu;
std::unordered_map<const void*, Bf16Entry> g_bf16_to_fp16_weight;

/* GC-on-free public API. evict_weight: the residence layer calls this when a
 * model's weights are freed (completing the bind_model lifecycle). Removes the
 * Layer-1 surrogate entry and all Layer-2 kit slots for the pointer AND the
 * surrogate, retiring every buffer. Idempotent; safe for unknown ptrs.
 * Lock order g_bf16_weight_mu → g_weight_mu (sequential, never nested) →
 * g_retire_mu (leaf, inside evict_g_weights_for_ptr_locked / retire_buf). */
extern "C" void cipher_rt_marlin_engine_evict_weight(const void *app_w_ptr)
{
    if (!app_w_ptr) return;
    void *surrogate = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_bf16_weight_mu);
        auto it = g_bf16_to_fp16_weight.find(app_w_ptr);
        if (it != g_bf16_to_fp16_weight.end()) {
            surrogate = it->second.surrogate;
            g_bf16_to_fp16_weight.erase(it);
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_weight_mu);
        evict_g_weights_for_ptr_locked(app_w_ptr);                 /* fp16-direct kit */
        if (surrogate) evict_g_weights_for_ptr_locked(surrogate);  /* bf16 surrogate kit */
        g_wptr_model.erase(app_w_ptr);
    }
    if (surrogate) retire_buf(surrogate);                          /* the fp16 surrogate */
}

/* Free all retired buffers after a quiescent device-sync barrier — every
 * Marlin kernel launched before now has completed, so no in-flight launch
 * still references a retired buffer (the safe-reclaim invariant). The
 * residence layer / churn harness calls this at a per-cycle barrier (after its
 * own torch.cuda.synchronize). If cudaDeviceSynchronize is unavailable the
 * buffers stay retired (bounded leak) rather than risk a UAF. Returns the
 * count freed, 0 if nothing pending, -1 if reclaim was skipped. */
extern "C" int cipher_rt_marlin_engine_reclaim_retired(void)
{
    if (!g_api.rt_free) return -1;
    std::vector<void*> local;
    {
        std::lock_guard<std::mutex> lk(g_retire_mu);
        if (g_retired.empty()) return 0;
        local.swap(g_retired);
    }
    if (!g_api.rt_devsync) {
        std::lock_guard<std::mutex> lk(g_retire_mu);
        for (void *p : local) g_retired.push_back(p);
        return -1;
    }
    /* (1) borrow barrier: wait for all in-flight kit borrows to drain. A
     * retired buffer is no longer reachable by any NEW lookup, so once
     * borrow==0 every thread that borrowed one of THESE buffers (necessarily
     * before its eviction) has reached dispatch() and released it. Bounded so a
     * (non-actuator) leaked borrow cannot hang reclaim — real dispatches drop
     * borrow to 0 between launches, so this returns within microseconds. */
    for (long i = 0; i < 50000000L &&
                     g_dispatch_borrow.load(std::memory_order_acquire) > 0; ++i) {
        /* busy-wait */
    }
    /* (2) device-sync: every launched Marlin kernel has completed, so no
     * in-flight kernel still reads any retired buffer. */
    (void)g_api.rt_devsync();
    int n = 0;
    for (void *p : local) { if (p) { g_api.rt_free(p); n++; } }
    return n;
}

/* Observability: count of fingerprint sample failures while gc_fp is on
 * (each leaves the affected slot on dim-revalidation only). 0 in a healthy run. */
extern "C" unsigned long cipher_rt_marlin_engine_gc_fp_failures(void)
{
    return g_fp_fail.load(std::memory_order_relaxed);
}

/* Release the dispatch borrow taken in lookup(). Called by the actuator after
 * dispatch()/dispatch_bf16() returns, on BOTH success and error branches, so
 * the borrow is leak-proof against dispatch_bf16's early-error returns. */
extern "C" void cipher_rt_marlin_engine_release_dispatch_borrow(void)
{
    g_dispatch_borrow.fetch_sub(1, std::memory_order_acq_rel);
}

/* Observability/self-check: current in-flight dispatch borrow count. Must
 * return to 0 when no Marlin GEMM is mid lookup→launch (assert after a soak). */
extern "C" int cipher_rt_marlin_engine_dispatch_borrow(void)
{
    return g_dispatch_borrow.load(std::memory_order_acquire);
}

extern "C" void *cipher_rt_marlin_engine_bf16_surrogate(const void *bf16_w_ptr,
                                                        int K, int N)
{
    /* GC-on-free validate-on-use (Layer-1, the dominant bf16 production
     * surface). Resolve the surrogate + check (K,N); for a same-shape hit,
     * re-verify content identity via a fingerprint (DtoH) OUTSIDE the lock.
     * On staleness, evict both layers and return null so the actuator
     * re-quantizes the current model. */
    void *surrogate = nullptr;
    bool  stale = false;
    {
        std::lock_guard<std::mutex> lk(g_bf16_weight_mu);
        auto it = g_bf16_to_fp16_weight.find(bf16_w_ptr);
        if (it == g_bf16_to_fp16_weight.end()) return nullptr;
        surrogate = it->second.surrogate;
        if (it->second.K != K || it->second.N != N) stale = true;
    }
    if (!stale && g_gc_fp_enabled) {
        uint64_t live = weight_fingerprint(bf16_w_ptr, K, N, 2);
        uint64_t stored = 0;
        {
            std::lock_guard<std::mutex> lk(g_bf16_weight_mu);
            auto it = g_bf16_to_fp16_weight.find(bf16_w_ptr);
            if (it == g_bf16_to_fp16_weight.end()) return nullptr;  /* raced an evict */
            stored = it->second.fp;
        }
        if (live && stored && live != stored) stale = true;
    }
    if (stale) {
        cipher_rt_marlin_engine_evict_weight(bf16_w_ptr);
        return nullptr;
    }
    return surrogate;
}

/* W.3 (2026-05-27) Sub-step 2: shared bf16→fp16 cast allocator for EDMD
 * sample-collection wiring in may13_cipher_dispatch.cpp's
 * edmd_live_post_relaunch_hook. Allocates a device temp buffer + runs
 * the W.2 NVRTC cast kernel. Returns nullptr on failure (caller falls
 * through; collect is best-effort). */
extern "C" void *cipher_rt_marlin_engine_cast_bf16_to_fp16_alloc(
    const void *in_bf16, long long n_elements, void *stream)
{
    if (!g_quant_fn_bf16_to_fp16 || !in_bf16 || n_elements <= 0) return nullptr;
    void *out_fp16 = nullptr;
    if (g_api.rt_malloc(&out_fp16, (size_t)n_elements * 2) != 0) return nullptr;
    if (launch_bf16_to_fp16(in_bf16, out_fp16, n_elements, stream) != 0) {
        g_api.rt_free(out_fp16);
        return nullptr;
    }
    return out_fp16;
}

extern "C" void cipher_rt_marlin_engine_free_cast_temp(void *p)
{
    if (p) g_api.rt_free(p);
}

/* W.2 (2026-05-27): bf16 Marlin dispatch wrapper.
 *
 * Casts bf16 activation A to fp16 temp; calls existing fp16 dispatch with
 * Marlin INT4 weights; casts fp16 output back to bf16 C buffer.
 * Activation+output temps are per-call (small; cheap to allocate). */
/* G-O7 graph gate: capture-safe cast alloc. When CIPHER_MARLIN_CAPTURE_SAFE=1, the per-call bf16<->fp16 cast
 * temps use stream-ordered cudaMallocAsync/FreeAsync (capture-legal — records as a graph mem node) instead of
 * cudaMalloc/Free (which throw cudaErrorStreamCaptureInvalidated inside a CUDA-graph capture). Per-call +
 * stream-ordered = correct under the N-stream concurrent multiplex (no shared-buffer race). Default-OFF;
 * OFF path is byte-identical to the prior plain cudaMalloc/Free. Set in cipher_rt_marlin_engine_init.
 * (g_capture_safe declared with the globals near g_marlin_state.) */
static inline int marlin_cast_alloc(void **p, size_t bytes, void *stream) {
    if (g_capture_safe.load(std::memory_order_relaxed) && g_api.rt_malloc_async)
        return g_api.rt_malloc_async(p, bytes, stream);
    return g_api.rt_malloc(p, bytes);
}
static inline void marlin_cast_free(void *p, void *stream) {
    if (g_capture_safe.load(std::memory_order_relaxed) && g_api.rt_free_async)
        (void)g_api.rt_free_async(p, stream);
    else (void)g_api.rt_free(p);
}

extern "C" int cipher_rt_marlin_engine_dispatch_bf16(
    const void *a_bf16, const void *marlin_B, const void *marlin_S,
    void *c_bf16, int M, int N, int K, int G, void *stream)
{
    /* TEST-ONLY: force the early-error path on ~1/3 of calls to exercise the
     * actuator's error-branch borrow release. Mirrors a malloc-fail/return -1
     * BEFORE cipher_rt_marlin_engine_dispatch() runs. */
    if (g_gc_fault_bf16 &&
        (g_bf16_call_ctr.fetch_add(1, std::memory_order_relaxed) % 3 == 0))
        return -1;
    if (!g_quant_fn_bf16_to_fp16 || !g_quant_fn_fp16_to_bf16) return -1;
    long long a_elems = (long long)M * (long long)K;
    long long c_elems = (long long)M * (long long)N;
    void *a_fp16 = nullptr;
    void *c_fp16 = nullptr;
    if (marlin_cast_alloc(&a_fp16, (size_t)a_elems * 2, stream) != 0) return -1;
    if (marlin_cast_alloc(&c_fp16, (size_t)c_elems * 2, stream) != 0) {
        marlin_cast_free(a_fp16, stream); return -1;
    }
    if (launch_bf16_to_fp16(a_bf16, a_fp16, a_elems, stream) != 0) {
        marlin_cast_free(a_fp16, stream); marlin_cast_free(c_fp16, stream); return -1;
    }
    int rc = cipher_rt_marlin_engine_dispatch(a_fp16, marlin_B, marlin_S,
                                              c_fp16, M, N, K, G, stream);
    if (rc != 0) {
        marlin_cast_free(a_fp16, stream); marlin_cast_free(c_fp16, stream); return rc;
    }
    rc = launch_fp16_to_bf16(c_fp16, c_bf16, c_elems, stream);
    marlin_cast_free(a_fp16, stream);
    marlin_cast_free(c_fp16, stream);
    return rc;
}
