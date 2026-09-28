// CIPHER FP8 Compute Substitution — Stage 13 implementation.
//
// Three pieces:
//   1) NVRTC-compiled per-tensor absmax + FP8 (E4M3) quantize kernels
//      for both weights (one-time) and activations (per-call, ~2us at
//      decode M).
//   2) Per-weight registry keyed by fp16 device pointer.
//   3) cublasLtMatmul wrapper that feeds (FP8 weight, FP8 activation,
//      A_scale, B_scale) into a CUBLAS_COMPUTE_32F matmul producing fp16.
//
// Layout follows PyTorch's standard cublasGemmEx convention for
// `y = x @ W^T` with W [N, K] row-major:
//   m_cublas = N, n_cublas = M, k_cublas = K, transA = T, transB = N.
// In storage terms this means A is N×K row-major (= K×N col-major) and
// B is M×K row-major (= K×M col-major), and our FP8 path mirrors that.

#include "cipher_fp8_compute.h"
#include "cipher_substitute_v2.h"

#include <cuda_runtime.h>

#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

// ── Constants we cannot rely on cublasLt.h for ─────────────────────────
// (we go through dlopen so we never link the header.)
constexpr int kCudaR16F   = 2;     // CUDA_R_16F
constexpr int kCudaR32F   = 0;     // CUDA_R_32F
constexpr int kCudaR8FE4M3 = 28;   // CUDA_R_8F_E4M3
constexpr int kCublasOpN  = 0;
constexpr int kCublasOpT  = 1;
constexpr int kCublasCompute32F = 68;

constexpr int kAttrTransA       = 3;
constexpr int kAttrTransB       = 4;
constexpr int kAttrAScalePtr    = 17;
constexpr int kAttrBScalePtr    = 18;

// Engine-level limits.
constexpr int    MAX_WEIGHTS    = 512;
constexpr size_t LT_WS_BYTES    = 32 * 1024 * 1024;   // 32 MB cublasLt workspace
constexpr size_t MAX_M          = 256;                // FP8 path M cap
constexpr size_t MAX_K          = 32 * 1024;          // covers Mistral 14336

// ── State ──────────────────────────────────────────────────────────────

std::atomic<int>      g_enabled{0};
std::atomic<uint64_t> g_weights_quantized{0};
std::atomic<uint64_t> g_matmul_calls{0};
std::atomic<uint64_t> g_passthroughs{0};
std::atomic<uint64_t> g_correctness_failures{0};
std::atomic<size_t>   g_bytes_fp16{0};
std::atomic<size_t>   g_bytes_fp8{0};

struct WeightEntry {
    void*  fp16_key;     // canonical pointer used by cublasGemmEx
    void*  fp8_buf;      // device fp8 e4m3 buffer, rows*cols bytes
    void*  scale_dev;    // device fp32 scalar = absmax/448
    int    rows;         // (= m_cublas)
    int    cols;         // (= k_cublas)
    int    hits;         // observation count for stability detection
};
WeightEntry g_weights[MAX_WEIGHTS]{};
std::mutex  g_weights_mu;

int find_weight_locked(void* key) {
    for (int i = 0; i < MAX_WEIGHTS; ++i) {
        if (g_weights[i].fp16_key == key) return i;
    }
    return -1;
}
int alloc_weight_slot_locked() {
    for (int i = 0; i < MAX_WEIGHTS; ++i) {
        if (g_weights[i].fp16_key == nullptr) return i;
    }
    return -1;
}

// ── cublasLt resolver ──────────────────────────────────────────────────

typedef int (*pf_LtCreate)(void**);
typedef int (*pf_LtDestroy)(void*);
typedef int (*pf_LtMatmul)(
    void* /*lt*/, void* /*matmul_desc*/,
    const void* /*alpha*/,
    const void* /*A*/, void* /*A_layout*/,
    const void* /*B*/, void* /*B_layout*/,
    const void* /*beta*/,
    const void* /*C*/, void* /*C_layout*/,
    void* /*D*/, void* /*D_layout*/,
    const void* /*algo*/,
    void* /*workspace*/, size_t /*ws_size*/,
    void* /*stream*/);
typedef int (*pf_LtMatmulDescCreate)(void**, int /*compute*/, int /*scale*/);
typedef int (*pf_LtMatmulDescDestroy)(void*);
typedef int (*pf_LtMatmulDescSetAttribute)(void*, int, const void*, size_t);
typedef int (*pf_LtMatrixLayoutCreate)(void**, int /*type*/,
                                        uint64_t, uint64_t, int64_t);
typedef int (*pf_LtMatrixLayoutDestroy)(void*);

// Heuristic preference + algo selection.
typedef int (*pf_LtPrefCreate)(void**);
typedef int (*pf_LtPrefDestroy)(void*);
typedef int (*pf_LtPrefSetAttribute)(void*, int, const void*, size_t);
typedef int (*pf_LtAlgoGetHeuristic)(
    void* /*lt*/, void* /*desc*/,
    void* /*A*/, void* /*B*/, void* /*C*/, void* /*D*/,
    void* /*pref*/, int /*requested*/,
    void* /*results*/, int* /*returned*/);

constexpr int kPrefAttrMaxWorkspaceBytes = 0;  // CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES

struct LtFns {
    void* lib = nullptr;
    pf_LtCreate                  create = nullptr;
    pf_LtDestroy                 destroy = nullptr;
    pf_LtMatmul                  matmul = nullptr;
    pf_LtMatmulDescCreate        desc_create = nullptr;
    pf_LtMatmulDescDestroy       desc_destroy = nullptr;
    pf_LtMatmulDescSetAttribute  desc_set_attr = nullptr;
    pf_LtMatrixLayoutCreate      layout_create = nullptr;
    pf_LtMatrixLayoutDestroy     layout_destroy = nullptr;
    pf_LtPrefCreate              pref_create = nullptr;
    pf_LtPrefDestroy             pref_destroy = nullptr;
    pf_LtPrefSetAttribute        pref_set_attr = nullptr;
    pf_LtAlgoGetHeuristic        algo_heuristic = nullptr;
};
LtFns g_lt;
void* g_lt_handle    = nullptr;     // cublasLtHandle_t
void* g_lt_workspace = nullptr;     // 32 MB
void* g_act_fp8_ws   = nullptr;     // MAX_M * MAX_K bytes
void* g_act_scale_dev = nullptr;    // fp32 scalar
std::mutex g_lt_mu;

bool resolve_lt() {
    if (g_lt.create) return true;
    g_lt.lib = dlopen("libcublasLt.so.12", RTLD_LAZY | RTLD_LOCAL);
    if (!g_lt.lib) g_lt.lib = dlopen("libcublasLt.so", RTLD_LAZY | RTLD_LOCAL);
    if (!g_lt.lib) return false;

    g_lt.create        = (pf_LtCreate)                 dlsym(g_lt.lib, "cublasLtCreate");
    g_lt.destroy       = (pf_LtDestroy)                dlsym(g_lt.lib, "cublasLtDestroy");
    g_lt.matmul        = (pf_LtMatmul)                 dlsym(g_lt.lib, "cublasLtMatmul");
    g_lt.desc_create   = (pf_LtMatmulDescCreate)       dlsym(g_lt.lib, "cublasLtMatmulDescCreate");
    g_lt.desc_destroy  = (pf_LtMatmulDescDestroy)      dlsym(g_lt.lib, "cublasLtMatmulDescDestroy");
    g_lt.desc_set_attr = (pf_LtMatmulDescSetAttribute) dlsym(g_lt.lib, "cublasLtMatmulDescSetAttribute");
    g_lt.layout_create = (pf_LtMatrixLayoutCreate)     dlsym(g_lt.lib, "cublasLtMatrixLayoutCreate");
    g_lt.layout_destroy= (pf_LtMatrixLayoutDestroy)    dlsym(g_lt.lib, "cublasLtMatrixLayoutDestroy");
    g_lt.pref_create   = (pf_LtPrefCreate)             dlsym(g_lt.lib, "cublasLtMatmulPreferenceCreate");
    g_lt.pref_destroy  = (pf_LtPrefDestroy)            dlsym(g_lt.lib, "cublasLtMatmulPreferenceDestroy");
    g_lt.pref_set_attr = (pf_LtPrefSetAttribute)       dlsym(g_lt.lib, "cublasLtMatmulPreferenceSetAttribute");
    g_lt.algo_heuristic= (pf_LtAlgoGetHeuristic)       dlsym(g_lt.lib, "cublasLtMatmulAlgoGetHeuristic");
    return g_lt.create && g_lt.matmul && g_lt.desc_create && g_lt.desc_set_attr
        && g_lt.layout_create && g_lt.pref_create && g_lt.algo_heuristic;
}

bool ensure_lt_handle() {
    if (g_lt_handle) return true;
    std::lock_guard<std::mutex> lk(g_lt_mu);
    if (g_lt_handle) return true;
    if (!resolve_lt()) return false;
    if (g_lt.create(&g_lt_handle) != 0 || !g_lt_handle) return false;
    if (cudaMalloc(&g_lt_workspace, LT_WS_BYTES) != cudaSuccess) return false;
    if (cudaMalloc(&g_act_fp8_ws, MAX_M * MAX_K) != cudaSuccess) return false;
    if (cudaMalloc(&g_act_scale_dev, sizeof(float)) != cudaSuccess) return false;
    return true;
}

// ── NVRTC kernels: per-tensor absmax + FP8 E4M3 quantize ────────────────

const char kFp8KernelSrc[] = R"NVRTC(
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Per-tensor absmax via per-block reduction + atomicMax on the bit
// pattern of a non-negative float (monotonic for x >= 0 in IEEE-754).
//
// Caller MUST cudaMemset *absmax to zero before launch.
extern "C" __global__ void cipher_fp8_absmax(
    const __half* __restrict__ in,
    int* __restrict__ absmax_bits,    // fp32 reinterpreted as int
    int n_elem)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float local = 0.0f;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n_elem; i += stride) {
        float v = __half2float(in[i]);
        float av = v < 0.0f ? -v : v;
        if (av > local) local = av;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid], b = sdata[tid + s];
            if (b > a) sdata[tid] = b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int my_bits = __float_as_int(sdata[0]);
        atomicMax(absmax_bits, my_bits);
    }
}

// Convert in-place: scale[0] currently holds absmax (as fp32 after the
// reinterpret), turn it into absmax/448.0f.
extern "C" __global__ void cipher_fp8_finalize_scale(float* scale)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float a = scale[0];
        scale[0] = (a > 0.0f) ? (a / 448.0f) : 1.0f;
    }
}

// Quantize fp16 -> FP8 e4m3 using the device-resident scale (= absmax/448).
// Reads scale_dev[0] once per thread; could be cached in shmem but the
// kernel is HBM-bound on n_elem.
extern "C" __global__ void cipher_fp8_quantize(
    const __half* __restrict__ in,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ scale_dev,
    int n_elem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elem) return;
    float s = scale_dev[0];
    float inv = 1.0f / (s > 0.0f ? s : 1.0f);
    float v = __half2float(in[idx]) * inv;
    // Saturate to E4M3 representable range.
    if (v >  448.0f) v =  448.0f;
    if (v < -448.0f) v = -448.0f;
    out[idx] = __nv_fp8_e4m3(v);
}
)NVRTC";

unsigned long g_cubin_absmax  = 0;
unsigned long g_cubin_finalize = 0;
unsigned long g_cubin_quant   = 0;
std::mutex    g_compile_mu;

bool ensure_kernels_compiled() {
    std::lock_guard<std::mutex> lk(g_compile_mu);
    if (g_cubin_absmax && g_cubin_finalize && g_cubin_quant) return true;
    if (!cipher_substitute_v2_enabled()) return false;
    if (!g_cubin_absmax)
        g_cubin_absmax = cipher_substitute_v2_compile(kFp8KernelSrc, "cipher_fp8_absmax");
    if (!g_cubin_finalize)
        g_cubin_finalize = cipher_substitute_v2_compile(kFp8KernelSrc, "cipher_fp8_finalize_scale");
    if (!g_cubin_quant)
        g_cubin_quant = cipher_substitute_v2_compile(kFp8KernelSrc, "cipher_fp8_quantize");
    return g_cubin_absmax && g_cubin_finalize && g_cubin_quant;
}

// libcuda launcher resolver.
typedef int (*pf_cuLaunchKernel)(void*, unsigned, unsigned, unsigned,
                                  unsigned, unsigned, unsigned,
                                  unsigned, void*, void**, void**);
pf_cuLaunchKernel g_cu_launch = nullptr;

bool resolve_launcher() {
    if (g_cu_launch) return true;
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) return false;
    g_cu_launch = (pf_cuLaunchKernel)dlsym(lib, "cuLaunchKernel");
    return g_cu_launch != nullptr;
}

// Run absmax + finalize + quantize over a contiguous fp16 buffer.
// `scale_dev` must be a device fp32 scalar; on entry contents are
// overwritten.  `stream` is the launch stream (may be 0 for default).
bool quantize_to_fp8(const void* in_fp16, void* out_fp8, void* scale_dev,
                     int n_elem, void* stream) {
    if (!ensure_kernels_compiled()) return false;
    if (!resolve_launcher())        return false;

    void* fn_absmax   = cipher_substitute_v2_get_function(g_cubin_absmax);
    void* fn_finalize = cipher_substitute_v2_get_function(g_cubin_finalize);
    void* fn_quant    = cipher_substitute_v2_get_function(g_cubin_quant);
    if (!fn_absmax || !fn_finalize || !fn_quant) return false;

    // Zero the scale buffer (4 bytes) on the stream.
    if (cudaMemsetAsync(scale_dev, 0, sizeof(float),
                         (cudaStream_t)stream) != cudaSuccess) return false;

    // absmax: 256 threads/block, up to 256 blocks, persistent loop.
    {
        int bx = 256;
        int gx = (n_elem + bx - 1) / bx;
        if (gx > 256) gx = 256;
        void* args[] = { (void*)&in_fp16, &scale_dev, &n_elem };
        if (g_cu_launch(fn_absmax, gx, 1, 1, bx, 1, 1, 0, stream, args, nullptr) != 0)
            return false;
    }
    // finalize: scale = absmax/448
    {
        void* args[] = { &scale_dev };
        if (g_cu_launch(fn_finalize, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr) != 0)
            return false;
    }
    // quantize
    {
        int bx = 256;
        int gx = (n_elem + bx - 1) / bx;
        void* args[] = { (void*)&in_fp16, &out_fp8, &scale_dev, &n_elem };
        if (g_cu_launch(fn_quant, gx, 1, 1, bx, 1, 1, 0, stream, args, nullptr) != 0)
            return false;
    }
    return true;
}

}  // namespace

// ── Public API ──────────────────────────────────────────────────────────

extern "C" int cipher_fp8_compute_init(void) {
    if (g_enabled.load(std::memory_order_relaxed)) return 1;
    const char* env = getenv("CIPHER_FP8_COMPUTE");
    int on = env && (env[0] == '1' || env[0] == 'o' || env[0] == 'O' ||
                     !strcmp(env, "on") || !strcmp(env, "ON"));
    if (!on) return 0;
    g_enabled.store(1, std::memory_order_relaxed);
    if (getenv("CIPHER_FP8_VERBOSE"))
        fprintf(stderr, "[CIPHER FP8] enabled\n");
    return 1;
}

extern "C" int cipher_fp8_compute_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_fp8_compute_observe(void* weight_fp16, int rows, int cols) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!weight_fp16) return 0;
    std::lock_guard<std::mutex> lk(g_weights_mu);
    int idx = find_weight_locked(weight_fp16);
    if (idx >= 0) {
        if (g_weights[idx].rows != rows || g_weights[idx].cols != cols) {
            // shape changed under us — reset.  cudaFree-then-realloc the
            // buffers happens lazily inside _quantize_weight.
            if (g_weights[idx].fp8_buf)   cudaFree(g_weights[idx].fp8_buf);
            if (g_weights[idx].scale_dev) cudaFree(g_weights[idx].scale_dev);
            g_weights[idx] = WeightEntry{};
            g_weights[idx].fp16_key = weight_fp16;
            g_weights[idx].rows = rows;
            g_weights[idx].cols = cols;
        }
        return ++g_weights[idx].hits;
    }
    idx = alloc_weight_slot_locked();
    if (idx < 0) return 0;
    g_weights[idx].fp16_key = weight_fp16;
    g_weights[idx].rows = rows;
    g_weights[idx].cols = cols;
    g_weights[idx].hits = 1;
    return 1;
}

extern "C" int cipher_fp8_compute_quantize_weight(
    void* weight_fp16, int rows, int cols, void* stream)
{
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!weight_fp16 || rows <= 0 || cols <= 0) return 0;

    // Idempotent on (ptr, rows, cols).  PyTorch's caching allocator can
    // hand back the same address for a different tensor — invalidate on
    // shape mismatch.
    int idx;
    {
        std::lock_guard<std::mutex> lk(g_weights_mu);
        idx = find_weight_locked(weight_fp16);
        if (idx >= 0) {
            // Shape mismatch?  Release stale buffers and fall through to
            // re-quantize in the same slot.
            if (g_weights[idx].rows != rows || g_weights[idx].cols != cols) {
                if (g_weights[idx].fp8_buf)   cudaFree(g_weights[idx].fp8_buf);
                if (g_weights[idx].scale_dev) cudaFree(g_weights[idx].scale_dev);
                g_weights[idx].fp8_buf = nullptr;
                g_weights[idx].scale_dev = nullptr;
                g_weights[idx].rows = rows;
                g_weights[idx].cols = cols;
            }
            // If already quantized and shape matches, we're done.
            if (g_weights[idx].fp8_buf) return 1;
            // Else fall through, keeping the slot index — the slot was
            // pre-reserved by an _observe() call.
        } else {
            idx = alloc_weight_slot_locked();
            if (idx < 0) return 0;
            g_weights[idx].fp16_key = weight_fp16;
            g_weights[idx].rows = rows;
            g_weights[idx].cols = cols;
        }
    }

    size_t n_elem = (size_t)rows * (size_t)cols;
    void* fp8_buf = nullptr;
    void* scale_dev = nullptr;
    if (cudaMalloc(&fp8_buf, n_elem) != cudaSuccess) {
        std::lock_guard<std::mutex> lk(g_weights_mu);
        g_weights[idx] = WeightEntry{};
        return 0;
    }
    if (cudaMalloc(&scale_dev, sizeof(float)) != cudaSuccess) {
        cudaFree(fp8_buf);
        std::lock_guard<std::mutex> lk(g_weights_mu);
        g_weights[idx] = WeightEntry{};
        return 0;
    }

    if (!quantize_to_fp8(weight_fp16, fp8_buf, scale_dev, (int)n_elem, stream)) {
        cudaFree(fp8_buf);
        cudaFree(scale_dev);
        std::lock_guard<std::mutex> lk(g_weights_mu);
        g_weights[idx] = WeightEntry{};
        return 0;
    }

    {
        std::lock_guard<std::mutex> lk(g_weights_mu);
        g_weights[idx].fp8_buf = fp8_buf;
        g_weights[idx].scale_dev = scale_dev;
    }
    g_weights_quantized.fetch_add(1, std::memory_order_relaxed);
    g_bytes_fp16.fetch_add(n_elem * 2, std::memory_order_relaxed);
    g_bytes_fp8.fetch_add(n_elem,     std::memory_order_relaxed);
    if (getenv("CIPHER_FP8_VERBOSE"))
        fprintf(stderr, "[CIPHER FP8] quantized weight %p shape %dx%d -> fp8 %p scale %p\n",
                weight_fp16, rows, cols, fp8_buf, scale_dev);
    return 1;
}

extern "C" int cipher_fp8_compute_weight_ready(void* weight_fp16) {
    std::lock_guard<std::mutex> lk(g_weights_mu);
    int i = find_weight_locked(weight_fp16);
    return (i >= 0 && g_weights[i].fp8_buf != nullptr) ? 1 : 0;
}

extern "C" int cipher_fp8_compute_lookup(
    void* weight_fp16,
    void** out_fp8_buf, void** out_scale_dev,
    int* out_rows, int* out_cols)
{
    std::lock_guard<std::mutex> lk(g_weights_mu);
    int i = find_weight_locked(weight_fp16);
    if (i < 0 || !g_weights[i].fp8_buf) return 0;
    if (out_fp8_buf)   *out_fp8_buf   = g_weights[i].fp8_buf;
    if (out_scale_dev) *out_scale_dev = g_weights[i].scale_dev;
    if (out_rows)      *out_rows      = g_weights[i].rows;
    if (out_cols)      *out_cols      = g_weights[i].cols;
    return 1;
}

extern "C" int cipher_fp8_compute_matmul(
    void* weight_fp16_key,
    const void* activation_fp16,
    void* c_fp16,
    int m_cublas, int n_cublas, int k_cublas,
    void* stream)
{
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!ensure_lt_handle())                        return 0;

    void* fp8_w = nullptr;
    void* w_scale = nullptr;
    int   w_rows = 0, w_cols = 0;
    {
        std::lock_guard<std::mutex> lk(g_weights_mu);
        int i = find_weight_locked(weight_fp16_key);
        if (i < 0 || !g_weights[i].fp8_buf) return 0;
        // Sanity: cuBLAS-style m_cublas == weight rows, k_cublas == weight cols
        if (g_weights[i].rows != m_cublas || g_weights[i].cols != k_cublas) {
            g_passthroughs.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        fp8_w   = g_weights[i].fp8_buf;
        w_scale = g_weights[i].scale_dev;
        w_rows  = g_weights[i].rows;
        w_cols  = g_weights[i].cols;
    }

    // Capacity guard for the activation workspace.
    size_t act_n = (size_t)n_cublas * (size_t)k_cublas;
    if (act_n > MAX_M * MAX_K) {
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // Quantize activation fp16 -> fp8 inline.  In cuBLAS terms B is k×n
    // col-major == n×k row-major; for absmax it's just a contiguous span
    // of n*k fp16 values.
    if (!quantize_to_fp8(activation_fp16, g_act_fp8_ws, g_act_scale_dev,
                          (int)act_n, stream)) {
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // Build matmul descriptor + layouts.  cublasLt is happy to be created
    // and torn down per call (the heavy state lives in the LtHandle and
    // the workspace).  At ~64 layers x 7 linears = 448 calls / token even
    // at decode this is on the order of 100 ns per descriptor — well
    // below the FP8 GEMM time.  We can cache later if needed.
    void* desc = nullptr;
    if (g_lt.desc_create(&desc, kCublasCompute32F, kCudaR32F) != 0) {
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    int transA = kCublasOpT;     // weight is N×K row-major == K×N col-major,
                                  // op(A) = A^T gives the N×K we want.
    int transB = kCublasOpN;     // activation is M×K row-major == K×M col-major,
                                  // op(B) = B already gives K×M we want.
    g_lt.desc_set_attr(desc, kAttrTransA, &transA, sizeof(transA));
    g_lt.desc_set_attr(desc, kAttrTransB, &transB, sizeof(transB));
    g_lt.desc_set_attr(desc, kAttrAScalePtr, &w_scale,         sizeof(void*));
    g_lt.desc_set_attr(desc, kAttrBScalePtr, &g_act_scale_dev, sizeof(void*));

    // Layouts (col-major, default).  Storage shapes:
    //   A_fp8 storage: N×K row-major == K×N col-major  -> rows=K, cols=N, ld=K
    //   B_fp8 storage: M×K row-major == K×M col-major  -> rows=K, cols=M, ld=K
    //   C_fp16 storage: M×N row-major == N×M col-major -> rows=N, cols=M, ld=N
    void* layoutA = nullptr;
    void* layoutB = nullptr;
    void* layoutC = nullptr;
    int rcA = g_lt.layout_create(&layoutA, kCudaR8FE4M3,
                                  (uint64_t)k_cublas, (uint64_t)m_cublas, k_cublas);
    int rcB = g_lt.layout_create(&layoutB, kCudaR8FE4M3,
                                  (uint64_t)k_cublas, (uint64_t)n_cublas, k_cublas);
    int rcC = g_lt.layout_create(&layoutC, kCudaR16F,
                                  (uint64_t)m_cublas, (uint64_t)n_cublas, m_cublas);
    if (rcA || rcB || rcC) {
        if (layoutA) g_lt.layout_destroy(layoutA);
        if (layoutB) g_lt.layout_destroy(layoutB);
        if (layoutC) g_lt.layout_destroy(layoutC);
        g_lt.desc_destroy(desc);
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // Heuristic algo selection — required for FP8 GEMM. Without this,
    // cublasLt's default algo path silently produces wrong results for
    // many M>1 / large-N FP8 configurations.
    void* pref = nullptr;
    if (g_lt.pref_create(&pref) != 0 || !pref) {
        g_lt.layout_destroy(layoutA); g_lt.layout_destroy(layoutB);
        g_lt.layout_destroy(layoutC); g_lt.desc_destroy(desc);
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    {
        size_t ws = LT_WS_BYTES;
        g_lt.pref_set_attr(pref, kPrefAttrMaxWorkspaceBytes, &ws, sizeof(ws));
    }

    // cublasLtMatmulHeuristicResult_t is { algo (8 size_t = 64B), ws (8B),
    // state (4B), wavesCount (4B), reserved (4*4B) } = 96 bytes total in
    // CUDA 12.x.  Allocate generously.
    char heuristic[256];
    int returned = 0;
    int hrc = g_lt.algo_heuristic(g_lt_handle, desc,
                                   layoutA, layoutB, layoutC, layoutC,
                                   pref, 1, heuristic, &returned);
    if (hrc != 0 || returned == 0) {
        if (getenv("CIPHER_FP8_VERBOSE"))
            fprintf(stderr, "[CIPHER FP8] AlgoGetHeuristic rc=%d returned=%d "
                    "for (%d,%d,%d)\n", hrc, returned, m_cublas, n_cublas, k_cublas);
        g_lt.pref_destroy(pref);
        g_lt.layout_destroy(layoutA); g_lt.layout_destroy(layoutB);
        g_lt.layout_destroy(layoutC); g_lt.desc_destroy(desc);
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    // The first field of cublasLtMatmulHeuristicResult_t is algo (the 64-byte
    // cublasLtMatmulAlgo_t struct).  Pass &heuristic directly.
    void* algo_ptr = heuristic;

    float h_alpha = 1.0f;
    float h_beta  = 0.0f;
    int rc = g_lt.matmul(g_lt_handle, desc,
                          &h_alpha,
                          fp8_w,         layoutA,
                          g_act_fp8_ws,  layoutB,
                          &h_beta,
                          /* C = NULL when beta==0 */ nullptr, layoutC,
                          c_fp16,        layoutC,
                          algo_ptr,
                          g_lt_workspace, LT_WS_BYTES,
                          stream);

    g_lt.pref_destroy(pref);
    g_lt.layout_destroy(layoutA);
    g_lt.layout_destroy(layoutB);
    g_lt.layout_destroy(layoutC);
    g_lt.desc_destroy(desc);

    if (rc != 0) {
        if (getenv("CIPHER_FP8_VERBOSE"))
            fprintf(stderr, "[CIPHER FP8] cublasLtMatmul rc=%d for (%d,%d,%d)\n",
                    rc, m_cublas, n_cublas, k_cublas);
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    g_matmul_calls.fetch_add(1, std::memory_order_relaxed);
    (void)w_rows; (void)w_cols;
    return 1;
}

extern "C" int cipher_fp8_compute_stats(CipherFp8ComputeStats* out) {
    if (!out) return 0;
    out->enabled            = g_enabled.load(std::memory_order_relaxed);
    out->weights_quantized  = g_weights_quantized.load(std::memory_order_relaxed);
    out->matmul_calls       = g_matmul_calls.load(std::memory_order_relaxed);
    out->passthroughs       = g_passthroughs.load(std::memory_order_relaxed);
    out->correctness_failures = g_correctness_failures.load(std::memory_order_relaxed);
    out->bytes_fp16_weights = g_bytes_fp16.load(std::memory_order_relaxed);
    out->bytes_fp8_weights  = g_bytes_fp8.load(std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_fp8_compute_report(void) {
    CipherFp8ComputeStats s; cipher_fp8_compute_stats(&s);
    FILE* f = fopen("/tmp/cipher_fp8_compute_report.json", "w");
    if (!f) return;
    fprintf(f, "{\n  \"enabled\": %d,\n"
               "  \"weights_quantized\": %llu,\n"
               "  \"matmul_calls\": %llu,\n"
               "  \"passthroughs\": %llu,\n"
               "  \"correctness_failures\": %llu,\n"
               "  \"bytes_fp16\": %zu,\n"
               "  \"bytes_fp8\": %zu\n}\n",
            s.enabled,
            (unsigned long long)s.weights_quantized,
            (unsigned long long)s.matmul_calls,
            (unsigned long long)s.passthroughs,
            (unsigned long long)s.correctness_failures,
            s.bytes_fp16_weights,
            s.bytes_fp8_weights);
    fclose(f);
}

// Self-init at priority 113 (after thermal_feedback at 111 and kv_redirect
// modules).  Idempotent, no-op when env flag is off.
__attribute__((constructor(113)))
static void cipher_fp8_compute_autoinit(void) {
    cipher_fp8_compute_init();
}
