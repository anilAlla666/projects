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
#include "cipher_op_counters.h"

#include <cuda_runtime.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

// Hot-path cached env check — getenv is a syscall and was previously
// being called once per matmul / quantize / passthrough event. With
// 270k+ GEMMs in a typical decode, that costs hundreds of ms of pure
// libc overhead on stderr-disabled paths. Cache once.
inline bool cipher_fp8_verbose() {
    static const bool s_on = []() {
        const char* e = std::getenv("CIPHER_FP8_VERBOSE");
        return e && *e && e[0] != '0';
    }();
    return s_on;
}

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
// CUBLASLT_MATMUL_DESC_A_SCALE_MODE = 31 (cu13).
// Value 3 = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F: scale is an
// M-element fp32 vector applied per-row of A. Used for FP8 per-row
// quantization on small models (Task 2) where per-tensor absmax
// over-compresses smaller-magnitude rows and breaks output coherence.
constexpr int kAttrAScaleMode   = 31;
constexpr int kAttrBScaleMode   = 32;
constexpr int kScaleModeOuterVec32F = 3;

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
    void*    fp16_key;     // canonical pointer used by cublasGemmEx
    void*    fp8_buf;      // device fp8 e4m3 buffer, rows*cols bytes
    void*    scale_dev;    // device fp32 scalar = absmax/448
    int      rows;         // (= m_cublas)
    int      cols;         // (= k_cublas)
    int      hits;         // observation count for stability detection
    uint64_t content_hash; // FNV-1a hash of first 128 bytes (only for hits 1-2).
    bool     transient;    // true once we've detected content drift at this
                           // address (e.g. bnb's recycled dequant scratch).
                           // Future observes return 0 immediately — no hash,
                           // no FP8 — so the hot path stays sync-free.
    bool     per_row_scale; // true if scale_dev is an M-element vector
                            // (CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F).
                            // Set when CIPHER_FP8_PER_ROW=on at quant time.
};

// FNV-1a 64-bit hash.
uint64_t fnv1a_64(const uint8_t* data, size_t n) {
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME  = 0x00000100000001B3ULL;
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)data[i];
        h *= FNV_PRIME;
    }
    return h;
}

// Read 128 bytes from device pointer, hash them.  ~1us D2H per call.
constexpr size_t HASH_PROBE_BYTES = 128;
uint64_t probe_content_hash(const void* dev_ptr) {
    if (!dev_ptr) return 0;
    uint8_t host[HASH_PROBE_BYTES];
    cudaError_t err = cudaMemcpy(host, dev_ptr, HASH_PROBE_BYTES,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return 0;
    return fnv1a_64(host, HASH_PROBE_BYTES);
}
WeightEntry g_weights[MAX_WEIGHTS]{};
std::mutex  g_weights_mu;

int find_weight_locked(void* key) {
    for (int i = 0; i < MAX_WEIGHTS; ++i) {
        if (g_weights[i].fp16_key == key) return i;
    }
    return -1;
}

// Content-hash secondary lookup. Same logical weight that landed at a
// different pointer due to allocator reuse / model relocation still
// matches by content. Filters by shape so distinct weights with rare
// hash collisions don't trip into each other.
int find_weight_by_hash_locked(uint64_t hash, int rows, int cols) {
    if (hash == 0) return -1;
    for (int i = 0; i < MAX_WEIGHTS; ++i) {
        if (g_weights[i].fp16_key != nullptr
            && g_weights[i].content_hash == hash
            && g_weights[i].rows == rows
            && g_weights[i].cols == cols) {
            return i;
        }
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

// Per-shape cache. Llama-class models hit ~6 distinct (m,n,k) at decode
// (q/k/v/o + gate/up/down) so 64 entries is comfortable headroom.
// Reused across all (weight, activation) pairs — only the AScalePtr varies
// per weight and is set per-call (cheap, ~100 ns).
struct ShapeCache {
    void*  desc      = nullptr;     // cublasLtMatmulDesc_t
    void*  layoutA   = nullptr;     // FP8 K×M
    void*  layoutB   = nullptr;     // FP8 K×N
    void*  layoutC   = nullptr;     // FP16 M×N
    void*  pref      = nullptr;     // cublasLtMatmulPreference_t
    char   algo[256] = {};          // cublasLtMatmulHeuristicResult_t
    bool   ready     = false;
    // Per-shape activation scale with periodic recalibration. Every
    // RECAL_PERIOD calls we re-run the full absmax+finalize+quant
    // pipeline (3 kernels) to track activation drift across layers and
    // tokens. Other calls run just the quant kernel against the cached
    // scale (1 kernel). Net amortized cost: 1 + 2/N kernels per call
    // (~1.07 at N=32) vs the original 3.
    void*    act_scale_dev    = nullptr;   // device fp32 scalar (absmax/448)
    void*    absmax_dev       = nullptr;   // device int32 scratch for fused kernel
};
std::unordered_map<uint64_t, ShapeCache> g_shape_cache;
std::mutex g_shape_cache_mu;

static inline uint64_t pack_mnk(int m, int n, int k) {
    return ((uint64_t)(uint32_t)m << 40)
         ^ ((uint64_t)(uint32_t)n << 20)
         ^ (uint64_t)(uint32_t)k;
}

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

// ── Per-row variants for Task 2: small-model FP8 coherence ────────────
// One row per block. Block reduces row[i] absmax across cols,
// writes scale[i] = absmax/448 (or 1.0 if row is all zeros), then
// each thread quantizes row[i, c] = round_to_e4m3(in[i,c] / scale[i]).
// Final scale_dev is consumed by cublasLt with
// A_SCALE_MODE = OUTER_VEC_32F (cu13 native per-row scaling).

extern "C" __global__ void cipher_fp8_absmax_per_row(
    const __half* __restrict__ in,
    float*       __restrict__ scale_out,   // M floats, cleared by caller
    int rows, int cols)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = __half2float(in[(size_t)r * cols + c]);
        float a = v < 0.0f ? -v : v;
        if (a > local) local = a;
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
        float a = sdata[0];
        scale_out[r] = (a > 0.0f) ? (a / 448.0f) : 1.0f;
    }
}

extern "C" __global__ void cipher_fp8_quantize_per_row(
    const __half* __restrict__ in,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ scale_dev,   // M-element vector
    int rows, int cols)
{
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    float s = scale_dev[r];
    float inv = 1.0f / (s > 0.0f ? s : 1.0f);
    float v = __half2float(in[(size_t)r * cols + c]) * inv;
    if (v >  448.0f) v =  448.0f;
    if (v < -448.0f) v = -448.0f;
    out[(size_t)r * cols + c] = __nv_fp8_e4m3(v);
}
)NVRTC";

unsigned long g_cubin_absmax        = 0;
unsigned long g_cubin_finalize       = 0;
unsigned long g_cubin_quant          = 0;
unsigned long g_cubin_absmax_per_row = 0;
unsigned long g_cubin_quant_per_row  = 0;
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

bool ensure_per_row_kernels_compiled() {
    std::lock_guard<std::mutex> lk(g_compile_mu);
    if (g_cubin_absmax_per_row && g_cubin_quant_per_row) return true;
    if (!cipher_substitute_v2_enabled()) return false;
    if (!g_cubin_absmax_per_row)
        g_cubin_absmax_per_row = cipher_substitute_v2_compile(
            kFp8KernelSrc, "cipher_fp8_absmax_per_row");
    if (!g_cubin_quant_per_row)
        g_cubin_quant_per_row = cipher_substitute_v2_compile(
            kFp8KernelSrc, "cipher_fp8_quantize_per_row");
    return g_cubin_absmax_per_row && g_cubin_quant_per_row;
}

// Cached env-gate for per-row quantization. Default OFF.
inline bool cipher_fp8_per_row_enabled() {
    static const bool s_on = []() {
        const char* e = std::getenv("CIPHER_FP8_PER_ROW");
        return e && *e && e[0] != '0';
    }();
    return s_on;
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

// Single-launch fused absmax+finalize+quant. Implementation in
// src/cipher_fp8_fused_quant.cu; uses cooperative_groups grid sync to
// resolve the per-tensor absmax inside the same kernel that quantizes.
// Pointer types kept opaque here to avoid pulling cuda_fp8.h into the .cpp.
extern "C" int cipher_fused_fp16_to_fp8(
    const void* input, void* output, float* scale_out,
    int numel, int* absmax_workspace, cudaStream_t stream);

// Quantize-only path: scale_dev MUST already hold (absmax/448) from a prior
// calibration pass. One kernel launch, no absmax / no finalize.
bool quantize_to_fp8_only(const void* in_fp16, void* out_fp8,
                          const void* scale_dev, int n_elem, void* stream) {
    if (!ensure_kernels_compiled()) return false;
    if (!resolve_launcher())        return false;
    void* fn_quant = cipher_substitute_v2_get_function(g_cubin_quant);
    if (!fn_quant) return false;
    int bx = 256;
    int gx = (n_elem + bx - 1) / bx;
    void* args[] = { (void*)&in_fp16, &out_fp8, (void*)&scale_dev, &n_elem };
    return g_cu_launch(fn_quant, gx, 1, 1, bx, 1, 1, 0, stream, args, nullptr) == 0;
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

// Per-row variant. scale_dev is an M-element fp32 vector (NOT 1 scalar).
bool quantize_to_fp8_per_row(const void* in_fp16, void* out_fp8,
                              void* scale_dev, int rows, int cols,
                              void* stream) {
    if (!ensure_per_row_kernels_compiled()) return false;
    if (!resolve_launcher())                return false;
    void* fn_absmax = cipher_substitute_v2_get_function(g_cubin_absmax_per_row);
    void* fn_quant  = cipher_substitute_v2_get_function(g_cubin_quant_per_row);
    if (!fn_absmax || !fn_quant) return false;

    // No memset needed — per-row absmax kernel writes scale[r] directly.
    {
        // One block per row; 256 threads reduce across cols.
        void* args[] = { (void*)&in_fp16, &scale_dev, &rows, &cols };
        if (g_cu_launch(fn_absmax, rows, 1, 1, 256, 1, 1, 0, stream, args, nullptr) != 0)
            return false;
    }
    {
        // 2D grid: (cols/bx, rows). Each thread writes one element.
        int bx = 256;
        int gx = (cols + bx - 1) / bx;
        void* args[] = { (void*)&in_fp16, &out_fp8, &scale_dev, &rows, &cols };
        if (g_cu_launch(fn_quant, gx, rows, 1, bx, 1, 1, 0, stream, args, nullptr) != 0)
            return false;
    }
    return true;
}

}  // namespace

// ── Public API ──────────────────────────────────────────────────────────

// OP 21 — auto-disable on torch 2.7+ (which has its own native FP8 path
// that out-performs CIPHER's substitute on H100). Probe by trying to
// dlsym torch's `at::_scaled_mm` symbol from libtorch_cpu.so or
// libtorch_python.so; presence implies torch has shipped its own FP8
// matmul.  Override via CIPHER_FP8_FORCE=on.
static bool torch_native_fp8_present(void) {
    const char* libs[] = {
        "libtorch_cpu.so", "libtorch_python.so", "libtorch.so", nullptr
    };
    for (const char** p = libs; *p; ++p) {
        void* h = dlopen(*p, RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen(*p, RTLD_NOW);
        if (!h) continue;
        // Mangled name for at::_scaled_mm in torch 2.6+.
        // We don't know the full mangling so probe a few candidates.
        // Discriminate torch ≥ 2.7 from earlier versions.  Both torch 2.6
        // and 2.7 ship `f8f8bf16_rowwise` and `aoti_torch__scaled_mm`, but
        // only torch 2.7 added `f8f8bf16_grouped_mm` — that's our marker.
        // CIPHER FP8 measurably loses to torch 2.7's native path on H100
        // (per stress2/SUBSTITUTION_ANALYSIS.md) and measurably wins on
        // torch ≤ 2.6.
        const char* candidates[] = {
            "_ZN2at4cuda6detail19f8f8bf16_grouped_mmENS_6TensorES2_S2_S2_St8optionalIS2_ES4_bRS2_",
            nullptr,
        };
        for (const char** c = candidates; *c; ++c) {
            if (dlsym(h, *c)) return true;
        }
    }
    return false;
}

extern "C" int cipher_fp8_compute_init(void) {
    if (g_enabled.load(std::memory_order_relaxed)) return 1;
    const char* env = getenv("CIPHER_FP8_COMPUTE");
    int on = env && (env[0] == '1' || env[0] == 'o' || env[0] == 'O' ||
                     !strcmp(env, "on") || !strcmp(env, "ON"));
    if (!on) return 0;

    // Auto-disable when torch ships native FP8 (torch 2.7+ on Hopper),
    // unless the operator force-enables.
    const char* force = getenv("CIPHER_FP8_FORCE");
    bool forced = force && (force[0] == '1' || force[0] == 'o' ||
                              force[0] == 'O');
    if (!forced && torch_native_fp8_present()) {
        fprintf(stderr,
            "[CIPHER FP8] torch-native FP8 detected — auto-disabled "
            "(set CIPHER_FP8_FORCE=on to override)\n");
        return 0;
    }

    g_enabled.store(1, std::memory_order_relaxed);
    if (cipher_fp8_verbose())
        fprintf(stderr, "[CIPHER FP8] enabled\n");
    return 1;
}

extern "C" int cipher_fp8_compute_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_fp8_compute_observe(void* weight_fp16, int rows, int cols) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!weight_fp16) return 0;

    // Fast path: take the lock first, look up the slot.  Only the FIRST
    // and SECOND observations of a given pointer trigger a D2H hash; once
    // a slot is past hits=2 (cached) or has been marked transient (bnb
    // recycle detected), we skip hashing entirely.  The hot path —
    // thousands of GEMMs/sec — must stay synchronous-free.
    //
    // FIX 1 — content-hash secondary lookup: if the pointer doesn't
    // match any slot but the same content hash already does (allocator
    // reuse / model relocation), bump that slot's hits and rebind its
    // pointer key. Without this, every PyTorch decode step that gets a
    // fresh pointer for the same logical weight starts fresh at hits=1
    // and FP8 stability never trips.
    int idx;
    {
        std::lock_guard<std::mutex> lk(g_weights_mu);
        idx = find_weight_locked(weight_fp16);
    }
    if (idx < 0) {
        // Pointer miss → check content-hash table before allocating.
        uint64_t h = probe_content_hash(weight_fp16);
        if (h != 0) {
            std::lock_guard<std::mutex> lk(g_weights_mu);
            int hidx = find_weight_by_hash_locked(h, rows, cols);
            if (hidx >= 0 && !g_weights[hidx].transient) {
                // Same content seen at a different pointer earlier. Rebind
                // and bump. If the slot already had hits>=2, the FP8 cache
                // (fp8_buf + scale_dev) is still valid, so we keep it.
                g_weights[hidx].fp16_key = weight_fp16;
                if (g_weights[hidx].hits < 2) g_weights[hidx].hits = 2;
                else                          g_weights[hidx].hits++;
                return g_weights[hidx].hits;
            }
        }
    }

    if (idx >= 0) {
        // Existing entry.
        std::lock_guard<std::mutex> lk(g_weights_mu);
        if (g_weights[idx].transient) {
            // Recycled buffer detected previously — don't FP8, don't hash.
            return 0;
        }
        bool shape_changed = (g_weights[idx].rows != rows ||
                              g_weights[idx].cols != cols);
        if (shape_changed) {
            // Different tensor at this address (allocator reuse).  Reset
            // and re-observe from scratch.  Hash on the next call.
            if (g_weights[idx].fp8_buf)   cudaFree(g_weights[idx].fp8_buf);
            if (g_weights[idx].scale_dev) cudaFree(g_weights[idx].scale_dev);
            g_weights[idx] = WeightEntry{};
            g_weights[idx].fp16_key = weight_fp16;
            g_weights[idx].rows     = rows;
            g_weights[idx].cols     = cols;
            g_weights[idx].hits     = 0;
            // Drop the lock before D2H, then re-acquire to commit.
        }
        if (g_weights[idx].hits >= 2) {
            // Already content-verified.  Just bump and return — no D2H.
            return ++g_weights[idx].hits;
        }
        // hits == 0 (just reset) or hits == 1 (need second-obs verify).
        // Drop the lock before the D2H.
    } else {
        // No slot yet — allocate one synchronously, hash later.
        std::lock_guard<std::mutex> lk(g_weights_mu);
        idx = alloc_weight_slot_locked();
        if (idx < 0) return 0;
        g_weights[idx].fp16_key = weight_fp16;
        g_weights[idx].rows     = rows;
        g_weights[idx].cols     = cols;
        g_weights[idx].hits     = 0;
    }

    // D2H hash — outside the lock.  This costs ~1us synchronously but is
    // taken at most TWICE per (ptr, shape) pair (first observation to
    // record, second to verify).  After hits >= 2 we never come here.
    uint64_t hash_now = probe_content_hash(weight_fp16);

    {
        std::lock_guard<std::mutex> lk(g_weights_mu);
        // Re-validate the slot (could have been recycled by another thread).
        if (g_weights[idx].fp16_key != weight_fp16) return 0;
        if (g_weights[idx].rows != rows || g_weights[idx].cols != cols)
            return 0;

        if (g_weights[idx].hits == 0) {
            // First observation — store hash.
            g_weights[idx].content_hash = hash_now;
            g_weights[idx].hits         = 1;
            return 1;
        }
        // hits == 1: second observation — verify.
        if (g_weights[idx].content_hash != 0 &&
            hash_now != 0 &&
            g_weights[idx].content_hash != hash_now) {
            // Content drifted at the same address → recycled buffer.
            // Mark transient so all future observes here are O(1) no-ops.
            if (g_weights[idx].fp8_buf)   cudaFree(g_weights[idx].fp8_buf);
            if (g_weights[idx].scale_dev) cudaFree(g_weights[idx].scale_dev);
            g_weights[idx].fp8_buf      = nullptr;
            g_weights[idx].scale_dev    = nullptr;
            g_weights[idx].content_hash = 0;
            g_weights[idx].hits         = 0;
            g_weights[idx].transient    = true;
            return 0;
        }
        g_weights[idx].hits = 2;
        return 2;
    }
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
    // Per-row mode allocates M floats; per-tensor mode allocates 1 float.
    bool per_row = cipher_fp8_per_row_enabled();
    size_t scale_bytes = per_row ? (size_t)rows * sizeof(float) : sizeof(float);
    if (cudaMalloc(&scale_dev, scale_bytes) != cudaSuccess) {
        cudaFree(fp8_buf);
        std::lock_guard<std::mutex> lk(g_weights_mu);
        g_weights[idx] = WeightEntry{};
        return 0;
    }

    bool qok = per_row
        ? quantize_to_fp8_per_row(weight_fp16, fp8_buf, scale_dev,
                                    rows, cols, stream)
        : quantize_to_fp8(weight_fp16, fp8_buf, scale_dev,
                          (int)n_elem, stream);
    if (!qok) {
        cudaFree(fp8_buf);
        cudaFree(scale_dev);
        std::lock_guard<std::mutex> lk(g_weights_mu);
        g_weights[idx] = WeightEntry{};
        return 0;
    }

    {
        std::lock_guard<std::mutex> lk(g_weights_mu);
        g_weights[idx].fp8_buf       = fp8_buf;
        g_weights[idx].scale_dev     = scale_dev;
        g_weights[idx].per_row_scale = per_row;
    }
    g_weights_quantized.fetch_add(1, std::memory_order_relaxed);
    g_bytes_fp16.fetch_add(n_elem * 2, std::memory_order_relaxed);
    g_bytes_fp8.fetch_add(n_elem,     std::memory_order_relaxed);
    if (cipher_fp8_verbose())
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

    // Bug-3 fix: small-GEMM bypass. Tiny matmuls (small encoders, low-rank
    // adapters) do not amortize the FP8 quant kernel + descriptor lookup
    // cost. Threshold tunable via CIPHER_FP8_MIN_ELEMENTS, default 4 M
    // = a 1024×1024×4 GEMM. Sentence-transformers' MiniLM-L6 has 384²
    // attention projections (~150 K elements) and lands well below this.
    static size_t s_min_elements = []() {
        const char* env = getenv("CIPHER_FP8_MIN_ELEMENTS");
        if (!env || !*env) return (size_t)4'000'000;
        char* end = nullptr;
        unsigned long long v = strtoull(env, &end, 10);
        return (v > 0) ? (size_t)v : (size_t)4'000'000;
    }();
    size_t total_elems = (size_t)m_cublas * (size_t)n_cublas * (size_t)k_cublas;
    if (total_elems < s_min_elements) {
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // Skip the entire FP8 substitute path while a CUDA graph is being
    // captured on this stream. cublasLt FP8 internally does lazy
    // allocations that aren't graph-capture-safe; let baseline cuBLAS
    // run instead, which IS capture-compatible. Replays of the captured
    // graph will run as fp16. CIPHER's pre-capture warmup carries the
    // weights' fp8 buffers; once we fix the capture-safety of our path
    // we can re-enable FP8 substitution inside captured graphs.
    cudaStreamCaptureStatus cap;
    if (cudaStreamIsCapturing((cudaStream_t)stream, &cap) == cudaSuccess
        && cap != cudaStreamCaptureStatusNone) {
        return 0;
    }

    void* fp8_w = nullptr;
    void* w_scale = nullptr;
    int   w_rows = 0, w_cols = 0;
    int   w_idx = -1;
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
        w_idx   = i;
    }
    bool per_row_w = false;
    {
        std::lock_guard<std::mutex> lk(g_weights_mu);
        per_row_w = g_weights[w_idx].per_row_scale;
    }

    // Per-call content-hash verification. ~1 µs per matmul because
    // probe_content_hash does a synchronous cudaMemcpy DToH that blocks
    // the launch stream. Production-mode warmup gate: only verify
    // during the first N matmuls, then trust the pointer-keyed cache
    // for the rest of the process. Override via CIPHER_FP8_HASH_FOREVER=1
    // for processes that swap models mid-flight (the original use case).
    static std::atomic<uint64_t> s_matmul_count{0};
    static const uint64_t s_hash_until = []() {
        const char* e = std::getenv("CIPHER_FP8_HASH_UNTIL");
        if (e && *e) return (uint64_t)atoll(e);
        return (uint64_t)1000;
    }();
    static const bool s_hash_forever = []() {
        const char* e = std::getenv("CIPHER_FP8_HASH_FOREVER");
        return e && *e && e[0] != '0';
    }();
    uint64_t mc = s_matmul_count.fetch_add(1, std::memory_order_relaxed) + 1;
    bool do_hash = s_hash_forever || (mc <= s_hash_until);
    uint64_t now_hash = do_hash ? probe_content_hash(weight_fp16_key) : 0;
    bool stale = false;
    {
        std::lock_guard<std::mutex> lk(g_weights_mu);
        if (g_weights[w_idx].fp16_key == weight_fp16_key &&
            g_weights[w_idx].content_hash != 0 &&
            now_hash != 0 &&
            g_weights[w_idx].content_hash != now_hash) {
            stale = true;
            // Free the stale fp8 buffer + scale; clear the entry so
            // subsequent observe()s re-establish content stability and
            // the next matmul re-quantizes through the slow path. This
            // is the conservative re-use-prevention path; the call
            // returns passthrough so cuBLAS fp16 still runs and the
            // model produces correct output this iteration.
            if (g_weights[w_idx].fp8_buf)   cudaFree(g_weights[w_idx].fp8_buf);
            if (g_weights[w_idx].scale_dev) cudaFree(g_weights[w_idx].scale_dev);
            g_weights[w_idx].fp8_buf      = nullptr;
            g_weights[w_idx].scale_dev    = nullptr;
            g_weights[w_idx].content_hash = 0;
            g_weights[w_idx].hits         = 0;
            // Do NOT mark transient — this is a legitimate model swap,
            // not the bnb-style intra-step buffer reuse that transient
            // is designed for. Fresh observe() will re-detect stability.
        }
    }
    if (stale) {
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        if (cipher_fp8_verbose())
            fprintf(stderr,
                    "[CIPHER FP8] stale weight detected at %p (m=%d k=%d) — "
                    "invalidated, re-quantize on next observe\n",
                    weight_fp16_key, m_cublas, k_cublas);
        return 0;
    }

    // Capacity guard for the activation workspace.
    size_t act_n = (size_t)n_cublas * (size_t)k_cublas;
    if (act_n > MAX_M * MAX_K) {
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // Activation quantization fp16 -> fp8.
    // We defer the cache lookup briefly: build the cache below, then
    // quantize using the cache's per-shape scale buffer. On the FIRST
    // call for a shape, run the full absmax+finalize+quant pipeline (3
    // kernels). On every subsequent call, run only the quant kernel,
    // reusing the cached scale. Activation magnitudes are stable across
    // tokens during inference.
    // Quantization itself is moved below the cache build.

    // Per-shape cache lookup. Build descriptor + layouts + pref + algo
    // ONCE per (m, n, k) and reuse them on every subsequent call.
    // Layouts and algo are pointer-independent. AScalePtr (per-weight) is
    // set per-call below; BScalePtr (global g_act_scale_dev) is baked in
    // on first build.
    uint64_t shape_key = pack_mnk(m_cublas, n_cublas, k_cublas);
    ShapeCache* sc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_shape_cache_mu);
        sc = &g_shape_cache[shape_key];
    }
    if (!sc->ready) {
        std::lock_guard<std::mutex> lk(g_shape_cache_mu);
        if (!sc->ready) {  // double-check inside lock
            void* desc = nullptr;
            if (g_lt.desc_create(&desc, kCublasCompute32F, kCudaR32F) != 0) {
                g_passthroughs.fetch_add(1, std::memory_order_relaxed);
                return 0;
            }
            int transA = kCublasOpT;
            int transB = kCublasOpN;
            g_lt.desc_set_attr(desc, kAttrTransA, &transA, sizeof(transA));
            g_lt.desc_set_attr(desc, kAttrTransB, &transB, sizeof(transB));
            // FP8 × FP8 path: per-shape activation scale (fp32 scalar) AND
            // absmax scratch (int32) for the fused-kernel quantize. cublasLt
            // BScalePtr is baked into the desc so the dequant uses the
            // freshest value each call.
            void* sc_act_scale = nullptr;
            void* sc_absmax    = nullptr;
            if (cudaMalloc(&sc_act_scale, sizeof(float)) != cudaSuccess) {
                g_lt.desc_destroy(desc);
                g_passthroughs.fetch_add(1, std::memory_order_relaxed);
                return 0;
            }
            if (cudaMalloc(&sc_absmax, sizeof(int)) != cudaSuccess) {
                cudaFree(sc_act_scale);
                g_lt.desc_destroy(desc);
                g_passthroughs.fetch_add(1, std::memory_order_relaxed);
                return 0;
            }
            g_lt.desc_set_attr(desc, kAttrBScalePtr, &sc_act_scale, sizeof(void*));

            // FIX (Task 2): set A_SCALE_MODE before heuristic — algo
            // selection must know about vector scaling up front. Otherwise
            // a scalar-mode algo gets cached and the later mode change
            // is silently ignored, producing garbled output. mode=3 is
            // CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F (cu13+).
            if (per_row_w) {
                int mode = kScaleModeOuterVec32F;
                g_lt.desc_set_attr(desc, kAttrAScaleMode, &mode, sizeof(int));
            }

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
            void* pref = nullptr;
            if (g_lt.pref_create(&pref) != 0 || !pref) {
                g_lt.layout_destroy(layoutA); g_lt.layout_destroy(layoutB);
                g_lt.layout_destroy(layoutC); g_lt.desc_destroy(desc);
                g_passthroughs.fetch_add(1, std::memory_order_relaxed);
                return 0;
            }
            size_t ws = LT_WS_BYTES;
            g_lt.pref_set_attr(pref, kPrefAttrMaxWorkspaceBytes, &ws, sizeof(ws));
            int returned = 0;
            int hrc = g_lt.algo_heuristic(g_lt_handle, desc,
                            layoutA, layoutB, layoutC, layoutC,
                            pref, 1, sc->algo, &returned);
            // Always log the first few heuristic failures — this is the
            // first observable signal whether cuBLAS Lt accepts FP8×FP16
            // mixed-precision on this device.
            static std::atomic<int> s_fail_log{0};
            if (hrc != 0 || returned == 0) {
                int fl = s_fail_log.fetch_add(1, std::memory_order_relaxed);
                if (fl < 8 || cipher_fp8_verbose())
                    fprintf(stderr, "[CIPHER FP8-MIX] AlgoGetHeuristic rc=%d "
                            "returned=%d for (m=%d, n=%d, k=%d) — FP8×FP16 "
                            "mixed not supported by this cuBLAS Lt\n",
                            hrc, returned, m_cublas, n_cublas, k_cublas);
                g_lt.pref_destroy(pref);
                g_lt.layout_destroy(layoutA); g_lt.layout_destroy(layoutB);
                g_lt.layout_destroy(layoutC); g_lt.desc_destroy(desc);
                g_passthroughs.fetch_add(1, std::memory_order_relaxed);
                return 0;
            }
            sc->desc           = desc;
            sc->layoutA        = layoutA;
            sc->layoutB        = layoutB;
            sc->layoutC        = layoutC;
            sc->pref           = pref;
            sc->act_scale_dev  = sc_act_scale;
            sc->absmax_dev     = sc_absmax;
            sc->ready          = true;
            if (cipher_fp8_verbose())
                fprintf(stderr,
                    "[CIPHER FP8] cache built for (%d,%d,%d) "
                    "(total cache entries=%zu)\n",
                    m_cublas, n_cublas, k_cublas, g_shape_cache.size());
        }
    }

    // Per-call AScalePtr update — different weight, same shape.
    g_lt.desc_set_attr(sc->desc, kAttrAScalePtr, &w_scale, sizeof(void*));
    // Mode (scalar vs OUTER_VEC_32F) was baked into desc at cache-build
    // time; setting it again per-call would be a no-op for the algo.

    // Activation quantization. cudaLaunchCooperativeKernel cannot be captured
    // by CUDA graphs (NVIDIA documented limitation), so use the 3-kernel path
    // when capturing, fused single-launch kernel otherwise.
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing((cudaStream_t)stream, &capture_status);
    bool in_capture = (capture_status != cudaStreamCaptureStatusNone);
    bool qok;
    if (in_capture) {
        // 3-kernel path: cudaMemsetAsync + absmax + finalize + quant.
        // All standard kernels — fully graph-capturable. Per-call launch
        // overhead is paid ONCE during capture and amortized across replays.
        qok = quantize_to_fp8(activation_fp16, g_act_fp8_ws,
                               sc->act_scale_dev, (int)act_n, stream);
    } else {
        // Eager mode: use the single-launch cooperative fused kernel.
        qok = cipher_fused_fp16_to_fp8(
                activation_fp16, g_act_fp8_ws,
                (float*)sc->act_scale_dev,
                (int)act_n,
                (int*)sc->absmax_dev,
                (cudaStream_t)stream);
    }
    if (!qok) {
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    float h_alpha = 1.0f;
    float h_beta  = 0.0f;
    int rc = g_lt.matmul(g_lt_handle, sc->desc,
                          &h_alpha,
                          fp8_w,         sc->layoutA,
                          g_act_fp8_ws,  sc->layoutB,
                          &h_beta,
                          /* C = NULL when beta==0 */ nullptr, sc->layoutC,
                          c_fp16,        sc->layoutC,
                          sc->algo,
                          g_lt_workspace, LT_WS_BYTES,
                          stream);

    if (rc != 0) {
        if (cipher_fp8_verbose())
            fprintf(stderr, "[CIPHER FP8] cublasLtMatmul rc=%d for (%d,%d,%d)\n",
                    rc, m_cublas, n_cublas, k_cublas);
        g_passthroughs.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    g_matmul_calls.fetch_add(1, std::memory_order_relaxed);
    cipher_op_inc(OP_SUBSTITUTE_FP8);
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
