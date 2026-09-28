// CIPHER fused-kernel substitutes — implementation.

#include "cipher_fusion_kernels.h"
#include "cipher_substitute_v2.h"

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

namespace {

// ── Pattern 1: Fused RMSNorm ───────────────────────────────────────────────
// Two-pass per-row reduction: variance = mean(x^2), then write x*rsqrt(var+eps)*w.
// blockDim.x = 256 threads per row; rows = grid.x.
const char kRmsNormSrc[] = R"NVRTC(
#include <cuda_fp16.h>

extern "C" __global__ void cipher_rmsnorm_fp16(
    const __half* __restrict__ x,        // (rows, hidden_dim)
    const __half* __restrict__ weight,   // (hidden_dim,)
    __half*       __restrict__ out,      // (rows, hidden_dim)
    int rows, int hidden_dim, float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    // Pass 1: per-row sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = __half2float(x[row * hidden_dim + i]);
        sum_sq += v * v;
    }
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);

    // Cross-warp reduce in shared memory (up to 32 warps = 1024 threads)
    __shared__ float warp_sum[32];
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) warp_sum[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (blockDim.x + 31) / 32;
        float v = (lane < n_warps) ? warp_sum[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) warp_sum[0] = v;
    }
    __syncthreads();

    float total    = warp_sum[0];
    float variance = total / (float)hidden_dim;
    float rsq      = rsqrtf(variance + eps);

    // Pass 2: write normalized * weight
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = __half2float(x[row * hidden_dim + i]);
        float w = __half2float(weight[i]);
        out[row * hidden_dim + i] = __float2half_rn(v * rsq * w);
    }
}
)NVRTC";

// ── Pattern 2: Fused SiLU(gate) * up ──────────────────────────────────────
const char kSiLUMulSrc[] = R"NVRTC(
#include <cuda_fp16.h>

extern "C" __global__ void cipher_silu_mul_fp16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half*       __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu_g = g / (1.0f + __expf(-g));     // x * sigmoid(x)
    out[idx] = __float2half_rn(silu_g * u);
}
)NVRTC";

// ── Pattern 3: Element-wise residual add ─────────────────────────────────
const char kResAddSrc[] = R"NVRTC(
#include <cuda_fp16.h>

extern "C" __global__ void cipher_residual_add_fp16(
    const __half* __restrict__ x,
    const __half* __restrict__ residual,
    __half*       __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float a = __half2float(x[idx]);
    float b = __half2float(residual[idx]);
    out[idx] = __float2half_rn(a + b);
}
)NVRTC";

// Per-device CUmodule/CUfunction map.  cipher_substitute_v2_compile loads
// the cubin into whatever CUDA primary context is current at compile time.
// On a multi-GPU system the caller (e.g. via torch.cuda.device(d)) sets
// the current device before invoking us; we compile lazily per device so
// each GPU has its own valid CUfunction handle.
constexpr int MAX_DEVICES = 16;

struct DeviceFns {
    unsigned long      rmsnorm_id  = 0;
    unsigned long      silu_mul_id = 0;
    unsigned long      res_add_id  = 0;
    void*              rmsnorm_fn  = nullptr;
    void*              silu_mul_fn = nullptr;
    void*              res_add_fn  = nullptr;
    std::atomic<bool>  compiled{false};
};

DeviceFns     g_per_device[MAX_DEVICES];
std::mutex    g_compile_mu;

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned long long> g_rmsnorm_calls{0};
std::atomic<unsigned long long> g_silu_mul_calls{0};
std::atomic<unsigned long long> g_res_add_calls{0};

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

int current_device_or_zero() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess) return 0;
    if (d < 0 || d >= MAX_DEVICES) return 0;
    return d;
}

bool ensure_compiled_for(int dev) {
    if (dev < 0 || dev >= MAX_DEVICES) return false;
    DeviceFns& d = g_per_device[dev];
    if (d.compiled.load(std::memory_order_acquire)) return true;
    std::lock_guard<std::mutex> lk(g_compile_mu);
    if (d.compiled.load(std::memory_order_relaxed)) return true;
    if (!cipher_substitute_v2_enabled()) return false;

    // The caller is responsible for making `dev` the current CUDA device
    // (torch.cuda.device(d) on the Python side).  cipher_substitute_v2_compile
    // calls cudaFree(0) which forces the primary context up — for the
    // *current* device — so the resulting CUmodule/CUfunction lives in
    // dev's context, exactly what we need.
    if (d.rmsnorm_id == 0)
        d.rmsnorm_id = cipher_substitute_v2_compile(kRmsNormSrc, "cipher_rmsnorm_fp16");
    if (d.silu_mul_id == 0)
        d.silu_mul_id = cipher_substitute_v2_compile(kSiLUMulSrc, "cipher_silu_mul_fp16");
    if (d.res_add_id == 0)
        d.res_add_id = cipher_substitute_v2_compile(kResAddSrc, "cipher_residual_add_fp16");

    if (d.rmsnorm_id)  d.rmsnorm_fn  = cipher_substitute_v2_get_function(d.rmsnorm_id);
    if (d.silu_mul_id) d.silu_mul_fn = cipher_substitute_v2_get_function(d.silu_mul_id);
    if (d.res_add_id)  d.res_add_fn  = cipher_substitute_v2_get_function(d.res_add_id);

    bool ok = d.rmsnorm_fn && d.silu_mul_fn && d.res_add_fn;
    if (ok) {
        d.compiled.store(true, std::memory_order_release);
        fprintf(stderr,
            "[CIPHER FUSION] cubins compiled for cuda:%d "
            "(rmsnorm=%lu silu_mul=%lu res_add=%lu)\n",
            dev, d.rmsnorm_id, d.silu_mul_id, d.res_add_id);
    }
    return ok;
}

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T'
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

} // namespace

extern "C" int cipher_fusion_kernels_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_FUSION"));
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        fprintf(stderr,
            "[CIPHER FUSION] init — Pattern 1 (RMSNorm), Pattern 2 (SiLU·Mul), "
            "Pattern 3 (Residual+) NVRTC-compiled lazily.\n");
    }
    return on;
}

extern "C" int cipher_fusion_kernels_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_fused_rmsnorm(void* x, void* weight, void* out,
                                     int rows, int hidden_dim, float eps,
                                     void* stream) {
    if (!x || !weight || !out || rows <= 0 || hidden_dim <= 0) return 0;
    if ((hidden_dim & 31) != 0) return 0;            // require ≥ warp aligned
    if (!resolve_launcher()) return 0;
    int dev = current_device_or_zero();
    if (!ensure_compiled_for(dev)) return 0;
    void* fn = g_per_device[dev].rmsnorm_fn;
    if (!fn) return 0;

    int bx = (hidden_dim >= 1024) ? 1024 : (hidden_dim >= 256 ? 256 : hidden_dim);
    int gx = rows;
    void* args[] = { &x, &weight, &out, &rows, &hidden_dim, &eps };
    if (g_cu_launch(fn, gx, 1, 1, bx, 1, 1, 0, stream, args, nullptr) != 0) return 0;
    g_rmsnorm_calls.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_fused_silu_mul(void* gate, void* up, void* out,
                                      int numel, void* stream) {
    if (!gate || !up || !out || numel <= 0) return 0;
    if (!resolve_launcher()) return 0;
    int dev = current_device_or_zero();
    if (!ensure_compiled_for(dev)) return 0;
    void* fn = g_per_device[dev].silu_mul_fn;
    if (!fn) return 0;

    int bx = 256;
    int gx = (numel + bx - 1) / bx;
    void* args[] = { &gate, &up, &out, &numel };
    if (g_cu_launch(fn, gx, 1, 1, bx, 1, 1, 0, stream, args, nullptr) != 0) return 0;
    g_silu_mul_calls.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_fused_residual_add(void* x, void* residual, void* out,
                                          int numel, void* stream) {
    if (!x || !residual || !out || numel <= 0) return 0;
    if (!resolve_launcher()) return 0;
    int dev = current_device_or_zero();
    if (!ensure_compiled_for(dev)) return 0;
    void* fn = g_per_device[dev].res_add_fn;
    if (!fn) return 0;

    int bx = 256;
    int gx = (numel + bx - 1) / bx;
    void* args[] = { &x, &residual, &out, &numel };
    if (g_cu_launch(fn, gx, 1, 1, bx, 1, 1, 0, stream, args, nullptr) != 0) return 0;
    g_res_add_calls.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_fusion_kernels_stats(CipherFusionStats* out) {
    if (!out) return 0;
    out->enabled            = g_enabled.load(std::memory_order_relaxed);
    out->rmsnorm_calls      = g_rmsnorm_calls.load(std::memory_order_relaxed);
    out->silu_mul_calls     = g_silu_mul_calls.load(std::memory_order_relaxed);
    out->residual_add_calls = g_res_add_calls.load(std::memory_order_relaxed);
    return 1;
}

__attribute__((constructor(112)))
static void cipher_fusion_kernels_autoinit() { cipher_fusion_kernels_init(); }
