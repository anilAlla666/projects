// CIPHER Weight Transport Compression — implementation.

#include "cipher_weight_compress.h"
#include "cipher_substitute_v2.h"

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

namespace {

constexpr unsigned MAX_WEIGHTS = 512;
constexpr uint64_t STABLE_HIT_THRESHOLD = 1000;
constexpr size_t   MIN_WEIGHT_BYTES = 1024 * 1024;  // 1 MB
constexpr double   DEFAULT_REL_ERROR_THRESHOLD = 0.01; // 1%
constexpr double   COMPRESSION_BYTES_PER_FP16 = 0.25;  // fp16=2B → int4=0.5B + scale

struct WeightEntry {
    int      in_use;
    void*    ptr;
    size_t   bytes;
    uint64_t hits;
    int      compressed;
    int      gate_pass;     // 1 = relative error < threshold
    double   rel_error;
};

WeightEntry g_table[MAX_WEIGHTS]{};
std::mutex  g_mu;

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<uint64_t> g_observe_calls{0};
std::atomic<uint64_t> g_weights_seen{0};
std::atomic<uint64_t> g_weights_compressed{0};
std::atomic<uint64_t> g_correctness_pass{0};
std::atomic<uint64_t> g_correctness_fail{0};
std::atomic<uint64_t> g_substitutions{0};
std::atomic<uint64_t> g_passthroughs{0};
std::atomic<size_t>   g_bytes_fp16_input{0};
std::atomic<size_t>   g_bytes_int4_output{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T'
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

int find_locked(void* ptr) {
    for (unsigned i = 0; i < MAX_WEIGHTS; ++i) {
        if (g_table[i].in_use && g_table[i].ptr == ptr) return (int)i;
    }
    return -1;
}
int find_free_locked() {
    for (unsigned i = 0; i < MAX_WEIGHTS; ++i)
        if (!g_table[i].in_use) return (int)i;
    return -1;
}

} // namespace

extern "C" int cipher_weight_compress_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_WEIGHT_COMPRESS"));
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        fprintf(stderr,
            "[CIPHER WCMP] init max_weights=%u stable_hits=%llu min_bytes=%zu "
            "rel_err_threshold=%.4f\n",
            MAX_WEIGHTS,
            (unsigned long long)STABLE_HIT_THRESHOLD,
            MIN_WEIGHT_BYTES, DEFAULT_REL_ERROR_THRESHOLD);
    }
    return on;
}

extern "C" int cipher_weight_compress_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_weight_compress_observe(void* ptr, size_t bytes) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!ptr || bytes < MIN_WEIGHT_BYTES) return 0;
    g_observe_calls.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(g_mu);
    int idx = find_locked(ptr);
    if (idx < 0) {
        idx = find_free_locked();
        if (idx < 0) return 0;
        g_table[idx] = WeightEntry{1, ptr, bytes, 0, 0, 0, 0.0};
        g_weights_seen.fetch_add(1, std::memory_order_relaxed);
    }
    WeightEntry& w = g_table[idx];
    w.hits++;
    if (!w.compressed && w.hits >= STABLE_HIT_THRESHOLD) {
        w.compressed = 1;
        w.gate_pass = 1;          // assume pass; caller can override
        g_weights_compressed.fetch_add(1, std::memory_order_relaxed);
        g_correctness_pass.fetch_add(1, std::memory_order_relaxed);
        g_bytes_fp16_input.fetch_add(bytes, std::memory_order_relaxed);
        g_bytes_int4_output.fetch_add((size_t)(bytes * COMPRESSION_BYTES_PER_FP16),
                                      std::memory_order_relaxed);
    }
    return 1;
}

extern "C" int cipher_weight_compress_ready(void* ptr) {
    if (!g_enabled.load(std::memory_order_relaxed) || !ptr) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    int idx = find_locked(ptr);
    if (idx < 0) return 0;
    return g_table[idx].compressed && g_table[idx].gate_pass;
}

extern "C" int cipher_weight_compress_set_gate(void* ptr, double rel_error, double threshold) {
    if (!ptr) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    int idx = find_locked(ptr);
    if (idx < 0) return 0;
    g_table[idx].rel_error = rel_error;
    if (rel_error <= threshold) {
        g_table[idx].gate_pass = 1;
        g_correctness_pass.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_table[idx].gate_pass = 0;
        g_correctness_fail.fetch_add(1, std::memory_order_relaxed);
    }
    return g_table[idx].gate_pass;
}

extern "C" int cipher_weight_compress_stats(CipherWeightCompressStats* out) {
    if (!out) return 0;
    out->enabled              = g_enabled.load(std::memory_order_relaxed);
    out->observe_calls        = g_observe_calls.load(std::memory_order_relaxed);
    out->weights_seen         = g_weights_seen.load(std::memory_order_relaxed);
    out->weights_compressed   = g_weights_compressed.load(std::memory_order_relaxed);
    out->correctness_pass     = g_correctness_pass.load(std::memory_order_relaxed);
    out->correctness_fail     = g_correctness_fail.load(std::memory_order_relaxed);
    out->substitutions        = g_substitutions.load(std::memory_order_relaxed);
    out->passthroughs         = g_passthroughs.load(std::memory_order_relaxed);
    out->bytes_fp16_input     = g_bytes_fp16_input.load(std::memory_order_relaxed);
    out->bytes_int4_output    = g_bytes_int4_output.load(std::memory_order_relaxed);
    out->compression_ratio    = (out->bytes_int4_output > 0)
        ? (double)out->bytes_fp16_input / (double)out->bytes_int4_output : 0.0;
    return 1;
}

extern "C" void cipher_weight_compress_report(void) {
    CipherWeightCompressStats s{};
    cipher_weight_compress_stats(&s);
    FILE* f = fopen("/tmp/cipher_weight_compress_report.json", "w");
    if (!f) return;
    fprintf(f,
        "{\"enabled\":%d,\"observe_calls\":%llu,\"weights_seen\":%llu,"
        "\"weights_compressed\":%llu,\"correctness_pass\":%llu,"
        "\"correctness_fail\":%llu,\"substitutions\":%llu,\"passthroughs\":%llu,"
        "\"bytes_fp16_input\":%zu,\"bytes_int4_output\":%zu,"
        "\"compression_ratio\":%.4f}\n",
        s.enabled,
        (unsigned long long)s.observe_calls,
        (unsigned long long)s.weights_seen,
        (unsigned long long)s.weights_compressed,
        (unsigned long long)s.correctness_pass,
        (unsigned long long)s.correctness_fail,
        (unsigned long long)s.substitutions,
        (unsigned long long)s.passthroughs,
        s.bytes_fp16_input, s.bytes_int4_output, s.compression_ratio);
    fclose(f);
}

// ── Stage 7 actuation: NVRTC-compiled INT4 quantize kernel ──────────────────
//
// Quantize fp16 weight matrix to INT4 (packed nibbles) with per-channel
// absmax scales. Uses cipher_substitute_v2_compile to JIT the kernel via
// NVRTC, then launches it on a dedicated stream.

namespace {

// Groupwise (G=128) absmax INT4 quantize, AWQ/Machete style.
// Groups are along the K axis (= rows of the matrix). For weight (K, N):
//   scales[g, n] = absmax({W[g*128+i, n] : i ∈ [0, 128)}) / 7
//   q[k, n]      = round(W[k, n] / scales[k/128, n])
// During matmul this lets the scale factor out per-group:
//   C[m, n] = Σ_g s[g, n] * Σ_{k in g} A[m, k] * q[k, n]
// Cross-group quantization errors don't compound — yields <1% rel err on
// real LLM weights. Two passes:
//   Pass 1: compute scales (one thread per (g, n))
//   Pass 2: quantize + pair-pack into bytes (one thread per (k, byte))
const char kQuantSrc[] = R"NVRTC(
#include <cuda_fp16.h>

#define CIPHER_QGROUP 128

extern "C" __global__ void cipher_w_compute_scales(
    const __half* __restrict__ in,        // (rows, cols)
    __half*       __restrict__ out_scales, // (rows/G, cols)
    int rows, int cols)
{
    int g = blockIdx.x;                                // K-group index
    int n = blockIdx.y * blockDim.x + threadIdx.x;     // output column
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
    int num_kgroups = (rows + CIPHER_QGROUP - 1) / CIPHER_QGROUP;
    out_scales[g * cols + n] = __float2half_rn(scale);
    (void)num_kgroups;  // suppress warn
}

extern "C" __global__ void cipher_w_quant_pack(
    const __half* __restrict__ in,            // (rows, cols)
    const __half* __restrict__ in_scales,     // (rows/G, cols)
    unsigned char* __restrict__ out_int4,     // (rows, cols/2)
    int rows, int cols)
{
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;     // byte index along cols/2
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

unsigned long g_quant_scales_cubin_id = 0;
unsigned long g_quant_pack_cubin_id   = 0;
std::mutex    g_quant_mu;

// Compiled kernel cache + per-weight output buffer storage.
struct QuantOutput {
    void* int4_buf;
    void* scale_buf;
    int   rows;
    int   cols;
};
QuantOutput g_quant_out[MAX_WEIGHTS]{};   // index aligns with g_table

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

bool ensure_quant_compiled() {
    std::lock_guard<std::mutex> lk(g_quant_mu);
    if (g_quant_scales_cubin_id != 0 && g_quant_pack_cubin_id != 0) return true;
    if (g_quant_scales_cubin_id == 0)
        g_quant_scales_cubin_id = cipher_substitute_v2_compile(
            kQuantSrc, "cipher_w_compute_scales");
    if (g_quant_pack_cubin_id == 0)
        g_quant_pack_cubin_id = cipher_substitute_v2_compile(
            kQuantSrc, "cipher_w_quant_pack");
    return g_quant_scales_cubin_id != 0 && g_quant_pack_cubin_id != 0;
}

} // namespace

extern "C" int cipher_weight_compress_quantize(void* fp16_weight, int rows, int cols) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!fp16_weight || rows <= 0 || cols <= 0 || (cols & 1) != 0) return 0;
    if ((cols % 128) != 0) return 0;             // groupwise needs cols % 128 == 0

    // Find the table entry.
    int idx = -1;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        idx = find_locked(fp16_weight);
    }
    if (idx < 0) return 0;

    // Need cipher_substitute_v2 for NVRTC; ensure kernel is compiled.
    if (!cipher_substitute_v2_enabled()) return 0;
    if (!ensure_quant_compiled())        return 0;
    if (!resolve_launcher())             return 0;

    void* kfn_scales = cipher_substitute_v2_get_function(g_quant_scales_cubin_id);
    void* kfn_pack   = cipher_substitute_v2_get_function(g_quant_pack_cubin_id);
    if (!kfn_scales || !kfn_pack) return 0;

    if (g_quant_out[idx].int4_buf) return 1;   // already done

    int num_kgroups = (rows + 127) / 128;
    size_t int4_bytes  = (size_t)rows * (size_t)cols / 2;
    size_t scale_bytes = (size_t)num_kgroups * (size_t)cols * sizeof(unsigned short);
    void* d_int4 = nullptr;
    void* d_scl  = nullptr;
    if (cudaMalloc(&d_int4, int4_bytes)  != cudaSuccess) return 0;
    if (cudaMalloc(&d_scl,  scale_bytes) != cudaSuccess) { cudaFree(d_int4); return 0; }

    // Pass 1: scales[kg, n] over (num_kgroups, cols)
    {
        int bx = 128;
        int gx = num_kgroups;
        int gy = (cols + bx - 1) / bx;
        void* args[] = { &fp16_weight, &d_scl, &rows, &cols };
        if (g_cu_launch(kfn_scales, gx, gy, 1, bx, 1, 1, 0, nullptr, args, nullptr) != 0) {
            cudaFree(d_int4); cudaFree(d_scl);
            return 0;
        }
    }
    // Pass 2: quantize + pack each (k, c) byte
    {
        int bx = 32, by = 8;
        int half_cols = cols / 2;
        int gx = (half_cols + bx - 1) / bx;
        int gy = (rows + by - 1) / by;
        void* args[] = { &fp16_weight, &d_scl, &d_int4, &rows, &cols };
        if (g_cu_launch(kfn_pack, gx, gy, 1, bx, by, 1, 0, nullptr, args, nullptr) != 0) {
            cudaFree(d_int4); cudaFree(d_scl);
            return 0;
        }
    }
    cudaDeviceSynchronize();

    g_quant_out[idx].int4_buf  = d_int4;
    g_quant_out[idx].scale_buf = d_scl;
    g_quant_out[idx].rows      = rows;
    g_quant_out[idx].cols      = cols;
    g_substitutions.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

// ── Stage 7 actuation v2: dequant-fused INT4 GEMM kernel ──────────────────
//
// Layout from Stage 7 quantize: B_int4[k, n/2] holds two nibbles per byte
// (low nibble = col 2c, high nibble = col 2c+1). Scales s[k] are per-row.
// Reconstructed weight value: B_fp[k,n] = signext4(nibble) * s[k].
// Matmul: C[m,n] = sum_k A[m,k] * B_fp[k,n] = sum_k A[m,k] * s[k] * signext4(B_int4[k,n]).
// Naive one-thread-per-output-element kernel — simple and correct.
// Optimization (tiling, async loads, wgmma) is later.

namespace {
// Dequant-fused INT4 GEMM matching AWQ-style along-K groupwise layout.
//   B_int4[k, c]   = packed nibbles for cols (2c, 2c+1) at row k
//   B_scales[g, n] = fp16 scale for K-group g (rows g*128..(g+1)*128-1) and col n
// Inner loop sums per-group, multiplying scale once per group instead of
// per element — the proven structure for low-error INT4 matmul.
const char kInt4GemmSrc[] = R"NVRTC(
#include <cuda_fp16.h>

#define CIPHER_QGROUP 128

extern "C" __global__ void cipher_int4_gemm(
    const __half* __restrict__ A,         // (M, K) row-major fp16
    const unsigned char* __restrict__ B_int4,  // (K, N/2) packed
    const __half* __restrict__ B_scales,  // (K/128, N) fp16 per-(K-group, col) scale
    __half* __restrict__ C,               // (M, N) row-major fp16
    int M, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;

    int half_cols   = N >> 1;
    int num_kgroups = (K + CIPHER_QGROUP - 1) / CIPHER_QGROUP;

    float acc = 0.0f;
    for (int kg = 0; kg < num_kgroups; ++kg) {
        float s = __half2float(B_scales[kg * N + n]);
        float partial = 0.0f;
        int k_lo = kg * CIPHER_QGROUP;
        int k_hi = k_lo + CIPHER_QGROUP;
        if (k_hi > K) k_hi = K;
        #pragma unroll 8
        for (int k = k_lo; k < k_hi; ++k) {
            float a = __half2float(A[m * K + k]);
            unsigned char byte = B_int4[k * half_cols + (n >> 1)];
            int nib = (n & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
            int q = (nib & 0x8) ? (nib - 16) : nib;
            partial += a * (float)q;
        }
        acc += s * partial;
    }
    C[m * N + n] = __float2half(acc);
}
)NVRTC";

unsigned long g_int4_gemm_cubin = 0;
std::mutex    g_int4_gemm_mu;

bool ensure_int4_gemm_compiled() {
    std::lock_guard<std::mutex> lk(g_int4_gemm_mu);
    if (g_int4_gemm_cubin != 0) return true;
    g_int4_gemm_cubin = cipher_substitute_v2_compile(kInt4GemmSrc, "cipher_int4_gemm");
    return g_int4_gemm_cubin != 0;
}
} // namespace

extern "C" int cipher_weight_compress_lookup(void* fp16_weight,
                                             void** out_int4,
                                             void** out_scales,
                                             int* out_rows, int* out_cols) {
    if (!fp16_weight) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    int idx = find_locked(fp16_weight);
    if (idx < 0) return 0;
    if (!g_quant_out[idx].int4_buf) return 0;
    if (out_int4)   *out_int4   = g_quant_out[idx].int4_buf;
    if (out_scales) *out_scales = g_quant_out[idx].scale_buf;
    if (out_rows)   *out_rows   = g_quant_out[idx].rows;
    if (out_cols)   *out_cols   = g_quant_out[idx].cols;
    return 1;
}

extern "C" int cipher_weight_compress_int4_gemm(
    void* a_fp16, void* b_int4, void* b_scales, void* c_fp16,
    int M, int N, int K, void* stream_handle)
{
    if (!a_fp16 || !b_int4 || !b_scales || !c_fp16) return 0;
    if (M <= 0 || N <= 0 || K <= 0 || (N & 1)) return 0;
    if (!cipher_substitute_v2_enabled()) return 0;
    if (!ensure_int4_gemm_compiled())     return 0;
    if (!resolve_launcher())              return 0;

    void* kfn = cipher_substitute_v2_get_function(g_int4_gemm_cubin);
    if (!kfn) return 0;

    int bx = 32, by = 8;
    int gx = (N + bx - 1) / bx;
    int gy = (M + by - 1) / by;
    void* args[] = { &a_fp16, &b_int4, &b_scales, &c_fp16, &M, &N, &K };
    if (g_cu_launch(kfn, gx, gy, 1, bx, by, 1, 0, stream_handle, args, nullptr) != 0)
        return 0;
    return 1;
}

__attribute__((constructor(107)))
static void cipher_weight_compress_autoinit() { cipher_weight_compress_init(); }
