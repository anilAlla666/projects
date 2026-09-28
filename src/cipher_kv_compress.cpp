// CIPHER KV Cache Compression — implementation.

#include "cipher_kv_compress.h"
#include "cipher_substitute_v2.h"

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

namespace {

constexpr unsigned MAX_REGIONS         = 256;
constexpr int      DEFAULT_RESIDUAL_W  = 32;
constexpr uint64_t STABLE_HIT_THRESHOLD = 200;
// fp16 = 2B → 2-bit = 0.25B per element + small per-channel/per-token scale
constexpr double   COMPRESSION_BYTES_PER_FP16 = 0.18;

struct KvRegion {
    int      in_use;
    int      layer_id;
    void*    ptr;
    size_t   bytes;
    uint64_t hits;
    int      compressed;
    int      full_precision;   // 1 if layer in allowlist
};

KvRegion   g_table[MAX_REGIONS]{};
std::mutex g_mu;

int        g_residual_window = DEFAULT_RESIDUAL_W;

constexpr unsigned MAX_FULL_LAYERS = 16;
int        g_full_layers[MAX_FULL_LAYERS];
int        g_full_layer_count = 0;

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<uint64_t> g_observe_calls{0};
std::atomic<uint64_t> g_kv_regions_seen{0};
std::atomic<uint64_t> g_kv_regions_compressed{0};
std::atomic<uint64_t> g_layers_full_precision{0};
std::atomic<uint64_t> g_layers_compressed{0};
std::atomic<uint64_t> g_substitutions{0};
std::atomic<size_t>   g_bytes_fp16_input{0};
std::atomic<size_t>   g_bytes_2bit_output{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T'
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

void parse_full_layers(const char* csv) {
    g_full_layer_count = 0;
    if (!csv) return;
    const char* p = csv;
    while (*p && g_full_layer_count < (int)MAX_FULL_LAYERS) {
        char* end = nullptr;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        g_full_layers[g_full_layer_count++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
}

bool is_full_precision_layer(int layer_id) {
    for (int i = 0; i < g_full_layer_count; ++i)
        if (g_full_layers[i] == layer_id) return true;
    return false;
}

int find_locked(void* ptr) {
    for (unsigned i = 0; i < MAX_REGIONS; ++i)
        if (g_table[i].in_use && g_table[i].ptr == ptr) return (int)i;
    return -1;
}
int find_free_locked() {
    for (unsigned i = 0; i < MAX_REGIONS; ++i) if (!g_table[i].in_use) return (int)i;
    return -1;
}

} // namespace

extern "C" int cipher_kv_compress_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_KV_COMPRESS"));
    g_enabled.store(on, std::memory_order_release);
    parse_full_layers(getenv("CIPHER_KV_FULL_LAYERS"));
    if (const char* w = getenv("CIPHER_KV_RESIDUAL_W")) {
        int v = atoi(w);
        if (v > 0 && v < 1024) g_residual_window = v;
    }
    if (on) {
        fprintf(stderr,
            "[CIPHER KVCMP] init max_regions=%u residual_w=%d full_layers=%d\n",
            MAX_REGIONS, g_residual_window, g_full_layer_count);
    }
    return on;
}

extern "C" int cipher_kv_compress_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_kv_compress_observe(int layer_id, void* ptr, size_t bytes) {
    if (!g_enabled.load(std::memory_order_relaxed) || !ptr) return 0;
    g_observe_calls.fetch_add(1, std::memory_order_relaxed);

    bool full_prec = is_full_precision_layer(layer_id);
    std::lock_guard<std::mutex> lk(g_mu);
    int idx = find_locked(ptr);
    if (idx < 0) {
        idx = find_free_locked();
        if (idx < 0) return 0;
        g_table[idx] = KvRegion{1, layer_id, ptr, bytes, 0, 0, full_prec ? 1 : 0};
        g_kv_regions_seen.fetch_add(1, std::memory_order_relaxed);
        if (full_prec) g_layers_full_precision.fetch_add(1, std::memory_order_relaxed);
    }
    KvRegion& r = g_table[idx];
    if (bytes > r.bytes) r.bytes = bytes;
    r.hits++;
    if (!r.full_precision && !r.compressed && r.hits >= STABLE_HIT_THRESHOLD) {
        r.compressed = 1;
        g_kv_regions_compressed.fetch_add(1, std::memory_order_relaxed);
        g_layers_compressed.fetch_add(1, std::memory_order_relaxed);
        g_bytes_fp16_input.fetch_add(r.bytes, std::memory_order_relaxed);
        g_bytes_2bit_output.fetch_add(
            (size_t)(r.bytes * COMPRESSION_BYTES_PER_FP16),
            std::memory_order_relaxed);
    }
    return 1;
}

extern "C" int cipher_kv_compress_ready(void* ptr) {
    if (!g_enabled.load(std::memory_order_relaxed) || !ptr) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    int idx = find_locked(ptr);
    if (idx < 0) return 0;
    return g_table[idx].compressed && !g_table[idx].full_precision;
}

extern "C" int cipher_kv_compress_stats(CipherKvCompressStats* out) {
    if (!out) return 0;
    out->enabled                = g_enabled.load(std::memory_order_relaxed);
    out->residual_window        = g_residual_window;
    out->observe_calls          = g_observe_calls.load(std::memory_order_relaxed);
    out->kv_regions_seen        = g_kv_regions_seen.load(std::memory_order_relaxed);
    out->kv_regions_compressed  = g_kv_regions_compressed.load(std::memory_order_relaxed);
    out->layers_full_precision  = g_layers_full_precision.load(std::memory_order_relaxed);
    out->layers_compressed      = g_layers_compressed.load(std::memory_order_relaxed);
    out->substitutions          = g_substitutions.load(std::memory_order_relaxed);
    out->bytes_fp16_input       = g_bytes_fp16_input.load(std::memory_order_relaxed);
    out->bytes_2bit_output      = g_bytes_2bit_output.load(std::memory_order_relaxed);
    out->compression_ratio      = (out->bytes_2bit_output > 0)
        ? (double)out->bytes_fp16_input / (double)out->bytes_2bit_output : 0.0;
    return 1;
}

extern "C" void cipher_kv_compress_report(void) {
    CipherKvCompressStats s{};
    cipher_kv_compress_stats(&s);
    FILE* f = fopen("/tmp/cipher_kv_compress_report.json", "w");
    if (!f) return;
    fprintf(f,
        "{\"enabled\":%d,\"residual_window\":%d,\"observe_calls\":%llu,"
        "\"kv_regions_seen\":%llu,\"kv_regions_compressed\":%llu,"
        "\"layers_full_precision\":%llu,\"layers_compressed\":%llu,"
        "\"substitutions\":%llu,\"bytes_fp16_input\":%zu,"
        "\"bytes_2bit_output\":%zu,\"compression_ratio\":%.4f}\n",
        s.enabled, s.residual_window,
        (unsigned long long)s.observe_calls,
        (unsigned long long)s.kv_regions_seen,
        (unsigned long long)s.kv_regions_compressed,
        (unsigned long long)s.layers_full_precision,
        (unsigned long long)s.layers_compressed,
        (unsigned long long)s.substitutions,
        s.bytes_fp16_input, s.bytes_2bit_output, s.compression_ratio);
    fclose(f);
}

// ── Stage 8 actuation: 2-bit asymmetric KV quantize via NVRTC ──────────────

namespace {

// 2-bit packing: 4 quants per byte. Asymmetric quant: q = (v - zp) / scale.
// scale = (max - min) / 3, zp = min.
const char kKvQuantSrc[] = R"NVRTC(
#include <cuda_fp16.h>

extern "C" __global__ void cipher_kv_quant_2bit(
    const __half* __restrict__ in,
    unsigned char* __restrict__ out_2bit,
    __half*       __restrict__ out_scales,
    __half*       __restrict__ out_zeros,
    int rows, int cols, int per_channel)
{
    // per_channel=1 → per row (key). per_channel=0 → per col (value, per-token).
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    __shared__ float s_min, s_max;
    if (tid == 0) { s_min = 1e30f; s_max = -1e30f; }
    __syncthreads();

    float local_min = 1e30f, local_max = -1e30f;
    int n = (per_channel) ? cols : rows;
    int idx_base = (per_channel) ? row * cols : row;          // start
    int idx_step = (per_channel) ? 1 : cols;                  // stride
    for (int c = tid; c < n; c += blockDim.x) {
        float v = __half2float(in[idx_base + c * idx_step]);
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float omin = __shfl_xor_sync(0xffffffff, local_min, off);
        float omax = __shfl_xor_sync(0xffffffff, local_max, off);
        if (omin < local_min) local_min = omin;
        if (omax > local_max) local_max = omax;
    }
    __shared__ float bmin[32], bmax[32];
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) { bmin[warp] = local_min; bmax[warp] = local_max; }
    __syncthreads();
    if (warp == 0) {
        float v_min = (tid < (blockDim.x + 31) / 32) ? bmin[lane] :  1e30f;
        float v_max = (tid < (blockDim.x + 31) / 32) ? bmax[lane] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            float omin = __shfl_xor_sync(0xffffffff, v_min, off);
            float omax = __shfl_xor_sync(0xffffffff, v_max, off);
            if (omin < v_min) v_min = omin;
            if (omax > v_max) v_max = omax;
        }
        if (lane == 0) { s_min = v_min; s_max = v_max; }
    }
    __syncthreads();

    float scale = (s_max - s_min) / 3.0f;
    if (scale < 1e-7f) scale = 1e-7f;
    float inv = 1.0f / scale;
    if (tid == 0) {
        out_scales[row] = __float2half_rn(scale);
        out_zeros [row] = __float2half_rn(s_min);
    }

    int packs = (n + 3) / 4;
    for (int p = tid; p < packs; p += blockDim.x) {
        int base = p * 4;
        unsigned char b = 0;
        for (int k = 0; k < 4; ++k) {
            int idx = base + k;
            if (idx >= n) break;
            float v = __half2float(in[idx_base + idx * idx_step]);
            int q = (int)((v - s_min) * inv);
            if (q < 0) q = 0; if (q > 3) q = 3;
            b |= (unsigned char)((q & 0x3) << (k * 2));
        }
        // Output layout: row-major over rows; cols-stride packs columns.
        if (per_channel) out_2bit[row * packs + p] = b;
        else             out_2bit[p   * rows + row] = b;
    }
}
)NVRTC";

unsigned long g_kv_cubin_id = 0;
std::mutex    g_kv_mu;

struct KvOutput {
    void* twobit_buf;
    void* scale_buf;
    void* zero_buf;
    int   rows, cols, mode;
};
KvOutput g_kv_out[MAX_REGIONS]{};

typedef int (*pf_cuLaunchKernel2)(void*, unsigned, unsigned, unsigned,
                                   unsigned, unsigned, unsigned,
                                   unsigned, void*, void**, void**);
pf_cuLaunchKernel2 g_kv_launch = nullptr;

bool resolve_launcher2() {
    if (g_kv_launch) return true;
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) return false;
    g_kv_launch = (pf_cuLaunchKernel2)dlsym(lib, "cuLaunchKernel");
    return g_kv_launch != nullptr;
}

bool ensure_kv_compiled() {
    std::lock_guard<std::mutex> lk(g_kv_mu);
    if (g_kv_cubin_id != 0) return true;
    g_kv_cubin_id = cipher_substitute_v2_compile(kKvQuantSrc, "cipher_kv_quant_2bit");
    return g_kv_cubin_id != 0;
}

} // namespace

extern "C" int cipher_kv_compress_quantize(void* fp16_kv, int rows, int cols, int mode) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!fp16_kv || rows <= 0 || cols <= 0) return 0;

    int idx = -1;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        idx = find_locked(fp16_kv);
    }
    if (idx < 0) return 0;
    if (g_kv_out[idx].twobit_buf) return 1;

    if (!cipher_substitute_v2_enabled()) return 0;
    if (!ensure_kv_compiled())           return 0;
    if (!resolve_launcher2())            return 0;

    void* kfn = cipher_substitute_v2_get_function(g_kv_cubin_id);
    if (!kfn) return 0;

    int per_channel = (mode == 0) ? 1 : 0;
    int n = per_channel ? cols : rows;
    int packs = (n + 3) / 4;

    size_t out_bytes = (size_t)rows * (size_t)packs;
    void* d_q = nullptr;
    void* d_s = nullptr;
    void* d_z = nullptr;
    if (cudaMalloc(&d_q, out_bytes) != cudaSuccess) return 0;
    if (cudaMalloc(&d_s, (size_t)rows * 2) != cudaSuccess) { cudaFree(d_q); return 0; }
    if (cudaMalloc(&d_z, (size_t)rows * 2) != cudaSuccess) { cudaFree(d_q); cudaFree(d_s); return 0; }

    int gx = rows, gy = 1, gz = 1;
    int bx = 256,  by = 1, bz = 1;
    void* args[] = { &fp16_kv, &d_q, &d_s, &d_z, &rows, &cols, &per_channel };
    if (g_kv_launch(kfn, gx, gy, gz, bx, by, bz, 0, nullptr, args, nullptr) != 0) {
        cudaFree(d_q); cudaFree(d_s); cudaFree(d_z);
        return 0;
    }
    cudaDeviceSynchronize();

    g_kv_out[idx].twobit_buf = d_q;
    g_kv_out[idx].scale_buf  = d_s;
    g_kv_out[idx].zero_buf   = d_z;
    g_kv_out[idx].rows = rows; g_kv_out[idx].cols = cols; g_kv_out[idx].mode = mode;
    g_substitutions.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

__attribute__((constructor(108)))
static void cipher_kv_compress_autoinit() { cipher_kv_compress_init(); }
