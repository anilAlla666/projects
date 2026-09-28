// CIPHER KV-Cache Redirect — Path A.
//   V1: memcpy redirect (verified — output identical to baseline).
//   V2: replace memcpy with quant→dequant round-trip (compression noise gate).
#include "cipher_kv_redirect.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>

extern "C" unsigned long cipher_substitute_v2_compile(const char* source, const char* kernel_name);
extern "C" int  cipher_substitute_v2_enabled(void);
extern "C" void* cipher_substitute_v2_get_function(unsigned long cubin_id);

namespace {

constexpr int    MAX_LAYERS    = 64;        // upper bound (Mistral 7B has 32)
constexpr size_t BUF_BYTES     = 1 << 20;   // 1 MB per buffer; covers Mistral
                                            // GQA at max_cache_len up to ~512.

// V3: per-layer persistent compressed cache. ROW=128 = head_dim for Mistral
// (and most modern LLMs). 8 heads × MAX_TOKENS = total rows per layer.
constexpr int    MAX_TOKENS  = 4096;        // max cache len (per layer)
constexpr int    KV_HEADS    = 8;           // Mistral GQA num_kv_heads

struct LayerBufs {
    // Staging scratch for FA struct rewrite (1 MB each).
    void* k_buf  = nullptr;
    void* v_buf  = nullptr;
    // V2 round-trip scratch (still used in v2_mode=1 path).
    void* k_q    = nullptr;
    void* v_q    = nullptr;
    void* k_meta = nullptr;
    void* v_meta = nullptr;
    // V3 persistent compressed cache: KV_HEADS × MAX_TOKENS rows of 128 fp16,
    // packed to 2-bit (32 bytes/row) plus per-row min/scale (4 bytes/row).
    void*   cache_k_q    = nullptr;          // 8 * 4096 * 32 = 1 MB
    void*   cache_v_q    = nullptr;
    void*   cache_k_meta = nullptr;          // 8 * 4096 * 4  = 128 KB
    void*   cache_v_meta = nullptr;
};

LayerBufs                 g_buf[MAX_LAYERS];
std::atomic<int>          g_enabled{0};
std::atomic<int>          g_v2_mode{0};   // 0=memcpy, 1=quant+dequant round-trip
std::atomic<int>          g_v3_mode{0};   // 0=off, 1=V3 (per-layer compressed cache)
std::mutex                g_init_mu;
std::atomic<uint64_t>     g_fa_seen{0};
std::atomic<uint64_t>     g_fa_rewritten{0};
std::atomic<uint64_t>     g_bytes_copied{0};
unsigned long             g_quant_cubin        = 0;
unsigned long             g_dequant_cubin      = 0;
unsigned long             g_dequant_perm_cubin = 0;
std::mutex                g_kern_mu;

// V3 state: layer position tracker + KV GEMM call counter.
int                       g_layer_pos[MAX_LAYERS] = {0};
std::atomic<int>          g_kv_gemm_count{0};   // 0=L0 K, 1=L0 V, 2=L1 K, ...

// Number of bytes to copy/quantize per K/V buffer. FA reads up to ~1 MB
// out of each staging buffer (verified empirically: smaller sizes break
// correctness, 1 MB matches OFF baseline exactly). Configurable via env
// CIPHER_KV_RDR_BYTES.
size_t g_copy_bytes = 1 << 20;        // 1 MB default
#define COPY_BYTES (g_copy_bytes)

// V2 kernels: 2-bit asymmetric quant + dequant on a contiguous fp16 buffer.
// Tile = 128 elements per row → one fp16 (min, scale) pair per row, 32 packed
// 2-bit values per uint8 row-chunk. We use ROW = 128 hardcoded — divides 1 MB
// fp16 (524288 elements) into 4096 rows, no remainder.
constexpr int    KV_ROW            = 128;
constexpr size_t FP16_ELEMENTS_1MB = (1u << 20) / 2;  // 524288

// NVRTC source: quantizer fills q[ROWS, ROW/4] uint8 + meta[ROWS] fp16x2.
const char kKvQuantSrc[] = R"NVRTC(
#include <cuda_fp16.h>

#define ROW 128

extern "C" __global__ void cipher_kv_q2(
    const __half* __restrict__ in,         // [N_ROWS, ROW] fp16
    unsigned char* __restrict__ q,         // [N_ROWS, ROW/4] uint8 (4 vals/byte)
    __half2* __restrict__ meta,            // [N_ROWS] (min, inv_scale)
    int n_rows)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;
    const __half* r = in + row * ROW;

    // Pass 1: per-row min and max.
    float vmin = 1e30f, vmax = -1e30f;
    int tid = threadIdx.x;
    for (int j = tid; j < ROW; j += blockDim.x) {
        float v = __half2float(r[j]);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    // Block reduction
    __shared__ float smin[32];
    __shared__ float smax[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        float a = __shfl_down_sync(0xffffffff, vmin, off);
        float b = __shfl_down_sync(0xffffffff, vmax, off);
        if (a < vmin) vmin = a;
        if (b > vmax) vmax = b;
    }
    if (lane == 0) { smin[warp] = vmin; smax[warp] = vmax; }
    __syncthreads();
    if (warp == 0) {
        vmin = (tid < (blockDim.x >> 5)) ? smin[lane] : 1e30f;
        vmax = (tid < (blockDim.x >> 5)) ? smax[lane] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            float a = __shfl_down_sync(0xffffffff, vmin, off);
            float b = __shfl_down_sync(0xffffffff, vmax, off);
            if (a < vmin) vmin = a;
            if (b > vmax) vmax = b;
        }
        if (lane == 0) { smin[0] = vmin; smax[0] = vmax; }
    }
    __syncthreads();
    vmin = smin[0]; vmax = smax[0];
    float scale = (vmax - vmin) / 3.0f;
    float inv   = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;

    // Pass 2: quantize. Each thread packs 4 vals → 1 byte.
    for (int j = tid * 4; j < ROW; j += blockDim.x * 4) {
        unsigned int b = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float v = __half2float(r[j + k]);
            int   q2 = (int)__float2int_rn((v - vmin) * inv);
            if (q2 < 0) q2 = 0; else if (q2 > 3) q2 = 3;
            b |= (q2 & 0x3) << (k * 2);
        }
        q[row * (ROW/4) + (j >> 2)] = (unsigned char)b;
    }
    if (tid == 0) {
        meta[row] = __floats2half2_rn(vmin, scale);   // (min, scale)
    }
}
)NVRTC";

const char kKvDequantSrc[] = R"NVRTC(
#include <cuda_fp16.h>

#define ROW 128

extern "C" __global__ void cipher_kv_dq2(
    __half* __restrict__ out,              // [N_ROWS, ROW] fp16
    const unsigned char* __restrict__ q,   // [N_ROWS, ROW/4]
    const __half2* __restrict__ meta,      // [N_ROWS] (min, scale)
    int n_rows)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;
    __half2 m = meta[row];
    float vmin  = __half2float(m.x);
    float scale = __half2float(m.y);

    int tid = threadIdx.x;
    for (int j = tid * 4; j < ROW; j += blockDim.x * 4) {
        unsigned int b = q[row * (ROW/4) + (j >> 2)];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int q2 = (int)((b >> (k * 2)) & 0x3);
            float v = vmin + (float)q2 * scale;
            out[row * ROW + j + k] = __float2half(v);
        }
    }
}

// Permuted dequant: reads compressed in (token, head, dim) order (matching
// cuBLAS K_proj output layout), writes uncompressed in (head, token, dim)
// order (the layout PyTorch's MEA expects in its key_ptr buffer). gridDim.x
// = n_heads * n_tokens; blockIdx.x = head*n_tokens + token (output row).
extern "C" __global__ void cipher_kv_dq2_perm(
    __half* __restrict__ out,              // [n_heads, n_tokens, ROW] fp16
    const unsigned char* __restrict__ q,   // [n_tokens, n_heads, ROW/4]
    const __half2* __restrict__ meta,      // [n_tokens, n_heads]
    int n_heads, int n_tokens)
{
    int row_out = blockIdx.x;
    int h = row_out / n_tokens;
    int t = row_out - h * n_tokens;
    int row_in = t * n_heads + h;

    __half2 m = meta[row_in];
    float vmin  = __half2float(m.x);
    float scale = __half2float(m.y);

    int tid = threadIdx.x;
    for (int j = tid * 4; j < ROW; j += blockDim.x * 4) {
        unsigned int b = q[row_in * (ROW/4) + (j >> 2)];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int q2 = (int)((b >> (k * 2)) & 0x3);
            float v = vmin + (float)q2 * scale;
            out[row_out * ROW + j + k] = __float2half(v);
        }
    }
}
)NVRTC";

bool ensure_kv_kernels() {
    if (g_quant_cubin && g_dequant_cubin && g_dequant_perm_cubin) return true;
    std::lock_guard<std::mutex> lk(g_kern_mu);
    if (!cipher_substitute_v2_enabled()) return false;
    if (!g_quant_cubin)
        g_quant_cubin = cipher_substitute_v2_compile(kKvQuantSrc, "cipher_kv_q2");
    if (!g_dequant_cubin)
        g_dequant_cubin = cipher_substitute_v2_compile(kKvDequantSrc, "cipher_kv_dq2");
    if (!g_dequant_perm_cubin)
        g_dequant_perm_cubin = cipher_substitute_v2_compile(
            kKvDequantSrc, "cipher_kv_dq2_perm");
    return g_quant_cubin && g_dequant_cubin && g_dequant_perm_cubin;
}

typedef int (*pf_cuLaunchKernel)(void*, unsigned, unsigned, unsigned,
                                  unsigned, unsigned, unsigned,
                                  unsigned, void*, void**, void**);
pf_cuLaunchKernel g_kv_launcher = nullptr;
bool resolve_kv_launcher() {
    if (g_kv_launcher) return true;
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) return false;
    g_kv_launcher = (pf_cuLaunchKernel)dlsym(lib, "cuLaunchKernel");
    return g_kv_launcher != nullptr;
}

bool quant_dequant_buffer(void* src_fp16, void* dst_fp16,
                           void* qbuf, void* mbuf,
                           int n_rows, void* stream)
{
    if (!ensure_kv_kernels() || !resolve_kv_launcher()) return false;

    void* qfn = cipher_substitute_v2_get_function(g_quant_cubin);
    void* dfn = cipher_substitute_v2_get_function(g_dequant_cubin);
    if (!qfn || !dfn) return false;

    void* qargs[] = { &src_fp16, &qbuf, &mbuf, &n_rows };
    if (g_kv_launcher(qfn, n_rows, 1, 1, 128, 1, 1, 0, stream, qargs, nullptr) != 0)
        return false;
    void* dargs[] = { &dst_fp16, &qbuf, &mbuf, &n_rows };
    if (g_kv_launcher(dfn, n_rows, 1, 1, 128, 1, 1, 0, stream, dargs, nullptr) != 0)
        return false;
    return true;
}

bool quant_buffer_only(void* src_fp16, void* qbuf, void* mbuf,
                        int n_rows, void* stream)
{
    if (!ensure_kv_kernels() || !resolve_kv_launcher()) return false;
    void* qfn = cipher_substitute_v2_get_function(g_quant_cubin);
    if (!qfn) return false;
    void* qargs[] = { &src_fp16, &qbuf, &mbuf, &n_rows };
    return g_kv_launcher(qfn, n_rows, 1, 1, 128, 1, 1, 0, stream, qargs, nullptr) == 0;
}

bool dequant_buffer_only(void* dst_fp16, void* qbuf, void* mbuf,
                          int n_rows, void* stream)
{
    if (!ensure_kv_kernels() || !resolve_kv_launcher()) return false;
    void* dfn = cipher_substitute_v2_get_function(g_dequant_cubin);
    if (!dfn) return false;
    void* dargs[] = { &dst_fp16, &qbuf, &mbuf, &n_rows };
    return g_kv_launcher(dfn, n_rows, 1, 1, 128, 1, 1, 0, stream, dargs, nullptr) == 0;
}

bool dequant_buffer_perm(void* dst_fp16, void* qbuf, void* mbuf,
                          int n_heads, int n_tokens, void* stream)
{
    if (!ensure_kv_kernels() || !resolve_kv_launcher()) return false;
    void* dfn = cipher_substitute_v2_get_function(g_dequant_perm_cubin);
    if (!dfn) return false;
    int n_rows = n_heads * n_tokens;
    void* dargs[] = { &dst_fp16, &qbuf, &mbuf, &n_heads, &n_tokens };
    return g_kv_launcher(dfn, n_rows, 1, 1, 128, 1, 1, 0, stream, dargs, nullptr) == 0;
}

bool ensure_buffers() {
    if (g_buf[0].k_buf) return true;
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (g_buf[0].k_buf) return true;
    // V2 q/meta scratch.
    const size_t qbuf_bytes = BUF_BYTES / 8;
    const size_t meta_rows  = BUF_BYTES / (KV_ROW * 2);
    const size_t meta_bytes = meta_rows * 4;
    // V3 persistent cache size:
    //   compressed = KV_HEADS × MAX_TOKENS × 32 bytes
    //   meta       = KV_HEADS × MAX_TOKENS × 4 bytes (fp16 min, fp16 scale)
    const size_t cache_q_bytes  = (size_t)KV_HEADS * MAX_TOKENS * 32;
    const size_t cache_md_bytes = (size_t)KV_HEADS * MAX_TOKENS * 4;
    for (int i = 0; i < MAX_LAYERS; ++i) {
        if (cudaMalloc(&g_buf[i].k_buf,  BUF_BYTES)  != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].v_buf,  BUF_BYTES)  != cudaSuccess) return false;
        // Initialize staging to zero so beyond-actual-data positions read as
        // 0.0 fp16 (no NaN propagation through FA's wmma tile reads).
        cudaMemset(g_buf[i].k_buf, 0, BUF_BYTES);
        cudaMemset(g_buf[i].v_buf, 0, BUF_BYTES);
        if (cudaMalloc(&g_buf[i].k_q,    qbuf_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].v_q,    qbuf_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].k_meta, meta_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].v_meta, meta_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].cache_k_q,    cache_q_bytes)  != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].cache_v_q,    cache_q_bytes)  != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].cache_k_meta, cache_md_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].cache_v_meta, cache_md_bytes) != cudaSuccess) return false;
    }
    return true;
}

} // namespace

extern "C" int cipher_kv_redirect_init(void) {
    const char* e = getenv("CIPHER_KV_REDIRECT");
    int on = (e && *e && strcmp(e, "0") != 0 && strcmp(e, "off") != 0) ? 1 : 0;
    g_enabled.store(on, std::memory_order_relaxed);
    if (const char* b = getenv("CIPHER_KV_RDR_BYTES")) {
        long v = strtol(b, nullptr, 0);
        if (v > 0 && (size_t)v <= BUF_BYTES) g_copy_bytes = (size_t)v;
    }
    if (const char* m = getenv("CIPHER_KV_RDR_V2")) {
        if (m[0] != '0' && strcmp(m, "off") != 0)
            g_v2_mode.store(1, std::memory_order_relaxed);
    }
    if (const char* m = getenv("CIPHER_KV_RDR_V3")) {
        if (m[0] != '0' && strcmp(m, "off") != 0)
            g_v3_mode.store(1, std::memory_order_relaxed);
    }
    if (on) {
        fprintf(stderr, "[CIPHER KV-RDR] init enabled v2_mode=%d v3_mode=%d max_layers=%d buf_bytes=%zu\n",
                g_v2_mode.load(), g_v3_mode.load(), MAX_LAYERS, BUF_BYTES);
    }
    return 1;
}

extern "C" void cipher_kv_redirect_reset(void) {
    g_kv_gemm_count.store(0, std::memory_order_relaxed);
    for (int i = 0; i < MAX_LAYERS; ++i) g_layer_pos[i] = 0;
}

extern "C" int cipher_kv_redirect_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_kv_redirect_on_fa_launch(void* args0, int layer_idx,
                                                 void* stream) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!args0) return 0;
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return 0;

    g_fa_seen.fetch_add(1, std::memory_order_relaxed);

    if (!ensure_buffers()) return 0;

    uintptr_t* st  = (uintptr_t*)args0;
    void* orig_K   = (void*)st[1];
    void* orig_V   = (void*)st[2];
    void* my_K     = g_buf[layer_idx].k_buf;
    void* my_V     = g_buf[layer_idx].v_buf;

    if (!orig_K || !orig_V) return 0;

    cudaStream_t s = (cudaStream_t)stream;
    int v3 = g_v3_mode.load(std::memory_order_relaxed);
    int v2 = g_v2_mode.load(std::memory_order_relaxed);

    if (v3) {
        // V3: dequant from per-layer compressed cache directly into my_K/V.
        int cur_pos = g_layer_pos[layer_idx];
        if (cur_pos > MAX_TOKENS) cur_pos = MAX_TOKENS;

        // V3 sub-mode: bytes beyond `cur_pos*16KB` come from the original
        // materialize-filled staging (so we can isolate "RoPE skip" vs "stale
        // tail"). CIPHER_KV_RDR_V3_FILL=fallback prepends a full memcpy first;
        // the dequant then overwrites the active prefix.
        static int fill_mode = -1;
        if (fill_mode < 0) {
            const char* m = getenv("CIPHER_KV_RDR_V3_FILL");
            fill_mode = (m && strcmp(m, "fallback") == 0) ? 1 : 0;
        }
        if (fill_mode == 1) {
            cudaMemcpyAsync(my_K, orig_K, COPY_BYTES,
                             cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(my_V, orig_V, COPY_BYTES,
                             cudaMemcpyDeviceToDevice, s);
        }
        if (cur_pos == 0 && fill_mode == 0) {
            if (cudaMemcpyAsync(my_K, orig_K, COPY_BYTES,
                                  cudaMemcpyDeviceToDevice, s) != cudaSuccess)
                return 0;
            if (cudaMemcpyAsync(my_V, orig_V, COPY_BYTES,
                                  cudaMemcpyDeviceToDevice, s) != cudaSuccess)
                return 0;
        } else if (cur_pos > 0) {
            static int skip_dq = -1;
            if (skip_dq < 0) skip_dq = getenv("CIPHER_KV_RDR_V3_SKIP_DEQ") ? 1 : 0;
            if (!skip_dq) {
                // Permuted dequant: cache stored in (token, head, dim) order
                // (matching cuBLAS K_proj output layout); my_K must be in the
                // (head, token, dim) layout PyTorch SDPA hands to FA.
                bool ok_k = dequant_buffer_perm(my_K, g_buf[layer_idx].cache_k_q,
                                                  g_buf[layer_idx].cache_k_meta,
                                                  KV_HEADS, cur_pos, s);
                bool ok_v = dequant_buffer_perm(my_V, g_buf[layer_idx].cache_v_q,
                                                  g_buf[layer_idx].cache_v_meta,
                                                  KV_HEADS, cur_pos, s);
                if (!ok_k || !ok_v) return 0;
            }
        }
    } else if (v2 == 0) {
        cudaError_t e1 = cudaMemcpyAsync(my_K, orig_K, COPY_BYTES,
                                           cudaMemcpyDeviceToDevice, s);
        cudaError_t e2 = cudaMemcpyAsync(my_V, orig_V, COPY_BYTES,
                                           cudaMemcpyDeviceToDevice, s);
        if (e1 != cudaSuccess || e2 != cudaSuccess) return 0;
    } else {
        int n_rows = (int)(COPY_BYTES / (KV_ROW * 2));
        if (!quant_dequant_buffer(orig_K, my_K, g_buf[layer_idx].k_q,
                                    g_buf[layer_idx].k_meta, n_rows, s))
            return 0;
        if (!quant_dequant_buffer(orig_V, my_V, g_buf[layer_idx].v_q,
                                    g_buf[layer_idx].v_meta, n_rows, s))
            return 0;
    }

    st[1] = (uintptr_t)my_K;
    st[2] = (uintptr_t)my_V;


    g_fa_rewritten.fetch_add(1, std::memory_order_relaxed);
    g_bytes_copied.fetch_add(2 * COPY_BYTES, std::memory_order_relaxed);
    return 1;
}

// V3: hook called from cublasGemmEx after K_proj/V_proj completes. Quantizes
// the n new tokens into per-layer cache slot at the layer's current write
// position. Layer + K-vs-V are inferred from a global counter.
extern "C" int cipher_kv_redirect_quant_kv(void* C_ptr, int m, int n, void* stream)
{
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!g_v3_mode.load(std::memory_order_relaxed)) return 0;
    if (m != KV_HEADS * KV_ROW) return 0;          // not K/V proj for Mistral
    if (n <= 0 || n > MAX_TOKENS) return 0;
    if (!ensure_buffers()) return 0;

    int idx     = g_kv_gemm_count.fetch_add(1, std::memory_order_relaxed);
    int layer   = (idx / 2) % MAX_LAYERS;
    int kv_idx  = idx & 1;                          // 0=K, 1=V

    int pos = g_layer_pos[layer];
    if (pos + n > MAX_TOKENS) return 0;

    // Cache slot pointers, offset to current write position (per-row offset).
    void* cache_q   = (kv_idx == 0) ? g_buf[layer].cache_k_q
                                     : g_buf[layer].cache_v_q;
    void* cache_md  = (kv_idx == 0) ? g_buf[layer].cache_k_meta
                                     : g_buf[layer].cache_v_meta;
    char* dst_q   = (char*)cache_q  + (size_t)KV_HEADS * pos * 32;
    char* dst_md  = (char*)cache_md + (size_t)KV_HEADS * pos * 4;

    int n_rows = KV_HEADS * n;
    if (!quant_buffer_only(C_ptr, dst_q, dst_md, n_rows, stream)) return 0;

    if (kv_idx == 1) g_layer_pos[layer] = pos + n;
    return 1;
}

extern "C" int cipher_kv_redirect_stats(CipherKvRedirectStats* out) {
    if (!out) return 0;
    out->enabled               = g_enabled.load(std::memory_order_relaxed);
    out->fa_launches_seen      = g_fa_seen.load(std::memory_order_relaxed);
    out->fa_launches_rewritten = g_fa_rewritten.load(std::memory_order_relaxed);
    out->bytes_copied          = g_bytes_copied.load(std::memory_order_relaxed);
    return 1;
}

__attribute__((constructor(112)))
static void cipher_kv_redirect_autoinit() { cipher_kv_redirect_init(); }
