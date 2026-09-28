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

constexpr int    MAX_LAYERS    = 32;        // Mistral 7B has 32 (was 64)
constexpr size_t BUF_BYTES     = 256ull << 20;// 256 MB per buffer — covers
                                              // FA K/V staging at B=64,
                                              // P=1024 (128 MB per K/V).
                                              // (was 32 MB, was 1 MB)

// V3: per-layer persistent compressed cache. ROW=128 = head_dim for Mistral.
// Cache laid out as [layer][batch][head][max_tokens][32 bytes].
//   batch dim  = g_batch_max     (env CIPHER_KV_BATCH, default 1)
//   max_tokens = g_max_tokens    (env CIPHER_KV_MAX_CACHE_LEN, default 4096)
// At B=8 P=2048 D=320 we need max_tokens >= 2368.
constexpr int    KV_HEADS      = 8;           // Mistral GQA num_kv_heads
int              g_max_tokens  = 4096;        // env CIPHER_KV_MAX_CACHE_LEN
int              g_batch_max   = 1;           // env CIPHER_KV_BATCH (cache dim)
// Legacy alias for old code paths that still use the (token,head)-flat
// layout — only used at g_batch_max == 1.
#define MAX_TOKENS_LEGACY (g_max_tokens)

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
unsigned long             g_quant_cubin            = 0;
unsigned long             g_dequant_cubin          = 0;
unsigned long             g_dequant_perm_cubin     = 0;
unsigned long             g_dequant_perm_rope_cubin = 0;
unsigned long             g_quant_b_cubin          = 0;
unsigned long             g_dequant_b_cubin        = 0;
unsigned long             g_dequant_b_rope_cubin   = 0;
std::mutex                g_kern_mu;
float                     g_rope_base = 10000.0f;   // Mistral-7B-v0.1 default

// V3 state: layer position tracker + KV GEMM call counter.
int                       g_layer_pos[MAX_LAYERS] = {0};
std::atomic<int>          g_kv_gemm_count{0};   // 0=L0 K, 1=L0 V, 2=L1 K, ...
// Number of model layers used to index the per-layer cache. The FA hook in
// cipher_intercept_cudart.cpp uses `fa_pre % 32` (Mistral-7B). quant_kv must
// match. Configurable via CIPHER_KV_NUM_LAYERS for other models.
int                       g_num_layers = 32;

// Running batch size — used by V3 to gate to B=1 only.  V3's existing
// quant/dequant layout is (token, head)-flat with no batch dimension, so
// running it at B>1 would scramble inter-batch attention.  The proper fix
// (per-(layer, batch) cache + B-aware dequant + bypassing PyTorch's
// fp16 materialize kernel) is a separate workstream.
//
// Set via CIPHER_KV_BATCH at process start.  Default 1.
int                       g_running_batch = 1;

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

// Permuted dequant with RoPE applied inline. Used for K only (V doesn't get
// RoPE). RoPE convention matches HuggingFace LlamaRotaryEmbedding /
// apply_rotary_pos_emb: for last dim D, pair (i, i+D/2) with theta_i =
// base^(-2i/D), at position p:
//   new[i]      = x[i]      * cos(p*theta_i) - x[i+D/2] * sin(p*theta_i)
//   new[i+D/2]  = x[i+D/2]  * cos(p*theta_i) + x[i]     * sin(p*theta_i)
// Position p == cache token index t (V3 cache stores tokens in RoPE-order).
// `pos_offset` is added to t in case the cache base differs (default 0).
extern "C" __global__ void cipher_kv_dq2_perm_rope(
    __half* __restrict__ out,              // [n_heads, n_tokens, ROW] fp16
    const unsigned char* __restrict__ q,   // [n_tokens, n_heads, ROW/4]
    const __half2* __restrict__ meta,      // [n_tokens, n_heads]
    int n_heads, int n_tokens,
    float rope_base,                       // 10000.0 for Mistral-7B-v0.1
    int   pos_offset)                      // RoPE position = t + pos_offset
{
    int row_out = blockIdx.x;
    int h = row_out / n_tokens;
    int t = row_out - h * n_tokens;
    int row_in = t * n_heads + h;
    int p = t + pos_offset;

    __half2 m = meta[row_in];
    float vmin  = __half2float(m.x);
    float scale = __half2float(m.y);

    // Pass 1: dequant the whole row to shared memory as fp32.
    __shared__ float row_buf[ROW];
    int tid = threadIdx.x;
    for (int j = tid * 4; j < ROW; j += blockDim.x * 4) {
        unsigned int b = q[row_in * (ROW/4) + (j >> 2)];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int q2 = (int)((b >> (k * 2)) & 0x3);
            row_buf[j + k] = vmin + (float)q2 * scale;
        }
    }
    __syncthreads();

    // Pass 2: apply RoPE on (i, i+D/2) pairs and store fp16.
    const int half_dim = ROW / 2;
    for (int i = tid; i < half_dim; i += blockDim.x) {
        float xa = row_buf[i];
        float xb = row_buf[i + half_dim];
        // theta_i = rope_base^(-2i/D); use exp/log to avoid powf in hot loop.
        float theta = expf(-(float)(2 * i) * logf(rope_base) / (float)ROW);
        float angle = (float)p * theta;
        float cs, sn;
        __sincosf(angle, &sn, &cs);
        float new_a = xa * cs - xb * sn;
        float new_b = xb * cs + xa * sn;
        out[row_out * ROW + i]            = __float2half(new_a);
        out[row_out * ROW + i + half_dim] = __float2half(new_b);
    }
}

// ─── B-aware variants ───────────────────────────────────────────────────
// Cache layout (per layer):
//     q[B][H][max_T][ROW/4]    flat index = ((b*H)+h)*max_T + t
//     meta[B][H][max_T]
// Quant input is K_proj output: row-major (B*S_per_call, H*ROW)
//   sub-row R = b*S_per_call*H + s*H + h    where s in [0, S_per_call)
//   maps to cache (b, h, pos+s)

extern "C" __global__ void cipher_kv_q2_b(
    const __half* __restrict__ in,       // (B*S*H, ROW) fp16 sub-rows
    unsigned char* __restrict__ q,       // (B, H, max_T, ROW/4)
    __half2* __restrict__ meta,          // (B, H, max_T)
    int B, int S, int H, int max_T, int pos)
{
    int row_in = blockIdx.x;
    if (row_in >= B * S * H) return;
    int b   = row_in / (S * H);
    int rem = row_in - b * (S * H);
    int s   = rem / H;
    int h   = rem - s * H;
    int t   = pos + s;
    if (t >= max_T) return;
    int row_out = (b * H + h) * max_T + t;

    const __half* r = in + row_in * ROW;

    // Pass 1: per-row min and max (warp reduce -> shared mem -> warp reduce).
    float vmin = 1e30f, vmax = -1e30f;
    int tid = threadIdx.x;
    for (int j = tid; j < ROW; j += blockDim.x) {
        float v = __half2float(r[j]);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    __shared__ float smin[32], smax[32];
    int lane = tid & 31, warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        float a = __shfl_down_sync(0xffffffff, vmin, off);
        float b2 = __shfl_down_sync(0xffffffff, vmax, off);
        if (a < vmin) vmin = a;
        if (b2 > vmax) vmax = b2;
    }
    if (lane == 0) { smin[warp] = vmin; smax[warp] = vmax; }
    __syncthreads();
    if (warp == 0) {
        vmin = (tid < (blockDim.x >> 5)) ? smin[lane] : 1e30f;
        vmax = (tid < (blockDim.x >> 5)) ? smax[lane] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            float a = __shfl_down_sync(0xffffffff, vmin, off);
            float b2 = __shfl_down_sync(0xffffffff, vmax, off);
            if (a < vmin) vmin = a;
            if (b2 > vmax) vmax = b2;
        }
        if (lane == 0) { smin[0] = vmin; smax[0] = vmax; }
    }
    __syncthreads();
    vmin = smin[0]; vmax = smax[0];
    float scale = (vmax - vmin) / 3.0f;
    float inv   = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;

    for (int j = tid * 4; j < ROW; j += blockDim.x * 4) {
        unsigned int packed = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float v = __half2float(r[j + k]);
            int   q2 = (int)__float2int_rn((v - vmin) * inv);
            if (q2 < 0) q2 = 0; else if (q2 > 3) q2 = 3;
            packed |= (q2 & 0x3) << (k * 2);
        }
        q[row_out * (ROW/4) + (j >> 2)] = (unsigned char)packed;
    }
    if (tid == 0) {
        meta[row_out] = __floats2half2_rn(vmin, scale);
    }
}

// B-aware dequant: produces FA-expected (B, H, n_tokens, ROW) output from
// the (B, H, max_T) cache.  No RoPE.
extern "C" __global__ void cipher_kv_dq2_b(
    __half* __restrict__ out,            // (B, H, n_tokens, ROW)
    const unsigned char* __restrict__ q, // (B, H, max_T, ROW/4)
    const __half2* __restrict__ meta,    // (B, H, max_T)
    int B, int H, int n_tokens, int max_T)
{
    int row_out = blockIdx.x;
    if (row_out >= B * H * n_tokens) return;
    int b   = row_out / (H * n_tokens);
    int rem = row_out - b * (H * n_tokens);
    int h   = rem / n_tokens;
    int t   = rem - h * n_tokens;
    int row_in = (b * H + h) * max_T + t;

    __half2 m = meta[row_in];
    float vmin = __half2float(m.x);
    float scale = __half2float(m.y);

    int tid = threadIdx.x;
    for (int j = tid * 4; j < ROW; j += blockDim.x * 4) {
        unsigned int packed = q[row_in * (ROW/4) + (j >> 2)];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int q2 = (int)((packed >> (k * 2)) & 0x3);
            float v = vmin + (float)q2 * scale;
            out[row_out * ROW + j + k] = __float2half(v);
        }
    }
}

// B-aware dequant with inline RoPE for K (V uses cipher_kv_dq2_b).
extern "C" __global__ void cipher_kv_dq2_b_rope(
    __half* __restrict__ out,            // (B, H, n_tokens, ROW)
    const unsigned char* __restrict__ q, // (B, H, max_T, ROW/4)
    const __half2* __restrict__ meta,    // (B, H, max_T)
    int B, int H, int n_tokens, int max_T,
    float rope_base, int pos_offset)
{
    int row_out = blockIdx.x;
    if (row_out >= B * H * n_tokens) return;
    int b   = row_out / (H * n_tokens);
    int rem = row_out - b * (H * n_tokens);
    int h   = rem / n_tokens;
    int t   = rem - h * n_tokens;
    int row_in = (b * H + h) * max_T + t;
    int p = t + pos_offset;

    __half2 m = meta[row_in];
    float vmin = __half2float(m.x);
    float scale = __half2float(m.y);

    __shared__ float row_buf[ROW];
    int tid = threadIdx.x;
    for (int j = tid * 4; j < ROW; j += blockDim.x * 4) {
        unsigned int packed = q[row_in * (ROW/4) + (j >> 2)];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int q2 = (int)((packed >> (k * 2)) & 0x3);
            row_buf[j + k] = vmin + (float)q2 * scale;
        }
    }
    __syncthreads();
    const int half_dim = ROW / 2;
    for (int i = tid; i < half_dim; i += blockDim.x) {
        float xa = row_buf[i];
        float xb = row_buf[i + half_dim];
        float theta = expf(-(float)(2 * i) * logf(rope_base) / (float)ROW);
        float angle = (float)p * theta;
        float cs, sn;
        __sincosf(angle, &sn, &cs);
        out[row_out * ROW + i]            = __float2half(xa * cs - xb * sn);
        out[row_out * ROW + i + half_dim] = __float2half(xb * cs + xa * sn);
    }
}
)NVRTC";

// Define the launcher resolver here so the B-aware launchers below can
// reach it (they used to be defined further down).
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

bool ensure_kv_kernels() {
    if (g_quant_cubin && g_dequant_cubin && g_dequant_perm_cubin
        && g_dequant_perm_rope_cubin
        && g_quant_b_cubin && g_dequant_b_cubin
        && g_dequant_b_rope_cubin) return true;
    std::lock_guard<std::mutex> lk(g_kern_mu);
    if (!cipher_substitute_v2_enabled()) return false;
    if (!g_quant_cubin)
        g_quant_cubin = cipher_substitute_v2_compile(kKvQuantSrc, "cipher_kv_q2");
    if (!g_dequant_cubin)
        g_dequant_cubin = cipher_substitute_v2_compile(kKvDequantSrc, "cipher_kv_dq2");
    if (!g_dequant_perm_cubin)
        g_dequant_perm_cubin = cipher_substitute_v2_compile(
            kKvDequantSrc, "cipher_kv_dq2_perm");
    if (!g_dequant_perm_rope_cubin)
        g_dequant_perm_rope_cubin = cipher_substitute_v2_compile(
            kKvDequantSrc, "cipher_kv_dq2_perm_rope");
    if (!g_quant_b_cubin)
        g_quant_b_cubin = cipher_substitute_v2_compile(
            kKvDequantSrc, "cipher_kv_q2_b");
    if (!g_dequant_b_cubin)
        g_dequant_b_cubin = cipher_substitute_v2_compile(
            kKvDequantSrc, "cipher_kv_dq2_b");
    if (!g_dequant_b_rope_cubin)
        g_dequant_b_rope_cubin = cipher_substitute_v2_compile(
            kKvDequantSrc, "cipher_kv_dq2_b_rope");
    return g_quant_cubin && g_dequant_cubin && g_dequant_perm_cubin
        && g_dequant_perm_rope_cubin
        && g_quant_b_cubin && g_dequant_b_cubin && g_dequant_b_rope_cubin;
}

// Launcher: B-aware quant.
bool quant_buffer_b(void* src_fp16, void* qbuf, void* mbuf,
                     int B, int S, int H, int max_T, int pos, void* stream)
{
    if (!ensure_kv_kernels() || !resolve_kv_launcher()) return false;
    void* fn = cipher_substitute_v2_get_function(g_quant_b_cubin);
    if (!fn) return false;
    void* args[] = {&src_fp16, &qbuf, &mbuf, &B, &S, &H, &max_T, &pos};
    int n_blocks = B * S * H;
    return g_kv_launcher(fn, n_blocks, 1, 1, 128, 1, 1, 0, stream, args, nullptr) == 0;
}

// Launcher: B-aware dequant (no RoPE, for V).
bool dequant_buffer_b(void* dst_fp16, void* qbuf, void* mbuf,
                       int B, int H, int n_tokens, int max_T, void* stream)
{
    if (!ensure_kv_kernels() || !resolve_kv_launcher()) return false;
    void* fn = cipher_substitute_v2_get_function(g_dequant_b_cubin);
    if (!fn) return false;
    void* args[] = {&dst_fp16, &qbuf, &mbuf, &B, &H, &n_tokens, &max_T};
    int n_blocks = B * H * n_tokens;
    return g_kv_launcher(fn, n_blocks, 1, 1, 128, 1, 1, 0, stream, args, nullptr) == 0;
}

// Launcher: B-aware dequant with inline RoPE (for K).
bool dequant_buffer_b_rope(void* dst_fp16, void* qbuf, void* mbuf,
                            int B, int H, int n_tokens, int max_T,
                            float rope_base, int pos_offset, void* stream)
{
    if (!ensure_kv_kernels() || !resolve_kv_launcher()) return false;
    void* fn = cipher_substitute_v2_get_function(g_dequant_b_rope_cubin);
    if (!fn) return false;
    void* args[] = {&dst_fp16, &qbuf, &mbuf, &B, &H, &n_tokens, &max_T,
                    &rope_base, &pos_offset};
    int n_blocks = B * H * n_tokens;
    return g_kv_launcher(fn, n_blocks, 1, 1, 128, 1, 1, 0, stream, args, nullptr) == 0;
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

// V3 K-path: dequant + RoPE in one kernel. RoPE position for cache index t
// is t + pos_offset.
bool dequant_buffer_perm_rope(void* dst_fp16, void* qbuf, void* mbuf,
                               int n_heads, int n_tokens, int pos_offset,
                               float rope_base, void* stream)
{
    if (!ensure_kv_kernels() || !resolve_kv_launcher()) return false;
    void* dfn = cipher_substitute_v2_get_function(g_dequant_perm_rope_cubin);
    if (!dfn) return false;
    int n_rows = n_heads * n_tokens;
    void* dargs[] = { &dst_fp16, &qbuf, &mbuf, &n_heads, &n_tokens,
                       &rope_base, &pos_offset };
    return g_kv_launcher(dfn, n_rows, 1, 1, 128, 1, 1, 0, stream, dargs, nullptr) == 0;
}

bool ensure_buffers() {
    if (g_buf[0].k_buf) return true;
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (g_buf[0].k_buf) return true;
    // V2 q/meta scratch (per layer — kept for the v2_mode path).
    const size_t qbuf_bytes = BUF_BYTES / 8;
    const size_t meta_rows  = BUF_BYTES / (KV_ROW * 2);
    const size_t meta_bytes = meta_rows * 4;
    // V3 persistent cache (per layer; sized for max batch & max tokens):
    const size_t cache_q_bytes  = (size_t)g_batch_max * KV_HEADS * g_max_tokens * 32;
    const size_t cache_md_bytes = (size_t)g_batch_max * KV_HEADS * g_max_tokens * 4;

    // FIX (a): ONE shared FA-staging buffer across all layers.  FA reads
    // one layer at a time in eager mode, so the staging can be re-used
    // (the prior per-layer 256 MB × 32 = 8 GB allocation was wasteful).
    // Sized to fit max batch × heads × max_tokens × head_dim × 2 bytes.
    static void* g_shared_k_buf = nullptr;
    static void* g_shared_v_buf = nullptr;
    size_t shared_bytes = (size_t)g_batch_max * KV_HEADS *
                           (size_t)g_max_tokens * KV_ROW * 2;  // fp16 = 2B
    if (shared_bytes < BUF_BYTES) shared_bytes = BUF_BYTES;
    if (cudaMalloc(&g_shared_k_buf, shared_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&g_shared_v_buf, shared_bytes) != cudaSuccess) return false;
    cudaMemset(g_shared_k_buf, 0, shared_bytes);
    cudaMemset(g_shared_v_buf, 0, shared_bytes);

    for (int i = 0; i < MAX_LAYERS; ++i) {
        // All layers point to the same staging.
        g_buf[i].k_buf = g_shared_k_buf;
        g_buf[i].v_buf = g_shared_v_buf;
        // V2 scratch is small (per-layer is fine and avoids contention if
        // we later move to multi-stream).
        if (cudaMalloc(&g_buf[i].k_q,    qbuf_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].v_q,    qbuf_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].k_meta, meta_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&g_buf[i].v_meta, meta_bytes) != cudaSuccess) return false;
        // V3 cache is per-layer (each layer has independent KV history).
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
    if (const char* m = getenv("CIPHER_KV_ROPE_BASE")) {
        double v = atof(m);
        if (v > 0) g_rope_base = (float)v;
    }
    if (const char* m = getenv("CIPHER_KV_NUM_LAYERS")) {
        long v = strtol(m, nullptr, 0);
        if (v > 0 && v <= MAX_LAYERS) g_num_layers = (int)v;
    }
    if (const char* m = getenv("CIPHER_KV_BATCH")) {
        long v = strtol(m, nullptr, 0);
        if (v >= 1) {
            g_running_batch = (int)v;
            // The cache must be sized for at least the running batch.
            if (g_batch_max < g_running_batch) g_batch_max = g_running_batch;
        }
    }
    if (const char* m = getenv("CIPHER_KV_MAX_CACHE_LEN")) {
        long v = strtol(m, nullptr, 0);
        if (v > 0 && v <= (1 << 20)) g_max_tokens = (int)v;
    }
    if (on) {
        fprintf(stderr, "[CIPHER KV-RDR] init enabled v2_mode=%d v3_mode=%d "
                "rope_base=%.0f max_layers=%d buf_bytes=%zu batch=%d\n",
                g_v2_mode.load(), g_v3_mode.load(), g_rope_base,
                MAX_LAYERS, BUF_BYTES, g_running_batch);
        if (g_v3_mode.load() && g_running_batch > 1) {
            fprintf(stderr, "[CIPHER KV-RDR] V3 gated OFF: B=%d > 1 — V3's "
                    "(token,head)-flat layout has no batch dimension; "
                    "falling through to original PyTorch FA staging.\n",
                    g_running_batch);
        }
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

    // V3: B-aware dequant for any batch size (B=1 included).  Cache is
    // laid out as (B, H, max_T, ROW/4) with corresponding (B, H, max_T)
    // meta.  Output is the FA-expected (B, H, n_tokens, ROW) layout.
    if (v3) {
        int cur_pos = g_layer_pos[layer_idx];
        if (cur_pos > g_max_tokens) cur_pos = g_max_tokens;
        if (cur_pos == 0) return 0;     // empty cache → fall through

        static int rope_mode = -1;
        if (rope_mode < 0) {
            const char* m = getenv("CIPHER_KV_RDR_V3_ROPE");
            rope_mode = (m && *m && strcmp(m, "0") != 0
                          && strcmp(m, "off") != 0) ? 1 : 0;
        }
        int B = g_running_batch < 1 ? 1 : g_running_batch;
        bool ok_k;
        if (rope_mode) {
            ok_k = dequant_buffer_b_rope(my_K,
                        g_buf[layer_idx].cache_k_q,
                        g_buf[layer_idx].cache_k_meta,
                        B, KV_HEADS, cur_pos, g_max_tokens,
                        g_rope_base, /*pos_offset=*/0, s);
        } else {
            ok_k = dequant_buffer_b(my_K,
                        g_buf[layer_idx].cache_k_q,
                        g_buf[layer_idx].cache_k_meta,
                        B, KV_HEADS, cur_pos, g_max_tokens, s);
        }
        bool ok_v = dequant_buffer_b(my_V,
                        g_buf[layer_idx].cache_v_q,
                        g_buf[layer_idx].cache_v_meta,
                        B, KV_HEADS, cur_pos, g_max_tokens, s);
        if (!ok_k || !ok_v) return 0;
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
    if (n <= 0) return 0;
    if (!ensure_buffers()) return 0;

    int B = g_running_batch;
    if (B < 1) B = 1;
    if (n % B != 0) return 0;                      // call must be a multiple of B
    int S_per_call = n / B;                         // 1 at decode, prefill_len at prefill

    int idx     = g_kv_gemm_count.fetch_add(1, std::memory_order_relaxed);
    int layer   = (idx / 2) % g_num_layers;        // must match FA hook (32 for Mistral-7B)
    int kv_idx  = idx & 1;                          // 0=K, 1=V

    int pos = g_layer_pos[layer];
    if (pos + S_per_call > g_max_tokens) return 0;

    void* cache_q   = (kv_idx == 0) ? g_buf[layer].cache_k_q
                                     : g_buf[layer].cache_v_q;
    void* cache_md  = (kv_idx == 0) ? g_buf[layer].cache_k_meta
                                     : g_buf[layer].cache_v_meta;

    // B-aware path (works for B==1 too; legacy path retained as fallback).
    if (B >= 1) {
        if (!quant_buffer_b(C_ptr, cache_q, cache_md,
                             B, S_per_call, KV_HEADS, g_max_tokens,
                             pos, stream))
            return 0;
        if (kv_idx == 1) g_layer_pos[layer] = pos + S_per_call;
        return 1;
    }
    return 0;
}

extern "C" int cipher_kv_test_qdq_rope(void* in_fp16, void* out_fp16,
                                         int n_heads, int n_tokens,
                                         int apply_rope, int pos_offset,
                                         float rope_base, void* stream) {
    if (!in_fp16 || !out_fp16) return 0;
    if (n_heads <= 0 || n_tokens <= 0) return 0;
    if (!ensure_kv_kernels() || !resolve_kv_launcher()) return 0;
    int n_rows = n_heads * n_tokens;
    size_t qb = (size_t)n_rows * (KV_ROW / 4);   // 32 bytes/row
    size_t mb = (size_t)n_rows * 4;              // 4 bytes/row (fp16x2)
    void* q  = nullptr;
    void* m  = nullptr;
    if (cudaMalloc(&q, qb) != cudaSuccess) return 0;
    if (cudaMalloc(&m, mb) != cudaSuccess) { cudaFree(q); return 0; }
    int rc = 0;
    if (!quant_buffer_only(in_fp16, q, m, n_rows, stream)) goto done;
    if (apply_rope) {
        if (!dequant_buffer_perm_rope(out_fp16, q, m,
                                       n_heads, n_tokens, pos_offset,
                                       rope_base, stream)) goto done;
    } else {
        if (!dequant_buffer_perm(out_fp16, q, m, n_heads, n_tokens, stream))
            goto done;
    }
    rc = 1;
done:
    cudaFree(q);
    cudaFree(m);
    return rc;
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
