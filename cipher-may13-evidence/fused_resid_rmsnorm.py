"""TASK A: Fused residual-add + RMSNorm kernel.

Math (per row of D=4096 fp16):
    x_post[i]  = residual[i] + output[i]                           (saved for next residual)
    rms        = sqrt(mean(x_post^2) + eps)
    x_norm[i]  = x_post[i] * gamma[i] / rms                        (output to next op)

One kernel launch instead of two:
    Before: (residual + output) → write x_post → read x_post → rms_norm → write x_norm
    After:  one launch, x_post stays in registers/shared

Used twice per layer:
  1. After attention:  x_post = residual_pre_attn + attn_out;  x_norm = post_attn_layernorm(x_post)
  2. Top of MLP:       (same pattern) — but Mistral's structure has post_attn_ln OUTSIDE the residual,
                       so this fusion replaces "x = res+a; x_n = ln(x)" with one kernel.
"""
import os, ctypes
os.environ.setdefault("CIPHER_SUBSTITUTE_V2", "on")

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_substitute_v2_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
rt.cipher_substitute_v2_compile.restype = ctypes.c_ulong
rt.cipher_substitute_v2_get_function.argtypes = [ctypes.c_ulong]
rt.cipher_substitute_v2_get_function.restype = ctypes.c_void_p
rt.cipher_substitute_v2_init()


KERNEL_SRC = r"""
#include <cuda_fp16.h>

// One block per (batch * seq) row of D elements. D must be multiple of blockDim.x.
// Each thread handles D/blockDim.x consecutive elements.
extern "C" __global__ void cipher_fused_resid_rmsnorm(
    __half*       __restrict__ x_norm_out,    // [B*S, D] - normalized output
    __half*       __restrict__ x_post_out,    // [B*S, D] - unnormalized x = residual+output (for next residual)
    const __half* __restrict__ residual,      // [B*S, D]
    const __half* __restrict__ output,        // [B*S, D]
    const __half* __restrict__ gamma,         // [D]
    int D,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int N = blockDim.x;             // threads per block, e.g. 256
    int per_thread = D / N;         // e.g. 4096/256 = 16

    const __half* res_row = residual + row * D;
    const __half* out_row = output   + row * D;
    __half* xn_row = x_norm_out + row * D;
    __half* xp_row = x_post_out + row * D;

    // Pass 1: compute x = residual + output, accumulate sum of squares
    float local_sumsq = 0.0f;
    float xs[32];   // up to 32 elements per thread
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (i >= per_thread) break;
        int idx = i * N + tid;          // strided across threads for coalesced reads
        if (idx < D) {
            float r = __half2float(res_row[idx]);
            float o = __half2float(out_row[idx]);
            float x = r + o;
            xs[i] = x;
            xp_row[idx] = __float2half(x);   // save unnormalized
            local_sumsq += x * x;
        }
    }

    // Block reduce sum of squares
    __shared__ float shared_sumsq[32];   // up to 32 warps
    int lane = tid & 31;
    int warp = tid >> 5;
    // warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        local_sumsq += __shfl_xor_sync(0xffffffff, local_sumsq, off);
    }
    if (lane == 0) shared_sumsq[warp] = local_sumsq;
    __syncthreads();
    // first warp reduces across warps
    if (warp == 0) {
        local_sumsq = (tid < (N + 31) / 32) ? shared_sumsq[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            local_sumsq += __shfl_xor_sync(0xffffffff, local_sumsq, off);
        }
        if (lane == 0) shared_sumsq[0] = local_sumsq;
    }
    __syncthreads();
    float total_sumsq = shared_sumsq[0];
    float inv_rms = rsqrtf(total_sumsq / (float)D + eps);

    // Pass 2: write normalized output
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (i >= per_thread) break;
        int idx = i * N + tid;
        if (idx < D) {
            float g = __half2float(gamma[idx]);
            float result = xs[i] * inv_rms * g;
            xn_row[idx] = __float2half(result);
        }
    }
}
"""

_cubin = rt.cipher_substitute_v2_compile(KERNEL_SRC.encode(), b"cipher_fused_resid_rmsnorm")
assert _cubin != 0, "compile failed"
_kfn = rt.cipher_substitute_v2_get_function(_cubin)
assert _kfn

_libcuda = ctypes.CDLL("libcuda.so.1")
_cu_launch = _libcuda.cuLaunchKernel
_cu_launch.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
]
_cu_launch.restype = ctypes.c_int


def fused_resid_rmsnorm(residual, output, gamma, eps=1e-5,
                        x_norm_out=None, x_post_out=None):
    """Fused residual + rmsnorm. Returns (x_norm, x_post)."""
    assert residual.shape == output.shape
    *batch, D = residual.shape
    BS = 1
    for b in batch: BS *= b
    flat_res = residual.reshape(BS, D).contiguous()
    flat_out = output.reshape(BS, D).contiguous()
    if x_norm_out is None:
        x_norm_out = torch.empty_like(residual)
    if x_post_out is None:
        x_post_out = torch.empty_like(residual)
    flat_xn = x_norm_out.reshape(BS, D)
    flat_xp = x_post_out.reshape(BS, D)
    block = 256
    args = [
        ctypes.c_void_p(flat_xn.data_ptr()),
        ctypes.c_void_p(flat_xp.data_ptr()),
        ctypes.c_void_p(flat_res.data_ptr()),
        ctypes.c_void_p(flat_out.data_ptr()),
        ctypes.c_void_p(gamma.data_ptr()),
        ctypes.c_int(D),
        ctypes.c_float(eps),
    ]
    arr = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = _cu_launch(_kfn, BS, 1, 1, block, 1, 1, 0, stream, arr, None)
    if rc != 0:
        raise RuntimeError(f"fused_resid_rmsnorm launch failed rc={rc}")
    return x_norm_out, x_post_out


# ===== Reference =====
def ref_resid_rmsnorm(residual, output, gamma, eps=1e-5):
    x_post = residual + output
    var = x_post.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_post * torch.rsqrt(var + eps)
    x_norm = x_norm.to(gamma.dtype) * gamma
    return x_norm, x_post


if __name__ == "__main__":
    print(f"=== Task A: fused residual+RMSNorm ===\n", flush=True)
    print(f"[compile] cubin_id = {_cubin}\n", flush=True)

    # Correctness
    print(f"--- Correctness ---", flush=True)
    print(f"  {'B':>3} {'S':>4} {'D':>5}  {'norm_max_err':>13} {'post_max_err':>13}  {'status':<6}", flush=True)
    for B, S, D in [(1, 1, 4096), (1, 17, 4096), (8, 1, 4096), (32, 1, 4096)]:
        torch.manual_seed(0)
        residual = torch.randn(B, S, D, device="cuda", dtype=torch.float16) * 0.5
        output   = torch.randn(B, S, D, device="cuda", dtype=torch.float16) * 0.5
        gamma    = torch.randn(D, device="cuda", dtype=torch.float16) * 0.5 + 1.0
        ref_norm, ref_post = ref_resid_rmsnorm(residual, output, gamma)
        f_norm, f_post = fused_resid_rmsnorm(residual, output, gamma)
        torch.cuda.synchronize()
        norm_err = (ref_norm - f_norm).abs().max().item()
        post_err = (ref_post - f_post).abs().max().item()
        ok = "PASS" if (norm_err < 5e-3 and post_err < 1e-3) else "FAIL"
        print(f"  {B:>3} {S:>4} {D:>5}  {norm_err:>13.6f} {post_err:>13.6f}  {ok:<6}", flush=True)

    # Speed: fused vs separate (residual add + rmsnorm)
    print(f"\n--- Speed (median of 1000 iters) ---", flush=True)
    print(f"  {'B':>3} {'S':>4} {'D':>5}  {'separate_us':>12} {'fused_us':>10}  {'speedup':>8}", flush=True)
    for B, S, D in [(1, 1, 4096), (8, 1, 4096), (32, 1, 4096)]:
        residual = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        output   = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        gamma    = torch.randn(D, device="cuda", dtype=torch.float16) + 1.0
        x_norm_out = torch.empty_like(residual); x_post_out = torch.empty_like(residual)
        # warmup
        for _ in range(20):
            _ = ref_resid_rmsnorm(residual, output, gamma)
            _ = fused_resid_rmsnorm(residual, output, gamma, 1e-5, x_norm_out, x_post_out)
        torch.cuda.synchronize()

        N = 1000
        t_sep = []
        for _ in range(N):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = ref_resid_rmsnorm(residual, output, gamma)
            e.record()
            torch.cuda.synchronize()
            t_sep.append(s.elapsed_time(e) * 1000.0)
        t_f = []
        for _ in range(N):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = fused_resid_rmsnorm(residual, output, gamma, 1e-5, x_norm_out, x_post_out)
            e.record()
            torch.cuda.synchronize()
            t_f.append(s.elapsed_time(e) * 1000.0)
        t_sep.sort(); t_f.sort()
        m_sep = t_sep[N // 2]; m_f = t_f[N // 2]
        sp = m_sep / m_f if m_f > 0 else 0
        print(f"  {B:>3} {S:>4} {D:>5}  {m_sep:>12.1f} {m_f:>10.1f}  {sp:>7.2f}x", flush=True)
