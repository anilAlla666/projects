import os, ctypes
import torch
ROOT = "/home/ubuntu/op31-prod-fix"
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"))
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_weight_compress_init.restype = ctypes.c_int
rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
rt.cipher_weight_compress_observe.restype = ctypes.c_int
rt.cipher_weight_compress_quantize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
rt.cipher_weight_compress_quantize.restype = ctypes.c_int
rt.cipher_weight_compress_lookup.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
rt.cipher_weight_compress_lookup.restype = ctypes.c_int
rt.cipher_weight_compress_int4_gemm.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
rt.cipher_weight_compress_int4_gemm.restype = ctypes.c_int
rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()

import warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16, device_map="cuda")

# Pick a few different weight tensors
candidates = []
for name, p in m.named_parameters():
    if p.dim() == 2 and p.shape[1] % 128 == 0 and p.shape[0] >= 128:
        candidates.append((name, p))
        if len(candidates) >= 4:
            break

print(f"Testing {len(candidates)} real Mistral weight tensors:\n")
for name, W in candidates:
    Wt = W.detach().contiguous()
    # nn.Linear stores weight as (out, in) and computes X @ W.T
    # We'll quantize as-is and matmul as A @ Wt.T to match nn.Linear semantics
    # Actually our kernel computes A @ B with B[K,N], so we use Wt.T:
    # But Wt.T isn't contiguous. Let's just quantize Wt as (K, N) where K=out, N=in
    # and test C = A @ Wt where A is (M, K)=(M, out)? No that's not natural.
    # 
    # Cleaner: directly test the matmul our kernel does.
    # Quantize Wt as (K, N) = (out_features, in_features). Our kernel computes
    # C[m,n] = sum_k A[m,k] * Wt[k,n]. So A is shape (M, out_features) and
    # output is (M, in_features). That's not nn.Linear semantics but it tests
    # the kernel's correctness.
    K, N = Wt.shape[0], Wt.shape[1]
    if N % 128 != 0:
        continue

    for _ in range(1001):
        rt.cipher_weight_compress_observe(Wt.data_ptr(), K * N * 2)
    rc = rt.cipher_weight_compress_quantize(Wt.data_ptr(), K, N)
    if rc != 1:
        print(f"  {name:40s}  quantize FAILED")
        continue

    b4 = ctypes.c_void_p(); bs = ctypes.c_void_p(); br = ctypes.c_int(); bc = ctypes.c_int()
    rt.cipher_weight_compress_lookup(Wt.data_ptr(),
        ctypes.byref(b4), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))

    M = 4
    A = torch.randn(M, K, dtype=torch.float16, device='cuda') * 0.5  # realistic activation magnitude
    C_ref = A @ Wt
    C_int4 = torch.empty(M, N, dtype=torch.float16, device='cuda')
    rt.cipher_weight_compress_int4_gemm(
        A.data_ptr(), b4.value, bs.value, C_int4.data_ptr(), M, N, K, None)
    torch.cuda.synchronize()

    abs_err = (C_int4.float() - C_ref.float()).abs()
    rel_err = abs_err / (C_ref.float().abs() + 1e-6)
    print(f"  {name:40s}  shape={tuple(Wt.shape)}")
    print(f"    rel_err mean={rel_err.mean():.4f} median={rel_err.median():.4f} p99={rel_err.flatten().kthvalue(int(0.99*rel_err.numel())).values.item():.4f}")
    print(f"    abs_err mean={abs_err.mean():.4f}  C_ref max_abs={C_ref.abs().max():.3f}  C_int4 max_abs={C_int4.abs().max():.3f}")
