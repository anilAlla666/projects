"""A6: deep Monarch tests.

  - Phase 1 on a KNOWN structured matrix (Kronecker product) — should fit at rank=1.
  - Phase 2 singular spectrum of the reshape for one Mistral weight (top-20).
  - Phase 3 per-iteration error trace (does Procrustes monotonically decrease?).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from monarch_phase1 import monarch_factorize, monarch_reconstruct, monarch_matvec
from monarch_phase3b import procrustes_rotation, baseline_err

print("=== A6.1: Monarch on KNOWN Kronecker-structured matrix ===\n", flush=True)
torch.manual_seed(0)
device = "cuda"

# Construct W = kron(A, B) where A is (p, p) = (64, 64) and B is (q_n, q_m) = (64, 64)
# In Monarch parameterization with our reshape:
#   F[i*p + k, j*q_m + l] = W[i*q_n + j, k*q_m + l]
# kron(A, B)[i*q_n + j, k*q_m + l] = A[i, k] * B[j, l]
# So F[i*p + k, j*q_m + l] = A[i, k] * B[j, l] — this is a rank-1 outer product in F
A = torch.randn(64, 64, device=device, dtype=torch.float64)
B = torch.randn(64, 64, device=device, dtype=torch.float64)
W_kron = torch.kron(A, B)            # shape (64*64, 64*64) = (4096, 4096)
print(f"  W = kron(A_{tuple(A.shape)}, B_{tuple(B.shape)}) → shape {tuple(W_kron.shape)}", flush=True)

p, q_n, q_m = 64, 64, 64
for R in [1, 2, 4]:
    Af, Bf = monarch_factorize(W_kron, p, q_n, q_m, R)
    W_rec = monarch_reconstruct(Af, Bf, p, q_n, q_m)
    err = ((W_kron - W_rec).norm() / W_kron.norm()).item()
    x = torch.randn(p * q_m, device=device, dtype=torch.float64)
    y_full = W_kron @ x
    y_mono = monarch_matvec(Af, Bf, x, p, q_n, q_m)
    mv_err = ((y_full - y_mono).norm() / y_full.norm()).item()
    print(f"  R={R}: reconstruct_err={err:.2e}  matvec_err={mv_err:.2e}  (should be ~0 at R=1)", flush=True)

print("\n=== A6.2: singular spectrum of reshape(F) for layer 0 q_proj ===\n", flush=True)
from transformers import AutoModelForCausalLM
MODEL = "mistralai/Mistral-7B-v0.1"
print(f"  loading {MODEL}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
model.train(False)
W = model.model.layers[0].self_attn.q_proj.weight.detach().to(torch.float32)
print(f"  q_proj shape: {tuple(W.shape)}", flush=True)
n, m = W.shape

for blocking in [32, 64, 128]:
    p = blocking
    if n % p != 0 or m % p != 0:
        continue
    q_n = n // p; q_m = m // p
    F = W.reshape(p, q_n, p, q_m).permute(0, 2, 1, 3).reshape(p*p, q_n*q_m)
    U, S, Vh = torch.linalg.svd(F, full_matrices=False)
    var = (S ** 2)
    cum = torch.cumsum(var, dim=0) / var.sum()
    print(f"  blocking p={p:>3}, F shape={tuple(F.shape)}", flush=True)
    print(f"    top-20 singular values:", flush=True)
    print(f"    {[f'{v:.3f}' for v in S[:20].tolist()]}", flush=True)
    print(f"    cum-var at k=1, 4, 16, 64, 256, 1024:", flush=True)
    ks = [1, 4, 16, 64, 256, 1024]
    for k in ks:
        if k <= len(cum):
            print(f"      k={k:>5}: {cum[k-1].item():.6f}", flush=True)
    print(f"    rank-deficient ratio (S[-1]/S[0]): {S[-1].item()/S[0].item():.6e}", flush=True)
    print(flush=True)

print("=== A6.3: Procrustes per-iteration error trace ===\n", flush=True)
print("  rank=4, p=64, layer 15 q_proj:", flush=True)
W15 = model.model.layers[15].self_attn.q_proj.weight.detach().to(torch.float32)
Q, side, err, hist = procrustes_rotation(W15, 64, 4, n_iter=10)
print(f"  side='{side}'", flush=True)
print(f"  baseline (no rotation, R=4): {baseline_err(W15, 64, 64, 64, 4):.6f}", flush=True)
print(f"  Procrustes error per iter: " + " → ".join(f"{e:.4f}" for e in hist), flush=True)
print(f"  best_err={err:.4f}", flush=True)
print(f"  monotonically decreasing? {'YES' if all(hist[i+1] <= hist[i] for i in range(len(hist)-1)) else 'NO'}", flush=True)
