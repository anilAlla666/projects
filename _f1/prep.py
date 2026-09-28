#!/usr/bin/env python3
# F1 repro prep: dump Mistral-7B layer-0 q_proj weight (FP16) + a fixed
# random activation to raw binary, for the standalone C++ Marlin test.
import json, os, struct
import torch
from safetensors import safe_open

MODEL = "/home/ubuntu/models/Mistral-7B-v0.1"
OUT   = "/home/ubuntu/_f1"
TENSOR = "model.layers.0.self_attn.q_proj.weight"
os.makedirs(OUT, exist_ok=True)

idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))
shard = idx["weight_map"][TENSOR]
with safe_open(f"{MODEL}/{shard}", framework="pt", device="cpu") as f:
    w = f.get_tensor(TENSOR)            # (out=N, in=K) row-major, bf16/fp16
print(f"{TENSOR}: shape={tuple(w.shape)} dtype={w.dtype}")
N, K = w.shape
w16 = w.to(torch.float16).contiguous()
w16.numpy().tofile(f"{OUT}/w0.fp16")   # N*K fp16, row-major (N,K)

torch.manual_seed(1234)
M = 1
a = torch.randn(M, K, dtype=torch.float16) * 0.1   # realistic activation scale
a.numpy().tofile(f"{OUT}/a.fp16")      # M*K fp16, row-major (M,K)

# fp32 host-reference output C[m,n] = sum_k A[m,k]*W[n,k]
cref = (a.float() @ w16.float().t())   # (M,N)
cref.numpy().tofile(f"{OUT}/c_ref_fp32.bin")

meta = dict(M=M, N=int(N), K=int(K), G=128,
            w_sum=float(w16.float().sum()), a_sum=float(a.float().sum()),
            cref_first8=[round(float(v),5) for v in cref[0,:8]])
json.dump(meta, open(f"{OUT}/meta.json","w"), indent=1)
print("meta:", meta)
print("wrote w0.fp16, a.fp16, c_ref_fp32.bin, meta.json")
