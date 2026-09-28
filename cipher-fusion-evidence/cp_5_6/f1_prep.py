"""CP 5.6 / F1 — minimal-repro input prep.

Extracts Mistral-7B layer-0 q_proj weight (BF16 -> FP16), builds a fixed
M=1 decode-shaped input A, and produces two references:
  f1_Yref.bin    — cuBLAS-FP16 GEMM (torch.F.linear, fp16, fp32-accum) = TRUTH
  f1_Yint4.bin   — A @ dequant(int4 g128 sym) W^T = the *expected* INT4 floor

Layout note: q_proj.weight on disk is PyTorch (out=N, in=K) row-major =
[4096,4096]. The libcipher_rt engine's quantize_repack expects exactly that
and transposes internally. So f1_W.bin is the verbatim weight; K=N=4096.
"""
import json, struct, numpy as np, torch

SHARD = 'models/Mistral-7B-v0.1/model-00001-of-00002.safetensors'
KEY   = 'model.layers.0.self_attn.q_proj.weight'
G     = 128

with open(SHARD, 'rb') as fh:
    n = struct.unpack('<Q', fh.read(8))[0]
    hdr = json.loads(fh.read(n)); base = 8 + n
    o0, o1 = hdr[KEY]['data_offsets']
    fh.seek(base + o0); raw = fh.read(o1 - o0)

W = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(4096, 4096).to(torch.float16)
K = W.shape[1]; N = W.shape[0]                       # in-features, out-features
assert (K, N) == (4096, 4096)

torch.manual_seed(42)
A = torch.randn(1, K, dtype=torch.float16)           # M=1 decode-shaped activation

# --- reference 1: cuBLAS-FP16 GEMM (the truth oracle) ---
Wg, Ag = W.cuda(), A.cuda()
Yref = torch.nn.functional.linear(Ag, Wg).float().cpu()   # [1,N]

# --- reference 2: expected INT4 floor (engine's scheme: per-128-K-group, absmax/7) ---
Wf = W.float()
Wg128 = Wf.view(N, K // G, G)
absmax = Wg128.abs().amax(dim=2, keepdim=True)
scale  = (absmax / 7.0).clamp_min(1e-8)
q = torch.clamp(torch.round(Wg128 / scale), -8, 7)
Wdq = (q * scale).view(N, K)
Yint4 = torch.nn.functional.linear(A.float(), Wdq).cpu()  # [1,N]

W.cpu().numpy().astype(np.float16).tofile('cipher-fusion-evidence/cp_5_6/f1_W.bin')
A.cpu().numpy().astype(np.float16).tofile('cipher-fusion-evidence/cp_5_6/f1_A.bin')
Yref.numpy().astype(np.float32).tofile('cipher-fusion-evidence/cp_5_6/f1_Yref.bin')
Yint4.numpy().astype(np.float32).tofile('cipher-fusion-evidence/cp_5_6/f1_Yint4.bin')

rel = lambda y: float(((y - Yref) ** 2).sum() ** 0.5 / ((Yref ** 2).sum() ** 0.5 + 1e-12))
print(f'K={K} N={N} M=1 G={G}')
print(f'Yref   first6: {[round(float(v),3) for v in Yref[0,:6]]}')
print(f'Yint4  first6: {[round(float(v),3) for v in Yint4[0,:6]]}')
print(f'INT4-floor rel_err vs cuBLAS-FP16 = {rel(Yint4):.4f}  (this is what "correct" looks like)')
