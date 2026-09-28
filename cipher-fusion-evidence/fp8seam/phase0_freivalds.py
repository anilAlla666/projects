# Phase 0 micro: Freivalds checksum for FP8 cutlass_scaled_mm. Settles (0b) cost — does it actually
# save for DECODE (M=1)? — and (0c/0d) the clean-noise tolerance + fault catch rate / small-fault floor.
# Real op: out = scale_a*scale_b*(a_fp8.float() @ b_fp8.float()) -> bf16 (per-tensor scales).
# Freivalds: residual = | out@V  -  scale_a*scale_b*(a_fp8.float() @ (b_fp8.float()@V)) |, V in [N,R].
import torch, time, vllm._custom_ops
dev = "cuda"
_real = torch.ops._C.cutlass_scaled_mm
torch.manual_seed(0)

def mk(M, K, N):
    a = (torch.randn(M, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device=dev) * 0.1).to(torch.float8_e4m3fn).t()   # [K,N] col-major weight
    sa = torch.tensor(0.7, device=dev); sb = torch.tensor(1.3, device=dev)
    return a, b, sa, sb

def realmm(a, b, sa, sb):
    o = torch.empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16, device=dev)
    _real(o, a, b, sa, sb, None); return o

def ev(fn, it=1000):
    for _ in range(30): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it

# ===== 0b COST (decode M=1, a heavy down/up-proj-ish K,N) =====
M, K, N = 1, 4096, 14336
a, b, sa, sb = mk(M, K, N)
af = a.float(); bf = b.float()
R = 4
V = torch.randn(N, R, device=dev)
bV_pre = (bf @ V)                                  # PRECOMPUTED b@V (fixed V), amortized once
def bare(): return realmm(a, b, sa, sb)
def fresh_freivalds():                             # fresh v each call -> b@v recomputed (full weight pass)
    o = realmm(a, b, sa, sb); v = torch.randn(N, 1, device=dev)
    bv = bf @ v; av = (sa*sb)*(af @ bv); ov = o.float() @ v; return (ov-av).abs().max()
def precomp_freivalds():                           # fixed V, b@V precomputed -> per-step = 2 tiny mat-vecs
    o = realmm(a, b, sa, sb)
    av = (sa*sb)*(af @ bV_pre); ov = o.float() @ V; return (ov-av).abs().max()
t_bare = ev(bare); t_fresh = ev(fresh_freivalds); t_pre = ev(precomp_freivalds)
print("P0-COST", {"shape":f"M{M}xK{K}xN{N}","bare_ms":round(t_bare,4),
    "fresh_freivalds_ms":round(t_fresh,4),"fresh_x":round(t_fresh/t_bare,2),
    "precomp_freivalds_ms":round(t_pre,4),"precomp_x":round(t_pre/t_bare,2),"R":R})

# ===== 0c CLEAN TOLERANCE (fp8 numerical noise floor) over many fresh operands =====
cleans = []
for _ in range(200):
    aa, bb, s1, s2 = mk(M, K, N); o = realmm(aa, bb, s1, s2)
    Vc = torch.randn(N, R, device=dev)
    av = (s1*s2)*(aa.float() @ (bb.float() @ Vc)); ov = o.float() @ Vc
    cleans.append((ov-av).abs().max().item())
ct = torch.tensor(cleans)
# scale-aware relative floor: residual ~ ||out@V|| * eps; report both abs and normalized
norm_cleans = []
for _ in range(50):
    aa, bb, s1, s2 = mk(M, K, N); o = realmm(aa, bb, s1, s2); Vc = torch.randn(N, R, device=dev)
    ov = o.float() @ Vc; av = (s1*s2)*(aa.float() @ (bb.float() @ Vc))
    norm_cleans.append(((ov-av).abs().max() / (ov.abs().max()+1e-6)).item())
nt = torch.tensor(norm_cleans)
print("P0-CLEAN", {"abs_mean":round(ct.mean().item(),5),"abs_max":round(ct.max().item(),5),
    "abs_p99":round(ct.quantile(0.99).item(),5),"rel_mean":round(nt.mean().item(),5),
    "rel_max":round(nt.max().item(),5)})
TOL = ct.max().item() * 3                           # principled: 3x the observed clean max (abs)
RELTOL = nt.max().item() * 3
print("P0-TOL", {"abs_tol(3x_cleanmax)":round(TOL,5),"rel_tol":round(RELTOL,5)})

# ===== 0d FAULT CATCH RATE + SMALL-FAULT FLOOR (fixed-V, precomputed, the viable mode) =====
def catch_rate(fault_fn, trials=300):
    hits = 0
    for _ in range(trials):
        aa, bb, s1, s2 = mk(M, K, N); o = realmm(aa, bb, s1, s2)
        bVp = bb.float() @ V
        o = fault_fn(o)                            # inject into the (already-computed) output
        ov = o.float() @ V; av = (s1*s2)*(aa.float() @ bVp)
        rel = ((ov-av).abs().max() / (ov.abs().max()+1e-6)).item()
        if rel > RELTOL: hits += 1
    return hits/trials
import math
faults = {
  "one_elem_flip_bigbit": lambda o:(o.view(-1).__setitem__(123,o.view(-1)[123]+ (o.abs().max()) ) or o),
  "one_elem_+0.5": lambda o:(o.view(-1).__setitem__(123,o.view(-1)[123]+0.5) or o),
  "one_elem_+0.05": lambda o:(o.view(-1).__setitem__(123,o.view(-1)[123]+0.05) or o),
  "row_scale_1.001": lambda o:(o.mul_(1.001) or o),
  "tiny_perturb_1e-3_allelem": lambda o:(o.add_(torch.randn_like(o)*1e-3) or o),
}
rates = {name: round(catch_rate(fn),3) for name,fn in faults.items()}
print("P0-CATCH", rates)

# ===== 0e CSE/folding guard: perturb a's operand for the av path -> residual must respond =====
aa, bb, s1, s2 = mk(M, K, N); o = realmm(aa, bb, s1, s2); bVp = bb.float() @ V
ov = o.float() @ V
av_clean = (s1*s2)*(aa.float() @ bVp)
av_pert  = (s1*s2)*((aa.float()*1.05) @ bVp)
print("P0-CSE", {"resid_clean":round((ov-av_clean).abs().max().item(),5),
    "resid_operand_perturbed":round((ov-av_pert).abs().max().item(),5),
    "responds_to_operand": (ov-av_pert).abs().max().item() > 10*(ov-av_clean).abs().max().item()})
