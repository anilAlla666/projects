#!/usr/bin/env python3
# STEP 1 repair+correctness: with CIPHER_SUBSTITUTE_V2=1 enabling the NVRTC path, do the may13
# fused kernels (a) return success (rc=1, i.e. actually launch) and (b) match a torch reference?
import os, ctypes, torch
import warnings; warnings.filterwarnings("ignore")
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL("/home/ubuntu/cipher-may13-evidence/libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
for fn,arg in [("cipher_substitute_v2_init",[]),("cipher_substitute_v2_enabled",[]),
               ("cipher_fusion_kernels_init",[]),("cipher_fusion_kernels_enabled",[]),
               ("cipher_fused_rmsnorm",[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p]),
               ("cipher_fused_silu_mul",[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p])]:
    f=getattr(rt,fn); f.restype=ctypes.c_int; f.argtypes=arg

print("CIPHER_SUBSTITUTE_V2=", os.environ.get("CIPHER_SUBSTITUTE_V2"), " CIPHER_FUSION=", os.environ.get("CIPHER_FUSION"))
print("substitute_v2_init ->", rt.cipher_substitute_v2_init(), " enabled ->", rt.cipher_substitute_v2_enabled())
print("fusion_kernels_init ->", rt.cipher_fusion_kernels_init(), " enabled ->", rt.cipher_fusion_kernels_enabled())
torch.manual_seed(0)
def sp(): return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

# RMSNorm: rows=8 (B=8 decode), hidden=4096 (Mistral)
R,H = 8,4096
x = (torch.randn(R,H,device="cuda",dtype=torch.float16)*0.5)
w = (torch.randn(H,device="cuda",dtype=torch.float16)*0.3+1.0)
eps = 1e-5
out = torch.empty_like(x)
rc = rt.cipher_fused_rmsnorm(x.data_ptr(), w.contiguous().data_ptr(), out.data_ptr(), R, H, float(eps), sp())
torch.cuda.synchronize()
# torch reference RMSNorm (Mistral: fp32 internal)
xf = x.float(); var = xf.pow(2).mean(-1,keepdim=True); ref = (xf*torch.rsqrt(var+eps)).to(torch.float16)*w
rms_rel = (out.float()-ref.float()).abs().max().item()/ref.float().abs().max().item()
print(f"RMSNORM   rc={rc}  max_rel_err={rms_rel:.3e}  PASS={rms_rel<1e-2 and rc==1}")

# SiLU*mul: numel = 8*14336
g = torch.randn(8,14336,device="cuda",dtype=torch.float16)*0.5
u = torch.randn(8,14336,device="cuda",dtype=torch.float16)*0.5
o = torch.empty_like(g)
rc2 = rt.cipher_fused_silu_mul(g.data_ptr(), u.data_ptr(), o.data_ptr(), g.numel(), sp())
torch.cuda.synchronize()
refsm = (torch.nn.functional.silu(g.float())*u.float()).to(torch.float16)
sm_rel = (o.float()-refsm.float()).abs().max().item()/refsm.float().abs().max().item()
print(f"SILU_MUL  rc={rc2}  max_rel_err={sm_rel:.3e}  PASS={sm_rel<1e-2 and rc2==1}")
print("STEP1_CORRECTNESS_PASS=", bool(rms_rel<1e-2 and sm_rel<1e-2 and rc==1 and rc2==1))
