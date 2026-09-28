#!/usr/bin/env python3
# Coverage spot-check INSIDE a real Mistral linear: clean -> 0 false positive; one Step-A-class
# bit-14 flip in the output -> caught. fp32 accumulation. (Full 200/200 coverage settled prior;
# this only confirms the check still flags when run on a real layer's real activations.)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0); dev='cuda'
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).cuda().eval()
# grab a real layer's gate_proj weight + a real activation (hidden states into the MLP of layer 0)
layer=m.model.layers[0]
ids=tok("The quick brown fox jumps over the lazy dog. SDC detection in transformers.",return_tensors="pt").input_ids.cuda()
acts={}
h=layer.mlp.gate_proj.register_forward_hook(lambda mod,inp,out: acts.__setitem__('A',inp[0].detach()))
with torch.no_grad(): m(ids)
h.remove()
A=acts['A'].reshape(-1,4096)              # [M,4096] real activations
Wt=layer.mlp.gate_proj.weight.detach()    # [14336,4096]
M,K=A.shape; N=Wt.shape[0]
C=torch.nn.functional.linear(A,Wt)        # [M,14336] fp16 output
g=torch.randn(N,device=dev)               # random probe
u=(Wt.float().t()@g)                       # [K] fp32  (W^T g)
def freivalds_resid(Cmat):
    v=A.float()@u; cg=Cmat.float()@g; return (v-cg).abs()
# threshold from clean (max residual over rows) -> 0 clean FP by construction
T=freivalds_resid(C).max().item()*1.0
clean_max=freivalds_resid(C).max().item()
# inject one Step-A harmful flip: bit-14 (top exponent) of one output element
Cbad=C.clone()
mrow,nrow=M//2, N//2
orig=Cbad[mrow,nrow].clone()
bits=Cbad[mrow,nrow].view(torch.int16); bits^=(1<<14); Cbad[mrow,nrow]=bits.view(torch.float16)
delta=(Cbad[mrow,nrow].float()-orig.float()).abs().item()
resid_bad=freivalds_resid(Cbad)
caught = resid_bad[mrow].item() > T or torch.isnan(resid_bad).any().item() or torch.isinf(resid_bad).any().item()
clean_fp = (freivalds_resid(C) > T).sum().item()
print(f"real layer0 gate_proj  M={M} K={K} N={N}")
print(f"  clean max |v-Cg| = {clean_max:.4e}   T = {T:.4e}   clean false positives (>T): {clean_fp}/{M}")
print(f"  injected bit-14 flip at C[{mrow},{nrow}]: {orig.item():.4f} -> {Cbad[mrow,nrow].item():.4f}  (delta={delta:.3g})")
print(f"  Freivalds residual at corrupted row = {resid_bad[mrow].item():.4e}   signal/T margin = {resid_bad[mrow].item()/T:.1f}x")
print(f"  CAUGHT = {caught}   (fp32 accum, 0 clean FP)")
import json; json.dump(dict(M=M,K=K,N=N,T=T,clean_max=clean_max,clean_fp=clean_fp,
    delta=delta,resid_corrupted=resid_bad[mrow].item(),margin=resid_bad[mrow].item()/T,caught=bool(caught)),
    open('coverage_spotcheck.json','w'),indent=1)
