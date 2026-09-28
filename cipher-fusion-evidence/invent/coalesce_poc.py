# MFU PRODUCT PoC: do N isolated batch-1 decode "tenants" contend for bandwidth (aggregate ~1x),
# while COALESCING them into one batch-N GEMM gives ~Nx the useful work (MFU)?
# Real Mistral-7B linear weights, fp16, the decode GEMV sequence (224 GEMMs/step). Substrate NOT loaded.
import torch, glob, time, json
from safetensors.torch import load_file
DEV="cuda"; N_PARAMS=7.24e9; PEAK=989.5e12
MIST=glob.glob("/home/ubuntu/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/*/")[0]
sd={}
for s in sorted(glob.glob(MIST+"*.safetensors")): sd.update(load_file(s,device="cpu"))
L=32; W=[]
for i in range(L):
    p=f"model.layers.{i}."
    W.append([sd[p+n].to(DEV,torch.float16) for n in
              ["self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight",
               "mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"]])
del sd; torch.cuda.synchronize()
H,I=4096,14336
def decode_step(B, stream=None):
    # one decode step at batch B: run the 7 linear GEMMs x 32 layers (the weight-streaming work)
    ctx = torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.current_stream())
    with ctx:
        x=torch.randn(B,H,device=DEV,dtype=torch.float16)
        for q,k,v,o,g,u,d in W:
            _=x@q.t(); _=x@k.t(); _=x@v.t(); _=x@o.t()
            gi=torch.nn.functional.silu(x@g.t())*(x@u.t())
            x=gi@d.t()
    return
def bench_steps(fn, steps=30, warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(steps): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/steps
def mfu(tok_s): return 2*N_PARAMS*tok_s/PEAK

# 1) single batch-1 (one tenant)
t1=bench_steps(lambda: decode_step(1)); tps1=1/t1
print(f"single batch-1 tenant     : {tps1:6.0f} tok/s   MFU={100*mfu(tps1):.2f}%   ({t1*1000:.2f} ms/step)")
# 2) N isolated batch-1 tenants on N concurrent streams (real bandwidth contention)
for Ntenant in [8,16,32]:
    streams=[torch.cuda.Stream() for _ in range(Ntenant)]
    def isolated():
        for s in streams: decode_step(1, s)
    tN=bench_steps(isolated); agg=Ntenant/tN
    # 3) coalesced: one batch-Ntenant step (weights streamed ONCE)
    tC=bench_steps(lambda B=Ntenant: decode_step(B)); coal=Ntenant/tC
    print(f"  {Ntenant:>2} ISOLATED batch-1 (concurrent): {agg:6.0f} tok/s agg  MFU={100*mfu(agg):.2f}%   <- bandwidth-capped?")
    print(f"  {Ntenant:>2} COALESCED -> batch-{Ntenant:<2}        : {coal:6.0f} tok/s      MFU={100*mfu(coal):.2f}%   = {coal/agg:.1f}x the isolated aggregate")
