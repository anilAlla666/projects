# Phase 0: frequency-reduction via mechanism (B) — move recompute OUT of the captured graph, run it
# host-side every Nth step on BUFFERED I/O. Unchecked-step cost = only cheap in-graph COPIES of the
# layer input x and output out into static module buffers; checked step = host-side eager recompute.
# Gating question: are the in-graph copies cheap (single-digit) on the fast decode GEMM? Also confirm
# host-side recompute from buffers detects. Plus a baseline for (A) device-skip and the always-on ref.
import torch, vllm._custom_ops
dev="cuda"; _real=torch.ops._C.cutlass_scaled_mm; torch.manual_seed(0)
M,K,N=1,4096,14336
a=(torch.randn(M,K,device=dev)*0.1).to(torch.float8_e4m3fn)
b=(torch.randn(N,K,device=dev)*0.1).to(torch.float8_e4m3fn).t()
sa=torch.tensor(0.7,device=dev); sb=torch.tensor(1.3,device=dev)
# static buffers we control (the (B) channel): copy x(as a here, the fp8 act) and out into them
CX=torch.zeros_like(a); COUT=torch.zeros((M,N),dtype=torch.bfloat16,device=dev)

def bare():
    o=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None); return o
def copy2():                              # (B) always-on cost: GEMM + copy x + copy out (NO recompute)
    o=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None)
    CX.copy_(a); COUT.copy_(o); return o
def always_on_recompute():                # Addendum-4 reference: GEMM + full recompute every step
    o=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None)
    o2=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o2,a,b,sa,sb,None)
    COUT.copy_((o2.float()-o.float()).abs().to(torch.bfloat16)); return o

def cgtime(fn,it=4000):
    cf=torch.compile(fn,fullgraph=True)
    for _ in range(5): cf()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): cf()
    torch.cuda.synchronize()
    for _ in range(200): g.replay()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it
tb=cgtime(bare); tc=cgtime(copy2); tr=cgtime(always_on_recompute)
print("PER2-COST", {"bare_ms":round(tb,5),
    "copy2_ms":round(tc,5),"copy2_ovh%":round((tc/tb-1)*100,1),
    "alwayson_recompute_ms":round(tr,5),"alwayson_ovh%":round((tr/tb-1)*100,1),
    "B_per_step_floor_vs_alwayson":f"{round((tc-tb)/(tr-tb)*100,1)}% of always-on extra"})

# Estimate periodic effective overhead: always-on copies + recompute amortized every N steps.
# per-step extra = copy_extra + (recompute_extra)/N
copy_extra = tc - tb; recomp_extra = tr - tc       # recompute extra beyond the copies
for Nn in [8,16,32,45,64]:
    eff = tb + copy_extra + recomp_extra/Nn
    print("PER2-NSWEEP", {"N":Nn,"eff_ms":round(eff,5),"overhead%":round((eff/tb-1)*100,1)})

# ---- detection from BUFFERED I/O (host-side recompute) ----
def bitflip(o,bit,idx=123):
    v=o.view(-1); iv=v[idx].view(torch.int16); v[idx]=(iv ^ (1<<bit)).view(torch.bfloat16); return o
# simulate: graph copied a->CX, out->COUT (with a fault injected into out); host recompute compares
o=bare(); CX.copy_(a)
hits={}
for name,bit in [("sign",15),("bit14",14),("bit5_mant",5),("bit0_mant",0)]:
    of=bitflip(o.clone(),bit); COUT.copy_(of)
    o2=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o2,CX,b,sa,sb,None)  # host eager recompute from buffer
    resid=(o2.float()-COUT.float()).abs().max().item()
    hits[name]={"resid":round(resid,4),"caught(exact,resid>0)":resid>0}
print("PER2-DETECT", hits)
