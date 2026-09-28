# Phase 0 micro: ABFT row+column checksum for FP8 cutlass_scaled_mm. Settles whether ABFT beats
# Freivalds on BOTH detection (single-element SDC) and cost (under cudagraph), for DECODE (M=1) and
# PREFILL (M=512) shapes. Real op: out = sa*sb*(a_fp8.float() @ b_fp8.float()) -> bf16, per-tensor scales.
#   row checksum:  out @ 1_N   ?=  sa*sb * a @ (b @ 1_N)        [b@1_N is a CONSTANT -> precomputable]
#   col checksum:  1_M^T @ out ?=  sa*sb * (1_M^T @ a) @ b      [1_M^T@a depends on a -> NOT precomputable]
import torch, vllm._custom_ops
dev="cuda"; _real=torch.ops._C.cutlass_scaled_mm; torch.manual_seed(0)
def mk(M,K,N):
    a=(torch.randn(M,K,device=dev)*0.1).to(torch.float8_e4m3fn)
    b=(torch.randn(N,K,device=dev)*0.1).to(torch.float8_e4m3fn).t()
    return a,b,torch.tensor(0.7,device=dev),torch.tensor(1.3,device=dev)
def realmm(a,b,sa,sb):
    o=torch.empty((a.shape[0],b.shape[1]),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None); return o
def bitflip(o,bit,idx):
    v=o.view(-1); iv=v[idx].view(torch.int16); v[idx]=(iv ^ (1<<bit)).view(torch.bfloat16); return o

def clean_tol(M,K,N,trials=120):
    rr=[]; cr=[]
    for _ in range(trials):
        a,b,sa,sb=mk(M,K,N); o=realmm(a,b,sa,sb)
        oneN=torch.ones(N,1,device=dev); oneM=torch.ones(1,M,device=dev)
        rc_o=o.float()@oneN; rc_r=(sa*sb)*(a.float()@(b.float()@oneN))
        cc_o=oneM@o.float(); cc_r=(sa*sb)*((oneM@a.float())@b.float())
        rr.append((rc_o-rc_r).abs().max().item()/(rc_o.abs().max().item()+1e-6))
        cr.append((cc_o-cc_r).abs().max().item()/(cc_o.abs().max().item()+1e-6))
    return torch.tensor(rr).max().item()*3, torch.tensor(cr).max().item()*3   # 3x clean-max relative tol

def catch(M,K,N,bit,row_tol,col_tol,trials=300):
    oneN=torch.ones(N,1,device=dev); oneM=torch.ones(1,M,device=dev); rh=ch=0
    for _ in range(trials):
        a,b,sa,sb=mk(M,K,N); o=realmm(a,b,sa,sb)
        idx=torch.randint(0,M*N,(1,)).item(); o=bitflip(o,bit,idx)
        rc_o=o.float()@oneN; rc_r=(sa*sb)*(a.float()@(b.float()@oneN))
        cc_o=oneM@o.float(); cc_r=(sa*sb)*((oneM@a.float())@b.float())
        rrel=((rc_o-rc_r).abs()/(rc_o.abs().max()+1e-6)).max().item()
        crel=((cc_o-cc_r).abs()/(cc_o.abs().max()+1e-6)).max().item()
        if rrel>row_tol: rh+=1
        if crel>col_tol: ch+=1
    return round(rh/trials,3), round(ch/trials,3)

for (M,K,N,tag) in [(1,4096,14336,"DECODE"),(512,4096,14336,"PREFILL")]:
    rtol,ctol=clean_tol(M,K,N)
    print(f"ABFT-TOL-{tag}", {"row_tol":round(rtol,5),"col_tol":round(ctol,5)})
    tbl={}
    for bit in [15,14,13,10,7,5,2,0]:
        r,c=catch(M,K,N,bit,rtol,ctol)
        lab=f"bit{bit}({'sign' if bit==15 else 'exp' if bit>=7 else 'mant'})"
        tbl[lab]={"row_only":r,"col_only":c,"ABFT(row|col)":round(max(r,c),3)}
    print(f"ABFT-CATCH-{tag}", tbl)

# ===== COST under cudagraph =====
def cost(M,K,N,tag):
    a,b,sa,sb=mk(M,K,N); oneN=torch.ones(N,1,device=dev); oneM=torch.ones(1,M,device=dev)
    bN=(b.float()@oneN).contiguous()   # precomputed b@1_N (row-checksum reference part)
    R_ROW=torch.zeros(M,1,device=dev); R_COL=torch.zeros(1,N,device=dev)
    def bare():
        o=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None); return o
    def row_only():   # cheap path: precomputed b@1_N
        o=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None)
        R_ROW.copy_((o.float()@oneN - (sa*sb)*(a.float()@bN)).abs()); return o
    def full_abft():  # row + col (col NOT precomputable: (1_M@a)@b)
        o=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None)
        R_ROW.copy_((o.float()@oneN - (sa*sb)*(a.float()@bN)).abs())
        R_COL.copy_((oneM@o.float() - (sa*sb)*((oneM@a.float())@b.float())).abs()); return o
    fns={"bare":bare,"row_only":row_only,"full_abft":full_abft}
    cg={}
    for nm,fn in fns.items():
        cf=torch.compile(fn,fullgraph=True)
        for _ in range(5): cf()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): cf()
        torch.cuda.synchronize()
        for _ in range(100): g.replay()
        torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
        for _ in range(3000): g.replay()
        e.record(); torch.cuda.synchronize(); cg[nm]=s.elapsed_time(e)/3000
    print(f"ABFT-COST-{tag}", {"bare_ms":round(cg['bare'],5),
        "row_only_ms":round(cg['row_only'],5),"row_only_ovh%":round((cg['row_only']/cg['bare']-1)*100,1),
        "full_abft_ms":round(cg['full_abft'],5),"full_abft_ovh%":round((cg['full_abft']/cg['bare']-1)*100,1)})
cost(1,4096,14336,"DECODE"); cost(512,4096,14336,"PREFILL")

# CSE guard (decode)
a,b,sa,sb=mk(1,4096,14336); o=realmm(a,b,sa,sb); oneM=torch.ones(1,1,device=dev)
cc_o=oneM@o.float()
rc=(cc_o-(sa*sb)*((oneM@a.float())@b.float())).abs().max().item()
rp=(cc_o-(sa*sb)*((oneM@(a.float()*1.05))@b.float())).abs().max().item()
print("ABFT-CSE",{"resid_clean":round(rc,5),"resid_operand_perturb":round(rp,5),"responds":rp>10*rc})
