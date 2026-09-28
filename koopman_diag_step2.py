#!/usr/bin/env python3
# READ-ONLY. Koopman diagnostic Step 2 (load-bearing): does the WORKING config's MATH fire+work?
# Working op = the MLP BLOCK (cipher_wrapper.py install_all_mlp_hooks: mlp.forward replaced by a rank-r Koopman map,
# X=MLP input hidden (4096) -> Y=MLP output (4096); matches LayerPtrs (16,4096)). NOT a single linear.
# 2x2: {NARROW repetitive, DIVERSE} calibration x {r=16 (working), r=64 (shipped)}, on SELECTED + CONTRAST layers.
# Distinguish the TWO sub-questions the gate ladder separates:
#   FIRES?  -> input-subspace per-row-max residual  max_i ||x_i - Vx Vx^T x_i|| / ||x_i||   vs OOD gate beta=0.05
#             (this is what cipher_koopman_fp16_ood_max_residual computes; my prior 0.96 was THIS metric)
#   CORRECT?-> output residual ||Y - X A_hat|| / ||Y|| (Tikhonov), + fit_error<0.08 (working selection criterion)
# exact torch SVD => optimistic vs shipped randomized SVD. No .so, no GPU clock change.
import os, sys, json
import torch
torch.manual_seed(0)
MODEL=os.environ.get("MODEL","/home/ubuntu/models/Mistral-7B-v0.1")
SEL=[0,8,12,23]      # subset of the working selected set {0,2,4,7,8,9,10,11,12,13,15,23}
CONTRAST=[16,31]     # non-selected, for contrast
RANKS=[16,64]
NTR=int(os.environ.get("NTR","2048")); NTE=int(os.environ.get("NTE","512")); SEQ=512
LAM=float(os.environ.get("TIKHONOV","1e-2"))   # Tikhonov (working config used regularization)
dev="cuda"
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,device_map=dev).eval()
H=model.config.hidden_size
LAYERS=sorted(set(SEL+CONTRAST))
mlps={L: model.model.layers[L].mlp for L in LAYERS}

cap={L:{"X":[],"Y":[]} for L in LAYERS}
_on={"v":False}
def mk(L):
    def hook(m,i,o):
        if not _on["v"]: return
        cap[L]["X"].append(i[0].detach().reshape(-1,H).float().cpu())
        cap[L]["Y"].append(o.detach().reshape(-1,H).float().cpu())
    return hook
hs=[mlps[L].register_forward_hook(mk(L)) for L in LAYERS]

def run(text,ntok,seqlen=SEQ):
    ids=tok(text,return_tensors="pt").input_ids[0]; got=0;pos=0
    while got<ntok and pos+2<ids.numel():
        ch=ids[pos:pos+seqlen].unsqueeze(0).to(dev);pos+=seqlen
        _on["v"]=True
        with torch.no_grad(): model(ch)
        _on["v"]=False; got+=ch.shape[1]
def wiki():
    try:
        from datasets import load_dataset
        return "\n\n".join(t for t in load_dataset("wikitext","wikitext-2-raw-v1",split="test")["text"] if t.strip())
    except Exception:
        import glob; return "".join(open(p,errors="ignore").read() for p in glob.glob('/home/ubuntu/*.md')[:8])
# NARROW repetitive: one short sentence repeated -> low-diversity activation manifold (the working-config condition)
NARROW=("The quick brown fox jumps over the lazy dog. "*4000)

print("[step2] capturing NARROW (repetitive) ...",flush=True)
run(NARROW, NTR+NTE, SEQ); narrow_end={L:sum(x.shape[0] for x in cap[L]["X"]) for L in LAYERS}
print("[step2] capturing DIVERSE (wikitext) ...",flush=True)
run(wiki(), NTR+NTE, SEQ)
for h in hs: h.remove()

def split(L):
    X=torch.cat(cap[L]["X"]).to(dev); Y=torch.cat(cap[L]["Y"]).to(dev); n=narrow_end[L]
    Xn,Yn=X[:n],Y[:n]; Xd,Yd=X[n:],Y[n:]
    return (Xn[:NTR],Yn[:NTR],Xn[NTR:NTR+NTE],Yn[NTR:NTR+NTE]),(Xd[:NTR],Yd[:NTR],Xd[NTR:NTR+NTE],Yd[NTR:NTR+NTE])

def eff_rank(S):
    p=(S**2); p=p/p.sum(); return (p.sum()**2/(p**2).sum()).item(), int((torch.cumsum(p,0)<0.90).sum())+1

def input_gate_resid(Vr, Xe):   # max & mean over rows of ||x - Vr Vr^T x||/||x||  (the OOD-kernel metric)
    proj=(Xe@Vr)@Vr.T
    num=torch.linalg.norm(Xe-proj,dim=1); den=torch.linalg.norm(Xe,dim=1).clamp_min(1e-6)
    r=(num/den)
    return r.max().item(), r.mean().item()

def out_resid(Vr,S,Ur,Ytr,Xe,Ye,lam):  # Tikhonov reduced-rank regression, applied to held-out
    Wk=( (Ur.T@Ytr) * (S/(S**2+lam*S[0]**2)).unsqueeze(1) )   # (r,N), ridge on 1/sigma
    out=(Xe@Vr)@Wk
    return (torch.linalg.norm(Ye-out)/torch.linalg.norm(Ye)).item()

results={}
for L in LAYERS:
    (nar),(div)=split(L)
    sets={"narrow":nar,"diverse":div}
    results[L]={"selected":L in SEL,"cells":[]}
    # report effective rank of each manifold
    for cname,(Xtr,Ytr,Xte,Yte) in sets.items():
        U,S,Vh=torch.linalg.svd(Xtr,full_matrices=False)
        er_pr,er90=eff_rank(S)
        for r in RANKS:
            Vr=Vh[:r].T.contiguous(); Ur=U[:,:r]
            # FIRES: input gate residual on held-out SAME corpus + the OTHER corpus (OOD)
            other = "diverse" if cname=="narrow" else "narrow"
            Xo_te=sets[other][2]; Yo_te=sets[other][3]
            g_max,g_mean=input_gate_resid(Vr,Xte)
            go_max,go_mean=input_gate_resid(Vr,Xo_te)
            # CORRECT: output residual held-out same + other
            o_in =out_resid(Vr,S[:r],Ur,Ytr,Xtr,Ytr,LAM)   # IN-SAMPLE = the "fit_error" the working config reported (memorization)
            o_same=out_resid(Vr,S[:r],Ur,Ytr,Xte,Yte,LAM)   # held-out same corpus = generalization within narrow
            o_other=out_resid(Vr,S[:r],Ur,Ytr,Xo_te,Yo_te,LAM)
            gi_max,gi_mean=input_gate_resid(Vr,Xtr)          # in-sample gate
            cell=dict(calib=cname,rank=r,eff_rank_pr=round(er_pr,1),eff_rank90=er90,
                      gate_max_insample=round(gi_max,3),gate_mean_insample=round(gi_mean,3),
                      gate_resid_max_same=round(g_max,3),gate_resid_mean_same=round(g_mean,3),
                      fires_same=bool(g_max<=0.05),fires_same_mean=bool(g_mean<=0.05),
                      gate_resid_max_OOD=round(go_max,3),fires_OOD=bool(go_max<=0.05),
                      out_resid_insample=round(o_in,3),out_resid_same=round(o_same,3),out_resid_OOD=round(o_other,3),
                      fit_ok_008_insample=bool(o_in<0.08),fit_ok_008_heldout=bool(o_same<0.08))
            results[L]["cells"].append(cell)
            print(f"  L{L:<2}{'*' if L in SEL else ' '} {cname:<7} r={r:<2} effrank={er_pr:4.1f} | GATE in/held(max)={gi_max:.3f}/{g_max:.3f} mean={g_mean:.3f} fire(held)?{cell['fires_same']} | OUTresid in={o_in:.3f}(fit<.08?{cell['fit_ok_008_insample']}) held={o_same:.3f}(<.08?{cell['fit_ok_008_heldout']}) OOD={o_other:.3f}",flush=True)
    del U,S,Vh; torch.cuda.empty_cache()

json.dump(dict(model=os.path.basename(MODEL),op="MLP_block",selected_layers=SEL,contrast=CONTRAST,
               ood_beta=0.05,tikhonov=LAM,note="exact-SVD optimistic; gate metric=input-subspace per-row-max; fit_error=output resid",
               results={str(k):v for k,v in results.items()}),
          open("/home/ubuntu/koopman_step2_results.json","w"),indent=1)
print("\nKOOPMAN_STEP2_JSON written",flush=True)
sys.stdout.flush(); os._exit(0)
