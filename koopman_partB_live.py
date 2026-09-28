#!/usr/bin/env python3
# READ-ONLY. Part B -- live shipped-reality probe. Drives the DEPLOYED .so's real kernels via direct C-API:
#   cipher_edmd_live_collect (accumulate 2000 real-activation rows) -> background randomized-SVD rank-64 fit + register
#   -> cipher_koopman_fp16_ood_max_residual (the runtime OOD gate value vs beta=0.05)
#   -> cipher_koopman_fp16_launch_shape (the actual .cu surrogate kernel: quality + wall-clock at the shipped r=64).
# No .so change. Confirms the offline frontier on the real shipped kernels (randomized SVD, not exact).
import os, sys, time, json, ctypes, subprocess
os.environ.setdefault("CIPHER_EDMD_LIVE", "1")
os.environ.setdefault("CIPHER_USE_CACHE", "0")   # measure real kernel compute, not the pointer-cache memcpy
os.environ.setdefault("CIPHER_KOOPMAN", "1")
import torch
torch.manual_seed(0)
SO = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
MODEL = os.environ.get("MODEL", "/home/ubuntu/models/Mistral-7B-v0.1")
LAYER = int(os.environ.get("LAYER", "16"))
LOCK_MHZ = int(os.environ.get("LOCK_MHZ", "1200"))
DT_FP16 = 2
dev = "cuda"

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=dev).eval()
cfg = model.config; H, V = cfg.hidden_size, cfg.vocab_size
TARGETS = {"lm_head": (model.lm_head, (H, V)),
           "q_proj":  (model.model.layers[LAYER].self_attn.q_proj, (H, H))}
print(f"[B] H={H} V={V} layer={LAYER} so={os.path.basename(SO)}", flush=True)

# ---- capture real activations (X=input) + true outputs (Y) for each target ----
cap = {n: {"X": [], "Y": []} for n in TARGETS}
_on = {"v": False}
def mk(n):
    def hook(m, i, o):
        if not _on["v"]: return
        cap[n]["X"].append(i[0].detach().reshape(-1, i[0].shape[-1]).half())
        cap[n]["Y"].append(o.detach().reshape(-1, o.shape[-1]).half())
    return hook
hs = [mod.register_forward_hook(mk(n)) for n,(mod,_) in TARGETS.items()]
def feed(text, ntok, seqlen=1024):
    ids = tok(text, return_tensors="pt").input_ids[0]; got=0; pos=0
    while got < ntok and pos+2 < ids.numel():
        ch = ids[pos:pos+seqlen].unsqueeze(0).to(dev); pos+=seqlen
        _on["v"]=True
        with torch.no_grad(): model(ch)
        _on["v"]=False; got += ch.shape[1]
def wiki():
    try:
        from datasets import load_dataset
        return "\n\n".join(t for t in load_dataset("wikitext","wikitext-2-raw-v1",split="test")["text"] if t.strip())
    except Exception:
        import glob; return "".join(open(p,errors="ignore").read() for p in glob.glob("/home/ubuntu/*.md")[:8])
def code():
    import glob; t=""
    for p in glob.glob("/home/ubuntu/*.py"):
        t+=open(p,errors="ignore").read()+"\n"
        if len(t)>300000: break
    return t
wt=wiki(); cd=code(); half=len(wt)//2
print("[B] capturing real activations...", flush=True)
feed(wt[:half], 2800); ntr={n:sum(x.shape[0] for x in cap[n]["X"]) for n in TARGETS}
feed(wt[half:], 600);  nte={n:sum(x.shape[0] for x in cap[n]["X"]) for n in TARGETS}
feed(cd, 600)
for h in hs: h.remove()
def slabs(n):
    X=torch.cat(cap[n]["X"]); Y=torch.cat(cap[n]["Y"]); a,b=ntr[n],nte[n]
    return (X[:a].contiguous(),Y[:a].contiguous()),(X[a:b].contiguous(),Y[a:b].contiguous()),(X[b:].contiguous(),Y[b:].contiguous())

# ---- load shipped .so, bind C-API ----
lib = ctypes.CDLL(SO)
lib.cipher_edmd_live_collect.restype = ctypes.c_bool
lib.cipher_edmd_live_collect.argtypes = [ctypes.c_int]*3 + [ctypes.c_int, ctypes.c_void_p]*1 + \
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.cipher_edmd_live_is_registered.restype = ctypes.c_bool
lib.cipher_edmd_live_is_registered.argtypes = [ctypes.c_int, ctypes.c_int]
lib.cipher_edmd_live_get_stats.restype = ctypes.c_bool
lib.cipher_edmd_live_get_stats.argtypes = [ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
lib.cipher_koopman_fp16_ood_max_residual.restype = ctypes.c_float
lib.cipher_koopman_fp16_ood_max_residual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.cipher_koopman_fp16_launch_shape.restype = ctypes.c_int
lib.cipher_koopman_fp16_launch_shape.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

def smlock(m):
    if m>0: subprocess.run(["sudo","-n","nvidia-smi","-lgc",f"{m},{m}"],capture_output=True); time.sleep(0.3)
def smreset(): subprocess.run(["sudo","-n","nvidia-smi","-rgc"],capture_output=True)
def bench(fn, it=50, wm=10):
    for _ in range(wm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3

out_report = {}
smlock(LOCK_MHZ)
for name,(mod,(K,N)) in TARGETS.items():
    (Xtr,Ytr),(Xte,Yte),(Xc,Yc) = slabs(name)
    Wt = mod.weight.detach().t().contiguous().half()   # (K,N) row-major; collect ignores it but needs valid ptr
    print(f"\n[B {name}] K={K} N={N} train_rows={Xtr.shape[0]} -- feeding shipped collect (8 rows/call, distinct ptrs)", flush=True)
    chunks=[]   # keep alive until registered (async D2H reads these)
    fed=0; i=0
    while fed < 2200 and (i+1)*8 <= Xtr.shape[0]:
        xc = Xtr[i*8:(i+1)*8].clone(); yc = Ytr[i*8:(i+1)*8].clone()   # clone => distinct device ptr
        chunks.append((xc,yc))
        lib.cipher_edmd_live_collect(8, K, N, DT_FP16, ctypes.c_void_p(Wt.data_ptr()),
                                     DT_FP16, ctypes.c_void_p(xc.data_ptr()),
                                     DT_FP16, ctypes.c_void_p(yc.data_ptr()))
        torch.cuda.synchronize(); fed+=8; i+=1
    # poll for background fit/register
    reg=False; t0=time.time()
    while time.time()-t0 < 90:
        if lib.cipher_edmd_live_is_registered(K,N): reg=True; break
        time.sleep(0.5)
    rr=ctypes.c_float(); en=ctypes.c_float(); md=ctypes.c_float(); rows=ctypes.c_int()
    has=lib.cipher_edmd_live_get_stats(K,N, ctypes.byref(rr),ctypes.byref(en),ctypes.byref(md),ctypes.byref(rows))
    rec=dict(K=K,N=N,registered=bool(reg),fed_rows=fed,stats_energy=round(en.value,4),stats_rows=rows.value)
    print(f"[B {name}] registered={reg} fed={fed} shipped_energy_captured(r=64)={en.value:.4f} rows={rows.value}", flush=True)
    if reg:
        # runtime OOD gate value on held-out + cross-corpus (the real fire/no-fire number vs beta=0.05)
        def ood(X):
            mrows=min(512, X.shape[0]); Xb=X[:mrows].contiguous()
            return float(lib.cipher_koopman_fp16_ood_max_residual(ctypes.c_void_p(Xb.data_ptr()), mrows, K, N))
        ood_h=ood(Xte); ood_c=ood(Xc)
        rec.update(ood_heldout=round(ood_h,4), ood_crosscorpus=round(ood_c,4),
                   fires_at_default_gate=bool(ood_h<=0.05))
        print(f"[B {name}] runtime OOD residual held={ood_h:.4f} cross={ood_c:.4f}  (gate beta=0.05 -> fires={ood_h<=0.05})", flush=True)
        # real .cu surrogate kernel: quality vs true GEMM + wall-clock vs cuBLAS, at shipped r=64
        M=min(512, Xte.shape[0]); Xb=Xte[:M].contiguous(); Yt=Yte[:M].contiguous()
        outk=torch.empty(M,N, device=dev, dtype=torch.float16)
        rc=lib.cipher_koopman_fp16_launch_shape(ctypes.c_void_p(Xb.data_ptr()), ctypes.c_void_p(outk.data_ptr()), M, K, N)
        rel=(torch.linalg.norm((outk.float()-Yt.float()))/torch.linalg.norm(Yt.float())).item()
        rec.update(launch_rc=rc, kernel_rel_err=round(rel,4))
        if name=="lm_head":
            top1=(outk.argmax(-1)==Yt.argmax(-1)).float().mean().item(); rec["kernel_top1"]=round(top1,4)
        # wall-clock: real .cu surrogate vs cuBLAS full GEMM
        Wf = mod.weight.detach().half()  # (N,K) for F.linear
        t_full = bench(lambda: torch.matmul(Xb, Wt))
        t_kern = bench(lambda: lib.cipher_koopman_fp16_launch_shape(ctypes.c_void_p(Xb.data_ptr()), ctypes.c_void_p(outk.data_ptr()), M, K, N))
        rec.update(full_ms=round(t_full,4), kernel_ms=round(t_kern,4), realized_speedup=round(t_full/t_kern,3),
                   flop_ceiling=round(K*N/(64*(K+N)),1))
        print(f"[B {name}] kernel rel_err={rel:.4f}{' top1='+str(rec.get('kernel_top1')) if name=='lm_head' else ''} | full={t_full:.4f}ms kernel={t_kern:.4f}ms realized={t_full/t_kern:.2f}x (FLOPceil={rec['flop_ceiling']}x)", flush=True)
    out_report[name]=rec
    del chunks; torch.cuda.empty_cache()
smreset()
json.dump(dict(model=os.path.basename(MODEL), ood_beta=0.05, kr_rank=64, results=out_report),
          open("/home/ubuntu/koopman_partB_results.json","w"), indent=1)
print("\nKOOPMAN_B_JSON written /home/ubuntu/koopman_partB_results.json", flush=True)
sys.stdout.flush(); os._exit(0)
