#!/usr/bin/env python3
# READ-ONLY. Koopman/rank measurement -- Part A (offline frontier) + the wall-clock microbench.
# Faithful replication of the SHIPPED actuator's fit (cipher_edmd_live.cpp:284-360):
#   data-driven reduced-rank regression on the ACTIVATION manifold (NOT weight SVD).
#   X (m,K)=real activations, Y (m,N)=real GEMM outputs. SVD of X => U S V^T.
#   surrogate map A_hat = V_r diag(1/sigma_r) U_r^T Y  (rank r);  out = X @ A_hat.
#   energy_captured(r)=sum(sigma_r^2)/sum(sigma^2);  residual_ratio=||Y-X@A_hat||/||Y||.
# Shipped reality: KR_RANK=64, OOD gate beta=0.05 (fires only when residual<=5%).
# We use EXACT torch SVD (actuator uses randomized rank-r) -> our residual is an OPTIMISTIC LOWER BOUND.
# Sweep r; report energy, residual {in-sample / held-out same-corpus / held-out CROSS-corpus},
# FLOP-ratio speedup (ceiling) AND measured wall-clock speedup (realized), lm_head logit-KL/top1/PPL-delta.
import os, sys, json, time, subprocess
import torch
torch.manual_seed(0)
MODEL = os.environ.get("MODEL", "/home/ubuntu/models/Mistral-7B-v0.1")
LAYER = int(os.environ.get("LAYER", "16"))
N_TRAIN = int(os.environ.get("N_TRAIN", "6144"))
N_TEST  = int(os.environ.get("N_TEST",  "2048"))
SEQLEN  = int(os.environ.get("SEQLEN",  "1024"))
RANKS = [int(x) for x in os.environ.get("RANKS", "4,8,16,32,64,128,256,512,1024,2048").split(",")]
LOCK_MHZ = int(os.environ.get("LOCK_MHZ", "1200"))
dev = "cuda"

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=dev).eval()
cfg = model.config
H, I_, V = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
print(f"[cfg] H={H} I={I_} V={V} layer={LAYER}", flush=True)

# ---- locate the 3 target Linear modules ----
lm_head   = model.lm_head
down_proj = model.model.layers[LAYER].mlp.down_proj
q_proj    = model.model.layers[LAYER].self_attn.q_proj
TARGETS = {"lm_head": lm_head, "down_proj": down_proj, "q_proj": q_proj}
SHAPES  = {"lm_head": (H, V), "down_proj": (I_, H), "q_proj": (H, H)}  # (K,N)

# ---- corpora: wikitext (train+test, same dist) + a CROSS-corpus (code) for the OOD point ----
def wikitext_text():
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(t for t in ds["text"] if t.strip())
    except Exception as e:
        print(f"[warn] datasets load failed ({str(e)[:60]}); using local fallback text", flush=True)
        # fallback: concatenate repo prose (README-like). Still real English tokens.
        import glob
        txt = ""
        for p in glob.glob("/home/ubuntu/*.md")[:8]:
            txt += open(p, errors="ignore").read() + "\n\n"
        return txt
def code_text():
    import glob
    txt = ""
    for p in (glob.glob("/home/ubuntu/*.py") + glob.glob("/usr/lib/python3.10/*.py")):
        try: txt += open(p, errors="ignore").read() + "\n\n"
        except Exception: pass
        if len(txt) > 400000: break
    return txt

# ---- capture buffers ----
cap = {name: {"X": [], "Y": []} for name in TARGETS}     # per corpus run we collect, then stack
lm_targets = []   # next-token ids aligned to lm_head rows (for PPL)
_active = {"on": False, "ids": None}
def mk_hook(name):
    def hook(mod, inp, out):
        if not _active["on"]: return
        X = inp[0].detach().reshape(-1, inp[0].shape[-1])     # (T, K) fp16
        Y = out.detach().reshape(-1, out.shape[-1])           # (T, N) fp16
        if name == "lm_head":
            # drop last position; target = next token
            X = X[:-1]; Y = Y[:-1]
            lm_targets.append(_active["ids"][0, 1:].detach())
        cap[name]["X"].append(X.float().cpu())
        cap[name]["Y"].append(Y.float().cpu())
    return hook
handles = [m.register_forward_hook(mk_hook(n)) for n, m in TARGETS.items()]

def run_corpus(text, n_rows, tag):
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    got = 0; pos = 0
    rows_before = {n: sum(x.shape[0] for x in cap[n]["X"]) for n in TARGETS}
    while got < n_rows and pos + 2 < ids_all.numel():
        chunk = ids_all[pos:pos+SEQLEN].unsqueeze(0).to(dev); pos += SEQLEN
        _active["on"] = True; _active["ids"] = chunk
        with torch.no_grad(): model(chunk)
        _active["on"] = False
        got += chunk.shape[1] - 1
    added = {n: sum(x.shape[0] for x in cap[n]["X"]) - rows_before[n] for n in TARGETS}
    print(f"[capture {tag}] ~{got} rows  added={added}", flush=True)

print("[phase] capturing activations (train=wikitext, test=wikitext-heldout, code=cross-corpus)", flush=True)
wt = wikitext_text(); code = code_text()
# split wikitext into train + heldout-test by position (disjoint)
half = len(wt)//2
run_corpus(wt[:half], N_TRAIN, "train")
splits = {n: sum(x.shape[0] for x in cap[n]["X"]) for n in TARGETS}      # boundary train|test (ROW counts)
run_corpus(wt[half:], N_TEST, "test")
test_end = {n: sum(x.shape[0] for x in cap[n]["X"]) for n in TARGETS}
run_corpus(code, N_TEST, "code")
for h in handles: h.remove()

# assemble per-shape tensors on GPU fp32, sliced into train/test/code
def slabs(name):
    Xall = torch.cat(cap[name]["X"]).to(dev); Yall = torch.cat(cap[name]["Y"]).to(dev)
    s, te = splits[name], test_end[name]
    return (Xall[:s], Yall[:s]), (Xall[s:te], Yall[s:te]), (Xall[te:], Yall[te:])
lm_tgt = torch.cat(lm_targets)  # aligned row-for-row to lm_head captured rows
_ls, _le = splits["lm_head"], test_end["lm_head"]
lm_tgt_train, lm_tgt_test, lm_tgt_code = lm_tgt[:_ls], lm_tgt[_ls:_le], lm_tgt[_le:]

def eff_rank_stats(S):
    p = (S**2); p = p/p.sum()
    part = (p.sum()**2 / (p**2).sum()).item()      # participation ratio
    csum = torch.cumsum(p, 0)
    r90 = int((csum < 0.90).sum().item())+1
    r99 = int((csum < 0.99).sum().item())+1
    return part, r90, r99

def ppl(logits, targets):
    lp = torch.log_softmax(logits.float(), -1)
    nll = -lp[torch.arange(logits.shape[0], device=logits.device), targets]
    return torch.exp(nll.mean()).item()

results = {}
for name, (K, N) in SHAPES.items():
    (Xtr, Ytr), (Xte, Yte), (Xc, Yc) = slabs(name)
    m = Xtr.shape[0]
    print(f"\n[fit {name}] K={K} N={N} train={m} test={Xte.shape[0]} code={Xc.shape[0]}", flush=True)
    U, S, Vh = torch.linalg.svd(Xtr, full_matrices=False)   # U(m,k0) S(k0) Vh(k0,K), k0=min(m,K)
    part, r90, r99 = eff_rank_stats(S)
    print(f"[spectrum {name}] participation_ratio={part:.1f}  rank@90%={r90}  rank@99%={r99}  (of K={K})", flush=True)
    rows = []
    for r in RANKS:
        if r > S.shape[0]: continue
        Vr = Vh[:r].T.contiguous()            # (K,r)
        Ur = U[:, :r]                          # (m,r)
        Wk = (Ur.T @ Ytr) / S[:r].unsqueeze(1) # (r,N)
        energy = (S[:r]**2).sum().item() / (S**2).sum().item()
        def resid(Xe, Ye):
            out = (Xe @ Vr) @ Wk
            return (torch.linalg.norm(Ye-out)/torch.linalg.norm(Ye)).item(), out
        ri, _   = resid(Xtr, Ytr)
        rh, oute = resid(Xte, Yte)
        rc, outc = resid(Xc, Yc)
        flop_speed = K*N/(r*(K+N))
        row = dict(rank=r, energy=round(energy,4), resid_insample=round(ri,4),
                   resid_heldout=round(rh,4), resid_crosscorpus=round(rc,4),
                   flop_speedup=round(flop_speed,2), passes_ood_005=bool(rh<=0.05))
        if name == "lm_head":
            kl = torch.nn.functional.kl_div(torch.log_softmax(oute.float(),-1),
                                            torch.log_softmax(Yte.float(),-1),
                                            log_target=True, reduction="batchmean").item()
            top1 = (oute.argmax(-1) == Yte.argmax(-1)).float().mean().item()
            ppl_true = ppl(Yte, lm_tgt_test.to(dev)); ppl_sur = ppl(oute, lm_tgt_test.to(dev))
            ppl_true_c = ppl(Yc, lm_tgt_code.to(dev)); ppl_sur_c = ppl(outc, lm_tgt_code.to(dev))
            row.update(kl_heldout=round(kl,5), top1_heldout=round(top1,4),
                       ppl_true=round(ppl_true,3), ppl_surrogate=round(ppl_sur,3),
                       ppl_delta_pct=round(100*(ppl_sur-ppl_true)/ppl_true,3),
                       ppl_delta_pct_crosscorpus=round(100*(ppl_sur_c-ppl_true_c)/ppl_true_c,3))
        rows.append(row)
        extra = f" KLheld={row.get('kl_heldout','')} top1={row.get('top1_heldout','')} dPPL%={row.get('ppl_delta_pct','')}" if name=="lm_head" else ""
        print(f"  r={r:>5} energy={energy:.3f} resid[in/held/cross]={ri:.3f}/{rh:.3f}/{rc:.3f} OOD<=.05held={row['passes_ood_005']} FLOPx={flop_speed:.1f}{extra}", flush=True)
    results[name] = dict(K=K, N=N, train_rows=m, participation_ratio=round(part,1), rank90=r90, rank99=r99, sweep=rows)
    del U, S, Vh; torch.cuda.empty_cache()

# ---- wall-clock microbench: realized speedup vs FLOP ceiling (advisor #1) ----
print("\n[phase] wall-clock microbench (locked clock, cuBLAS for both arms)", flush=True)
def smlock(mhz):
    if mhz>0: subprocess.run(["sudo","-n","nvidia-smi","-lgc",f"{mhz},{mhz}"],capture_output=True); time.sleep(0.3)
def smreset(): subprocess.run(["sudo","-n","nvidia-smi","-rgc"],capture_output=True)
def bench(fn, it=50, wm=10):
    for _ in range(wm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it
smlock(LOCK_MHZ)
micro = {}
for name,(K,N) in SHAPES.items():
    micro[name] = {}
    for M in [1, 512, 2048]:
        A = torch.randn(M,K, device=dev, dtype=torch.float16)
        W = torch.randn(K,N, device=dev, dtype=torch.float16)
        t_full = bench(lambda: torch.matmul(A,W))
        micro[name][M] = {"full_ms": round(t_full*1e3,4), "ranks": []}
        for r in RANKS:
            Vr = torch.randn(K,r, device=dev, dtype=torch.float16)
            Wk = torch.randn(r,N, device=dev, dtype=torch.float16)
            t_sur = bench(lambda: torch.matmul(torch.matmul(A,Vr), Wk))
            micro[name][M]["ranks"].append(dict(rank=r, sur_ms=round(t_sur*1e3,4),
                realized_speedup=round(t_full/t_sur,3), flop_speedup=round(K*N/(r*(K+N)),2)))
        best = micro[name][M]["ranks"]
        print(f"  {name} M={M:>4} full={t_full*1e3:.3f}ms | r=64 realized={[x for x in best if x['rank']==64][0]['realized_speedup'] if any(x['rank']==64 for x in best) else 'NA'}x (FLOPceil={K*N/(64*(K+N)):.1f}x)", flush=True)
smreset()

out = dict(model=os.path.basename(MODEL), layer=LAYER, ranks=RANKS, ood_beta=0.05, kr_rank_shipped=64,
           note="exact-SVD => residual is OPTIMISTIC lower bound vs shipped randomized rank-r SVD",
           frontier=results, microbench=micro)
json.dump(out, open("/home/ubuntu/koopman_rank_results.json","w"), indent=1)
print("\nKOOPMAN_JSON written /home/ubuntu/koopman_rank_results.json", flush=True)
sys.stdout.flush(); os._exit(0)
