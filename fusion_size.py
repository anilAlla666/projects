#!/usr/bin/env python3
# STEP 2 — size the fusion lever (RMSNorm + SiLU.mul), Marlin OFF, graph-captured, on cu13.
# Run: CIPHER_SUBSTITUTE_V2=1 CIPHER_FUSION=1 B=8 python3 fusion_size.py   (CIPHER_MARLIN unset, no injection)
# PART A: teacher-forced correctness (fusion vs fp16 native) — gate ~100% argmax, KL ~ fp16 round-off.
# PART B: GRAPH INTEGRITY — fusion-graph greedy tokens must == fusion-eager greedy tokens (else stale-buffer artifact).
# PART C: speed — baseline-graph vs fusion-graph tok/s + tok/W (headline); eager as context.
import os, ctypes, time, threading
import numpy as np, torch
import warnings; warnings.filterwarnings("ignore")
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL("/home/ubuntu/cipher-may13-evidence/libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_substitute_v2_init(); rt.cipher_fusion_kernels_init()
rt.cipher_fused_rmsnorm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p]; rt.cipher_fused_rmsnorm.restype=ctypes.c_int
rt.cipher_fused_silu_mul.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p]; rt.cipher_fused_silu_mul.restype=ctypes.c_int
class Stats(ctypes.Structure): _fields_=[("enabled",ctypes.c_int),("rmsnorm_calls",ctypes.c_ulonglong),("silu_mul_calls",ctypes.c_ulonglong),("residual_add_calls",ctypes.c_ulonglong)]
rt.cipher_fusion_kernels_stats.argtypes=[ctypes.POINTER(Stats)]
def fstats(): s=Stats(); rt.cipher_fusion_kernels_stats(ctypes.byref(s)); return (s.rmsnorm_calls,s.silu_mul_calls)

import pynvml; pynvml.nvmlInit(); H=pynvml.nvmlDeviceGetHandleByIndex(0)
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP
import transformers; print("transformers", transformers.__version__)

B=int(os.environ.get("B","8")); MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
def sp(): return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)  # stream-fix: capture-stream so kernels become graph nodes
class Power:
    def __init__(s): s.on=False; s.s=[]
    def _l(s):
        while s.on:
            try: s.s.append(pynvml.nvmlDeviceGetPowerUsage(H)/1000.0)
            except: pass
            time.sleep(0.02)
    def __enter__(s): s.on=True; s.t=threading.Thread(target=s._l); s.t.start(); return s
    def __exit__(s,*a): s.on=False; s.t.join()
    def mean(s): return float(np.mean(s.s)) if s.s else 0.0

tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="cuda").eval()

_orms=MistralRMSNorm.forward; _omlp=MistralMLP.forward
def f_rms(self,x):
    o=torch.empty_like(x); flat=x.reshape(-1,x.shape[-1]).contiguous(); of=o.reshape(-1,x.shape[-1])
    rc=rt.cipher_fused_rmsnorm(flat.data_ptr(),self.weight.contiguous().data_ptr(),of.data_ptr(),flat.shape[0],flat.shape[-1],float(self.variance_epsilon),sp())
    if rc!=1: raise RuntimeError("fused_rmsnorm rc=%d"%rc)   # never silently return uninit buffer
    return o
def f_mlp(self,x):
    g=self.gate_proj(x).contiguous(); u=self.up_proj(x).contiguous(); o=torch.empty_like(g)
    rc=rt.cipher_fused_silu_mul(g.data_ptr(),u.data_ptr(),o.data_ptr(),g.numel(),sp())
    if rc!=1: raise RuntimeError("fused_silu_mul rc=%d"%rc)
    return self.down_proj(o)
def patch(on):
    MistralRMSNorm.forward = f_rms if on else _orms
    MistralMLP.forward     = f_mlp if on else _omlp

base=["The history of computing spans several distinct eras, each defined by","In a quiet village nestled between two mountains, there lived",
 "The fundamental theorem of calculus connects the concept of","Once the spacecraft cleared the atmosphere, the crew began",
 "Economists have long debated whether monetary policy can","The recipe calls for fresh basil, ripe tomatoes, and a generous",
 "Deep beneath the ocean surface, bioluminescent creatures drift","When the ancient library was finally excavated, archaeologists found"]
prompts=[base[i%len(base)] for i in range(B)]
enc=tok(prompts,return_tensors="pt",padding=True); ids=enc.input_ids.cuda(); attn=enc.attention_mask.cuda()
plen=ids.shape[1]

# ---------- PART A: teacher-forced correctness (single forward, fp16 native vs fusion) ----------
patch(False)
with torch.no_grad(): gen=model.generate(ids,attention_mask=attn,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
full=gen  # [B, plen+24] teacher sequence
fa=torch.ones_like(full)
with torch.no_grad():
    lb=model(full,attention_mask=fa).logits.float()   # baseline logits
patch(True)
with torch.no_grad():
    lf=model(full,attention_mask=fa).logits.float()    # fusion logits, same input
patch(False)
gp=slice(plen-1,plen+23)  # positions predicting the generated tokens
amb=lb[:,gp].argmax(-1); amf=lf[:,gp].argmax(-1)
agree=(amb==amf).float().mean().item()
P=torch.softmax(lb[:,gp],-1); Q=torch.softmax(lf[:,gp],-1)
kl=(P*(P.clamp_min(1e-12).log()-Q.clamp_min(1e-12).log())).sum(-1)
print(f"[A correctness] teacher-forced argmax_agree={agree*100:.2f}%  kl_mean={kl.mean():.3e} kl_max={kl.max():.3e}  GATE={'PASS' if agree==1.0 else 'FAIL'}")

# ---------- graph helpers (B-batched single-step decode graph) ----------
def build_graph_and_replay(n_replay, collect=False):
    cache=StaticCache(config=model.config,max_batch_size=B,max_cache_len=plen+n_replay+8,device="cuda",dtype=torch.float16)
    with torch.no_grad():
        cp=torch.arange(plen,device="cuda"); o=model(input_ids=ids,attention_mask=attn,cache_position=cp,past_key_values=cache,use_cache=True,return_dict=True)
    nt=o.logits[:,-1:].argmax(-1); input_ids=nt.detach().clone(); cache_pos=torch.tensor([plen],device="cuda")
    fmask=torch.ones(B,plen+n_replay+8,device="cuda",dtype=torch.long)
    outl=torch.empty(B,1,model.config.vocab_size,device="cuda",dtype=torch.float16)
    side=torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                oo=model(input_ids=input_ids,cache_position=cache_pos,past_key_values=cache,use_cache=True,return_dict=True)
            outl.copy_(oo.logits); input_ids.copy_(outl.argmax(-1)); cache_pos+=1
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            oo=model(input_ids=input_ids,cache_position=cache_pos,past_key_values=cache,use_cache=True,return_dict=True)
        outl.copy_(oo.logits); input_ids.copy_(outl.argmax(-1))
    toks=[]
    with Power() as pw:
        t0=time.time()
        for _ in range(n_replay):
            cache_pos+=1; g.replay()
            if collect: torch.cuda.synchronize(); toks.append(input_ids[:, 0].clone())
        torch.cuda.synchronize(); dt=time.time()-t0
    return dt, pw.mean(), (torch.stack(toks,1) if collect else None)

def eager_greedy(n):
    with torch.no_grad(): o=model.generate(ids,attention_mask=attn,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
    return o[:,plen:]

# ---------- PART B: GRAPH INTEGRITY — test BOTH arms (disambiguate harness-drift vs fusion-not-captured) ----------
def integrity(on,label):
    patch(on); r0=fstats()
    eg = eager_greedy(24)
    _,_,gg = build_graph_and_replay(24, collect=True)
    r1=fstats(); patch(False)
    m=(eg==gg).float().mean().item() if gg is not None else -1
    print(f"[B {label}] eager==graph token match={m*100:.1f}%  (rmsnorm {r0[0]}->{r1[0]} silu {r0[1]}->{r1[1]})")
    return m
mb=integrity(False,"baseline ")   # if THIS is also low, the harness greedy isn't eager-equivalent (affects both arms equally)
mf=integrity(True ,"fusion   ")
verdict = "PASS (both match; graph faithful)" if (mb>0.99 and mf>0.99) else ("HARNESS-DRIFT (baseline also diverges -> tok/s still matched-fair, integrity inconclusive)" if mb<0.99 else "FUSION-NOT-CAPTURED (baseline OK, fusion diverges -> fake lift)")
print(f"[B verdict] baseline={mb*100:.1f}% fusion={mf*100:.1f}% -> {verdict}")

# ---------- PART B2: DECISIVE single-step integrity — capture with t1, replay with t2, output must reflect t2 ----------
# If the fused kernel is NOT in the replay (stale buffer), changing the input won't change the logits correctly.
def prefill_cache(maxlen):
    c=StaticCache(config=model.config,max_batch_size=B,max_cache_len=maxlen,device="cuda",dtype=torch.float16)
    with torch.no_grad(): model(input_ids=ids,attention_mask=attn,cache_position=torch.arange(plen,device="cuda"),past_key_values=c,use_cache=True,return_dict=True)
    return c
def ss_integrity(on,label):
    patch(on); ML=plen+8
    cg=prefill_cache(ML)
    iid=torch.full((B,1),100,device="cuda",dtype=torch.long); cpos=torch.tensor([plen],device="cuda")
    outl=torch.empty(B,1,model.config.vocab_size,device="cuda",dtype=torch.float16)
    side=torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad(): oo=model(input_ids=iid,cache_position=cpos,past_key_values=cg,use_cache=True,return_dict=True)
            outl.copy_(oo.logits)
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad(): oo=model(input_ids=iid,cache_position=cpos,past_key_values=cg,use_cache=True,return_dict=True)
        outl.copy_(oo.logits)
    iid.copy_(torch.full((B,1),5000,device="cuda",dtype=torch.long)); g.replay(); torch.cuda.synchronize()
    Lg=outl.float().clone()
    cr=prefill_cache(ML)
    with torch.no_grad(): Lr=model(input_ids=torch.full((B,1),5000,device="cuda",dtype=torch.long),cache_position=torch.tensor([plen],device="cuda"),past_key_values=cr,use_cache=True,return_dict=True).logits.float()
    patch(False)
    a=(Lg.argmax(-1)==Lr.argmax(-1)).float().mean().item(); r=(Lg-Lr).abs().max().item()/Lr.abs().max().item()
    print(f"[B2 {label}] replay(t2)==eager(t2) argmax={a*100:.1f}% max_rel={r:.3e}")
    return a
ssb=ss_integrity(False,"baseline"); ssf=ss_integrity(True,"fusion  ")
print(f"[B2 verdict] baseline={ssb*100:.1f}% fusion={ssf*100:.1f}% -> "+("BOTH FAIL = harness cache-state issue, not fusion (graph still does real work, tok/s matched-fair)" if ssb<0.99 else ("graph FAITHFUL" if ssf>0.99 else "FUSION stale-buffer = graph lift FAKE")))

# ---------- PART C: speed (graph headline + eager context), both arms ----------
NREP=int(os.environ.get("NREP","300"))
def time_eager(n):
    with Power() as pw:
        t0=time.time()
        with torch.no_grad(): model.generate(ids,attention_mask=attn,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); dt=time.time()-t0
    return B*n/dt, pw.mean()
def runarm(on):
    patch(on)
    for _ in range(1): build_graph_and_replay(8)   # warm
    dt,W,_ = build_graph_and_replay(NREP)
    g_tps=B*NREP/dt; e_tps,eW=time_eager(48)
    patch(False)
    return dict(graph_tps=g_tps,graph_W=W,graph_tokW=g_tps/W if W else 0,eager_tps=e_tps,eager_W=eW)
bse=runarm(False); fus=runarm(True)
def d(a,b): return 100*(b/a-1) if a else 0
print(f"[C baseline]  graph tps={bse['graph_tps']:.1f} W={bse['graph_W']:.1f} tok/s/W={bse['graph_tokW']:.4f} | eager tps={bse['eager_tps']:.1f}")
print(f"[C fusion ]  graph tps={fus['graph_tps']:.1f} W={fus['graph_W']:.1f} tok/s/W={fus['graph_tokW']:.4f} | eager tps={fus['eager_tps']:.1f}")
print(f"[C LIFT B={B}] GRAPH tok/s {d(bse['graph_tps'],fus['graph_tps']):+.1f}%  tok/s/W {d(bse['graph_tokW'],fus['graph_tokW']):+.1f}%  | EAGER tok/s {d(bse['eager_tps'],fus['eager_tps']):+.1f}% (context, incl launch savings)")
