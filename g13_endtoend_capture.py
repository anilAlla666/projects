#!/usr/bin/env python3
# GATE-1 g1.3 END-TO-END (the test the brief asks): the SHIPPED Koopman path inside the ENGINE's real static-KV
# torch.cuda.graph capture. LD_PRELOAD=$SO + CIPHER_KOOPMAN={1 test|0 control}. Real prefill -> graph-decode capture
# (the inc-1/g1.2 harness). Engaged the way it really engages (NOT hand-fed distinct pointers). Witness = the SHIPPED
# cipher_edmd_live_report() (per-shape rows/registered/failed) + koopman counters -> did any shape engage or go inert?
# Two honest outcomes: (a) capture THROWS -> name the op from THAT failure; (b) capture SUCCEEDS because actuator went
# INERT (distinct-input gate never passes on fixed-pointer static-KV activations) -> "doesn't compose, never engages".
# NO CIPHER_RT_DISABLE_AUTO_INIT here -- we WANT koopman_init to run under the preload.
import os, sys, ctypes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
KOOP=os.environ.get("CIPHER_KOOPMAN","?")
PATH=os.environ.get("G13_MODEL","/home/ubuntu/models/TinyLlama-1.1B"); N=48
lib=ctypes.CDLL(SO)
for s in ("cipher_rt_koopman_calls_total","cipher_rt_koopman_calls_handled","cipher_rt_koopman_is_active"):
    getattr(lib,s).restype=ctypes.c_ulong
torch.manual_seed(0)
tok=AutoTokenizer.from_pretrained(PATH); V=tok.vocab_size
m=AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float16, device_map="cuda").eval()
prompt="The history of artificial intelligence began in the 1950s, when researchers first"
pids=tok(prompt,return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; dev=pids.device; Lc=P+N+64
def cl(t): return int(max(0,min(V-1,int(t))))
print(f"[g13-e2e] CIPHER_KOOPMAN={KOOP} koopman_is_active={lib.cipher_rt_koopman_is_active()} model={os.path.basename(PATH)}",flush=True)

@torch.no_grad()
def prefill(c): return m(pids,cache_position=torch.arange(P,device=dev),past_key_values=c,use_cache=True).logits[0,-1].argmax()
# ---- EAGER decode (real warmup; this is where collect would accumulate snapshots if it engages) ----
with torch.no_grad():
    c=StaticCache(config=m.config,max_cache_len=Lc); nt=prefill(c); es=[]
    for i in range(N):
        es.append(cl(nt))
        nt=m(torch.tensor([[cl(nt)]],device=dev),cache_position=torch.tensor([P+i],device=dev),past_key_values=c,use_cache=True).logits[0,-1].argmax()
ct_after_eager=lib.cipher_rt_koopman_calls_total()
print(f"[g13-e2e] after eager warmup: koopman calls_total={ct_after_eager} handled={lib.cipher_rt_koopman_calls_handled()}",flush=True)

# ---- CAPTURE the decode step into the engine's static-KV graph (fixed activation buffer = the real regime) ----
# Instrument: report() BEFORE and AFTER the captured forward; diff per-shape calls= tells us whether the shipped
# collect fires DURING capture (mechanism check -- do NOT assert "drains pre-capture" without this).
sys.stderr.write("[g13-e2e] --- report() BEFORE capture block ---\n"); sys.stderr.flush()
try: lib.cipher_edmd_live_report()
except Exception: pass
cap_err=None; cap_ok=False; match=None
try:
    cache=StaticCache(config=m.config,max_cache_len=Lc); t1=prefill(cache)
    sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(cl(t1)); spos=torch.tensor([P],device=dev)
    g=torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        slog=m(sin,cache_position=spos,past_key_values=cache,use_cache=True).logits
    gs=[cl(t1)]; sin.fill_(cl(t1)); spos.fill_(P)
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize(); nt=cl(slog[0,-1].argmax()); gs.append(nt); sin.fill_(nt); spos.add_(1)
    match=sum(a==b for a,b in zip(gs,es)); cap_ok=True
    print(f"[g13-e2e] CAPTURE+REPLAY SUCCEEDED. graph-vs-eager match={match}/{N}",flush=True)
    sys.stderr.write("[g13-e2e] --- report() AFTER capture block (diff calls= vs BEFORE = collect fired during capture) ---\n"); sys.stderr.flush()
    try: lib.cipher_edmd_live_report()
    except Exception: pass
except Exception as e:
    cap_err=f"{type(e).__name__}: {str(e)[:240]}"
    print(f"[g13-e2e] CAPTURE FAILED -> {cap_err}",flush=True)

ct=lib.cipher_rt_koopman_calls_total(); hd=lib.cipher_rt_koopman_calls_handled()
print(f"[g13-e2e] FINAL koopman calls_total={ct} handled={hd} (substitutions={hd})",flush=True)
# SHIPPED witness: per-shape EDMD-live state (rows/registered/failed) -> engaged or inert?
sys.stderr.write("[g13-e2e] cipher_edmd_live_report() (shipped per-shape witness):\n"); sys.stderr.flush()
try:
    lib.cipher_edmd_live_report()
except Exception as e:
    print(f"  report() unavailable: {e}",flush=True)
print(f"[g13-e2e VERDICT] KOOP={KOOP} capture_ok={cap_ok} capture_err={cap_err}",flush=True)
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
