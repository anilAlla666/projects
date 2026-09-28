#!/usr/bin/env python3
# G3 Marlin @ B>=8 bf16 DECODE — torch-HF single-process arm.
#   ARM=off : bf16 reference. Generates the reference token seq + saves per-decode-step
#             reference logits. Times free-running decode (tok/s, tok/W).
#   ARM=on  : CIPHER_MARLIN=on. Times free-running decode. Then TEACHER-FORCES over the
#             OFF arm's reference token seq (identical inputs both arms) and saves ON logits.
#             Reads Marlin/dispatch counters (handled>0 = substitutes; M-gate is call->n=batch).
# Correctness is judged teacher-forced (advisor): matched inputs, per-step argmax-agreement + KL.
# The free-running gen-token match is reported too (the user's literal "output matches OFF ref").
import os, sys, json, time, ctypes, threading
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ARM   = os.environ["ARM"]                      # "off" | "on"
MODEL = os.environ.get("MODEL", "/home/ubuntu/models/TinyLlama-1.1B")
B     = int(os.environ.get("B", "8"))
NEW   = int(os.environ.get("NEW", "96"))       # decode steps (>=75 so koopman obs reaches 10000-ish elsewhere)
WARM  = int(os.environ.get("WARM", "8"))       # warmup decode steps to pass STABILITY_THRESHOLD=4 + quantize
OUT   = os.environ.get("OUT", "/home/ubuntu/marlin_b8_torch")
SO    = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
os.makedirs(OUT, exist_ok=True)

lib = ctypes.CDLL(SO)
def _u(name):
    try:
        f = getattr(lib, name); f.restype = ctypes.c_ulong; return int(f())
    except Exception: return -1
def counters():
    return {
        "matmul_total":      _u("cipher_rt_matmul_calls_total"),
        "matmul_handled":    _u("cipher_rt_matmul_calls_handled"),
        "matmul_passthrough":_u("cipher_rt_matmul_calls_passthrough"),
        "marlin_handled":    _u("cipher_rt_marlin_calls_handled"),
        "marlin_bf16_observed":   _u("cipher_rt_marlin_calls_bf16_observed"),
        "marlin_bf16_substituted":_u("cipher_rt_marlin_calls_bf16_substituted"),
        "marlin_skipped":    _u("cipher_rt_marlin_calls_skipped"),
        "cublas_shim_calls": _u("cipher_rt_cublas_shim_calls"),
        "cublaslt_shim_calls": _u("cipher_rt_cublaslt_shim_calls"),
        "marlin_active":     _u("cipher_rt_marlin_is_active"),
    }

# --- power sampler ---
import pynvml; pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
class Power:
    def __init__(s): s.on=False; s.samples=[]
    def _loop(s):
        while s.on:
            try: s.samples.append(pynvml.nvmlDeviceGetPowerUsage(H)/1000.0)
            except Exception: pass
            time.sleep(0.02)
    def __enter__(s): s.on=True; s.t=threading.Thread(target=s._loop); s.t.start(); return s
    def __exit__(s,*a): s.on=False; s.t.join()
    def mean(s): return float(np.mean(s.samples)) if s.samples else 0.0

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
prompt = "The history of computing spans several distinct eras, each defined by"
ids = tok([prompt]*B, return_tensors="pt", padding=True).input_ids.cuda()
gen_kw = dict(max_new_tokens=NEW, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id)

# --- warmup: trigger lazy-quantize (>=4 obs/weight) so timed run is fully substituted ---
with torch.no_grad():
    m.generate(ids, max_new_tokens=WARM, do_sample=False, pad_token_id=tok.pad_token_id)
torch.cuda.synchronize()
c_after_warm = counters()

# --- timed free-running decode: tok/s + tok/W (matched OFF vs ON) ---
torch.cuda.synchronize()
with Power() as pw:
    t0 = time.time()
    with torch.no_grad():
        out = m.generate(ids, **gen_kw, return_dict_in_generate=True, output_scores=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
seq = out.sequences                         # [B, prompt+NEW]
gen_tokens = seq[:, ids.shape[1]:]          # [B, NEW]
n_decoded = B * NEW
tok_s = n_decoded / dt
meanW = pw.mean()
c_after_timed = counters()

# --- teacher-forced per-step logit capture (decode regime, M=B) ---
# OFF: build reference token seq (greedy, manual decode loop) + save ref logits.
# ON : load OFF ref tokens, feed identical tokens, capture ON logits.
ref_path = f"{OUT}/ref_tokens.npy"
def manual_decode_capture(teacher=None):
    """Returns (chosen_tokens[B,T], logits[B,T,V]). If teacher given, feed those tokens."""
    with torch.no_grad():
        o = m(ids, use_cache=True)
        past = o.past_key_values
        last = o.logits[:, -1, :]                      # [B,V]
        chosen, logmat = [], []
        T = NEW if teacher is None else teacher.shape[1]
        for t in range(T):
            logmat.append(last.float().cpu().numpy())  # logits that PREDICT step t
            nxt = teacher[:, t:t+1] if teacher is not None else last.argmax(-1, keepdim=True)
            chosen.append(nxt.squeeze(1).cpu().numpy())
            o = m(nxt, past_key_values=past, use_cache=True)
            past = o.past_key_values
            last = o.logits[:, -1, :]
    return np.stack(chosen, 1), np.stack(logmat, 1)    # [B,T], [B,T,V]

if ARM == "off":
    ref_tokens, ref_logits = manual_decode_capture(teacher=None)
    np.save(ref_path, ref_tokens)
    np.save(f"{OUT}/off_logits.npy", ref_logits)
else:
    ref_tokens = np.load(ref_path)
    teacher = torch.tensor(ref_tokens, device="cuda")
    _, on_logits = manual_decode_capture(teacher=teacher)
    np.save(f"{OUT}/on_logits.npy", on_logits)
c_final = counters()

res = {
    "arm": ARM, "model": MODEL, "B": B, "new_tokens": NEW, "warm": WARM,
    "wall_s": round(dt,4), "tok_s": round(tok_s,2), "tokens": n_decoded,
    "mean_power_W": round(meanW,2),
    "tok_per_s_per_W": round(tok_s/meanW,5) if meanW else 0,
    "joule_per_tok": round(meanW*dt/n_decoded,5) if n_decoded else 0,
    "gen_tokens": gen_tokens[:, :12].cpu().tolist(),
    "counters_after_warmup": c_after_warm,
    "counters_after_timed": c_after_timed,
    "counters_final": c_final,
}
json.dump(res, open(f"{OUT}/{ARM}_metrics.json","w"), indent=2)
print(f"[{ARM}] tok/s={tok_s:.1f} meanW={meanW:.1f} tok/s/W={res['tok_per_s_per_W']} "
      f"matmul handled={c_final['matmul_handled']} pass={c_final['matmul_passthrough']} "
      f"marlin_bf16_sub={c_final['marlin_bf16_substituted']} lt_shim={c_final['cublaslt_shim_calls']}", flush=True)
