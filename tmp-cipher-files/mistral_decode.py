"""V2+V3+V4 combined for Mistral-7B autoregressive decode B=1 and B=8.

This is Marlin's designed regime: M_marlin (cuBLAS N) = batch ≤ 64.
Decode at B=8 → cuBLAS N=8 per step → Marlin gate passes.
"""
import os, time, json, sys
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "The future of GPU computing is to make every joule"
MODE = sys.argv[1] if len(sys.argv) > 1 else "fp16"  # fp16 or marlin
OUT  = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/mistral_decode_{MODE}.pt"
NTOK = int(os.environ.get("WL_NTOK", "32"))

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"[{MODE}] load...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
mdl.train(False)
print(f"[{MODE}] loaded {time.time()-t0:.1f}s", flush=True)

def decode(batch, ntok):
    p = tok(PROMPT, return_tensors="pt").to("cuda")
    ids = p.input_ids.repeat(batch, 1)
    msk = p.attention_mask.repeat(batch, 1)
    with torch.no_grad():
        out = mdl.generate(
            input_ids=ids, attention_mask=msk,
            max_new_tokens=ntok, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize()
    return out

# Warmup: decode B=1 (3 times) then B=8 (3 times) to cross stability + quantize
print(f"[{MODE}] === warmup decode B=1 × 3 ===", flush=True)
for i in range(3):
    t0 = time.time()
    out = decode(1, NTOK)
    ms = (time.time()-t0)*1000
    print(f"  warm B=1 #{i+1}: {ms:.0f}ms argmax_last={int(out[0,-1])}", flush=True)
print(f"[{MODE}] === warmup decode B=8 × 3 ===", flush=True)
for i in range(3):
    t0 = time.time()
    out = decode(8, NTOK)
    ms = (time.time()-t0)*1000
    print(f"  warm B=8 #{i+1}: {ms:.0f}ms argmax_b0_last={int(out[0,-1])}", flush=True)

# Measure steady state: 5 each
print(f"[{MODE}] === measure decode B=1 × 5 ===", flush=True)
ms_b1 = []
out_b1 = None
for i in range(5):
    t0 = time.time()
    out_b1 = decode(1, NTOK)
    ms_b1.append((time.time()-t0)*1000)
print(f"  B=1 mean: {sum(ms_b1)/len(ms_b1):.0f}ms tokens={out_b1.shape[-1]-tok(PROMPT,return_tensors='pt').input_ids.shape[-1]}", flush=True)

print(f"[{MODE}] === measure decode B=8 × 5 ===", flush=True)
ms_b8 = []
out_b8 = None
for i in range(5):
    t0 = time.time()
    out_b8 = decode(8, NTOK)
    ms_b8.append((time.time()-t0)*1000)
print(f"  B=8 mean: {sum(ms_b8)/len(ms_b8):.0f}ms tokens_per_batch_member={out_b8.shape[-1]-tok(PROMPT,return_tensors='pt').input_ids.shape[-1]}", flush=True)

# Compute tok/s — total tokens generated per second across all batch members
import statistics as _s
prompt_len = tok(PROMPT, return_tensors="pt").input_ids.shape[-1]
new_tok_b1 = (out_b1.shape[-1] - prompt_len) * 1  # batch 1
new_tok_b8 = (out_b8.shape[-1] - prompt_len) * 8  # batch 8

result = {
    "mode": MODE,
    "ntok_per_member": NTOK,
    "prompt_len": int(prompt_len),
    "b1_ms_mean": round(_s.mean(ms_b1), 1),
    "b1_ms_stddev": round(_s.pstdev(ms_b1), 2),
    "b1_new_tokens_total": int(new_tok_b1),
    "b1_tok_per_s": round(new_tok_b1 * 1000.0 / _s.mean(ms_b1), 2),
    "b8_ms_mean": round(_s.mean(ms_b8), 1),
    "b8_ms_stddev": round(_s.pstdev(ms_b8), 2),
    "b8_new_tokens_total": int(new_tok_b8),
    "b8_tok_per_s": round(new_tok_b8 * 1000.0 / _s.mean(ms_b8), 2),
    "b1_first_3_tokens": [int(x) for x in out_b1[0, prompt_len:prompt_len+3]],
    "b8_first_3_tokens_b0": [int(x) for x in out_b8[0, prompt_len:prompt_len+3]],
}
torch.save({"result": result,
            "out_b1": out_b1.cpu(),
            "out_b8": out_b8.cpu()}, OUT)
print(json.dumps(result, indent=2), flush=True)
print(f"[{MODE}] saved {OUT}", flush=True)
