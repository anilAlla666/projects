"""CP 4.7 — launch-count spot-check (grounds the design-memo bound).

Mistral-7B fp16 B=1 decode under the cipher_rt_phase4 substrate (c2c5d313).
The substrate's CUPTI hook counts every kernel launch and the matmul substrate
counts every cublasGemmEx; both print at teardown (DIAG-T4.2.4d total_launches,
MATMUL exit totals). We decode a known token count and divide.

fp16 (not Marlin INT4) is deliberate: the launch topology -- specifically the
elementwise/reduce kernels that are the fusion targets (RMSNorm, RoPE, SiLU,
residual add) -- is identical under Marlin; Marlin only swaps the GEMM kernel
itself (one cublasGemmEx-equivalent either way). fp16 avoids the Marlin B=1
hang risk (CP 2.4 MARLIN_HANG) for a pure launch-count measurement.
"""
import os, time, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "/home/ubuntu/models/Mistral-7B-v0.1"
PREFILL = 128
WARMUP = 8
DECODE = 256
FWD_TOTAL = 1 + WARMUP + DECODE          # prefill + warmup + measured decode
OUT = "/home/ubuntu/cipher-fusion-evidence/cp_4_7/cp47_launch_probe_result.json"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                             device_map="cuda:0")
model.eval()
base = tok("Energy efficiency means doing more useful work per watt. ",
           return_tensors="pt").input_ids
reps = (PREFILL // base.shape[1]) + 1
prompt = base.repeat(1, reps)[:, :PREFILL].to("cuda:0")

cache = StaticCache(config=model.config, max_batch_size=1,
                    max_cache_len=PREFILL + DECODE + 8,
                    device="cuda:0", dtype=torch.float16)

with torch.no_grad():
    out = model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
    nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        out = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(DECODE):
        out = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

tok_s = DECODE / wall
ms_per_tok = wall / DECODE * 1000
doc = {
    "cp": "4.7", "probe": "launch-count spot-check",
    "model": "Mistral-7B-v0.1 fp16", "batch": 1,
    "prefill_tokens": PREFILL, "warmup_steps": WARMUP,
    "decode_tokens_measured": DECODE,
    "forward_passes_total": FWD_TOTAL,
    "decode_tok_s": round(tok_s, 2),
    "ms_per_token": round(ms_per_tok, 3),
    "note": "launches/forward = substrate DIAG-T4.2.4d total_launches / "
            "forward_passes_total (read from stderr runlog)",
}
json.dump(doc, open(OUT, "w"), indent=2)
print(f"[cp47] decode {tok_s:.2f} tok/s  {ms_per_tok:.3f} ms/token  "
      f"(measured {DECODE} steps; total fwd passes = {FWD_TOTAL})", flush=True)
print(f"[cp47] wrote {OUT}", flush=True)
