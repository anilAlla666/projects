"""T4.2.4e prefill-with-latency driver.

Same shape as wl_noisy_neighbor_bomb.py (Mistral-7B prefill B=8) but
captures per-prefill wall-clock latencies via time.perf_counter()
around the m(**inputs) call. Used as both tenants in the SM-bound
2-prefill noisy-neighbor A/B.

Outputs:
  /tmp/cipher_tenant_<tid>_progress  (same as wl01)
  /tmp/<tid>_prefill_latencies.json  (per-prefill wall-clock ms)
"""
import sys, os, time, json
import torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "120"))
MODEL = os.environ.get("WL_MODEL", "mistralai/Mistral-7B-v0.1")
TID = os.environ.get("CIPHER_TENANT_ID", "tpf")
PROMPT_LEN = 1024
LATENCY_JSON = f"/tmp/{TID}_prefill_latencies.json"

tenant = tenant_register.register(TID)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
m.train(False)

long_prompt = ("The quick brown fox jumps over the lazy dog. " * 30)[:PROMPT_LEN]
prompts = [long_prompt] * 8
inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
             max_length=PROMPT_LEN).to("cuda")

# Per-prefill wall-clock latency (ms), measured around the synchronous-completed
# forward pass. cuda synchronize is needed because m(...) returns when the
# kernel is queued, not when it completes.
all_prefill_latencies_ms = []


def step():
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(**inputs)
    # Force completion so the wall-clock measures actual GPU time.
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    all_prefill_latencies_ms.append((t1 - t0) * 1000.0)
    return 8, int(inputs.input_ids.numel())


run_for_duration.run(step, DURATION, "WLNN_TP", tenant, "prefills")

if all_prefill_latencies_ms:
    payload = {
        "tenant_id": TID,
        "model": MODEL,
        "batch_size": 8,
        "prompt_len": PROMPT_LEN,
        "duration_s": DURATION,
        "n_prefills": len(all_prefill_latencies_ms),
        "prefill_latency_ms": all_prefill_latencies_ms,
    }
else:
    payload = {"tenant_id": TID, "error": "no prefills recorded"}

with open(LATENCY_JSON, "w") as f:
    json.dump(payload, f)

print(f"[wl_prefill_with_latency] wrote {LATENCY_JSON} with "
      f"{len(all_prefill_latencies_ms)} latency samples",
      file=sys.stderr, flush=True)
