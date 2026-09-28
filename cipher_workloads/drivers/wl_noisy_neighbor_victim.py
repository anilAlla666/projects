"""T4.2.4c noisy-neighbor VICTIM driver.

Fork of wl01_decode_b1.py that captures per-token wall-clock timestamps
via a custom HF streamer. Used as the latency-sensitive tenant in the
T4.2.4c partition-ON vs partition-OFF A/B.

Outputs:
  /tmp/cipher_tenant_<tid>_progress  (same as wl01)
  /tmp/<tid>_latencies.json          (per-token wall-clock arrival times)
"""
import sys, os, time, json
import torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "120"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TID = os.environ.get("CIPHER_TENANT_ID", "nn_victim")
LATENCY_JSON = f"/tmp/{TID}_latencies.json"

tenant = tenant_register.register(TID)
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
prompt = tok("Hello, world.", return_tensors="pt").to("cuda")
PROMPT_LEN = int(prompt.input_ids.shape[-1])
MAX_NEW = 32

# Wall-clock arrival times for each generated token, across the whole run.
all_token_times = []


class _TimestampStreamer:
    """Streamer called once per generated token. First put() is the prompt;
    we skip it. Subsequent put() calls receive 1-element tensors per token."""

    def __init__(self, sink):
        self._sink = sink
        self._first = True

    def put(self, tok_tensor):
        if self._first:
            self._first = False
            return
        self._sink.append(time.perf_counter())

    def end(self):
        pass


def step():
    streamer = _TimestampStreamer(all_token_times)
    with torch.no_grad():
        out = m.generate(
            **prompt,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            streamer=streamer,
        )
    return 1, int(out.shape[-1]) - PROMPT_LEN


run_for_duration.run(step, DURATION, "WLNN_V", tenant, "decodes")

if all_token_times:
    t0 = all_token_times[0]
    rel_ms = [round((t - t0) * 1000.0, 3) for t in all_token_times]
    payload = {
        "tenant_id": TID,
        "model": MODEL,
        "max_new_tokens": MAX_NEW,
        "total_new_tokens_recorded": len(all_token_times),
        "duration_s": DURATION,
        "wall_clock_first_token_ts": t0,
        "token_arrival_rel_ms": rel_ms,
    }
else:
    payload = {"tenant_id": TID, "error": "no tokens recorded"}

with open(LATENCY_JSON, "w") as f:
    json.dump(payload, f)

print(f"[wl_noisy_neighbor_victim] wrote {LATENCY_JSON} with "
      f"{len(all_token_times)} token timestamps",
      file=sys.stderr, flush=True)
