"""Per-tenant worker for the N=2 same-prompt KV-dedup test.

Launched as a subprocess by sc_kvdedup_n2.py. Loads vLLM TinyLlama,
prefills the shared system prompt, decodes a few tokens, then waits
for SIGUSR1 to flush via cipher_vllm_kvdedup.

Args (positional):
  tenant_num     int   — CIPHER_TENANT_NUM env value (1 or 2)
  out_dir        path  — where to write the result file
"""
import argparse, json, os, signal, sys, time

p = argparse.ArgumentParser()
p.add_argument("tenant_num", type=int)
p.add_argument("out_dir")
args = p.parse_args()

os.environ["CIPHER_KV_ALLOC"]    = "1"
os.environ["CIPHER_KVDEDUP"]     = "1"
os.environ["CIPHER_TENANT_NUM"]  = str(args.tenant_num)

# Shared system prompt — long enough to fill multiple 2 MiB KV pages.
# At 4 KiB/token/layer × 22 layers (TinyLlama) = ~88 KiB/token across all layers.
# 4 KiB pattern per 2 MiB page = 512 tokens per page per layer.
# For a 2K-token prompt across 22 layers: 22 × (2K × 4 KiB / 2 MiB) = ~88 pages.
SHARED = ("The CIPHER substrate provides cross-tenant key-value cache "
          "deduplication via a kernel-resident refcount table and content-"
          "addressable physical pages. " * 32)  # ~1500 tokens

print(f"[t{args.tenant_num}] start; prompt={len(SHARED)} chars")

import vllm
MODEL = os.environ.get("CIPHER_KVDEDUP_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MAX_LEN = int(os.environ.get("CIPHER_KVDEDUP_MAX_LEN", "2048"))
GPU_UTIL = float(os.environ.get("CIPHER_KVDEDUP_GPU_UTIL", "0.18"))
print(f"[t{args.tenant_num}] model={MODEL} max_len={MAX_LEN} gpu_util={GPU_UTIL}")
llm = vllm.LLM(
    model=MODEL,
    max_model_len=MAX_LEN,
    gpu_memory_utilization=GPU_UTIL,
    enforce_eager=True,
)
print(f"[t{args.tenant_num}] vllm loaded")

# Prefill + decode 16 tokens. Greedy + same prompt = identical token_ids
# across tenants (the bit-identity regression guard).
sp = vllm.SamplingParams(max_tokens=16, temperature=0)
out = llm.generate([SHARED], sp)
tokens = list(out[0].outputs[0].token_ids)
text = out[0].outputs[0].text
print(f"[t{args.tenant_num}] decode ok: tokens={tokens} text={text!r}")

# Write a 'ready' marker so the orchestrator can sequence the SIGUSR1
# flushes (tenant 1 first, then tenant 2 — so tenant 2 sees tenant 1's
# pages in the kmod table).
ready_file = os.path.join(args.out_dir, f"t{args.tenant_num}_ready.txt")
with open(ready_file, "w") as f:
    f.write(f"{os.getpid()}\n")
    f.write(",".join(str(t) for t in tokens) + "\n")
    f.write(text + "\n")
print(f"[t{args.tenant_num}] ready: {ready_file}")

# Wait for the orchestrator to send SIGUSR1 via cipher_vllm_kvdedup
# (which writes the dedup_now() result to /tmp/cipher_kvdedup_result_t<N>.json),
# then for an explicit 'done' marker so we can exit cleanly without
# tearing down vLLM mid-flush.
done_file = os.path.join(args.out_dir, f"t{args.tenant_num}_done.signal")
print(f"[t{args.tenant_num}] waiting for {done_file}...")
while not os.path.exists(done_file):
    time.sleep(0.2)
print(f"[t{args.tenant_num}] done signal received; exiting")
