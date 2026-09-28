"""Per-tenant Mistral-7B worker for Step 3.C KL-gate test.

Same as sc_kvdedup_worker.py but loads Mistral-7B-v0.1 with logprobs=20
capture per token. Writes the 20-logprobs-per-step matrix as npz
alongside the ready file."""
import argparse, json, os, signal, sys, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("tenant_num", type=int)
p.add_argument("out_dir")
args = p.parse_args()

os.environ["CIPHER_KV_ALLOC"]    = "1"
os.environ["CIPHER_KVDEDUP"]     = "1"
os.environ["CIPHER_TENANT_NUM"]  = str(args.tenant_num)
os.environ.setdefault("CIPHER_KVDEDUP_GPU_UTIL", "0.20")

# Shared system prompt — long enough to actually fill content pages
# in the KV cache (Mistral-7B uses 4 KiB/token/layer; we want ~256 KiB
# of content per layer to occupy multiple 2 MiB pages per layer).
# 1500 tokens × 4 KiB × 32 layers / 2 MiB = ~94 content pages across
# 32 layers per tenant.
SHARED = ("The CIPHER substrate provides cross-tenant key-value cache "
          "deduplication via a kernel-resident refcount table and content-"
          "addressable physical pages. Each 2 MiB page is hashed using "
          "xxhash64; collisions are verified with full-page memcmp. " * 24)

K_DECODE  = 32
LOGPROBS  = 20

print(f"[t{args.tenant_num}] start; prompt={len(SHARED)} chars; decode={K_DECODE} tokens; logprobs={LOGPROBS}")

import vllm
llm = vllm.LLM(
    model="mistralai/Mistral-7B-v0.1",
    max_model_len=2048,
    gpu_memory_utilization=float(os.environ.get("CIPHER_KVDEDUP_GPU_UTIL", "0.20")),
    enforce_eager=True,
)
print(f"[t{args.tenant_num}] mistral-7b loaded")

sp = vllm.SamplingParams(max_tokens=K_DECODE, temperature=0, logprobs=LOGPROBS)
out = llm.generate([SHARED], sp)
tokens = list(out[0].outputs[0].token_ids)
text = out[0].outputs[0].text
print(f"[t{args.tenant_num}] decode ok: tokens[:8]={tokens[:8]} text={text[:60]!r}")

# Extract per-step logprobs into a stable matrix. vLLM's output.logprobs
# is List[dict[token_id, Logprob]] of length max_tokens; each dict has
# up to LOGPROBS entries. We collect:
#   chosen_token_ids[step]      — the greedy choice (==tokens[step])
#   topk_token_ids[step, 0..K]  — sorted by logprob descending
#   topk_logprobs[step, 0..K]   — corresponding logprob values
# Across tenants, IF Mistral is deterministic at fp16 with same prompt,
# the matrices should be byte-identical.
lp_list = out[0].outputs[0].logprobs
n_steps = len(lp_list)
topk_ids = np.full((n_steps, LOGPROBS), -1, dtype=np.int64)
topk_lps = np.full((n_steps, LOGPROBS), -1e9, dtype=np.float64)
for step, lp_dict in enumerate(lp_list):
    if lp_dict is None: continue
    # lp_dict: dict[int token_id -> Logprob obj with .logprob]
    sorted_items = sorted(lp_dict.items(),
                          key=lambda kv: kv[1].logprob, reverse=True)
    for k, (tid, lp) in enumerate(sorted_items[:LOGPROBS]):
        topk_ids[step, k] = tid
        topk_lps[step, k] = lp.logprob

npz_path = os.path.join(args.out_dir, f"t{args.tenant_num}_logprobs.npz")
np.savez(npz_path, tokens=np.array(tokens, dtype=np.int64),
         topk_ids=topk_ids, topk_logprobs=topk_lps)
print(f"[t{args.tenant_num}] logprobs saved: {npz_path}")

ready_file = os.path.join(args.out_dir, f"t{args.tenant_num}_ready.txt")
with open(ready_file, "w") as f:
    f.write(f"{os.getpid()}\n")
    f.write(",".join(str(t) for t in tokens) + "\n")
    f.write(text + "\n")
print(f"[t{args.tenant_num}] ready: {ready_file}")

done_file = os.path.join(args.out_dir, f"t{args.tenant_num}_done.signal")
print(f"[t{args.tenant_num}] waiting for {done_file}...")
while not os.path.exists(done_file):
    time.sleep(0.2)
print(f"[t{args.tenant_num}] done signal received; exiting")
