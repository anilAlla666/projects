"""Op 9 AUDIT smoke. modes: baseline | audit | audit_s2b.

baseline   : no preload — reference token hashes
audit      : libcipher_rt LD_PRELOAD, CIPHER_AUDIT on
audit_s2b  : audit + op#1 S2b CIPHER KV cache (prior-op regression)

6 new tokens x 3 prompts keeps total dispatch events < 8192 so the audit
ring holds the whole history and the chain is fully verifiable.
"""
import os, sys, json, hashlib, torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "baseline"
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")

if MODE == "audit_s2b":
    import cipher_kv_cache
    assert cipher_kv_cache.install()

from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

PROMPTS = ["The capital of France is",
           "Quantum entanglement is",
           "The Roman Empire collapsed because"]
MAX_NEW = 6

tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/Mistral-7B-v0.1")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/models/Mistral-7B-v0.1", torch_dtype=torch.float16,
    attn_implementation="sdpa").cuda()
model.train(False)

results = []
with torch.inference_mode():
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        cache = StaticCache(config=model.config,
                            max_cache_len=int(ids.shape[1]) + MAX_NEW + 1)
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                             past_key_values=cache,
                             pad_token_id=tok.eos_token_id)
        h = hashlib.sha256(json.dumps(out[0].tolist()).encode()).hexdigest()[:16]
        results.append({"prompt": p, "hash": h})
        print(f"[{MODE}] {p!r:40s} hash={h}")

with open(f"/tmp/op2_{MODE}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"[{MODE}] wrote /tmp/op2_{MODE}.json")
