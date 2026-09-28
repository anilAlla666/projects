"""T4.6.1 S4 composition check.

Runs Mistral-7B generate(max_new_tokens=32) and emits the output token IDs +
final logits-tip hash. Run once with substrate (CIPHER_ATTN_TEST=on +
CIPHER_VOLT off + CIPHER_MARLIN off) and once baseline (no preload).
Output must be byte-identical token-for-token.
"""
import os, sys, time, json, hashlib, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

mp = "/home/ubuntu/models/Mistral-7B-v0.1"
tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
out_path = f"/tmp/comp_{tag}.json"

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(mp)
m = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float16,
                                         attn_implementation="sdpa").cuda()
m.train(False)

PROMPTS = [
    "The capital of France is",
    "Quantum entanglement is",
    "The Roman Empire collapsed because",
]
results = []
with torch.inference_mode():
    for p in PROMPTS:
        inp = tok(p, return_tensors="pt").input_ids.cuda()
        torch.cuda.synchronize(); t0 = time.time()
        out = m.generate(inp, max_new_tokens=32, do_sample=False,
                         use_cache=True, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); dt = time.time() - t0
        ids = out[0].tolist()
        h = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        results.append({"prompt": p, "token_ids": ids, "hash": h, "gen_s": dt,
                        "text": tok.decode(out[0], skip_special_tokens=True)})
        print(f"[{tag}] {p!r:40s} hash={h} dt={dt:.2f}s")

with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"[{tag}] wrote {out_path}")
