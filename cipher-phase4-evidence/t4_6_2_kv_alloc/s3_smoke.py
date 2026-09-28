"""T4.6.2 S3 — three-indicator smoke test.

Modes (argv[1]):
  baseline : stock transformers StaticCache (torch.zeros KV)
  cipher   : CIPHER-VMM-backed StaticCache via cipher_kv_cache.install()

Both use cache_implementation=static so the ONLY difference is who owns
the KV memory. Emits token-id hashes (byte-identical gate) + the three
indicators in cipher mode.
"""
import os, sys, json, hashlib, torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "baseline"
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")

if MODE == "cipher":
    import cipher_kv_cache
    assert cipher_kv_cache.install(), "install() returned False"

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StaticCache

PROMPTS = [
    "The capital of France is",
    "Quantum entanglement is",
    "The Roman Empire collapsed because",
]
MAX_NEW = 32


def run_model(path, tag, exercise_taginfo=False):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, attn_implementation="sdpa").cuda()
    model.train(False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base_mem = torch.cuda.memory_allocated()

    results = []
    peak_cache = None
    with torch.inference_mode():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            max_len = int(ids.shape[1]) + MAX_NEW + 1
            cache = StaticCache(config=model.config, max_cache_len=max_len)
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 past_key_values=cache,
                                 pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            toks = out[0].tolist()
            h = hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]
            results.append({"prompt": p, "hash": h,
                            "text": tok.decode(out[0], skip_special_tokens=True)})
            print(f"[{tag}] {p!r:40s} hash={h}")
            peak_cache = cache

    mem_after = torch.cuda.memory_allocated()
    kv_mem = mem_after - base_mem
    print(f"[{tag}] torch.cuda.memory_allocated KV delta = "
          f"{kv_mem/1048576:.1f} MiB")

    # Indicator (b): page-tag readback on the last cache's layers.
    if exercise_taginfo and MODE == "cipher":
        import cipher_kv_bridge
        n_layers = len(peak_cache.layers)
        ok = 0
        for i in (0, n_layers // 2, n_layers - 1):
            lyr = peak_cache.layers[i]
            ki = cipher_kv_bridge.page_info(lyr.keys.data_ptr())
            vi = cipher_kv_bridge.page_info(lyr.values.data_ptr())
            kok = ki and ki["layer"] == i and ki["role"] == 0
            vok = vi and vi["layer"] == i and vi["role"] == 1
            ok += int(bool(kok)) + int(bool(vok))
            print(f"[{tag}] layer {i:2d}: K tag={ki}  V tag={vi}")
        print(f"[{tag}] page-tag readback: {ok}/6 correct")

    return results, kv_mem


print(f"=== T4.6.2 S3 smoke — mode={MODE} ===")
mistral, mistral_kv = run_model("/home/ubuntu/models/Mistral-7B-v0.1",
                                f"{MODE}/mistral", exercise_taginfo=True)

if MODE == "cipher":
    import cipher_kv_bridge
    print(f"[{MODE}] allocator stats: {cipher_kv_bridge.get_stats()}")

# Exercise the plain (non-sliding) StaticLayer path via TinyLlama.
print(f"--- {MODE}: TinyLlama (plain StaticLayer path) ---")
tiny, tiny_kv = run_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                          f"{MODE}/tinyllama")

out = {"mode": MODE, "mistral": mistral, "mistral_kv_mib": mistral_kv/1048576,
       "tinyllama": tiny, "tinyllama_kv_mib": tiny_kv/1048576}
with open(f"/tmp/s3_{MODE}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"[{MODE}] wrote /tmp/s3_{MODE}.json")
