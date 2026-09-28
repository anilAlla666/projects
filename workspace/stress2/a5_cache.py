"""A5 — multi-model cache isolation.

Llama-3.2-1B → gen 50 → del → Llama-3.1-8B → gen 50 → del → Llama-3.2-1B → gen 50.
Third pass output token sequence must match first (temp=0).
"""
import os, sys, json, time, gc, hashlib, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
PROMPT = "The future of GPU computing is"


def gen_pass(rt, model_path, tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map={"": "cuda:0"})
    model.requires_grad_(False); model.train(False)
    load_s = time.perf_counter() - t0
    if USE_CIPHER and rt is not None:
        try:
            sc.patch_fusion(rt, model)
        except Exception as e:
            print(f"  [A5 {tag}] patch_fusion error: {e}", flush=True)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id,
                            use_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(ids, attention_mask=attn, max_new_tokens=50,
                              do_sample=False, pad_token_id=tok.pad_token_id,
                              use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    seq = out[0, ids.shape[1]:].tolist()
    text = tok.decode(seq, skip_special_tokens=False)[:120]
    h = hashlib.sha256(json.dumps(seq).encode()).hexdigest()[:16]
    del model, tok, ids, attn, out
    gc.collect(); torch.cuda.empty_cache()
    return dict(tag=tag, model_path=model_path, load_s=load_s,
                elapsed_s=elapsed, hash=h, seq=seq, text=text)


def main():
    rt, fp8_stats, _ = (sc.init_cipher() if USE_CIPHER
                       else (None, lambda: {}, lambda: {}))
    print(f"[A5] cipher={USE_CIPHER}", flush=True)
    rows = []
    P1B = "/home/ubuntu/models/Llama-3.2-1B"
    P8B = "/home/ubuntu/models/Llama-3.1-8B"
    for tag, path in [("first_1B", P1B), ("middle_8B", P8B), ("third_1B", P1B)]:
        print(f"[A5] {tag}", flush=True)
        try:
            r = gen_pass(rt, path, tag)
        except Exception as e:
            r = dict(tag=tag, error=f"{type(e).__name__}: {str(e)[:200]}")
        rows.append(r)
        if "hash" in r:
            print(f"  hash={r['hash']} text={r['text']!r}", flush=True)

    same_hash = (rows[0].get("hash") == rows[2].get("hash")
                 if "hash" in rows[0] and "hash" in rows[2] else False)
    fp8 = fp8_stats() if USE_CIPHER else {}

    suffix = "_baseline" if not USE_CIPHER else ""
    out_path = os.path.join(os.path.dirname(__file__), f"a5_cache{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(dict(cipher=USE_CIPHER, rows=rows,
                       first_third_match=same_hash, fp8_final=fp8),
                  f, indent=2)
    print(f"[A5] first vs third hash match: {same_hash}", flush=True)
    print(f"[A5] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
