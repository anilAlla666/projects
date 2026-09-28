"""A4 — fp16 vs bf16 with CIPHER.

Loads Llama-3.1-8B in each dtype, generates 50 tokens, reports whether
each works and what (if anything) crashes.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
PROMPT = "The future of GPU computing is"


def run_dtype(dtype_name):
    import torch, ctypes
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    tdtype = dtype_map[dtype_name]
    rt, fp8_stats, fus_stats = (sc.init_cipher() if USE_CIPHER
                                else (None, lambda: {}, lambda: {}))
    tok = AutoTokenizer.from_pretrained(sc.MODEL_PATH)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    try:
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            sc.MODEL_PATH, torch_dtype=tdtype,
            device_map={"": "cuda:0"})
        model.requires_grad_(False); model.train(False)
        load_s = time.perf_counter() - t0
        if USE_CIPHER and rt is not None and tdtype == torch.float16:
            sc.patch_fusion(rt, model)  # patches assume fp16; bf16 falls through
    except Exception as e:
        return dict(dtype=dtype_name, load_error=f"{type(e).__name__}: {str(e)[:200]}")

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)
    try:
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
        n_new = out.shape[1] - ids.shape[1]
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        result = dict(dtype=dtype_name, load_s=load_s, elapsed_s=elapsed,
                      tokens=n_new, tps=n_new/elapsed,
                      first_text=text[:120],
                      coherent=("!!!!" not in text),
                      fp8_delta=fp8_stats(),
                      fusion_delta=fus_stats())
    except Exception as e:
        result = dict(dtype=dtype_name, load_s=load_s,
                      gen_error=f"{type(e).__name__}: {str(e)[:200]}")
    del model
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    print(f"[A4] cipher={USE_CIPHER}", flush=True)
    rows = []
    for dt in ("fp16", "bf16"):
        print(f"[A4] running {dt}", flush=True)
        r = run_dtype(dt)
        rows.append(r)
        print(f"  -> {r}", flush=True)

    suffix = "_baseline" if not USE_CIPHER else ""
    out_path = os.path.join(os.path.dirname(__file__), f"a4_dtype{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(dict(cipher=USE_CIPHER, rows=rows), f, indent=2)
    print(f"[A4] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
