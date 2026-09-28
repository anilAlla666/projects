#!/usr/bin/env python3
"""Heavy-prefill tenant: 2048-token prompt, generate 1 token, repeat.

Used as the noisy neighbor for the SHIELD test.
"""
import os, json, time, ctypes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings; warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    tid = int(os.environ.get("CIPHER_TENANT_ID", "0"))
    duration_s = float(os.environ.get("MT_DURATION_S", "30"))
    out_path = os.environ["MT_OUT_PATH"]
    model_path = os.environ.get("MT_MODEL_PATH",
                                "/home/ubuntu/models/Llama-3.2-1B")
    prefill_len = int(os.environ.get("MT_HEAVY_PREFILL", "2048"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    if os.environ.get("MT_LOAD_RT", "1") != "0":
        try:
            ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"),
                         mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0})

    BASE = ("Energy efficiency means doing more useful work per watt. "
            "The future of GPU computing is to make every joule count. ")
    ids_one = tok(BASE, return_tensors="pt").input_ids[0]
    n_tile = (prefill_len + len(ids_one) - 1) // len(ids_one)
    prompt = ids_one.repeat(n_tile)[:prefill_len].unsqueeze(0).to("cuda:0")

    n_iter = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration_s:
        cache = StaticCache(config=model.config, max_cache_len=prefill_len + 4)
        with torch.no_grad():
            cp = torch.arange(prefill_len, device="cuda:0", dtype=torch.long)
            o = model(input_ids=prompt, cache_position=cp,
                      past_key_values=cache, use_cache=True, return_dict=True)
            _ = o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        n_iter += 1
    elapsed = time.perf_counter() - t_start

    json.dump(dict(
        tenant_id=tid, mode="heavy_prefill",
        prefill_len=prefill_len, n_iter=n_iter,
        elapsed_s=elapsed,
        prefill_per_sec=n_iter/elapsed if elapsed > 0 else 0,
    ), open(out_path, "w"), indent=2)
    print(f"[heavy prefill tenant {tid}] {n_iter} iters in {elapsed:.1f}s "
          f"({n_iter/elapsed:.2f} prefills/sec, prefill_len={prefill_len})")


if __name__ == "__main__":
    main()
