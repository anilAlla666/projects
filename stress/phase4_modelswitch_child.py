#!/usr/bin/env python3
"""Phase 4 — model switch child. Loads Llama-3.2-1B, generates briefly,
unloads, then loads Llama-3.1-8B, generates briefly. Triggers CONTINUITY
checkpoint events on the switch (FP8 cache invalidation, EDMD shape
flush, predictor reset)."""
import os, sys, time, json, gc, ctypes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"]   = "error"
import warnings; warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
RT_PATH = os.path.normpath(os.path.join(ROOT, "..", "libcipher_rt.so"))


def gen(model_path, decode_steps, fusion_on=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0})
    for p in model.parameters(): p.requires_grad_(False)
    PROMPT_LEN = 32
    ids = tok("Energy efficiency means doing more useful work per watt. ",
              return_tensors="pt").input_ids[0]
    ids = ids.repeat(PROMPT_LEN // len(ids) + 1)[:PROMPT_LEN].unsqueeze(0).to("cuda:0")
    n = 0; t0 = time.perf_counter()
    cache = StaticCache(config=model.config,
                        max_cache_len=PROMPT_LEN + decode_steps + 4)
    cp = torch.arange(PROMPT_LEN, device="cuda:0", dtype=torch.long)
    with torch.no_grad():
        o = model(input_ids=ids, cache_position=cp,
                  past_key_values=cache, use_cache=True, return_dict=True)
        input_ids = o.logits[:, -1:].argmax(-1).clone()
        cache_pos = torch.tensor([PROMPT_LEN], device="cuda:0", dtype=torch.long)
        for _ in range(decode_steps):
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
            input_ids.copy_(o.logits.argmax(-1))
            cache_pos += 1
            n += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sample = tok.decode(input_ids[0].tolist(), skip_special_tokens=True)[:60]
    del model, tok, cache
    gc.collect(); torch.cuda.empty_cache()
    return n, elapsed, sample


def main():
    out_path = os.environ["PHASE4_OUT_PATH"]
    duration = float(os.environ.get("PHASE4_DURATION_S", "60"))
    load_rt   = os.environ.get("MT_LOAD_RT", "1") != "0"

    if load_rt and os.path.exists(RT_PATH):
        try:
            ctypes.CDLL(RT_PATH, mode=ctypes.RTLD_GLOBAL)
        except Exception as ex:
            print(f"CIPHER rt load failed: {ex}", file=sys.stderr)

    # Run model A for half the duration, switch, then model B for the rest.
    half = max(20, int(duration / 2))
    print(f"[switch] phase A: Llama-3.2-1B for ~{half}s")
    n1, e1, s1 = gen("/home/ubuntu/models/Llama-3.2-1B", decode_steps=200)
    print(f"[switch] phase A done: {n1} tokens in {e1:.1f}s, sample='{s1}'")

    print(f"[switch] freeing 1B; loading 8B (CONTINUITY checkpoint event)")
    n2, e2, s2 = gen("/home/ubuntu/models/Llama-3.1-8B", decode_steps=50)
    print(f"[switch] phase B done: {n2} tokens in {e2:.1f}s, sample='{s2}'")

    json.dump(dict(
        phase_a_tokens=n1, phase_a_elapsed=e1, phase_a_sample=s1,
        phase_b_tokens=n2, phase_b_elapsed=e2, phase_b_sample=s2,
    ), open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
