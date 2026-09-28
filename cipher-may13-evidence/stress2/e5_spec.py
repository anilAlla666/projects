"""E5 — speculative decoding (assisted generation).

Llama-3.2-1B draft + Llama-3.1-8B target.
Run model.generate(..., assistant_model=draft, max_new_tokens=200)
and compare against an identical greedy run without the draft.
Acceptance rate measured indirectly via wall-clock speedup.
"""
import os, sys, json, time, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
PROMPT = "Energy efficiency means doing more useful work per watt. The future of GPU computing is"


def main():
    rt, fp8_stats, fus_stats = (sc.init_cipher() if USE_CIPHER
                                else (None, lambda: {}, lambda: {}))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[E5] cipher={USE_CIPHER}", flush=True)

    tok = AutoTokenizer.from_pretrained(sc.MODEL_PATH)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        sc.MODEL_PATH, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    target.requires_grad_(False); target.train(False)
    if USE_CIPHER and rt is not None:
        try: sc.patch_fusion(rt, target)
        except Exception as e: print(f"  patch_fusion err: {e}", flush=True)

    draft = AutoModelForCausalLM.from_pretrained(
        "/home/ubuntu/models/Llama-3.2-1B", torch_dtype=torch.float16,
        device_map={"": "cuda:0"})
    draft.requires_grad_(False); draft.train(False)

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)

    # Warmup target.
    with torch.no_grad():
        _ = target.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        _ = draft.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()

    # Greedy 200-token target only.
    t0 = time.perf_counter()
    with torch.no_grad():
        out_target = target.generate(ids, attention_mask=attn, max_new_tokens=200,
                                     do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    elapsed_target = time.perf_counter() - t0
    text_target = tok.decode(out_target[0, ids.shape[1]:], skip_special_tokens=False)

    # Assisted: draft generates, target verifies.
    t0 = time.perf_counter()
    with torch.no_grad():
        out_assist = target.generate(
            ids, attention_mask=attn, max_new_tokens=200,
            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True,
            assistant_model=draft)
    torch.cuda.synchronize()
    elapsed_assist = time.perf_counter() - t0
    text_assist = tok.decode(out_assist[0, ids.shape[1]:], skip_special_tokens=False)

    speedup = elapsed_target / elapsed_assist
    same = (out_target[0].tolist() == out_assist[0].tolist())

    suffix = "_baseline" if not USE_CIPHER else ""
    payload = dict(
        cipher=USE_CIPHER,
        target_elapsed_s=elapsed_target,
        assist_elapsed_s=elapsed_assist,
        speedup_assist_vs_target=speedup,
        same_output=same,
        target_text=text_target[:160],
        assist_text=text_assist[:160])
    with open(os.path.join(os.path.dirname(__file__), f"e5_spec{suffix}.json"),
              "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[E5] target={elapsed_target:.2f}s assist={elapsed_assist:.2f}s "
          f"speedup={speedup:.2f}x same={same}", flush=True)
    print(f"  target text: {text_target[:80]!r}", flush=True)
    print(f"  assist text: {text_assist[:80]!r}", flush=True)


if __name__ == "__main__":
    main()
