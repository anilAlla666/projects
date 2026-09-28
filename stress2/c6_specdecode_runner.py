"""C6 — speculative-decoding integration harness for CIPHER.

Wraps HF assisted_generation behind two CIPHER-style env knobs:
    CIPHER_SPEC_DRAFT=/path/to/draft/model    (e.g. Llama-3.2-1B)
    CIPHER_SPEC_TARGET=/path/to/target/model  (e.g. Llama-3.1-8B)

Run:
    LD_PRELOAD=...libcipher_hook.so... \\
    CIPHER_FP8_COMPUTE=on CIPHER_FUSION_KERNELS=on \\
    CIPHER_SPEC_DRAFT=/home/ubuntu/models/Llama-3.2-1B \\
    CIPHER_SPEC_TARGET=/home/ubuntu/models/Llama-3.1-8B \\
    python3 stress2/c6_specdecode_runner.py

Reports tok/s and tok/W under three modes:
    - target alone
    - target + draft assisted
    - target + draft assisted + CIPHER FP8

The expected stack:
    target_alone (FP16 baseline) → target_assisted (HF spec) → target_assisted+CIPHER
"""
import os, sys, time, json, threading, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
TARGET_PATH = os.environ.get("CIPHER_SPEC_TARGET",
                              "/home/ubuntu/models/Llama-3.1-8B")
DRAFT_PATH  = os.environ.get("CIPHER_SPEC_DRAFT",
                              "/home/ubuntu/models/Llama-3.2-1B")
N_TOKENS    = int(os.environ.get("N_TOKENS", "200"))
PROMPT      = ("Energy efficiency means doing more useful work per watt. "
               "The future of GPU computing is")


def measure(model, tok, ids, attn, label, assistant=None):
    samples = []; stop = threading.Event()
    th = threading.Thread(target=sc.power_sampler,
                          args=(stop, samples, 1, 0.15), daemon=True)
    th.start()
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        kw = dict(attention_mask=attn, max_new_tokens=N_TOKENS,
                  do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        if assistant is not None:
            kw["assistant_model"] = assistant
        out = model.generate(ids, **kw)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    pw = [s[1] for s in samples[3:]] or [0]
    mean_w = sum(pw)/len(pw)
    n = out.shape[1] - ids.shape[1]
    tps = n / elapsed
    return dict(label=label, tokens=n, elapsed_s=elapsed, tps=tps,
                mean_w=mean_w, tok_w=tps/mean_w if mean_w > 0 else 0,
                first_text=tok.decode(out[0, ids.shape[1]:],
                                      skip_special_tokens=False)[:120])


def main():
    rt, _, _ = (sc.init_cipher() if USE_CIPHER
                else (None, lambda: {}, lambda: {}))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[C6] cipher={USE_CIPHER} target={TARGET_PATH} draft={DRAFT_PATH}",
          flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET_PATH)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    target.requires_grad_(False); target.train(False)
    if USE_CIPHER and rt is not None:
        try: sc.patch_fusion(rt, target)
        except Exception as e: print(f"  patch_fusion err: {e}", flush=True)

    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    draft.requires_grad_(False); draft.train(False)

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)
    # Warmup both.
    with torch.no_grad():
        for m in (target, draft):
            _ = m.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()

    rows = []
    rows.append(measure(target, tok, ids, attn, "target_alone"))
    rows.append(measure(target, tok, ids, attn, "target_assisted",
                         assistant=draft))

    out_path = os.path.join(os.path.dirname(__file__),
                             f"c6_specdecode{'_baseline' if not USE_CIPHER else ''}.json")
    with open(out_path, "w") as f:
        json.dump(dict(cipher=USE_CIPHER, target=TARGET_PATH, draft=DRAFT_PATH,
                       N_TOKENS=N_TOKENS, rows=rows), f, indent=2)
    for r in rows:
        print(f"  {r['label']:<18} tps={r['tps']:7.2f} W={r['mean_w']:5.0f} "
              f"tok/W={r['tok_w']:.4f}", flush=True)
    if rows[0]['tps'] > 0:
        print(f"  speedup assist/alone = {rows[1]['tps']/rows[0]['tps']:.2f}x",
              flush=True)
    print(f"[C6] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
