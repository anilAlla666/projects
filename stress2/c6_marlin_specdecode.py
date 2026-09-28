"""C6 — speculative decoding with Marlin INT4 substitution on both
draft (Llama-3.2-1B) and target (Llama-3.1-8B) models.

Runs three modes for tok/W comparison:
   target alone (Marlin-patched)
   target + draft assisted (both Marlin-patched)
"""
import os, sys, json, time, ctypes, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc
sys.path.insert(0, os.path.dirname(__file__))
from c2_marlin import setup_rt, patch_model, Counters

sc._setup_alloc_env()
TARGET = "/home/ubuntu/models/Llama-3.1-8B"
DRAFT  = "/home/ubuntu/models/Llama-3.2-1B"
N_TOKENS = int(os.environ.get("N_TOKENS", "200"))
PROMPT = ("Energy efficiency means doing more useful work per watt. "
          "The future of GPU computing is")


def measure(model, tok, ids, attn, label, assistant=None):
    samples = []; stop = threading.Event()
    th = threading.Thread(target=sc.power_sampler,
                          args=(stop, samples, 1, 0.15), daemon=True)
    th.start()
    import torch
    Counters.marlin = 0; Counters.fallback_largeM = 0; Counters.fallback_rc = 0
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
                marlin=Counters.marlin,
                fb_largeM=Counters.fallback_largeM,
                first_text=tok.decode(out[0, ids.shape[1]:],
                                      skip_special_tokens=False)[:120])


def main():
    rt = setup_rt()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[C6-M] torch={torch.__version__}", flush=True)

    tok = AutoTokenizer.from_pretrained(TARGET)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        TARGET, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    target.requires_grad_(False); target.train(False)
    n_t, _ = patch_model(rt, target)
    print(f"[C6-M] target: Marlin-patched {n_t} linears", flush=True)

    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    draft.requires_grad_(False); draft.train(False)
    n_d, _ = patch_model(rt, draft)
    print(f"[C6-M] draft : Marlin-patched {n_d} linears", flush=True)

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)
    # Warmup both.
    with torch.no_grad():
        for m in (target, draft):
            _ = m.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()

    rows = []
    rows.append(measure(target, tok, ids, attn, "target_alone_marlin"))
    rows.append(measure(target, tok, ids, attn, "target_assist_marlin", assistant=draft))

    out_path = os.path.join(os.path.dirname(__file__),
                             "c6_marlin_specdecode.json")
    with open(out_path, "w") as f:
        json.dump(dict(rows=rows, n_target=n_t, n_draft=n_d), f, indent=2)

    for r in rows:
        print(f"  {r['label']:<22} tps={r['tps']:>7.2f} W={r['mean_w']:>5.0f} "
              f"tok/W={r['tok_w']:.4f} marlin={r['marlin']:>6}", flush=True)
    if rows[0]['tps'] > 0:
        print(f"  speedup assist/alone   = {rows[1]['tps']/rows[0]['tps']:.2f}x")
        print(f"  tok/W ratio assist/alone = {rows[1]['tok_w']/rows[0]['tok_w']:.2f}x")


if __name__ == "__main__":
    main()
