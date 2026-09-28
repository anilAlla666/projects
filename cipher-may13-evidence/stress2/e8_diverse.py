"""E8 — 200 diverse prompts, T=0, all coherent.

Uses 200 prompts spanning code, prose, math, instructions, dialogue, etc.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
N = int(os.environ.get("N", "200"))


PROMPT_BANK = [
    "Explain photosynthesis in one paragraph.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is the capital of France?",
    "Summarise the plot of Hamlet.",
    "Translate to French: 'Good morning, how are you?'",
    "Why is the sky blue?",
    "What is 3 * 7 + 2?",
    "Define machine learning.",
    "List the planets in the solar system.",
    "Who wrote 'Pride and Prejudice'?",
    "Solve: x^2 - 5x + 6 = 0.",
    "Give three uses for a hashmap.",
    "What is HTTP?",
    "Compose a haiku about autumn.",
    "Explain TCP vs UDP.",
    "What is the speed of light?",
    "Recipe for a simple omelet:",
    "What is the difference between a list and a tuple in Python?",
    "Name three causes of climate change.",
    "Describe the Big Bang theory in two sentences.",
]


def main():
    rt, fp8_stats, fus_stats = (sc.init_cipher() if USE_CIPHER
                                else (None, lambda: {}, lambda: {}))
    import torch
    print(f"[E8] cipher={USE_CIPHER} N={N}", flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)

    prompts = [PROMPT_BANK[i % len(PROMPT_BANK)] + f" (variant {i})" for i in range(N)]

    rows = []
    coh = 0
    crashes = 0
    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
        attn = torch.ones_like(ids)
        try:
            with torch.no_grad():
                out = model.generate(ids, attention_mask=attn,
                                     max_new_tokens=48, do_sample=False,
                                     pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
            coherent = "!!!!" not in text and len(text.strip()) > 3
            coh += int(coherent)
            if i < 5 or i % 50 == 0:
                rows.append(dict(i=i, prompt=p[:60], text=text[:80],
                                 coherent=coherent))
                print(f"  [{i:>3}] coh={coherent} text={text[:60]!r}", flush=True)
        except Exception as e:
            crashes += 1
            rows.append(dict(i=i, error=f"{type(e).__name__}: {str(e)[:120]}"))
    elapsed = time.perf_counter() - t0

    suffix = "_baseline" if not USE_CIPHER else ""
    payload = dict(cipher=USE_CIPHER, N=N, coherent=coh, crashes=crashes,
                   elapsed_s=elapsed, samples=rows)
    with open(os.path.join(os.path.dirname(__file__), f"e8_diverse{suffix}.json"),
              "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[E8] coherent={coh}/{N} crashes={crashes} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
