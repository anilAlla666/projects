"""A3 — batch sweep B=1,2,4,8,16,32 on Llama-3.1-8B with CIPHER.

For each B: prefill=128, generate ~64 tokens, report tps/W/tok-W/MFU.
MFU = (45.0e9 * tok/s) / 989e12  (per the spec).
"""
import os, sys, json, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()

USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
BATCHES = [int(b) for b in os.environ.get("BATCHES", "1,2,4,8,16,32").split(",")]
PREFILL = int(os.environ.get("PREFILL", "128"))
TARGETS = {1:200, 2:128, 4:96, 8:96, 16:64, 32:64}
PROMPT = ("Energy efficiency means doing more useful work per watt. "
          "The future of GPU computing is to make every joule count. ")


def main():
    rt, fp8_stats, fus_stats = (sc.init_cipher() if USE_CIPHER
                                else (None, lambda: {}, lambda: {}))
    import torch
    print(f"[A3] cipher={USE_CIPHER} batches={BATCHES} prefill={PREFILL}",
          flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)

    ids_one = tok(PROMPT, return_tensors="pt").input_ids[0]
    n_tile = (PREFILL + len(ids_one) - 1) // len(ids_one)
    big = (ids_one.repeat(n_tile))[:PREFILL]
    p1 = big.unsqueeze(0).to("cuda:0")

    rows = []
    for B in BATCHES:
        prompt_ids = p1.expand(B, -1).contiguous()
        attn = torch.ones_like(prompt_ids)
        target = TARGETS.get(B, 64)
        try:
            with torch.no_grad():
                _ = model.generate(prompt_ids, attention_mask=attn,
                                   max_new_tokens=4, do_sample=False,
                                   pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()

            samples = []
            stop = threading.Event()
            th = threading.Thread(target=sc.power_sampler,
                                  args=(stop, samples, 1, 0.15), daemon=True)
            th.start()

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(prompt_ids, attention_mask=attn,
                                     max_new_tokens=target, do_sample=False,
                                     pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            stop.set(); th.join(timeout=2)

            n_new = out.shape[1] - prompt_ids.shape[1]
            tps = (B * n_new) / elapsed
            pw = [s[1] for s in samples[3:]] or [0]
            mean_w = sum(pw)/len(pw)
            tok_w = tps / mean_w if mean_w > 0 else 0
            mfu = (45.0e9 * tps) / 989e12 * 100  # %
            tail = tok.decode(out[0, prompt_ids.shape[1]:],
                              skip_special_tokens=False)[:60]
            rows.append(dict(B=B, tps=tps, mean_w=mean_w, tok_w=tok_w,
                             mfu_pct=mfu, elapsed_s=elapsed,
                             new_tokens=n_new, last_text=tail))
            print(f"  B={B:>3} tps={tps:7.2f} W={mean_w:6.1f} "
                  f"tok/W={tok_w:.4f} MFU={mfu:.2f}% tail={tail!r}",
                  flush=True)
            del out, prompt_ids, attn
            import gc; gc.collect(); torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            rows.append(dict(B=B, error="OOM"))
            print(f"  B={B}: OOM ({e})", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            rows.append(dict(B=B, error=f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"  B={B}: {type(e).__name__}: {str(e)[:200]}", flush=True)

    suffix = "_baseline" if not USE_CIPHER else ""
    out_path = os.path.join(os.path.dirname(__file__), f"a3_batch_sweep{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(dict(cipher=USE_CIPHER, prefill=PREFILL, rows=rows), f, indent=2)
    print(f"[A3] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
