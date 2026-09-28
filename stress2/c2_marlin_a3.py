"""C2 — Marlin batch sweep B=1,2,4,8,16,32 on system torch 2.7."""
import os, sys, json, time, ctypes, threading, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
sys.path.insert(0, os.path.dirname(__file__))
from c2_marlin import setup_rt, patch_model, Counters

PROMPT = ("Energy efficiency means doing more useful work per watt. "
          "The future of GPU computing is to make every joule count. ")
BATCHES = [int(b) for b in os.environ.get("BATCHES", "1,2,4,8,16,32").split(",")]
PREFILL = int(os.environ.get("PREFILL", "128"))
TARGETS = {1:200, 2:128, 4:96, 8:96, 16:64, 32:64}


def main():
    rt = setup_rt()
    import torch
    print(f"[C2-A3] torch={torch.__version__} batches={BATCHES}", flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=False, rt=None)
    n_c, n_s = patch_model(rt, model)
    print(f"[C2-A3] Marlin-patched {n_c} linears, skipped {n_s}", flush=True)

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
            Counters.marlin = 0
            Counters.fallback_largeM = 0
            Counters.fallback_rc = 0

            samples = []; stop = threading.Event()
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
            tail = tok.decode(out[0, prompt_ids.shape[1]:],
                              skip_special_tokens=False)[:60]
            rows.append(dict(B=B, tps=tps, mean_w=mean_w,
                             tok_w=tps/mean_w if mean_w > 0 else 0,
                             marlin=Counters.marlin,
                             fb_largeM=Counters.fallback_largeM,
                             fb_rc=Counters.fallback_rc,
                             new_tokens=n_new, last_text=tail,
                             coherent=("!!!!" not in tail)))
            r = rows[-1]
            print(f"  B={B:>3} tps={tps:>8.1f} W={mean_w:>5.0f} "
                  f"tok/W={r['tok_w']:.4f} marlin={Counters.marlin} "
                  f"fb_largeM={Counters.fallback_largeM} coh={r['coherent']}",
                  flush=True)
            del out, prompt_ids, attn
            gc.collect(); torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            rows.append(dict(B=B, error="OOM"))
            print(f"  B={B}: OOM", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            rows.append(dict(B=B, error=f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"  B={B}: {type(e).__name__}: {str(e)[:200]}", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "c2_marlin_a3.json")
    with open(out_path, "w") as f:
        json.dump(dict(rows=rows, n_compressed=n_c, n_skipped=n_s), f, indent=2)
    print(f"[C2-A3] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
