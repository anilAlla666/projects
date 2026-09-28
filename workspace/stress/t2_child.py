"""Test 2 child — load Llama-3.1-8B and generate continuously.

Reads STRESS_T2_DEADLINE (epoch seconds) and stops afterwards.
Writes a per-child JSON with completion stats and stderr counts.
"""
import os, sys, json, time, gc, traceback
sys.path.insert(0, os.path.dirname(__file__))
import stress_common as sc

sc._setup_alloc_env()

USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
DEADLINE   = float(os.environ["STRESS_T2_DEADLINE"])
CHILD_ID   = os.environ.get("STRESS_T2_CHILD", "0")
DEVICE     = os.environ.get("STRESS_T2_DEVICE", "cuda:0")
SUFFIX     = os.environ.get("STRESS_SUFFIX", "")
PROMPT     = ("Energy efficiency means doing more useful work per watt. "
              "The future of GPU computing is to make every joule count. ")
# Each loop iter generates this many tokens; small enough to bound
# cuBLAS workspace per-iter, large enough to make per-call overhead
# negligible.
N_TOKENS   = int(os.environ.get("STRESS_T2_TOKENS_PER_CALL", "64"))


def main():
    err_path = os.path.join(os.path.dirname(__file__),
                            f"t2_child_{CHILD_ID}{SUFFIX}.err")
    out_path = os.path.join(os.path.dirname(__file__),
                            f"t2_child_{CHILD_ID}{SUFFIX}.json")
    err_log = open(err_path, "w")
    err_log.write(f"[child {CHILD_ID}] start cipher={USE_CIPHER} "
                  f"deadline={DEADLINE}\n")
    err_log.flush()
    try:
        rt, _, _ = (sc.init_cipher() if USE_CIPHER
                    else (None, lambda: {}, lambda: {}))
        import torch
        model, tok, load_s = sc.load_model(DEVICE, patch=USE_CIPHER, rt=rt)
        err_log.write(f"[child {CHILD_ID}] loaded in {load_s:.1f}s\n")
        err_log.flush()
        ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
        attn = torch.ones_like(ids)

        with torch.no_grad():
            _ = model.generate(ids, attention_mask=attn,
                               max_new_tokens=8, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        torch.cuda.synchronize()

        n_calls = 0; n_tokens = 0; cuda_errs = 0
        first_text = None; last_text = None
        t0 = time.time()
        while time.time() < DEADLINE:
            try:
                with torch.no_grad():
                    out = model.generate(ids, attention_mask=attn,
                                         max_new_tokens=N_TOKENS,
                                         do_sample=False,
                                         pad_token_id=tok.pad_token_id,
                                         use_cache=True)
                torch.cuda.synchronize()
                new = out.shape[1] - ids.shape[1]
                n_tokens += new
                if first_text is None:
                    first_text = tok.decode(out[0, ids.shape[1]:],
                                            skip_special_tokens=False)[:80]
                last_text = tok.decode(out[0, ids.shape[1]:],
                                       skip_special_tokens=False)[:80]
                del out
                n_calls += 1
                if n_calls % 50 == 0:
                    err_log.write(
                        f"[child {CHILD_ID}] {n_calls} calls, "
                        f"{n_tokens} tokens, "
                        f"alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB\n")
                    err_log.flush()
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:200]}"
                err_log.write(f"[child {CHILD_ID}] err: {msg}\n")
                err_log.flush()
                if "CUDA" in msg or "cuda" in msg:
                    cuda_errs += 1
                # Stop loop on any error so we don't spin forever.
                break
        elapsed = time.time() - t0
        ok = True
    except Exception as e:
        ok = False
        traceback.print_exc(file=err_log)
        err_log.write(f"[child {CHILD_ID}] fatal: {type(e).__name__}: "
                      f"{str(e)[:200]}\n")
        err_log.flush()
        n_calls = 0; n_tokens = 0; cuda_errs = 1; elapsed = 0.0
        first_text = None; last_text = None

    payload = dict(
        child_id=CHILD_ID, cipher=USE_CIPHER, device=DEVICE,
        ok=ok, calls=n_calls, tokens=n_tokens, cuda_errs=cuda_errs,
        elapsed_s=elapsed, tps=n_tokens/elapsed if elapsed>0 else 0.0,
        first_text=first_text, last_text=last_text,
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    err_log.write(f"[child {CHILD_ID}] DONE calls={n_calls} "
                  f"tokens={n_tokens} elapsed={elapsed:.1f}s\n")
    err_log.close()


if __name__ == "__main__":
    main()
