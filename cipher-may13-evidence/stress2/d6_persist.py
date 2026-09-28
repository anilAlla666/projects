"""D6 v2 — Persist convergence (kernel-flow promotion count).

Reads `cipher_persist_promotion_count` from the LD_PRELOAD-loaded hook.
"""
import os, sys, json, time, ctypes, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()


def main():
    rt, fp8_stats, _ = sc.init_cipher()
    # The promotion count lives in the hook .so (LD_PRELOAD'd).
    libc = ctypes.CDLL(None)
    libc.cipher_persist_promotion_count.restype = ctypes.c_uint64
    libc.cipher_persist_observe_count.restype = ctypes.c_uint64
    libc.cipher_persist_fast_path_count.restype = ctypes.c_uint64

    def stats():
        return dict(promoted=libc.cipher_persist_promotion_count(),
                    observed=libc.cipher_persist_observe_count(),
                    fast_path=libc.cipher_persist_fast_path_count())

    import torch
    print(f"[D6] loading model", flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=True, rt=rt)
    PROMPT = "The future of GPU computing is"
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)

    # Warmup.
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()

    pers_at = {0: stats()}
    waypoints = [100, 1000, 5000]
    cur = 0
    last_ids = ids
    with torch.no_grad():
        for wp in waypoints:
            n_to_gen = wp - cur
            cur_attn = torch.ones_like(last_ids)
            out = model.generate(last_ids, attention_mask=cur_attn,
                                 max_new_tokens=n_to_gen, do_sample=False,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            pers_at[wp] = stats()
            last_ids = out
            cur = wp
            print(f"[D6 @ {wp:>5}]: {pers_at[wp]}", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "d6_persist.json")
    with open(out_path, "w") as f:
        json.dump(dict(pers_at=pers_at), f, indent=2, default=str)
    print(f"[D6] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
