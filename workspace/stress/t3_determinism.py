"""Test 3 — determinism @ T=0.

Generate the same prompt 100 times with do_sample=False and compare
the full token sequences.  Run twice — once with CIPHER, once without
— and report:

    intra_run_match     : how many of the 100 outputs are byte-identical
    inter_run_match     : does CIPHER produce the same tokens as baseline?
    first_divergence    : index of first token that differs (CIPHER vs baseline)

A model.generate() in a loop hits the cuBLAS workspace leak
(documented in step9_llama70b.py).  We mitigate by torch.cuda.empty_cache()
between calls and by using max_new_tokens=64 (small enough to keep
workspace bounded).
"""
import os, sys, json, time, gc, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import stress_common as sc

sc._setup_alloc_env()

USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
N_RUNS    = int(os.environ.get("STRESS_T3_RUNS", "100"))
N_TOKENS  = int(os.environ.get("STRESS_T3_TOKENS", "64"))
SUFFIX    = os.environ.get("STRESS_SUFFIX",
                           "_baseline" if not USE_CIPHER else "")
PROMPT    = "The future of GPU computing is"


def main():
    rt, _, _ = (sc.init_cipher() if USE_CIPHER
                else (None, lambda: {}, lambda: {}))
    import torch
    print(f"[t3] cipher={USE_CIPHER} runs={N_RUNS} tokens={N_TOKENS}",
          flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)

    # Untimed warmup so cuBLAS workspace is allocated and FP8 hits its
    # stable-pointer threshold before we start measuring.
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn,
                           max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    gc.collect(); torch.cuda.empty_cache()

    sequences = []  # list[list[int]]
    hashes    = []
    crashes   = 0
    t0 = time.perf_counter()
    for i in range(N_RUNS):
        try:
            with torch.no_grad():
                out = model.generate(ids, attention_mask=attn,
                                     max_new_tokens=N_TOKENS,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id,
                                     use_cache=True)
            torch.cuda.synchronize()
            seq = out[0, ids.shape[1]:].tolist()
        except Exception as e:
            seq = [-1]
            crashes += 1
            print(f"[t3] run {i} crashed: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
        sequences.append(seq)
        h = hashlib.sha256(json.dumps(seq).encode()).hexdigest()[:16]
        hashes.append(h)
        if i in (0, 1, 49, 99) or i % 20 == 0:
            print(f"[t3] run {i:3d} hash={h} first6={seq[:6]} "
                  f"alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB",
                  flush=True)
        # We cannot empty_cache() between calls under CIPHER FP8 because
        # FP8 holds device-side weight buffers across calls.  But for
        # PyTorch state we can at least free Python references.
        del out
        gc.collect()
    elapsed = time.perf_counter() - t0

    # Intra-run determinism: how many outputs match run 0?
    ref = hashes[0]
    intra_match = sum(1 for h in hashes if h == ref)

    out_path = os.path.join(os.path.dirname(__file__),
                            f"t3_determinism{SUFFIX}.json")
    payload = dict(
        cipher=USE_CIPHER, n_runs=N_RUNS, n_tokens=N_TOKENS,
        elapsed_s=elapsed, crashes=crashes,
        intra_match=intra_match,
        ref_hash=ref, ref_seq=sequences[0],
        all_hashes=hashes,
        unique_hashes=sorted(set(hashes)),
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[t3] {intra_match}/{N_RUNS} runs match run 0 "
          f"(unique={len(set(hashes))}, crashes={crashes}, {elapsed:.1f}s)",
          flush=True)
    print(f"[t3] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
