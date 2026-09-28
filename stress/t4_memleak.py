"""Test 4 — 1000 sequential generate() calls × 100 tokens.

Records nvidia-smi memory.used every 10 calls.  Reports growth.
Documents whether the cuBLAS-workspace leak (per step9_llama70b.py)
shows up.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(__file__))
import stress_common as sc

sc._setup_alloc_env()

USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
N_CALLS   = int(os.environ.get("STRESS_T4_CALLS", "1000"))
N_TOKENS  = int(os.environ.get("STRESS_T4_TOKENS", "100"))
SUFFIX    = os.environ.get("STRESS_SUFFIX",
                           "_baseline" if not USE_CIPHER else "")
PROMPT    = ("Energy efficiency means doing more useful work per watt. "
             "The future of GPU computing is to make every joule count. ")


def main():
    rt, _, _ = (sc.init_cipher() if USE_CIPHER
                else (None, lambda: {}, lambda: {}))
    import torch
    print(f"[t4] cipher={USE_CIPHER} calls={N_CALLS} tokens={N_TOKENS}",
          flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)

    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn,
                           max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()

    pre_smi = sc.nvsmi_mem_used_mib(0)
    pre_alloc = torch.cuda.memory_allocated(0) // (1024*1024)
    pre_reserved = torch.cuda.memory_reserved(0) // (1024*1024)
    print(f"[t4] start: nvsmi={pre_smi}MiB alloc={pre_alloc}MiB "
          f"reserved={pre_reserved}MiB", flush=True)

    log = []  # list[dict]
    crashes = 0
    last_err = None
    completed = 0
    t0 = time.perf_counter()
    for i in range(N_CALLS):
        try:
            with torch.no_grad():
                out = model.generate(ids, attention_mask=attn,
                                     max_new_tokens=N_TOKENS,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id,
                                     use_cache=True)
            torch.cuda.synchronize()
            del out
            completed += 1
        except Exception as e:
            crashes += 1
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[t4] crash @ call {i}: {last_err}", flush=True)
            log.append(dict(call=i, crash=last_err))
            break  # leak test is over once it crashes
        if i % 10 == 9:
            smi = sc.nvsmi_mem_used_mib(0)
            alloc = torch.cuda.memory_allocated(0) // (1024*1024)
            res   = torch.cuda.memory_reserved(0)  // (1024*1024)
            log.append(dict(call=i+1, nvsmi_mib=smi,
                            alloc_mib=alloc, reserved_mib=res))
            if i % 100 == 99 or i < 30:
                print(f"[t4] call {i+1:4d}: nvsmi={smi}MiB "
                      f"alloc={alloc}MiB reserved={res}MiB", flush=True)
        # Don't gc/empty_cache between calls — we want to see the leak.
    elapsed = time.perf_counter() - t0

    post_smi = sc.nvsmi_mem_used_mib(0)
    post_alloc = torch.cuda.memory_allocated(0) // (1024*1024)
    post_reserved = torch.cuda.memory_reserved(0) // (1024*1024)
    growth_smi = post_smi - pre_smi
    growth_alloc = post_alloc - pre_alloc

    payload = dict(
        cipher=USE_CIPHER, target_calls=N_CALLS, completed_calls=completed,
        crashes=crashes, last_err=last_err, elapsed_s=elapsed,
        pre_nvsmi_mib=pre_smi, post_nvsmi_mib=post_smi,
        pre_alloc_mib=pre_alloc, post_alloc_mib=post_alloc,
        pre_reserved_mib=pre_reserved, post_reserved_mib=post_reserved,
        growth_nvsmi_mib=growth_smi, growth_alloc_mib=growth_alloc,
        log=log,
    )
    out_path = os.path.join(os.path.dirname(__file__),
                            f"t4_memleak{SUFFIX}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[t4] completed={completed}/{N_CALLS} crashes={crashes} "
          f"nvsmi {pre_smi}->{post_smi} (+{growth_smi}MiB) "
          f"alloc {pre_alloc}->{post_alloc} (+{growth_alloc}MiB) "
          f"in {elapsed:.1f}s",
          flush=True)
    print(f"[t4] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
