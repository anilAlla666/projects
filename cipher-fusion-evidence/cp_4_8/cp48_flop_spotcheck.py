"""CP 4.8 section-3 — FLOP cross-check spot-check (Option C, 3 representative WLs).

cipher_flopd (the CUPTI PM-sampling daemon behind /proc/cipher/flops) is not
deployed, so the live PMU cross-check has no source. This spot-check supplies
the independent FLOP count instead via PyTorch FlopCounterMode, which traces
the actual ATen op graph (matmul/addmm/bmm + scaled_dot_product_attention) --
independent of mfu_compute.tensor_flops()'s 2N/6N-geometry method, which is
what section 3 validates.

3 WLs (user-selected): WL01 memory-bound 2N; WL17 training 6N; WL13 mixed,
exercises the attention term + the Mistral geometry. crosscheck() at +/-5%.
>5% on any -> methodology bug -> pause + investigate before G1.
"""
import os, json, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import mfu_compute as M

TINY = "/home/ubuntu/models/TinyLlama-1.1B"
MISTRAL = "/home/ubuntu/models/Mistral-7B-v0.1"
results = []
INFER = torch.inference_mode


def measure(fn):
    """Run fn() under FlopCounterMode; return total FLOPs (independent count)."""
    fc = FlopCounterMode(display=False)
    with fc:
        fn()
    return fc.get_total_flops()


def load(path, train=False):
    # eager attention: FlopCounterMode's sdpa_flop_count mishandles GQA
    # (mismatched Q/KV head counts); eager runs attention as explicit
    # repeat_kv + bmm, which FlopCounterMode counts via the matmul path --
    # identical algorithmic FLOPs, no GQA-shape assertion.
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                             device_map="cuda:0",
                                             attn_implementation="eager")
    m.train(train)
    return m


# ---- WL01 -- TinyLlama, one B=1 decode step (memory-bound, 2N) ------------
def wl01():
    tok = AutoTokenizer.from_pretrained(TINY)
    m = load(TINY)
    CTX = 128
    ids = tok("Energy efficiency. " * 40, return_tensors="pt").input_ids[:, :CTX].cuda()
    cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=CTX + 8,
                        device="cuda:0", dtype=torch.float16)
    with torch.no_grad():
        out = m(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        measured = measure(lambda: m(nxt, past_key_values=cache,
                                     use_cache=True, logits_to_keep=1))
    analytical = M.tensor_flops("WL01", units=1, mean_context=CTX)
    del m, cache
    torch.cuda.empty_cache()
    return measured, analytical, dict(units=1, mean_context=CTX)


# ---- WL17 -- TinyLlama, one full training step fwd+bwd (training, 6N) -----
def wl17():
    tok = AutoTokenizer.from_pretrained(TINY)
    m = load(TINY, train=True)
    B, S = 1, 256
    ids = tok("The future of GPU computing. " * 80,
              return_tensors="pt").input_ids[:, :S].cuda().repeat(B, 1)
    def step():
        out = m(ids, labels=ids)
        out.loss.backward()
        m.zero_grad(set_to_none=True)
    measured = measure(step)
    analytical = M.tensor_flops("WL17", units=B * S, mean_context=S / 2)
    del m
    torch.cuda.empty_cache()
    return measured, analytical, dict(units=B * S, mean_context=S / 2)


# ---- WL13 -- Mistral-7B, prefill (mixed; exercises the attention term) ----
def wl13():
    tok = AutoTokenizer.from_pretrained(MISTRAL)
    m = load(MISTRAL)
    S = 8192
    ids = tok("This is a long context test sentence. " * 2000,
              return_tensors="pt").input_ids[:, :S].cuda()
    with torch.no_grad():
        measured = measure(lambda: m(ids, use_cache=False, logits_to_keep=1))
    # FlopCounterMode counts DENSE attention (no causal mode). The gate uses
    # causal-half (mean_context=S/2) — what the real SDPA/FlashAttention
    # workload executes. Cross-check against the DENSE value (mean_context=S)
    # to validate the attention *geometry*; report the causal gate value too.
    analytical_dense  = M.tensor_flops("WL13", units=S, mean_context=S)
    analytical_causal = M.tensor_flops("WL13", units=S, mean_context=S / 2)
    del m
    torch.cuda.empty_cache()
    return measured, analytical_dense, dict(
        units=S, crosscheck="dense (matches FlopCounterMode)",
        analytical_causal_gate_value=analytical_causal,
        note="gate uses causal-half; FlopCounterMode validates dense geometry")


for wl, fn in [("WL01", wl01), ("WL17", wl17), ("WL13", wl13)]:
    print(f"[spotcheck] running {wl} ...", flush=True)
    measured, analytical, params = fn()
    agree, rel = M.crosscheck(wl, analytical, measured, tol=0.05)
    row = dict(wl=wl, measured_flops=measured, analytical_flops=analytical,
               rel_delta=rel, agree=agree, params=params)
    results.append(row)
    print(f"[spotcheck] {wl}: measured={measured:.4e}  analytical={analytical:.4e}  "
          f"rel_delta={rel*100:.2f}%  agree(<=5%)={agree}", flush=True)

allok = all(r["agree"] for r in results)
doc = dict(cp="4.8", check="section-3 FLOP cross-check spot-check",
           method="independent count via torch FlopCounterMode (ATen op-graph trace)",
           tolerance_pct=5.0, all_within_tolerance=allok, results=results,
           note="cipher_flopd not deployed -- live PMU path unavailable; this "
                "is the representative-sample cross-check (Option C).")
json.dump(doc, open("/home/ubuntu/cipher-fusion-evidence/cp_4_8/"
                     "cp48_flop_spotcheck_result.json", "w"), indent=2, default=str)
print(f"\n[spotcheck] ALL WITHIN +/-5%: {allok}")
print("[spotcheck] wrote cp48_flop_spotcheck_result.json")
sys.exit(0 if allok else 1)
