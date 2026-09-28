"""ACTION 3: Joint KV projection with prefill+decode calibration, held-out prompt.

Calibration captures KVs at BOTH prefill positions and decode positions.
Sweep K_rank x V_rank. Held-out prompt eval with teacher-forcing.
"""
import os, sys, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL = "mistralai/Mistral-7B-v0.1"
N_LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
JOINT = NUM_KV_HEADS * HEAD_DIM  # 1024
N_DECODE_CALIB = 200
N_DECODE_EVAL  = 200

CALIB_PROMPTS = [
    "Explain how a CPU executes instructions step by step",
    "Write a story about a detective solving a mystery in Tokyo",
    "What are the economic implications of rising interest rates",
    "Describe the process of photosynthesis in detail",
    "Compare and contrast democracy and authoritarianism",
    "Write a Python function that implements binary search",
    "Explain the theory of general relativity to a high school student",
    "Describe the history of the Roman Empire from founding to fall",
    "What causes weather patterns and how do meteorologists predict them",
    "Write a recipe for chicken tikka masala with detailed instructions",
]
EVAL_PROMPT = "Explain the differences between TCP and UDP networking protocols"

# Sweep ranks (K_rank, V_rank) pairs
RANKS = [(192, 240), (256, 320), (384, 480), (512, 512)]


def calibrate_kv_pre_dec(model, tok):
    """Capture KV at BOTH prefill positions and decode positions."""
    caps_k = [[] for _ in range(N_LAYERS)]
    caps_v = [[] for _ in range(N_LAYERS)]
    capture_enabled = [False]

    def make_patched(cache):
        orig = cache.update
        def patched(key_states, value_states, layer_idx, cache_kwargs=None):
            if capture_enabled[0]:
                # key_states: [B=1, num_kv=8, S, D=128]
                S = key_states.shape[2]
                for s in range(S):
                    k_flat = key_states[0, :, s, :].reshape(-1).detach().to(torch.float32).cpu()
                    v_flat = value_states[0, :, s, :].reshape(-1).detach().to(torch.float32).cpu()
                    caps_k[layer_idx].append(k_flat)
                    caps_v[layer_idx].append(v_flat)
            return orig(key_states, value_states, layer_idx, cache_kwargs)
        return patched

    for p_idx, prompt in enumerate(CALIB_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        cache = DynamicCache()
        cache.update = make_patched(cache)
        # Capture both prefill AND decode
        capture_enabled[0] = True  # ON for prefill
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(N_DECODE_CALIB):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        capture_enabled[0] = False
        print(f"[calib]   prompt {p_idx+1}/{len(CALIB_PROMPTS)} done; layer 0: {len(caps_k[0])} captures", flush=True)

    print(f"[calib] total per layer: {len(caps_k[0])}", flush=True)
    return caps_k, caps_v


def compute_qkv(caps_k, caps_v, max_rank):
    """Returns qK[layer], qV[layer]: each [JOINT, max_rank] cpu fp32."""
    qK, qV = [], []
    for li in range(N_LAYERS):
        XK = torch.stack(caps_k[li], dim=0).to("cuda")
        XV = torch.stack(caps_v[li], dim=0).to("cuda")
        UK, SK, VhK = torch.linalg.svd(XK, full_matrices=False)
        UV, SV, VhV = torch.linalg.svd(XV, full_matrices=False)
        rK = min(max_rank, VhK.shape[0])
        rV = min(max_rank, VhV.shape[0])
        qK.append(VhK[:rK, :].t().contiguous().to("cpu"))
        qV.append(VhV[:rV, :].t().contiguous().to("cpu"))
        del XK, XV, UK, SK, VhK, UV, SV, VhV
    torch.cuda.empty_cache()
    return qK, qV


def baseline_run(model, prompt_ids, n_decode):
    decoded = []
    logits = []
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoded.append(nxt.item())
        for _ in range(n_decode):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits.append(out.logits[0, -1, :].detach().to(torch.float32).cpu())
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded.append(nxt.item())
    return decoded, logits


def projected_run(model, prompt_ids, forced, qK_g, qV_g, K_rank, V_rank):
    """Run with projection on K (top K_rank) and V (top V_rank). Returns logits."""
    cache = DynamicCache()
    orig_update = cache.update
    def patched(key_states, value_states, layer_idx, cache_kwargs=None):
        b, h, s, d = key_states.shape
        k_flat = key_states.reshape(b, s, h * d)  # [1, s, 1024]
        v_flat = value_states.reshape(b, s, h * d)
        QK = qK_g[layer_idx][:, :K_rank]
        QV = qV_g[layer_idx][:, :V_rank]
        k_proj = (k_flat @ QK) @ QK.T
        v_proj = (v_flat @ QV) @ QV.T
        return orig_update(
            k_proj.reshape(b, h, s, d).contiguous(),
            v_proj.reshape(b, h, s, d).contiguous(),
            layer_idx, cache_kwargs)
    cache.update = patched

    logits = []
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
        past = out.past_key_values
        for tok_id in forced:
            inp = torch.tensor([[tok_id]], dtype=torch.long, device=prompt_ids.device)
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits.append(out.logits[0, -1, :].detach().to(torch.float32).cpu())
    return logits


def main():
    print(f"[load] {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    print(f"\n[calib] {len(CALIB_PROMPTS)} prompts (prefill + {N_DECODE_CALIB} decode each)", flush=True)
    caps_k, caps_v = calibrate_kv_pre_dec(model, tok)
    qK, qV = compute_qkv(caps_k, caps_v, max_rank=512)
    print(f"[calib] qK[0]={tuple(qK[0].shape)}, qV[0]={tuple(qV[0].shape)}", flush=True)

    qK_g = [q.to("cuda", dtype=torch.float16) for q in qK]
    qV_g = [q.to("cuda", dtype=torch.float16) for q in qV]

    eval_ids = tok(EVAL_PROMPT, return_tensors="pt").input_ids.to("cuda")
    print(f"\n[baseline] HELD-OUT prompt: {EVAL_PROMPT!r}", flush=True)
    base_tokens, base_logits = baseline_run(model, eval_ids, N_DECODE_EVAL)
    base_text = tok.decode(base_tokens)
    print(f"[baseline] {base_text[:200]!r}", flush=True)
    forced = base_tokens[:N_DECODE_EVAL]

    results = []
    print(f"\n  {'(K_rank, V_rank)':<18} {'top1':>7} {'mean_kl':>8} {'p99_kl':>8} {'[0..50)':>8} {'[100..150)':>10}", flush=True)
    print("  " + "-" * 70, flush=True)
    for K_rank, V_rank in RANKS:
        proj_logits = projected_run(model, eval_ids, forced, qK_g, qV_g, K_rank, V_rank)
        n = min(len(base_logits), len(proj_logits))
        base_argmax = [base_logits[i].argmax().item() for i in range(n)]
        proj_argmax = [proj_logits[i].argmax().item() for i in range(n)]
        correct = [int(b == p) for b, p in zip(base_argmax, proj_argmax)]
        top1 = sum(correct) / n
        kls = []
        for i in range(n):
            p_log = F.log_softmax(base_logits[i], dim=-1)
            q_log = F.log_softmax(proj_logits[i], dim=-1)
            p = p_log.exp()
            kls.append((p * (p_log - q_log)).sum().item())
        kls.sort()
        mean_kl = sum(kls)/n
        p99_kl = kls[min(n-1, int(0.99 * n))]
        bin_05 = sum(correct[0:50])/50
        bin_100 = sum(correct[100:150])/50
        print(f"  ({K_rank:>3}, {V_rank:>3})         {top1:>7.4f} {mean_kl:>8.4f} {p99_kl:>8.4f} {bin_05:>8.4f} {bin_100:>10.4f}", flush=True)
        results.append({
            "K_rank": K_rank, "V_rank": V_rank,
            "top1": top1, "mean_kl": mean_kl, "p99_kl": p99_kl,
            "bin_0_50": bin_05, "bin_100_150": bin_100,
            "correct": correct,
        })

    # Find min rank meeting >95% top1
    print(f"\n=== ACTION 3 verdict ===", flush=True)
    passed = [r for r in results if r["top1"] >= 0.95]
    if passed:
        best = min(passed, key=lambda r: r["K_rank"] + r["V_rank"])
        bw = (best["K_rank"] + best["V_rank"]) / (1024 + 1024)
        print(f"  Min rank passing >95% top1: K={best['K_rank']}, V={best['V_rank']}", flush=True)
        print(f"  Compression ratio: {1.0/bw:.2f}x KV bandwidth reduction", flush=True)
    else:
        best = max(results, key=lambda r: r["top1"])
        print(f"  No rank reaches 95% top1. Best: K={best['K_rank']}, V={best['V_rank']} -> {best['top1']:.4f}", flush=True)

    out_path = "/home/ubuntu/op31-prod-fix/action3_kv_projection_results.json"
    with open(out_path, "w") as f:
        json.dump([{kk: vv for kk, vv in r.items() if kk != "correct"} for r in results], f, indent=2)
    print(f"\n[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
