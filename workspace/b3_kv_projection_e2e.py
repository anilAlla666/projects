"""B3: end-to-end joint-KV-projection quality test.

Calibrate per-layer joint K and V (post-RoPE for K, post-projection for V),
fit PCA at top-k. At eval time, monkey-patch DynamicCache.update to project
incoming key_states/value_states to top-k and back (round-trip). Compare:
  - top-1 token match rate vs baseline (teacher-forced)
  - mean KL divergence on logits

Per prior PCA: K-95 needs 154-211 of 1024 joint dims; V-95 needs 192-238. Test K=192.
Unlike weight projection, KV projection error doesn't compound through layers
because each layer's KV is independent.
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
JOINT_DIM = NUM_KV_HEADS * HEAD_DIM  # 1024
EVAL_PROMPT = "Explain how a CPU executes instructions step by step"
N_EVAL_DECODE = 200
N_CALIB_DECODE = 200    # capture more to cover the eval window
RANK_K = 192            # 95% var per prior PCA
RANK_V = 240

CALIB_PROMPTS = [
    "Explain how a CPU executes instructions step by step",
    "Write a story about a detective solving a mystery in Tokyo",
    "What are the economic implications of rising interest rates",
    "Describe the process of photosynthesis in detail",
    "Compare and contrast democracy and authoritarianism",
]


def calibrate_kv(model, tok):
    """Returns Q_K[layer], Q_V[layer]: each [JOINT_DIM, rank]."""
    caps_k = [[] for _ in range(N_LAYERS)]
    caps_v = [[] for _ in range(N_LAYERS)]
    capture_enabled = [False]

    def make_patched(cache):
        orig = cache.update
        def patched(key_states, value_states, layer_idx, cache_kwargs=None):
            if capture_enabled[0] and key_states.shape[-2] == 1:
                # [1, 8, 1, 128] → flatten heads → [1024]
                k_flat = key_states[0, :, 0, :].reshape(-1).detach().to(torch.float32).cpu()
                v_flat = value_states[0, :, 0, :].reshape(-1).detach().to(torch.float32).cpu()
                caps_k[layer_idx].append(k_flat)
                caps_v[layer_idx].append(v_flat)
            return orig(key_states, value_states, layer_idx, cache_kwargs)
        return patched

    for prompt in CALIB_PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        cache = DynamicCache()
        cache.update = make_patched(cache)
        with torch.no_grad():
            capture_enabled[0] = False
            out = model(input_ids=ids, past_key_values=cache, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            capture_enabled[0] = True
            for _ in range(N_CALIB_DECODE):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            capture_enabled[0] = False

    print(f"[calib] captured {len(caps_k[0])} per layer", flush=True)

    # Compute Q per layer per K/V (uncentered SVD on stacked tensors)
    qK = []
    qV = []
    for li in range(N_LAYERS):
        XK = torch.stack(caps_k[li], dim=0).to("cuda")  # [N, 1024]
        XV = torch.stack(caps_v[li], dim=0).to("cuda")
        UK, SK, VhK = torch.linalg.svd(XK, full_matrices=False)
        UV, SV, VhV = torch.linalg.svd(XV, full_matrices=False)
        # Q = top-r right singular vectors, [1024, r]
        rK = min(RANK_K, VhK.shape[0])
        rV = min(RANK_V, VhV.shape[0])
        qK.append(VhK[:rK, :].t().contiguous().to("cpu"))
        qV.append(VhV[:rV, :].t().contiguous().to("cpu"))
        del XK, XV, UK, SK, VhK, UV, SV, VhV
    torch.cuda.empty_cache()
    return qK, qV


def baseline_run(model, tok, prompt_ids, n_decode):
    """Greedy decode; return decoded tokens + per-step logits."""
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


def projected_teacher_forced_run(model, tok, prompt_ids, forced_tokens, qK, qV):
    """Project K,V at every cache.update by Q Q^T round-trip."""
    qK_g = [q.to("cuda", dtype=torch.float16) for q in qK]
    qV_g = [q.to("cuda", dtype=torch.float16) for q in qV]

    cache = DynamicCache()
    orig_update = cache.update
    def patched(key_states, value_states, layer_idx, cache_kwargs=None):
        # Project both K and V (joint, 1024-dim) per layer
        # key_states / value_states: [1, 8, seq, 128]
        b, h, s, d = key_states.shape
        k_flat = key_states.reshape(b, s, h * d)            # [1, seq, 1024]
        v_flat = value_states.reshape(b, s, h * d)
        QK = qK_g[layer_idx]   # [1024, rK]
        QV = qV_g[layer_idx]
        # round trip: x' = (x @ Q) @ Q.T
        k_proj = (k_flat @ QK) @ QK.T
        v_proj = (v_flat @ QV) @ QV.T
        k_back = k_proj.reshape(b, h, s, d).contiguous()
        v_back = v_proj.reshape(b, h, s, d).contiguous()
        return orig_update(k_back, v_back, layer_idx, cache_kwargs)
    cache.update = patched

    logits = []
    with torch.no_grad():
        # Prefill (with projection on)
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
        past = out.past_key_values
        for tok_id in forced_tokens:
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

    print(f"[calib] {len(CALIB_PROMPTS)} prompts x {N_CALIB_DECODE} decode tokens", flush=True)
    qK, qV = calibrate_kv(model, tok)
    print(f"[calib] Q_K[0] shape = {tuple(qK[0].shape)}, Q_V[0] shape = {tuple(qV[0].shape)}", flush=True)

    # Baseline
    eval_ids = tok(EVAL_PROMPT, return_tensors="pt").input_ids.to("cuda")
    print(f"\n[baseline] free run {N_EVAL_DECODE} tokens", flush=True)
    baseline_tokens, baseline_logits = baseline_run(model, tok, eval_ids, N_EVAL_DECODE)
    base_text = tok.decode(baseline_tokens)
    print(f"[baseline] {base_text[:150]!r}", flush=True)

    # Projected
    print(f"\n[projected] K=top-{RANK_K}, V=top-{RANK_V} joint per-layer", flush=True)
    forced = baseline_tokens[:N_EVAL_DECODE]
    proj_logits = projected_teacher_forced_run(model, tok, eval_ids, forced, qK, qV)

    # Metrics
    n = min(len(baseline_logits), len(proj_logits))
    base_argmax = [baseline_logits[i].argmax().item() for i in range(n)]
    proj_argmax = [proj_logits[i].argmax().item() for i in range(n)]
    correct = [int(b == p) for b, p in zip(base_argmax, proj_argmax)]
    top1 = sum(correct) / n

    kls = []
    for i in range(n):
        p_log = F.log_softmax(baseline_logits[i], dim=-1)
        q_log = F.log_softmax(proj_logits[i], dim=-1)
        p = p_log.exp()
        kls.append((p * (p_log - q_log)).sum().item())
    kls.sort()
    mean_kl = sum(kls)/n
    p99_kl = kls[min(n-1, int(0.99 * n))]

    print(f"\n=== B3 RESULT ===", flush=True)
    print(f"  top-1 match rate: {top1:.4f}  ({sum(correct)}/{n})", flush=True)
    print(f"  mean KL:          {mean_kl:.4f}", flush=True)
    print(f"  p99 KL:           {p99_kl:.4f}", flush=True)

    # Per-position breakdown
    print(f"\n  position breakdown:", flush=True)
    for lo, hi in [(0, 50), (50, 100), (100, 150), (150, 200)]:
        sub = correct[lo:hi]
        if sub:
            print(f"    [{lo:>3} .. {hi:>3}): {sum(sub)/len(sub):.4f}  ({sum(sub)}/{len(sub)})", flush=True)

    if top1 >= 0.90 and mean_kl < 0.1:
        print(f"\n  --> JOINT KV PROJECTION VIABLE.", flush=True)
    elif top1 >= 0.85:
        print(f"\n  --> Marginal — preserves most tokens but quality budget tight.", flush=True)
    else:
        print(f"\n  --> Joint KV projection at this rank does not preserve quality.", flush=True)

    out_path = "/home/ubuntu/op31-prod-fix/b3_kv_projection_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "rank_K": RANK_K, "rank_V": RANK_V,
            "top1": top1, "mean_kl": mean_kl, "p99_kl": p99_kl,
            "n_eval": n,
        }, f, indent=2)
    print(f"\n[save] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
