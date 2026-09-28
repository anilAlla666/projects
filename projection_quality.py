"""Subspace-projection quality test for Mistral-7B.

Replaces every nn.Linear in the 32 decoder layers with a low-rank approximation
y = (x @ Q) @ M.t() where M = W @ Q and Q is the top-k right singular vectors
of that linear's *input* distribution observed during calibration.

Compares baseline (full W) vs projected (low-rank) on identical decode tokens:
  - top-1 token match rate
  - mean / p99 KL divergence on logits
  - per-layer residual cosine similarity
  - free-running coherence check

Uses the adaptive-k profile derived from the residual-stream PCA:
  v1: k=16 for layers 2-22, k=64 for 23-29, k=256 for 0,1,30,31
  v2: k=32 for layers 2-22, k=128 for 23-29, k=256 for 0,1,30,31
"""
import os
import sys
import time
import json
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
N_LAYERS = 32

CALIB_PROMPTS = [
    "Explain how a CPU executes instructions step by step",
    "Write a story about a detective solving a mystery in Tokyo",
    "What are the economic implications of rising interest rates",
    "Describe the process of photosynthesis in detail",
    "Compare and contrast democracy and authoritarianism",
]
N_CALIB_DECODE = 100
EVAL_PROMPT = CALIB_PROMPTS[0]
N_EVAL_DECODE = 200

LINEAR_TO_POINT = {
    "q_proj": "qkv_in", "k_proj": "qkv_in", "v_proj": "qkv_in",
    "o_proj": "o_in",
    "gate_proj": "gate_in", "up_proj": "gate_in",
    "down_proj": "down_in",
}
ATTN_LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_LINEARS = {"gate_proj", "up_proj", "down_proj"}

K_PROFILES = {
    "v1": {**{i: 256 for i in (0, 1, 30, 31)},
           **{i: 16 for i in range(2, 23)},
           **{i: 64 for i in range(23, 30)}},
    "v2": {**{i: 256 for i in (0, 1, 30, 31)},
           **{i: 32 for i in range(2, 23)},
           **{i: 128 for i in range(23, 30)}},
    # uniform k sanity probes: does the algorithm work at all when k is large?
    "v3_uniform_256": {i: 256 for i in range(N_LAYERS)},
    "v4_uniform_512": {i: 499 for i in range(N_LAYERS)},  # 500 samples => max rank 499
}
K_MAX = 499  # store up to 499 PCs per (layer, point) — full rank given 500 calibration samples


# -------------------- Phase 1: load --------------------
def load_model():
    print(f"[load] {MODEL} fp16 -> cuda")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)
    print(f"[load] done in {time.time()-t0:.1f}s")
    return model, tok


# -------------------- Phase 2: calibrate --------------------
def calibrate(model, tok):
    """Capture per-linear inputs across CALIB_PROMPTS x N_CALIB_DECODE decode tokens."""
    captures = {}  # (layer_idx, point) -> list of cpu fp32 tensors
    capture_enabled = [False]

    def make_pre(layer_idx, point):
        def hook(module, args):
            if not capture_enabled[0]:
                return None
            if len(args) == 0 or not isinstance(args[0], torch.Tensor):
                return None
            x = args[0]
            if x.dim() != 3 or x.shape[1] != 1:
                return None
            captures.setdefault((layer_idx, point), []).append(
                x[0, 0, :].detach().to(torch.float32).cpu()
            )
            return None
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(make_pre(i, "qkv_in")))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_pre(i, "o_in")))
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(make_pre(i, "gate_in")))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_pre(i, "down_in")))
    print(f"[calib] hooks={len(handles)}; running {len(CALIB_PROMPTS)} prompts x {N_CALIB_DECODE} decode tokens")

    t0 = time.time()
    for p_idx, prompt in enumerate(CALIB_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            capture_enabled[0] = False
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            capture_enabled[0] = True
            for s in range(N_CALIB_DECODE):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            capture_enabled[0] = False
        print(f"[calib]   prompt {p_idx+1}/{len(CALIB_PROMPTS)} done")
    for h in handles:
        h.remove()
    counts = [len(captures[(i, p)]) for i in range(N_LAYERS) for p in ("qkv_in", "o_in", "gate_in", "down_in")]
    print(f"[calib] capture counts min={min(counts)} max={max(counts)} elapsed={time.time()-t0:.1f}s")
    return captures


# -------------------- Phase 3: SVD per (layer, point) --------------------
def compute_qs(captures, k_max=K_MAX):
    """Returns {(layer, point): {'Q': cpu fp32 [in, k_max], 'cum': cpu fp32 [k_max]}}"""
    qs = {}
    print(f"[svd] computing top-{k_max} singular vectors (uncentered) for {len(captures)} (layer, point) pairs")
    t0 = time.time()
    for (li, pt), samples in sorted(captures.items()):
        X = torch.stack(samples, dim=0)
        # Uncentered SVD: at k = min(N, D), col(Q) spans the calibration row space exactly,
        # so Q Q^T x = x for x in {calibration inputs} and W Q Q^T x = W x.
        # Centering would drop the mean direction even at full rank (silent bug).
        Xg = X.to("cuda")
        U, S, Vh = torch.linalg.svd(Xg, full_matrices=False)
        rank = min(Vh.shape[0], k_max)
        Q = Vh[:rank, :].t().contiguous().to("cpu")  # [in_features, rank]
        var = (S ** 2)
        cum = (torch.cumsum(var, dim=0) / var.sum())[:rank].to("cpu")
        qs[(li, pt)] = {"Q": Q, "cum": cum}
        del Xg, U, S, Vh
    torch.cuda.empty_cache()
    print(f"[svd] done in {time.time()-t0:.1f}s")
    return qs


# -------------------- ProjectedLinear --------------------
class ProjectedLinear(nn.Module):
    def __init__(self, orig: nn.Linear, Q: torch.Tensor, dtype=torch.float16):
        super().__init__()
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.k = Q.shape[1]
        device = orig.weight.device
        with torch.no_grad():
            W = orig.weight.data.to(torch.float32)  # [out, in]
            Qf = Q.to(device=device, dtype=torch.float32)
            M = W @ Qf  # [out, k]
        self.register_buffer("Q", Qf.to(dtype))
        self.register_buffer("M", M.to(dtype))
        self.bias = orig.bias  # Mistral linears have bias=None

    def forward(self, x):
        # x: [..., in]
        z = x @ self.Q          # [..., k]
        y = z @ self.M.t()      # [..., out]
        if self.bias is not None:
            y = y + self.bias
        return y


def patch_model(model, qs, k_profile, store):
    """Replace nn.Linear children of decoder layers with ProjectedLinear. store is a dict to record originals."""
    cum_vars = {}  # (layer, linear) -> cum_var at chosen k
    for li, layer in enumerate(model.model.layers):
        k = k_profile[li]
        for name in list(LINEAR_TO_POINT):
            point = LINEAR_TO_POINT[name]
            parent = layer.self_attn if name in ATTN_LINEARS else layer.mlp
            orig = getattr(parent, name)
            assert isinstance(orig, nn.Linear), f"layers.{li}.{name} not nn.Linear (already patched?)"
            entry = qs[(li, point)]
            Q_full = entry["Q"]
            cum = entry["cum"]
            k_eff = min(k, Q_full.shape[1])
            Q_k = Q_full[:, :k_eff].contiguous()
            new_mod = ProjectedLinear(orig, Q_k).to(orig.weight.device)
            store[f"layers.{li}.{name}"] = (parent, name, orig)
            setattr(parent, name, new_mod)
            cum_vars[(li, name)] = cum[k_eff - 1].item() if k_eff <= len(cum) else 1.0
    return cum_vars


def unpatch_model(model, store):
    while store:
        key, (parent, name, orig) = store.popitem()
        setattr(parent, name, orig)


# -------------------- Sanity test --------------------
def sanity_test_projected_linear():
    print("[sanity] testing ProjectedLinear math")
    torch.manual_seed(0)
    in_f, out_f = 4096, 1024
    orig = nn.Linear(in_f, out_f, bias=False).to("cuda").half()

    # Random low-rank-ish input distribution
    R = 32
    basis = torch.randn(in_f, R, device="cuda", dtype=torch.float32)
    coefs = torch.randn(500, R, device="cuda", dtype=torch.float32)
    X = (coefs @ basis.t()).to(torch.float16)  # [500, in]

    # SVD on input
    Xc = X.to(torch.float32) - X.to(torch.float32).mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    cum = torch.cumsum(var, dim=0) / var.sum()

    for k in [4, 8, 16, 32, 64]:
        Q = Vh[:k, :].t().contiguous()  # [in, k]
        proj = ProjectedLinear(orig, Q).to("cuda")
        with torch.no_grad():
            y_full = orig(X)             # [500, out]
            y_proj = proj(X)
            err_norm = (y_full.float() - y_proj.float()).norm() / y_full.float().norm()
            cum_var = cum[k - 1].item()
            unexplained = 1.0 - cum_var
        print(f"[sanity]   k={k:3d}  cum_var={cum_var:.6f}  ||err||/||y||={err_norm:.6f}  expected~{(unexplained**0.5):.6f}")
    # Full rank test
    Q_full = Vh.t().contiguous()  # [in, min(N, in)]
    proj_full = ProjectedLinear(orig, Q_full).to("cuda")
    with torch.no_grad():
        y_full = orig(X)
        y_proj_full = proj_full(X)
        err = (y_full.float() - y_proj_full.float()).norm() / y_full.float().norm()
    print(f"[sanity]   full rank err={err:.2e}  (should be ~0 — input lies in row span of Vh)")
    del orig, proj
    torch.cuda.empty_cache()


# -------------------- Free-run baseline --------------------
def free_run_baseline(model, prompt_ids, n_decode):
    layer_inputs = [[] for _ in range(N_LAYERS)]
    capture_enabled = [False]

    def make_hook(li):
        def pre(module, args, kwargs):
            if not capture_enabled[0]:
                return None
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if not isinstance(hs, torch.Tensor) or hs.dim() != 3 or hs.shape[1] != 1:
                return None
            layer_inputs[li].append(hs[0, 0, :].detach().to(torch.float32).cpu())
            return None
        return pre

    handles = [layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
               for i, layer in enumerate(model.model.layers)]

    decoded_tokens = []  # tokens produced (T_P, T_{P+1}, ..., T_{P+n_decode})
    decode_logits = []   # logits at decode-step output positions
    with torch.no_grad():
        capture_enabled[0] = False
        out = model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoded_tokens.append(nxt.item())  # T_P

        capture_enabled[0] = True
        for s in range(n_decode):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            decode_logits.append(out.logits[0, -1, :].detach().to(torch.float32).cpu())
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded_tokens.append(nxt.item())  # T_{P+s+1}
        capture_enabled[0] = False
    for h in handles:
        h.remove()
    return {
        "decoded_tokens": decoded_tokens,    # length n_decode + 1
        "decode_logits": decode_logits,      # length n_decode
        "layer_inputs": layer_inputs,        # 32 lists of length n_decode
    }


# -------------------- Teacher-forced run --------------------
def teacher_forced_run(model, prompt_ids, forced_tokens, prefill_past=None):
    """Feeds prompt then forced_tokens[0..len-1] one at a time. Returns logits AFTER each forced input.

    If prefill_past is provided, skip the prefill forward (use the supplied KV cache instead).
    This lets the caller patch the model AFTER prefill so prefill uses full W (decode-only patching).
    """
    layer_inputs = [[] for _ in range(N_LAYERS)]
    capture_enabled = [False]

    def make_hook(li):
        def pre(module, args, kwargs):
            if not capture_enabled[0]:
                return None
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if not isinstance(hs, torch.Tensor) or hs.dim() != 3 or hs.shape[1] != 1:
                return None
            layer_inputs[li].append(hs[0, 0, :].detach().to(torch.float32).cpu())
            return None
        return pre

    handles = [layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
               for i, layer in enumerate(model.model.layers)]

    logits_list = []
    with torch.no_grad():
        if prefill_past is None:
            capture_enabled[0] = False
            out = model(input_ids=prompt_ids, use_cache=True)
            past = out.past_key_values
        else:
            past = prefill_past
        capture_enabled[0] = True
        for tok_id in forced_tokens:
            inp = torch.tensor([[tok_id]], dtype=torch.long, device=prompt_ids.device)
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits_list.append(out.logits[0, -1, :].detach().to(torch.float32).cpu())
        capture_enabled[0] = False
    for h in handles:
        h.remove()
    return {"logits": logits_list, "layer_inputs": layer_inputs}


# -------------------- Free-run for projected (qualitative) --------------------
def free_run_projected_text(model, tok, prompt_ids, n_decode):
    out_ids = []
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out_ids.append(nxt.item())
        for s in range(n_decode):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out_ids.append(nxt.item())
    return tok.decode(out_ids)


# -------------------- Metrics --------------------
def compute_metrics(base, proj):
    n = min(len(base["decode_logits"]), len(proj["logits"]))
    base_argmax = [base["decode_logits"][i].argmax().item() for i in range(n)]
    proj_argmax = [proj["logits"][i].argmax().item() for i in range(n)]
    top1 = sum(int(b == p) for b, p in zip(base_argmax, proj_argmax)) / n

    kls = []
    for i in range(n):
        p_log = F.log_softmax(base["decode_logits"][i], dim=-1)
        q_log = F.log_softmax(proj["logits"][i], dim=-1)
        p = p_log.exp()
        kl = (p * (p_log - q_log)).sum().item()
        kls.append(kl)
    kls_sorted = sorted(kls)
    mean_kl = sum(kls) / len(kls)
    p99_kl = kls_sorted[min(len(kls) - 1, int(0.99 * len(kls)))]

    layer_cos = []
    for li in range(N_LAYERS):
        bi = base["layer_inputs"][li]
        pi = proj["layer_inputs"][li]
        m = min(len(bi), len(pi))
        if m == 0:
            layer_cos.append(float("nan"))
            continue
        cs_per = []
        for s in range(m):
            cs = F.cosine_similarity(bi[s].unsqueeze(0), pi[s].unsqueeze(0), dim=-1).item()
            cs_per.append(cs)
        layer_cos.append(sum(cs_per) / m)

    return {
        "n": n,
        "top1_match": top1,
        "mean_kl": mean_kl,
        "p99_kl": p99_kl,
        "layer_cos": layer_cos,
        "kls": kls,
    }


def run_projected_variant(model, tok, qs, k_profile, prompt_ids, baseline, name, decode_only=False):
    print(f"\n========== Projected {name}{' [decode-only]' if decode_only else ''} ==========")
    prefill_past = None
    if decode_only:
        # Run prefill with full W, save KV cache, then patch and decode.
        with torch.no_grad():
            out = model(input_ids=prompt_ids, use_cache=True)
            prefill_past = out.past_key_values
        print(f"[{name}] prefill done with full W; KV cache saved")

    store = {}
    cum_vars = patch_model(model, qs, k_profile, store)
    print(f"[{name}] patched {len(store)} linears. Per-linear cum-variance at chosen k:")
    items = sorted(cum_vars.items(), key=lambda kv: kv[1])
    print(f"[{name}] 5 worst (layer, linear) by cum-var:")
    for (li, ln), cv in items[:5]:
        k = k_profile[li]
        print(f"           layer {li:>2} {ln:<10} k={k:>3} cum_var={cv:.4f}")
    by_layer = {}
    for (li, _), cv in cum_vars.items():
        by_layer.setdefault(li, []).append(cv)
    avg_per_layer = {li: sum(vs)/len(vs) for li, vs in by_layer.items()}

    try:
        proj = teacher_forced_run(model, prompt_ids,
                                  baseline["decoded_tokens"][:N_EVAL_DECODE],
                                  prefill_past=prefill_past)
        metrics = compute_metrics(baseline, proj)
        proj_text = free_run_projected_text(model, tok, prompt_ids, n_decode=80)
    finally:
        unpatch_model(model, store)
        torch.cuda.empty_cache()

    return {
        "metrics": metrics,
        "proj_text": proj_text,
        "cum_vars": cum_vars,
        "avg_per_layer": avg_per_layer,
        "k_profile": dict(k_profile),
        "decode_only": decode_only,
    }


def print_metrics(name, result, k_profile):
    m = result["metrics"]
    print(f"\n[{name}] === metrics over {m['n']} decode steps ===")
    print(f"[{name}] top-1 match rate: {m['top1_match']:.4f}")
    print(f"[{name}] mean KL(base||proj): {m['mean_kl']:.4f}")
    print(f"[{name}] p99 KL:              {m['p99_kl']:.4f}")
    print(f"[{name}] per-layer residual cosine sim (and k):")
    for li in range(N_LAYERS):
        cs = m["layer_cos"][li]
        cv = result["avg_per_layer"][li]
        print(f"           layer {li:>2}  k={k_profile[li]:>3}  cum_var_avg={cv:.4f}  cos_sim={cs:.4f}")
    print(f"\n[{name}] free-running projected text (first 80 decoded tokens):")
    print(f"    {result['proj_text']!r}")


# -------------------- Main --------------------
def main():
    model, tok = load_model()
    sanity_test_projected_linear()

    # Calibration
    captures = calibrate(model, tok)
    qs = compute_qs(captures, k_max=K_MAX)
    del captures
    gc.collect()

    # Baseline run
    print(f"\n========== Baseline (full W) ==========")
    eval_ids = tok(EVAL_PROMPT, return_tensors="pt").input_ids.to("cuda")
    print(f"[baseline] prompt={EVAL_PROMPT!r} prompt_len={eval_ids.shape[1]}")
    t0 = time.time()
    baseline = free_run_baseline(model, eval_ids, N_EVAL_DECODE)
    print(f"[baseline] generated {len(baseline['decoded_tokens'])} tokens in {time.time()-t0:.1f}s")
    base_text = tok.decode(baseline["decoded_tokens"])
    print(f"[baseline] free-running text:")
    print(f"    {base_text!r}")

    results = {}

    # DIAGNOSTIC: k=499 (full rank), decode-only patching.
    # Tests whether the algorithm composes through 32 layers when calibration is aligned with eval.
    # If top-1 ~ 100%: algorithm is sound, sweep k down to find quality threshold.
    # If top-1 << 90%: error compounds even with aligned calibration; algorithm has a real ceiling.
    res_diag = run_projected_variant(model, tok, qs, K_PROFILES["v4_uniform_512"], eval_ids, baseline,
                                     "diag_k499_decode_only", decode_only=True)
    print_metrics("diag_k499_decode_only", res_diag, K_PROFILES["v4_uniform_512"])
    results["diag_k499_decode_only"] = res_diag

    diag_top1 = res_diag["metrics"]["top1_match"]
    print(f"\n[diag] top-1 = {diag_top1:.4f}")
    print("[diag] sweeping k to map the quality vs k curve under decode-only patching")
    sweep_profiles = [
        ("k256_decode_only", K_PROFILES["v3_uniform_256"]),
        ("k128_decode_only", {i: 128 for i in range(N_LAYERS)}),
        ("k64_decode_only",  {i: 64  for i in range(N_LAYERS)}),
        ("k32_decode_only",  {i: 32  for i in range(N_LAYERS)}),
        ("v1_decode_only",   K_PROFILES["v1"]),
        ("v2_decode_only",   K_PROFILES["v2"]),
    ]
    for sname, sprof in sweep_profiles:
        r = run_projected_variant(model, tok, qs, sprof, eval_ids, baseline, sname, decode_only=True)
        print_metrics(sname, r, sprof)
        results[sname] = r

    # Quality verdict
    print(f"\n========== VERDICT ==========")
    for name in results:
        m = results[name]["metrics"]
        passed = (m["top1_match"] > 0.90) and (m["mean_kl"] < 0.1)
        print(f"  {name}: top1={m['top1_match']:.4f}  mean_kl={m['mean_kl']:.4f}  -> {'PASS' if passed else 'FAIL'}")

    # Save
    out_path = "/home/ubuntu/op31-prod-fix/projection_quality_results.json"
    serial = {}
    for name, r in results.items():
        m = r["metrics"]
        serial[name] = {
            "k_profile": {str(k): v for k, v in r["k_profile"].items()},
            "top1_match": m["top1_match"],
            "mean_kl": m["mean_kl"],
            "p99_kl": m["p99_kl"],
            "n": m["n"],
            "layer_cos": m["layer_cos"],
            "avg_cum_var_per_layer": r["avg_per_layer"],
            "cum_vars_per_linear": {f"L{li}.{ln}": cv for (li, ln), cv in r["cum_vars"].items()},
            "proj_text": r["proj_text"],
        }
    serial["baseline_text"] = base_text
    with open(out_path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"\n[save] results -> {out_path}")


if __name__ == "__main__":
    main()
