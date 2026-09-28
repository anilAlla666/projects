"""ACTION 2: subspace projection with 2000+ calibration samples.

10 prompts × 200 decode tokens = 2000 samples per (layer, point).
Uncentered SVD at k = 64, 128, 256, 499.
Held-out prompt for eval (NOT in calibration).
Decode-only patching (prefill unpatched).
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
N_CALIB_DECODE = 200
N_EVAL_DECODE = 200

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

LINEAR_TO_POINT = {
    "q_proj": "qkv_in", "k_proj": "qkv_in", "v_proj": "qkv_in",
    "o_proj": "o_in",
    "gate_proj": "gate_in", "up_proj": "gate_in",
    "down_proj": "down_in",
}
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
N_LAYERS = 32
K_VALUES = [64, 128, 256, 499]


def calibrate(model, tok):
    captures = {}  # (layer, point) -> list[fp32 cpu tensor]
    capture_enabled = [False]

    def make_pre(li, pt):
        def hook(module, args):
            if not capture_enabled[0]: return
            if not args or not isinstance(args[0], torch.Tensor): return
            x = args[0]
            if x.dim() != 3 or x.shape[1] != 1: return
            captures.setdefault((li, pt), []).append(
                x[0, 0, :].detach().to(torch.float32).cpu())
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(make_pre(i, "qkv_in")))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_pre(i, "o_in")))
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(make_pre(i, "gate_in")))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_pre(i, "down_in")))

    print(f"[calib] {len(CALIB_PROMPTS)} prompts × {N_CALIB_DECODE} decode tokens", flush=True)
    for p_idx, prompt in enumerate(CALIB_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            capture_enabled[0] = False
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            capture_enabled[0] = True
            for _ in range(N_CALIB_DECODE):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            capture_enabled[0] = False
        print(f"[calib]   prompt {p_idx+1}/{len(CALIB_PROMPTS)} done; {len(captures.get((0, 'qkv_in'), []))} samples at L0_qkv", flush=True)
    for h in handles: h.remove()
    return captures


def compute_qs(captures, k_max=499):
    """Uncentered SVD per (layer, point); store top-k_max right singular vectors as Q."""
    qs = {}
    for (li, pt), samples in sorted(captures.items()):
        X = torch.stack(samples, dim=0).to("cuda")
        # UNCENTERED SVD
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        rank = min(Vh.shape[0], k_max)
        Q = Vh[:rank, :].t().contiguous().to("cpu")  # [in, rank]
        qs[(li, pt)] = Q
        del X, U, S, Vh
    torch.cuda.empty_cache()
    return qs


class ProjectedLinear(nn.Module):
    def __init__(self, orig: nn.Linear, Q: torch.Tensor):
        super().__init__()
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        device = orig.weight.device
        with torch.no_grad():
            W = orig.weight.data.to(torch.float32)
            Qf = Q.to(device, dtype=torch.float32)
            M = W @ Qf
        self.register_buffer("Q", Qf.to(torch.float16))
        self.register_buffer("M", M.to(torch.float16))
        self.bias = orig.bias

    def forward(self, x):
        z = x @ self.Q
        y = z @ self.M.t()
        if self.bias is not None:
            y = y + self.bias
        return y


def patch_model(model, qs, k):
    store = {}
    for li, layer in enumerate(model.model.layers):
        for name, point in LINEAR_TO_POINT.items():
            parent = layer.self_attn if name in ATTN else layer.mlp
            orig = getattr(parent, name)
            if not isinstance(orig, nn.Linear): continue
            Q_full = qs[(li, point)]
            k_eff = min(k, Q_full.shape[1])
            Q_k = Q_full[:, :k_eff].contiguous()
            new = ProjectedLinear(orig, Q_k).to(orig.weight.device)
            store[f"L{li}.{name}"] = (parent, name, orig)
            setattr(parent, name, new)
    return store


def unpatch(model, store):
    while store:
        key, (parent, name, orig) = store.popitem()
        setattr(parent, name, orig)


def free_run(model, prompt_ids, n_decode):
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


def teacher_forced(model, prompt_ids, forced, prefill_past=None):
    logits = []
    with torch.no_grad():
        if prefill_past is None:
            out = model(input_ids=prompt_ids, use_cache=True)
            past = out.past_key_values
        else:
            past = prefill_past
        for tok_id in forced:
            inp = torch.tensor([[tok_id]], dtype=torch.long, device=prompt_ids.device)
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits.append(out.logits[0, -1, :].detach().to(torch.float32).cpu())
    return logits


def metrics(base_logits, proj_logits, label):
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
    mean_kl = sum(kls) / n
    print(f"  [{label}] top1={top1:.4f}  mean_kl={mean_kl:.4f}", flush=True)
    bins = [(0, 50), (50, 100), (100, 150), (150, 200)]
    for lo, hi in bins:
        sub = correct[lo:hi]
        if sub:
            print(f"    [{lo:>3}..{hi:>3}): top1={sum(sub)/len(sub):.4f}", flush=True)
    return top1, mean_kl, correct


def main():
    print(f"[load] {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    captures = calibrate(model, tok)
    n_per = len(captures[(0, "qkv_in")])
    print(f"\n[calib] total: {n_per} samples per (layer, point)", flush=True)
    qs = compute_qs(captures, k_max=max(K_VALUES))
    del captures

    # Held-out prompt
    eval_ids = tok(EVAL_PROMPT, return_tensors="pt").input_ids.to("cuda")
    print(f"\n[baseline] HELD-OUT prompt: {EVAL_PROMPT!r}", flush=True)
    base_tokens, base_logits = free_run(model, eval_ids, N_EVAL_DECODE)
    base_text = tok.decode(base_tokens)
    print(f"[baseline] {base_text[:200]!r}", flush=True)

    forced = base_tokens[:N_EVAL_DECODE]

    results = {}
    for k in K_VALUES:
        print(f"\n=== k={k} (decode-only patching, teacher-forced) ===", flush=True)
        # PREFILL UNPATCHED, save past
        with torch.no_grad():
            out = model(input_ids=eval_ids, use_cache=True)
            prefill_past = out.past_key_values
        # Patch
        store = patch_model(model, qs, k)
        try:
            proj_logits = teacher_forced(model, eval_ids, forced, prefill_past=prefill_past)
            top1, mean_kl, correct = metrics(base_logits, proj_logits, f"k={k}")
            results[k] = {"top1": top1, "mean_kl": mean_kl, "correct": correct}
        finally:
            unpatch(model, store)
            torch.cuda.empty_cache()

    # Free-running test at k=256
    if 256 in K_VALUES:
        print(f"\n=== k=256 FREE-RUNNING (no teacher forcing) ===", flush=True)
        with torch.no_grad():
            out = model(input_ids=eval_ids, use_cache=True)
            prefill_past = out.past_key_values
        store = patch_model(model, qs, 256)
        try:
            with torch.no_grad():
                past = prefill_past
                # First decode token: use the prefill's last logit (which used unpatched W) — but we want PROJECTED model behavior.
                # In actual deployment, we'd PREFILL with projected too. For this test, use baseline first token.
                nxt = torch.tensor([[forced[0]]], dtype=torch.long, device="cuda")
                proj_tokens = [forced[0]]
                for _ in range(80):
                    out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    proj_tokens.append(nxt.item())
            text = tok.decode(proj_tokens)
            print(f"  free-run text: {text!r}", flush=True)
        finally:
            unpatch(model, store)
            torch.cuda.empty_cache()

    out_path = "/home/ubuntu/op31-prod-fix/action2_projection_results.json"
    with open(out_path, "w") as f:
        json.dump({"k_values": K_VALUES,
                   "n_calib_samples": n_per,
                   "results": {str(k): {kk: v for kk, v in r.items() if kk != "correct"} for k, r in results.items()},
                   "baseline_text": base_text[:300],
                   }, f, indent=2)
    print(f"\n[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
