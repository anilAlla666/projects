"""ACTION D: Joint KV projection tok/W measurement at batch=1, 2K context, 200 tokens.

Calibrate qK[layer], qV[layer] on prefill+decode of diverse prompts, then run
decode with cache.update intercepted to project KV into top-rank subspace and
back. Compare tok/s and tok/W vs baseline.

Note: this round-trip projection (Q @ Q.T x) doesn't actually shrink the cache
storage — it just adds projection error. To measure actual bandwidth savings,
we'd need to store ONLY the projected k-dim representation. For this experiment
we test the COMPUTE/QUALITY tradeoff; storage savings are theoretical (2.37x).
"""
import os, sys, time, json, threading, subprocess
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL = "mistralai/Mistral-7B-v0.1"
N_DECODE_CALIB = 200
N_DECODE_EVAL  = 200
PROMPT_LEN = 2048

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

N_LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
JOINT = NUM_KV_HEADS * HEAD_DIM


def calibrate(model, tok):
    caps_k = [[] for _ in range(N_LAYERS)]
    caps_v = [[] for _ in range(N_LAYERS)]
    capture_enabled = [False]

    def make_patched(cache):
        orig = cache.update
        def patched(key_states, value_states, layer_idx, cache_kwargs=None):
            if capture_enabled[0]:
                S = key_states.shape[2]
                for s in range(S):
                    caps_k[layer_idx].append(key_states[0, :, s, :].reshape(-1).detach().to(torch.float32).cpu())
                    caps_v[layer_idx].append(value_states[0, :, s, :].reshape(-1).detach().to(torch.float32).cpu())
            return orig(key_states, value_states, layer_idx, cache_kwargs)
        return patched

    for prompt in CALIB_PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        cache = DynamicCache()
        cache.update = make_patched(cache)
        capture_enabled[0] = True
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(N_DECODE_CALIB):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        capture_enabled[0] = False
    print(f"[calib] captured {len(caps_k[0])} per layer", flush=True)

    qK, qV = [], []
    for li in range(N_LAYERS):
        XK = torch.stack(caps_k[li], dim=0).to("cuda")
        XV = torch.stack(caps_v[li], dim=0).to("cuda")
        UK, SK, VhK = torch.linalg.svd(XK, full_matrices=False)
        UV, SV, VhV = torch.linalg.svd(XV, full_matrices=False)
        rK = min(512, VhK.shape[0])
        rV = min(512, VhV.shape[0])
        qK.append(VhK[:rK, :].t().contiguous().to(torch.float16))
        qV.append(VhV[:rV, :].t().contiguous().to(torch.float16))
        del XK, XV, UK, SK, VhK, UV, SV, VhV
    torch.cuda.empty_cache()
    return qK, qV


def power_sampler(stop_evt, samples):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                samples.append((float(parts[0]), int(parts[1])))
        except: pass
        time.sleep(0.2)


def run_decode_timing(model, tok, qK_g, qV_g, K_rank, V_rank, label):
    """Run decode with optional KV projection. Returns (tps, mean_W, tok_W)."""
    cfg = model.config
    torch.manual_seed(0)
    prompt = torch.randint(low=10, high=cfg.vocab_size - 10, size=(1, PROMPT_LEN), device="cuda")

    if K_rank is None:
        cache = DynamicCache()
    else:
        cache = DynamicCache()
        orig = cache.update
        def patched(key_states, value_states, layer_idx, cache_kwargs=None):
            b, h, s, d = key_states.shape
            QK = qK_g[layer_idx][:, :K_rank]
            QV = qV_g[layer_idx][:, :V_rank]
            k_flat = key_states.reshape(b, s, h * d)
            v_flat = value_states.reshape(b, s, h * d)
            k_proj = (k_flat @ QK) @ QK.T
            v_proj = (v_flat @ QV) @ QV.T
            return orig(
                k_proj.reshape(b, h, s, d).contiguous(),
                v_proj.reshape(b, h, s, d).contiguous(),
                layer_idx, cache_kwargs)
        cache.update = patched

    with torch.no_grad():
        out = model(input_ids=prompt, past_key_values=cache, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    # Sample power
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter()
    for _ in range(N_DECODE_EVAL):
        with torch.no_grad():
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

    powers = [s[0] for s in samples[1:]]
    mean_pw = sum(powers)/len(powers) if powers else 0
    tps = N_DECODE_EVAL / elapsed
    tokw = tps / mean_pw if mean_pw > 0 else 0
    print(f"  {label:<35} tps={tps:>6.2f}  W={mean_pw:>6.1f}  tok/W={tokw:.4f}", flush=True)
    return tps, mean_pw, tokw


def main():
    print(f"[load] {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    print(f"\n[calib] {len(CALIB_PROMPTS)} prompts (prefill + {N_DECODE_CALIB} decode)", flush=True)
    qK, qV = calibrate(model, tok)
    qK_g = [q.to("cuda") for q in qK]
    qV_g = [q.to("cuda") for q in qV]
    print(f"[calib] qK[0]={tuple(qK[0].shape)}, qV[0]={tuple(qV[0].shape)}", flush=True)

    print(f"\n=== Decode timing at batch=1, 2K context, {N_DECODE_EVAL} tokens ===", flush=True)
    print(f"  {'config':<35} {'tps':>6}  {'W':>6}  {'tok/W':>7}", flush=True)
    print("  " + "-" * 65, flush=True)

    results = []
    # Baseline (no projection)
    r = run_decode_timing(model, tok, qK_g, qV_g, None, None, "baseline (no projection)")
    results.append({"config": "baseline", "tps": r[0], "watts": r[1], "tokw": r[2]})

    # KV projection at (384, 480) - the 95% top-1 config
    r = run_decode_timing(model, tok, qK_g, qV_g, 384, 480, "KV proj (K=384, V=480) [95% top1]")
    results.append({"config": "kv_384_480", "tps": r[0], "watts": r[1], "tokw": r[2]})

    # KV projection at (256, 320) - higher compression, 91% top-1
    r = run_decode_timing(model, tok, qK_g, qV_g, 256, 320, "KV proj (K=256, V=320) [91% top1]")
    results.append({"config": "kv_256_320", "tps": r[0], "watts": r[1], "tokw": r[2]})

    # Compare
    print(f"\n=== Comparison ===", flush=True)
    base_tokw = results[0]["tokw"]
    for r in results[1:]:
        ratio = r["tokw"] / base_tokw if base_tokw > 0 else 0
        print(f"  {r['config']:<25} tok/W={r['tokw']:.4f}  ratio={ratio:.3f}x", flush=True)

    print(f"\nNOTE: This measurement uses Q @ Q.T round-trip in PyTorch.", flush=True)
    print(f"  ADDS compute (one extra matmul per cache.update) without storage savings.", flush=True)
    print(f"  Real deployment would store ONLY projected k-dim repr, eliminating the round-trip.", flush=True)
    print(f"  Theoretical bandwidth saving: 2.37x KV cache (1024 -> ~432 dims).", flush=True)

    with open("/home/ubuntu/op31-prod-fix/action_d_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
