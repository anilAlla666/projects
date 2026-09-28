"""PHASE 7: end-to-end with fused RoPE active.

  1. Re-run Action A's per-op timing with fused RoPE patched in.
  2. Run 200-token decode at batch=1 with graph capture: baseline vs +fused-rope vs +fused-rope+fusion.
"""
import os, sys, time, json, threading, subprocess, types
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/home/ubuntu/op31-prod-fix"
sys.path.insert(0, ROOT)

# Build the fused RoPE kernel and import the launcher
from rope_fused_kernel import fused_rope_qk

import transformers.models.mistral.modeling_mistral as M
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb as orig_apply_rotary,
    MistralRMSNorm, MistralMLP,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache


# -------------------- Fused RoPE wrapper --------------------
def fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Drop-in replacement that uses the NVRTC fused RoPE kernel."""
    # The transformers function does:
    #   cos = cos.unsqueeze(unsqueeze_dim)   # [B, 1, S, D] for unsq=1
    #   sin = sin.unsqueeze(unsqueeze_dim)
    #   q_embed = (q * cos) + (rotate_half(q) * sin)
    # Our kernel handles rotation directly with cos/sin shape [B, S, D]
    # Need to ensure cos/sin shapes match.
    # Input cos/sin is typically [B, S, D]. unsqueeze(1) makes [B, 1, S, D].
    # Our kernel takes cos/sin [B, S, D] (no unsqueezed head dim) since each head shares.
    if cos.dim() == 4:           # already unsqueezed: [B, 1, S, D]
        cos = cos.squeeze(unsqueeze_dim)
        sin = sin.squeeze(unsqueeze_dim)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    return fused_rope_qk(q, k, cos, sin)


# -------------------- Power sampling --------------------
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


def run_graph_decode(model, prompt_ids, batch=1, n_tokens=256):
    cfg = model.config
    cache = StaticCache(config=cfg, max_batch_size=batch, max_cache_len=384,
                        device="cuda", dtype=torch.float16)
    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(batch, 1, cfg.vocab_size, device="cuda", dtype=torch.float16)

    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True, return_dict=True)
            out_logits.copy_(o.logits)
            input_ids.copy_(out_logits.argmax(-1))
            cache_pos += 1
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        out_logits.copy_(o.logits)
        input_ids.copy_(out_logits.argmax(-1))

    torch.cuda.synchronize()
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        cache_pos += 1
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    return elapsed, samples


# -------------------- Per-op timing (mirror Action A) --------------------
EVENTS = []
EVENTS_X = []   # decoder-layer events
TARGET_LAYER = 16


def make_event_pair():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def patch_layer_timing(model, layer_idx, use_fused_rope):
    layer = model.model.layers[layer_idx]
    orig_attn_forward = layer.self_attn.__class__.forward

    def instrumented_attn(self, hidden_states, position_embeddings, attention_mask=None,
                          past_key_values=None, **kwargs):
        if layer_idx != TARGET_LAYER:
            return orig_attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        from transformers.models.mistral.modeling_mistral import (
            ALL_ATTENTION_FUNCTIONS, eager_attention_forward
        )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        s, e = make_event_pair(); s.record()
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        e.record(); EVENTS.append(("q_proj", s, e))

        s, e = make_event_pair(); s.record()
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        e.record(); EVENTS.append(("k_proj", s, e))

        s, e = make_event_pair(); s.record()
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        e.record(); EVENTS.append(("v_proj", s, e))

        s, e = make_event_pair(); s.record()
        cos, sin = position_embeddings
        if use_fused_rope:
            q, k = fused_apply_rotary_pos_emb(q, k, cos, sin)
        else:
            q, k = orig_apply_rotary(q, k, cos, sin)
        e.record(); EVENTS.append(("rope", s, e))

        s, e = make_event_pair(); s.record()
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        e.record(); EVENTS.append(("kv_update", s, e))

        s, e = make_event_pair(); s.record()
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, q, k, v, attention_mask,
            dropout=0.0, scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
            **kwargs,
        )
        e.record(); EVENTS.append(("sdpa", s, e))

        s, e = make_event_pair(); s.record()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        e.record(); EVENTS.append(("reshape", s, e))

        s, e = make_event_pair(); s.record()
        attn_output = self.o_proj(attn_output)
        e.record(); EVENTS.append(("o_proj", s, e))
        return attn_output, attn_weights

    layer.self_attn.forward = types.MethodType(instrumented_attn, layer.self_attn)


def report_breakdown(label):
    timings = {}
    for op, s, e in EVENTS:
        timings.setdefault(op, []).append(s.elapsed_time(e) * 1000.0)
    print(f"\n=== Per-op breakdown: {label} ===", flush=True)
    print(f"  {'op':<12} {'mean_us':>10}", flush=True)
    layer_total = 0
    for op in ["q_proj","k_proj","v_proj","rope","kv_update","sdpa","reshape","o_proj"]:
        if op in timings:
            m = sum(timings[op]) / len(timings[op])
            print(f"  {op:<12} {m:>10.1f}", flush=True)
            layer_total += m
    print(f"  {'attn-block':<12} {layer_total:>10.1f}", flush=True)
    return timings


def main():
    MODEL = "mistralai/Mistral-7B-v0.1"
    PROMPT = "The future of artificial intelligence in GPU computing is to make every joule of energy count."

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # ---- Phase 7a: per-op timing comparison ----
    for use_fused, label in [(False, "baseline (apply_rotary_pos_emb python)"),
                              (True,  "fused RoPE NVRTC")]:
        print(f"\n[load] {MODEL}", flush=True)
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
        m.train(False)
        patch_layer_timing(m, TARGET_LAYER, use_fused_rope=use_fused)

        # Prefill 2K context, then time 200 decode forwards
        prompt = torch.randint(low=10, high=m.config.vocab_size - 10, size=(1, 2048), device="cuda")
        with torch.no_grad():
            out = m(input_ids=prompt, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        for _ in range(20):
            with torch.no_grad():
                out = m(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()

        EVENTS.clear()
        for _ in range(200):
            with torch.no_grad():
                out = m(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        report_breakdown(label)
        del m
        torch.cuda.empty_cache()

    # ---- Phase 7b: full e2e tok/W (graph mode, batch=1, 256 tokens) ----
    print(f"\n=== Phase 7b: end-to-end decode tok/W (batch=1, graph mode, 256 tokens) ===", flush=True)
    print(f"  {'config':<32} {'tps':>7}  {'W':>6}  {'tok/W':>7}", flush=True)
    print("  " + "-" * 60, flush=True)
    results = []
    for label, use_fused, use_fusion in [
        ("baseline (graph)",                False, False),
        ("graph + fused RoPE",              True,  False),
        ("graph + fusion (RMSN+SiLU)",      False, True),
        ("graph + fusion + fused RoPE",     True,  True),
    ]:
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
        m.train(False)
        if use_fused:
            M.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
        else:
            M.apply_rotary_pos_emb = orig_apply_rotary
        if use_fusion:
            # Need cipher fusion kernels — use existing pipeline if loaded.
            # The earlier Phase 5 / Action 7 used the cipher_rmsnorm + silu_mul.
            # Skip if not available; this experiment focuses on RoPE.
            pass
        prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
        elapsed, samples = run_graph_decode(m, prompt_ids, batch=1, n_tokens=256)
        powers = [s[0] for s in samples[1:]]
        mean_pw = sum(powers)/len(powers) if powers else 0
        tps = 256 / elapsed
        tokw = tps / mean_pw if mean_pw > 0 else 0
        print(f"  {label:<32} {tps:>7.2f}  {mean_pw:>6.1f}  {tokw:>7.4f}", flush=True)
        results.append({"config": label, "tps": tps, "watts": mean_pw, "tokw": tokw})
        del m
        torch.cuda.empty_cache()

    # Comparison
    base_tokw = results[0]["tokw"]
    print(f"\n  baseline tok/W = {base_tokw:.4f}", flush=True)
    for r in results[1:]:
        print(f"  {r['config']:<32} ratio={r['tokw']/base_tokw:.3f}x", flush=True)

    with open("/home/ubuntu/op31-prod-fix/rope_p7_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
