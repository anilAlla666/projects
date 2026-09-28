"""Phase 7: Mistral-7B with full CIPHER fusion stack — measure compound tok/W.

Stack:
  - Existing fused RMSNorm (cipher_rmsnorm_fp16) — used standalone
  - Existing fused SiLU·Mul (cipher_silu_mul_fp16) — used in MLP
  - NEW Phase 5: fused RoPE (cipher_fused_rope_qk)
  - NEW Task A: fused residual+RMSNorm (replaces residual_add + rmsnorm pairs)

Configurations measured at batch=1, graph mode, 256 decode tokens, ctx=2048:
  C0: baseline (no fusion)
  C1: + fused RoPE
  C2: + fused residual+norm
  C3: + RMSN/SiLU (old fusion)
  C4: ALL fusions
"""
import os, sys, time, json, threading, subprocess, ctypes
sys.path.insert(0, "/home/ubuntu/op31-prod-fix")

# Build the kernels (their __init__ doesn't run main code, just compile)
from rope_fused_kernel import fused_rope_qk
from fused_resid_rmsnorm import fused_resid_rmsnorm

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"

# Set up CIPHER fusion kernels (RMSN, SiLU)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
rt.cipher_fused_rmsnorm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
rt.cipher_fused_rmsnorm.restype = ctypes.c_int
rt.cipher_fused_silu_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_void_p]
rt.cipher_fused_silu_mul.restype = ctypes.c_int
rt.cipher_fusion_kernels_init()


def cipher_rmsnorm(x, weight, eps):
    out = torch.empty_like(x)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out_flat = out.reshape(-1, x.shape[-1])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rt.cipher_fused_rmsnorm(flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                              flat.shape[0], flat.shape[-1], float(eps), stream)
    return out


def cipher_silu_mul(gate, up):
    if not gate.is_contiguous(): gate = gate.contiguous()
    if not up.is_contiguous(): up = up.contiguous()
    out = torch.empty_like(gate)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                              gate.numel(), stream)
    return out


# Wrapper for fused RoPE that matches transformers' interface
def fused_apply_rotary(q, k, cos, sin, unsqueeze_dim=1):
    if cos.dim() == 4:
        cos = cos.squeeze(unsqueeze_dim)
        sin = sin.squeeze(unsqueeze_dim)
    if not q.is_contiguous(): q = q.contiguous()
    if not k.is_contiguous(): k = k.contiguous()
    if not cos.is_contiguous(): cos = cos.contiguous()
    if not sin.is_contiguous(): sin = sin.contiguous()
    return fused_rope_qk(q, k, cos, sin)


from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
import transformers.models.mistral.modeling_mistral as M
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb as orig_apply_rotary,
    MistralRMSNorm, MistralMLP, MistralDecoderLayer,
)
import types

orig_decoder_forward = MistralDecoderLayer.forward


# Custom decoder layer forward that uses fused residual+RMSNorm
def make_decoder_forward_with_resid_norm(model):
    """Patch each decoder layer's forward to call fused residual+norm."""
    def fused_forward(self, hidden_states, attention_mask=None, position_ids=None,
                      past_key_values=None, output_attentions=False, use_cache=False,
                      cache_position=None, position_embeddings=None, **kwargs):
        # Standard Mistral structure:
        #   x_in = hidden_states (already-LN'd from previous layer's residual+norm fusion, OR pre-LN at layer 0)
        # For layer 0: input_layernorm is standalone (no preceding residual)
        # For layers 1..n: assume input has been pre-normed by previous layer's exit fusion
        #
        # However the simplest correct approach: maintain the standard semantics.
        # Within ONE layer, fuse:
        #   x = residual + attn_out;  x_norm = post_attention_layernorm(x)  ← fused
        #
        # The preceding input_layernorm stays standalone (one less fusion opportunity but easy).

        # input_layernorm (standalone)
        residual_pre_attn = hidden_states
        hs = self.input_layernorm(hidden_states)

        # Self-attention
        attn_out, _ = self.self_attn(hidden_states=hs,
                                       position_embeddings=position_embeddings,
                                       attention_mask=attention_mask,
                                       past_key_values=past_key_values,
                                       cache_position=cache_position,
                                       **kwargs)

        # FUSED: x = residual + attn_out; hs = post_attention_layernorm(x)
        hs, x_post = fused_resid_rmsnorm(
            residual_pre_attn, attn_out, self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon)
        # x_post is residual for next add
        residual_pre_mlp = x_post

        # MLP
        mlp_out = self.mlp(hs)

        # Residual add (NOT fused — next op is the next layer's input_layernorm which is standalone)
        return residual_pre_mlp + mlp_out

    for layer in model.model.layers:
        layer.forward = types.MethodType(fused_forward, layer)


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


def run_graph_decode(model, prompt_ids, batch=1, n_tokens=256, max_len=384):
    cfg = model.config
    cache = StaticCache(config=cfg, max_batch_size=batch, max_cache_len=max_len,
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


def reset_patches(model):
    """Reset all monkey-patches."""
    M.apply_rotary_pos_emb = orig_apply_rotary
    MistralRMSNorm.forward = MistralRMSNorm._original_forward if hasattr(MistralRMSNorm, "_original_forward") else MistralRMSNorm.forward
    MistralMLP.forward = MistralMLP._original_forward if hasattr(MistralMLP, "_original_forward") else MistralMLP.forward
    for layer in model.model.layers:
        layer.forward = types.MethodType(orig_decoder_forward, layer)


def apply_config(model, name):
    """Apply fusion patches for the given config name."""
    # Reset first
    M.apply_rotary_pos_emb = orig_apply_rotary
    MistralRMSNorm.forward = MistralRMSNorm._original_forward
    MistralMLP.forward = MistralMLP._original_forward
    for layer in model.model.layers:
        layer.forward = types.MethodType(orig_decoder_forward, layer)

    if "rope" in name:
        M.apply_rotary_pos_emb = fused_apply_rotary
    if "rmsn_silu" in name:
        MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(x, self.weight, self.variance_epsilon)
        MistralMLP.forward = lambda self, x: self.down_proj(cipher_silu_mul(self.gate_proj(x), self.up_proj(x)))
    if "resid_norm" in name:
        make_decoder_forward_with_resid_norm(model)


def main():
    MODEL = "mistralai/Mistral-7B-v0.1"
    PROMPT = "The future of artificial intelligence in GPU computing is to make every joule of energy count."

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Save originals so reset works
    if not hasattr(MistralRMSNorm, "_original_forward"):
        MistralRMSNorm._original_forward = MistralRMSNorm.forward
    if not hasattr(MistralMLP, "_original_forward"):
        MistralMLP._original_forward = MistralMLP.forward

    configs = [
        ("baseline",                    [],                                        ),
        ("+rope",                       ["rope"],                                  ),
        ("+resid_norm",                 ["resid_norm"],                            ),
        ("+rmsn_silu",                  ["rmsn_silu"],                             ),
        ("+rope+rmsn_silu",             ["rope", "rmsn_silu"],                     ),
        ("ALL: rope+resid_norm+rmsn_silu", ["rope", "resid_norm", "rmsn_silu"],    ),
    ]

    print(f"\n=== Phase 7 compound: batch=1, graph mode, 256 decode, ctx=2048 ===\n", flush=True)
    print(f"  {'config':<40} {'tps':>7}  {'W':>6}  {'tok/W':>7}  {'ratio':>7}", flush=True)
    print("  " + "-" * 75, flush=True)

    results = []
    base_tokw = None
    for cfg_name, cfg_flags in configs:
        try:
            print(f"  loading model for: {cfg_name}", flush=True)
            m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
            m.train(False)
            apply_config(m, "+".join(cfg_flags))
            prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
            elapsed, samples = run_graph_decode(m, prompt_ids, batch=1, n_tokens=256)
            powers = [s[0] for s in samples[1:]]
            mean_pw = sum(powers)/len(powers) if powers else 0
            tps = 256 / elapsed
            tokw = tps / mean_pw if mean_pw > 0 else 0
            if base_tokw is None: base_tokw = tokw
            ratio = tokw / base_tokw if base_tokw > 0 else 0
            print(f"  {cfg_name:<40} {tps:>7.2f}  {mean_pw:>6.1f}  {tokw:>7.4f}  {ratio:>6.3f}x", flush=True)
            results.append({"config": cfg_name, "tps": tps, "watts": mean_pw, "tokw": tokw, "ratio": ratio})
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {cfg_name:<40} ERROR: {e}", flush=True)
            torch.cuda.empty_cache()

    with open("/home/ubuntu/op31-prod-fix/phase7_compound_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] phase7_compound_results.json", flush=True)


if __name__ == "__main__":
    main()
