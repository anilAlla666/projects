"""AWQ activation-aware calibration for Mistral-7B INT4 quantization.

Per-channel activation absmax is collected during a calibration pass across
several decoded prompts. AWQ scale per input channel:

    s_j = clip( |x_j|_amax^alpha / |W[:, j]|_amax^(1-alpha) , min=1e-4 )
    s   = s / geo_mean(s)         # normalize so magnitudes don't drift

After scaling, W' = W * s[None, :]  (each input column scaled), and at forward
time the activation is divided by s before the linear:  x' = x / s.

Equivalence:  y = x @ W^T = (x / s) @ (W * s)^T  ✓ for any positive s.

The scaled W' has flatter per-column magnitudes (outlier channels are tamed by
s_j), so absmax INT4 quantization wastes fewer bits on the outliers and the
remaining channels keep more dynamic range.

Saves:
    /home/ubuntu/op31-prod-fix/awq_scales.pt  -> {(layer_idx, point) -> [in_features]}

Where point ∈ {"qkv_in", "o_in", "gate_in", "down_in"} (the 4 unique input
distributions per Mistral decoder layer).
"""
import os
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
OUT_PATH = "/home/ubuntu/op31-prod-fix/awq_scales.pt"
ALPHA = float(os.environ.get("AWQ_ALPHA", "0.5"))
N_DECODE = 64    # calibration decode tokens per prompt
SCALE_MIN = 1e-4

CALIB_PROMPTS = [
    "Explain how a CPU executes instructions step by step",
    "Write a story about a detective solving a mystery in Tokyo",
    "What are the economic implications of rising interest rates",
    "Describe the process of photosynthesis in detail",
    "Compare and contrast democracy and authoritarianism",
    "List the steps to bake sourdough bread at home",
    "Translate to French: I would like to learn more about your country",
    "Summarize the plot of Hamlet in three sentences",
]

LINEAR_TO_POINT = {
    "q_proj": "qkv_in", "k_proj": "qkv_in", "v_proj": "qkv_in",
    "o_proj": "o_in",
    "gate_proj": "gate_in", "up_proj": "gate_in",
    "down_proj": "down_in",
}
ATTN_LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def main():
    print(f"[load] {MODEL}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)
    n_layers = len(model.model.layers)
    print(f"[load] done in {time.time()-t0:.1f}s, {n_layers} layers")

    # Per-(layer, point) running absmax, shape [in_features]
    amax = {}  # (layer_idx, point) -> torch.Tensor on cuda

    def make_pre(layer_idx, point):
        def hook(module, args):
            if not capture_enabled[0]:
                return None
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            x = args[0]
            # x shape during prefill: [1, prompt_len, in]; during decode: [1, 1, in]
            x_abs = x.detach().abs().float().reshape(-1, x.shape[-1])  # [tokens, in]
            cur = x_abs.amax(dim=0)  # [in]
            key = (layer_idx, point)
            prev = amax.get(key)
            if prev is None:
                amax[key] = cur
            else:
                amax[key] = torch.maximum(prev, cur)
            return None
        return hook

    capture_enabled = [False]
    handles = []
    for i, layer in enumerate(model.model.layers):
        for name, point in LINEAR_TO_POINT.items():
            parent = layer.self_attn if name in ATTN_LINEARS else layer.mlp
            mod = getattr(parent, name)
            handles.append(mod.register_forward_pre_hook(make_pre(i, point)))
    print(f"[hook] registered {len(handles)} forward_pre_hooks (4 unique points x 32 layers covers all 7 linears)")

    # Capture across BOTH prefill and decode — AWQ wants the full activation distribution
    print(f"[calib] {len(CALIB_PROMPTS)} prompts x ({{prefill}} + {N_DECODE} decode tokens)")
    capture_enabled[0] = True
    for p_idx, prompt in enumerate(CALIB_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for s in range(N_DECODE):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        print(f"[calib]   prompt {p_idx+1}/{len(CALIB_PROMPTS)} done")
    capture_enabled[0] = False
    for h in handles:
        h.remove()

    # ---- Compute AWQ scales per linear (each linear keyed by the layer's point) ----
    # We need W column absmax per linear. Collect from model.
    print(f"[awq] computing per-linear AWQ scales (alpha={ALPHA})")
    scales = {}  # (layer_idx, name) -> [in_features] cpu fp32 tensor
    for li, layer in enumerate(model.model.layers):
        for name, point in LINEAR_TO_POINT.items():
            parent = layer.self_attn if name in ATTN_LINEARS else layer.mlp
            mod = getattr(parent, name)
            W = mod.weight.detach().float()  # [out, in]
            x_amax = amax[(li, point)]  # [in], cuda fp32
            w_amax = W.abs().amax(dim=0)  # [in], cuda fp32

            # Avoid div-by-zero / log of zero
            x_amax = x_amax.clamp(min=SCALE_MIN)
            w_amax = w_amax.clamp(min=SCALE_MIN)

            s = (x_amax.pow(ALPHA) / w_amax.pow(1.0 - ALPHA))
            s = s.clamp(min=SCALE_MIN)
            # Geometric-mean normalize so the s tensor has unit geomean (keeps W' magnitudes balanced)
            log_s = s.log()
            s = (log_s - log_s.mean()).exp()
            scales[(li, name)] = s.cpu()

    # Sanity: print scale distribution stats per layer for the 4 input types (one example per type per layer)
    print("\n[awq] per-layer scale distribution stats (one representative linear per input type):")
    print(f"  {'layer':>5}  {'q_proj scale':>20}  {'o_proj scale':>20}  {'gate_proj scale':>22}  {'down_proj scale':>22}")
    for li in [0, 5, 15, 25, 31]:
        def stat(name):
            s = scales[(li, name)]
            return f"min={s.min():.3f} max={s.max():.3f} std={s.std():.3f}"
        print(f"  {li:>5}  {stat('q_proj'):>20}  {stat('o_proj'):>20}  {stat('gate_proj'):>22}  {stat('down_proj'):>22}")

    # ---- Verify: compare quantization rel-err WITH vs WITHOUT AWQ scales on a sample weight ----
    print("\n[verify] simulating INT4 quantization rel-err with/without AWQ on layer 15 down_proj:")
    layer = model.model.layers[15]
    W = layer.mlp.down_proj.weight.detach().float()  # [4096, 14336]
    s = scales[(15, "down_proj")].to(W.device)  # [14336]
    W_scaled = W * s.unsqueeze(0)  # [4096, 14336]

    # Use same calibration activation as input (decode-time average)
    x_amax = amax[(15, "down_in")]  # [14336]
    # Synthetic input matching observed magnitudes
    torch.manual_seed(0)
    X = (torch.randn(32, 14336, device="cuda", dtype=torch.float32) * x_amax * 0.3)

    # Naive INT4: groupwise (G=128), per (k_group, n) absmax, scale = absmax/7, q = round(W/scale)
    def quant_int4_groupwise(W_in):
        G = 128
        K, N = W_in.shape  # K is "rows" of the to-be-quantized matrix; here for our [out, in]
                           # we treat groups along the input dim
        # Reshape to (out, in_groups, 128)
        n_groups = K // G if K >= G else 1
        # We're quantizing along the larger dim. For W shape [4096, 14336] take the
        # 14336 dim as group-axis since AWQ scaling is per-input-channel.
        return W_in
        # actually skip detailed groupwise sim — just do per-row absmax for clarity
    # Simpler simulation: per-channel absmax with 16 levels (signed nibble = 8 bins on each side)
    def absmax_int4_perrow(W_in):
        # W_in: [rows, cols] — quantize along cols (each row gets one scale)
        amax_row = W_in.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = amax_row / 7.0
        q = (W_in / scale).round().clamp(-8, 7)
        return (q * scale)

    W_naive_q = absmax_int4_perrow(W)
    W_awq_q   = absmax_int4_perrow(W_scaled)

    y_full      = X @ W.T                   # baseline output
    y_naive     = X @ W_naive_q.T
    # AWQ inference: x_scaled = X / s, output = x_scaled @ W_awq_q.T
    X_scaled = X / s.unsqueeze(0)
    y_awq    = X_scaled @ W_awq_q.T

    rel_naive = (y_naive - y_full).norm() / y_full.norm()
    rel_awq   = (y_awq   - y_full).norm() / y_full.norm()
    print(f"  naive INT4 (per-row absmax) rel_err: {rel_naive:.4f}")
    print(f"  AWQ-scaled INT4 rel_err:             {rel_awq:.4f}")
    print(f"  improvement: {rel_naive / rel_awq:.2f}x lower error")

    # ---- Save scales ----
    serializable = {f"L{li}_{name}": s for (li, name), s in scales.items()}
    torch.save(serializable, OUT_PATH)
    print(f"\n[save] {len(serializable)} per-linear scale tensors -> {OUT_PATH}")


if __name__ == "__main__":
    main()
