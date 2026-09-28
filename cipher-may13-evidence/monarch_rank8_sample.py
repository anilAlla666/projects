"""Rank=8 sample test on selected layers — confirm the STOP criterion.

User's gate: if avg_rotated_err >= 15% at rank=8, STOP. We test on 6 representative
layers (0, 5, 12, 18, 25, 31) at rank=8 with rotation. If even the best layer
fails, the negative result is confirmed.
"""
import os, sys, time
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monarch_phase1 import monarch_factorize, monarch_reconstruct
from monarch_phase3b import procrustes_rotation, baseline_err

MODEL = "mistralai/Mistral-7B-v0.1"
SAMPLE_LAYERS = [0, 5, 12, 18, 25, 31]
LINEARS = [
    ("q_proj", "self_attn"), ("k_proj", "self_attn"), ("v_proj", "self_attn"),
    ("o_proj", "self_attn"), ("gate_proj", "mlp"), ("up_proj", "mlp"),
    ("down_proj", "mlp"),
]


def main():
    print(f"[load] {MODEL}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    print(f"\n[rank8] testing rank=8 with rotation on {len(SAMPLE_LAYERS)} sample layers (n_iter=5, p=64)\n", flush=True)
    print(f"  {'L':>3} {'name':<10} {'shape':>14} {'before':>9} {'after':>9} {'side':>7}", flush=True)

    all_after_R4 = []
    all_after_R8 = []
    for li in SAMPLE_LAYERS:
        layer = model.model.layers[li]
        for name, parent_name in LINEARS:
            mod = getattr(getattr(layer, parent_name), name)
            W = mod.weight.detach().to(torch.float32)
            n, m = W.shape
            if n % 64 != 0 or m % 64 != 0:
                continue
            before = baseline_err(W, 64, n // 64, m // 64, 8)
            Q, side, after, _ = procrustes_rotation(W, 64, 8, n_iter=5)
            all_after_R8.append(after)
            shape_s = f"{n}x{m}"
            print(f"  {li:>3} {name:<10} {shape_s:>14} {before:>9.4f} {after:>9.4f} {side:>7}", flush=True)

    n_total = len(all_after_R8)
    avg_R8 = sum(all_after_R8) / n_total
    n_pass_15 = sum(1 for e in all_after_R8 if e < 0.15)
    print(f"\n[rank8 SUMMARY] {n_total} weights tested at rank=8 with rotation:")
    print(f"  avg rel_err: {avg_R8:.4f}")
    print(f"  pass 15% bar: {n_pass_15}/{n_total}")
    print(f"  pass  7% bar: {sum(1 for e in all_after_R8 if e < 0.07)}/{n_total}")
    if avg_R8 >= 0.15:
        print(f"\n  ==> STOP criterion CONFIRMED: avg = {avg_R8:.4f} >= 15% at rank=8 even with rotation.")
    elif avg_R8 < 0.07:
        print(f"\n  ==> avg < 7% — PROCEED with rank=8 in Phase 5")
    else:
        print(f"\n  ==> marginal: 7% < avg < 15% at rank=8")


if __name__ == "__main__":
    main()
