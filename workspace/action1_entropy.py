"""ACTION 1: Von Neumann entropy + effective rank profile of Mistral-7B weights.

Information-theoretic floor for ANY tensor-network compression. No algorithm can
do better than chi_min = 2^S_entropy bond dimension.
"""
import math
import torch
from transformers import AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
LINEARS = [
    ("q_proj", "self_attn"), ("k_proj", "self_attn"), ("v_proj", "self_attn"),
    ("o_proj", "self_attn"),
    ("gate_proj", "mlp"), ("up_proj", "mlp"), ("down_proj", "mlp"),
]


def main():
    print(f"[load] {MODEL}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    print(f"\n=== ACTION 1: Von Neumann entropy of weight matrices ===", flush=True)
    print(f"\n  {'L':>3} {'name':<10} {'shape':>14} {'S_ent':>7} {'S_max':>7} {'S/Smax':>7} {'chi_min':>8} {'k95':>5} {'k99':>5}", flush=True)
    print("  " + "-" * 84, flush=True)

    rows = []
    for li in LAYERS:
        layer = model.model.layers[li]
        for name, parent_name in LINEARS:
            mod = getattr(getattr(layer, parent_name), name)
            W = mod.weight.detach().to(torch.float32)
            n, m = W.shape
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            del U, Vh
            # Probability distribution from squared singular values
            p = S * S
            p = p / p.sum()
            # Von Neumann entropy in bits
            S_ent = -(p * (p + 1e-30).log2()).sum().item()
            S_max = math.log2(min(n, m))
            chi_min = 2.0 ** S_ent
            ratio = S_ent / S_max if S_max > 0 else 0
            cumvar = torch.cumsum(S * S, 0) / (S * S).sum()
            k95_idx = (cumvar >= 0.95).nonzero()
            k99_idx = (cumvar >= 0.99).nonzero()
            k95 = k95_idx[0, 0].item() + 1 if k95_idx.numel() > 0 else len(S)
            k99 = k99_idx[0, 0].item() + 1 if k99_idx.numel() > 0 else len(S)
            shape_s = f"{n}x{m}"
            print(f"  {li:>3} {name:<10} {shape_s:>14} {S_ent:>7.2f} {S_max:>7.2f} {ratio:>7.4f} {chi_min:>8.0f} {k95:>5d} {k99:>5d}", flush=True)
            rows.append({
                "layer": li, "name": name, "shape": [n, m],
                "S_ent": S_ent, "S_max": S_max, "ratio": ratio,
                "chi_min": chi_min, "k95": k95, "k99": k99,
            })
            del S, p, cumvar, W
        torch.cuda.empty_cache()

    print(f"\n=== SUMMARY ===", flush=True)
    n_total = len(rows)
    avg_ratio = sum(r["ratio"] for r in rows) / n_total
    avg_chimin = sum(r["chi_min"] for r in rows) / n_total
    avg_k95 = sum(r["k95"] for r in rows) / n_total
    avg_k99 = sum(r["k99"] for r in rows) / n_total
    print(f"  weights tested: {n_total}", flush=True)
    print(f"  avg S_ent / S_max:           {avg_ratio:.4f}  (0=perfectly compressible, 1=incompressible random)", flush=True)
    print(f"  avg chi_min (effective rank): {avg_chimin:.0f}", flush=True)
    print(f"  avg k_for_95:                 {avg_k95:.0f}", flush=True)
    print(f"  avg k_for_99:                 {avg_k99:.0f}", flush=True)

    print(f"\n  Distribution of chi_min:", flush=True)
    n_lt_64  = sum(1 for r in rows if r["chi_min"] < 64)
    n_lt_128 = sum(1 for r in rows if r["chi_min"] < 128)
    n_lt_256 = sum(1 for r in rows if r["chi_min"] < 256)
    n_lt_512 = sum(1 for r in rows if r["chi_min"] < 512)
    print(f"    chi_min < 64:   {n_lt_64}/{n_total}", flush=True)
    print(f"    chi_min < 128:  {n_lt_128}/{n_total}", flush=True)
    print(f"    chi_min < 256:  {n_lt_256}/{n_total}", flush=True)
    print(f"    chi_min < 512:  {n_lt_512}/{n_total}", flush=True)

    print(f"\n  Distribution of k_for_95:", flush=True)
    for thr in [64, 128, 256, 512, 1024]:
        c = sum(1 for r in rows if r["k95"] <= thr)
        print(f"    k95 <= {thr:>4}:  {c}/{n_total}", flush=True)

    import json
    with open("/home/ubuntu/op31-prod-fix/action1_entropy_results.json", "w") as f:
        json.dump({"layers_tested": LAYERS, "rows": rows,
                   "avg_ratio": avg_ratio, "avg_chimin": avg_chimin,
                   "avg_k95": avg_k95, "avg_k99": avg_k99}, f, indent=2)


if __name__ == "__main__":
    main()
