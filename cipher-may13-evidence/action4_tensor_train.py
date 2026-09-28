"""ACTION 4: Tensor Train (MPO) decomposition quality on Mistral weights.

Reshape W into 8D tensor and TT-decompose at ranks 2, 4, 8, 16, 32, 64.
Compare to Monarch at the same compression ratio.
"""
import os, json
import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import tensor_train
tl.set_backend("numpy")

from transformers import AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
LAYERS = [0, 8, 16, 24]
LINEARS = [
    ("q_proj", "self_attn"), ("k_proj", "self_attn"),
    ("v_proj", "self_attn"), ("o_proj", "self_attn"),
    ("gate_proj", "mlp"), ("up_proj", "mlp"), ("down_proj", "mlp"),
]


def factor_to_8(n):
    """Find an 8-tuple of factors whose product is n. Prefer balanced."""
    # Try 8 base factors near n^(1/8)
    target = n ** (1/8)
    # Simple: factor 4096 = 8^4, 14336 = 8^4 * 28/8... 14336 / 4096 = 3.5
    # 14336 = 2^11 * 7 = 8 * 8 * 8 * 28
    # 1024 = 2^10 = 4 * 4 * 4 * 4 * 4. As 8d: 2*4*4*4*4*1*1*1 — but TT wants nontrivial dims
    # We'll prefer 4D for 1024, 4D for 4096 each side, total 8D for square 4096x4096
    # And use the natural reshape per shape
    if n == 4096:
        return (8, 8, 8, 8)
    if n == 1024:
        return (8, 8, 4, 4)
    if n == 14336:
        return (8, 8, 8, 28)
    raise ValueError(f"unknown n={n}")


def tt_test(W: np.ndarray, ranks_list, label):
    """Decompose W via TT at multiple ranks; report (rank, err, params, compression)."""
    n, m = W.shape
    # Reshape into 8D: (n_factors..., m_factors...)
    nf = factor_to_8(n)
    mf = factor_to_8(m)
    full_dims = nf + mf
    # Verify product
    assert np.prod(full_dims) == n * m, f"{full_dims} product != {n}*{m}"
    Wt = W.reshape(full_dims)
    rows = []
    for R in ranks_list:
        # tensor_train wants ranks of length d+1 = 9 with [1, R, R, R, R, R, R, R, 1]
        rank_seq = [1] + [R] * (len(full_dims) - 1) + [1]
        try:
            cores = tensor_train(Wt, rank=rank_seq)
            W_recon = tl.tt_to_tensor(cores).reshape(n, m)
            err = float(np.linalg.norm(W - W_recon) / np.linalg.norm(W))
            n_params = int(sum(c.size for c in cores))
            compression = (n * m) / max(n_params, 1)
            rows.append({"rank": R, "err": err, "n_params": n_params, "compression": compression})
        except Exception as e:
            rows.append({"rank": R, "err": None, "n_params": 0, "compression": 0, "error": str(e)[:80]})
    return rows


def main():
    print(f"[load] {MODEL}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    print(f"\n=== ACTION 4: Tensor Train decomposition ===", flush=True)
    print(f"  ranks tested: 2, 4, 8, 16, 32, 64\n", flush=True)
    print(f"  {'L':>3} {'name':<10} {'shape':>14}  {'rank':>5}  {'rel_err':>9}  {'n_params':>10}  {'compression':>11}", flush=True)
    print("  " + "-" * 75, flush=True)

    all_rows = []
    for li in LAYERS:
        layer = model.model.layers[li]
        for name, parent in LINEARS:
            mod = getattr(getattr(layer, parent), name)
            W_fp32 = mod.weight.detach().to(torch.float32).cpu().numpy()
            n, m = W_fp32.shape
            shape_s = f"{n}x{m}"
            tt_rows = tt_test(W_fp32, [2, 4, 8, 16, 32, 64], f"{li}_{name}")
            for r in tt_rows:
                if r.get("err") is None:
                    print(f"  {li:>3} {name:<10} {shape_s:>14}  {r['rank']:>5}  FAILED: {r.get('error','?')[:50]}", flush=True)
                else:
                    print(f"  {li:>3} {name:<10} {shape_s:>14}  {r['rank']:>5}  {r['err']:>9.4f}  {r['n_params']:>10d}  {r['compression']:>10.2f}x", flush=True)
                all_rows.append({"layer": li, "name": name, "shape": [n, m], **r})

    # Aggregate: for each rank, mean err
    print(f"\n=== AGGREGATE ===", flush=True)
    print(f"  {'rank':>5}  {'mean_err':>9}  {'mean_compression':>17}  {'<7% (count/total)':>17}", flush=True)
    for R in [2, 4, 8, 16, 32, 64]:
        sub = [r for r in all_rows if r.get("rank") == R and r.get("err") is not None]
        if sub:
            me = sum(r["err"] for r in sub) / len(sub)
            mc = sum(r["compression"] for r in sub) / len(sub)
            below_7 = sum(1 for r in sub if r["err"] < 0.07)
            print(f"  {R:>5}  {me:>9.4f}  {mc:>16.2f}x  {below_7}/{len(sub)}", flush=True)

    # Compare to Monarch (Phase 2 numbers)
    print(f"\n=== Compare TT vs Monarch (Phase 2 raw, no rotation) ===", flush=True)
    print(f"  Monarch results from monarch_phase2_log.txt: avg rel_err 0.999 at R=1, 0.995 at R=4, 0.980 at R=16", flush=True)
    print(f"  TT: see table above", flush=True)
    print(f"  Verdict: at any compression ratio, which decomposition wins?", flush=True)

    with open("/home/ubuntu/op31-prod-fix/action4_tt_results.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n[save] action4_tt_results.json", flush=True)


if __name__ == "__main__":
    main()
