"""A1 CRITICAL TESTS:
  - Q.T @ Q vs identity (column orthonormality — should be ~0)
  - Q @ Q.T vs identity (subspace projector — only ~I if k = in_features)
  - For ONE Mistral weight: ||W - W @ Q @ Q.T|| / ||W|| (how much W lies in col(Q))
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import projection_quality as pq


def main():
    print("=== A1 critical tests ===\n", flush=True)
    model, tok = pq.load_model()
    captures = pq.calibrate(model, tok)
    qs = pq.compute_qs(captures, k_max=pq.K_MAX)
    del captures

    # Pick representative Q: layer 15, qkv_in (post input_layernorm input to q/k/v_proj)
    rep_key = (15, "qkv_in")
    Q_full = qs[rep_key]["Q"]   # [in_features, K_MAX] fp32 cpu
    print(f"\nRepresentative Q: layer 15, qkv_in. Shape = {tuple(Q_full.shape)}", flush=True)

    in_features = Q_full.shape[0]
    K_MAX = Q_full.shape[1]
    print(f"  in_features = {in_features},  K_MAX = {K_MAX}", flush=True)

    Q_full_g = Q_full.to("cuda")

    # Test 1: Q.T @ Q == I_k  (column orthonormality)
    QtQ = Q_full_g.T @ Q_full_g
    I_k = torch.eye(K_MAX, device="cuda")
    err_QtQ = (QtQ - I_k).norm().item() / (K_MAX ** 0.5)
    print(f"\n[orthonormality] ||Q.T @ Q - I_k||_F / sqrt(k) = {err_QtQ:.2e}   (should be ~1e-7 fp32)", flush=True)

    # Test 2: Q @ Q.T  — this is the PROJECTOR onto col(Q), shape [in, in]
    QQt = Q_full_g @ Q_full_g.T
    I_n = torch.eye(in_features, device="cuda")
    err_QQt_full = (QQt - I_n).norm().item()
    # The expected magnitude: rank-K_MAX projector vs n-dim identity has Frobenius diff = sqrt(n - K_MAX)
    expected = (in_features - K_MAX) ** 0.5
    print(f"[projector]      ||Q @ Q.T - I_n||_F = {err_QQt_full:.2f}   (expected sqrt(n - k) = {expected:.2f})", flush=True)
    print(f"  (this CANNOT be ~0 unless k == in_features. With 500 calibration samples, k_max=499 < n=4096.", flush=True)
    print(f"   Q @ Q.T is a rank-{K_MAX} projector onto a {K_MAX}-dim subspace of {in_features}-dim ambient.)", flush=True)

    # Test 3: For one Mistral weight whose INPUT is layer 15 qkv_in (q_proj, k_proj, v_proj),
    # measure ||W - W @ Q @ Q.T|| / ||W||.
    layer = model.model.layers[15]
    W = layer.self_attn.q_proj.weight.detach().to(torch.float32)   # [out=4096, in=4096]
    print(f"\n[W projection] q_proj layer 15, W shape = {tuple(W.shape)}", flush=True)

    # W @ Q @ Q.T is the projection of W's ROWS onto col(Q)
    Q_g = Q_full_g.float()
    W_g = W.to("cuda")
    W_proj_g = (W_g @ Q_g) @ Q_g.T
    rel_err = ((W_g - W_proj_g).norm() / W_g.norm()).item()
    print(f"  ||W - W @ Q @ Q.T||_F / ||W||_F = {rel_err:.6f}", flush=True)
    print(f"  This is the fraction of W that does NOT lie in col(Q).", flush=True)
    print(f"  At k=499 / n=4096: theoretical lower bound depends on W's row distribution in calibration subspace.", flush=True)
    print(f"  If W was learned to fit activation patterns ⊆ col(Q), this could be small. Empirically:", flush=True)

    # Test 4: For the same q_proj, measure ||W x - W Q Q.T x|| / ||W x|| on actual decode-time activations
    # (not synthetic; capture from a small decode pass)
    print(f"\n[decode-time test] capture qkv_in at layer 15 over 32 decode tokens, measure W x vs W Q Q.T x", flush=True)
    cap = []
    capture_enabled = [False]
    def hook(module, args):
        if capture_enabled[0] and len(args) > 0 and isinstance(args[0], torch.Tensor):
            x = args[0]
            if x.dim() == 3 and x.shape[1] == 1:
                cap.append(x[0, 0, :].detach().to(torch.float32).cpu())
    h = layer.self_attn.q_proj.register_forward_pre_hook(hook)
    ids = tok("Explain how a CPU executes instructions step by step",
              return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        capture_enabled[0] = True
        for _ in range(32):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    h.remove()
    print(f"  captured {len(cap)} decode-time activations", flush=True)

    Wcuda = W.to("cuda")
    rel_errs = []
    rel_errs_proj = []
    for i, x in enumerate(cap):
        xg = x.to("cuda")
        y_full = Wcuda @ xg
        # Q-projection in fp32
        z = Q_g.T @ xg               # [k]
        x_proj = Q_g @ z              # [in], the projection of x onto col(Q)
        y_proj = Wcuda @ x_proj
        # Direct Wx vs Wproj_x
        rel_errs.append((y_full - y_proj).norm().item() / y_full.norm().item())
        # Also: how much of x lies in col(Q)?
        rel_errs_proj.append((xg - x_proj).norm().item() / xg.norm().item())

    print(f"  decode activation: mean ||x - Q Q.T x|| / ||x|| = {sum(rel_errs_proj)/len(rel_errs_proj):.6f}", flush=True)
    print(f"  decode output:     mean ||W x - W Q Q.T x|| / ||W x|| = {sum(rel_errs)/len(rel_errs):.6f}", flush=True)


if __name__ == "__main__":
    main()
