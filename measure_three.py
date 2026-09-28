#!/usr/bin/env python3
"""
Three measurement passes on Mistral-7B-v0.1 (decode), one model load.

M1: SiLU/MLP intermediate sparsity
    - Per layer: fraction of gate_proj outputs < -3.0  (silu(<-3) < 0.05)
    - Per layer: fraction of silu(gate)*up outputs with |x| < 0.01
    - Distribution: mean across 200 decode tokens

M2: Cross-token output stability per decoder layer
    - Per layer: cos_sim(out_t, out_{t-1}) averaged over 199 pairs
    - Per layer: ||out_t - out_{t-1}|| / ||out_t||  averaged

M3: Per-head attention concentration / top-64 stability
    - Per layer / head: entropy of attention distribution (avg over 100 tokens)
    - Per layer / head: fraction of mass on top-64 positions (3% of 2K)
    - Per layer / head: avg Jaccard of top-64 sets across consecutive tokens

Hook strategy
-------------
PyTorch forward hooks (not cuBLAS intercept):
  - mlp.gate_proj          forward hook  -> gate (pre-SiLU)
  - mlp.down_proj          forward pre-hook -> silu(gate) * up   (= input)
  - layers[i]              forward hook  -> residual-stream output of the layer
  - output_attentions=True (eager attention) -> per-layer per-head softmax weights

Captures the same tensors a cuBLAS shim would observe; runs in-process, no LD_PRELOAD.

Context: 2048-token prompt prefill, then 200 greedy decode tokens.
M1+M2 over all 200, M3 over first 100.
"""

import json, math, time, gc, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID    = "mistralai/Mistral-7B-v0.1"
PROMPT_LEN  = 2048
DECODE_M12  = 200
DECODE_M3   = 100
TOP_K       = 64

torch.manual_seed(0)


def build_prompt(tok, target_len):
    # Varied natural-prose chunks. Avoids the repetition loop a single repeated
    # sentence would produce under greedy decode.
    chunks = [
        "The Atlantic is the second-largest of the world's oceans. It covers "
        "an area of about 41 million square miles and separates the so-called "
        "Old World from the New World. Trade winds blow from east to west "
        "across the tropics, and the Gulf Stream carries warm water north. ",
        "In computer architecture, the memory hierarchy organizes storage "
        "from the smallest, fastest registers near the processor down to "
        "large, slow disks. Each level trades capacity against latency, and "
        "good algorithms exploit locality of reference to keep working sets "
        "in caches close to the processing units. ",
        "Photosynthesis converts light energy into chemical energy in plants, "
        "algae and certain bacteria. The reactions take place in chloroplasts, "
        "where chlorophyll absorbs photons. Carbon dioxide and water become "
        "glucose and oxygen, fueling almost every food chain on Earth. ",
        "The Roman Republic transitioned to an empire after a long period of "
        "civil war. Julius Caesar crossed the Rubicon in 49 BC and his "
        "adopted heir Augustus eventually consolidated power, opening the "
        "Pax Romana, a two-century span of relative stability. ",
        "Quantum mechanics describes nature at the smallest scales. Particles "
        "exhibit wave-like interference; their position and momentum cannot "
        "both be known exactly, and measurement collapses superpositions. "
        "These ideas underpin lasers, transistors and atomic clocks. ",
        "The composer Johann Sebastian Bach worked mostly in Leipzig, where "
        "he served as cantor of St Thomas Church. He wrote the Mass in B "
        "minor, the Brandenburg Concertos and many cantatas, and was largely "
        "forgotten for decades after his death until Mendelssohn revived him. ",
        "Climate models simulate the atmosphere, oceans, ice sheets and land "
        "surface. They are validated against historical observations and used "
        "to project future temperatures under different emission scenarios. "
        "Uncertainty in clouds and aerosols remains a major research focus. ",
        "Beneath the icy crust of Jupiter's moon Europa, scientists believe "
        "a global ocean of liquid water sloshes against a rocky seafloor. "
        "Tidal heating from Jupiter keeps it from freezing solid, and the "
        "ocean is one of the most promising places to look for life. ",
    ]
    text = " ".join(chunks)
    while True:
        ids = tok(text, return_tensors="pt").input_ids[0]
        if ids.numel() >= target_len:
            break
        text = text + " " + " ".join(chunks)
    return ids[:target_len].unsqueeze(0)


def main():
    print(f"Loading {MODEL_ID} (eager attention, fp16)...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.train(False)   # inference mode (PyTorch's "eval" setter)
    cfg = model.config
    n_layers   = cfg.num_hidden_layers
    n_q_heads  = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim   = cfg.hidden_size // n_q_heads
    inter_dim  = cfg.intermediate_size
    hidden     = cfg.hidden_size
    print(f"layers={n_layers} q_heads={n_q_heads} kv_heads={n_kv_heads} "
          f"head_dim={head_dim} hidden={hidden} inter={inter_dim}")

    layers = model.model.layers
    assert len(layers) == n_layers

    # ---------------- Hook state ----------------
    gate_lt_neg3   = np.zeros(n_layers, dtype=np.float64)
    gate_total     = np.zeros(n_layers, dtype=np.int64)
    intr_zero_cnt  = np.zeros(n_layers, dtype=np.float64)
    intr_total     = np.zeros(n_layers, dtype=np.int64)
    intr_cdf_bins  = np.zeros((n_layers, 8), dtype=np.float64)
    bin_edges      = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, float("inf")]

    prev_layer_out = [None] * n_layers
    cos_sum        = np.zeros(n_layers, dtype=np.float64)
    rel_l2_sum     = np.zeros(n_layers, dtype=np.float64)
    cos_pairs      = np.zeros(n_layers, dtype=np.int64)

    entropy_sum     = np.zeros((n_layers, n_q_heads), dtype=np.float64)
    top64_mass_sum  = np.zeros((n_layers, n_q_heads), dtype=np.float64)
    prev_top64_set  = [None] * n_layers
    jaccard_sum     = np.zeros((n_layers, n_q_heads), dtype=np.float64)
    jaccard_pairs   = np.zeros((n_layers, n_q_heads), dtype=np.int64)
    m3_token_count  = 0

    state = {"step": 0, "in_decode": False}

    def make_gate_hook(layer_idx):
        def hook(_mod, _inp, out):
            if not state["in_decode"]:
                return
            xf = out.detach().float().reshape(-1)
            gate_lt_neg3[layer_idx] += int((xf < -3.0).sum().item())
            gate_total[layer_idx]   += int(xf.numel())
        return hook

    def make_down_pre_hook(layer_idx):
        def pre_hook(_mod, inputs):
            if not state["in_decode"]:
                return None
            xa = inputs[0].detach().float().abs().reshape(-1)
            intr_zero_cnt[layer_idx] += int((xa < 0.01).sum().item())
            intr_total[layer_idx]    += int(xa.numel())
            for i in range(8):
                lo, hi = bin_edges[i], bin_edges[i+1]
                if math.isinf(hi):
                    intr_cdf_bins[layer_idx, i] += int((xa >= lo).sum().item())
                else:
                    intr_cdf_bins[layer_idx, i] += int(((xa >= lo) & (xa < hi)).sum().item())
            return None
        return pre_hook

    def make_layer_hook(layer_idx):
        def hook(_mod, _inp, out):
            if not state["in_decode"]:
                return
            h = out[0] if isinstance(out, (tuple, list)) else out
            cur = h.detach().float().reshape(-1)
            prev = prev_layer_out[layer_idx]
            if prev is not None:
                cs = F.cosine_similarity(cur, prev, dim=0).item()
                rel = (cur - prev).norm().item() / (cur.norm().item() + 1e-12)
                cos_sum[layer_idx]    += cs
                rel_l2_sum[layer_idx] += rel
                cos_pairs[layer_idx]  += 1
            prev_layer_out[layer_idx] = cur
        return hook

    handles = []
    for i, L in enumerate(layers):
        handles.append(L.mlp.gate_proj.register_forward_hook(make_gate_hook(i)))
        handles.append(L.mlp.down_proj.register_forward_pre_hook(make_down_pre_hook(i)))
        handles.append(L.register_forward_hook(make_layer_hook(i)))

    # ---------------- Prefill ----------------
    input_ids = build_prompt(tok, PROMPT_LEN).cuda()
    assert input_ids.shape[1] == PROMPT_LEN
    print(f"Prefill: {input_ids.shape[1]} tokens")

    state["in_decode"] = False
    t0 = time.time()
    with torch.inference_mode():
        out = model(input_ids, use_cache=True, output_attentions=False)
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    print(f"Prefill done in {time.time()-t0:.2f}s; "
          f"first decoded token id={next_id.item()}")

    # ---------------- Decode ----------------
    state["in_decode"] = True
    decoded_ids = []
    t0 = time.time()
    for step in range(DECODE_M12):
        state["step"] = step
        do_attn = step < DECODE_M3
        with torch.inference_mode():
            out = model(
                next_id,
                past_key_values=past,
                use_cache=True,
                output_attentions=do_attn,
            )
        past = out.past_key_values
        if do_attn:
            atts = out.attentions
            assert len(atts) == n_layers
            kv_len = atts[0].shape[-1]
            top_k = min(TOP_K, kv_len)
            for li, a in enumerate(atts):
                p = a.float().reshape(n_q_heads, kv_len)
                logp = torch.log(p.clamp_min(1e-12))
                ent  = -(p * logp).sum(dim=1)
                entropy_sum[li] += ent.cpu().numpy()
                vals, idx = torch.topk(p, top_k, dim=1)
                top_mass = vals.sum(dim=1).cpu().numpy()
                top64_mass_sum[li] += top_mass
                idx_np = idx.cpu().numpy()
                if prev_top64_set[li] is not None:
                    prev = prev_top64_set[li]
                    for hi in range(n_q_heads):
                        a_set = set(int(x) for x in idx_np[hi])
                        b_set = set(int(x) for x in prev[hi])
                        union = len(a_set | b_set)
                        inter = len(a_set & b_set)
                        if union:
                            jaccard_sum[li, hi]   += inter / union
                            jaccard_pairs[li, hi] += 1
                prev_top64_set[li] = idx_np
            m3_token_count += 1
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        decoded_ids.append(next_id.item())
    torch.cuda.synchronize()
    state["in_decode"] = False
    dt = time.time() - t0
    print(f"Decode {DECODE_M12} tokens in {dt:.2f}s = {DECODE_M12/dt:.1f} tok/s "
          f"(eager attn for M3 capture)")

    snippet = tok.decode(decoded_ids[:50])
    print(f"first 50 decoded tokens: {snippet!r}")

    for h in handles: h.remove()

    # ===== REPORTING =====
    print("\n" + "="*78)
    print("MEASUREMENT 1 -- SiLU activation sparsity")
    print("="*78)
    print(f"{'lay':>3} | {'gate<-3 frac':>12} | {'|silu*up|<0.01':>14} | "
          f"{'frac |x|<1e-3':>14} | {'frac |x|<1e-2':>14}")
    print("-"*78)
    for i in range(n_layers):
        f1 = gate_lt_neg3[i] / max(gate_total[i], 1)
        f2 = intr_zero_cnt[i] / max(intr_total[i], 1)
        hist = intr_cdf_bins[i] / max(intr_total[i], 1)
        cdf_lt_1em3 = hist[0] + hist[1]
        cdf_lt_1em2 = cdf_lt_1em3 + hist[2]
        print(f"{i:>3} | {f1:>12.4f} | {f2:>14.4f} | {cdf_lt_1em3:>14.4f} | {cdf_lt_1em2:>14.4f}")

    skip_qual_30 = int((intr_zero_cnt / np.maximum(intr_total, 1) > 0.30).sum())
    skip_qual_50 = int((intr_zero_cnt / np.maximum(intr_total, 1) > 0.50).sum())
    avg_intr_zero = float((intr_zero_cnt / np.maximum(intr_total, 1)).mean())
    print()
    print(f"Layers with >30% of intermediate near-zero (|x|<0.01): {skip_qual_30}/32")
    print(f"Layers with >50% of intermediate near-zero (|x|<0.01): {skip_qual_50}/32")
    print(f"Mean across layers: {avg_intr_zero:.3f}")
    per_layer_w = (4096*14336 + 4096*14336 + 14336*4096) * 2
    saved_bytes = avg_intr_zero * per_layer_w * n_layers
    print(f"Estimated HBM saved if column-sparsity is exploitable: "
          f"{saved_bytes/1e9:.2f} GB / token")

    print("\n" + "="*78)
    print("MEASUREMENT 2 -- Per-layer cross-token output stability")
    print("="*78)
    print(f"{'lay':>3} | {'avg cos_sim':>11} | {'avg rel_L2 ||d||/||x||':>22} | {'pairs':>5}")
    print("-"*78)
    cos_avg = cos_sum / np.maximum(cos_pairs, 1)
    rel_avg = rel_l2_sum / np.maximum(cos_pairs, 1)
    for i in range(n_layers):
        print(f"{i:>3} | {cos_avg[i]:>11.6f} | {rel_avg[i]:>22.6f} | {cos_pairs[i]:>5}")

    cache_99   = int((cos_avg > 0.99).sum())
    cache_995  = int((cos_avg > 0.995).sum())
    cache_999  = int((cos_avg > 0.999).sum())
    print()
    print(f"Layers with avg cos_sim > 0.990: {cache_99}/32  (output-cache candidates)")
    print(f"Layers with avg cos_sim > 0.995: {cache_995}/32")
    print(f"Layers with avg cos_sim > 0.999: {cache_999}/32  (near-static layer)")

    print("\n" + "="*78)
    print(f"MEASUREMENT 3 -- Per-head attention concentration "
          f"({m3_token_count} tokens, kv_len ~ {PROMPT_LEN}+t)")
    print("="*78)
    avg_ent      = entropy_sum / max(m3_token_count, 1)
    avg_top64_m  = top64_mass_sum / max(m3_token_count, 1)
    avg_jaccard  = jaccard_sum / np.maximum(jaccard_pairs, 1)
    print(f"{'lay':>3} | {'avg ent':>9} | {'mean top64':>10} | "
          f"{'min top64':>10} | {'max top64':>10} | {'avg jacc':>9}")
    print("-"*78)
    for i in range(n_layers):
        print(f"{i:>3} | {avg_ent[i].mean():>9.3f} | "
              f"{avg_top64_m[i].mean():>10.4f} | "
              f"{avg_top64_m[i].min():>10.4f} | "
              f"{avg_top64_m[i].max():>10.4f} | "
              f"{avg_jaccard[i].mean():>9.4f}")

    head_top64_gt_90 = int((avg_top64_m > 0.90).sum())
    head_top64_gt_95 = int((avg_top64_m > 0.95).sum())
    head_jacc_gt_8   = int((avg_jaccard > 0.80).sum())
    head_jacc_gt_9   = int((avg_jaccard > 0.90).sum())
    total_heads = n_layers * n_q_heads
    print()
    print(f"Heads where top-64 holds >90% of mass: {head_top64_gt_90}/{total_heads}")
    print(f"Heads where top-64 holds >95% of mass: {head_top64_gt_95}/{total_heads}")
    print(f"Heads top-64 jaccard > 0.80 across consecutive tokens: "
          f"{head_jacc_gt_8}/{total_heads}")
    print(f"Heads top-64 jaccard > 0.90 across consecutive tokens: "
          f"{head_jacc_gt_9}/{total_heads}")
    kv_len_mid = PROMPT_LEN + (m3_token_count // 2)
    bytes_full_kv = n_layers * n_kv_heads * kv_len_mid * head_dim * 2 * 2
    bytes_top64   = n_layers * n_kv_heads * TOP_K * head_dim * 2 * 2
    print(f"Full KV per token (kv_len~{kv_len_mid}): {bytes_full_kv/1e6:.1f} MB read")
    print(f"Top-64 KV per token: {bytes_top64/1e6:.2f} MB read")
    print(f"Potential KV-read reduction: {bytes_full_kv/bytes_top64:.1f}x")

    out_dict = {
        "model": MODEL_ID,
        "prompt_len": PROMPT_LEN,
        "decode_m12": DECODE_M12,
        "decode_m3": m3_token_count,
        "m1": {
            "gate_lt_neg3_frac": (gate_lt_neg3 / np.maximum(gate_total,1)).tolist(),
            "intr_abs_lt_0_01_frac": (intr_zero_cnt / np.maximum(intr_total,1)).tolist(),
        },
        "m2": {
            "cos_sim_avg": cos_avg.tolist(),
            "rel_l2_avg":  rel_avg.tolist(),
        },
        "m3": {
            "avg_entropy_per_head":     avg_ent.tolist(),
            "avg_top64_mass_per_head":  avg_top64_m.tolist(),
            "avg_jaccard_per_head":     avg_jaccard.tolist(),
        },
    }
    out_path = "/home/ubuntu/op31-prod-fix/measure_three_results.json"
    with open(out_path, "w") as f:
        json.dump(out_dict, f)
    print(f"\nresults saved: {out_path}")


if __name__ == "__main__":
    main()
