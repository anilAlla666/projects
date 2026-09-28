#!/usr/bin/env python3
"""Mistral-7B with AWQ-calibrated INT4 substitute. End-to-end quality + perf check.

Loads per-linear AWQ scales from awq_scales.pt. For each nn.Linear in the 32
decoder layers, builds AwqInt4Linear which:
  - In __init__: scales weight columns by s before quantization (so absmax/groupwise
    quantization sees flatter per-channel magnitudes)
  - In forward: divides activation by s before INT4 GEMV

Runs three configurations on the same prompt with greedy decode, teacher-forced
against the fp16 baseline:
  1. fp16 baseline (no CIPHER)
  2. naive INT4 (current shipped path, no AWQ)
  3. AWQ INT4 (new path)

Reports per-config: top-1 token match rate, mean KL divergence on logits, tok/s.
"""
import os, sys, ctypes, time
# Must set env BEFORE loading libcipher_rt.so — flags read in init constructor
os.environ.setdefault("CIPHER_SUBSTITUTE_V2",   "on")
os.environ.setdefault("CIPHER_WEIGHT_COMPRESS", "on")
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_weight_compress_init.restype = ctypes.c_int
rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
rt.cipher_weight_compress_observe.restype = ctypes.c_int
rt.cipher_weight_compress_quantize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
rt.cipher_weight_compress_quantize.restype = ctypes.c_int
rt.cipher_weight_compress_lookup_T.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
rt.cipher_weight_compress_lookup_T.restype = ctypes.c_int
rt.cipher_weight_compress_int4_gemv.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]  # a, b_T, scales, c, M, N, K, stream
rt.cipher_weight_compress_int4_gemv.restype = ctypes.c_int
rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()


class Int4Linear(nn.Module):
    """Naive INT4 substitute (no AWQ scaling). Matches existing run_mistral_int4.py path."""
    def __init__(self, orig_linear: nn.Linear):
        super().__init__()
        self.in_features  = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.bias         = orig_linear.bias
        wt = orig_linear.weight.detach().t().contiguous()
        self.wt_fp16 = nn.Parameter(wt, requires_grad=False)
        for _ in range(1001):
            rt.cipher_weight_compress_observe(self.wt_fp16.data_ptr(), wt.numel() * 2)
        rc = rt.cipher_weight_compress_quantize(
            self.wt_fp16.data_ptr(), self.in_features, self.out_features)
        self._compressed = (rc == 1)
        if self._compressed:
            bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
            br = ctypes.c_int(); bc = ctypes.c_int()
            ok = rt.cipher_weight_compress_lookup_T(self.wt_fp16.data_ptr(),
                ctypes.byref(bT), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))
            if ok != 1:
                self._compressed = False
            else:
                self._bT_ptr = bT.value
                self._bs_ptr = bs.value

    def forward(self, x):
        if self._compressed and x.dim() >= 2 and x.shape[-2] == 1:
            xb = x.reshape(-1, self.in_features).contiguous()
            assert xb.shape[0] == 1
            out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                1, self.out_features, self.in_features, stream)
            if rc != 1:
                out = (x.reshape(-1, self.in_features) @ self.wt_fp16).reshape(
                    x.shape[:-1] + (self.out_features,))
            else:
                out = out.reshape(x.shape[:-1] + (self.out_features,))
        else:
            out = x @ self.wt_fp16
        if self.bias is not None:
            out = out + self.bias
        return out


class AwqInt4Linear(nn.Module):
    """AWQ-scaled INT4 substitute.

    Pre-divides activation by s and quantizes W * s. Mathematically equivalent to
    the naive linear (y = x @ W^T = (x/s) @ (W*s)^T) but the INT4 quantization of
    W*s sees flatter per-channel magnitudes, so the absmax-groupwise quantizer
    wastes fewer bits on outlier channels.
    """
    def __init__(self, orig_linear: nn.Linear, scale_s: torch.Tensor):
        super().__init__()
        self.in_features  = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.bias         = orig_linear.bias
        assert scale_s.shape == (self.in_features,), \
            f"scale must be [{self.in_features}], got {tuple(scale_s.shape)}"
        device = orig_linear.weight.device
        s = scale_s.to(device=device, dtype=torch.float32).clamp(min=1e-6)

        # Build scaled weight: W'[i, j] = W[i, j] * s[j].
        # Then transpose to wt'[j, i] = wt[j, i] * s[j], i.e. wt' = wt * s[:, None].
        with torch.no_grad():
            W = orig_linear.weight.detach().to(torch.float32)  # [out, in]
            W_scaled = W * s.unsqueeze(0)                      # broadcast over out dim
            wt_scaled = W_scaled.t().contiguous().to(torch.float16)  # [in, out]

        self.wt_fp16 = nn.Parameter(wt_scaled, requires_grad=False)
        # Inverse scale used in forward: x' = x * inv_s
        self.register_buffer("inv_s", (1.0 / s).to(torch.float16))

        for _ in range(1001):
            rt.cipher_weight_compress_observe(self.wt_fp16.data_ptr(),
                                              self.wt_fp16.numel() * 2)
        rc = rt.cipher_weight_compress_quantize(
            self.wt_fp16.data_ptr(), self.in_features, self.out_features)
        self._compressed = (rc == 1)
        if self._compressed:
            bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
            br = ctypes.c_int(); bc = ctypes.c_int()
            ok = rt.cipher_weight_compress_lookup_T(self.wt_fp16.data_ptr(),
                ctypes.byref(bT), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))
            if ok != 1:
                self._compressed = False
            else:
                self._bT_ptr = bT.value
                self._bs_ptr = bs.value

    def forward(self, x):
        # Divide activation by s (multiply by inv_s)
        x_scaled = x * self.inv_s
        if self._compressed and x_scaled.dim() >= 2 and x_scaled.shape[-2] == 1:
            xb = x_scaled.reshape(-1, self.in_features).contiguous()
            assert xb.shape[0] == 1
            out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                1, self.out_features, self.in_features, stream)
            if rc != 1:
                out = (x_scaled.reshape(-1, self.in_features) @ self.wt_fp16).reshape(
                    x.shape[:-1] + (self.out_features,))
            else:
                out = out.reshape(x.shape[:-1] + (self.out_features,))
        else:
            out = x_scaled @ self.wt_fp16
        if self.bias is not None:
            out = out + self.bias
        return out


# -------------------- Driver --------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "Explain how a CPU executes instructions step by step"
N_DECODE = 200

LINEAR_TO_POINT = {
    "q_proj": "qkv_in", "k_proj": "qkv_in", "v_proj": "qkv_in",
    "o_proj": "o_in",
    "gate_proj": "gate_in", "up_proj": "gate_in",
    "down_proj": "down_in",
}
ATTN_LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def build_model(mode, tok, scales=None):
    """mode in {'fp16', 'int4', 'awq'}"""
    print(f"[build] {mode}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)
    if mode == "fp16":
        return model
    n_replaced = 0
    for li, layer in enumerate(model.model.layers):
        for name in LINEAR_TO_POINT:
            parent = layer.self_attn if name in ATTN_LINEARS else layer.mlp
            old = getattr(parent, name)
            if old.in_features % 128 != 0 or old.out_features % 8 != 0:
                continue
            if mode == "int4":
                new_mod = Int4Linear(old).to("cuda")
            else:  # awq
                key = f"L{li}_{name}"
                s = scales[key]
                new_mod = AwqInt4Linear(old, s).to("cuda")
            setattr(parent, name, new_mod)
            n_replaced += 1
    print(f"[build] replaced {n_replaced} linears")
    return model


def free_run(model, prompt_ids, n_decode):
    """Greedy decode. Returns (decoded_tokens [n_decode+1], decode_logits [n_decode], elapsed_sec).

    decoded_tokens[0] = T_P from prefill; decoded_tokens[i+1] = argmax of decode step i.
    decode_logits[i]  = logits AFTER decode step i (used for KL comparison).
    """
    decoded = []
    logits_list = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoded.append(nxt.item())
        for s in range(n_decode):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits_list.append(out.logits[0, -1, :].detach().to(torch.float32).cpu())
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded.append(nxt.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return decoded, logits_list, elapsed


def teacher_forced(model, prompt_ids, forced_tokens):
    """Feed prompt + forced_tokens one at a time. Returns list of logits AFTER each forced input."""
    logits_list = []
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        for tok_id in forced_tokens:
            inp = torch.tensor([[tok_id]], dtype=torch.long, device=prompt_ids.device)
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits_list.append(out.logits[0, -1, :].detach().to(torch.float32).cpu())
    return logits_list


def compute_metrics(base_logits, proj_logits):
    n = min(len(base_logits), len(proj_logits))
    base_argmax = [base_logits[i].argmax().item() for i in range(n)]
    proj_argmax = [proj_logits[i].argmax().item() for i in range(n)]
    top1 = sum(int(b == p) for b, p in zip(base_argmax, proj_argmax)) / n
    kls = []
    for i in range(n):
        p_log = F.log_softmax(base_logits[i], dim=-1)
        q_log = F.log_softmax(proj_logits[i], dim=-1)
        p = p_log.exp()
        kls.append((p * (p_log - q_log)).sum().item())
    kls.sort()
    return {
        "top1_match": top1,
        "mean_kl": sum(kls) / len(kls),
        "p99_kl": kls[min(len(kls) - 1, int(0.99 * len(kls)))],
        "n": n,
    }


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    print(f"[setup] prompt={PROMPT!r} prompt_len={prompt_ids.shape[1]} N_DECODE={N_DECODE}")

    # ===== Phase 1: fp16 baseline (free-run) =====
    print("\n========== fp16 baseline ==========")
    m_fp16 = build_model("fp16", tok)
    decoded, base_logits, elapsed = free_run(m_fp16, prompt_ids, N_DECODE)
    base_text = tok.decode(decoded)
    print(f"[fp16] {N_DECODE} tokens in {elapsed:.2f}s = {N_DECODE/elapsed:.2f} tok/s")
    print(f"[fp16] free-running text: {base_text[:300]!r}")
    forced_tokens = decoded[:N_DECODE]   # tokens to feed during teacher-forced runs
    del m_fp16
    torch.cuda.empty_cache()

    results = {"fp16": {"text": base_text, "tok_s": N_DECODE/elapsed}}

    # ===== Phase 2: naive INT4 (teacher-forced + free-run) =====
    print("\n========== naive INT4 (no AWQ) ==========")
    m_int4 = build_model("int4", tok)
    proj_logits = teacher_forced(m_int4, prompt_ids, forced_tokens)
    metrics = compute_metrics(base_logits, proj_logits)
    print(f"[int4-teacher] top1={metrics['top1_match']:.4f} mean_kl={metrics['mean_kl']:.4f} p99_kl={metrics['p99_kl']:.4f}")
    decoded2, _, elapsed2 = free_run(m_int4, prompt_ids, N_DECODE)
    int4_text = tok.decode(decoded2)
    print(f"[int4-free] {N_DECODE} tokens in {elapsed2:.2f}s = {N_DECODE/elapsed2:.2f} tok/s")
    print(f"[int4-free] text: {int4_text[:300]!r}")
    results["int4"] = {"text": int4_text, "tok_s": N_DECODE/elapsed2, **metrics}
    del m_int4
    torch.cuda.empty_cache()

    # ===== Phase 3: AWQ INT4 (teacher-forced + free-run) =====
    print("\n========== AWQ INT4 ==========")
    scales = torch.load("/home/ubuntu/op31-prod-fix/awq_scales.pt", map_location="cpu", weights_only=True)
    print(f"[awq] loaded {len(scales)} per-linear scale tensors")
    m_awq = build_model("awq", tok, scales=scales)
    proj_logits = teacher_forced(m_awq, prompt_ids, forced_tokens)
    metrics = compute_metrics(base_logits, proj_logits)
    print(f"[awq-teacher] top1={metrics['top1_match']:.4f} mean_kl={metrics['mean_kl']:.4f} p99_kl={metrics['p99_kl']:.4f}")
    decoded3, _, elapsed3 = free_run(m_awq, prompt_ids, N_DECODE)
    awq_text = tok.decode(decoded3)
    print(f"[awq-free] {N_DECODE} tokens in {elapsed3:.2f}s = {N_DECODE/elapsed3:.2f} tok/s")
    print(f"[awq-free] text: {awq_text[:300]!r}")
    results["awq"] = {"text": awq_text, "tok_s": N_DECODE/elapsed3, **metrics}
    del m_awq
    torch.cuda.empty_cache()

    # ===== Verdict =====
    print("\n========== SUMMARY ==========")
    print(f"  {'mode':>8}  {'tok/s':>8}  {'top1':>8}  {'mean_kl':>8}  {'p99_kl':>8}")
    print(f"  {'fp16':>8}  {results['fp16']['tok_s']:>8.2f}       --        --        --")
    for name in ("int4", "awq"):
        r = results[name]
        print(f"  {name:>8}  {r['tok_s']:>8.2f}  {r['top1_match']:>8.4f}  {r['mean_kl']:>8.4f}  {r['p99_kl']:>8.4f}")

    # Improvement: AWQ vs naive INT4
    int4_top1 = results["int4"]["top1_match"]
    awq_top1 = results["awq"]["top1_match"]
    int4_kl = results["int4"]["mean_kl"]
    awq_kl = results["awq"]["mean_kl"]
    if int4_kl > 0:
        kl_reduction = (int4_kl - awq_kl) / int4_kl * 100
    else:
        kl_reduction = 0
    print(f"\n  AWQ vs naive INT4: top1 {int4_top1:.4f} -> {awq_top1:.4f} ({(awq_top1-int4_top1)*100:+.1f} pp)")
    print(f"                     mean_kl {int4_kl:.4f} -> {awq_kl:.4f} ({kl_reduction:+.1f}% reduction)")

    import json
    with open("/home/ubuntu/op31-prod-fix/awq_results.json", "w") as f:
        # strip non-serializable text fields to short snippets
        ser = {}
        for k, v in results.items():
            ser[k] = {kk: (vv[:200] if isinstance(vv, str) else vv) for kk, vv in v.items()}
        json.dump(ser, f, indent=2)
    print(f"\n[save] awq_results.json")


if __name__ == "__main__":
    main()
