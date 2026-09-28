"""MFU computation from cipher-exporter /metrics.

MFU approximation:
  MFU_tenant = (sm_util_pct / 100) * (sustained_clock_mhz / peak_clock_mhz) * 100

This is a coarse measure — it doesn't separate tensor-core ops from generic
SM cycles. For comparing CIPHER's per-tenant lift (relative delta), the
approximation is sufficient. For absolute B200-equivalent claims it would
need a tensor-op-ratio refinement (Phase 4.8 work).

For H100 SXM5: peak boost clock = 1980 MHz.
"""
import json
import re
import urllib.request
import time


H100_PEAK_CLOCK_MHZ = 1980
EXPORTER_DEFAULT = "http://localhost:9402"


def fetch_metrics(exporter=EXPORTER_DEFAULT, timeout=2.0):
    try:
        with urllib.request.urlopen(f"{exporter}/metrics", timeout=timeout) as r:
            return r.read().decode("utf-8")
    except Exception as e:
        return None


def parse_per_tenant(metrics_text):
    """Return list of dicts: [{tenant, pid, sm_util_pct, mem_util_pct, launches_total}, ...]"""
    by_key = {}
    if not metrics_text:
        return []

    re_sm  = re.compile(r'cipher_tenant_sm_util_pct\{tenant="([^"]+)",pid="([^"]+)",[^}]*\}\s+(\S+)')
    re_mem = re.compile(r'cipher_tenant_mem_util_pct\{tenant="([^"]+)",pid="([^"]+)",[^}]*\}\s+(\S+)')
    re_la  = re.compile(r'cipher_tenant_launches_total\{tenant="([^"]+)",pid="([^"]+)",[^}]*\}\s+(\S+)')

    for m in re_sm.finditer(metrics_text):
        key = (m.group(1), m.group(2))
        by_key.setdefault(key, {"tenant": m.group(1), "pid": m.group(2)})
        by_key[key]["sm_util_pct"] = float(m.group(3))
    for m in re_mem.finditer(metrics_text):
        key = (m.group(1), m.group(2))
        by_key.setdefault(key, {"tenant": m.group(1), "pid": m.group(2)})
        by_key[key]["mem_util_pct"] = float(m.group(3))
    for m in re_la.finditer(metrics_text):
        key = (m.group(1), m.group(2))
        by_key.setdefault(key, {"tenant": m.group(1), "pid": m.group(2)})
        by_key[key]["launches_total"] = float(m.group(3))

    return list(by_key.values())


def parse_device(metrics_text):
    if not metrics_text:
        return {}
    out = {}
    for key, name in [
        ("power_watts",       r"cipher_gpu_power_watts"),
        ("temp_celsius",      r"cipher_gpu_temp_celsius"),
        ("sm_clock_mhz",      r"cipher_gpu_sm_clock_mhz"),
        ("mem_clock_mhz",     r"cipher_gpu_mem_clock_mhz"),
        ("sm_util_pct",       r"cipher_gpu_sm_util_pct"),
        ("mem_util_pct",      r"cipher_gpu_mem_util_pct"),
        ("fb_used_mb",        r"cipher_gpu_fb_used_mb"),
        ("state_age_seconds", r"cipher_gpu_state_age_seconds"),
    ]:
        m = re.search(rf'{name}\{{[^}}]*\}}\s+(\S+)', metrics_text)
        if m:
            out[key] = float(m.group(1))
    return out


def compute_mfu(sm_util_pct, sustained_clock_mhz, peak_clock_mhz=H100_PEAK_CLOCK_MHZ):
    if peak_clock_mhz <= 0:
        return 0.0
    return (sm_util_pct / 100.0) * (sustained_clock_mhz / peak_clock_mhz) * 100.0


def sample(exporter=EXPORTER_DEFAULT):
    """Returns dict {tenants: [...], device: {...}, timestamp: t}."""
    text = fetch_metrics(exporter)
    return {
        "timestamp": time.time(),
        "tenants":   parse_per_tenant(text),
        "device":    parse_device(text),
    }


# ===========================================================================
# CP 4.8 — analytical FLOP-roofline tensor_mfu_pct  (B5 fold-in, D1 Option A)
# ===========================================================================
# The legacy compute_mfu() above is the SM-cycle proxy: it counts an SM
# stalled on HBM as "utilised", so it reads ~100% for memory-bound decode
# where the tensor cores are ~99% idle. CP 4.8 gates on tensor_mfu_pct:
#
#     tensor_mfu_pct = workload_tensor_FLOPs / (wallclock_s * PEAK_TENSOR_FLOPS)
#
# Computed analytically (zero runtime overhead, soak-safe — no DCGM, no
# perturbing CUPTI profiling) and cross-checked against the cipher_flops.c
# PMU counter (CP 3.3). Two physics corrections vs the original D1 spec:
#   * inference forward = 2N FLOP/param, not the Lambda training 6N;
#   * Hopper has NO INT4 tensor path — Marlin dequantises to fp16, peak is
#     989.4 TFLOPS fp16 (the INT4 win is bandwidth/AI, not compute rate).

H100_PEAK_FP16_FLOPS = 989.4e12      # dense fp16/bf16 tensor, H100 SXM5
H100_HBM_BW          = 3.35e12       # HBM3 bytes/s
AI_RIDGE_FP16        = H100_PEAK_FP16_FLOPS / H100_HBM_BW   # ~295 FLOP/byte

# Transformer geometry of the models the WL drivers actually run (B5
# substitutes: TinyLlama for Llama-3.2-1B, Mistral-7B for Llama-3.1-8B).
# embed_params = vocab x hidden, the INPUT embedding lookup table: it does
# 0 matmul FLOPs (it is a gather), so the 2N/6N coefficient must exclude it.
# lm_head stays in `params` — it IS a real matmul. (Cross-check finding,
# CP 4.8 §3 spot-check: WL01 analytical was high by exactly 2*embed_params.)
MODEL_GEOMETRY = {
    "tinyllama-1.1b": dict(params=1.10e9, embed_params=32000 * 2048,
                           layers=22, hidden=2048,
                           heads=32, kv_heads=4, head_dim=64),
    "mistral-7b":     dict(params=7.24e9, embed_params=32000 * 4096,
                           layers=32, hidden=4096,
                           heads=32, kv_heads=8, head_dim=128),
    "minilm-l6":      dict(params=22.7e6, embed_params=30522 * 384,
                           layers=6,  hidden=384,
                           heads=12, kv_heads=12, head_dim=32),
}

# Per-WL FLOP accounting. coeff: 2 = inference forward, 6 = training
# (fwd+bwd+grad). basis: "exact" = derived from transformer geometry;
# "estimate" = per-architecture documented estimate (vision/diffusion/
# speech) — flagged so the report states cross-check coverage honestly.
WL_FLOP = {
    # LLM inference — exact 2N geometry
    "WL01": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL02": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL03": dict(model="mistral-7b",     coeff=2, basis="exact", unit="tokens"),
    "WL04": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL05": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL06": dict(model="minilm-l6",      coeff=2, basis="exact", unit="tokens"),
    "WL10": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL11": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL12": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL13": dict(model="mistral-7b",     coeff=2, basis="exact", unit="tokens"),
    "WL14": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL16": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL21": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    "WL22": dict(model="tinyllama-1.1b", coeff=2, basis="exact", unit="tokens"),
    # LLM training — exact 6N geometry
    "WL07": dict(model="tinyllama-1.1b", coeff=6, basis="exact", unit="tokens"),
    "WL17": dict(model="tinyllama-1.1b", coeff=6, basis="exact", unit="tokens"),
    # vision / diffusion / speech / multimodal — documented per-unit estimates
    # (FLOP/unit; basis "estimate" — PMU cross-check is partial here, §3)
    "WL08": dict(model=None, basis="estimate", unit="images",
                 flops_per_unit=2.7e14,   # SDXL: 20 steps x2 CFG UNet + VAE
                 note="SDXL UNet ~6.6 TFLOP/fwd x 40 + VAE decode"),
    "WL09": dict(model=None, basis="estimate", unit="clips",
                 flops_per_unit=1.2e12,   # whisper-large-v3 enc+dec per clip
                 note="whisper-large-v3 encoder(1500 frames)+decoder"),
    "WL19": dict(model=None, basis="estimate", unit="images",
                 flops_per_unit=1.6e11,   # CLIP ViT-L/14 @224px
                 note="ViT-L/14 ~80 GMAC/image"),
    "WL20": dict(model=None, basis="estimate", unit="generations",
                 flops_per_unit=8.8e12,   # CLIP-L vision + Vicuna-7B 20-tok gen
                 note="CLIP-L vision ~0.16 TFLOP + 7B decode ~616 tok x 2N"),
}


def _attention_flops(geo, tokens, mean_context):
    """Attention FLOPs (QK^T + AV), summed over `tokens` query tokens, each
    attending to `mean_context` keys: 4 * layers * hidden * mean_context per
    token. For a causal prefill of S tokens pass tokens=S, mean_context=S/2
    (the average causal context — what a causal Flash/SDPA kernel executes).
    For a decode step pass tokens=1, mean_context=KV-length. For the DENSE
    (non-causal-optimised) count pass mean_context=S — used only to match
    FlopCounterMode, which has no causal mode (§3 cross-check)."""
    return 4.0 * geo["layers"] * geo["hidden"] * mean_context * tokens


def tensor_flops(wl_id, units, mean_context=128):
    """Total tensor-eligible FLOPs for a WL given the unit count measured by
    the harness (tokens for LLM WLs; images/clips/generations otherwise).

    The coefficient term uses (params - embed_params): the input embedding
    is a gather (0 matmul FLOPs); lm_head stays in params (real matmul).
    Attention is added for inference (x1) and training (x3 = fwd + bwd)."""
    spec = WL_FLOP.get(wl_id)
    if spec is None:
        return None
    if spec["basis"] == "estimate":
        return spec["flops_per_unit"] * units
    geo = MODEL_GEOMETRY[spec["model"]]
    n_flops = geo["params"] - geo["embed_params"]
    flops = spec["coeff"] * n_flops * units
    attn = _attention_flops(geo, units, mean_context)
    flops += attn * (1.0 if spec["coeff"] == 2 else 3.0)
    return flops


def compute_tensor_mfu(wl_id, units, elapsed_s, mean_context=128,
                       peak_flops=H100_PEAK_FP16_FLOPS):
    """Analytical tensor_mfu_pct for a WL run.
    units: tokens / images / clips / generations (harness total).
    Returns percent, or None if the WL has no FLOP accounting (e.g. WL23)."""
    f = tensor_flops(wl_id, units, mean_context)
    if f is None or elapsed_s <= 0:
        return None
    return f / (elapsed_s * peak_flops) * 100.0


def read_pmu_flops(exporter=EXPORTER_DEFAULT):
    """Cross-check path: cumulative FLOPs from the cipher_flops.c PMU counter
    (CP 3.3), exposed by cipher-exporter. Returns float FLOPs or None."""
    text = fetch_metrics(exporter)
    if not text:
        return None
    m = re.search(r'cipher_(?:gpu_)?flops_total\{[^}]*\}\s+(\S+)', text)
    return float(m.group(1)) if m else None


def crosscheck(wl_id, analytical_flops, pmu_flops, tol=0.05):
    """§3 cross-check: analytical vs PMU FLOP count. Returns
    (agree:bool|None, rel_delta:float|None)."""
    if not analytical_flops or not pmu_flops:
        return (None, None)
    rel = abs(analytical_flops - pmu_flops) / analytical_flops
    return (rel <= tol, rel)
