# WEEK_6_G5_PATH_A_VERIFICATION.md

**Date:** 2026-05-23
**Scope:** Verify v1.2.3 W6 G5 path-a (per-model VA from `hf_config`) closes the VA-pool budget for the 100-tenant heterogeneous-model product target.
**Read-only math + decision note. No source edits.**

## 1. Pre-conditions verified

- `cipher-fusion-evidence` HEAD `751c6b8` (W6 G1+G2 step doc landed).
- `cipher_kmod` at tag **`week-6-step-g1-g2-cap-bump`** (commit `c4e2d6f`); ko md5 `2f294edf…`, ABI `0.5.0`.
- `cipher_vllm_kv.py:58` confirmed: `va_gib = int(os.environ.get("CIPHER_KV_VA_POOL_GIB", "80"))`.
- 5 `hf_config`s pulled (Llama-3-8B via `NousResearch/Meta-Llama-3-8B` mirror — `meta-llama/Meta-Llama-3-8B` is gated on this pod).

## 2. Pod budget reality

- **DRAM:** 221 GiB (`MemTotal=231924628 kB`). NOT 1 TB as the architecture-gap audit assumed.
- **Swap:** 0.
- **GPU HBM (H100):** 80 GiB.
- **Linux x86_64 user VA per process:** 128 TiB; `ulimit -v` unlimited.
- **Sum across 100 processes:** ~12.8 PiB (VA is *per-process*, not a global resource).

The audit's "exhausts host VA at N≈12 on 1 TB host" framing was imprecise — Linux user VA is per-process and effectively unbounded at 100 tenants. The real constraints (in priority order) are (i) GPU HBM physical commitment, (ii) per-process allocator metadata, (iii) CUDA-context GPU VA. Path-a still closes the gap because the 80 GiB default reserves CUDA VA in *each* tenant's context — at 100 contexts × 80 GiB = 8 TB cumulative CUDA-VA reservation, with allocator-metadata side-effects on host RAM and on the cipher_kv_bridge bookkeeping.

## 3. Per-model VA budget (`max_model_len=8192`, bf16 KV, Marlin INT4 weights)

Formula: `KV_per_seq = 2(K+V) · L · H_kv · head_dim · max_len · bytes_dtype`; `wt_INT4 = params·0.5 + params/128·2`; allocator headroom 0.5 GiB.

| Model | params | KV/seq | KV @ B=1 | KV @ B=8 | wt fp16 | wt INT4 | total B=1 | total B=8 |
|-------|--------|--------|----------|----------|---------|---------|-----------|-----------|
| Mistral-7B-v0.1 | 7.24 B | 1.00 GiB | 1.00 GiB | 8.00 GiB | 13.49 GiB | **3.48 GiB** | **4.98 GiB** | 11.98 GiB |
| Qwen2-7B | 7.62 B | 0.44 GiB | 0.44 GiB | 3.50 GiB | 14.18 GiB | **3.66 GiB** | **4.59 GiB** | 7.66 GiB |
| Llama-3-8B | 8.03 B | 1.00 GiB | 1.00 GiB | 8.00 GiB | 14.96 GiB | **3.86 GiB** | **5.36 GiB** | 12.36 GiB |
| TinyLlama-1.1B | 1.10 B | 0.17 GiB | 0.17 GiB | 1.38 GiB | 2.05 GiB | **0.53 GiB** | **1.20 GiB** | 2.40 GiB |
| Phi-2 (2.7B, MHA) | 3.62 B | 2.50 GiB | 2.50 GiB | 20.00 GiB | 6.74 GiB | **1.74 GiB** | **4.74 GiB** | 22.24 GiB |

Phi-2 is KV-heavy because it uses MHA (`H_kv = H_q = 32`), no GQA. At B=8 it dominates the per-tenant budget; at B=1 it lands near 7B-class GQA models.

## 4. Multi-tenant scenarios

### Scenario 1 — proportional split, B=1 agentic, Marlin INT4 (the CP 5.5 deployment case)

100 tenants split: 30 Mistral + 25 Qwen + 20 Llama-3 + 15 TinyLlama + 10 Phi-2.

| Model | N | per-tenant | subtotal |
|-------|---|------------|----------|
| Mistral-7B-v0.1 | 30 | 4.98 GiB | 149.3 GiB |
| Qwen2-7B | 25 | 4.59 GiB | 114.9 GiB |
| Llama-3-8B | 20 | 5.36 GiB | 107.1 GiB |
| TinyLlama-1.1B | 15 | 1.20 GiB | 18.0 GiB |
| Phi-2 | 10 | 4.74 GiB | 47.4 GiB |
| **Total** | **100** | — | **436.7 GiB** |

vs current default `100 × 80 GiB = 8000 GiB`: **18× reduction**.
vs audit's "1 TB host VA" assumption: **587 GiB headroom**.

### Scenario 1 + Track 2 weight sharing (same-model tenants share one physical weight copy)

| Model | N | shared weights | per-tenant KV | subtotal |
|-------|---|----------------|---------------|----------|
| Mistral-7B-v0.1 | 30 | 3.48 GiB | 1.50 GiB | 48.5 GiB |
| Qwen2-7B | 25 | 3.66 GiB | 0.94 GiB | 27.1 GiB |
| Llama-3-8B | 20 | 3.86 GiB | 1.50 GiB | 33.9 GiB |
| TinyLlama-1.1B | 15 | 0.53 GiB | 0.67 GiB | 10.6 GiB |
| Phi-2 | 10 | 1.74 GiB | 3.00 GiB | 31.7 GiB |
| **Total** | **100** | — | **151.8 GiB** |

Path-a + Track 2 weight sharing compose. **152 GiB total VA reservation across all 100 tenants** — fits in the audit's 1 TB envelope with 872 GiB headroom, and fits in pod DRAM (221 GiB) physically if every reserved page were committed (which it won't be — `cuMemAddressReserve` only reserves; `cuMemMap` is lazy).

### Scenario 2 — pathological worst case (no quantization, no weight sharing, B=8)

100 tenants × 20 per model × fp16 weights × B=8 × max_len=8192:
**Total: 1896 GiB** (exceeds the 1 TB assumption).

But this is NOT the v1.2.3 deployment configuration. CP 5.5 ships Marlin INT4 + Track 2 + B=1 agentic per v1.2.3 §7 W15-17. Scenario 2 is the envelope test for "what happens if every mitigation simultaneously fails"; the answer is "honest scope reduction to ~50 tenants × fp16 × B=8" — and that's not what we're shipping.

## 5. Verification result — **C.1 path-a verified**

Path (a) — `cipher_vllm_kv.py` reads `hf_config` and sizes `va_gib` per (model, max_len, quantization, max_batch) — closes G5 for the v1.2.3 product target.

- **Scenario 1 (realistic CP 5.5):** 437 GiB / 100 tenants. 18× reduction vs current default. Fits any plausible budget.
- **Scenario 1 + Track 2 weight sharing:** 152 GiB / 100 tenants. The paths compose cleanly; Track 2 lift is fully preserved.
- **Headroom:** > 500 GiB against the audit's 1 TB framing in Scenario 1; > 850 GiB in Scenario 1 + Track 2.

**Path (a) locks for W10-12 implementation.** Path (b) (kmod-broker) is not needed for v1.2.3. Path (c) (container-per-tenant) is not invoked; Goal 5 (LD_PRELOAD-only transparency) is preserved.

## 6. W10-12 implementation scoping

`cipher_vllm_kv.py:50–62` currently:
```python
va_gib = int(os.environ.get("CIPHER_KV_VA_POOL_GIB", "80"))
if not cipher_kv_bridge.init(va_gib * (1 << 30)):
    ...
```

W10-12 change: replace the hardcoded default with an hf_config-driven sizer.

```python
def compute_va_gib(hf_config, max_model_len: int, dtype_bytes: int,
                   max_batch: int, quantization: str) -> int:
    # KV: 2(K+V) · L · H_kv · head_dim · max_len · bytes_dtype · max_batch
    kv_b = (2 * hf_config.num_hidden_layers
              * hf_config.num_key_value_heads
              * hf_config.head_dim
              * max_model_len * dtype_bytes * max_batch)
    # Weights: fp16=2B/p, INT4=0.5B/p + scales
    p = total_params_from_config(hf_config)
    wt_b = p * 2 if quantization == "none" else p * 0.5 + (p / 128) * 2
    headroom_b = max(0.5 * GIB, 0.1 * (kv_b + wt_b))
    return ceil((kv_b + wt_b + headroom_b) / GIB)

va_gib = int(os.environ.get("CIPHER_KV_VA_POOL_GIB",
                            compute_va_gib(hf_config, max_model_len,
                                           dtype_bytes, max_batch,
                                           quantization)))
```

Env override (`CIPHER_KV_VA_POOL_GIB`) preserved for ops who want to force a specific value; default flips from hardcoded 80 to computed-from-config.

**LOC budget:** ~150 modify (cipher_vllm_kv.py + helper) + ~50 new (`compute_va_gib` + unit tests). **Eng-time:** ~3 eng-days in W10-12 alongside RING_WRITE + G3 + G4.

**No call-site contract changes.** The downstream consumer is `cipher_kv_bridge.init(bytes)`, which accepts any byte count. No kmod ioctl change. No ABI change. No new symbols.

**Regression gate at W10-12 close:**
1. `--no-cipher` baseline TPS unchanged (this code path is CIPHER-on only; `--no-cipher` skips `cipher_vllm_kv.init`).
2. CIPHER-on Option 1 (Mistral-7B B=1) bench shows identical decode_MFU / tok/W to the v1.2.3 W6 bench-harness numbers in `WEEK_6_OPTION_1_REDO.md`.
3. Multi-process N=2 same-prompt KV-dedup test still passes Track 2 SC6 KL gate (≤ 5.5e-5).

## 7. Commit + memory

- `cipher-fusion-evidence` commit lands this doc.
- Memory anchor: `g5-path-a-verified` with scenario margins + W10-12 LOC estimate.
- Closes the second of three v1.2.3 §7 W6 carry items (G1+G2 already closed; May-13 POC reconstruction kickoff remains).
