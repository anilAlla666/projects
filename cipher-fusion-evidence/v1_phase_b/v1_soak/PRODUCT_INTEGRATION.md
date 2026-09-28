# CIPHER PRODUCT INTEGRATION — V.1 Tier A (what composes today vs the gated roadmap)

**HONEST HEADLINE:** CIPHER composes a **capture-safe product across two regimes** — decode/agent (engine ⊕ DVFS ⊕ NF4) and compute (FP8 substitution) — under one routing surface (`cipher_product.py`), default-OFF, OFF byte-identical, anchor `1f305ce6` untouched. It does **NOT** yet deliver: non-GEMM driver-level substitution, attention fusion, compute-actuators-inside-the-engine-graph, 85% MFU, or sub-0.37% FP8 quality — **each is gated for an earned reason (Tier B roadmap below), NOT wired here.** This is the maximal integration *within the discipline*, not 'all 5 goals delivered'.

## TIER A — COMPOSES & VALIDATED LIVE (this product run)

| Regime (router) | Composable set | Live result this run | Capture-safe? | Tags |
|---|---|---|---|---|
| **agent/decode** | engine (pager + static-KV graph-decode + coalescing) ⊕ DVFS ⊕ NF4 | **100/100 agents FAULT=0** (teacher-forced KL=0 LIVE: exact=93 tie=7); P99 4.1s/10.5s; DVFS tok/W 2.002 | YES | `b6508bb` `4e6f77b` `a182daa` |
| **agent/decode: density** | NF4 packed weights ⊕ graph-decode ⊕ DVFS | **3 NF4 models co-resident, all KL=0=True** (~5.6GB/7B) | YES | `a182daa` |
| **compute/prefill** | FP8 E4M3 substitution at cublasGemmEx (n>64) | **MFU 65.3% @ batch=8**, FP8 substituted (handled=11475) | eager (not graph) | `aab6ea6` |

Routing is by **declared regime** (the `serve_manifest` list), not closed-loop: the live classifier *classifies* (A4_BATCH_INFERENCE etc., shown in V.0v2) but **classifier-driven dispatch is itself Tier B** — the router hands each job to the regime's validated handler (subprocess in the correct injection state, because engine = auto-init-disabled for capture-safety vs FP8 = cublasGemmEx actuator conflict). **What is NEW = the orchestration layer only** (`cipher_product.py` router + `cipher_product_report.py`); **no new `.so` / kernel / substrate capability** — this packages the V.0-validated composable set as ONE product surface.

## DISCIPLINE & NO-REGRESSION (live, this run)

- **OFF byte-identical:** PASS — `.so` injected with all actuators OFF is BIT-IDENTICAL to baseline (baseline=('-1336048.875000', '450'), OFF=('-1336048.875000', '450')).
- **Engine modules UNCHANGED (byte-identical):** PASS — cipher_engine.py/a6538d743cee, cipher_engine_batched.py/1cb85eb60ddf, cipher_inc4.py/ebd4b76f03b3 (match pre-Tier-A md5s).
- **Anchor / `.so` UNCHANGED:** deployed `libcipher_rt.so` (2026-06-01) + git `b2304d3`; NO `.so` source change (read-only counters + additive harness only).
- **Engine gate LIVE:** inc-4 100-agent (inc-1..inc-4 + g1.1 stack) re-run this product run → `[GATE inc4 K=4] PASS` (FAULT=0, 4th independent live pass this session).
- **Honest scope of this no-regression:** K=4 → swap is INACTIVE, so the K<4 **swap path is both gated (un-rooted debt) AND untested by this product run** — surfaced in Tier B, not silently covered.

## TIER B — THE GATED ROADMAP (real increments, NOT wired here — discipline)

Each needs a `.so` source change, a rejected mechanism, or a capture-illegal op → surfaced, not faked. (My decision rule.)

| Gate | Why gated (earned THIS session) | What it actually needs | Evidence |
|---|---|---|---|
| Non-GEMM **driver-level** substitution (RMSNorm/SiLU fused kernels exist) | matcher detects RMSNorm seqs but `real_substituted=0` — X/W/Y device pointers are in opaque ATen arg structs, not `args[i]` | reverse-engineer ATen kernel arg layout per build | `CIPHER_FLOW_STAGE2_STAGE3.md:5,61` |
| **Attention** fused kernel (`cipher_attn_fused_kernel` built) | FSM detects QK→softmax→AV but `"fused_path_not_yet_wired"`; the (V_T,K_op) registry needs Koopman EDMD = capture-illegal cusolver | capture-safe low-rank registry + prefill-end hook | `cipher_attn_koopman.cpp:495-499` |
| Compute actuators (Marlin/FP8/Koopman) **inside the engine graph** | their NVRTC/cudaMalloc/cudaHostAlloc/cusolver are capture-illegal in `torch.cuda.graph` | offline weight-prequant before capture, or a capture-legal kernel path | `edc0b8b` (g1.3) |
| FP8 **quality** (+0.72% PPL > 0.37% bound) | per-tensor scalar scheme (the only fused cuBLASLt path on cu13) breaks the bound | custom CUTLASS rowwise (per-channel-W + per-token-A) fused kernel = v1.5 | `aab6ea6` `cublaslt-cu13-fp8-pertensor-only` |
| **85% MFU** (single-GPU ~68%) | bf16 GEMMs already 95-100% of peak → forward is non-GEMM-bound; 85% needs NCCL-overlap | multi-GPU (8+) / attention-norm fusion | `aab6ea6` |
| Engine **swap** root-cause | re-capture-per-serve works (REQ=200) but corruption mechanism un-characterized | white-box capture/VMM mechanism analysis | `6c08fee` `54d6f5f` |
| STAGE13 monkeypatch 2× tok/W (fused RMSNorm+SiLU RAN there) | application-level forward-replacement = Mem #24-rejected for production + carried a quality divergence | a driver-level capture-safe path (= the non-GEMM-driver-substitution gate above) | `STAGE13_2X_ACHIEVED.md:25,42` |

## BOTTOM LINE

**Composes today (Tier A, validated live):** the multi-model engine ⊕ DVFS energy ⊕ NF4 density in the decode/agent regime, and FP8 substitution in the compute regime — under one product router, default-OFF, OFF byte-identical, zero regression, anchor untouched. **Gated (Tier B, honestly surfaced):** non-GEMM driver substitution, attention fusion, in-graph compute actuators, sub-0.37% FP8, 85% MFU, swap root-cause — each a real increment requiring work the discipline forbids faking. This is everything we built, fused as far as the discipline allows — and an honest map of the rest.