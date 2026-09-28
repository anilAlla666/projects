---
name: cipher-cp51-closed
description: "CIPHER CP 5.1 (vLLM KV integration) CLOSED 2026-05-17 — Option A buffer-ownership hook, correctness gate PASS on TinyLlama + Mistral-7B"
metadata: 
  node_type: memory
  type: project
  originSessionId: 12c1ae17-5dfa-47e3-b06f-7f8aeda1f2c3
---

CP 5.1 (vLLM live-decode integration, Phase 5 — see [[cipher-phase5-scoped]]) **CLOSED 2026-05-17**.

**What shipped:** Option A — `cipher_vllm_plugin/` (pip `cipher-vllm-kv 0.1.0`) monkey-patches `GPUModelRunner._allocate_kv_cache_tensors` to source vLLM 0.20.2's raw int8 KV buffers from `cipher_kv_bridge.vmm_zeros` instead of `torch.zeros`; installed via the `vllm.general_plugins` entry point so `register()` fires in the EngineCore subprocess; env gate `CIPHER_KV_ALLOC`. Coverage-immune: CIPHER *is* the buffer.

**Gate (memo §4 criterion #2):** `cp_5_1/cp51_kv_gate.py` PASS on **both** TinyLlama-1.1B (MHA, 22 KV buffers) and Mistral-7B-v0.1 (GQA 32q/8kv, 32 KV buffers) — token-ID identity substrate-off vs -on (4 prompts × 64 tok greedy) + hook confirmed in EngineCore. No OOM at util=0.30 (KU#4 budget-neutrality empirically confirmed).

**Why:** campaign runs design-memo→approve→build per atomic step ([[cipher-fusion-campaign]]); adjudication required the GQA model before close.

**How to apply:**
- Anchors held: kmod `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`. `cipher_kv_bridge` is a NON-anchor build artifact, now `8d6ffe3f` (int8 dtype added for vLLM's flat-int8 KV buffer); prior `.so` preserved `cipher_kv_bridge.so.pre_cp51`.
- The `cipher_kmod/cipher_proc.c` /proc banner `0.4.5`→`0.4.8` edit is STILL staged, NOT built — rebuilding rotates anchor `e2f50452`; rides the next *legitimate* kmod rebuild ([[cipher-kbuild-clean-wipes-ko]] / [[cipher-devnode-codified]]).
- Two gaps deferred OUT of CP 5.1 scope: KU#1 page-tag granularity (per-layer-buffer, K/V interleaved by stride — affects T4.6.3 dedup keying); KU#3 uniform-kv path (`allocate_uniform_kv_caches`, connector-only, unpatched — CIPHER bypassed under disaggregated serving).
- CP 5.1 close certifies memo §4 criterion **#2 only** (correctness). Criteria #1 (CP 4.8 env-fail workloads), #3 (operator deployment recipe), #4 (MFU/tok-W on live continuous-batching decode) are NOT asserted by this close.
- Build log: `cipher-fusion-evidence/cp_5_1/CP_5_1_STEP_3_BUILD_LOG.md`.
