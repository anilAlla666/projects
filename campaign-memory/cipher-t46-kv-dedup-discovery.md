---
name: cipher-t46-kv-dedup-discovery
description: Phase 4.6.0 KV-dedup discovery. cuIpc cross-process GPU memory verified working on this pod; substrate architecture concrete; impl is 2.5-4 weeks.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

Phase 4.6.0 (2026-05-14 night, 3h discovery cap) mapped the
multi-tenant moat technical landscape.

**Critical architectural verification:** `cuIpcGetMemHandle` /
`cuIpcOpenMemHandle` works on this pod. Cross-process GPU memory
sharing is viable → kmod-managed cross-tenant page pool architecture
is open. Test probe: producer process writes magic value 0xCAFEBABE...
at one device pointer; consumer process (separate pid) opens IPC
handle, reads MATCH. This is the foundation for the L3
"cross-tenant KV page sharing" moat that distinguishes CIPHER from
llm-d (which only does cross-instance ROUTING, not cross-tenant
shared GPU memory).

**Modern attention dispatch path:** PyTorch 2.11 ships `pytorch_flash`
internally in libtorch_cuda.so. transformers ≥4.40 uses
`sdpa_attention_forward` → `torch.nn.functional.scaled_dot_product_attention`
→ `aten::_scaled_dot_product_flash_attention_forward` → 
`pytorch_flash::run_mha_*`. transformers already has `sdpa_paged`
integration + `PagedAttentionCache` ready for activation.

**Substrate-pattern interception points** (analog to T4.5.1 cuBLAS-routing):
1. `.symver`-tagged LD_PRELOAD of `pytorch_flash::run_mha_*` C++ mangled
   symbols (primary; mirrors T4.5.1 pattern exactly)
2. `aten::_scaled_dot_product_flash_attention_forward` dispatcher
   op-overlay via Python init module (backup)
3. `Cache.update` monkey-patch via PYTHONSTARTUP/libcipher_rt early load
   (operator-side, framework-aware)

**Existing designs surveyed:** vLLM PagedAttention (16-token blocks,
hash-table prefix dedup); SGLang RadixAttention (token-level radix tree);
llm-d (Kubernetes-level prefix-aware routing — orchestration not GPU-memory
dedup). CIPHER's unique angle: operator-context cross-tenant KV page
sharing in HBM. Stackable on top of llm-d.

**Mistral-7B KV math corrected (advisor catch):** 8 KV heads GQA, not 32.
Per-request at 8K context = 1 GB (not 2 GB). H100 80 GB baseline: ~76
concurrent requests. With 5× L1 dedup: ~380 concurrent requests.

**Implementation plan, honest budget:**
- T4.6.0.5 measurement-of-opportunity gate (3h, fresh session)
- T4.6.1 attention-routing substrate (1-2 sessions)
- T4.6.2 KV page allocator + per-process table (2-3 sessions)
- T4.6.3 paged-attention kernel adapter (3-5 sessions — highest risk)
- T4.6.4 L1 block-hash prefix matching (1-2 sessions)
- T4.6.5 L3 cross-tenant page sharing (2-3 sessions)
- T4.6.6 multi-tenant validation (1-2 sessions)
Total: 11-18 sessions, 56-86 hours, calendar 2.5-4 weeks.

**Recommended next session:** T4.6.0.5 measurement. Build a probe that
hashes 16-token KV blocks during representative workloads (chat with
shared sys prompt, RAG with shared context, agent with shared tool
defs). Measure bytes-deduped / total-KV-bytes. Result branches:
- <2× → pivot to different moat
- 2-10× → ship L1 only
- >10× → full L3 implementation justified

**Substrate-contract findings applied:** Hash kernels must be GPU-side
(µs budget on hot path); setup kernels must push primary context
(not inherit GREEN_CTX 8-SM); pre-deploy common sys-prompt hashes at
service-start where possible.

**Discipline:** ABI 12/12 PASS, fallback md5s unchanged, taint 12288,
NO production artifacts modified. Discovery probe in
`cipher-phase4-evidence/t4_6_0_discovery/cuda_ipc_test.c`. Full report
`PHASE_4_T4_6_0_DISCOVERY.md`.

Linked: [[cipher-t45-substrate-marlin]] (the substrate pattern T4.6
extends), [[cipher-t432-kmod-volt-ioctl]] (ioctl ABI pattern T4.6.2
will mirror).
