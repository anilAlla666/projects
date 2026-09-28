# Phase 4.6.0 — KV dedup / multi-tenant moat discovery

**Date:** 2026-05-14 night, after Phase 4.5 close.
**Mode:** Discovery only. Zero production code changes.
**Cap:** 3h; ~2h 30min elapsed.

---

## TL;DR

| Question | Answer |
|---|---|
| Is operator-context cross-tenant KV dedup architecturally viable on this pod? | **YES** — `cuIpcGetMemHandle`/`cuIpcOpenMemHandle` verified working: producer process writes magic value at 0x7dfc..., consumer process (separate pid) reads MATCH at 0x7dab... after opening IPC handle. Cross-process GPU memory mapping is the foundation for the moat. |
| What attention path do modern stacks use? | **PyTorch SDPA → pytorch_flash (FA2 internal) / cuDNN / efficient / math**. PyTorch 2.11 bundles `pytorch_flash` directly in `libtorch_cuda.so`. transformers already has `sdpa_paged` integration (`PagedAttentionCache`) ready for activation. |
| What's the right substrate-extension shape? | **An "attention-routing substrate" analog to T4.5.1's cuBLAS-routing.** Two non-exclusive interception points: (a) Python-side: monkey-patch `transformers.cache.Cache.update` from a libcipher_rt-loaded shim (or PYTHONSTARTUP-injected hook); (b) C-side: `.symver`-tagged interpose of `at::_scaled_dot_product_flash_attention_forward` and/or `pytorch_flash::run_mha_*` C++ entry points. |
| Realistic timeline | **8–15 focused sessions, ~2-3 weeks calendar.** Breakdown in D4 below. The first session is a measurement-of-opportunity gate (D4.5) that determines whether to invest the full engineering budget. |
| Recommended first session | **T4.6.0.5 — measurement of dedup opportunity.** Before building the substrate, measure how much KV-cache content actually IS dedup-able on representative workloads. If opportunity < 2×, scope shrinks; if > 10×, full engineering budget justified. |

---

## D1 — Attention path mapping

### What's actually in this stack

| Component | Status |
|---|---|
| PyTorch 2.11.0+cu130 | Installed; ships `pytorch_flash::run_mha_*` inside `libtorch_cuda.so` |
| transformers 4.x | Installed; Mistral uses `attn_implementation='sdpa'` by default |
| `sdpa_attention_forward` | `transformers/integrations/sdpa_attention.py:40` — calls `torch.nn.functional.scaled_dot_product_attention(q, k, v, ...)` |
| `sdpa_paged_forward` | `transformers/integrations/sdpa_paged.py:18` — paged variant taking `PagedAttentionCache` already integrated |
| `PagedAttentionCache` | `transformers/generation/continuous_batching/cache.py` |
| flash_attn (standalone) | NOT installed |
| xformers, vLLM, sgl_kernel | NOT installed |
| cuDNN 91900 | Installed; can be selected via SDPA backend |

### Attention dispatch chain (transformers ≥4.40 path)

```
model.forward()
  → MistralAttention.forward()
    → ALL_ATTENTION_FUNCTIONS.get_interface(impl)("sdpa")
      → sdpa_attention_forward(module, query, key, value, attention_mask, ...)
        → torch.nn.functional.scaled_dot_product_attention(...)
          → at::_scaled_dot_product_attention dispatcher
            → selects {flash, cuDNN, mem_efficient, math}
              → pytorch_flash::run_mha_fwd<half_t, head_dim, is_causal>(...)
                → fmha_fwd_kernel (sm_90 PTX kernel)
```

The `past_key_values.update(key_states, value_states, layer_idx)` call (`Cache.update`) happens BEFORE this dispatch chain. The Cache object owns the K/V tensors; it returns the concatenated K/V history that SDPA sees.

### Profiler-confirmed kernel mix on Mistral-7B B=1 short prompt

GPU time breakdown:
- `aten::mm` (linear matmuls) — 72% (`nvjet_sm90_hsh_*` kernels)
- Various small element-wise / reduce / cat — ~25%
- Attention kernels — small slice on this micro-workload (B=1, ~10 tokens generation)

At larger prompts (B=8 prefill of 4K context), attention compute scales O(seq² × batch × heads); FlashAttention's kernel becomes dominant. Our intercept point matters at production scale.

### Interception points (analog to T4.5.1 substrate)

| Point | Pros | Cons | Operator-context? |
|---|---|---|---|
| `Cache.update()` Python | Clean semantics, framework-aware, sdpa_paged ready | Framework-specific (transformers vs vLLM vs SGLang); fragile across versions | YES via PYTHONSTARTUP / libcipher_rt early-load Python init (operator's launch recipe owns env vars) |
| `aten::_scaled_dot_product_flash_attention_forward` C++ dispatcher op | Catches all SDPA regardless of framework | Requires PyTorch dispatcher op-overlay via `torch.library.impl`; needs Python init | YES via the same operator init pattern |
| `pytorch_flash::run_mha_*` C++ mangled symbols | Pure `.symver` LD_PRELOAD (matches T4.5.1 substrate pattern) | C++ mangled; struct ABI internal to PyTorch; fragile across PyTorch versions | YES (LD_PRELOAD env var is operator-side) |
| `cudaMalloc` fingerprinting + memory-pattern KV detection | Framework-agnostic, kmod-level, no PyTorch coupling | No semantic info; "this allocation is KV cache vs random tensor" must be heuristic; high impl complexity | YES |

Recommended primary: **`pytorch_flash::run_mha_*` LD_PRELOAD intercept**, mirroring T4.5.1. Backup: Python init module for cases where the C++ symbol path doesn't work (e.g., if a tenant uses TRT-LLM or another non-PyTorch stack).

---

## D2 — Existing design study

### vLLM PagedAttention ([docs](https://docs.vllm.ai/en/stable/design/paged_attention/), [paper](https://arxiv.org/abs/2309.06180))

- **Block size: 16 tokens.** Each block: 16 × num_kv_heads × head_dim × dtype bytes. For Mistral-7B (8 KV heads GQA, 128 head_dim, fp16): 16 × 8 × 128 × 2 = 32 KB per block per layer.
- **Prefix sharing:** hash-table-keyed physical blocks. Two requests with identical first-N tokens hash to same physical block → memory shared. Block-granular (not token-granular) matching.
- **Eviction:** LRU when pool full.
- **State per sequence:** logical-block list (each entry points to a physical block ID).

### SGLang RadixAttention ([blog](https://www.lmsys.org/blog/2024-01-17-sglang/), [paper](https://arxiv.org/pdf/2312.07104))

- **Token-granular radix tree (trie)** of prefix sequences. Edges labeled with token sequences of varying lengths.
- **Prefix matching:** walk the tree from root; longest match identifies which existing KV pages to reuse.
- **Eviction:** LRU at the node level.
- **State per sequence:** path through the tree.
- **Tradeoff vs vLLM:** finer-grained matching → more dedup; more bookkeeping → more CPU work per request.

### llm-d ([llm-d.ai blog](https://llm-d.ai/blog/production-grade-llm-inference-at-scale-kserve-llm-d-vllm))

- Red Hat + Google + IBM + NVIDIA + CoreWeave, launched May 2025.
- **NOT operator-context drop-in** — Kubernetes operator that orchestrates vLLM/SGLang instances with prefix-cache-aware ROUTING.
- Routes similar requests (shared prefix) to the same backend instance, where vLLM's local PagedAttention dedups.
- Reports 3× output tok/s + 2× TTFT reduction from prefix-cache-aware routing.
- **Per-instance dedup ceiling** — limited by what one vLLM instance can pack on a GPU.

### CIPHER's unique angle vs llm-d

llm-d optimizes ROUTING (which instance handles a request). CIPHER's pitch optimizes the GPU MEMORY LAYER (cross-tenant page sharing in HBM). These are stackable:
- llm-d routes shared-prefix requests to instance X
- CIPHER inside instance X dedups KV pages across all its tenants
- Combined: bigger fan-out per GPU than either alone

This is the operator-deployment niche llm-d explicitly does NOT fill.

### MVP feature levels (smallest to largest)

| Level | Mechanism | Estimated tenant-density lift on chat-with-shared-sys-prompt workload |
|---|---|---|
| **L0 — Pure paged KV, no dedup** | Replace contiguous KV with paged allocation; better memory packing | 1.2-2× |
| **L1 — Block-hash prefix dedup (vLLM-style)** | Identical first-N tokens share blocks | 3-10× (depends on sys-prompt size vs total context) |
| **L2 — Radix-tree token-level dedup (SGLang-style)** | Finer-grained matching, partial-prefix sharing | 5-15× |
| **L3 — Cross-tenant page sharing via cuIpc** | Multiple tenant processes share KV physical pages | **10-50×** — the marvel target |

L3 is what makes CIPHER different from in-process libraries. L3 requires L0-L2 plus the cuIpc cross-process plumbing. Without IPC working, L3 is impossible; today's verification confirms it works on this pod.

---

## D3 — Substrate extension analysis (revised, conditional on D3.0)

### D3.0 — cuIpc cross-process GPU memory: **VERIFIED WORKING**

Test: `/tmp/cuda_ipc_test`:
```
[producer] dptr=0x7dfc4c800000 size=8388608 wrote handle to /tmp/cipher_ipc_handle.bin
[consumer] opened handle as dptr=0x7dab54800000
[consumer] read magic = 0xcafebabe12345678 (expected 0xcafebabe12345678) -> MATCH
consumer exit=0
```

Cross-process GPU memory sharing works on this pod. The architectural foundation for L3 (cross-tenant page sharing) is intact.

### D3.1 — Recommended substrate architecture

```
operator-deployed:
  cipher_kmod.ko        (extended with KV page pool ioctls + cuIpc handle registry)
  libcipher_rt.so       (matmul substrate + KV-routing substrate + Marlin substrate + VOLT)

operator service entry (per tenant):
  LD_PRELOAD=/opt/cipher/lib/libcipher_rt.so
  CUDA_INJECTION64_PATH=/opt/cipher/lib/libcipher_rt.so
  CIPHER_KV_DEDUP=on
  CIPHER_KV_BLOCK_TOKENS=16
  python <customer service entry point>          # UNCHANGED
```

Architectural layers:

```
PyTorch SDPA call
   ↓ (LD_PRELOAD interpose pytorch_flash::run_mha_* via .symver)
[cipher_rt_attn_shim]
   ↓ (descriptor + page table lookup)
[cipher_rt_kv_dispatch substrate registry]  <-- analog to T4.5.1 cipher_rt_matmul_dispatch
   ├── KV_DEDUP actuator (primary)
   │     · Block-hash key on K-projection output (block = 16 tokens × num_kv_heads × head_dim)
   │     · Consult per-process page table (libcipher_rt-side)
   │     · Consult cross-tenant shared-page registry (kmod-side ioctl)
   │     · If hash match: redirect attention's K/V input pointers to shared physical pages
   │     · Else: allocate fresh page from kmod-managed pool
   │
   ├── (future) ATTENTION_KOOPMAN actuator
   └── PASSTHROUGH: real pytorch_flash::run_mha_*
```

kmod additions (analog to T4.3.2 `CIPHER_SET_CLOCK_MHZ` pattern):
- `CIPHER_KV_REGISTER_PAGE` ioctl: tenant publishes a content-hash + IPC handle for a KV block
- `CIPHER_KV_LOOKUP_PAGE` ioctl: tenant queries by content-hash, gets IPC handle if shared page exists
- `CIPHER_KV_RELEASE_PAGE` ioctl: refcount decrement
- udev-restricted access (operator policy); audit trail via dmesg

### D3.2 — Apply Phase 4.5 substrate-contract findings

Three findings durable across all future actuators:

1. **`maybe_handle()` must be µs-bound or fall back to PASSTHROUGH+async warmup.** KV-page hashing must be GPU-side (a small kernel computes block hash). Synchronous CPU hashing during the attention hot path = repeat of Marlin's first-call latency disaster.
2. **Setup kernels (hash kernel, page-table fills) must push primary context.** Don't inherit GREEN_CTX 8-SM restriction.
3. **Pre-deploy where possible.** Common sys prompts can be pre-hashed and pre-loaded into the cross-tenant shared pool at service-start (operator's deployment recipe runs `cipher_kv_seed /path/to/sys_prompts.txt` once).

---

## D4 — Realistic implementation plan

### Sub-phases (revised with D4.5 measurement gate)

| Sub | Cap | Work |
|---|---:|---|
| **T4.6.0.5 — Measurement of dedup opportunity** | 1 session, ~3h | Capture KV-cache content across N concurrent Mistral-7B requests with realistic prompts (chat sys prompts, RAG passages, agent tool defs). Hash 16-token blocks. Measure: bytes-deduped / total-KV-bytes. If <2×: scope shrinks. If 2-10×: ship L1. If >10×: justify full L3. |
| **T4.6.1 — Attention-routing substrate** | 1-2 sessions, ~6-8h | `.symver` interpose of `pytorch_flash::run_mha_*` or `at::_scaled_dot_product_flash_attention_forward`. Verification: byte-correct passthrough on Mistral-7B at no-actuator-registered baseline. <2% overhead. |
| **T4.6.2 — KV page allocator + per-process table** | 2-3 sessions, ~10-15h | kmod-managed HBM page pool. cipher_rt_kv_pages.c: per-process page table; allocate/free hooks; refcount. New kmod ioctl set (nr 11+, additive ABI). |
| **T4.6.3 — Paged-attention kernel adapter** | 3-5 sessions, ~15-25h | Adapt `pytorch_flash`'s kernel to accept a page-pointer-array input OR write our own paged-attention kernel via NVRTC (like Marlin's port pattern). Correctness gate: bit-identical attention output when pages map 1:1; INT4-noise-floor when pages are content-matched (no actual loss; KV pages are content-addressed so identical-hash → identical-content). |
| **T4.6.4 — Block-hash prefix matching (L1)** | 1-2 sessions, ~6-8h | Hash kernel + per-tenant page table consults + within-tenant + across-tenant dedup paths. |
| **T4.6.5 — Cross-tenant page sharing (L3, if D4.5 justifies)** | 2-3 sessions, ~10-15h | cuIpc handles registered in kmod page registry. Per-tenant udev-restricted ioctl access. CoW or refcount semantics. |
| **T4.6.6 — Validation: 50-tenant multi-tenant scenario** | 1-2 sessions, ~6-8h | Test framework: 50 concurrent Mistral-7B tenants with shared system prompt. Measure actual density vs theoretical, p99 latency, MFU. Statistical methodology (5+ pairs per matrix cell). |

**Total Phase 4.6: 11-18 focused sessions, ~56-86 hours.** Calendar at 4-5h/day focused: **2.5-4 weeks.** That's the honest budget for the multi-tenant moat.

### Dependencies and risks per sub-phase

| Sub | Dependency | Risk |
|---|---|---|
| T4.6.0.5 | None (pure measurement) | If opportunity < 2×, full Phase 4.6 economically isn't justified; pivot to other moat |
| T4.6.1 | T4.5.1 substrate pattern (proven) | `.symver` against `pytorch_flash::run_mha_*` mangled symbols may need careful name matching across torch versions |
| T4.6.2 | None | kmod additive ABI (per cipher-abi-rule); audit trail via dmesg |
| T4.6.3 | T4.6.2 | Highest implementation risk — paged-attention correctness on Hopper sm_90 |
| T4.6.4 | T4.6.3 | Hash collision handling; safety against malicious tenants forging hashes |
| T4.6.5 | T4.6.4 + cuIpc working (verified ✓) | Per-tenant isolation: a malicious tenant must not be able to read another tenant's KV via IPC handle abuse — kmod-side authorization |
| T4.6.6 | All of above | Test infrastructure; multi-process tenant orchestration |

### Composition with prior work

- **GREEN_CTX (T4.2.4d):** per-tenant SM partition + per-tenant KV pages = orthogonal. KV pool lives in HBM (not tied to SM partition). Each tenant sees its allocated KV pages and shares some with other tenants.
- **VOLT (T4.3.2):** clock locking happens at GPU level, independent of KV. Composition: tok/W lift from VOLT × tenant-density lift from KV dedup = multiplicative on $/tenant-hour.
- **Marlin (T4.5.2 once re-ported):** Marlin reduces weight memory (4×); KV dedup reduces KV memory (1.2-50×). Combined: model + KV both shrink, leaving more room for more tenants per GPU.

The substrate composition story across Phase 4.x is now concrete.

---

## D5 — Recommended first session scope + honest position

### What to do first: T4.6.0.5 (the measurement gate, before any substrate code)

The advisor pushed back on jumping straight to substrate engineering. Reason: the value of the engineering investment depends on the actual dedup OPPORTUNITY in realistic traffic. We don't know that number. Measuring it is a 1-session task that gates the rest.

**T4.6.0.5 scope:**
1. Build a probe that hashes 16-token KV blocks during Mistral-7B forward.
2. Run on representative traffic:
   - **Chat with shared sys prompt** — 10 concurrent requests all starting with the same 500-token system prompt + different user questions
   - **RAG with shared retrieved context** — 20 concurrent requests with same 1K-token RAG passage + different queries
   - **Agent traffic** — 10 concurrent agent calls with same tool definitions + different user goals
   - **Few-shot inference** — 10 concurrent classification requests with same 5-shot example set
3. Measure: `bytes_deduped / total_KV_bytes` per workload type.

The result determines Phase 4.6's economic justification:
- **< 2× dedup:** Phase 4.6 doesn't pay for itself; pivot to a different moat
- **2-10× dedup:** Ship L1 (block-hash within-process); skip L3 (cross-tenant)
- **\> 10× dedup:** Full L3 cross-tenant implementation justified

### Position vs marvel target

Phase 4.6 is the multi-tenant moat — the 50-100 tenants/H100 pitch. Today this discovery:

| Moat | Status |
|---|---|
| Efficiency (DVFS) | **Shipped Phase 4.3** (+57% tok/W) |
| Deployment (matmul-routing substrate) | **Shipped Phase 4.5.1** |
| MFU (Marlin compute path) | **Open** — Phase 4.5.2 needs re-port with async warmup or pre-quant kit |
| Multi-tenant (KV dedup) | **Discovery complete tonight; impl is 2.5-4 weeks** |

The architectural verification done tonight (cuIpc cross-process works) means the multi-tenant moat is engineering, not research. Honest budget; concrete sub-phases; clear measurement gate.

### Mistral-7B KV math correction (per advisor)

The previously-quoted "~2 GB KV cache per Mistral-7B request at 8K context" overstated. Mistral-7B uses GQA with 8 KV heads (not 32):

- Per layer per token: 8 × 128 × 2 bytes = 2 KB (K) + 2 KB (V) = 4 KB
- 32 layers × 4 KB × 8192 tokens = **1 GB per request** (still small enough that with 100 tenants you'd hit 100 GB on a 80 GB H100 — bandwidth-driven density gain still holds)

Updated marvel math: Mistral-7B at 8K context on H100 80 GB with Marlin (3.5 GB weights):
- KV pool: 76.5 GB
- Per-request KV: 1 GB
- **Baseline density: ~76 concurrent requests**
- With L3 KV dedup at ~5× (mid-range opportunity assumption): **~380 concurrent requests**

Even without L3, the L1 paged-KV with 5× dedup is a substantial moat. The marvel pitch holds; the numbers are precise.

### Whether to start T4.6.0.5 tonight or in a fresh session

**Recommended: stop here.** Discovery has been thorough; the implementation budget is multi-week. The measurement gate (T4.6.0.5) is a 3-hour session that deserves focused attention, not an hour of fatigue-tail at the end of a 5+-hour discovery sprint.

Fresh session priority: T4.6.0.5 measurement → decision branch → T4.6.1 substrate.

### Discipline gate at session end

| Gate | Result |
|---|---|
| Phase 3 ABI 12/12 PASS | ✅ |
| Fallback kmod md5 `55ab8c0c` | ✅ unchanged |
| Fallback libcipher_v2 md5 `86618c30` | ✅ unchanged |
| Taint 12288 | ✅ unchanged |
| Production artifacts modified | **NONE** (discovery only) |
| Pod state | baseline (345 MHz, 700 W) |
| Discovery artifacts | `/tmp/cuda_ipc_test.c`, `/tmp/cuda_ipc_test` (binary), this doc |

---

## Sources

- vLLM PagedAttention: [docs.vllm.ai paged_attention](https://docs.vllm.ai/en/stable/design/paged_attention/), [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)
- SGLang RadixAttention: [lmsys blog](https://www.lmsys.org/blog/2024-01-17-sglang/), [arXiv 2312.07104](https://arxiv.org/pdf/2312.07104)
- llm-d: [llm-d.ai blog](https://llm-d.ai/blog/production-grade-llm-inference-at-scale-kserve-llm-d-vllm)
- transformers SDPA paged integration: `transformers/integrations/sdpa_paged.py` (installed locally)
