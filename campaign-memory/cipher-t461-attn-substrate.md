---
name: cipher-t461-attn-substrate
description: T4.6.1 attention-routing substrate shipped. Plain-mangled LD_PRELOAD on ATen SDPA dispatcher (no .symver — ATen ops un-versioned). cuDNN is hot path on H100/cu13; flash/eff trampolines built but unverified at runtime.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

T4.6.1 (2026-05-14) shipped the attention-routing substrate — twin of
T4.5.1 cuBLAS substrate for PyTorch SDPA.

**Architecture shipped:**
- `cipher_rt_attn_dispatch.{cpp,h}` — 16-actuator priority-ordered
  registry + 3 LD_PRELOAD trampolines (flash / efficient / cuDNN
  ::call symbols in libtorch_cpu.so).
- `cipher_rt_attn_test_actuator.c` — smoke actuator (env-gated by
  `CIPHER_ATTN_TEST=on`).
- libcipher_rt.so v0.3.0_T4_6_1 (md5 22ce23bbfc25bb9100c845ac6900c053).
- Pre-T4_6_1 saved as libcipher_rt.so.pre_T4_6_1 (md5 9ee93ec8958ab0de868c0bc0506c1cfa).

**Key architectural findings vs phase prompt assumptions:**

1. **Intercept symbol is in libtorch_cpu.so, NOT libtorch_cuda.so.**
   ATen op-dispatcher (`at::_ops::*::call`) is universal-backend and
   lives in libtorch_cpu.so. Phase prompt said libtorch_cuda.so — wrong.
2. **No `.symver` directive needed.** ATen ops are plain mangled C++
   symbols, no `@@` version tags (unlike libcublas.so.13 which T4.5.1
   does need .symver for). DO NOT add SDPA symbols to a version map's
   global clause with local:*; — would hide them and silently break
   interposition.
3. **cuDNN is the default backend on H100 + cu13 + torch 2.11.**
   `_scaled_dot_product_cudnn_attention::call` fires for all Mistral-7B
   attention calls; flash and efficient trampolines never hit on this
   stack. Substrate hooks all three for portability (pre-Hopper or
   non-cuDNN stacks would hit flash/eff).
4. **Return tuples use `c10::SymInt`, not `int64_t`** for max_q/max_k.
   Substituting int64_t silently corrupts return slot. Use ATen header
   schema typedef.
5. **`--no-as-needed` is mandatory at link time** for `-ltorch_cpu`.
   Without it the linker drops libtorch_cpu from DT_NEEDED (our
   trampolines satisfy the symbols), and dlsym RTLD_NEXT returns nothing.

**Verified-against stack:** torch 2.11.0+cu130, Hopper H100, kernel
6.8.0-1046-nvidia. Header has `CIPHER_RT_ATTN_TORCH_VERIFIED = "2.11.0+cu130"`.

**Drift-handling contract:** lazy per-call dlsym RTLD_NEXT; abort() on
failure (NOT "register inactive and fall through" — that's impossible
once the mangled symbol is exported from libcipher_rt). Loud crash is
the correct failure mode vs silent model corruption. Documented in
PHASE_4_T4_6_1_SIGNATURE.md.

**REDIRECTED mechanism deferred to T4.6.3.** Substrate wires the enum
+ telemetry but currently treats REDIRECTED as PASSTHROUGH with a
stderr note. T4.6.3 (L1 dedup actuator) commits the choice between (a)
alloc-new-tensor with shared storage or (b) ATen storage-swap.

**Verification on Mistral-7B (cuDNN backend):**
- 3 prompts × 32 tokens hash-matched substrate vs baseline (byte-identical)
- 32,896 cuDNN calls intercepted in composition run
- T4.5.1 cuBLAS substrate: 21,600 passthrough calls, both substrates
  compose without conflict
- VOLT interference A/B (pre-T4_6_1 vs T4_6_1, both with VOLT on):
  tok/W 0.128 vs 0.127 — within 0.8% noise. No measurable interference.

**Side observation (not T4.6.1 issue):** VOLT lift vs baseline on B=1
decode dropped to +2.4% on this pod today (0.128 vs 0.125), far below
T4.3.2's claimed +57%. Pod-state regression in VOLT path, NOT
composition-related. Flagged for follow-up.

**Coverage gap:** flash and efficient trampolines are built and
symbol-exported, signature-matched against ATen headers, but were
never hit at runtime on this stack. First pre-Hopper workload will be
the empirical smoke test for those paths.

**Discipline:** ABI surface (ioctl nrs 1-10) untouched. No kmod changes.
Fallback md5s 55ab8c0c / 86618c30 unchanged. Taint 12288.

**Next: T4.6.2 — KV page allocator** (per-process page table, block=16,
FNV-1a hash → device pointer). 2-3 sessions, 10-15h. Lives behind the
T4.6.1 substrate.

Linked: [[cipher-t46-measurement]] (outcome-c commit T4.6.1 implements),
[[cipher-t45-substrate-marlin]] (cuBLAS substrate pattern T4.6.1 mirrors),
[[cipher-t46-kv-dedup-discovery]] (the discovery this substrate executes on).

Full docs: PHASE_4_T4_6_1_REPORT.md (engineering+strategic),
PHASE_4_T4_6_1_SIGNATURE.md (mangled names + drift contract).
Kit: cipher-phase4-evidence/t4_6_1_substrate/kit/.
