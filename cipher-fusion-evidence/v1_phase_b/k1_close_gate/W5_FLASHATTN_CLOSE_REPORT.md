# W.5 FLASHATTENTION 4-PATTERN INTERCEPT — CLOSE REPORT

**Date:** 2026-05-28
**Substrate anchor:** `cipher_rt_phase4/build_cuda13/libcipher_rt.so` md5 `fa9fd8f71870944862443f7cb588dd7b`
**Side anchor preserved:** `build_cuda13/libcipher_rt.so.w5` (identical bytes)

## Verdict
**W.5 PASS.** 4 of 6 attention patterns GOT-patched (FA2/FA3/FlashMLA/
PagedAttn-v2); FlashInfer P3/P4 = JIT residue (§D, Anil-adjudicated
2026-05-28). FA3 (the H100 FLASH_ATTN hot path) intercepts 5280/5280/
5120/4352 calls on P1/P2/T1/C1. K.1+W.1+W.2+W.3+W.5 9/9 close gate
PASS, ZERO segfaults. Substitution = passthrough (byte-identical to
baseline → Memory #11 KL gate trivially holds).

## A — 6 patterns + symbol resolution

| Pattern | Entry symbol | Library | Status |
|---------|-------------|---------|--------|
| P1 FlashAttention-2 | `flash::run_mha_fwd` (`_ZN5flash11run_mha_fwd...`) | _vllm_fa2_C.abi3.so | ARMED |
| P2 FlashAttention-3 | `run_mha_fwd` (`_Z11run_mha_fwd...`) | _vllm_fa3_C.abi3.so | **ARMED + intercepting (H100 hot path)** |
| P3 FlashInfer prefill | JIT-compiled (no static symbol) | flashinfer (csrc + build_backend.py) | §D residue |
| P4 FlashInfer decode | JIT-compiled (no static symbol) | flashinfer | §D residue |
| P5 FlashMLA | `FMHACutlassSM100FwdRun` | _flashmla_C.abi3.so | ARMED (DeepSeek; 0 on ref workloads) |
| P6 PagedAttention v2 | `paged_attention_v2` | _C.abi3.so | ARMED (0 on FLASH_ATTN-backend ref workloads) |

`nm -D` enumeration in cipher_rt_attn_6pattern.c::SYM_P{1,2,5,6}.

## B — Trampolines + dispatch wire

### File: `cipher_rt_phase4/cipher_rt_attn_6pattern.c` (new)

- 4 naked-asm GOTPCREL trampolines (`cipher_attn_p{1,2,5,6}_tramp`),
  reusing the W.2 Machete pattern: only r11 (SysV scratch) touched;
  arg registers pass through ABI-faithfully for the per-pattern
  signatures. Each bumps intercepts + passthrough then tail-jmps to
  the resolved original.
- `arm_pattern()` dlopens the specific .so + dlsym the mangled symbol
  (real original) + `cipher_rt_got_register(mangled, trampoline)`.
- `cipher_rt_attn_6pattern_init()` arms all 4; retry-hook in
  `observe_launch_v2` (1024-tick) re-arms as vLLM lazy-loads the libs.

### Wire
- `cipher_inject.c`: `cipher_rt_attn_6pattern_init()` after Machete init.
- `Makefile`: `cipher_rt_attn_6pattern.o` added to OBJS.
- CLASSIFY log: `attn_p1 / attn_p2 / attn_p5 / attn_p6` intercept counters.

## C — Per-pattern correctness gate + Memory #11

v1 ships intercept + passthrough only — the trampoline forwards every
call to the original kernel, so output is **byte-identical to baseline**.
Memory #11 KL gate is therefore trivially satisfied (no substitution =
no divergence). The per-pattern KL acceptance gate facilities land with
the substitution kernels (§D, v1.x).

**Substrate-level Memory #11 evidence**: vLLM completes all 9 close-gate
cells (4 with active FA3 interception) without crash or output
corruption. The segfault iteration below is itself a Memory #11 catch —
the close gate stopped a broken build from shipping.

### Segfault iteration (CRITICAL — Memory #11 working as designed)

- **v1 md5 de0a20b7**: trampolines defined AS the mangled target symbols
  with default visibility. This SHADOWED the library exports, so
  `dlsym(lib_handle, mangled)` returned OUR trampoline instead of the
  real function → trampoline tail-jmp'd to itself → unbounded recursion
  → stack overflow → **segfault** (E3 cell: `attn_p2=1` then
  `!!!!!!! Segfault encountered !!!!!!!`). Close gate FAILED 4/9. HARD STOP.
- **v2 md5 fa9fd8f7**: trampolines renamed to UNIQUE HIDDEN symbols
  (`cipher_attn_p{1,2,5,6}_tramp`, `visibility("hidden")`). They no longer
  shadow lib exports → dlsym resolves the real original → no recursion.
  Defensive `if (orig == trampoline)` skip added. `nm -D` confirms no
  dynamic export of run_mha_fwd / FMHACutlass / paged_attention_v2.
  E3 smoke: `attn_p2=21` clean, EXIT=0, no segfault. Full gate: 9/9 PASS.

## D — 4-workload engagement table + per-pattern residue

| Cell | Class | attn_p1 (FA2) | **attn_p2 (FA3)** | attn_p5 (MLA) | attn_p6 (PagedAttn) |
|------|-------|---------------|---------------------|----------------|----------------------|
| P1 Llama-3-8B B=8 bf16   | A4 | 0 | **5280** | 0 | 0 |
| P2 Mistral-7B B=1 bf16   | A4 | 0 | **5280** | 0 | 0 |
| T1 Llama-3 transition    | A3 | 0 | **5120** | 0 | 0 |
| C1 Llama-3 calibration   | A4 | 0 | **4352** | 0 | 0 |
| P3 TinyLlama-AWQ B=1     | A3 | 0 | (FA3 path; AWQ attention is dtype-agnostic) | 0 | 0 |
| N1 / E1 / E2 / E3        | UNK | 0 | 0 | 0 | 0 |

K.1+W.1+W.2+W.3+W.5 close gate: **9/9 PASS**, zero segfaults.
All prior counters preserved (VOLT 5/5 engage, Marlin bf16 20k+ subs,
Machete 14k intercepts on P3, Koopman bf16 obs).

### Per-pattern residue (§D — substitution kernels + FlashInfer)

1. **FlashInfer P3/P4 JIT intercept** (Anil-adjudicated 2026-05-28):
   flashinfer ships csrc + build_backend.py, no static .so — JIT-compiled
   at runtime. GOT-patch can't reach JIT modules statically. Needs an
   NVRTC-cache hook or dlopen-callback on the JIT module load. v1.x.
2. **Substitution kernels** for all 4 armed patterns: v1 is intercept +
   passthrough. Substrate-side optimized attention kernels (FA3-shape-
   matched, MLA, PagedAttn) are v1.x. Per spec sub-step (i), the W.5 bar
   is "patterns intercept + passthrough (byte-identical correctness)";
   substitution kernels are explicit §D items, NOT silent narrowing.
3. **FA2/FlashMLA/PagedAttn-v2 intercept=0 on ref workloads**: vLLM 0.21
   FLASH_ATTN backend on H100 routes to FA3, so the other 3 armed
   patterns never fire on Llama/Mistral/TinyLlama. They are ARMED and
   would intercept on workloads that use them (FA2 fallback, DeepSeek
   MLA, PagedAttn-v2 backend). Coverage validated via arming logs.
4. **Hopper FA3 TMA descriptor handling** for substitution kernel: v1.x.
5. **Heterogeneous head_dim batching** (W.4 POOL co-engagement): v1.x.
6. **100K+ ctx scale validation**: V.1 CP 5.5.

## E — Anchors

| Component | Pre-W.5 (post-W.3) | Post-W.5 |
|-----------|---------------------|----------|
| `cipher_rt_phase4` HEAD              | e992457 (w3-koopman-bf16-edmd-wire) | (commit pending) tag `w5-flashattn-6pattern` |
| `cipher_rt_phase4` libcipher_rt.so md5 | 8b2e14cf                          | **fa9fd8f71870944862443f7cb588dd7b** |
| `cipher-fusion-evidence` HEAD        | ce63dd1 (w3-koopman-bf16-edmd-close) | (commit pending) tag `w5-flashattn-close` |
| `cipher_kmod` HEAD                   | 8c643fc (UNCHANGED)                | 8c643fc (UNCHANGED) |
| `cipher-platform` .deb               | rev8 md5 5603f72d (UNCHANGED)      | rev8 md5 5603f72d (UNCHANGED) |

W.5 iteration trail (2 builds to PASS):
- v1 md5 de0a20b7: trampoline-shadows-lib-symbol → recursion → E3 segfault → 4/9 FAIL (Memory #11 HARD STOP)
- **v2 md5 fa9fd8f7: unique hidden trampolines → dlsym resolves real original → 9/9 PASS, FA3 intercept 5280/cell, zero segfaults**

## F — Memory #29 next-substep

Per resequenced Memory #29 binding (W.5 → W.6 sub-B/C → W.4 → W.7 → W.8):
- **W.6 sub-B**: NR 27 REGISTER_MODEL plugin Python-heap crash fix
  (Memory cipher-audit-2026-05-16; ~2-3 ED).

## G — Goal 3 MFU lever status

| Goal | Lever | Status |
|------|-------|--------|
| Goal 3 | Attention intercept       | substrate-wired W.5 ✓ INTERCEPTING (FA3 5280/cell); substitution kernels v1.x |
| Goal 3 | POOL cross-tenant batching | pending W.4 |
| Goal 3 | SM packer                 | pending P.3 |
| Goal 2 | DVFS / Marlin / Machete    | wired W.1/W.2 (engaging) |
| Goal 4 | Koopman O(1) bf16         | wired W.3 (armed; v2 broader-rank) |
| Goal 1 | POOL multiplexing         | pending W.4 (needs W.6 sub-C signal) |
| Goal 5 | zero-touch CDI            | shipped CP 2.5 |

Goal 3 (85% MFU) is a composition target: attention intercept (W.5) +
POOL cross-tenant batching (W.4) + SM packer (P.3) stack. W.5 wires the
attention lever; the MFU headline needs W.4 + P.3 to compose.
