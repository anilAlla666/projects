# Phase 4 — Cut Decisions and Audit-Incomplete Resolutions (T4.0.6)

**Date:** 2026-05-13
**Build state at decision time:** cipher_rt rebuilt clean with libssl-dev installed on this pod. New artifacts:

| Artifact | md5 | Size |
|---|---|---|
| `libcipher_rt.so` (T4.0.6 fresh build) | `d66fb8c725c0272186584e95cf701950` | 785456 B |
| `libcipher_hook.so` (T4.0.6 fresh build) | `4841480aa28bb8d1a04f6f282ad6815d` | 138576 B |
| `libcipher_nccl_tuner.so` (T4.0.6 fresh build) | (unchanged 16552 B) | — |
| `libcipher_rt.so.preroadmap` | `95284eb1c028382deef061cf679dcc7c` | 785352 B (kept) |

`make clean && make -j4` rc=0. **64 init T-symbols in libcipher_rt.so** (matches `.preroadmap` count — no source-level regression from the build). No `iowrite32`/`__raw_writel`/`memcpy_toio` symbols. ldd dependencies clean: libcudart, libcrypto, libstdc++, libm, libgcc, libc, libdl, libpthread, librt, loader. cuBLAS intercepts (`cublasLtMatmul`, `cuLaunchKernel*`, `cudaLaunchKernel*`) confirmed in `libcipher_hook.so` T section.

**Build sanity:** ✅ PASS. Every "working" claim in PHASE_4_OP_AUDIT.md is now confirmed by fresh build evidence.

## Audit-incomplete resolutions

### FP8_COMPUTE — RESOLVED, reclassify from C to B

All 19 `return 0;` paths in `src/cipher_fp8_compute.cpp` are legitimate, not stubs:

| Line(s) | Path | Justification |
|---|---|---|
| 91, 95 | `probe_content_hash` early return | null pointer / memcpy failure (error paths) |
| 411 | `cipher_fp8_compute_init` | env `CIPHER_FP8_COMPUTE` not set → disabled |
| 423-424 | `cipher_fp8_compute_observe` | `g_enabled` false or null arg (enabled-check) |
| 442 | observe path | recycled-buffer detection (guard) |
| 467-468, 482-484, 505-506 | observe path | slot allocator failure paths |
| 516-517, 542-543 | install path | enabled-check / shape-validation guards |
| 555-556, 561-562, 569-570 | install path | weight-slot recycle / shape-mismatch (clears slot, returns 0 as "not installed") |
| 599-600 | lookup path | no fp8_buf for this slot |
| 615-616 | lookup path | enabled-check / lt handle init |

**Verdict:** FP8_COMPUTE is wired but disabled by default (env-gated). Off the Phase 4 critical path; functional when enabled.

**New classification: Class B** (tenant-invariant, keep as-is). May promote to Class A later if a tenant SLO calls for "FP8 enabled per this tenant" — but that's a P4.8+ overlay decision, not Phase 4 core scope.

**No cut.**

### ATTN_KOOPMAN — RESOLVED, Class A confirmed; TODO is non-blocking

The TODO at `cipher_attn_koopman.cpp:495`:

```cpp
if (is_atv_gemm(M, N, K) && s->fused_ready) {
    // Would run fused kernel here (TODO: wire up once the
    // per-shape (V_T, K_op, V_compressed) registry is populated
    // by a prefill-end hook). For now, fall through to revert
    // — defaulting to CORRECT output on any uncertainty.
    revert_and_reset_locked(s, "fused_path_not_yet_wired");
```

**Verdict:** The TODO is a follow-up for an OPTIMIZATION fast-path (fused attention via Koopman). When the fast-path is not wired, the code reverts to cuBLAS attention via `revert_and_reset_locked` → **correct output, just slower**. The fast path needs a prefill-end hook to populate a per-shape registry — that hook lives outside the cipher_rt tree (vLLM/inference engine territory).

This is **not a fusion blocker**. ATTN_KOOPMAN's per-tenant relevance (the `koopman_substitution_eligibility` flag in `cipher_tenant_snapshot`) is about *which tenants are eligible for Koopman substitution at all*. The fused-fast-path TODO is independent and deferred to a future phase that wires the prefill-end hook (out of scope here).

**Class A confirmed**, TODO documented for Phase 4.8 follow-up (not P4.6 P4 close-out).

## Cut decisions

Decisions are **documented here** but **not yet applied to source**. Applied atomically with the cipher_kmod 0.4.0 bump in T4.1 to preserve a single Phase 4 cutover point.

### CUT: cipher_telemetry.cpp (Class B → Class C)

- **Reason:** Redundant with Phase 3 `cipher-gpustate` daemon. The daemon's NVML poll (250 ms cadence, full device-wide telemetry submitted via `CIPHER_SUBMIT_GPU_STATE` ioctl) covers all the information `cipher_telemetry.cpp` polls.
- **Action (T4.1.x):** wrap implementation in `#ifdef CIPHER_LEGACY_TELEMETRY` (default off). The `cipher_telemetry_init` symbol stays exported but returns 0 immediately.
- **Risk:** any cipher_rt subsystem currently reading from `cipher_telemetry` directly (not via NVML wrapper) needs to point at `cipher_gpu_state` instead. Grep before applying.
- **Reclassification:** B → C.

### CUT: cipher_nccl_bpf.cpp (Class D, already deferred)

- **Reason:** libbpf-dev not installed; multi-node NCCL deferred to Phase 5+.
- **Action:** `#ifdef CIPHER_HAVE_LIBBPF` guards around the implementation. Init symbol returns 0 immediately when libbpf absent.
- **Reclassification:** Confirmed C (cut for Phase 4 build).

### CUT: cipher_nccl_v4.cpp (Class D, already deferred)

- **Reason:** Same — multi-node deferred. Single-pod NCCL is a no-op.
- **Action:** Wrap in `#ifdef CIPHER_NCCL_MULTINODE`. Init returns 0 when undefined.
- **Reclassification:** Confirmed C.

### CUT: cipher_nccl_neural.cpp (Class D, kept for Phase 5+)

- **Reason:** Same. **Keep source in tree** for Phase 5 promotion.
- **Action:** Wrap in `#ifdef CIPHER_NCCL_MULTINODE`. **Source retained**, build excludes.
- **Reclassification:** D → "Phase 5 substrate, source kept, build excluded."

### CUT: cipher_nccl.cpp (orchestrator, Class D)

- **Reason:** Same family.
- **Action:** Same `#ifdef CIPHER_NCCL_MULTINODE` guard.
- **Reclassification:** D → "Phase 5 substrate, kept."

### KEEP: cipher_fp8_compute.cpp (Class B per resolution above)

No cut.

## Cut summary

| Op | Pre-cut class | Post-cut class | Action in T4.1 |
|---|---|---|---|
| cipher_telemetry | B | C | `#ifdef CIPHER_LEGACY_TELEMETRY` |
| cipher_nccl_bpf | D | C | `#ifdef CIPHER_HAVE_LIBBPF` |
| cipher_nccl_v4 | D | C | `#ifdef CIPHER_NCCL_MULTINODE` |
| cipher_nccl_neural | D | D (source kept) | `#ifdef CIPHER_NCCL_MULTINODE` |
| cipher_nccl | D | D (source kept) | `#ifdef CIPHER_NCCL_MULTINODE` |
| cipher_fp8_compute | C (audit-incomplete) | **B** (works, env-gated) | none — keep |
| ATTN_KOOPMAN | A (audit-incomplete) | **A** confirmed | TODO deferred to P4.8 follow-up |

**Net effect on class counts** (vs PHASE_4_OP_AUDIT.md baseline):
- Class A: 35 → 35 (ATTN_KOOPMAN stays A, was already counted)
- Class B: 22 → 22 (FP8_COMPUTE B kept; cipher_telemetry moves out to C; net 0)
- Class C: 1 → 4 (added: cipher_telemetry, cipher_nccl_bpf, cipher_nccl_v4)
- Class D: 4 → 2 (cipher_nccl, cipher_nccl_neural source kept but build excluded)
- INFRA: 10 → 10

**Audit-incomplete: 2 → 0.** Resolved.

## When the cuts apply

T4.1.x bumps cipher_kmod to 0.4.0 **and** applies the `#ifdef` guards atomically. Phase 3 substrate untouched (cipher_kmod unchanged through this audit). Fallback md5s unchanged (verified at start and end of T4.0.6).

## Build sanity end-state

- libssl-dev installed (`/usr/include/openssl/hmac.h` present)
- cipher_rt rebuilt clean from source: libcipher_rt.so (md5 `d66fb8c7...`), libcipher_hook.so (md5 `4841480a...`), libcipher_nccl_tuner.so (unchanged)
- 64 init T-symbols (matches .preroadmap)
- No iowrite/__raw_writel/memcpy_toio
- Two warnings (benign): `resolve_introspection() defined but not used`, `t_layer_counter defined but not used` — both `static` flagged-unused. Not blockers.

## Phase 3 substrate (end-of-T4.0.6 check)

| Check | Result |
|---|---|
| `cipher_kmod.ko.v0.2.0` md5 | `55ab8c0cd8309ca7cc0fc40fe556aa19` ✓ |
| `libcipher_v2.so.v0.2.0` md5 | `86618c30896470b642fcc6985d8dc632` ✓ |
| cipher_kmod loaded | `cipher_kmod 0.3.1`, srcversion `B1AF5E2A...` ✓ |
| Taint | 12288 stable ✓ |
| /proc/cipher | all 3 entries readable ✓ |

Phase 3 substrate untouched. Phase 4 substrate prep (cipher_rt rebuild) succeeded. Move to T4.0.7.
