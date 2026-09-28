# Week 2 Step 1 — LP-2 SDPA Trampoline Refactor — RESULT

**Status:** PASS
**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 1 of 7
**Wave 5 finding addressed:** LP-2 (CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md §5.4)
**Tree touched:** `cipher_rt_phase4` (Tree B1 — userspace runtime)
**Anchors:**
  - Pre:  `bcf8a83b` (`week-1-complete`)
  - Post: `465b244e` (`week-2-step-1-lp2-sdpa-refactor`)
  - libcipher_rt.so md5: `88ed35bb13b0524e889bb5b56d610fd9` → `56291439c7fd941ca37cf86c482948d7`

---

## 1. Headline

The three SDPA trampolines in `cipher_rt_attn_dispatch.cpp` (flash, efficient, cuDNN) now honor `route()`'s return value. The HANDLED branch is guarded by a defensive `abort()` until v1.5 provides the actuator-side substitute-attention construction logic. No actuator returns HANDLED today, so the branch is unreachable at runtime — every kernel still reaches the real SDPA function, runtime behavior unchanged. The refactor unlocks v1.5 attention substitute actuators (KV dedup L1, paged-attn adapter, FlashSwiftKey) without further substrate edits.

---

## 2. What changed

### 2.1 `cipher_rt_attn_dispatch.h` — Cb.2-style reserved-tail bump

Added `void *out_status_devptr` to `struct cipher_rt_attn_call` (L96), consuming one 8-byte slot of the existing `uint64_t reserved[6]` tail. Result: `uint64_t reserved[5]` at L97, `sizeof(cipher_rt_attn_call) = 408 B` preserved.

The new field is the slot where a future HANDLED actuator writes a device pointer to the substitute SDPA output. The trampoline (in v1.5) will reconstruct the libtorch return tuple from it. For v1, the field is allocated but unused.

Cb.2 invariants verified:
  - `sizeof(struct cipher_rt_attn_call) = 408` (unchanged)
  - `offsetof(struct cipher_rt_attn_call, backend) = 0` (unchanged)
  - `offsetof(struct cipher_rt_attn_call, q) = 8` (unchanged)
  - ... (all existing field offsets unchanged through `attn_bias_rank`)
  - `offsetof(struct cipher_rt_attn_call, out_status_devptr) = 360` (new)
  - `offsetof(struct cipher_rt_attn_call, reserved) = 368` (shifted +8 by design — consumes 8 B of the previously-reserved tail)

### 2.2 `cipher_rt_attn_dispatch.cpp` — three trampolines refactored

For each of the three trampoline functions (flash @ L289, efficient @ L324, cuDNN @ L362), the existing call:

```c
route(c);
return orig(...);
```

is replaced with:

```c
int r = route(c);
if (r == CIPHER_RT_ATTN_HANDLED) {
    fprintf(stderr, "[cipher-attn] FATAL: <backend> trampoline saw HANDLED "
                    "but substitute-return construction is v1.5 work. "
                    "Aborting.\n");
    abort();
}
return orig(...);
```

The `abort()` is intentional and defensive: HANDLED-return-construction requires deserializing `out_status_devptr` into the libtorch return tuple (which differs per backend — a tuple of {output, log_sumexp, debug_mask, seed, offset} for flash; differs again for efficient and cuDNN). That work is deferred to v1.5. Until then, any actuator that returns HANDLED is a bug, and crashing loudly beats silently corrupting attention output.

PASSTHROUGH (1), REDIRECTED (2), ERROR (3): unchanged — fall through to `orig()`.

---

## 3. Build verification

```
$ cd /home/ubuntu/cipher_rt_phase4 && ./build.sh
... 18 warnings (same as baseline) ...
rc=0
```

Warning delta: **0** (18 pre → 18 post; identical text).

libcipher_rt.so md5: `88ed35bb13b0524e889bb5b56d610fd9` → `56291439c7fd941ca37cf86c482948d7`.

---

## 4. Runtime verification

### 4.1 SDPA smoke (cuDNN backend)

A 1×4×32×64 fp16 SDPA call under the post-refactor `libcipher_rt.so`:

```
torch 2.11.0+cu130 cuda available True
SDPA output shape: torch.Size([1, 4, 32, 64]) dtype: torch.float16
SDPA output mean: 0.01479339599609375
PASS: SDPA path executes cleanly under post-LP-2-refactor libcipher_rt.so

[cipher-attn] exit totals — tramp_calls=1 tramp_fake=0 observed=1
              handled=0 passthrough=1 redirected=0
              (flash=0 eff=0 cudnn=1)
```

Reading the counters:
  - `tramp_calls=1`: the cuDNN trampoline was entered once (this is the SDPA backend selected by torch for this small workload)
  - `handled=0`: **no actuator returned HANDLED** — the defensive `abort()` guard never fired
  - `passthrough=1`: route() chain returned PASSTHROUGH (AUDIT observer @ priority 0); orig() called
  - `cudnn=1`: backend tag correctly identified

This confirms the design invariant: the LP-2 refactor preserves runtime behavior because no registered actuator returns HANDLED today.

### 4.2 CP 5.4 isolation regression

CP 5.4 SM-arbitration isolation suite executed against post-refactor libcipher_rt.so:

```
PASS 15/15
```

Output diff against Week 1 closeout baseline: **empty**. Byte-identical.

### 4.3 Tree-wide HANDLED grep

```
$ grep -rn "return CIPHER_RT_ATTN_HANDLED" cipher_rt_phase4/
(zero matches outside the enum definition itself)
```

Confirms: no actuator returns HANDLED in v1. The new guard is structurally unreachable.

---

## 5. Tag chain

```
week-1-complete                  → bcf8a83b
week-2-step-1-lp2-sdpa-refactor  → 465b244e   (this step)
```

Rollback: `git reset --hard week-1-complete`.

---

## 6. What this unlocks

The substrate now correctly distinguishes "actuator handled the call" from "actuator wants passthrough." Future v1.5 work — Op-3 FAVOR+ attention substitute (KV dedup live, FlashSwiftKey, paged-attention adapter) — can register as an actuator that returns HANDLED without further trampoline edits, once the per-backend return-tuple reconstruction is built.

Out of scope of this step (deferred to v1.5):
  - Per-backend return-tuple reconstruction from `out_status_devptr` (replaces the `abort()` with real construction)
  - Actuators that populate `out_status_devptr` and return HANDLED
  - Test coverage for the HANDLED path (no actuator returns it today, so no test can exercise it)

The Wave 5 LP-2 finding is resolved at the substrate level. The actuator side ships in v1.5.

---

## 7. Discipline notes

  - Single-commit step; no in-flight rebase.
  - Cb.2 pattern reused from Week 1 Step 4 v2 (kmod snapshot reserved-tail).
  - Defensive `abort()` over silent fall-through — failing loudly is the v1 contract.
  - No mitigation in this document.
