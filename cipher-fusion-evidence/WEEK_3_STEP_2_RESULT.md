# Week 3 Step 2 — Observer Extension: TLS `substitute_hint` Publish — RESULT

**Status: PASS.**

Observer's `observe()` extended with hint-publish logic. Per-thread `cipher_rt_substitute_hint` populated on every GEMM-classified launch when `cipher_rt_dispatch_lookup()` returns a non-PASS_THROUGH action. Marlin doesn't consume the hint yet (Step 3); SC6 TinyLlama bit-identical confirmed in both modes — substrate is observe-only at runtime.

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 3, Step 2 of 5
**Anchors:**
  - Pre:  `0b6effdb` (`week-3-step-1-dispatch-scaffold`)
  - Post: `71bf5523` (`week-3-step-2-observer-hint-publish`)
  - libcipher_rt.so md5: `896bf19d` → `ca3492e3` (nvcc non-deterministic; source identity git-verified)

---

## A — Pre-edit verification + snapshot — PASS

| signal | value |
| --- | --- |
| HEAD | `0b6effdb` ✓ |
| working tree | clean |
| Pre-build rc | 0 |
| Pre-build md5 | `896bf19d` (informational; nvcc non-deterministic) |
| Pre-build warnings | 78 |
| Symbol count baseline | 310 |
| step2_pre snapshot | preserved at `/tmp/week3_step2/libcipher_rt.so.step2_pre` |

---

## B — Oracle PERMIT decision path — v1 hardcoded TRUE

### Inspection findings

`cipher::oracle_decide(state, query)` returns `CipherOracleResult{decision, reason}` where `decision ∈ {CIPHER_ORACLE_PERMIT(0), CIPHER_ORACLE_DENY(1)}`.

Reads `CipherOracleState->initialized` at `cipher_oracle.cpp:270` (and `:310`) and returns early on uninitialized state. cipher_rt_phase4's deferred-init model (no `cipher_init()` call from native code per impedance check §3.4) leaves that state uninitialized → undefined-ish behavior on call.

OpClass::GEMM = 0 (uint8_t enum from `cipher_classify.hpp`).

### Decision: hardcode `permit = 1` in v1

- Wave 4 brief is the canonical place to wire real oracle (when deferred-init is reconciled OR when v1.5 attn substitute requires oracle gating).
- Marlin actuator's own gate at `maybe_handle_marlin` L100-114 (shape + `MARLIN_MAX_M_GATE` + N/K≥1024 filters) enforces the real safety constraints. The classifier-published hint is advisory; the actuator's gate is binding.
- The semantic in v1 is "always permit GEMM substitution attempts" — same as the existing Marlin behavior pre-Week-3.

Documented inline in `cipher_rt_classify_observer.c` as v1 rationale; Week 4 deliverable to wire real oracle is listed in WEEK_2_STEP_7_CLOSEOUT.md §7.

---

## C — `cipher_rt_classify_observer.h` extension (+21 lines)

```c
/* Week 3 Step 2 — per-thread substitute hint published by observer
 * when oracle PERMIT + op_class==GEMM + dispatch_lookup() returns a
 * non-PASS_THROUGH action. Marlin actuator reads this in Step 3;
 * the hint is acted upon when CIPHER_DISPATCH_LIVE=1 (Step 4).
 *
 * Wave 5 W3-1 SYNTHESIS-HYPOTHESIS resolution: ... folded into the
 * existing `observe()` per scope-lock §2 Step 2. */
struct cipher_rt_substitute_hint {
    uint8_t  action;        /* enum cipher_rt_dispatch_action */
    uint16_t actuator_hint; /* opaque hint slot from dispatch table */
    uint8_t  valid;         /* 1 = hint applies; 0 = no substitute */
    uint32_t reserved;
};

const struct cipher_rt_substitute_hint *
cipher_rt_classify_observer_get_tls_hint(void);
```

---

## D — `cipher_rt_classify_observer.c` extension (+65 lines)

### D.1 Include + TLS storage

```c
#include "cipher_rt_dispatch.h"   /* lookup() + decision struct */

static __thread struct cipher_rt_substitute_hint g_tls_hint;
```

`__thread` is per-thread BSS-zero on thread creation. No locks needed (per-thread by definition).

### D.2 Publish block in `observe()` (after existing 3 atomic increments + per-op-class bucket)

```c
{
    const uint8_t OPCLASS_GEMM = 0;
    int permit = 1;  /* v1: hardcoded TRUE; see rationale comment */

    if (result == CIPHER_RT_CLASSIFY_HANDLED
        && out->op_class == OPCLASS_GEMM
        && permit) {
        struct cipher_rt_dispatch_decision decision;
        int rc = cipher_rt_dispatch_lookup(out->op_class, &decision);
        if (rc == 0
            && decision.action != CIPHER_RT_DISPATCH_PASS_THROUGH) {
            g_tls_hint.action        = decision.action;
            g_tls_hint.actuator_hint = decision.actuator_hint;
            g_tls_hint.valid         = 1;
            return;
        }
    }
}
g_tls_hint.valid = 0;
```

Semantics:
- HANDLED + GEMM + permit + ROUTE_MARLIN (v1 table) → valid=1, action=ROUTE_MARLIN
- Any other path → valid=0
- The hint is published on every observe() call; consumers (Step 3) read valid first.

### D.3 Getter

```c
const struct cipher_rt_substitute_hint *
cipher_rt_classify_observer_get_tls_hint(void)
{
    return &g_tls_hint;
}
```

No locks; hot-path safe.

---

## E — Makefile diff (+1 line)

```
  cipher_rt_classify_observer.o: cipher_rt_classify_observer.c \
                                  cipher_rt_classify_observer.h \
                                  cipher_rt_classify_substrate.h \
+                                 cipher_rt_dispatch.h
      $(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<
```

Forces rebuild of observer.o when dispatch.h changes.

---

## F — Build + nm diff + symbol audit — PASS

| signal | value |
| --- | --- |
| build rc | 0 |
| warning count | 78 → 78 (zero delta) |
| libcipher_rt.so md5 | `896bf19d` → `ca3492e3` (nvcc non-deterministic) |
| Symbol count | 310 → 311 |
| Symbols removed | 0 |

### Added (1 T-symbol)

```
T cipher_rt_classify_observer_get_tls_hint
```

(The `__thread g_tls_hint` storage is internal-linkage; correctly absent from the dynamic symbol table.)

### Undef audit

Clean — only expected system/weak surface; no new CIPHER undefineds.

---

## G — Runtime smoke + SC6 gate (LOAD-BEARING) — PASS

### G.1 Loader smoke

```
$ LD_PRELOAD=$(realpath libcipher_rt.so) /bin/true
[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)
[cipher_v2] DISPATCH: substrate registered, LIVE=0 (0 = observe-only until Step 4 flip; set CIPHER_DISPATCH_LIVE=1 to engage)
rc=0
```

Both substrates' load banners fire as designed.

### G.2 SDPA shim injection smoke

```
rc=0; trampoline accounting:
  tramp_calls=5 tramp_fake=0 observed=5 handled=0 passthrough=5
  redirected=0 (flash=0 eff=0 cudnn=5)
MATMUL: exit totals — calls=0 handled=0 passthrough=0 (actuators=0)
DIAG-T4.2.4d: total_launches=30
```

LP-2 invariant holds (`handled=0 passthrough=5`). The observer's `observe()` runs per-launch under CUPTI; each call publishes-or-clears the TLS hint, but no consumer reads it. Hint side-effect is purely TLS-resident.

### G.3 SC6 TinyLlama bit-identical gate

**Vanilla baseline (no CUDA_INJECTION64_PATH):**

```
$ python3 sc6_run.py TinyLlama shared
SC6 TinyLlama/shared: PASS
Runtime: 34s (17:59:16 → 17:59:50)
JSON: all4_fwd1_bit_identical=true, all4_fwd2_bit_identical_after_producer_death=true,
      all 7/7 sub-checks PASS
```

**CIPHER-injected:**

```
$ CUDA_INJECTION64_PATH=$(realpath libcipher_rt.so) python3 sc6_run.py TinyLlama shared
SC6 TinyLlama/shared: PASS
Runtime: 33s (17:59:59 → 18:00:32)
JSON: all4_fwd1_bit_identical=true, all4_fwd2_bit_identical_after_producer_death=true,
      all 7/7 sub-checks PASS
```

**Both modes 7/7 PASS.** Observer publishes per-thread substitute_hint on every classify call; no consumer reads it; bit-identity preserved across all 4 consumers (pre + post producer death). Substrate observe-only at runtime as designed.

---

## H — CP 5.4 regression — PASS

```
=== Phase A result: 15 PASS, 0 FAIL ===
rc=0
```

Byte-identical to Step 1 baseline.

---

## I — Commit + tag — PASS

```
3 files changed, 88 insertions(+), 1 deletion(-)
  Makefile                      |  3 +-
  cipher_rt_classify_observer.c | 65 ++++++++++++++++++++++++++++++++++++++
  cipher_rt_classify_observer.h | 21 +++++++++++++

HEAD: 71bf552321e0ee75ba94840868951ed54328361f
Tag:  week-3-step-2-observer-hint-publish
Pre tag: week-3-step-1-dispatch-scaffold (0b6effdb)
```

---

## Discipline notes

- Single commit. Tag chain: `week-3-step-1-dispatch-scaffold` → `week-3-step-2-observer-hint-publish`.
- Rollback: `git -C cipher_rt_phase4 reset --hard week-3-step-1-dispatch-scaffold`.
- Resolves Wave 5 W3-1 SYNTHESIS-HYPOTHESIS: Wave 5's `maybe_handle` prescription folded into Step-4's existing `observe()` function name.
- Oracle PERMIT hardcoded TRUE in v1 (rationale documented inline + here in §B). Week 4 brief is the canonical place to wire real oracle.
- Observer publishes hint on every call; Marlin doesn't read it yet (Step 3). CIPHER_DISPATCH_LIVE still 0.
- TLS storage (`__thread`) is per-thread BSS-zero, correctly absent from the dynamic symbol table.

---

## What this unlocks

- **Week 3 Step 3** — Marlin actuator reads `cipher_rt_classify_observer_get_tls_hint()` on entry to `maybe_handle_marlin`; if hint.valid + hint.action == ROUTE_MARLIN, short-circuit the STABILITY_THRESHOLD warmup. VOLT decode-band 1200 MHz lock added.
- **Week 3 Step 4** — Flip `CIPHER_DISPATCH_LIVE=1` default; Marlin hint consumption becomes routing-decision-making (not just warmup-skip).
- **Week 3 Step 5** — Closeout.

---

## Anchors at close

| tree | HEAD | tag |
| --- | --- | --- |
| cipher_rt_phase4 | `71bf5523` | `week-3-step-2-observer-hint-publish` |
| cipher_kmod | `0ce4b8e2` | (unchanged from week-2-complete) |
| cipher-may13-evidence | `fc8a9ae6` | (unchanged) |
| libcipher_rt.so md5 | `ca3492e3` | (this build; nvcc non-deterministic) |
| libcipher_rt.so.step2_pre snapshot | `896bf19d` | preserved at `/tmp/week3_step2/` |

Week 3 progress: **2/5 steps complete**. Step 3 brief draftable.
