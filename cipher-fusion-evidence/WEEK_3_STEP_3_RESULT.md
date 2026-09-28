# Week 3 Step 3 — Marlin Hint Consumption + VOLT Discrepancy — RESULT

**Status: PASS.**

Marlin actuator reads TLS `substitute_hint` from observer at entry; advisory at `LIVE=0`, binding at `LIVE=1` (Step 4). VOLT 1200 MHz lock NOT modified — surfaced as v1.2.2-vs-may13-measured discrepancy (LUT honors measured optimum 1000 MHz). SC6 TinyLlama bit-identical in both modes (substrate observe-only at LIVE=0).

**Date:** 2026-05-20
**Phase:** v1.2.2 §7 Week 3, Step 3 of 5
**Anchors:** `71bf5523` → `4f1a86ab` (tag `week-3-step-3-marlin-hint-volt-lock`)
**libcipher_rt.so md5:** `c41182e0` → `f63e1f2e` (nvcc non-deterministic)

## A — Baseline + snapshot — PASS

HEAD `71bf5523` ✓, pre-build rc=0, md5 `c41182e0`, warnings 78, 311 symbols, `step3_pre` snapshot preserved.

## B — Marlin entry path inspection

`maybe_handle_marlin()` at `cipher_rt_marlin_actuator.c:62+` (post #include addition). Pre-existing entry checks: `g_enabled` atomic at L92-93, dtype check at L95-99 (CUDA_R_16F), shape gates at L107-114 (`MARLIN_MAX_M_GATE`, `marlin_N >= 1024`, `marlin_K >= 1024`). New hint-read block inserted **between** `g_enabled` check and dtype check.

## C — Marlin extension (+22 LOC)

```c
#include "cipher_rt_dispatch.h"
#include "cipher_rt_classify_observer.h"
...

if (!atomic_load(&g_enabled))
    return CIPHER_RT_MATMUL_PASSTHROUGH;

/* Week 3 Step 3 — read TLS substitute_hint... */
{
    const struct cipher_rt_substitute_hint *hint =
        cipher_rt_classify_observer_get_tls_hint();
    if (cipher_rt_dispatch_is_live() && hint && hint->valid) {
        if (hint->action != CIPHER_RT_DISPATCH_ROUTE_MARLIN) {
            atomic_fetch_add(&g_calls_skipped, 1);
            return CIPHER_RT_MATMUL_PASSTHROUGH;
        }
        /* hint says ROUTE_MARLIN — existing gates stay binding */
    }
}

if (call->Atype != CUDA_R_16F || ...)  /* existing dtype check */
```

`cipher_rt_dispatch_is_live()` returns 0 at default → entire block inert. Step 4 flips default to 1 → block becomes active. Existing gates remain binding even when hint is honored (Marlin can still skip on geometry incompatibility).

## D — VOLT 1200 MHz lock: NOT MODIFIED (discrepancy surfaced)

**Inspection at `cipher_rt_volt.c:54-65`:**

```c
static unsigned int batch_to_mhz(int batch) {
    switch (batch) {
    case 1:   return 1000;   /* B=1 — may13 measured optimum */
    case 8:   return 1600;
    case 32:  return 1980;
    case 64:  return 1980;
    default:  return 0;
    }
}
```

**Discrepancy:**

- **v1.2.2 §7 L1303**: "VOLT engagement: trigger 1200 MHz lock on detect-decode-band."
- **Wave 5 §5.5 W3 L736**: "trigger `CIPHER_VOLT_BATCH=1` (**1000 MHz** lock) when classifier detects sustained decode."
- **Existing LUT**: B=1 → **1000 MHz** (sourced from `cipher-may13-evidence/p5_optimal_clocks.json` measured optimum).

The measured optimum on H100 is 1000 MHz at B=1; the spec named 1200 MHz. Wave 5 agrees with the LUT; v1.2.2 prescribes a different value. The `cipher-t43-envelope` memory's +14% tok/W lift claim was measured at the existing 1000 MHz operating point.

**Decision (no code change):**

Honor the measured-optimum LUT. Surface the discrepancy for adjudication; do NOT auto-modify the LUT to match the spec's named value. If +14% tok/W is the binding claim, the existing LUT is correct. If a stronger lock to 1200 MHz is the intent (sacrificing measured optimum for spec compliance), Step 4 or a future step can re-tune. Surfacing here, not mitigating.

The classifier observer's hint can still trigger VOLT engagement (Step 4 work); the locked clock value is independently chosen by `batch_to_mhz()`.

## E — Makefile dep update (+1 line per rule)

```
cipher_rt_marlin_actuator.o: ... cipher_rt_dispatch.h cipher_rt_classify_observer.h
    $(CC) ...
```

Forces rebuild when dispatch/observer headers change.

## F — Build + nm diff

| signal | value |
| --- | --- |
| build rc | 0 |
| warning count | 78 → 78 |
| md5 | `c41182e0` → `f63e1f2e` |
| Symbol count | 311 → 311 (changes internal to existing TU) |
| Symbols added | 0 |
| Symbols removed | 0 |
| Undef audit | clean |

## G — Runtime smoke + SC6 gate — PASS

**G.1 Loader:** rc=0; CLASSIFY + DISPATCH banners; LIVE=0.

**G.3 SC6 TinyLlama bit-identical:**
- Vanilla baseline: 33s; 7/7 PASS; `all4_fwd1_bit_identical=true`; `all4_fwd2_bit_identical_after_producer_death=true`
- CIPHER-injected: 33s; 7/7 PASS; same bit-identity

Both modes match. Marlin's hint-read is advisory at LIVE=0; runtime behavior unchanged from Step 2.

## H — CP 5.4 — 15/15 PASS byte-identical

## I — Commit + tag

`4f1a86abd34dd7ae3fca42af7a832eb5b429ee5c`, tag `week-3-step-3-marlin-hint-volt-lock`.

## Discipline notes

- VOLT discrepancy documented as adjudication item (no auto-mitigation).
- Substrate now reaches into Marlin's entry path; LIVE=0 keeps it dormant.
- Step 4 flips LIVE=1 — Marlin's hint-read becomes binding. Mistral-7B SC6 is the load-bearing gate.

## Anchors

- cipher_rt_phase4 HEAD: `71bf5523` → `4f1a86ab` (tag `week-3-step-3-marlin-hint-volt-lock`)
- Other trees: unchanged
- libcipher_rt.so md5: `c41182e0` → `f63e1f2e`

Week 3 progress: **3/5**. Step 4 unblocked.
