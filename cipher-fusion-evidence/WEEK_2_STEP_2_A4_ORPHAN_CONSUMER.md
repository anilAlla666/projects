# Week 2 Step 2 A4 — Orphan Consumer Inspection (`cipher_intercept_stats`)

**HEADLINE STATUS: H1 RECOMMENDED.** Consumer is **PURE-LOG**.

The sole call site (`cipher_runtime.cpp:123`) is inside `cipher_report()` — a user-invoked diagnostic stderr dump. All 6 counter fields are emitted via a single `fprintf` with two defensive guarded divisions. No threshold checks, no decision-gating, no downstream storage. A zero-counter stub produces truthful telemetry ("no F1 activity") for the cipher_rt_phase4 build (where the F1 hook layer is intentionally not installed per CP 2.5 D2(iii) — GOT patcher is the canonical interceptor instead).

**Date:** 2026-05-20
**Read-only diagnostic.** No source modifications.

---

## Step 1 — Call site inspection

**File:** `cipher-may13-evidence/src/cipher_runtime.cpp`
**Containing function:** `void cipher_report(void)` — diagnostic telemetry dumper, lines 107–145.
**Call site:** L123 (`const CipherInterceptStats* is = cipher_intercept_stats();`).

### Surrounding context (lines 107–145)

```c
// ---------------------------------------------------------------------------
// cipher_report
// ---------------------------------------------------------------------------

void cipher_report(void) {
    fprintf(stderr,
        "\n═══════════════════════════════════════════════════════\n"
        "  CIPHER Runtime Report — v%s\n"
        "═══════════════════════════════════════════════════════\n",
        CIPHER_VERSION_STR);

    cipher_green_ctx_report(&g_cipher.green_ctx);
    fprintf(stderr, "\n");
    cipher_l2_persist_report(&g_cipher.l2_persist);
    fprintf(stderr, "\n");
    cipher_liquid_state_report(&g_cipher.liquid);
    fprintf(stderr, "\n");
    cipher_telemetry_report(&g_cipher.telemetry);
    cipher_layer3_report();

    const CipherInterceptStats* is = cipher_intercept_stats();    // ← L123
    uint64_t avg_ns = is->total_intercepts > 0
        ? is->overhead_ns_sum / is->total_intercepts : 0;
    fprintf(stderr,
        "\n[F1] Intercept Stats\n"
        "  Total intercepts:  %lu\n"
        "  Substitutions:     %lu  (%.1f%%)\n"
        "  Passthroughs:      %lu\n"
        "  Deferred:          %lu\n"
        "  Avg overhead:      %lu ns\n"
        "  Max overhead:      %lu ns\n",
        is->total_intercepts,
        is->substitutions,
        is->total_intercepts > 0
            ? (double)is->substitutions * 100.0 / is->total_intercepts : 0.0,
        is->passthroughs,
        is->deferred,
        avg_ns,
        is->overhead_ns_max);

    fprintf(stderr,
        "═══════════════════════════════════════════════════════\n\n");
}
```

**Function purpose:** human-readable runtime telemetry. Stitches together output from `cipher_green_ctx_report`, `cipher_l2_persist_report`, `cipher_liquid_state_report`, `cipher_telemetry_report`, `cipher_layer3_report` — and appends the F1 intercept summary as the final block before the trailing rule.

**Call site characteristics:**
- Local pointer-receive variable `is` — never escapes the function scope.
- The function returns `void`. No struct is copied or forwarded elsewhere.
- The function is called manually (no `__attribute__((destructor))`, no signal handler, no atexit registration in this TU). It exists as a diagnostic surface a developer or a smoke test invokes.

---

## Step 2 — Per-field usage trace

`CipherInterceptStats` exposes 6 counter fields. Tracing each use in the call site:

| field | uses in this function | role |
| --- | --- | --- |
| `total_intercepts` | L124 guard, L128 `%lu` direct print, L136 guard, L137 ratio denominator | denominator for avg_ns + substitution_pct; printed directly; guarded against div-by-zero |
| `substitutions`     | L129 `%lu` direct print, L137 ratio numerator | numerator in `substitution * 100.0 / total`; printed directly |
| `passthroughs`      | L130 `%lu` direct print | direct print only |
| `deferred`          | L131 `%lu` direct print | direct print only |
| `overhead_ns_sum`   | L125 numerator in `sum / total` average computation | feeds `avg_ns`; printed via avg_ns |
| `overhead_ns_max`   | L133 `%lu` direct print | direct print only |

**Derived values:**
- `avg_ns` (L124–125) — division `overhead_ns_sum / total_intercepts`, guarded with `total_intercepts > 0` ternary returning 0 on zero-counter. Defensive against div-by-zero. Output channel: stderr only.
- `substitution_pct` (L136–137) — division `substitutions * 100.0 / total_intercepts`, guarded with `total_intercepts > 0` ternary returning `0.0` on zero-counter. Output channel: stderr only via `%.1f%%` format.

**No other usages of any field anywhere in `cipher_runtime.cpp`.** Verified by:

```bash
$ grep -n 'is->' cipher_runtime.cpp
# Only the lines shown above (124–141). No other escapes.
```

**No threshold comparisons.** No `if (is->X > N)` / `while (is->X < M)` / etc.
**No storage.** Nothing of the form `g_cipher.X = is->Y` or `cache->stats = *is`.
**No further function call.** Nothing of the form `report_to_dashboard(is)` or `audit_emit(*is)`.

---

## Step 3 — Consumer behavior classification

Per the rubric:

| category | match? | reasoning |
| --- | --- | --- |
| **PURE-LOG** | ✓ YES | All 6 fields land in one `fprintf(stderr, ...)` block. Two defensive guarded divisions produce 0 when counters are 0 — output remains well-formed and truthful. |
| PURE-REPORT | partial | The function is called `cipher_report` and the categorization rubric uses "PURE-REPORT" for "stored elsewhere but not used for control flow." Here the function is the report — there is no further storage step. The output flows to stderr. The label PURE-LOG matches more precisely because nothing leaves the function. |
| DECISION-INPUT | ✗ NO | Zero comparison/branch operators on any field. Zero downstream control flow uses any field. |

**Verdict: PURE-LOG.** Unambiguous.

The two ternary `total_intercepts > 0 ? ... : default` patterns are robust to zero-state — the guards exist precisely to handle "no F1 activity recorded" gracefully, which is the exact state cipher_rt_phase4 produces (no F1 hook layer installed; canonical interceptor is the GOT patcher).

---

## Step 4 — Handling recommendation

Per Step 3 classification (PURE-LOG): **H1 (5-LOC zero-return stub)** is correct.

The 5-LOC stub:

```c
// In cipher_rt_phase4/cipher_rt_may13_stubs.cpp (new file, ~30 LOC total for A4 stubs)
#include "may13/cipher_intercept.h"

const CipherInterceptStats* cipher_intercept_stats(void) {
    /* F1 hook layer is intentionally not installed in cipher_rt_phase4
     * (CP 2.5 D2(iii) — GOT patcher is the canonical interceptor instead;
     * see WEEK_2_STEP_2_V3_PREFLIGHT.md §4.1). Returning zero counters
     * is the truthful state: no F1 intercept events occurred. */
    static const CipherInterceptStats zero = {0};
    return &zero;
}
```

What `cipher_report()` will output on cipher_rt_phase4:

```
[F1] Intercept Stats
  Total intercepts:  0
  Substitutions:     0  (0.0%)
  Passthroughs:      0
  Deferred:          0
  Avg overhead:      0 ns
  Max overhead:      0 ns
```

This is *correct*. No F1 hooks installed → no intercepts recorded. The two guarded divisions correctly produce `0` and `0.0` when `total_intercepts == 0`. A human reading the report sees "the F1 hook layer is not active" — exactly what cipher_rt_phase4's substrate architecture is.

### Why H2 is not necessary

H2 (translate to cipher_rt_phase4 native counters from `cipher_rt_got_patch.c`) would only matter if the consumer used the values for decision-making — e.g., "if intercept rate < 0.5, fall back to passthrough". The audit shows zero such usage. The native GOT-patcher counters (call counts of `cublasGemmEx`, SDPA trampolines, etc.) measure a *different* mechanism. Routing them into the F1 stats struct would either:

- **Mislabel them** (the user reads "F1 Intercept Stats" but the numbers are actually F4.5/F4.6 GOT-patcher hits — a documentation-vs-behavior split is worse than honest zeroes), OR
- **Require renaming/restructuring the report** — out of scope for a 1-symbol stub fix.

H1's zero-counter truthful telemetry is the cleaner answer.

### Why H3 cascade is not necessary

H3 (cascade-exclude `cipher_runtime.cpp`) re-introduces a different stub need (`g_cipher` definition), shrinks closure by 2 files (loses `cipher_telemetry.cpp`), and removes the native `CipherRuntime` definition from scope. Same stub count, smaller scope but loses the F4/F5 init wiring that `cipher_runtime.cpp` provides. No upside given H1 is sufficient.

---

## Summary

| signal | value |
| --- | --- |
| Containing function | `cipher_report()` (diagnostic, returns void) |
| Consumer kind | PURE-LOG — single `fprintf(stderr, ...)` block |
| Decision-input fields | none |
| Storage of fields | none |
| Downstream propagation | none |
| Defensive guards on zero | yes (two `... > 0 ? ... : 0` ternaries) |
| **Recommended handling** | **H1 — 5-LOC zero stub** |
| Resulting telemetry | "F1 Intercept Stats: 0 intercepts" — truthful and unambiguous |

A4-closure status: ready to draft the Step 2 v3 port brief. **11-file scope, 1 stub (5 LOC), 3 .cu files, 0 FLAG static-inits, 0 cascading exclusions needed.**

---

## Discipline notes

- Read-only diagnostic. No source-tree changes.
- Decision is unambiguous (PURE-LOG with defensive guards) — no need to default to conservative handling.
- The CLAUDE.md context attached to this session (Stage 1-12 build history) is informational about a different snapshot of cipher-may13-evidence (April 28, op31-prod-fix pod, different from this audit's purpose) and does not change the analysis of this specific call site.

Awaiting confirmation to proceed with H1 + Step 2 v3 port brief.
