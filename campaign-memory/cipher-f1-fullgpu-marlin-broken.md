---
name: cipher-f1-fullgpu-marlin-broken
description: CIPHER finding F1 — shipped libcipher_rt c2c5d313 full-GPU Marlin produces degenerate decode; needs its own audit STEP
metadata: 
  node_type: memory
  type: project
  originSessionId: fc3a14bf-83b4-4034-8d55-c6111b810f83
---

**Finding F1**, surfaced by CP 5.3 STEP 2's A-tok investigation (2026-05-18).

The shipped `libcipher_rt.so` **`c2c5d313`** (campaign anchor, post-CP-2.5)
produces **degenerate output when Marlin INT4 is engaged on a real transformer
decode** — the full-GPU path (`grid=132`, CP 2.4 `PrimaryCtxGuard` pinning the
GEMM to the primary context). TinyLlama → repetition gibberish; Mistral-7B →
`the 197imi` then 122× token-0 `<unk>` (the NaN-logits→argmax(0) signature).

Reproduced **six ways**: TinyLlama and Mistral × {original A-tok run, fresh
re-run, B=8 batch, tenant-id unset}. Marlin engaged every time
(`MATMUL handled=13875`/`28125`). **Not** transient, **not** M=1-specific,
**not** tenant-id-dependent. Evidence: `cipher-fusion-evidence/cp_5_3/`
`cp53_atok_*_{fullgpu,fullgpu_rerun,fullgpu_b8,fullgpu_notenant}.*`,
`cp53_atok_investigate.runlog`; analysis in `CP_5_3_STEP_2_REPORT.md` §5.

Leading hypothesis (for F1's audit, NOT proven): CP 2.4's `PrimaryCtxGuard`
held across the GEMM launch makes the GEMM run on the primary context,
decoupled from the app's compute stream → cross-stream race. CP 5.3 STEP 2
scopes the guard to the quant phase so the GEMM runs on the app's stream — and
its decode is coherent. Consistent with CP 2.4's gate having measured tok/W /
tok/s (numbers a degenerate decode still produces) without a token-correctness
check.

**Why:** a blocker traced to a prior change ([[cipher-regression-discipline]]:
audit-first-in-source, as its own work — never reframe, never silently absorb).

**Records review (2026-05-18, `F1_TPW_RECORDS_REVIEW.md`):** the anchored
**3.617× tok/W** headline is **SUSPECT**. The CP 2.4 composed gate engaged the
F1 path (Marlin full-GPU on, green ctx active, `on_null_stream` accounting)
with **zero output-correctness gate** — `spec_varied_driver.py` records only
tok/s, never token IDs; all-on arm's n-gram `accept_rate=1.000` is a loop
signature. CP 4.8 Task B re-ran it on `c2c5d313` and "reproduced" 3.6166× —
but that just reproduced the timing-only blind spot. The F1 audit / CP 5.6
must carry **re-measurement scope** (on `dc804eb3`, with a correctness gate);
the 3.617× number must not be re-asserted in the investor one-pager / Ditlev
brief without a "throughput-only, correctness unverified" caveat.

**How to apply:** F1 is **NOT** a CP 5.3 deliverable — do not fold it into
STEP 2 or STEP 2B. It needs a dedicated regression-audit STEP/CP, now also
scoped to re-measure 3.617×. STEP 2's shipped `dc804eb3` fixes the
*partitioned* path only; the full-GPU path is still broken. Related:
[[cipher-phase5-scoped]], [[cipher-marlin-primary-ctx-pin]],
[[cipher-cp24-closed]].
