# CP 4.7 — kernel-fusion throughput lever — build progress

Atomic STEP, 8 items. Migrate the canonical kernel-fusion actuator into
`cipher_rt_phase4`, wire into the cuBLAS GEMM dispatch path, validate lift on
the CP 2.4 composed stack. Durable cross-session checkpoint;
`CP_4_7_REPORT.md` is the report at close (if a build proceeds).

Anchors: kmod 0.4.8 (`E427CAFA4E94D548233DC7A` / `e2f50452`), libcipher_v2
`86618c30`, libcipher_rt `c2c5d313` (rollback point).

## Status — CP 4.7 CLOSED 2026-05-17 (memo, path 7.2)

- **Item 1 — design memo — DONE.** `CP_4_7_DESIGN_MEMO.md` (canonical artifact).
- **Launch-count spot-check — DONE** (authorized concurrent characterization):
  `cp47_launch_probe.py` / `cp47_launch_probe_result.json` /
  `cp47_launch_probe.runlog`.
- **§6 adjudicated — path 7.2 (close at memo).** Bound decisive + may13-confirmed.
- **Items 2–8 — NOT EXECUTED.** No build, no migration. Closure note: memo §9.
- Anchors unchanged — no rebuild. Fusion actuator NOT in `cipher_rt_phase4`.

## The bound (memo §1–§3)

Measured on c2c5d313: **2,789 kernel launches/token** (Mistral-7B B=1 decode) —
225 GEMM (8 %), 32 SDPA (1 %), ≈2,532 elementwise/reduce (91 %). The cuBLAS
GEMM dispatch hook the CP scope specifies reaches only the 8 % GEMM surface;
bias-free Mistral/Llama projections prune that to ≈32 launches (1.1 %) — the
gate_proj SiLU epilogue. Direct may13 evidence: canonical fusion-only measured
**B=1 +0.88 %, B=8 −0.23 %** — characterized-null. Predicted CP 4.7 lift
+0.1–0.9 %, inside n=5 noise.

**Memo recommends path 7.2 — close CP 4.7 at memo.** Same pattern as CP 4.4.
Secondary if 7.1/7.3: D1 Marlin epilogue variant, D2 Llama fp16 arm.

## Session log

### 2026-05-16 — STEP started. Item 1 (design memo) done; held at §6.
Launch-count spot-check run on c2c5d313 (2,789 launches/token). Canonical
fusion actuator + may13 driver-fusion discovery doc audited; matmul dispatch
substrate surveyed for the hook-reachability bound. Awaiting §6 adjudication.

### 2026-05-17 — §6 adjudicated path 7.2; CP 4.7 CLOSED at memo.
User selected 7.2 (close at memo) — two independent lines of evidence
(measured 1.1% hook-reachable surface; may13 +0.88%/−0.23% direct measurement)
converge on characterized-null; a build would be documentation theater.
Closure note appended memo §9. Items 2–8 not executed. Anchors unchanged.
STEP COMPLETE. Phase 4: only 4.8 (integration soak) remains.
