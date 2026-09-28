# CP 4.4 — L2 persistent weight pin — build progress

Atomic STEP, 8 items. Migrate the canonical L2-weight-pin actuator into
`cipher_rt_phase4`, wire into the cuBLAS GEMM dispatch path, validate lift on
top of the CP 2.4 composed stack. This file is the durable cross-session
checkpoint — `CP_4_4_REPORT.md` is the report at close.

Anchors: kmod 0.4.8 (`E427CAFA4E94D548233DC7A` / `e2f50452`), libcipher_v2
`86618c30`, libcipher_rt `c2c5d313` (rollback point). New libcipher_rt anchor
will result on close (links `cipher_rt_l2_persist.o`).

## Status — CP 4.4 CLOSED 2026-05-16 (memo, path 6.2)

- **Item 1 — design memo — DONE.** `CP_4_4_DESIGN_MEMO.md` (canonical artifact).
- **§6 adjudicated — path 6.2 (skip the build).** Bound is decisive.
- **Items 2–8 — NOT EXECUTED.** No code, no build, no measurement.
- Closure note: memo §9. Anchors unchanged — no rebuild, no new module.
  L2-weight-pin actuator NOT in `cipher_rt_phase4`; deferred indefinitely.

## The §6 decision blocking the build

The working-set bound (memo §1) is decisive: the H100 persisting-L2 budget is
**31.25 MiB** (hardware cap, live-queried), the B=1-decode weight working set
is **3.7 GB (Mistral INT4) – 16 GB (Llama fp16)** — so `S/W` = 0.2–0.9 %.
Predicted tok/s lift **+0.1 % to +0.6 %** — a measured null inside n=5 noise.
The canonical actuator was an LNN lever (3 MB hot set, fits L2); the 7B
transformer-decode regime is 100–460× over budget.

Three adjudication paths in memo §6:
- **6.1 build as characterized-null** (recommended) — ~1 week, ships the
  module, closes the question with a measured data-point.
- **6.2 skip the build** — bound is sufficient; frees the week for CP 4.7.
- **6.3 re-scope** to a small-`W` regime where the lever earns.

Secondary (if 6.1/6.3): D1 Marlin hook point (shim sees fp16 B, Marlin reads
its own packed INT4 buffer → pin inside the Marlin actuator); D2 Llama fp16
arm in/out; D3 canonical `apply()` singular-window bug → per-GEMM window-set.

## Session log

### 2026-05-16 — STEP started. Item 1 (design memo) done; held at §6.
Bound computed pre-build; canonical `cipher_l2_persist.cu` located + read;
matmul dispatch substrate + inject init chain surveyed for the migration plan.
Awaiting user adjudication of §6 before item 2.

### 2026-05-16 (cont.) — §6 adjudicated path 6.2; CP 4.4 CLOSED at memo.
User selected 6.2 (skip the build) — the working-set bound (Mistral-7B INT4
118× over the 31.25 MiB L2 persistent budget, Llama-3.1-8B fp16 524× over) is
decisive; no measurement adds information. Closure note appended to memo §9.
Items 2–8 not executed. Anchors unchanged. STEP COMPLETE.
