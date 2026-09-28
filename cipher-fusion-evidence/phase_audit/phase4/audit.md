# CIPHER Phase 4 — Canonical Checkpoint Evidence Audit (CP 4.1–CP 4.3)

**Auditor scope:** read-only evidence audit of the three CANONICAL Phase 4 checkpoints only.
**Date of audit:** 2026-05-15
**Method:** file reads + `md5sum` / `wc -l` / `ls -l` / `grep` only. No code written, no workloads run.
**Out of scope (audited separately):** the T4.x work series (T4.2 contention/GREEN_CTX, T4.3 VOLT/DVFS, T4.5 .symver matmul substrate + Marlin, T4.6 KV dedup). T4.x is referenced here only to test equivalence claims.

---

## 1. CP-by-CP status table

| CP | One-line scope | STATUS |
|----|----------------|--------|
| CP 4.1 | L2 weight pinning, 1.3–1.6× tok/W on Mistral-7B decode | **PARTIAL** |
| CP 4.2 | TMA-based memory substitution, 1.5–2× lift on memory-bound elementwise | **NOT DONE** |
| CP 4.3 | Per-shape persistent kernel routing, 1.2–1.4× lift on common matmul shapes | **PARTIAL** |

---

## 2. Per-CP evidence treatment

### CP 4.1 — L2 weight pinning → 1.3–1.6× tok/W on Mistral-7B decode — **PARTIAL**

**What the canonical gate requires:** an L2-weight-pinning mechanism that delivers a *measured 1.3–1.6× tok/W on Mistral-7B decode*, attributed to that mechanism.

**Evidence that exists on disk:**

- `cipher-may13-evidence/src/cipher_l2_persist.cu` — the L2-weight-pinning mechanism.
  - md5 `c2fcdce1464036b2b2e77fae8317284d`, 238 lines, mtime 2026-04-27 (file stamp 2026-05-13 04:27 = copy time of the may13 evidence snapshot; content dated 2026-04-28 per CLAUDE.md).
  - Implements `cipher_l2_persist_register/apply/unpin/reset/report` using `cudaAccessPolicyWindow` with `hitProp = cudaAccessPropertyPersisting`, `missProp = cudaAccessPropertyStreaming`, `hitRatio = 1.0`. Budget-capped against `CIPHER_L2_PERSIST_MAX_BYTES`. This is a real, correct L2-pinning implementation.
- `cipher-may13-evidence/src/cipher_persist_engine.cpp` (md5 `b1a5cc057afe7164907eb51c29ebbc77`, 13177 B) — the production wiring: 64-region table, fractional-knapsack budget vs `silicon->l2_persist_max` (31.2 MB on this H100), per-launch hot-path actuation in `cuLaunchKernelEx`.
- may13 `CLAUDE.md` "Stage 3 — Persistence Engine": firing verified — "M4 run = 48,848 window_hits / 48,797 launches in 10 s." The mechanism *fires*.
- `cipher-may13-evidence/cipher_metrics_receipts/d1_single_60s.json`, `d2_n4_60s.json` — persist-engine test receipts (57 KB / 50 KB).
- `cipher-may13-evidence/action_b_l2.py` (md5 `0734794cf9dc98e347c43a2b3b41f96d`) + `action_b_results.json` (77 lines) + `action_b_log.txt` — an explicit L2-cache cold-vs-hot GEMM microbench on the four Mistral linear shapes.

**Measured numbers (action_b_results.json):** cold-vs-hot *GEMM microbench* speedup:
- fp16: q/o_proj 1.37×, k/v_proj 1.08×, gate/up_proj 1.17×, down_proj 1.16×.
- INT4: q/o_proj 1.61×, k/v_proj 1.68×, gate/up_proj 1.21×, down_proj 1.48×.

**Why this is PARTIAL, not SHIPPED:**

1. **The measured numbers are not the canonical gate.** `action_b_l2.py` measures isolated GEMM cold-vs-hot cache-bandwidth speedup (a microbench of L2 residency), not *tok/W on Mistral-7B decode*. The 1.08–1.68× numbers happen to straddle the "1.3–1.6×" band, but they measure a different quantity on a different workload. The canonical gate is an end-to-end tok/W lift on Mistral-7B decode.
2. **No end-to-end Mistral-7B decode tok/W measurement attributes any lift to L2 pinning.** may13 `BUILD_STATE`/`CLAUDE.md` explicitly states Stage 3 (L2 persist) M4 ΔTFLOPS = **0%** ("compute-bound at 700 W cap — wrong probe; correctness only").
3. The Mistral-7B tok/W numbers that *do* exist in may13 CLAUDE.md (1.79× tok/W at B=8) are attributed to **Marlin INT4 + fused RMSNorm/SiLU**, explicitly *not* L2 weight pinning. The 1.62× INT4 figure is Marlin's. L2 pinning's isolated contribution to end-to-end Mistral decode tok/W was never measured.

**Gap to close:** an A/B Mistral-7B decode run with the persist engine the *only* variable, reporting tok/W, showing 1.3–1.6×. No such capture exists on disk.

**Reproducibility:** `action_b_l2.py` is re-runnable (`python action_b_l2.py`) but it does not test the canonical gate. The persist-engine tests (`tests/test_persist_engine.py`) are re-runnable and verify firing/correctness, not tok/W.

**Verdict: PARTIAL.** Mechanism ships and is verified firing. The canonical numeric gate — tok/W lift on Mistral-7B decode attributed to L2 pinning — was never measured.

---

### CP 4.2 — TMA-based memory substitution → 1.5–2× lift on memory-bound elementwise — **NOT DONE**

**What the canonical gate requires:** a TMA (Tensor Memory Accelerator) based memory-substitution path, delivering 1.5–2× on memory-bound elementwise kernels.

**Evidence:** none. The opposite — explicit, dated evidence that TMA is *not* implemented:

- `PHASE_4_DEPTH_AUDIT.md` (2026-05-13), "TMA descriptors" section, verbatim:
  > `grep cuTensorMap on cuda.h: empty` … "**TMA (Tensor Memory Accelerator) is NOT exposed in this `cuda.h` version.** It's accessed through PTX/SASS directly, not via Driver API. cipher_rt's substitute_v2 emits PTX via NVRTC; it does NOT emit TMA descriptor setup. **Hopper-specific FLOPS lift gap.**"
- Same doc, Hopper-feature table: "TMA bulk async copy — **NOT** [used]"; "TMA descriptors — NOT mentioned [in src/]".
- `PHASE_4_ARCHITECTURE.md` lists "TMA descriptors + thread block clusters" only as a *future* "depth win 4" for P4.5, not as shipped work.
- `grep -rln "TMA|cuTensorMap|tensor memory accelerator"` across `cipher-phase4-evidence/` and `cipher-phase4-cutover-evidence/`: **0 hits**.

**Equivalence check:** none of the T4.x work touches TMA. T4.5's substrate emits PTX via NVRTC but explicitly does not emit TMA descriptor setup. There is no substitute deliverable to argue.

**Verdict: NOT DONE.** No TMA code, no TMA descriptor emission, no memory-bound elementwise lift measurement. The project's own depth audit documents TMA as an unaddressed gap.

---

### CP 4.3 — Per-shape persistent kernel routing → 1.2–1.4× lift on common matmul shapes — **PARTIAL**

**What the canonical gate requires:** per-shape *persistent kernel* routing, delivering 1.2–1.4× on common matmul shapes. ("Persistent kernel" in standard usage = a resident kernel that survives across launches via the persistent-thread-block / cooperative-groups pattern.)

**Evidence that exists on disk:**

- `cipher-may13-evidence/src/cipher_kernel_table.cpp` — md5 `0128be6650f8269ca109c3c4bbd3a060`, 317 lines. Per `STAGE14_KERNEL_TABLE.md` (2026-04-30): a kernel *name-recognition / classification* table. It **observes and categorizes** every distinct CUfunction (GEMM/RMSNorm/SiLU/FlashAttn/etc.) — it does **not route or substitute**. STAGE14 documents the blocker: `cuFuncGetParamInfo` returns 0 params for all 62 PyTorch kernels, so per-shape substitution from the table was **not** built (Tasks 2–5 explicitly not shipped).
- T4.5 `.symver` matmul-routing substrate — `PHASE_4_T4_5_REPORT.md`:
  - `cipher-phase4-evidence/t4_5_substrate/perf_summary.json`: substrate routing verified byte-identical on TinyLlama, perf delta **−1.69% ± 1.17%** (95% CI [−4.0%, +0.6%]) — i.e. the substrate itself is a near-zero-overhead passthrough, no lift.
  - Marlin INT4 actuator as the first per-shape-gated dispatch (gate `N≥1024 && K≥1024`, ~45% of matmuls routed). `cipher-phase4-evidence/t4_5_marlin/summary.json`: on TinyLlama B=1 decode, tok/s **−32.4%**, tok/W **−26.9%** — a regression.
- may13 `CLAUDE.md` "Phase 2 — Marlin": Mistral-7B B=8 tok/s 1.38× / tok/W 1.79× — but this is a *prior-session (2026-04-28) cipher-may13 measurement*, predating the T4.5 substrate, and was not re-validated under T4.5's measurement harness (the T4.5 report explicitly flags this as deferred next-session work).

**Why this is PARTIAL, not SHIPPED and not SUPERSEDED:**

1. **"Persistent kernel" framing does not fit.** Neither `cipher_kernel_table.cpp` (observation only) nor the T4.5 substrate + Marlin (shape-gated INT4 GEMM *dispatch*) is a persistent-kernel pattern. The closest real work is per-shape *routing* (which exists), but not *persistent kernel* routing.
2. **The numeric gate (1.2–1.4× on common matmul shapes) is met only in one prior-session config.** The 1.38× Mistral-7B B=8 number exists (may13 CLAUDE.md) but predates and was not reproduced under the canonical/T4.5 measurement. Under the actual T4.5 measurement (TinyLlama B=1) Marlin *regresses* −32%. The substrate itself delivers −1.69% (no lift).
3. **Equivalence to the canonical CP cannot be fully argued.** T4.5+Marlin is the only borderline candidate. It is genuinely per-shape routing of matmuls, and the may13 1.38× B=8 figure lands inside the 1.2–1.4× band — but (a) it is INT4-quantization dispatch, not a persistent-kernel mechanism, and (b) the lift number is from a superseded prior session, not the current substrate's own measurement, which shows no lift / regression. Honest accounting forbids SHIPPED or SUPERSEDED.

**Gaps to close:** (a) reframe — if the project means "per-shape matmul routing," T4.5 substrate is the substrate, but it must be re-measured to show 1.2–1.4× on common matmul shapes; (b) the per-shape lift must be demonstrated under the T4.5 harness on Marlin's validated regime (Mistral-7B B≥8), which the T4.5 report itself flags as deferred. No persistent-kernel implementation exists at all.

**Reproducibility:** T4.5 pair runs under `cipher-phase4-evidence/t4_5_substrate/` and `t4_5_marlin/` are re-runnable per `PHASE_4_T4_5_REPORT.md`; they reproduce the −1.69% substrate / −32% Marlin-TinyLlama numbers, not a 1.2–1.4× lift.

**Verdict: PARTIAL.** Per-shape matmul *routing* exists (T4.5 substrate, byte-correct, zero-overhead; Marlin as first actuator). But it is not a "persistent kernel" mechanism, and the 1.2–1.4× lift is met only in a superseded prior-session config (Mistral B=8, 1.38×), not under the current substrate's own measurement.

---

## 3. Honest accounting — drift summary

Canonical Phase 4 targets **memory-system optimizations**: L2 weight pinning (CP 4.1), TMA memory substitution (CP 4.2), per-shape persistent kernels (CP 4.3).

The actual Phase 4 work (the T4.x series) targeted **different problems**: T4.2 = SM/compute partitioning (GREEN_CTX), T4.3 = DVFS clock-lock (VOLT), T4.5 = a `.symver` matmul-routing substrate + Marlin INT4 quantization, T4.6 = KV-cache dedup. These T4.x deliverables are real and independently audited — but they are **not equivalent substitutes** for the canonical CPs:

- CP 4.1's L2-pinning mechanism *does* exist (`cipher_l2_persist.cu`, `cipher_persist_engine.cpp`) and predates the canonical-plan re-framing — but its canonical tok/W gate on Mistral-7B decode was never measured in isolation. PARTIAL.
- CP 4.2 (TMA) is the cleanest miss: the project's own `PHASE_4_DEPTH_AUDIT.md` documents TMA as not exposed, not emitted, not used. NOT DONE.
- CP 4.3's only borderline equivalent is T4.5+Marlin, which is per-shape matmul *routing* (not persistent kernels) and shows no lift / a regression under its own current measurement; the 1.38× number is a superseded prior-session figure. PARTIAL.

**Single most important honest finding:** Canonical Phase 4 is the phase where the project drifted hardest. Of the three memory-system checkpoints, exactly **zero shipped to gate**: one mechanism (L2 pinning) is built and fires but was never measured against its canonical tok/W gate (PARTIAL), one (TMA) was never started and is documented by the project itself as an open Hopper gap (NOT DONE), and one (persistent-kernel routing) exists only as per-shape matmul *routing* whose lift is unproven under current measurement (PARTIAL). The substantial T4.x work that consumed Phase 4's calendar time solved compute-partitioning, DVFS, matmul-substrate and KV-dedup problems — all real, none of them the canonical Phase 4 memory-system checkpoints.
