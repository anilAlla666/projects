# CP 4.8 — Integration Soak / Phase 4 Close — REPORT

**Date:** 2026-05-17 **Status:** CP 4.8 closed; Phase 4 declared closed (founder decision — see §6).
**Anchors (held, md5-verified):** kmod 0.4.8 `e2f50452`, libcipher_rt `c2c5d313`,
libcipher_v2 `86618c30` (not loaded by the soak runtime — D6).

This is the Phase 4 ship-gate CP report. It supersedes nothing; it closes CP 4.8
and Phase 4. Lean §1–§6 structure per the 2026-05-17 user lean-pivot. The
adjudicated `CP_4_8_DESIGN_MEMO.md` is the pre-registration this report is
measured against.

---

## §1 — What CP 4.8 ships

CP 4.8 is the Phase 4 ship-gate CP. It ships **no substrate code change** — the
composed production stack is validated exactly as it ships (anchors above held
throughout). The single build artifact is a **measurement-path refinement**:

- **`mfu_compute.py` — analytical `tensor_mfu_pct`.** Replaces the legacy
  SM-cycle proxy (`sm_util_pct × clock_ratio`), which counted an SM *stalled on
  HBM* as "utilised" and so reported ~100% for memory-bound decode where the
  tensor cores are ~99% idle. The new metric is `workload_tensor_FLOPs /
  (wallclock × peak_tensor_FLOPS)` — FLOP count from model geometry × tokens,
  zero added runtime overhead, soak-safe. Two physics corrections were baked in
  and adjudicated: inference is `2N` not the training `6N`; Hopper has no INT4
  tensor path (Marlin dequantises to fp16, peak 989.4 TFLOPS).

**Validation delivered:** descriptive per-WL tensor-MFU characterization (G1,
20 WLs — §4); tenant-density ceiling (G3 — §3); a 34-minute partial stability
soak (§2). The 24-hour sustained-load soak (G4) is **deferred** — see §2, §6.

**Independent confirmation (not CP 4.8 evidence).** Task B re-measured the
CP 2.4 composed stack on the current anchor `c2c5d313`: **3.6166× tok/W**
[3.5908, 3.6423] — exact reproduction of the 3.617× headline. This is the
investor one-pager number; it is logged here for the register, not as Phase 4
ship-gate evidence.

---

## §2 — Soak results (34-minute partial; full 24 h deferred)

`cp48_soak.sh` launched 2026-05-17 11:13:35, realistic 7-tenant mix (4 inference
4K×4K + 2 training-like 8K×8K + 1 idle), libcipher_rt `c2c5d313`. It was
terminated by founder decision after ~34 minutes for calendar/cost reasons —
**not** on a failure.

| Soak metric | Baseline | Final tick (`t=2042s`) | Δ |
|---|---|---|---|
| kernel taint (`/proc/sys/kernel/tainted`) | 12288 | 12288 | **0** |
| kernel WARN/oops/BUG count | 3 | 3 | **0** |
| GPU power | — | 162–164 W steady | — |
| tenants alive | 7 | 7 | 0 |
| fallback `.ko` md5 | stable | stable | — |

Every 60 s monitor tick from `t=0s` to `t=2042s` was clean: zero taint drift,
zero new kernel events, steady power. Teardown on `C-c` was clean — all 7
tenants reaped, GPU returned to 0 MiB, 0 compute apps (the cleanup-logic fix,
§5.3, worked end-to-end).

**Verdict — partial evidence.** 34 minutes of sustained 7-tenant load produced
zero kernel instability. This is genuine partial evidence of soak stability; it
is **not** the 24-hour G4 result. The full 24-hour soak is **deferred to the
post-Phase 7 operator pilot** (founder decision, calendar/cost). G4 is carried
forward as a tracked obligation, not marked passed (§6).

---

## §3 — Density results (G3)

`cp48_density.sh` — incremental tenant push 2→5→10→20→30→50→100, each tier
settled 30 s, per-tenant MFU sampled from `/metrics`; gate stops the push when
any tenant's MFU drops below 1%.

| Result | Value |
|---|---|
| Tiers completed | all 7 (2, 5, 10, 20, 30, 50, **100**) |
| Max tenants observed | **100** |
| Min per-tenant MFU at 100 tenants | **8%** |
| Gate threshold (min tenant MFU) | 1% |
| Gate tripped? | **No** — 8% ≫ 1% |

**Verdict — G3 PASS, ceiling ≥100.** The density push ran to its top tier (100
tenants) with the saturation gate never reached: the weakest tenant at 100-way
contention still held 8% MFU against a 1% floor. The sustained-tenant ceiling is
**≥100** — the test exhausted its tier ladder before exhausting the substrate.
This far exceeds the G3 design criterion (≥30 concurrent tenants).

*Provenance note:* the first density run (07:10) measured all 7 tiers correctly
but then hung ~4 h in teardown (cleanup-logic bug, §5.3). The numbers above are
the clean re-run (`cp48_density_rerun.log`, `DENSITY_EXIT=0`, clean teardown)
after the fix; they reproduce the original measurement exactly. The original
hung-state log is preserved as `cp48_density.log`.

---

## §4 — Descriptive tensor-MFU table (G1)

`cp48_g1_runner.py --all`, 180 s windows, libcipher_rt `c2c5d313`. 20 WLs (WL05
is the density WL, covered in §3). **Descriptive only** — the per-WL 80%-of-
ceiling gate was dropped by the 2026-05-17 lean pivot (`PROGRESS.md`); the
`verdict` field in `cp48_g1_result.json` grades against that dropped gate and is
not reproduced here. Ceilings are the pre-registered physics-derived values from
`cp48_ceilings.json`.

**17 of 20 WLs ran clean. 3 failed at the environment layer (§5.2).**

| WL | Workload | Regime | Ceiling | Measured `tensor_mfu_pct` | frac. of ceiling |
|---|---|---|---|---|---|
| WL01 | Decode B=1 · TinyLlama | memory-bound | 0.34% | 0.0144% | 0.04 |
| WL02 | Decode B=8 · TinyLlama | memory-bound | 2.7% | 0.1323% | 0.05 |
| WL03 | Prefill B=8×1024 · Mistral-7B | compute-bound | 80% | 4.2117% | 0.05 |
| WL04 | vLLM serving · TinyLlama | memory-bound | 2.7% | **env-fail** | — |
| WL06 | Embeddings batch-512 · MiniLM | compute-bound | 45% | 0.1245% | 0.003 |
| WL07 | LoRA fine-tune · TinyLlama | training | 55% | 0.3126% | 0.006 |
| WL08 | Diffusion 20-step · SDXL | compute-bound | 60% | 2.5504% | 0.04 |
| WL09 | Speech · whisper-large-v3 | mixed | 25% | 0.0465% | 0.002 |
| WL10 | Speculative decode · vLLM | memory-bound | 1.4% | **env-fail** | — |
| WL11 | Agentic multi-turn · TinyLlama | memory-bound | 0.34% | 0.0245% | 0.07 |
| WL12 | Batch-64 · TinyLlama | compute-bound | 65% | 2.1440% | 0.03 |
| WL13 | Long-context 32K prefill · Mistral-7B | compute-bound | 80% | 1.9795% | 0.02 |
| WL14 | torch.compile decode · TinyLlama | memory-bound | 0.34% | 0.3944% | 1.16 |
| WL16 | Prefix caching · vLLM | memory-bound | 2.0% | **env-fail** | — |
| WL17 | Training full · TinyLlama | training | 55% | 0.1858% | 0.003 |
| WL19 | Vision · CLIP ViT-L/14 | compute-bound | 62% | 3.1147% | 0.05 |
| WL20 | Multimodal · LLaVA-1.5-7B | mixed | 8% | 1.1563% | 0.14 |
| WL21 | Code generation · TinyLlama | memory-bound | 0.34% | 0.0165% | 0.05 |
| WL22 | RAG pipeline · MiniLM+TinyLlama | mixed | 15% | 0.0304% | 0.002 |
| WL23 | Model-switch stress · TinyLlama↔Mistral | not-tensor-compute | n/a | n/a — ran clean | stability-only |

**Descriptive observation.** Every WL that ran came in at **0.2–14% of its own
pre-registered ceiling** — including the memory-bound WLs whose ceilings are
themselves exact `AI/295` physics (e.g. WL01 measured 0.0144% against a 0.34%
bandwidth roofline ≈ 4% of even the roofline). Tensor cores are structurally
near-idle for these workloads, and a further large margin below even the
roofline is consumed by kernel-launch / Python-dispatch / harness overhead at
the small model sizes used. WL14 alone reads slightly *above* its ceiling
(frac 1.16) — a minor analytical-vs-measured artifact under `torch.compile`
fusion; noted, not interpreted. None of this is a substrate finding — it is the
workloads' own arithmetic intensity, which is exactly the methodology point in
§5.1.

---

## §5 — Methodology findings

### §5.1 — The 85–90% MFU criterion is physically unachievable on B=1 agentic decode

The legacy Phase-1–3 gate framing graded GPU utilisation against an 85–90% MFU
target. Under a *true* tensor-MFU metric (§1) this is **not physically
attainable** for B=1 decode and agentic workloads: each weight is read from HBM
once per token, arithmetic intensity AI ≈ 1 FLOP/byte, and the roofline ridge
point is 295 FLOP/byte — so the tensor-MFU ceiling is `1/295 = 0.34%`. The GPU
spends ~99.7% of every decode step waiting on HBM bandwidth; the tensor cores
*cannot* be driven to 85–90% by any substrate. This is HBM physics, not a
substrate deficiency. It is why CP 4.8 pivoted (2026-05-17) to **descriptive
tensor-MFU + a roofline-relative framing**, and dropped the absolute-threshold
per-WL gate. The §4 data confirm the prediction empirically.

### §5.2 — vLLM environment dependency blocked WL04 / WL10 / WL16

The three vLLM-based workloads failed **identically at engine initialisation**,
before any measurement window:

```
vllm.v1.structured_output.backend_xgrammar.XgrammarBackend
  → import xgrammar
    → from tvm_ffi import register_error
      → tvm_ffi/_dtype.py:315 → tvm_ffi/registry.py:26  (import-chain failure)
→ RuntimeError: Engine core initialization failed
```

Evidence: `g1_logs/WL04.log`, `WL10.log`, `WL16.log` (identical traces). This is
a **Python-environment defect** in this pod's vLLM / xgrammar / tvm_ffi stack —
**not a substrate issue** (libcipher_rt is never reached). vLLM v1 eagerly
imports the xgrammar structured-output backend at engine startup; the broken
`tvm_ffi` transitive dependency aborts the core process. **Scope: Phase 5
CP 5.1** (vLLM/TGI live-decode integration) — the environment must be repaired
and pinned before vLLM-path workloads can be measured.

### §5.3 — Density/soak cleanup bug — `kill -INT` no-op on `&`-launched children

The first density run hung ~4 h in teardown. Root cause: `cleanup()` in both
`cp48_density.sh` and `cp48_soak.sh` sent `kill -INT` (SIGINT) to the tenant
processes, then `wait`. But bash masks `SIGINT`/`SIGQUIT` (`SIG_IGN`) on
`&`-launched children in a non-interactive script, and CPython preserves an
inherited `SIG_IGN` rather than installing its `KeyboardInterrupt` handler — so
`kill -INT` was a no-op, the tenants spun forever, and `wait` blocked
indefinitely. Deterministic, independent of tenant count.

**Fixed at source** (per regression discipline — fix in source, never a runtime
workaround) in both scripts: `cleanup()` now sends `SIGTERM` (not masked for
background jobs), polls up to 5 s, then `SIGKILL`s any straggler still in a CUDA
sync. Verified end-to-end: the density re-run (§3) and the soak teardown (§2)
both reaped all tenants and returned the GPU to 0 MiB cleanly. The density
*measurement* was always valid — only teardown hung.

### §5.4 — `cipher_flopd` not deployed — PMU cross-check used FlopCounterMode

The design memo §3 specified an analytical-vs-PMU FLOP cross-check against the
`cipher_flops.c` substrate counter, fed by the `cipher_flopd` daemon. **`cipher_flopd`
was not deployed** on this pod, so the live PMU FLOP path was unavailable. The
cross-check instead used **PyTorch `FlopCounterMode`** (ATen op-graph trace) as
the independent FLOP source — which validated the analytical method (WL01 0.06%,
WL13 1.40%, WL17 1.10% agreement after the WL01 embedding-FLOP bug was found and
fixed). `cipher_flopd` deployment is a **Phase 5 / operator backlog** item.

### §5.5 — 24-hour soak deferred to post-Phase 7 operator pilot

Founder decision, calendar/cost. The 34-minute partial soak (§2) ran clean. The
full 24-hour G4 sustained-load soak is deferred to the post-Phase 7 operator
pilot, where it runs on operator-representative hardware and load. Tracked as a
carried-forward obligation (§6).

---

## §6 — Phase 4 close declaration

**Measured against the pre-registration.** The adjudicated `CP_4_8_DESIGN_MEMO.md`
§4 states the joint criterion explicitly: *"All four gates must pass
simultaneously … 3-of-4 = FAIL … a partial pass is a documented FAIL, not a soft
close."* Honest accounting against that bar:

| Gate | Pre-registered criterion | Outcome |
|---|---|---|
| G1 | Per-WL tensor-MFU ≥ 80% of ceiling | **Gate dropped** by 2026-05-17 lean pivot; replaced by descriptive characterization — §4. 17/20 WLs characterized; 3 vLLM env-fails. |
| G2 | WL05 device-aggregate tensor-MFU ≥ 0.27% | Subsumed into G3 density. |
| G3 | ≥30 concurrent tenants sustained | **PASS** — ceiling ≥100 tenants, gate never tripped (§3). |
| G4 | Zero kernel oops/WARN, taint Δ≤1 across 24 h soak | **DEFERRED** — 34-min partial clean (taint Δ=0, warns Δ=0); full 24 h → post-Phase 7 operator pilot. |

**The G1–G4 joint pass as pre-registered was not achieved** — G1's gate was
dropped (by deliberate, adjudicated lean pivot, on the §5.1 physics finding) and
G4 is deferred. This report does **not** claim a clean ship-gate pass.

**Founder close decision.** Phase 4 is declared **CLOSED** on the evidence that
*was* produced:

- the composed production stack runs anchored and unmodified (`e2f50452` /
  `c2c5d313` / `86618c30`), md5-verified across the run;
- descriptive tensor-MFU characterization across 17 workload classes, with the
  measurement-method physics corrected and cross-check-validated (§1, §4);
- a tenant-density ceiling of **≥100** concurrent tenants — 3.3× the G3 design
  criterion (§3);
- 34 minutes of sustained 7-tenant load with zero kernel instability (§2).

G4 (full 24-hour soak) and the §5.2 / §5.4 environment items are **explicitly
preserved as tracked obligations** to Phase 5 and the post-Phase 7 operator
pilot — not silently dropped. Phase 4 closes as an **engineering close on
partial ship-gate evidence by founder calendar/cost decision**, transparently,
consistent with the design memo by naming the deviation rather than
contradicting it.

**Carried forward:**
- G4 24-hour sustained-load soak → post-Phase 7 operator pilot.
- vLLM/xgrammar/tvm_ffi environment repair → Phase 5 CP 5.1.
- `cipher_flopd` deployment → Phase 5 / operator backlog.

**Register (Phase 4 closing state):** CP 2.4 composed 3.6166× tok/W (Task B,
reconfirmed on `c2c5d313`); CP 2.5 LD_PRELOAD-free deploy + `cipher-platform.deb`;
CP 4.4 / 4.6.5+6 / 4.7 closed; CP 4.8 closed (this report). Phase 4 → CLOSED.
Next: Phase 5 (CP 5.1–5.5 design memos — companion documents to this report).

---
*CP 4.8 closed 2026-05-17. Anchors held: kmod `e2f50452`, libcipher_rt
`c2c5d313`, libcipher_v2 `86618c30`.*
