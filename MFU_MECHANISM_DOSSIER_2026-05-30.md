# CIPHER MFU MECHANISM DOSSIER — consolidated, cited-from-evidence

**Date:** 2026-05-30. **Type:** READ-ONLY inventory. Nothing built, no `.so` change, nothing committed,
anchors unchanged. Every number cited to a file:line / commit / result-file per Mem #10. Where a recalled
number is not in a result-file, it is marked **NOT FOUND IN EVIDENCE** rather than recalled.
**Sources:** first-hand reads of `cipher_rt_phase4` + a 7-reader parallel sweep of `cipher-may13-evidence`,
`cipher-fusion-evidence`, `cipher_rt_phase4`, `cipher_kmod`.

---

## 0. HEADLINE — recall vs. evidence (the load-bearing finding)

Three pillars of the recalled framing are **not corroborated by the current evidence**:

1. **The additive "FFN +30 / L2 +6 / fusion +7 / NCCL +8" decomposition does NOT exist in evidence.**
   `CIPHER_REENGINEERING_PLAN.md:988-993` carries a *different* 4-lever decomposition (Marlin INT4
   already-baseline, TMA+clusters post-CP-5.5, L2-pinning v1.5, L2-prefetch partial-v1) summing ~17-28
   pts above a **67% baseline**. The plan explicitly states **"85% gate is the right product target, but
   does NOT land in v1 window"** (`:986`) and **"no live claim of universal 85% MFU remains; per-WL
   framing is the honest claim"** (`:48`, `:1159`).

2. **"85% is a MULTI-GPU (8+ H100) cluster metric" is only partially in evidence — and not as recalled.**
   What the plan actually says: **three** NCCL ops are deferred to **v2 on hardware grounds** —
   `NCCL_P2P` (Op 33), `OVERLAP` (compute-comm scheduler), `STRAGGLER-cross-rank` (Op 32) —
   *"single-GPU has no AllReduce traffic to overlap"* (`PLAN.md:1721`, `:1701-1728`, `:110`). The **rest
   of the 85% target is single-GPU depth-wins** (TMA+clusters, L2) deferred post-CP-5.5, **not** a cluster
   measurement. The honest **single-GPU BF16 ceiling = ~67% (≈660 TFLOPS @ 700W)** (`PLAN.md:970-971`,
   `CIPHER_PLAN_EXECUTIVE_SUMMARY.md`).

3. **"Nebius pilot = an 8-GPU MFU validation plan" — NOT FOUND.** `v1_phase_b/DEVANG_NEBIUS_UPDATE.md`
   (2026-05-26) is entirely a **cipher-platform `.deb` deployment-design** message (apt install on
   neocloud H100 hosts via CDI; two ops questions about DKMS + nvidia-container-toolkit version). There is
   **no MFU cluster-validation pilot** in it.

4. **"Koopman LM-head 7.43%" — NOT FOUND.** Flagged "provenance not located" in
   `V1_CAPABILITY_AUDIT…md:292-293` and "informational historical context… not located in evidence"
   (`WEEK_13_14_SCOPE_LOCK.md §9`).

**Net:** the *mechanisms* are real and mostly POC-built; the *85%-additive-decomposition* and
*multi-GPU-cluster-gate* framings as recalled are largely **not in the written evidence** — the evidence
frames 85% as a deferred (post-CP-5.5) depth-win target over a measured ~67% single-GPU BF16 ceiling,
with only the NCCL/overlap slice being genuinely multi-GPU.

---

## 1. WIRED-vs-UNWIRED TRUTH TABLE (current runtime = `cipher_rt_phase4`, HEAD `2909eb3`)

"Wired" = initialized in `cipher_inject.c:cipher_v2_init_body()` AND able to fire. "POC" =
built+measured in `cipher-may13-evidence` (op31-prod snapshot, commit `83b76da`).

| MFU mechanism | current-runtime state | POC measured (result-file) |
|---|---|---|
| **FFN substitution — Marlin INT4** | **(a) BUILT+WIRED** `cipher_inject.c:62` env `CIPHER_MARLIN`; M≤64 decode gate | 1.38× B=8 / 1.38-1.55× decode M≤32 vs cuBLAS (may13 `BUILD_STATE.md`) |
| **FP8 (tonight, Path A)** | **(a) BUILT+WIRED** `cipher_inject.c:63` env `CIPHER_FP8`; priority 20 | 64.3-68.2% MFU vs 989, 1.32×@700W, +0.567% PPL FAIL (`D9_FP8_ACTUATOR_BUILD_REPORT.md`) |
| **Koopman / FFN-substitution** | **(a) WIRED but NEVER FIRES** `cipher_inject.c:66` env `CIPHER_KOOPMAN`; **FP16-only dtype gate blocks bf16 prod (m_total=11658, k_total=0)** | in-distribution top-1 90%, rel_err 2e-4 (`WEEK_14_STEP_2_KOOPMAN_TIER.md:89`) |
| **VOLT (DVFS — tok/W, NOT MFU)** | **WIRED but DEGRADED on this pod** (`audit_section_1b.md:46`: NVML clock-set NOT_SUPPORTED → classifier-only) | +57% TinyLlama-B1 / −14% Mistral-7B (`cipher-t43-envelope`) |
| **L2 weight persistence** | **(c) FILE-EXISTS-UNWIRED** — `may13_cipher_l2_persist.o` compiled into the `.so`; ACCESS_POLICY_WINDOW injection code at `cipher_intercept_cudart.cpp:2817` — but **`cipher_init()` never called from init → window injection is a permanent no-op** | 48,848 window_hits/10s firing; **0% ΔTFLOPS on M4 (compute-bound — wrong probe); NO MFU gain measured** (`CLAUDE.md:158-161`) |
| **Kernel fusion (SwiGLU/RMSNorm/RoPE)** | **(c) FILE-EXISTS-UNWIRED** — names detected (`cipher_kernel_table.cpp:78-84`) but `apply_recipe()` returns FALSE (`cipher_dispatch.cpp:481` "substitution happens at Python wrapper level"); **no `cipher_fusion*.o` in OBJS** | RoPE+RMSNorm+SiLU combined **1.074× tps / 1.043× tok/W** (Python-patched, graph-captured) (`rope_p7_results.json:9-12`); Llama-8B B=8 361.88 tps (`step9_llama8b_fusion_only.json`) |
| **CUDA Graph promotion (REMEMBER/SPECULATE)** | **(c) INSPECT-ONLY** — `cipher_graph_inspect.cpp` compiled (Makefile:429), reads captured graphs but **does NOT rewrite nodes (Phase-4 deferred)**; real replay opt-in behind `CIPHER_GRAPH_REPLAY`, observation-only default. SPECULATE: no evidence either repo | **4.07× (64-kernel) / 2.05× Mistral-7B decode / +35% tok/W** (`BUILD_STATE.md:413-414`); full-stack fusion+INT4+graph 8.70× tps (`:559`) |
| **CUTLASS persistent GEMM** | **(d) NOT BUILT** — zero references in either repo (Marlin INT4 is bandwidth-reduction, a different mechanism) | — (NOT FOUND) |
| **NCCL AllReduce overlap** | **(c) FILE-EXISTS-UNWIRED / SPEC-ONLY** — `libcipher_nccl_tuner.so` is a *separate* plugin; **zero `nccl*.o` in the `.so` OBJS**; "cuBLAS/Lt/NCCL dropped from may13 g_patches". W.7 validated plugin **primitives** (ABI/load/decision/fallback PASS) | **27% = NCCLbpf *paper* floor, NOT a CIPHER measurement** (`cipher_nccl.h:9,22-23`); multi-GPU p99 ≥20% gate **NOT MEASURED (hardware-deferred)** (`W7_NCCL_CLOSE_REPORT`) |

**Summary by bucket (the gap to a multi-GPU 85% measurement):**
- **(a) built + wired + can-fire:** Marlin INT4, FP8 (Koopman wired but inert on bf16; VOLT degraded).
- **(b) built + gain-needs-multi-GPU:** NCCL overlap primitive (plugin validated; ≥20% AllReduce p99 gate hardware-deferred).
- **(c) file-exists-but-unwired in current runtime:** L2 persist, kernel fusion, CUDA-graph promotion (inspect-only).
- **(d) not built:** CUTLASS persistent GEMM; SPECULATE/prefetch.

---

## 2. PER-MECHANISM DETAIL (file · wiring · measured · citation)

### L2 weight persistence
- POC (`cipher-may13-evidence` `83b76da`): `src/cipher_persist_engine.cpp:201` `hitProp = cudaAccessPropertyPersisting`; `src/cipher_l2_persist.cu` (239 lines); `tests/test_persist_engine.py` 5/5 PASS.
- Firing: 48,848 window_hits / 48,797 launches in 10s (`CLAUDE.md:161`). **MFU delta: 0% on M4 4096³ (compute-bound, wrong probe)** (`CLAUDE.md:158-159`). **No throughput-gain number exists** (NOT FOUND).
- Current runtime: `may13_cipher_l2_persist.o` in OBJS; injection code `cipher_intercept_cudart.cpp:2817`; **but `cipher_init()` is never called from `cipher_v2_init_body()` → permanent no-op** (`cipher_inject.c:50-101`). Wiring is the **Week-12-Step-6 deferred** item.

### Kernel fusion / ORCHESTRATE
- POC: NVRTC fused RMSNorm/SiLU-Mul/residual/RoPE kernels (`include/cipher_fusion_kernels.h`); correctness rel_err ~6e-5 (`test_fusion_correctness.py`); **RoPE+RMSNorm+SiLU = 1.074× tps, 1.043× tok/W** (`rope_p7_results.json`).
- Key blocker doc: `CIPHER_DRIVER_FUSION_DISCOVERY.md` — *"PyTorch eager decomposes RMSNorm/SiLU/RoPE into N independent kernels; no single target to intercept"* — POC gains came via **Python-level patching**, not driver interception.
- Current runtime: pattern names detected (`cipher_kernel_table.cpp:78-84`) but `apply_recipe(type=2)` returns FALSE (`cipher_dispatch.cpp:481`); **no fusion kernel wired**.

### NCCL AllReduce overlap
- POC: full implementation (NCCLbpf, neural policy, compute-overlap, tuner ABI) in may13; **27% AllReduce floor is the NCCLbpf *paper* threshold, not a CIPHER benchmark** (`cipher_nccl.h:9,22-23`).
- W.7 (`v1_phase_b/w7_nccl/W7_NCCL_CLOSE_REPORT`): plugin **primitive** validation PASS (ABI bind NCCL 2.26.2, init rc=0, getCollInfo 24 defined/0 bad, 1-rank bit-identical, clean fallback). **Item 5 (multi-GPU p99 ↓ ≥20%): NOT MEASURED — hardware-deferred** (single H100 can't run a 2-rank inter-GPU collective).
- Current runtime: **separate `.so` plugin only; not in libcipher_rt OBJS; compute-overlap path NOT actuated** (tuner-only).

### CUDA Graph promotion + CUTLASS persistent GEMM
- Graph POC: **4.07× (64-kernel seq), 2.05× Mistral-7B decode, +35% tok/W** (`BUILD_STATE.md:413-414`); full stack (fusion+INT4+graph) **8.70× tps / 10.66× tok/W** (`:559`).
- Current runtime: `cipher_graph_inspect.cpp` compiled — **inspection only (node-rewrite Phase-4 deferred)**; replay opt-in `CIPHER_GRAPH_REPLAY` (observation default).
- **CUTLASS persistent GEMM: NOT BUILT** (zero refs). **SPECULATE: no evidence** in either repo.

### FP8
- WIRED tonight (`2909eb3`). Measured: per-call **64.3%** / shared **68.2%** MFU vs 989; **1.32×@700W**; vLLM sustained GEMM **1.83×**; +24% tok/W B=8 (`STAGE13_FP8_REPORT.md:111`); vLLM full-coverage **91.7-98.3%** MFU (reachability) but PPL +0.37% FAIL; Path A all-layers PPL **+0.567%** / MMLU **−1.6pp** FAIL (`D9_FP8_ACTUATOR_BUILD_REPORT.md`, `D9_BUILD_REPORT.md`). Amdahl ceiling ~74% (31% non-GEMM).

### Koopman / FFN substitution
- WIRED (`cipher_inject.c:66`, env-gated) **but never fires on bf16 production**: `WEEK_14_FOLLOWUP_OPTION_2_STEP_0_5…` — m_total=11658, k_total=0; **FP16-only dtype gate** (kernel `CUDA_R_16F` vs vLLM `CUDA_R_16BF`). In-distribution narrow-domain top-1 90%. **LM-head 7.43%: NOT FOUND.**

### BF16 single-instance ceiling (the "is there headroom" baseline)
- `WEEK_6_OPTION_1_REDO.md`: B=1 prefill **59.57% vanilla / 59.09% CIPHER** (−0.81% delta); across batches **60-66%**; **CIPHER NEUTRAL** ≤0.31% TPS / ≤0.79% decode MFU; tok/W regression −0.25% to −0.59% single-instance. → single-instance is textbook compute-bound BF16; **CIPHER adds no MFU single-instance; product value is N≥2 cross-tenant.**

---

## 3. THE HONEST ANSWER — what is BUILT and ready to fire on a multi-GPU cluster, and what remains

**Ready to fire today (a-bucket, single- or multi-GPU):**
- **Marlin INT4** (decode M≤64) and **FP8** (large-M forward, 64-68% MFU) — both wired, env-gated, measured. These are the only two MFU actuators that actually engage in the current runtime.

**Built but gain-unmeasurable on this single-GPU pod (b-bucket — the genuine multi-GPU items):**
- **NCCL AllReduce overlap** — plugin primitive validated (W.7); the ≥20% AllReduce-p99 gain gate is hardware-deferred (needs ≥2 ranks / NVLink). This is the *only* mechanism whose gate is truly multi-GPU.

**File-exists-but-UNWIRED — these need a wiring step before any multi-GPU (or single-GPU) measurement (c-bucket):**
- **L2 weight persistence** — needs the `cipher_init()` callsite added to `cipher_v2_init_body()` (Week-12-Step-6 deferred); machinery + window-injection present, just disconnected.
- **Kernel fusion** — needs driver-level wiring (POC was Python-patched; `apply_recipe` returns FALSE); the `CIPHER_DRIVER_FUSION_DISCOVERY.md` blocker (eager decomposes into N kernels) is unresolved at the driver boundary.
- **CUDA Graph promotion** — inspect-only; node-rewrite/replay actuation deferred (Phase 4).

**Not built (d-bucket):**
- **CUTLASS persistent GEMM** (the "persistent kernel dispatch" the capability audit names as the MISSING Goal-3 lever, `V1_CAPABILITY_AUDIT…:316`); **SPECULATE/prefetch**.

**Scoping the next step (multi-GPU MFU measurement), from the real inventory:**
- A multi-GPU 85% measurement is **not gated by a single missing kernel** — it is gated by (1) the **NCCL overlap** path (the only true multi-GPU lever, primitive-ready, gain-unmeasured), AND (2) the **single-GPU depth-wins still unwired or unbuilt** (L2-persist wiring, fusion driver-wiring, persistent-GEMM build) that the plan counts toward the 85% target over the ~67% baseline.
- The evidence does **not** support "flip on the 8 mechanisms and measure 85% on 8 GPUs." It supports: **Marlin + FP8 fire today; L2/fusion/graph are unwired single-GPU levers; CUTLASS/SPECULATE unbuilt; NCCL is the lone multi-GPU lever (primitive-ready, gain-deferred); the additive-decomposition and cluster-gate framings are not in the written record (§0).**

**Recommended honest framing for the next step:** before any multi-GPU run, decide which of the unwired
single-GPU levers (L2-persist wiring, fusion, graph-replay, persistent-GEMM) to actually build/wire — and
treat NCCL-overlap as the separate multi-GPU validation that needs ≥2 ranks. The 85% number itself is a
*deferred program target over ~67%*, not a measured or single-equation result in the evidence.

---

## 4. EVERY MFU REPORT SURFACED (path · what it concluded)
- `CIPHER_REENGINEERING_PLAN.md:986-1019,1701-1728` — 85% deferred post-CP-5.5; 4-lever decomp; 3 NCCL ops v2-deferred; ~67% single-GPU ceiling.
- `V1_CAPABILITY_AUDIT_AND_PRODUCT_ARCHITECTURE.md:271-318` (2026-05-27) — Goal 3 "85% MFU **cross-tenant**" gated by POOL executor + persistent-kernel dispatch (MISSING); NCCL tuner SPEC-ONLY; fusion PARTIAL/observation-only.
- `WEEK_6_OPTION_1_REDO.md` — BF16 single-instance 59-66%, CIPHER neutral (≤0.31% TPS).
- `WEEK_5_POSTCLOSE_OPTION_1_MFU.md` — B=1 decode CIPHER −4.91% (early harness; later corrected).
- `D9_BUILD_REPORT.md` / `D9_FP8_ACTUATOR_BUILD_REPORT.md` — FP8 reachability 91.7-98.3% (vLLM) vs delivered 64-68% (torch); quality fails.
- `STAGE13_FP8_REPORT.md`, `step7_fp8.json` — FP8 +24% tok/W B=8.
- may13 `BUILD_STATE.md` / `CLAUDE.md` — graph 4.07×/2.05×, full-stack 8.70×; L2 48,848 hits/0% ΔTFLOPS.
- `W7_NCCL_CLOSE_REPORT` — NCCL plugin primitives PASS; multi-GPU gain hardware-deferred.
- `WEEK_14_FOLLOWUP_OPTION_2_STEP_0_5…` — Koopman never routes on bf16 (k_total=0).

**NOT FOUND IN EVIDENCE (recalled but uncited):** the additive "FFN+30/L2+6/fusion+7/NCCL+8" table;
"85% requires 8-GPU cluster" as a stated gate (only the NCCL slice is multi-GPU); a Nebius MFU-pilot
plan (Nebius = `.deb` deployment); the 27% AllReduce as a CIPHER measurement (it is a paper floor);
the Koopman LM-head 7.43% provenance.
