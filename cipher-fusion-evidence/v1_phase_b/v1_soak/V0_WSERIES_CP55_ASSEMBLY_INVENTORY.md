# READ-ONLY — W.1–W.8 → CP-5.5 ASSEMBLY INVENTORY (what's wired + fires in the ONE production .so)

**Date:** 2026-05-31. **Type:** READ-ONLY (no build, no commit, anchors unchanged). Production runtime =
`cipher_rt_phase4` HEAD `936402c`, `libcipher_rt.so`. Every claim cited to `file:line` in that tree (init wire
`cipher_inject.c:cipher_v2_init_body` :50-101; Makefile OBJS; per-actuator env gates; `nm -D` symbol checks).
**W-numbering provenance:** W.1/W.2/W.5 are **code-self-labeled** (the `W.1`/`W.2`/`W.5` comments cited);
W.3/W.4/W.6/W.7/W.8 are **mapped by function** (the code doesn't self-label them W.N) — if Anil's numbering
differs, the row *labels* may shift but the cited per-actuator facts stand. **Headline: the
*substrate* (classifier + matmul/attn dispatch registries + telemetry) is always-on and assembled; the
*perf actuators* are independent, default-OFF env flags — there is NO single controller that fires them as one
product. Of W.1–W.8, only VOLT/Marlin/POOL/attn-detect are wired-and-can-fire (each env-gated, and 3 of the 4
are gate-blocked or degraded in real serving); Koopman is wired-but-inert on bf16; NCCL and the persistent
kernel are NOT in the .so.**

---

## THE TABLE

| W-step | actuator (obj) | in `.so`? (Makefile + init wire) | FIRES in real serving? | closed-gate (rig) | goal |
|---|---|---|---|---|---|
| **W.1 VOLT** | `cipher_rt_volt.o` (Makefile:59) | **YES** — init `cipher_inject.c:60`; env `CIPHER_VOLT` (`cipher_rt_volt.c:323`), classifier-gated (reads `volt_engage`, `:28,:82`) | **degraded** — fires (classifier+env) but NVML clock-set NOT_SUPPORTED on this pod → classifier/observe-only (dossier §1b) | TinyLlama B=1 clock-lock rig (+55% tok/W, memory-bound only; doesn't generalize to 7B) | tok/W (not MFU) |
| **W.2 Marlin** | `cipher_rt_marlin_{engine,actuator,kernel_src}.o` + `cipher_rt_machete_intercept.o` (Makefile:64-65,79) | **YES** — init `:62` (+ Machete GOT `:64`); env `CIPHER_MARLIN` (`:339`) default-OFF; classifier-gated (`marlin_actuator.c:153-154`); on matmul-dispatch (`:61`) | **eager-only + REGRESSES** — substitutes ~97% on vLLM B≥8 but −32%→−53% tok/s, and **graph-bypassed** in prod (CUDA-graph replay skips the cuBLAS GOT-patch) — `V0_G3_MARLIN_VLLM_BATCHED_DECODE_MEASUREMENT.md` | torch B=8 microbench (1.38× — disproven end-to-end) | G3 GEMM-substitution / MFU |
| **W.3 Koopman** | `cipher_rt_koopman_engine.o` + `may13_cipher_koopman_runtime.o` (Makefile:67,101) | **YES** — init `:66`; env `CIPHER_KOOPMAN` (`koopman_engine.cpp:274`) default-OFF; classifier flag `koopman_engage` exists | **INERT on bf16** — fp16-only dtype gate; vLLM decode is bf16 → `koopman_total=0` (WEEK_14_FOLLOWUP §0.5); also a rank-wall on fp16 general decode | WikiText autocal rig (sub=0, residual_ratio>0.5) | G4 (weakest) |
| **W.4 POOL** | `cipher_rt_green_ctx.o`, `cipher_rt_partition_router.o`, `may13_cipher_green_ctx.o` (Makefile:57,59,98) | **YES** — init green-ctx CP54 `:54`, SM_PACKER `:56`, partition-router `:57`; env `CIPHER_QOS_CLASS`/`CIPHER_SM_COUNT` (`green_ctx.c:121,148`) | **fires as substrate** — SM-partition / green-ctx transport works; but the *isolation* lever is contention-gated (CP54 closeout: OP-2 loses 0.256 vs 0.160) | synthetic SM-partition bomb/throughput rig | G1 density / multi-tenant |
| **W.5 attn (FlashAttn/geom)** | `cipher_rt_attn_6pattern.o` + `cipher_rt_attn_dispatch.o` + `cipher_rt_geom_capture.o` (Makefile:80,68,56) | **YES** — 6-pattern init `:65`, attn-dispatch `:67`, geom via `cipher_cupti.c`; 6-pattern **not** env-gated (arms by `dlopen(RTLD_NOLOAD)` the exact vLLM FA3 `.so`, `attn_6pattern.c:135,162`); geom env `CIPHER_GEOM_ATTN` default-OFF (`geom_capture.c:57`) | **observe/detect ONLY** — no attention-math substitution (W.5 wall). 6-pattern needs the V1 worker-init hook to arm; geom-attn is the staging observe-only counter (`geom-attn-detection-staging`) | byte-identical + agnostic-proof rig (observe-only) | G5 KV / attention |
| **W.6 NR27 / process-registry** | `cipher_stream_resolver.o` (Makefile:83); model_uuid fields in `cipher_rt_commit.h:59`, `koopman.h:17`, `kv_alloc.h:134` | **partial** — stream-resolver (NR29 `CIPHER_REGISTER_STREAMS`, `stream_resolver.c:22`) init `:71`; **NR27 REGISTER_MODEL is kmod-side** (G10 ABI), NOT in this `.so` | **fires as telemetry** — multi-tenant view-slot mirror (N=128 soak PASS); observability substrate, not a perf actuator | N=128 resolver soak (coherence/fairness) | per-tenant observability |
| **W.7 NCCL** | **NONE** — no `nccl*.o` in OBJS; `cipher_inject.c:99` is a comment ("cuBLAS/Lt/NCCL dropped from may13 g_patches") | **NO** — separate plugin `libcipher_nccl_tuner.so`, not in the production `.so`; no init call | **NOT WIRED** here | 1-rank plugin-primitive rig (ABI/load/fallback PASS; multi-GPU p99 gate hardware-deferred) | multi-GPU AllReduce overlap |
| **W.8 persistent kernel** | L2-persist: `may13_cipher_l2_persist.o` (the `.cu`, Makefile:98); fusion/megakernel: NONE in OBJS | **NO** — hot path dlsyms `cipher_persist_engine_*` (`cipher_intercept_cudart.cpp:589-595`) but **`nm -D` shows `cipher_persist_engine_*` ABSENT** from the `.so` → dlsym(RTLD_DEFAULT) returns NULL → `cipher_persist_maybe_apply` (`:2508`) no-ops. (Only `cipher_l2_persist_*` *primitives* are present, and they're never init'd — no call in `cipher_v2_init_body`.) Fusion/megakernel = may13-archive-only | **NOT WIRED / no-op** — engine absent; fused kernels are archive (`V0_FUSION_LEVER_SIZING.md`: ~9-12% graph, decaying, needs a port) | may13 microbench / megakernel-correctness rig (op31-prod) | G1 depth / MFU |

---

## THE ASSEMBLY GAP

### 1. WIRED + can-FIRE in the production `.so` (the real CP-5.5 starting set)
All are **default-OFF env flags** (no auto-fire): **W.1 VOLT** (`:60`, `CIPHER_VOLT`), **W.2 Marlin** (`:62`,
`CIPHER_MARLIN`), **W.3 Koopman** (`:66`, `CIPHER_KOOPMAN`), **FP8** (`:63`, `CIPHER_FP8`; the GEMM complement
to Marlin, n>64), **W.4 POOL** green-ctx/partition (`:54,:56,:57`), **W.5 attn** 6-pattern+geom (`:65,:67`).
Plus always-on **substrate**: classifier `cipher_workload_detect_init` (`:59`), matmul-dispatch (`:61`),
attn-dispatch (`:67`), telemetry COMMIT/RING_WRITE/AUDIT/stream-resolver (`:69-72`). **But "wired+can-fire" ≠
"delivers": of the perf actuators, only POOL fires usefully-as-substrate; VOLT is DVFS-degraded on this pod,
Marlin regresses+graph-bypassed, Koopman inert on bf16, attn observe-only.** The honest firing-and-useful set
today is thin.

### 2. BUILT but NOT WIRED into `cipher_rt_phase4` (archive/rig only — needs a port)
- **W.7 NCCL** — lives in `libcipher_nccl_tuner.so` (separate plugin); port = link the tuner decision path
  into the runtime (and the gain gate needs ≥2 ranks — hardware-deferred regardless).
- **W.8 persistent kernel / fusion** — may13-archive (`cipher-may13-evidence/src/cipher_fusion_kernels.cpp`,
  `cipher_weight_compress.cpp` megakernels, `cipher_graph.cpp` capture); port = compile the kernels into
  `cipher_rt_phase4` OBJS + wire the NVRTC path (env `CIPHER_SUBSTITUTE_V2`) + a per-framework vLLM-module
  monkey-patch so vLLM's graph captures them (`V0_FUSION_LEVER_SIZING.md`).
- **W.8 L2-persist** — `.cu` kernel is compiled but the **engine `.o` is unlinked** → hot-path dlsym no-ops;
  port = link `cipher_persist_engine` + add its init to `cipher_v2_init_body` (Week-12-Step-6 deferred).

### 3. CLOSED-IN-RIG but GRAPH-BYPASSED in real vLLM (the gate problem)
Any actuator on the **cuBLAS/SDPA-symbol GOT-patch path** is bypassed when vLLM captures decode into CUDA
graphs (replay re-runs the original kernel, not the intercept): **W.2 Marlin**, **FP8**, and the **W.5
6-pattern attn** intercept. Confirmed for Marlin (`V0_G3_…`: substitutes in eager, 0 under graphs). Opening
the gate = either monkey-patch CIPHER kernels into vLLM's modules so vLLM's own graph captures them, or build
graph-node-rewrite (Phase-4-deferred, INERT today — `V0_FUSED_MEGAKERNEL_GRAPH_INVENTORY.md`). Koopman is
gate-irrelevant (inert on bf16 first).

### 4. ORCHESTRATION SPINE — is there a "fire together" layer?
**Partial, advisory — not a unified controller.** The pieces:
- **Classifier substrate** `cipher_workload_detect` (`:59`) classifies the workload into classes
  (`workload_detect.cpp:1063-1091`: A1/A2/A3/A4/B1/B2/C1) and sets per-class engage flags `volt_engage /
  marlin_engage / koopman_engage` (`:833-915`).
- **Dispatch registries** — matmul-dispatch (`:61`) registers Marlin (prio 10) + FP8 (prio 20) and iterates
  them per `cublasGemmEx` (`cipher_rt_matmul_dispatch.c:try_actuators`); attn-dispatch (`:67`) is the parallel
  attention registry.
- **But each actuator gates ITSELF independently**: `enabled = env-flag AND classifier-engage AND shape-gate`
  (e.g. `marlin_actuator.c:142` env, `:153-154` classifier, `:200-208` shape). There is **no central
  ORCHESTRATE op** that reads the classifier and coordinates "fire actuators {X,Y,Z} together" — **`nm -D`
  shows no `orchestrate` symbol in the `.so`** and no orchestrate wiring in the init path. Assembly = set N env flags; the classifier then
  advisorily vetoes per-actuator. So "fire as one product" today = **N independent opt-in flags + an advisory
  classifier veto + two dispatch registries**, not a single orchestrated controller.

---

## NET (the real CP-5.5 starting point)
- **Assembled + firing substrate (always-on):** classifier, matmul/attn dispatch, telemetry (COMMIT/RING_WRITE/
  AUDIT/resolver), POOL green-ctx — these run together.
- **Wired actuators, env-gated, but each gate-blocked/degraded in real serving:** VOLT (DVFS-degraded),
  Marlin+FP8 (graph-bypassed; Marlin regresses at B≥8), attn (observe-only), Koopman (inert on bf16).
- **Not wired (needs a port):** NCCL (separate plugin + multi-GPU hardware), persistent-kernel/fusion (may13
  archive; ~9-12% decaying lever), L2-persist (engine unlinked).
- **CP-5.5 work = (a) open the vLLM-graph gate** so the GEMM/attn intercepts fire in production (the single
  highest-leverage blocker — it gates Marlin, FP8, attn at once), **(b) port the unwired levers** (fusion,
  L2-persist, NCCL) into the one `.so`, **(c) add a real orchestrate layer** if "fire together" is required
  beyond N env flags + advisory classifier, **(d) then measure the assembled product** — not relitigate
  components. Anchors unchanged; no code touched.
