# V.0 engagement map — ALL THREE SUBSTRATES, stock-vLLM DECODE path

**Date:** 2026-05-30. **Type:** READ-ONLY correction of the `V0_WEIGHTED_MIX_HARNESS_DESIGN_MEMO`
engagement map (§3.0), which was **matmul/GEMM-substrate-scoped** — it marked G3/G4/G5 "nvjet casualty"
from the cuBLAS intercept alone. CIPHER is **three substrates** (matmul/cuBLAS · attention/SDPA+vLLM-kernels ·
memory/cuMemMap) attacking **three HBM levers** (weights · KV · activations). The nvjet bypass hits
**only matmul on PREFILL** — and **prefill→v1.5, 85%-magnitude→multi-GPU** (adjudicated, not re-opened
here). This redraws the map on the **stock-vLLM DECODE** path. No build, no commit, anchors unchanged.
Every claim file:line cited; "deferred/untested" ≠ "impossible".

**Headline correction:** on **vLLM DECODE**, the matmul intercept **FIRES** (`0→11658` cuBLAS calls in a
real vLLM-V1 worker — `WEEK_14_FOLLOWUP_OPTION_2_STEP_0_VLLM_WORKER_HOOK.md:43`); the nvjet bypass was
**prefill-specific** (`D9_CIPHER_DELIVERY_REPORT.md:25` "a full 8×2048 vLLM **prefill**"). The attention
substrate **intercepts vLLM's own FA3 kernel** (5280/5280 — `W5_FLASHATTN_CLOSE_REPORT.md`), and KV-dedup
**engages on vLLM via plugin** (measured). **No goal "does not engage at all" on vLLM decode.** The real
open items are *actuation-fire on the live intercept* (G3/G4 substitution = 0 handled today) and *fusion
not-built* (G2's third lever) — each with a **named next step**, not a casualty.

---

## PART 1 — THE THREE SUBSTRATES on vLLM DECODE (cited)

### 1. matmul substrate (cuBLAS, cublasGemmEx/cublasLtMatmul) — INTERCEPT ENGAGES on decode
- **vLLM DECODE GEMMs hit the public symbol — intercept FIRES (measured):**
  `WEEK_14_FOLLOWUP_OPTION_2_STEP_0_VLLM_WORKER_HOOK.md:43` — *"`cipher_rt_cublas_shim_calls`: **0 → 11658**
  over the 128-token decode burst in worker pid 2033010 … GOT patches are intercepting `cublasGemmEx` in
  the worker process"*; `:47` *"MATMUL: exit totals — calls=11658 handled=0 passthrough=11658"*. (Real
  vLLM-V1 AsyncLLMEngine TinyLlama, worker-spawn, LD_PRELOAD + the cipher_vllm_plugin worker-init hook.)
- **The nvjet bypass was PREFILL-only:** `D9_CIPHER_DELIVERY_REPORT.md:25,33-37` — *"a full 8×2048 vLLM
  **prefill** (real big GEMMs) … vLLM/torch on cu13 dispatch the **prefill** GEMMs through libcublas-
  internal `nvjet`"*. Decode GEMMs (small M) go through public `cublasGemmEx` (the 11658 above).
- **Corroborated on torch HF decode:** the Marlin churn test engaged on torch HF decode via public cuBLAS
  (`gate_churn.json` `A_solo_sub`: mistral 3600 / llama8b 3584 / qwen2 1792 — `sub>0`).
- **BUT actuation does NOT fire on vLLM decode today** — `handled=0 passthrough=11658` (`:47`); Gate-3
  (`:51-57`): *"Koopman … handled … stayed at 0 … even with β=0.99 … the MATMUL dispatcher classified 100%
  of decode GEMMs as PASSTHROUGH"* — either a classifier filter at the dispatcher or OOD>0.99. **Named
  next** (`:57` + `D10_LT_ROUTE_MEMO`): route the typed cuBLASLt variants (`cublasLtHSHMatmul` fp16)
  through `cipher_rt_matmul_dispatch` (as `cublasGemmEx` already does at `cipher_rt_cublas_shim.c:221`) and
  diagnose the decode-shape passthrough (Step 1). **⇒ intercept ENGAGES; actuator-substitution = OPEN
  (named path), not a bypass.**

### 2. attention substrate — ENGAGES on vLLM via the vLLM-kernel GOT-patcher (NOT torch SDPA)
- **Two distinct intercept points — don't conflate:**
  - `cipher_rt_attn_dispatch.cpp` / `.h:11-27` patches the **torch ATen SDPA dispatcher**
    (`_scaled_dot_product_*_attention::call` in libtorch_cpu) — the **HF path**; vLLM does NOT use torch
    SDPA, so this one is **bypassed on vLLM** (`SUBSTRATE_STATE_OF_RECORD.md:82` / `WEEK_12_STEP_5_D14_BACKFILL`:
    *"the vLLM real path uses xformers/paged-attention which bypasses the ATen SDPA trampolines"*).
  - `cipher_rt_attn_6pattern.c:5,20-26` GOT-patches **vLLM 0.21's OWN attention libs**: P1 FA2
    `_vllm_fa2_C.abi3.so`, **P2 FA3 `_vllm_fa3_C.abi3.so` (Hopper)**, P5 FlashMLA `_flashmla_C.abi3.so`,
    P6 PagedAttn `_C.abi3.so`. **This one ENGAGES on vLLM.**
- **W.5 CLOSED (`w5-flashattn-close`) with FA3 actively intercepting on vLLM 0.21 / H100:**
  `W5_FLASHATTN_CLOSE_REPORT.md:8-10` *"FA3 (the H100 FLASH_ATTN hot path) intercepts 5280/5280…"*; `:55`
  *"vLLM completes all 9 close-gate cells (4 with active FA3 interception)"*; `:101-102` *"vLLM 0.21
  FLASH_ATTN backend on H100 routes to FA3"*. FlashInfer P3/P4 = JIT residue (no static symbol → not
  GOT-interceptable, v1.x — `:93,97`). **v1 = intercept+passthrough only** (`:49` *"the trampoline forwards
  every call"*; substitution v1.x `:97-98`). **⇒ attention intercept ENGAGES on vLLM (FA3 live);
  KV-via-attention SUBSTITUTION (FA3-shape-matched / MLA / PagedAttn) is v1.x.**

### 3. memory substrate (cuMemMap, KV-dedup) — ENGAGES on vLLM (plugin, measured)
- **Delivered via the vLLM PLUGIN vector (NOT the cuBLAS/SDPA intercept):** `cipher_vllm_kvdedup.py:1-7,
  23-29` — registered under `vllm.general_plugins`, runs in the EngineCore subprocess, **monkey-patches
  `GPUModelRunner._allocate_kv_cache_tensors`** (after CP 5.1's CIPHER-VMM-backed KV tensors); does an
  in-place `cuMemUnmap`+`cuMemRelease`+`cuMemMap` 2 MiB-page swap post-prefill (`:118-169`). **nvjet-
  irrelevant** (operates on KV-cache buffer ownership, not GEMMs).
- **Engages under container vLLM:** `D9_CIPHER_DELIVERY_REPORT.md:21-22` *"CIPHER's KV-dedup/telemetry
  plugins can now run under the container vLLM for V.1"*. **Measured on stock vLLM decode KV:** N=2 cross-
  tenant hits (`tests/sc_kvdedup_n2.py:139-150`), N=4 total_hits≥3 + **≥30 GiB saved** TinyLlama
  (`sc_kvdedup_n4_tinyllama.py:119-132`), 50.62% cross-tenant dedup ratio toolagent (`t4_6_3_dedup_report.md`).

---

## PART 2 — THE THREE HBM LEVERS on vLLM decode (cited)

| Lever | Substrate | vLLM-decode verdict |
|---|---|---|
| **1 weights** (GEMM / Koopman) | matmul | **INTERCEPT ENGAGES** (11658 calls); **actuation OPEN** (0 handled — named next: LT-route + passthrough diagnosis). PREFILL→nvjet (v1.5). |
| **2 KV** (compression / dedup) | memory (plugin) + attention (FA3) | **ENGAGES** — KV-dedup measured on vLLM (memory plugin, NON-GEMM); attention FA3 intercept live (5280/5280), KV-via-attn substitution v1.x. |
| **3 activations** (RMSNorm/RoPE/SwiGLU fusion) | fusion | **NOT_BUILT** — CP 4.7 closed at bound (`CP_4_7_DESIGN_MEMO.md`; ~1.1% hook-reachable, not built). v-next; not a vLLM-specific blocker. |

---

## PART 3 — THE CORRECTED PER-GOAL ENGAGEMENT MAP (deliverable)

| Goal | Substrate(s) / lever(s) | vLLM-DECODE verdict (cited) |
|---|---|---|
| **G1 density** | POOL batching + KV-dedup (memory, non-GEMM) | **ENGAGES** — KV-dedup measured on vLLM (`sc_kvdedup_n2/n4`, ≥30 GiB, plugin vector); POOL is CIPHER's own batched executor (engages independent of the intercept). |
| **G2 tok/W** | VOLT (clock) + KV-compress (Lever 2) + activation-fusion (Lever 3) | **ENGAGES via 2 of 3 levers** — VOLT clock auto-activates (`W1_VOLT_CLOSE_REPORT.md`, intercept-independent) + KV (memory plugin); **fusion NOT_BUILT** (CP 4.7) → the third lever is v-next. The ≥1.5× stacks VOLT+KV today; fusion adds headroom later. |
| **G3 MFU machinery** | matmul + attention + fusion (magnitude→multi-GPU) | **machinery ENGAGES on decode** — matmul intercept live (11658) + attention FA3 live (5280/5280); **GEMM-substitution actuation = 0 handled today** (named next: LT-route/Step-1); fusion NOT_BUILT. CUPTI measures vLLM MFU regardless. (Magnitude 85% = multi-GPU, out of scope.) |
| **G4 Koopman** | weight-sub GEMM (matmul), decode | **INTERCEPT ENGAGES; ACTUATION does NOT fire yet** — `WEEK_14…:51-57` Koopman handled=0 even at β=0.99 (decode GEMMs 100% passthrough). **Specific blocker:** dispatcher classifies vLLM decode GEMMs PASSTHROUGH (classifier filter OR Koopman OOD>0.99 OR typed-cuBLASLt-variant unrouted). **Named next:** D10 LT-route + Step-1 passthrough diagnosis. NOT a casualty — the intercept sees every call. |
| **G5 zero-touch** | K.1 classifier (across substrates) | **OBSERVATION ENGAGES on decode** — the dispatcher classified the 11658 decode GEMMs (as passthrough), so the classifier IS fed on vLLM decode; the `obs=0` was the **prefill-bypass** test (`D9:28`). The engagement-gap reframe (`WEEK_14…:82`, Mem #17): *"LD_PRELOAD + drop-in cipher_vllm_plugin, zero application code changes"* — the worker-init hook is the fix (already landed, `option-2-step-0-vllm-worker-init-hook`). **End-to-end profile-report <30s = UNTESTED** (named next: time the cipher-platform profile on a stock serve). |

---

## PART 4 — THE HONEST VERDICT

**Given prefill→v1.5 and 85%-magnitude→multi-GPU: YES — all 5 goals have a LIVE engagement path on the
stock-vLLM DECODE path, single-GPU, across the three substrates.** The GEMM-only "nvjet casualty" framing
was wrong for decode. Precise status, distinguishing *engages* / *actuation-pending* / *not-built*:

- **ENGAGES — live & measured on vLLM decode:** KV-dedup (memory plugin, ≥30 GiB), VOLT (clock),
  matmul cuBLAS intercept (11658 calls), attention FA3 intercept (5280/5280). → **G1, G2 (via VOLT+KV)
  are deliverable on stock vLLM decode now.**
- **ENGAGES (intercept) but ACTUATION-PENDING — named next step, NOT impossible:** the GEMM-substitution
  actuators (Marlin/Koopman, G3-lift + G4) fire **0 handled** on vLLM decode today (100% passthrough,
  `WEEK_14…:47,51-57`) — the intercept sees the calls; the dispatcher passes them through. Named path:
  **route the typed cuBLASLt variants through `cipher_rt_matmul_dispatch`** (`D10_LT_ROUTE_MEMO`) +
  diagnose the decode-shape passthrough (Step 1). This is "engages-but-doesn't-substitute-yet", not "does
  not engage." G5's classifier observes decode (the 11658 were classified); end-to-end profile-time UNTESTED.
- **NOT_BUILT (v-next, not a vLLM-blocker):** activation-fusion (CP 4.7 at bound) — G2's third lever;
  KV-via-attention *substitution* (W.5 v1.x — intercept is live, the HANDLED-substitution is v1.5).
- **DEFERRED (scope-locked, not re-opened):** matmul on PREFILL (nvjet → v1.5); 85% MFU magnitude
  (multi-GPU); FlashInfer P3/P4 attention (JIT, no static symbol → v1.x).

**Nothing "does not engage at all" on vLLM decode across the three substrates.** The corrected founder
picture: **G1 density + G2 tok/W are deliverable on stock vLLM decode now** (KV-dedup + VOLT, both
measured/intercept-independent); **G3/G4 GEMM-substitution and G5 profile-report are intercept-live with a
named actuation-fire next step** (LT-route / passthrough-diagnosis / profile-timing), **not** matmul-nvjet
casualties. V.0 should therefore build toward 5-goal **decode** engagement, with the §Part-3 named-next
items as its pre-flight, and the V0 design memo's §3.0 map **superseded by this table**.

---

## NOTE
Supersedes `V0_WEIGHTED_MIX_HARNESS_DESIGN_MEMO.md` §3.0/§3.2 engagement framing (which read the bypass as
matmul-only-and-decode-inclusive). The evict-wiring conditional in that memo's §3.2 is **unchanged in
logic** but now scoped to where the **matmul actuators actually substitute** — i.e. once the LT-route/
passthrough-diagnosis lands so Marlin/Koopman fire on vLLM decode (today 0 handled), the surrogate-memory
evict-wiring becomes load-bearing on the vLLM path too; until then it is load-bearing only on the torch-HF
decode path (where Marlin fires, `sub>0`). No build, no commit, anchors unchanged.
