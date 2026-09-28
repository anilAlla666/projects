# D.9 FP8 ACTUATOR — DESIGN MEMO (write-first; STOP for approval before any code)

**Date:** 2026-05-30. **Status:** DESIGN ONLY. No code written. No anchors rotated. Awaiting Anil's
approval of this memo before the build.

**Goal of the build this memo precedes:** convert the D.9 probe's *viable* verdict into a *delivered*
one — CIPHER executes FP8 at its own `cublasGemmEx` driver-boundary intercept, app unmodified, on the
torch-training / large-batch path the probe proved actuatable. This is the actuator that the probe
report's own "Next" section names (`D9_TRAINING_INTERCEPT_PROBE_REPORT.md:51-55`).

---

## 0. Anchors (verified first-hand this session)

| anchor | claimed | verified | note |
|---|---|---|---|
| `cipher_rt_phase4` HEAD / `d7-rh1-close` | `ed130e7` | ✅ `git rev-parse d7-rh1-close = ed130e760b…` = HEAD | **HEAD subject is "D.10 LT-ROUTE … (default-OFF)"** — i.e. the D.10 cublasLt layout-decode build sits *additively, default-OFF* atop the D.7 R-H1 close (`7f9f98c`). The tag rides the latest commit; this is the campaign anchor. |
| `01d4effb` | rt_phase4 ref | ⚠️ **not a git object** in rt_phase4 | Almost certainly the deployed `libcipher_rt.so` content hash (the `D9_BUILD_PREREG.md:69` and probe report both cite `01d4effb`/`d7-rh1-close` as the live-`.so`+tag pair). Not blocking — **anchors stay UNCHANGED this run** (Mem #16), nothing rotates. |
| kmod | `0.7.0` | ✅ `MODULE_VERSION("0.7.0")` `cipher_main.c` | D.8 FAIRNESS+SHIELD; not touched by this build. |
| evidence | `9230b42` | ✅ HEAD of `cipher-fusion-evidence` = the probe commit | |
| FP8 kernel `ba873f44` | "per-channel-W + per-token-A E4M3 lineage" | ⚠️ **see §5 — the only concrete lineage code (may13 `cipher_fp8_compute`) is per-TENSOR, not per-channel.** `ba873f44` is a symbolic lineage label in the D9 docs, not a file/commit on disk. | This is the single biggest build risk; called out explicitly below. |

**Discipline locks for this run (Mems):** #24 intercept-level (CIPHER's `cublasGemmEx` FP8, no
monkeypatch, no torch/vLLM source); #11 KL=0-on-output per declined layer + HARD STOP if any engaged
layer diverges beyond the bar; #16 net-new default-OFF, anchors unchanged, **no `.so` rotation this
run**; #13 additive-only (Marlin's gate is untouched; FP8 is a new path).

---

## 1. Where the FP8 actuator sits — alongside Marlin, on the existing substrate

The substrate was **purpose-built for exactly this**. `cipher_rt_matmul_dispatch.h:5-7` states it
verbatim: *"Marlin is the first actuator on it. Future actuators (FP8, fusion, speculative decode…)
plug in via the same registry without touching the cuBLAS shim … Marlin keys on B-ptr, FP8 keys on
(A-ptr, B-ptr, dtype)."*

Call path (verified first-hand):

```
torch nn.Linear  →  public cublasGemmEx
   │  (GOT-patched to cipher_rt_cublasGemmEx_impl, cipher_rt_cublas_shim.c:325-335)
   ▼
cipher_rt_cublasGemmEx_impl            (cublas_shim.c:137)   builds struct cipher_rt_matmul_call
   │                                                          {m,n,k,A/Atype,B/Btype,C/Ctype,stream}
   ▼
cipher_rt_matmul_dispatch(&call, g_real_gemmEx)   (cublas_shim.c:231)
   │
   ▼
cipher_rt_matmul_try_actuators()       (matmul_dispatch.c:88-110)  ← actuators in PRIORITY order
   ├─ priority 10  MARLIN_INT4   maybe_handle_marlin   (gate: M≤64 decode)
   ├─ priority 20  FP8_E4M3      maybe_handle_fp8      (gate: large-M, NEW — this build)
   └─ none HANDLED → passthrough_fn == g_real_gemmEx   (Stage 0 sacred, byte-identical)
```

**The FP8 actuator is a NEW `struct cipher_rt_matmul_actuator`** registered via
`cipher_rt_matmul_register_actuator()` from a new `cipher_rt_fp8_init()` (mirrors
`cipher_rt_marlin_init`, `cipher_rt_marlin_actuator.c:320-352`). New files:
`cipher_rt_fp8_actuator.c` (the actuator + gate + policy) and a kernel/engine TU
(`cipher_rt_fp8_engine.*`, see §5). **No edit to `cipher_rt_cublas_shim.c` or
`cipher_rt_matmul_dispatch.c` is required** for the actuator itself — registration is the only
contact point (the one shim edit we *do* add is the optional telemetry of §7).

**Marlin is untouched (Mem #13).** Its `MARLIN_MAX_M_GATE` (64) and its `maybe_handle_marlin` are
not modified. We do not "lift Marlin's gate"; we **add a second actuator whose gate is the
complement** of Marlin's. That is what "lift the M≤64 gate for the FP8 path only" means concretely.

---

## 2. The M-gate split (Marlin M≤64 ; FP8 large-M)

The cuBLAS↔Marlin axis mapping is load-bearing and verified at `cipher_rt_marlin_actuator.c:108-122,
245-247`:

```
weight  = call->A      activation = call->B      output = call->C
Marlin/FP8 M (batch rows)   = call->n     ← the gate dimension
        N (out_features)    = call->m
        K (in_features)     = call->k
```

- **Marlin** engages `call->n ∈ [1, 64]` (decode regime; `marlin_actuator.c:250`,
  `MARLIN_MAX_M_GATE=64`). Returns PASSTHROUGH for `call->n > 64`.
- **FP8** engages **`call->n > 64`** — the large-batch / prefill / training-forward regime the probe
  proved fires (Mistral B=64 → `call->n = B·S = thousands`). For `call->n ≤ 64`, FP8 returns
  PASSTHROUGH and the GEMM is Marlin's (or cuBLAS's).

Because the two gates are **disjoint by construction**, registering FP8 at **priority 20** (after
Marlin's 10) gives a clean, race-free split: a given GEMM is offered to Marlin first; Marlin's
PASSTHROUGH on large-M hands it to FP8; small-M never reaches FP8. (`try_actuators` iterates in
priority order and stops at the first HANDLED — `matmul_dispatch.c:95-108`.) No GEMM is eligible for
both.

---

## 3. Per-layer (per-GEMM) bounded-quality engagement — the decline mechanism

### 3.1 What the recorded bar actually requires (corrected)

The locked gate (`D9_BUILD_PREREG.md`): **PPL Δ ≤ 0.3%** (HARD), **MMLU Δ ≤ 0.5%** (HARD), KL ≤ 0.01
nats (diagnostic, non-binding). The per-layer back-off measurement
(`d9_perlayer_backoff_result.json`) recorded two operating points:

| config | engaged set | flops-coverage | PPL Δ | within 0.3% bar | proj MFU vs 989 |
|---|---|---|---|---|---|
| `all_fp8` | {q,k,v,o,gate,up,**down**} = 224 | 100% | **+0.37%** | ❌ (needs amendment) | 98.3% |
| **`drop_downproj`** (chosen) | {q,k,v,o,gate,up} = 192 | 73% | **+0.2465%** | ✅ | 85.9% |

**Corrected engaged-set definition (advisor catch — this was nearly wrong):**
- `224 = 32 layers × 7` linears; `192 = 32 × 6` (drops down_proj). **`lm_head` is in NEITHER set**
  (224 ≠ 225) — it always declines.
- So the engaged set is **exactly `{q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj}`**, with
  **`down_proj` and `lm_head` declined** (down_proj declined only in the `drop_downproj` policy;
  lm_head declined in both).
- A naïve `k > m` predicate is **WRONG**: with GQA, `k_proj`/`v_proj` are 4096→1024 (k>m) and would
  be wrongly declined; a "large-N, fp16-in" rule would wrongly grab `lm_head`. Either error moves the
  engaged set off the measured one and **voids the "one config" correspondence with the bar.**

### 3.2 Driver-pure discriminator (Mem #24-clean)

down_proj and lm_head are isolated by **exact shape against per-model dims**, not by a heuristic:

- **down_proj** ⟺ `call->k == intermediate_size` (Mistral 14336; the unique linear whose *contracting*
  dim is the FFN width).
- **lm_head** ⟺ `call->m == vocab_size` (Mistral 32000).

Source of `intermediate_size` / `vocab_size`, in preference order (all driver-side, no framework
tagging):
1. **Model registry (G10 `CIPHER_REGISTER_MODEL`, `model_uuid`) + hf_config dims.** Primary.
   *Build sub-item:* confirm the registry actually carries `intermediate_size` and `vocab_size`; the
   G5 `compute_va_gib` path ingests KV dims but may not carry these two — if absent, add them
   (additive, no ABI break to existing NRs).
2. **Workload classifier signals** — *already present and driver-pure*: `cipher_workload_detect.cpp:1635-1639`
   documents `max_k` ≙ `intermediate_size` (FFN) and `max_m` ≙ `vocab_size`. The actuator can decline
   `call->k == max_k` (down_proj) and `call->m == max_m` (lm_head) once the classifier has settled.
   Fallback when the registry lacks the fields; needs brief warmup (state it).
3. **Per-family recipe table** — prior art exists (`src/may13/cipher_recipes.cpp:388-395` lm-head
   table; 430/450 FFN-shape comments). Static fallback for known families.

The decline is a **shape predicate evaluated per GEMM inside `maybe_handle_fp8`** → `return
CIPHER_RT_MATMUL_PASSTHROUGH` for declined layers, which falls through to **real cuBLAS in the app's
fp16/bf16** ⇒ those layers are byte-identical (KL=0-on-output, Mem #11). Engaged layers run FP8.

### 3.3 Which policy ships — a DECISION FOR APPROVAL (not pre-decided)

Both projMFUs above are **vLLM-prefill projections**, not the torch-path delivery number this build
measures. The trade-off:

- **(A) `drop_downproj` (recommended default):** stays within the *locked* 0.3% bar with **no bar
  amendment**. Risk: projMFU **85.9% barely clears 85%**, and running down_proj in fp16 (≈27% of FFN
  flops) plus per-token quant overhead could push the *real torch-path* number **below 85%**.
- **(B) `all_fp8`:** projMFU 98.3%, comfortable MFU margin — **but requires the 0.3%→0.37% bar
  amendment**, recorded with rationale *before* any MFU+quality claim. The amendment is defensible
  (the literature basis cited in the prereg, arXiv 2411.02355, calls per-channel/per-token E4M3
  "essentially lossless"; +0.37% PPL with MMLU +0.0/−0.4 and true-KL 0.00357 nats is within that
  envelope).

**Recommendation:** build the per-GEMM decline mechanism so the policy is a single switch
(`CIPHER_FP8_POLICY=drop_downproj|all_fp8`, default `drop_downproj`), measure the **real** torch-path
MFU+quality for the chosen config (§6), and let the measured numbers — not the projection — settle
A-vs-B. **Anil decides at approval whether to pre-authorize the 0.3%→0.37% amendment** so the build
may report `all_fp8` if `drop_downproj` misses 85% on the real path.

---

## 4. Where weight-prequant happens (one-time, static) — mirrors Marlin's lazy cache

Per-channel **weight** absmax is computed **once per `(model_id, weight_ptr)`**, exactly like Marlin's
proven lazy path (`marlin_actuator.c:267-288`, engine `ensure_weight_quantized_repacked`
`marlin_engine.cpp:1008-1069`):

1. `maybe_handle_fp8` observes the weight pointer; after `STABILITY_THRESHOLD` (reuse Marlin's 4)
   stable observations (content-hash verified, as may13 does via a 128-byte D2H probe ~1µs), it
   computes the **per-output-channel** absmax → E4M3 weight + a per-channel scale vector, allocates
   device buffers, caches them in an FP8 weight registry keyed on `(model_id, w_ptr)`.
2. Every subsequent call for that weight hits the cache (O(1), sync-free).

**Key separation from Marlin (advisor gotcha):** FP8's cache is **distinct** from Marlin's INT4 slots
(different scheme, different key per the header: FP8 keys on `(A-ptr, B-ptr, dtype)`). Do **not** reuse
Marlin's `g_weights` — a shared key would let one scheme's buffers be read under the other's
interpretation (silent corruption). Separate registry, separate TU.

**Static-weight scope (advisor scope catch):** static prequant assumes the weight is constant. This
holds for the **forward / large-batch inference** GEMMs (the MFU-relevant target). For a **real train
step**, weights mutate each optimizer step → the content-hash re-verify forces re-quant (per-step
overhead), and the **backward** GEMMs (wgrad/dgrad) have *no* static weight operand at all. **v1
scopes FP8 to the static-weight forward/large-batch GEMMs.** The train-step measurement (§6) is framed
as a **firing + MFU demonstration** (the intercept fires, the forward GEMMs actuate), explicitly **not**
a converged-training-with-FP8 claim. Stating this prevents an over-claim.

---

## 5. Where per-token activation quant happens (inline) + THE build risk: per-tensor → per-channel

### 5.1 The lineage and the gap

The only concrete FP8 code on disk is the may13 `cipher_fp8_compute` (`cipher-may13-evidence/`,
md5s `7cf19003`/`01a4da59`/`0ceac54a` — **none is `ba873f44`; that label is symbolic**). It gives us
the reusable lineage:
- NVRTC fp16→E4M3 quant kernels + a fused single-launch activation quant
  (`cipher_fused_fp16_to_fp8`, `cipher_fp8_fused_quant.cu:93`, ~2µs at decode M, cooperative grid);
- `cublasLtMatmul` FP8×FP8→**fp16** with descriptor `A_SCALE_PTR`/`B_SCALE_PTR` doing the implicit
  rescale (`cipher_fp8_compute.cpp:868-877`) — **fp16-in / fp16-out preserved end-to-end**;
- shape-cached descriptor/algo pool; lazy stable-weight gate; `CIPHER_FP8_COMPUTE=on` default-OFF.

**The gap (advisor: THE load-bearing risk).** may13's scheme is **per-TENSOR weight + per-shape
activation** (scalar scales via cuBLASLt `*_SCALE_POINTER`) — the STAGE13 report says so outright:
*"No per-channel / blocked scales"* (`STAGE13_FP8_REPORT.md:178-180`). But the recorded **bar**
(+0.37%/+0.2465%) was measured with **per-channel-W + per-token-A rowwise** via torch `_scaled_mm`
(`d9_hf_quality_result.json`: *"per_channel_W+per_token_A rowwise FP8 E4M3"*) — **not a CIPHER
kernel.** Per-tensor is measurably worse (`d9_pertensor_downproj_result.json`: all-fp8 per-tensor
**+0.478%** vs per-channel **+0.37%**).

⇒ **The actuator MUST execute per-channel-W + per-token-A**, or its quality will not correspond to the
bar — this is precisely the *"per-tensor-MFU / per-channel-quality scheme mismatch, the
apples-to-oranges trap"* the charter forbids.

### 5.2 How activation per-token quant sits inline

Inside `maybe_handle_fp8`, on `call->stream`, before the GEMM: launch the per-**row** (per-token)
absmax→E4M3 quant of the activation `call->B` (extend may13's fused kernel from one global absmax to
one absmax per M-row). Output: device E4M3 activation buffer + per-token scale vector. Then the FP8
GEMM with per-channel weight scale (cached, §4) + per-token activation scale → **fp16/bf16 C**
written into `call->C`. `Ctype` is never modified ⇒ the app sees its original dtype (the shim forwards
`Ctype` verbatim to passthrough on decline, and the actuator writes the same dtype on engage).

### 5.3 Build-time feasibility gate (do this FIRST in the build, before wiring the actuator)

Per-channel/per-token ("rowwise"/"outer-vector") FP8 scaling on Hopper via cuBLASLt is **genuinely
uncertain on this cu13 build** and must be **verified empirically before committing the GEMM path**:
- **Option 1 (preferred):** cuBLASLt outer-vector / `*_SCALE_MODE` per-row(A)/per-col(B) FP32 scale
  mode, if present and correct on `libcublas.so.13` here. Verify with a standalone correctness probe
  (FP8 rowwise vs fp16 reference) before integration.
- **Option 2 (fallback, name it now):** custom CUTLASS FP8 epilogue (or a hand dequant epilogue
  kernel after a per-tensor cuBLASLt FP8 GEMM) that applies the per-channel × per-token rescale. More
  code, but removes the cuBLASLt-feature dependency.
- **Worst case at every layer:** `CIPHER_RT_MATMUL_PASSTHROUGH` → real cuBLAS fp16 (Stage 0 sacred).
  If neither option validates, the actuator declines and the build reports the intercept finding
  (charter's third outcome).

### 5.4 Quality is RE-MEASURED with CIPHER's kernel

The +0.37%/+0.2465% numbers are a **feasibility prior, not the delivery quality** (they came from
`_scaled_mm`, not CIPHER). The "one config" rule means the build **re-measures PPL + MMLU + KL on the
exact CIPHER-engaged config** (same kernel, same scheme, same engaged set). The memo does **not** claim
the bar is already met; the build proves it on CIPHER's own path. "Same config" is **proven by
telemetry, not assumed** (§6(d)): the engaged set must be identical *and non-empty* across the MFU and
quality runs, else the quality number is silently a partial-fp16 measurement.

---

## 6. Measurement plan — ONE config, CIPHER-engaged, torch large-batch, driver boundary

**Scope = torch / HF nn.Linear path** (where `cublasGemmEx` fires, proven: 3375 calls = 15×32×7 +
15 lm_head). **NOT vLLM prefill** — that bypasses to libcublas-internal nvjet (sees 0 public-symbol
calls; `D9_CIPHER_DELIVERY_REPORT.md:33-42`). The vLLM-prefill delivery is a *separate* track
(framework-boundary override `f194bd4`, or the D.10 nvjet depth) and is **explicitly out of scope
here**. We also do **not** conflate with §1b's "PARTIAL@64%": that was vLLM-prefill *under the strict
token-identity contract*; this is the **torch path under the bounded-quality gate** — different path
*and* different gate.

Single config (CIPHER_FP8 on, chosen policy from §3.3), Mistral-7B:
- **(a) MFU** — Mistral-7B **B=64 forward** + **a real train step** (firing+MFU demo per §4), CIPHER
  doing the FP8 at the driver boundary. Report **MFU vs the 989 bf16 reference at sustained 700 W**
  (watts + clock logged, sustained not burst; prefix-cache OFF / distinct prompts per the prereg
  confounds). This is the **delivery number**.
- **(b) Quality** — **same CIPHER-engaged config**: WikiText-2 PPL Δ (50×2048 = 102,400 tok), MMLU
  500q (the locked eval sets), KL vs fp16 ref. KL=0-on-output asserted on declined layers
  (down_proj/lm_head); engaged layers reported at the measured bar.
- **(c) max_n / shape evidence** — confirm the big GEMMs (N=4096, N=14336) actually went FP8 by
  **direct shape log** (§7), closing the probe's count-identity-only residue.
- **(d) engagement-integrity assertion (measurement-integrity guard).** The MFU run (B=64 forward,
  `call->n` in the thousands) is guaranteed to engage. The **quality run is a separate execution with
  *small* inputs** — PPL windows are 2048 tokens (`call->n=2048>64`, fine), but **MMLU 0-shot prompts
  can be short**, and *any* eval call landing at `call->n ≤ 64` silently runs Marlin/fp16, **not FP8**.
  Then PPL/MMLU could "pass the bar" while partly or wholly measuring the **fp16** path — the inverse
  of the apples-to-oranges trap (FP8 simply not firing in the quality run). **Guard:** capture the §7
  engaged/declined per-class telemetry in **both** the MFU and the quality runs, and assert
  **(i) engaged-count > 0 during the quality run** and **(ii) the engaged set (per-class FP8 vs
  declined) is identical across both runs.** That telemetry-derived identical-engaged-set is the
  *airtight* proof of "one config" — stronger than asserting it by construction (same binary/env).

---

## 7. Closing the `max_n` telemetry residue (build step 3)

The probe could only argue big-GEMM inclusion from the **count identity** (3375) because the
classifier's `max_n` summary didn't emit on the short runs (`probe report:60-64`). The infra exists:
`cipher_workload_detect.cpp:246-252` already tracks `max_m/max_n/max_k` atomics. This build adds the
actuator's **own** per-shape evidence so inclusion rests on a *direct read*, not an identity:
- the FP8 actuator maintains a small set of distinct engaged `(M,N,K)` tuples + an engaged-`max_n`,
  emitted at teardown (atexit diag, like `matmul_dispatch.c:31-38`), and per-class counters
  (engaged vs declined-down_proj vs declined-lm_head);
- one optional `cipher_dbg` of `(M,N,K)` on the first engaged FP8 call per distinct shape.

Result: the report shows "FP8 engaged on N=14336 (gate/up) and N=4096 (q/o); declined K=14336
(down_proj) and M=32000 (lm_head)" by direct log.

---

## 8. fp16-in / fp16-out preservation + the decline-to-fp16 path (summary)

- **Engage:** read `call->B` (fp16/bf16) → per-token E4M3 quant inline; FP8 GEMM with cached
  per-channel weight E4M3; write fp16/bf16 into `call->C`. `Ctype` unchanged. App sees identical
  dtype, never knows FP8 happened.
- **Decline (per-layer):** `return CIPHER_RT_MATMUL_PASSTHROUGH` → substrate calls the **real
  `cublasGemmEx`** with the original 18 args (`matmul_dispatch.c:127-132`) → byte-identical fp16/bf16,
  KL=0. Used for down_proj (policy-dependent) and lm_head (always), and for any shape the FP8 path
  can't service.
- **Disabled (`CIPHER_FP8` unset):** `maybe_handle_fp8` returns PASSTHROUGH on the first line
  (`env_on` idiom, `marlin_actuator.c:130-131`) ⇒ **byte-identical to CIPHER-absent** (Mem #16
  worst-case == current behavior).
- **Error:** if an engaged FP8 dispatch fails, `return CIPHER_RT_MATMUL_ERROR` → substrate falls to
  real cuBLAS (logged). Safe fallback, never a wrong answer.
- **HARD STOP (Mem #11):** if the workload-level gate shows any engaged layer's output diverges beyond
  the recorded bar, the run STOPS and the config is rejected — same enforcement point Marlin uses
  (workload-test gate, not in-band per-matmul).

---

## 9. Default-OFF, additive, anchors-unchanged (Mems #16/#13/#24)

- New env `CIPHER_FP8` (+ `CIPHER_FP8_VERBOSE`, `CIPHER_FP8_POLICY`), `env_on` idiom ("on"/"1"/"ON");
  default-OFF.
- New TUs only (`cipher_rt_fp8_actuator.*`, `cipher_rt_fp8_engine.*`); registration in
  `cipher_rt_fp8_init()` called from `cipher_inject.c` next to the other actuator inits. **Marlin and
  the shim are not modified** (the only shim-adjacent change is the optional §7 teardown diag, which
  is additive and OFF-safe).
- **No `.so` rotation this run.** Anchors (`cipher_rt_phase4 ed130e7`/`d7-rh1-close` + `01d4effb`,
  kmod `0.7.0`, evidence) stay UNCHANGED until a *later* close gate. The close (every-prior-model
  KL=0 OFF byte-identical + 30-min N=128 soak + tag) is the **follow-on, not this run**.

---

## 10. Build order (after approval) + STOP

1. **Feasibility-first:** standalone probe — does cu13 cuBLASLt do correct per-channel/per-token
   (rowwise) FP8 on this H100? (§5.3). Pick Option 1 or fall to Option 2. *Gate the rest on this.*
2. FP8 engine TU: per-channel weight prequant (static, cached, §4) + per-token activation quant
   (inline, §5.2) + the FP8 GEMM (fp16/bf16 out).
3. FP8 actuator TU: gate (large-M, §2) + per-GEMM decline predicate (down_proj/lm_head via
   registry/classifier, §3.2) + policy switch (§3.3) + per-shape/`max_n` telemetry (§7); register at
   priority 20.
4. Wire `cipher_rt_fp8_init()` into `cipher_inject.c`. Build. Confirm OFF == byte-identical.
5. Measure §6 (one config): MFU vs 989 @700W + PPL/MMLU/KL same config + max_n shape evidence.

**Measurement gate (NOT a close):**
- CIPHER-engaged MFU ≥ 85% vs 989 @700W **AND** quality at the recorded bar, one config →
  **DELIVERY DEMONSTRATED.** Report it (close is the follow-on).
- < 85% or quality miss → report the measured CIPHER-engaged number + the binding constraint
  (large-M FP8 GEMM rate / per-channel-quant overhead at the intercept / coverage). Honest waypoint.
- FP8-at-intercept doesn't work (substitution breaks output, or large-M FP8 through cuBLASLt is
  slower than the path it replaces) → report it as a real finding about the intercept.

The 0.3%→0.37% bar amendment, **if** Anil pre-authorizes policy (B), is recorded with rationale
**before** any MFU+quality claim.

**Commit (post-build only):** as Anil `<anil.0666369@gmail.com>`, no co-author trailer. **This memo is
not committed** — it is written for review.

---

### STOP — awaiting Anil's approval of this memo before any code.

Open decisions for approval:
1. **Policy default:** `drop_downproj` (within locked 0.3% bar, MFU-gate risk) vs pre-authorize
   `all_fp8` + the 0.3%→0.37% amendment (MFU-safe, quality-amendment). (§3.3)
2. **Per-channel feasibility fallback:** OK to spend Option-2 (custom epilogue) effort if cuBLASLt
   rowwise FP8 isn't available/correct on this build? (§5.3)
3. **Train-step scope:** confirm v1 = static-weight forward GEMMs only; train step is a firing+MFU
   demo, not converged-FP8-training. (§4)
