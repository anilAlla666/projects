# CIPHER MFU DELIVERY — LANE 1 REPORT (2026-06-10)

Three route-deciding measurements, no kernel work. Anchor `libcipher_rt.so` md5
`2edba0d2136f8ede4713d90a8f7cd55f` — **entry == exit** (verified at close). Clocks DEFAULT (unlocked entry+exit,
345 MHz idle). Fork-1/snapshot/mfu_audit/anchor/vLLM-source READ-ONLY; scratch only in `mfu_lane1/`. Every vLLM
lane `VLLM_PLUGINS=""` with stderr substrate-line count reported. MFU = analytic FLOPs ÷ 989.5 TFLOP/s bf16 dense,
**attention FLOPs excluded ⇒ the absolute 1a MFU levels are analytic lower bounds** (the load-bearing quantity is
the V/E tax delta, identical model both arms, so the exclusion cancels). These absolute levels are NOT comparable
to the 88–98 % FP8 figure (different denominator, attention-inclusive).

## THE ROUTE RECOMMENDATION (the point of this run)

**Build Lane 2's first ≥85 % config on EAGER vLLM prefill (or torch prefill), NOT cudagraph vLLM — and wire
interception via LD_PRELOAD-style symbol interposition, with substrate-side interception as Lane 2's first
gate.** Basis, separating what is *measured this run* from what *remains to verify*:

- **Necessary condition MET, measured (1b):** eager vLLM fp16 prefill **calls the public `cublasGemmEx` symbol
  ≥1900×/run, `cublasLtMatmul`=0** (LD_PRELOAD census). The symbol is NOT bypassed at the API level — the nvjet
  kernels are what `cublasGemmEx` launches internally. This is the condition cudagraph mode removes (replays issue
  kernels without re-calling the public symbol), so eager is the right mode.
- **Hosting tax is negligible there, measured (1a):** eager vLLM prefill **+0.86 %**, torch prefill **+1.69 %** —
  neither gates Goal-3. So if interception is wired, the substrate overhead does not eat the FP8 margin.
- **OPEN — substrate-side interception on eager vLLM is NOT proven this run (corrected after panel):** my census
  catches the call via **LD_PRELOAD interposition**; that does NOT prove the substrate's **GOT-patch**
  (`cipher_rt_cublas_shim.c:327`) catches it. In fact the D.9 record's **eager** telemetry run logged the
  substrate's GemmEx interception = 0 ("nvjet bypass"), even though (as 1b now shows) the symbol IS called. The
  resolution is the interception MECHANISM, not cudagraph: a global LD_PRELOAD interposer catches what the
  GOT-patch missed. The substrate's GOT-patch DID work on **torch** (mfu_audit FP8 handled=1125), so the gap is
  vLLM-specific (forked worker / handle / GOT-coverage). **Lane 2's first step: arm FP8 on eager vLLM and confirm
  `fp8_calls_handled > 0` — this run deliberately armed no actuator, so it is unverified here.**
- **cudagraph is a SEPARATE wall** (the FP8 actuator's NVRTC/cudaMalloc are capture-illegal), but it is NOT the
  explanation for D.9's GemmEx=0. The small eager-vs-cudagraph prefill penalty (~1.6 %, prior `d9_s1b_vllm_prefill.json`,
  not measured this run) means eager prefill is a cheap host.
- **Do NOT target training as the first config:** the substrate hosting tax on the LoRA train step is **+150 %
  (2.5× slowdown)** — launch-density, not logging (proven below). Training is the wrong phase for this substrate.

## LANE 1a — SUBSTRATE HOSTING TAX (loaded, zero actuators armed)

Interleaved V/E ×3, default clocks. Engaged = unmodified anchor via `CUDA_INJECTION64_PATH` + **`CIPHER_VOLT=off`**
(disarm: source `cipher_rt_volt.c:344` — UNSET defaults VOLT to ARMED/classifier-self-engage, which is what fired
in the 2026-06-10 contaminated run; `off` ⇒ "staying OFF, classifier override denied"). No actuator env set.

| Workload | Vanilla | Engaged (loaded, 0 actuators) | **Tax** | Engagement-state proof | Verdict |
|---|---|---|---|---|---|
| torch-HF prefill 8×2048×5 | MFU 0.4858 (reps .4859/.4857/.4858) | MFU 0.4777 (.4771/.4777/.4799) | **+1.69 %** | substrate lines 1536–1566; `VOLT: CIPHER_VOLT=off — staying OFF`; `FP8/MARLIN actuator DISABLED`; `DISPATCH LIVE=1`; FP8 counters 0/0/0; "first registration awaited" (0 actuators registered/handled; one "Substitution ENABLED" line = phase-capability declaration, not a fired substitution) | **does NOT gate** |
| eager vLLM prefill 8×2048×10 | MFU 0.6002 (.6009/.5988/.6002) | MFU 0.5950 (.5944/.5966/.5950) | **+0.86 %** | worker stderr: substrate lines ~480; VOLT off; FP8 DISABLED; DISPATCH LIVE=1; MATMUL "first registration awaited" (0 actuators); no MATMUL exit-totals (so substrate-side GemmEx interception count is unestablished here — see route rec). Injection reached the V1 EngineCore worker via env inheritance — no plugin path needed | **does NOT gate** |
| LoRA train B=1 s512 ×40 | 0.1401 s/step (MFU 0.1070) | 0.3506 s/step (MFU 0.0428) | **+150 % (2.5× slowdown)** | substrate lines 23,766; VOLT off; 0 actuators registered | **GATES HARD** |

**Mechanism (measured, not asserted):** per-launch CUPTI + classify overhead. It gates **launch-DENSE** phases
(training: fwd+bwd+optimizer = thousands of small launches/step, ~2.5×, mirroring the ~1.9× eager-decode tax) but
not **launch-sparse large-GEMM prefill** (~1–2 %). **Logging ruled out:** engaged train with stderr→/dev/null =
0.3427 s/step vs →file median 0.3506 — within ~2 % (devnull even slightly faster), i.e. far smaller than the
+150 % tax, so the 23,766 log lines cannot explain it; the per-launch CUPTI+classify work is the cost.
This refines the prereg, which expected train ~1–5 %: **SURPRISE** — training inherits the launch-density tax
(prereg correctly predicted prefill ~1–3 %, both confirmed).

**Disclosed teardown artifact:** the engaged torch and train processes **segfault at process teardown** (anchor
destructor × `CUDA_INJECTION64_PATH` exit-ordering) **after** the timed window closes and the result JSON is
written + printed. Every result above is from a completed window with a written JSON; medians over 3 reps are tight
(torch vanilla σ<0.0001). The vLLM worker is SIGKILLed normally, so teardown is moot there. Not a measurement defect.

## LANE 1b — vLLM GEMM ENTRY-SYMBOL COVERAGE

**(1) Source side — what the substrate intercepts:**

| Symbol | Mechanism | file:line |
|---|---|---|
| `cublasGemmEx` | GOT-patch (`cipher_rt_got_register`) | `cipher_rt_cublas_shim.c:327` |
| `cublasLtMatmul` | GOT-patch (umbrella) | `cipher_rt_cublas_shim.c:331` |
| `cublasLt{ACC,BII,…,HSH,…,ZZZ}Matmul` (17 private dtype variants) | dlsym-hook trampolines; **HSH** is ROUTED (full shim, CIPHER_LT_ROUTE-gated, telemetry+passthrough OFF) | `cipher_rt_cublaslt_variants.c:181-198,236` |
| `machete::mm_dispatch`, FlashAttn/cuDNN attn | GOT-patch (not GEMM-MFU) | `cipher_rt_machete_intercept.c:116`, `cipher_rt_attn_dispatch.cpp:564` |

**(2) Runtime side — scratch ldd-clean census shim (libc-only, no `/dev/cipher`, no dlsym-hook), public entries:**

| Symbol | torch-HF prefill | eager vLLM prefill | intercepted by substrate | covered |
|---|---|---|---|---|
| `cublasGemmEx` | **900** (3 reps×300) | **≥1900** (periodic-flush floor; worker SIGKILLed) | GOT-patch target exists (`cipher_rt_cublas_shim.c:327`); **whether it FIRES on eager vLLM is unverified this run** (D.9 eager telemetry: substrate count=0) | **called: Y; substrate-intercepted: UNVERIFIED** |
| `cublasLtMatmul` | 0 | 0 | GOT-patch target exists (`:331`) | not called |
| `cublasGemmStridedBatchedEx` | 0 | 0 | no | not called |
| nvjet_sm90_**hsh**_* compute kernels | dominant (mfu_audit profile) | dominant | these are the kernels `cublasGemmEx` *launches internally*; entry = GemmEx (called) | n/a (compute kernel, not an entry) |

The dtype-specialized private variants were identified by profiler kernel-name census (`nvjet_sm90_hsh_*` = HSH =
fp16-in/fp32-scale/fp16-out), not the dlsym shim: a global dlsym interposer **hung CUDA lazy resolution** (caught
and dropped; the public-symbol interposer is sufficient and answers the deciding question).

**Reconciliation of the D.9 "vLLM GemmEx=0" probe (CORRECTED after panel — the cause is the interception
mechanism, not cudagraph):** the panel found D.9's substrate-count=0 was logged in an **eager** D.9 run too
(attributed there to "nvjet bypass"), not only cudagraph. My LD_PRELOAD census shows the public symbol IS called
in eager vLLM (≥1900). So the two facts coexist because they measure different interception points: **LD_PRELOAD
interposition (my census) catches the call; the substrate's GOT-patch did not** (in D.9's eager run, count=0).
The GOT-patch works on torch (mfu_audit FP8 handled=1125) but evidently missed vLLM's eager GemmEx calls —
plausibly because vLLM's forked V1 worker resolves cublas through a GOT the patcher didn't walk, or a separate
handle. cudagraph is a separate, real wall (replays would drop the call count and the FP8 actuator is
capture-illegal), but it is NOT the explanation for the D.9 eager zero. **What Lane 1b establishes: the symbol is
called in volume on eager vLLM (so interception is achievable, via LD_PRELOAD if not GOT-patch); what it does NOT
establish: that the current substrate intercepts it on vLLM — Lane 2 must confirm with FP8 armed.**

## LANE 1c — THE QUALITY-PASSING RECIPE (Lane-2 kernel contract)

Full contract: `l1c_recipe/LANE2_KERNEL_CONTRACT.md` (RECORD vs ASSUMPTION-FROM-VLLM-SOURCE per line; 5-agent
extraction + adversarial verify). Summary + the three honesty flags:

- **Recipe (RECORD):** per-output-channel weight scale + per-token dynamic activation scale, rowwise **E4M3**;
  `w_scale=amax(|W|,dim=in)/448` fp32 clamp 1e-12; `a_scale=amax(|x_row|)/448` dynamic; `torch._scaled_mm(x8, W8ᵀ,
  scale_a, scale_b, out_dtype=fp16, use_fast_accum=True)`; bias post-GEMM fp16; **224 linears (q,k,v,o,gate,up,down
  × 32), lm_head EXCLUDED**. Quality: PPL +0.37 %, MMLU −0.4 pp, KL 0.00357 nats on WikiText-2 50×2048 windows.
- **FLAG 1 — not one config:** +0.37 % is HF per-channel; 88–98 % MFU is **vLLM per-tensor** derated ×0.886
  (where 0.886 = 1265/1427 TFLOP/s from an un-logged prior single-shape GEMM check). Strict arithmetic gives
  ~81–87 % for the per-channel band (0.886×91.7…98.3); "88 %" is a ~1 pp generous lower bound. Never co-measured.
  No single config has shown ≥85 % at ≤0.3 %.
- **FLAG 2 — fails the locked bar:** +0.37 % is over the HARD 0.3 % bar (`WITHIN_BAR:false`); passes only the
  pre-authorized 0.37 % amendment. The unamended-pass config is per-channel **drop_downproj** (192 linears,
  +0.2465 % PPL, modeled projMFU ~80–86 %).
- **FLAG 3 — Path B never built; accumulator precision UNRESOLVED.** Building the CIPHER-native rowwise kernel,
  pinning the accumulator, and co-measuring quality+MFU in ONE engine is exactly Lane 2's job.

## Pre-reg diffs

- 1a prefill ~1–3 % → **confirmed** (torch 1.69 %, vLLM 0.86 %). Train ~1–5 % → **SURPRISE, +150 %** (launch-density
  transfer; mechanism measured). 1a vLLM-injection-reaches-worker → **confirmed** (no WALL; env inheritance works).
- 1b "vLLM GEMM entry UNKNOWN, D.9 probe said GemmEx=0" → **partially resolved: eager vLLM CALLS GemmEx in volume
  (≥1900, LD_PRELOAD census); but the D.9 substrate-side zero was an EAGER run too (not cudagraph), so the gap is
  the interception mechanism (GOT-patch missed vLLM; LD_PRELOAD catches). Substrate-side vLLM interception remains
  to be confirmed by Lane 2 (this corrects the report's first-draft cudagraph attribution).**
- 1c "expected per-channel W + per-token A E4M3" → **confirmed as the scheme**, plus the three unflagged-until-now
  honesty issues (not-one-config, fails-locked-bar, Path-B-never-built).

## Integrity (exit)

- Anchor md5 entry == exit `2edba0d2136f8ede4713d90a8f7cd55f` (verified at close). cipher_rt_phase4 HEAD unchanged
  (`b2304d3`); fork-1/snapshot/anchor/vLLM-source unedited.
- Clocks default throughout (never locked); `-rgc` no-op confirmation at exit; 0 leftover compute procs.
- **Disclosed discipline slips (panel-caught, both remediated/benign):**
  (1) the torch symbol-census run wrote one scratch JSON (`m_torchpf_cen.json`) into `mfu_audit/` because the
  reused harness's default result path points there; it overwrote no existing artifact and was **relocated to
  `mfu_lane1/l1b_symbols/`** — `mfu_audit/` is now byte-unchanged this run (0 files modified).
  (2) a subagent ran a `git stash` push+pop in `cipher_rt_phase4/.git` during the run (09:47:38): **HEAD,
  branches, refs, and the anchor .so are all unchanged** (0 stash entries remain, 0 refs moved); 4 dangling
  objects left in place (pruning would be a further write to the protected repo).
- Census shim ldd = libc only; its single "cipher" string is its own output path (no substrate, no `/dev/cipher`,
  no dlsym hook). A first census-shim version with a global dlsym hook hung CUDA and was dropped before any
  measurement.
- Every number → a `mfu_lane1/` JSON: `l1a_tax/lane1a_tax.json`; **`l1b_symbols/census_torch.json` (torch) and
  `l1b_symbols/census.json` (vLLM — written to the shim's default path because RV_OUT collides with the harness's
  token-count env)**; `l1c_recipe/extraction.json`. No retracted number used (verified absent: 3.617×, 14×, 2.96×,
  7.43×, 0.127 %, unscoped 57.3 %, and 85 % appears only as a Lane-2 target).

## Verification panel (mandatory adversarial pass — 4 read-only dimensions; record `verification_panel.json`)

**Result: 4 material + 4 minor + 5 note findings — all applied.** The two material HONESTY findings reshaped the
route recommendation; the two material INTEGRITY findings are disclosed above.

| Dim | Passed | Findings → action |
|---|---|---|
| 1 NUMBERS | 17 (every 1a tax recomputed from per-rep JSONs to the printed digit; 1b census counts; 1c +0.37 %/WITHIN_BAR:false/224-linears/0.886; retracted-number scan clean; report never states 88–98 %/+0.37 % as delivered) | substrate-line counts → ranges; "0 ENGAGED/substitution" → "0 actuators registered/handled (one phase-capability line)"; 88–98 % → ~81–87 % arithmetic note; MFU absolute → lower-bound note **(all applied)** |
| 2 ENGAGEMENT | 14 (each engaged run loaded+VOLT-off+actuators-DISABLED from its own stderr; each vanilla 0 lines; shim ldd-clean, no /dev/cipher; injection reached the V1 worker) | census filename citation fixed (`census.json` for vLLM); "Substitution ENABLED" clarified as phase-capability not a fired substitution **(applied)** |
| 3 HONESTY | 16 | **MATERIAL ×2:** (a) cudagraph misattribution of D.9 GemmEx=0 → corrected to interception-mechanism (D.9 eager run saw substrate=0 too); (b) "coverage exists" conflated called-vs-GOT-intercepted → route rec now separates "symbol called (proven)" from "substrate intercepts (unverified, Lane-2 gate)". Minors: "identical"→"within ~2 %"; eager-vs-cudagraph prefill penalty cited as prior data **(all applied)** |
| 4 INTEGRITY | 7 (anchor md5 live-verified; no vLLM-source edit; clocks default; no leftover procs; census/run confined to mfu_lane1/) | **MATERIAL ×2:** mfu_audit scratch-write (`m_torchpf_cen.json`, relocated) and in-run `git stash` push+pop → both disclosed in Integrity; HEAD/anchor/refs verified unchanged **(applied)** |

**Exit state (at close):** anchor md5 `2edba0d2136f8ede4713d90a8f7cd55f` (entry==exit); HEAD `b2304d3`; clocks
default/unlocked (345 MHz idle), `-rgc` no-op; 0 compute procs; `mfu_audit/` 0 files modified this run.
