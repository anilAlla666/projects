# CIPHER MFU DELIVERY — LANE 2 GATE-ZERO REPORT (2026-06-10)

## VERDICT: (i) THE SUBSTRATE INTERCEPTS EAGER vLLM PREFILL. **Lane 2 host = eager vLLM prefill.**

One bit read from the substrate's own behaviour: with the existing FP8 actuator armed (`CIPHER_FP8=1`) via
`CUDA_INJECTION64_PATH` into the eager vLLM V1 worker, the actuator **registers, prequantizes weights, and engages
real GEMM shapes** — `handled > 0`. No new kernel, no anchor change, no plugin path. G0-C (interception-fix) was
**not run** because the verdict is (i), not (ii).

> **Label on every MFU number in this report:** engagement-demo, per-tensor E4M3 path, which **FAILS the PPL bar
> (+0.567–0.72 %)**. This run makes **no quality claim and no deliverable-MFU claim**. The quality-passing kernel
> is Lane 2 proper, now gated to **eager vLLM** by this result.

Anchor `libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — **entry == exit** (verified at close). Clocks
DEFAULT throughout. Read-only fork-1/snapshot/mfu_audit/mfu_lane1/anchor/vLLM-source; scratch only in
`mfu_lane2_gate0/`. Every vLLM lane `VLLM_PLUGINS=""`; vanilla arm proven 0 substrate lines. VOLT off, Marlin/
Koopman unset — FP8 isolated. Interleaved vanilla/FP8 ×3. MFU = analytic FLOPs ÷ 989.5 TFLOP/s bf16 dense,
attention excluded ⇒ absolute levels are lower bounds; the load-bearing quantity is **engagement**, not the MFU.

## G0-A — TORCH POSITIVE CONTROL (rig known-good)

torch-HF prefill 8×2048×5, vanilla vs `CUDA_INJECTION64_PATH + CIPHER_FP8=1`, in-process `RTLD_NOLOAD` counter read:

| Rep | fp8 handled | total | skipped | MFU vanilla→fp8 |
|---|---|---|---|---|
| ×3 | **1125 / 1125 / 1125** | 1350 | 225 | 0.486 → 0.546 (engagement-demo, clock-confounded) |

**Reproduces mfu_audit exactly** (handled=1125/1350, 5 ENGAGED shapes). Vanilla: 0 substrate lines. Rig is valid →
proceed to vLLM. (Engaged runs segfault at process teardown — anchor dtor × injection exit-ordering — **after** the
timed window + result JSON are written; not a measurement defect, same artifact documented in Lane 1.)

## G0-B — vLLM THE DECIDER

eager vLLM prefill 8×2048×10, `VLLM_PLUGINS=""` both arms; engaged = `CUDA_INJECTION64_PATH + CIPHER_FP8=1`
reaching the EngineCore worker via env inheritance. Read from **worker stderr** (mid-run, survives the worker
SIGKILL):

**Engagement proof (the deliverable):**
- `MATMUL: actuator 'FP8_E4M3' registered at priority 20 (slot 0/1)` — the actuator entered the dispatch chain
  (contrast Lane 1's FP8-off run: "first registration awaited", 0 actuators).
- `FP8: engine init OK (cuBLASLt+driver+nvrtc resolved)`.
- **5 `FP8: ENGAGED shape` lines** per rep (×3): 4 linear projections at batch(n)=16384 — out(m)=6144/k4096
  (fused qkv), 4096/4096 (o), 28672/4096 (gate_up), 4096/14336 (down) — **PLUS lm_head out(m)=32000/k4096 at
  batch(n)=1024**. (lm_head being engaged is consistent with — and reinforces — the per-tensor-all-layers
  +0.567 % PPL-FAILING label: the contract excludes lm_head precisely because it is disproportionately costly;
  this engagement-demo path does not.)
- **129 `FP8: prequantized W=…` lines** (r1) — weights actually quantized = 32 layers × 4 fused projections
  (128) + 1 lm_head = 129.
- substrate lines: **804 (fp8) vs 0 (vanilla)**; vanilla ENGAGED lines = 0 (clean control).

**`handled > 0` is established — and the ENGAGED line provably implies handling (source-verified, not inferred):**
in `cipher_rt_fp8_actuator.c:maybe_handle_fp8`, the `FP8: ENGAGED shape` log (`fp8_log_shape`, line 49) is reached
**only on the success path** — after `cipher_rt_fp8_engine_matmul()` returns 0. If the FP8 GEMM had declined, the
code logs `FP8: matmul declined — passthrough`, increments `g_calls_skipped`, and returns `PASSTHROUGH` **before**
the ENGAGED line; the very next statement after `fp8_log_shape` is `g_calls_handled++` / return `HANDLED`. So each
ENGAGED line ⟺ a real FP8 GEMM executed and was counted handled. Likewise `FP8: prequantized` prints only after
`quantize_weight` succeeds. With 5 unique shapes engaged and 129 weights quantized, `handled` is large (every
covered prefill GEMM × 10 reps), bounded well above zero.

The exact final `fp8_calls_handled` integer is **unrecoverable via the counter dump** this run: the worker is
SIGKILLed before the atexit dump, and the `CIPHER_RT_COUNTER_DUMP_PATH` file written is the **"install"-tag**
snapshot (pre-GEMM, all-zero; pid == EngineCore worker). Recoverable in Lane 2 via SIGUSR1-before-kill; the exact
integer is not needed for the verdict (handled>0 is what gate-0 asks).

**MFU vanilla vs FP8 — with the power-relief confound made visible (NOT hidden):**

| Rep | vanilla MFU / clk / power | fp8 MFU / clk / power |
|---|---|---|
| r1 | 0.602 / 1470 MHz / 639 W | 0.826 / 1695 MHz / 594 W |
| r2 | 0.599 / 1455 MHz / 636 W | 0.837 / 1680 MHz / 600 W |
| r3 | 0.602 / 1455 MHz / 637 W | 0.826 / 1695 MHz / 597 W |

**The 0.60 → 0.83 jump is CLOCK-INFLATED, not a clean FP8 efficiency number.** FP8 draws ~40 W less → SwPowerCap
eases → SM clock rises ~230 MHz (1460 → 1690) → both tok/s and the bf16-denominator MFU rise. This is the exact
power-relief confound from the MFU audit; it is reported, not used as a Goal-3 result. The honest engagement signal
is the ENGAGED/prequantized lines, not this MFU. **The same confound is present and LARGER in the torch G0-A
control** (vanilla ~1530–1620 MHz/665 W → fp8 ~1935–1950 MHz/600 W: −65 W, +330 MHz), so neither G0-A nor G0-B MFU
is an iso-conditions number — both are engagement demos only.

## G0-C — NOT RUN

Verdict is (i): the substrate already intercepts eager vLLM. No interception fix is needed, so no scratch
LD_PRELOAD interposer was built.

## Reconciliation with Lane 1 (this run RESOLVES Lane 1's open question)

Lane 1 (FP8 **off**, observe-only) saw `MATMUL: first registration awaited` with 0 actuators and — citing D.9's
eager telemetry of substrate-count 0 — cautiously concluded the GOT-patch *might miss* vLLM, leaving
"substrate-intercept UNVERIFIED" as gate-0's job. **Gate-0's answer to the question it asked: with FP8 armed, the
substrate's FP8 actuator demonstrably FIRES on the eager V1 EngineCore worker** (registered + 5 shapes + 129
weights, source-proven to imply real handling). So the operative bit — *can the shipped substrate intercept and
substitute eager vLLM prefill GEMMs?* — is **YES**. Lane 1's "first registration awaited" was simply the
no-actuator-armed state, not evidence of a miss.

**What this does NOT resolve (honesty, panel-corrected):** it does not explain *why* D.9's eager probe recorded
substrate-count 0. That contradiction is left OPEN, not "superseded" — plausible (unverified) causes: D.9 read a
different counter (the `cublasGemmEx`-shim's own call counter, which is a distinct interception point from the
matmul-dispatch *actuator* path that this run exercised), or a different vLLM version/config. This run proves the
actuator path works on eager vLLM today; it does not back-explain the D.9 telemetry. (The cudagraph path remains a
separate, real wall — the FP8 actuator's NVRTC/cudaMalloc are capture-illegal — but it never bore on the eager
question.)

## Lane 2 host decision + next concrete step

- **Host: eager vLLM prefill.** Interception is proven there with the shipped substrate (no LD_PRELOAD workaround,
  no GOT-coverage substep needed), and Lane 1 measured its hosting tax at **+0.86 %** — negligible against the FP8
  margin.
- **Next step (Lane 2 proper):** build the **quality-passing** per-channel-W + per-token-A rowwise E4M3 kernel
  (the `LANE2_KERNEL_CONTRACT.md` recipe) on this eager-vLLM host, co-measuring **quality (≤0.3 % PPL HARD) AND
  MFU in ONE config** — the gap the entire D.9 record left open. First sub-step: re-run this exact armed config
  with a SIGUSR1-before-kill to capture the exact `fp8_calls_handled` integer, then swap the per-tensor engine for
  the per-channel recipe and run the contract's WikiText-2 quality gate.

## Pre-reg diffs
- G0-A expected handled ~1125-scale → **confirmed exactly (1125)**.
- G0-B expected the deciding bit → **handled>0 on vLLM (verdict i)**; the prereg flagged either outcome as valid;
  this corrects Lane 1's cautious lean toward (ii).
- Power-relief confound expected visible → **confirmed and quantified** (−40 W, +230 MHz).

## Integrity (exit)
- Anchor md5 entry == exit `2edba0d2136f8ede4713d90a8f7cd55f`; cipher_rt_phase4 HEAD unchanged (`b2304d3`). No
  anchor rebuild, no new kernel, no vLLM-source edit, no plugin path. Scratch only in `mfu_lane2_gate0/`
  (mfu_audit/mfu_lane1 cited not written — 0 files modified outside the lane dir).
- **Disclosed (panel-caught):** a subagent ran a `git stash` push+pop in `cipher_rt_phase4/.git` during the run
  (4 dangling objects at 10:54:58); **HEAD, refs, and the anchor .so are unchanged** (0 stash entries remain, 0
  refs moved). Content-neutral; objects left in place (pruning = a further write to the protected repo).
- Clocks default (never locked); `-rgc` no-op at exit; 0 leftover compute procs.
- Every number → `mfu_lane2_gate0/gate0_result.json` + the per-run `torch_*.json` / `vllm_*.json`.
- No retracted number used; the MFU figures (torch 0.55, vLLM 0.83) are explicitly labeled clock-confounded +
  per-tensor + PPL-failing, never as a Goal-3 result. The verdict (handled>0) does not rest on any MFU number.

## Verification panel (mandatory adversarial pass — 4 read-only dimensions; record `verification_panel.json`)

**Result: 1 material + 6 minor + 5 note — all applied. The verdict (handled>0 on eager vLLM) was found
over-determined** (registration + 5 ENGAGED shapes + 129 prequantized weights + source-proof that ENGAGED ⟹ real
handle + the in-process hard counter 1125 on the torch control).

| Dim | Passed | Findings → action |
|---|---|---|
| 1 VERDICT | 11 (5 ENGAGED + 129 prequant in every fp8 rep; lines in the EngineCore WORKER mid-run; vanilla 0/0 clean control; install-dump-zero honestly disclosed; **source-proof ENGAGED⟹handled at `cipher_rt_fp8_actuator.c`**; real cuBLASLt E4M3 GEMM not a stub; torch hard counter 1125 corroborates) | lm_head 5th shape omission → enumeration corrected (note) |
| 2 NUMBERS | 10 (torch 1125/1350/225 exact; vLLM MFU/clk/power all trace to JSONs; confound JSON-backed; 0.826 < 0.85 so no stray 85% claim; retracted-number scan clean) | lm_head/"all batch=16384" error → fixed; |
| 3 HONESTY | 10 | **MATERIAL:** "D.9 superseded" overclaimed → rewritten to leave the D.9 eager-zero contradiction OPEN (different counter/config, unverified), claiming only that the actuator path fires on eager vLLM today. Minors: torch confound now quantified (−65 W/+330 MHz); lm_head PPL-risk noted; verdict humility on "why D.9 saw 0" kept |
| 4 INTEGRITY | 9 (anchor md5 live; no vLLM-source edit; clocks default; no leftover procs; scratch confined) | git-stash in cipher_rt_phase4/.git → disclosed; HEAD/anchor verified unchanged |

**Exit state:** anchor `2edba0d2…` entry==exit; HEAD `b2304d3`; clocks default/unlocked (345 MHz idle), `-rgc`
no-op; 0 compute procs; 0 files modified outside `mfu_lane2_gate0/`.
