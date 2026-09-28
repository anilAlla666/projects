# CIPHER MFU DELIVERY — LANE 2 PROPER (2026-06-10)

## STATUS: STOPPED AT PHASE 0 (GO/NO-GO) — surfaced to Anil. **No kernel built.**

The Rule-4 closed-form + iso-clock proxy say the FP8-GEMM-substitution ceiling on this workload is **below the 85 %
target at quality**. Per the binding no-auto-degrade gate (Phase 0c: "if iso-clock lands materially below 85 % …
STOP, surface to Anil … do not build the kernel into a ceiling that misses"), Phases 1–3 were **not entered**. This
is a valid DONE: *Phase 0 resolved → STOP-surfaced.*

**Phase-4 verdict (written plainly): NO — one config does not reach ≥85 % MFU at ≤0.3 % PPL via FP8 GEMM
substitution on eager vLLM Mistral-7B prefill.** The measured ceiling and why it misses are below; options are for
Anil. No silent scope cut, no kernel built into a missing ceiling.

Frozen anchor `libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — **entry == exit, byte-identical, never
rebuilt** (no successor artifact was produced because the gate said STOP before Phase 3). Clocks reset at exit.
MFU = analytic FLOPs ÷ 989.5 TFLOP/s bf16 dense (prefill 2·N·T, N=7,241,732,096), attention excluded ⇒ lower bound.

---

## PHASE 0a — CLOSED-FORM REACHABILITY (Amdahl)

e2e prefill MFU lift from speeding GEMMs by factor `s`: `new_MFU = V · 1/(g/s + (1−g))`, with vanilla iso-clock
prefill MFU `V≈0.60` and GEMM kernel-time share `g` (measured ≥ **0.646**, mfu_audit e3 profile, a lower bound):

| fp8_speedup s | e2e MFU @ g=0.646 | @ g=0.70 |
|---|---|---|
| 1.4 | 0.736 | 0.750 |
| 1.6 | 0.792 | 0.814 |
| 1.8 | **0.842** | 0.871 |
| 2.0 | 0.886 | 0.923 |

**85 % needs fp8_speedup ≈ 1.8–2.0× at g=0.646** (≈1.7–1.8× at g=0.70). That is the bar the kernel's GEMM speedup
must clear for the *e2e* number to reach target.

## PHASE 0b — ISO-CLOCK PROXY (the decisive measurement; existing per-tensor FP8 engine, frozen anchor)

eager vLLM Mistral-7B prefill 8×2048×10, vanilla vs `CIPHER_FP8=1` (per-tensor E4M3, via `CUDA_INJECTION64_PATH`,
`CIPHER_VOLT=off`, `VLLM_PLUGINS=""`), a tight 0.2 s background relock loop targeting 1455 MHz (the plain `-lgc`
lock releases mid-run on driver 580 — Lane-1 finding — and FP8's lower power lets it boost; the loop fights that).
**FP8 held exactly 1455/1455/1455 (0 W headroom issue, 548 W); vanilla power-limited to ~1440 median (632 W — the
loop could not hold vanilla at the 1455 ceiling because it's at the power cap).** The 15 MHz (1.0 %) residual gap
is handled by clock-normalizing FP8 *down* to vanilla's 1440 — a correction that is **conservative for the STOP**
(a power-limited, slightly-slower vanilla inflates the measured FP8/vanilla ratio, so the true iso-clock speedup is
≤ the 1.32× reported).

| Arm | iso-clock MFU (2 reps) | clk med | power | default-clock MFU (unpinned) | default clk |
|---|---|---|---|---|---|
| vanilla | **0.603 / 0.602** | 1440 | 632 W | 0.602 / 0.599 | 1455–1470 |
| per-tensor FP8 | **0.812 / 0.800** | 1455 | 548 W | **0.827 / 0.828** (mean 0.828) | 1680–1695 |

- **Iso-clock e2e prefill MFU (per-tensor, clock-normalized to vanilla's 1440): 0.797.** Compute-only e2e speedup
  **1.32×** ⇒ implied per-tensor FP8 **GEMM speedup ≈ 1.61×** (from `s = g/(1/1.32 − (1−g))`, g=0.646).
- **Default-clock (power-relief): 0.828.** FP8 draws ~90 W less → SwPowerCap eases → clock rises 1455→1695. This is
  a *real* product benefit (the GPU genuinely clocks higher within the 700 W cap when running FP8), so the
  product-realistic number is 0.83 — but it is still below 85 %, and it is the *per-tensor* (quality-failing) path.
- Engagement confirmed: 5 FP8 ENGAGED shapes, 0 substrate lines in vanilla (clean control).

## PHASE 0c — VERDICT: STOP

**The decisive fact needs no assumption about the rowwise kernel: the fastest available FP8 path (per-tensor
E4M3, the vendor-fused cuBLASLt scalar GEMM) already reaches only 0.80 iso-clock / 0.83 default-clock e2e prefill
MFU — below the 85 % target.** Since per-tensor is the *upper bound* on any FP8 substitution speed here, the
quality-passing **rowwise** kernel (per-channel weight + per-token activation, the only PPL-passing scheme,
contract `LANE2_KERNEL_CONTRACT.md`) can only be **≤ per-tensor** and therefore also misses — its ceiling is
estimated ~0.75–0.79 iso-clock / ~0.80–0.82 default-clock. Two concrete reasons rowwise ≤ per-tensor (panel-refined,
strongest first): (1) on this cu13 cuBLASLt rowwise is **NOT_SUPPORTED as a fused path** (`cipher_rt_fp8_engine.cpp:5-8`,
measured 2026-05-30), so it must run on a *less-vendor-tuned* `torch._scaled_mm`/CUTLASS kernel; (2) the quality
config **excludes lm_head** (less FP8 coverage than the per-tensor proxy, which quantizes it — proxy shape #4
m=32000), so the proxy is an even more generous upper bound; the per-row/per-col dequant epilogue is a real but
second-order extra cost. The measured per-tensor GEMM speedup (~1.61×) is already below the ~1.8–2.0× the closed
form requires for 0.85.

**Mechanism (WALL): the GEMM-share Amdahl wall.** GEMMs are only ~65 % of prefill kernel time; the ~35 % non-GEMM
(elementwise, RMSNorm/SiLU, rotary, attention, copies) does not speed up under FP8 GEMM substitution, capping the
e2e lift. The operative limit is the **achievable** GEMM speedup: measured ~1.61× (per-tensor) → e2e 0.80;
reaching 0.85 needs ~1.8–2.0×, which FP8 GEMM substitution does not deliver here. (Asymptotically, even a *free*
GEMM (s→∞) leaves the ~35 % non-GEMM time, so e2e speedup is bounded by 1/(1−g)=1/0.354≈2.8× — but that is an
unreachable asymptote and MFU is capped at 1.0 regardless; the binding constraint is the realistic ~1.6×.)
**85 % at quality is outside the reachable band on this workload with FP8-GEMM-substitution alone.**

Per the no-auto-degrade rule, the kernel was **not built**. Confidence on the STOP: **high** — it rests on a
direct iso-clock measurement of the *fast* path (0.80) that already misses, plus the strict-slower argument for the
quality path, plus the closed form, three independent lines agreeing.

## OPTIONS FOR ANIL (no silent scope cut)

1. **Accept the measured ceiling as the honest deliverable:** FP8 prefill MFU **~0.80 iso-clock / ~0.83
   default-clock** (per-tensor; rowwise ~0.75–0.82), i.e. **Goal-3's 85 % is not reachable by FP8-GEMM
   substitution on eager-vLLM Mistral-7B prefill** — the GEMM-share Amdahl wall is the reason. Report the number,
   close the goal as "ceiling characterized, target missed by ~3–10 pts."
2. **L6 / non-GEMM stack** (RMSNorm+SiLU fusion + tactic-pin + PDL): the only lever that attacks the ~35 %
   non-GEMM share the Amdahl wall is made of. Ledger estimate **+3–8 pts, low–med confidence**
   (`D9_PERSISTENT_KERNEL_MEMO.md:83-85`). **Caveat (Lane-1/mfu-audit):** the fusion was never built in the
   substrate (POC monkeypatch only, evaporates under graph capture). Closing even half the non-GEMM share could
   push e2e toward 0.85, but confidence is low and it is net-new engineering.
3. **Re-evidence the 85 % target itself:** the D.9 "88–98 %" that anchored the target was vLLM *per-tensor*
   *cudagraph* reachability (derated ×0.886), never a delivered co-measured per-channel number (gate-0/Lane-1
   findings). The target may have been premised on a reachability figure that does not survive the quality +
   iso-clock + single-config constraints. Worth confirming the target is real before more engineering.
4. **Different workload regime:** larger batch/seq raises GEMM efficiency and share somewhat; but g is already
   ~0.65 and the marginal room is small — unlikely to add the needed ~5–10 pts alone.
5. **0.37 % PPL amendment:** does NOT help — it is a quality knob, orthogonal to the MFU ceiling.
6. **Build the rowwise kernel anyway** purely to capture the product-realistic default-clock artifact (~0.80–0.82,
   power-relief included), knowing it misses 85 % — only if the artifact itself has value to you.

## Pre-reg diffs
- Phase 0a expected the 85 % band to need ~1.8–2.0× → **confirmed**.
- Phase 0b expected (from gate-0 confound math) iso-clock ~0.71 → **measured higher, 0.80** (per-tensor is faster
  iso-clock than the crude gate-0 linear scaling implied), but **still below 85 %** → STOP stands.
- The STOP outcome itself was pre-registered as the no-auto-degrade branch; it fired on measured evidence.

## Integrity (exit)
- Frozen anchor md5 entry == exit `2edba0d2136f8ede4713d90a8f7cd55f` — never rebuilt (STOP before Phase 3); **no
  successor artifact created**. No new kernel, no vLLM-source edit, no monkeypatch, no plugin path.
- Scratch only in `mfu_lane2/`; vanilla arms 0 substrate lines; FP8 via injection only. Clocks reset (`-rgc`) at
  exit; relock loop killed; 0 leftover compute procs.
- Every number → `mfu_lane2/phase0_result.json` + per-run `iso_*.json` / `isoL_*.json`. No retracted number; the
  MFU figures are labeled per-tensor / quality-failing / clock-regime-explicit, 85 % only as the target.

## Verification panel (mandatory adversarial pass — 3 read-only dimensions; record `verification_panel.json`)

**Result: 0 material + 4 minor + 4 note — all applied. The STOP verdict was found OVER-DETERMINED and sound.**

| Dim | Passed | Findings → action |
|---|---|---|
| 1 ISO-CLOCK VALIDITY | 15 (FP8 pinned 1455/1455/1455 confirmed; vanilla a genuine 0-substrate control; every derived number — 0.6026, 0.8058, 0.7975, 1.323, s=1.61 — recomputed exactly; clock-norm is fair+conservative; **2-rep spread (1.5 %) is an order of magnitude smaller than the 5–13 pt gap to 0.85**, even the most-PASS-favorable rep combo gives s=1.64 < 1.8) | iso-clock header overstated "both held 1455" → reworded (FP8 pinned, vanilla power-limited ~1440, gap conservative for STOP); default-clock cell 0.828/0.826 → 0.827/0.828 |
| 2 STOP vs GREEN | 10 (Amdahl table reproduces; STOP robust across g=0.646–0.72; the "rowwise ≤ per-tensor" direction sound) | mechanism reframed — rowwise ≤ per-tensor rests on (1) rowwise off the fused cuBLASLt path onto a less-tuned kernel + (2) lm_head excluded, not the prologue cost; **strengthened: the STOP holds even if rowwise *equals* per-tensor, since per-tensor itself (0.80) already misses** |
| 3 HONESTY + INTEGRITY | 17 (frozen anchor md5 live-verified, no successor .so built, both clock regimes on every number, STOP surfaced with 6 options no silent degrade, per-tensor-is-a-speed-proxy-that-fails-PPL clear, 0 procs, clocks reset, no git writes, scratch confined) | default-clock cell fix (dup of dim1); asymptote sentence "1.69×" confusing (>1 MFU) → rewritten with the correct 1/(1−g)≈2.8× unreachable-asymptote framing + the corrected arithmetic |

**Exit state:** frozen anchor `2edba0d2136f8ede4713d90a8f7cd55f` entry==exit (never rebuilt, no successor); HEAD
`b2304d3`; clocks reset (345 MHz idle); 0 compute procs; relock loop killed; 0 files outside `mfu_lane2/`.
