# R.A+R.B GOODPUT-AND-RECOVERY CAPSTONE — useful-work-under-faults + dispatch-boundary quarantine

**Date:** 2026-06-09. **Model:** mistralai/Mistral-7B-v0.1 fp16. **Stack:** vLLM 0.20.2 / torch 2.11 / CUDA 13,
real KV cache, real FlashAttention, `enforce_eager` (the only validated host; cudagraph-on hosting is walled at Path-1
— a cuBLAS interceptor never fires under graph replay). **GPU:** H100 80GB `GPU-993dd5d1-…`. **Clock:** requested/locked
1980 MHz; **achieved ~1830 MHz under `SwPowerCap`** during compute (a power-cap, not a fault) → `-rgc` at exit (report
post-reset). **Power:** NVML `power.draw` sampled at 50 ms, trapezoid-integrated over the *measured generate window*
only (model-load excluded). **Decode:** **batch=1** so the single-element GEMM corruption corrupts THE workload (no
7/8-of-batch dilution), greedy (temperature=0, bit-deterministic run-to-run — verified clean-A==clean-B).

**Discipline:** anchor `2edba0d2…` md5 **UNCHANGED entry+exit**; SDC signal = scratch `LD_PRELOAD` cuBLAS shim
**copied from rc_pergpu/sdc_shim.c** and extended **only in rc_ab/** (no edit to the anchor, ra_e2e/, ra_realvllm/,
rc_pergpu/, …); `libcipher_rt.so` NOT loaded into vLLM; `CUDA_INJECTION64_PATH` unset, `VLLM_PLUGINS=""`,
`VLLM_USE_DEEP_GEMM=0`; GPU exclusive at each run; children reaped. **Kmod-decoupled:** the shim was copied from the
R.C `rc_pergpu` shim, which carried a W.6 cohort-registry `ioctl` heartbeat on `/dev/cipher`; that heartbeat is
**DISABLED here** (`RV_HEARTBEAT=0` default, gated out at build) — the shim never opens `/dev/cipher`, so this run is
fully decoupled from the live CIPHER kmod (the substrate is `.so` + kmod; neither is touched). The verification panel
flagged the inherited heartbeat as undisclosed substrate coupling; it was removed and all runs re-measured kmod-free
(token-level results are deterministic and unchanged; timing/power were re-integrated). **Pre-registration:** `rc_ab/PREREG.md` (goodput
arithmetic + quarantine bound, written before measurement). **Evidence:** `rc_ab/{ab_shim.c, ab_driver.py, ab_analyze.py,
serve_gate.py, g_*.json, q_*.json, *_events.jsonl, goodput.json, quarantine_demo.json, ENTRY_MD5.txt, EXIT_MD5.txt}`.

---

## BOTTOM LINE (the honest answer)

**CIPHER does NOT deliver net-positive useful goodput under fault — it slightly REDUCES useful goodput (detector
overhead) — and that is the honest, pre-registered result.** Under a persistent SDC, *both* WITH and WITHOUT deliver the
**same 17 useful tokens** (the pre-fault prefix); greedy decoding then cascades, so the rest of the output is corrupt
regardless of CIPHER — CIPHER cannot *recover* goodput because it does not correct the GEMM. **What CIPHER buys is
safety, not goodput:** it converts **239 silently-served garbage tokens (93% of the output) into ≤25 (N=45) or ≤4 (N=8)**
(streaming serve-prefix) — or **0** under batch withhold-whole — by quarantining the corrupt response at the dispatch
boundary, at a **+4.0% (N=45) to +12.6% (N=8)** throughput cost. **Recovery (dispatch-boundary quarantine) is REAL for the batch case**: a persistent SDC is caught within N steps
and the corrupt response is withheld before serve, with **0 false quarantine** on clean output. **The honest bound:** a
**transient single-step fault that lands between checks is MISSED entirely** (the GEMM is clean again at the next check;
catch-rate ~1/N) even though it cascades to corrupt 237/256 tokens — and preventing an *already-streamed* token needs
vLLM cooperation (WALL). So R.A+R.B at the dispatch boundary is a **detection-and-quarantine value proposition for
persistent faults, paid for in throughput, blind to between-check transients** — not a goodput win.

---

## RULE-4 PRE-REGISTRATION (committed before measurement — `rc_ab/PREREG.md`)
Predicted: (1) useful-goodput **WITH ≤ WITHOUT** in every regime (CIPHER costs overhead + conservatively quarantines;
it never increases useful tokens); (2) CIPHER's value is **garbage-elimination** (`T−f` silent-wrong → ≤N); (3)
tok/joule **WITH ≤ WITHOUT**; (4) persistent quarantined **within ≤N, 0 false quarantine**; (5) quarantine-as-flag is
substrate-legal for batch, mid-stream is a WALL; (6) **transient between checks MISSED** (not merely late), catch ~1/N.
**All six were confirmed by measurement** (regime 2: divergence-at-onset). The one refinement vs the task's framing:
the transient is not "detected-after-emission," it is **not detected at all** unless it lands on a check step.

## PART 1 — GOODPUT UNDER A PERSISTENT SDC (`goodput.json`)
Injection: bit-14 flip of layer-0 down_proj output element 0, persistent from onset (shim `cur_step=20` = decode-token
**17**; prefill→cur_step offset = 3, derived from divergence alignment). **Determinism + non-mutation asserted and
PASSED:** the faulted token stream is **byte-identical** across WITHOUT / WITH-N8 / WITH-N45 (detector recomputes into
scratch, never mutates the output), and WITHOUT[0,17) == clean[0,17). **Useful = matching-PREFIX length** vs the clean
reference (after the first divergence the greedy cascade makes later coincidental matches meaningless). Quarantine =
labeling (full T generated in every run, so time/energy denominators match).

The Part-1 table uses the **streaming serve-prefix policy** (serve tokens as produced; on detection at S' stop serving;
tokens emitted in [onset, S') escape). The **batch withhold-whole policy** (§ Part 2) instead withholds the *entire*
flagged response → 0 garbage served but also 0 useful served. Both are reported; they trade prefix-delivery vs
garbage-elimination.

| Config (streaming serve-prefix) | useful tok | garbage **served** | quarantined | gen s | Joules | **useful tok/s** | **useful tok/J** | raw tok/s |
|---|---|---|---|---|---|---|---|---|
| CLEAN (no fault, ref) | 256 | 0 | 0 | 3.38 | 864 | **75.7** | **0.296** | 75.7 |
| **WITHOUT-CIPHER** (fault, all served) | **17** | **239** | 0 | 3.37 | 877 | **5.05** | **0.0194** | 76.1 |
| **WITH-CIPHER N=45** | 17 | **25** | 214 | 3.50 | 887 | **4.86** | 0.0192 | 73.1 |
| **WITH-CIPHER N=8** | 17 | **4** | 235 | 3.79 | 975 | **4.49** | 0.0174 | 67.6 |

Read it honestly:
- **Useful goodput is essentially flat across WITHOUT and WITH** — all three deliver the same **17** correct tokens (the
  fault corrupts every token from #17 onward via greedy cascade, whether or not CIPHER is present). WITH-CIPHER's useful
  tok/s and tok/J are **slightly LOWER** (overhead: **+4.0% time @N=45, +12.6% @N=8**; energy +1.2% / +11.2%). **CIPHER
  is not a goodput gain; it is a small goodput cost.** (As pre-registered.)
- **The decisive metric is garbage served:** WITHOUT silently ships **239 wrong tokens (93%** of a 256-token response).
  Under **streaming serve-prefix**, WITH ships **25 (N=45)** or **4 (N=8)** before quarantine — an **89.5–98.3%
  reduction**. Under **batch withhold-whole** (Part 2), WITH ships **0** garbage (entire flagged response withheld) at
  the cost of also discarding the 17-token correct prefix. Either way CIPHER converts *unknown-wrong* output into
  *known-bad / withheld* output; the policy choice is prefix-delivery (streaming, ≤N garbage) vs zero-garbage (batch, 0
  useful from the faulted response).
- **The N trade (streaming):** N=45 costs +4.0% and cuts garbage 239→25 (89.5%); N=8 cuts garbage to 4 (98.3%) but costs
  +12.6%. The best trade here is **N=45** (modest overhead, ~90% of garbage eliminated). Smaller N buys lower latency at higher
  overhead. (`overhead_time_pct` in `goodput.json`.)
- **Absolute useful goodput is low (≈5 tok/s) because a persistent SDC is catastrophic for greedy decode** (one wrong
  token cascades), not because of CIPHER — and CIPHER cannot fix it (it detects, it does not correct). batch=1 is
  launch-bound (~78 tok/s clean); a saturated batch changes absolute tok/s but not the trade structure.
- **Deployment note (early-stop, not in the fair table):** generating all 256 tokens after detection wastes compute on
  output that will be quarantined; stopping at detection (token 21 @N=8) would avoid generating ~235 garbage tokens —
  a real time/energy saving CIPHER enables, reported separately from the held-T comparison.

## PART 2 — DISPATCH-BOUNDARY QUARANTINE (`quarantine_demo.json`)
The shim detects SDC at the cuBLAS-interception boundary (residual>0, T=0) and records a detection signal; a scratch
**serve-gate** (NOT vLLM source) consumes it and decides SERVE vs QUARANTINE.

| Config | truly corrupt | detected@token | decision | outcome |
|---|---|---|---|---|
| WITHOUT (persistent, no detector) | yes (from 17) | — | SERVE | **SHIPS SILENT GARBAGE** |
| WITH N=8 (persistent) | yes (from 17) | 21 | **QUARANTINE** | corrupt response withheld |
| WITH N=45 (persistent) | yes (from 17) | 42 | **QUARANTINE** | corrupt response withheld |
| CLEAN + detector (no fault) | no | — | SERVE | served (genuinely clean) — **0 false quarantine** |
| TRANSIENT on-check + detector | yes (from 21) | 21 | **QUARANTINE** | withheld (detected at the corrupted token) |
| TRANSIENT between-check + detector | yes (from 19) | — | SERVE | **SHIPS SILENT GARBAGE (missed)** |

- **Persistent SDC quarantined within ≤N:** detection latency = **4 steps @N=8**, **25 steps @N=45** (both ≤N), counted
  from onset. For a batch/non-streaming response, detection at S' < T occurs *during* generation, so the serve-gate
  withholds the whole response **before it leaves the box**. **0 false quarantine** (clean + detector = 0 detections
  over 33 checks).
- **Transient honest bound (the periodic-detection / fork-2 wall, MEASURED):** a transient single-step fault **on a check
  step** is caught (detected at the corrupted token — i.e. *after* it was computed); a transient **between checks is
  MISSED entirely** — it corrupts **237/256** tokens via greedy cascade yet the detector sees a clean GEMM at the next
  check (the propagated KV-cache corruption is invisible to GEMM-recompute). Catch-rate for a transient ≈ **1/N**.

## QUARANTINE MECHANISM — what CIPHER can do vs the WALL
- **CAN, substrate-legally:** detect at the cuBLAS boundary + flag the response corrupt (a poison signal the serve-gate
  reads) → refuse to serve a flagged **batch** response before it returns. Demonstrated above. The serve-gate is scratch
  code (my application layer), not a vLLM edit.
- **WALL-WITH-MECHANISM:** scrubbing/poisoning an **already-streamed** token mid-flight, or reaching into vLLM's
  in-flight response buffer to invalidate it, requires vLLM cooperation (the shim cannot reach vLLM's response object
  from the cuBLAS boundary without a source hook). So for **token-streaming** serve, tokens emitted between onset and
  detection escape — CIPHER flags the response corrupt but cannot un-send those tokens. The **fleet-recovery** half
  (drain / failover / quarantine-node) is orchestrator-layer, fed by the R.C event — **out of scope here**, stated.

## NAMED WALLS / BOUNDS
- **No goodput gain:** CIPHER reduces useful goodput (overhead); its value is garbage-elimination/safety, not throughput.
- **Eager host only:** cudagraph-on hosting walled at Path-1 (+51% eager tax documented); these are eager numbers.
- **Periodic, not per-step:** transient between-check faults MISSED (~1/N catch); only persistent (or on-check) faults
  are reliably caught within N.
- **Linear-GEMM-only coverage:** detector checks qkv/o/gate_up/down via cublasGemmEx; attention (FlashAttn) + lm_head
  uncovered (R.A blind spot).
- **Mercurial-core blind:** same-GPU recompute reproduces a deterministic core fault → not caught.
- **One GPU, batch=1:** not a fleet, not a saturated multi-stream serve; the goodput *trade structure* generalizes, the
  absolute tok/s does not.
- **Streaming-serve quarantine is a WALL** (above); only batch/held-response quarantine is demonstrated before-serve.

## FORK-1 INTEGRITY (md5, entry vs exit) — `ENTRY_MD5.txt` / `EXIT_MD5.txt`
| File | entry | exit | match |
|---|---|---|---|
| `cipher_rt_phase4/libcipher_rt.so` (anchor) | `2edba0d2136f8ede4713d90a8f7cd55f` | `2edba0d2136f8ede4713d90a8f7cd55f` | ✅ |
| `ra_e2e/common.py` | `8276b0814020a8d9167a0edb2da58e85` | = | ✅ |
| `ra_e2e/run_e2e.py` | `2de53982eaf1aef14234edd8a5f6819a` | = | ✅ |
| `ra_e2e/calib.json` | `ade20e25357d108423210b4c2fea3ac9` | = | ✅ |
| `ra_e2e/trace.json` | `7d55b9120d05a74249db3dcc7618bb4d` | = | ✅ |
| `ra_e2e/run_e2e_summary.json` | `e5ac91b2d54407ab85f1d3ee82251d57` | = | ✅ |
| `ra_e2e/throughput.json` | `8131a2bd8f7ed7fd0e3821fc879cb98a` | = | ✅ |
| `ra_e2e/detect.json` | `cefa60651cc8cb56885923164c070a4c` | = | ✅ |

`diff(ENTRY_MD5.txt, EXIT_MD5.txt)` = empty.

## VERIFICATION PANEL (4-dimension adversarial pass, read-only, no GPU)
**NUMBERS — clean** (36/36 numeric checks traced to the rc_ab JSON; useful=matching-PREFIX confirmed genuine, not
positional; no retracted number reused). **LOGIC — clean** (WITH-vs-WITHOUT fair: same fault/onset/T, overhead in the
denominator — `gemms_checked` 0 WITHOUT vs 768/4224 WITH; useful=min(f,served) structurally ≤ useful_WITHOUT so no
offset can manufacture a gain; report claims no goodput gain). **RECOVERY — honest** (within-N quarantine real, 0 false
quarantine genuinely 0 across thousands of clean recomputes, transient-miss measured + emphasized, streaming WALL
stated). **INTEGRITY — 1 material + fixes applied:**
- **[MATERIAL → FIXED] Undisclosed kmod coupling.** The shim (copied from the R.C `rc_pergpu` shim) carried a W.6
  cohort `ioctl` heartbeat on `/dev/cipher`, coupling the live CIPHER kmod into vLLM — undisclosed under the
  "substrate-clean" framing (numbers were *not* contaminated: throttled ≥1s, result ignored, off the token/recompute
  path). **Fix:** heartbeat gated out (`RV_HEARTBEAT=0` default); shim rebuilt so it never opens `/dev/cipher`; **all 7
  runs re-measured kmod-free** (deterministic token-level results unchanged; timing/power re-integrated — overhead
  refreshed to +4.0% @N=45 / +12.6% @N=8). Disclosed above.
- **[minor → FIXED] Stray output in rc_pergpu.** The WITHOUT run wrote a counts JSON to `rc_pergpu/` via a stale default
  path. **Fix:** stray deleted (it was an untracked, non-deliverable artifact created by this run, not a pre-existing
  R.C file), shim default `RV_OUT` repointed to `rc_ab/`, all runs given explicit `rc_ab/` output paths; `rc_pergpu/`
  has no new files.
- **[minor → FIXED] Forward-referenced EXIT_MD5 / empty sections.** `EXIT_MD5.txt` generated; this panel + GUARDRAILS
  sections filled; the integrity table now shows exit hashes.
- **[minor → FIXED] Serve-policy conflation.** Part-1 (streaming serve-prefix, ≤N garbage) vs Part-2 (batch
  withhold-whole, 0 garbage) now labeled distinctly; the panel noted the original wording *understated* CIPHER.

## GUARDRAILS (exit)
Anchor `2edba0d2…` unchanged (entry==exit md5, table above). No vLLM source edit; shim copied+extended in `rc_ab/`
only; `libcipher_rt.so` not loaded into vLLM and the shim links only libc/libdl + dlopens only libcublas/libcudart
(`ldd` confirmed, no cipher link); kmod heartbeat disabled (no `/dev/cipher` access). `CUDA_INJECTION64_PATH` unset,
`VLLM_PLUGINS=""`, `VLLM_USE_DEEP_GEMM=0`. No files written outside `rc_ab/`. No leftover compute processes
(`nvidia-smi` empty). Clock requested 1980 / achieved ~1830 (SwPowerCap) → reset `-rgc` at exit (post-reset confirmed).

## HONEST BOTTOM LINE
Does CIPHER deliver net-positive useful goodput under faults? **No** — it costs a little (overhead) and recovers none
(it cannot correct the GEMM; greedy cascade corrupts the suffix regardless). What dispatch-boundary recovery actually
prevents: **shipping a persistent-fault's silent garbage** (239→≤25 tokens, 0 false quarantine, caught within N, for
batch responses). What it does NOT prevent: **between-check transients** (missed, ~1/N catch) and **already-streamed
tokens** (needs vLLM); fleet drain/failover is the orchestrator's job. The defensible R.A+R.B claim is **a
detection-and-quarantine safety layer for persistent linear-GEMM SDC, paid for in throughput** — valuable exactly when
silently-wrong output is costlier than the overhead, and explicitly not a goodput or a transient-coverage win.
