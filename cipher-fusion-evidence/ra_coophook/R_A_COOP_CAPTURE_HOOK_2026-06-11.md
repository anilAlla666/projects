# R.A vLLM-COOP CAPTURE HOOK — cudagraph-frame detector marginal, MEASURED (2026-06-11)

## VERDICT: **MEASURED — replaces the WI-4 memo's "~6% est. UNMEASURED".** Hosting the SDC detector inside the
cudagraph frame via a vLLM-cooperative capture hook costs **−3.9 % @ N=45 / −14.1 % @ N=8** (clean throughput,
0 false positives, catch-at-first-check ≤ N) — and **eliminates the +51 % eager tax** (1252 tok/s checked vs
fork-1's 609.5 eager+detector = 2.05×). The cheap design is **not free of correctness**: the naive periodic sidecar
is **measurably WRONG** (false positives from cudagraph memory-pool buffer reuse); correctness forces either
always-on inline recompute (**−50.5 %**) or a **dual full-graph** (the −3.9 % number, at **2× decode-graph
memory** for each captured size). The mechanism question is also settled: **capture-TIME insertion SURVIVES**
where Path-1's post-hoc graph mutation walled. This remains **vLLM-coop, NOT substrate-external** — the wrapper
patch is a runtime monkeypatch of vLLM's `CUDAGraphWrapper.__call__` standing in for a vLLM-side capture hook;
for the shipping A3 design that cooperation is **not one line** (it captures a second full graph and dispatches
between vanilla/checked per step — tens of lines of vLLM-internal logic). The substrate line (no vLLM-source
edits, Path-1 wall) is unchanged.

Pre-registered BEFORE measurement (`PREREG.md`, predictions 1–5 all addressed below). Scratch only in
`ra_coophook/`; frozen anchor md5 `2edba0d2136f8ede4713d90a8f7cd55f` entry == exit; fork-1 sources
(`ra_realvllm/`, `ra_path1/`, `ra_e2e/`) md5-unchanged; every run JSON records `anchor_loaded:false` (RTLD_NOLOAD
probe — the scratch shim is LD_PRELOADed, the anchor is NOT in-process). Default app clocks 1980, sampled per run.

---

## Setup (matches fork-1 ra_realvllm for comparability)
Mistral-7B-v0.1 fp16, vLLM 0.20.2 v1, **cudagraph ON** (FULL_AND_PIECEWISE), B=8, OUT=64, max_model_len=2048,
gpu_mem_util=0.85, T=0 greedy, `VLLM_ENABLE_V1_MULTIPROCESSING=0` (the coop patch + LD_PRELOAD shim must live in
the engine process). `VLLM_PLUGINS=` (cipher KV plugins off — they would pull the anchor in) and
`VLLM_DEEP_GEMM_WARMUP=skip` (this box has no deep_gemm; the warmup else segfaults — a vLLM-env issue, not ours).
Detector predicate identical to fork-1: linear GEMMs `k∈{4096,14336} ∧ m≠32000`; independent same-args-same-algo
`cublasGemmEx` recompute into preallocated scratch; `axpy(−1)` + `nrm2` (fp16 result, DEVICE pointer-mode in-graph);
T=0 ⇒ clean recompute bit-identical ⇒ residual exactly 0. Injection: a captured `xor14` CUDA kernel flips bit-14
of C[0] of one linear ordinal every replay (persistent SDC, fork-1 fault class).

**The coop hook:** `vllm.compilation.cuda_graph.CUDAGraphWrapper.__call__` is monkeypatched (version-pinned
semantics, copied from the 0.20.2 source) to (a) arm the shim around vLLM's FULL-decode capture so detector ops are
recorded **into vLLM's own graph**, and (b) for the dual-graph arm, capture a **second** checked full graph and
choose which to replay per step. This monkeypatch **stands in for a vLLM-side capture hook** (for A3, tens of
lines: a second-graph capture + per-step vanilla/checked dispatch — not a one-liner) — it is the measurement
vehicle for "what does cooperation buy", NOT a substrate-legal external mechanism.

## Arms and results (tok/s = best of 2 timed generates; marginal vs A0-restricted mean 1303.2)

| arm | design | N | tok/s | marginal | clean FP | catch (resid @catch step) | correct output |
|---|---|---|---|---|---|---|---|
| **A0** | cudagraph baseline, no shim (restricted capture_sizes=[8]) | — | **1303.2** (1302.8, 1303.6) | — | — | — | md5 `4e155f90` |
| A0u | cudagraph baseline, unrestricted capture sizes | — | 1324.9 | +1.7 % | — | — | `4e155f90` |
| **A1** | in-frame **always-on** inline recompute | — | **645.6** | **−50.5 %** | **0** (resid 0.0) | yes @inject (max resid 34976) | `4e155f90` clean |
| **A2** | periodic **sidecar** graph (frozen ptrs) | 45 | 1249.9 | −4.1 % | **4/4 FALSE POS** (resid 30656 on CLEAN) | — INVALID DESIGN — | — |
| **A3** | periodic **dual full-graph** | 45 | **1251.7** (1252.0, 1251.4) | **−3.9 %** | **0** (resid 0.0) | yes @ step 45 (resid 25520) | `4e155f90` clean |
| A3 | periodic dual full-graph | 8 | 1118.6–1119.4 | **−14.1 %** | **0** (resid 0.0) | yes @ step 8 (resid 8280) | `4e155f90` clean |
| A3-nr | dual full-graph, **D2H read skipped** (cost isolation) | 45 | 1262.5 | −3.1 % | n/a (no read) | n/a | `4e155f90` clean |
| A3-long | dual full-graph, **long clean** (OUT=512, FP bound) | 8 | 1127.1 | — | **0 over 191 checks = 24,448 slot-cmp** | — | clean |

(Injection runs corrupt the token stream as expected: A1, A3@N45-v2 and A3@N8-v2 all `7178ead0` — identical
because the persistent fault is now captured into **both** graphs (see Finding 1), so every production step is
corrupted exactly as in the always-on case; distinct from clean `4e155f90`, confirming the fault reached the output
AND the periodic check caught it at the first checked step. The earlier A3-inject v1 runs, with the fault in the
checked graph only, gave `b244a0cd`/`bfe00532` and are superseded.)

## Findings

**1. MECHANISM (PREREG pred 1 — CONFIRMED): capture-time insertion SURVIVES; Path-1's wall does not generalize.**
A1 and A3 both capture detector ops (recompute + axpy + nrm2, and the inject kernel) **during** vLLM's own
`torch.cuda.graph` capture, and the runs complete with correct output and replay every step. This is the precise
inverse of Path-1, where adding even a trivial memset node to vLLM's **already-captured** graph crashed init
`profile_cudagraph_memory`. The dividing line is **authorship/timing**: ops emitted inside the capture become part
of the vLLM-authored graph (legal); structural mutation of a finished graph vLLM re-processes is rejected. So the
"vLLM-coop hook" the arc proposed is viable — provided the cooperation is a **capture-time** hook, not a post-hoc
one.

**2. THE CHEAP PERIODIC SIDECAR (A2) IS MEASURABLY WRONG — the in-context-activation wall, now seen in cudagraph.**
A2 replays a separate sidecar graph every Nth step that recomputes each logged GEMM from its **frozen input
pointers**. On a **clean** run (RV_INJECT=−1) it fires **4/4 false positives, residual 30656**, 127/128 slots
nonzero. Cause (measured, not speculated): vLLM's FULL-decode cudagraph allocates intermediate activations from a
**reused memory pool**, so a linear's input buffer is overwritten later in the same forward pass; by the time the
sidecar replays, the frozen input pointer no longer holds that layer's activations ⇒ D′ ≠ C on clean data. This is
the same "recompute needs in-context activations" wall the e2e arc flagged for vLLM — here **quantified inside the
cudagraph frame**. It cannot be fixed by timing; only by recomputing while the buffer is live.

**3. CORRECTNESS forces inline recompute ⇒ two viable designs, both measured:**
- **A1 always-on inline (−50.5 %, 0 FP, catches):** recompute captured immediately behind every linear, every
  replay. Correct because it reads live activations; expensive because it doubles linear-GEMM work every step (r =
  T_checked/T_vanilla = 1303.2/645.6 = **2.02**). Not shippable; measured as the correctness anchor and the
  per-checked-step cost.
- **A3 dual full-graph (−3.9 % @N45, 0 FP, catches):** capture a **second** full decode graph with inline checks;
  replay vanilla on normal steps, the checked graph every Nth step (both are full correct forwards, so both
  produce correct logits — checked-step output md5 == vanilla). Correct AND cheap, at the cost of a second decode
  graph per captured size.

**The persistent-fault catch is rigorous (panel-corrected).** The injection is captured into **both** the vanilla
and the checked graph (`rv_set_inject` armed around both captures), so the fault corrupts **every production
step**, not just the periodic checked one — the corrupted output md5 (`7178ead0`) is identical to the always-on A1
injection, confirming the fault is genuinely persistent across all steps. The periodic checked step still catches
it at the **first** check (step 45 @N=45, step 8 @N=8; residual 25520 / 8280 at the catch step), latency ≤ N. (An
earlier version injected only into the checked graph, which made the catch trivially co-located; that run is
superseded and the rigorous re-run is the one reported.)

**FP bound (panel-corrected).** Beyond the short runs, a long clean run (OUT=512, N=8) performs **191 clean checks
= 24,448 clean slot-comparisons with max residual exactly 0.0 and 0 detections** — a *stronger* clean sample than
fork-1's eager 5,376 GEMM-checks. The "0 FP" claim is bounded by this, not by the 4–23 checks of the short runs.

**4. PERIODIC COST is Amdahl over the two measured per-step costs, plus an excess that ISOLATES into two parts
(PREREG pred 3 — D2H attribution corrected by measurement).** Analytic dual-graph marginal = N / ((N−1) + r),
r = 2.02: predicts −2.21 % @N45, −11.29 % @N8. **Measured −3.95 % / −14.10 %** — excess 1.7 pp / 2.8 pp above
Amdahl. I priced the D2H readback directly with an isolating run (**A3-nr**: replay the checked graph every Nth
step but **skip** `rv_read`): −3.12 % @N45. So at N=45 the 1.74 pp excess **decomposes**: ≈ **0.83 pp is the
legacy-stream `cudaMemcpy` D2H readback** (3.95 − 3.12) and ≈ **0.91 pp is replay-path overhead** above Amdahl
(python dispatch of the checked-graph branch + second-graph replay), independent of the readback. **The earlier
"the excess IS the D2H sync" was wrong** — the readback is only about half of it, and a device-pointer result kept
on-GPU with a deferred async read would recover only that ~0.8 pp, not the whole excess. (The per-check excess is
also larger at N=45 than N=8, inconsistent with a pure fixed-per-check cost — corroborating that a per-step
component, not only a per-check one, is present.)

**5. MEMORY (PREREG pred 5):** A3 captures one extra FULL decode graph per captured batch size; with
capture_sizes=[8] that is a single extra graph and did not block at 0.85 util. For production (many capture sizes)
it is **2× the decode-graph memory** — a real, named cost of the correct periodic design.

**6. BASELINE ties to fork-1.** A0 cudagraph 1303 tok/s (restricted) / 1325 (unrestricted) ≈ fork-1's 1289
cudagraph (within ~1–3 %). The capture-size restriction (used in all shimmed arms for a clean dual-graph capture)
costs ~1.7 % vs unrestricted; A3 is compared only to the **same-restriction** A0, so the −3.9 % is not
contaminated by the restriction.

## What this means for the product line (updates the WI-4 memo)
The memo's load-bearing **"capture hook → cudagraph-frame marginal ~6 % est. UNMEASURED"** is replaced by a
**measured** number: **−3.9 % @N45 (−14.1 % @N8)** via a **dual full-graph** design behind a **capture-time
vLLM-coop hook**, with **0 FP** and **catch ≤ N**, **eliminating the +51 % eager tax** (2.05× the eager+detector
throughput). The old "~6 %" estimate sits between the measured N=45 and N=8 points. **Two hard caveats travel with
the number:** (a) it is **vLLM-coop, not substrate-external** — Path-1 proved the substrate cannot inject this
without vLLM source cooperation, and this report measures the value of that cooperation, not a way around it; (b)
the **cheap sidecar is incorrect** — correctness in the cudagraph frame requires inline recompute, i.e. a second
full graph (2× decode-graph memory) or always-on (−50 %). The detector remains **persistent-within-N** (transient
between-check faults still ~1/N catch, unchanged from fork-1/fork-2).

## E2E PRODUCT TEST — A0 vs A3 on a realistic mixed serving trace (added 2026-06-11)
The −3.9 % above is the **saturated pure-decode (B=8) worst case**. The arc's product test is an E2E real serving
trace (cf. `ra_e2e`). I drove real vLLM (cudagraph on, `cudagraph_capture_sizes=[1,2,4,8]`, fixed seed → identical
trace for A0/A3) with 32 requests of varied prompt length + varied generation, plus a decode-dominated variant
(`e2e_driver.py`, `e2e_*.json`). **Two findings: the E2E detector cost scales with the decode-FULL fraction of the
trace, and coverage is partial (decode-FULL-at-captured-sizes only).**

**Disclosure (self-caught via verification): the first E2E pass was run WITHOUT `LD_PRELOAD`**, so the shim's
`cublasGemmEx` interceptor never fired (`ge_total=0, slots_used=0`) and the "checked" graph was an empty duplicate
forward — the ≈0 % deltas it produced measured empty-second-graph dispatch, not the detector. The runs below are
the corrected re-runs **with the interceptor active** (`ge_total` 6k–54k, `slots_used=128`, recompute confirmed).
The superseded no-PL JSONs are kept and labeled.

| trace | A0 tok/s | A3 tok/s (PL) | Δ | decode-FULL replays | checks | covered? |
|---|---|---|---|---|---|---|
| mixed (32 req, prefill-heavy) | 1170.1 / 1161.1 | 1163.0 / 1165.9 | **−0.1 %** | 256 | 5 | decode-FULL only |
| decode-dominated (8 req, batch≤cap) | 866.8 / 867.8 | 842.4 / 841.9 | **−2.9 %** | 1022 | 22 | decode-FULL only |
| decode-heavy (32 req, batch 32 > cap 8) | 1900.9 | 1874.1 | −1.4 % **(CONFOUND)** | 0 | 0 | **NONE** |

1. **E2E throughput cost scales with the decode-FULL fraction.** On decode-dominated covered traffic the cost is
   **−2.9 %** (1022 FULL replays, 22 checks @N=45) — approaching the −3.9 % saturated bound. On the prefill-heavy
   mixed trace it is **−0.1 %** (within noise): decode is a small fraction of wall time and only 5 checks land, so
   the per-check cost is diluted. So the detector is *cheap but not free* E2E — its cost is the saturated −3.9 %
   scaled by the share of the trace that is checked-eligible FULL decode.
2. **Coverage is the binding limit, MEASURED.** The FULL-decode hook protects only decode steps whose batch size
   matches a captured FULL graph. (a) **Prefill is uncovered** — on these traces prefill ran eager (`PIECEWISE|NONE`
   / `FULL|NONE` modes, 33,792 + 1,024 wrapper calls in the decode-heavy run), not through the FULL-decode hook.
   (b) **Decode batches larger than the captured set are uncovered** — the batch-32 trace ran entirely in NONE mode
   (`full_decode_replays=0, checks=0`), so no checked graph ran. Its −1.4 % Δ is a **confound, NOT a checked-graph
   cost**: with the interceptor preloaded the shim still passes through every prefill `cublasGemmEx`
   (`ge_total=202,370`) at a small per-call overhead, plus the 33.8 MB graph memory — both artifacts of the
   LD_PRELOAD *measurement vehicle*, which a real vLLM-integrated hook would not incur. Full-trace coverage would
   need hooking the PIECEWISE prefill capture too **and** capturing all decode sizes — strictly more vLLM
   cooperation than the decode-FULL hook.
3. **Memory MEASURED (sharpens Finding 5):** the 4 checked decode graphs (sizes 1,2,4,8) cost **+33.8 MB total**
   (~8.5 MB each) — the "2× decode-graph" cost is real but small in absolute terms (0.04 % of 80 GB); it scales
   with the number of captured sizes.
4. **0 FP on the E2E traces (now real):** the covered runs ran genuine recompute (`ge_total` 6k–54k,
   `slots_used=128`); across them 5 + 5 + 22 + 22 = 54 clean checks = **6,912 clean slot-comparisons, 0 detections,
   max residual 0.0** — corroborating the long-clean bound on real mixed/decode traffic.

**E2E bottom line:** in the cudagraph regime the detector's E2E cost scales with the decode-FULL fraction (−2.9 %
decode-dominated → −0.1 % prefill-heavy; −3.9 % saturated worst case), and the binding question is **coverage** —
as-built it protects decode at captured batch sizes, leaving prefill and over-cap-size decode unprotected. So the
product claim is "low-cost decode-SDC coverage under a vLLM-coop capture hook," explicitly *not* whole-forward
coverage.

## Integrity
- Frozen anchor md5 `2edba0d2136f8ede4713d90a8f7cd55f` entry == exit (`integrity_entry_md5.txt` /
  `integrity_exit.txt`). Fork-1 sources md5-unchanged (`ra_realvllm/detector_shim.c` `0c10c1ee…`,
  `ra_path1/path1_kernels.cu` `a269b8b5…`). NO anchor rebuild, NO successor shipped — this is a measurement study.
- Scratch only in `ra_coophook/` (shim `rv_coop_shim.c`+`.so`, `xor14.cu`+`.cubin`, `coop_driver.py`, run JSONs).
  The substrate anchor is NOT loaded into the engine (`anchor_loaded:false` in every run JSON via RTLD_NOLOAD).
  The monkeypatch touches a vLLM module **in-process at runtime only** — no vLLM file on disk was edited (verify:
  vLLM package mtimes unchanged).
- Default clocks 1980 (sampled per run, recorded in each JSON `clocks_after`); GPU idle + 0 leftover processes at
  exit. Disclosed env deviations (necessary, not cosmetic): `VLLM_PLUGINS=` (else the cipher KV plugin loads the
  anchor), `VLLM_DEEP_GEMM_WARMUP=skip` (else vLLM's deep_gemm warmup segfaults on this box), multiprocessing off
  (the patch must be in-process). None of these touch the GEMM math the detector checks.
- Numbers: every cell → a `run_*.json`; tok/s = best-of-2; A0 and A3@N45 each repeated (1302.8/1303.6,
  1252.0/1251.4) — variance < 0.1 %.

## Verification panel (2026-06-11)

4-lens adversarial panel (numbers / mechanism+logic / honesty / discipline), each material finding then attacked
by an independent skeptic (9 agents). Full findings + verdicts: `panel_findings_full.json`. **13 findings: 3
confirmed material (ALL FIXED BY RE-MEASUREMENT, not just prose), 2 downgraded, 0 refuted; minors applied.**

**Confirmed material — each fixed with a NEW measurement, not a softened sentence:**
1. **[mechanism] A3 injection was captured into the checked graph only** — so the fault was absent from vanilla
   (production) steps and the catch was trivially co-located. **FIX (re-measured):** added an independent
   `rv_set_inject` flag and armed it around **both** captures; re-ran A3-inject @N45/@N8. The persistent fault now
   corrupts every step (output md5 `7178ead0` == always-on A1 injection), and the periodic check still catches at
   the first checked step (45 / 8). The favorable-case runs are superseded. (`run_a3_n*_inject40_v2.json`.)
2. **[mechanism/numbers] "The excess IS the D2H sync" was asserted, not isolated, and the per-check pattern
   contradicted it.** **FIX (re-measured):** isolating run A3-nr (replay checked graph, skip `rv_read`) =
   −3.12 % @N45, so the 1.74 pp excess decomposes into ≈0.83 pp D2H + ≈0.91 pp replay-path overhead. Claim
   corrected from "is the D2H" to the measured split. (`run_a3_n45_noread.json`.)
3. **[honesty] "0 FP" lacked a clean-check bound** and was thinner than fork-1's 5,376. **FIX (re-measured):** long
   clean run (OUT=512, N=8) = 191 clean checks = **24,448 clean slot-comparisons, 0 detections, max resid 0.0** —
   now a stronger bound than fork-1. (`run_a3_n8_out512_clean.json`.)

**Downgraded (addressed):** the "one-line vLLM source hook" characterization understated an ~84-line monkeypatch
doing second-graph capture + dispatch — corrected to "tens of lines of vLLM-internal logic" at both sites. The D2H
finding was independently raised by the numbers lens (skeptic called it minor) and the mechanism lens (skeptic
confirmed material) — treated as material and isolated.

**Numbers lens: every headline reproduced bit-exactly** (A0 1303.2, marginals −50.5/−4.1/−3.9/−14.1 %, r 2.02,
Amdahl −2.21/−11.29 %, eager elimination 1252/609.5 = 2.05×, fork-1 baselines 609.5 + 1289). Minor table fix:
residual figures now report the value AT the catch step (25520 @45 / 8280 @8), not the run max. **Discipline lens:
0 findings** — anchor md5 entry==exit==live, all run JSONs `anchor_loaded:false`, vLLM on-disk unedited (May-13
mtime), no fork-1 file touched, clocks default, 0 leftover processes.

(Self-caught: the eager-elimination factor is 1252/609.5 = 2.05× using fork-1's precise eager+detector throughput
609.5; an interim edit using a rounded 609 briefly wrote 2.06× — corrected back to 2.05×.)

**E2E section (added after the panel) — independently verified, one material bug caught + fixed:** an adversarial
read of the E2E section found the first E2E pass had been run **without `LD_PRELOAD`**, so the interceptor never
fired and the "checked" graph held no detector ops (`ge_total=0`) — the ≈0 % deltas measured empty-second-graph
dispatch, not the detector. **Re-run with the interceptor active**: the corrected numbers (−2.9 % decode-dominated,
−0.1 % mixed, real 0 FP over 6,912 clean slot-comparisons) replace the broken ones; the superseded no-PL JSONs are
kept and labeled. The verifier also confirmed the coverage and memory findings hold. This correction is the reason
the E2E "bottom line" reads "cost scales with decode-FULL fraction," not the earlier overclaim "nearly free E2E."
