# R.A FULL-COVERAGE CLOSURE — STATUS + PRE-REGISTERED VERDICTS

**Resume-from-cold file.** Session keeps disconnecting; this captures the mandate, the on-disk
state, the pre-registered verdicts (mandate requires these BEFORE measurement), and the plan.
Last updated: 2026-06-08 (3rd session continuing the run).

## MANDATE (verbatim intent)
Close ALL SDC coverage gaps the persistent-SDC R.A detector left open, in ONE run, producing ONE
consolidated report `R_A_FULL_COVERAGE_CLOSURE_2026-06-07.md` in `cipher-fusion-evidence/`.
READ-ONLY: no production .so change (anchor `2edba0d2136f8ede4713d90a8f7cd55f` frozen, confirm at
exit), no application-layer edit, no vLLM source edit, no monkeypatch. Rule 4: derive analytical
floor ON PAPER before any kernel. Rule 3: a wall is recorded WITH its mechanism named
(WALL-WITH-MECHANISM) and the run CONTINUES — never halt all, never paper a gap. Every number
from THIS run. fp32 accumulation throughout. Clock 1980 (ran ~1830/1590 under load). Reset -rgc
at exit, no procs left.

## GUARDRAILS (checked 2026-06-08 session start)
- anchor md5 `2edba0d2136f8ede4713d90a8f7cd55f` UNCHANGED; LD_PRELOAD/CUDA_INJECTION64_PATH empty.
- clock locked 1830/max 1980, 0 compute procs, 35C idle.
- Measurement path uninjected EXCEPT Gap 6's explicit vLLM test (CUDA_INJECTION64_PATH side-channel,
  NOT production .so). Confirm production .so md5 unchanged regardless.

## SINGLE-GPU ORCHESTRATION RULE
One H100. GPU MEASUREMENT MUST BE SERIAL (concurrent probes contaminate each other's numbers).
"Parallel agents" applies ONLY to NON-GPU work: Step-0 paper floors, vLLM source root-cause
reading, attn-ABFT math re-verification, adversarial verification of computed results, report
synthesis. GPU probes run serially in-loop in the advisor order 4 -> 5 -> 3 -> 2 -> 1 -> 6.

## PRE-REGISTERED VERDICTS (written BEFORE this session's measurement)
- **GAP 1** phase-matched per-step (compute-heavy fused checksum in decode rides idle tensor cores;
  bandwidth-heavy separate-GEMV in prefill rides idle HBM). PRE-REG = **CLOSED-WITH-BOUND**: decode
  per-step CLOSED <3% (matches prior fused-checksum +1.1%), prefill WALL-WITH-MECHANISM (bandwidth
  check >=6%, substitution gap / Freivalds re-reads A+C). CLOSED only if BOTH <3%.
- **GAP 2** fused-checksum GEMM to cuBLAS parity. PRE-REG = **WALL-WITH-MECHANISM**: cuBLASLt cannot
  host the row-sum epilogue (BGRADA/B reduce over K not N = Finding 1); custom CUTLASS prefill
  gate/up +4.1% FAILS <3% at real shapes; decode GEMV +10.1% deployable substitution gap. FLOP
  floor <0.1% NOT realizable.
- **GAP 3** attn ABFT (QK^T + PV checksums). PRE-REG = **CLOSED-WITH-BOUND**: score-flip (S-check)
  + PV-output-flip (O-check) caught ~100%, 0 clean FP; softmax-INTERNAL finite residual BLIND by
  construction (scales O and ref identically) — proven, not a miss. FLOP floor <0.2%; realizable
  needs FA-epilogue fusion (analytical only, no production kernel).
- **GAP 4** non-GEMM per-step recompute. PRE-REG (reconstructed) = CLOSED if <3% both phases.
  ACTUAL on disk: +71.4% decode / +54.1% prefill, catch 40/40, T=0. **NOT SUBSUMED by GEMM checksum
  (advisor-corrected):** a storage flip in a non-GEMM OUTPUT (h=RMSNorm, act=SiLU, q/k=RoPE) is read
  identically by the forward GEMM and by the GEMM-recompute (same cached corrupted input) -> both
  agree -> BLIND. Non-GEMM outputs are GEMM *inputs* = a DIFFERENT fault site than GEMM-output
  storage faults. Catching them REQUIRES regenerating the op from ITS input (h'=RMSNorm(x)) = exactly
  Gap 4's cost. So Gap 4 is a genuine separate gap = its OWN coverage-matrix row ("non-GEMM output
  storage fault"). DIAGNOSIS (must CONFIRM by profiling/kernel-count, not assert): bandwidth floor
  from activation traffic (MLP gate/up/act [M,14336] dominate); unfused probe issues ~700 tiny
  reduction kernels -> Step-A kernel-COUNT wall (~9.5us min-duration each survives graph). Revised
  PRE-REG = **CLOSED-WITH-BOUND**: floor TBD on paper (decode likely ~1-2%, prefill traffic 4x ->
  may exceed 3% even fused = a real finding), realization needs FUSED recompute+compare; FLIP = if
  fused/floor still <3% both phases it CLOSES, if prefill floor >3% it WALLS-WITH-MECHANISM (BW).
- **GAP 5** disjoint-SM via green contexts. PRE-REG = **CLOSED-WITH-BOUND**: cuGreenCtxCreate +
  cuDevSmResourceSplitByCount give disjoint SM sets; deterministic SM-localized fault caught by
  recompute on the disjoint partition; BOUNDED to SM-localized faults (shared-L2/HBM-path faults
  recur, out of scope); same-SM reproduce-and-miss reported CONDITIONALLY on measured block->SM
  stability (never asserted).
- **GAP 6** vLLM root-cause + co-run. PRE-REG = **WALL-WITH-MECHANISM** (likely): vLLM CUDA-graphed
  weight-streaming decode incompatible with in-context recompute (recompute needs in-context
  activations+weights, no separate process hosts it). Root-cause the death precisely; CLOSED only
  if a co-run path works without crossing the substrate line.

## ON-DISK STATE (full_coverage/)
- gap3_attn_abft/attn_abft_probe.py — READY, NOT RUN. Implements S-check + O-check + sites A/B/C +
  FA-streaming + FLOP floor. (advisor: S=QK^T checksum mandatory — present.)
- gap4_nongemm/gap4.py + gap4_result.json — MEASURED +71/+54 (kernel-count artifact, see above).
- gap5_disjoint_sm/disjoint_sm_probe.py + kernels.cu + kernels.cubin — READY, NOT RUN.
- gap1, gap2, gap6 — NOT STARTED.
- No consolidated report yet.

## EXECUTION ORDER (this session)
0. advisor on continuation plan. 1. Wave-0 parallel NON-GPU floors+root-cause (workflow).
2. Serial GPU: Gap 4 diagnose/fix -> Gap 5 run -> Gap 3 run -> Gap 2 floor+reconfirm wall ->
   Gap 1 phase-matched build+measure -> Gap 6 vLLM. 3. Wave-2 unified scorecard + coverage matrix.
4. Parallel adversarial verification (workflow). 5. Write consolidated report. 6. Reset -rgc,
   confirm anchor, no procs.

## PRIOR-MEMORY CROSS-REFS (pre-answers to reconcile, re-measure fresh per mandate)
step-a (ABFT coverage YES/overhead NO, fused-only path), step-b-spike (decode fused +1.1% PASS /
deployable +10.1%), step-b-freivalds (prefill gate FAILS >=6% unconditional), step-b-prefill
(CUTLASS gate/up +4.1% FAIL), step-b-e2e (periodic recompute path), ra-e2e (+2.96% shipped).

## RESULTS (live, append per gap)
### GAP 4 — DONE 2026-06-08. Verdict: decode CLOSED-WITH-BOUND / prefill WALL-WITH-MECHANISM.
Files: gap4_nongemm/{gap4.py (catch), gap4_diag.py (floor+kernel-count), gap4_result.json, gap4_diag_result.json}.
- Genuine separate fault-site (advisor): non-GEMM OUTPUT storage flip is BLIND to the shipped GEMM-output
  recompute (both read same cached corrupted input); caught only by regenerating op from its input = Gap 4.
- COVERAGE: 40/40 non-GEMM flips caught (rms1/rope_q/rope_k/rms2/silu), T=0, 0 clean FP.
- COST unfused: +72.3% decode / +54.2% prefill. KERNEL-COUNT WALL CONFIRMED (measured): +2305 kernels
  IDENTICAL both M (M-independent => count not work), slope linear 0.21/0.29 ms per checked-layer, per-kernel
  2.91/3.99us. = Step-A "graph removes launch-gaps not min-duration".
- Rule-4 BW FLOOR (separate-pass re-read of cached non-GEMM I/O @2.97TB/s): decode 0.237ms=**2.56%** (<3%,
  CLOSES if fused to 1 streaming pass) ; prefill 0.949ms=**5.59%** (>3%, WALLS even fully fused — MLP gate/up/act
  [M,14336] re-read ~2.5GB dominates; only inline-epilogue fusion overlaps = substitution gap). Fused kernel
  NOT built (floor analytical per Rule-4 discipline; kernel-count mechanism measured). FLIP HIT: prefill>3%=WALL.
- MATRIX ROW: "non-GEMM output storage fault" — decode caught <3% (fused), prefill bandwidth-walled.
### GAP 5 — DONE 2026-06-08. Verdict: CLOSED-WITH-BOUND.
Files: gap5_disjoint_sm/{disjoint_sm_probe.py, kernels.cu, kernels.cubin, disjoint_sm_result.json}.
- MECHANISM PROVEN: green-ctx split 132 SM -> disjoint 64+64 (S_A ∩ S_B = empty, verified; 4 stranded).
  Deterministic SM-localized fault (FAULTY_SM=64 in S_A, not in S_B): DISJOINT-partition recompute catches
  GUARANTEED (faulty SM absent by construction). This is the ONLY catch for deterministic-COMPUTE faults that
  recompute is BLIND to (recompute on same HW reproduces them).
- HONESTY (advisor): same-SM recompute caught BY ACCIDENT — block->SM map NOT stable (recur=0.0156~1/64,
  roving); principled catch = disjoint partition, not same-SM reproduce-and-miss.
- BOUNDS: (a) SM-localized only (shared L2/HBM-path faults recur on both -> blind); (b) fault SIMULATED by
  smid-conditioned flip (demonstrates spatial-redundancy MECHANISM, not that real faults are SM-localized);
  (c) cost ~2x SMs (dual redundant exec) = a redundancy MODE, not a <3% detector.
- MATRIX ROW: "deterministic SM-localized compute fault" — caught via disjoint-SM redundancy, ~2x cost.
### GAP 3 — DONE 2026-06-08. Verdict: CLOSED-WITH-BOUND.
Files: gap3_attn_abft/{attn_abft_probe.py, attn_abft_result.json}.
- COVERAGE (the named Step-A blind spot attn QK^T/PV, NOW COVERED): S-check (raw QK^T row-sum vs Q@colsum_K)
  catches SCORE flips 200/200=100%; O-check (O row-sum vs P@rowsum_V) catches PV-OUTPUT flips 200/200=100%.
  Margins ~1e4-1e6x both regimes; naninf caught by isnan/isinf guard.
- 0 CLEAN FP over 100 runs both regimes, with T=8x roundoff floor. (Attention checksum ref uses a DIFFERENT
  fp eval order than the output => clean roundoff NONZERO, unlike bit-exact GEMM recompute T=0 => needs a
  safety-factored T, not T=0.)
- BOUND/RESIDUAL (pre-registered BLIND): softmax-INTERNAL finite corruption (P probs, online-softmax l_i/m_i
  bookkeeping) scales O and the checksum ref IDENTICALLY => algebraically invisible. ~81-88% caught
  INCIDENTALLY by fp-eval-order divergence (NOT guaranteed), ~12-19% genuinely blind. NaN/inf softmax-internal
  IS caught. FA-streaming verified: rowsum_V appended col survives online rescaling (PRESERVED); l_i-fault BLIND
  (O wrong by 7e-2 yet checkcol matches).
- COST: FLOP floor combined 0.488% (analytical); decode reductions MUST be incremental (from-scratch=100%);
  realizable needs FA-epilogue fusion (rowsum_V as checksum column) — NOT built (analytical per Rule-4).
- MATRIX ROWS: "attn QK^T score storage fault" caught 100%; "attn PV output storage fault" caught 100%;
  "softmax-internal finite fault" = BLIND residual (partial incidental ~85%).
### GAP 2 — DONE 2026-06-08. Verdict: WALL-WITH-MECHANISM.
Files: gap2_fused_gemm/{gap2_abft.py, gap2_abft_result.json}.
- Per-GEMM row-sum ABFT chk=Y.sum(1) == ref=x@(W.sum0), s precomputed once.
- FLOP floor 0.028% (checksum ~free arithmetically). BW floor (sep-reduction re-read X+Y @2.97TB/s):
  decode 2.72% (<3%) / prefill 5.89% (>3%).
- MEASURED sep-reduction over PLAIN cuBLAS: decode +76.8% / prefill +58.3% = KERNEL-COUNT WALL
  (+2016/+1824 kernels, ~3.2/4.9us each), nowhere near BW floor.
- WALL MECHANISM: to hit the BW floor the row-sum must FUSE into the GEMM epilogue, but cuBLASLt
  CANNOT host row-sum-over-N (Finding 1: BGRAD reduces over K not out) => custom CUTLASS GEMM =
  slower than cuBLAS by more than the check saves (substitution gap: prior +4.1% gate/up, +10.1%
  decode GEMV). No per-GEMM checksum is <3% in BOTH phases over cuBLAS.
- COVERAGE = Step-A-settled (200/200 harmful, same row-sum checksum the shipped detector uses); this
  run's quick re-confirm 21/30 bit-14 flips = harmful caught, 9 benign below fp16-output roundoff T
  (threshold-muddied, NOT a coverage result).
- FEEDS GAP 1: the check is BANDWIDTH-heavy not compute-heavy => decode (mem-bound) collides,
  prefill (compute-bound) may OVERLAP on a concurrent stream.
### GAP 1 — DONE 2026-06-08 (THE HEART). Verdict: WALL-WITH-MECHANISM. Hypothesis FALSIFIED.
Files: gap1_phasematched/{gap1.py, gap1_result.json}.
- Mandate hypothesis (match check to idle resource: compute-check in decode, BW-check in prefill) is
  FALSIFIED. Measured concurrent-stream overlap: PREFILL +58.7% (vs serial +68.4%) — check did NOT hide
  under 50% spare HBM; DECODE overlap +183% WORSE than serial +163% — collides for scarce BW.
- Rule-4 roofline does the work: check BW = prefill 5.6% peak / 11.2% of spare; decode 2.4% peak. BW is
  NOT the limiter. The limiter is KERNEL-COUNT (~1800 tiny reduction kernels), which a 2nd stream cannot
  co-schedule into tensor-core-bound GEMMs => phase-matching to the idle RESOURCE is moot.
- Only fix = fuse the check into few/epilogue kernels so spare HBM is usable = the SUBSTITUTION GAP
  (cuBLASLt can't host row-sum epilogue; custom kernel slower; fused check itself is cheap +1.1% decode
  prior but deployable +10.1%). Both phases wall on ONE mechanism (kernel-count -> fusion -> subst gap).
- base eager~graph: prefill 4.6% (eager valid, compute-bound), decode 12% (launch-sensitive; cite graph).
- RESIDUAL (unbuilt): a custom fused-to-~32-kernels BW-bound check MIGHT overlap prefill spare HBM
  (0.9GB/0.6ms in 15ms) IF SMs co-schedule it — needs a custom kernel + multi-stream graph (not built;
  substitution-gap-adjacent). Stated, not closed.
- MATRIX: "linear GEMM-output transient, PER-STEP" — NOT <3% deployable either phase (kernel-count/subst
  gap). Persistent-within-N is the shipped R.A path (+2.96%); per-step remains walled.
### GAP 6 — DONE 2026-06-08 (mechanism; source citations from agent). Verdict: WALL-WITH-MECHANISM (cudagraphed vLLM) / co-run via enforce_eager (caveated).
Files: gap6_vllm/{gap6_mechanism.py, gap6_mechanism_result.json}. vLLM 0.20.2 installed.
- MECHANISM MEASURED (minimal cudagraph probe, no vLLM run): (1) FROZEN GRAPH (MEASURED: replay re-ran
  captured op w/ NEW input ybuf=14, but post-capture kernel ABSENT zbuf=300 not 700); (2a) host-sync
  .item() during capture -> cudaErrorStreamCaptureInvalidated; (2b) raw cudaHostAlloc during capture ->
  err 900/cudaErrorStreamCaptureUnsupported ("operation not permitted when stream is capturing") ->
  invalidates capture = EXACT injected-lib failure ("vLLM died under injected cipher"; Koopman err 900).
- VERIFIED by 7-agent adversarial panel (Gaps 1-5 + Wave-2 sound; Gap6 frozen-graph re-measured + minor
  wording fixes applied to report).
### WAVE-2 CLOSE-OUT (advisor) — DONE 2026-06-08. Headline CORRECTED.
Files: wave2_unified/{wave2_cov_cost.py, wave2_cov_cost_result.json}. Clock re-locked 1980 for clean
ratio then -rgc. Advisor caught: the unified ~3.0%@N=45 headline was ASSERTED not measured (same flaw
fixed in Gap6). Measured both:
- (A) PROPAGATION COVERAGE: recompute-FROM-INPUT + GEMM-output compare, T_clean=0 (0 FP by construction).
  non-GEMM h 12/20, act 17/20 (class the shipped from-cached-h detector is BLIND to), GEMM q 14/20.
  Misses BENIGN by T=0 logic (not-caught <=> no checked GEMM output changed <=> absorbed in fp16 rounding
  <=> output uncorrupted). Harmful non-GEMM faults that propagate ARE caught. Propagation CONFIRMED.
- (B) REAL COST: recompute-from-input r=2.59 dec/2.62 pre (>> ra-e2e from-cached-h r=1.49). Wall-weighted
  E2E (r-1)/N: N=45 -> 3.57% (OVER 3%), N=64 -> 2.51% (under). Synthetic = UPPER bound (GEMM-heavy base,
  no real attention); real-trace lower but NOT re-measured. => deployable N in 45-64 band, latency 44-63
  steps; full coverage costs MORE than linear-only +2.96%; "<3%@N=45" does NOT carry to full coverage.
- Report headline/matrix/residual/bottom-line ALL corrected to the measured r=2.6 / N=45-64 band.
RUN COMPLETE. Guardrails exit: anchor 2edba0d2 UNCHANGED, env uninjected, -rgc reset (375/1980), 0 procs.
- ROOT CAUSE: an injected in-context recompute detector cannot be hosted in vLLM's CUDA-graphed decode:
  can't add to a frozen captured graph; injecting during capture (alloc/host-decision) invalidates the
  capture; a separate process lacks in-context activations. CO-RUN: enforce_eager=True disables cudagraph
  -> detector hookable (eager forward) BUT vLLM loses cudagraph speedup (decode launch-bound); the +2.96%
  (custom graph harness) does NOT transfer. [source file:line citations pending agent.]
- MATRIX: "detector hosting in production serving" — WALL for cudagraphed vLLM; eager co-run caveated.
- SOURCE CITATIONS (agent, vLLM 0.20.2): default cudagraph ON (O2->FULL_AND_PIECEWISE config/vllm.py:336,228);
  decode capture cuda_graph.py:308-312 dedicated graph_pool/single stream; frozen replay cuda_graph.py:355
  (no kernel insertion); external alloc/stream during capture -> cudaErrorStreamCaptureInvalidated(900);
  enforce_eager config/vllm.py:895-901 -> cudagraph_mode=NONE, capture skipped gpu_model_runner.py:6050-6056
  (eager forward hookable), costs ~10-30% decode throughput.
### WAVE 2 — UNIFIED periodic detector. DONE 2026-06-08.
Files: wave2_unified/{wave2.py, wave2_result.json}.
- Per-step walls (Gap 1) => deployable unified detector is PERIODIC (every N steps).
- Measured (synthetic, GEMM-heavy base = UPPER bound on r vs ra-e2e real r=1.49/1.33): r_gemm 2.11dec/2.02pre,
  r_unified(w/ redundant non-GEMM compares) 2.57/2.35; synth E2E@N45 unified 3.50%dec/3.01%pre.
- REFINEMENT: explicit non-GEMM compares are REDUNDANT — non-GEMM-output corruption propagates to the
  downstream GEMM/attention, so recompute-from-INPUT + GEMM-output compare + attn-ABFT catches it WITHOUT
  separate non-GEMM compare kernels (the +46%-of-base synthetic add is the unfused artifact, not required).
- DEPLOYABLE unified (anchored ra-e2e real trace): +2.96% (GEMM-recompute) + ~0.01% (attn ABFT FLOP/N) +
  small fused non-GEMM (recompute-from-input via model's fused fwd) ~= +3.0% E2E @N=45; N=64 -> ~2.1-2.7%
  margin. Catches PERSISTENT linear-GEMM + non-GEMM + attention faults within N. Latency N-1 steps.
- BOTTOM LINE: full SDC coverage is NOT one <3% always-on detector. Persistent-within-N unified ~3% (band
  N=45-64). PER-STEP / transient walls. Residuals: transient single-step, softmax-internal finite (Gap3),
  deterministic-compute non-SM-localized (Gap5=SM-localized@2x), cudagraphed-vLLM hosting (Gap6).
