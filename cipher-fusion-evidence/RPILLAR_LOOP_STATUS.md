# CIPHER R-PILLAR GOODPUT CLOSE-OUT — LOOP STATUS

## LOOP SUMMARY (final — all four items resolved; loop ran 2026-06-09, completed on resume 2026-06-10)
1. **All four WIs DONE, all panel-verified:** WI-1 (panel 06-09: 1 material fixed), WI-2 (panel 06-09: ceiling
   retracted; +06-10 source correction: attainment floor 0.988→0.98 per its own JSONs), WI-3 (panel 06-10: 4
   material fixed → collector v2 + report rewrite), WI-4 memo (3-dim panel 06-10: ~45 values traced, 2 material +
   completeness gaps fixed).
2. **Correctness goodput (R_AB): NOT a win, by mechanism** — same 17 useful tokens WITH/WITHOUT; value = garbage
   served 239→25/4/0 at +4.0%/+12.6% (batch=1 greedy bound; persistent/on-check faults only).
3. **TRAINING goodput (WI-1): conditional WIN** — within-N catch (4@N8/25@N45, 0 FP/3,064) nets +3.7→+26.8pp @N=8
   p=1 upper bound in the silent regime, −4.8pp for fast-discovered; single-GPU LoRA bound; breakeven
   1/17,820 (N8) / 1/98,121 (N45) GPU-steps.
4. **SERVING SLO goodput (WI-2): mux WALL (substrate + vLLM-coop), unmeasured** — baseline latency-comfortable
   (≥0.98 to λ=64, no knee, no ceiling — transient runs); 3.06–3.30× stays in the tok/W lane.
5. **Fleet (WI-3): rc.v1 aggregation LOGIC validated end-to-end** (negatively-tested schema 4/4, totals-level
   idempotence, 1000-GPU simulated replay); depicted 2% fleet NET-NEGATIVE (−5.11pp@N8/−0.80pp@N45); PREREG
   "WITH>WITHOUT" falsified as stated (healthy-overhead term omitted) — disclosed.
6. **Decision rule (WI-4 memo):** fleet NET>0 iff fault rate > breakeven; Meta anchor (~1/50,700 GPU-h all-cause)
   is orders below ⇒ fleet-wide always-on NOT evidenced; deployment is targeted (suspect GPUs/jobs, triage,
   safety-critical withhold).
7. **Standing walls:** eager +51.2% (Path-1 mutation wall; cudagraph-frame marginal ~6% est. UNMEASURED);
   between-check transients missed (~1/N); linear-GEMM-only coverage; mercurial-core; per-tenant causation (v2);
   streaming quarantine; mux-SLO.
8. **Deliverable:** `wi4_memo/RPILLAR_GOODPUT_CLOSEOUT_MEMO_2026-06-10.md` (panel-verified, provenance table).
9. **Integrity:** anchor `2edba0d2` md5 entry==exit every WI; fork-1 dirs untouched; scratch only in wi*/ dirs
   (+ disclosed WI-2 prose correction); clock `-rgc` reset at loop exit; no leftover procs.
10. **Unbuilt/next (Anil's call):** vLLM-coop capture hook (then MEASURE cudagraph-frame marginal); per-tenant v2
    causation signal; deployed collector + real multi-GPU; steady-state SLO soak; opt-in DMR (≥14%).

## Per-item status
- **WI-1 TRAINING-BADPUT** — **DONE** (panel passed: numbers/honesty/integrity clean, 1 material logic finding
  [coverage p omitted] FIXED + 2 minor fixed). Real Mistral-7B LoRA step loop (t_s=0.1389s); SDC caught within N
  (latency 4@N8 / 25@N45, max_det 124, 0 FP over 3064 checks); training-goodput WIN conditional on silent-discovery
  AND covered-op (+3.7→+26.8pp eff-time @p=1 silent regime; +9.5pp @p=0.5; −4.8pp for fast-discovered rare faults);
  overhead +0.99%@N45 / +5.59%@N8. NaN-bug (fp16 random-token divergence miscounted) found+fixed. Anchor md5 entry==exit.
- **WI-2 SERVING-GOODPUT** — **DONE** (panel passed 06-09: 1 material [no capacity ceiling measured — transient
  runs, no knee] FIXED by retraction + minors). WITHOUT-mux baseline latency-comfortable: strict-SLO attainment
  ≥0.98 across λ=1–64 (0.988 @λ=32/64, 0.995 sustained; floor corrected 06-10 — prose said 0.988, own table/JSONs
  say 0.98 at λ=1–16; dated correction note in report), TTFT p90 ≤76ms, TPOT ≈14ms. Mux = WALL-WITH-MECHANISM
  (compiled substrate + vLLM-coop host needed); 3.06–3.30× kept in tok/W lane. Server reaped.
- **WI-3 FLEET-AGG** — **DONE** (panel 06-10: NUMBERS exact/byte-identical re-run + INTEGRITY pass; 2 HONESTY +
  2 LOGIC material ALL FIXED → collector v2: negative schema tests 4/4 + ValueError(-O-safe) + totals-level
  idempotence + prereg'd fleet_effective_training_time computed — depicted 2% fleet NET-NEGATIVE −5.11pp@N8/
  −0.80pp@N45, PREREG expectation 2 FALSIFIED as stated + disclosed; "real-derived onset" reworded — steps are
  synthetic 24+7i; gross/upper-bound labels added). fleet_view.json md5 c451b691.
- **WI-4 MEMO** — **DONE** (`wi4_memo/RPILLAR_GOODPUT_CLOSEOUT_MEMO_2026-06-10.md`; 3-dimension panel: ~45 values
  traced, 0 untraceable; materials fixed: WI-2 floor inheritance, capture-hook ~3%→~6%-unmeasured, deploy-at-N45 →
  rate>breakeven decision rule + Meta-anchor calibration; completeness bounds restored at all pitch sites).
