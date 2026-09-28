# WI-3 — R.C FLEET-AGGREGATION SCHEMA END-TO-END

**Date:** 2026-06-09 (verification panel + fixes applied 2026-06-10). **Goal:** prove the rc.v1 per-GPU event is
fleet-consumable end-to-end, without claiming real N-GPU scale. No GPU used (anchor md5 still recorded).
**Discipline:** scratch code in `wi3_fleetagg/` only; the real rc.v1 event copied read-only from
`rc_pergpu/inject_verdict.json` (md5-identical copy, `2e9d75c5`); anchor `2edba0d2…` md5 entry==exit.
**Pre-reg:** `wi3_fleetagg/PREREG.md` (unmodified; deviations disclosed below, never edited into the prereg).
**Evidence:** `wi3_collector.py` (v2, post-panel), `rcv1_event_real.json`, `fleet_view.json` (md5 `c451b691`).

---

## BOTTOM LINE
The rc.v1 per-GPU event is **fleet-aggregatable end-to-end.** A collector ingests rc.v1 events keyed by `gpu.uuid`
and emits a `fleet_reliability_view` over a **1000-GPU simulated fleet (2% degraded = 20 GPUs)**: it lists each
degraded GPU's **synthetically-varied onset step (24+7i replay variation; the real event's onset is 64/72 — the wall
stamp and descriptor are replayed verbatim from the single real event)**, severity, corroboration, and
process/context descriptor (`VLLM::EngineCore`, tgid 3320702), attributes WI-1's modeled-accounting badput-avoided
per fault (**138.35 GPU-s GROSS, WI-1's MOST FAVORABLE scenario** — silent-1000-step discovery, N=8, p=1, where p=1
is WI-1's labeled upper bound), and sums a **fleet GROSS total of 0.77 GPU-hours avoided** at 20 faults.
Schema-conformance (now **negatively tested**: 4/4 malformed classes rejected, view uncorrupted), uuid-keyed
aggregation, and idempotence (now tested at the **totals** level, including changed-verdict update) all hold.
**The depicted fleet is NET-NEGATIVE:** the pre-registered `fleet_effective_training_time` is now computed WITH the
healthy-GPU overhead term the PREREG formula omitted — at the 2% mix, fleet WITH 0.9470 < WITHOUT 0.9981 (@N=8,
−5.11pp) and 0.9901 < 0.9981 (@N=45, −0.80pp); **PREREG expectation 2's WITH>WITHOUT prediction is FALSIFIED as
stated** (see PREREG DEVIATIONS). **Real vs simulated is explicit:** the per-GPU EVENT is the validated real
DEGRADED verdict; the multi-GPU population is SIMULATED by replay; the **aggregation LOGIC** is what is validated,
not fleet scale.

## COLLECTOR + FLEET VIEW (`fleet_view.json` md5 `c451b691`, `wi3_collector.py` v2)
- **Ingest:** rc.v1 events, schema-checked (10 required fields + `schema_version=="rc.v1"` + `gpu.uuid` + valid
  verdict + severity ∈ {critical,warning,info} + DEGRADED⇒fault_onset present), keyed by `gpu.uuid` (fleet-unique,
  not the ordinal — uuid-keying proven non-ordinal: synthetic `gpu_index` collides at i%8 while uuids stay unique).
  Rejection raises `ValueError` (the v1 `assert` was stripped under `python -O` — fixed and re-verified under `-O`).
  Idempotent: re-ingesting a uuid updates, never double-counts (tested, see Validation).
- **Fleet view output:** `total_gpus=1000`, `n_healthy=980`, `n_degraded=20`, `degraded_fraction=0.02`;
  `degraded[]` = per-GPU {uuid, node, fault_onset_step (synthetic), fault_onset_wall_replayed, severity,
  corroboration, descriptor tgid + cmdline, badput_avoided_gpu_seconds_gross};
  `fleet_badput_avoided_gpu_seconds_total_gross=2767` (= 20×138.35 exactly), `gpu_hours=0.769`;
  `fleet_effective_training_time` block (window 10k steps, per-N WITHOUT/WITH/delta, overhead GPU-h, breakeven,
  depicted fault rate, assumptions, prereg note).
- **Validation block:** `schema_rejections_in_replay=0` (all replayed events conform — one-sided by construction);
  `schema_negative_tests: 4/4 rejected` (missing required field / missing gpu.uuid / invalid verdict /
  DEGRADED-with-null-onset), `view_uncorrupted_by_rejection=true`; `idempotent_reingest_count=true` plus
  `update_semantics_tests`: identical re-ingest leaves totals unchanged AND changed-verdict re-ingest updates without
  double-count (n_degraded 20→19, gross total 2767.0→2628.7 = 19×138.35); `all_keyed_by_uuid=true`.

A fleet reliability service consumes exactly this: a stream of per-(node, gpu.uuid) rc.v1 events → which GPUs are
producing silent corruption, when each began, what was running, and the (gross, scenario-labeled) reliability badput
each represents — plus the fleet NET economics block that says whether the detector fleet-wide is paying for itself.

## REAL vs SIMULATED (binding boundary)
- **REAL:** `rcv1_event_real.json` = the validated rc.v1 DEGRADED verdict from R.C (real onset step 64/72, real
  descriptor `VLLM::EngineCore`, real signal-sources, `uncorroborated_by_hardware_counters`).
- **SIMULATED:** the 1000-GPU population — the one real event replayed across deterministic synthetic uuids with a
  formula-varied onset STEP (24+7i → 24..157; none equals the real 64/72) and a 2% degraded mix. Severity is NOT
  varied (all degraded carry the real event's `critical` — a PREREG deviation, disclosed below). The wall stamp is
  the single replayed real value on all 20 (a replay artifact, labeled `fault_onset_wall_replayed`). The per-GPU
  SIGNAL is the single real detection; the COLLECTOR/aggregation logic + schema are what is validated.

## TIE TO WI-1 (badput attribution + fleet NET economics)
Each degraded GPU is attributed WI-1's **modeled-accounting** per-fault badput-avoided (138.35 GPU-s = measured step
time 0.1389 s × modeled silent-1000-step discovery delay net of measured latency-4; per `wi1_accounting.json` —
**not a measured quantity**: WI-1 classifies the discovery delay as MODELED). This is WI-1's most favorable row and
p=1 upper bound; the fleet total is **GROSS**. The fleet total is linear in `n_degraded` **by construction**
(constant per-fault attribution — consistent with WI-1's onset-independent model, but not an emergent validated
property). **Fleet NET economics (computed, v2):** with overhead paid continuously on ALL 1000 GPUs and
badput-avoided accruing only on the 20 that fault, the depicted fleet is NET-NEGATIVE — gross 0.77 GPU-h avoided vs
**21.6 GPU-h (@N=8) / 3.8 GPU-h (@N=45)** fleet detector overhead over the same 10k-step window; fleet effective
time WITH 0.9470/0.9901 < WITHOUT 0.9981. The depicted **fleet fault rate** (defined: faults per GPU-step) is
20/10⁷ = **1 per 500,000 GPU-steps**, far below WI-1's breakeven (**1/17,820 @N=8, 1/98,121 @N=45**); under the
stated homogeneity assumptions overhead and badput-avoided are both additive per GPU-step, so **fleet NET>0 iff the
fleet fault rate exceeds WI-1's per-GPU breakeven** (cross-check: 138.35/(0.0559×0.1389)=17,818 ≈ 17,820 ✓).

## PREREG DEVIATIONS (disclosed; PREREG.md itself unmodified)
1. **`fleet_effective_training_time` was pre-registered as a collector output and initially NOT emitted** (v1
   deferred to WI-1's breakeven with an "avoid double-counting" rationale the panel rejected — computing it once
   double-counts nothing; the real obstacle was that the PREREG formula omitted the 980 healthy GPUs' overhead).
   **v2 computes it** (with the healthy-overhead term): at the depicted 2% mix, **fleet WITH < WITHOUT at both N**
   — i.e. **pre-registered expectation 2's "WITH > WITHOUT" prediction is FALSIFIED as stated.** The qualitative
   breakeven claim that replaces it is arithmetically consistent with WI-1 (see TIE TO WI-1).
2. **"Varied severity" was pre-registered but not delivered** — all 20 degraded events carry the real event's
   `critical`; only the onset step varies. (Variation would have been synthetic invention anyway; disclosed rather
   than fabricated.)
3. **PREREG expectation 1's phrase "real-derived onset"** carried the same overclaim the panel flagged in the v1
   bottom line; resolved by wording (synthetic step / replayed wall), not by editing the prereg.

## BOUNDS
- One real event replayed ⇒ per-GPU signal diversity is synthetic and THIN (varied onset step only; severity/wall/
  descriptor identical across degraded entries). Real fleet heterogeneity (distinct models, distinct real faults)
  not represented.
- Badput numbers inherit WI-1's bounds (TRAINING-goodput definition; covered-op fraction p; single-GPU step model)
  and use WI-1's most favorable scenario — gross figures are an upper-bound illustration, not a central estimate.
- Fleet NET economics assume homogeneity (every fault silent-1000/N-covered/p=1) and a 10k-step window; the
  breakeven equivalence is exact only under those assumptions.
- No real multi-GPU run; no deployed collector service — aggregation logic + schema validated offline.

## FORK-1 INTEGRITY
Anchor `libcipher_rt.so`: entry `2edba0d2136f8ede4713d90a8f7cd55f` == exit (verified again post-panel) ✅ (no GPU
used; recorded per discipline). New work only in `wi3_fleetagg/`; real rc.v1 event consumed read-only (copy
md5-identical to source, `2e9d75c5`). **Artifact hygiene (panel finding, adopted):** v1's `fleet_view.json` had been
regenerated in place after the report was written (harmless ONLY because the collector is deterministic — a
path-redirected /tmp copy reproduced it byte-identically, md5 `82d13867`); v2 records the artifact md5 in the report
(`c451b691`) and future re-runs write to /tmp and diff. Ambient `.git` stash-style loose objects written at 19:05
during the WI window were inspected by the panel: pre-existing phase_c drift snapshots, NOT WI-3 content; refs
unchanged since `a15e59b`.

## VERIFICATION PANEL (4-dimension adversarial pass, read-only — run 2026-06-10, fixes applied)
**NUMBERS — clean on arithmetic** (every load-bearing number reproduced to the digit: 1000/980/20/0.02; 2767.0 =
20×138.35 exact; 0.769 h; 138.35 traced verbatim to `wi1_accounting.json` silent_10C=1000.N8 — WI-1's 138.4 is its
rounding; onset 64/72 + tgid 3320702 in the real event; copy md5-identical to source; degraded list = 20 unique
uuids × 9 fields; collector re-run byte-identical). 3 minor + 2 nit labeling findings — all folded into the fixes
below. **INTEGRITY — passes all six checks** (anchor md5 now-verified; no fork-1 writes in the WI window; /tmp
determinism proof; no leftover procs; constants tie to WI-1 artifacts); 2 minor: in-place artifact regeneration
(hygiene adopted above) + the undisclosed prereg deviation (now §PREREG DEVIATIONS). **HONESTY — 2 MATERIAL, both
FIXED:** (1) bottom-line "real-derived fault onset (step+wall)" contradicted by the artifact in 20/20 entries
(steps synthetic 24+7i, none = real 64/72; single replayed wall) → reworded everywhere, view field renamed
`fault_onset_wall_replayed`, provenance block added; (2) prereg-promised fleet effective-time silently dropped while
its computation falsifies prereg expectation 2 → computed in v2 + §PREREG DEVIATIONS. Minors fixed: most-favorable/
upper-bound/gross labels on 138.35 & 0.77; "measured"→"modeled-accounting"; "varied severity" struck + disclosed;
schema-conformance claim re-scoped. **LOGIC — 2 MATERIAL, both FIXED:** (1) schema validation was vacuous (rejection
path structurally dead in the replay; `assert` stripped under `python -O`; DEGRADED-with-null-onset accepted with
full badput) → negative tests 4/4 rejected + view-corruption check + `ValueError` + tightened `schema_ok`
(schema_version/severity/DEGRADED⇒onset), verified under `-O`; (2) the deferral rationale ("double-counting") was
wrong and the prereg formula omitted healthy-GPU overhead → fleet eff-time computed (−5.11pp @N=8 / −0.80pp @N=45 at
the depicted mix), fleet fault rate defined (per GPU-step) with window field, breakeven equivalence stated under
explicit homogeneity assumptions (cross-check 17,818 ≈ 17,820 ✓). Minors fixed: idempotence now tested at totals
level incl. changed-verdict update (panel had probe-verified the code was already correct — confirmed 20→19,
2767.0→2628.7); linearity relabeled by-construction. Panel verdict after fixes: the narrow validated claim — the
uuid-keyed aggregation logic + schema work offline over a simulated population — **stands**; the headline now
carries the same honesty standard as WI-1/WI-2.

## HONEST BOTTOM LINE
The rc.v1 per-GPU reliability event is fleet-consumable: a uuid-keyed collector produces a correct fleet reliability
view (degraded list with synthetic-onset/replayed-wall labeling + descriptor + WI-1-tied GROSS badput attribution +
fleet totals + computed fleet NET economics) with negatively-tested schema conformance and totals-level idempotence.
What is validated is the aggregation LOGIC + schema over a SIMULATED fleet (one real per-GPU event, replayed
population); real multi-GPU scale and a deployed collector service remain unbuilt. The fleet NET story is now
explicit and unflattering at the depicted mix: 2% degraded over a 10k-step window is NET-NEGATIVE (−5.11pp @N=8,
−0.80pp @N=45; gross 0.77 GPU-h avoided vs 21.6/3.8 GPU-h overhead) — the detector pays fleet-wide only above
WI-1's breakeven fault rate (1/17,820 @N=8, 1/98,121 @N=45 GPU-steps), i.e. in the silent-SDC regime at realistic
fault rates on large fleets, exactly as WI-1 bounded it.
