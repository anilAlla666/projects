# Week 3 — CLOSEOUT

**Headline: WEEK 3 COMPLETE.** All 4 implementation steps + 1 closeout PASS. Classifier-driven routing now LIVE by default (CIPHER_DISPATCH_LIVE=1); SENSE classify pipeline + transition-detection wrapper + DSM PROPOSE infrastructure all functional end-to-end. Mistral-7B SC6 bit-identical at LIVE=1 confirmed (the load-bearing correctness gate).

**Date:** 2026-05-20
**Phase:** v1.2.2 §7 Week 3 → Week 4 boundary

---

## §1 — Steps summary

| # | step | status | anchor (tree:tag) | result doc |
| ---:| --- | --- | --- | --- |
| 1 | `cipher_rt_dispatch.{cpp,h}` scaffold + CIPHER_DISPATCH_LIVE env | PASS | `cipher_rt_phase4 @ 0b6effdb` (`week-3-step-1-dispatch-scaffold`) | WEEK_3_STEP_1_RESULT.md |
| 2 | Observer extension: TLS substitute_hint publish | PASS | `cipher_rt_phase4 @ 71bf5523` (`week-3-step-2-observer-hint-publish`) | WEEK_3_STEP_2_RESULT.md |
| 3 | Marlin hint consumption + VOLT discrepancy surfaced (no edit) | PASS | `cipher_rt_phase4 @ 4f1a86ab` (`week-3-step-3-marlin-hint-volt-lock`) | WEEK_3_STEP_3_RESULT.md |
| 4 | Step 4 (initial) Shape-3 STOP → Step 4 Option II-a (LIVE flip + SENSE + invented transition wrapper + DSM PROPOSE) | PARTIAL × 1 → PASS | `cipher_rt_phase4 @ 79c1b4f9` (`week-3-step-4-opt2a-dispatch-live-sense`) + `cipher_kmod @ a21a45ee` (`week-3-step-4-opt2a-dsm-propose`) | WEEK_3_STEP_4_RESULT.md (PARTIAL) + WEEK_3_STEP_4_OPT2_PREFLIGHT.md + WEEK_3_STEP_4_OPT2A_RESULT.md |
| 5 | This closeout | — | all three trees: `week-3-complete` | (this document) |

---

## §2 — Tree state at closeout

| tree | HEAD | tags | artifact md5 |
| --- | --- | --- | --- |
| `cipher_rt_phase4` | `79c1b4f9` | `week-3-step-4-opt2a-dispatch-live-sense`, `week-3-complete` | libcipher_rt.so: `daaccc40` (this build; nvcc non-deterministic) |
| `cipher_kmod` | `a21a45ee` | `week-3-step-4-opt2a-dsm-propose`, `week-3-complete` | cipher_kmod.ko: `8401f31a` (deterministic) |
| `cipher-may13-evidence` | `fc8a9ae6` | `week-3-complete` (also `week-2-complete` and `week-1-step-1-lp7-rename`) | (no Week 3 source change) |

---

## §3 — Anchor evolution

### cipher_rt_phase4

```
Week 2 close      f9c32322   (week-2-complete)
Step 1 dispatch   0b6effdb   (week-3-step-1-dispatch-scaffold)
Step 2 observer   71bf5523   (week-3-step-2-observer-hint-publish)
Step 3 Marlin     4f1a86ab   (week-3-step-3-marlin-hint-volt-lock)
Step 4 II-a       79c1b4f9   (week-3-step-4-opt2a-dispatch-live-sense, week-3-complete)
```

libcipher_rt.so md5 progression (nvcc non-deterministic; commit-time snapshots):
```
Week 2 close      f6251e62
Step 1            201dc1a1
Step 2            ca3492e3
Step 3            f63e1f2e
Step 4 II-a       daaccc40
```

### cipher_kmod

```
Week 2 close      0ce4b8e2   (week-2-complete)
Step 4 II-a       a21a45ee   (week-3-step-4-opt2a-dsm-propose, week-3-complete)
```

cipher_kmod.ko md5: `8d1f1fc4 → 8401f31a`. Kmod build deterministic; reproduces.

---

## §4 — Rollback paths verified

| target | command |
| --- | --- |
| pre-Week-3 | `git -C cipher_rt_phase4 reset --hard week-2-complete` |
| post Step 1 | `git -C cipher_rt_phase4 reset --hard week-3-step-1-dispatch-scaffold` |
| post Step 2 | `git -C cipher_rt_phase4 reset --hard week-3-step-2-observer-hint-publish` |
| post Step 3 | `git -C cipher_rt_phase4 reset --hard week-3-step-3-marlin-hint-volt-lock` |
| post Step 4 | `git -C cipher_rt_phase4 reset --hard week-3-step-4-opt2a-dispatch-live-sense` + `git -C cipher_kmod reset --hard week-3-step-4-opt2a-dsm-propose` |
| Week 3 complete | `git -C cipher_rt_phase4 reset --hard week-3-complete` + `git -C cipher_kmod reset --hard week-3-complete` |
| Disable LIVE without rollback | export `CIPHER_DISPATCH_LIVE=0` → reverts to Week 2 routing semantic |
| Disable SENSE without rollback | unset `CIPHER_SENSE` (default OFF) → no transition observations, no proposals |

Every tag is a clean rollback point; `make clean && make` succeeds at each.

---

## §5 — Findings surfaced during Week 3

### 5.1 Wave 5 §5.5 W3 SYNTHESIS-HYPOTHESIS items resolved inline

| item | how resolved |
| --- | --- |
| W3-1 observer `maybe_handle` vs Step 4 `observe()` naming drift | Step 2 folded the hint-publish prescription into the actual function name; no rename needed |
| W3-7 CIPHER_DISPATCH_LIVE env undefined | Step 1 added the env-read + cached atomic; default flipped to 1 in Step 4 |

### 5.2 Step 4 Shape-3 → Option II-a invented Sub-4

The original Step 4 brief assumed SENSE → DSM PROPOSE wire-up was a mechanical exercise. Pre-flight Sub-4 audit found: **SENSE classifies sessions, doesn't produce migration proposals.** Wave 5 prescribed a behavior that didn't exist in cipher-may13-evidence.

User adjudicated Option II-a (build the transition-detection wrapper as new code). Result: ~230 LOC of new `cipher_rt_sense_transition.{c,h}` with hysteresis state machine + tool-idle heuristic + bounded proposals queue. Research-grade thresholds (N=8 stability, M=4 transition-debounce, K=100ms agent-idle). Successfully fires (5 transitions emitted in SC6 end-to-end smoke).

### 5.3 Sub-4 threshold tuning deferred

N=8, M=4, K=100ms are documented inline as v1 best-guesses. Measurement-driven tuning deferred:
- Week 4 observability tier (LOOP/PIPELINE/PULSE) may produce empirical evidence on transition cadences.
- Phase 5 CP 5.5 100-tenant benchmark is the validation surface.

### 5.4 VOLT 1000 vs 1200 MHz discrepancy

v1.2.2 §7 L1303 prescribes "1200 MHz lock on decode-band"; existing LUT B=1 → **1000 MHz** (sourced from cipher-may13-evidence/p5_optimal_clocks.json measured optimum; Wave 5 L736 agrees). Step 3 honored the measured optimum (no LUT modification); discrepancy surfaced but NOT resolved.

Open question: is 1200 MHz a spec mistake or a deliberate tuning target? Defer to future weeks (likely Week 4 or measurement work post-CP 5.5).

### 5.5 Sub-4 research-grade caveats

- SENSE classifications (HUMAN/AGENT/BATCH/UNKNOWN) are heuristic per may13 README ("ring-buffer timing and shape sequences").
- The transition wrapper inherits this heuristic provenance.
- Proposals are observability-only (Pattern a); kmod accepts + queues + emits at /proc but does NOT act.
- Pattern (b) auto-action would require SENSE to be production-quality; out of scope for v1.

### 5.6 Failure-mode-of-record (paid forward to Week 4 scope-lock)

Wave 5 prescriptions naming "X → Y" (where X is a may13 actuator and Y is a kmod ioctl) need explicit pre-flight checks on **both**:

- Producer's actionable-output exposure (is there an API that produces what Y consumes?)
- Consumer's ioctl/symbol existence

Two examples this week: Sites 1+2 cuBLAS/SDPA CLASSIFY (Week 2 Step 6 pre-flight); SENSE → DSM PROPOSE (Step 4 pre-flight).

---

## §6 — Week 4 entry readiness

### What Week 3 delivered that Week 4 needs

- **Classifier-driven routing LIVE**: every kernel launch fires CLASSIFY route(); GEMM-class kernels publish substitute_hint; Marlin actuator reads + consumes hint.
- **SENSE observation pipeline functional**: cipher_sense_observe per launch produces session classifications (under CIPHER_SENSE=1).
- **DSM PROPOSE infrastructure shipped**: ioctl ABI + kmod ring + /proc emitter all functional. Producer (transition wrapper) populated; operator can read.
- **Substrate triple complete**: matmul + attn + classify + dispatch all in place.

### Week 4 scope per v1.2.2 §7 + Wave 5 §5.5

- AUDIT lockless refactor (Wave 5 W4 names a target file that doesn't exist; C3 finding from §5.5 verification)
- Observability tier port (TRACE, RECEIPT, CARBON, FAIRNESS, GUARD, COMPLY, LOOP, PIPELINE, DETERMINISM, CONTINUITY, PULSE)
- LP-8 allocator retirement (cipher_partition_allocator)
- Prometheus exporter additions

LOOP and PIPELINE port may produce additional transition signals that complement (or replace) the Sub-4 invented wrapper. Worth checking at Week 4 entry.

---

## §7 — Open items for Week 4 entry window

- **Week 4 pre-flight verification sweep**: mirror Week 3 pre-flight pattern. Particularly check C3/C4 findings from `WAVE_5_S5_5_VERIFICATION.md`:
  - C3: `cipher_audit.cpp` doesn't exist (AUDIT lives inline in `cipher_10ops_impl.cpp:341-378`)
  - C4: `cipher_partition_slot[32]` naming + count drift (LP-8 retirement target)
- **Real `cipher::oracle_decide` wiring**: deferred from Step 2 hardcode `permit=1`. Decision required: extend deferred-init to allow oracle decide, OR keep permit hardcoded (Marlin's own gate is binding anyway).
- **VOLT 1000 vs 1200 MHz discrepancy**: still unresolved from Step 3.
- **Sub-4 threshold tuning**: research-grade thresholds in `cipher_rt_sense_transition.c` need measurement-driven calibration.
- **LOOP/PIPELINE port assessment**: do these may13 actuators produce transition signals that obsolete the Sub-4 invented wrapper?
- **Sites 1+2 cuBLAS/SDPA CLASSIFY wiring revisit**: still N/A (geometry mismatch persists); CUPTI catches downstream launches.
- **Mistral-7B SC6 with CIPHER_SENSE=1 baseline**: bit-identical confirmed at SC6; future workload tests should re-confirm at scale.

---

## §8 — Honest accounting

### Time

| step | budget | actual |
| ---:| ---:| ---:|
| 1 dispatch scaffold | 2-3h | ~2h |
| 2 observer hint | 2-3h | ~2h |
| 3 Marlin hint + VOLT | 3-4h | ~2h (VOLT not modified) |
| 4 (initial Shape-3 STOP) | 0 | ~1h (pre-flight + STOP) |
| 4 Option II pre-flight | 0 | ~1.5h |
| 4 Option II-a impl + gates | 6.5-9h | ~5-6h |
| 5 closeout | 1-2h | ~1h |
| **total** | **14-20h** | **~14-15h** |

Step 4 PARTIAL + Option II pre-flight added ~2.5h of paperwork beyond the original Step 4 budget but the actual Option II-a implementation came in below the pre-flight estimate (the invented Sub-4 wrapper was ~230 LOC vs ~250-300 estimated; build clean on first try).

Week 3 total under budget despite the Shape-3 detour.

12-14 week trajectory: ~1.5 weeks of plan reserve absorbed cumulatively (Week 2 Step 2 audit chain + Week 3 Step 4 Shape-3 detour). 10.5-12.5 weeks remain for Weeks 4-14. Plan reserve still sufficient.

### Quality

- Every step cleared SC6 bit-identical gate (TinyLlama for Steps 1-3; TinyLlama + Mistral-7B for Step 4)
- LP-2 SDPA invariant tracked across all 4 steps; held at every gate
- Step 1 → Step 4 anchor chain is clean rollback-by-tag
- 1 STOP-and-surface event (Step 4 Shape-3); resolved through user adjudication (Option II-a)
- 0 mitigation-by-default

### Substantive deliverables

Three substrates now active in cipher_rt_phase4:
- **matmul** (cuBLAS GemmEx GOT-patched; Marlin actuator + classifier hint)
- **attn** (SDPA trampolines route()-aware; LP-2 invariant preserved)
- **classify** (CUPTI per-launch → may13_default → observer; TLS hint publish; Marlin consumes)

Plus the new **dispatch** substrate (Step 1) and **SENSE+DSM** observation pipeline (Step 4 Option II-a).

`/proc/cipher/` exposes 8 nodes (was 7): + dsm_proposals.
Classifier-driven routing is LIVE by default. Operators can disable via `CIPHER_DISPATCH_LIVE=0` without rebuild.

---

## Tags at close

```
cipher_rt_phase4 tags:
  week-3-complete            = week-3-step-4-opt2a-dispatch-live-sense
                             = 79c1b4f996fd0ddd9fce86b0b9ae8c4e1cd3271b

cipher_kmod tags:
  week-3-complete            = week-3-step-4-opt2a-dsm-propose
                             = a21a45ee7a1fa8afb1277fac20a6d49e621fe5a2

cipher-may13-evidence tags:
  week-3-complete            = fc8a9ae6 (unchanged from Week 2)
```

Week 3 closed. Week 4 brief pending.
