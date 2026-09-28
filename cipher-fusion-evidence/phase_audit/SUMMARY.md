# CIPHER phase-by-phase evidence audit — SUMMARY

**Date:** 2026-05-15. Read-only audit. Six parallel audits cross-checked
report claims against on-disk artifacts (md5sum, line counts, file
timestamps, logs, /dev nodes, lsmod). Per-phase detail:
`phase_audit/phase{0,1,2,3,4}/audit.md`, `phase_audit/non_canonical/audit.md`.

No git history exists; file timestamps are the dating evidence.

## Canonical checkpoint scoreboard (23 CPs, Phase 0–4)

| Phase | CPs | SHIPPED | PARTIAL | NOT DONE | SUPERSEDED |
|---|---|---|---|---|---|
| 0 | 6 | 1 | 1 | 3 | 1 |
| 1 | 5 | 3 | 2 | 0 | 0 |
| 2 | 5 | 1 | 1 | 3 | 0 |
| 3 | 4 | 0 | 2 | 2 | 0 |
| 4 | 3 | 0 | 2 | 1 | 0 |
| **Total** | **23** | **5** | **8** | **9** | **1** |

SHIPPED: 0.3 · 1.2 · 1.4 · 1.5 · 2.2. SUPERSEDED: 0.1.
PARTIAL: 0.2 · 1.1 · 1.3 · 2.3 · 3.1 · 3.2 · 4.1 · 4.3.
NOT DONE: 0.4 · 0.5 · 0.6 · 2.1 · 2.4 · 2.5 · 3.3 · 3.4 · 4.2.

## Non-canonical T4.x.x work (19 deliverables)

| Status | Count | Deliverables |
|---|---|---|
| SHIPPED (several caveated) | 11 | T4.2.1, T4.2.4a (plumbing only), T4.2.4d (enforcement; isolation falsified), T4.3.0, T4.3.1 (user-process path degraded), T4.3.2, T4.3.x envelope, T4.5 .symver substrate, T4.6.0/0.5, T4.6.2, T4.6.4 |
| PARTIAL | 6 | T4.2.2, T4.2.3, T4.2.4b, T4.2.4e, T4.6.1, T4.6.3 |
| NOT DONE / regressed | 2 | T4.2.4c (hypothesis unsupported), T4.5 Marlin (regressed −32% / hung) |

**Every T4.x.x deliverable is PARALLEL — none supersedes a canonical
checkpoint.** The canonical Phase 0–9 plan contains no SM-partition,
DVFS, matmul-routing, or KV-dedup checkpoint, so equivalence cannot be
argued for any of them.

## The honest big picture

**The canonical plan and the actual work are two largely-disjoint
tracks.** Of 23 canonical Phase 0–4 checkpoints, **5 shipped clean, 1
was superseded, 8 are partial, 9 never ran.** Meanwhile the project's
substantial real effort — 19 T4.x.x deliverables — ran *entirely
parallel* to the canonical plan and substitutes for none of it.

The drift is documented, not accidental: Phase 0's own
`DIAGNOSTIC_REPORT_2026_05_13.md` records a deliberate mid-phase pivot
(the density sweeps were judged to be measuring a prototype-harness
ceiling, so the team jumped to "Phase 1 — cipher_kmod skeleton"). From
there the project organized around "Phase 1 / 1.5 / 2 / 3 / 4" milestones
and then the T4.x.x series — never the canonical CP structure. The
canonical plan was effectively abandoned after Phase 0 without being
formally retired.

What *is* real and solid: the kmod loads and observes (Phase 1); the
v2 CUDA-injection + tenant-identity plumbing (Phase 2); a Prometheus
telemetry substrate (Phase 3, though not the canonical Grafana/ClickHouse
gate); and the strongest recent work — T4.3.2 (airtight n=5 DVFS stats),
T4.5.1 (.symver substrate), T4.6.2 (VMM KV allocator), T4.6.4
(cross-process kmod dedup, 5/5 indicators). What is *not* real: the
canonical memory-system Phase 4 (L2 pinning / TMA / persistent-kernel
routing — 0/3 to gate), continuous PMU FLOP counting, and any product
deliverable (`cipher-platform.deb` does not exist).

## Cross-phase honesty flags (surfaced by the audit)

1. **The 2.96× tok/W figure is misattributed.** It appears in
   `cipher-may13-evidence/INVESTOR_REPORT.md` and `SCORECARD.md` but was
   measured 2026-05-02 on the *old LD_PRELOAD stack* (Llama-3.1-8B, B=1)
   — the very stack Phase 2 was meant to replace. It was never
   reproduced through the v2 dispatch; the v2 library contains zero
   Marlin/DVFS/speculative code. T4.3.x's own envelope work indicates a
   DVFS headline like this does not generalize past small models. If
   the investor report is used externally, this number is stale.
2. **`cipher_main.c`'s `MODULE_DESCRIPTION` advertises "PMU telemetry"**
   — there is no PMU code, no `perf_event_open`, no FLOP counter in the
   kmod. (CP 3.3 NOT DONE.)
3. **The T4.2.2 / T4.2.3 WL05 headline tables were retracted** by a
   2026-05-14 correction (the benchmark loaded `libcipher_v2`, not
   `libcipher_rt` — WL05 was never actually measured under the router).
4. **T4.x reliably ships substrates/mechanisms but reliably fails to
   demonstrate the headline product lift** — SM-partition product
   hypothesis refuted/unsupported across all five T4.2.4 sub-phases;
   VOLT +57% tok/W real but TinyLlama-only (negative on 7B+); Marlin
   never cleared a gate; the attention substrate sees only ~14% of
   compiled decode. The multi-tenant KV-dedup moat's real-KV product
   value remains unmeasured pending T4.6.5.

## NOT DONE canonical CPs — what closing each requires

| CP | Scope | What would close it |
|---|---|---|
| **0.4** | Test A continuous decode sweep N=4…128 | The harness (`density_harness.py`, continuous path) exists and is verified — run the sweep, capture Story-A numbers. |
| **0.5** | Test B agentic burst sweep N=8…192 | Harness burst path exists — run it; capture Story-B numbers. |
| **0.6** | Phase 0 final report (md+JSON, both stories, MFU/HFU/TPW/density per step, $/M-tok power-only) | Requires 0.4 + 0.5 data first, then author the report. The metrics collector already emits the canonical metric set. |
| **2.1** | libcipher_hook port/keep/delete inventory | Produce the actual symbol-level inventory artifact (none exists; the insight report only argues strategy). |
| **2.4** | Marlin+DVFS+speculative ported, 2.96× reproduces through v2 dispatch | Port the three actuators into the v2/substrate path and re-measure. **Honest expectation:** the envelope evidence says 2.96× will not reproduce — closing this is an honest re-measurement, not a number to assume. |
| **2.5** | LD_PRELOAD dropped, `cipher-platform.deb` | Build the `.deb` packaging; dropping LD_PRELOAD depends on the substrate being the sole proven path (T4.6.1 coverage gap currently blocks that). |
| **3.3** | PMU continuous FLOP counting | Implement `perf_event_open` / GPU PMU programming for a continuous FLOP series. Nothing exists today. |
| **3.4** | Grafana + ClickHouse dashboard | Stand up ClickHouse + Grafana — or formally re-scope to "the Prometheus `/metrics` exporter is the telemetry surface" and retire the canonical gate. |
| **4.2** | TMA-based memory substitution | `PHASE_4_DEPTH_AUDIT.md` states TMA is not exposed in this `cuda.h` and the substitution engine emits no TMA descriptors. This CP likely needs to be **retired**, not closed, in the current toolchain. |

PARTIAL CPs (0.2, 1.1, 1.3, 2.3, 3.1, 3.2, 4.1, 4.3) have specific
named gaps in their per-phase `audit.md` files — e.g. 1.3 has no DKMS
packaging, 3.1 has no external review, 4.1's L2 mechanism fires but its
tok/W gate was never measured.

## For adjudication

The audit's job was to find out what happened, not to make it look
orderly. The finding: **CIPHER has real, verified substrate engineering
and a documented history of drift from its own canonical plan.** The
decision to make together is which canonical CPs to (a) actually run
and close, (b) formally retire/re-scope as superseded-by-direction, or
(c) leave open — and whether the canonical Phase 0–9 plan should be
re-baselined around the T4.x architecture that the project actually
built.
