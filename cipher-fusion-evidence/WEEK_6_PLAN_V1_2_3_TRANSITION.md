# WEEK_6_PLAN_V1_2_3_TRANSITION.md

**Date:** 2026-05-23
**Purpose:** Document the v1.2.2 → v1.2.3 transition for the binding plan (`CIPHER_REENGINEERING_PLAN.md`), in language suitable for posterity, investor data-room consultation, and NVIDIA ICC / Devang / neocloud-operator communication.
**Trigger:** `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` (md5 `c5d2d4ad68796578caaae022ff25e743`, 640 lines, dated 2026-05-23, `cipher-fusion-evidence` commit `07c2212`) surfaced six load-bearing gaps unscheduled by v1.2.2 §7.
**Co-document:** `CIPHER_REENGINEERING_PLAN.md` v1.2.3 (lands in same commit).
**Baseline preserved:** `CIPHER_REENGINEERING_PLAN.md.v1.2.2.archive` (md5 `40722374f7a9b4b93d56381add208c92`, 1748 lines, byte-identical to v1.2.2 disk content at audit-time).

---

## 1. Why v1.2.3

The v1.2.2 plan §7 sequenced 12-14 weeks ending in CP 5.5: the headline 100-tenant heterogeneous benchmark on the 30-of-33-op unified runtime. Three threads of work landed in Week 6 (2026-05-21 through 2026-05-23) that revealed a structural gap in that plan:

1. **The Week-6 architecture-gap audit** (`07c2212`, this week) decomposed the product target — 100 concurrent agents per H100, each potentially running a different model architecture, driver-level multiplexing via LD_PRELOAD, zero application code changes — into 18 stable-IDed requirements and audited the v1.2.2 substrate against each. Six load-bearing gaps surfaced as unscheduled by v1.2.2 §7: G1 (CP54 ALLOCATE table cap = 64, the single structural blocker), G2 (weight-arena slots = 16), G3 (KV-dedup hash content-only → silent cross-model corruption), G4 (Marlin weight kit process-global, single-model), G5 (VA pool 80 GiB per-process → fails at N≈12), G6 (no kmod-resident AUDIT chain). Plus G10 (CIPHER_REGISTER_MODEL ABI missing, prerequisite for G3/G4/G12) and G12 (Koopman registry model-keying missing).

2. **The Week-6 bench harness work** (`cc913a6`, this week) replaced `bench_mfu.py` with an industry-standard harness (`bench_llm.py`, md5 `558865fd914c48ab843b3a5cfb10a452`) mapped to MLPerf Inference v5.1 + Karpathy/PaLM MFU + TokenPowerBench + vLLM streaming. The harness's Option-1 redo confirmed directly: single-process single-instance B-sweep CIPHER is **neutral** vs vanilla. The real lever for tok/W is cross-tenant batching. This in turn revealed that v1.2.2's "100-tenant heterogeneous" framing in CP 5.5 was workload-class heterogeneity (5 prefill + 80 decode + 15 burst), **not** model-architecture heterogeneity — the difference is load-bearing for the product target.

3. **The reframe:** The product target — 100 different model architectures coexisting on one H100 — is in the NEOCLOUD B2-B + B3 envelope (different-model two-customer + 5-customer 5-model dynamic load), which `NEOCLOUD_SUBSTRATE_AUDIT.md:501-563` already documents as "~0% KV-dedup hits, 0 GB weight savings" without the substrate keying that G3 + G4 + G12 + G10 land. The v1.2.2 plan does not schedule those gaps. v1.2.3 folds them in.

**The decision:** B.1 — re-sequence not redesign. Same architecture, same op surface (30 of 33), same actuator-registry pattern, same five product goals. Timeline extends from 12-14 weeks to 15-17 weeks to absorb the six gaps inside the existing track structure. CP 5.5 reframes from workload-class heterogeneous to hybrid workload-class + model-architecture heterogeneous on the unified runtime.

---

## 2. What stays the same

v1.2.3 is **a re-sequence**, not a redesign. Everything below is preserved verbatim from v1.2.2:

- **Architecture.** Tree A (`cipher-may13-evidence/`) classifier + observer brain. Tree B (`cipher_rt_phase4/` + `cipher_kmod/`) actuator hands. Bridged by the priority-ordered actuator-registry pattern (matmul + attention dispatch, plus the classifier substrate added in W1). No change.
- **v1 op surface: 30 of 33.** A1 stands — Koopman tier in v1, NCCL family deferred to v2 on hardware grounds. A4 (NCCL formal deferral, §8.4) stands. A5 (v1 shippability table) stands.
- **§4 unified architecture design.** No change. §4.0 actuator-registry contract (4-value enum from attn + single-arg `maybe_handle` + snapshot-under-lock + break-on-ERROR) stands. §4.8 COMMIT atomic primitive (A2) stands; it absorbs G6's per-tenant AUDIT chain head as the natural insertion point in the deterministic state-update order. §4.9 RING_WRITE lock-free substrate (A3) stands; it absorbs G3 + G4 + G5 in the same window because they share the kmod hot-path-write pattern.
- **The 5 product goals** are all retained. v1.2.3 explicitly names five (Goal 1 = 100-agent heterogeneous-model multiplexing per H100 — the new headline; Goal 2 = MFU per workload-class roofline; Goal 3 = TPW composed lift; Goal 4 = O(1) Koopman compute substitution — retained from v1.2.2 A1; Goal 5 = LD_PRELOAD-only deployment transparency). The v1.2.2 framing was "three product goals" with Koopman-as-fourth and transparency-as-fifth implicit. v1.2.3 makes them explicit.
- **The retracted claims discipline.** The 3.617× single-instance, 14× synthetic, and universal 85% MFU retractions stand. The Week-6 Option-1 redo (`WEEK_6_OPTION_1_REDO.md`) directly confirms the single-instance retraction by reproducing it on the corrected bench harness.
- **The Weeks 1-5 work** is **DONE** with commit anchors recorded in the §7 v1.2.3 amendment (W1 → `fc8a9ae6`; W3 → `79c1b4f9`; W4 → `850bd8b`; W5 → `cipher_rt_phase4` `ec0e005` + cipher-fusion-evidence `4302079`). No re-work.
- **The §8.4 NCCL environment-bound deferral** stands. The 3 deferred ops (NCCL_P2P, OVERLAP, STRAGGLER cross-rank) remain v2 on hardware grounds.
- **ADJUDICATIONS A1, A2, A3, A4, A5** all stand. v1.2.3 reverses no prior adjudication. The Koopman tier moves from W11-12 to W13-14 in calendar position only — A1 is in force.

---

## 3. What changes

| Aspect | v1.2.2 | v1.2.3 |
|--------|--------|--------|
| Timeline | 12-14 weeks | **15-17 weeks** (+3 weeks calendar) |
| CP 5.5 position | W13-14 | **W15-17** (+1 week) |
| CP 5.5 scope | Workload-class heterogeneity (5 prefill + 80 decode + 15 burst, single model family) | **Hybrid:** workload-class heterogeneity **+ model-architecture heterogeneity (≥ 5 different model families coexisting: Mistral-7B + Qwen-7B + Llama-3-8B + 2 SLMs)** |
| W6 use | Reserved-TBD placeholder; two candidate uses (absorb-overflow / fold-forward) | **In-flight (KV-dedup auto-trigger + bench harness audit/rewrite/Option-1 redo + architecture gap audit) + carry (G1+G2 cap bumps, G5 audit, May-13 POC reconstruction kickoff)** |
| W7-8 → W7-9 | COMMIT atomic state-transition primitive (10 eng-days) | COMMIT primitive **+ G6 kmod-resident AUDIT chain (~5 eng-days, folds into the COMMIT contract) + G10 CIPHER_REGISTER_MODEL ABI at NR 27 (~2 eng-days, shares kmod-ABI-bump pattern)** |
| W9-10 → W10-12 | RING_WRITE substrate (8-10 eng-days) | RING_WRITE substrate **+ G3 KV-dedup model-aware keying (~3 eng-days, closes silent cross-model corruption defect) + G4 Marlin tenant-scoped weight kit (~5 eng-days) + G5 VA pool per-tenant sizing (~3 eng-days nominal, path-dependent)** |
| W11-12 → W13-14 | Koopman tier integration (unchanged scope) | Koopman tier integration **+ G12 Koopman registry model-keying (~3 eng-days, piggybacks on recipe-table port)** |
| W13-14 → W15-17 | CP 5.5 100-tenant workload-class heterogeneous | **CP 5.5 hybrid** workload-class + model-architecture heterogeneous + per-model agents/GPU breakdown + per-(model-pair) KV-dedup hit-rate + May-13 POC reconstruction |
| §0 version history | v1.2.2 row "Current" | v1.2.2 row "Superseded by v1.2.3"; new v1.2.3 row "Current" |
| §1 product target | Implicit across multiple docs | **Explicit headline sentence** at top of "Document role" block + 5-goal structure explicit |
| §8 risk register | §8.1–§8.4 + risks-by-tier/deployment/strategic | **§8.5 new** — five architecture-gap risks (R-G3.1, R-G5.1, R-G5.2, R-G6.1, R-G10.1). Plus 5 new per-week risks (R-W7.4, R-W7.5, R-W10.1, R-W10.2, R-W10.3). |

---

## 4. Engineering-day budget delta

Per the architecture-gap audit's per-gap engineering sketches (`WEEK_6_ARCHITECTURE_GAP_AUDIT.md` §3.1):

| Gap | Scheduled week | Eng-days | Notes |
|-----|----------------|----------|-------|
| G1 (CP54 cap 64 → ≥128) | W6 carry | ~1 | kmod 0.5 ABI bump; ~50 LOC |
| G2 (WA cap 16 → ≥100) | W6 carry | ~0.5 | Same kmod 0.5 bump; ~30 LOC |
| G3 (KV-dedup model-keying) | W10-12 | ~3 | ~200 LOC; HARD correctness gate (KL ≤ 5.5e-5 across 6 cross-model pairs) |
| G4 (Marlin tenant-scoped) | W10-12 | ~5 | ~500 LOC; cubin JIT cache stays (K, N)-keyed to avoid thrash |
| G5 (VA pool per-tenant) | W6 audit + W10-12 impl | ~3 nominal | Path-a (plugin-side per-model sizing) is ~150 LOC; path-b (kmod broker) adds ~80 LOC; path-c is foreclosure of Goal 5 (LAST RESORT) |
| G6 (kmod AUDIT chain) | W7-9 | ~5 | ~400 LOC; folds into COMMIT contract — incremental cost from COMMIT alone |
| G10 (CIPHER_REGISTER_MODEL) | W7-9 | ~2 | ~200 LOC kmod + plugin; ioctl NR 27 (additive per `[[cipher-abi-rule]]`) |
| G12 (Koopman registry keying) | W13-14 | ~3 | ~300 LOC; piggybacks on recipe-table port |
| May-13 POC reconstruction | W6 kickoff + parallel | 6-10 | Parallel to W7+ substrate work; not on the critical path |
| **Total net add over v1.2.2 budget** | | **~20-25 eng-days** | |

Calendar shift: **+3 weeks** (12-14 → 15-17). The eng-days do not equate 1:1 to calendar because:
- W6 carry items happen alongside post-Week-5 bench harness + audit landings already in flight (zero new calendar cost).
- W7-9 absorbs G6 + G10 inside the existing COMMIT track (+1 week).
- W10-12 absorbs G3 + G4 + G5 inside the existing RING_WRITE track (+2 weeks).
- W13-14 holds Koopman one position (0 weeks added — pure shift).
- W15-17 absorbs the heterogeneous-model fold-in to CP 5.5 (+1 week, partly for measurement scope, partly for May-13 reconstruction parallel work to land before CP 5.5 runs).

---

## 5. Timeline shift summary

| Milestone | v1.2.2 calendar | v1.2.3 calendar | Shift |
|-----------|------------------|------------------|-------|
| COMMIT primitive (A2) closed | W8 | W9 | +1 week |
| RING_WRITE substrate (A3) closed | W10 | W12 | +2 weeks |
| Koopman tier closed | W12 | W14 | +2 weeks |
| **CP 5.5 headline benchmark complete** | **W14** | **W17** | **+3 weeks** |

If a stakeholder has v1.2.2's "W14 CP 5.5" date in mind, the v1.2.3 substitute is **W17**. If a stakeholder has v1.2.2's "12-14 weeks" envelope in mind, the v1.2.3 substitute is **15-17 weeks**.

---

## 6. Communication implications

The v1.2.3 timeline shift moves the CP 5.5 headline-benchmark date out by ~3 calendar weeks vs v1.2.2. Communication implications:

- **No external timing commitment surfaced in current memory.** The most recent memory anchors (`[[week6-bench-harness]]`, `[[week6-arch-gap-audit]]`, `[[week5-complete]]`, `[[week6-entry-fold-forward]]`, `[[neocloud-substrate-audit]]`) do not reference any pre-committed CP 5.5 date with Array, EverGiven, NVIDIA ICC, Devang, Nebius, or any other external party. **Before any external communication, the user must verify whether any such commitment exists in channels not represented in CIPHER memory.**
- **The data room.** Investor data-room materials that quoted v1.2.2's "12-14 weeks" to CP 5.5 will need a one-line update to "15-17 weeks." The scope expansion (workload-class + model-heterogeneous) is a strict improvement to the demonstration — the headline becomes more product-relevant, not less.
- **The NVIDIA ICC channel** (if relevant). The ABI bump at NR 27 (CIPHER_REGISTER_MODEL, G10) is purely additive per `[[cipher-abi-rule]]`. No NVIDIA ABI is touched. The `CUDA_INJECTION64_PATH` contract is unchanged. The Marlin × primary-context-pin restriction (`[[cipher-marlin-primary-ctx-pin]]`) is unchanged — Phase 6 partition-aware Marlin remains a Song Han engagement at v2.
- **The neocloud-operator channel.** The v1.2.3 plan strengthens the per-tenant cryptographic billing story (G6 kmod-resident AUDIT chain land at W7-9). If a neocloud-operator has been told to expect billing receipts post-CP-5.5, the date moves out 3 weeks but the receipts become production-grade rather than aspirational.

---

## 7. Pre-emptive correction language

In case any external party has heard a specific v1.2.2 date or scope and asks why it has changed, the honest single-paragraph framing is:

> *"We extended the v1 plan by 2-3 weeks to fold in six substrate gaps surfaced by an architecture audit. Same architecture, same goals, larger demonstration scope. CP 5.5 will now include both workload-class heterogeneity (prefill + decode + burst mix) AND model-architecture heterogeneity (≥ 5 different model families coexisting on one H100) instead of workload-class only. The Koopman O(1) substitution tier stays in v1. The six gaps were not in v1.2.2 because they were not surfaced until the bench-harness audit and architecture-gap audit landed this week; we caught them before CP 5.5, not after. We have a full file:line-cited audit doc available in the data room if anyone wants to read through it."*

Variants for specific audiences:

- **Investor:** *"The plan tightened. Three more weeks. We caught six substrate gaps before the headline benchmark instead of after; the benchmark itself got bigger (different models, not just different workloads)."*
- **NVIDIA ICC:** *"We added one additive ioctl NR (CIPHER_REGISTER_MODEL, NR 27). No NVIDIA ABI is touched. Plan timeline extends 12-14 → 15-17 weeks to cover heterogeneous-model multi-tenant correctness."*
- **Neocloud operator:** *"The per-tenant billing chain (kmod-resident AUDIT chain) moves to W7-9, lands in production form three weeks later than originally scoped. The CP 5.5 demonstration will include external HMAC-chain verifiability."*

The "pre-emptive correction" framing assumes the listener has heard a specific v1.2.2 date; if they have not, no correction is needed. **The user should confirm before initiating any external communication based on this framing.**

---

## 8. Closing — audit-trail integrity

The v1.2.2 baseline is preserved at `CIPHER_REENGINEERING_PLAN.md.v1.2.2.archive` (md5 `40722374f7a9b4b93d56381add208c92`, 1748 lines, mtime preserved from the 2026-05-21 W5 Step 0 doc-cleanup commit `5fdc075`). The v1.2.3 plan is reproducible: every change carries a `<!-- v1.2.3: source -->` HTML comment in the body, and every change is enumerated in §0 "What changed in v1.2.3" + the V1.2.3 AUDIT TRAIL block at the bottom of the plan. Readers who want to reconstruct what moved between v1.2.2 and v1.2.3 can `diff` the archive against the live document.

The B.1 task brief pre-condition cited expected v1.2.2 md5 `f6146a6cae2628fd5e3486cc647401d7` (1743 lines). The on-disk md5 at task time was `40722374f7a9b4b93d56381add208c92` (1748 lines). Git history showed the last edit was commit `5fdc075` on 2026-05-21 (Week 5 Step 0 doc-cleanup) — predating the 2026-05-23 architecture gap audit at commit `07c2212`. The audit's file:line citations (`L1515`, `L1521`, `L1548`, `L1593`, `L1726-1748`) are consistent with the 1748-line on-disk version. The spec md5 was a stale pre-computed value, not evidence of post-audit edits. User reconciled via AskUserQuestion: proceed with disk md5 as the v1.2.2 baseline. This reconciliation is recorded in the v1.2.3 audit trail block.
