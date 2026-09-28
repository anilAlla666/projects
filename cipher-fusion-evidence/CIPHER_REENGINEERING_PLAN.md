# CIPHER REENGINEERING PLAN — Unification of may13 + cipher_rt_phase4 + cipher_kmod into one workload-aware control-plane runtime

**Date:** 2026-05-20
**Type:** Binding architecture plan. Planning only — no code changes during this document's authorship.
**Status:** v1.2.3 — B.1 timeline extension. Folds six architecture gaps from `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` (cipher-fusion-evidence commit `07c2212`) into the existing v1.2.2 track structure. Timeline extends from 12-14 weeks to **15-17 weeks**. CP 5.5 headline benchmark moves from W13-14 to **W15-17** and reframes as hybrid workload-class + heterogeneous-model on the unified runtime. Five product goals all retained including Koopman tier (Goal 4, held to W13-14; reverses no prior adjudication). Same architecture, same op surface (30 of 33), same actuator-registry pattern. Folds onto v1.2.2 (Option B scope expansion + COMMIT/RING_WRITE adjudications) which folded onto v1.2.1 (doc-only) which folded onto v1.2 (Deep Inspection + Wave 1-5 audits).
**Author:** Claude Opus 4.7 (1M context), under user direction.
**Document length:** see audit trail at bottom for v1.2 / v1.2.1 / v1.2.2 / v1.2.3 byte/line counts and md5.
**Audit trail:** The bottom of this document records the md5 of the file content. v1.1 sealed `e1f047d17d4b6d6424331c301e0052c2`. v1.2 sealed `33080cbd5b6e6d94247a0e8ea0ac6515`. v1.2.1 sealed `8502b12b5cf10daaf99153e5076c7604`. v1.2.2 sealed `40722374f7a9b4b93d56381add208c92` (recomputed from disk after the 2026-05-21 W5 Step 0 doc-cleanup; this supersedes the v1.2.2 audit trail's "recorded in PRE_WEEK_1_ADJUDICATION_CLOSURE.md" deferral). v1.2.3 md5 recorded in the v1.2.3 audit trail block below after this revision is sealed.

---

<a id="section-0"></a>
## SECTION 0 — VERSION HISTORY

<!-- v1.2: new section added per task brief; carries the v1.0 → v1.1 → v1.2 lineage and the high-level changelog so downstream readers can see what moved between revisions. -->

### Lineage

| Version | Date | Origin | Status |
|---|---|---|---|
| v1.0 | 2026-05-20 | Initial draft. Pre-stamp md5 `ee3bc026e98dc98414ab634ec78e8bb6`. | Superseded. |
| v1.1 | 2026-05-20 | Advisor corrections folded back: §5.1 MFU timeline honesty (post-CP-5.5 depth-win program), §4.3.6 DVFS scope rule (fleet-policy multi-tenant, per-batch single-tenant), §5.6 framing reset ("the regime-specific caveats ARE the architecture"). v1.1 md5 `e1f047d17d4b6d6424331c301e0052c2`. | Superseded by v1.2. Preserved at `CIPHER_REENGINEERING_PLAN.md.v1.1.bak`. |
| v1.2 | 2026-05-20 | Deep Inspection Report (9 corrections + 6 new findings) and Wave 1-5 Logic Audit (43 fusion contracts, 16 lossy fusion points, 8 NF-5.x findings) folded back. v1.2 sealed md5 `33080cbd5b6e6d94247a0e8ea0ac6515`. | Superseded by v1.2.1. |
| v1.2.1 | 2026-05-20 | Documentation-only patch: 3 corrections folded from PRE_WEEK_1_ADJUDICATION_CLOSURE.md Part 3 (C1 libcipher_v2 path confirmation; C2 Appendix A kmod anchor 285d102e → 008b3c66; C3 §7 Week 1 behavioral-test concretization with actual runner paths). No body logic changes. v1.2.1 sealed md5 `8502b12b5cf10daaf99153e5076c7604`. | Superseded by v1.2.2. |
| v1.2.2 | 2026-05-20 | **Option B scope expansion** (5 adjudications folded from OP_INTENT_VS_IMPLEMENTATION.md + the v1.2.2 round of PRE_WEEK_1_ADJUDICATION_CLOSURE.md): A1 scope to 30/33 ops with Koopman tier in v1 and NCCL family deferred to v2 on hardware grounds; A2 COMMIT promoted from implicit dispatch-return to atomic state-transition primitive (new §4.8); A3 RING_WRITE promoted from "userspace ring without consumers" to lock-free inline telemetry substrate (new §4.9); A4 formal NCCL deferral (new §8.4); A5 reclassification of v1 shippability (4 REQUIRES-FIX in v1: CLASSIFY/ORACLE/COMMIT/RING_WRITE, plus 5 Koopman ops promoted from V2-SCOPE). Timeline extended from 5 to **12-14 weeks**. §7 sequence updated with Weeks 6-14. R-W7.x and R-W9.x risks added. v1.2.2 sealed md5 `40722374f7a9b4b93d56381add208c92` (recomputed externally from disk after the 2026-05-21 W5 Step 0 doc-cleanup; supersedes the "recorded in PRE_WEEK_1_ADJUDICATION_CLOSURE.md" footnote in the v1.2.2 audit trail at L1745). | Superseded by v1.2.3. |
| v1.2.3 | 2026-05-23 | **B.1 timeline extension** (re-sequence not redesign). Folds six architecture gaps from `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` (md5 `c5d2d4ad68796578caaae022ff25e743`, commit `07c2212`) into the existing track structure: G1 (`CIPHER_CP54_MAX_ALLOCS=64`), G2 (`CIPHER_WA_MAX_ARENAS=16`), G3 (KV-dedup hash content-only), G4 (Marlin weight kit single-model), G5 (VA pool 80 GiB per-process), G6 (no kmod-resident AUDIT chain), plus G10 (CIPHER_REGISTER_MODEL ABI at NR 27) and G12 (Koopman registry model-keying). Timeline extends from 12-14 weeks to **15-17 weeks**. CP 5.5 headline benchmark moves from W13-14 to **W15-17** and reframes as **hybrid workload-class + heterogeneous-model** on the unified runtime. Koopman tier retained in v1 (held from W11-12 to W13-14; reverses no prior adjudication; A1 still in force). Five product goals all retained. §7 sequence updated. New §8.5 risk register entries for the architecture-gap-driven risks (R-G3.1, R-G5.1, R-G5.2, R-G6.1, R-G10.1). v1.2.3 audit trail at the bottom of this document. | Current. |

### What changed in v1.2

The following bullets summarise the substantive changes. Every changed block in the body of the document carries an inline `<!-- v1.2: ... -->` HTML comment that cites the audit source.

- **§1.1 / §2.1 line citations corrected.** `cipher_dispatch.cpp` PASS_THROUGH/SUBSTITUTED line numbers in v1.0/v1.1 (L541/L560-561/L570-573/L577-579/L584/L589) matched the silently-excluded `src/` shadow copy (616 LOC). v1.2 updates to the TOP-LEVEL live file (543 LOC): L448 (Layer 3 not initialized), L468 (ORACLE DENY), L480 (registry MISS — the load-bearing Goal-4 gate), L486 (error_bound > 0.01), L500 (substitute failed), L515 (lone SUBSTITUTED success). The top-level file also contains a separate classify-only branch at L338-426 with 3 additional PASS_THROUGH + 3 additional SUBSTITUTED paths. Source: Deep Inspection Report §B.1.7, §D.1, §D.1.NEW.
- **§2.1 recipe registry updated.** "32 hardcoded Llama-3-70B-scale shapes" replaced by "mixed 32+ entries spanning gemm / Chebyshev / HyperFlux / A100 / shape-parametric categories" per Deep Inspection §D.3.
- **§2.3 COMMIT FSM scoped to cp54_sched.c.** The v1.1 claim that COMMIT lived as both kmod cp54_sched DSM token AND weight-arena commit-on-publish was wrong: weight_arena's "commit" is only a code-comment string at L262, not a state-machine token. v1.2 restricts the COMMIT claim to `cipher_cp54_sched.c` only. Source: Deep Inspection §B.1.
- **§3.5 kmod ABI extension table.** v1.1 listed only generic "future need" rows; v1.2 lands the specific NR 25-29 proposals (FAIRNESS_SET_QUOTA / PREDICT_PUSH_HOT_REGION / L2_BUDGET_SET / CARBON_REGION_SET / KOOPMAN_SURROGATE_ADD) with v1/v1.5/v2 scoping. Cross-referenced to Wave 3's live ABI table; all NRs 1-24 are confirmed in-use, leaving 25+ free for additive extensions per [[cipher-abi-rule]]. Source: Deep Inspection §C.3.
- **§4.0 actuator-registry semantic divergences.** The v1.0/v1.1 plan declared the matmul and attn registries as "the same pattern." Deep Inspection §A.3 surfaced 5 specific divergences (enum cardinality 3 vs 4, single-arg vs two-arg maybe_handle, no-lock vs snapshot-under-lock, break-on-ERROR vs continue-on-ERROR, and the F2-X attn HANDLED-discard). v1.2 names the chosen classifier-substrate contract: 4-value enum (from attn), single-arg maybe_handle (from attn), snapshot-under-lock (from attn), break-on-ERROR (from matmul). The classifier substrate matches the **attn** contract on three of four dimensions plus matmul ERROR-break.
- **§4.7 (new subsection) — 43 fusion contracts.** Wave 5 §5.1 catalogued every meeting point between Tree A (classifier brain + observers) and Tree B (substrate + kmod) as 43 contracts in five categories: 12 Ca (classifier→actuator), 5 Cb (classifier→kmod), 11 Cc (actuator→kmod), 8 Cd (observer→{classifier, actuator, kmod}), 7 Ce (kmod-internal cross-TU). v1.2 inserts §4.7 summarising the contract count per category and referencing Wave 5 §5.1 by name and line range for the full per-contract detail.
- **§5.6 R-A3 KV-dedup risk clarified.** v1.1 implied the kvdedup substrate work was incomplete. Deep Inspection §C.4 verifies the kmod substrate is COMPLETE (5 ioctls wired, no TODOs); the integration risk is upstream at the `cipher_vllm_plugin` layer. v1.2 reframes R-A3 accordingly.
- **§7 week sequence corrected.** Wave 5 §5.5 identified five concrete holes in v1.1's Section 7 sequence. v1.2 folds:
  - **Week 1:** adds LP-7 `CipherKernelEntry` struct rename (compile blocker) and the snapshot reserved-tail extension (Cb.2).
  - **Week 2:** adds LP-2 attn trampoline refactor (required to unblock Week 3 attn lane).
  - **Week 3:** scoped to GEMM-only routing in v1; attn dispatch routing defers to v1.5.
  - **Week 4:** adds AUDIT lockless refactor for N=100 scale (note: Wave 5 LP-4 originally deferred this to v1.5; v1.2 pulls into Week 4 per task brief — flip back to v1.5 if hot-path measurement does not justify).
  - **Week 5:** unchanged.
- **§6 / §8 honest classifier hot-path budget.** Wave 5 NF-5.1 established the "12 ns" budget in v1.1 §6 R-C1 is unsupportable; the honest budget is 100-200 ns per launch, accept 1-2% per-tenant overhead as the primary case. v1.2 updates the budget in §6 (C2 row), §7 Week 2 R-W2.1, and §8 R-C1.
- **§8 risk register augmented.** v1.2 adds two tables: (a) the 16 lossy fusion points (LP-1 through LP-16) from Wave 5 §5.4 with severity rollup (2 blocking, 5 requires-mitigation, 9 minor); (b) the 6 new Deep Inspection findings (F1 classify-only branch, F2 shape-parametric latent bug, F3 AUDIT sync-on-hot-path, F4 CUPTI-warmup race, F5 sysfs DSM tunables, F6 undocumented env vars) with the severities verbatim from the report.
- **Inline citations.** Every substantive change is tagged with a `<!-- v1.2: source -->` HTML comment immediately preceding the changed block. Sources appear as either "Deep Inspection §X.Y" or "Wave N §X.Y" or "Wave 5 NF-5.x" — any reviewer can trace each change back to its audit document.
- **Retracted claims discipline.** Per [[cipher-phase-a-multitenant]] and [[cipher-cp56-closed]]: the 3.617× single-instance, 14× cross-tenant on synthetic, and universal 85% MFU framings have been removed wherever they appeared in v1.0. v1.1 already corrected §5.1 (MFU per-WL) and §5.2 (3.617× retraction). v1.2 verified by grep that no live claim of these retracted figures remains.

### What changed in v1.2.2

<!-- v1.2.2: 5 adjudications folded from OP_INTENT_VS_IMPLEMENTATION.md (the per-op intent-vs-implementation audit) and the v1.2.2 round of PRE_WEEK_1_ADJUDICATION_CLOSURE.md. Every substantive change tagged with `<!-- v1.2.2: source -->` HTML comment. -->

- **A1 — Scope expanded to Option B (30 of 33 ops in v1).** OP_INTENT_VS_IMPLEMENTATION D3.5 surfaced the honest commit-able list. v1.2.1 implicitly carried only 18 SHIP-READY + 2 REQUIRES-FIX = ~20 ops with 8 V2-SCOPE. v1.2.2 moves the 6 Koopman learning ops (REMEMBER, VALIDATE, SPECULATE, ADAPT, SAMPLE/GENERATE, SUBSTITUTE-Koopman lane) and the 4 v1.5-deferred overlay ops (PREDICT, SHIELD, SUSTAIN, THERMOSTAT) into v1 scope. The 3 NCCL multi-GPU ops (NCCL_P2P, OVERLAP, STRAGGLER cross-rank) stay deferred to v2 on **hardware grounds** (single-H100 pod cannot exercise multi-GPU NCCL communication), not engineering grounds — formally documented in new §8.4. Source: OP_INTENT_VS_IMPLEMENTATION.md D3.4 + D3.5.
- **A2 — COMMIT promoted from implicit dispatch-return to atomic state-transition primitive (new §4.8).** OP_INTENT_VS_IMPLEMENTATION Item I-1 (D2 register) surfaced COMMIT as UNDOCUMENTED-INTENT with no design document specifying behavior beyond "return actuator status." v1.2.2 resolves the adjudication by promoting COMMIT to an atomic primitive that updates per-tenant state in a deterministic order (AUDIT chain → FAIRNESS → CARBON → RECEIPT → kmod-resident tenant context) and produces a canonical post-kernel snapshot for observers to read. New §4.8 documents the state update order, the atomicity mechanism (per-tenant sequence counter + release-fence discipline, not a global lock), and the observer protocol change (observers read post-COMMIT snapshot rather than mutate per-kernel state independently). Engineering estimate 10 days (Weeks 7-8). Risks R-W7.1/2/3 added to §8.
- **A3 — RING_WRITE promoted from "userspace ring without consumers" to lock-free inline telemetry substrate (new §4.9).** OP_INTENT_VS_IMPLEMENTATION Item I-2 (D2 register) surfaced RING_WRITE as REQUIRES-INTENT-CLARIFICATION because the may13 ring has no spawning consumer in production (CUPTI supersedes for per-launch counters). v1.2.2 resolves the adjudication by treating RING_WRITE as a distinct substrate from CUPTI: sub-microsecond per-tenant ring written inline by the LD_PRELOAD-equivalent injection, consumed by background threads serving REMEMBER (kernel pattern memory), AUDIT post-processing, and classifier feedback. New §4.9 documents the ring structure, producer/consumer protocols, coexistence with CUPTI (which serves callback-based post-launch counters), and the migration path from dormant may13 infrastructure to live consumers. Engineering estimate 8-10 days (Weeks 9-10). Risks R-W9.1/2/3 added to §8.
- **A4 — Three NCCL ops formally deferred to v2 on hardware grounds (new §8.4).** NCCL_P2P (Op 33), OVERLAP (the NCCL compute-comm overlap scheduler at `cipher_nccl_neural.cpp:259-336` + `cipher_layer2.cpp:22-84`), and STRAGGLER cross-rank attribution (Op 32 NCCL-side; local detection stays in v1). The deferral reason is that the v1 environment is a single-H100 pod without multi-node deployment infrastructure. Acquiring multi-GPU hardware + multi-node deployment is the v2 gate — estimated post-seed quarter 2. Not engineering work; environment work.
- **A5 — v1 shippability reclassification (18 SHIP-READY + 4 REQUIRES-FIX + 5 + 3 deferred).** §2.1 op tier table now carries a "v1 shippability" column. SHIP-READY (18 ops): SUBSTITUTE-Marlin, AUDIT, ARBITRATE, SENSE, GUARD, RECEIPT, CONTINUITY, DETERMINISM, PULSE, CARBON, FAIRNESS, TOPOLOGY, LOOP, PIPELINE, TRACE, COMPLY, VOLT, HIBERNATE-detection. REQUIRES-FIX in v1 (4 ops, primary work): CLASSIFY (port), ORACLE (port + LP-6 fix), COMMIT (build §4.8 primitive), RING_WRITE (build §4.9 substrate). REQUIRES-FIX in v1 (5 Koopman ops moved from V2-SCOPE per A1): REMEMBER, VALIDATE, SPECULATE, ADAPT, SAMPLE/GENERATE, plus the Koopman lane of SUBSTITUTE. REQUIRES-FIX in v1 (4 overlay ops moved from v1.5): PREDICT, SHIELD, SUSTAIN, THERMOSTAT. DEFERRED-TO-V2 (3 NCCL ops on hardware grounds per A4): NCCL_P2P, OVERLAP, STRAGGLER cross-rank.
- **Timeline.** §7 extended from 5 weeks to 12-14 weeks. New Weeks 6-14: W6 (KV-dedup live wire — added between v1.2.1's W4 observability and CP 5.5), W7-8 (COMMIT primitive build), W9-10 (RING_WRITE substrate build), W11-12 (Koopman tier integration: EDMD pipeline real-input wiring, recipe registry seeding for narrow workload domain, SUBSTITUTE-Koopman lane validation), W13-14 (CP 5.5 headline benchmark on full unified runtime with 30 of 33 ops firing).

### What changed in v1.2.3

<!-- v1.2.3: B.1 timeline extension that folds six architecture gaps surfaced in WEEK_6_ARCHITECTURE_GAP_AUDIT.md (cipher-fusion-evidence commit `07c2212`, doc md5 `c5d2d4ad68796578caaae022ff25e743`, 640 lines, dated 2026-05-23) into the existing v1.2.2 track structure. RE-SEQUENCE not redesign — same architecture, same op surface (30 of 33), same actuator-registry pattern, same five product goals. Every substantive change tagged with `<!-- v1.2.3: source -->` HTML comment. -->

- **B.1 — Re-sequence, not redesign.** v1.2.3 retains the v1.2.2 architecture wholesale: actuator-registry pattern (§4.0), 30-of-33 op surface (A1 in force), COMMIT primitive (§4.8), RING_WRITE substrate (§4.9), Koopman tier in v1 (A1 NOT reversed), three NCCL ops deferred (§8.4). What changes is the §7 timeline only — it extends from 12-14 weeks to 15-17 weeks to absorb six architecture-gap closures inside the existing track structure. Source: WEEK_6_ARCHITECTURE_GAP_AUDIT.md §7.2 + §5.4 recommended-option-(b).
- **Six architecture gaps folded into v1.2.3 §7.** Each gap maps to an existing v1.2.2 track and is closed inside that track's extended window:
  - **G1** (`CIPHER_CP54_MAX_ALLOCS=64` at `cipher_kmod/cipher_cp54_sched.c:123`) — the single structural blocker. Cap bump to ≥128 in W6 (carry from this week's audit-landing). ~50 LOC, 1 eng-day.
  - **G2** (`CIPHER_WA_MAX_ARENAS=16` at `cipher_kmod/cipher_ioctl.h:481`) — weight-arena cap raised to ≥100 in W6 alongside G1; kmod 0.5 ABI bump. ~30 LOC, half eng-day.
  - **G3** (KV-dedup hash content-only at `cipher_rt_kv_alloc.h:112`; correctness defect — silent cross-model attention corruption at heterogeneous N>1) — re-keyed to `(model_uuid, layer_idx, head_idx, dtype, content_hash)` in W10-12 alongside RING_WRITE. ~200 LOC, 3 eng-days, teacher-forced KL gate ≤ 5.5e-5 on synthetic Mistral-7B + Llama-3.1-8B N=2 cross-tenant pair test.
  - **G4** (Marlin INT4 weight kit keyed by `weight_ptr` alone at `cipher_rt_marlin_actuator.c:150-169`; process-global, single-model) — re-keyed to `(model_uuid, layer_idx, K, N)` in W13-14 alongside the Koopman tier (G12 pairs with this). ~500 LOC, 5 eng-days. Marlin stays full-GPU-only per [[cipher-marlin-primary-ctx-pin]].
  - **G5** (VA pool 80 GiB per-process at `cipher_vllm_kv.py:58`; fails at N≈12 on 1 TB host) — architecture audit in W6 to pick fix path (a: shrink default + per-model sizing from `hf_config`; b: move VA reservation into the kmod weight-arena fd custodian as authoritative broker; c: container-per-tenant — LAST RESORT, violates R-Dep1 LD_PRELOAD-only goal). Implementation lands in W10-12.
  - **G6** (no kmod-resident AUDIT chain — zero grep hits for AUDIT in `cipher_kmod/`) — folds cleanly into the existing W7-9 COMMIT track (was W7-8 in v1.2.2; extended to absorb G6). Per-tenant HMAC chain head in `struct cipher_pid_stats` (~64 B addition) + append-only per-tenant ring in shared memory. ~400 LOC, 5 eng-days. Synergistic with v1.2.2 W9-10 RING_WRITE substrate.
  - **G10** (CIPHER_REGISTER_MODEL ABI missing) — new ioctl at NR 27 (additive per [[cipher-abi-rule]]), keyed to `model_path + hf_config_hash → model_uuid`. Folds into W7-9 alongside COMMIT (both touch the kmod ioctl surface). ~200 LOC, 2 eng-days. Unblocks G3 + G4 + G12.
  - **G12** (Koopman registry model-keying) — recipe-table lookups in `cipher_recipes.cpp` and `cipher_kernel_table.cpp` take `model_uuid` prefix. Folds into W13-14 Koopman tier integration. ~300 LOC, 3 eng-days.
- **Timeline extends from 12-14 weeks to 15-17 weeks.** W7-8 → W7-9 (+1 week for G6 + G10 fold-in alongside COMMIT). W9-10 → W10-12 (+2 weeks for G3 + G4 + G5 implementation alongside RING_WRITE). W11-12 → W13-14 (Koopman held one position, no scope expansion of its own — G12 keying piggybacks). W13-14 → W15-17 (+1 week for CP 5.5 reframe to hybrid workload-class + heterogeneous-model). Net add: +3 weeks calendar, ~20-25 eng-days net add over v1.2.2 budget. Koopman tier RETAINED in v1 — reverses no prior adjudication. CP 5.5 reframes from "100-tenant workload-class heterogeneous benchmark" to "**100-tenant hybrid benchmark: workload-class heterogeneity (5 prefill + 80 decode + 15 burst) + model-architecture heterogeneity (≥5 different model families coexisting)** on full unified runtime." Source: WEEK_6_ARCHITECTURE_GAP_AUDIT.md §5.4 option (b).
- **Product target language clarified.** Document role section (below) now leads with the explicit 100-agent heterogeneous-model product target: "100 concurrent agents per H100, each potentially running a different model architecture (Mistral / Qwen / Llama / SLMs), driver-level multiplexing via LD_PRELOAD, zero application code changes." This was implicit in v1.2.2; v1.2.3 makes it explicit because the architecture-gap audit revealed that v1.2.2's "100-tenant heterogeneous" language meant workload-class heterogeneity, not model-architecture heterogeneity, and the difference is load-bearing.
- **§1 goals reframed to five.** The v1.2.2 document role section listed "all three product goals" (MFU / TPW / multi-tenant density) with vLLM-composition transparency as an implicit fourth and Goal-4 Koopman O(1) substitution as the Weeks 11-12 work item. v1.2.3 makes the five-goal structure explicit (see Document role below): Goal 1 = 100-agent heterogeneous-model multiplexing per H100 (was "multi-tenant density" in v1.2.2, reframed to lead with the explicit product target); Goal 2 = MFU per workload-class roofline; Goal 3 = TPW (tok/W) composed lift via DVFS + density + cross-tenant batching; Goal 4 = O(1) Koopman substitution (retained in v1, held to W13-14); Goal 5 = LD_PRELOAD-only transparency (vLLM-composition surface preserved, zero application code changes). No goal removed; ordering reflects v1 product priority.
- **§8.5 architecture-gap risks** (new subsection): R-G3.1 (silent cross-model KV corruption until G3 lands), R-G5.1 (VA pool exhaustion at N>~12), R-G5.2 (container-per-tenant foreclosure of Goal 5), R-G6.1 (AUDIT chain non-kmod-resident until G6 lands), R-G10.1 (no model-identity at kmod boundary until G10 lands). Each entry cites WEEK_6_ARCHITECTURE_GAP_AUDIT.md §3 gap-matrix row.
- **§7 W1-W5 marked DONE with commit anchors.** W1 → `fc8a9ae6` (week-1-step-1-lp7-rename), W3 → `79c1b4f9` (week-3-step-4-opt2a-dispatch-live-sense), W4 → `850bd8b` (W4 scope-lock), W5 → `ec0e005` (week-5-complete on `cipher_rt_phase4`) + `4302079` (cipher-fusion-evidence post-close).
- **§7 W6 marked IN-FLIGHT** with three landed sub-items (KV-dedup auto-trigger; bench harness audit + rewrite + Option 1 redo at `cc913a6`; architecture gap audit at `07c2212`) and three carry items (G1 + G2 cap bumps; G5 architecture audit; May-13 POC reconstruction kickoff). The May-13 POC reconstruction is documented as `~6-10 eng-days parallel` work, not on the W6 critical path.
- **Inline citations.** Every v1.2.3 substantive change is tagged with `<!-- v1.2.3: source -->` HTML comment pointing back to WEEK_6_ARCHITECTURE_GAP_AUDIT.md or to the substrate file:line. The full enumeration is this section; readers can grep for `v1.2.3:` to find every changed block.

---

## Document role

This document is the unification plan that merges:

- **Tree A** `/home/ubuntu/cipher-may13-evidence/` (the 33-op canonical codebase, 74 source files in `src/`, plus top-level `cipher_dispatch.cpp` + `cipher_oracle.cpp`)
- **Tree B1** `/home/ubuntu/cipher_rt_phase4/` (the deployed userspace runtime, ~16 active sources building `libcipher_rt.so`, anchor `83afd1ca` as of 2026-05-19)
- **Tree B2** `/home/ubuntu/cipher_kmod/` (the deployed kernel module, 14 sources building `cipher_kmod.ko`, anchor `008b3c66` post-Track-2 SC5)<!-- v1.2.1: anchor corrected per PRE_WEEK_1_ADJUDICATION_CLOSURE.md Part 3 C2 — Track 3 SC3 was userspace-only (libcipher_rt-side only) and did not rebuild the kmod, so the post-Track-2-SC5 anchor `008b3c66` remains the in-tree anchor. ->
- **Tree C** orphan/auxiliary trees: `cipher-phase1-evidence/` (original kmod), `cipher_vllm_plugin/` (vLLM KV-offload bridge), `cipher_kv_bridge` (Track 2 weight arena Python C-ext)

into one coherent runtime that satisfies **all five product goals simultaneously** and ships **30 of 33 canonical ops at marvel depth in v1**:

<!-- v1.2.3: explicit product target sentence added at the top of the goals block per the v1.2.3 §1 reframe (B.1). The v1.2.2 framing was three goals (MFU/TPW/density) with Koopman-as-fourth implicit in Weeks 11-12 and LD_PRELOAD-only transparency as Phase 3 wording. WEEK_6_ARCHITECTURE_GAP_AUDIT.md §7.2 surfaced that the v1 product target had not been stated explicitly anywhere — the 100-agent heterogeneous-model framing was implied across multiple documents (NEOCLOUD_SUBSTRATE_AUDIT B2-A/B2-B/B3, CIPHER_WORKLOAD_ARCHITECTURE Class D) but never the headline of the binding plan. v1.2.3 makes it explicit. -->

> **v1 product target:** 100 concurrent agents per H100, each potentially running a different model architecture (Mistral / Qwen / Llama / SLMs, ~5+ model families coexisting at headline-CP-5.5 density), driver-level multiplexing via `LD_PRELOAD` / `CUDA_INJECTION64_PATH`, **zero application code changes**. "100-tenant heterogeneous" in v1.2.3 includes BOTH workload-class heterogeneity (prefill + decode + burst mix) AND model-architecture heterogeneity (≥5 different model families coexisting at CP 5.5).

1. **100-agent heterogeneous-model multiplexing per H100** — the headline product goal: agents-per-GPU at density (weight sharing across same-model tenants via Track 2 weight arena, KV-prefix dedup via cipher_kvdedup, SM partitioning via CP 5.4 green-context ledger, dynamic SM migration via Track 3 DSM). Customer metric: agents/GPU, cost per agent-hour, per-agent p99. Workload class D per `CIPHER_WORKLOAD_ARCHITECTURE.md:38`. The 100-different-models product framing maps to `NEOCLOUD_SUBSTRATE_AUDIT.md:501-563` B2-B + B3 envelopes. <!-- v1.2.3: Goal 1 reframed from v1.2.2's "Multi-tenant density" to lead with the explicit 100-agent heterogeneous-model framing per WEEK_6_ARCHITECTURE_GAP_AUDIT.md §1.1 + §7.2. -->
2. **MFU per workload-class roofline** — compute-bound prefill: ~67% sustained at the 700 W power cap, 85% post-CP-5.5 depth-win program with partition-aware Marlin + L2 budget enforcement + TMA clusters; memory-bound decode: roofline-limited but optimal via batching + quantization (Marlin INT4 actuator, full-GPU lane per `[[cipher-marlin-primary-ctx-pin]]`).
3. **TPW (tok/W) composed lift** — DVFS via VOLT (+13.9% single-instance memory-bound decode under vLLM at 1200 MHz lock per `FUTURE_SCOPE_A_PHASE_3_5_RESULTS.md`); density via Track 2 weight-share (76% HBM saved at Mistral-7B N=4 per `phase_c/TRACK_2_CLOSEOUT.md`); cross-tenant batching (the real-and-defensible **3.06× Mistral-7B N=4 → 5.98× TinyLlama-1.1B N=16** scaling curve per `cp_5_6/TPW_RETEST_2026_05_19.md`).
4. **O(1) Koopman compute substitution** — RETAINED in v1 per v1.2.2 A1 and v1.2.3 §7 re-sequence. EDMD pipeline with snapshot-feeder fix; recipe registry seeding for narrow workload domain (per-layer rank-parameterized recovery, Tikhonov regularization α = 0.01·σ₁², spectral radius ≤ 1); SUBSTITUTE-Koopman lane validation. **Held from W11-12 to W13-14** in v1.2.3 §7 re-sequence (no scope expansion, just position-shift). v1 scope is the narrow-domain Koopman; wide-domain Koopman remains v2 research. <!-- v1.2.3: Goal 4 retained per v1.2.2 A1 (NOT reversed); the v1.2.3 re-sequence only moves Koopman from W11-12 to W13-14 to make room for G6/G10 in COMMIT and G3/G4/G5 in RING_WRITE. -->
5. **LD_PRELOAD-only deployment transparency** — customer applications run unchanged. Library is injected via `CUDA_INJECTION64_PATH` at install time; the substrate intercepts `cublasGemmEx` and SDPA via GOT-patch; `−0.39%` measured latency overhead under vLLM in graph mode (Phase 3). At v1.2.3 CP 5.5 this composes with the heterogeneous-model multiplexing — zero-app-code transparency is preserved across all five model families.

<!-- v1.2.2: scope expanded to Option B per ADJUDICATION 1 in PRE_WEEK_1_ADJUDICATION_CLOSURE.md (v1.2.2 round). The 6 Koopman learning ops (REMEMBER, VALIDATE, SPECULATE, ADAPT, SAMPLE/GENERATE, SUBSTITUTE-Koopman lane) are IN v1 scope. The 3 NCCL multi-GPU ops (NCCL_P2P, OVERLAP, NCCL-side STRAGGLER) are deferred to v2 because the v1 environment is a single-H100 pod that cannot exercise multi-GPU NCCL communication — environment-bound deferral, not engineering deferral. -->

> **Koopman scope reconciliation (2026-05-23, user adjudication).** v1.2.2 ADJUDICATION A1 reactivated Koopman in v1 narrow-domain scope. §3.4 + §4 engineering analyses below remain authoritative on the substrate challenges (recipe registry currently never matches real workloads at `cipher_recipes.cpp:346`; `cipher_koopman_runtime.cpp` is dead code with zero callers at audit time; nvcc compilation rules absent from `cipher_rt_phase4/Makefile`). v1 scope addresses these in narrow-domain form only at Weeks 13-14 per §7 line 1258 (EDMD pipeline real-input fix + recipe registry narrow-domain seeding + SUBSTITUTE-Koopman lane validation). Wide-domain Koopman (dynamic operator discovery via `CIPHER_KOOPMAN_SURROGATE_ADD` ioctl NR 29 at line 549, runtime-derived operators) remains v2 research. Earlier prose in this plan flagged Koopman as v2-deferred (v1.2.0/v1.2.1 framing); those statements are superseded by ADJUDICATION A1 and reconciled to the narrow-domain-in-v1 framing in this revision. Closes `WEEK_12_SCOPE_DRIFT_AUDIT.md` item D9.

**v1 op surface: 30 of 33.** Deferred to v2 on hardware grounds (multi-GPU deployment not in v1 environment): NCCL_P2P (Op 33), OVERLAP (NCCL compute-comm overlap scheduler), STRAGGLER-cross-rank (Op 32 NCCL-side; local-detection half is in v1). See §8.4 for the formal deferral.

This document is the binding artifact for the next ~15-17 weeks of engineering AND the technical backbone of the investor data room. The 15-17 week sequence is detailed in §7. <!-- v1.2.3: 12-14 → 15-17 per B.1 timeline extension. -->


---

## Constraints on this plan itself

- **No assumed work.** Every claim grounded in inspected code or documented measurement.
- **No optimistic estimates.** Risks named honestly.
- **No compromises hidden.** Where any goal cannot be achieved without compromise in any regime, the compromise is surfaced explicitly with name and location.
- **No invention.** The plan integrates what exists. New work named explicitly where required.
- **No ambiguity in dispatch routing or op placement.** Every op has a defined home.

---

## Table of contents

<!-- v1.2: Section 0 added to TOC -->
- [Section 0 — Version history](#section-0)
- [Section 1 — Complete code inventory](#section-1)
- [Section 2 — The 33-op tier map with current state](#section-2)
- [Section 3 — Compatibility analysis](#section-3)
- [Section 4 — Unified architecture design](#section-4)
- [Section 5 — Workload impact verification](#section-5)
- [Section 6 — No-compromise constraint verification](#section-6)
- [Section 7 — Integration sequence (15-17 week)](#section-7)
- [Section 8 — Risk register and honest gaps](#section-8)
- [Appendix A — Anchor manifest at planning baseline](#appendix-a)
- [Appendix B — Cross-references](#appendix-b)

---

<a id="section-1"></a>
## SECTION 1 — COMPLETE CODE INVENTORY

[To be enriched with full per-file detail from background inspection agents. Skeleton below.]

### 1.1 — Tree A: cipher-may13-evidence

**Build products:** three DSOs from one source tree:
- `libcipher_hook.so` (LD_PRELOAD shim) — 7 sources: `cipher_intercept_cudart.cpp`, `cipher_persist.cpp`, `cipher_graph_inspect.cpp`, `cipher_kernel_table.cpp`, `cipher_flow_recorder.cpp`, `cipher_flow_patterns.cpp`, `cipher_flow_substitute.cpp`.
- `libcipher_rt.so.preroadmap` — all src/*.cpp + src/*.cu MINUS the 3 hook-only-or-standalone files (intercept_cudart, persist, nccl_tuner) MINUS src/cipher_dispatch.cpp, src/cipher_oracle.cpp (which are filtered out per Makefile L29), PLUS top-level `cipher_dispatch.cpp` + `cipher_oracle.cpp`.
- `libcipher_nccl_tuner.so` + symlink `libnccl-tuner-cipher.so` — standalone, single source `cipher_nccl_tuner.cpp`.

**Build state:** Makefile is intact; binaries `libcipher_hook.so`, `libcipher_rt.so`, `libcipher_rt.so.preroadmap` are on disk. NOT the deployed runtime — `nm -D` on production `libcipher_rt.so` (`c2c5d313`) finds zero of the 33 ops' symbols.

**Source tree summary (74 entries in src/, plus 2 top-level):**
- 67 `cipher_*.cpp` + 6 `cipher_*.cu` = 73 buildable sources
- 1 file is `.bak_1431` (cipher_fp8_compute.cpp.bak_1431) — dead code
- **2 files in src/ are SILENTLY EXCLUDED by Makefile L29:** `src/cipher_dispatch.cpp` and `src/cipher_oracle.cpp`. The live, canonical versions live at top-level `/home/ubuntu/cipher-may13-evidence/{cipher_dispatch.cpp,cipher_oracle.cpp}` and DIFFER from the src/ copies per `diff -q`.
- **5 source files are built into BOTH `libcipher_hook.so` AND `libcipher_rt.so`** (because the Makefile excludes only 3 hook-side files from RT_CPP_SRC, not all 7): `cipher_kernel_table.cpp`, `cipher_graph_inspect.cpp`, `cipher_flow_recorder.cpp`, `cipher_flow_patterns.cpp`, `cipher_flow_substitute.cpp`. **Dual-loaded DSOs cause LD_PRELOAD-position symbol resolution** — hook-side wins. **The unified runtime must resolve this duplication explicitly.**
- 23 headers in `include/` define inter-op ABI.
- Total LOC: ~29,387 lines across .cpp + .cu.
- Test coverage: 53 test files in tests/; every overlay op (13-31) has a 1:1 `test_<op>.py`; integration via `test_actuation_full.py`, `test_layer3.cpp`, `test_int4_mistral.py`, `test_fusion_correctness.py`, `test_fp8_correctness.py`.

**Top-level (canonical, live) source files:**

| File | LOC | Purpose | Tier | Key PASS_THROUGH points |
|---|---|---|---|---|
<!-- v1.2: line citations updated per Deep Inspection §B.1.7 / §D.1 — v1.1 cited the src/ shadow (616 LOC); the TOP-LEVEL live file (543 LOC) is the canonical port target and uses different line numbers. -->
| `cipher_dispatch.cpp` | 543 | Hot-path dispatcher: CLASSIFY → struct_lookup → ORACLE → registry → recipe → SUBSTITUTE. **Five PASS_THROUGH early-exits in the main path** all return `CIPHER_PASS_THROUGH` (top-level live file): line 448 (Layer 3 not initialized), 468 (ORACLE DENY), 480 (registry MISS — "L3.5 EDMD pipeline — wired in Week 4-5" — the load-bearing Goal-4-never-fires path), 486 (entry->error_bound > 0.01), 500 (substitute failed). Only one code path in the main spine reaches `return CIPHER_SUBSTITUTED` at line 515. Empty registry + no EDMD discovery → all kernels exit at L480. Additionally, the top-level file carries a separate **classify-only branch at L338-426** that fires when `!g_cipher.initialized`, with 3 additional PASS_THROUGH paths (L373/L425/L445) and 3 additional SUBSTITUTED paths (L388/L403/L418). Total: 8 PASS_THROUGH + 4 SUBSTITUTED returns. The src/ shadow at `cipher-may13-evidence/src/cipher_dispatch.cpp` is **silently excluded by Makefile L29** and is strictly older (616 LOC, missing the EDMD-live hook). Action: port the top-level file; drop the src/ shadow. | classifier+actuator spine |
| `cipher_oracle.cpp` | — | Safety gate. 5-gate logic per audit_section_1a (phase / min-confidence / structural-lookup / EMA-demotion / N≤4). Functions: `cipher_oracle_init` (L80), `cipher_oracle_update_gradients` (L265), `cipher_oracle_decide` (L293), `cipher_oracle_record_substitution` (L383), `cipher_oracle_set_phase` (L410), `cipher_oracle_report` (L422). | classifier |

**src/ tree — categorized inventory (load-bearing summary):**

*Core 12-op spine (Stage 0/1/2 implementations):*
- `cipher_10ops_impl.cpp` (1080 LOC) — Stage 1+2 background threads implementing REMEMBER/VALIDATE/AUDIT/SPECULATE/ADAPT/ARBITRATE. **Lazy-spawn-gated**: at line 949-965, threads spawn only if `observers_enabled > 0` (sum of 20 observer `_init()` returns); default-off observers → no threads → 6 core ops NEVER FIRE in production.
- `cipher_intercept.cpp` — F1 cuLaunchKernel/Ex intercept via cuGetProcAddress reroute; RING_WRITE (Op 7); ships into libcipher_rt (NOT hook DSO).
- `cipher_intercept_cudart.cpp` (3006 LOC) — the new dispatch spine (hook DSO): PLT exports + ELF GOT patching via dl_iterate_phdr; integrates persist/kernel_table/flow_*/attn_koopman.

*Classifier/dispatcher substrate:*
- `cipher_classify.hpp` (header at `include/cipher_classify.hpp`) — `cipher::OpClass` enum + `classify_launch()`.
- `cipher_structural_lookup.cpp` — L3.8 fast structural rule table; CLASSIFY backing.
<!-- v1.2: registry count corrected per Deep Inspection §D.3 — actually 32+ mixed entries, not exclusively Llama-3-70B-scale. -->
- `cipher_recipes.cpp` — recipe library + registry; **mixed 32+ entries seeded at init (L346)** spanning Llama-3-70B GEMMs, SOMA motor-control, Llama-3 attention shapes, RMSNorm/elementwise Chebyshev shapes, HyperFlux gaming physics surrogates, A100 (sm_80) versions of core shapes, and 10 shape-parametric entries with `hash_shape(0, K, N)` keys; `cipher_registry_init/lookup/insert` (L346/486/502). The 10 shape-parametric entries at the tail are a latent bug (Deep Inspection §D.3.NEW): keyed with M=0 while the lookup at `cipher_dispatch.cpp:153` calls `fnv_shape(m, n, k)` with the actual M ≥ 1 — these entries never match real workloads. Cosmetic; v2 cleanup.
- `cipher_kernel_table.cpp` — kernel-name recognition (FLASHATTN/GEMM/RMSNORM/…); CLASSIFY backing (DUAL-BUILT, hook+RT).

*SUBSTITUTE actuators (compute paths):*
- `cipher_substitute_v2.cpp` — NVRTC compile pool; up to 64 cubins.
- `cipher_weight_compress.cpp` + `cipher_marlin_src.cpp` (33.8 KB Apache-2.0 string literal from IST-DASLab) — Stage 7 W4A16 + Marlin INT4 GEMM kernel.
- `cipher_fusion_kernels.cpp` — NVRTC RMSNorm/SiLU·mul/residual_add fused kernels.
- `cipher_attn_koopman.cpp` + `cipher_attn_koopman_kernel.cu` — fused-attention FSM + CUDA kernel.
- `cipher_block_sub_kernel.cu` — Block-level O(Kr) + WMMA Koopman substitute kernel.
- `cipher_fp8_compute.cpp` + `cipher_fp8_fused_quant.cu` — Stage 13 FP8 quant via cooperative_groups single-launch.
- `cipher_kv_compress.cpp` + `cipher_kv_redirect.cpp` — Stage 8 KIVI 2-bit KV compression + redirect (Path A).
- `cipher_param_recovery.cpp` — Stage 1 fatbin/cubin EIATTR_KPARAM_INFO parser; CLASSIFY backing.

*Koopman/EDMD learner (Goal-4 substrate — dead code at audit-time, zero callers; narrow-domain reactivation at Weeks 13-14 per v1.2.2 ADJUDICATION A1 + §7 line 1258):*
- `cipher_lnn.cpp` — CfC LNN forward; REMEMBER/SPECULATE backing.
- `cipher_edmd.cpp` + `cipher_edmd_live.cpp` — EDMD pipeline; ADAPT backing. The cipher_koopman_runtime.cpp module has zero callers anywhere in the runtime.
- `cipher_koopman_runtime.cpp` — L1.1 Runtime Koopman derivation; dead code (audit Section 4).

*F-layer substrate (hardware/state/init):*
- `cipher_runtime.cpp` — cipher_init/teardown/report; wires F2→F3→F4→F5→Layer-3.
- `cipher_silicon.cpp` — hardware capability model (static + atomic dynamic fields).
- `cipher_hw_desc.cpp` — H100/H200 descriptor + cross-hardware Koopman transfer.
- `cipher_liquid_state.cu` — F4 shared liquid state; per-device EMA on grad-norm/HW-util/NCCL.
- `cipher_telemetry.cpp` — F5 NVML/CUPTI 500 Hz poll.
- `cipher_green_ctx.cu` — F2 Green Context allocator + SM band priorities.
- `cipher_l2_persist.cu` + `cipher_persist_engine.cpp` — F3 L2 persistence + Stage-3 admission engine.
- `cipher_vmm.cpp` — Stage-4 VMM pool (cuMemAddressReserve/Map).
- `cipher_graph.cpp` + `cipher_graph_inspect.cpp` — Stage-5 graph capture + audit (graph_inspect DUAL-BUILT).
- `cipher_partition_router.cpp` — Stage-10 partition router (ARBITRATE actuator).
- `cipher_thermal_feedback.cpp` — Stage-11 thermal-substitution feedback thread.
- `cipher_power_cap.cpp` — per-batch power-cap LUT (VOLT co-actuator).

*Overlay ops (one file each, 21):* `cipher_sense.cpp` (13), `cipher_shield.cpp` (14), `cipher_sustain.cpp` (15), `cipher_guard.cpp` (16), `cipher_predict.cpp` (17), `cipher_receipt.cpp` (18), `cipher_continuity.cpp` (19), `cipher_thermostat.cpp` (20), `cipher_determinism.cpp` (21), `cipher_pulse.cpp` (22), `cipher_carbon.cpp` (23), `cipher_fairness.cpp` (24) + `cipher_fairness_shm.cpp` (cross-process), `cipher_topology.cpp` (25), `cipher_loop.cpp` (26), `cipher_pipeline.cpp` (27), `cipher_trace.cpp` (28), `cipher_comply.cpp` (29), `cipher_volt.cpp` (30), `cipher_hibernate.cpp` (31), `cipher_straggler.cpp` (Phase-3).

*Layer-2 orchestrator + NCCL family:* `cipher_layer2.cpp`, `cipher_sm_packer.cpp`, `cipher_mem_layout.cpp`, `cipher_fusion.cpp`, `cipher_nccl.cpp`, `cipher_nccl_bpf.cpp`, `cipher_nccl_neural.cpp`, `cipher_nccl_v4.cpp`, `cipher_nccl_tuner.cpp` (standalone DSO).

*Flow/pattern hook substrate (DUAL-BUILT into hook AND RT):* `cipher_flow_recorder.cpp`, `cipher_flow_patterns.cpp`, `cipher_flow_substitute.cpp`.

*Persistent fast-path (hook DSO):* `cipher_persist.cpp`.

*Cross-tenant weight-share (Phase 2 IPC predecessor to Track 2 SC5):* `cipher_weight_share.cpp` — cuIpcGetMemHandle/OpenMemHandle + /dev/shm coordination. **Superseded by Track 2 SC5 weight-arena fd custodian in cipher_kmod.**

**Dead code in src/:**
1. `src/cipher_fp8_compute.cpp.bak_1431` — .bak suffix excluded by wildcard. Action: delete.
2. `src/cipher_dispatch.cpp` — silently excluded by Makefile L29; canonical version is top-level. Action: discard, use top-level.
3. `src/cipher_oracle.cpp` — same. Action: discard, use top-level.

### 1.2 — Tree B1: cipher_rt_phase4

**Build product:** `libcipher_rt.so` (anchor `83afd1ca` post-Track 3 SC3 / 2026-05-19; `c2c5d313` was the CP 2.5 anchor; `a7ac8e97` was the CP 5.4 Step 1.3 anchor). Plus separate `cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` (Python C-ext for vLLM KV) built via `build_kv_bridge.sh`.

**Init order in `cipher_inject.c` `cipher_v2_init_body` (load-bearing — every unification op port must respect this sequence):**

1. `cipher_v2_tenant_register` — opens /dev/cipher, ioctl nr 1.
2. `cipher_rt_green_ctx_cp54_init` — ioctl nr 13 (CP54_ALLOCATE); reads `CIPHER_QOS_CLASS`, `CIPHER_SM_COUNT`, `CIPHER_MIGRATABLE`. Optionally ioctl nr 16 (SUBSCRIBE_MIGRATE).
3. `cipher_rt_smp_init` — zero-init 32-slot streak cache.
4. `cipher_rt_pr_init` — zero-init 256-slot stream cache, refresh tenant snapshot.
5. `cipher_v2_cupti_init` — opens /dev/cipher (long-lived write-only fd), cuptiSubscribe + 8 callbacks (cudaLaunchKernel_v7000, cudaLaunchKernelExC_v11060, cuLaunchKernel, cuStreamCreate*, cudaStreamCreate*).
6. `cipher_rt_volt_init` — env-gated; resolves NVML or probes ioctl nr 10; locks clock if requested; installs atexit + signal handlers (SIGTERM/INT/SEGV/ABRT/BUS).
7. `cipher_rt_matmul_dispatch_init` — empty substrate init.
8. `cipher_rt_marlin_init` — env-gated by `CIPHER_MARLIN`; NVRTC compile, registers MARLIN_INT4 actuator priority 10 on matmul substrate.
9. `cipher_rt_attn_dispatch_init` — empty substrate init.
10. `cipher_rt_attn_test_actuator_init` — env-gated by `CIPHER_ATTN_TEST`; registers test actuator priority 0.
11. `cipher_rt_audit_init` — env-gated by `CIPHER_AUDIT`; registers priority-0 observer actuator on BOTH matmul and attn substrates.
12. `cipher_rt_cublas_shim_register_got` — registers `cublasGemmEx` → `cipher_rt_cublasGemmEx_impl`.
13. `cipher_rt_attn_register_got` — registers 3 SDPA mangled names → trampolines.
14. `cipher_rt_got_patch_init` — walks `dl_iterate_phdr`, patches every loaded module's GOT slot via mprotect dance (RELRO-safe).

Returns 1 to CUDA driver. cuInit proceeds.

**Active source files (Makefile `OBJS`, 16 entries):**

| Source | Role | Hot path? |
|---|---|---|
| `cipher_inject.c` | CUDA_INJECTION64_PATH entry point. Defines `InitializeInjection` / `InitializeInjection2`. Calls `cipher_rt_green_ctx_cp54_init()` at cuInit (CP 5.4 Step 1.3). | setup (cuInit) |
| `cipher_tenant.c` | Tenant identity registration (CIPHER_REGISTER_TENANT, ioctl nr 1). | setup |
| `cipher_cupti.c` | CUPTI launch-counter daemon thread (nr 7 SUBMIT_LAUNCH_STATS). | background |
| `cipher_rt_tenant.{cpp,h}` | Userspace snapshot client (ioctl nr 8 CIPHER_GET_TENANT_SNAPSHOT, three-access-mode API). | varies |
| `cipher_rt_partition_router.{c,h}` | Per-tenant CUDA stream creation with priority + mem_sync_domain (Depth Win #1). | per-stream setup |
| `cipher_rt_sm_packer.{c,h}` | SM-pack hint computation. | setup |
| `cipher_rt_green_ctx.{c,h}` | CP 5.4 green-context allocation client (CIPHER_CP54_ALLOCATE nr 13). | cuInit |
| `cipher_rt_volt.{c,h}` | DVFS actuator. Calls NVML first, falls through to kmod nr 10 CIPHER_SET_CLOCK_MHZ. | setup + periodic |
| `cipher_rt_matmul_dispatch.{c,h}` | Per-kernel matmul classifier + dispatch (Marlin INT4 vs cuBLAS shim vs passthrough). | hot path |
| `cipher_rt_cublas_shim.c` | GOT-patched cuBLAS interception. | hot path |
| `cipher_rt_got_patch.{c,h}` | GOT/PLT patcher (CP 2.5 — replaced LD_PRELOAD link-order with rewrite at injection time). | setup |
| `cipher_rt_marlin_kernel_src.cpp` + `cipher_rt_marlin_engine.cpp` + `cipher_rt_marlin_actuator.c` + `cipher_rt_marlin_perms.h` + `cipher_rt_marlin.h` | Marlin INT4 kernel + engine + actuator. **Structurally full-GPU (grid=132 SMs, persistent split-K).** Does NOT compose with green-context partitioning per [PHASE_5_ARCHITECTURE_REVISION.md](./PHASE_5_ARCHITECTURE_REVISION.md). Pinned to primary context via Fix A (`PrimaryCtxGuard`). | hot path (B≥8) |
| `cipher_rt_attn_dispatch.{cpp,h}` | ATen SDPA trampoline (T4.6.1, GOT-patched). Three trampolines (math/efficient/flash). REDIRECT path returns PASSTHROUGH (`cipher_rt_attn_dispatch.cpp:158-164`). | hot path |
| `cipher_rt_attn_test_actuator.c` | Test actuator for the attention substrate. | test-only |
| `cipher_rt_audit.{c,h}` | HMAC-SHA256 audit chain (Fusion Op 9 AUDIT — links libcrypto). | per-launch |

**Retired source (in-tree, unbuilt):**

- `cipher_rt_arbitrate.{c,h}` — RETIRED at CP 5.4 Step 1.3 (2026-05-18). Ioctl nr 9 (legacy CIPHER_REQUEST_SM_PARTITION) now returns `-ENOSYS`. Allocation moved to ledger nrs 13-15. File preserved for historical reference.

**Python bridges (not part of libcipher_rt build, but part of the deployed product):**

- `cipher_kv_bridge.cpp` (anchor `c04b0c39` post-Track 2 SC3) — Python C-ext for vLLM KV integration.
- `cipher_kv_cache.py`, `cipher_spec_decode.py`, `pillar_driver.py`, `tf_gate_driver.py` — Python harnesses.

### 1.3 — Tree B2: cipher_kmod

**Build product:** `cipher_kmod.ko` (anchor `008b3c66` post-Track 2 SC5; was `8d777dfb` after CP 5.4 Step 1.3 Phase A, `e2f50452` at CP 2.5).<!-- v1.2.1: anchor corrected per PRE_WEEK_1_ADJUDICATION_CLOSURE.md Part 3 C2; see footnote at Appendix A. -->

**Kbuild source list (14 entries):**

| Source | Role |
|---|---|
| `cipher_main.c` | Module init/exit, srcversion, taint setup. |
| `cipher_probe.c` | kprobe on `nvidia_unlocked_ioctl` — observes all CUDA driver ioctls per-process. |
| `cipher_ioctl_decode.c` | The dispatch table for `/dev/cipher` ioctls (NRs 1-24). |
| `cipher_proc.c` | `/proc/cipher/{stats, bar0_state, gpu_state, migrations, ...}` nodes. |
| `cipher_dev.c` | chardev `/dev/cipher` (mode 0666 via devnode callback, kmod 0.4.8+). |
| `cipher_bar0.c` | BAR0 mapping for hardware register reads. Note: PMC_BOOT_1=0x00000000 is expected bare-metal on H100 ([[cipher-pmc-boot-1-bare-metal]]). |
| `cipher_tenant_snapshot.c` | `cipher_get_current_tenant_snapshot()` / `_by_id()` / `_enumerate_tenants()` — the sub-200ns RCU read API for in-kernel consumers. |
| `cipher_state_updater.c` | 1 kHz kthread that mirrors per-tenant derived fields from `cipher_pid_stats` into snapshot view. |
| `cipher_partition_allocator.c` | Legacy nr 9 allocator. Returns `-ENOSYS` post-CP-5.4. |
| `cipher_cp54_sched.c` | CP 5.4 SM-arbitration ledger (nrs 13-20). 15 × 8-SM groups (Step 1.3a measured). Two-clause disjointness invariant. Track 3 migration state machine. |
| `cipher_weight_arena.c` | Track 2 SC5 fd custodian for VMM POSIX fd cross-process weight sharing (nrs 21-24). |
| `cipher_clock.c` | DVFS via `call_usermodehelper("nvidia-smi -lgc")` (nr 10). |
| `cipher_kvdedup.{c,h}` | KV-dedup L3 cross-process via cuIpc + content hash (T4.6.4). |
| `cipher_flops.c` | CP 3.3 FLOP telemetry — accepts CUPTI PM Sampling submissions (nr 11), serves per-tenant queries (nr 12). |

**Kmod ABI — public ioctls (cipher_ioctl.h):**

NRs 1-24 wired. NRs 2/3/4 reserved (`-ENOSYS`). Strict additive-only rule per [[cipher-abi-rule]]:

| Nr | Name | Purpose | Verified by |
|---|---|---|---|
| 1 | REGISTER_TENANT | Tenant identity (Phase 2 baseline) | Phase 2 |
| 5 | SUBMIT_GPU_STATE | NVML device-state push from `cipher-gpustate` daemon | Phase 3 |
| 6 | SUBMIT_PROCESS_UTIL | Per-process GPM util | Phase 3 |
| 7 | SUBMIT_LAUNCH_STATS | CUPTI launch counters from workload | Phase 3 |
| 8 | GET_TENANT_SNAPSHOT | Cross-process per-tenant query (sub-200ns target) | Phase 4 T4.1.7 |
| 9 | REQUEST_SM_PARTITION | **DEACTIVATED** (`-ENOSYS`) — replaced by nrs 13-15 | CP 5.4 Step 1.3 |
| 10 | SET_CLOCK_MHZ | DVFS via privileged usermode helper | T4.3.2 |
| 11 | SUBMIT_FLOP_SAMPLE | CUPTI PM Sampling daemon push | CP 3.3 |
| 12 | QUERY_FLOPS | Device + per-tenant FLOP/MFU query | CP 3.3 |
| 13 | CP54_ALLOCATE | SM partition allocation (15 × 8-SM ledger) | CP 5.4 |
| 14 | CP54_FREE | Release partition | CP 5.4 |
| 15 | CP54_QUERY | Ledger state read | CP 5.4 |
| 16 | CP54_SUBSCRIBE_MIGRATE | Opt-in to live migration | Track 3 SC2 |
| 17 | CP54_POLL_MIGRATE | Read migration state | Track 3 SC2 |
| 18 | CP54_START_MIGRATE | PROPOSED → MIGRATING | Track 3 SC2 |
| 19 | CP54_ACK_MIGRATE | COMMIT or NACK | Track 3 SC2 |
| 20 | CP54_COMPACT_MIGRATE | Force compaction pass | Track 3 SC2 |
| 21 | ARENA_REGISTER | Weight arena fd registration | Track 2 SC5 |
| 22 | ARENA_IMPORT | Consumer fd import | Track 2 SC5 |
| 23 | ARENA_LEAVE | Consumer leaves arena | Track 2 SC5 |
| 24 | ARENA_QUERY | Ledger query | Track 2 SC5 |

### 1.4 — Tree C: orphan and auxiliary trees

- `cipher-phase1-evidence/` — frozen snapshot of Phase 1 kmod (cipher_main/dev/probe/proc/ioctl_decode/internal.h). Identical srcversion-mapped sources are now in `cipher_kmod/`. **Action: archive only, not part of v1.**
- `cipher_vllm_plugin/` — Python package (`cipher_vllm_kv.py`, `cipher_kv_offload.py`, `setup.py`). Bridges CIPHER's KV-dedup to vLLM as an offload backend. **Action: this is the load-bearing v1 bridge — must remain part of the deployed package.**
- `cipher_kv_bridge_src_track2_sc3.tar.gz` — Track 2 SC3 kv_bridge source archive (12877 bytes). **Action: source-of-truth for cipher_kv_bridge.cpp.**

### 1.5 — Compilation status, current state

- **may13 `libcipher_rt.so.preroadmap`** — built historically; not regenerated since the may13 snapshot. Not deployed.
- **`cipher_rt_phase4/libcipher_rt.so`** — actively built, anchor `83afd1ca` (Track 3 SC3). Production runtime.
- **`cipher_kmod/cipher_kmod.ko`** — actively built, anchor `008b3c66`. DKMS-installed to `/lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko` (DKMS md5 `6654d9e5`, stale per WEEK_1_PRE_FLIGHT.md §1.4: size 122,987 vs in-tree 2,631,632; mtime 2026-05-16 predates Track 2 SC5; DKMS re-install deferred as deployment hardening with no Week 1 impact).<!-- v1.2.1: anchor corrected per PRE_WEEK_1_ADJUDICATION_CLOSURE.md Part 3 C2. -->
- **`/home/ubuntu/libcipher_v2/libcipher_v2.so.v0.2.0`** anchor `86618c30` — Phase 3 substrate, still loaded. Companion to libcipher_rt.<!-- v1.2.1: full path confirmed per PRE_WEEK_1_ADJUDICATION_CLOSURE.md Part 3 C1. Earlier task briefs referenced `/home/ubuntu/cipher_rt_phase4/libcipher_v2.so` as the canonical path; that path does not exist on disk. The Phase 3 substrate has always lived at `/home/ubuntu/libcipher_v2/`. A second binary `/home/ubuntu/libcipher_v2/libcipher_v2.so` md5 `cc0479b8...` (Track 2 close anchor) exists at the same directory. -->

### 1.6 — Test coverage map

| Area | Test source | Pass status |
|---|---|---|
| Phase 4 contention (per-thread fd) | `cipher_phase4_tests/cipher_test_phase4_partition_contention.c` | PASS — 1.4× p99 ratio per-thread vs 44.2× shared |
| Phase 4 happy-path | `cipher_test_happy` (17/17) | PASS through CP 5.6 |
| CP 5.4 isolation | `cp_5_4/step1_6/` runtime probes | 15/15 PASS, two-clause disjointness verified |
| Track 2 SC6 | bit-identical forward, 40 forwards, two models | PASS |
| Track 3 SC2 | 29/29 state machine assertions | PASS |
| Track 3 SC5/SC6 | A/B sweep (10 scenarios), within-A 17 migrations | PASS |
| Per-op (may13) | `stress2/per_op_validation.json` | 8/11 canonical-overlay FIRE, 2 SILENT-by-design, 6 NEVER (Stage 1/2 threads not spawned) |
| Goal-4 Koopman substitutions | All stress/stress2 logs | **0 substitutions ever fire** (zero matches in `[L3.2] SUBSTITUTE`, `[O(1)-block]`, etc.) |
| 24-hour soak (G4) | `cipher_measurement/soak_24h.sh` | **NEVER RUN** |
| MFU 85% gate (G1) | per-WL mfu_per_tenant.sh | **NEVER RUN at full WL01-24 coverage** |

---

<a id="section-2"></a>
## SECTION 2 — THE 33-OP TIER MAP WITH CURRENT STATE

Source for tiers: PHASE_4_OP_WORKLOAD_MATRIX.md groups the 33 ops by Stage (0/1/2 + overlay). For the unification plan we re-tier them by their architectural role in the unified control plane:

- **Classifier** — decides workload regime, drives dispatch. Must run BEFORE actuator selection.
- **Actuator** — changes execution (substitution, partition, frequency, etc.). Driven by classifier output.
- **Observability** — passive measurement / accounting / audit. Cross-cutting.
- **Learning** — speculative / adaptive / model-update. Off hot path; deferred to v2.

### 2.1 — Core 12 (Stage 0/1/2)

| # | Op | Tier | may13 location | rt_phase4 location | State | Hot-path | Regime |
|---|---|---|---|---|---|---|---|
| 1 | CLASSIFY | classifier | `cipher_classify.hpp:94-224` (called from `cipher_dispatch.cpp:406-411,523-530`) | **NOT PORTED** | WORKING in may13 | every kernel launch | universal |
| 2 | ORACLE | classifier | `cipher_oracle.cpp:293-377` (called from `cipher_dispatch.cpp:496,558`) | **NOT PORTED** | WORKING in may13 | every kernel launch | universal |
| 3 | SUBSTITUTE | actuator | `cipher_dispatch.cpp:200-395` + `cipher_recipes.cpp:346` (32 static shapes) + `cipher_block_sub_kernel.cu` + `cipher_attn_koopman_kernel.cu` | Partial: `cipher_rt_matmul_dispatch.c` + Marlin + cuBLAS shim + attn_dispatch (separate compute paths, no Koopman) | PARTIAL — Marlin lane works for B≥8 compute; Koopman lane NEVER FIRES (audit Section 4) | hot path | compute-bound |
| 4 | COMMIT / ORCHESTRATE | actuator | dispatch return path (`cipher_dispatch.cpp:587-589`) | implicit in matmul_dispatch (v1.2.1 state); **v1.2.2 promotes to atomic state-transition primitive — see §4.8** | v1.2.2: REQUIRES-FIX (build the atomic primitive in Weeks 7-8) | every launch | universal |
| 5 | SAMPLE / GENERATE | observability | `cipher_intercept_cudart.cpp:2353-2425`. Hard-stops after 500 launches (`:2409-2411`). | not present | PARTIAL — warmup-only | per-launch | universal |
| 6 | RING_WRITE | observability | `cipher_10ops.h:83-100`, call sites `cipher_intercept.cpp:180-181,374-375` | not present (v1.2.1 state); **v1.2.2 promotes to lock-free inline telemetry substrate distinct from CUPTI — see §4.9** | v1.2.2: REQUIRES-FIX (build the substrate in Weeks 9-10) | per-launch | universal |
| 7 | REMEMBER | learning | `cipher_10ops_impl.cpp:428-458` (CfC LNN forward) | not present | WORKING code, NEVER FIRES (Stage 1 thread not spawned) | shadow-thread | universal |
| 8 | VALIDATE | observability | `cipher_10ops_impl.cpp:460-475` | not present | PARTIAL — Welford stats real; `rs_ok` 3-σ detector never called | shadow-thread | universal |
| 9 | AUDIT | observability | `cipher_10ops_impl.cpp:341-378`, live path `:472-474` | `cipher_rt_audit.{c,h}` (HMAC-SHA256 chain — links libcrypto) | PARTIAL (may13: HMAC call commented out `:472-473`); WORKING (rt_phase4) | shadow-thread / per-launch | universal |
| 10 | SPECULATE | learning | write `cipher_10ops_impl.cpp:477-505`; check `cipher_intercept.cpp:127-149` | not present | WORKING code, NEVER FIRES | shadow-thread | universal |
| 11 | ADAPT | learning | `cipher_10ops_impl.cpp:597-765`, KEN `:99-285` | not present | PARTIAL — EDMD solver real but fed degenerate `h_before==h_after` synthetic input (`:693-701, :663-669`); NEVER FIRES | background-thread | universal |
| 12 | ARBITRATE | actuator | `cipher_10ops_impl.cpp:767-814` (writes SHM demand int; SM rebalance `TODO`) | **NEW: kmod-resident** `cipher_cp54_sched.c` (real partition allocation, 15-group ledger, two-clause disjointness, do_exit reaper, Track 3 migration state machine) | UPGRADED — real partition arbitration ships in kmod (not as the may13 op); the may13 op is superseded. | every partition-change | multi-tenant |

<!-- v1.2: PASS_THROUGH line citations corrected per Deep Inspection §B.1.7 / §D.1 — top-level file is 543 LOC and uses these line numbers; src/ shadow at 616 LOC uses different numbers and is silently excluded. -->
**Stage-0 dispatch reality check (verified at planning time, top-level live file).** The may13 hot-path dispatcher `cipher-may13-evidence/cipher_dispatch.cpp` (top-level, 543 LOC) has **5 distinct PASS_THROUGH early-exits in its main path** verified by direct inspection:
- **L448** — `if (!g_layer3_initialized) return CIPHER_PASS_THROUGH` (Layer 3 not initialized).
- **L468** — `if (oracle.decision == CIPHER_ORACLE_DENY) return CIPHER_PASS_THROUGH`.
- **L480** — Registry MISS, the documented "L3.5 EDMD pipeline — wired in Week 4-5" deferral. THE LOAD-BEARING PATH: empty registry → every kernel exits here.
- **L486** — `if (entry->error_bound > 0.01f) return CIPHER_PASS_THROUGH`.
- **L500** — `if (!substituted) return CIPHER_PASS_THROUGH`.

Only **one** code path in the main spine reaches `return CIPHER_SUBSTITUTED` at line 515. Additionally, the file carries a **classify-only branch at L338-426** (Deep Inspection §D.1.NEW) that fires when `!g_cipher.initialized` and contains 3 additional PASS_THROUGH paths (L373/L425/L445) plus 3 additional SUBSTITUTED paths (L388/L403/L418) — when ported in Week 1, this entire second dispatch path must come across with the main spine.

The registry is seeded with mixed 32+ entries (`cipher_recipes.cpp:346`) spanning gemm / Chebyshev / HyperFlux / A100 / shape-parametric categories; no real workload's M/N/K hash matches them; therefore `apply_recipe()` and the Koopman substitution branch never ran at the audit-time substrate. **This is the central engineering reality the unified runtime must address.** v1.2.2 ADJUDICATION A1 (re-stated in §1 line 103) places this in Weeks 13-14 v1 narrow-domain scope: replace the 32 mismatching seed entries with narrow-domain recipes that DO match real workloads (LM head and small-attention shape class), then validate the SUBSTITUTE-Koopman lane fires on at least one matching workload (per `WEEK_13_14_SCOPE_LOCK.md` Step 2). Wide-domain dynamic operator discovery remains v2 research.

### 2.2 — Overlay 21

| # | Op | Tier | may13 file | rt_phase4 equivalent | State | Regime |
|---|---|---|---|---|---|---|
| 13 | SENSE | classifier | `cipher_sense.cpp:171-273` | not present | WORKING in may13 | universal (session classifier) |
| 14 | SHIELD | actuator | `cipher_shield.cpp:90-146` | not present | PARTIAL — ITL/jitter detector real; Protections 2/3 flags never consumed | latency-bound |
| 15 | SUSTAIN | actuator | `cipher_sustain.cpp:90-105,177-183` | not present (but kmod kv_alloc covers similar ground) | PARTIAL — KV-pressure slope real; `sustain_compress` flag has no consumer | memory-bound + multi-tenant |
| 16 | GUARD | observability | `cipher_guard.cpp:99-130` | not present | WORKING — cross-session leak detector | multi-tenant |
| 17 | PREDICT | actuator | `cipher_predict.cpp:160-245` | not present | WORKING — hot-pointer promotion into persist engine | memory-bound |
| 18 | RECEIPT | observability | `cipher_receipt.cpp:123-225` | not present | WORKING — HMAC-SHA256 proof-of-compute | universal (billing) |
| 19 | CONTINUITY | observability | `cipher_continuity.cpp:130-175` | not present | PARTIAL — region tracking only; no manifest writer | multi-tenant (checkpoint) |
| 20 | THERMOSTAT | actuator | `cipher_thermostat.cpp:166-236` | not present (but VOLT consumes via env coupling) | WORKING | compute-bound + DVFS |
| 21 | DETERMINISM | observability | `cipher_determinism.cpp:50-63` | not present | WORKING — FIRES (dispatch_count=36450 in stress2) | universal |
| 22 | PULSE | observability | `cipher_pulse.cpp:177-358` | not present | PARTIAL — Signal 2 deferred, severity ceiling capped at 2 | universal (hw fault) |
| 23 | CARBON | observability | `cipher_carbon.cpp:84-169` | not present | WORKING — FIRES | universal (sustainability) |
| 24 | FAIRNESS | observability+actuator | `cipher_fairness.cpp:88-117`, `cipher_fairness_shm.cpp` | not present | WORKING — per-tenant quota | multi-tenant |
| 25 | TOPOLOGY | classifier | `cipher_topology.cpp:26-55` | not present | WORKING (static peer-adj; degenerate on single H100) | multi-GPU |
| 26 | LOOP | observability | `cipher_loop.cpp:148-229` | not present | WORKING — FIRES (runaway_count=7) | agentic |
| 27 | PIPELINE | observability | `cipher_pipeline.cpp:117-239` | not present | WORKING — FIRES (edge_count=10) | agentic |
| 28 | TRACE | observability | `cipher_trace.cpp:50-116` | not present | WORKING — FIRES (written=8192) | universal |
| 29 | COMPLY | observability | `cipher_comply.cpp:46-82` | not present | WORKING — fires-as-designed | universal (compliance) |
| 30 | VOLT | actuator | `cipher_volt.cpp:307-366` (NVML SetGpuLockedClocks; NOT_SUPPORTED on this pod) | `cipher_rt_volt.{c,h}` + kmod nr 10 SET_CLOCK_MHZ + cipher_clock.c | WORKING in rt_phase4 — NVML path falls through to kmod ioctl; +13.9% measured under vLLM at 1200 MHz | compute-bound, decode-with-headroom |
| 31 | HIBERNATE | actuator | `cipher_hibernate.cpp:139-170` | not present | PARTIAL — power-limit actuator gated off (NVML NOT_SUPPORTED) | bursty workloads |
| 32 | STRAGGLER | observability | `cipher_straggler.cpp:178-261` | not present | PARTIAL — local detection real; cross-rank attribution not implemented | multi-GPU |
| 33 | NCCL_P2P | actuator | `cipher_nccl.cpp:34-291` + `cipher_nccl_bpf.cpp` + `cipher_nccl_neural.cpp` + `cipher_nccl_v4.cpp` + `cipher_nccl_tuner.cpp` | not present (Class D deferred per PHASE_4_CUTS.md) | PARTIAL — algo-selection real; eBPF path simulated | multi-GPU (deferred) |

### 2.3 — Tier summary

| Tier | Count | Working | Partial | Not firing | Not ported (to rt_phase4) |
|---|---|---|---|---|---|
| **Classifier** | 5 (CLASSIFY, ORACLE, SENSE, PREDICT, TOPOLOGY) | 5 | 0 | 0 | 5 (all in may13 only) |
| **Actuator** | 11 (SUBSTITUTE, COMMIT, ARBITRATE, SHIELD, SUSTAIN, THERMOSTAT, VOLT, HIBERNATE, NCCL_P2P, FAIRNESS-quota, GENERATE) | 4 | 5 | 2 | 8 (VOLT & ARBITRATE in rt_phase4/kmod; rest stranded) |
| **Observability** | 14 (SAMPLE, RING_WRITE, VALIDATE, AUDIT, GUARD, RECEIPT, CONTINUITY, DETERMINISM, PULSE, CARBON, FAIRNESS-counters, LOOP, PIPELINE, TRACE, COMPLY, STRAGGLER) | 10 | 4 | 0 | 13 (AUDIT in rt_phase4; rest stranded) |
| **Learning** | 3 (REMEMBER, SPECULATE, ADAPT) | 0 | 3 (Stage 1/2 threads never spawned) | 3 | 3 (all in may13 only) |
| **Total** | 33 | 19 | 14 | 5 | 29 of 33 NOT in deployed runtime |

**The unification verdict (verified by Tree B inspection 2026-05-20).** Of 33 canonical ops, exactly **6 fire on Tree B's hot path today**:
1. **SUBSTITUTE** — via the single registered MARLIN_INT4 actuator on `cipher_rt_matmul_dispatch.c`; Koopman/EDMD lane unwired (no callers of `cipher_rt_substitute_v2` equivalent).
2. **ARBITRATE** — via kmod `cipher_cp54_sched.c` (15×8-SM ledger, ioctl nrs 13-15) + `cipher_partition_allocator.c` (legacy 4-SM-slot still gets called for do_exit reaper).
3. **VOLT** — via `cipher_rt_volt.c` + kmod `cipher_clock.c` (ioctl nr 10).
4. **AUDIT** — via `cipher_rt_audit.c` (HMAC-SHA256 chain, env-gated).
<!-- v1.2: COMMIT scope corrected per Deep Inspection §B.1 — only the cp54_sched Track 3 DSM "COMMIT" state-machine token is real; weight-arena's "commit" is a code-comment string at L262, not a state-machine token. -->
5. **COMMIT** — as a state-machine TOKEN in kmod **scoped to `cipher_cp54_sched.c` only** (Track 3 DSM ACK-commit at `cipher_cp54_sched.c:383`, `CIPHER_CP54_MIGOUT_COMMITTED` at L395). The earlier v1.0 framing included weight-arena commit-on-publish in this row; that was an over-claim — "commit" in `cipher_weight_arena.c:262` is a code-comment string, not a state-machine token. Drop the weight-arena half.
6. **CLASSIFY** — referenced as comment-only ("classify the stream this launch is on" at `cipher_cupti.c:166`); the substantive classifier logic is NOT ported.

The other **27 ops are NOT-PORTED to the deployed runtime** at audit-time. The brain (CLASSIFY/ORACLE/SENSE/PREDICT/DETERMINISM/TOPOLOGY — the 6 classifiers) is fully stranded in may13. The 11 observability ops that work in may13 (CARBON, FAIRNESS, RECEIPT, GUARD, COMPLY, LOOP, PIPELINE, TRACE, CONTINUITY, PULSE, STRAGGLER) have zero presence in the runtime. Goal-4's Koopman runtime module is dead code with zero callers in either tree at audit-time; narrow-domain reactivation at Weeks 13-14 per v1.2.2 ADJUDICATION A1 + §1 line 103.

**The marvel is an integration problem.** The pieces work in their natal tree. They were never connected to the actuator scaffolding that ships.

**The reusable abstraction that makes this tractable:** Tree B's `cipher_rt_matmul_dispatch.c:52` `cipher_rt_matmul_register_actuator` (priority-ordered actuator registry, 16-slot, returns HANDLED/PASSTHROUGH/REDIRECTED). The pattern repeats in `cipher_rt_attn_dispatch.cpp` for SDPA. The unified runtime adds a **CLASSIFY substrate** with the same shape: priority-ordered classifier registry that takes a kernel descriptor and returns a kernel_class. Every may13 classifier op ports as a classifier-registry entry; every observer ports as a priority-0 substrate actuator (the AUDIT template).

### 2.4 — The spine

The unified runtime cannot ship without these:

| Op | Why spine |
|---|---|
| CLASSIFY | The dispatcher input. No dispatch without it. |
| ORACLE | The safety gate. No dispatch without it. |
| SENSE | The session-band classifier (HUMAN/AGENT/BATCH). Drives SHIELD, FAIRNESS, LOOP. |
| SUBSTITUTE (Marlin lane) | The compute actuator. Already in rt_phase4. |
| ARBITRATE (kmod ledger) | The partition actuator. Already in kmod. |
| VOLT | The TPW actuator. Already in rt_phase4. |
| AUDIT | The integrity chain. Already in rt_phase4. |
| TRACE + RECEIPT | The observability surfaces customers pay for. |

Everything else is enhancement of the spine.

[Background agent will refine this with byte-level symbol cross-references.]

---

<a id="section-3"></a>
## SECTION 3 — COMPATIBILITY ANALYSIS

For each op-port from Tree A into Tree B's build, verified compatibility along six axes.

### 3.1 — Header compatibility

The may13 tree uses a single-file `include/cipher_10ops.h`, classify in `cipher_classify.hpp`, oracle in `cipher_oracle.cpp` (declarations inline). The rt_phase4 tree uses `cipher_v2_internal.h` + per-module headers (`cipher_rt_tenant.h`, `cipher_rt_volt.h`, etc).

**Mismatches found:**

| Conflict | Location | Class | Resolution |
|---|---|---|---|
| `struct cipher_tenant_snapshot` — userspace homonym | `cipher_rt_tenant.h` declares `cipher_tenant_snapshot_user`, kernel `cipher_ioctl.h` declares `cipher_tenant_snapshot` (different layouts) | requires-shim | Already handled: rt_phase4 currently uses `cipher_tenant_snapshot_user` to avoid the conflict ([cipher_rt_arbitrate.c historic comment](file:///home/ubuntu/cipher_rt_phase4/cipher_rt_arbitrate.c)). When porting may13 ops that touch the snapshot, use the userspace mirror struct. |
| `cipher_classify_t` enum | may13 has 7 classes; rt_phase4 has no equivalent | requires-port | Port may13's `cipher_classify.hpp` into rt_phase4 as `cipher_rt_classify.h`. |
| HMAC chain types | may13 `cipher_10ops_impl.cpp` AUDIT internal struct vs rt_phase4 `cipher_rt_audit.h` standalone | requires-shim | Pick one — rt_phase4's AUDIT is leaner and already shipped. Retire may13's AUDIT internals. |
| 32-shape recipe registry | may13 `cipher_recipes.cpp:346` static array | minor | Carry as-is into rt_phase4; the registry-population gap is its own work item (Section 4). |

### 3.2 — Symbol compatibility

`nm -D` cross-check (audit 2026-05-16, §1.5):

- `nm -D cipher_rt_phase4/libcipher_rt.so | grep cipher_(sense|shield|...)_ → 0 matches` — no symbol collisions because Tree A symbols don't exist in Tree B at all.
- `nm -D libcipher_rt.so.preroadmap | grep -c overlay-ops → 30` — Tree A's old build had them.

**Collision-free.** Port via additive new objects added to the rt_phase4 `OBJS`. Single new symbol-naming convention: prefix may13 op names with `cipher_rt_` (e.g. `cipher_sense_classify` → `cipher_rt_sense_classify`) to mark them as the production-tree variant.

### 3.3 — Build-system compatibility

**Tree A build:** GNU Make + nvcc, builds 3 DSOs. CUDA SDK at `/usr/local/cuda`, includes `include/`.

**Tree B build:** GNU Make, gcc/g++ + libcupti + libcuda + libpthread + libc10 (NOT libtorch_cpu — severed at CP 2.5 D2(iii)) + libcrypto. ATen headers via Python torch path.

**Path to merge:** rt_phase4's Makefile is the keeper (it's simpler, version-controlled by anchor). For each may13 op being ported:

1. Move source into `cipher_rt_phase4/` (renamed `cipher_rt_<op>.{c,cpp}`).
2. Add to `OBJS` in `cipher_rt_phase4/Makefile`.
3. Add per-source build rule (each source has its own rule in current Makefile).
4. Re-link `libcipher_rt.so`.
5. New anchor md5 recorded.

The may13 CUDA sources (.cu files) introduce nvcc to the rt_phase4 build — currently rt_phase4 is plain gcc/g++ only. **Adding nvcc is required if Koopman O(1) substitute kernels (`cipher_block_sub_kernel.cu`, `cipher_attn_koopman_kernel.cu`) are ported.** Per v1.2.2 ADJUDICATION A1 + §1 line 103, the narrow-domain Koopman tier lands at Weeks 13-14 (§7 line 1258); the W13-14 scope-lock (`WEEK_13_14_SCOPE_LOCK.md` Step 2 + risk R-W13.2) plans nvcc rule addition if the narrow-domain seeding needs `cipher_block_sub_kernel.cu`. A host-only EDMD fallback is documented in the scope-lock as the rollback path if .cu port is too disruptive in the W13-14 eng-day budget.

### 3.4 — Runtime compatibility

**Threading model.**

- Tree A: Three lazy-start threads (Stage 1 shadow, Stage 2 background, dispatch). Spawned only when observer env vars set. In practice never spawn in production runs (`per_op_validation.log:20` — "Stage 1/2 threads skipped").
- Tree B: One CUPTI thread (`cipher_cupti.c` daemon), one arbitrate poll thread (retired with arbitrate.c at CP 5.4 Step 1.3), the GPU-state daemon process (cipher-gpustate, separate process).

**Conflict:** none. Tree A's threads simply have never spawned in deployment. Porting CLASSIFY (the only Stage 0 op needed for the hot path) does not require spawning any thread — it runs synchronously per launch.

**State requirements.**

- CLASSIFY needs `g_classify_cache` (512-slot lock-free hash). Standalone — no kmod calls.
- ORACLE needs `g_recipe_registry` (32 static entries from `cipher_recipes.cpp`). Standalone.
- SENSE needs per-tenant session state — fits cleanly under `cipher_rt_tenant` (snapshot-backed).
- SHIELD/SUSTAIN/PREDICT need per-tenant state — same.
- AUDIT in rt_phase4 already has its own state — port leaves it alone.

### 3.5 — Kmod ABI compatibility

The kmod ABI (NRs 1-24) is **stable** and additive-only ([[cipher-abi-rule]]). The port does not require new kmod ioctls in v1.

<!-- v1.2: ABI extension table expanded per Deep Inspection §C.3 — specific NR 25-29 proposals with v1/v1.5/v2 scoping. Verified against Wave 3 ABI table (Section 1.3): NRs 1-24 are live or reserved, 25-29 are free. -->

**Future ABI extensions — specific NR proposals (NR 25-29).** Per Deep Inspection §C.3, every extension below takes a fresh NR; NRs 1-24 are live or reserved per [[cipher-abi-rule]] and Section 1.3's table.

| Proposed NR | Name | Direction | Payload | Op / purpose | Userspace caller | When |
|---|---|---|---|---|---|---|
| **25** | `CIPHER_FAIRNESS_SET_QUOTA` | `_IOW` | `struct cipher_fairness_quota` (tenant_id_str, quota_units, period_ns) | FAIRNESS — push per-tenant quota for state_updater to consume | userspace FAIRNESS op | **v1 Week 4** only if FAIRNESS goes beyond observe-only (per Wave 5 Cb.3); else defer to v1.5 |
| **26** | `CIPHER_PREDICT_PUSH_HOT_REGION` | `_IOW` | `struct cipher_hot_region` (devptr, size_kb, layer_idx) | PREDICT — informs kmod of hot L2 regions; state_updater writes `predicted_hot_regions[8]` slots | userspace PREDICT op | **v1.5** |
| **27** | `CIPHER_L2_BUDGET_SET` | `_IOW` | `__u32 budget_kb` | depth-win program — per-tenant `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` enforcement | userspace runtime | **post-CP-5.5 depth-win program** |
| **28** | `CIPHER_CARBON_REGION_SET` | `_IOW` | `struct cipher_carbon_region` (region_name[32], grams_co2_per_kwh) | CARBON+VOLT — operator-supplied marginal carbon factor for per-tenant carbon budget | userspace CARBON op | **v2** |
| **29** | `CIPHER_KOOPMAN_SURROGATE_ADD` | `_IOW` | `struct cipher_surrogate` (op_class, shape_hash, kernel_id, error_bound) | SUBSTITUTE — Goal-4 v2 dynamic registry insertion | userspace EDMD pipeline | **v2 / research** |

**v1 scope note.** Of these 5, only NR 25 (FAIRNESS_SET_QUOTA) is even a candidate for v1, and only IF FAIRNESS moves past observe-only in Week 4. If FAIRNESS ships observe-only in v1 (per Wave 5 §5.5 Week 4 default scope), zero new NRs are added in v1.

All extensions are additive. NRs 1-24 frozen.

<!-- v1.2: NF-5.4 snapshot reserved-tail seam — Wave 5 / Wave 3 — additive classifier→kmod fields without a new ioctl nr. -->

**Snapshot reserved-tail seam (Cb.2, NF-5.4).** The kmod's user-facing snapshot struct `cipher_tenant_snapshot_user` (`cipher_ioctl.h:178`) carries a `__u32 reserved[16]` tail by design — additive classifier-side fields can land there without bumping any ioctl NR. Wave 5 §5.5 schedules this as a Week-1 ABI bump (not a new NR; just a struct field rename within the reserved tail) for the classifier→kmod handshake. Proposed Week-1 layout consumes 4 of the 16 reserved slots:

| Field | Type | Purpose | Producer | Consumer |
|---|---|---|---|---|
| `recommended_sm_count` | `__u32` | Classifier-derived partition hint | state_updater kthread (Cb.5) | CP54 ALLOCATE handler (Cc.7) |
| `slo_priority` | `__u32` | SHIELD-derived priority class | state_updater | partition_router stream priority |
| `session_band` | `__u8` (4-pack with 3-byte padding) | HUMAN / AGENT / BATCH / UNKNOWN | SENSE → state_updater | every overlay op via snapshot |
| `tenant_billing_class` | `__u8` (in same 4-pack) | corp / free / research | operator config → state_updater | RECEIPT, FAIRNESS |

Remaining: `__u32 reserved[12]` for future extensions. ABI rule: any future struct change adds fields to the reserved-tail, never reorders or shrinks; field meanings frozen once shipped. Size invariant: `sizeof(struct cipher_tenant_snapshot_user) == 336` bytes — verified by T-W1.3 in §7 Week 1.

### 3.6 — CUDA driver compatibility

- Tree A: assumes `CUDA_INJECTION64_PATH=libcipher_rt.so.preroadmap` AND `LD_PRELOAD=libcipher_hook.so`.
- Tree B: post-CP 2.5, `CUDA_INJECTION64_PATH=libcipher_rt.so` ONLY (LD_PRELOAD-free deploy). GOT-patched cuBLAS at injection time.

**Operational invariant** ([[cipher-cp54-step1-3]]): libcipher_rt-loaded tenants (PARTITION / SHARED qos) MUST be launched via `CUDA_INJECTION64_PATH`, NOT plain LD_PRELOAD. The green-context init runs in `InitializeInjection/InitializeInjection2`, which the driver fires only for the injection lib.

**Path to merge:** Port classifier ops into the InitializeInjection callback chain (currently calls `cipher_rt_green_ctx_cp54_init()`). Add `cipher_rt_classify_init()`, `cipher_rt_sense_init()`, etc., in dependency order.

### 3.7 — Summary table of incompatibilities

| # | Item | Class | Work to resolve |
|---|---|---|---|
| 1 | `cipher_tenant_snapshot` struct name collision | requires-shim | already mitigated; use `_user` suffix when porting may13 ops that touch the snapshot |
| 2 | Stage-0 dispatch entry symbol (`cipher_dispatch_kernel`) — not present in rt_phase4 | requires-port | Port `cipher_dispatch.cpp` + `cipher_classify.hpp` + `cipher_oracle.cpp` as `cipher_rt_dispatch.{cpp,h}` |
| 3 | CUDA `.cu` source compilation | requires-build-system change | Add nvcc rules to `cipher_rt_phase4/Makefile`. v1 scope per v1.2.2 ADJUDICATION A1 + §7 W13-14 (Koopman narrow-domain ports); W13-14 scope-lock R-W13.2 plans rule addition. Host-only EDMD fallback documented in scope-lock if .cu port disruptive in budget. |
| 4 | may13 LD_PRELOAD `libcipher_hook.so` vs rt_phase4 LD_PRELOAD-free deploy | requires-port | Drop libcipher_hook.so; classifier ops port via InitializeInjection chain |
| 5 | 32-entry static recipe registry | minor | Port as-is; the registry-population gap is its own Section 4 work item |
| 6 | Tree A's three-thread Stage 1/2 architecture (never spawned in prod) | minor | Do not port the threads; port the ops as synchronous per-tenant hooks |
| 7 | `libcrypto` linkage | already-resolved | rt_phase4/Makefile already links `-lcrypto` for AUDIT |
| 8 | NCCL multi-node ops (NCCL_P2P) | out-of-v1 | Defer to Phase 6. Tree A's `cipher_nccl_*.cpp` files stay in may13. |
| 9 | **Dual-built source files** (cipher_kernel_table, cipher_graph_inspect, cipher_flow_recorder, cipher_flow_patterns, cipher_flow_substitute) compiled into BOTH libcipher_hook.so AND libcipher_rt.so in may13 | requires-port-decision | The unified runtime drops libcipher_hook.so entirely (CP 2.5 LD_PRELOAD-free deploy). Each dual-built source ports to exactly ONE location in cipher_rt_phase4 as a non-duplicated object. |
| 10 | Top-level vs src/ split for `cipher_dispatch.cpp` and `cipher_oracle.cpp` in may13 (live versions at top-level, src/ copies are SILENTLY EXCLUDED by Makefile L29 `filter-out` and differ from top-level per `diff -q`) | requires-port-decision | Port the TOP-LEVEL versions ONLY. Delete the src/ shadow copies. Anyone naively rsync-ing `src/*` introduces stale dispatch code. |
| 11 | may13's `cipher_weight_share.cpp` (cuIpc-based) vs Track 2 SC5's cipher_kmod weight_arena fd custodian (VMM POSIX fd, kmod-resident) | requires-supersession-decision | Track 2 SC5 supersedes may13's IPC weight-share. The may13 file does NOT port — Track 2 ships. |
| 12 | License boundary: `cipher_marlin_src.cpp` carries Apache-2.0 string-literal kernel (33.8 KB upstream IST-DASLab Marlin) | minor | Carry through; already in rt_phase4 via `cipher_rt_marlin_kernel_src.cpp`. License compliance unchanged. |

**Blocking conflicts: zero.** Every mismatch is either already resolved, additive, or a deliberate v1.5/v2 deferral. Two require explicit port decisions (#9 dual-built, #10 top-level vs src/) — both resolved in this plan.

---

<a id="section-4"></a>
## SECTION 4 — UNIFIED ARCHITECTURE DESIGN

### 4.0 — The reusable abstraction (key architectural finding)

The deployed runtime ALREADY contains the substrate pattern the unified runtime extends: **priority-ordered actuator registries**. Two instances ship today:

- `cipher_rt_matmul_dispatch.c:52-122` — `cipher_rt_matmul_register_actuator(priority, name, maybe_handle_fn)` + `cipher_rt_matmul_dispatch(call, real_fn_passthrough)`. 3-value enum (HANDLED / PASSTHROUGH / ERROR). Walks 16-slot priority-sorted actuator array; first HANDLED returns; ERROR breaks the loop and falls through to real cuBLAS; PASSTHROUGH tries next; all-PASSTHROUGH calls real `cublasGemmEx`.
- `cipher_rt_attn_dispatch.cpp:211-242` — same shape for SDPA (flash / efficient / cudnn backends), but with a **4-value enum** (HANDLED / PASSTHROUGH / REDIRECTED / ERROR), snapshot-under-lock at dispatch time, and ERROR semantics that continue (not break) the loop.

<!-- v1.2: 5 semantic divergences between matmul and attn substrate registries surfaced per Deep Inspection §A.3 — v1.1 implied uniformity; Deep Inspection §A.3 verified they are nearly the same but NOT identical. Classifier substrate must pick a contract explicitly. -->

**Semantic divergences (5) between the two substrates** (Deep Inspection §A.3). The unified runtime cannot pretend uniformity; the classifier substrate added in Week 1 must pick a contract per dimension:

| # | Dimension | matmul behavior | attn behavior | Class |
|---|---|---|---|---|
| 1 | Result enum cardinality | 3 values (HANDLED / PASSTHROUGH / ERROR) | **4 values** (HANDLED / PASSTHROUGH / REDIRECTED / ERROR). Actuator returning `2` means PASSTHROUGH on attn but ERROR on matmul. | SEMANTIC |
| 2 | `maybe_handle` signature | `(call, *out_status)` — actuator writes cuBLAS-style status | `(call)` — actuator returns enum only | SEMANTIC |
| 3 | Lock-during-dispatch | NO lock (assumes registry append-only after init) | LOCKS to memcpy snapshot, then releases | SEMANTIC (matmul faster, attn safer) |
| 4 | ERROR handling | `break` — exits loop, falls through to real cuBLAS | continues to next actuator (the `break;` inside the switch case exits only the switch) | SEMANTIC |
| 5 | HANDLED bypass | Substrate returns `*out_status` directly; real cuBLAS not called | **Trampoline ALWAYS calls `orig(...)` regardless** of route() returning HANDLED — T4.6.1 observe-only contract; HANDLED is structurally discarded today | BLOCKING for future attn substitute actuators (LP-2; Wave 5 §5.4) |

**Chosen classifier-substrate contract (hybrid).** Per Deep Inspection §A.3 recommendation, the new classifier substrate matches:
- **Enum** (4 values: HANDLED / PASSTHROUGH / REDIRECTED / ERROR) — follow **attn** (superset is safer; REDIRECTED reserved for v2 surrogate path).
- **`maybe_handle` signature** (single-arg) — follow **attn** (classifier does not need cuBLAS-style status propagation).
- **Lock-during-dispatch** (snapshot under lock) — follow **attn** (safer; ~16 × sizeof(actuator) memcpy per call acceptable).
- **ERROR handling** (break loop, fall through to default routing) — follow **matmul** (classifier errors should be terminal not retried).
- **HANDLED semantics** — defined explicitly here: classifier HANDLED means "this classifier produced a confident result and downstream routing accepts it"; PASSTHROUGH means "no opinion, try next or default".

**This pattern is the unification join point.** Every may13 op that needs to observe or modify dispatch ports as a registered actuator on one of these substrates (or on the new classifier-substrate added in Week 1). Specifically:
- **Observers** (AUDIT-style): register at priority 0, always return PASSTHROUGH, do their telemetry / HMAC / etc. work as a side effect. AUDIT is the canonical template (`cipher_rt_audit.c:132-137`).
- **Classifiers** (CLASSIFY / SENSE-style): register on the new classifier substrate added in Week 1; return classify_result enum; downstream dispatch routes on the highest-confidence classification.
- **Actuators** (SUBSTITUTE / Marlin-style): register at high priority on matmul / attn substrate; gate on workload / regime; return HANDLED only when actuator's regime conditions are met. Marlin INT4 is the template (`cipher_rt_marlin_actuator.c:170`).

<!-- v1.2: attn HANDLED-discard structural note added per Deep Inspection §A.3 finding #5 / Wave 5 LP-2. -->

**Attn HANDLED-discard structural note (Wave 5 LP-2).** The attn substrate's three SDPA trampolines (`cipher_rt_attn_dispatch.cpp:321 / 359 / 398`) ALWAYS call `orig(q,k,v,...)` regardless of route() result. T4.6.1 ships observe-only by design and a HANDLED return is structurally discarded. v1 observers (priority 5 classifier, priority 0 audit) never return HANDLED so this is invisible. When the first attn substitute actuator (Op-3 FAVOR+ / FlashSwiftKey, v1.5) lands, the trampolines must be refactored to branch on route()'s return value — see §7 Week 2 LP-2 fold (refactor required before Week 3 attn lane). The refactor adds an `out_status_devptr` field to `cipher_rt_attn_call` and ~20 LOC across the 3 trampolines.

**Dual device note.** Two char devices coexist in the kmod:
- `/dev/cipher` — magic byte `'C'`, 24 ioctl NRs (Section 1.3 table). The main ABI.
- `/dev/cipher_kvdedup` — magic byte `'K'`, 5 ioctl NRs (INIT/PUT/CONFIRM/FREE/STATS). The L3 cross-process KV-page dedup substrate (T4.6.4). Owns refcount table + holds cuIpc POSIX-fd handles (kmod outlives tenants).

The unified runtime preserves both. The KV bridge `cipher_kv_bridge.cpp` is the Python C-ext that wires vLLM to the kvdedup substrate.

### 4.1 — Process model

The unified runtime is a **single shared library `libcipher_rt.so`** loaded into every tenant process via `CUDA_INJECTION64_PATH`. Threading model:

- **Per-tenant tenant process** = one CUDA process (vLLM worker, agent, etc.) that loads libcipher_rt at cuInit.
- **Inside each tenant process:**
  - One **CUPTI thread** (`cipher_cupti.c`) — submits launch counters to kmod nr 7.
  - The **main CUDA stream(s)** — per-stream operation; hot-path classify/dispatch runs on the launching thread synchronously.
  - **No Stage 1/2 background threads** in v1 (per [[cipher-cp54-step1-3]] — they are never load-bearing in production).
- **Kmod** is the **cross-process coordinator** — owns the SM-arbitration ledger, the FLOP samples ring, the weight arena custodian, the kvdedup table. All cross-tenant state lives in the kernel.
- **Two daemon processes outside any tenant:**
  - `cipher-gpustate` — NVML poller @ 250ms, submits nr 5/6.
  - `cipher_flopd` — CUPTI PM Sampling daemon (root), submits nr 11 @ ~10Hz.

Per-thread fd rule (PHASE_4_ARCHITECTURE.md §"Binding deployment requirement"): every actuator path that issues `/dev/cipher` ioctls from N parallel threads MUST hold its own fd. The contention sweep (kmod 0.4.3) showed 44.2× p99 amplification under shared-fd that collapses to 1.4× per-thread.

### 4.2 — Dispatch pipeline per kernel

The unified per-kernel hot path:

```
CUDA kernel launch (cuLaunchKernel / cudaLaunchKernel intercept via GOT patch)
  ↓
[1] CLASSIFY                                     ← geometry fingerprint → kernel_class
  ↓ (returns kernel_class enum)
[2] RING_WRITE                                   ← SPMC ring entry (timestamp, params_hash, class)
  ↓
[3] ORACLE                                       ← 5-gate safety: phase / min-confidence / structural lookup / EMA demotion / N≤4
  ↓ (returns SUBSTITUTE | PASS_THROUGH | DEMOTE)
  ↓
[4] If SUBSTITUTE:
      [4a] SUBSTITUTE dispatch table lookup       ← keyed on (kernel_class, params_hash, tenant_band)
        ↓
      [4b] Route to actuator:
              ├─ Marlin INT4 lane (B≥8, weight-quantized, full-GPU primary ctx)
              ├─ cuBLAS shim lane (FP16 GEMM, partition-aware via stream priority)
              ├─ Attention substrate lane (T4.6.1 — SDPA dispatcher trampoline)
              ├─ FP8 lane (env-gated, T8.x)
              └─ Koopman O(1) lane (narrow-domain v1 W13-14; wide-domain v2)
        ↓
      [4c] COMMIT — launch the chosen kernel
  ↓
[5] Post-launch observability fan-out (asynchronous, off critical path):
      ├─ SAMPLE (warmup window) → ring entry
      ├─ AUDIT → HMAC chain advance
      ├─ TRACE → bounded JSONL emit
      ├─ DETERMINISM → fingerprint
      ├─ FAIRNESS → per-tenant quota debit (FAIRNESS_SHM cross-process)
      ├─ CARBON → energy estimator
      ├─ RECEIPT → per-session proof
      ├─ GUARD → cross-session leak check (every 100 launches)
      └─ LOOP / PIPELINE / PULSE / COMPLY → session-band observers
```

<!-- v1.2: Stage 0 budget corrected per Wave 5 NF-5.1 (LP-14) — the 12 ns figure is unsupportable; honest budget is 100-200 ns / launch, accept 1-2% per-tenant overhead. -->

**Total Stage 0 budget (honest).** Wave 1 control-flow analysis and `cipher_classify.hpp` header note ("~160 ns cache-hit") put the realistic budget at **100-200 ns / launch**, not the 12 ns figure originally written in PHASE_4_OP_WORKLOAD_MATRIX.md. With cublasGemmEx rate ~10K/s/tenant × 16 tenants × 200 ns ≈ 4% CPU on one core. The honest framing is **accept 1-2% per-tenant overhead as the primary case** (not the fallback). R-C1 in §8 is updated accordingly; Week 2 verifies under cycle counter with budget < 500 ns/launch as the alarm threshold.

<!-- v1.2: NF-5.6 classifier confidence is binary (40/85) — surfaced per Wave 5 LP-11 / Wave 1 control flow. -->

**Classifier output characterisation (NF-5.6 / Wave 5 LP-11).** The Wave 1 brain emits **binary** confidence values — 40 for ITERATIVE_CUSTOM op class, 85 for everything else — not a continuous [0, 1] score. With ORACLE's default `min_confidence = 60`, every ITERATIVE_CUSTOM kernel exits early at the main path L468 (PASS_THROUGH). The fallback IS the existing PASS_THROUGH — already lossless against any substitution. The "loss" is opportunity cost (no observer fires either for that small fraction of launches). Optional v1.5 mitigation: env `CIPHER_OBSERVE_LOW_CONFIDENCE=1` to still publish the TLS hint for observer-only consumption while keeping SUBSTITUTE off. This means downstream actuators reading "confidence < 0.9" in any planning diagram must read it as "confidence < 60 / 100" rather than as a continuous score — the routing tables in §4.5 are exact-match, not soft-routed.

### 4.3 — Per-regime pipeline specialization

The dispatch above specializes per workload regime detected by CLASSIFY + SENSE:

#### Regime 1: Compute-bound prefill (large GEMM, B≥8)
```
kernel_class = LARGE_GEMM
SENSE.band = BATCH or AGENT(prefill phase)
ORACLE.gate = SUBSTITUTE
→ Marlin INT4 lane (full-GPU primary ctx) OR cuBLAS FP16 partition-aware lane
→ TARGET: MFU 85% at sustained 10+ min (G1 gate)
→ Concurrent VOLT engages: high-clock (default), no DVFS cut
→ Concurrent PREDICT: L2 hot-region preload via cuMemPrefetchAsync
```

#### Regime 2: Memory-bound decode (B=1 single or batched via POOL)
```
kernel_class = SMALL_GEMM or MATVEC or ATTN_DECODE
SENSE.band = HUMAN(decoding) or AGENT(decoding)
ORACLE.gate = PASS_THROUGH (Marlin REGRESSES at B=1 per CP_2_4_REPORT.md:168-172)
→ cuBLAS lane (FP16) OR attention substrate lane
→ TARGET: TPW lift via DVFS (VOLT 1200 MHz sweet spot = +14% under vLLM)
→ Concurrent SUSTAIN: monitor KV-pressure slope
→ Concurrent kvdedup (kmod L3) for shared prefixes
→ Concurrent batching via POOL (libcipher_v2 batched executor)
```

#### Regime 3: Multi-tenant agentic burst (N concurrent agents)
```
SENSE.band = AGENT
ARBITRATE: kmod CP54 ledger allocates 8-SM (or N×8) partition per agent
Track 3 DSM: migratable opt-in, migrates idle partitions to busy
SUBSTITUTE per tenant: routed through tenant's partition (cuBLAS shim, partition-aware)
→ Marlin route GATED OFF (structurally full-GPU, doesn't compose with partition per PHASE_5_ARCHITECTURE_REVISION.md)
→ Weight sharing (Track 2 arena) — single physical weight copy across N same-model tenants (76% HBM saved)
→ KV-prefix dedup (kmod kvdedup) — cross-process shared system prompt stored once
→ FAIRNESS per-tenant quota; SHIELD per-tenant priority
→ Cross-tenant batching (libcipher_v2 POOL) — N tenants' decode steps fused, 3.06-5.98× scaling
→ **DVFS is FLEET-POLICY in this regime, NOT per-tenant** — see §4.3.6 below
→ TARGET: agents/GPU = 100 (G3 gate); fleet tok/W
```

#### Regime 4.3.6: DVFS scope — a structural caveat the architecture must own

The kmod's `CIPHER_SET_CLOCK_MHZ` actuator (ioctl nr 10) shells `nvidia-smi -lgc <mhz>` via `call_usermodehelper` (`cipher_kmod/cipher_clock.c:72`). **This is device-global — the H100 has no per-SM-group clock domain accessible to third parties.** The unified architecture cannot give Tenant A's prefill partition a 1980 MHz clock while Tenant B's decode partition sits at 1200 MHz simultaneously; both share whatever the global lock is set to.

**Architectural rule (binding):**

- **In multi-tenant deployments (Regime 3/4/5): DVFS is FLEET-POLICY, not per-tenant.** The fleet-level clock-lock is computed at deployment time (or periodically by a fleet-policy worker) to maximize **aggregate fleet tok/W across the mix of tenants currently active**. The choice is a function of the workload-class distribution: if the fleet is decode-heavy (most agents), 1200 MHz is right (+14% per-tenant tok/W via [[cipher-t431-volt-shipped]] / +13.9% under vLLM per FUTURE_SCOPE_A_PHASE_3_5_RESULTS.md); if the fleet is prefill-heavy (RAG burst), 1980 MHz is right.
- **In single-tenant or solo-batch deployments (Regimes 1/2 alone): DVFS is per-batch.** The B=1/B=8/B≥32 LUT in `cipher_rt_volt.c` (1000/1600/1980 MHz) drives the lock.
- **The fleet-policy decision is taken by a new component:** `cipher_rt_fleet_dvfs_policy` (Section 7 Week 3 / 4 deliverable). It reads `/proc/cipher/stats` for the current tenant-class distribution, applies the policy, calls ioctl nr 10. Cadence: every 10 seconds (slow enough to avoid clock thrash).
- **The customer-facing TPW claim under multi-tenant** is built around the multi-tenant batching curve (3.06-5.98×) and the density story, NOT around per-tenant DVFS. DVFS contributes a single fleet-level constant multiplier.

This is not a defeat — it is a structural property of H100 clock-domain reality and the architecture owns it explicitly.

#### Regime 4: Tool-call idle (agent paused, GPU has free SMs)
```
SENSE.band = AGENT
LOOP detects tool-call boundary; PIPELINE correlates multi-agent session
DSM (Track 3) — migrate idle agent's 8-SM-group to busy agents
Idle partition stays IDLE (no waste — its SMs are reallocated)
HIBERNATE — optionally lower clock on idle partition's group (v1.5; v1 = no-op)
→ TARGET: zero GPU-idle-during-tools
```

#### Regime 5: Mixed real production
All four regimes interleaved. CLASSIFY runs per kernel; SENSE runs per session; phase-detection drives transitions:
- Each tenant's prefill kernels → Regime 1 actuators
- Each tenant's decode kernels → Regime 2 actuators
- Cross-tenant aggregate → Regime 3 actuators
- Tool-call gaps → Regime 4 actuators
- All five observability streams run cross-cutting

**This is the marvel.** Every kernel is classified live, every actuator selection is regime-correct, every customer metric is the right one for that regime.

### 4.4 — Op placement matrix (where each op runs)

Per [[cipher-cp54-step1-3]] / PHASE_4_CONTRACT.md Mode 1/2/3 access pattern:

| Op | Path | Mode | Synchrony | Periodicity | State scope |
|---|---|---|---|---|---|
| CLASSIFY | hot path | Mode 3 TLS | sync | per launch | per-tenant cache |
| ORACLE | hot path | Mode 3 TLS | sync | per launch | global registry + per-tenant phase |
| SUBSTITUTE | hot path | Mode 3 TLS | sync | per launch | per-tenant snapshot |
| COMMIT | hot path | inline | sync | per launch | none (dispatch return) |
| SAMPLE | hot path | inline | sync | per launch (warmup) | per-tenant ring |
| RING_WRITE | hot path | inline | sync | per launch | per-tenant ring |
| AUDIT | post-launch | async (deferred) | async | per launch | per-session HMAC chain |
| TRACE | post-launch | async | async | sampled | bounded JSONL buffer |
| DETERMINISM | post-launch | async | async | per launch | per-tenant counter |
| SENSE | per-session entry | Mode 2 /proc poll | sync at REGISTER_TENANT, periodic refresh | per session | per-tenant snapshot |
| SHIELD | hot path | Mode 3 TLS | sync | per launch | per-tenant SLO state |
| SUSTAIN | post-launch | async | async | per N launches | per-tenant KV state |
| PREDICT | setup + post-launch | Mode 2 /proc poll | async | per N launches | per-tenant L2 region table |
| GUARD | per-N-launches | Mode 2 /proc poll | sync at boundary | every ~100 launches | per-session pointer set |
| THERMOSTAT | periodic | Mode 1 ioctl (nr 5) | sync at cadence | 1 Hz | global thermal state |
| VOLT | setup + on-thermal-event | ioctl nr 10 | sync (cold path) | event-driven | global clock lock |
| HIBERNATE | event-driven | (v1.5+) | async | event | per-partition |
| ARBITRATE | partition-change | ioctl nr 13-15 | sync at cuInit + on resize | event-driven | kmod ledger |
| DSM | partition-change | ioctl nr 16-20 | sync | event-driven | kmod migration FSM |
| FAIRNESS | per-N-launches | Mode 3 TLS | async | per N launches | per-tenant quota |
| CARBON | post-launch | async | async | per launch | per-session energy accum |
| RECEIPT | session-boundary | sync at end | sync | per session | per-session HMAC |
| LOOP | post-launch | async | async | per N launches | per-session runaway latch |
| PIPELINE | post-launch | async | async | per N launches | per-session multi-agent corr |
| PULSE | periodic | Mode 1 ioctl | sync at cadence | 1 Hz | global hw fault state |
| CONTINUITY | per-N-launches | Mode 2 /proc poll | sync at boundary | per N launches | per-tenant region tracking |
| COMPLY | session-boundary | sync at end | sync | per session | per-session verdict |
| TOPOLOGY | init only | sync at init | sync | once | global topology |
| STRAGGLER | post-launch | async | async | per N launches | per-tenant runtime stats |
| NCCL_P2P | (deferred Phase 6) | — | — | — | — |
| REMEMBER, VALIDATE, SPECULATE, ADAPT | (v2 learning tier, deferred) | — | — | — | — |

### 4.5 — Routing tables

#### Dispatch table (CLASSIFY → actuator)

| kernel_class | SENSE.band | tenant context | → actuator lane |
|---|---|---|---|
| LARGE_GEMM (M,N,K ≥ 512) | BATCH or AGENT-prefill | full-GPU primary ctx | Marlin INT4 (if weight-quantized AND B≥8) OR cuBLAS FP16 |
| LARGE_GEMM | AGENT-prefill | partitioned (8-SM) | cuBLAS partition-aware (Marlin gated OFF) |
| SMALL_GEMM (M,N,K < 512) | AGENT-decode or HUMAN-decode | partitioned | cuBLAS lane |
| MATVEC (M=1 or N=1) | AGENT-decode | partitioned | cuBLAS lane |
| ATTN_DECODE (Q/K/V dispatch) | AGENT-decode | partitioned | attention substrate lane (T4.6.1 SDPA trampoline) |
| ATTN_PREFILL | AGENT-prefill | full-GPU | attention substrate lane |
| FP8_COMPUTE | env-gated | any | FP8 substitute lane (RECIPES path) |
| KOOPMAN_ELIGIBLE | any | any | **narrow-domain v1** (Koopman O(1) lane — W13-14 per v1.2.2 ADJUDICATION A1 and §7 line 1258; wide-domain dynamic discovery remains v2) |
| (default) | any | any | PASS_THROUGH (existing cuBLAS/cuDNN runs unmodified) |

#### Safety gate (ORACLE → SUBSTITUTE | PASS_THROUGH | DEMOTE)

| ORACLE check | SUBSTITUTE if all pass | else |
|---|---|---|
| phase == STEADY (not warmup) | yes | DEMOTE |
| min-confidence ≥ 0.9 | yes | PASS_THROUGH |
| structural lookup HIT in registry | yes | PASS_THROUGH (queue for v2 surrogate population) |
| EMA error < threshold | yes | DEMOTE (drop confidence) |
| outstanding substitutions N ≤ 4 | yes | PASS_THROUGH (rate-limit) |

#### Workload-state gate (SENSE → per-tenant resource state)

| SENSE input | output |
|---|---|
| `CIPHER_REGISTER_TENANT(tenant_id_str)` payload | session_fp (FNV-64), handle_u32 (murmur32), session_band (HUMAN/AGENT/BATCH) |
| Per-launch frequency, kernel-class distribution | phase-detect (prefill/decode/tool-idle) |
| `cipher_pid_stats` SM/MEM util history | resource state (thermal_headroom, power_headroom) |

### 4.6 — State management

**Per-tenant state ownership:**

| State | Owner | Producer | Consumers |
|---|---|---|---|
| tenant_id_str, session_fp, handle_u32 | kmod `cipher_pid_stats` | REGISTER_TENANT (nr 1) | SENSE, FAIRNESS, RECEIPT, AUDIT |
| sm_util_pct, mem_util_pct, launches_total | kmod `cipher_pid_stats` | SUBMIT_GPU_STATE, SUBMIT_PROCESS_UTIL, SUBMIT_LAUNCH_STATS | every actuator via snapshot |
| sm_partition_mask, sm_partition_count | kmod cp54 ledger | CP54_ALLOCATE (nr 13) | green_ctx client, ARBITRATE |
| voltage_envelope_mv, sustained_clock_mhz | kmod state_updater | SET_CLOCK_MHZ (nr 10) + state_updater | VOLT consumer, THERMOSTAT |
| l2_residency_kb, hot_region_count | kmod state_updater | PREDICT pushes (v1.5) | SUSTAIN, PREDICT |
| kv_cache_size_mb, kv_compression_ratio_pct | userspace kv_alloc client | userspace allocation events | SUSTAIN, GUARD |
| weight_dedup_savings_mb | kmod weight_arena | ARENA_REGISTER/IMPORT (nrs 21-24) | observability |
| session_band, slo_priority, fairness_quota_remaining | kmod state_updater (writes from SENSE/SHIELD/FAIRNESS userspace pushes) | userspace ops | SHIELD, FAIRNESS, observability |
| graph_capture_state, koopman_substitution_eligibility | userspace per-tenant | dispatch lane decisions | observability |

**Cross-op state flow:** CLASSIFY output → ORACLE input → SUBSTITUTE input → COMMIT. SENSE updates session_band on each REGISTER and periodically — read by every overlay op via snapshot.

**State teardown on tenant exit:** kmod's `do_exit` reaper releases:
- `cipher_pid_stats` hashtable entry
- CP54 partition allocation (Track 3 migration FSM finalizes any in-flight migration)
- weight_arena consumer membership (last consumer triggers arena reap workqueue, 5s cadence)
- HMAC chain checkpoint (AUDIT)

Per Track 3 SC1: do_exit reaper handles concurrent crashes correctly (verified at 5-seed integration).

### 4.7 — Fusion contract summary (43 contracts across 5 categories)

<!-- v1.2: new subsection per Wave 5 §5.1 — the 43 fusion contracts catalogued between Tree A (classifier brain + observers) and Tree B (substrate + kmod) are too long to inline; this subsection summarises per category and points to the source. -->

Wave 5 §5.1 ("FUSION CONTRACTS") catalogues every meeting point between the trees as a contract with stable ID `C<class>.<num>`, error semantics, concurrency assumptions, fusion class (PORT-AS-IS / SHIM-REQUIRED / REFACTOR-REQUIRED / VESTIGIAL / REFACTOR-RETIRE), and severity. Total **43 contracts** across 5 categories. The full per-contract table is in Wave 5 §5.1.a through §5.1.e (file `CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md`, L113-194). Summary here:

| Category | Contracts | What it covers | Source range (Wave 5) |
|---|---|---|---|
| (a) Classifier → Actuator | **12** (Ca.1 – Ca.12) | New pre-launch hook, weak-symbol re-entry replacements (Ca.2 / Ca.3 supersede Tree A's `tls_get_gemm_shape` / `tls_relaunch`), classifier observer registration on matmul (Ca.7) and attn (Ca.8) substrates, Marlin TLS-hint read (Ca.9), 80-layer drift fix (Ca.10), `classify_launch` header port (Ca.11), `kernel_name` wire into structural_lookup (Ca.12). | §5.1.a (L113-130) |
| (b) Classifier → Kmod | **5** (Cb.1 – Cb.5) | Brain reads `cipher_dev_get_tenant_snapshot` nr 8 (Cb.1), consumes new fields in snapshot `reserved[16]` tail (Cb.2), emits classifier decisions via proposed nr 25 (Cb.3 — v1.5+), registers per-pid hook into kprobe pre-handler (Cb.4), state_updater per-entry brain hook (Cb.5). | §5.1.b (L132-141) |
| (c) Actuator → Kmod | **11** (Cc.1 – Cc.11) | Every Wave 2 substrate to Wave 3 ABI call: nr 1 (Cc.1 REGISTER_TENANT), nr 7 (Cc.2 SUBMIT_LAUNCH_STATS), nr 8 (Cc.3 GET_TENANT_SNAPSHOT), nr 10 (Cc.4 SET_CLOCK_MHZ), nr 11/12 (Cc.5/Cc.6 FLOPS), nr 13 + nrs 14-20 (Cc.7/Cc.8 CP54), kvdedup nrs 1-5 (Cc.9), weight-arena nrs 21-24 (Cc.10), stream-create CB (Cc.11). All PORT-AS-IS. | §5.1.c (L144-159) |
| (d) Observer → {Classifier, Actuator, Kmod} | **8** (Cd.1 – Cd.8) | SHIELD priority band write (Cd.1), fairness_shm hot-path consumer (Cd.2 — LP-5 v1.5 migration to kmod nrs 25-27), THERMOSTAT liquid-state read (Cd.3), PULSE NVML ECC (Cd.4), HIBERNATE NVML power limit (Cd.5 — LP-9 v1.5 kmod-clock fallback), SENSE TLS session read (Cd.6), Stage-1 SPMC ring fan-out (Cd.7), AUDIT record() shim for substitute-lane (Cd.8). | §5.1.d (L162-177) |
| (e) Kmod-internal cross-TU | **7** (Ce.1 – Ce.7) | Invariants surviving the port unchanged: `cipher_pid_table` RCU discipline (Ce.1), do_exit two-step reaper (Ce.2), exit-order invariant (Ce.3), gpu_state spinlock (Ce.4), CP54 lock vs lock-free reaper (Ce.5), weight-arena lock + reaper (Ce.6), kvdedup lock + release fop (Ce.7). | §5.1.e (L180-194) |

**Lossless vs lossy.** Wave 5 §5.3 classifies 22 of these as lossless (compose without information loss). Wave 5 §5.4 surfaces 16 lossy fusion points (LP-1 through LP-16) across the 43 contracts with severity rollup 2 blocking / 5 requires-mitigation / 9 minor — these are folded into §8 risk register.

### 4.8 — COMMIT as atomic state-transition primitive (v1.2.2 ADJUDICATION 2)

<!-- v1.2.2: new subsection per OP_INTENT_VS_IMPLEMENTATION Item I-1 (D2 register). COMMIT was UNDOCUMENTED-INTENT in the audit; v1.2.2 adjudicates by promoting it from implicit dispatch-return to atomic state-transition primitive. Engineering Weeks 7-8. -->

The v1.2.1 plan treated COMMIT as the implicit dispatch return phase: when an actuator returns HANDLED, the substrate returns the cuBLAS status to the caller, and individual observers update their own state independently. That phrasing tolerated drift: any two observers reading per-tenant state could see partially-updated views of a single kernel launch's effects, because no order or barrier discipline governed the cross-observer state transition. OP_INTENT_VS_IMPLEMENTATION Item I-1 surfaced this as undocumented intent; v1.2.2 resolves the adjudication by promoting COMMIT to a named atomic primitive with a documented contract.

**Contract.** Post-actuator-return, COMMIT updates the following per-tenant state in a deterministic order, atomically with respect to any downstream observer read:

1. **AUDIT chain HMAC entry** (`cipher_rt_audit_record`, per `cipher_rt_audit.c`): advance the chain with the canonical 96-byte block.
2. **FAIRNESS quota and arbitration counters** (`cipher_fairness_observe` + `cipher_fairness_shm`-side aggregate): increment per-tenant work-units, update quota overrun flag if crossed.
3. **CARBON power accounting** (`cipher_carbon_observe`): increment per-tenant energy accumulator.
4. **RECEIPT billing surface** (`cipher_receipt_observe`): advance the per-session FNV chain + launch counter.
5. **kmod-resident tenant state** (`cipher_pid_stats` per-launch fields via `state_updater` 1 kHz mirror): launches_total, sm_partition_mask snapshot, derived fields.

**Output.** A canonical post-kernel state snapshot stored in `cipher_pid_stats` (read by observers via `cipher_get_current_tenant_snapshot()`). Observers read from this snapshot rather than mutate their own per-kernel state independently.

**Atomicity mechanism.** Per-tenant sequence counter incremented before the 5-step update, with release-fence on the snapshot publish and acquire-fence on the consumer read. This is **not** a global lock — at N=100 tenants a global lock would serialise the entire dispatch hot path. The per-tenant sequence counter is a single `__atomic_fetch_add` (relaxed) on the write side and a `__atomic_load_explicit(acquire)` on the read side; the snapshot publish uses `__atomic_store_explicit(release)`. Contention is bounded to the per-tenant slot — no cross-tenant ordering is enforced (correct: cross-tenant launches are independent).

**Observer protocol change.** Existing v1.2.1 observers each call their own `_observe(ev)` hook with per-kernel state updates. The v1.2.2 unified observer protocol routes through the COMMIT primitive: per-kernel hooks remain (for the per-launch counters that are not in the snapshot), but the post-COMMIT snapshot is the read source for any cross-observer state. Specifically:

- AUDIT chain remains the authoritative HMAC of decision history; readers verify chain via `audit_verify.py` and read the published snapshot for per-launch decisions.
- FAIRNESS quota crossings are observed by AUDIT (Wave 5 Cd.8) and by COMPLY at report time via the snapshot.
- CARBON / RECEIPT / TRACE read the post-COMMIT snapshot at report time.

**Migration path.** Existing ad-hoc state updates (each observer calling its own `_observe` independently) → COMMIT-mediated atomic updates (observers still call their per-launch hooks, but the snapshot publish is the single source of cross-observer truth). The 21 overlay ops port unchanged at the per-launch level; their report-time consumers switch from reading global counters directly to reading the snapshot. Verified backward-compatible because no overlay op currently mutates another op's state (per OP_CONTRACT.md I2 "Pure Stage 1 observer").

**Engineering estimate.** 10 days (Weeks 7-8 in §7). Touches all 21 overlay ops' report paths (mechanical) + the snapshot writer in the state_updater kthread (~50 LOC kmod change) + the substrate's actuator-return path (~20 LOC userspace change to call the COMMIT publish after each HANDLED/PASSTHROUGH).

### 4.9 — RING_WRITE as lock-free inline telemetry substrate (v1.2.2 ADJUDICATION 3)

<!-- v1.2.2: new subsection per OP_INTENT_VS_IMPLEMENTATION Item I-2 (D2 register). RING_WRITE was REQUIRES-INTENT-CLARIFICATION because the may13 ring has no spawning consumer in production. v1.2.2 adjudicates by treating RING_WRITE as a distinct substrate from CUPTI. Engineering Weeks 9-10. -->

The v1.2.1 plan surfaced RING_WRITE as a v1 substrate without a spawning consumer (Stage 1/2 threads default-off), and noted that CUPTI already provides per-launch telemetry. OP_INTENT_VS_IMPLEMENTATION Item I-2 framed the user-adjudication choice. v1.2.2 resolves the adjudication by ruling that **RING_WRITE and CUPTI cover different telemetry needs and the unified runtime ships both**.

**Why both.** CUPTI's `cuptiActivityKernel7` callback fires post-launch from a CUPTI-owned thread; it's reliable but adds tens to hundreds of nanoseconds of overhead per launch on the CUPTI thread's queue, and the callback runs asynchronously to the launch thread (so it cannot inform a pre-launch decision). RING_WRITE fires synchronously from the LD_PRELOAD-equivalent injection (the GOT-patched cuLaunchKernel intercept) at sub-microsecond cost, on the launch thread, with the kernel descriptor in hand. The two substrates feed different consumers:

| Consumer | Telemetry source | Rationale |
|---|---|---|
| CLASSIFY hot-path cache | RING_WRITE (pre-launch, on-thread) | Classifier reads cached fingerprint per launch; CUPTI callback is too late |
| ORACLE phase detection | RING_WRITE post-dispatch entry | Oracle's phase state advances on every dispatch decision |
| AUDIT chain | RING_WRITE post-dispatch entry | The (decision, kernel_class, params_hash) tuple must be in the chain |
| REMEMBER (Koopman tier) | RING_WRITE post-dispatch entry | The CfC hidden-state update needs the per-launch class + decision |
| Per-tenant kmod telemetry | CUPTI activity queue → ioctl nr 7 SUBMIT_LAUNCH_STATS | Aggregate counters at 256× batching; off-thread |
| MFU / FLOP attribution | CUPTI PM Sampling → ioctl nr 11 SUBMIT_FLOP_SAMPLE | NVPM hardware counters; CUPTI is the only access |
| Per-launch hardware event correlation | CUPTI (kernel duration, occupancy) | CUPTI is the only post-launch hardware event source |

**Ring buffer structure.** Per-tenant SPMC (single-producer multi-consumer) ring buffer in process-anonymous shared memory (TLS-resident; the producer is the launching thread, the consumers are this tenant's REMEMBER/AUDIT/classifier-feedback background threads). Size 65,536 entries per tenant (matches may13 `cipher_10ops.h:83-100`). Entry layout per may13 `CipherRingEntry` (sequence, timestamp_ns, timestamp_delta, kernel_class, grid/block dims, fn_hash, confidence, decision, params_hash). Lock-free MPMC unnecessary because each tenant has exactly one producer thread (the launching tenant); the multiple consumers all advance their own read sequence (`read_seq_s1` for Stage 1 / REMEMBER, `read_seq_s2` for Stage 2 / classifier-feedback, additional sequences added per consumer).

**Producer protocol.** Inside the GOT-patched cuLaunchKernel intercept, after `cipher_dispatch` returns:

1. Compute `seq = atomic_load_explicit(&ring->write_seq, relaxed)`.
2. Compute minimum of read sequences (Stage 1 / Stage 2 / AUDIT / classifier-feedback).
3. If `seq - min_read >= 65536`, the ring is full: **drop the entry, increment per-tenant drop counter, return** (telemetry loss is bounded and reported).
4. Otherwise memcpy the entry into `buf[seq & MASK]` and `atomic_store_explicit(&ring->write_seq, seq + 1, release)`.

Total cost on x86 TSO: ~10 ns (one relaxed load + comparison + memcpy of ~64 B + one release store). Budget per OP_CONTRACT.md I5 ("No CUDA / syscalls on the observe path").

**Consumer protocol.** Each consumer thread (REMEMBER thread, AUDIT post-processor, classifier-feedback) maintains its own `read_seq_<consumer>` advanced after consuming each entry. Consumers acquire-load `ring->write_seq` to discover new entries, then process up to a configurable batch size before yielding. Consumers are responsible for releasing entries back to the producer by advancing `read_seq_<consumer>`; a consumer that falls behind is its own problem — the producer drops, the slow consumer misses those events. This is the intended trade-off (telemetry loss > producer blocking).

**Coexistence with CUPTI.** RING_WRITE and CUPTI run in parallel. CUPTI submits aggregate launch stats to the kmod via ioctl nr 7 (256-launch batches); RING_WRITE delivers per-launch records to userspace consumers. The two paths never block each other. The kmod's per-tenant snapshot is populated by CUPTI's aggregate; the userspace observers (REMEMBER, AUDIT, classifier-feedback) consume RING_WRITE's per-launch stream.

**Migration path.** may13's existing `cipher_ring_write` inline + ring struct + consumer thread skeletons port into `cipher_rt_phase4/cipher_rt_ring.{c,h}`. The Stage 1/2 consumer thread spawn predicate is replaced from "any observer env var set" (may13's lazy-spawn) to "if any of REMEMBER, AUDIT, or classifier-feedback consumers are enabled" (explicit env per consumer). The producer side wires into `cipher_inject.c`'s GOT-patched cuLaunchKernel intercept at the post-dispatch site.

**Engineering estimate.** 8-10 days (Weeks 9-10 in §7). Port may13 ring (~100 LOC mechanical) + wire producer in `cipher_inject.c` (~50 LOC) + write the AUDIT consumer (~80 LOC) + write the classifier-feedback consumer (~100 LOC) + N=100 contention validation harness (~200 LOC). REMEMBER consumer is part of the Koopman tier in Weeks 11-12.

---

<a id="section-5"></a>
## SECTION 5 — WORKLOAD IMPACT VERIFICATION

For each regime, the unified design's path to all three goals. Where any goal cannot be achieved without compromise, the compromise is surfaced.

### 5.1 — Regime 1: Compute-bound prefill

**Goal — MFU 85%+ (G1 gate).**

**Current measured baseline:** CP 3.3 measured 745 TFLOPS on a compute-bound workload at 59.8% MFU; device sum 887 TFLOPS, NVML cross-check 916. Power-capped at 700 W → ~660 TFLOPS sustained ceiling = ~67% MFU. The 745 figure was likely a brief uncapped burst or proxy-counter over-read.

**Mechanism in unified runtime:**

1. CLASSIFY detects LARGE_GEMM kernel class.
2. SENSE detects BATCH or AGENT-prefill band.
3. ORACLE permits SUBSTITUTE (steady phase, high confidence, registry hit on common shapes).
4. SUBSTITUTE routes to Marlin INT4 lane (if weight-quantized model AND batch ≥ 8) OR cuBLAS FP16 partition-aware.
5. PREDICT proactively cuMemPrefetchAsync's hot weight regions to L2.
6. THERMOSTAT keeps clock at max (no DVFS cut for compute-bound).
7. VOLT engages high-clock lock (default 1980 MHz, not the 1200 sweet-spot).
8. AUDIT/TRACE/DETERMINISM observe post-launch.

**Timeline reality check — what v1 (5 weeks) does and what comes after:**

The 85% gate is the right product target, but it does NOT land in the 5-week v1 window. Honest decomposition:

| Lever | Status | In v1? | Lift contribution |
|---|---|---|---|
| (a) Marlin INT4 quantization | already ships in cipher_rt_phase4 | already counted in v1 | already in baseline measurement |
| (b) TMA + thread-block-cluster attention kernels (Depth Win #4) | **never built**; substantial CUDA work; requires Hopper-specific kernel programming | **NO — post-CP-5.5 program** | ~5-10 points on WL13/WL08 |
| (c) L2-pinning via `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` per tenant | **requires v1.5 kmod ABI extension** (Section 3.5 future NR) | **NO — v1.5** | ~5 points |
| L2 prefetch via `cuMemPrefetchAsync` (Depth Win #2) | userspace, smaller lift than (c) | **partial in v1** (Week 4 PREDICT port is deferred to v1.5) | ~2-3 points |

**v1's MFU deliverable (5 weeks):** integrate the classifier+dispatch+observers, ship the substrate that makes the current Marlin-on-prefill actuator workload-class-correct, and **measure the per-WL MFU baseline at CP 5.5**. The number CP 5.5 produces will be **the honest current MFU per-WL across WL01-24** — likely 60-75% on compute-bound prefill WLs where Marlin engages, 15-40% on decode WLs.

**Path to 85% — the depth-win program, named, scoped, AFTER CP 5.5:**

| Program item | Owner | Calendar | Output |
|---|---|---|---|
| D-W4 TMA + thread-block-cluster attention | external CUDA engagement (Song Han scope §6 of PHASE_5_ARCHITECTURE_REVISION.md) | post-CP-5.5; 4-8 weeks | +5-10 points WL13/WL08 |
| D-W3 L2 budget enforcement (new kmod NR) | kmod owner + rt | post-CP-5.5; 2-3 weeks | +5 points L2-bound WLs |
| D-W2 L2 prefetch via PREDICT | engineering | v1.5; 2 weeks | +2-3 points |
| Power-cap raise to enable closer to 989 TFLOPS theoretical | deployment / operator | deployment-time | unlocks remainder |

**Risk:**

- **R-1.1** [HIGH IMPACT]: The 85% gate is **post-CP-5.5**, not a v1 deliverable. v1 measures the per-WL baseline; the depth-win program delivers 85% on the declared subset. **DD-honest claim: "v1 delivers the substrate; 85% on compute-bound WLs is the depth-win program goal, named and scoped separately."**
- **R-1.2** [MEDIUM]: 745 TFLOPS proxy-counter reading was likely over-counted vs HFU denominator. Honest sustained ceiling ~660 TFLOPS / 67% on H100 at 700W. The 85% gate requires either (a) raised power-cap deployment, or (b) re-base to "approach the 67% ceiling under power-cap" as the customer-facing claim.
- **R-1.3** [HIGH IMPACT]: 85% is a per-WL claim, not aggregate. Decode WLs (WL01/WL02) are roofline-bound and cannot reach 85% — physics. **Per-WL measurement set must report which WLs hit gate and which do not, and the customer-facing claim must scope to "on declared compute-bound WLs."**

**Verification in unified runtime:**

- Per-launch FLOP attribution via kmod nr 11/12 (CP 3.3 telemetry).
- Per-tenant MFU reported in `/proc/cipher/stats` MFU column.
- Aggregate device MFU via cipher-exporter `/metrics` Prometheus `cipher_tenant_mfu_pct{tenant=X}`.
- Per-WL gate measurement script: `cipher_measurement/mfu_per_tenant.sh` (already written, never run on full WL set).

**Verdict for Regime 1:** 85% MFU on declared compute-bound WLs is the **post-CP-5.5 depth-win-program goal**, not a v1 deliverable. v1's job is to ship the substrate and measure the per-WL baseline. The depth-win program (TMA clusters, L2 budget enforcement, PREDICT L2 prefetch) closes the gap from baseline to 85% on the named WL subset. **Surfaced explicitly: the document's earlier "v1 reaches 85%" framing is corrected here.**

### 5.2 — Regime 2: Memory-bound decode

**Goal — TPW ≥ 2× (composed).**

**Current measured:**
- DVFS-alone under vLLM at 1200 MHz: +13.9% tok/W (1.14×, FUTURE_SCOPE/A Phase 3.5).
- Cross-tenant batching (CIPHER-native path, not vLLM-composed): 3.06× (Mistral-7B N=4), 3.30× (TinyLlama N=8), 5.98× (N=16). The "3.6×" lives at N≈9-10 on the curve.
- Composed in CP 5.6 V_C: 1.54× (the F1-corrected number; the 3.617× was timing-only on the broken path).

**Mechanism in unified runtime:**

1. CLASSIFY detects SMALL_GEMM/MATVEC/ATTN_DECODE.
2. SENSE detects HUMAN-decode or AGENT-decode.
3. ORACLE detects steady phase, but Marlin lane is GATED OFF for B=1 decode (regresses per CP 2.4 §168).
4. SUBSTITUTE routes to cuBLAS lane (FP16) OR attention substrate lane.
5. SUSTAIN monitors KV pressure slope.
6. **kvdedup (kmod L3)** shares prefix KV across tenants.
7. **POOL batched executor (libcipher_v2)** fuses N tenants' decode steps — this is where the 3-6× lift comes from at scale.
8. **VOLT** engages 1200 MHz lock — +14% sweet spot.
9. AUDIT/TRACE/DETERMINISM observe.

**Path to 2× composed:**

- DVFS lane: +14%.
- Cross-tenant POOL batching: 3-6× when ≥ N=4 tenants share the GPU.
- Multiplicative if independent: 1.14 × 3.0 = ~3.4× at N=4.
- **Honest claim: ≥ 2× is achievable when ≥ N≈4 same-model tenants share the GPU. Single-instance decode TPW is ≤ 1.14× — DVFS-only.**

**Risk:**

- **R-2.1** [HIGH]: "Single-instance TPW 2× under vLLM" is NOT achievable. The compute-acceleration actuators (Marlin, persistent dispatch) have no interception surface under vLLM. Single-instance ≤ 1.14× is the honest ceiling. **The 2× claim is conditioned on N≥4 tenant density, not single stream.**
- **R-2.2** [MEDIUM]: The POOL batched executor's 3-6× was measured on CIPHER-native execution, not under vLLM. Phase 4 of FUTURE_SCOPE/A measures the composed number — that's the headline experiment.

**Verification:**

- Per-tenant tok/W in `/proc/cipher/stats` (CP 3.3 substrate telemetry + power from NVML).
- N-tenant scaling curve via `density_pack.sh`.
- vLLM-composed measurement: FUTURE_SCOPE/A Phase 4 (next checkpoint).

**Verdict for Regime 2:** TPW ≥ 2× target achievable as a multi-tenant composed claim (N ≥ 4), NOT as a single-stream claim. **Surfaced explicitly: single-instance under vLLM is +14% DVFS-only; the headline lives at multi-tenant density.**

### 5.3 — Regime 3: Multi-tenant agentic (N=100 agents)

**Goal — 100 agents per GPU at maintained per-tenant performance (G3 gate).**

**Current measured:**
- Track 2: 76% HBM saved at Mistral-7B N=4. Asymptotic 95% at large N per the formula `savings(N) = N·W / (N·W + (N+1)·C)`.
- CP 5.4: 15 × 8-SM groups validated; isolation 15/15 PASS; two-clause disjointness invariant.
- Track 3: ~1.26 ms migration cost; ~70% POOL stranding reduction under churn.
- kvdedup: cross-process verified on synthesized pages; never wired to live decode (T4.6.5 not done).

**Mechanism in unified runtime:**

1. Each agent process loads libcipher_rt via injection.
2. Each agent calls CIPHER_CP54_ALLOCATE with QoS=PARTITION (own 8-SM group) OR QoS=SHARED (pool member).
3. Weight arena (Track 2 SC5): N same-model agents IMPORT one producer's exported VMM POSIX fd. Weight HBM is 1 copy.
4. kvdedup (kmod nr — handled via cipher_kvdedup hook): shared system-prompt KV is stored once across all processes.
5. POOL batched executor (libcipher_v2) fuses concurrent decode steps for N pool members.
6. ARBITRATE (kmod cp54_sched): allocates partitions, runs do_exit reaper.
7. DSM (Track 3): migrates idle partitions to busy on opt-in.
8. SHIELD enforces per-tenant SLO priority via stream priority.
9. FAIRNESS per-tenant quota.
10. RECEIPT/AUDIT per-session billing.

**Path to 100 tenants:**

- 15 × 8-SM PARTITION slots + 1 POOL slot (the singleton) → 15 PARTITION tenants + N POOL members (no per-member SM cost).
- POOL handles 85-100 of the 100 agents (low-latency-tolerant ones, batched).
- 15 PARTITION slots for latency-sensitive agents (e.g., interactive tool-using agents).
- Weight sharing → all 100 agents share 1 weight copy → ~76-95% HBM saved.
- kvdedup → all 100 agents share their common prefix KV.

**Risk:**

- **R-3.1** [HIGH]: 100-tenant integration measurement (CP 5.5) has not run. Track 2 verified at N=4; extrapolating to N=100 carries scale risk (kmod hashtable contention, fd custodian fan-out, weight-arena 16-slot limit, kvdedup table size).
- **R-3.2** [MEDIUM]: The 16-arena hard limit (CIPHER_WA_MAX_ARENAS) — 100 same-model agents share 1 arena, OK. 16+ DIFFERENT models is the cap. v2 raises the cap.
- **R-3.3** [MEDIUM]: kvdedup never wired to live decode (T4.6.5 not done). The 100-tenant prefix-share story is **architectural — not measured.**

**Verification:**

- `density_pack.sh` 2→5→10→20→30→50→100 sweep on Nemotron Nano or Mistral.
- Per-tenant latency at p50/p99 reported per agent.
- Aggregate fleet tok/W.
- HBM utilization.

**Verdict for Regime 3:** 100-agent target is **structurally achievable** with current substrate (Track 2 + CP 5.4 + Track 3 + kvdedup) but **never measured at scale**. **Surfaced explicitly: the headline density figure is real at N=4; CP 5.5 is the next-step measurement that validates at N=100.**

### 5.4 — Regime 4: Tool-call idle filling

**Goal — zero GPU-idle-during-tools waste.**

**Current measured:**
- Track 3 DSM: ~1.26ms migration, opt-in, ~70% POOL-stranding reduction.
- LOOP op (may13): runaway detector, FIRES (runaway_count=7 in stress2).
- PIPELINE op (may13): multi-agent session correlation, FIRES (edge_count=10).

**Mechanism:**

1. SENSE detects AGENT band per session.
2. LOOP/PIPELINE detect tool-call boundary (signaled by long inter-launch gap, kernel inactivity).
3. DSM (Track 3 nr 16-20) — agent on tool-call subscribes-migratable, kmod PROPOSEs migration of its 8-SM-group to a busy agent.
4. Idle partition reallocated; on tool return, agent migrates back.

**Risk:**

- **R-4.1** [MEDIUM]: Tool-call boundary detection is not yet implemented at op level. LOOP/PIPELINE detect agentic activity broadly; specific tool-call-idle signaling requires app-level cooperation OR inference from kernel-inactivity (heuristic).
- **R-4.2** [LOW]: HIBERNATE op (idle SM power-gating) is PARTIAL (gated off on Lambda pods). v1.5 wiring brings it online for tool-call power savings.

**Verification:**

- Agentic benchmark with mock tool-call delays (sleep 100-500ms mid-iteration).
- Measure SM utilization gap before/after DSM enablement.
- Per-agent tail latency unchanged.

**Verdict for Regime 4:** Tool-call idle filling is **substrate-ready** (DSM works, LOOP/PIPELINE detect boundaries) but **needs the op-level "tool-call detected" signal**. Surfaced as Section 7 Week 3 deliverable.

### 5.5 — Regime 5: Mixed real production

**Goal — all three product goals simultaneously on the same substrate.**

The unified runtime, by construction:
- Classifies each kernel per-launch → routes to regime-appropriate actuator.
- Maintains per-tenant snapshot in kmod → every actuator reads consistent state.
- Composes the regime-specific actuators without conflict (Marlin GATED OFF in partitioned tenants; DVFS engages on both partitioned and full-GPU contexts; weight sharing transparent to dispatch).

**Risk:**

- **R-5.1** [HIGH]: The mixed-regime measurement has never been run. CP 5.5 is exactly this. **The marvel hasn't been measured.**
- **R-5.2** [MEDIUM]: Cross-regime actuator interactions — e.g., what happens when Tenant A is in prefill (wants full-GPU Marlin) at the same instant Tenant B is in decode (wants partition + DVFS cut)? Architecture answer: Marlin pins to primary ctx; Tenant B's partition has its own clock domain — they coexist by spatial separation. **Validated by CP 5.4 isolation test (15/15) but not under heterogeneous workload.**

**Verification:** CP 5.5 100-tenant heterogeneous benchmark.

**Verdict for Regime 5:** Achievable in principle by construction (every per-regime mechanism works alone; the architecture composes them spatially). **Measurement at the marvel scale is exactly what CP 5.5 delivers — the headline experiment.**

### 5.6 — The "no compromise" table

**Framing — the caveats ARE the architecture, not concessions.** CIPHER does not claim every metric on every regime. CIPHER **classifies the workload per kernel and applies the metric and lever that fit that regime.** That selection IS the architectural product — it is what the dispatcher does. The "regime-specific" framing in this table is therefore a *description of which regime each goal lives in*, not a list of things sacrificed. The marvel is exactly this dispatch: the right lever, the right metric, the right tenant, the right kernel, at runtime.

A universal claim ("85% MFU on everything, 2× TPW on everything") would be the compromise — because it would be false on at least one regime and DD would find it. The dispatch architecture lets every claim be true on the regime where it lives, and silent on the regime where it doesn't apply. That is the difference between a substrate and a marketing slide.

| Regime | Goal | Lever (what fires) | Where this goal LIVES | Where it does NOT (silent) |
|---|---|---|---|---|
| 1. Compute-bound prefill (LARGE_GEMM, B≥8) | MFU on the per-WL gate, target 85% **post-CP-5.5 depth-win program** (not v1; §5.1 timeline) | Marlin INT4 (full-GPU primary ctx) + L2 pinning + TMA clusters (post-CP-5.5) | Where CLASSIFY=LARGE_GEMM and tenant is full-GPU | Decode WLs (Regime 2) — physics says no |
| 2. Memory-bound decode (B=1, SMALL_GEMM / MATVEC / ATTN_DECODE) | TPW lift, target ≥2× **composed at N ≥ 4 tenant density** | cuBLAS-FP16 + POOL cross-tenant batching (3-6×) + fleet-policy DVFS (+14% at 1200 MHz) | Where CLASSIFY=decode AND multi-tenant present | Single-stream decode under vLLM (≤1.14× DVFS-only — by design, not failure) |
| 3. Multi-tenant agentic (N≈100 agents) | density: 100 agents/GPU at maintained per-agent p99 | weight sharing (Track 2, 76% HBM saved at N=4, 95% asymptote) + KV-prefix dedup + SM partitioning (15 × 8-SM) + DSM | Where SENSE=AGENT and tenant count > 1 | Single-stream (Regime 2 alone) |
| 4. Tool-call idle | zero GPU-idle-during-tools waste | DSM live migration + LOOP/PIPELINE tool-call detection | Where SENSE=AGENT and tool-call boundary detected | Continuous compute (Regime 1) |
| 5. Mixed real production | all four goals on the SAME GPU SAME workload | per-kernel CLASSIFY → per-tenant ARBITRATE → per-regime SUBSTITUTE → fleet-policy DVFS → cross-tenant density | by construction (every per-regime mechanism composes spatially) | Never silent — this regime is the proof |

**This is the no-compromise architecture.** Every goal lives where it physically applies. The dispatcher decides. CP 5.5 is the measurement that proves Regime 5 (mixed real) is structurally achievable.

What v1 ships in 5 weeks: the substrate, the dispatch, the per-regime routing, the baseline measurement. What v1.5/depth-win program ships post-CP-5.5: the closing-the-gap programs (TMA clusters, L2 budget enforcement, partition-aware Marlin) named and scoped in Section 8. **Honest scope, complete architecture, no caveat hidden.**

---

<a id="section-6"></a>
## SECTION 6 — NO-COMPROMISE CONSTRAINT VERIFICATION

### C1 — Three product goals achievable simultaneously

**Satisfied** with explicit scope:
- MFU 85% on compute-bound WLs (per-WL, not aggregate).
- TPW ≥ 2× at multi-tenant density (not single-stream).
- 100 tenants per GPU (CP 5.5 measures).

The three goals do not conflict in the unified architecture because the actuators that serve each are regime-disjoint:
- MFU lever (Marlin/cuBLAS-FP16 with L2 pinning) fires on prefill kernels.
- TPW lever (DVFS + POOL batching) fires on decode kernels.
- Density lever (weight sharing + partition) fires across tenants.

These coexist on the same GPU because the dispatch classifies per-kernel and routes per-regime.

### C2 — vLLM transparency preserved

**Satisfied.** Phase 3 measured -0.39% in graph mode. The unified runtime does not change the LD_PRELOAD-free deploy invariant — every new op ports through the InitializeInjection chain, none of which adds hot-path latency beyond the existing CLASSIFY+ORACLE budget (~12 ns).

### C3 — 33 ops have defined homes

**Satisfied.** Section 2.3 places every op in a tier. Section 4.4 places every op in a path (hot / setup / async / periodic). The learning tier (REMEMBER, SPECULATE, ADAPT) is deferred to v2 — but their place in the architecture is reserved (Section 4.4 row: "v2 learning tier"). They are not orphaned.

### C4 — Correctness gates preserved

**Satisfied.** Existing gates:
- max_diff = 0 vs ground truth (Track 2 SC6: 40 bit-identical forwards).
- 100% intercept rate (CP 5.4: 15/15 isolation).
- Bit-identical multi-tenant output (Track 2 SC4: tier-1 fingerprint mismatch detected pre-commit).

The integration plan (Section 7) gates every week's port on these tests passing.

### C5 — kmod ABI stability

**Satisfied.** NRs 1-24 frozen and additive-only. v1 unification does not require any new ioctl. v1.5/v2 extensions (Section 3.5) take fresh NRs 25+.

### C6 — Additive integration (no big-bang replacement)

**Satisfied.** Each week's port is additive:
- Week 1: classifier ops compile into libcipher_rt build, but do not fire.
- Week 2: classifier wired to hot-path, but only in observe-only mode (no actuator routing change).
- Week 3: dispatch routing goes live, but fallback to existing PASS_THROUGH if classifier returns UNKNOWN.
- Week 4: observability ports.
- Week 5: CP 5.5 measurement.

At any point in this sequence, the deployed runtime remains functional. Anchor snapshots taken every week; rollback returns to the prior week's anchor.

---

<a id="section-7"></a>
## SECTION 7 — INTEGRATION SEQUENCE (15-17 WEEK)

<!-- v1.2.3: §7 header extended from "12-14 WEEK" to "15-17 WEEK" per B.1 timeline extension. The v1.2.2 amendment subsection below is preserved historically and the v1.2.3 amendment subsection extends it. -->

> **Week schedule reconciled 2026-05-23 per v1.2.3 B.1 timeline extension** (W6 absorbs G1+G2+G5-audit carry; W7-9 absorbs G6+G10 alongside COMMIT; W10-12 absorbs G3+G4+G5 alongside RING_WRITE; W13-14 holds Koopman tier + G12 keying; W15-17 hosts the hybrid workload-class + heterogeneous-model CP 5.5). Prior schedule: W7-8 COMMIT, W9-10 RING_WRITE, W11-12 Koopman, W13-14 CP 5.5 (v1.2.2 reconciled 2026-05-21).

### v1.2.3 amendment — B.1 timeline extension (architecture gap fold-in)

<!-- v1.2.3: new amendment subsection per B.1 plan. Folds six gaps from WEEK_6_ARCHITECTURE_GAP_AUDIT.md into the v1.2.2 track structure. -->

§7 extended from v1.2.2's 12-14-week sequence to **15-17 weeks** per:

- **WEEK_6_ARCHITECTURE_GAP_AUDIT.md** (cipher-fusion-evidence `07c2212`, doc md5 `c5d2d4ad`, 640 lines, 2026-05-23) surfaced six load-bearing gaps unscheduled by v1.2.2 §7: G1 (CP54 cap 64), G2 (weight-arena cap 16), G3 (KV-dedup content-only hash → silent cross-model corruption), G4 (Marlin single-model weight kit), G5 (VA pool 80 GiB per-process), G6 (no kmod-resident AUDIT chain), plus G10 (CIPHER_REGISTER_MODEL ABI) and G12 (Koopman registry model-keying).
- **B.1 decision (2026-05-23):** re-sequence not redesign. Hold Koopman tier in v1 (reverses no prior adjudication; A1 in force). Fold gaps into existing track structure. Extend timeline 2-3 weeks. CP 5.5 reframes as hybrid workload-class + heterogeneous-model.
- Gap-to-week mapping (full detail per-week below):
  - **W6** (in-flight + carry): G1 + G2 cap bumps (1.5 eng-days, kmod 0.5 ABI bump); G5 architecture audit (read-only research, 2-3h); May-13 POC reconstruction kickoff (parallel ~6-10 eng-days).
  - **W7-9** (was W7-8 COMMIT, extended one week): G6 kmod-resident AUDIT chain folds into the COMMIT contract; G10 CIPHER_REGISTER_MODEL ABI at NR 27.
  - **W10-12** (was W9-10 RING_WRITE, extended two weeks): G3 KV-dedup model-aware keying; G4 Marlin tenant-scoped weight kit; G5 VA pool implementation per W6 audit decision.
  - **W13-14** (was W11-12 Koopman tier, held one position): Koopman tier integration unchanged scope; G12 Koopman registry model-keying piggybacks on the recipe-table port.
  - **W15-17** (was W13-14 CP 5.5, extended one week): hybrid workload-class + model-architecture heterogeneous benchmark on full unified runtime.

Week-by-week breakdown under v1.2.3:

| week | scope | status |
|---|---|---|
| 1 | Compile-level classifier port + LP-7 struct collision fix + Cb.2 snapshot reserved-tail | **DONE** — anchor `fc8a9ae6` (week-1-step-1-lp7-rename) |
| 2 | Hot-path classifier wiring + LP-2 attn trampoline refactor | **DONE** — W2 closeouts (Steps 1-6) landed Week 2 cycle |
| 3 | Dispatch routing goes live | **DONE** — anchor `79c1b4f9` (week-3-step-4-opt2a-dispatch-live-sense) |
| 4 | Observability integration + LP-8 retirement + Prometheus extension | **DONE** — W4 scope-lock at `850bd8b`; cipher_kmod tag `week-4-step-4-lp8-retired` (`158ad96`) |
| 5 | KV-dedup live wire + v1 substrate consolidation | **DONE** — `cipher_rt_phase4` `ec0e005` (week-5-complete); cipher-fusion-evidence post-close `4302079` (post-Step 1b: `cipher_kv_bridge.so` `c04b0c39` → `f041789c`) |
| **6** | **DONE** (2026-05-23) — in-flight: KV-dedup auto-trigger (cd8c826f → 438e4023, 2026-05-21); bench harness audit + rewrite + Option 1 redo (`cc913a6`); architecture gap audit (`07c2212`). **Carry closed:** G1+G2 kmod cap bumps (cipher_kmod `c4e2d6f` tag `week-6-step-g1-g2-cap-bump` + cipher-fusion-evidence `751c6b8`; ko md5 `2f294edf`, ABI 0.5.0); G5 path-a verified (cipher-fusion-evidence `35f9b6c`); May-13 POC reconstruction plan (cipher-fusion-evidence `970694b`). **W7-9 scope-lock landed** at cipher-fusion-evidence following this row update — 5-step implementation sequence (G10 ABI → COMMIT primitive → G6 AUDIT chain → 21 overlay-ops port → N=128 soak), ~11 eng-days across W7-W9. | v1.2.3 |
| **7-9** | **COMMIT atomic state-transition primitive build + G6 kmod-resident AUDIT chain + G10 CIPHER_REGISTER_MODEL ABI (NR 27)** *(extended from W7-8 to W7-9 per B.1)* | v1.2.3 |
| **10-12** | **RING_WRITE lock-free inline telemetry substrate build + G3 KV-dedup model-aware keying + G4 Marlin tenant-scoped weight kit + G5 VA pool implementation per W6 audit decision** *(extended from W9-10 to W10-12 per B.1)* | v1.2.3 |
| **13-14** | **DONE** (2026-05-24) — Koopman tier integration (EDMD pipeline real-input fix + six substrate fixes: H2 CGS2 / GPU upload / POWER_ITERS / deterministic cuSOLVER SVD / β OOD detector / Tikhonov α=0.01·σ₁²) + recipe registry narrow-domain seeding + SUBSTITUTE-Koopman lane validation (S3.C two-pass full-substrate harness on real TinyLlama LM head: fire mode top-1 = 0.9000 reproduces D1.3 architectural ceiling, passthrough mode KL = 0.0 byte-identical) + G12 Koopman registry model-keying (`cipher_rt_recipe_model_key()` at `cipher_recipes.cpp:557-575`) + N=128 30-min soak both modes ALL GATES PASS (Mode A 11.73 M/s, Mode B 11.22 M/s, fairness ≥ 0.879, 0 incoherent of 5.3T reads). Two-mode KL gate addendum (option α adjudication; scope-lock preserved) lands the v1 contract per plan §1 line 103. Substrate exits at cipher_rt_phase4 `25970f3` alias `week-14-complete` + `week-13-14-complete`; libcipher_rt.so md5 `097cf8d9`; cipher_kmod `8c643fc` unchanged. Closeout doc `WEEK_13_14_COMPLETE.md`. Honest residue: substrate ships, real-workload tok/s measurement campaign queued (memory `w14-step-3-followup-mistral-tok-s`). | v1.2.3 |
| **15-17** | **CP 5.5 hybrid headline benchmark on full unified runtime** — workload-class heterogeneity (5 prefill + 80 decode + 15 burst) **+ model-architecture heterogeneity (≥5 different model families coexisting: Mistral-7B + Qwen-7B + Llama-3-8B + 2 SLMs)** at the highest density the closed gaps permit. 24-hour soak. Per-tenant FAIRNESS + CARBON + RECEIPT cryptographic billing chain verified. *(extended from W13-14 to W15-17 per B.1)* | v1.2.3 |

### v1.2.2 amendment — Week 5 / CP 5.5 schedule (preserved historically — superseded by v1.2.3 amendment above)

<!-- v1.2.3: v1.2.2 amendment block preserved for audit trail. Schedule supplanted by the v1.2.3 amendment above; this block is the historical record of the v1.2.2 schedule. -->

§7 was extended from v1.2.1's 5-week sequence to 12-14 weeks per:

- **ADJUDICATION 1** — Option B scope (30 of 33 ops in v1).
- **ADJUDICATION 2** — COMMIT primitive build (Weeks 7-8).
- **ADJUDICATION 3** — RING_WRITE substrate build (Weeks 9-10).
- Koopman tier integration (Weeks 11-12).
- **CP 5.5 final benchmark moved from v1.2.1's Week 5 to Weeks 13-14**, because CP 5.5 ships on the full 30-of-33-ops surface (which Weeks 7-12 build).

**Week 5 (originally v1.2.1's CP 5.5) is repurposed as the KV-dedup live wire and v1 substrate consolidation step.** The CP-5.5 content that was at Week 5 in v1.2.1 now lives at Weeks 13-14 in v1.2.2 / Weeks 15-17 in v1.2.3 (verbatim retention + heterogeneous-model fold-in; see W15-17 below).

---

Each week: goal, files moved, compile verification, behavioral test, rollback path, risk register.

### Week 1 — Compile-level classifier port + struct collision fix + snapshot reserved-tail bump — **DONE** (`fc8a9ae6` — week-1-step-1-lp7-rename)

<!-- v1.2.3: DONE marker added per B.1 §7 re-sequence. Anchor `fc8a9ae6` is the cipher-fusion-evidence tag week-1-step-1-lp7-rename; the rename is mechanical and was completed on schedule with W1 regression PASS within ±3% per the original behavioral test gate. -->

<!-- v1.2: Week 1 augmented per Wave 5 §5.5 — adds LP-7 CipherKernelEntry rename (compile blocker) and snapshot reserved-tail extension (Cb.2 lossless seam). -->

**Goal:** Classifier headers / sources compile into the rt_phase4 build. Do not fire yet. **Plus the LP-7 struct-collision fix (compile blocker; must precede any port that pulls both headers) and the Cb.2 snapshot reserved-tail bump (additive without new NRs).**

**Source-of-truth note:** Port the **TOP-LEVEL** dispatch and oracle files, NOT the `src/` copies (which are silently excluded by may13/Makefile L29 and differ from the live versions). Delete the `.bak_1431` while moving — these are dead artifacts that will cause confusion if carried.

**LP-7 struct collision fix (REQUIRED first, before any port that touches both headers).** Per Wave 1 F1-STRUCT-COLLISION and Wave 5 LP-7: `struct CipherKernelEntry` is defined with different fields in two headers (`include/cipher_kernel_table.h:53-64` vs `include/cipher_param_recovery.h:30-38`). Both cannot be `#include`d in the same TU. Rename one to break the cycle before any other Week 1 file touches both:
- `include/cipher_kernel_table.h::struct CipherKernelEntry` → `struct CipherKtEntry` (9 caller sites — Wave 1 §DEPENDENCIES).
- `include/cipher_param_recovery.h::struct CipherKernelEntry` → `struct CipherParamEntry` (6 caller sites).
- One-pass `sed` across both headers and all callers; mechanical rename.

**Cb.2 snapshot reserved-tail bump (additive, no new NR).** Per §3.5's seam note: add classifier-side fields to `cipher_internal.h::struct cipher_pid_stats` AND the userspace mirror `cipher_ioctl.h::struct cipher_tenant_snapshot_user`, all in the `reserved[16]` tail. Fields per §3.5 table: `recommended_sm_count`, `slo_priority`, `session_band`, `tenant_billing_class`, plus padding; remaining `reserved[12]`. Size invariant: `sizeof(struct cipher_tenant_snapshot_user) == 336` bytes — verified by T-W1.3.

**Files ported:**
- `cipher-may13-evidence/cipher_dispatch.cpp` (TOP-LEVEL, 543 LOC) → `cipher_rt_phase4/cipher_rt_dispatch.cpp` (declarations only — no hot-path entry yet)
- `cipher-may13-evidence/cipher_oracle.cpp` (TOP-LEVEL) → `cipher_rt_phase4/cipher_rt_oracle.cpp`
- `cipher-may13-evidence/include/cipher_classify.hpp` → `cipher_rt_phase4/cipher_rt_classify.h`
- `cipher-may13-evidence/src/cipher_recipes.cpp` (32-shape registry) → `cipher_rt_phase4/cipher_rt_recipes.cpp`
- `cipher-may13-evidence/src/cipher_sense.cpp` → `cipher_rt_phase4/cipher_rt_sense.cpp`
- `cipher-may13-evidence/src/cipher_structural_lookup.cpp` (L3.8 fast bypass) → `cipher_rt_phase4/cipher_rt_structural_lookup.cpp`
- `cipher-may13-evidence/include/cipher_recipes.h`, `cipher_classify.hpp`, `cipher_oracle.h`, `cipher_sense.h`, `cipher_structural_lookup.h` → `cipher_rt_phase4/` (header port)

**Build-system change:** Add a new substrate file `cipher_rt_classify_substrate.cpp` mirroring the matmul/attn dispatch pattern:
```c
typedef struct { const char *name; int priority; enum cipher_rt_classify_result (*maybe_classify)(...); } cipher_rt_classifier_t;
int cipher_rt_classify_register(const cipher_rt_classifier_t *cls);
enum cipher_rt_classify_result cipher_rt_classify_dispatch(const cipher_rt_classify_call *call, cipher_rt_classify_out *out);
```

**Makefile changes:** Add 7 new objects to `OBJS`; add per-source build rules. No nvcc added (no .cu files ported in Week 1).

**Compile verification:** `make clean && make` produces `libcipher_rt.so` that links cleanly.

**Behavioral test:** No behavior change. Run the following concrete regression runners (no perf delta beyond ±3% expected):

- **W1 regression (multi-tenant substrate sweep)** — `/home/ubuntu/cipher_measurement/density_pack.sh` (sweep N=2/5/10/20/30/50/100 with per-tenant MFU gate); for a fixed-N regression that matches a 15- or 16-slot scenario, use `/home/ubuntu/cipher-fusion-evidence/cp_5_4/step1_6/cp54_s16_orchestrator.py` (16-slot, two-clause disjointness + KL gates).
- **Isolation 15/15** — `/home/ubuntu/cipher-fusion-evidence/cp_5_4/step1_3/cp54_isolation_test` (compiled binary at md5 prefix `8f395f7e`; six sub-tests, 15 sub-assertions, requires `/dev/cipher` loaded).
- **Track 2 SC6 bit-identical forward** — `/home/ubuntu/cipher-fusion-evidence/phase_c/sc6_run.py` driver + `sc6_models.py` + `sc6_consumer.py` + `sc6_independent_tenant.py` + `sc6_aggregate.py`; baseline JSON at `phase_c/sc6_Mistral-7B_independent_result.json` (5-tenant independent loaded=73,332 MiB, PASS=true).

<!-- v1.2.1: behavioral-test step concretized per PRE_WEEK_1_ADJUDICATION_CLOSURE.md Part 3 C3. The earlier abstract phrasing ("Run W1 regression") was paraphrased by the pre-flight task brief as "Phase 5 baseline 15-tenant FAIRNESS run", a runner name that does not exist in the repository. The bullets above name the actual runners that exercise the substrate behavior the abstract phrase intended; future pre-flight prompts and operator-runbooks should cite these paths directly. -->

**Rollback path:** Revert Makefile change; rebuild from anchor `83afd1ca`. Fallback `libcipher_rt.so.pre_week1` preserved.

**Risk register:**
- R-W1.1 [LOW]: ABI clash on snapshot struct (handled per Section 3.1).
- R-W1.2 [LOW]: nvcc not added at W1 — Koopman .cu files not ported in W1. Per v1.2.2 ADJUDICATION A1, narrow-domain Koopman ports land at Weeks 13-14 (§7 line 1258); W13-14 scope-lock R-W13.2 plans the nvcc rule addition.
- R-W1.3 [MEDIUM]: Symbol naming convention enforcement — ensure no may13 symbol shadows rt_phase4 symbol.

### Week 2 — Hot-path classifier wiring + LP-2 attn trampoline refactor — **DONE**

<!-- v1.2.3: DONE marker added per B.1 §7 re-sequence. W2 closeout landed across W2 Step 1-6 closeouts in 2026-05 cycle; CLASSIFY + SENSE + ORACLE telemetry firing per `/proc/cipher/classify_stats`, cache hit-rate gate passed. -->

<!-- v1.2: Week 2 augmented per Wave 5 §5.5 — adds LP-2 attn trampoline refactor (required to unblock Week 3 attn lane in v1.5 future). -->

**Goal:** CLASSIFY + SENSE + ORACLE fire on every kernel launch. Telemetry only — no actuator routing change. **Plus the LP-2 attn trampoline refactor (required so the attn lane is unblockable when the first attn substitute actuator lands in v1.5).**

**LP-2 attn trampoline refactor.** Per Wave 5 LP-2 (blocking for any future attn substitute actuator): modify `cipher_rt_attn_dispatch.cpp::flash_call`, `eff_call`, `cudnn_call` (L289, L324, L362) to branch on `route()`'s return value. New control flow:

```
cipher_rt_attn_result r = route(c);
if (r == CIPHER_RT_ATTN_HANDLED) {
    /* actuator stored substitute result into call.out_status_devptr */
    return; /* skip orig() */
}
return orig(q, k, v, ...);
```

Add field `void* out_status_devptr` to `cipher_rt_attn_call` (header L80+); actuators write the substitute result here when returning HANDLED. About 20 LOC across the 3 trampolines. The Track 2 SC6 bit-identical-attention gate (T-W2.4) is the regression check — for v1 (no attn substitute actuator yet), the refactor is a no-op behavioural change because no actuator returns HANDLED on attn.

**Hook point:** `cipher_inject.c`'s GOT-patched cuLaunchKernel intercept. Add a pre-launch hook:
```c
cipher_rt_classify_fingerprint_t fp = cipher_rt_classify(kernel_meta);
cipher_rt_oracle_decision_t dec = cipher_rt_oracle_check(fp, snapshot);
// Decision NOT yet consumed — log only.
cipher_rt_classify_record(fp, dec);
```

**Behavioral test:**
- W1 regression PASS (no perf change beyond ±3%).
- `/proc/cipher/classify_stats` (new node) reports non-zero classification counts per kernel launch.
- ORACLE EMA stays in `STEADY` after warmup.
- CLASSIFY cache hit rate ≥ 95% after 10s steady-state.

**Rollback path:** Stub the pre-launch hook to `return CIPHER_PASS_THROUGH` — equivalent to Week 1 state. No anchor change needed for rollback.

**Risk register:**
<!-- v1.2: R-W2.1 budget corrected per Wave 5 NF-5.1 / LP-14 — 12 ns is unsupportable; honest budget is 100-200 ns/launch, alarm at 500 ns. -->
- R-W2.1 [MEDIUM]: Hot-path latency spike — measure with cycle counter. **Honest budget is 100-200 ns/launch (Wave 1 control-flow analysis + cipher_classify.hpp "~160 ns cache-hit" header note), accept 1-2% per-tenant overhead as primary case.** Alarm threshold: 99th-percentile > 500 ns/launch.
- R-W2.2 [MEDIUM]: Per-tenant cache contention — CLASSIFY uses 512-slot lock-free cache; verify under contention with 33-thread test (existing harness `cipher_test_phase4_partition_contention`).

### Week 3 — Dispatch routing goes live — **DONE** (`79c1b4f9` — week-3-step-4-opt2a-dispatch-live-sense)

<!-- v1.2.3: DONE marker added per B.1 §7 re-sequence. Anchor `79c1b4f9` is the cipher-fusion-evidence tag week-3-step-4-opt2a-dispatch-live-sense; Step 4 Option II-a (SENSE bridge + DSM_PROPOSE ioctl nr 26) landed; dispatch routing live with PASS_THROUGH fallback retained as Week 3 rollback gate. -->

**Goal:** Classifier output drives actuator selection. Per-regime routing per Section 4.3.

**Changes:**
- Dispatch table from Section 4.5 implemented in `cipher_rt_dispatch.cpp`.
- For each kernel: CLASSIFY → ORACLE → SUBSTITUTE-table lookup → route to {Marlin, cuBLAS shim, attn dispatch, PASS_THROUGH}.
- Marlin lane GATED on (kernel_class == LARGE_GEMM) AND (tenant has full-GPU primary ctx, not partitioned).
- VOLT engagement: trigger 1200 MHz lock on detect-decode-band.
- SENSE phase transitions trigger DSM PROPOSE for tool-idle detection.

**Behavioral test:**
- Regime 1: WL03 prefill — measure MFU; expect ≥ 60% baseline approach toward 85% with Marlin engaged.
- Regime 2: WL01 decode — measure tok/W; expect +14% with DVFS engaged.
- Regime 3: WL05 multi-tenant — measure isolation, density.
- Regime 5: Heterogeneous mini-benchmark (1 prefill + 4 decode tenants concurrent) — verify all four work simultaneously.
- W1 regression PASS.
- Track 2 SC6 PASS.

**Rollback path:** Configuration env var `CIPHER_DISPATCH_LIVE=0` falls back to all-PASS_THROUGH (Week 2 state). Anchor `libcipher_rt.so.week3_pre` preserved.

**Risk register:**
- R-W3.1 [HIGH]: Mis-routing — a decode kernel routed to Marlin would regress. Mitigated by ORACLE gate + explicit B<8 filter in dispatch table.
- R-W3.2 [HIGH]: DVFS lock interaction with concurrent prefill tenant — must verify VOLT clock-lock doesn't degrade compute-bound tenant. Test in mixed regime.
- R-W3.3 [MEDIUM]: Tool-call detection heuristic false positives — LOOP/PIPELINE may signal too aggressively; DSM rate-limit (10s/tenant) prevents thrashing.

### Week 4 — Observability integration + LP-8 retirement + Prometheus exporter — **DONE** (`850bd8b` W4 scope-lock; `cipher_kmod` tag `week-4-step-4-lp8-retired` at `158ad96`; W4 closeout `9230b5c`)

<!-- v1.2.3: DONE marker added per B.1 §7 re-sequence. Week 4 complete per [[week4-complete]] memory: 6 implementation steps + closeout PASS; `cipher_rt_phase4` final `3c5ddaa`; `cipher_kmod` final `2fc70c3`; `cipher-may13-evidence` final `fc8a9ae`; loaded kmod srcversion `CECE94921DE1F43F04E452F`; +75 new T-symbols. -->

**Goal:** Port the observability tier. Per-tenant billing/fairness/audit becomes customer-facing.

**Files ported:**
- `cipher_audit.cpp` (may13) — RETIRE in favor of existing `cipher_rt_audit.{c,h}`.
- `cipher_trace.cpp` → `cipher_rt_trace.cpp`
- `cipher_receipt.cpp` → `cipher_rt_receipt.cpp`
- `cipher_carbon.cpp` → `cipher_rt_carbon.cpp`
- `cipher_fairness.cpp` + `cipher_fairness_shm.cpp` → `cipher_rt_fairness.cpp`
- `cipher_guard.cpp` → `cipher_rt_guard.cpp`
- `cipher_comply.cpp` → `cipher_rt_comply.cpp`
- `cipher_loop.cpp` → `cipher_rt_loop.cpp`
- `cipher_pipeline.cpp` → `cipher_rt_pipeline.cpp`
- `cipher_determinism.cpp` → `cipher_rt_determinism.cpp`
- `cipher_continuity.cpp` → `cipher_rt_continuity.cpp`
- `cipher_pulse.cpp` → `cipher_rt_pulse.cpp`
- (Defer: PREDICT, SHIELD, SUSTAIN, THERMOSTAT — partial ops; v1.5 work.)

**Prometheus exporter additions:** Update cipher-exporter `/metrics` with:
- `cipher_tenant_fairness_quota{tenant=X}`
- `cipher_tenant_carbon_grams_co2{tenant=X}`
- `cipher_tenant_receipt_hash{tenant=X}` (compact form)
- `cipher_tenant_session_band{tenant=X}` (label: HUMAN/AGENT/BATCH)

**Behavioral test:**
- Per-tenant FAIRNESS quota debits visible at `/proc/cipher/fairness`.
- AUDIT chain advances per launch (kmod nr 8 telemetry).
- TRACE bounded buffer rotates without drops at sustained load.
- RECEIPT per-session HMAC verifiable.

**Rollback path:** Each observability op can be individually disabled via env var. No anchor needed; ops are observe-only.

**Risk register:**
- R-W4.1 [LOW]: Observer overhead — these are async/post-launch; budget < 1 µs/launch combined.
- R-W4.2 [LOW]: Prometheus cardinality — at N=100 tenants, ~10 metrics × 100 = 1000 series. Within `:9402` capacity.

### Week 5 — KV-dedup live wire + v1 substrate consolidation — **DONE** (`cipher_rt_phase4` `ec0e005` week-5-complete; cipher-fusion-evidence post-close `4302079`; `cipher_kv_bridge.so` `c04b0c39` → `f041789c` at Step 1b)

<!-- v1.2.3: DONE marker added per B.1 §7 re-sequence. Week 5 complete per [[week5-complete]] memory: KV-dedup live wire + Mistral N=4 KL=0 + 45.8 GiB saved at TinyLlama N=4; week-5-complete tag on all 3 trees. -->

> **Schedule note (2026-05-21 reconciliation):** This slot originally hosted CP 5.5 100-tenant benchmark in v1.2.1; the v1.2.2 amendment (see §7 preamble above) repurposed it as KV-dedup live wire and v1 substrate consolidation. The CP-5.5 content moved to Weeks 13-14 in v1.2.2 (and to Weeks 15-17 in v1.2.3).

**Goal:** Wire the `cipher_vllm_plugin` integration layer to drive the kmod kvdedup substrate against live vLLM decode KV pages. The kmod substrate is complete (5 ioctls verified, plan §1.3); the Python bridge layer is the v1 work item.

**Files ported:**

- `cipher_vllm_plugin/cipher_vllm_kv.py` — bridge from vLLM's KV offload backend to the kmod's `/dev/cipher_kvdedup` ioctls.
- `cipher_vllm_plugin/cipher_kv_offload.py` — content-hash + page-handle marshaling.
- New test harness `tests/test_kvdedup_live_decode.py` for the bit-identical KV-dedup gate at N=4 same-prompt tenants (TinyLlama or Mistral).

**Behavioral test:**

- Bit-identical KV across N=4 same-prompt tenants (KL gate ≤ 5.5e-5).
- Dedup hit rate ≥ 60% for the shared system prompt portion.
- W1 regression PASS.
- Track 2 SC6 PASS unchanged.

**Rollback path:** `CIPHER_KVDEDUP_LIVE=0` falls back to non-dedup KV allocation per tenant. The kmod kvdedup ioctls stay live for synthetic harness use.

**Risk register:**

- R-W5.1 [HIGH]: bit-identical correctness gate at decode is tighter than the synthesized-page T4.6.3/T4.6.4 gates; surprises possible at live attention shapes. (Was R-W6.1 in pre-2026-05-21 doc; renumbered with the section.)
- R-W5.2 [MEDIUM]: vLLM upstream API stability (offload backend interface) — the bridge may need pin to a specific vLLM minor version. (Was R-W6.2 pre-reconciliation.)

### Week 6 — In-flight (post-Week-5) + architecture-gap carry — **IN FLIGHT** (this week)

<!-- v1.2.3: Week 6 reframed per B.1. Pre-2026-05-21 Week 6 was a "reserved TBD" placeholder; the v1.2.2 amendment had two candidate uses (absorb KV-dedup overflow vs fold forward). v1.2.3 picks "absorb": Week 6 hosts the architecture-gap carry from WEEK_6_ARCHITECTURE_GAP_AUDIT.md. The "fold forward to 13 weeks" candidate is explicitly NOT taken — v1.2.3 instead extends to 15-17 weeks to fold the six gaps cleanly. -->

**Goal:** Land the post-Week-5 measurement and audit work; close the two structural-cap kmod gaps (G1 + G2); audit the VA-pool architecture (G5) read-only ahead of W10-12 implementation; kick off the May-13 POC reconstruction in parallel to W7+ substrate work.

**In-flight (already landed this week):**

- **KV-dedup auto-trigger (plugin-only update).** `cipher_vllm_kvdedup.py` `cd8c826f → 438e4023`. Time + pressure modes default-off. Unblocks B1/B2 unattended. Substrate anchors unchanged. Per [[week6-kvdedup-autotrigger]].
- **Bench harness audit + rewrite + Option 1 redo.** `cipher-fusion-evidence` commit **`cc913a6`** lands `WEEK_6_BENCH_HARNESS_AUDIT.md` + `bench_llm.py` (md5 `558865fd914c48ab843b3a5cfb10a452`, 883 LOC) + `WEEK_6_BENCH_HARNESS_REWRITE.md` + `WEEK_6_OPTION_1_REDO.md`. Headline: single-instance B-sweep CIPHER is NEUTRAL vs vanilla; the lift mechanism is cross-tenant batching. Per [[week6-bench-harness]].
- **Architecture gap audit.** `cipher-fusion-evidence` commit **`07c2212`** lands `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` (md5 `c5d2d4ad`, 640 lines). Six load-bearing gaps surfaced: G1 (CP54 cap 64 — single blocker), G2 (weight-arena cap 16), G3 (KV-dedup content-only hash), G4 (Marlin single-model weight kit), G5 (VA pool per-process), G6 (no kmod-resident AUDIT chain). Per [[week6-arch-gap-audit]]. Drives this v1.2.3 plan revision.

**Carry items (to land in remainder of W6):**

- **G1 + G2 kmod cap bumps (kmod 0.5 ABI bump).** Raise `CIPHER_CP54_MAX_ALLOCS` 64 → ≥128 at `cipher_kmod/cipher_cp54_sched.c:123` (file:line confirmed by audit §8.1). Raise `CIPHER_WA_MAX_ARENAS` 16 → ≥100 at `cipher_kmod/cipher_ioctl.h:481` (file:line confirmed). Audit every `for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++)` loop body for µs-level cost at the new size. Re-run CP 5.4 Step 1.3b' POOL regression at N=128 and Track 3 DSM SC1-SC6 at the new cap. Land as `cipher_kmod` 0.5 bump with new srcversion recorded in memory. **LOC: ~80. Eng time: 1.5 days.**
- **G5 VA-pool architecture audit (read-only research).** Decide fix path among: (a) shrink the default `va_gib=80` at `cipher_vllm_kv.py:58` and let the plugin compute from `hf_config` per actual model footprint; (b) move VA reservation to the kmod weight-arena fd custodian as authoritative broker; (c) container-per-tenant orchestration **(LAST RESORT — violates R-Dep1 LD_PRELOAD-only Goal 5)**. Output: `WEEK_6_G5_VA_POOL_AUDIT.md` with recommended path + LOC + eng-time. **Read-only research, 2-3 hours.** Implementation lands in W10-12.
- **May-13 POC reconstruction kickoff.** Per WEEK_6_ARCHITECTURE_GAP_AUDIT.md §6: the May-13 POC's op31-prod dispatch surface has been superseded by the Week-5 consolidation. Reconstructing the POC's 7-problem evaluation on the v1.2.3 substrate is **~6-10 eng-days parallel** work to W7+ substrate work, not on the W6 critical path. W6 deliverable is the reconstruction *plan* (which POC actuators map cleanly to v1.2.3 substrate primitives vs which need fresh ports), not the reconstruction itself.

**Behavioral test (W6 close):**

- Cap-bump regression: CP 5.4 Step 1.3b' POOL test passes at N=128; Track 3 DSM SC1-SC6 PASS at N=128.
- G5 audit doc committed with one recommended path and LOC + eng-time estimate.
- May-13 POC reconstruction plan committed (which actuators reproduce, which need fresh ports).

**Rollback path:** kmod 0.5 ABI bump is additive (only constants raised; struct layouts unchanged). Rollback = re-deploy the prior `cipher_kmod.ko` (`2fc70c3`). No userspace impact (libcipher_rt does not depend on cap values).

**Risk register:**

- **R-W6.1 [LOW]:** CP 5.4 migration-FSM evaluator (`cp54_eval_migration` at `cipher_cp54_sched.c:445+`) may have O(N²) loop bodies that pass at N=64 but stall at N=128 (audit §7.3 R1). Mitigation: cycle-counter benchmark before merge; target ≤ 500 µs per call.
- **R-W6.2 [LOW]:** PID hashtable (1024 buckets at `cipher_internal.h:73-74`) may have collision-chain depth > 4 at N=100 (audit §7.3 R2). Mitigation: synthetic PID-collision stress test; if depth > 4, raise hashtable to 4096 buckets in the same kmod 0.5 bump (one-line change).
- **R-W6.3 [MEDIUM]:** G5 audit may conclude > 4 weeks of substrate work is needed for path (b) kmod-broker, in which case CP 5.5 density may need scope reduction (e.g., 30-50 tenants instead of 100). Mitigation: if audit recommends path (b), this v1.2.3 plan extends further or scope reduces; surface explicitly at W6 close. **HARD ESCALATION** per §8.5 R-G5.1.

### Weeks 7-9 — COMMIT atomic state-transition primitive + G6 kmod-resident AUDIT chain + G10 CIPHER_REGISTER_MODEL ABI

<!-- v1.2.3: Weeks 7-8 extended to Weeks 7-9 per B.1. G6 (kmod-resident AUDIT chain) folds into the COMMIT contract because the COMMIT primitive's deterministic state update is the natural place to write the per-tenant HMAC chain entry. G10 (CIPHER_REGISTER_MODEL ABI at NR 27) folds into the same window because it shares the kmod-ABI-bump pattern (both add to the ioctl surface) and is a prerequisite for G3, G4, G12. The +1 week of calendar absorbs G6 (~5 eng-days) + G10 (~2 eng-days) inside the COMMIT track. -->

**Goal:** Build §4.8's COMMIT primitive **AND** the kmod-resident per-tenant AUDIT chain head (G6) **AND** the CIPHER_REGISTER_MODEL ABI at ioctl NR 27 (G10). Promote dispatch return from implicit phase to named atomic primitive. Switch all 21 overlay ops' report-time consumers to read the post-COMMIT snapshot. Land model-identity propagation through the kmod boundary so W10-12 (G3 KV-dedup model-keying + G4 Marlin tenant-scoped weight kit) and W13-14 (G12 Koopman registry model-keying) have a `model_uuid` to key on.

**Files added:**

- `cipher_rt_phase4/cipher_rt_commit.{c,h}` — the COMMIT primitive itself (per-tenant sequence counter + 5-step state update + release-fence snapshot publish). ~150 LOC.
- `cipher_kmod/` state_updater 1 kHz mirror updated to publish post-COMMIT snapshot atomically. ~50 LOC kmod change.
- **G6:** `cipher_kmod/cipher_audit_chain.c` (new) — per-tenant HMAC-SHA256 chain head in shared memory mapped to userspace at REGISTER_TENANT time. Storage: append-only per-tenant ring buffer (mmap'd via the existing tenant snapshot shm pattern at `cipher_tenant_snapshot.c`). HMAC accumulator added to `struct cipher_pid_stats` (~64 B addition at `cipher_kmod/cipher_internal.h:172-234`). Per-tenant chain head seeded with FNV-prefixed HMAC at REGISTER_TENANT (ioctl nr 1). Hot-path write rate at N=100 × decode × ~8 actuator calls/token = ~64 000 writes/s aggregate — each write is a sub-µs RING_WRITE (synergistic with W10-12 RING_WRITE substrate; the RING_WRITE producer is the AUDIT-chain producer). ~400 LOC.
- **G10:** new ioctl `CIPHER_REGISTER_MODEL` at NR 27 (additive per [[cipher-abi-rule]]). Payload: `model_path` (PATH_MAX), `hf_config_hash` (32 B sha256), `model_arch` (enum: MISTRAL / QWEN / LLAMA / GPT_NEOX / OTHER). Return: `model_uuid` (128-bit). Storage: kmod-side model-registry hashtable keyed on `hf_config_hash`. Per-tenant binding: tenant's current `model_uuid` is a field in `struct cipher_pid_stats` (added in the reserved-tail bump from W1). Plugin-side call: `cipher_vllm_kv.py` calls REGISTER_MODEL at engine init, then propagates `model_uuid` into every `vmm_zeros` call (replaces the existing `tenant_id`-only call at `cipher_vllm_kv.py:87`). ~200 LOC (kmod + plugin). Cite WEEK_6_ARCHITECTURE_GAP_AUDIT.md §3.1 G10 sketch.

**Files modified (mechanical):** all 21 overlay ops' `_report()` functions to read the snapshot via `cipher_get_current_tenant_snapshot()` rather than read global counters directly. No per-launch hook changes; only report-time readers.

**COMMIT × AUDIT chain composition:** the v1.2.2 §4.8 COMMIT contract specifies the deterministic state-update order AUDIT chain → FAIRNESS → CARBON → RECEIPT → kmod-resident tenant context. In v1.2.3 this becomes the natural insertion point for G6: COMMIT's AUDIT-chain step writes one entry into the kmod-resident per-tenant chain head and advances the HMAC accumulator under release-fence discipline. The chain head is a per-tenant ring buffer; consumer is offline (the neocloud operator's billing-export tooling).

**Behavioral test:**

- W1 regression PASS within ±3%.
- 21 overlay op self-tests pass under COMMIT-mediated state.
- Multi-tenant N=128 contention test verifies snapshot consistency across observer reads (N=128 per the W6 G1 cap bump).
- **G6 gate:** per-tenant HMAC chain head advances on every COMMIT; chain is externally verifiable by re-computing the HMAC sequence from the kmod-published ring buffer (offline verifier).
- **G10 gate:** REGISTER_MODEL ioctl round-trip < 100 µs (it's a one-shot at engine init, not a hot path); model-registry hashtable scales to ≥ 100 distinct `hf_config_hash`es; concurrent REGISTER_MODEL from 5 tenants returns 5 distinct `model_uuid`s.
- Track 2 SC6 PASS.

**Verification gate:** N=128 concurrent tenants run a contention harness that races two observer reads of the same snapshot from different threads; both reads must see the same per-launch sequence number AND the same HMAC chain head pointer (atomicity check covers both COMMIT and G6). Per-tenant sequence counter discipline verified by `__atomic_compare_exchange` no-spin assertion.

**Rollback path:** `CIPHER_COMMIT_MODE=legacy` falls back to v1.2.1 per-observer state mutation; G6 chain head writes become no-ops; G10 ioctl returns -ENOSYS and the plugin falls back to tenant-id-only KV bridge calls (current v1.2.2 behavior). Anchor `libcipher_rt.so.pre_week7` preserved.

**Risk register:**

- **R-W7.1 [HIGH]:** Observer protocol change touches all 21 overlay ops' report paths. Validation work scales with op count. Mitigation: structured port checklist; per-op self-test must pass before integration.
- **R-W7.2 [HIGH]:** Atomicity guarantee at N=128 multi-tenant may surface contention not visible at N=15. Mitigation: Week 9 includes an explicit N=100 (and N=128) stress run before declaring COMMIT closed.
- **R-W7.3 [MEDIUM]:** Existing per-observer state mutation paths may have hidden invariants (e.g., implicit ordering assumed by the observer's internal logic) not captured in the COMMIT design. Mitigation: read every overlay op's report code in advance of the Week 7 port (Wave 4 audit covered this; surfacing any remaining hidden invariants is a Week 7 R-W7.3 audit item).
- **R-W7.4 [MEDIUM]** (NEW v1.2.3, G6 fold-in): The kmod-resident AUDIT chain head's HMAC accumulator + per-tenant ring buffer adds ~64 B to `struct cipher_pid_stats` (sized at ~1.7 KB today, audit §2.5). The reserved-tail bump from W1 already added ~16 B for snapshot fields; the G6 fold needs another ~64 B which exceeds the reserved-tail headroom. Mitigation: bump `cipher_pid_stats` size, version the struct layout in `cipher_internal.h:32` (per the existing ABI versioning pattern); userspace mirror `cipher_tenant_snapshot_user` in `cipher_ioctl.h` size invariant goes from 336 → 400 bytes. Verified by a new T-W7.4 size-invariant test. Cite WEEK_6_ARCHITECTURE_GAP_AUDIT.md §3.1 G6 sketch (~400 LOC).
- **R-W7.5 [MEDIUM]** (NEW v1.2.3, G10 fold-in): If REGISTER_MODEL races with REGISTER_TENANT on a new process startup (e.g., vLLM engine subprocess fork), the `model_uuid` may not be bound at the moment the first kernel launches. Mitigation: COMMIT primitive's snapshot publication defers to a sentinel `model_uuid = MODEL_UNKNOWN` until REGISTER_MODEL lands; downstream model-keyed actuators (G3, G4, G12) treat MODEL_UNKNOWN as a pass-through (vanilla behavior, no dedup / Marlin / Koopman engagement). Documented in v1.2.3 §8.5 R-G10.1.

### Weeks 10-12 — RING_WRITE lock-free inline telemetry substrate + G3 KV-dedup model-aware keying + G4 Marlin tenant-scoped weight kit + G5 VA pool implementation

<!-- v1.2.3: Weeks 9-10 extended to Weeks 10-12 per B.1. G3 (KV-dedup model-keying) folds here because it shares the kmod hot-path-write pattern with RING_WRITE. G4 (Marlin tenant-scoped weight kit) folds here because the Marlin engine touches the shm + per-tenant-snapshot infrastructure RING_WRITE establishes. G5 (VA pool per-tenant sizing) folds here because the W6 audit recommends a substrate path that is independent of the COMMIT track and ships cleanly inside the RING_WRITE-extended window. The +2 weeks of calendar absorb G3 (~3 eng-days) + G4 (~5 eng-days) + G5 (~3 eng-days, path-dependent) inside this track. -->

**Goal:** Build §4.9's RING_WRITE substrate. Port may13's ring buffer + write inline producer + add AUDIT and classifier-feedback consumers. REMEMBER consumer is Weeks 13-14 work (Koopman tier). **AND** re-key the kmod's KV-dedup xxhash64 lookup from content-only to `(model_uuid, layer_idx, head_idx, dtype, content_hash)` (G3 — closes the silent cross-model corruption defect). **AND** refactor the Marlin engine's weight cache from `weight_ptr` key to `(model_uuid, layer_idx, K, N)` (G4 — closes the single-model assumption). **AND** implement the W6-audit-selected G5 fix path (VA pool per-tenant sizing — closes the N≈12 process-count exhaustion).

**Files added:**

- `cipher_rt_phase4/cipher_rt_ring.{c,h}` — the per-tenant SPMC ring + producer inline + consumer base class. Port from may13's `cipher_10ops.h:83-100` + `cipher_intercept.cpp:169-184`. ~200 LOC.
- `cipher_rt_phase4/cipher_rt_ring_audit_consumer.c` — the AUDIT consumer that reads ring entries and advances the HMAC chain (composes with G6 from W7-9). ~80 LOC.
- `cipher_rt_phase4/cipher_rt_ring_classify_feedback.c` — the classifier-feedback consumer that updates the ORACLE EMA from ring entries. ~100 LOC.
- New test harness `cipher_test_ring_contention.c` — N=128 producer + 4 consumers stress test. ~200 LOC.
- **G3:** Re-keyed KV-dedup hashtable in `cipher_kmod/cipher_kvdedup.c`. The new key is the 5-tuple `(model_uuid, layer_idx, head_idx, dtype, content_hash)`. The `model_uuid` field already exists in `struct cipher_rt_kv_page_tag` (`cipher_rt_kv_alloc.h:43`) from G10 in W7-9; G3 wires it into the hash lookup path at `cipher_rt_kv_alloc.h:112`. ~200 LOC. **Correctness gate** (HARD): teacher-forced KL gate ≤ 5.5e-5 across 6 cross-model pairs (Mistral-7B vs Llama-3.1-8B; Mistral-7B vs Qwen-7B; Llama-3.1-8B vs Qwen-7B; plus 3 SLM × LLM pairs). The dedup engine must refuse to share pages across non-matching `(model, layer, head)` triples. Cite WEEK_6_ARCHITECTURE_GAP_AUDIT.md §3.1 G3 sketch.
- **G4:** Marlin tenant-scoped weight kit. Refactor `cipher_rt_phase4/cipher_rt_marlin_engine.cpp` weight cache from `weight_ptr`-keyed to `(model_uuid, layer_idx, K, N)`-keyed. The model registration ABI (G10 from W7-9) provides `model_uuid`; the runtime stamps every quantize-repack call with `(model_uuid, layer_idx, K, N)` extracted from the launch metadata. Per-shard scale: ≥100 unique `(model_uuid, K, N)` tuples without contention (per-key shard or RCU). **Sharing optimization:** cubin JIT cache should remain keyed on `(K, N)` only (model_uuid in weight cache, NOT in cubin cache) — different models with the same `(K, N)` GEMM share the same cubin, avoiding JIT thrash per audit §7.3 R4. ~500 LOC. **Marlin remains full-GPU-only per [[cipher-marlin-primary-ctx-pin]]** — G4 does not change the Marlin × partition restriction (Phase 6 Song Han engagement remains the v2 path for partition-aware Marlin).
- **G5:** VA pool per-tenant sizing per the W6 audit decision. Path (a) — default: shrink the per-process `va_gib` at `cipher_vllm_kv.py:58` from 80 GiB to the actual model's KV-max footprint, computed from `hf_config` (num_hidden_layers × num_kv_heads × head_dim × max_model_len × bytes_per_dtype × 2 for K+V). At Mistral-7B with max_model_len=8192, this is ~6 GiB; the 80 GiB default was a worst-case-everything assumption. Path (b) — alternative: kmod weight-arena fd custodian becomes authoritative VA broker; plugin requests per-model size. The W6 audit doc picks one path; if path (b), additional ~80 LOC in kmod broker ABI. Default LOC budget: ~150 (plugin) + (0 or ~80) (kmod). Cite WEEK_6_ARCHITECTURE_GAP_AUDIT.md §3.1 G5 sketch.

**Files modified:** `cipher_inject.c`'s GOT-patched cuLaunchKernel intercept gains a post-dispatch ring write call (~10 LOC); `cipher_rt_kv_alloc.h:112` xxhash64 lookup re-keyed (G3, ~10 LOC modify); `cipher_rt_marlin_actuator.c:150-169` weight observation path now stamps `(model_uuid, layer_idx, K, N)` (G4, ~30 LOC modify); `cipher_vllm_plugin/cipher_vllm_kv.py:50-62` VA reservation now computed from `hf_config` (G5 path a, ~50 LOC modify).

**Behavioral test:**

- W1 regression PASS within ±3%.
- N=128 producer/4-consumer contention harness reports zero data races, drop rate < 0.1% under steady-state.
- 99th-percentile producer overhead ≤ 30 ns at the alarm threshold (warn at 50 ns, fail at 100 ns).
- AUDIT chain advances correctly per ring entry under N=128 load.
- **G3 correctness gate (HARD):** teacher-forced KL gate ≤ 5.5e-5 across 6 cross-model pairs verifies the dedup engine refuses cross-model page sharing. **If gate fails, G3 is reverted and CIPHER_KVDEDUP=0 is the default for heterogeneous-model deployments.**
- **G3 hit-rate gate:** for same-model multi-tenant (Mistral-7B N=4 shared system prompt), dedup hit rate ≥ 60% (matches v1.2.2 W5 gate). The model-keying refinement must NOT regress the existing same-model hit rate.
- **G4 gate:** 5 distinct (model, K, N) tuples can coexist in the Marlin engine without JIT cache thrash; Mistral-7B B=1 decode behavior unchanged (existing single-model regression).
- **G5 gate:** at N=100 vLLM tenant processes each with a 6-GiB per-tenant VA pool, total VA reservation = 600 GiB ≤ 1 TB host VA limit. Cold-start latency per tenant ≤ 100 ms (the VA reservation is `cuMemAddressReserve`, not allocation, so this is mostly bookkeeping).
- Track 2 SC6 PASS.

**Rollback path:** `CIPHER_RING_WRITE=0` disables the producer call entirely; consumers idle. `CIPHER_KVDEDUP_MODEL_KEYED=0` falls back to content-only keying (regression to silent cross-model corruption — acceptable only at same-model deployments). `CIPHER_MARLIN_TENANT_SCOPED=0` falls back to weight_ptr keying (single-model only). `CIPHER_VA_PER_TENANT=0` falls back to the 80 GiB default. Anchors `libcipher_rt.so.pre_week10`, `cipher_kmod.ko.pre_week10` preserved.

**Risk register:**

- **R-W9.1 [HIGH]:** Lock-free correctness under contention requires careful validation. Test infrastructure must exercise N=128 producer/consumer concurrency. Mitigation: dedicated Week 11 day on the contention harness; pthread thread-sanitizer in CI.
- **R-W9.2 [HIGH]:** Sub-microsecond budget is tight. Inline calls from the LD_PRELOAD-equivalent injection path cannot afford any allocation, any system call, any cache miss. Mitigation: hot-path budget alarm at 50 ns/launch (10× the 5 ns target); fall back to a no-op write if profiler detects budget breach.
- **R-W9.3 [MEDIUM]:** Consumer thread protocol must handle ring overflow gracefully (telemetry loss vs producer blocking trade-off). Mitigation: explicit drop counter per tenant + observability `/proc/cipher/ring_stats`. Producer never blocks; drops are reported.
- **R-W10.1 [HIGH]** (NEW v1.2.3, G3 fold-in): The re-keyed KV-dedup lookup may inflate per-page xxhash64 cost from O(1) to multi-key trie walk, breaking the dedup hot-path budget. Mitigation: microbenchmark the new lookup vs the old at 8192 keys × 100 lookups/s; if > 1 µs per lookup, switch to a hash-of-hashes scheme keyed on `model_uuid` first (audit §7.3 R3). HARD STOP: if the budget cannot be made, the KV-dedup substrate ships single-model-only in v1 and the heterogeneous-model framing degrades CP 5.5's per-model-density story by ~76 → ~0 GiB saved at different-model.
- **R-W10.2 [HIGH]** (NEW v1.2.3, G4 fold-in): Marlin cubin JIT cache thrash at 100 unique `(model_uuid, K, N)` tuples. Mitigation: per audit §7.3 R4, keep cubin cache keyed on `(K, N)` only; model_uuid is in the weight cache (which can be ~100s of entries cheaply) but not in the JIT cache (which is expensive to invalidate). Test at 5 distinct models × Mistral/Qwen/Llama shape sets: count unique cubins should be ≤ 32 across the family.
- **R-W10.3 [MEDIUM]** (NEW v1.2.3, G5 fold-in): If the W6 audit picks path (b) kmod-broker, the implementation may take longer than 3 days (substrate ABI change). Mitigation: surface at W6 close; if path (b) cannot land in Weeks 10-12, extend to Week 13 (further pushing Koopman from W13-14 to W14-15 and CP 5.5 to W16-18 — surfaced in v1.2.3 §8.5 R-G5.1).

### Weeks 13-14 — Koopman tier integration + G12 Koopman registry model-keying

<!-- v1.2.3: Weeks 11-12 held to Weeks 13-14 per B.1. No scope expansion of the Koopman work itself; G12 (registry model-keying) piggybacks on the recipe-table port because that port already touches `cipher_recipes.cpp` and `cipher_kernel_table.cpp`. The hold is position-only — A1 (Koopman in v1) is NOT reversed. -->

**Goal:** Wire the Goal-4 Koopman learning tier so 6 ops (REMEMBER, VALIDATE, SPECULATE, ADAPT, SAMPLE/GENERATE, SUBSTITUTE-Koopman lane) ship in v1. This is the largest scope expansion in v1.2.2 (Goal 4 of the five product goals). **AND** key the Koopman recipe registry on `(model_uuid, layer_idx, shape)` instead of `shape`-only (G12 — closes the cross-model-mis-keying defect that would otherwise let a Mistral-7B recipe fire on a Qwen-7B layer with the same `(K, N)`).

**Files ported:**

- `cipher-may13-evidence/src/cipher_10ops_impl.cpp` (sections for REMEMBER/VALIDATE/SPECULATE/ADAPT) → `cipher_rt_phase4/cipher_rt_learning_tier.cpp`. ~800 LOC port.
- `cipher-may13-evidence/src/cipher_intercept_cudart.cpp:2353-2425` (SAMPLE/GENERATE) → `cipher_rt_phase4/cipher_rt_sample.cpp`. ~80 LOC port, fix `CIPHER_SAMPLE_DIM=1` to actual tensor sample (~64 strided floats) + remove 500-launch hard-stop.
- `cipher-may13-evidence/src/cipher_lnn.cpp` (CfC forward) → `cipher_rt_phase4/cipher_rt_lnn.cpp`.
- `cipher-may13-evidence/src/cipher_edmd.cpp` + `cipher_edmd_live.cpp` → `cipher_rt_phase4/cipher_rt_edmd.cpp`. **Fix the synthetic-input bug at `:663-669` / `:693-701`** — feed actual (h_before, h_after) snapshots from REMEMBER, not the near-identity synthetic.
- Recipe registry seeding (`cipher-may13-evidence/src/cipher_recipes.cpp:346`) — replace the 32+ stale entries (Llama-3-70B / HyperFlux / A100 / SOMA) with **a narrow workload domain across the 5 v1 model families** (Mistral-7B + Qwen-7B + Llama-3-8B + 2 SLMs on H100 SXM5) so EDMD has shape matches in production. ~250 entries (~50 per model family) with real measured M/N/K hashes from CP 5.5 baseline traces, each keyed on `(model_uuid, layer_idx, shape)` per G12. <!-- v1.2.3: registry seeding expanded from 50 entries × 2 models to 250 entries × 5 models per CP 5.5 hybrid heterogeneous-model scope. Keying on (model_uuid, layer_idx, shape) per G12 from WEEK_6_ARCHITECTURE_GAP_AUDIT.md §3.1 G12 sketch (~300 LOC). -->

**Files modified:**

- `cipher_rt_dispatch.cpp` — wire the Koopman lane in `apply_recipe` to actually call the SUBSTITUTE-Koopman path when EDMD pipeline status is SOLVED.
- `cipher_inject.c` — spawn the Stage 1 shadow thread (for REMEMBER/VALIDATE/SPECULATE) and Stage 2 background thread (for ADAPT) at injection time, gated by `CIPHER_LEARNING_TIER=on` (default off in v1 ship until the CP 5.5 gate passes; default on after).
- SPECULATE check-side bug fix at `cipher_intercept.cpp:129`: read `desc.op_class` AFTER CLASSIFY runs, not before (the bug audit_section_1a:45 flagged).

**Behavioral test:**

- W1 regression PASS within ±3%.
- EDMD pipeline reaches SOLVED status on at least one op_class within 500 launches at steady-state Mistral-7B B=8 prefill (validates the snapshot-feeder fix).
- SUBSTITUTE-Koopman lane produces bit-identical decode on at least one shape (validates the Koopman surrogate generates correct outputs).
- Track 2 SC6 PASS.
- 24-hour soak (G4) at N=15 with learning tier ON — no kernel oops, no thread crashes.

**Verification gate:** real EDMD convergence on a Mistral-7B GEMM shape, validated by a separate offline replay tool that re-runs the Koopman surrogate on the same training set and produces fit_error < 0.05. If no shape converges within 500-launch warmup, the v1.2.2 Koopman claim is retracted and the 6 ops fall back to V2-SCOPE.

**Rollback path:** `CIPHER_LEARNING_TIER=off` disables the shadow + background threads entirely. The Koopman lane in dispatch falls through to PASS_THROUGH (current v1.2.1 behavior). Anchor `libcipher_rt.so.pre_week11` preserved.

**Risk register:**

- R-W11.1 [HIGH]: The snapshot-feeder fix (h_before / h_after) and the recipe registry reseed are both load-bearing. If either fails, no Koopman lane fires and the 6 ops effectively fall back to V2-SCOPE. Mitigation: prove convergence on at least one shape before committing the Koopman claim; an unconvergent Koopman tier is a documented retraction, not a silent regression.
- R-W11.2 [HIGH]: The Stage 1/2 thread spawn was never tested in production. Crash modes are unknown. Mitigation: 24-hour soak before declaring Weeks 11-12 done; signal-handler discipline copied from VOLT's pattern.
- R-W11.3 [MEDIUM]: Koopman fit_error < 0.05 is the canonical gate but Wave 5 + audit_section_4 documented that the registry lookup never matched any production shape historically. The Weeks 11-12 reseed makes shapes match, but actual surrogate quality on real shapes is unmeasured. Mitigation: the verification gate above (bit-identical decode on at least one shape) is the operational test; if it fails, the Koopman claim retracts cleanly.

### Weeks 15-17 — CP 5.5 hybrid headline benchmark on full unified runtime (workload-class + model-architecture heterogeneous)

<!-- v1.2.3: Weeks 13-14 extended to Weeks 15-17 per B.1. The +1 week absorbs the heterogeneous-model fold-in (≥5 model families coexisting). The v1.2.2 CP 5.5 measured workload-class heterogeneity (prefill / decode / burst mix); v1.2.3 CP 5.5 measures BOTH workload-class AND model-architecture heterogeneity, on the same 100-tenant H100 substrate, with the closed gaps from W6+W7-9+W10-12+W13-14. -->

**Goal:** The full **100-tenant hybrid heterogeneous benchmark** on the 30-of-33-op unified runtime. This is the final v1 deliverable. v1.2.3 reframes from v1.2.2's workload-class-only heterogeneity to hybrid: workload-class heterogeneity (5 prefill + 80 decode + 15 burst per the v1.2.2 mix) **PLUS** model-architecture heterogeneity (≥ 5 different model families coexisting: Mistral-7B + Qwen-7B + Llama-3-8B + 2 SLMs).

**Preconditions:**

- All 30 v1 ops firing (verified by per-op telemetry in `/proc/cipher/stats`).
- Nemotron Nano + Qwen-7B + Llama-3-8B + 2 SLMs on the pod (Weeks 6 + 14 install if not already done; HF cache pre-warmed).
- 100-tenant launcher hybrid-mode (5 prefill + 80 decode + 15 burst tenants distributed across 5 model families).
- All v1.2.3 weeks (1-14) closed.
- G1 (CP54 cap ≥ 128), G2 (WA cap ≥ 100), G3 (KV-dedup model-keyed), G4 (Marlin tenant-scoped), G5 (VA pool per-tenant sized), G6 (kmod AUDIT chain), G10 (CIPHER_REGISTER_MODEL ABI), G12 (Koopman registry model-keyed) all landed.

**Measurement:**

- Same as v1.2.1's Week 5 / v1.2.2's Weeks 13-14 CP 5.5 measurement: agents/GPU, fleet tok/W, per-agent p99, per-agent fairness, MFU per-WL, weight HBM saved, KV-prefix dedup hit rate, 24-hour soak.
- Plus: COMMIT atomicity verified at N=100 (per-tenant sequence counter discipline check).
- Plus: RING_WRITE producer/consumer throughput at N=100.
- Plus: SUBSTITUTE-Koopman lane fires on at least one tenant per workload class **per model family** (validates the W13-14 wiring + G12 model-keying).
- Plus: per-tenant FAIRNESS quota enforcement (not just observation — Wave 5 Cb.3 ioctl nr 25 if landed; else observation-only fallback).
- **NEW v1.2.3 — model-heterogeneity metrics:** per-model agents/GPU breakdown (Mistral-7B 30 agents, Qwen-7B 25 agents, Llama-3-8B 20 agents, SLMs 15+10 agents — exact ratio determined by the closed gaps' density envelope, surfaced honestly in the CP 5.5 report); per-(model-pair) cross-tenant KV-dedup hit rate (expected: ~0% across different-model pairs per NEOCLOUD B2-B; ≥ 60% within same-model pairs per Week 5); per-tenant cryptographic billing receipts (G6 chain head verified externally by re-computing the HMAC sequence from the kmod-published ring buffer).
- **NEW v1.2.3 — May-13 POC reproduction (parallel work landed pre-W15):** the May-13 "7 problems addressed at 15 tenants × Llama-3.2-1B" demonstration **reconstructed** on the v1.2.3 substrate. Each of the 7 problems explicitly mapped to current substrate primitives (which reproduce, which were superseded by W1-W5 consolidation, which are CP-5.5 first-time-demonstrations). Output: `WEEK_15_MAY13_RECONSTRUCTION.md`.

**Verification gate:** the headline result is reproducible from a fresh boot in ≤ 30 minutes setup. The 30-of-33-op surface is verified by a `/proc/cipher/op_status` enumeration that lists every op + its firing status. The 7-problems reproduction PASSES on the v1.2.3 substrate (or each non-reproducing problem is honestly attributed to a known supersedence/replacement, not a regression).

**Rollback path:** If CP 5.5 fails the headline, the runtime is still functional — only the marvel-scale measurement is incomplete. Iterations cost 1-2 days each; the engineering work is not lost. If the heterogeneous-model fold-in fails (e.g., G3 KL gate cannot pass), CP 5.5 falls back to v1.2.2's workload-class-only scope with explicit notation that model-heterogeneity claim is retracted to "≤ 2 model families" or similar honest scope reduction.

**Outputs:** `WEEK_15_CP_5_5_BENCHMARK_RESULT.md` (operator + investor data-room artifact); `WEEK_15_MAY13_RECONSTRUCTION.md`; raw per-tenant per-second telemetry parquet + HMAC chain export for external audit.

**Risk register:** See R-C2 (CP 5.5 scaling cliff) + the per-week risks accumulated in Weeks 6-14 + the v1.2.3 §8.5 architecture-gap risks (R-G3.1, R-G5.1, R-G5.2, R-G6.1, R-G10.1).

---

<a id="section-8"></a>
## SECTION 8 — RISK REGISTER AND HONEST GAPS

### Catastrophic risks (could prevent any goal)

| # | Risk | Prob | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R-C1 | Classifier hot-path latency exceeds 12 ns budget | medium | blocking R-2/R-3 (every launch slowed) | Profile in Week 2. Fall back to coarser classify if needed. If 30 ns realistic, accept 1-2% per-tenant overhead. | engineering |
| R-C2 | CP 5.5 measurement reveals scaling-cliff at N≈30-50 (kmod hashtable, weight_arena slot exhaustion, or kvdedup table size) | medium | blocking R-3 | Stress-test kmod at N=30 in Week 4 before CP 5.5. Pre-emptive: bump hashtable to dynamic-resize; bump arena slots from 16 to 32 (kmod 0.5). | kmod owner |
| R-C3 | Marlin × partitioning composition cannot be made to work in v1 → MFU 85% target undeliverable | low (R-1 mitigated by per-WL claim) | degrading | Accept Marlin as full-GPU-only lane (per [[cipher-marlin-primary-ctx-pin]]); MFU 85% claim restricted to non-partitioned compute-bound WLs. Phase 6 = partition-aware Marlin via Song Han engagement (4-12 weeks). | external |

### Per-actuator risks

| # | Risk | Prob | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R-A1 | DVFS over-cut (810 MHz) regresses tok/W (-12.7% measured) | low | cosmetic if guarded | Hard-bound VOLT to [1200, 1980] MHz range. Validated in code (already enforced 210-1980 range). | done |
| R-A2 | DSM migration races with kernel-in-flight on the source partition | medium | correctness | Track 3 SC2 verified at 29/29 assertions; SC3 e2e verified disjointness clauses 1+2 hold under churn. v1 ships migratable opt-in only. | done (Track 3) |
<!-- v1.2: R-A3 clarified per Deep Inspection §C.4 — kmod substrate is COMPLETE (5 ioctls wired, no TODOs); risk is at cipher_vllm_plugin integration layer, not the kmod substrate. -->
| R-A3 | kvdedup live-wiring breaks bit-identical correctness (T4.6.5 not done). **Scope clarification:** the kmod substrate is COMPLETE (5 ioctls INIT/PUT/CONFIRM/FREE/STATS wired through `cipher_rt_kv_alloc.c:490,563,602,650,669`; zero TODOs; PUT/CONFIRM two-phase handshake is correctness-gated). The integration risk lives upstream at the `cipher_vllm_plugin` layer (Python bridge), not the kernel substrate. | medium | blocking R-3 prefix story | Wire the cipher_vllm_plugin bridge in Week 4 with bit-identical gate; abort and ship 2026-05-19 substrate state if gate fails. The kmod substrate stays unchanged. | engineering (vllm plugin layer) |
| R-A4 | Weight-arena 5s reap window leaks fd on rapid producer-consumer churn | low | degrading | Track 2 SC5 verified at producer-die-first; 5s window accepted per closeout §3. v1.5 = grace window. | done |
| R-A5 | (Superseded by v1.2.2 ADJUDICATION A1.) Original v1.2.1 framing held Goal-4 Koopman to v2; A1 (re-stated in §1 line 103) moves narrow-domain Koopman into v1 W13-14 scope. Wide-domain dynamic operator discovery remains v2 research. The dispatch table reserves the lane (Section 4.5); W13-14 scope-lock (`WEEK_13_14_SCOPE_LOCK.md` Step 2) wires it for the narrow-domain seed set. | (was v2; now W13-14 v1 narrow-domain) | scope-defined | Architecture preserves the lane in the dispatch table; W13-14 implements narrow-domain seeding + SUBSTITUTE-Koopman lane validation. | v1 W13-14 (narrow); v2 (wide) |

### Per-tier risks

| # | Risk | Prob | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R-T1 | Classifier port surfaces unexpected may13 dependencies (e.g., cipher_flow_substitute STUB declarations cascading) | medium | degrading W1 | Audit_section_4 identified the disconnected substrates; the W1 port carried CLASSIFY + ORACLE only, NOT cipher_flow_substitute or cipher_koopman_runtime. The Koopman dead code in may13 ports at W13-14 per v1.2.2 ADJUDICATION A1 + §7 line 1258 (narrow-domain v1; wide-domain remains v2). | engineering |
| R-T2 | Stage 1/2 ops (REMEMBER/VALIDATE/ADAPT/SPECULATE) were never tested in production; porting them risks introducing unknown failure modes | (deliberate v2 deferral) | scope | Section 4.4 deliberately defers these to v2. Code stays in may13 unbuilt. | v2 |
| R-T3 | Observability ops at N=100 produce log spam / metric explosion at `/proc/cipher` | low | cosmetic | TRACE is bounded-buffer (`written=8192` then drop). Prometheus cardinality bounded (Section 7 W4). | done |

### Per-deployment risks

| # | Risk | Prob | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R-D1 | Lambda pod NVML restrictions (VOLT NVML NOT_SUPPORTED, HIBERNATE NOT_SUPPORTED) → some actuators degraded on Lambda | known | degrading on Lambda only | VOLT falls through to kmod nr 10 path (validated). HIBERNATE deferred to v1.5 on Lambda. | accepted |
| R-D2 | DKMS srcversion drift (DKMS-built kmod md5 differs from in-tree, srcversion identical) | low | cosmetic | Verified at anchor manifest; documented. | done |
| R-D3 | per-thread fd rule violated by lazy maintainers (44.2× p99 amplification) | medium | degrading | Test harness `cipher_test_phase4_partition_contention` MUST pass on every release. | regression gate |

### Strategic / scope risks

| # | Risk | Prob | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R-S1 | The 3.617× headline retraction discovered by investor DD before being proactively corrected | medium | strategic blocking | **THIS IS THE MOST URGENT NON-ENGINEERING RISK.** Reframe the investor pitch this week (see [F1_TPW_RECORDS_REVIEW.md](./F1_TPW_RECORDS_REVIEW.md)). Lead with 76% density (Track 2), real cross-tenant batching curve, integration thesis. | founder |
| R-S2 | "MFU 85%" universal claim made before per-WL caveat surfaced | medium | strategic | Per-WL claim explicit in Section 5.1. Investor pitch must say "85% on compute-bound prefill WLs." | founder |
| R-S3 | CP 5.5 100-tenant measurement reveals the real number is 30-50 tenants, not 100 | medium | degrading | Run smaller scoping pass (N=30) in Week 4; iterate. Honest scale claim = "demonstrated at N=X, projects to Y." | engineering |
| R-S4 | Partition-aware Marlin Phase 6 engagement (Song Han) doesn't materialize → v2 stalled | known | scope | v1 ships Marlin-as-full-GPU-only; Marlin × partition is v2 hardening, not v1 blocker. | external |
| R-S5 | The integration plan slips beyond 5 weeks | medium | strategic | Risk-managed per-week rollback path; even partial completion at Week 3 leaves a coherent substrate. | engineering |

### 8.3a — v1.2.2 new risks (per Weeks 7-8 COMMIT + Weeks 9-10 RING_WRITE)

| # | Risk | Prob | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R-W7.1 | Observer protocol change touches all 21 overlay ops' report paths; validation work scales with op count | medium | degrading W7-8 | Structured per-op port checklist; per-op self-test must pass before integration; Wave 4 covered the observer surface so the change set is bounded | engineering |
| R-W7.2 | Atomicity guarantee at N=100 multi-tenant may surface contention not visible at N=15 | medium | blocking CP 5.5 if N=100 fails | Week 8 includes explicit N=100 stress run; per-tenant sequence counter discipline keeps contention bounded to the tenant slot, not global | engineering |
| R-W7.3 | Existing per-observer state mutation paths may have hidden invariants (implicit ordering) not captured in the COMMIT design | medium | degrading | Read every overlay op's report code in advance of Week 7 port; surface any hidden invariants as Week 7 R-W7.3 audit items before the protocol switch | engineering |
| R-W9.1 | Lock-free correctness under contention requires careful validation; N=100 producer/consumer concurrency | high | blocking W9-10 + CP 5.5 | Dedicated Week 10 day on the contention harness; pthread thread-sanitizer in CI; lock-free SPMC pattern is well-understood (Disruptor) and may13 already shipped the producer side | engineering |
| R-W9.2 | Sub-microsecond producer budget is tight; inline calls cannot afford allocation, syscall, or cache miss | high | blocking RING_WRITE if budget breached | Profile in Week 9 with cycle counter; hot-path budget alarm at 50 ns/launch; fallback to no-op write if profiler detects breach (telemetry loss > correctness regression) | engineering |
| R-W9.3 | Consumer thread protocol must handle ring overflow gracefully (telemetry loss vs producer blocking) | low | cosmetic (drops reported) | Explicit per-tenant drop counter + `/proc/cipher/ring_stats` observability; producer never blocks by design; consumer-falls-behind is a consumer concern, not a producer concern | done by design |

### 8.4 — Environment-bound v2 deferrals (v1.2.2 ADJUDICATION 4)

<!-- v1.2.2: new subsection per OP_INTENT_VS_IMPLEMENTATION D3.4 + ADJUDICATION 4. Distinguishes engineering-deferred (which v1.2.2 moved into v1) from environment-deferred (which v1.2.2 keeps deferred because the v1 hardware cannot exercise them). -->

The v1.2.2 scope expansion moves all engineering-deferred ops (the 6 Koopman tier + 4 v1.5 overlay ops) into v1. The remaining 3 deferrals are **environment-bound**, not engineering-bound:

| Op | Deferral reason | v2 gate |
|---|---|---|
| **NCCL_P2P** (Op 33) | Multi-GPU NCCL communication; single-H100 v1 pod cannot exercise any NCCL traffic pattern. L2.4 eBPF P2P-routing path is also gated on multi-node deployment. | Multi-GPU hardware acquisition + multi-node deployment infrastructure |
| **OVERLAP** (NCCL compute-comm overlap scheduler at `cipher_nccl_neural.cpp:259-336` + `cipher_layer2.cpp:22-84`) | Multi-GPU AllReduce windows; single-GPU has no AllReduce traffic to overlap | Same multi-GPU gate as NCCL_P2P |
| **STRAGGLER cross-rank attribution** (Op 32 NCCL-side; local-detection half **is in v1**) | Cross-rank attribution requires multi-rank workloads; the v1 pod has rank=1 only. The local detector (per-bucket EMA + algo hint) ships in v1 because it's reachable from the NCCL plugin shim even in single-GPU mode. | Same multi-GPU gate |

**Estimated v2 gate timing:** post-seed quarter 2 — when multi-GPU hardware and multi-node deployment infrastructure land. The deferral is **not engineering work**; it is environment work (acquiring the right hardware + deploying the multi-node tooling). When the environment lands, the 3 ops port mechanically (Wave 5 §5.1.c Cc.2 NCCL plugin interface is already documented; Wave 3 contracts already cover the kmod side).

**Not deferred in v1:**

- **OVERLAP's design intent has no v1 equivalent on single-GPU.** Cannot ship a no-op version because the per-bucket schedule has no semantic meaning without inter-rank traffic.
- **NCCL_P2P's algorithm-selection policy is real and fires** from the ncclAllReduce shim when NCCL traffic is present. The shim is built and linked; on the v1 pod it never receives a call. v2 = pod that does.

### 8.5 — Architecture-gap risks (v1.2.3 — from WEEK_6_ARCHITECTURE_GAP_AUDIT.md)

<!-- v1.2.3: new subsection per B.1. Each entry cites the file:line evidence in the architecture gap audit. -->

These risks are introduced by the v1.2.3 fold-in of six gaps from `WEEK_6_ARCHITECTURE_GAP_AUDIT.md`. Each is a *known structural condition* that remains until the corresponding gap closes; the closure is scheduled per §7 above.

| # | Risk | Prob | Impact | Mitigation | Owner | Source |
|---|---|---|---|---|---|---|
| **R-G3.1** | **Silent cross-model KV corruption.** Until G3 lands (W10-12), any heterogeneous-model test with KV-dedup enabled risks silent wrong-token output (no crash, no exception, just incorrect generation). The kmod's xxhash64 lookup at `cipher_rt_kv_alloc.h:112` is content-only — two different models with different KV tensor layouts but identical 2 MiB page content will dedup, producing silent attention errors. | high (if enabled at heterogeneous N>1) | correctness blocking the entire heterogeneous-model framing | G3 closure gated on teacher-forced KL gate ≤ 5.5e-5 across 6 cross-model pairs (W10-12 gate). **Until G3 lands, all heterogeneous-model tests run with `CIPHER_KVDEDUP=0`.** The audit also documents this is consistent with NEOCLOUD_SUBSTRATE_AUDIT.md:508-509 (~0% cross-model dedup hits in practice). | engineering | WEEK_6_ARCHITECTURE_GAP_AUDIT.md §2.7, §3.1 G3 |
| **R-G5.1** | **VA pool exhaustion.** At N > ~12 tenant processes each reserving 80 GiB host VA at `cipher_vllm_kv.py:58`, host address space exhausts. CP 5.5 at N=100 will fail at process startup before any work runs. | high (until G5 lands) | structural blocker for N > ~12 multi-process tenancy | G5 W6 audit produces fix path; implementation lands in W10-12. **HARD ESCALATION:** if W6 audit concludes > 4 weeks of substrate work needed for G5 (e.g., path-b kmod-broker), this v1.2.3 plan extends further OR CP 5.5 density scope reduces (e.g., 30-50 tenants × 5 models instead of 100). Surface at W6 close. | engineering | WEEK_6_ARCHITECTURE_GAP_AUDIT.md §2.7, §3.1 G5 |
| **R-G5.2** | **VA architecture path forecloses Goal 5.** If G5 W6 audit picks path (c) container-per-tenant, the LD_PRELOAD-only deployment property (Goal 5) is at risk — containers need orchestration which is application-layer (Docker / Kubernetes / Podman / containerd), not driver-layer. Customer applications would need wrapping. | low (path-c is last resort) | strategic — invalidates Goal 5 if path-c chosen | Path (c) is the LAST resort. W6 audit MUST clearly show why path (a) per-model sizing and path (b) kmod-broker are both infeasible before recommending (c). Document the foreclosure explicitly if (c) is picked. | founder + engineering | WEEK_6_ARCHITECTURE_GAP_AUDIT.md §3.1 G5 |
| **R-G6.1** | **AUDIT chain non-kmod-resident.** Per-tenant cryptographic billing receipts require kmod-resident AUDIT chains. Until G6 lands (W7-9), AUDIT lives in userspace and is per-process, not per-tenant in kmod. Billing receipts produced before W7-9 may not pass external cryptographic audit (the chain head is in userspace and lost on process crash). | medium | strategic — neocloud per-tenant billing not production-grade until G6 | G6 folds into the COMMIT track at W7-9; chain head moves to kmod-resident shared memory. Pre-W7-9 demos can show AUDIT functionality but the billing-export protocol is not finalized until G6 ships. | engineering | WEEK_6_ARCHITECTURE_GAP_AUDIT.md §2.5 (no AUDIT in kmod), §3.1 G6 |
| **R-G10.1** | **Tenant identity propagation incomplete.** Until CIPHER_REGISTER_MODEL ABI lands at NR 27 (G10, W7-9), CIPHER has no model-identity concept at the kmod boundary. All model-scoped actuators (G3 KV-dedup, G4 Marlin, G12 Koopman) must treat tenants as opaque processes. The W10-12 KV-dedup model-keying and Marlin tenant-scoped weight kit work, and the W13-14 Koopman registry model-keying work, all DEPEND on G10 closing first. | high (chained dependency) | blocking W10-12 and W13-14 | G10 ABI bump scheduled W7-9 alongside COMMIT (shared kmod-ABI-bump pattern). If G10 slips past W9, W10-12 work is held until G10 lands. Documented in v1.2.3 §7 COMMIT track. | engineering | WEEK_6_ARCHITECTURE_GAP_AUDIT.md §3.1 G10 |

### Surfaced honest gaps (not risks — definite states)

1. **Goal-4 Koopman O(1) substitution — narrow-domain v1 W13-14 per v1.2.2 ADJUDICATION A1 + §1 line 103.** (Original v1.2.1 framing "NEVER built into v1" was superseded by v1.2.2 ADJUDICATION 1; v1.2.3 §7 re-sequence holds Koopman at W13-14, no scope expansion.) The math exists (cipher_edmd, cipher_lnn). The runtime module exists (cipher_koopman_runtime — dead code, zero callers at audit time). The hot-path lane is reserved (Section 4.5 dispatch table). v1.2.2 moved the tier into v1 scope (originally Weeks 11-12; v1.2.3 holds at Weeks 13-14). This gap statement is preserved historically; the binding disposition is v1.2.2 ADJUDICATION A1 + v1.2.3 §1 line 103: narrow-domain in v1 W13-14, wide-domain in v2.
2. **The deployed runtime currently ships 4 of 33 canonical ops** (VOLT, ARBITRATE-as-kmod, AUDIT, partial SUBSTITUTE-as-Marlin). Unification adds ~14 more ops in 5 weeks → 18 of 33. Remaining 15 are either v1.5 enhancements (PREDICT/SHIELD/SUSTAIN/THERMOSTAT) or v2 (learning tier + NCCL_P2P).
3. **24-hour soak (G4) has never been run.** CP 5.5 includes it as a measurement, but if it surfaces a kernel issue at hour 12 we have no historical data to compare against.
4. **MFU 85% has never been measured on this pod under sustained 10-min windows on the full WL01-24 set.** CP 5.5 measures it.
5. **The kvdedup → live decode wire** (T4.6.5) has never been done. CP 5.5 depends on this Week 4 work landing.
6. **Nemotron Nano is not on the pod.** Install in Week 4.

---

<a id="appendix-a"></a>
## APPENDIX A — ANCHOR MANIFEST AT PLANNING BASELINE (2026-05-20)

| Artifact | Md5 | Source |
|---|---|---|
| `cipher_kmod/cipher_kmod.ko` | `008b3c66` (post-Track 2 SC5)<sup>v1.2.1</sup> | Kbuild from 14 sources |
| `/lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko` | `6654d9e5` (stale, mtime 2026-05-16) | DKMS rebuild predates Track 2 SC5; deferred per PRE_WEEK_1_ADJUDICATION_CLOSURE.md |

<!-- v1.2.1 footnote (Appendix A kmod anchor correction).
The plan's v1.0/v1.1/v1.2 row asserted `285d102e` post-Track-3-SC5 as the in-tree kmod anchor. That was wrong: Track 3 SC3 (the close-out for Dynamic SM Migration) was userspace-only — the substrate work landed entirely in `cipher_rt_phase4/cipher_rt_green_ctx.c` plus libcipher_v2 wiring; no kmod source was touched, so no new kmod md5 was produced. The post-Track-2-SC5 anchor `008b3c66` is therefore the in-tree kmod anchor at the post-Track-3-SC5 timestamp as well. WEEK_1_PRE_FLIGHT.md §1.3 documents the on-disk md5 reproducibly (clean rebuild matches `008b3c66...` byte-identically). Auto-memory entries `[CIPHER Track 3 DSM]` and `[CIPHER Track 2 weight-sharing]` will be reconciled separately; the on-disk byte is the source of truth.
-->

| `cipher_rt_phase4/libcipher_rt.so` | `83afd1ca` (post-Track 3 SC3) | OBJS from 16 active sources |
| `libcipher_v2/libcipher_v2.so.v0.2.0` | `86618c30` | Phase 3 substrate |
| `cipher_rt_phase4/cipher_kv_bridge.so` (Track 2 SC3) | `c04b0c39` | Python C-ext |
| `cipher-fusion-evidence/CIPHER_WORKLOAD_ARCHITECTURE.md` | (markdown) | 2026-05-19 |
| `cipher-fusion-evidence/CIPHER_REENGINEERING_PLAN.md` | (this document) | 2026-05-20 |

---

<a id="appendix-b"></a>
## APPENDIX B — CROSS-REFERENCES

- **Workload architecture & class-D declaration:** `cipher-fusion-evidence/CIPHER_WORKLOAD_ARCHITECTURE.md`
- **Phase 4 binding architecture:** `/home/ubuntu/PHASE_4_ARCHITECTURE.md`
- **Phase 4 contract (kernel ABI):** `/home/ubuntu/PHASE_4_CONTRACT.md`
- **33-op × 24-workload matrix:** `/home/ubuntu/PHASE_4_OP_WORKLOAD_MATRIX.md`
- **2026-05-16 audit (op surface ground truth):** `/home/ubuntu/AUDIT_REPORT_2026_05_16.md`
- **Audit section 4 (Goal-4 deep dive):** `cipher-fusion-evidence/audit_section_4.md`
- **F1 records review (3.617× retraction):** `cipher-fusion-evidence/F1_TPW_RECORDS_REVIEW.md`
- **Phase 5 architecture revision (Marlin × partition):** `cipher-fusion-evidence/PHASE_5_ARCHITECTURE_REVISION.md`
- **Track 2 weight-sharing closeout:** `cipher-fusion-evidence/phase_c/TRACK_2_CLOSEOUT.md`
- **Track 3 DSM closeout:** `cipher-fusion-evidence/phase_c/track_3/TRACK_3_CLOSEOUT.md`
- **CP 5.4 partial closeout:** `cipher-fusion-evidence/cp_5_4/CP_5_4_CLOSEOUT.md`
- **CP 5.1 (vLLM KV integration):** `cipher-fusion-evidence/PHASE_5_CP_5_1_DESIGN_MEMO.md`
- **CP 5.6 closeout (F1 fix):** `cipher-fusion-evidence/cp_5_6/CP_5_6_CLOSEOUT.md`
- **FUTURE_SCOPE/A Phase 3.5 (DVFS under vLLM):** `cipher-fusion-evidence/future_scope_a/FUTURE_SCOPE_A_PHASE_3_5_RESULTS.md`
- **Memory pointers (active campaign state):** `/home/ubuntu/.claude/projects/-home-ubuntu/memory/MEMORY.md`

---

## CLOSING NOTE

This plan is binding for the next 5 weeks of engineering AND the technical backbone of the investor data room. It surfaces every honest gap (Section 8) and names every regime-specific caveat (Section 5.6) explicitly.

The single most important architectural finding: **CIPHER is not a broken system — it is a half-assembled one.** Tree A contains the brain (classifier + observers + Koopman dead code). Tree B contains the hands (actuators in userspace + kmod). They were never connected.

The single most important *integration* finding: **the runtime already has the substrate pattern that connects them.** The priority-ordered actuator registry on the cuBLAS and SDPA dispatch substrates (`cipher_rt_matmul_dispatch.c`, `cipher_rt_attn_dispatch.cpp`) is the canonical join point. Adding a classifier substrate in Week 1 lets every may13 op port as a registered entry — observers at priority 0, classifiers on the new substrate, actuators at high priority on matmul/attn. AUDIT is the template; Marlin is the template; everything else follows the same shape.

The marvel is integration, not invention. Five weeks. Six sections of risk surfaced. Three goals achievable on the regimes where each is physically valid. **One coherent control-plane runtime.**

**File md5 at this snapshot (pre-stamp):** `ee3bc026e98dc98414ab634ec78e8bb6` — pre-revision baseline. Document was revised after advisor review (Section 5.1 timeline correction, Section 4.3.6 DVFS scope rule, Section 5.6 framing reset). Post-revision md5 to be recorded by user out-of-band after final read-through (the act of writing the md5 changes the md5; the convention is: pre-stamp md5 sealed in this line; post-stamp re-computed externally and noted in the next revision).

---

## V1.2 AUDIT TRAIL (promised at §0 line 7)

<!-- v1.2: closing-out the §0 promise of an audit trail. Block inserted immediately after v1.2 body was sealed; no body changes accompany this insertion. Same pre-/post-stamp md5 convention as the v1.0 pre-stamp paragraph above. -->

| Field | Value |
|---|---|
| Version sealed | v1.2 |
| Seal date | 2026-05-20 |
| Line count at v1.2 seal (pre-audit-trail insert) | 1406 |
| Byte count at v1.2 seal (pre-audit-trail insert) | 123433 |
| v1.2 pre-stamp md5 | `fec71406e4ab1b007661f26dfb3ecd1a` (snapshot of v1.2 immediately before this audit trail block was appended; the act of writing the audit trail changes the file md5 per the convention restated below) |
| v1.1 md5 (sealed in §0 lineage) | `e1f047d17d4b6d6424331c301e0052c2` |
| v1.0 pre-stamp md5 (sealed in closing note above) | `ee3bc026e98dc98414ab634ec78e8bb6` |
| md5 stamping convention | Pre-stamp md5 sealed inline; post-stamp md5 recomputed externally and recorded in the next revision (same convention used for v1.0 in the closing-note paragraph above). |
| v1.2 lineage source | §0 "VERSION HISTORY" table (this document, lines 19–23). |
| Per-change citation source | Every body change in v1.2 carries an inline `<!-- v1.2: source -->` HTML comment pointing back to either Deep Inspection §X.Y or Wave N §X.Y. The full enumeration is at §0 "What changed in v1.2" (lines 25–45). |

— end of CIPHER_REENGINEERING_PLAN.md (v1.2, 2026-05-20) —

---

## V1.2.1 AUDIT TRAIL (documentation-only patch)

<!-- v1.2.1: appended after the v1.2 audit trail per closing-note convention. Three documentation corrections folded from PRE_WEEK_1_ADJUDICATION_CLOSURE.md Part 3. No body logic changes. -->

| Field | Value |
|---|---|
| Version sealed | v1.2.1 |
| Seal date | 2026-05-20 |
| Patch type | Documentation-only (Part 3 C1, C2, C3 corrections from PRE_WEEK_1_ADJUDICATION_CLOSURE.md) |
| C1 | libcipher_v2 path footnote added at §1.2 (the plan already cited the correct `/home/ubuntu/libcipher_v2/...` paths; the footnote confirms vs the wrong path that appeared in earlier task briefs). |
| C2 | kmod anchor `285d102e` corrected to `008b3c66` at 4 occurrences (§1 Tree B2 intro, §1.3 Build product, §1.5 Compilation status, Appendix A). Footnote at Appendix A explains Track 3 SC3 was userspace-only. |
| C3 | §7 Week 1 behavioral-test step concretized with actual runner paths (`density_pack.sh`, `cp54_s16_orchestrator.py`, `cp54_isolation_test`, `sc6_run.py`). |
| v1.2 sealed md5 (pre-this-patch) | `33080cbd5b6e6d94247a0e8ea0ac6515` |
| v1.2.1 pre-stamp md5 | recorded externally in PRE_WEEK_1_ADJUDICATION_CLOSURE.md per closing-note convention (the act of writing the md5 changes the md5; convention restated below) |
| md5 stamping convention | Pre-stamp md5 sealed in the parallel PRE_WEEK_1_ADJUDICATION_CLOSURE.md Part 3 closure block; post-stamp md5 recomputed externally and recorded in the next revision. |
| v1.2.1 lineage source | §0 "VERSION HISTORY" table, v1.2.1 row. |

— end of CIPHER_REENGINEERING_PLAN.md (v1.2.1, 2026-05-20) —

---

## V1.2.2 AUDIT TRAIL (Option B scope expansion + COMMIT and RING_WRITE adjudications)

<!-- v1.2.2: appended after the v1.2.1 audit trail per closing-note convention. Five ADJUDICATIONS folded; 2 new §4 subsections + 1 new §8 subsection + 1 new §8 risk subsection + Weeks 6-14 in §7 + reclassification of v1 op surface to 30 of 33. v1.2.1 sealed md5 `8502b12b5cf10daaf99153e5076c7604`; v1.2.2 md5 recorded in the parallel PRE_WEEK_1_ADJUDICATION_CLOSURE.md (v1.2.2 round). -->

| Field | Value |
|---|---|
| Version sealed | v1.2.2 |
| Seal date | 2026-05-20 |
| Patch type | Architecture lock for the 12-14 week integration sequence; folds 5 adjudications from OP_INTENT_VS_IMPLEMENTATION.md + the v1.2.2 round of PRE_WEEK_1_ADJUDICATION_CLOSURE.md |
| A1 | Option B scope: v1 ships 30 of 33 ops; 6 Koopman ops + 4 v1.5 overlay ops moved into v1; 3 NCCL ops deferred to v2 on hardware grounds |
| A2 | COMMIT promoted to atomic state-transition primitive (new §4.8). Engineering Weeks 7-8. |
| A3 | RING_WRITE promoted to lock-free inline telemetry substrate (new §4.9). Engineering Weeks 9-10. |
| A4 | NCCL family formally deferred (new §8.4). |
| A5 | v1 shippability reclassification (18 SHIP-READY + 4 REQUIRES-FIX primary + 5 Koopman + 4 v1.5 overlay = 31 with the SUBSTITUTE dual-path counted twice; canonical 30 of 33). |
| Timeline | 5 weeks → 12-14 weeks; new Weeks 6-14 added to §7. |
| §7 new weeks | W6 (KV-dedup live), W7-8 (COMMIT), W9-10 (RING_WRITE), W11-12 (Koopman tier), W13-14 (CP 5.5 on full 30-op surface). |
| §8 new risks | R-W7.1, R-W7.2, R-W7.3, R-W9.1, R-W9.2, R-W9.3 |
| §8 new subsection | §8.4 environment-bound v2 deferrals (3 NCCL ops) |
| v1.2.1 sealed md5 (pre-this-patch) | `8502b12b5cf10daaf99153e5076c7604` |
| v1.2.2 sealed md5 | recorded in PRE_WEEK_1_ADJUDICATION_CLOSURE.md (v1.2.2 round); convention: post-stamp md5 recomputed externally per the closing-note rule |
| v1.2.2 lineage source | §0 "VERSION HISTORY" table, v1.2.2 row. |

— end of CIPHER_REENGINEERING_PLAN.md (v1.2.2, 2026-05-20) —

---

## V1.2.3 AUDIT TRAIL (B.1 timeline extension — architecture gap fold-in)

<!-- v1.2.3: appended after the v1.2.2 audit trail per closing-note convention. Six architecture gaps from WEEK_6_ARCHITECTURE_GAP_AUDIT.md folded into the §7 track structure. Timeline extends 12-14 → 15-17 weeks. No source code touched in this revision; the plan is binding-doc-only. -->

| Field | Value |
|---|---|
| Version sealed | v1.2.3 |
| Seal date | 2026-05-23 |
| Patch type | B.1 timeline extension — re-sequence not redesign. Folds six gaps + G10 + G12 (eight total architecture-gap items) from `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` into the existing v1.2.2 track structure. |
| Audit source | `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` md5 `c5d2d4ad68796578caaae022ff25e743`, 640 lines, dated 2026-05-23, cipher-fusion-evidence commit `07c2212`. |
| v1.2.2 sealed md5 (pre-this-patch) | `40722374f7a9b4b93d56381add208c92` (recomputed from disk 2026-05-23; supersedes the v1.2.2 audit trail's "recorded in PRE_WEEK_1_ADJUDICATION_CLOSURE.md" deferral that was never closed in-document). The v1.2.2 disk content includes the 2026-05-21 W5 Step 0 doc-cleanup (commit `5fdc075`) — a documentation-only patch that does not alter v1.2.2's architectural decisions. |
| md5 spec/disk discrepancy note | The B.1 task brief pre-condition cited expected v1.2.2 md5 `f6146a6cae2628fd5e3486cc647401d7` (1743 lines). On-disk md5 at task time was `40722374f7a9b4b93d56381add208c92` (1748 lines). Git history shows last edit at commit `5fdc075` on 2026-05-21 (Week 5 Step 0 doc-cleanup), which PREDATES the 2026-05-23 architecture gap audit at commit `07c2212`. The audit's file:line citations (L1515, L1521, L1548, L1593, L1726-1748) are consistent with the 1748-line on-disk version — i.e. the audit was written against the current content. The spec md5 is therefore a stale/pre-computed value from before the W5 Step 0 doc-cleanup, not evidence of post-audit edits. User reconciled by selecting "proceed with disk md5 as v1.2.2 baseline" via AskUserQuestion (2026-05-23). The v1.2.2 baseline is preserved at `CIPHER_REENGINEERING_PLAN.md.v1.2.2.archive` (md5 `40722374f7a9b4b93d56381add208c92` byte-identical). |
| Baseline preservation | `CIPHER_REENGINEERING_PLAN.md.v1.2.2.archive` — `cp -p` with mtime preserved; md5 byte-identical to v1.2.2 disk content. |
| B.1 — Re-sequence | v1.2.3 folds eight architecture-gap items into the existing v1.2.2 track structure. No architecture changes. No goal changes (five product goals retained; Goal 4 Koopman retained per v1.2.2 A1, NOT reversed). No op-surface changes (30 of 33). No actuator-registry-pattern changes. |
| Gaps folded | G1 (CP54 cap 64 → ≥128, W6 carry); G2 (WA cap 16 → ≥100, W6 carry); G3 (KV-dedup model-keying, W10-12 fold); G4 (Marlin tenant-scoped weight kit, W10-12 fold); G5 (VA pool per-tenant sizing, W6 audit + W10-12 implementation); G6 (kmod-resident AUDIT chain, W7-9 fold); G10 (CIPHER_REGISTER_MODEL ABI at NR 27, W7-9 fold); G12 (Koopman registry model-keying, W13-14 fold). |
| Timeline change | 12-14 weeks → 15-17 weeks. W7-8 → W7-9 (+1 week, G6+G10). W9-10 → W10-12 (+2 weeks, G3+G4+G5). W11-12 → W13-14 (held one position, G12 piggybacks on Koopman recipe-table port). W13-14 → W15-17 (+1 week, hybrid heterogeneous-model fold-in to CP 5.5). Net add: +3 weeks calendar, ~20-25 eng-days net add over v1.2.2 budget. |
| CP 5.5 reframe | From "workload-class heterogeneous benchmark" (v1.2.2: 5 prefill + 80 decode + 15 burst on one model family) to **hybrid heterogeneous benchmark** (v1.2.3: same workload-class mix + ≥ 5 different model families coexisting: Mistral-7B + Qwen-7B + Llama-3-8B + 2 SLMs). Per-model agents/GPU breakdown surfaced in CP 5.5 report; honest scope reduction documented if any closed gap falls below density target. |
| §1 goals reframed | From "three product goals" (v1.2.2: MFU / TPW / multi-tenant density, with Koopman-as-fourth implicit) to **five product goals** explicit: Goal 1 = 100-agent heterogeneous-model multiplexing per H100 (the new headline); Goal 2 = MFU per workload-class roofline; Goal 3 = TPW composed lift; Goal 4 = O(1) Koopman compute substitution (retained from v1.2.2); Goal 5 = LD_PRELOAD-only deployment transparency. |
| §0 entry | "What changed in v1.2.3" subsection added (after "What changed in v1.2.2"). |
| §7 changes | New v1.2.3 amendment subsection. v1.2.2 amendment preserved historically. Each of W1-W5 marked DONE with commit anchors. W6 reframed from placeholder to in-flight + carry (G1+G2+G5-audit+May13-recon). W7-8 → W7-9 with G6+G10 fold-in (+ R-W7.4, R-W7.5 risks). W9-10 → W10-12 with G3+G4+G5 fold-in (+ R-W10.1, R-W10.2, R-W10.3 risks). W11-12 → W13-14 with G12 fold-in. W13-14 → W15-17 with hybrid heterogeneous-model reframe + per-model metrics + May-13 reconstruction. |
| §8 new subsection | §8.5 — Architecture-gap risks (R-G3.1, R-G5.1, R-G5.2, R-G6.1, R-G10.1), each citing the audit's §2.x and §3.1 evidence. |
| §8 risk count | 5 new entries in §8.5 (R-G3.1, R-G5.1, R-G5.2, R-G6.1, R-G10.1) + 2 new entries in W7-9 risk register (R-W7.4 G6 size-invariant; R-W7.5 G10 race) + 3 new entries in W10-12 risk register (R-W10.1 G3 budget; R-W10.2 G4 cubin thrash; R-W10.3 G5 path-b extension). Total: 10 new risk entries. |
| Co-document | `WEEK_6_PLAN_V1_2_3_TRANSITION.md` lands in this commit alongside this revision; it documents the transition for investor + NVIDIA ICC + Devang communication purposes. |
| Engineering-day budget delta | G1+G2: 1.5 days (W6). G3+G10: 5 days (W7-9 alongside COMMIT). G4+G12: 8 days (W13-14 + W10-12 carries). G5: 3 days nominal (path-a per W6 audit; path-b adds ~3 more). G6: 0 incremental days (folds into COMMIT scope). May-13 reconstruction: 6-10 days parallel to W7+ substrate work. Net add: ~20-25 eng-days over v1.2.2 budget. |
| Communication implications | If any external timing commitment exists with Array / EverGiven / NVIDIA ICC / Nebius, the CP 5.5 date moves ~3 weeks. **No such commitment surfaced in current memory** — verify with user before any external communication. Pre-emptive correction language documented in `WEEK_6_PLAN_V1_2_3_TRANSITION.md` §7. |
| What v1.2.3 does NOT change | Architecture, op surface (30 of 33), actuator-registry pattern, ADJUDICATIONS A1 (Koopman in v1) / A2 (COMMIT primitive) / A3 (RING_WRITE substrate) / A4 (NCCL deferral) / A5 (v1 shippability), §4 unified architecture, §5 workload-impact verification, §6 no-compromise constraints, NCCL-family v2 deferral (still environment-bound), the L1389 reconciliation in v1.2.2's §7 amendment (preserved historically). |
| What v1.2.3 explicitly does NOT do | Re-open A1 (Koopman stays in v1). Re-open A4 (NCCL stays v2). Change the substrate code (zero source-tree changes in this revision — plan-only). Change the bench harness from W6 (Option 1 redo numbers stand). Touch any cipher_rt_phase4 or cipher_kmod tree. |
| v1.2.3 pre-stamp md5 | (sealed in this audit trail block before final write; the act of writing the md5 changes the md5 per closing-note convention. Post-stamp md5 recomputed externally and recorded in the v1.2.3 commit message body in `cipher-fusion-evidence`.) |
| v1.2.3 lineage source | §0 "VERSION HISTORY" table (this document, v1.2.3 row) + "What changed in v1.2.3" subsection. |

— end of CIPHER_REENGINEERING_PLAN.md (v1.2.3, 2026-05-23) —
