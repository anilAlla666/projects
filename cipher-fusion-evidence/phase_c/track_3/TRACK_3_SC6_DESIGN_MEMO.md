# Track 3 — Dynamic SM Migration — SC6 DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no source
modified, no GPU, no measurement. STOP for adjudication before SC6-2.
**Predecessor:** SC5 closed, F1 (the headline) adjudicated ACCEPTED.

SC6 is the **final** Track 3 sub-component: it closes the two SC5 coverage
gaps, characterizes migration cost, measures the SC2 item-4 quiet-period
worst case, and writes `TRACK_3_CLOSEOUT.md`. **SC6 is measurement +
documentation only — no source change, no anchor rotation** (kmod `285d102e`,
libcipher_rt `83afd1ca` remain loaded throughout). The one build activity is
*test-harness* refinement, not substrate code.

---

## §0 — What SC6 must deliver (and what it must NOT reopen)

F1 is adjudicated. SC6 does **not** re-litigate it. SC6 produces the
*completeness of characterization* SC5 deferred:
1. harness refinement — POOL process, mixed migratable/pinned tenants,
   clause-2 runtime probe;
2. within-A migratable-vs-pinned latency (the 1.6B-4 per-opt-in-class split);
3. migration-cost curves (per-migration cost, resolved directly);
4. worst-case PROPOSED→ABORT under a quiet period (SC2 item-4 PUSH);
5. production deployment guidance + CP 5.5 integration;
6. `TRACK_3_CLOSEOUT.md` — the final closeout with a claim→measurement
   provenance table (no overclaiming, per the SC5-adjudication constraint).

## §1 — Harness refinement (SC6-2, ~½ day)

Three changes to the SC5 harness (`sc5_tenant.py` / `sc5_churn.py` →
`sc6_*`), test-code only:

- **POOL process.** The orchestrator launches a `qos=pool` tenant that
  consumes the low prefix. This **fixes the F2 artifact**: with the POOL
  claiming the residual (constrained-low-prefix grant), there is no static
  free region — free groups exist only transiently from churn, so the
  stranding metric measures *churn-induced* fragmentation, and the free-order
  seed becomes the dominant variable it should be.
- **Mixed migratable/pinned tenants.** Per scenario: **3 migratable + 2
  pinned** PARTITION tenants (migratable set `CIPHER_MIGRATABLE=1`, pinned
  leave it unset). Enables the within-A per-class comparison (§2).
- **Clause-2 runtime probe.** Reuse the proven `cp54_s16_orchestrator.py`
  per-round lockstep barrier: each tenant runs its `%smid` probe per round,
  drops a sentinel; the orchestrator collects the contemporaneous observed
  SM sets and checks **pairwise-disjoint across all tenants** (clause 2).
  Clause 1 (observed ⊆ allocated) stays the per-tenant L2 check. SC6 thus
  runtime-probes **both** clauses.

## §2 — Within-A migratable-vs-pinned measurement (SC6-2, ~½ day)

5 seeds, **config A only** (`gap_min_grps=1`), 3 migratable + 2 pinned tenants.
Report **per opt-in class, separately** (the SC1 item-4 adjudication / the
1.6B-4 headline):
- migratable class — decode-latency p50/p95/p99, *including* migration-round
  spikes;
- pinned class — decode-latency p50/p95/p99, the zero-migration-cost baseline.

The honest expectation (stated before measurement): the pinned class is the
clean baseline; the migratable class shows the same p50 with a p95/p99 tail
slightly raised by migration rounds — *if* migration cost rises above the
workload noise floor (SC5 found it did not at aggregate; §3 resolves it
directly). A finding either way: "migration cost is invisible even per-class"
is a strong v1 result; "migratable p99 measurably higher" quantifies the
opt-in cost.

## §3 — Migration-cost curves (SC6-3, ~½ day)

SC5's aggregate p99 could not resolve migration cost (below the noise floor).
SC6 resolves it **directly**: `sc6_tenant` records, per decode round, whether
`handler.step()` migrated, and the round's wall latency. Migration cost =
(migration-round latency) − (that tenant's own non-migration-round p50).
Report:
- per-migration cost **p50/p95/p99** (the direct measurement);
- cost **vs stranding severity** at PROPOSE — expectation: ≈ flat, because
  migration is count-preserving (one green-ctx rebuild regardless of how far
  the partition moves); a *non-flat* curve would be a finding;
- empirical check of the ~1.7 ms prior (Step 1.5) — confirm or surface.

## §4 — Worst-case PROPOSED→ABORT under a quiet period (SC6-2, ~½ day)

SC2 item-4 PUSH. The lazy timeout fires only on the next FREE/COMPACT after
PROPOSE+30 s. SC6 measures:
- **bounded case** — a churning deployment: time from PROPOSE to the next
  natural ledger-mutating ioctl (short — sub-second under load);
- **worst case** — a quiesced ledger: hold a PROPOSED migration, issue **no**
  FREE/COMPACT, and confirm the abort does **not** fire until activity resumes
  — i.e. the latency is **unbounded-until-activity**, exactly the item-4
  trade-off.

**Recommendation framing for the closeout:** if a production deployment can
sit quiet for long stretches with a PROPOSED migration pending, the lazy
timeout strands that tenant's reservation → **recommend the v1.5 explicit
timer / eventfd**. If deployments are always churning (the multi-tenant target
regime), v1 lazy timeout is acceptable. SC6 measures, then `TRACK_3_CLOSEOUT.md`
records the explicit recommendation.

## §5 — Production deployment guidance (SC6-3, ~½ day)

- **Which workloads opt in:** high-churn multi-tenant (partition lifetimes
  varied, frequent alloc/free) — `CIPHER_MIGRATABLE=1` pays off (the F1
  ~70 % stranding reduction). Low-churn / long-lived partitions — `pinned`
  is fine; migration would rarely trigger anyway.
- **Policy values:** SC5 found the SC4 defaults (gap 1 / sustain 2 s / rate
  10 s) did not compound cost; SC6's denser within-A + cost data either
  confirms them or recommends a retune (Item-c PUSH).
- **CP 5.5 integration:** the 100-tenant soak should run a realistic
  migratable/pinned mix; `/proc/cipher/migrations` is the operator's live
  view (counters + per-tenant state).
- **Operator runbook:** the module params, `/proc/cipher/migrations`, and the
  rollback path.

## §6 — `TRACK_3_CLOSEOUT.md` (SC6-4, ~½ day)

The final Track 3 document:
- the architectural claim, with a **claim→measurement provenance table** —
  every claim cites the SC and run that substantiates it (SC5 §9 addendum is
  the seed of this);
- **v1 boundaries — prominently, up front** (SC1 item-5 PUSH): single-GPU
  H100; 15-group ledger (≤14 partitions + POOL); count-preserving migration;
  poll-based delivery; opt-in default-pinned; rate ≤ 1/tenant/10 s; lazy
  timeout (§4 caveat);
- **v2 deferrals named:** multi-GPU/NCCL migration; sub-ms eventfd push;
  resize-via-migration; POOL/SHARED-client migration; numeric per-tenant cost
  budgets; MIGRATING-state watchdog;
- the CP 5.5 integration plan;
- anchor lineage (kmod `8d777dfb`→`98da2d1f`→`285d102e`; libcipher_rt
  `ebc0baaa`→`83afd1ca`).

## §7 — SC6-2 / SC6-3 / SC6-4 plan & discipline

| phase | scope | est. |
|---|---|---|
| **SC6-1** | this design memo | done |
| **SC6-2** | refine harness (§1); run within-A (§2) + quiet-period (§4) | ~1 d |
| **SC6-3** | cost-characterization analysis (§3) + deployment guidance (§5) | ~½ d |
| **SC6-4** | `TRACK_3_CLOSEOUT.md` (§6) — Track 3 closes | ~½ d |

**Regression discipline — no anchor rotation in SC6.** At SC6-2 start, a
sanity check: W1/W2/W3 + isolation 15/15 + SC2-unit 29/29 + SC3 7/7 still
PASS on the unchanged substrate (`285d102e` / `83afd1ca`). dmesg clean
throughout. **Stop conditions:** a clause-2 runtime-probe violation ⇒ STOP
(substrate bug, investigate before Track 3 closes); migration cost or pinned-
tenant latency materially worse than expected ⇒ surface as a finding, do not
paper over.

**Anchors:** none rotate. kmod `285d102e`, libcipher_rt `83afd1ca`,
libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d` — all unchanged through
SC6.

---

## §8 — Adjudication ask

**STOPPING — no source modified, no build, no GPU.** Decisions:

1. **Harness refinement (§1)** — accept POOL process + 3-migratable/2-pinned
   mix + the `cp54_s16`-style per-round barrier for the clause-2 probe.
2. **Within-A plan (§2)** — accept per-opt-in-class latency reporting, A
   config, 5 seeds.
3. **Cost curves (§3)** — accept the direct per-migration-round measurement
   (migration-round latency − tenant's own non-migration p50).
4. **Quiet-period (§4)** — accept measuring both the bounded (churning) and
   unbounded-until-activity (quiesced) cases, with the v1.5-timer
   recommendation decided from the result.
5. **Closeout (§6)** — accept the `TRACK_3_CLOSEOUT.md` structure
   (provenance table, v1 boundaries up front, v2 deferrals).

On adjudication: proceed to **SC6-2**. After SC6-4, **Track 3 is complete.**
