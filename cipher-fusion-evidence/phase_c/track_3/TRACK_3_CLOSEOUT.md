# TRACK 3 — DYNAMIC SM MIGRATION — CLOSEOUT

**Date:** 2026-05-19. **Status: TRACK 3 COMPLETE.** Final document.
Anchors at close: kmod `285d102e`, libcipher_rt `83afd1ca`, libcipher_v2
`86618c30`, cipher_kv_bridge `fca6843d`.

---

## 1 — Headline

Track 3 (Dynamic SM Migration) delivers **live tenant SM migration under
arbitrary partition churn at bounded substrate cost (~1.26 ms per migration),
transparent to tenant code via opt-in, with all correctness invariants —
disjointness clauses 1+2, teacher-forced KL gates — verified at runtime across
5 free-order patterns.** It is the substrate-layer primitive that keeps a
multi-tenant GPU's batch POOL compact (≈70 % less stranded) as partitions
allocate and free in any order — a primitive no production GPU stack provides.

## 2 — Measurement provenance

Every Track 3 claim mapped to the measurement that substantiates it. No buried
caveats — a claim not in this table is not substantiated.

| Claim | Measurement | Document |
|---|---|---|
| Live SM migration → ~70 % POOL-stranding reduction under churn (A 2.78 vs B 9.49 mean, 5 seeds) | SC5-3 A/B sweep, 10 scenarios | `TRACK_3_SC5_MEASUREMENTS.md` |
| Migration primitive cost ~1.26 ms p50, flat across seeds (count-preserving ⇒ constant) | SC6-2 within-A sweep, 17 migrations | `TRACK_3_SC6_MEASUREMENTS.md` §4 |
| Migratable opt-in carries no measurable decode-latency penalty | SC6-2 within-A migratable-vs-pinned, 5 seeds | `TRACK_3_SC6_MEASUREMENTS.md` §2 |
| Disjointness clause 2 (cross-tenant) holds under churn + live migration | SC6-2 5-seed barriered runtime probe (0 fails) | `TRACK_3_SC6_MEASUREMENTS.md` §3 |
| Disjointness clause 1 (observed ⊆ allocated) holds every migration | SC3 e2e + SC5/SC6 live L2 `%smid` (~80 migrations) | `TRACK_3_SC3_*` / `_SC5_` / `_SC6_` |
| Quiet-period PROPOSED→ABORT is unbounded-until-activity (lazy timeout) | SC6-2 quiet-period test (quiesced 75 s still PROPOSED; churning 30.03 s) | `TRACK_3_SC6_MEASUREMENTS.md` §5 |
| Destroy-failure commit-and-leak works at scale | SC5-3 destroy-fault scenario (3 faulted + 2 clean, all KL pass) | `TRACK_3_SC5_MEASUREMENTS.md` §5 |
| Migration state machine + reaper correct (PROPOSE/START/ACK/COMMIT/ABORT, crash in any phase) | SC2 29/29 unit assertions | `TRACK_3_SC2_*` |
| No regression to any prior verified primitive | per-SC smoke W1/W2/W3 + isolation 15/15 + SC2 29/29 + SC3 7/7 | each SC `_REGRESSION.md` |

## 3 — v1 boundaries

What v1 Dynamic SM Migration **is and is not**, stated prominently so a
production reviewer sees the envelope before relying on it:

- **Single-GPU, H100.** No multi-GPU / NCCL migration.
- **15-group ledger** — the device partitions into 15 × 8-SM groups
  (`cuDevSmResourceSplitByCount`, Step 1.3a). A full ledger holds the POOL +
  up to ~14 PARTITION tenants.
- **Count-preserving migration only** — a migration moves a partition to a
  different group set of the **same size**; it does not resize a partition.
- **Rate-limited** — ≤ 1 migration per tenant per 10 s (operator-tunable via
  `cipher_cp54_mig_ratelimit_ms`).
- **Quiet-period PROPOSED→ABORT is lazy** — the timeout fires on the next
  FREE/COMPACT, not on a timer. Under a churning deployment the worst case is
  ~31 s (bounded); under a **quiesced** ledger it is **unbounded-until-
  activity** (a pending PROPOSE strands its reservation until activity
  resumes). Accepted for the churning multi-tenant target regime; see v1.5.
- **L2 verification probe is oversized** in the current implementation
  (`probe_smid_k<<<4096,64>>>` — ~27 ms/migration). Functionally correct but
  ~10× larger than needed for an 8-SM partition. **Right-size to ~tens of
  blocks before production** — drops total per-migration cost from ~28 ms to
  ~4 ms (≈0.04 % per-tenant overhead at the v1 rate limit).
- **Opt-in, default-pinned** — a tenant migrates only if it sets
  `CIPHER_MIGRATABLE`; with zero migratable tenants the substrate is
  byte-identical to pre-Track-3.

## 4 — v2 deferrals (named)

- **Multi-GPU migration** (NCCL / cross-device) — the D10 v2 bundle.
- **Sub-ms eventfd notification** — replaces poll-based delivery; the v1.5
  candidate **if** deployments can quiesce with a migration pending (§3).
- **Explicit timer for the quiet-period abort** — the other half of the v1.5
  quiescence fix.
- **Resize-via-migration** — relaxing the count-preserving constraint so a
  migration can also change a partition's SM count.
- **POOL/SHARED-client migration** — v1 migrates PARTITION tenants only.
- **Per-request dynamic arbitration** — v1 arbitrates per session, not per
  request.
- **Numeric per-tenant cost budgets** — v1 uses the boolean opt-in + the
  operator-wide rate limit.

## 5 — Production deployment guidance

- **Opt in (`CIPHER_MIGRATABLE=1`)** — latency-tolerant tenants in a high-churn
  multi-tenant deployment: the ≈70 % stranding reduction is the payoff, and
  SC6-2 measured no decode-latency penalty in aggregate.
- **Stay pinned (default)** — latency-sensitive tenants, or low-churn /
  long-lived partitions where migration would rarely trigger.
- **Policy values** — the SC4 defaults (`gap_min_grps=1`, `sustain_ms=2000`,
  `ratelimit_ms=10000`) showed no cost compounding under churn; **no retune
  indicated**. All operator-tunable at `/sys/module/cipher_kmod/parameters/`.
- **Before production: right-size the L2 probe** (§3).
- **Operator visibility** — `/proc/cipher/migrations` (policy, gap state,
  aggregate counters, per-tenant rows); dmesg verbosity via the
  `cipher_cp54_mig_verbose` module parameter.

## 6 — CP 5.5 integration plan

- The DSM substrate is ready for the **CP 5.5 100-tenant integration soak**.
- The POOL-served path (constrained-low-prefix grant + migration compaction)
  is the architectural enabler for a 100-tenant deployment (Step 1.6P scope) —
  the POOL stays compact under churn, which is what serving ~95 % of tenants
  from the batch pool requires.
- **Sequencing after Track 3:** Track 2 weight-sharing SC3–SC6 → FUTURE_SCOPE/A
  (CIPHER composed with vLLM) → CP 5.4 closure (Steps 1.6B-3 / 1.6B-4 measured
  on the Track 3 substrate, then 1.7 / 1.8) → CP 5.5.

## 7 — Sub-components

| SC | scope | anchor outcome |
|---|---|---|
| SC1 | design memo (paperwork) | none |
| SC2 | kmod migration state machine (ioctls 16–20, RSVD state, reaper) | kmod `8d777dfb` → `98da2d1f` |
| SC3 | libcipher_rt tenant primitive (`cipher_rt_green_ctx_migrate`) | libcipher_rt `ebc0baaa` → `83afd1ca` |
| SC4 | policy framework | folded into SC5 (too thin for its own cycle) |
| SC5 | headline measurement + `cp54_pool_claim_low_prefix` (the SC1 fact-2 constrained POOL grant) + policy module params | kmod `98da2d1f` → `285d102e` |
| SC6 | coverage-gap closure + cost characterization + this closeout | none (measurement/docs only) |

## 8 — Anchors at Track 3 close

| artifact | anchor | lineage |
|---|---|---|
| kmod `cipher_kmod.ko` | **`285d102e`** | `8d777dfb` → `98da2d1f` (SC2) → `285d102e` (SC5-2); loaded |
| libcipher_rt | **`83afd1ca`** | `ebc0baaa` → `83afd1ca` (SC3); loaded |
| libcipher_v2 | `86618c30` | unchanged throughout Track 3 |
| cipher_kv_bridge | `fca6843d` | unchanged throughout Track 3 |

All fallbacks preserved with md5 verification (`TRACK_3_ANCHORS.md`):
`cipher_kmod.ko.{pre_track3,pre_sc2,track3_sc2,pre_sc5,track3_sc5}`,
`libcipher_rt.so.{pre_track3,pre_sc3,track3_sc3}`; source tarballs
`cipher_kmod_src_track3_sc{2,5}.tar.gz`, `cipher_rt_green_ctx_track3_sc3.tar.gz`.

---

## TRACK 3 — CLOSED

Dynamic SM Migration is built, verified, and characterized. The substrate
keeps the multi-tenant batch POOL compact under arbitrary partition churn, at
a migration cost of ~1.26 ms (mechanism) with no measurable tenant-latency
penalty, every correctness invariant runtime-verified. The honest envelope —
v1 boundaries (§3), the one v1.5 item (quiescence timer) and the L2-probe
right-sizing — is documented up front. Anchors stable for Track 2.
