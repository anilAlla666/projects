# Track 3 — Dynamic SM Migration — SC5 DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no source
modified, no GPU, no measurement. STOP for adjudication before SC5-2 build.
**Predecessor:** SC4-1 adjudicated; SC4 build **folded into SC5** (kmod
`98da2d1f`, libcipher_rt `83afd1ca`).

SC5 is the **headline verification** of Track 3 — and it carries the folded
SC4 policy build. It rotates the kmod anchor once (`98da2d1f` → SC5) for both
the SC4 work and any SC5 kmod work.

---

## §0 — SC5 = the folded SC4 build + the headline measurement

**SC4 portion (folded — built first in SC5-2):**
- module parameters (mode 0644, `/sys/module/cipher_kmod/parameters/`):
  `cipher_cp54_mig_gap_min_grps` (default **1**), `cipher_cp54_mig_sustain_ms`
  (default **2000**), `cipher_cp54_mig_ratelimit_ms` (default **10000**),
  `cipher_cp54_mig_verbose` (default **1**);
- `cp54_eval_migration` reads the params instead of the SC2 `#define`s;
- per-tenant `migration_count`; the `/proc/cipher/migrations` node (§4);
- (`CIPHER_CP54_MIG_TIMEOUT_NS` 30 s stays a `#define` — not in the adjudicated
  param set; flagged, could be a 5th param if SC5 measurement wants it.)

**SC5 headline portion:** 5-partition randomized-free-order churn; A-vs-B;
per-tenant latency p50/p95/p99; migratable-vs-pinned; destroy-fault at scale.

---

## §1 — KEY FINDING: the constrained-low-prefix POOL grant is not yet built

SC1 **fact 2** (adjudicated) — *"B ⊆ A: A retains the constrained-low-prefix
POOL grant (`cp54_pool_claim_low_prefix`) as the POOL's safety floor during the
migration window."* SC1 §3f — *"the POOL keeps the constrained-low-prefix
grant."*

**Audit finding: it is not in the kmod.** SC2 built the migration state
machine; SC3 built the tenant primitive; neither touched the POOL claim.
`cipher_cp54_ioctl_allocate`'s POOL branch still does `cp54_claim(pid,
CIPHER_CP54_NUM_GROUPS)` — it claims **every** free group, contiguous or not.

Why this matters, concretely. Under out-of-order partition frees, a free group
can sit *above* a live partition. The unconstrained POOL claim grabs it in the
ledger — but the POOL's green context is a count-only contiguous-low-prefix
placement (Step 1.3b'), so the POOL's green ctx and its ledger-owned set
**diverge**, and the green ctx can collide with the partition in between →
disjointness break. This has been latent: W1/W2/W3 are static configs and never
exercise POOL `check_resize` under out-of-order churn — which is exactly what
SC5 introduces.

**Therefore SC5-2 must build `cp54_pool_claim_low_prefix`** — constrain the POOL
claim to the groups contiguous with its existing low prefix (stop at the first
group it does not already own and cannot reach). This is:
1. **load-bearing for A's correctness** under the SC5 churn workload (the SC1
   fact-2 safety floor — without it, the migration *window* is unsafe);
2. **the core of the B-branch** (see §2).

It is ~20–30 LOC in `cipher_cp54_sched.c` (the 1.6X-1 memo designed the logic;
SC1 fact 2 said "B's grant logic survives as a component of A"). **This expands
SC5-2 beyond the SC4 fold** — flagged honestly; the realistic SC5-2 estimate is
~2 days, not 1. Surfaced as the central adjudication item.

---

## §2 — A-vs-B is a CONFIG, not a separate build

The SC1 §4 plan called for a "throwaway B branch." With SC4 making the policy a
module parameter, **B becomes a throwaway *configuration* of the SC5 kmod, not
a throwaway *build*:**

| | A (live migration) | B (constrained grant only) |
|---|---|---|
| kmod | SC5 kmod | **same SC5 kmod** |
| `cipher_cp54_mig_gap_min_grps` | **1** (migration fires) | **16** (> 15 groups — no gap ever qualifies → migration never fires) |
| POOL claim | constrained-low-prefix (§1) | constrained-low-prefix (§1) — identical |
| behaviour | gaps are compacted away | gaps stay **stranded** (POOL valid + disjoint, capacity lost) |

This is cleaner than a separate branch: A and B are the *same binary*,
differing only by one module-param value — so the A-vs-B comparison is
perfectly controlled and trivially reproducible. The constrained-low-prefix
grant (§1) is the shared floor present in both. No throwaway kmod is built or
preserved; the "B artifact" is a documented module-param setting.

**Recommendation: adopt the config-based B.** (If adjudication prefers a
literally separate B binary, that is also possible — but the config approach is
strictly better-controlled.)

---

## §3 — SC5 measurement plan

**Workload.** 5 PARTITION decode tenants (TinyLlama, the SC3 e2e tenant +
`MigrateHandler`) + 1 POOL, allocate, then free in a randomized order while a
controller drives `COMPACT_MIGRATE` / the kmod evaluates on FREE.

**5 seeds — free-order patterns** (deterministic, named):
`LIFO` (4,3,2,1,0), `FIFO` (0,1,2,3,4), `MIDDLE_OUT` (2,1,3,0,4),
`RAND_A`, `RAND_B` (two fixed random permutations). LIFO is the benign case;
FIFO / MIDDLE_OUT / RAND_* create POOL-unreachable gaps.

**Metrics (→ `TRACK_3_SC5_MEASUREMENTS.md`):**
1. **Correctness** — per-tenant teacher-forced KL ≤ 0.1 (TFGATE), every seed.
2. **POOL stranding** — over the run, free-but-POOL-unreachable SM count.
   *The headline*: **A → ~0 stranded; B → N stranded** (seed-dependent). Read
   directly from `/proc/cipher/migrations` (§4) — the empirical no-stranding
   proof.
3. **Per-tenant decode latency p50/p95/p99** — reported **per opt-in class,
   `migratable` vs `pinned`, separately** (SC1 item-4 adjudication; the
   1.6B-4 headline). The migratable class carries the migration-cost variance
   spike; the pinned class does not.
4. **Migration cost** — per-migration wall time on the affected decode round
   (the SC3 green-ctx rebuild, ~1.7 ms prior); migration count per seed.
5. **A-vs-B** — same 5 seeds, both configs (§2): A's stranding vs B's; A's
   per-tenant latency (with migration cost) vs B's (no migration, but POOL
   capacity stranded). Headline claim: *"A delivers the same correctness as B
   with NO stranding under sustained churn, at a bounded, characterized
   migration cost."*
6. **Destroy-fault at scale** — `CIPHER_SC3_FAULT=destroy` across the
   5-partition churn: every tenant's KL gate must still pass; count the leaked
   old green-ctx handles (the commit-and-leak path, SC3). The leak *rate* under
   sustained driver failure is recorded here; **SC6 documents it as a v1
   boundary** (SC3 item-6 PUSH).

**Item-(c) PUSH:** SC5 empirically validates the production defaults
(gap=1 / sustain=2 s / rate=10 s) under sustained churn. If migration cost
compounds materially under realistic load, SC5 surfaces it and SC6's closeout
carries a retune recommendation.

**migratable-vs-pinned comparison (metric 3) is distinct from A-vs-B
(metric 5):** metric 3 is *within A* — a migratable tenant vs a pinned tenant
under the same churn (the per-opt-in-class variance split); metric 5 is *A the
substrate vs B the substrate*. Both are reported.

---

## §4 — `/proc/cipher/migrations` (Item-e PUSH — full contents)

Read-only, additive on `cipher_proc.c` (the existing `/proc/cipher` provider);
`cipher_cp54_sched.c` exposes a `seq_file` printer it calls. Contents:

```
policy:   gap_min_grps=1  sustain_ms=2000  ratelimit_ms=10000  verbose=1
gap:      stranded_groups=<n>  gap_since_ms=<age>
counters: proposals=<n>  commits=<n>
          aborts: timeout=<n>  tenant_nack=<n>  kmod_refused=<n>
tenants:  pid=<p> qos=<c> cur_mask=0x<m> migrate_state=<s>
          last_outcome=<o> migration_count=<n>
          ... one row per migratable tenant ...
```

Aggregate counters are module-static atomics in `cipher_cp54_sched.c`
(`cp54_stat_proposals` / `_commits` / `_abort_timeout` / `_abort_nack` /
`_abort_refused`), incremented at the PROPOSE / COMMIT / abort sites. This node
is **the SC5/SC6 measurement instrument** — stranding and counters are read
directly, no dmesg parsing.

---

## §5 — SC5-2 / SC5-3 / SC5-4 plan

| phase | scope | est. |
|---|---|---|
| **SC5-1** | this design memo | done |
| **SC5-2** | build (kmod-only): (a) SC4 fold — 4 module params + `cp54_eval_migration` reads them + per-tenant `migration_count` + `/proc/cipher/migrations`; (b) **`cp54_pool_claim_low_prefix`** (§1); (c) the 5-partition randomized-free-order test harness (extends `cp54_s16_orchestrator.py`). Anchor `98da2d1f` → `.pre_sc5` **before any edit**. | **~2 d** (not 1 — §1) |
| **SC5-3** | verify: pre-SC5 baseline → `TRACK_3_SC5_PRE_BASELINE.md`; **SC4-portion regression smoke FIRST** — 15 isolation + 29 SC2-unit + SC3 5/5 unit + SC3 e2e + W1/W2/W3 within ±3 %, dmesg clean (if it regresses → rollback to `.pre_sc5`, STOP, surface, *before* any measurement); then the headline measurement — 5 seeds, A-vs-B, migratable-vs-pinned, destroy-fault at scale → `TRACK_3_SC5_MEASUREMENTS.md`. | ~2–3 d |
| **SC5-4** | closeout: kmod `98da2d1f` → SC5 anchor; fallback preservation; `TRACK_3_SC5_CLOSEOUT.md` with the headline numbers + SC6 recommendations. | ~0.5 d |

**Regression discipline (matches SC2/SC3/SC4):** the SC4-portion smoke is run
**before** the measurement work begins — a regression there stops SC5 at the
build, not after wasted measurement. 15 isolation + 29 SC2-unit + SC3 5/5 +
SC3 e2e + W1/W2/W3 ±3 %, dmesg clean. Any regression → rollback `.pre_sc5`,
surface.

**Files SC5-2 touches:** `cipher_cp54_sched.c` (module params, counter,
constrained grant, eval reads params), `cipher_proc.c` (the /proc node), and
new test harnesses. No ABI change (`cipher_ioctl.h` untouched). libcipher_rt
`83afd1ca`, libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d` unchanged.

**Anchors:** kmod `98da2d1f` → `cipher_kmod_fallback/cipher_kmod.ko.pre_sc5`
before any edit; → SC5 anchor after the SC4-portion smoke PASS, before the
measurement work. The B-branch is a module-param config of the SC5 kmod (§2) —
no separate anchor.

---

## §6 — Adjudication ask

**STOPPING — no source modified, no build, no GPU.** Decisions:

1. **The §1 finding** — accept that SC5-2 builds `cp54_pool_claim_low_prefix`
   (the SC1 fact-2 constrained-low-prefix POOL grant, not yet in the kmod;
   load-bearing for A's correctness under churn). This makes SC5-2 ~2 days.
2. **A-vs-B as a config (§2)** — accept B = the SC5 kmod with
   `cipher_cp54_mig_gap_min_grps=16` (migration disabled), no separate binary —
   or request a literally separate B build.
3. **Measurement plan (§3)** — accept the 5 named seeds, the 6 metric groups,
   and that migratable-vs-pinned (metric 3) and A-vs-B (metric 5) are distinct
   comparisons both reported.
4. **`/proc` contents (§4)** — accept the node layout (policy + gap + aggregate
   counters + per-tenant rows).
5. **SC5-2 ~2-day estimate** — acknowledge the honest expansion from the §1
   finding.

On adjudication: proceed to **SC5-2** (build the SC4 fold + the constrained
grant + the harness), then the SC4-portion regression smoke, then the headline
measurement — closing with `TRACK_3_SC5_CLOSEOUT.md`.
