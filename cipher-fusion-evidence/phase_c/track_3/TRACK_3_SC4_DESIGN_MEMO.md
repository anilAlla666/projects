# Track 3 — Dynamic SM Migration — SC4 DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no source
modified, no GPU, no measurement. STOP for adjudication before SC4-2 build.
**Predecessor:** SC3 closed (`TRACK_3_SC3_CLOSEOUT.md`; kmod `98da2d1f`,
libcipher_rt `83afd1ca`).

SC4 is the **policy framework** layered on the SC2 mechanism + SC3 primitive.
Both lower layers are built and verified — SC4 adds no new migration capability;
it makes the policy SC2 shipped as fixed `#define` conservative defaults
**operator-tunable** and gives the operator **visibility**. SC4 is kmod-only
and rotates the kmod anchor `98da2d1f`.

---

## §0 — What is already done (SC4 is genuinely thin)

The SC1 memo scoped SC4 as "gap detection, rate limiting, migratable plumbing,
abort fallback." The adjudicated SC2/SC4 split and SC3 already shipped most of
that:
- gap-detection + rate-limit **mechanism** — SC2 (`cp54_eval_migration`);
- `migratable` opt-in **plumbing** — SC3 (`CIPHER_MIGRATABLE` env);
- abort fallback — SC2 (`ABORTED_TIMEOUT` / `ABORTED_TENANT_NACK`).

**What genuinely remains for SC4:** make SC2's three policy scalars tunable
instead of `#define`-frozen, and add operator visibility. That is the whole of
SC4. It is correctly a small SC — flagged honestly so the scope is not
inflated.

---

## §a — Operator interface — **recommend kmod module parameters**

The policy knobs are operator-level scalars (a gap threshold, a sustain
duration, a rate limit, a verbosity flag) — not per-process state.

| option | verdict |
|---|---|
| **`module_param`, perm 0644** *(recommend)* | exposed at `/sys/module/cipher_kmod/parameters/*`; **boot-time** settable via `insmod cipher_kmod.ko <name>=<v>` **and runtime** writable via `echo > /sys/...`. One mechanism, zero new ABI surface, no ioctl nr consumed. |
| new runtime-config ioctl | a new ABI nr (21+), a permanent ABI commitment, copy_from_user plumbing — unjustified for four operator scalars with no per-process semantics. Rejected. |
| both | redundant — 0644 module params already give boot + runtime. |

**Recommendation: module parameters, mode 0644.** The operator-forced
`COMPACT_MIGRATE` ioctl (nr 20) already exists for forcing an evaluation pass —
SC4 needs **no new ioctl**, so the ABI is frozen at nrs 1-20.

## §b — Per-tenant cost budget — **recommend defer to v2**

SC1 §3c already deferred numeric per-tenant cost budgets to v2. SC4 holds that
line. Reasons:
- A count-preserving migration's cost is ≈ one green-context rebuild — roughly
  fixed and small (~1.7 ms prior, Step 1.5; SC3 confirmed the path). A numeric
  budget discriminates poorly on a near-constant cost.
- The kmod cannot estimate a tenant's migration cost *before* proposing — it
  has no model of the tenant's kernel state. Option (i) (kmod skips
  over-budget proposals) needs information the kmod does not have.
- Option (ii) (tenant NACKs if too costly) — the tenant only learns the cost
  *after* running `migrate()`, by which point it is MIGRATING; a "too costly"
  late NACK is just the existing NACK path with no new budget machinery.

**v1 cost control = the two levers already in place:** the operator-wide
**rate limit** (now tunable, §a) bounds migration frequency; the **boolean
`migratable`/`pinned`** opt-in is the per-tenant escape — a tenant for which
migration is too costly simply does not set `CIPHER_MIGRATABLE` (the B-floor).
A numeric budget is a v2 refinement (it would extend the `SUBSCRIBE_MIGRATE`
payload or add a per-tenant env var — flagged, not built).

## §c — Production-realistic defaults — **recommend, with reasoning**

SC2's conservative defaults: gap ≥ 3 groups (24 SM), sustain ≥ 5 s, rate
≥ 1/tenant/10 s. SC4 ships these as module params; the **recommended default
values**:

| param | SC2 conservative | SC4 default | reasoning |
|---|---|---|---|
| `mig_gap_min_grps` | 3 (24 SM) | **1 (8 SM)** | Any POOL-unreachable free group is real lost capacity for the component serving ~95 % of tenants. **Incremental count-preserving compaction is monotone — a migrated partition moves toward the high end and never back; it cannot oscillate.** So an aggressive gap threshold cannot thrash; it only makes compaction prompt. |
| `mig_sustain_ms` | 5000 | **2000** | Filters transient gaps (a partition that frees and is immediately re-allocated). 2 s is well below session-scale partition lifetimes yet absorbs churn bursts. |
| `mig_ratelimit_ms` | 10000 | **10000** (unchanged) | Bounds worst-case per-tenant disruption regardless of churn; the SC1 §3g v1 boundary. Keep. |

**These are defaults, not hard-codes** — the operator dials conservatism per
deployment via §a. **SC4 ships the knobs + reasoned defaults; SC5's measured
A-vs-B and per-tenant latency characterization is what validates or retunes
them.** Flagged: the numbers above are reasoned, not yet measured — SC5 may
adjust, and that is expected, not a failure.

## §d — Multi-partition migration per pass — **recommend one-per-pass for v1**

SC2 proposes one partition (the lowest blocker) per evaluation pass; the next
FREE/COMPACT re-evaluates. SC4 **keeps one-per-pass for v1.** Reasons:
- It is the proven SC2/SC3 behaviour (the SC3 e2e migrated 2 tenants this way).
- The state machine already supports multiple tenants concurrently MIGRATING
  (per-tenant state) — one-per-pass limits only *proposals* per pass, and eval
  runs on **every FREE** plus any COMPACT, so compaction converges fast (the
  SC3 e2e controller issued 103 passes in ~12 s).
- Multi-per-pass trades a faster compaction front for a burst of simultaneous
  ~1.7 ms stalls — an optimization with a real downside.

Multi-per-pass is a **v2 item** (or an operator param if SC5 shows convergence
is too slow). v1 default: one-per-pass. Surfaced for adjudication.

## §e — Operator visibility — **recommend a read-only /proc node + counters**

Build, additively on `cipher_proc.c` (the existing `/proc/cipher` provider):
- **`/proc/cipher/migrations`** — read-only: the policy params in effect; the
  current POOL-unreachable gap state; and per-`migratable`-tenant rows
  (`pid`, `migrate_state`, `last_outcome`, `migration_count`).
- **per-tenant `migration_count`** — a new `u32` in `struct cipher_cp54_alloc`,
  incremented on each COMMIT. Additive, kmod-internal.
- **dmesg verbosity** — a `mig_verbose` module param (0/1, default 1) gating
  the `pr_info("… migrate PROPOSE …")` lines, so a busy production deployment
  can quiet them.

All read-only / additive — no ABI surface, no new ioctl.

## §f — Policy ↔ crash recovery — **document only, no code**

The do_exit reaper (SC2 two-cmpxchg sweep) frees a crashed tenant's PACK +
RSVD groups unconditionally — it is **policy-independent**. No policy param
gates or alters it. SC4 changes nothing here; `TRACK_3_SC4_CLOSEOUT.md` will
state explicitly that crash recovery is unaffected by policy tuning.

## §g — Env var alignment — **recommend no new env vars**

`CIPHER_MIGRATABLE` (tenant opt-in, SC3) stays the **only** Track 3 tenant env
var. Per §b no cost-budget env var is added. Clean separation: **policy is
operator-level (kmod module params); opt-in is tenant-level (one env var).**
A per-tenant budget, if v2 adds one, is where a second env var or a
`SUBSCRIBE_MIGRATE` payload extension would land — not v1.

---

## §1 — SC4-2 / SC4-3 / SC4-4 plan

| phase | scope | est. |
|---|---|---|
| **SC4-1** | this design memo | done |
| **SC4-2** | build (kmod-only): convert SC2's 3 policy `#define`s to `module_param` (0644) + `mig_verbose`; `cp54_eval_migration` reads the params; per-tenant `migration_count`; `/proc/cipher/migrations` node. Anchor `98da2d1f` → `.pre_sc4` **before any edit**. | ~1–2 d |
| **SC4-3** | verify: pre-SC4 baseline (W1/W2/W3) → `TRACK_3_SC4_PRE_BASELINE.md`; tests — threshold settings change the trigger rate as expected; rate limit prevents a migration storm; `/proc/cipher/migrations` reports correctly; **regression: 15 isolation + 29 SC2-unit + SC3 5/5 unit + SC3 e2e all re-PASS**; smoke W1/W2/W3 within ±3 %; dmesg clean. | ~0.5 d |
| **SC4-4** | closeout: kmod `98da2d1f` → SC4 anchor; fallback preservation; `TRACK_3_SC4_CLOSEOUT.md`; update `TRACK_3_ANCHORS.md` + `cipher_anchors_manifest.txt` + memory. | ~0.5 d |

**Files SC4-2 touches:** `cipher_cp54_sched.c` (module params + counter + eval
reads params), `cipher_proc.c` (new /proc node). Untouched: `cipher_ioctl.h`
(no ABI change), `cipher_dev.c`, libcipher_rt, libcipher_v2, cipher_kv_bridge.

**Regression discipline (matches SC2/SC3):** 15 isolation + 29 SC2-unit + SC3
5/5 unit + SC3 e2e all re-PASS; W1/W2/W3 within ±3 % of the SC4 pre-baseline;
dmesg clean. Any regression → STOP, rollback to `.pre_sc4` kmod + SC3
libcipher_rt, surface.

**Anchors:** kmod `98da2d1f` → `cipher_kmod_fallback/cipher_kmod.ko.pre_sc4`
before any edit; → SC4 anchor only after smoke PASS. libcipher_rt `83afd1ca`,
libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d` unchanged.

---

## §2 — Adjudication ask

**STOPPING — no source modified, no build, no GPU.** Decisions:

1. **Interface (§a)** — accept kmod **module parameters** (mode 0644), no new
   ioctl; ABI frozen at nrs 1-20.
2. **Cost budget (§b)** — accept **defer numeric per-tenant budget to v2**;
   v1 cost control = tunable rate limit + boolean opt-in.
3. **Defaults (§c)** — accept the reasoned production defaults (gap ≥ 1 group,
   sustain 2 s, rate 10 s) as module-param *defaults*, SC5 measurement may
   retune — or set different starting values.
4. **Multi-per-pass (§d)** — accept **one-per-pass for v1**, multi-per-pass
   deferred to v2.
5. **Visibility (§e)** — accept the read-only `/proc/cipher/migrations` node +
   per-tenant `migration_count` + `mig_verbose` param.
6. **Crash recovery (§f)** / **env vars (§g)** — accept: no policy/crash-
   recovery coupling (document only); no new env vars.

On adjudication: proceed to **SC4-2** (build), closing with the §1 regression
discipline and the kmod anchor rotation.
