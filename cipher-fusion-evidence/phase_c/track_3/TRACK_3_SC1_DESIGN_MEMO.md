# Track 3 — Dynamic SM Migration (DSM) — SC1 DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no GPU, no
source modified, no measurement. STOP for adjudication before SC2 build.
Anchors unchanged & preserved: kmod `8d777dfb` → `.pre_track3`, libcipher_rt
`ebc0baaa` → `.pre_track3` (`TRACK_3_ANCHORS.md`); libcipher_v2 `86618c30`,
cipher_kv_bridge `fca6843d` untouched.

---

## §0 — Decision, and three framing facts that make A tractable

**Decision (adjudicated 2026-05-19):** Dynamic SM Migration is the **v1**
substrate primitive — not a v2 deferral. CIPHER will support **live migration
of a PARTITION tenant's SM placement** so the kmod can compact the group ledger
and keep the batch POOL on a maximal contiguous low prefix **under arbitrary
partition churn, with no capacity stranding**. Step 1.6X (the constrained-grant
"B" fix) is **not shipped as the substrate answer**; Step 1.6 measurements
(1.6B-3/1.6B-4) are **deferred** and will be taken *on* the Track 3 substrate.

Three facts govern the whole design. Each is stated again where it bites:

1. **Migration is count-preserving.** A migration moves a partition from one
   group set to another group set **of the same size** — `popcount(new_mask)
   == popcount(old_mask)`, enforced by the kmod at PROPOSE time. Migration is
   **not** resize. This single constraint removes Marlin re-gridding, KV-buffer
   reshape, and the resize/migrate distinction from v1 scope. *Migration moves
   SMs; it does not change how many a tenant has.*

2. **B ⊆ A — the 1.6X-1 design is a component of Track 3, not throwaway.**
   A **retains** the constrained-low-prefix POOL grant from
   `CP_5_4_STEP_1_6X_1_DESIGN_MEMO.md` (`cp54_pool_claim_low_prefix`) as the
   POOL's **safety floor**: a migration takes a non-zero window (PROPOSE →
   COMMIT, bounded by a tenant's poll cadence — seconds), and *during that
   window the ledger layout is fragmented*. The constrained grant keeps the
   POOL valid and disjoint throughout that window. **A's migration is what then
   removes the stranding B alone would leave.** Two composing components, not
   two competing options: B is the floor, migration is the lift.

3. **A is strictly additive — opt-in, default-off.** A tenant is `pinned`
   (never migrated) unless it explicitly declares `migratable` *and* runs the
   poll-and-migrate loop. **With zero migratable tenants, the substrate behaves
   byte-identically to today** (`8d777dfb`/`ebc0baaa`). Every closed CP, every
   Phase B number, every Track 2 result was produced by pinned tenants and is
   reproduced by pinned tenants. This is the structural guarantee behind the
   no-regression constraint: the regression smoke tests (W1/W2/W3) run
   all-pinned and exercise the *unchanged* path.

---

## §1 — The problem (recap, grounded)

`cipher_cp54_sched.c` (`8d777dfb`): the POOL is granted groups; partitions take
8-SM groups exclusively; the green-ctx layer enacts a `grp_mask` as a physical
SM placement (Step 1.3, deterministic per `PHASE_1_3A_PROBE.md`). The POOL's
green context is built with the **count-only** `torch.cuda.GreenContext.create(N)`
(Step 1.3b'), which can only place the POOL on a **contiguous low prefix**.

When partitions free out of LIFO order, free groups appear *above* still-live
partitions. The POOL cannot reach them (count-only ⇒ contiguous-low only).
Result without migration: either the POOL fragments and disjointness breaks
(the unfixed kmod), or — with B — the POOL stays valid but **strands** that
capacity until the intervening partitions happen to exit. Under heterogeneous
partition lifetimes a long-lived low-index partition can strand a *majority* of
the GPU from the POOL — the component serving ~95 % of a 100-tenant deployment.

**Track 3's answer:** when a gap forms, the kmod *proposes* that a partition
above the gap migrate down into it. The partition, cooperatively and at a safe
point, rebuilds its green context on the new groups. The gap closes; the POOL
grows; no stranding. The partition keeps the same number of SMs throughout
(fact 1).

---

## §2 — Architecture: the migration state machine

Migration is a cooperative, kmod-coordinated protocol between the kmod ledger
and a `migratable` partition tenant. Per-tenant migration state in the kmod:

```
  IDLE ──PROPOSE(new_mask)──▶ PROPOSED ──tenant START_MIGRATE──▶ MIGRATING
    ▲                            │                                  │
    │                            │ timeout, no START                │ tenant
    │                            ▼                                   │ ACK_MIGRATE
    └────────────────────────── ABORT ◀──────┐                       ▼
    │                                        │                    COMMIT
    └──────────────────────── (COMMIT done) ─┴───────────────────────┘
```

- **PROPOSED** — the kmod has chosen `new_mask` for tenant *P* and **reserved**
  those groups (state `RESERVED(P)` — see §3d). *P* still owns `old_mask` and
  still runs there. The tenant has not yet acted.
- **MIGRATING** — *P* has called `START_MIGRATE`; the kmod will **no longer
  ABORT** this migration. *P* is now draining + rebuilding its green context.
- **COMMIT** — *P* has called `ACK_MIGRATE` ("I am now running on `new_mask`");
  the kmod atomically frees `old_mask` and promotes `RESERVED(P)`→`PACK(P)` on
  `new_mask`. State returns to IDLE.
- **ABORT** — only reachable from PROPOSED (tenant never called START within
  the timeout). The kmod releases the `RESERVED(P)` reservation; *P* keeps
  `old_mask` untouched. **ABORT is impossible once MIGRATING is entered** — this
  is what removes the post-action abort race (a tenant cannot finish migrating
  and then discover its migration was cancelled). START is the commit point of
  intent; ACK is the commit point of fact.

The protocol's safety rests on: *P* never touches CUDA state until START; the
kmod never reclaims `old_mask` until ACK; and between START and ACK the kmod
holds **both** `old_mask` (owned) and `new_mask` (reserved) for *P*, so no other
tenant can intrude on either.

---

## §3 — SC1 architectural answers (a)–(h)

### (a) kmod→tenant notification mechanism — **recommend poll-based**

**Recommendation: poll-based delivery, reusing the existing QUERY pattern.**
The substrate already has a verified poll loop: `cp54_pool.py.check_resize()`
calls `CIPHER_CP54_QUERY` every decode round (Step 1.3b'). Track 3 adds a
`migrate_state` + `target_mask` to the QUERY result (or a dedicated light
ioctl `CIPHER_CP54_POLL_MIGRATE`). A `migratable` partition tenant polls at its
own **decode-round boundary** — a point where it already synchronizes — and
acts on a PROPOSED state it sees there.

| mechanism | verdict |
|---|---|
| **poll (QUERY-class ioctl)** | **recommended v1** — zero new kmod async infrastructure, reuses a verified path, delivery latency bounded by one decode round (≪ the ≤1/tenant/10 s migration-rate v1 boundary, §3g) |
| eventfd / signalfd push | lower latency, but needs a new fd-registration ioctl, kmod-side `eventfd_signal`, fd-lifecycle failure modes. Unjustified for a ≤1/10 s rate. **v1.5 upgrade path** if SC2 measures poll latency unacceptable. |
| netlink | heaviest; multicast/socket lifecycle; rejected |
| shm ring buffer | needs a shared mapping + cross-process sync; rejected |

**Surface for adjudication:** the choice is poll. **SC2 must *measure* actual
migration-completion latency** (PROPOSE→COMMIT wall time) before SC3 commits to
poll permanently; if it exceeds ~5 s in practice and that is unacceptable,
eventfd is the scoped v1.5 upgrade. The protocol (§2) is delivery-agnostic —
poll vs push changes only how PROPOSED reaches the tenant, not the state
machine.

**Failure mode — tenant unresponsive:** a `migratable` tenant that stops
polling never sees PROPOSED → never calls START → the kmod times out → **ABORT**
(§3c). Safe: an unresponsive tenant is treated exactly as a pinned one for that
proposal. No forced action (§3c).

### (b) libcipher_rt green-ctx refresh primitive

**New API:** `int cipher_rt_green_ctx_migrate(u32 new_mask)` — **count-preserving**
(`popcount(new_mask)` must equal the current mask's popcount; the call rejects
otherwise — fact 1, enforced both kmod-side at PROPOSE and client-side here).

Sequence, all on the calling (tenant) thread, which is at a known safe point:

1. **Drain.** `cuStreamSynchronize` / `torch.cuda.synchronize()` on the
   tenant's green-ctx stream — no kernel of the old green context is in flight.
   (The Step 1.4C cross-stream-race discipline: sync before any swap.)
2. **Build.** Create the new green context from `new_mask` — the *exact* Step 1.3
   path: `cuDevSmResourceSplitByCount(minCount=8)` → select the `new_mask`
   set-bit group resources → `cuDevResourceGenerateDesc` → `cuGreenCtxCreate`.
3. **Swap.** libcipher_rt enforces the partition by setting the green context
   current per-launch (the T4.2.4d mechanism — `cuCtxSetCurrent(green)` in the
   CUPTI launch callback, [[cipher-t424d-enforcement-fixed]]). Migration =
   atomically swap the single "current green ctx" pointer that callback reads,
   and the tenant's green-ctx stream handle. Because step 1 drained and the
   tenant is at a safe point, no launch observes a half-swapped state.
4. **Release.** Destroy the old green context once its stream is idle (it is —
   step 1).
5. **`ACK_MIGRATE` ioctl** — tell the kmod "running on `new_mask`".

**Self-verify (reused):** after the swap, run the `%smid` probe kernel on the
new stream and confirm the observed SM set == `new_mask`'s groups — the same
self-verify `cp54_pool.py` already does (Step 1.3b'). A mismatch raises before
ACK → the kmod times out → ABORT → tenant stays on old. Fail-safe.

**Tenant-side cooperation model — "safe to migrate now":** the tenant **calls
`cipher_rt_green_ctx_migrate()` itself**, from its own loop, at a boundary
where it knows no kernel it cares about is mid-flight (between decode steps /
between `generate()` calls). It is never called asynchronously from a signal
handler. The opt-in loop a `migratable` tenant runs:

```
  at each decode-round boundary:
      st = poll_migrate()                       # QUERY-class ioctl
      if st.state == PROPOSED:
          start_migrate()                       # kmod: PROPOSED -> MIGRATING
          cipher_rt_green_ctx_migrate(st.target_mask)   # drain/build/swap/verify
          ack_migrate()                         # kmod: MIGRATING -> COMMIT
```

**Backward compatibility:** a tenant that never calls `poll_migrate()` /
`migrate()` keeps its first green context for its whole life — today's exact
behavior. The new API is **purely additive** to libcipher_rt; no existing entry
point changes signature or semantics.

### (c) migration policy framework

**WHEN the kmod proposes a migration.** After any FREE/reaper event, the kmod
evaluates ledger contiguity. A migration is proposed when **a `migratable`
partition sits above a free group the POOL cannot reach** — i.e. closing the
gap by moving that partition down would let the POOL's contiguous low prefix
grow. The compaction target is: **all live partitions densely packed against
the high end (group 14 downward)** ⇒ all free groups form one contiguous block
just above the POOL ⇒ POOL grows to maximum. One partition is proposed per
evaluation pass (incremental compaction; the next FREE/tick re-evaluates).

- **Trigger surface:** evaluated on FREE, on the do_exit reaper path (deferred —
  the reaper is atomic-context and lock-free; it sets a "re-evaluate" flag the
  next `cipher_partition_tick` consumes), and on a manual operator ioctl
  `CIPHER_CP54_COMPACT` (operator control / forcing a pass).
- **Rate limiting:** per-tenant `last_migrate_ns`; the kmod will not PROPOSE to
  a tenant within the v1 window (**≤ 1 migration / tenant / 10 s**, §3g). Bounds
  worst-case tenant disruption regardless of churn rate.

**HOW a tenant opts in/out.** A per-tenant flag, **`migratable` vs `pinned`**,
declared at registration (default **`pinned`** — additive, opt-in). v1 uses a
**boolean**, not a numeric budget: a count-preserving migration's cost is
essentially fixed (~one green-ctx rebuild, §3e), so "do you accept occasional
~2 ms stalls" is a yes/no. A **numeric per-tenant cost budget** is deferred to
v2 (§3g) — flagged, not built.

**WHAT on tenant non-cooperation.** **Abort-only.** A `migratable` tenant that
does not `START_MIGRATE` within the timeout → the kmod **ABORTs** the proposal;
the tenant keeps `old_mask`; the POOL accepts the stranding from that tenant
for now (B behavior — the graceful fallback, §3e). **Forced migration is
explicitly rejected for v1** (and likely permanently): yanking SMs from a
tenant that has not quiesced means a green context destroyed under in-flight
kernels — undefined behavior on H100, a correctness violation. The user's spec
listed "forced sync after timeout (degraded safety)" as an option; **this memo
rejects it** — there is no safe forced path; abort is the only sound response.
Surfaced for explicit confirm-or-override.

### (d) failure recovery — the RESERVED state and the reaper

Migration adds a third group state. Encoding (extends the `8d777dfb` scheme —
`0` free, `CIPHER_CP54_GRP_PACK(pid)` owned):

```
  state[g] = 0                              free
           = CIPHER_CP54_GRP_PACK(pid)      owned by pid
           = CIPHER_CP54_GRP_RSVD(pid)      reserved for pid's pending migration
  CIPHER_CP54_GRP_RSVD(pid) = RSVD_BIT | (pid & PID_MASK)   // distinct high bit
```

A migration in PROPOSED/MIGRATING holds, for tenant *P*: `old_mask` groups as
`PACK(P)`, `new_mask` groups as `RSVD(P)`.

**Reaper (the load-bearing correctness path).** `cipher_cp54_release(pid)` runs
in the `do_exit` kprobe — atomic context, lock-free, must not sleep. It must
free **every** group held by a dying pid in **any** migration state. The sweep
becomes, per group, two cmpxchg attempts:

```
  atomic_cmpxchg(&state[g], PACK(pid), 0);   // owned    -> free
  atomic_cmpxchg(&state[g], RSVD(pid), 0);   // reserved -> free
```

— so a crash in **any** phase (PROPOSED, MIGRATING, mid-`migrate()` sync, after
the green-ctx swap but before ACK) leaves no leaked group: both the old set and
the reserved set return to free. The kmod also clears *P*'s `migrate_state`.
This extends the verified Step 1.2 reaper pattern (which the isolation suite
Test 4 already exercises) with one extra cmpxchg per group — additive, same
lock-free model.

| failure | handling |
|---|---|
| tenant crash in PROPOSED | reaper frees `PACK(P)` (old) + `RSVD(P)` (new); migration discarded; POOL reclaims via constrained grant |
| tenant crash in MIGRATING (sync / build / swap) | identical reaper sweep — both sets freed; the tenant's half-built green context dies with the process |
| tenant crash after green-ctx swap, before ACK | identical — reaper frees both; the (now-orphaned) new groups go free, POOL reclaims |
| tenant calls START but never ACKs (hang, not crash) | **no ABORT once MIGRATING** (§2) — the kmod leaves the migration open; the tenant holds both sets; a hung tenant is a stuck tenant, not a ledger corruption. v1: surface via a `cipher_partition_tick` watchdog log. (A MIGRATING watchdog→reclaim is v2; v1 keeps it safe-but-stuck rather than risk reclaiming under a possibly-live green ctx.) |
| `migrate()` self-verify fails (wrong SM set) | tenant does **not** ACK; kmod times out the PROPOSED→… ; tenant stays on old (it never swapped if verify is placed before the pointer swap — see §3b ordering note) |
| COMMIT itself | a kmod operation under `cipher_cp54_lock`; frees `old_mask`, promotes `RSVD→PACK`; pure atomic-array writes, cannot partially fail |
| concurrent crashes (≥2 tenants) | each reaper touches only its own pid's groups (`cmpxchg` keyed on `PACK(pid)`/`RSVD(pid)`) — the D9 concurrent-crash case; disjoint by construction |

**§3b ordering note (correctness):** `migrate()` must run the `%smid`
self-verify **on the new green context before swapping the enforcement
pointer** — verify first, swap only on success. Then a verify failure leaves
the tenant cleanly on the old context. SC3 build must honor this order.

### (e) cost characterization methodology

**Migration cost** = the extra latency on the single decode round where
`migrate()` runs ≈ stream drain (`synchronize`) + green-ctx build + `%smid`
self-verify + stream rebind. Step 1.5 measured a POOL green-ctx rebuild at
**~1.7 ms**; a partition migration is the same operation class, so the prior is
**low single-digit ms, once per migration**.

- **Metric:** per-decode-step wall latency on a `migratable` tenant (the Step
  1.6 §11.2 instrument); identify the step where `migrate()` fired; report that
  step's latency against the tenant's own p50/p95/p99.
- **Comparison:** the *same* tenant, *same* workload, run **`migratable`** (it
  experiences migrations) vs **`pinned`** (it never does) — the delta is the
  migration cost, isolated.
- **Stop condition:** if measured migration cost exceeds the declared budget
  (v1: the boolean's implied "≲ a few ms" expectation), the tenant **falls back
  to `pinned`** — i.e. **B is the per-tenant graceful fallback inside A.** A
  never does worse than B for any individual tenant; it does strictly better
  for the POOL whenever migration succeeds.

**Surfaced for adjudication — the headline-metric consequence.** Migration
injects a variance spike on exactly the axis Step 1.6B-4 measures as the
substrate headline (per-tenant decode-latency variance). Therefore **1.6B-4
must report variance per opt-in class — `migratable` vs `pinned` separately,
not aggregated** — and the Track 3 headline claim becomes *"per-tenant
isolation **with bounded, priced migration cost**,"* not "isolation, full
stop." This is honest and still strong; it must be agreed now so it is not a
surprise in the 1.6B-4 report.

### (f) integration with existing primitives

- **POOL resize (`cp54_pool.check_resize`)** — **unchanged.** The POOL keeps
  the constrained-low-prefix grant (fact 2, the B floor) and its existing
  per-round `QUERY`→rebuild. After a partition migrates down, the POOL's next
  `check_resize` sees a larger contiguous mask and grows — the *existing* path,
  no new code. The POOL is never a migration *subject*; it reorganizes via the
  mechanism it already has.
- **Phase B cross-tenant batching** — **unaffected.** POOL clients are SHARED
  tenants (own no groups, not migrated). The batched `generate()` runs on
  whatever green context the POOL currently holds; Step 1.4's flat 120→40-SM
  lift curve already showed POOL SM count is ±noise in this range. The verified
  Phase B numbers (W1) are POOL-path and reproduce.
- **Track 2 weight sharing** — **orthogonal.** The VMM weight arena
  (`cuMemExport` POSIX-FD) is *device memory*; migration changes *SM placement*.
  A green-ctx swap does not touch memory mappings. SC2's `KL=0.0` weight-arena
  evidence (W3) is migration-neutral and must reproduce.
- **Partition-aware Marlin** — **count-preserving ⇒ grid unchanged** (fact 1).
  Marlin sizes its grid from `cipher_rt_green_ctx_sm_count()`; a count-preserving
  migration leaves that count identical, so the grid is unchanged — only the
  physical SMs differ, which Marlin does not see (it launches a grid; the green
  ctx places it). `migrate()` will defensively re-call Marlin's sizing only if
  a count *did* change — which v1 migration never does.
- **libcipher_v2 / kv_bridge** — untouched by SC1–SC3. **SC4's opt-in-flag
  plumbing** *may* touch the libcipher_v2 tenant-registration path (to carry
  `migratable`); §3h and §4-SC4 flag this — if so, libcipher_v2 `86618c30`
  rotates with preservation. (Alternative: carry `migratable` as an env var
  read by libcipher_rt, exactly as `CIPHER_QOS_CLASS` already is (Step 1.3 Q2)
  — then libcipher_v2 does **not** change. **SC4 recommendation: env var, no
  libcipher_v2 change** — decided at SC4 design, flagged here.)

### (g) v1 boundaries

**v1 supports:** single-GPU H100; the 15-group ledger (≤14 active partitions +
POOL); **count-preserving** migration (compaction only); **poll-based** delivery
at decode-round granularity; **opt-in** `migratable` tenants, `pinned` default;
migration **rate-limited ≤ 1/tenant/10 s**; abort-only on non-cooperation;
B-fallback per tenant if cost exceeds budget.

**v1 explicitly defers to v2** (each acknowledged, not silently dropped):
multi-GPU / NCCL migration; async sub-ms (eventfd push) delivery; **resize**-via-
migration (changing a partition's `sm_count` live — the count-preserving
constraint is a v1 line); migration of POOL/SHARED clients; numeric per-tenant
cost budgets; a MIGRATING-state watchdog→reclaim; KV-cache coherence under
migration *during* cross-tenant batching (v1 does not migrate POOL clients, so
this does not arise in v1).

### (h) backward compatibility verification

The additive/opt-in structure (fact 3) is the compatibility guarantee:

- A tenant with no migration support **never** declares `migratable`, **never**
  polls, **never** calls `migrate()` → it is `pinned` → the kmod never PROPOSEs
  to it → it keeps its first green context for life → **today's exact behavior**.
- All current Phase B, Step 1.6, and Track 2 harnesses run pinned and unchanged.
  The `cp54_s16_*` harnesses are touched only if a Track 3 *migration probe* is
  added — and that is **additive** (a new optional hook), per the user's
  constraint.
- The regression smoke test (W1/W2/W3, §4) is run **all-pinned after every SC**
  — it exercises precisely the unchanged path, so a smoke-test pass *is* the
  backward-compat proof at each milestone.

---

## §4 — SC2–SC6 plan & per-milestone regression discipline

Every SC closes with the adjudicated discipline: **(1)** anchor preservation
verified (pre-SC md5s recorded in `TRACK_3_ANCHORS.md`); **(2)** smoke test —
**W1** Phase B N=8 TinyLlama mean-vs-mean within ±3 % of 491.8 tok/s,
**W2** one PARTITION + POOL concurrent (1.6B-2A pattern) clause1=clause2=0 + KL
gates, **W3** Track 2 SC2 VMM weight-arena evidence chain, **dmesg clean**;
**(3)** rollback path documented (exact `rmmod`/`insmod` of `.pre_track3`, or
libcipher_rt `.pre_track3` restore, + post-rollback W1/W2/W3 re-pass);
**(4)** the SC's own primitive test; **(5)** adjudication checkpoint. **Any
smoke-test regression → STOP, rollback, surface — no next SC until resolved.**

| SC | scope | anchor | est. |
|---|---|---|---|
| **SC1** | this design memo | none | ~1–2 d (done) |
| **SC2** | kmod notification + migration state machine: `RSVD` group state, `migrate_state`/`target_mask` per-tenant, ioctls `SUBSCRIBE_MIGRATE`/`POLL_MIGRATE`/`START_MIGRATE`/`ACK_MIGRATE`/`COMPACT`, PROPOSE/timeout/COMMIT/ABORT logic, reaper extension (§3d). Additive ioctls only ([[cipher-abi-rule]]); existing ioctl semantics unchanged; `cipher_cp54_lock` discipline preserved. **Measure PROPOSE→COMMIT latency** (§3a). | kmod `8d777dfb`→SC2 | ~2–3 d |
| **SC3** | libcipher_rt `cipher_rt_green_ctx_migrate(new_mask)` (§3b): drain / build / verify / swap / release; the poll-and-migrate handler; opt-in flag. Verify-before-swap order (§3d). | libcipher_rt `ebc0baaa`→SC3 | ~3–4 d |
| **SC4** | migration policy: gap detection, rate limiting, `migratable` plumbing (**recommend env var — no libcipher_v2 change**, §3f), abort fallback. | kmod→SC4 (if policy kmod-side) | ~2 d |
| **SC5** | end-to-end: 5 partitions × 5 randomized free-order seeds; migration triggers; all TFGATE KL gates pass; per-tenant latency p50/p95/p99; `migratable` vs `pinned` comparison; POOL contiguous under arbitrary churn (no stranding); **vs B-baseline** (see open item below). | none (verification) | ~3 d |
| **SC6** | cost characterization + CP 5.5 integration: migration-cost curves vs churn rate; A-vs-B under sustained churn; deployment guidance (which tenants opt in); `TRACK_3_CLOSEOUT.md`. | none | ~2 d |

**Total Track 3 ≈ 13–18 days** build + verification (SC1 included). Consistent
with the ~15–20 day realistic estimate; **Track 3 will not complete for Peak XV**
(§6).

**Open item for SC1 adjudication — SC5's B-baseline.** SC5 compares A to "B as
baseline." Two ways to source it: **(i)** build B (`cp54_pool_claim_low_prefix`
alone, the 1.6X-1 memo) in a throwaway branch purely for the comparison — no
anchor rotation, ~0.5 d, gives measured A-vs-B numbers; **(ii)** characterize
B's stranding analytically (it is deterministic — POOL = `[0,K-1]`,
`K`=lowest-partition) and compare A's measured curve to that analytic curve —
cheaper, but invites "show the numbers." **Recommend (i)** — honest measured
comparison, and the B branch is anyway a subset of the SC2/SC4 kmod. Flagged so
SC5's scope is fixed before SC2 starts.

---

## §5 — Track 2 vs Track 3 sequencing

**Recommend serial: Track 3 primary, Track 2 SC3–SC6 after** (the user already
declared Track 3 the primary Phase 5 workstream). Rationale: Track 3 rotates
**kmod** and **libcipher_rt**; Track 2 SC3+ is likely to touch **libcipher_rt**
and/or **cipher_kv_bridge** — concurrent rotation of a shared artifact is a
merge hazard and breaks the clean per-SC anchor lineage the regression
discipline depends on.

**Carve-out:** parallel is safe **only if** Track 2 SC3–SC6 is provably pinned
to `cipher_kv_bridge` + weight-arena code and touches **neither** the
libcipher_rt SM/green-ctx path **nor** the kmod ledger. If the user wants
parallelism, that constraint must be confirmed first. Default recommendation:
**serial, for anchor-lineage clarity.** Adjudication item.

---

## §6 — Peak XV framing (timing acknowledged)

Peak XV is ~24 h out; Track 3 is ~13–18 days. **Track 3 will not be complete
for Peak XV** — and the honest pitch is the stronger one:

- **Present capability:** CP 5.1/5.2/5.3/5.6 closed + CP 5.4 through Step
  1.6B-2A — kmod SM-arbitration ledger, kmod-driven variable green contexts,
  POOL/executor binding, the flat 120→40-SM lift curve, ~1.7 ms green-ctx
  churn, verified mixed-deployment disjointness.
- **Architectural milestone in active build:** Track 3 DSM — *"live SM
  migration under arbitrary multi-tenant churn: a substrate-layer primitive no
  production GPU stack provides today."* This reframes Peak XV from
  "production-ready today" to **architectural-roadmap with a working substrate
  underneath** — genuine systems depth, not incremental-optimization claims.

The pitch is honest (the substrate exists; DSM is in build) and it is what
distinguishes CIPHER from application-layer multi-tenancy.

---

## §7 — Anchors

Preserved 2026-05-19 at authorization (`TRACK_3_ANCHORS.md`, md5-verified):
`cipher_kmod_fallback/cipher_kmod.ko.pre_track3` (`8d777dfb`),
`cipher_rt_fallback/libcipher_rt.so.pre_track3` (`ebc0baaa`). libcipher_v2
`86618c30` (`.v0.2.0`) and cipher_kv_bridge `fca6843d` unchanged; libcipher_v2
rotates only if SC4 rejects the env-var route (§3f — recommended route changes
nothing). SC1 rotates **nothing** — paperwork. Rotation log appended to
`TRACK_3_ANCHORS.md` per SC.

---

## §8 — Adjudication ask

**STOPPING HERE — no source modified, no build, no GPU.** Decisions:

1. **Architecture (§2/§3)** — accept the PROPOSE → START → MIGRATING → ACK →
   COMMIT state machine; the `RSVD(pid)` third group state + the two-cmpxchg
   reaper sweep (§3d); **count-preserving** migration (fact 1); **B ⊆ A** —
   the constrained-low-prefix POOL grant retained as the safety floor (fact 2).
2. **Notification (§3a)** — accept **poll-based** delivery for v1, with SC2
   *measuring* PROPOSE→COMMIT latency before SC3 commits; eventfd as the
   scoped v1.5 upgrade path.
3. **Non-cooperation (§3c)** — accept **abort-only**; confirm the explicit
   rejection of forced migration (no safe forced path exists).
4. **Headline-metric consequence (§3e)** — accept that 1.6B-4 reports
   variance **per opt-in class** (`migratable` vs `pinned`); the Track 3
   headline is "isolation + bounded, priced migration cost."
5. **v1 boundaries (§3g)** — confirm the v1/v2 line, especially
   count-preserving-only and rate ≤ 1/tenant/10 s.
6. **SC5 B-baseline (§4)** — choose (i) measured throwaway B branch
   [recommended] or (ii) analytic.
7. **Sequencing (§5)** — confirm Track 3 serial-primary, Track 2 after (or
   confirm the parallel carve-out constraint).
8. **SC4 opt-in plumbing (§3f)** — note the recommendation to carry
   `migratable` as an env var (no libcipher_v2 change); final call at SC4.

On adjudication: proceed to **SC2** (kmod notification + state machine), closing
with the §4 regression discipline. No code until this memo is adjudicated.
