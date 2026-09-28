# W.4b.7 — per-token rendezvous amortization (option 1) — DESIGN MEMO

**Date:** 2026-05-29. **Status:** DESIGN MEMO — awaiting Anil approval before build
(campaign discipline: design-memo → approve → build, one atomic step, wait between).
**Decision provenance:** W.4b.6 cleared the engagement gate for **branch A**
(homogeneous same-model density, full-handoff executor, 3.239× substrate-attributable).
Anil elected (2026-05-28/29) NOT to close W.4b yet but to **harden the result first
via option (1): amortize the per-token rendezvous** — the smallest atomic step, and
the de-risk for the shared component of branch B / R-4.

**Anchors this step will NOT touch:** cipher_rt_phase4 `8b5e928` (libcipher_rt.so md5
`9fe23143`), cipher_kmod `02fc2d1` (0.6.6), cipher_kv_bridge `5a3db034`. This is a
**userspace harness change only** (the `formb_*` per-token prototype). No substrate /
kmod / bridge edit ⇒ no 9-cell regression backfill required (discipline (a)). If the
amortization turns out to *require* a bridge primitive (e.g. a cuIPC event handle —
see §3 mechanism B), that is a STOP-and-re-scope point, not a silent anchor bump.

---

## §0 — Why this step (the de-risk argument)

The executor-owned-KV **per-token** path (`formb_executor.py` / `formb_tenant.py`,
shipped W.4b.2–.5) does the **identical GPU compute** as the W.4b.6 full-handoff
executor: the executor runs the batched B=N transformer body over executor-owned KV.
The only difference is *who drives the decode loop* — full-handoff drives all GEN
steps internally (banking the barrier), per-token has each tenant ship one embed and
await one logit per step. **Therefore, with a zero-cost rendezvous the per-token path
converges to full-handoff's 0.983× of the in-proc ceiling. The rendezvous IS the
entire gap.**

That same per-token cross-process embed/logit exchange is the component **R-4**
(tenant-owned cuIPC-mapped peer-KV attention, branch B) is built on top of. So:

- If a tight, profiled rendezvous **still** can't bring the per-token path near the
  in-proc ceiling → the heavier R-4 (same rendezvous + a net-new peer-KV attention
  path) is dead on arrival. **Kill R-4 cheaply, before any attention-architecture work.**
- If it **can** → green light for R-4, plus a reusable low-latency rendezvous
  primitive that R-4 inherits.

This is the cheapest possible test of branch B's load-bearing assumption.

---

## §1 — Where the floor is (the measured chain, grounded in source)

Per decode step `t`, steady state, the chain is a strictly serial CPU-mediated
ping-pong (`formb_tenant.py:158-166`, `formb_executor.py:242-256`):

| # | side | op | source | cost class |
|---|------|----|--------|-----------|
| 1 | tenant | `embed(forced)` → [1,h] | tenant:160 | small GPU |
| 2 | tenant | `embeds_out.write(e)` **sync=True ⇒ full `torch.cuda.synchronize()`** | tenant:161 → ipc:94-101 | **host sync (NOT amortized — W.4b.4 only fixed executor scatter)** |
| 3 | tenant | `send_msg({"phase":"decode","t":t})` | tenant:162 | socket syscall |
| 4 | executor | `recv_msg` × R (blocks until all R embeds signalled) | exec:243-244 | R socket syscalls |
| 5 | executor | `gather()` — R cuIPC reads + `.clone()` | exec:245,138-145 | R device copies |
| 6 | executor | batched forward (B=R) | exec:249-250 | the only "real" work |
| 7 | executor | `scatter()` — R cuIPC writes, sync=False | exec:252,147-151 | R device copies |
| 8 | executor | **one `torch.cuda.synchronize()`** | exec:253 | host sync |
| 9 | executor | `send_msg({"done_step":t})` × R | exec:254-255 | R socket syscalls |
| 10 | tenant | `recv_msg` (blocks) | tenant:163 | socket syscall |
| 11 | tenant | `logit_in.read([vocab])` | tenant:164 | device copy |

**Measured: ~79 ms/step at N=4 Mistral-7B (W.4b.5).** An in-process B=4 Mistral decode
forward (step 6 alone) is ~5–15 ms. **⇒ ~60–70 ms/step is pure rendezvous overhead** —
pathologically large for what is nominally a host sync + a few socket messages + small
cuIPC copies. **That gap is unexplained and MUST be decomposed before any fix is chosen.**
Likely suspects (to be confirmed, not assumed): the per-step host sync on *both* sides
draining the pipeline with zero overlap; Python/`pickle`+`struct` socket-message
overhead compounded over 2R messages/step; `gather()`'s `.clone()` forcing a copy +
implicit sync; serial CPU↔GPU latency with no async overlap. **We do not optimize a
number we have not decomposed.**

---

## §2 — Scope boundary (what this step is and is NOT)

**IS:** make the *per-token executor-owned-KV* rendezvous cheap, and re-measure
cross-process tok/W vs the in-proc ceiling at N=4 and N=8 Mistral-7B (the gate regime,
reachable via the W.4b.6 weight-share). Executor-owned-KV compute structure is
**unchanged**.

**IS NOT:** R-4 / branch B (tenant-owned cuIPC-mapped peer-KV attention) — no
attention-path change. Not full-handoff (that already banks the barrier; we are
validating the *per-token* mechanism specifically). Not a substrate/kmod/bridge edit.
Not a new correctness model — the W.4b.3 first-coalesce backstop, the audited W.4a
gate, and the teacher-forced KL / exact-greedy correctness gate are preserved verbatim
and re-asserted (§5).

---

## §3 — Candidate amortizations (chosen AFTER §1 decomposition, not before)

Listed cheapest/safest first. The §6 plan profiles first, then applies only what the
decomposition justifies.

- **A. Kill the redundant host syncs.** Tenant step-2 `write(sync=True)` is a full
  device drain every step; the executor needs the embed *visible before its forward*,
  not the tenant *host-synced*. Replace with stream-ordering: the executor's forward
  stream waits on embed-write completion. Removes 1 host sync/step/tenant. (Pure
  userspace; uses existing buffers.)
- **B. Replace socket signaling with a low-latency doorbell.** 2R `pickle`/socket
  messages/step → a shared-memory ring of atomic counters (`tenant_ready[r]`,
  `executor_done`) in an `mmap`'d `/dev/shm` region, polled (futex or short spin).
  No bridge change. *If* GPU-timeline ordering (not CPU polling) proves necessary,
  the clean form is a cuIPC **event** handle — which **does** touch cipher_kv_bridge
  ⇒ STOP-and-re-scope (it would no longer be an anchors-unchanged step). We try the
  host-doorbell form first precisely to stay userspace-only.
- **C. Overlap gather/scatter copies with compute** via a dedicated copy stream so the
  R cuIPC reads (step 5) and writes (step 7) hide behind the forward (step 6).
- **D. Drop `gather()`'s `.clone()`** if the read-only arena view can be consumed
  directly by the forward without a copy (verify aliasing safety first).

**Irreducible floor (honest):** the autoregressive dependency (tenant needs logit `t`
to embed token `t+1`) means a single sequence cannot pipeline across tokens — there is
a true serial chain tenant→executor→tenant per token. Option (1) makes each link cheap;
it does **not** remove the link. Multi-token-window / speculative pipelining would, but
that is branch-B scope, explicitly out.

---

## §4 — Decision gate (what this step must produce to greenlight or kill R-4)

Re-measure on the W.4b.6 weight-shared harness, **vanilla executor with
`CIPHER_RT_DISABLE_AUTO_INIT=1`** (the W.4b.6 CUPTI-artifact fix — must carry, or we
re-measure the artifact, not the rendezvous):

1. **ms/step decomposition table** (step-by-step attribution of the ~79 ms) — the
   primary scientific deliverable regardless of outcome.
2. **Post-amortization per-token cross-process tok/W** at N=4 and N=8 Mistral, vs the
   in-proc B=N ceiling and the naive-N-concurrent baseline (same triplet as W.4b.6).
3. **Verdict (one of):**
   - **GREENLIGHT R-4** if per-token cross/inproc rises to a meaningful fraction of
     full-handoff's 0.983× (target band to be set with Anil at approval — proposed
     ≥ 0.80× as "the rendezvous is no longer the bottleneck").
   - **KILL/DEFER R-4** if, even fully amortized, cross/inproc stays low (proposed
     < 0.50×) ⇒ the per-token mechanism is structurally rendezvous-bound and R-4's
     extra attention cost cannot help; branch B is v2-research, branch A stands as the
     v1 realization and W.4b closes on branch A.
   - **INCONCLUSIVE** (middle band) ⇒ report honestly, name the residual cost, let
     Anil adjudicate R-4 vs close.

**This step does not itself close or open W.4b** — it produces the number that informs
the R-4-vs-close adjudication, which remains Anil's per campaign discipline.

---

## §5 — Correctness invariant (Memory #11 — non-negotiable, re-asserted every run)

The amortization changes *only how/when signals and copies are ordered*, never the math.
Every post-change run must reproduce the W.4b.5/.6 correctness gate, else HARD STOP:

- exact-greedy token match per tenant (benign near-tie cascades adjudicated by the
  teacher-forced KL test, `formb6_tf_kl.py` pattern — solo top1−top2 gap < ~0.05 ⇒ benign);
- teacher-forced per-step KL ≤ ~7e-5 (the W.4b.6 bar);
- the W.4b.3 first-coalesce backstop still fires on the forced-fault test
  (`CIPHER_FORMB_FAULT`) → BLOCK → solo, no corruption;
- the negative control (distinct-fp model) still rejected by the audited gate
  (`distinct_rejected=1`, zero cross-fp coalesce).

A faster-but-wrong rendezvous (e.g. a missed ordering letting the forward read a
stale/half-written embed) is the central risk of this step and is exactly what these
gates catch.

---

## §6 — Execution plan (7 atomic items; approve before item 1)

1. **Decompose the 79 ms/step.** Instrument the existing per-token harness (per-op
   timers + a few `torch.cuda.Event` spans + an optional `nsys`/`pmon` window) at N=4
   Mistral, weight-shared, vanilla executor + `CIPHER_RT_DISABLE_AUTO_INIT=1`. Produce
   the §4-#1 attribution table. **systematic-debugging discipline: confirm where the
   time goes before touching anything.** (No behavior change; baseline re-confirmed.)
2. **Amortization A (host syncs).** Remove tenant `write(sync=True)`; make the
   executor forward stream-wait on embed visibility. Re-run correctness gate (§5).
3. **Amortization B (doorbell).** Replace socket per-step signaling with the
   `/dev/shm` atomic-counter doorbell (host-polled). Re-run correctness gate.
   *If* GPU-timeline ordering proves required → STOP, surface the bridge-event
   re-scope to Anil (no longer anchors-unchanged).
4. **Amortization C/D (copy overlap / clone elision)** as the decomposition justifies.
   Re-run correctness gate after each.
5. **Re-measure the engagement triplet** (cross-process / in-proc ceiling / naive) at
   **N=4 and N=8 Mistral**, weight-shared, with the artifact-fix env carried; compute
   cross/inproc, in-proc lever, substrate-attributable.
6. **Write `W4B_7_FINDINGS.md`** — the decomposition table, the per-amortization
   deltas, the final triplet, and the §4 verdict (greenlight / kill / inconclusive R-4)
   with the honest residual-cost statement.
7. **Commit** (cipher-fusion-evidence; userspace only, anchors unchanged; author
   `Anil <anil.0666369@gmail.com>`, no co-author trailer per repo convention; tag
   proposal `w4b-7-rendezvous-amort`). Update memory. **WAIT for Anil's R-4-vs-close
   adjudication** — do not proceed to R-4 or to close on my own.

---

## §7 — Risks

- **R-7.1 (correctness, HIGH-impact / LOW-prob-if-gated):** an async/doorbell ordering
  bug lets the forward consume a stale embed → silent corruption. *Mitigation:* §5 gate
  runs after every item; teacher-forced KL + forced-fault backstop catch it; build
  items 2→3→4 incrementally, never stacked.
- **R-7.2 (bridge creep):** the clean async form wants a cuIPC event ⇒ bridge edit ⇒
  anchor bump. *Mitigation:* host-doorbell first (§3-B); a bridge primitive is an
  explicit STOP-and-re-scope, not a silent step.
- **R-7.3 (irreducible floor):** even fully amortized, the autoregressive serial chain
  may dominate at small B. *Mitigation:* that IS a valid (kill/defer) verdict for R-4 —
  it is the answer, not a failure. N=8 (deeper batch, more compute to hide latency
  behind) is the favorable regime and is measured.
- **R-7.4 (doorbell power cost):** busy-spin polling burns CPU → hurts tok/W.
  *Mitigation:* prefer futex/short-spin-then-block; the GPU dominates power, but the
  §4 measurement is tok/W so any regression shows up honestly in the verdict number.
- **R-7.5 (N=8 OOM / harness):** the per-token harness at N=8 weight-shared is newer
  than the full-handoff path. *Mitigation:* reuse the W.4b.6 weight-arena import
  verbatim (lossless, 13.49 GiB proven); N=4 first, then N=8.

---

## §8 — One-line ask

Approve this memo (or adjust the §4 decision bands / the §6 sequence) and I execute
items 1→7, stopping at the §4 verdict for your R-4-vs-close adjudication. No substrate,
kmod, or bridge edit; correctness gate (§5) enforced after every item.
