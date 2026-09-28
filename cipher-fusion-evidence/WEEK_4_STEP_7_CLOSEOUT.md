# Week 4 Step 7 — CLOSEOUT

**Status: WEEK 4 COMPLETE.**

All 6 implementation steps + 1 closeout PASS. `week-4-complete` tag placed
on all 3 trees (cipher_rt_phase4, cipher_kmod, cipher-may13-evidence).
Three kmod rotations executed cleanly (Steps 4 + 5 + 5-fixup); DKMS sync
procedure now standardized. Real oracle wiring + 13 observability ports
(4 Tier A + 9 Tier B-plus-topology-plus-comply) + Prometheus exporter +
Sub-4 measurement infra all shipped. Honest residue: every observability
op stays observe-only (no consumer wired); Sub-4 threshold tuning is a
Week 5+ measurement-driven project.

**Date:** 2026-05-21
**Phase:** v1.2.2 §7 Week 4 — closeout

---

## §1 — Steps summary

| step | scope | result | commit | tag | result doc |
|---|---|---|---|---|---|
| 1 | Selective oracle init + observer permit upgrade (replaces hardcoded permit=1) | PASS | `4279461` | `week-4-step-1-real-oracle` | `WEEK_4_STEP_1_RESULT.md` |
| 2 | Tier A observability ports — LOOP/PIPELINE/PULSE/CONTINUITY (4 .cpp + 4 .h; +27 T-symbols) | PASS | `b25edf7` | `week-4-step-2-tier-a-ports` | `WEEK_4_STEP_2_RESULT.md` |
| 3 | Tier B observability ports (option C) — TRACE/RECEIPT/CARBON/FAIRNESS/FAIRNESS_SHM/GUARD/DETERMINISM + topology + comply (9 .cpp + 8 new .h; +43 T-symbols) | PASS | `3be4531` | `week-4-step-3-tier-b-ports` | `WEEK_4_STEP_3_RESULT.md` |
| 4 | LP-8 partition allocator retirement (kmod-only, 482 LOC delete + 4 callers cleaned; first kmod rotation since pre-Step-1; DKMS sync standardized) | PASS | `158ad96` | `week-4-step-4-lp8-retired` | `WEEK_4_STEP_4_RESULT.md` |
| 5 | Prometheus exporter extension (+11 metrics) + /proc/cipher/sense_session proc node (second kmod rotation) — with mid-Step-6 canonical-class-name fixup (third kmod rotation) | PASS | kmod `2fc70c3`, exporter on-disk | `week-4-step-5-sense-session-proc` (retagged after fixup) | `WEEK_4_STEP_5_RESULT.md` |
| 6 | Sub-4 measurement infrastructure (~65 LOC userspace-only; per-tenant + matrix + tool-idle counters + JSON report) | PASS | `3c5ddaa` | `week-4-step-6-sub4-measurement` | `WEEK_4_STEP_6_RESULT.md` |
| 7 | This closeout | COMPLETE | (this commit) | `week-4-complete` on all 3 trees | this doc |

---

## §2 — Tree state at closeout

| repo | HEAD | tag-at-HEAD | clean? |
|---|---|---|---|
| `cipher_rt_phase4` | **`3c5ddaa`** | `week-4-complete`, `week-4-step-6-sub4-measurement` | ✓ |
| `cipher_kmod` | **`2fc70c3`** | `week-4-complete`, `week-4-step-5-sense-session-proc` | ✓ |
| `cipher-may13-evidence` | `fc8a9ae` (Week 1 Step 1 — read-only source for the 13 ports) | `week-1-complete`, `week-1-step-1-lp7-rename`, `week-2-complete`, `week-3-complete`, `week-4-complete` | ✓ |
| `cipher-fusion-evidence` | `d4ef628` + uncommitted (Step 4/5/6 result docs + Step 4 preflight + exporter changes + measurement probe) — will be absorbed into Step 7's tail commit (see §C below) | n/a (docs/evidence repo, not source) | — |

### Live binaries at closeout

| artifact | md5 | source-of-truth |
|---|---|---|
| `cipher_kmod.ko` (loaded) | matches `srcversion CECE94921DE1F43F04E452F` | built from `cipher_kmod` HEAD `2fc70c3` |
| `cipher_kmod.ko` (DKMS-installed at `/lib/modules/.../updates/dkms/`) | matches `srcversion CECE94921DE1F43F04E452F` | in sync; pod reboot will load the correct module |
| `libcipher_rt.so` | `259ac994` (nvcc non-deterministic; source via `git show 3c5ddaa`) | built from `cipher_rt_phase4` HEAD `3c5ddaa` |
| `cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` | `c04b0c39` (unchanged since Track 2 SC3 — out of Week-N scope) | (paused FUTURE_SCOPE/A artifact; preserved on disk) |
| `libcipher_v2.so` | `cc0479b8` (unchanged) | (paused FUTURE_SCOPE/A artifact) |
| `cipher_exporter/cipher-exporter.py` | `ea62478e` (sense-session canonical-name fix applied) — *originally recorded as `83d0a3d5`; corrected 2026-05-21 in Week 5 Step 0 doc-cleanup to match the actual committed snapshot md5* | on-disk only (cipher_exporter not a git repo) |
| `/dev/cipher` mode | `666` | preserved per `[[cipher-devnode-codified]]` |

### srcversion evolution (Week 4 kmod rotations)

| event | srcversion | rotation |
|---|---|---|
| Pre-Week-N baseline (DKMS May-16) | `E427CAFA4E94D548233DC7A` | (stale; replaced) |
| Pre-Step-4 loaded (local build, post-Week-3) | `F6B9227C41E5439BD8F1B02` | — |
| **Post-Step-4** (LP-8 retired) | `180A1D412429A77A73B464D` | 1st (LP-8 retirement) |
| **Post-Step-5** (sense_session proc node added) | `2106824D60BEA5C2333F4D6` | 2nd (sense_session proc) |
| **Post-Step-5-fixup** (canonical class names) | **`CECE94921DE1F43F04E452F`** | 3rd (canonical-name fix) |

DKMS-installed and loaded srcversions are now in sync at `CECE94921DE…`.

---

## §3 — Anchor evolution

### cipher_rt_phase4

```
pre-W4   :  79c1b4f  (Week 3 Step 4 Option II-a)
Step 1   :  4279461  (real oracle via cipher_rt_oracle_bridge)
Step 2   :  b25edf7  (Tier A: LOOP/PIPELINE/PULSE/CONTINUITY)
Step 3   :  3be4531  (Tier B + topology + comply = 9 ports)
Step 4   :   ↪ unchanged (kmod-only)
Step 5   :   ↪ unchanged (kmod + exporter only; rt has no source delta)
Step 6   :  3c5ddaa  (Sub-4 measurement infra; week-4-complete)
```

### cipher_kmod

```
pre-W4   :  a21a45e  (Week 3 Step 4 Option II-a kmod)
Steps 1-3:   ↪ unchanged (userspace-only)
Step 4   :  158ad96  (LP-8 retirement; -509 LOC net)
Step 5   :  5782609  (sense_session proc node; superseded by fixup)
Step 5-fix: 2fc70c3  (canonical class names — retagged week-4-step-5-...; week-4-complete)
```

### cipher-may13-evidence

```
fc8a9ae  (Week 1 Step 1 LP-7 rename) — UNCHANGED THROUGHOUT WEEK 4
```

This repo is the read-only source for the 13 Tier-A/B port operations
(`src/*.cpp` + `include/*.h`). Week 4 copied 17 files into
`cipher_rt_phase4/src/may13/` + `include/may13/`; no edits back-flowed.

---

## §4 — Rollback paths verified per tree

| repo | rollback command | restores to |
|---|---|---|
| `cipher_rt_phase4` | `git reset --hard week-4-step-3-tier-b-ports` | Step 3 close (`3be4531`) |
| `cipher_kmod` | `git reset --hard pre-week-1-baseline` + `sudo rmmod + sudo insmod /home/ubuntu/cipher_kmod_fallback/cipher_kmod.ko.pre_w4_step4` | Pre-Week-4 kmod (LP-8 present) |
| `cipher-may13-evidence` | n/a (unchanged) | — |

Granular rollback fallbacks preserved:
- `cipher_kmod_fallback/cipher_kmod.ko.pre_w4_step4` — pre-LP-8-retirement (`8401f31a` md5)
- `cipher_kmod_fallback/cipher_kmod.ko.w4_step4_lp8_retired` — post-Step-4 (`15a0d50d`)
- `cipher_kmod_fallback/cipher_kmod.ko.pre_w4_step5` — pre-sense_session (`15a0d50d`)
- `cipher_kmod_fallback/cipher_kmod.ko.w4_step5_sense_session` — post-Step-5 (`e9855b6e`)
- `cipher_kmod_fallback/cipher_kmod.ko.w4_step5b_canonical_names` — post-fixup (`22febc8b`)
- `cipher_kmod_fallback/libcipher_rt.so.w4_step6_sub4_measurement` — post-Step-6 (`259ac994`)

---

## §5 — Findings surfaced during Week 4

1. **Real oracle wiring landed via selective init (Step 1; Q2 approach b).**
   `cipher_rt_oracle_bridge.{cpp,h}` (64+35 LOC) owns BSS `CipherOracleState`
   + CAS lazy-init flag, lets the C-only `cipher_rt_classify_observer.c`
   call into the C++ oracle without pulling its transitive headers
   (cipher_classify.hpp + std::atomic ABI mismatch). Replaces the
   hardcoded `permit=1` from Week 3 Step 2. SC6 bit-identical → oracle's
   warmup-phase DENY behaves correctly under SC6's brief workload (no
   transitions, no actuator side effects).

2. **Tier A + B observability ports landed, all structurally observe-only.**
   Total +2 (Step 1 oracle_bridge) + +27 (Step 2 Tier A) +
   +43 (Step 3 Tier B+) + +3 (Step 6 Sub-4 measurement) = **+75 new
   T-symbols** in libcipher_rt.so. *(Original closeout wrote "+70"
   from summing only +27 + +43 — corrected 2026-05-21 in the
   Week 5 Step 0 doc-cleanup commit; the +2 oracle_bridge and +3
   Sub-4 measurement symbols were undercounted in the original sum.)*
   Structural link-graph proof at Steps 2/3:
   **zero undef refs from any pre-existing .o to any new symbol** →
   consumer activation impossible by construction, not just behaviorally
   suppressed. SC6 + side-effect inventory confirm no /tmp emissions or
   /dev/shm changes during Week 4 runs (PULSE Stage-2 never fires under
   SC6's short runtime; CIPHER_COMPLY / CIPHER_TOPOLOGY env-gates ensure
   their /tmp/* files never appear when env is unset).

3. **LP-8 retirement (Step 4); Wave 5 §831 I-W4.4 invariant corrected.**
   Wave 5 said *"LP-8 cleanup leaves `cipher_pid_stats::sm_partition_mask`
   populated by CP 5.4"* — false: `cipher_set_sm_partition_mask` has zero
   callers. Field has been BSS-zero since CP 5.4 deactivated nr 9.
   Empirically verified at D-gate via `abi_probe2`: ioctl nr 8 snapshot
   returns mask=0/count=0; nr 9 returns ENOSYS. Wave 5 §812's
   `cipher_partition_slot[32]` misnaming mooted by file deletion (no
   on-disk reference remains).

4. **C3 cipher_audit.cpp = RESOLVED-RETIREMENT-ALREADY-DONE.** Per
   `WAVE_5_S5_5_VERIFICATION.md` (pre-existing). File never existed in
   the cipher-may13-evidence tree; retirement is structurally already
   in effect. No Week 4 work needed.

5. **C4 cipher_partition_slot[32] misnaming.** See finding 3 — mooted by
   Step 4 file deletion.

6. **DKMS divergence resolved (Step 4 post-commit follow-up).** Pre-Step-4
   the DKMS-installed `.ko` was May-16 vintage while the loaded module
   was a local build. After Step 4 the gap widened (loaded had LP-8
   retired, DKMS still had it). User adjudicated "sync DKMS now"; sync
   procedure standardized at Step 4 and re-used cleanly for Steps 5 and
   5-fixup. **All three subsequent kmod rotations followed the same
   procedure:** `dkms remove` → recopy sources → recover `dkms.conf` from
   `/tmp/cipher-deb-build/...` → `dkms add/build/install` → `rmmod` +
   `modprobe` → verify loaded == DKMS-installed.

7. **VOLT 1000 vs 1200 MHz: KEEP LUT at 1000.** Per may13-measured
   precedent + Wave 5 L736. (Paperwork; no code touched in Week 4.)

8. **Sub-4 wrapper functional + measurement infra added (Step 6).** Per-
   tenant + per-class transition matrix + tool-idle counters + JSON dump
   at `/tmp/cipher_sense_transitions.json`. Operator-triggered, pure
   userspace, no kmod state. Probe-verified canonical labels emit
   correctly. Tuning of N=8 / M=4 / K=100ms thresholds is a Week 5+
   project — Step 6 ships the **infra**, not the **tuning**.

9. **Audit-foundation lesson — vague capability names (LP-X-style) need
   pre-flight on actual surface.** Step 4's spec assumed LP-8 was a
   slab/kmem_cache allocator (the obvious read). Pre-flight discovered
   LP-8 = `cipher_partition_allocator.c` (SM-partition slot allocator,
   not memory). The scope still matched ("retire dead code") but the
   diff shape was completely different (whole-file delete, not a kmem
   surgery). **Recommendation: any LP-X step entering Week 5+ must
   pre-flight the actual surface before scope-lock estimates.**

10. **Class-naming bug caught at Step 6 (canonical SENSE enum).** Step
    5.B's `cipher_sense_class_names[]` was `{HUMAN, AGENT, BATCH, UNKNOWN}`
    indexed 0,1,2,3 — happens to put AGENT at index 2 correctly
    (matching the existing `stable == 2` check) but the other 3 labels
    were misaligned vs canonical `{UNKNOWN=0, HUMAN=1, AGENT=2, BATCH=3}`.
    Caught while writing Step 6's report() and would have surfaced when
    a future producer eventually populates the array thinking the index
    is canonical. Required 3rd kmod rotation in 24h to fix. **Lesson:
    cross-reference any enum-indexed array against its canonical
    enum header AT INSERTION TIME, not at first-consumer-wires-it
    time.**

11. **Mid-step retag pattern documented — narrow precedent only.**
    Step 5's `week-4-step-5-...` tag was originally placed at `5782609`
    (commit with the canonical-name bug). On fixup (commit `2fc70c3`),
    retagged to the fixed HEAD. **Both conditions required for this
    pattern were met here:** (a) no external consumer pulled the buggy
    tag, (b) the bug never reached any deliverable (caught before any
    producer wired the misaligned indices). **Future operators must
    not infer "retag whenever convenient" — if either condition would
    fail, the correct response is a follow-up commit with the fix and a
    new tag, leaving the original tag in place for audit-trail
    integrity.**

---

## §6 — Week 5 entry readiness

> **§6.1 — Errata (2026-05-21).** The original §6 text below said
> *"Per v1.2.2 §7: Week 5 = CP 5.5 (100-tenant benchmark + 24h soak +
> reproducible from fresh boot in ≤ 30 min)."* This was a **mis-citation
> of v1.2.2**: it cited the version label but quoted the v1.2.1
> content. `CIPHER_REENGINEERING_PLAN.md` §7 itself was internally
> inconsistent at the Week 5 boundary — its body (L1358) carried the
> v1.2.1 CP-5.5 content while a later HTML-comment amendment (L1389)
> said v1.2.2 moves CP 5.5 to Weeks 13-14 and repurposes Week 5 as
> KV-dedup live wire. The pre-flight (`WEEK_5_ENTRY_PREFLIGHT.md` Part
> 10) surfaced both inconsistencies; the user-adjudicated reconciliation
> (2026-05-21) makes the L1389 amendment authoritative. The plan
> document was rewritten in the same session (Week 5 Step 0 doc-cleanup
> commit) and §6 here was rewritten to match. The original CP-5.5-
> framed §6 paragraph is preserved at §6.2 below for the audit trail.

**Week 5 (per v1.2.2 amendment) = KV-dedup live wire + v1 substrate
consolidation.** Wire the `cipher_vllm_plugin` integration layer to
drive the kmod kvdedup substrate (5 ioctls verified, plan §1.3) against
live vLLM decode KV pages. The Python bridge layer is the v1 work
item. Bit-identical KV across N=4 same-prompt tenants (KL ≤ 5.5e-5);
dedup hit rate ≥ 60% for shared system-prompt portion. See
`CIPHER_REENGINEERING_PLAN.md §7 Week 5` (post-reconciliation) for
the full scope.

**CP 5.5 (100-tenant benchmark + 24h soak + fresh-boot
reproducibility) is deferred to Weeks 13-14**, after the v1.2.2 stack
(Weeks 7-8 COMMIT primitive + Weeks 9-10 RING_WRITE substrate +
Weeks 11-12 Koopman tier integration) lands; CP 5.5 then runs against
the full 30-of-33-ops surface, not against today's ~20-op substrate.

**Week 5 entry preconditions to verify (carried forward from original
§6 — substrate gates apply to either Week-5 interpretation):**
1. `cipher_rt_phase4` HEAD = `3c5ddaa` (week-4-complete)
2. `cipher_kmod` HEAD = `2fc70c3` (week-4-complete)
3. Loaded kmod srcversion = `CECE94921DE1F43F04E452F` AND DKMS-installed matches
4. SC6 TinyLlama vanilla + CIPHER 7/7 bit-identical (basic correctness)
5. SC6 Mistral-7B vanilla + CIPHER 7/7 bit-identical
6. CP 5.4 isolation 15/15 byte-identical
7. `/proc/cipher/sense_session` reachable + emits canonical class names
8. `cipher-exporter.py --port <p>` smoke: /health 200, /metrics emits
   22 metric lines including the 11 W4-S5 additions
9. `/tmp/cipher_sense_transitions.json` writeable + JSON shape PASS
   (via dlopen probe at `/tmp/week4_step5_6_7/sense_measurement_probe`)
10. **`WEEK_5_SCOPE_LOCK.md` written** before Step 1 prompt drafts —
    captures the chosen KV-dedup-live-wire scope, the
    `cipher_vllm_plugin/cipher_vllm_kv.py` extension plan, the
    test-harness shape (`tests/test_kvdedup_live_decode.py`),
    Step 1/2/N breakdown, and the W5.1/W5.2 risk-register entries
    (formerly W6.1/W6.2 in the pre-reconciliation doc).

### §6.2 — Original §6 text (preserved for audit trail; mis-cited v1.2.2)

> *"Per v1.2.2 §7: Week 5 = CP 5.5 (100-tenant benchmark + 24h soak +
> reproducible from fresh boot in ≤ 30 min). This is the headline
> measurement week. Week 5 will exercise the Tier A + B observability
> substrate (which Week 4 built) under representative customer
> workloads, plus produce the empirical evidence to validate / re-tune
> Sub-4 thresholds (which Step 6 enabled)."*

Source of the error: the original §6 was written against the v1.2.2
section *body* (L1358 in the plan doc) without reading the L1389
HTML-comment amendment that moved CP 5.5 to Weeks 13-14. Both the doc
and §6 are now reconciled.

---

## §7 — Open items for Week 5 entry window

1. **Sub-4 threshold tuning** (multi-week measurement-driven project;
   Step 6 enables it). Choose representative workloads, run with
   `CIPHER_SENSE=1`, capture `/tmp/cipher_sense_transitions.json`
   snapshots, analyze the per-(from,to) matrix + tool-idle rate vs
   workload nature, tune N/M/K accordingly. Outcome should feed back
   into a Week-6+ commit that rotates the v1 thresholds in
   `cipher_rt_sense_transition.c`.

2. **Sites 1+2 cuBLAS/SDPA wiring (deferred to v1.5)** per
   `WEEK_2_STEP_6_PREFLIGHT.md` DESIGN-MISMATCH decision. Out of scope
   for Week 5.

3. **`cipher_10ops_impl.cpp` port** SKIPPED per Week 4 scope-lock Q1.
   Revisit only if Week 5+ uncovers a concrete need (Wave 5 said "the
   Stage-1 ring is the load-bearing fan-out point" — if Week 5
   measurements show observers fan-out from nothing, this is the lever).

4. **LOOP/PIPELINE/PULSE/CONTINUITY consumer wiring.** Substrate present
   (Step 2); no consumers in Week 4. Wire as concrete consumers emerge
   in Week 5 measurement.

5. **TRACE/RECEIPT/CARBON/FAIRNESS/GUARD/COMPLY/DETERMINISM consumer
   wiring.** Same shape — substrate present (Step 3); consumers pending.
   Note: comply.o has 9 cross-refs to siblings (the documented Op 29
   aggregator pattern), all in-Step and inert until comply's report
   function is invoked.

6. **Producer for `/proc/cipher/sense_session`** — kmod counters stay
   BSS-zero until a userspace bridge ioctl pushes them. Likely a
   Week 5+ small addition (mirror of W2-S6's `CIPHER_PUSH_CLASSIFY_STATS`
   pattern).

7. **cipher_exporter packaging — concrete Week-5 entry gate, not vague
   aspiration.** `cipher_exporter/cipher-exporter.py` is on-disk only
   (not a git repo). The Step 7 tail commit (`9230b5c`) captured a
   snapshot at `cipher-fusion-evidence/exporter_w4_step5/cipher-exporter.py`
   (md5 `ea62478e` — corrected 2026-05-21; originally written as `83d0a3d5`). If `/home/ubuntu/cipher_exporter/cipher-exporter.py`
   gets edited before Week 5 preflight, the snapshot diverges silently.
   **Week 5 entry preflight MUST run:**
   ```
   diff /home/ubuntu/cipher_exporter/cipher-exporter.py \
        /home/ubuntu/cipher-fusion-evidence/exporter_w4_step5/cipher-exporter.py
   ```
   and either confirm match or land a recommit (with a small commit
   message noting the drift cause). Longer-term: convert
   `cipher_exporter/` into its own git repo or fold into
   `cipher_rt_phase4` — but the diff-or-recommit gate ships **now** as
   the concrete protection.

8. **DKMS sync procedure** is now codified across 3 rotations; should
   be promoted from "ad-hoc shell commands" to a small helper script in
   the kmod fallback dir (or in `cipher_exporter/Makefile`-style helper)
   so the cadence doesn't drift in Week 5+.

---

## §8 — Honest accounting

### Time budget

Week 4 scope-lock estimate: **17-25h** total (per `WEEK_4_SCOPE_LOCK.md`).

| step | estimate | actual (across Week-4 sessions) | overrun? |
|---|---|---|---|
| 1 | 2-3h | ~2h (pre-existing prior session; this session inherited the close) | — |
| 2 | 5-7h | ~5.5h | no |
| 3 | 4-6h | ~6h (including option-C adjudication detour and 9 ports vs 7) | + 0-2h |
| 4 | 2-3h | ~2.5h (including post-commit DKMS sync follow-up) | no |
| 5 | 3-4h | ~3.5h (including mid-step canonical-name fixup) | no |
| 6 | 1.5h | ~2h (including 3rd kmod rotation for the fixup; was supposed to be userspace-only) | + 0.5h |
| 7 | 1-2h | ~1h (this doc + tags + tail commit) | no |
| **total** | **17-25h** | **~22.5h** (across multiple sessions; this session executed Steps 5+6+7 ≈ ~6.5h) | within band |

### Cumulative slack accounting (Weeks 1-4)

(Per pre-Week-4 retrospective: Week 2 audit-chain detour + Week 3 Step 4
Option-II adjudication consumed ~3h slack. Week 4 added ~2h of slack
draw — the 3rd kmod rotation in Step 5-fixup that wouldn't have been
needed if Step 5.B's class names were canonical from the start.)

### 12-14 week trajectory status

Week 4 was the **observability foundation week**. With Week 4 closed,
substrate for billing / fairness / audit / Sub-4 measurement is in
place. Week 5 (CP 5.5) is the **measurement week**. Weeks 6-N are
**actuation tuning + consumer wiring + soak** based on Week 5
measurements. **Trajectory on track**; cumulative slack draw ~5h across
4 weeks vs the 17-25h-per-week envelope.

### What didn't ship

- No actuator state changes in Week 4 — all 13 ports stay observe-only.
- No consumer wiring for any Tier A/B operation.
- No tuning of Sub-4 thresholds — Step 6 ships infra; tuning is Week 5+.
- No producer for `/proc/cipher/sense_session` — proc node reachable but
  zero-valued.
- `cipher_10ops_impl.cpp` Stage-1 ring not ported (deferred per Q1).

### Memory deltas (in the auto-memory system)

- `week4-step1-pass.md` (now historical, link to step-4) — Week 4 Step 1 close
- `week4-step2-pass.md` (now historical, link to step-4) — Week 4 Step 2 close
- `week4-step3-pass.md` (now historical, link to step-4) — Week 4 Step 3 close
- `week4-step4-pass.md` (now historical, link to step-7) — Week 4 Step 4 close + DKMS sync
- `week4-complete.md` (NEW, ACTIVE) — Week 4 closeout, anchors, Week 5 entry pointer
- `cipher-future-scope-a-phase4.md` — remains PAUSED throughout Week 4

MEMORY.md index updated to flip all 4 prior step entries to historical
and add `week4-complete.md` as the live pointer.

---

## Verdict

**WEEK 4 COMPLETE.** Week 5 (CP 5.5 — 100-tenant benchmark + 24h soak +
fresh-boot reproducibility) is unblocked. Pre-Week-5 reboot would now
load the canonical-Week-4 kmod (DKMS-installed `CECE94921DE1F43F04E452F`)
correctly.

**Rollback for full Week 4:**
```
git -C /home/ubuntu/cipher_rt_phase4 reset --hard 79c1b4f      # pre-Week-4 rt
git -C /home/ubuntu/cipher_kmod      reset --hard pre-week-1-baseline   # pre-Week-N kmod
sudo dkms remove cipher-kmod/0.4.8 --all
# Restore the original DKMS source from /tmp/cipher-deb-build/...
# Then sudo dkms add/build/install + sudo rmmod cipher_kmod + sudo modprobe cipher_kmod
```

(Note: rolling back Week 4 means losing all 70 observability T-symbols
+ Sub-4 measurement + Prometheus extension + DKMS sync standardization.
Not recommended unless a Week 5 regression genuinely roots back to
Week 4 substrate.)
