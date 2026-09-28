# Session State Reconciliation — 2026-05-21

**HEADLINE: CROSS-CAMPAIGN-AMBIGUITY.**

Two campaigns coexist on this pod with **conflicting anchors and disjoint git
state**:

- **Workstream A — Week-N integration sequence** is the *committed, active*
  campaign. Week 4 Step 1 **PASSED** 2026-05-20 19:27–19:29; Week 4 Step 2
  (Tier A ports) is the next step per `WEEK_4_SCOPE_LOCK.md`.
- **Workstream B — FUTURE_SCOPE/A (vLLM-composition)** has real artifacts
  on disk (Track 2 SC1-SC6, Track 3 SC2/3/5, FUTURE_SCOPE/A Phases 1-4 docs)
  but **is uncommitted in the cipher-fusion-evidence git history** and
  references kmod/libcipher_rt anchors that **do not match the live binaries
  Week-N just built**. The Phase 4 design memo at
  `future_scope_a/FUTURE_SCOPE_A_PHASE_4_DESIGN_MEMO.md` (2026-05-20 19:49)
  is untracked, written *after* the Week 4 Step 1 commit, against stale
  auto-memory.

The auto-memory (`~/.claude/projects/-home-ubuntu/memory/MEMORY.md`) is
entirely Workstream B; there are **no Week-N memory entries**. The user's
prompt framing in this turn is Workstream A. The two workstreams are not
just diverged — they imply different next actions, and the disagreement
must be adjudicated before either resumes.

---

## PART 1 — Tree states across all repos

| repo | exists | git | HEAD | dirty | tags-at-HEAD |
|---|---|---|---|---|---|
| `/home/ubuntu/cipher_rt_phase4` | yes | yes | `4279461` *Week 4 Step 1: real cipher_oracle_decide via selective bridge* | **clean** | `week-4-step-1-real-oracle` |
| `/home/ubuntu/cipher_kmod` | yes | yes | `a21a45e` *Week 3 Step 4 Option II-a (kmod): CIPHER_DSM_PROPOSE ioctl + proposals queue* | **clean** | `week-3-complete`, `week-3-step-4-opt2a-dsm-propose` |
| `/home/ubuntu/cipher-may13-evidence` | yes | yes | `fc8a9ae` *Week 1 Step 1: LP-7 struct rename* | **clean** | `week-1-complete`, `week-1-step-1-lp7-rename`, `week-2-complete`, `week-3-complete` |
| `/home/ubuntu/cipher-fusion-evidence` | yes | yes | `e2084da` *W4 Step 1 result: real oracle wired via selective bridge* | **DIRTY — 56 files** (phase_c SC6 logs modified or deleted by Week-4-Step-1 SC6 reruns; plus the untracked `future_scope_a/FUTURE_SCOPE_A_PHASE_4_DESIGN_MEMO.md`) | none at HEAD; tags exist only for `pre-week-1-baseline*` |
| `/home/ubuntu/libcipher_v2` | yes | **NOT a git repo** | — | — | — |
| `/home/ubuntu/cipher_kv_bridge` (standalone) | **does not exist** | — | — | — | — |
| `/home/ubuntu/cipher_vllm_plugin` | yes | **NOT a git repo** | — | — | — |
| `/home/ubuntu/cipher_rt_fallback` | yes | **NOT a git repo** (2 .pre_* files) | — | — | — |
| `/home/ubuntu/vllm_env` | yes | **NOT a git repo** (venv) | — | — | — |

**Recent commits (cipher-fusion-evidence, last 15):**

```
e2084da W4 Step 1 result: real oracle wired via selective bridge
850bd8b W4 scope-lock: 6 steps, ~1315 LOC, ~17-25h
3ff4070 Week 4 pre-flight — WEEK 4 ENTRY READY (1 deferred decision)
c475058 Week 3 Steps 4 Option II-a + 5 Closeout: WEEK 3 COMPLETE
dca172d Week 3 Step 4 Option II pre-flight — SCOPE-COMPRESSED
ac4be8e Week 3 Step 3 RESULT (PASS) + Step 4 RESULT (PARTIAL — Shape 3 surfaced)
1521203 Week 3 Step 2 RESULT (PASS) — observer TLS substitute_hint publish landed
0f3c733 Week 3 Step 1 RESULT (PASS) — cipher_rt_dispatch scaffold landed
42092af Week 3 SCOPE LOCKED — 5-step decomposition + SC6 cadence + CIPHER_DISPATCH_LIVE progression
f47e54b Week 3 pre-flight — WEEK 3 ENTRY READY
bced53e Week 2 Steps 6 + 7 — Closeout: WEEK 2 COMPLETE
c40b0a8 Week 2 Step 6 pre-flight — SCOPE-COMPRESSED: drop Sites 1+2 (DESIGN-MISMATCH)
73db5ea Week 2 Steps 4+5 RESULT (both PASS)
a2756d3 Week 2 Step 3 RESULT (PASS) — classify_substrate.{cpp,h} scaffold landed
548eb19 Week 2 Step 3 — telemetry-substrate impedance check: COMPOSE-CLEAN
```

**All 15 are Week-N integration commits, all 2026-05-20.** No FUTURE_SCOPE/A,
phase_c, or Track 2/3 commits in the active lineage.

---

## PART 2 — Week 4 Step 1 outcome — **PASS**

| signal | result |
|---|---|
| **2.1 WEEK_4_STEP_1_RESULT.md** | exists — `/home/ubuntu/cipher-fusion-evidence/WEEK_4_STEP_1_RESULT.md` (13196 B, May 20 19:28). **HEADLINE: "Status: PASS."** Both load-bearing SC6 gates (TinyLlama + Mistral-7B, vanilla + CIPHER-injected) clear bit-identical; CP 5.4 isolation 15/15; real oracle confirmed live under CUPTI smoke. |
| **2.2 Tag** | `week-4-step-1-real-oracle` → commit `4279461743…` (msg: "Week 4 Step 1: real cipher_oracle_decide via selective bridge", 2026-05-20 19:27:29 UTC). |
| **2.3 Source landed** | `cipher_rt_phase4/cipher_rt_oracle_bridge.h` (1168 B, May 20 19:20) and `cipher_rt_oracle_bridge.cpp` (2003 B, May 20 19:21). |
| **2.4 observer.c diff vs `week-3-step-4-opt2a-dispatch-live-sense`** | 69-line diff. New include of `cipher_rt_oracle_bridge.h`; the hardcoded `permit=1` is replaced by a real call into `cipher_rt_oracle_bridge_decide()` (Q2 selective-init approach b — bridge owns BSS `CipherOracleState` + CAS lazy-init flag; `cipher_oracle_init(state, NULL, NULL)` is pure-function safe per `oracle.cpp:80-119`). |
| **2.5 `/tmp/week4_step1/` logs** | 6 SC6 .log files, **all end with `"PASS": true` followed by `"SC6 …: PASS"`**: TinyLlama vanilla+CIPHER (pre and post), Mistral-7B vanilla+CIPHER (post both). The user's framing in this prompt (*"46 sec into a 3-minute timeout, no output captured"*) is **contradicted by the on-disk evidence** — the Mistral-7B CIPHER-injected SC6 completed and PASSED at 19:26. |
| **2.6 libcipher_rt.so + nm** | md5 `d60c625692a1c2b9dcf56ce0ebbe8999` (post-Week-4-Step-1, May 20 19:22); `nm -D` shows **`cipher_rt_oracle_bridge_decide`** and **`cipher_rt_oracle_bridge_init_lazy`** exported (T) — the bridge was built into the live .so. |

**Conclusion for Part 2:** Week 4 Step 1 reached the result-doc + commit +
tag stage cleanly. The Mistral-7B SC6 run the user thought was interrupted
actually completed and PASSED. **Next per `WEEK_4_SCOPE_LOCK.md` §85: Step 2
— Tier A observability ports (4 files, ~970 source LOC + headers, ~5-7h).**

---

## PART 3 — FUTURE_SCOPE/A campaign inventory

**The directory `/home/ubuntu/cipher-fusion-evidence/future_scope_a/` exists**
with 39 entries (5 .md docs + JSON/log evidence from Phase 2/3/3.5 GPU runs).

**.md inventory (newest first):**

| file | size | mtime | verdict in head |
|---|---|---|---|
| `FUTURE_SCOPE_A_PHASE_4_DESIGN_MEMO.md` | 17501 B | **2026-05-20 19:49** | "paperwork only … STOP for adjudication before Phase 4 build" |
| `FUTURE_SCOPE_A_PHASE_3_5_RESULTS.md` | 5281 B | 2026-05-19 15:50 | "DVFS composes with vLLM … +13.9 %, below 1.5× gate" |
| `FUTURE_SCOPE_A_PHASE_3_RESULTS.md` | 5689 B | 2026-05-19 15:33 | "PARITY PASS (production mode) … vLLM-on-CIPHER within −0.39 % of vLLM-alone" |
| `FUTURE_SCOPE_A_PHASE_2_RESULTS.md` | 4712 B | 2026-05-19 15:18 | "TRANSPARENCY PASS — vLLM 0.21.0 runs coherently under CIPHER injection" |
| `FUTURE_SCOPE_A_DESIGN_MEMO.md` | 18367 B | 2026-05-19 15:03 | "Phase 1 of FUTURE_SCOPE/A. STOP for adjudication before Phase 2" |

**Phase C (Track 2/3) docs are present:** `phase_c/` has 148 files including
`TRACK_2_CLOSEOUT.md`, all SC1-SC6 closeouts, plus `track_3/` with
`TRACK_3_CLOSEOUT.md` and SC2/SC3/SC5 closeouts.

**Companion binaries the FUTURE_SCOPE/A docs reference:**

- `cipher_kv_bridge` (built into `cipher_rt_phase4/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so`, md5 `c04b0c39…`, 373752 B, May 19 10:19). The `.so` is a symlink to the cpython artifact — the "48 B" file size is the symlink, not a corrupt binary.
- `libcipher_v2/libcipher_v2.so` (md5 `cc0479b8…`, 16968 B, May 13 10:05). Plain dir, not a git repo.
- `cipher_vllm_plugin/` (CP 5.1 KV-allocator plugin) — plain dir, not a git repo.
- vLLM 0.21.0 isolated venv at `/home/ubuntu/vllm_env/`.

**git tracking status:** the FUTURE_SCOPE/A docs and `phase_c/` are present
in the working tree but **only the initial baseline commit `c43eec0`
("Baseline snapshot pre-Week-1 integration")** touches them. No subsequent
commit in cipher-fusion-evidence's lineage has modified, added to, or
referenced these directories. The Phase 4 design memo authored this session
(2026-05-20 19:49) is **untracked**.

**Conclusion for Part 3:** FUTURE_SCOPE/A is **real** — the artifacts are
authentic, not hallucinated. But it is *off-git* relative to the active
cipher-fusion-evidence commit lineage: present as working-tree files only,
preserved by the baseline import but never integrated into the
Week-N-as-the-active-campaign timeline.

---

## PART 4 — Cross-campaign artifact reconciliation

### 4.1 Commit-source classification

- Commits touching `future_scope_a/`: **1** — `c43eec0 Baseline snapshot pre-Week-1 integration`. Nothing since.
- Commits touching `phase_c/`: **1** — same baseline.
- Commits touching `WEEK_*` docs: **all 15 recent commits** are Week-N.
- `FUTURE_SCOPE_A_PHASE_4_DESIGN_MEMO.md` (this session's deliverable): `git status` reports **`Untracked`**.

### 4.2 Trees touched

| campaign | repos | shared with the other? |
|---|---|---|
| Week-N (committed) | `cipher_rt_phase4` (HEAD), `cipher_kmod` (HEAD), `cipher-may13-evidence` (HEAD), `cipher-fusion-evidence` (HEAD) | yes — same tree (cipher_rt_phase4) where Week-N rebuilt libcipher_rt is also where Track 2's `cipher_kv_bridge.so` lives |
| FUTURE_SCOPE/A (off-git) | `cipher-fusion-evidence` (untracked dirs `future_scope_a/`, `phase_c/`); `cipher_rt_phase4` (cipher_kv_bridge.so artifact); `libcipher_v2/`; `cipher_vllm_plugin/`; `vllm_env/` | yes — same trees, but cites *different kmod and libcipher_rt anchors* that no longer exist on disk |

### 4.3 Live-binary anchors vs each campaign's expectation

| binary | live md5 | Week-N HEAD-source-commit | FUTURE_SCOPE/A Phase 4 memo cite | match? |
|---|---|---|---|---|
| `cipher_kmod/cipher_kmod.ko` | **`8401f31a`** (mtime 2026-05-20 18:38) | `a21a45e` (Week 3 Step 4 II-a kmod) | `008b3c66` (Track 2 SC5) | **NEITHER** matches a labelled anchor; `8401f31a` is the build artifact from `a21a45e` source |
| `cipher_rt_phase4/libcipher_rt.so` | **`d60c6256`** (mtime 2026-05-20 19:22) | `4279461` (Week 4 Step 1) | `83afd1ca` (Track 3 SC3) | **Week-N matches** (rebuilt post-Step-1); FUTURE_SCOPE/A cite is **stale** |
| `cipher_rt_phase4/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` | `c04b0c39` | n/a (Week-N didn't touch) | `c04b0c39` (Track 2 SC3) | **FUTURE_SCOPE/A matches**; Week-N is silent |
| `libcipher_v2/libcipher_v2.so` | `cc0479b8` | n/a | `cc0479b8` | **matches FUTURE_SCOPE/A**; Week-N silent |
| **DKMS-loaded `cipher_kmod.ko`** | `6654d9e5` | **pre-Week-1 baseline** | n/a | The kernel-loaded module is **older than either campaign's HEAD source** — neither has been built-and-loaded since `pre-week-1-baseline`. |

**Anchor reconciliation:** the FUTURE_SCOPE/A Phase 4 memo's anchor block
(`kmod 008b3c66 / libcipher_rt 83afd1ca / libcipher_v2 cc0479b8 /
cipher_kv_bridge c04b0c39`) is **half stale**: `libcipher_v2` and
`cipher_kv_bridge` are correct (Week-N hasn't touched them), but `kmod` and
`libcipher_rt` no longer correspond to any artifact on disk — Week-N
overwrote them. The DKMS-loaded kmod is older still (pre-Week-1).

### 4.4 Timing

| event | timestamp |
|---|---|
| `cipher-may13-evidence` HEAD | 2026-05-20 (Week 1 Step 1 rename) |
| `cipher_kmod` HEAD | 2026-05-20 **18:46:06** (Week 3 Step 4 II-a) |
| Live `cipher_kmod.ko` mtime | 2026-05-20 18:38:51 (built from a slightly earlier checkout — pre-amend?) |
| `cipher_rt_phase4` HEAD | 2026-05-20 **19:27:29** (Week 4 Step 1) |
| Live `libcipher_rt.so` mtime | 2026-05-20 19:22:16 (built ~5 min before the commit was made — pre-commit build, consistent with the result-doc-then-commit sequence) |
| `cipher-fusion-evidence` HEAD | 2026-05-20 **19:29:10** (W4 Step 1 result) |
| **Phase 4 memo write** | 2026-05-20 **19:49** (untracked, ~20 minutes *after* Week 4 Step 1 was committed) |
| User prompt arrived | 2026-05-21 |

**Read:** Week-N work landed in a clean sequence ending 19:29. **20 minutes
later**, this session (operating from auto-memory that contains no
Week-N entries) wrote the FUTURE_SCOPE/A Phase 4 design memo citing
*pre-Week-N* anchors, without realising the live tree had moved.

---

## PART 5 — Honest summary

### Q1 — Did Week 4 Step 1 PASS, FAIL, or never complete?

**PASS.** Five independent signals: result doc with "Status: PASS"
(`WEEK_4_STEP_1_RESULT.md`, 13 196 B, 19:28); commit `4279461` tagged
`week-4-step-1-real-oracle` (19:27:29); source files
`cipher_rt_oracle_bridge.{h,cpp}` landed (19:20–19:21); observer.c diff vs
`week-3-step-4-opt2a-dispatch-live-sense` shows the real `cipher_oracle_decide`
wiring (69-line diff, hardcoded `permit=1` removed); `/tmp/week4_step1/` has
6 SC6 logs all ending "PASS"; nm exports `cipher_rt_oracle_bridge_decide` and
`_init_lazy`. **The user's framing in this prompt that the Mistral-7B SC6
run was interrupted at 46 s is contradicted by `/tmp/week4_step1/sc6_post_mistral_cipher.log`,
which contains the full PASS verdict at 19:26.**

### Q2 — Is FUTURE_SCOPE/A a separate active campaign, a remnant, or hallucinated?

**REAL artifacts, OFF-GIT, and stale-anchored.** The Track 2/3 + FUTURE_SCOPE/A
Phase 1–3.5 docs exist on disk; `cipher_kv_bridge.so` (`c04b0c39`) and
`libcipher_v2.so` (`cc0479b8`) match their cites verbatim. vLLM 0.21.0 is
genuinely installed at `/home/ubuntu/vllm_env`. But *no commit in
cipher-fusion-evidence's lineage after the initial baseline touches any of
this*; the directory has been carried in the working tree only. The Phase 4
design memo authored this session is **untracked**, references kmod and
libcipher_rt anchors that no longer exist on disk (Week-N rebuilt both), and
was written ~20 min *after* the Week 4 Step 1 commit by a session operating
from auto-memory that knew nothing about Week-N.

### Q3 — What is the user's most recent intended workstream?

**Week-N integration sequence — Week 4 Step 2 (Tier A observability ports) is
next.** Most recent commits (all 15 in cipher-fusion-evidence) are Week-N;
most recent tag (`week-4-step-1-real-oracle`) is Week-N; most recent
substantive result doc (`WEEK_4_STEP_1_RESULT.md`) is Week-N; live binaries
are Week-N; `WEEK_4_SCOPE_LOCK.md` §85 names Step 2 as "Tier A observability
ports (4 files, ~970 source LOC + headers, ~5-7h)". The user's framing in
this prompt — "Week 4 Step 1 was running selective oracle init + observer
permit upgrade … 46 sec into a 3-minute timeout" — is itself Week-N
vocabulary, confirming Week-N is the operating mental model. The Phase 4
memo this session produced is a **divergent artifact** that was not in the
user's intent for that session and is not in the user's intent now.

### Unresolved cross-campaign question (surfacing only, no mitigation)

Auto-memory and the committed cipher-fusion-evidence git history disagree
about which campaign is active. Auto-memory says FUTURE_SCOPE/A Phase 4
is the next step; git+filesystem+result-docs say Week 4 Step 2 is the next
step. The Phase 4 design memo authored 2026-05-20 19:49 is the concrete
collision point: it exists on disk, is untracked, and cites anchors that
the Week-N builds have invalidated. **Options for the user to pick from:**

- **(a) Auto-memory is stale.** Week-N is the real campaign; the Phase 4
  memo is the product of a session that drifted onto stale context. Resume
  Week-N at Step 2 (Tier A ports). Decide what to do with the
  `future_scope_a/` directory and the untracked Phase 4 memo (delete,
  archive, or keep as parked work).
- **(b) Auto-memory is correct; Week-N is itself the divergent branch.**
  Reconcile by acknowledging both: Week-N may need to integrate the Track
  2/3 substrate, or vice versa. The Phase 4 memo's anchor block must be
  updated to reference the live binaries (kmod `8401f31a`, libcipher_rt
  `d60c6256`), and the SC5 fd-custodian + Track 3 DSM features the memo
  assumes "already shipped" must be re-verified against the Week-N kmod
  source (which is at `a21a45e` Week 3 Step 4 II-a — no Track 2 SC5
  ioctls visible).
- **(c) Both campaigns are intended to coexist.** Then a reconciliation
  pass is required to identify which kmod/libcipher_rt build the operator
  intends to load at runtime, and how the Track 2/3 features stack with
  the Week-3-step-4 dispatch + Week-4-step-1 selective oracle wiring.

**This diagnostic does not propose which option is correct — only that the
disagreement is real, is concrete, and must be adjudicated before either
campaign resumes.**

---

**Methodology:** read-only. No commits, no edits to git history, no rebuilds.
Inspections via `git status / log / tag / diff / rev-parse`, `ls`, `stat`,
`md5sum`, `nm -D`, `file`, `grep`, `head/tail`. Total commands ≈ 25 across
4 batches.
