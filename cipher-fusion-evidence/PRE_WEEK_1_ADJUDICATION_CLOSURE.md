# Pre-Week-1 Adjudication Closure (v1.2.2 round)

**Date:** 2026-05-20
**Supersedes:** the v1.2.1 round of this document (which closed out the same six pre-flight items against the plan-v1.2.1 state).
**Predecessor:** WEEK_1_PRE_FLIGHT.md (md5 `db05944065dd9721ea21ceb2d51e2922`).
**Plan locked at:** CIPHER_REENGINEERING_PLAN.md v1.2.2 (md5 `f6146a6cae2628fd5e3486cc647401d7`, 1,743 lines, 166,043 bytes). Option B scope expansion (30 of 33 ops at marvel depth); COMMIT promoted to atomic state-transition primitive (new §4.8); RING_WRITE promoted to lock-free inline telemetry substrate (new §4.9); timeline extended from 5 weeks to 12-14 weeks; 3 NCCL ops formally deferred to v2 on hardware grounds (new §8.4).

---

## Adjudication status — all six pre-flight items

| # | Pre-flight item (from WEEK_1_PRE_FLIGHT.md) | v1.2.2 disposition | Where |
|---|---|---|---|
| 1 | libcipher_v2.so path drift (brief named `cipher_rt_phase4/libcipher_v2.so`; actual `/home/ubuntu/libcipher_v2/`) | **CORRECTED** | Folded into v1.2.1 §1.2 footnote, carried forward to v1.2.2 |
| 2 | kmod anchor disagreement (`285d102e` plan App A vs `008b3c66` brief/on-disk) | **CORRECTED** | Folded into v1.2.1 §1, §1.3, §1.5, Appendix A, carried forward to v1.2.2 |
| 3 | DKMS-installed kmod stale (md5 `6654d9e5`, mtime 2026-05-16, 21× smaller than in-tree) | **DEFERRED** | Deployment hardening; no Week 1 impact. Plan v1.2.1 §1.5 note preserved in v1.2.2. |
| 4 | No git repositories in any of the four trees | **CORRECTED** | D3 below — repos exist from the v1.2.1 round; tag and tarball baselines refreshed to capture the v1.2.2 plan state |
| 5 | Regression runner name drift ("Phase 5 baseline 15-tenant FAIRNESS" not a real runner) | **CORRECTED** | Folded into v1.2.1 §7 Week 1 behavioral-test step, carried forward to v1.2.2 |
| 6 | LP-7 rename scheme (two competing proposals + scope confusion about rt_phase4) | **CONFIRMED** | Plan v1.2.2 §7 Week 1 + Part 4 below — `CipherKtEntry` + `CipherParamEntry` within may13 only; 9 + 9 = 18 call sites enumerated |

---

## v1.2.2 ADJUDICATIONS — what landed in the plan

Five adjudications folded into plan v1.2.2. The full text and citations live in plan §0 "What changed in v1.2.2" + §4.8 + §4.9 + §7 Weeks 6-14 + §8.3a + §8.4. Brief summary for this closure document:

### ADJUDICATION 1: Scope expanded to Option B (30 of 33 ops in v1)

**v1.2.1 state.** 18 SHIP-READY + 2 REQUIRES-FIX = ~20 ops shipping in v1 + 8 V2-SCOPE (Goal-4 Koopman tier of 6 ops + NCCL family of 3 ops).
**v1.2.2 state.** 30 of 33 ops ship in v1. Specifically: 18 SHIP-READY (unchanged) + 4 primary REQUIRES-FIX (CLASSIFY, ORACLE, COMMIT, RING_WRITE) + 4 v1.5-deferred overlay ops promoted to v1 (PREDICT, SHIELD, SUSTAIN, THERMOSTAT) + 6 Koopman tier ops promoted to v1 (REMEMBER, VALIDATE, SPECULATE, ADAPT, SAMPLE/GENERATE, SUBSTITUTE-Koopman lane). The 3 NCCL ops (NCCL_P2P, OVERLAP, STRAGGLER cross-rank) stay deferred to v2 on **hardware grounds** (single-H100 pod cannot exercise multi-GPU NCCL traffic) per new §8.4 — environment work, not engineering work.
**Citation.** Plan v1.2.2 §0 lineage row + §0 "What changed in v1.2.2" A1 + the executive-summary intro update at lines ~59-67. OP_INTENT_VS_IMPLEMENTATION D3.4 + D3.5 establishes the per-op feasibility.

### ADJUDICATION 2: COMMIT as atomic state-transition primitive

**v1.2.1 state.** COMMIT = implicit dispatch-return phase; observers update their own state independently. OP_INTENT_VS_IMPLEMENTATION Item I-1 surfaced this as UNDOCUMENTED-INTENT.
**v1.2.2 state.** New §4.8. COMMIT is now a named atomic primitive that updates AUDIT chain → FAIRNESS → CARBON → RECEIPT → kmod-resident tenant state in a deterministic order, with release-fence snapshot publish and per-tenant sequence counter discipline (not a global lock). Observers read the post-COMMIT snapshot rather than mutate per-kernel state independently. Engineering Weeks 7-8 (10 days). Risks R-W7.1/2/3 added to §8.3a.
**Citation.** Plan v1.2.2 §4.8 (new subsection); §7 Weeks 7-8; §8.3a R-W7.x rows.

### ADJUDICATION 3: RING_WRITE as lock-free inline telemetry substrate

**v1.2.1 state.** RING_WRITE = userspace ring (may13) without spawning consumer threads in production. OP_INTENT_VS_IMPLEMENTATION Item I-2 surfaced this as REQUIRES-INTENT-CLARIFICATION.
**v1.2.2 state.** New §4.9. RING_WRITE and CUPTI are distinct substrates that ship in parallel. RING_WRITE is the per-tenant SPMC ring written inline by the GOT-patched cuLaunchKernel intercept, consumed by background threads serving CLASSIFY cache, ORACLE EMA, AUDIT chain, REMEMBER hidden-state updater. CUPTI is the post-launch callback path for hardware-counter aggregation. The two cover different telemetry needs and do not block each other. Engineering Weeks 9-10 (8-10 days). Risks R-W9.1/2/3 added to §8.3a.
**Citation.** Plan v1.2.2 §4.9 (new subsection); §7 Weeks 9-10; §8.3a R-W9.x rows.

### ADJUDICATION 4: NCCL family formally deferred to v2 on hardware grounds

**v1.2.1 state.** NCCL ops deferred without a formal subsection naming the reason.
**v1.2.2 state.** New §8.4. NCCL_P2P (Op 33) + OVERLAP (NCCL compute-comm overlap scheduler) + STRAGGLER cross-rank attribution (Op 32 NCCL-side; local-detection half remains in v1) are deferred because the v1 environment is a single-H100 pod that cannot exercise multi-GPU NCCL communication. The deferral is **environment work**, not engineering work — when multi-GPU hardware and multi-node deployment infrastructure land (estimated post-seed quarter 2), the 3 ops port mechanically because Wave 5 §5.1.c Cc.2 (NCCL plugin interface) and Wave 3 (kmod-side contracts) are already documented.
**Citation.** Plan v1.2.2 §8.4 (new subsection); executive-summary intro at lines ~59-67; §0 "What changed in v1.2.2" A4.

### ADJUDICATION 5: v1 shippability reclassification

**v1.2.1 state.** 18 SHIP-READY + 4 REQUIRES-FIX + 8 V2-SCOPE = 30 ops accounted for (with SUBSTITUTE counted as a single row).
**v1.2.2 state.** 18 SHIP-READY (unchanged) + 4 primary REQUIRES-FIX (CLASSIFY, ORACLE, COMMIT, RING_WRITE) + 4 v1.5 overlay ops promoted to v1 (PREDICT, SHIELD, SUSTAIN, THERMOSTAT) + 5 Koopman ops promoted to v1 (REMEMBER, VALIDATE, SPECULATE, ADAPT, SAMPLE/GENERATE) + 1 SUBSTITUTE-Koopman lane (the dual-path SUBSTITUTE op contributes one path to v1 here; the Marlin path is already SHIP-READY) + 3 V2-deferred on hardware grounds (NCCL family) = 30 + 3 = 33.
**Citation.** Plan v1.2.2 §0 "What changed in v1.2.2" A5; §2.1 op tier table v1-shippability column annotations.

---

## v1.2.2 timeline summary

| Weeks | Goal | New in v1.2.2? |
|---|---|---|
| **1** | Compile-level classifier port + LP-7 struct rename + snapshot reserved-tail bump | No (carried from v1.2.1) |
| **2** | Hot-path classifier wiring + LP-2 attn trampoline refactor | No (carried from v1.2.1) |
| **3** | Dispatch routing goes live (GEMM-only in v1) | No (carried from v1.2.1) |
| **4** | Observability integration | No (carried from v1.2.1) |
| **5** | KV-dedup `cipher_vllm_plugin` integration was implicit in v1.2.1 Week 5 CP 5.5; v1.2.2 splits it out | Reorganized — was part of v1.2.1 W5 CP 5.5 prep |
| **6** | KV-dedup live wire + v1 substrate consolidation | **NEW (v1.2.2)** |
| **7-8** | COMMIT atomic state-transition primitive | **NEW (v1.2.2 ADJUDICATION 2)** |
| **9-10** | RING_WRITE lock-free inline telemetry substrate | **NEW (v1.2.2 ADJUDICATION 3)** |
| **11-12** | Koopman tier integration (EDMD + SUBSTITUTE-Koopman lane + Stage 1/2 thread spawn) | **NEW (v1.2.2 ADJUDICATION 1 / Option B)** |
| **13-14** | CP 5.5 headline benchmark on full unified 30-of-33-op runtime | Moved from W5 (v1.2.1) to W13-14 (v1.2.2) to ship after the full op surface lands |

---

## PART 4 — LP-7 rename plan with full call-site enumeration

Identical to the v1.2.1 round of this document — the LP-7 scope did not change in v1.2.2. Restated here for the v1.2.2 closure record.

### 4.1 — Rename scheme (confirmed)

Within `cipher-may13-evidence/` only:

```
include/cipher_kernel_table.h:53
  typedef struct CipherKernelEntry { ... } CipherKernelEntry;
  →
  typedef struct CipherKtEntry     { ... } CipherKtEntry;

include/cipher_param_recovery.h:30
  typedef struct CipherKernelEntry { ... } CipherKernelEntry;
  →
  typedef struct CipherParamEntry  { ... } CipherParamEntry;
```

`cipher_rt_phase4` and `cipher_kmod` have **zero** references to `CipherKernelEntry` (re-verified). The rename is entirely within `cipher-may13-evidence`.

### 4.2 — Call-site enumeration

**Total references: 18, all within `cipher-may13-evidence/`.**

**Group A — `CipherKtEntry` rename target (`cipher_kernel_table.h::struct CipherKernelEntry`) — 9 sites:**

```
include/cipher_kernel_table.h:53   typedef struct CipherKernelEntry {            [DEFINITION]
include/cipher_kernel_table.h:64   } CipherKernelEntry;                          [DEFINITION typedef]
include/cipher_kernel_table.h:75   const CipherKernelEntry* cipher_kt_observe(   [function declaration]
src/cipher_kernel_table.cpp:22     CipherKernelEntry  g_kt[KT_SIZE]{};           [global array]
src/cipher_kernel_table.cpp:185    CipherKernelEntry& e = g_kt[slot];            [reference]
src/cipher_kernel_table.cpp:222    extern "C" const CipherKernelEntry* cipher_kt_observe(   [impl]
src/cipher_kernel_table.cpp:235    CipherKernelEntry& e = g_kt[slot];            [reference]
src/cipher_kernel_table.cpp:262    CipherKernelEntry& e = g_kt[slot];            [reference]
src/cipher_kernel_table.cpp:278    const CipherKernelEntry& e = g_kt[i];         [reference]
```

**Group B — `CipherParamEntry` rename target (`cipher_param_recovery.h::struct CipherKernelEntry`) — 9 sites:**

```
include/cipher_param_recovery.h:30   typedef struct CipherKernelEntry {                                            [DEFINITION]
include/cipher_param_recovery.h:38   } CipherKernelEntry;                                                          [DEFINITION typedef]
include/cipher_param_recovery.h:52   const CipherKernelEntry* cipher_param_lookup(const void* host_fun);           [decl]
include/cipher_param_recovery.h:58   const CipherKernelEntry* cipher_param_lookup_by_name(const char* kernel_name); [decl]
include/cipher_param_recovery.h:83   const CipherKernelEntry* cipher_param_lookup_cufunc(const void* cufunc);      [decl]
src/cipher_param_recovery.cpp:71     CipherKernelEntry e;                                                          [local]
src/cipher_param_recovery.cpp:455    extern "C" const CipherKernelEntry*                                           [impl return type]
src/cipher_param_recovery.cpp:527    extern "C" const CipherKernelEntry*                                           [impl return type]
src/cipher_param_recovery.cpp:585    extern "C" const CipherKernelEntry* cipher_param_lookup(const void* host_fun) {  [impl]
```

Match with plan §7 Week 1 expected counts: 9 sites for `CipherKtEntry` matches exactly; 9 sites for `CipherParamEntry` exceeds the plan's claimed 6 by 3 sites (the plan undercounted Wave 1 §DEPENDENCIES). Week 1 Step 1 budget: 18 mechanical sed substitutions across 4 files.

---

## Week 1 Step 1 readiness check

| Precondition | State | Citation |
|---|---|---|
| Plan v1.2.2 locked and on disk | DONE | `/home/ubuntu/cipher-fusion-evidence/CIPHER_REENGINEERING_PLAN.md` md5 `f6146a6cae2628fd5e3486cc647401d7` |
| 30-of-33-op surface explicitly declared | DONE | Plan §0 + executive intro + §2.1 v1-shippability column |
| COMMIT design documented before Week 7 build | DONE | Plan §4.8 |
| RING_WRITE design documented before Week 9 build | DONE | Plan §4.9 |
| NCCL family hardware-deferral formally surfaced | DONE | Plan §8.4 |
| Pre-Week-1 baseline tag exists in all four trees | DONE | D3 below |
| Rollback tarballs preserved outside the trees | DONE | D3 below (refreshed for v1.2.2 state) |
| LP-7 collision verified + 18 call sites enumerated | DONE | Part 4 above |
| Rename scheme adjudicated and named | CONFIRMED | `CipherKtEntry` + `CipherParamEntry` (plan §7 Week 1 + Part 4.1) |
| Live kmod loaded and `/dev/cipher` accessible | VERIFIED | WEEK_1_PRE_FLIGHT.md §3 (re-verified in this turn: `lsmod \| grep cipher` returns `cipher_kmod 1163264 0`) |
| Clean rebuild of `libcipher_rt.so` and `cipher_kmod.ko` reproducible byte-identically | VERIFIED | WEEK_1_PRE_FLIGHT.md §2.1, §2.2 |
| CP 5.4 isolation 15/15 regression PASS | VERIFIED | WEEK_1_PRE_FLIGHT.md §3.1 |

### Build verification protocol for Week 1 Step 1

1. **Pre-rename baseline:** record md5 of `cipher-may13-evidence/include/cipher_kernel_table.h` and `cipher_param_recovery.h` and their `.cpp` counterparts. Tag SHA from `cipher-may13-evidence` git baseline = `pre-week-1-baseline`.
2. **Apply rename:** single-pass `sed -i 's/CipherKernelEntry/CipherKtEntry/g'` on `cipher_kernel_table.h` + `cipher_kernel_table.cpp`. Then `sed -i 's/CipherKernelEntry/CipherParamEntry/g'` on `cipher_param_recovery.h` + `cipher_param_recovery.cpp`. (The two structs are in different files so the order doesn't matter; each sed sees only one of the two definitions.)
3. **Compile-test in cipher-may13-evidence:** `cd /home/ubuntu/cipher-may13-evidence && make clean && make` — the rename either compiles or it doesn't. No semantic gate; this is purely compile correctness.
4. **Post-rename verification:** `grep -rn CipherKernelEntry cipher-may13-evidence/` must return zero hits. `grep -rn 'struct CipherKtEntry\b' cipher-may13-evidence/` must return at least 3 hits (definition + closing brace + at least 1 caller). `grep -rn 'struct CipherParamEntry\b'` same.
5. **Commit and tag:** `git commit -m "Week 1 Step 1: LP-7 struct rename (CipherKernelEntry → CipherKtEntry + CipherParamEntry)"` in `cipher-may13-evidence`. Tag `week-1-step-1` (or similar; not finalized here).
6. **Rollback path:** `git reset --hard pre-week-1-baseline` in `cipher-may13-evidence` restores the pre-rename state. Tarball `/home/ubuntu/cipher-baselines/pre_week_1_cipher-may13-evidence_20260520.tar.gz` is the belt-and-suspenders artifact.

**Week 1 Step 1 prompt is ready to send.** The next prompt should reference this closure document, the v1.2.2 plan, the `pre-week-1-baseline` tag, and the LP-7 rename scheme.

---

## Version Control State (D3 — git baselines + tarball snapshots)

Repos were initialized in the v1.2.1 round of this closure document; the current task refreshed tarballs and moved the `pre-week-1-baseline` tag to point at the v1.2.2 plan state for `cipher-fusion-evidence`. The three other trees (cipher-may13-evidence, cipher_rt_phase4, cipher_kmod) had no source changes between v1.2.1 and v1.2.2 — their tags and SHAs are unchanged.

### Git commit SHAs per tree

| Tree | Pre-existing `.git`? | HEAD SHA at this closure | Tag `pre-week-1-baseline` SHA | Working tree pending |
|---|---|---|---|---|
| `/home/ubuntu/cipher-may13-evidence` | yes (from v1.2.1 round) | `83b76dabcf7444db2b9840fcfd0d364bb5533a66` | same as HEAD | none |
| `/home/ubuntu/cipher_rt_phase4` | yes (from v1.2.1 round) | `b621453c4602f78d4751502a6873032ca1999e7a` | same as HEAD | none |
| `/home/ubuntu/cipher_kmod` | yes (from v1.2.1 round) | `6e360f4d95047f6922c570c835a8250fb553b0b7` | same as HEAD | none |
| `/home/ubuntu/cipher-fusion-evidence` | yes (from v1.2.1 round) | `7945523bdec69218ffb898448d03928510b8271d` (post-v1.2.2 refresh) | `7945523bdec69218ffb898448d03928510b8271d` (tag moved forward from prior `c43eec0c...`) | none |

The `cipher-fusion-evidence` SHA and tag were refreshed in D3 to capture the v1.2.2 baseline. The prior v1.2.1 baseline SHA `c43eec0cc73335cb0bcf636f2f704c697d43363a` is preserved as an auxiliary tag `pre-week-1-baseline-v1.2.1` so a v1.2.1 rollback is still git-reachable from the same tree.

**Commit message for the v1.2.2 refresh (cipher-fusion-evidence HEAD `7945523b...`):**

```
Baseline snapshot pre-Week-1 integration (v1.2.2 refresh).

v1.2.2 plan locked. 12-14 week integration sequence committed.
30 of 33 ops in v1 scope.

Anchors:
  libcipher_rt.so: 83afd1ca (verified, see WEEK_1_PRE_FLIGHT.md §2.1)
  cipher_kmod.ko: 008b3c66 (verified, see WEEK_1_PRE_FLIGHT.md §2.2)

Plan changes from v1.2.1: A1 Option B / A2 COMMIT primitive /
A3 RING_WRITE substrate / A4 NCCL deferral / A5 reclassification.

Pre-flight verification: WEEK_1_PRE_FLIGHT.md
Adjudication closure: PRE_WEEK_1_ADJUDICATION_CLOSURE.md (this commit)
Per-op verification: CORE_12_OP_VERIFICATION.md
Intent reconciliation: OP_INTENT_VS_IMPLEMENTATION.md

Plan v1.2.1 sealed md5: 8502b12b5cf10daaf99153e5076c7604
Plan v1.2.2 sealed md5: f6146a6cae2628fd5e3486cc647401d7
```

### .gitignore content per tree

Inherited from the v1.2.1 round; not changed in v1.2.2. The four `.gitignore` files cover: C/C++ build artifacts (`*.o`, `*.so`, `*.a`, `*.lo`, `*.la`); Python cache (`__pycache__/`, `*.pyc`, `*.pyo`); kbuild artifacts in `cipher_kmod` (`*.ko`, `*.mod`, `*.mod.c`, `*.cmd`, `.*.cmd`, `*.d`, `.*.d`, `*.symvers`, `*.order`, `*.builtin*`, `.tmp_versions/`); editor / IDE files (`*.swp`, `*~`, `.vscode/`, `.idea/`); build output dirs (`build/`, `.deps/`, `.libs/`); test output (`tests/*.log`); tarball backups (`*.tar.gz`, `*.tar.bz2`, `*.tar.xz`); cipher-fusion-evidence adds markdown-backup variants (`*.md.v1.0.bak`, etc.).

### Tarball snapshots (refreshed for v1.2.2 baseline)

All four tarballs include their tree's `.git/` directory so a full git history rollback is possible from any tarball alone. The prior v1.2.1 tarballs were preserved by renaming to `*.v1.2.1.tar.gz` at the same directory before the v1.2.2 tarballs were written; both round artifacts are recoverable.

| Tarball | Bytes | `.git/` entries | md5 |
|---|---|---|---|
| `pre_week_1_cipher-may13-evidence_20260520.tar.gz` | 38,562,013 | 1,007 | `c46d2ecefef34f71c8996f41fae99cbd` |
| `pre_week_1_cipher_rt_phase4_20260520.tar.gz` | 1,720,490 | 147 | `57dc351077bc247c289ef44f82a9d17e` |
| `pre_week_1_cipher_kmod_20260520.tar.gz` | 2,634,505 | 82 | `23bf73a40e5f06bcff1e3c26256da1d8` |
| `pre_week_1_cipher-fusion-evidence_20260520.tar.gz` | 360,571,530 | 3,308 | `54539a639ee11d8a83022528958a5d15` |

| Prior v1.2.1 tarball (preserved) | Bytes | md5 |
|---|---|---|
| `pre_week_1_cipher-may13-evidence_20260520.v1.2.1.tar.gz` | 38,561,534 | `9b9d1a70c248a5e076606a2be976b185` |
| `pre_week_1_cipher_rt_phase4_20260520.v1.2.1.tar.gz` | 1,720,484 | `33dbc6e6659068edb40ff4cd354a51ea` |
| `pre_week_1_cipher_kmod_20260520.v1.2.1.tar.gz` | 2,634,514 | `927173bf8e4248baeb1120fa045055ba` |
| `pre_week_1_cipher-fusion-evidence_20260520.v1.2.1.tar.gz` | 360,375,347 | `6622ea67c5d9b8e3661f90649ca211fd` |

Total v1.2.2 tarball corpus: 403,488,538 bytes (~385 MB). Total preserved v1.2.1 corpus: 403,291,879 bytes (~384 MB). Combined snapshot footprint: ~770 MB on disk at `/home/ubuntu/cipher-baselines/`.

### Rollback path

- **v1.2.2 → v1.2.1:** `git reset --hard pre-week-1-baseline-v1.2.1` in `cipher-fusion-evidence` returns the tree to the v1.2.1 plan state (or equivalently, `tar -xzf /home/ubuntu/cipher-baselines/pre_week_1_cipher-fusion-evidence_20260520.v1.2.1.tar.gz -C /home/ubuntu/` if you want a tarball-restore that also reconstructs the prior tags).
- **Any tree → its pre-Week-1 baseline:** `git reset --hard pre-week-1-baseline` in that tree. The tag in `cipher-fusion-evidence` now points at the v1.2.2 baseline; the tag in the other three trees points at the unchanged baseline (no source changes occurred between v1.2.1 and v1.2.2 in those trees).
- **Cold restore from tarball:** `tar -xzf /home/ubuntu/cipher-baselines/pre_week_1_<tree>_20260520.tar.gz -C /home/ubuntu/`.

---

**End of PRE_WEEK_1_ADJUDICATION_CLOSURE.md (v1.2.2 round).**
