# Week 1 Closeout

**HEADLINE STATUS: WEEK 1 COMPLETE.**

**Date:** 2026-05-20
**Scope closed:** Week 1 of the 12-14 week integration sequence per `CIPHER_REENGINEERING_PLAN.md` v1.2.2 §7. All four steps executed; three of four passed on first attempt, one required a v2 retry under user adjudication.
**Output artifacts (this closeout):** `WEEK_1_CLOSEOUT.md` (this document), `WAVE_5_CB2_ADJUDICATION.md`, `WAVE_5_S5_5_VERIFICATION.md`.
**Tag landed on all three trees:** `week-1-complete`.

---

## §1 — Steps summary

| Step | Scope | Tree | Status | Commit SHA | Tag | Deliverable doc + md5 |
|---|---|---|---|---|---|---|
| 1 | LP-7 struct rename (`CipherKernelEntry` → `CipherKtEntry` + `CipherParamEntry`) within `cipher-may13-evidence` | cipher-may13-evidence | PASS | `fc8a9ae6025cff345e86d054b1df02d07a9fb1fe` | `week-1-step-1-lp7-rename` | `WEEK_1_STEP_1_RESULT.md` md5 `7559cb9798291fcc4468b34350ab1fa5` |
| 2 (first attempt) | 7-header may13 port into `cipher_rt_phase4/include/may13/` | cipher_rt_phase4 | **PARTIAL** (transitive deps + bare-name resolution gap) | — (no commit; left as uncommitted untracked) | — | `WEEK_1_STEP_2_RESULT.md` md5 `d4ff2fc5408f84edc26b5baa3bf8d678` |
| 2 v2 | 12-header Option B closure + 8 bare-name `may13/` rewrites | cipher_rt_phase4 | PASS | `50a6f2283032420d7bd664f29ba1d81a786f25c0` | `week-1-step-2-v2-header-port` | `WEEK_1_STEP_2_V2_RESULT.md` md5 `331cdddd005ab8c099e79de77e2b1b83` |
| 3 | Cross-tree compile harness (`src/cipher_may13_harness.cpp`) | cipher_rt_phase4 | PASS | `bcf8a83b10e5112a10e03afdb70cabb9b74d75c4` | `week-1-step-3-cross-tree-harness` | `WEEK_1_STEP_3_RESULT.md` md5 `1664a6f768ddede181c3b4d3c800aab5` |
| 4 (first attempt) | Cb.2 reserved-tail bump per Wave 5 §5.5 literal spec | cipher_kmod | **PARTIAL** (3 Wave-5/disk contradictions surfaced) | — (no commit) | — | `WEEK_1_STEP_4_RESULT.md` md5 `558869e65a0c960995e2f26027aa7d05` |
| 4 v2 | Cb.2 reserved-tail bump under Option A adjudication | cipher_kmod | PASS | `f8572ecf422050a488f1fb45f74534ecd1bde678` | `week-1-step-4-cb2-reserved-tail` | `WEEK_1_STEP_4_V2_RESULT.md` md5 `9d88eca12b51895b991e229cf8afd446` |

Supporting closure-walk + audit documents:

| Document | Purpose | md5 |
|---|---|---|
| `WEEK_1_STEP_2_CLOSURE.md` | 11-header transitive closure walk | `1f55748ac929aeb1ae2da749ad301515` |
| `WEEK_1_STEP_2_CLOSURE_DELTA.md` | Delta walk adding `cipher_recipes.h` → 12-header closure | `38423a8c65d8b1eae9a3ef430e9f5734` |
| `WAVE_5_CB2_ADJUDICATION.md` | Audit-correction trail for Cb.2 contradictions + Option A | (recorded at end) |
| `WAVE_5_S5_5_VERIFICATION.md` | Weeks 2-5 sweep for further synthesis errors | (recorded at end) |

---

## §2 — Tree state at closeout

### 2.1 — cipher-may13-evidence

| Field | Value |
|---|---|
| Working tree | clean |
| HEAD | `fc8a9ae6025cff345e86d054b1df02d07a9fb1fe` |
| `pre-week-1-baseline` tag | `83b76dabcf7444db2b9840fcfd0d364bb5533a66` |
| `week-1-step-1-lp7-rename` tag | `fc8a9ae6025cff345e86d054b1df02d07a9fb1fe` (same as HEAD) |
| `week-1-complete` tag | `fc8a9ae6025cff345e86d054b1df02d07a9fb1fe` (same as HEAD) |
| Artifact md5 | n/a (no shipped binary in this tree; build outputs live in `build/` and 3 DSOs from local make for verification) |

### 2.2 — cipher_rt_phase4

| Field | Value |
|---|---|
| Working tree | clean |
| HEAD | `bcf8a83b10e5112a10e03afdb70cabb9b74d75c4` |
| `pre-week-1-baseline` tag | `b621453c4602f78d4751502a6873032ca1999e7a` |
| `week-1-step-2-v2-header-port` tag | `50a6f2283032420d7bd664f29ba1d81a786f25c0` |
| `week-1-step-3-cross-tree-harness` tag | `bcf8a83b10e5112a10e03afdb70cabb9b74d75c4` (same as HEAD) |
| `week-1-complete` tag | `bcf8a83b10e5112a10e03afdb70cabb9b74d75c4` (same as HEAD) |
| `libcipher_rt.so` md5 | `88ed35bb13b0524e889bb5b56d610fd9` |

### 2.3 — cipher_kmod

| Field | Value |
|---|---|
| Working tree | clean |
| HEAD | `f8572ecf422050a488f1fb45f74534ecd1bde678` |
| `pre-week-1-baseline` tag | `6e360f4d95047f6922c570c835a8250fb553b0b7` |
| `week-1-step-4-cb2-reserved-tail` tag | `f8572ecf422050a488f1fb45f74534ecd1bde678` (same as HEAD) |
| `week-1-complete` tag | `f8572ecf422050a488f1fb45f74534ecd1bde678` (same as HEAD) |
| `cipher_kmod.ko` md5 | `09c6ded53e33d1ee418dcdc6d8605cd4` |
| Kernel-loaded module | live (`lsmod \| grep cipher_kmod` → loaded; reload-after-Cb.2 verified in Step 4 v2 §E) |

---

## §3 — Anchor evolution

### 3.1 — `libcipher_rt.so`

| Anchor | md5 | Cause |
|---|---|---|
| Pre-Week-1 (baseline) | `83afd1ca4118dc651854ef751e4ef82d` | The `pre-week-1-baseline` anchor reproducible from clean rebuild of cipher_rt_phase4 at the original 16-source build set. |
| Post-Step-2-v2 | `83afd1ca4118dc651854ef751e4ef82d` (UNCHANGED) | Step 2 v2 ported 12 headers to `include/may13/` without modifying any library TU and without touching the Makefile's `INCLUDES` or `OBJS`. The library compile-only invariant held: headers exist on disk but no library TU includes them, so libcipher_rt.so build is byte-identical to the anchor. |
| Post-Step-3 | `88ed35bb13b0524e889bb5b56d610fd9` | Step 3 added one new TU (`src/cipher_may13_harness.cpp`, 59 LOC) that includes `may13/cipher_classify.hpp` and declares 3 static markers. Plus 3 Makefile changes (`-Iinclude`, `cipher_may13_harness.o` in OBJS, new per-source rule). md5 changes expected — new code in the .so. nm-diff verified: 3 new internal-linkage symbols added, zero existing symbols removed or name-changed (the address shifts on `_DYNAMIC`/`__FRAME_END__`/`__GNU_EH_FRAME_HDR` are pure layout drift from +952 bytes). GNU Build-IDs identical between Step 2 v2 baseline and Step 3 post-harness; semantic surface preserved. |

The transition from `83afd1ca` → `88ed35bb` is the only md5 change for libcipher_rt.so during Week 1. Substrate runtime behavior is unchanged (CP 5.4 isolation 15/15 PASS line-for-line identical to pre-flight baseline).

### 3.2 — `cipher_kmod.ko`

| Anchor | md5 | Cause |
|---|---|---|
| Pre-Week-1 (baseline) | `008b3c66faa82c71c1615ddf9c87ec56` | Post-Track-2 SC5 close; reproducible from clean rebuild of cipher_kmod. |
| Post-Step-4-v2 | `09c6ded53e33d1ee418dcdc6d8605cd4` | Step 4 v2 applied the Cb.2 reserved-tail bump under Option A — added 2 `__u32` fields (`recommended_sm_count`, `tenant_billing_class`) at the head of the reserved tail in both `cipher_tenant_snapshot` (kmod-internal) and `cipher_tenant_snapshot_user` (userspace mirror); reduced `reserved[16]` → `reserved[14]` to absorb 8 bytes. sizeof preserved at 336 in both. All non-reserved field offsets unchanged. md5 changes expected — the struct layout edit affects every TU that includes the modified headers. |

The transition from `008b3c66` → `09c6ded5` is the only md5 change for cipher_kmod.ko during Week 1. Reload + CP 5.4 isolation verified runtime behavior byte-identical.

---

## §4 — Rollback paths verified

### 4.1 — Per-tree rollback to pre-Week-1 state

| Tree | Command | Returns to | Tarball belt-and-suspenders |
|---|---|---|---|
| cipher-may13-evidence | `git -C /home/ubuntu/cipher-may13-evidence reset --hard pre-week-1-baseline` | `83b76dabcf7444db2b9840fcfd0d364bb5533a66` | `/home/ubuntu/cipher-baselines/pre_week_1_cipher-may13-evidence_20260520.tar.gz` md5 `c46d2ecefef34f71c8996f41fae99cbd` |
| cipher_rt_phase4 | `git -C /home/ubuntu/cipher_rt_phase4 reset --hard pre-week-1-baseline` | `b621453c4602f78d4751502a6873032ca1999e7a` | `/home/ubuntu/cipher-baselines/pre_week_1_cipher_rt_phase4_20260520.tar.gz` md5 `57dc351077bc247c289ef44f82a9d17e` |
| cipher_kmod | `git -C /home/ubuntu/cipher_kmod reset --hard pre-week-1-baseline` | `6e360f4d95047f6922c570c835a8250fb553b0b7` | `/home/ubuntu/cipher-baselines/pre_week_1_cipher_kmod_20260520.tar.gz` md5 `23bf73a40e5f06bcff1e3c26256da1d8` |
| cipher-fusion-evidence | `git -C /home/ubuntu/cipher-fusion-evidence reset --hard pre-week-1-baseline-v1.2.1` (or `pre-week-1-baseline` for the v1.2.2 refresh) | preserved at two SHAs | `/home/ubuntu/cipher-baselines/pre_week_1_cipher-fusion-evidence_20260520.tar.gz` md5 `54539a639ee11d8a83022528958a5d15` |

### 4.2 — Per-step intermediate rollback

The per-step tags allow rolling back from any later state to any earlier completed state:

- cipher-may13-evidence: only one Week 1 step (Step 1), so `pre-week-1-baseline` ↔ `week-1-step-1-lp7-rename` ↔ `week-1-complete` is the full chain.
- cipher_rt_phase4: `pre-week-1-baseline` → `week-1-step-2-v2-header-port` → `week-1-step-3-cross-tree-harness` (= `week-1-complete`). `git reset --hard week-1-step-2-v2-header-port` rolls back Step 3 alone.
- cipher_kmod: `pre-week-1-baseline` → `week-1-step-4-cb2-reserved-tail` (= `week-1-complete`).

### 4.3 — Tarball corpus integrity

All four tarballs in `/home/ubuntu/cipher-baselines/` include the `.git/` directory; restoring from a tarball reconstructs the git history and tag set. Tarballs were generated at the v1.2.2 baseline refresh (commit `7945523b...` in cipher-fusion-evidence). The prior v1.2.1 tarballs are also preserved at `*.v1.2.1.tar.gz` for the four trees.

---

## §5 — Findings surfaced during Week 1

### 5.1 — Step 2 PARTIAL: closure walk required (scope 7 → 12 headers + 8 rewrites)

Step 2's brief listed 7 headers from Wave 5's literal Week 1 scope. The first attempt revealed that 4 of those 7 headers had unresolved transitive `#include` dependencies (`cipher_10ops.h`, `cipher_liquid_state.h`, `cipher_structural_lookup.h`) and that the bare-name include style inside may13 headers does not resolve under cipher_rt_phase4's strict include discipline. **Option B (user adjudication):** expand the port set to the full transitive closure (12 headers) and apply 8 bare-name → `may13/` prefix rewrites to the destination copies. Both deliverables landed cleanly in Step 2 v2.

The closure walk artifacts (`WEEK_1_STEP_2_CLOSURE.md` + `WEEK_1_STEP_2_CLOSURE_DELTA.md`) are the formal record of how the 12-header set was derived from the original 7 + 1 delta.

### 5.2 — Step 4 PARTIAL: Wave 5 §5.5 Cb.2 contained three contradictions; Option A adjudicated

The Wave 5 spec for the Cb.2 snapshot reserved-tail bump contained three errors:

1. **Target struct named incorrectly.** Wave 5 named `cipher_pid_stats` as one of the bump targets; that struct has no `reserved[N]` tail. The actual targets are `cipher_tenant_snapshot` (kmod-internal) and `cipher_tenant_snapshot_user` (userspace mirror).
2. **Two of four proposed fields already exist** at `__u32` size in the Agentic/SLO cluster (`session_band`, `slo_priority`), not in the reserved tail. Wave 5 proposed them as `__u8` in the reserved tail.
3. **Internal math contradiction**: field declarations sum to 12 bytes, "Remaining `reserved[12]` (was 16; now 12)" implies 16 bytes consumed, I-W1.2 says "new 8 B." Three different numbers for the same arithmetic.

**Option A** (user adjudication): add only the 2 truly-new fields (`recommended_sm_count`, `tenant_billing_class`) as `__u32` to the reserved tail of both target structs; reduce `reserved[16]` → `reserved[14]` to absorb 8 bytes; preserve sizeof at 336; leave `cipher_pid_stats` and the existing `session_band`/`slo_priority` fields untouched. Step 4 v2 applied this cleanly with all ABI invariants held.

Full audit trail in `WAVE_5_CB2_ADJUDICATION.md`.

### 5.3 — Test infrastructure gap: may13 tests hardcode `/home/ubuntu/op31-prod-fix/` paths

Step 1 §D surfaced that `cipher-may13-evidence/tests/test_kernel_table.py` and `tests/test_pattern6_paraminfo.py` reference an absolute path `/home/ubuntu/op31-prod-fix/libcipher_hook.so` that does not exist on this pod (the historical dev-pod path). The Mistral-7B generate portion of the kernel-table test ran successfully against the post-Step-1 binaries before hitting the unrelated ctypes-path error, so the rename is verified not to have broken upstream HuggingFace + PyTorch + CUDA paths. Documenting as a pre-existing test-infrastructure gap to be revisited in Week 4 observability port.

### 5.4 — Filename drift: `cipher_classify.hpp` vs Wave 5's `.h` citation

Step 2 §B surfaced that Wave 5 §5.5 cited `cipher_classify.h` but the canonical may13 file is `cipher_classify.hpp` (a C++ header with template / inline functions). The Step 2 v2 port preserved the `.hpp` extension at the destination. Documentation drift, no execution blocker.

### 5.5 — Wave 5 §5.5 Weeks 2-5 sweep surfaced 4 additional CONTRADICTION items

The verification sweep (`WAVE_5_S5_5_VERIFICATION.md`) — performed during this closeout — applied the same lens to Weeks 2-5 entries:

- **C1:** `/proc/cipher/classify_stats` (Wave 5 attributed to Week 1; not created)
- **C2:** `cipher_rt_classify_observer.c` (Wave 5 attributed to Week 1 as new empty stub; not created)
- **C3:** `cipher-may13-evidence/src/cipher_audit.cpp` (Wave 5 Week 4 names it as retirement target; the file does not exist — AUDIT lives inline in `cipher_10ops_impl.cpp`)
- **C4:** `cipher_partition_slot[32]` array (Wave 5 Week 4 names it; the actual variable is `cipher_slots[CIPHER_PARTITION_SLOTS_MAX]` where MAX=33)

Plus 4 SYNTHESIS-HYPOTHESIS items (off-by-one step numbering, paraphrased function names) that are non-blocking. See §7 below for the Week 2 entry-window adjudication implications.

### 5.6 — Wave 5 Week 1 scope wider than what Steps 1-4 v2 executed

Wave 5 §5.5 Week 1 implicitly attributed the creation of `cipher_rt_classify_substrate.cpp`, `cipher_rt_classify_observer.c`, and the `/proc/cipher/classify_stats` proc node to Week 1. None of those landed. The Steps 1-4 v2 prompts each had a tightly-scoped brief (LP-7 rename / 12-header port / cross-tree harness / Cb.2 bump) that did not include the substrate / observer files or the proc node. This is a Week 1 scope GAP relative to Wave 5, surfaced for Week 2 entry-window resolution.

---

## §6 — Week 2 entry readiness

Week 2 per plan v1.2.2 §7: **Hot-path classifier wiring + LP-2 SDPA trampoline refactor.**

Foundation for Week 2 is in place:

| Foundation item | State |
|---|---|
| LP-7 struct collision resolved | DONE (Step 1) |
| may13 headers reachable via `may13/` prefix from cipher_rt_phase4 TUs | DONE (Step 2 v2) |
| Cross-tree compile harness proves linkability | DONE (Step 3) |
| Cb.2 reserved-tail bump for COMMIT primitive input fields | DONE (Step 4 v2) |
| Library invariant (existing TU set produces unchanged libcipher_rt.so under -Iinclude) | DONE (Step 3 — substrate behavior unchanged) |
| Kmod ABI extended for classifier-driven snapshot fields | DONE (Step 4 v2 — `recommended_sm_count`, `tenant_billing_class` at offsets 272/276) |

Items deferred from Wave 5's broader Week 1 scope to Week 2 entry (§7 below).

---

## §7 — Open items for Week 2 entry window

### 7.1 — Week 1 scope gap items (Wave 5 attributed; not executed)

Three items Wave 5 attributed to Week 1 that need creation at Week 2 entry:

1. **`cipher_rt_phase4/cipher_rt_classify_substrate.cpp`** — new file, mirrors `cipher_rt_matmul_dispatch.{c,h}` shape. Exports `cipher_rt_classify_register` + `cipher_rt_classify_dispatch`. No actuator chain yet (substrate header only).
2. **`cipher_rt_phase4/cipher_rt_classify_observer.c`** — new empty stub. Will register against matmul/attn registries in Week 2 Step 1.
3. **`/proc/cipher/classify_stats`** — new kmod proc node. Reports non-zero classification counts per kernel launch (Week 2 I-W2.2 invariant).

These three together form the entry surface for the brain-hook wiring. Week 2 Step 1 should create them as a single coherent change before any other Week 2 work proceeds.

### 7.2 — Weeks 2-5 CONTRADICTION items requiring user adjudication

From `WAVE_5_S5_5_VERIFICATION.md` — four items the Week 2 entry-window prompt should address:

| # | Item | Decision needed |
|---|---|---|
| C1 | `/proc/cipher/classify_stats` (Wave 5 says created by Week 1) | Confirm Week 2 Step 1 creates the node, OR amend Wave 5's Week 1 attribution after the fact. |
| C2 | `cipher_rt_classify_observer.c` (Wave 5 says created by Week 1) | Same — Week 2 Step 1 creates, OR amend Wave 5. |
| C3 | `cipher-may13-evidence/src/cipher_audit.cpp` "retirement" in Week 4 | File does not exist; AUDIT inline in `cipher_10ops_impl.cpp:341-378`. Decide whether Week 4 retirement step is needed (no .cpp to delete) or whether Wave 5's framing is moot. |
| C4 | Slot array naming in Week 4 LP-8 retirement | Actual variable is `cipher_slots` (not `cipher_partition_slot[32]`); size 33 not 32. Wave 5 text needs correction for mechanical retirement to work. |

### 7.3 — Test infrastructure gap

`cipher-may13-evidence/tests/test_kernel_table.py` + `test_pattern6_paraminfo.py` hardcode `/home/ubuntu/op31-prod-fix/libcipher_hook.so`. Recommend deferring path-rewrite to Week 4 (observability port), unless an earlier need arises.

### 7.4 — DKMS image staleness

WEEK_1_PRE_FLIGHT.md §1.4 surfaced that the DKMS-installed kmod at `/lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko` is 21× smaller than the in-tree build (md5 `6654d9e5`, mtime 2026-05-16; predates Track 2 SC5). Documented as deployment hardening; no Week 1 impact. Week 5 CP 5.5 benchmark may want a DKMS refresh; surfaced for that planning window.

---

## §8 — What "Week 1 complete" actually means

**Compile-level integration achieved across three trees.** The two new structs (`recommended_sm_count`, `tenant_billing_class`) are in place; the 12-header may13 closure is reachable from cipher_rt_phase4 TUs via the `may13/` prefix; the LP-7 struct collision is resolved; the cross-tree compile harness proves linkability.

**No runtime behavior changed.** libcipher_rt.so md5 transitioned 83afd1ca → 88ed35bb (Step 3 harness only); cipher_kmod.ko md5 transitioned 008b3c66 → 09c6ded5 (Step 4 v2 struct extension only). Substrate behavior verified byte-identical via CP 5.4 isolation 15/15 PASS across Steps 1, 2 v2, 3, and 4 v2 — every step's regression smoke produced output line-for-line identical to the previous baseline.

**Honest scope.** Week 1 as executed did NOT create `cipher_rt_classify_substrate.cpp`, `cipher_rt_classify_observer.c`, or `/proc/cipher/classify_stats`. These were within Wave 5's broader Week 1 scope but outside the brief of any of the four executed steps. They are surfaced in §7 above for Week 2 entry-window resolution.

**The Week 2 entry-window prompt** should adjudicate the four CONTRADICTION items and authorize the three Week 1 scope-gap items (substrate + observer files + proc node) as the entry deliverable of Week 2 Step 1.

---

**End of WEEK_1_CLOSEOUT.md.**
