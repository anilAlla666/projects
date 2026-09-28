# Week 2 — CLOSEOUT

**Headline: WEEK 2 COMPLETE.** All 6 implementation steps PASS. Classify substrate built (Steps 3-5), wired through CUPTI (Step 6), end-to-end visible at `/proc/cipher/classify_stats`. Step 1 LP-2 SDPA invariant holds throughout. CP 5.4 isolation 15/15 PASS byte-identical at every step boundary.

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2 → Week 3 boundary

---

## §1 — Steps summary

| # | step | status | anchor (tree:tag) | result doc |
| ---:| --- | --- | --- | --- |
| 1 | LP-2 SDPA trampoline refactor | PASS | `cipher_rt_phase4 @ 465b244e` (`week-2-step-1-lp2-sdpa-refactor`) | WEEK_2_STEP_1_RESULT.md |
| 2 | may13 .cpp ports (A4+H1, ROOT variants) | PASS (after 4 audit rounds → D1 retry) | `cipher_rt_phase4 @ 23014c1e` (`week-2-step-2-may13-ports-a4-h1-root`) | WEEK_2_STEP_2_V3_D1_RESULT.md |
| 3 | classify_substrate scaffolding | PASS | `cipher_rt_phase4 @ 7629cf60` (`week-2-step-3-classify-substrate-scaffold`) | WEEK_2_STEP_3_RESULT.md |
| 4 | classify_observer stub | PASS | `cipher_rt_phase4 @ 7ee5b2a8` (`week-2-step-4-classify-observer-stub`) | WEEK_2_STEP_4_RESULT.md |
| 5 | /proc/cipher/classify_stats proc node | PASS | `cipher_kmod @ bc48590e` (`week-2-step-5-classify-proc-node`) | WEEK_2_STEP_5_RESULT.md |
| 6 | CUPTI hot-path wiring + ioctl bridge | PASS | rt_phase4 `f9c32322` + kmod `0ce4b8e2` (both `week-2-step-6-*`) | WEEK_2_STEP_6_RESULT.md |
| 7 | This closeout | — | all three trees: `week-2-complete` | (this document) |

---

## §2 — Tree state at closeout

| tree | HEAD | tag | artifact md5 |
| --- | --- | --- | --- |
| `cipher_rt_phase4` | `f9c32322` | `week-2-step-6-cupti-classify-wired`, `week-2-complete` | libcipher_rt.so: `f6251e62` (this build; nvcc-non-deterministic) |
| `cipher_kmod` | `0ce4b8e2` | `week-2-step-6-ioctl-classify-push`, `week-2-complete` | cipher_kmod.ko: `8d1f1fc4` |
| `cipher-may13-evidence` | `fc8a9ae6` | `week-2-complete` (also `week-1-step-1-lp7-rename`) | (no source change in Week 2) |
| `cipher-fusion-evidence` | (will be set by this doc commit) | — | (paperwork repo) |

`cipher-may13-evidence` HEAD is unchanged from Week 1 close — Week 2 ports COPIED from this tree into `cipher_rt_phase4/src/may13/` and `cipher_rt_phase4/include/may13/`, but the source tree itself was never modified.

---

## §3 — Anchor evolution

### `cipher_rt_phase4`

```
Week 1 close      bcf8a83b  (week-1-complete)
Step 1 LP-2       465b244e  (week-2-step-1-lp2-sdpa-refactor)
Step 2 may13 port 23014c1e  (week-2-step-2-may13-ports-a4-h1-root)
Step 3 substrate  7629cf60  (week-2-step-3-classify-substrate-scaffold)
Step 4 observer   7ee5b2a8  (week-2-step-4-classify-observer-stub)
Step 6 CUPTI      f9c32322  (week-2-step-6-cupti-classify-wired, week-2-complete)
```

(Step 5 was kmod-side only; rt_phase4 stayed at `7ee5b2a8` between Steps 4 and 6.)

libcipher_rt.so md5 progression (informational; nvcc non-determinism noted from Step 3 onward):
```
Week 1 close      88ed35bb
Step 1 LP-2       56291439     ← Step 1 commit-time
Step 2 may13      bc249c9d     ← Step 2 D1 commit-time
Step 3 substrate  c20717f3     ← Step 3 commit-time
Step 4 observer   59e77b42     ← Step 4 commit-time
Step 6 CUPTI      f6251e62     ← Step 6 commit-time (this build)
```

Each step's md5 is the commit-time snapshot; consecutive rebuilds produce different bytes due to nvcc fatbin timestamping. Source identity verified via `git diff <commit>` at each step.

### `cipher_kmod`

```
Week 1 close (Step 4 v2)  f8572ecf  (week-1-step-4-cb2-reserved-tail)
Step 5 proc node          bc48590e  (week-2-step-5-classify-proc-node)
Step 6 ioctl              0ce4b8e2  (week-2-step-6-ioctl-classify-push, week-2-complete)
```

cipher_kmod.ko md5: `09c6ded5 → f497b2ee → 8d1f1fc4`. Kmod build is deterministic; md5 reproduces.

---

## §4 — Rollback paths verified

| target step | per-tree command |
| --- | --- |
| pre-Week-2 | `git -C cipher_rt_phase4 reset --hard week-1-complete` |
| post Step 1 | `git -C cipher_rt_phase4 reset --hard week-2-step-1-lp2-sdpa-refactor` |
| post Step 2 | `git -C cipher_rt_phase4 reset --hard week-2-step-2-may13-ports-a4-h1-root` |
| post Step 3 | `git -C cipher_rt_phase4 reset --hard week-2-step-3-classify-substrate-scaffold` |
| post Step 4 | `git -C cipher_rt_phase4 reset --hard week-2-step-4-classify-observer-stub` |
| post Step 5 | `git -C cipher_kmod reset --hard week-2-step-5-classify-proc-node` |
| post Step 6 | `git -C cipher_rt_phase4 reset --hard week-2-step-6-cupti-classify-wired` + `git -C cipher_kmod reset --hard week-2-step-6-ioctl-classify-push` |
| Week 2 complete | `git -C cipher_rt_phase4 reset --hard week-2-complete` + `git -C cipher_kmod reset --hard week-2-complete` |

Every tag is a clean rollback point — each step's `make clean && make` succeeds at the tag's commit (verified at the time of that step's `make` invocation).

---

## §5 — Findings surfaced during Week 2

### 5.1 — Step 2 PARTIAL × 3 (closure → intercept → source variants)

The may13 ports needed **four** rounds of audit before landing:

- **Step 2 (initial)** PARTIAL — closure walker surfaced 6 new transitive headers required by `cipher_dispatch.cpp`. Audit chain triggered.
- **Step 2 v2 (extended closure)** PARTIAL — header port succeeded; build linked; runtime loader failed with `g_block_sub` + `cipher_block_sub_collect` undefs. Symbol-closure audit built.
- **Step 2 v2 sym-closure result**: brief's 6-file scope had **40-file transitive link-closure** (54% of the may13 codebase). Single g_cipher stub collapses to 12 files; 4 stubs to 9 files.
- **Static-init pre-flight (v3)** surfaced 2 FLAG files (`cipher_intercept.cpp` + `cipher_intercept_cudart.cpp`) that install duplicate GOT-patcher hooks at libcipher_rt.so load — would race with cipher_rt_phase4's existing CP 2.5 D2(iii) GOT patcher. **A4 adjudication: exclude both.**
- **A4 closure rerun** → 11-file closure + 1 orphan (`cipher_intercept_stats`). Orphan consumer at `cipher_runtime.cpp:123` proven PURE-LOG → **H1 5-LOC stub** resolves cleanly.
- **Step 2 v3 PARTIAL** — built with H1 stub but linker found unresolved `g_block_sub`. **Variant ambiguity discovered**: cipher-may13-evidence has *two* `cipher_dispatch.cpp` files (root-canonical and src/ variant); the prebuilt `.o` is byte-identical to root; my port had copied src/.
- **D3 variant history audit** → ROOT-CANONICAL definitively (Makefile filters src/; fresh ROOT compile md5 = prebuilt md5 byte-identical).
- **Step 2 v3 D1 retry** with ROOT variants → PASS.

**The audit chain is the deliverable.** It built three reusable diagnostic tools (see §5.4).

### 5.2 — Sites 1+2 DESIGN-MISMATCH dropped (Step 6 pre-flight)

The original Step 6 brief enumerated 4 wiring sites: cuBLAS GemmEx, SDPA trampolines, CUPTI callback, ioctl bridge. The first two have a **structural geometry gap**: their call descriptors (`cipher_rt_matmul_call`, `cipher_rt_attn_call`) live at API layers above cuLaunchKernel, so neither carries `fn` / grid / block / shared. The classifier substrate is fundamentally a cuLaunchKernel-level tool.

CUPTI catches every kernel launch — including those launched internally by cuBLAS GemmEx and torch SDPA — so dropping Sites 1+2 loses **zero** observability. Step 6 shipped with 3 landings instead of 5.

### 5.3 — nvcc build non-determinism (Step 3 finding)

Consecutive `make clean && make` produces different libcipher_rt.so md5s (nvcc fatbin embeds timestamps). The brief's `md5 reproduces X` check is unreliable for this tree. Source identity is git-verified at every step (`git rev-parse HEAD` + `git diff <commit>`). Functional invariants (loader smoke, SDPA shim accounting, CP 5.4 isolation) are the binding correctness checks.

**Future hardening (out of scope; deferred to ops work):** `SOURCE_DATE_EPOCH` or `-Xcompiler -D__DATE__=...` flags would make nvcc deterministic.

### 5.4 — Audit tooling built (reusable for Weeks 3-14)

- `/tmp/symbol_closure.py` — symbol-graph builder + closure walker. Reads prebuilt .o files via `nm`, builds provider/consumer maps, walks transitive symbol closure for any source-file subset. The artifact that *should* have existed before `WEEK_2_SCOPE_LOCK.md`.
- `/tmp/symbol_closure_a4.py` — extends closure walker with file-exclusion + orphan tracking. Used for the A4 adjudication.
- `/tmp/symbol_closure_v3.py` — greedy stub picker. For any start set + closure target, finds the minimum stub-symbol set.
- `/tmp/static_init_scan.py` + `/tmp/static_init_trace.py` — per-`.o` readelf + objdump + disassembly to surface static-init side effects. Catches the GOT-patcher-conflict class of failure that Step 2 v3 surfaced.

All five tools work against any source tree with prebuilt `.o` artifacts. They are recommended as standard pre-flight for any future "port N source files" or "wire substrate to actuator" brief.

### 5.5 — Honest residue (partial-batch loss in classify_stats)

`/proc/cipher/classify_stats` flushes every 256 launches via the existing `CIPHER_V2_FLUSH_MASK` cadence. The last 0-255 launches before a process exits are **not flushed** — they remain in the userspace observer's `g_stats` but never reach kmod. Acceptable for v1.2.2 single-tenant scope (SET semantics, observability-only).

Multi-tenant ADD semantics + on-exit flush would close this gap. Deferred to Phase 5+.

---

## §6 — Week 3 entry readiness

### What Week 2 delivered that Week 3 needs

- **Classifier substrate (Steps 3-6)**: every cuLaunchKernel fires `cipher_rt_classify_route()` → may13's `cipher::classify_launch()` → `OpClass` in the observer's atomic counters. Week 3's "GEMM dispatch routing" can consult per-launch classification via the CUPTI callback's TLS or stream cache (Step 6 leaves the per-launch classify result in the route's `out` struct; future work threads it into stream metadata).
- **Substrate pattern proven (3 substrates: matmul, attn, classify)**: any future actuator (GEMM dispatch in Week 3 included) plugs in via the same priority-ordered 16-slot registry.
- **Observer infrastructure**: per-class counters surface via /proc; Phase 5 dashboards can ingest them.
- **LP-2 invariant preserved**: SDPA trampolines correctly honor `route()` return value; future v1.5 attn substitute actuators (KV dedup live, FlashSwiftKey) can register without further substrate edits.

### Week 3 scope per v1.2.2 §7

Wave 5 §5.5 W3 hot-path code wires Op 3 SUBSTITUTE for GEMM:
- Add `cipher::oracle_decide()` call after CLASSIFY (existing in may13's `cipher_dispatch.cpp`; rt_phase4 needs to call it on the CUPTI/cuBLAS-shim path).
- Add `cipher_registry_lookup()` to find a recipe for the M/N/K shape.
- Add `apply_recipe()` to dispatch the substitute kernel.
- Verification gate: bit-identical forward pass on TinyLlama + Mistral-7B.

Week 2's `cipher_rt_classify_substrate` provides the per-launch `op_class` — Week 3 layers SUBSTITUTE decisions on top.

---

## §7 — Open items for Week 3 entry window

- **Wave 5 §5.5 Week 3 pre-flight** — verification sweep mirroring the Week 2 impedance check. Should catch any Week 3 design-mismatch BEFORE the Step-1 brief drafts.
- **Sites 1+2 revisit** — if Week 3 GEMM routing wants to interpose at the cuBLAS-shim level (rather than CUPTI), Site 1's design-mismatch needs adjudication. Option from pre-flight §5: 1B short-circuit (`op_class = GEMM` without classify_route).
- **Multi-tenant push semantics** — currently SET; Phase 5 CP 5.5 (cross-tenant weight-sharing) will land multi-tenant traffic. Revisit ADD semantics when CP 5.5 brief drafts.
- **Partial-batch loss at process exit** — current behavior loses 0-255 launches per process exit. Phase 5 dashboards may need an on-exit flush. Out of scope for Week 3.
- **Step 3 nvcc non-determinism** — informational; if Phase 5 ops require reproducible builds, add `SOURCE_DATE_EPOCH` / nvcc date-strip. Out of Week-N implementation scope.

---

## §8 — Honest accounting

### Time

| step | est | actual |
| ---:| ---:| ---:|
| 1 | 2-3h | ~2h |
| 2 (4 audit rounds + D1) | 4-6h | ~14-16h |
| 3 | 2-3h | ~2h |
| 4 | 1-2h | ~1h |
| 5 | 2-3h | ~2h |
| 6 | 5-8h | ~4-6h |
| 7 closeout | 1-2h | ~1h |
| **total** | **17-27h** | **~26-30h** |

Step 2 ran 2-3× over budget (4 audit rounds vs the brief's single PASS expectation). Net effect: the audit chain produced reusable tooling (§5.4) that should accelerate Weeks 3-14. The investment amortizes if even one future step's port lands cleanly because the tooling caught a problem at pre-flight cost.

12-14 week trajectory: holds. Week 2 actual ~30h vs 17-27h budget = ~1 week of slack consumed. Plan reserve is sufficient.

### Quality

- Every step CP 5.4 byte-identical regression check.
- LP-2 SDPA invariant tracked across Steps 1-6; held at every gate.
- The Step 2 PARTIAL chain exposed three classes of audit gap (closure-by-include vs symbol, static-init conflicts, source-variant ambiguity). Each was resolved through diagnostic-first, mitigation-second discipline.
- No mitigation surfaces in result docs; every adjudication routed through user.

### Surfacing discipline observed

- 7 "STOP and surface" events during Step 2 audit chain. Each produced a result doc with options surfaced (no recommendation) for user adjudication.
- 1 "SCOPE-COMPRESSED" surface at Step 6 pre-flight (Sites 1+2 dropped).
- 0 mitigation-by-default events.

---

## Tags at close

```
cipher_rt_phase4 tags: week-2-complete (= week-2-step-6-cupti-classify-wired = f9c32322)
cipher_kmod tags:      week-2-complete (= week-2-step-6-ioctl-classify-push = 0ce4b8e2)
cipher-may13-evidence tags: week-2-complete (= week-1-step-1-lp7-rename = fc8a9ae6, unchanged)
```

Week 2 closed. Week 3 brief pending.
