# Week 4 Step 2 — Entry Verification

**Date:** 2026-05-21. **Type:** read-only diagnostic — no source modified, no
build, no GPU. **All preconditions PASS — Step 2 (Tier A observability ports)
is unblocked.** Step 2 prompt may be drafted.

---

## B.1 — Tree HEADs match Step 1 close

| repo | HEAD | tags-at-HEAD | matches expected? |
|---|---|---|---|
| `cipher_rt_phase4` | `4279461743…` | **`week-4-step-1-real-oracle`** | ✅ |
| `cipher_kmod` | `a21a45ee7a…` | `week-3-complete`, **`week-3-step-4-opt2a-dsm-propose`** | ✅ |
| `cipher-may13-evidence` | `fc8a9ae602…` | `week-1-step-1-lp7-rename`, **`week-3-complete`** | ✅ |

All three trees clean (0 dirty files). No drift since Step 1 close (2026-05-20
19:27:29). `cipher-fusion-evidence` carries 57 dirty files but **all reside in
the off-path `phase_c/` directory** (FUTURE_SCOPE/A SC6 logs rewritten by
Week 4 Step 1's SC6 reruns) + the now-paused Phase 4 memo + this
session's `SESSION_STATE_2026-05-21.md`. **No Week-N path is dirty.**

## B.2 — libcipher_rt.so + oracle-bridge symbols present

```
md5sum: d60c625692a1c2b9dcf56ce0ebbe8999  libcipher_rt.so
nm -D : 0000000000013c10 T cipher_rt_oracle_bridge_decide
        0000000000013bd0 T cipher_rt_oracle_bridge_init_lazy
```

Both Week-4-Step-1 bridge symbols are exported (T = text). md5 unchanged
since the Step 1 commit-time build (mtime 19:22:16, commit 19:27:29 —
build-then-commit ordering). ✅

**Bonus check — loaded kmod is the right source.** `modinfo` srcversion
`F6B9227C41E5439BD8F1B02` matches `/sys/module/cipher_kmod/srcversion` —
the DKMS-recompiled loaded module is from the **same source** as on-disk
`cipher_kmod/cipher_kmod.ko` (HEAD `a21a45e`). The yesterday's diagnostic
flag about "loaded md5 6654d9e5 is older than HEAD source" was a false
alarm — md5 diverges because DKMS recompiles, but srcversion equality
proves source-identity. **Loaded kmod is Week 3 Step 4 II-a, as expected.**

## B.3 — Tier A sources present in `cipher-may13-evidence`

```
src/cipher_loop.cpp        (11145 B)    include/cipher_loop.h        (1770 B)
src/cipher_pipeline.cpp    ( 9396 B)    include/cipher_pipeline.h    (1004 B)
src/cipher_pulse.cpp       (16198 B)    include/cipher_pulse.h       (2240 B)
src/cipher_continuity.cpp  ( 8950 B)    include/cipher_continuity.h  ( 995 B)
```

All 4 .cpp + 4 .h present. Total source ≈ **45 689 B** (~970 source LOC band
the scope-lock projected). ✅

## B.4 — Symbol closure (re-verified at HEAD)

Ran `/tmp/closure_tierA.py` (wraps `/tmp/symbol_closure.py` machinery; same
build-tree provider/consumer graph used in WEEK_4_PREFLIGHT Part 4.4).

| file | undef-tot | non-system | tree-prov-still-needed | "unsat" |
|---|---:|---:|---:|---:|
| `cipher_loop.cpp` | 8 | 2 | **0** | 2 |
| `cipher_pipeline.cpp` | 9 | 3 | **0** | 2 |
| `cipher_pulse.cpp` | 11 | 2 | **0** | 2 |
| `cipher_continuity.cpp` | 9 | 3 | **0** | 2 |

The **2 "unsat" entries per file are always `__fprintf_chk` and
`__stack_chk_fail`** — the FORTIFY_SOURCE glibc wrappers, not may13 sources.
These are the "system glibc undefs" the spec expects (resolved by the
fortified libc at link time, no .cpp port required).

For `cipher_pipeline.cpp` and `cipher_continuity.cpp` the third non-system
undef is `cipher_sense_get_type`, **already exported by libcipher_rt** (ported
Week 2 Step 2; confirmed live: `nm -D libcipher_rt.so | grep cipher_sense_get_type`
returns `T cipher_sense_get_type` at `0x17730`). **No new transitive .cpp pull**
required for any of the 4 files. ✅

This matches WEEK_4_PREFLIGHT Part 4.4 verbatim for LOOP + PIPELINE and
confirms the preflight's "likely similar pattern" prediction for PULSE +
CONTINUITY.

---

## Verdict

| precondition | result |
|---|---|
| B.1 tree HEADs | ✅ all three at expected commits + tags |
| B.2 libcipher_rt.so + bridge symbols | ✅ d60c6256, both symbols exported |
| B.3 Tier A sources | ✅ 4 .cpp + 4 .h present in cipher-may13-evidence |
| B.4 closure check | ✅ all 4 self-contained on current libcipher_rt (only glibc fortify wrappers + the already-ported `cipher_sense_get_type`) |

**Step 2 may proceed.** Per `WEEK_4_SCOPE_LOCK.md` §85, Step 2 scope is the
port of LOOP/PIPELINE/PULSE/CONTINUITY (~970 source LOC + headers, ~5–7 h);
no /proc nodes (these are session-level observability monitors emitting to
internal substrate buffers consumed by Tier B + Step 5). The closure-clean
property means no transitive .cpp pulls beyond cipher_sense_get_type
(already shipped). Anchors will rotate `cipher_rt_phase4` (Step 2 is
userspace-only — kmod, may13-evidence, and libcipher_v2 untouched).

**No surfaces requiring adjudication.** Next: draft Step 2 prompt against
the scope-lock.
