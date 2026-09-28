# Week 2 Step 4 — `cipher_rt_classify_observer.{c,h}` Stub — RESULT

**Status: PASS.**

Atomic-counter observer placeholder for the classify substrate. 3 new T-symbols exported; counter storage BSS-resident zero-init. Hot-path safe (no locks, no allocation). Not yet wired — Step 6 calls `observe()` from the route loop; Step 5 `/proc/cipher/classify_stats` consumes `snapshot()` via ioctl bridge.

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 4 of 7
**Anchors:**
  - Pre:  `7629cf60` (`week-2-step-3-classify-substrate-scaffold`)
  - Post: `7ee5b2a8` (`week-2-step-4-classify-observer-stub`)

---

## A — Pre-edit verification — PASS

| signal | value | expected | match |
| --- | --- | --- | --- |
| HEAD | `7629cf60` | `7629cf60` | ✓ |
| working tree | clean | clean | ✓ |
| Pre-edit build rc | 0 | 0 | ✓ |
| Pre-edit md5 | `d7f8548e` (informational; nvcc-non-deterministic) | — | n/a |
| Pre-edit warnings | 78 | 78 | ✓ |
| Symbol count | 304 | 304 | ✓ |

Source identity is git-verified (`git rev-parse HEAD` matches); md5 reproducibility is informational only per Step 3 finding.

---

## B + C — Design + implementation

### Surface (`cipher_rt_classify_observer.h`, 58 lines)

```c
struct cipher_rt_classify_observer_stats {
    uint64_t total_classifications;
    uint64_t handled;
    uint64_t passthrough;
    uint64_t per_op_class_counts[16];  /* indexed by OpClass */
    uint64_t reserved[8];              /* Cb.2 future expansion */
};

int  cipher_rt_classify_observer_init(void);
void cipher_rt_classify_observer_observe(
    const struct cipher_rt_classify_call *call,
    const struct cipher_rt_classify_out  *out,
    int                                   result);
const struct cipher_rt_classify_observer_stats *
cipher_rt_classify_observer_snapshot(void);
```

### Implementation (`cipher_rt_classify_observer.c`, 72 lines)

- BSS-resident static `g_stats` (zero-init at .so load).
- `init()` returns 0 with no side effects (Step 4 stub; future revisions may need real init work).
- `observe()` uses `__atomic_fetch_add` (C11 atomics, RELAXED memory order) on three counters:
  1. `total_classifications` (always)
  2. `handled` if `result == CIPHER_RT_CLASSIFY_HANDLED`, else `passthrough`
  3. `per_op_class_counts[op]` where `op = out->op_class`, clamped to slot 15 if ≥ 16 (UNCLASSIFIED bucket; matches observer convention)
- `snapshot()` returns `&g_stats` — readers should treat fields as atomic loads (struct lifetime = .so lifetime).
- No locks. No allocation. Hot-path safe.

The `cast to atomic_ullong*` comment in the file documents the layout-compatibility assumption (uint64_t and atomic_ullong are layout-compatible on Linux x86_64; `__atomic_*` intrinsics work portably).

---

## D — Makefile diff

```
  cipher_rt_classify_substrate.o cipher_rt_classify_observer.o \
  cipher_may13_harness.o \
...
+# Week 2 Step 4 — classify observer stub. Atomic-counter sink for the
+# classify substrate. Step 6 wires observe() into the route loop;
+# Step 5 /proc/cipher/classify_stats pulls from snapshot() via ioctl.
+cipher_rt_classify_observer.o: cipher_rt_classify_observer.c \
+                                cipher_rt_classify_observer.h \
+                                cipher_rt_classify_substrate.h
+   $(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<
```

10 lines added. C-only build (matches the brief's `.c` not `.cpp` choice — Wave 5 §5.5 naming).

---

## E — Build + nm diff — PASS

| signal | value |
| --- | ---:|
| Build rc | 0 |
| Warning count | 78 → 78 (zero delta) |
| libcipher_rt.so md5 | `d7f8548e...` → `59e77b42...` (nvcc-non-deterministic) |
| Symbol count | 304 → 307 |
| Symbols removed | **0** |

### 3 new T-symbols (matches design)

```
T cipher_rt_classify_observer_init
T cipher_rt_classify_observer_observe
T cipher_rt_classify_observer_snapshot
```

(BSS-resident `g_stats` is internal-linkage and does NOT appear in the dynamic symbol table — by design.)

### Undef-symbol audit

Clean — only the expected system surface (`cuptiSubscribe`, weak edmd-family undefs, libgcc/glibc system symbols). No new CIPHER undefs introduced.

---

## F — Runtime smoke — PASS

### F.1 Loader smoke

```
$ LD_PRELOAD=$(realpath libcipher_rt.so) /bin/true
[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)
rc=0
```

CLASSIFY registration banner still fires (Step 3 substrate behavior unchanged).

### F.2 SDPA shim injection smoke

```
banner count: 17 (unchanged vs Step 3)
MATMUL: exit totals — calls=0 handled=0 passthrough=0 (actuators=0)
[cipher-attn] exit totals - tramp_calls=1 tramp_fake=0 observed=1
              handled=0 passthrough=1 redirected=0 (flash=0 eff=0 cudnn=1)

rc=0
```

LP-2 invariant holds (`tramp_calls=1 handled=0 passthrough=1`). Observer's `observe()` is not yet called from any hot path — counters remain at zero.

---

## G — CP 5.4 regression — PASS

```
=== Phase A result: 15 PASS, 0 FAIL ===
rc=0
```

Byte-identical to Step 3 baseline.

---

## H — Commit + tag — PASS

```
3 files changed, 139 insertions(+), 1 deletion(-)
  cipher_rt_classify_observer.c   | 72 ++++++++++++++++++++++++++
  cipher_rt_classify_observer.h   | 58 +++++++++++++++++++
  Makefile                        | 10 ++

HEAD: 7ee5b2a82b101393c0223671f4b59a70ca122741
Tag:  week-2-step-4-classify-observer-stub
```

---

## Discipline notes

- Single commit. Tag chain: `week-2-step-3-classify-substrate-scaffold` → `week-2-step-4-classify-observer-stub`.
- Rollback: `git reset --hard week-2-step-3-classify-substrate-scaffold`.
- Observer is BSS zero and is not on any hot path until Step 6. Counter values remain zero in any current invocation of the .so.

---

## Anchors at close

- `cipher_rt_phase4` HEAD: `7629cf60` → `7ee5b2a8` (tag `week-2-step-4-classify-observer-stub`)
- `cipher_kmod`, `cipher-may13-evidence`: unchanged

Week 2 progress: 4/7. Step 5 result document follows.
