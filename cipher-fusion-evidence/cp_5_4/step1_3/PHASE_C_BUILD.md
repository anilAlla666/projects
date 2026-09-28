# CP 5.4 Step 1.3 — Phase C: libcipher_rt client build — REPORT

**Date:** 2026-05-18. **Verdict: BUILD PASS.** `libcipher_rt.so.new`
(`ebc0baaa4e771abbdd20f6c4af645040`) built clean at a non-canonical path; the
canonical `libcipher_rt.so` is **untouched at `a7ac8e97`** pending Phase D.
Symbol-table diff is exactly the expected delta. No anchor rotated.

---

## Source modifications (libcipher_rt only)

### Sub-component 1 — `cipher_rt_green_ctx.c` (+ `.h`)

The green context is now **kmod-driven**, replacing the `pid ^ tenant_handle`
hash-pick:
- New `cipher_rt_green_ctx_cp54_init()` — reads `CIPHER_QOS_CLASS`
  (partition/shared/pool, default shared) and `CIPHER_SM_COUNT`, issues
  `CIPHER_CP54_ALLOCATE` (ioctl nr 13) on `/dev/cipher`, caches the granted
  `grp_mask`. One-shot; intended to run at injection-init (Q4). On any failure
  (`/dev/cipher` absent, ioctl error) it leaves a not-ok state so `ensure()`
  falls back — graceful, no crash.
- `cipher_rt_green_ctx_ensure()` reworked: `cuDevSmResourceSplitByCount(MIN_SM=8)`
  → the 8-SM group resources; the set bits of the cached `grp_mask` select this
  tenant's groups; `cuDevResourceGenerateDesc` over their **union** →
  variable-size green context. `qos=shared` (grp_mask 0) → **no green context**
  (runs on the primary context; pool binding is Step 1.3b′). Defensive: a
  `grp_mask` bit ≥ the split count falls back rather than indexing a
  nonexistent group. The hash-pick path is **retained** as the fallback.
- Group count is never hard-coded — `ensure()` uses the count
  `cuDevSmResourceSplitByCount` returns (15 on this H100).
- `cipher_ioctl.h` included for the CP 5.4 ABI (verified userspace-safe — the
  isolation test includes it; green_ctx.c does not pull `cipher_rt_tenant.h`,
  so no struct-name conflict).
- `cipher_rt_green_ctx.h`: declares `cipher_rt_green_ctx_cp54_init`.

### Sub-component 2 — `cipher_inject.c`

`cipher_v2_init_body()` calls `cipher_rt_green_ctx_cp54_init()` right after
`cipher_v2_tenant_register()` — the `pthread_once` injection-init thread is a
process-lifetime thread, so the kmod ledger entry's lifetime tracks the
process (Q4-4A). `#include "cipher_rt_green_ctx.h"` added.

### Sub-component 3 — ARB-poll retirement

ARB drove SM allocation through ioctl nr 9, which the CP 5.4 kmod deactivates
(`-ENOSYS`); its 30 s poll thread would fail every wake.
- `cipher_inject.c`: the `cipher_rt_arb_init()` call removed (the slot is now
  the `cp54_init()` call).
- `cipher_rt_partition_router.c`: the `cipher_rt_arb_request_partition()` call
  and its `#include "cipher_rt_arbitrate.h"` removed (orientation confirmed
  ARB output had **zero consumers**).
- `cipher_v2_internal.h`: `cipher_rt_arb_init` declaration removed.
- `Makefile`: `cipher_rt_arbitrate.o` dropped from `OBJS`.
- `cipher_rt_arbitrate.c`: kept in-tree, unbuilt, with a RETIRED banner.

Each sub-component was compile-checked in isolation before the next
(`cipher_rt_green_ctx.o` ✓, `cipher_inject.o` ✓).

## Build

`make TARGET=libcipher_rt.so.new` after `rm -f *.o` (full rebuild of all 17
TUs against current headers). Clean — **no errors; no warnings from any
modified TU** (`cipher_rt_green_ctx.c`, `cipher_inject.c`,
`cipher_rt_partition_router.c`). The warnings emitted are all pre-existing and
from untouched files (`cipher_rt_tenant.cpp` `_GNU_SOURCE`/parens/strncpy;
torch headers' unused-params; `cipher_rt_attn_dispatch.cpp` `env_on`). No
`undefined reference` at link — confirms nothing still referenced the removed
ARB symbols.

- New artifact: `cipher_rt_phase4/libcipher_rt.so.new`,
  md5 **`ebc0baaa4e771abbdd20f6c4af645040`**.
- Canonical `cipher_rt_phase4/libcipher_rt.so` verified **`a7ac8e97`** both
  before and after the build — untouched.

## Symbol-table diff (`nm -D --defined-only`, new vs `a7ac8e97`)

```
REMOVED (5):  cipher_rt_arb_close  cipher_rt_arb_init  cipher_rt_arb_last_count
              cipher_rt_arb_last_mask  cipher_rt_arb_request_partition
ADDED   (1):  cipher_rt_green_ctx_cp54_init
```
Exactly the expected delta — the 5 ARB symbols retired, the 1 new CP 5.4 entry
point added; nothing else changed (defined-symbol count 93 → 89). Newly
undefined: `strcasecmp`, `strtoul` (glibc — pulled by the qos/SM-count env
parsing); resolved by `libc`. `DT_NEEDED` unchanged (no new shared-lib
dependency); `ldd` resolves all.

## LOC

`cipher_rt_green_ctx.c` grew 289 → 395 lines (+106 file lines; ~60 net lines
of new executable logic — `cp54_init` + the ALLOCATE-driven selection — the
remainder is the rewritten header/inline documentation). This is consistent
with the Step 1.3 design memo's pre-stated estimate ("~40–60 LOC net … near
the >50 threshold; will report exact count"); the work landed where scoped —
no scope-undersizing surfaced. `cipher_inject.c`/`partition_router.c` net
near-zero (one call swapped, one call + include removed).

## Anchors

- `libcipher_rt` canonical **unchanged — `a7ac8e97`**. New build staged at
  `libcipher_rt.so.new` (`ebc0baaa`); promotion only on full Phase D PASS.
- Fallbacks `libcipher_rt.so.pre_cp5_4_step3` (`a7ac8e97`, both
  `cipher_rt_phase4/` and `cipher_rt_fallback/`) intact.
- kmod `8d777dfb`, libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d` —
  unchanged.

**Phase C BUILD PASS → proceeding to Phase D (integration tests A–F).**
