# Track 3 SC5-2 — BUILD LOG

**Date:** 2026-05-19. **Scope:** the folded SC4 policy build + the SC1 fact-2
constrained POOL grant + the 5-partition churn harness. kmod-only.

## Source changes (kmod — 3 files)

| file | change |
|---|---|
| `cipher_cp54_sched.c` | **SC4 fold** — 4 `module_param`s (0644): `cipher_cp54_mig_gap_min_grps`=1, `cipher_cp54_mig_sustain_ms`=2000, `cipher_cp54_mig_ratelimit_ms`=10000, `cipher_cp54_mig_verbose`=1; `cp54_eval_migration` reads them live; per-tenant `migration_count`; 5 aggregate stat atomics. **Constrained grant** — `cp54_pool_claim_low_prefix()`; the POOL branch of `cipher_cp54_ioctl_allocate` calls it instead of `cp54_claim(all 15)`. The `/proc` emitter `cipher_cp54_migration_proc_show()`. |
| `cipher_proc.c` | `/proc/cipher/migrations` node (open fn + `proc_ops` + `proc_create`/`proc_remove`) |
| `cipher_internal.h` | `CIPHER_PROC_MIGRATIONS` define; `cipher_cp54_migration_proc_show` prototype |

`cipher_ioctl.h` untouched — **no ABI change** (ABI frozen at nrs 1-20).
libcipher_rt `83afd1ca`, libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`
untouched.

## The constrained POOL grant (SC5 design memo §1 finding)

`cp54_pool_claim_low_prefix(pid)` claims the **maximal contiguous low prefix**
[0, K-1] — walking groups from 0, claiming each free-or-already-POOL group,
stopping at the first partition-owned or RSVD group. A free group above that K
is left free (POOL-unreachable; DSM compacts it back). This replaces the prior
`cp54_claim(pid, NUM_GROUPS)` which claimed *every* free group — the SC1 fact-2
B-floor that was specified but never built (verified absent at SC5 design,
`cipher_cp54_sched.c:527`). It is now in the kmod, present in **both** the A
config (migration on) and B config (migration off) — the shared safety floor.

## Module parameters — verified live

`/sys/module/cipher_kmod/parameters/`: `cipher_cp54_mig_gap_min_grps`,
`cipher_cp54_mig_sustain_ms`, `cipher_cp54_mig_ratelimit_ms`,
`cipher_cp54_mig_verbose` — all mode 0644, reading the production defaults
(1 / 2000 / 10000 / 1). A-vs-B is `gap_min_grps` = 1 vs 16, no rebuild.

## /proc/cipher/migrations — verified live

```
policy:   gap_min_grps=1 sustain_ms=2000 ratelimit_ms=10000 verbose=1
gap:      stranded_groups=0 gap_age_ms=0
counters: proposals=0 commits=0
          aborts: timeout=0 tenant_nack=0 kmod_refused=0
```
(per-tenant rows appear when migratable tenants are registered).

## Build

`make` — **clean**, no warning/error from `cipher_cp54_sched.c` or
`cipher_proc.c`. Built `cipher_kmod.ko` md5
**`285d102e86eafb70e95040fd1c2d44d0`** (SC5 anchor `285d102e`).

## 5-partition churn harness (SC5-3 measurement runner)

`sc5_tenant.py` (one PARTITION decode tenant, scheduled-lifetime exit =
the 'free' event) + `sc5_churn.py` (orchestrator: 5 seeds — LIFO / FIFO /
MIDDLE_OUT / RAND_A / RAND_B — × A/B config via the `gap_min_grps` param;
COMPACT + `/proc/cipher/migrations` sampling; the 6 metric groups;
`SC5_DESTROY_FAULT=1` for destroy-fault at scale). Syntax-checked
(`py_compile` clean). **Not run in SC5-2** — the measurement run is SC5-3.

## Verification

Regression smoke — `TRACK_3_SC5_REGRESSION.md` — PASS. Anchor rotation +
closeout details there and in `TRACK_3_ANCHORS.md`.
