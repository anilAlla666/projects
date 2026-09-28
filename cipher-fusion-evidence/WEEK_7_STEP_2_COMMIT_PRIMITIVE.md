# WEEK_7_STEP_2_COMMIT_PRIMITIVE.md

**Date:** 2026-05-23
**Step:** v1.2.3 §7 W7-9 Step 2 — COMMIT atomic state-transition primitive core (per `CIPHER_REENGINEERING_PLAN.md` §4.8).
**Pre-tag (cipher_rt_phase4):** `week-5-complete` / `ec0e005`.
**Close tag (cipher_rt_phase4):** **`week-7-step-2-commit-primitive`** (`24f906d`).
**cipher_kmod:** unchanged at `week-7-step-1-g10-abi-scaffold` (`a9d18aa`, 0.5.5) — **Case A** per Part C below.
**Scope-lock parent:** `WEEK_7_9_SCOPE_LOCK.md` §4 (cipher-fusion-evidence `58c03e4`).

---

## 1. Pre-condition verification

| Anchor | Expected | Disk | Match |
|---|---|---|---|
| cipher-fusion-evidence HEAD | post-Step-1 | `d3caa85` | ✓ |
| cipher_kmod | tag `week-7-step-1-g10-abi-scaffold` | `a9d18aa` (kmod 0.5.5, srcversion `C919E44DC2336F0F60A99A4`) | ✓ |
| cipher_rt_phase4 | `week-5-complete` `ec0e005` | `ec0e005…` | ✓ |
| `cipher_pid_stats` reserved tail post-Step-1 | `reserved[10]` = 40 B | confirmed at `cipher_internal.h:313` | ✓ |
| libcipher_rt.so md5 (pre) | W5-close canonical | `259ac994aead2da8289fc84d6116fbe9` | ✓ |
| cipher_rt_phase4 IS a git repo | per pre-condition `git rev-parse HEAD` | yes (can commit + tag) | ✓ |

**Baselines preserved at `/tmp/step2_baseline/`:**

| File | md5 |
|---|---|
| `libcipher_rt.so.pre` | `259ac994aead2da8289fc84d6116fbe9` |
| `cipher_kmod.ko.pre` | `2e36cd99276082eab228ded34121898b` (Step-1 close) |
| `cipher_inject.c.pre` | `71b4607e24b95ee88cd1959de991d071` |
| `Makefile.pre` | `840415fe0c7f8fd2eb4a32c8089c1cde` |
| `cipher_state_updater.c.pre` | `18dbd9ab5c64ffd0785b664333e184c7` |

---

## 2. §4.8 binding-spec extraction (Part A.3-A.4)

Read of `CIPHER_REENGINEERING_PLAN.md` v1.2.3 lines **893–919**:

- **5-step deterministic order** at L899–905 (unambiguous, owner-named):
  1. **AUDIT chain HMAC entry** — `cipher_rt_audit_record` (cipher_rt_audit.c)
  2. **FAIRNESS quota + arbitration** — `cipher_fairness_observe` + cipher_fairness_shm aggregate
  3. **CARBON power accounting** — `cipher_carbon_observe`
  4. **RECEIPT billing surface** — `cipher_receipt_observe`
  5. **kmod-resident tenant state** — `cipher_pid_stats` per-launch fields via `state_updater` 1 kHz mirror

- **Per-tenant sequence-counter discipline** (L909): `__atomic_fetch_add(relaxed)` on write open, `__atomic_load_explicit(acquire)` on read, `__atomic_store_explicit(release)` on publish. **Per-tenant slot — NOT a global lock**.

- **Rollback path** (L1528): `CIPHER_COMMIT_MODE=legacy` → COMMIT no-op; observers fall back to v1.2.1 per-observer state mutation.

- **Engineering estimate** (L919): 10 days W7-8 total for COMMIT primitive + 21 overlay ops port + kmod state_updater mirror. Step 2 scope is the primitive core only (~3 of those 10 days); Step 3 + Step 4 carry the rest.

**Doc-gap RESOLVED**: §4.8 L909 didn't name the per-tenant seq counter type. Step 2 implementation uses **`atomic_uint_least64_t`** (chosen for 64-bit width + maximum portability across compilers; on Linux x86_64 this is a 64-bit atomic, single instruction on TSO).

**No HARD STOP** on ambiguous spec — all 5 steps have well-defined owners.

---

## 3. Part C — Case A vs Case B determination

Read of `cipher_kmod/cipher_state_updater.c` (lines 100–137):

The state_updater kthread runs at 1 kHz and writes per-tenant derived fields via `WRITE_ONCE(e->field, value)` — `thermal_headroom_pct`, `power_headroom_w`, `sustained_clock_mhz`, `voltage_envelope_mv`, `fairness_quota_remaining_pct`, `snapshot_jiffies` (lines 110–136). `WRITE_ONCE` is relaxed-atomic on Linux kernel semantics.

**Case A applies**: the userspace COMMIT primitive's seqlock writer pattern wraps these writes atomically from the consumer's perspective. When COMMIT publishes its snapshot:
1. Reader acquire-loads `snap->seq` (sees the writer-done even value)
2. Reader reads `snap->kmod_tenant_state` (the field COMMIT copied at end-time from the kmod's latest state)
3. Reader acquire-loads `snap->seq` again to detect any concurrent writer

The kmod side does NOT need release-fence in Step 2 — its `WRITE_ONCE` writes are visible to userspace via mmap'd snapshot in the relaxed-coherent sense, and the userspace COMMIT primitive's release-store of `seq` is the synchronisation point.

Explicit release semantics on the kmod side will land in **Step 3 alongside G6** (kmod-resident HMAC chain head requires release ordering against the AUDIT chain write).

**Step 2 verdict**: no `cipher_kmod` change. cipher_kmod stays at tag `week-7-step-1-g10-abi-scaffold` (`a9d18aa`, 0.5.5).

---

## 4. The edits — cipher_rt_phase4 (4 files, +259 LOC)

### 4.1 `cipher_rt_commit.h` (new, 113 LOC)

Public contract. Exposes:

- `struct cipher_rt_snapshot` — 9 fields, ~72 B:
  - `atomic_uint_least64_t seq` — seqlock (odd = writer-in-progress, even = published)
  - `audit_chain_head` / `fairness_quota` / `carbon_joules_x1e6` / `receipt_seq` / `kmod_tenant_state` — the 5 §4.8 slots
  - `model_uuid_lo` / `model_uuid_hi` — W7-9 Step 1 G10 propagation (lo/hi split because struct members are simpler than `uint8_t[16]`)
  - `commits_total` — per-tenant counter visible to Step 4 _report() consumers
- `struct cipher_rt_commit_fields` — input bundle to `cipher_rt_commit_end` (the 5+G10 fields the caller pre-computes)
- `int  cipher_rt_commit_init(void)` — idempotent init via `atomic_compare_exchange_strong`
- `uint64_t cipher_rt_commit_begin(uint32_t tenant_id)` — seqlock open (bump to odd)
- `void cipher_rt_commit_end(uint32_t, uint64_t seq, const struct cipher_rt_commit_fields *)` — 5-step deterministic stores + release-store seqlock close
- `const struct cipher_rt_snapshot *cipher_get_current_tenant_snapshot(uint32_t)` — Step 4 _report() consumer accessor
- `static inline int cipher_rt_snapshot_consistent(snap, s1)` — seqlock reader helper (for test_commit_atomicity)
- `uint64_t cipher_rt_commit_total_count(void)` — global counter for throughput verification
- `#define CIPHER_RT_MAX_TENANTS 128` — matches W6 G1 CIPHER_CP54_MAX_ALLOCS

### 4.2 `cipher_rt_commit.c` (new, 140 LOC)

- BSS-resident `g_snapshots[CIPHER_RT_MAX_TENANTS]` cache-line-padded array (~9 KiB total at 128 × 72 B aligned to 64 B).
- `cipher_rt_commit_init`: atomic_compare_exchange-guarded idempotent init.
- `cipher_rt_commit_begin`: `atomic_fetch_add(&s->seq, 1, memory_order_relaxed)` → returns the now-odd seq.
- `cipher_rt_commit_end`: 5+G10 field stores (plain stores; the release-store of seq below makes them visible) → `s->commits_total = seq >> 1` → `atomic_store_explicit(&s->seq, seq + 1, memory_order_release)`.
- All public symbols (`cipher_rt_commit_init`, `_begin`, `_end`, `_total_count`, `cipher_get_current_tenant_snapshot`) are `T` (text, externally visible).

### 4.3 `cipher_inject.c` (+2 LOC)

Added `#include "cipher_rt_commit.h"` and one-line `cipher_rt_commit_init()` call in `cipher_v2_init_body()` after `cipher_rt_audit_init()`. Init order: tenant → CP54 → SMP → PR → CUPTI → VOLT → matmul dispatch → Marlin → attn dispatch → attn test → AUDIT → **COMMIT** → GOT patch. COMMIT init must precede any per-launch hook that will eventually call `cipher_rt_commit_*` (Step 4 work).

### 4.4 `Makefile` (+4 LOC)

`cipher_rt_commit.o` added to `OBJS` list (between `cipher_rt_sense_transition.o` and `cipher_may13_harness.o`); per-file build rule added before `cipher_rt_audit.o:`.

### 4.5 Diff stat

```
 Makefile           |   4 ++
 cipher_inject.c    |   2 +
 cipher_rt_commit.c | 140 +++++++++++++++++++++++++++++++++++++++++++++++++++++
 cipher_rt_commit.h | 113 ++++++++++++++++++++++++++++++++++++++++++
 4 files changed, 259 insertions(+)
```

---

## 5. Build + smoke (Part D)

### 5.1 Build

```
$ make
[...]
cc -O2 -Wall -Wextra -fPIC -I/home/ubuntu/cipher_kmod -I/usr/include -I/usr/include -Iinclude -c -o cipher_rt_commit.o cipher_rt_commit.c
g++ -shared -fPIC -o libcipher_rt.so [...] cipher_rt_commit.o [...]
```

- Errors: **0**
- Warnings: **0**
- `libcipher_rt.so` md5 (pre) `259ac994…` → (post) **`f858f17f98be4e0e170e373dc897eacf`**

### 5.2 Exported symbols verified

```
$ nm -D libcipher_rt.so | grep -E "cipher_rt_commit|cipher_get_current_tenant_snapshot"
0000000000016870 T cipher_get_current_tenant_snapshot
00000000000167c0 T cipher_rt_commit_begin
00000000000167f0 T cipher_rt_commit_end
0000000000016730 T cipher_rt_commit_init
00000000000168a0 T cipher_rt_commit_total_count
```

All 5 symbols `T` (text, exported, callable from external consumers like Step 4's overlay-op port).

### 5.3 `test_commit_atomicity.c` — 4/4 PASS

Test source: `/tmp/step2_baseline/test_commit_atomicity.c` (~200 LOC). Compiled standalone and linked against `libcipher_rt.so`.

| Case | Description | Result |
|---|---|---|
| 1 | Single-tenant 10000 commits: final seq even, fields coherent (last `audit_chain_head` == N-1) | **PASS** — final seq=20000 |
| 2 | N=15 concurrent tenants × 10000 commits each. Per-tenant `audit_chain_head` encodes `(tid << 32) \| i`; verifies zero cross-tenant interference | **PASS** — all 15 tenants final state correct |
| 3 | 1 writer + 2 readers contention (1 s sustained). Writer commits a self-consistent quintuple `(i, i+1, i+2, i+3, i+4)`; readers verify coherence under seqlock retry | **PASS** — 3.78 M writer commits, 11.59 M coherent reads, 34.86 M retries (retries normal — seqlock pattern absorbs writer-in-progress windows), zero torn-read sentinel triggers |
| 4 | Overhead microbench: 100 000 begin/end cycles, mean/p50/p99 | **PASS** — mean 47 ns, p50 47 ns, **p99 49 ns vs 200 ns budget = 4× under**; max 10 463 ns (scheduler-preemption blip) |

`cipher_rt_commit_total_count() = 4 047 182` after the run (sum of all Case 1+2+3+4 commits) — global counter incremented atomically across all tenants.

---

## 6. Regression gates (Part E) — all PASS

| Gate | Result |
|---|---|
| **E.1 W6 G1+G2 caps** | `test_cap65 N=128` 128/128 + `test_arena17 N=100` 100/100 byte-identical |
| **E.2 W7 Step 1 register_model** | 5/5 PASS, MODULE_VERSION still 0.5.5 (Case A — no kmod change) |
| **E.3 CP 5.4 isolation** | 15/15 PASS byte-identical to v1.2.2 baseline |
| **E.4 libcipher_rt.so loader smoke** | Implicit in E.5 (vLLM loaded the new libcipher_rt.so cleanly); symbols visible per §5.2 |
| **E.5 vLLM Mistral-7B B=1 decode smoke** | **163.5 tok/s** mean (163.6 / 163.4 / 163.5) — **EXACTLY matches W7 Step 1 baseline of 163.5**, delta **0.0%**. decode_MFU 0.249% byte-identical. REGISTER_MODEL fires (`arch=1 uuid=5c30f3f70c698687a10352d3836497ec`) — Step 1 functionality unchanged. |
| **E.6 N=15 contention** | Covered by `test_commit_atomicity` Case 2 — N=15 × 10000 commits with zero cross-tenant interference |

**No regression**. Step 2 doesn't wire COMMIT into the per-launch hot path (Step 4 does), so the 49 ns overhead is invisible to vLLM end-to-end TPS.

---

## 7. Final state + anchors

| Field | Value |
|---|---|
| `cipher_rt_phase4` HEAD | **`24f906d`** (tag `week-7-step-2-commit-primitive`) |
| `libcipher_rt.so` md5 (pre) | `259ac994aead2da8289fc84d6116fbe9` (W5-close canonical) |
| `libcipher_rt.so` md5 (post) | **`f858f17f98be4e0e170e373dc897eacf`** |
| New symbols (5) | `cipher_rt_commit_{init,begin,end,total_count}` + `cipher_get_current_tenant_snapshot` |
| `cipher_kmod` HEAD | `a9d18aa` (tag `week-7-step-1-g10-abi-scaffold`) — **unchanged (Case A)** |
| `cipher_kmod.ko` md5 | `2e36cd99276082eab228ded34121898b` (Step-1 close) — unchanged |
| MODULE_VERSION | 0.5.5 — unchanged (Case A) |
| COMMIT overhead p99 | **49 ns** (budget 200 ns; 4× under) |

---

## 8. Honest residue

- **COMMIT is not yet wired into the per-launch hot path.** Step 2 builds the primitive + init hook; Step 4 will wire `cipher_rt_commit_end()` into the GOT-patched dispatch return points (cublasGemmEx + SDPA shims). The 49 ns overhead measured in Case 4 is the *primitive cost*; the per-launch cost will be that plus a small wrapper. Step 4's regression gate is the in-vivo TPS check; Step 2's gate is the primitive correctness.
- **The 5 snapshot fields are populated with caller-supplied values in Step 2.** Step 4's overlay-op port will wire each `_report()` consumer to read the snapshot AND will wire each per-launch hook to populate the fields. Step 2's contract is "if you give me fields, I publish them atomically"; Step 4 fills in the per-op source plumbing.
- **`commits_total` is `seq >> 1` rather than its own counter** — saves one atomic per COMMIT. Reader who wants the total commit count for a tenant just reads `commits_total` from the snapshot under the seqlock pattern.
- **Step 3 G6 will replace the userspace `audit_chain_head` field source** with the kmod-resident chain head, requiring (a) the cipher_pid_stats HMAC accumulator addition (forcing the 336 → 400 B struct grow and ABI 0.5.5 → 0.6.0), (b) the new mmap'd shm device for the chain ring. Step 2 leaves `audit_chain_head` as a plain `uint64_t` field; Step 3 will read it from the new shm.

---

## 9. v1.2.3 §7 W7-9 progression

- ✓ **Step 1 G10 ABI scaffolding** (`week-7-step-1-g10-abi-scaffold` / `a9d18aa`; kmod 0.5.5)
- ✓ **Step 2 COMMIT primitive core** (this; `week-7-step-2-commit-primitive` / `24f906d` on cipher_rt_phase4)
- Step 3 G6 kmod-resident AUDIT chain — pre-tag `week-7-step-2-commit-primitive`; close tag `week-8-step-3-g6-audit-chain`. **ABI bump 0.5.5 → 0.6.0** (struct size grow).
- Step 4 21 overlay-ops `_report()` port — close tag `week-8-step-4-overlay-ops-port`
- Step 5 N=128 contention soak → `week-9-complete`

**Next implementation prompt**: Step 3 G6 kmod-resident AUDIT chain (~3 eng-days; ~550 LOC; ABI bump to 0.6.0; fresh mmap infrastructure for the per-tenant HMAC chain ring buffer).
