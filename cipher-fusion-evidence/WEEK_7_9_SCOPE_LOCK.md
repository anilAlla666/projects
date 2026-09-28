# WEEK_7_9_SCOPE_LOCK.md

**Date:** 2026-05-23
**Scope:** v1.2.3 §7 W7-9 implementation plan — COMMIT atomic state-transition primitive + G6 kmod-resident AUDIT chain + G10 CIPHER_REGISTER_MODEL ABI.
**Read-only research + planning deliverable. No source edits beyond a small v1.2.3 §7 W6-row close-marker.**

This scope-lock is the W4-pattern parent for **five implementation prompts** that follow (Step 1 → Step 5).

---

## 1. Pre-conditions verified

| Anchor | Expected | Disk | Match |
|---|---|---|---|
| `cipher-fusion-evidence` HEAD | post-W6 May-13 plan | **`970694b`** | ✓ |
| `cipher_kmod` | tag `week-6-step-g1-g2-cap-bump` | **`c4e2d6f`**, MODULE_VERSION 0.5.0 | ✓ |
| `cipher_rt_phase4` | `ec0e005` (week-5-complete) | `ec0e005…` | ✓ |
| `CIPHER_REENGINEERING_PLAN.md` | v1.2.3 (1907 lines) | 216 675 bytes | ✓ |
| `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` | md5 `c5d2d4ad…` | 67 712 bytes | ✓ |
| `WEEK_6_MAY13_RECONSTRUCTION_PLAN.md` | md5 `11e79cc8…` | 23 042 bytes | ✓ |
| Post-W6 kmod evidence | constants + MODULE_VERSION at v0.5.0 | `cipher_cp54_sched.c:123` = 128 / `cipher_ioctl.h:481` = 100 / `cipher_main.c:141` = "0.5.0" / `cipher_proc.c:155` banner = "0.5.0" | ✓ |

No HARD-STOP conditions met.

---

## 2. Binding-spec extraction (file:line citations)

Per Agent A read of `CIPHER_REENGINEERING_PLAN.md` v1.2.3:

- **§4.8 COMMIT atomic state-transition primitive** at **lines 893–920** (28 lines).
  - Deterministic step order at L899–905: **AUDIT chain HMAC entry → FAIRNESS quota/arbitration → CARBON power accounting → RECEIPT billing surface → kmod-resident tenant state** (via state_updater 1 kHz mirror).
  - Per-tenant sequence-counter discipline at L909: `__atomic_fetch_add` (relaxed) write side; `__atomic_load_explicit(acquire)` read side; `__atomic_store_explicit(release)` snapshot publish. No global lock.
  - Release-fence snapshot publish contract at L907, L911–915: canonical post-kernel state snapshot stored in `cipher_pid_stats`; observers read from snapshot.
  - Rollback path at L1528: `CIPHER_COMMIT_MODE=legacy` → per-observer state mutation (v1.2.1 behaviour).
  - Engineering estimate at L919: 10 days (Weeks 7-8) for the COMMIT primitive alone (pre-fold-in).
- **§4.9 RING_WRITE lock-free inline telemetry substrate** at **lines 921–957**. Per-tenant SPMC ring, 65 536 entries; producer ~10 ns on x86 TSO (one relaxed load + comparison + 64 B memcpy + one release store). **Critical for Step 3 design**: §4.9 specifies the per-tenant ring is the AUDIT-chain producer's natural sink (L940, L950, L952). G6's HMAC chain head is populated by RING_WRITE entries via an AUDIT consumer thread that lands in W10-12; Step 3 builds the producer + the storage substrate, W10-12 lands the consumer.
- **§7 W7-9 section** at **lines 1500–1537** (38 lines): verification gate (N=128 contention), rollback path (`CIPHER_COMMIT_MODE=legacy`), and all five risks R-W7.1 through R-W7.5 (see §8 below).
- **§8.5 R-G6.1 + R-G10.1 entries** at **lines 1729–1742**.

**Doc-gap (not blocking):** §4.8 L909 specifies "single `__atomic_fetch_add`" for the per-tenant sequence counter but does NOT name the type. Step 2 implementation should specify `atomic_uint64_t` (the architecturally-canonical choice for a 64-bit-counter per tenant slot). Flagged as **Step 2 sub-task documentation hardening**; not a substrate spec gap.

Per Agent C read of substrate:

- `struct cipher_pid_stats` at **`cipher_internal.h:245–311`**. Reserved tail: **`reserved[14]` = 56 bytes** (post-W1 Cb.2 bump). Userspace mirror `cipher_tenant_snapshot_user` at same layout, currently locked at 336 B.
- Ioctl NR 26 (`CIPHER_DSM_PROPOSE`) is the highest assigned. **NR 27 is unassigned** and additive-only per `[[cipher-abi-rule]]` (no NR repurposing).
- `cipher_tenant_snapshot.c` uses **per-CPU TLS assembly buffer + `copy_to_user` via ioctl NR 8**, NOT a persistent mmap'd shm region. **G6 requires fresh mmap infrastructure**, not reuse of this pattern — this is a sharpening of the audit's "reuse existing snapshot shm" framing.
- Ioctl dispatch insertion point: **`cipher_dev.c:289-290`** (after `case CIPHER_DSM_PROPOSE`, before the `-ENOSYS` block at L291).

Per Agent B read of overlay ops in `cipher_rt_phase4/src/may13/`:

- **18 standalone `_report()` functions + 2 sub-dispatchers (LAYER3 at `dispatch.cpp:525`, ORACLE_BILLING at `oracle.cpp:509`) + 3 infrastructure (L2_PERSIST header at `l2_persist.h:95`, LIQUID_STATE, GREEN_CTX) = 23 total report-like functions**, not 21 as the spec/audit cited. **Documenting this discrepancy in §6 below; Step 4's port surface is 18 + 2 = 20 ports plus 3 infra audits.**
- Most ports are mechanical (~5 LOC each: replace direct counter access with snapshot field access). Total mechanical port LOC ~95.
- **Two ports are non-mechanical**: RECEIPT (HMAC chain accumulator; composes with G6 chain head — handle in Step 3, not Step 4) and DETERMINISM (XOR'd dispatch-fingerprint accumulator; requires snapshot pin under COMMIT to guarantee deterministic hash under concurrent dispatch). Adds ~30-40 LOC for the refactor.
- KERNEL_TABLE (atexit dump at `cipher_kernel_table.cpp:312`) is **not** a standard `_report()` — lifecycle is different; not in Step 4 scope.
- FAIRNESS_SHM is referenced in the audit but not found as a separate `_report()` in may13/. Step 4 covers FAIRNESS only; FAIRNESS_SHM cross-tenant work is the W10-12 G3/G4 follow-on.

---

## 3. Step 1 — G10 ABI scaffolding (CIPHER_REGISTER_MODEL ioctl NR 27)

**Tag at close:** `week-7-step-1-g10-abi-scaffold` (on `cipher_kmod` + on `cipher_rt_phase4` for the plugin-side change).

### 3.1 Scope (~2 eng-days, ~200 LOC kmod + ~50 LOC plugin)

**kmod side (cipher_kmod tag week-7-step-1):**

- `cipher_kmod/cipher_ioctl.h`: new ioctl `CIPHER_REGISTER_MODEL` at NR 27 (insertion at L558 after `CIPHER_DSM_PROPOSE`). Payload struct:
  ```c
  struct cipher_register_model {
      char     model_path[PATH_MAX];     /* in : canonical model dir path */
      __u8     hf_config_hash[32];       /* in : sha256 over hf_config canonical JSON */
      __u32    model_arch;               /* in : enum MISTRAL/QWEN/LLAMA/GPT_NEOX/OTHER */
      __u8     model_uuid_out[16];       /* out: 128-bit UUID assigned by kmod */
      __u32    reserved[4];
  };
  #define CIPHER_REGISTER_MODEL _IOWR(CIPHER_IOCTL_MAGIC, 27, struct cipher_register_model)
  ```
- `cipher_kmod/cipher_model_registry.c` (new, ~150 LOC): hashtable keyed on `hf_config_hash` (xxhash64 of the 32 B sha256 → 64-bit bucket key), value is `{model_uuid, ref_count, model_arch}`. **Idempotent**: lookup before insert, same `hf_config_hash` always returns the same `model_uuid`. Reference-counted; `cipher_model_release(uuid)` on tenant exit (kprobe hook into existing `cipher_probe.c:184-221` do_exit reaper).
- `cipher_kmod/cipher_dev.c`: dispatch table gains NR 27 handler at L289-290 (insert after `CIPHER_DSM_PROPOSE`, before the `-ENOSYS` block).
- `cipher_kmod/cipher_internal.h`: `struct cipher_pid_stats` gains `__u8 model_uuid[16]` field. **Fits in current `reserved[14] = 56 B` tail**: consumes 16 B, leaves 40 B for Step 3 (which needs 64 B → forces 336→400 B struct growth at Step 3).
- `cipher_kmod/cipher_main.c:141`: `MODULE_VERSION("0.5.0") → MODULE_VERSION("0.5.5")` — additive ioctl + additive struct field within reserved tail; no userspace ABI break.
- `cipher_kmod/cipher_proc.c:155`: `/proc/cipher/stats` banner string `"cipher_kmod 0.5.0" → "cipher_kmod 0.5.5"` (matches the W6 G1+G2 pattern of bumping both MODULE_VERSION and the proc display string).

**Plugin side (cipher_rt_phase4 tag week-7-step-1):**

- `cipher_vllm_plugin/cipher_vllm_kv.py`: at engine init (before first `cipher_kv_bridge.init()`), call `CIPHER_REGISTER_MODEL` ioctl with the model's `hf_config_hash` (already computed for the existing tenant-id wiring). Cache the returned `model_uuid` in the plugin's engine state. Propagate `model_uuid` into the existing `cipher_kv_bridge.vmm_zeros(...)` call signature (replaces `_TENANT`-only argument with `(_TENANT, _MODEL_UUID)`).
- **Backward-compat**: if `REGISTER_MODEL` returns `-ENOSYS` (old kmod loaded), fall back to `model_uuid = MODEL_UNKNOWN` sentinel (all-zeros UUID). Downstream model-keyed actuators (G3 KV-dedup in W10-12, G4 Marlin in W10-12, G12 Koopman in W13-14) treat MODEL_UNKNOWN as pass-through.
- `cipher_rt_phase4/cipher_kv_bridge.cpp:43-64` (`struct cipher_rt_kv_page_tag`): the `model_uuid` field already exists per architecture-gap audit §2.3; Step 1 just wires it from the plugin side.

### 3.2 Behavioral gate

- `test_register_model.c`: `REGISTER_MODEL` with same `hf_config_hash` twice returns same `model_uuid`.
- 5 concurrent `REGISTER_MODEL` calls from 5 distinct PIDs return 5 idempotent (same-hash → same-uuid) `model_uuid`s.
- `REGISTER_MODEL` round-trip < 100 µs (`-Wframe-larger-than=1024` budget; payload `PATH_MAX + 32 + …` ≈ 4 KiB → already heap-allocate per the cipher_wa_ioctl_query kzalloc pattern landed in W6).
- Idempotency across kmod reload (close+reopen `/dev/cipher`; lookup by `hf_config_hash` returns same uuid).

### 3.3 Regression gate

- W6 G1+G2 caps regression: `test_cap65` N=128 + `test_arena17` N=100 still PASS.
- CP 5.4 isolation test 15/15 byte-identical to v1.2.2 baseline (`cipher_cp54_sched.c:445+ cp54_eval_migration` unchanged).
- `cipher_vllm_kv.py` vLLM smoke (single Mistral-7B B=1 decode via `bench_llm.py`) runs clean with the new REGISTER_MODEL call site; per-tenant `model_uuid` visible in `/proc/cipher/stats`.

### 3.4 Rollback path

Set `CIPHER_REGISTER_MODEL_ENABLE=0` → kmod's `cipher_dev_register_model` returns `-ENOSYS` → plugin falls back to MODEL_UNKNOWN sentinel → downstream actuators pass-through. `cipher_kmod.ko.pre_week7_step1` preserved. Anchor rolls back to `week-6-step-g1-g2-cap-bump` (`c4e2d6f`, ABI 0.5.0).

### 3.5 Per-actuator file:line table

| File | Lines | Change |
|------|-------|--------|
| `cipher_kmod/cipher_ioctl.h` | insert after L558 | new `struct cipher_register_model` + `#define CIPHER_REGISTER_MODEL` |
| `cipher_kmod/cipher_model_registry.c` | new file ~150 LOC | hashtable + uuid generator + ref-count |
| `cipher_kmod/cipher_dev.c` | insert after L290 | dispatch case for `CIPHER_REGISTER_MODEL` |
| `cipher_kmod/cipher_internal.h:245-311` | tail | `__u8 model_uuid[16]` added to `cipher_pid_stats` |
| `cipher_kmod/cipher_main.c:141` | replace | MODULE_VERSION 0.5.0 → 0.5.5 |
| `cipher_kmod/cipher_proc.c:155` | replace | banner 0.5.0 → 0.5.5 |
| `cipher_vllm_plugin/cipher_vllm_kv.py` | engine init | `CIPHER_REGISTER_MODEL` call + uuid propagation; backward-compat sentinel |
| `cipher_rt_phase4/cipher_kv_bridge.cpp:43-64` | tag field | wire `model_uuid` from plugin (struct already exists) |

### 3.6 Eng-days breakdown

| Sub-task | Days |
|---|---|
| ioctl scaffolding + payload struct | 0.5 |
| `model_registry.c` implementation | 0.5 |
| `cipher_pid_stats` `model_uuid` field + struct invariant check | 0.25 |
| plugin-side REGISTER_MODEL call + propagation + backward-compat | 0.5 |
| test harness + gate validation | 0.25 |
| **Total** | **2.0** |

---

## 4. Step 2 — COMMIT primitive core

**Tag at close:** `week-7-step-2-commit-primitive` (on `cipher_rt_phase4`; small kmod state_updater touch tagged on `cipher_kmod` as well).

### 4.1 Scope (~3 eng-days, ~200 LOC userspace + ~50 LOC kmod)

- `cipher_rt_phase4/cipher_rt_commit.c` (new, ~150 LOC):
  - `cipher_rt_commit_begin()` + `cipher_rt_commit_end()` entry points
  - Per-tenant sequence counter — **`atomic_uint64_t commit_seq` in `cipher_pid_stats`** (resolves the §4.8 doc-gap; type explicitly named here)
  - 5-step deterministic state update under release-fence:
    1. **AUDIT chain HMAC accumulator advance** — Step 2 builds the userspace half + contract; Step 3 wires the kmod-resident half via the chain ring buffer (composes cleanly).
    2. FAIRNESS quota + arbitration counter publish.
    3. CARBON power accounting publish.
    4. RECEIPT billing-surface publish (composes with G6 via Step 3 since RECEIPT's HMAC chain *is* the G6 chain).
    5. kmod-resident tenant context publish (via the state_updater 1 kHz mirror).
  - Each step has a single deterministic owner; ordering is load-bearing for snapshot consistency under N=128 readers.
- `cipher_rt_phase4/cipher_rt_commit.h`: public contract — `cipher_get_current_tenant_snapshot()` returns a const pointer to the post-COMMIT snapshot, valid until the next COMMIT-end on this tenant. Sequence counter exposed via `cipher_get_commit_seq()` for the contention harness in Step 5.
- `cipher_kmod/cipher_state_updater.c`: 1 kHz mirror updated to publish post-COMMIT snapshot atomically (`__atomic_store_explicit(release)`) instead of mutating individual counters as today.
- `cipher_rt_phase4/cipher_inject.c`: GOT-patched dispatch intercept gains COMMIT call after dispatch return (~5 LOC). Specifically the `cipher_inject.c:50-56` GOT-patch list gains a post-dispatch hook into `cipher_rt_commit_end()`.

### 4.2 Why second

Step 1 (G10) gives Step 2 a stable `model_uuid` to include in the snapshot. Step 3 (G6 kmod chain) and Step 4 (overlay-op port) both consume COMMIT's contract; landing COMMIT before them means they consume a stable snapshot interface.

### 4.3 Behavioral gate

- W1 regression PASS within ±3% (per-tenant baseline TPS).
- N=15 concurrent tenants × 2 reader threads per tenant: contention harness races two snapshot reads from different threads; both reads observe the same `commit_seq` (atomicity verified by `__atomic_compare_exchange` no-spin assertion).
- COMMIT call overhead p99 ≤ 200 ns per intercept (warn at 100 ns, fail at 500 ns). Measured via cycle counter in the bench harness extension.

### 4.4 Regression gate

- 21 overlay ops self-tests still PASS at single-tenant (Step 4 ports them to snapshot-read; Step 2 just establishes the snapshot publish contract).
- Marlin smoke + VOLT smoke + Track 2 SC6 PASS.
- vLLM Mistral-7B B=1 decode runs clean against the new COMMIT intercept (forward pass not regressed beyond the budget).

### 4.5 Rollback path

`CIPHER_COMMIT_MODE=legacy` → COMMIT becomes a no-op; overlay ops fall back to v1.2.1 per-observer state mutation. `libcipher_rt.so.pre_week7_step2` preserved. Anchor rolls back to `week-7-step-1-g10-abi-scaffold`.

### 4.6 Per-actuator file:line table

| File | Lines | Change |
|------|-------|--------|
| `cipher_rt_phase4/cipher_rt_commit.c` | new ~150 LOC | primitive entry points + 5-step state-update sequencer |
| `cipher_rt_phase4/cipher_rt_commit.h` | new ~50 LOC | public contract — `cipher_get_current_tenant_snapshot()` |
| `cipher_rt_phase4/cipher_inject.c:50-56` | extend GOT list | post-dispatch hook into `cipher_rt_commit_end()` |
| `cipher_kmod/cipher_state_updater.c` | ~50 LOC modify | 1 kHz mirror publishes post-COMMIT snapshot atomically |
| `cipher_kmod/cipher_internal.h:245-311` | `cipher_pid_stats` | add `atomic_uint64_t commit_seq` field (still within reserved tail — model_uuid 16 B + commit_seq 8 B = 24 B used; 32 B remaining for Step 3) |

### 4.7 Eng-days breakdown

| Sub-task | Days |
|---|---|
| `cipher_rt_commit.c` primitive | 1.0 |
| `cipher_rt_commit.h` contract + overlay-op consumer pattern | 0.5 |
| kmod `state_updater.c` mirror | 0.5 |
| `cipher_inject.c` COMMIT call site | 0.25 |
| Contention harness + N=15 race test | 0.75 |
| **Total** | **3.0** |

---

## 5. Step 3 — G6 kmod-resident AUDIT chain fold-in

**Tag at close:** `week-8-step-3-g6-audit-chain` (on `cipher_kmod`).

### 5.1 Scope (~2.5–3 eng-days, ~550 LOC kmod + ~50 LOC userspace)

**This step's audit-doc estimate of ~400 LOC underestimates by ~150 LOC** — the architecture-gap audit assumed reuse of the existing `cipher_tenant_snapshot.c` shm pattern, but Agent C found that `cipher_tenant_snapshot.c` uses **per-CPU TLS assembly + `copy_to_user` via ioctl NR 8**, NOT a mmap'd shm. G6's per-tenant append-only ring buffer requires **fresh mmap infrastructure** (`alloc_pages` + `remap_pfn_range` + `file_operations.mmap`). Adds ~150 LOC to the prior estimate.

- `cipher_kmod/cipher_audit_chain.c` (new, ~550 LOC):
  - Per-tenant HMAC-SHA256 chain head.
  - **NEW mmap'd shm region** (`alloc_pages` + `remap_pfn_range`; expose via new `/dev/cipher_audit` character device OR a `file_operations.mmap` hook on the existing `/dev/cipher`). Documented in step's design memo.
  - Append-only ring buffer per tenant. Sizing: 4096 entries × 64 B = 256 KiB per tenant × 128 tenants = 32 MiB total kmod-resident DRAM (well within budget).
  - HMAC accumulator added to `struct cipher_pid_stats` (~64 B addition). **Forces struct size growth 336 → 400 B** (the R-W7.4 ABI break) — Step 1 already consumed 16 B for `model_uuid` and Step 2 consumed 8 B for `commit_seq` (24 B used), leaving 32 B in the original `reserved[14]`. 64 B more = total 88 B addition; the `reserved[14]` tail cannot absorb this without growing the struct. **ABI bump 0.5.5 → 0.6.0** is the explicit signal.
  - Per-tenant chain head seeded with FNV-prefixed HMAC at REGISTER_TENANT (existing ioctl NR 1, the seeding lands in the existing `cipher_dev.c:85-93` handler tail).
  - Hot-path write rate at N=128 × ~80 tok/s × ~8 actuator calls/token = ~82 000 writes/s aggregate. Each write designed as **sub-µs RING_WRITE-pattern producer** (per §4.9 spec: relaxed-load + memcpy + release-store ≈ 10 ns on x86 TSO). **Consumers wait for W10-12.**
- `cipher_rt_phase4/cipher_rt_commit.c` (modify): COMMIT's step 1 (AUDIT chain) writes one entry into the kmod-resident chain head and advances the HMAC accumulator under release-fence. The userspace half from Step 2 redirects to the kmod-resident half when `CIPHER_AUDIT_CHAIN_KMOD=1` (default 1 after Step 3 lands; 0 for legacy rollback).
- `cipher_kmod/cipher_ioctl.h:179-190` (cipher_tenant_snapshot_user mirror): size invariant updated 336 → 400 B. Userspace consumers re-validate at `size_t` load time.
- `cipher_kmod/cipher_internal.h:245-311`: struct layout versioned (existing ABI versioning pattern at L32; bump the layout version field).
- `cipher_kmod/cipher_main.c:141`: `MODULE_VERSION("0.5.5") → MODULE_VERSION("0.6.0")` — userspace ABI break in `cipher_tenant_snapshot_user`.
- `cipher_kmod/cipher_proc.c:155`: banner 0.5.5 → 0.6.0.

### 5.2 Why third

COMMIT contract from Step 2 is stable; G6 plugs into COMMIT's step-1 AUDIT slot naturally. Step 4 (overlay-op port) consumes the merged COMMIT+G6 snapshot.

### 5.3 Behavioral gate

- Per-tenant HMAC chain head advances on every COMMIT (`cipher_get_commit_seq()` and chain-head pointer advance in lock-step; verified via contention harness).
- Chain is **externally verifiable**: offline verifier in `cipher_kmod/tools/verify_audit_chain.c` (~80 LOC) recomputes the HMAC sequence from the kmod-published ring buffer; result matches the in-flight chain head value at any sample point.
- Chain head **survives** REGISTER_TENANT close+reopen (kmod-resident, not userspace-only — this is the key R-G6.1 mitigation).
- 24-hour soak: chain head never wraps under N=128 × 8000 tok/s aggregate; ring overflow handled gracefully (drop counter advances, no producer block).

### 5.4 Regression gate

- Step 1 + Step 2 regression gates still PASS.
- `cipher_tenant_snapshot_user` size invariant verified by new test `T-W7.4` (size_t = 400 B post-Step-3, was 336 B post-Step-2).
- `libcipher_rt.so` loader smoke still PASS against new kmod 0.6.0.
- vLLM Mistral-7B B=1 decode runs clean with the new chain-head writes (forward-pass cost unchanged within budget).

### 5.5 Rollback path

`CIPHER_AUDIT_CHAIN_KMOD=0` → COMMIT's step 1 writes to userspace chain head (Step-2 behaviour). `cipher_audit_chain.c` remains compiled but the ring buffer is unused. `cipher_kmod.ko.pre_week8_step3` preserved. Anchor rolls back to `week-7-step-2-commit-primitive`.

### 5.6 Per-actuator file:line table

| File | Lines | Change |
|------|-------|--------|
| `cipher_kmod/cipher_audit_chain.c` | new ~400 LOC | per-tenant HMAC chain + ring buffer storage |
| `cipher_kmod/cipher_audit_mmap.c` | new ~150 LOC | alloc_pages + remap_pfn_range + file_operations.mmap |
| `cipher_kmod/cipher_dev.c:85-93` | extend | REGISTER_TENANT handler tail seeds chain head |
| `cipher_kmod/cipher_internal.h:245-311` | `cipher_pid_stats` | add HMAC accumulator (~64 B); bump layout version |
| `cipher_kmod/cipher_ioctl.h:179-190` | `cipher_tenant_snapshot_user` | size 336 → 400 B (userspace ABI break) |
| `cipher_kmod/cipher_main.c:141` | replace | MODULE_VERSION 0.5.5 → 0.6.0 |
| `cipher_kmod/cipher_proc.c:155` | replace | banner 0.5.5 → 0.6.0 |
| `cipher_rt_phase4/cipher_rt_commit.c` | step-1 wire | redirect AUDIT step to kmod chain via env gate |
| `cipher_kmod/tools/verify_audit_chain.c` | new ~80 LOC | offline HMAC sequence verifier |

### 5.7 Eng-days breakdown

| Sub-task | Days |
|---|---|
| `audit_chain.c` HMAC accumulator + ring | 1.0 |
| `audit_mmap.c` fresh mmap infrastructure (~150 LOC) | 0.5 |
| `cipher_pid_stats` HMAC accumulator field + struct size grow | 0.25 |
| COMMIT integration in `cipher_rt_commit.c` | 0.5 |
| `verify_audit_chain.c` + 24-hour soak | 0.5 |
| Size invariant T-W7.4 test | 0.25 |
| **Total** | **3.0** (high end of audit's 2.5–3 range, sharpened for the fresh mmap infra) |

---

## 6. Step 4 — Overlay-op `_report()` port to snapshot reads

**Tag at close:** `week-8-step-4-overlay-ops-port` (on `cipher_rt_phase4`).

### 6.1 Scope (~2 eng-days, ~125–175 LOC)

**Count discrepancy surfaced**: Agent B found **18 standalone `_report()` functions + 2 sub-dispatchers + 3 infra = 23**, not the spec's "21". Documenting the actual surface here for the Step-4 implementation prompt.

| Op | File:Line | LOC | Counters read | Mechanical? |
|----|-----------|-----|----------------|-------------|
| **SENSE** | `cipher_sense.cpp:296` | 44 | 5 | ✓ |
| **ORACLE** | `cipher_oracle.cpp:429` | 34 | 14 (already snapshot-param) | ✓ |
| **ORACLE_BILLING** | `cipher_oracle.cpp:509` (sub-disp) | 20 | 6 | ✓ |
| **CARBON** | `cipher_carbon.cpp:113` | 56 | 8 | ✓ |
| **FAIRNESS** | `cipher_fairness.cpp:131` | 54 | 7 | ✓ |
| **GUARD** | `cipher_guard.cpp:151` | 57 | 7 | ✓ |
| **DETERMINISM** | `cipher_determinism.cpp:73` | 15 | 2 | **NON-MECHANICAL** — XOR'd fingerprint accumulator; needs COMMIT-time snapshot pin to guarantee determinism under concurrent dispatch |
| **TOPOLOGY** | `cipher_topology.cpp:66` | 30 | 3 | ✓ |
| **COMPLY** | `cipher_comply.cpp:46` | 37 | 9 (calls 9 sub-funcs) | ✓ |
| **LOOP** | `cipher_loop.cpp:250` | 37 | 6 | ✓ |
| **PIPELINE** | `cipher_pipeline.cpp:151` | 100+ | 10+ (pairwise Jaccard) | ✓ |
| **PULSE** | `cipher_pulse.cpp:393` | 23 | 6 (ECC + dispatch) | ✓ |
| **RECEIPT** | `cipher_receipt.cpp:165` | 60 | 8 (HMAC chain accumulator) | **NON-MECHANICAL** — HMAC chain composes with G6 from Step 3; refactor handled in Step 3, not Step 4 |
| **CONTINUITY** | `cipher_continuity.cpp:185` | 52 | 9 | ✓ |
| **TRACE** | `cipher_trace.cpp:77` | 40 | 4 | ✓ |
| **TELEMETRY** | `cipher_telemetry.cpp:379` | 32 | 12 (already snapshot-param) | ✓ |
| **RECIPES** | `cipher_recipes.cpp:510` | 13 | 3 | ✓ |
| **STRUCTURAL_LOOKUP** | `cipher_structural_lookup.cpp:281` | 20 | 5 | ✓ |
| **RUNTIME** | `cipher_runtime.cpp:107` | 39 | 9 (aggregator) | ✓ |
| **LAYER3 (dispatch)** | `cipher_dispatch.cpp:525` (sub-disp) | 6 | 0 (pure aggregator) | ✓ |
| **L2_PERSIST** | `cipher_l2_persist.h:95` | TBD | TBD | ? **Confirm in impl prompt** — header decl only; implementation may be inline |

**Sub-totals:** 18 standalone + 2 sub-dispatchers + 1 infra-TBD = 21 surface entries to port. (The audit's "21" framing was numerically correct in count but conflated standalone + sub-dispatcher; documented here.)

- For each mechanical-port op: replace direct counter access with snapshot field access via `cipher_get_current_tenant_snapshot()`. Each op typically reads 3–8 counters → ~5 LOC per op. Total mechanical port LOC ~95.
- **DETERMINISM refactor** (~20 LOC): snapshot must pin XOR'd dispatch fingerprint at COMMIT-end; report reads the pinned value. Add a `dispatch_fp_xor` field to the snapshot.
- **RECEIPT refactor** (~20 LOC): the HMAC chain is the G6 chain from Step 3; RECEIPT's `_report()` reads the chain head and finalizes the per-session HMAC. This is the natural place for the RECEIPT/G6 seam. The G6 fold-in is already in Step 3; Step 4 wires RECEIPT to read the kmod-resident head.

### 6.2 Why fourth

COMMIT + G6 + REGISTER_MODEL all stable. Overlay ops port cleanly because the snapshot they read is well-defined (post-COMMIT, post-G6-chain-update). DETERMINISM/RECEIPT refactor lands cleanly because the substrate guarantees they need (snapshot pin + kmod chain) exist now.

### 6.3 Behavioral gate

- All 21 surface entries' `_report()` values **byte-identical** to pre-port output at single-tenant (regression check: per-op snapshot of `_report()` output before/after the port).
- All 21 surface entries' `_report()` values **consistent across N=100 concurrent observers** (atomic snapshot read = no torn reads). Contention harness asserts.
- DETERMINISM: dispatch_fp_xor at any post-COMMIT sample is deterministic across concurrent dispatch (verified by 5 runs of the same workload at N=15 producing identical fingerprint).
- RECEIPT: per-session HMAC chain externally verifiable (composes with Step 3's `verify_audit_chain.c`).

### 6.4 Regression gate

- Steps 1–3 regression gates still PASS.
- Per-op behavioral test pass rate: 21/21 (or document any port that requires re-design).

### 6.5 Rollback path

`CIPHER_OVERLAY_REPORT_VIA_SNAPSHOT=0` → overlay ops fall back to direct counter access; per-op behaviour matches v1.2.1. The env gate is per-op via a bitmask if needed.

### 6.6 Eng-days breakdown

| Sub-task | Days |
|---|---|
| 18 mechanical ports (4–5 ops/day) | 1.0 |
| DETERMINISM refactor (~20 LOC) | 0.25 |
| RECEIPT refactor wiring to G6 chain (~20 LOC) | 0.25 |
| L2_PERSIST audit + decision (in-scope vs defer) | 0.25 |
| Per-op self-test validation + N=100 atomicity | 0.25 |
| **Total** | **2.0** |

---

## 7. Step 5 — N=128 contention soak

**Tag at close:** `week-9-step-5-n128-soak` (== **`week-9-complete`** on `cipher_rt_phase4` + `cipher_kmod`).

### 7.1 Scope (~1 eng-day, ~300 LOC test harness)

- New harness `cipher_test_commit_n128.c` (~300 LOC): N=128 concurrent tenants × 4 reader threads per tenant racing snapshot reads. Verify:
  - **Snapshot atomicity**: no torn reads at sequence-counter discipline (1000/1000 iterations).
  - **HMAC chain head atomicity**: G6 release-fence verified — concurrent COMMITs from 128 tenants each advance their own chain head without cross-contamination.
  - **REGISTER_MODEL idempotency under 128 concurrent calls** (5 distinct `hf_config_hash`es split across 128 tenants → 5 distinct `model_uuid`s, all matching).
  - **5-step COMMIT order**: instrument each step with a microsecond timestamp; assert ordering AUDIT < FAIRNESS < CARBON < RECEIPT < kmod-state in every COMMIT cycle.
- 24-hour soak at N=128. Verify:
  - Memory leak rate ≤ 1 MB/hr (within v1.2.2 envelope).
  - Thread starvation: per-tenant FAIRNESS ratio ≥ 0.95 across all 128 tenants.
  - No silent regressions in 21 overlay ops' `_report()` output (sampled hourly, compared to single-tenant golden).
  - `cipher_kmod.ko` remains loaded clean (no oops, no `KERN_WARNING` in dmesg).
  - Audit chain externally re-verifiable from the kmod-published ring at 24 h.

### 7.2 Why fifth

All four primitives (COMMIT + G6 + G10 + overlay ops) integrated. The soak verifies they compose under stress before W10-12 piles RING_WRITE + G3 + G4 + G5 on top. **The N=128 number is intentionally the post-G1 cap** — we soak at the new structural ceiling, not the v1.2.2 ceiling.

### 7.3 Behavioral gate

- N=128 atomicity test 1000/1000 PASS.
- 24-hour soak completes with all gates green.
- Per-tenant per-second telemetry recorded for post-soak analysis (parquet export via existing bench harness reporter).

### 7.4 Regression gate

- All four prior steps' regression gates PASS at N=128.
- W6 G1+G2 baseline (CP 5.4 isolation 15/15) still PASS.
- vLLM Mistral-7B B=1 single-tenant baseline (per Week-6 Option 1 redo: agg_tps ≈ 165 ± 0.2) unchanged within ±3%.

### 7.5 Rollback path

N=128 soak failure with a specific signature (snapshot torn read, HMAC chain corruption, FAIRNESS starvation) surfaces back to the failing Step. Rollback to that Step's tagged commit; do not advance to W10-12 until the soak passes.

### 7.6 Eng-days breakdown

| Sub-task | Days |
|---|---|
| N=128 contention harness | 0.5 |
| 24-hour soak run + analysis + parquet export | 0.5 |
| **Total** | **1.0** |

---

## 8. Cross-step composition + ordering rationale

### 8.1 Dependency DAG

```
Step 1 (G10) ──┐
               ├──> Step 3 (G6) ────┐
Step 2 (COMMIT)─┴──> Step 4 (ops)  ─┴──> Step 5 (N=128 soak)
                          │
                          └──> RECEIPT refactor wires to G6 chain (Step 3 substrate)
```

- Step 1 produces `model_uuid` that Step 3 (HMAC chain entries reference it) and Step 4 (snapshot includes it) consume.
- Step 2 produces the COMMIT contract that Step 3 (G6 plugs into AUDIT step 1) and Step 4 (overlay ops read via snapshot) consume.
- Step 3 produces the kmod-resident chain that Step 4's RECEIPT refactor reads from.
- Step 4 produces all-21-ops report consistency that Step 5 validates at N=128.
- Step 5 produces soak evidence that W10-12 starts from.

Re-ordering breaks dependencies.

### 8.2 Composition with W6 deliverables

- **G6 AUDIT chain** (Step 3) is designed as a **RING_WRITE producer** (per v1.2.3 §4.9 L939). W10-12 builds the AUDIT consumer thread. Step 3 design is forward-compatible: the ring entry format matches §4.9's `CipherRingEntry` (sequence, timestamp_ns, kernel_class, params_hash, …).
- **G10 REGISTER_MODEL** (Step 1) unblocks W10-12 **G3** (KV-dedup model-keying) + **G4** (Marlin tenant-scoped) + W13-14 **G12** (Koopman registry model-keying). The `model_uuid` is stable across the entire 17-week timeline by design (idempotent registry).
- **May-13 reconstruction Phase 2** (CARBON + FAIRNESS_SHM + RECEIPT) composes with G6 AUDIT chain. Phase 2 should land in the same window as Step 3 (mid-W7-9 to early-W10-12). Phase 1 (3–4 days) can land in W7 in parallel to Step 1 because it doesn't depend on the AUDIT chain.

### 8.3 Total calendar

| Step | Calendar | Eng-days |
|------|----------|----------|
| Step 1 (G10) | W7 first half | 2.0 |
| Step 2 (COMMIT) | W7 second half + W8 first half | 3.0 |
| Step 3 (G6) | W8 second half + W9 first half | 3.0 |
| Step 4 (overlay ops) | W9 first half (can parallelize with Step 3 if capacity) | 2.0 |
| Step 5 (N=128 soak) | W9 last days | 1.0 |
| **Total** | **W7–W9 (3 calendar weeks)** | **~11.0 eng-days** |

Parallel lane: **May-13 reconstruction Phase 1 (3–4 days)** runs in W7. **Phase 2 (5–7 days)** runs W8–W9. **Phase 3 (1 day)** runs in W9.

---

## 9. Risk register additions

### 9.1 Adopted from v1.2.3 §7 W7-9 (lines 1531–1536)

- **R-W7.1** [HIGH] Observer protocol change touches all 21 overlay ops; validation scales with op count. **Mitigation:** structured per-op port checklist; per-op self-test must pass before integration. **Sharpened:** the actual surface is 18 standalone + 2 sub-dispatchers + 1 infra-TBD = 21 entries; the 21 framing is numerically correct but conflated standalone + sub-dispatcher. Documented at §6.1.
- **R-W7.2** [HIGH] Atomicity at N=128 may surface contention invisible at N=15. **Mitigation:** explicit N=128 stress run before close (Step 5).
- **R-W7.3** [MEDIUM] Hidden invariants in per-observer state mutation. **Mitigation:** audit overlay ops in advance (already done by Agent B at §2.4 + §6.1 of this scope-lock).
- **R-W7.4** [MEDIUM] (v1.2.3 new) G6 fold-in needs ~64 B addition to `cipher_pid_stats`; the W1 reserved-tail bump already used most of the headroom. **Mitigation:** Step 1 + Step 2 fit in `reserved[14] = 56 B` (model_uuid 16 B + commit_seq 8 B = 24 B); Step 3's 64 B HMAC accumulator forces struct size grow to 400 B (ABI bump 0.5.5 → 0.6.0). Userspace mirror `cipher_tenant_snapshot_user` re-validates at size_t load time via new T-W7.4 test.
- **R-W7.5** [MEDIUM] (v1.2.3 new) REGISTER_MODEL race with REGISTER_TENANT. **Mitigation:** sentinel `model_uuid = MODEL_UNKNOWN` until REGISTER_MODEL lands; downstream actuators treat MODEL_UNKNOWN as pass-through.

### 9.2 New W7-9 risks surfaced in this scope-lock

- **R-W7.6** [MEDIUM] (NEW): existing `cipher_tenant_snapshot.c` is **per-CPU TLS + ioctl `copy_to_user`**, NOT a mmap'd shm region. G6's per-tenant chain ring needs **fresh mmap infrastructure** (~150 LOC: alloc_pages + remap_pfn_range + file_operations.mmap, possibly via a new `/dev/cipher_audit` character device). **Mitigation:** Step 3 LOC budget sharpened from audit's 400 LOC to 550 LOC; design memo at Step 3 entry documents the chosen device path (extend `/dev/cipher` with mmap hook vs new `/dev/cipher_audit` device).
- **R-W7.7** [LOW] (NEW): **DETERMINISM `_report()`** at `cipher_determinism.cpp:73` XOR-accumulates dispatch fingerprints — concurrent dispatch can alias the hash. **Mitigation:** snapshot must pin `dispatch_fp_xor` at COMMIT-end; per-op self-test verifies 5 runs of the same workload at N=15 produce identical fingerprint. Refactor scoped in Step 4 (~20 LOC).
- **R-W7.8** [LOW] (NEW): **RECEIPT/G6 chain coupling** — RECEIPT's per-session HMAC chain *is* the G6 chain head. RECEIPT refactor in Step 4 (~20 LOC) reads the kmod-resident chain head and finalizes per-session HMAC. **Mitigation:** Step 3 closes the chain substrate; Step 4 wires RECEIPT to it; offline verifier (`verify_audit_chain.c`) covers both at the same time.

---

## 10. Implementation-prompt sequencing

Five implementation prompts follow this scope-lock:

| # | Prompt | Pre-condition tag | Close tag |
|---|--------|-------------------|-----------|
| 1 | Step 1 G10 ABI scaffolding | `week-6-step-g1-g2-cap-bump` (`c4e2d6f`) | `week-7-step-1-g10-abi-scaffold` |
| 2 | Step 2 COMMIT primitive core | `week-7-step-1-g10-abi-scaffold` | `week-7-step-2-commit-primitive` |
| 3 | Step 3 G6 kmod AUDIT chain | `week-7-step-2-commit-primitive` | `week-8-step-3-g6-audit-chain` |
| 4 | Step 4 21 overlay-ops port | `week-8-step-3-g6-audit-chain` | `week-8-step-4-overlay-ops-port` |
| 5 | Step 5 N=128 contention soak | `week-8-step-4-overlay-ops-port` | `week-9-step-5-n128-soak` = `week-9-complete` |

Each prompt is mechanical given this scope-lock. Each ships a tagged commit on `cipher_kmod` and/or `cipher_rt_phase4` + a deliverable doc on `cipher-fusion-evidence`. Each has hard-stop pre-conditions tied to the prior step's tag.

---

## 11. v1.2.3 §7 W6 closure (small follow-on edit folded into this commit)

v1.2.3 plan `CIPHER_REENGINEERING_PLAN.md` §7 W6 row updated from **"IN-FLIGHT + CARRY"** to **"DONE"** with the three closing commit refs:

- ✓ G1+G2 cap bumps — `cipher_kmod` `c4e2d6f` (tag `week-6-step-g1-g2-cap-bump`); `cipher-fusion-evidence` `751c6b8` (`WEEK_6_STEP_G1_G2_CAP_BUMP.md` md5 `ce8adb6a`)
- ✓ G5 path-a verification — `cipher-fusion-evidence` `35f9b6c` (`WEEK_6_G5_PATH_A_VERIFICATION.md` md5 `504d3007`)
- ✓ May-13 POC reconstruction plan — `cipher-fusion-evidence` `970694b` (`WEEK_6_MAY13_RECONSTRUCTION_PLAN.md` md5 `11e79cc8`)

**v1.2.3 §7 status post-scope-lock:**

- W1–W5: DONE (W1 `fc8a9ae6`; W3 `79c1b4f9`; W4 `850bd8b`; W5 `ec0e005` + `4302079`)
- W6: **DONE** (this scope-lock close-marker)
- **W7–W9: SCOPE-LOCKED** (this doc) — 5 implementation steps follow in separate prompts
- W10–W12: open (G3 + G4 + G5 + RING_WRITE substrate)
- W13–W14: open (Koopman tier + G12)
- W15–W17: open (CP 5.5 hybrid heterogeneous benchmark)

---

## 12. Commit + memory anchor

This doc lands as a `cipher-fusion-evidence` commit alongside the §7 W6 row close-marker edit. Memory anchor `w7-9-scope-lock` records the 5-step structure, tag-chain progression, and ~11 eng-day total.

After this scope-lock lands, the next prompt is **Step 1: G10 ABI scaffolding implementation** (cipher_kmod 0.5.0 → 0.5.5; ~2 eng-days; tag `week-7-step-1-g10-abi-scaffold`).
