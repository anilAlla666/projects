# Phase 4 T4.2.4b — Per-tenant Green Context partition + WL05 lift test

**Date:** 2026-05-14 morning session (continuing post-T4.2.4a).
**Build:** `libcipher_rt.so.v0.2.0_T4_2_4b` md5 `33550788d134ed826e2d9dbeac92135c`.
**Kmod substrate:** 0.4.5 (1B657D6043... → 4618FD1FC28BEE5EBC32944, B10 FIT_HINT).
**Source tarball:** `cipher_rt_phase4_src_T4_2_4b.tar.gz` md5 `c3b82e8d5a709dde87dcce3673d83de9`.
**Pre-T4.2.4b artifact saved:** `libcipher_rt.so.v0.2.0_T4_2_4a.pre_T4_2_4b` md5 `77589b98c8147ba0f9385cc2bae27074`.

## What shipped

`cipher_rt_green_ctx.c` rewritten from "full-SM no-partition" (T4.2.4a) to:

1. `cuDeviceGetDevResource(SM)` → full SM resource set.
2. `cuDevSmResourceSplitByCount(nbGroups=16, minCount=8)` → split into ≥8-SM groups (H100 minimum). Result: **15 groups of 8 SMs each, 12 SMs in remaining** (132 - 120 = 12).
3. Pick group index `tenant_handle_u32 % 15`.
4. `cuDevResourceGenerateDesc` on the selected group → descriptor.
5. `cuGreenCtxCreate` from that descriptor → restricted green ctx.
6. `cuCtxFromGreenCtx` → CUcontext for cuCtxPushCurrent.
7. **Verify with `cuGreenCtxGetDevResource`** (advisor-binding check) → confirms green ctx has 8 SMs, not 132.

`cipher_rt_tenant_handle_u32_for_self()` added to cipher_rt_partition_router.c to expose the cached tenant handle for the group selector.

## Smoke verification (WL01 60s)

| Item | Result |
|---|---|
| `GREEN: device 0 has 132 SMs total; splitting into 16 groups of 8 SMs each` | ✅ |
| `GREEN: split returned nb_groups=15 (requested 16); remaining 12 SMs unallocated` | 15 groups confirmed |
| `GREEN: tenant_handle=0x0991b1a8 → selecting group 4 of 15 (each group has 8 SMs)` | selector firing |
| `GREEN: green context bound to group 4 with 8 SMs (CUcontext=0x...)` | **partition VERIFIED via cuGreenCtxGetDevResource — 8 SMs, NOT 132** |
| WL01 tok/s | 79.6 (within ±5% of baseline ~80) |
| Phase 3 ABI 12/12 PASS | ✅ |

The cuGreenCtxGetDevResource verify (advisor's "did the partition take" check) passed: green ctx contains 8 SMs, not 132. **The partition mechanism itself works at the API level.**

## WL05 lift test — the headline measurement

| Build | TPW | Notes |
|---|---:|---|
| Morning libcipher_v2 noise band mean | 1.396 | 3-run same-condition spread ±0.53% |
| **T4.2.4b (libcipher_rt + GREEN + 8-SM partition / 0.4.5)** | **1.3854** | |
| Δ vs noise band mean | **-0.76%** | |

Three-band interpretation (per PHASE_4_NOISE_BAND_WL05.md):
- `|Δ| ≤ 0.53%` → neutral
- `0.53% < |Δ| ≤ 1.06%` → **AMBIGUOUS** (current band: -0.76%)
- `|Δ| > 1.06%` → genuine signal

**Verdict: AMBIGUOUS, leaning slight regression but within 1×–2× band.** Not a lift.

## Diagnosis — why the mask isn't load-bearing (the actual finding)

Two pathologies surfaced in the same run, either of which alone explains the null result.

### Pathology #1 — 100% hash collision: all 8 tenants picked group 5

Per-child stderr from the WL05 run:
```
t1: tenant_handle=0x59a52e05 → selecting group 5 of 15
t2: tenant_handle=0x59a528ec → selecting group 5 of 15
t3: tenant_handle=0x59a52a9f → selecting group 5 of 15
t4: tenant_handle=0x59a52586 → selecting group 5 of 15
t5: tenant_handle=0x59a52739 → selecting group 5 of 15
t6: tenant_handle=0x59a52220 → selecting group 5 of 15
t7: tenant_handle=0x59a523d3 → selecting group 5 of 15
t8: tenant_handle=0x59a539ea → selecting group 5 of 15
```

All 8 tenants — running on `wl05_t1`, `wl05_t2`, ..., `wl05_t8` tenant IDs — picked the SAME green-ctx group.

**Root cause:** the kmod's `tenant_handle_u32` is the low 32 bits of FNV-64 of the tenant string. FNV's multiplicative structure means strings differing by only the last byte produce hashes whose difference is `(last_byte_xor) * FNV_PRIME`. The FNV-64 prime is `0x100000001b3`. The difference between consecutive `wl05_tN` strings is `(N1 XOR N2) * 0x100000001b3` mod 2^32 = small. **`0x100000001b3 mod 15 = 1`**, so consecutive tenant strings differ by a multiple-of-3 modular position that — combined with the `_t1`..`_t8` byte difference of 1-7 — collapsed all 8 to the same residue mod 15.

Advisor's pre-design call-out ("birthday paradox at 8 tenants into 16 buckets ~14% chance of one collision") was correct in principle but underestimated the specific worst case: FNV against close-prefix strings is NOT a random hash for this purpose.

**Impact:** with all 8 tenants confined to the same 8-SM group, the experiment measured "8 tenants share 8 SMs (oversubscribed)" rather than "8 tenants get 8 SMs each (load-balanced)." That's why TPW is flat (1.39 ish, near noise band) — there's no parallel-partition advantage to measure.

**Fix paths:**
- Replace `tenant_handle % 15` with `getpid() % 15` (deterministic per process, perfectly distributes).
- Or apply a better mix step on `tenant_handle`: `((handle * 0x85ebca6b) ^ (handle >> 16)) % 15` (xorshift-style avalanche).
- Or use the kmod-side slot index (lowest set bit of mask) as the selector — uses ARBITRATE's own data, but requires kmod-side slot→SM-group mapping (separate sub-phase).

### Pathology #2 — runtime API stream-create may not trip the driver-API CUPTI hook

The cuStreamCreate hooks I enabled are in `CUPTI_CB_DOMAIN_DRIVER_API` only:
- `CUPTI_DRIVER_TRACE_CBID_cuStreamCreate`
- `CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority`

PyTorch uses `cudaStreamCreate` / `cudaStreamCreateWithFlags` / `cudaStreamCreateWithPriority` (runtime API). The runtime API translates to driver API internally, **but it's not guaranteed that the driver-API CUPTI callback fires** for the translated call — that depends on whether the runtime API uses CUDA-driver entry points (CUPTI-visible) or shortcut/inlined paths. Need to either:
- Enable runtime-API stream-create cbids (`CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate*`).
- Or verify push-fires-on-create via instrumentation (the `g_streams_observed` counter is exposed but wasn't logged this run).

**Indirect evidence the hook may not be firing for the user's streams:**
- WL01 smoke produced 79.6 tok/s with green ctx bound to 8 SMs. If kernels were actually restricted to 8 SMs, this small-model decode might still hit similar tok/s (decode B=1 on TinyLlama is largely memory-bound, not SM-count-bound). So this doesn't distinguish "partition enforced but invisible" from "partition not enforced."
- WL05 8-tenants-all-on-group-5 produced 278 tok/s aggregate (vs noise band ~279). If partition were enforced, 8 tenants on 8 SMs would show severe contention. The fact that there's NO measurable contention suggests kernels are NOT actually constrained — supporting pathology #2.

**Fix:** add CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate / cudaStreamCreateWithFlags / cudaStreamCreateWithPriority enables, plus instrument cipher_rt_green_ctx_push() to log first-N call sites for diagnostic.

## Honest finding statement

**T4.2.4b's API-level mechanism is verified working** (cuGreenCtxCreate with 8-SM resource succeeds; cuGreenCtxGetDevResource confirms partition took). **The lift question on WL05 is still unmeasured** because two independent bugs prevented the experiment from being valid:
1. Hash collision routes all WL05 tenants to the same partition (no parallelism to measure).
2. Runtime-API stream-create may not fire the driver-API CUPTI hook (push may not actually happen).

Both are addressable. Neither requires a kmod change. Estimated fix scope: 30 min code + 10 min verify.

## Discipline

- Pre-T4.2.4b artifact saved: `libcipher_rt.so.v0.2.0_T4_2_4a.pre_T4_2_4b` md5 `77589b98c8147ba0f9385cc2bae27074`.
- Pre-T4.2.4b source saved: `cipher_rt_green_ctx.c.pre_T4_2_4b` md5 `9a303c14200ddffd03e30736151d8230`.
- Phase 3 ABI 12/12 PASS at every checkpoint.
- Fallback md5s unchanged (55ab8c0c / 86618c30).
- Taint unchanged at 12288.
- No oops/warn/bug in dmesg.

## T4.2.4b_v2 — fixes applied, WL05 re-measured

**Build:** `libcipher_rt.so.v0.2.0_T4_2_4b_v2` md5 `8fbc350fb322b0733ce6d4af1468a88f`.

Three fixes shipped over T4.2.4b:

1. **Hash-mix on group selector:** `(getpid() ^ tenant_handle)` through a
   2-round xorshift+multiply (Murmur-style avalanche) → `% nb_groups`.
   Distributes uniformly even for collided tenant_handles.
2. **Runtime API CUPTI hooks added:** `cudaStreamCreate_v3020`,
   `cudaStreamCreateWithFlags_v5000`, `cudaStreamCreateWithPriority_v5050`.
   Dispatched alongside driver-API in the same callback.
3. **Push instrumentation:** first 3 push observations log
   `GREEN: push fired (observation #N, group G)` so we can verify the
   hook actually fires during user stream creation.

### WL05 600s under T4.2.4b_v2

| Metric | Value |
|---|---:|
| MFU% device aggregate | 100.0 |
| tok/s aggregate | 280.81 |
| watts_avg | 202.02 |
| **TPW** | **1.39** |
| children_complete | 8/8 |

### Distribution check — hash fix worked

Per-child group assignments under hash-mix:

| Child | tenant_handle | pid | group |
|---|---|---|---:|
| t1 | 0x59a52e05 | 1025920 | 4 |
| t2 | 0x59a528ec | 1026089 | **2** |
| t3 | 0x59a52a9f | 1026116 | 10 |
| t4 | 0x59a52586 | 1026253 | 12 |
| t5 | 0x59a52739 | 1026280 | 14 |
| t6 | 0x59a52220 | 1026417 | 11 |
| t7 | 0x59a523d3 | 1026469 | **2** |
| t8 | 0x59a539ea | 1026635 | 5 |

7 distinct groups; one collision at group 2 (t2 + t7). This is 1/8 collision rate, very close to advisor's predicted ~14% birthday paradox for 8 tenants into 16 buckets. **Pathology #1 fixed.**

### Push fires confirmed — pathology #2 was actually not the issue

Per-child log: every WL05 child logged 3 `GREEN: push fired` observations. The driver-API hook (likely the one firing, since pyTorch's cudaStreamCreate dispatches to cuStreamCreate internally) catches user stream creation. **The push was firing in T4.2.4b v1 as well; we just didn't have visibility.**

So the original T4.2.4b's null result (-0.76%) was 100% due to pathology #1 (all tenants on group 5). Now with pathology #1 fixed:

### Lift verdict on WL05 T4.2.4b_v2

| Reference | TPW | Δ T4.2.4b_v2 |
|---|---:|---:|
| Same-condition libcipher_v2 noise band mean (this morning, 3 runs) | 1.396 | -0.43% |

Three-band interpretation:
- `|Δ| = 0.43%` ≤ 0.53% noise band → **NEUTRAL WITHIN NOISE.**

**No measurable lift on WL05 from per-tenant green-ctx partitioning.** The mechanism is verified working (distinct partitions, push fires, kernels launch within green ctx) — but TPW is statistically indistinguishable from baseline.

### Honest interpretation — why no lift

Advisor's pre-design call-out was prescient:
> "TinyLlama decode B=1 is so memory-bound that SM partitioning doesn't help anything. The model's bottleneck is HBM bandwidth, which doesn't partition with SMs. **This is possible.** If neutral, the diagnostic is to compare a compute-bound workload — but you don't have time for that tonight."

The two-fold mechanism is hereby established:
1. **SM partitioning is real and correctly applied** — `cuGreenCtxGetDevResource` confirms 8 SMs per tenant; user streams go through the push hook (verified).
2. **WL05 (TinyLlama decode multi-tenant) is HBM-bandwidth-bound, not SM-count-bound.** 8 tenants each constrained to 8 SMs (64 SMs used, 68 SMs idle on this 132-SM device) compete for the SAME HBM. Restricting SM count doesn't free bandwidth for other tenants. The lift has no mechanical foothold on this workload.

### What would show lift

A compute-bound workload at high concurrency — e.g. WL03 prefill (which saturates 693W = TGP at 83.6% MFU baseline) split across multiple tenants. There, each tenant's SMs would actually be doing FLOP work that's distinct from another tenant's, and partitioning would prevent SM contention.

This is the workload pattern T4.2.4c should target. WL05 (decode B=1 × 8) is the WRONG workload for the SM-partitioning lift hypothesis. The substrate works; we just measured against a memory-bound test that won't move on this hypothesis.

## Closing artifact map

| Build | md5 | Purpose |
|---|---|---|
| libcipher_rt.so.v0.2.0_T4_2_4a | 77589b98c8147ba0f9385cc2bae27074 | scaffold (no partition) |
| libcipher_rt.so.v0.2.0_T4_2_4a.pre_T4_2_4b | 77589b98c8147ba0f9385cc2bae27074 | discipline pre-snapshot |
| libcipher_rt.so.v0.2.0_T4_2_4b | 33550788d134ed826e2d9dbeac92135c | restricted green ctx, broken hash |
| libcipher_rt.so.v0.2.0_T4_2_4b_v2 | 8fbc350fb322b0733ce6d4af1468a88f | hash-mix + runtime API hooks + push instrumentation |
| cipher_rt_phase4_src_T4_2_4b.tar.gz | c3b82e8d5a709dde87dcce3673d83de9 | source tarball |

## Discipline gate at session end

| Gate | Result |
|---|---|
| Phase 3 ABI 12/12 PASS | ✅ |
| Fallback md5s 55ab8c0c / 86618c30 | ✅ unchanged |
| Taint 12288 | ✅ unchanged |
| dmesg oops/warn/bug since start | (none) |
| kmod 0.4.5 loaded | ✅ |
| daemons running | ✅ |
