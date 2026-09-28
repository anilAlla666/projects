# Week 1 Pre-Flight Verification

**Date:** 2026-05-20
**Scope:** Read-only baseline verification before Week 1 integration begins. No code changes; the only filesystem mutations are the explicitly requested `make clean && make -j` rebuilds in `cipher_rt_phase4` and `cipher_kmod`, and the binary backups to `/tmp/preflight_backup/` taken before the cleans.
**Output:** This file.
**Pod:** Lambda H100 80GB SXM5, driver 580.105.08, kernel 6.8.0-1046-nvidia.
**Discipline rule applied:** drift and infrastructure gaps are surfaced verbatim with file paths and measured values. No mitigations are proposed inside this document (Step 5 LP-7 rename was explicitly requested by the brief and is included).

---

## Status table

| Check | Result | Notes |
|---|---|---|
| Step 1 — `libcipher_rt.so` anchor | PASS | md5 `83afd1ca4118dc651854ef751e4ef82d` = documented `83afd1ca` |
| Step 1 — `cipher_kmod.ko` anchor | PASS | md5 `008b3c66faa82c71c1615ddf9c87ec56` = brief's `008b3c66` (Track 2 SC5 close) |
| Step 1 — `cipher_kv_bridge` anchor | PASS | md5 `c04b0c39d8282daee66b3b865a1849a9` = documented `c04b0c39` |
| Step 1 — `libcipher_v2.so` anchor | DRIFT | Brief named path `/home/ubuntu/cipher_rt_phase4/libcipher_v2.so` does not exist. Two libcipher_v2 binaries exist at `/home/ubuntu/libcipher_v2/`: `libcipher_v2.so` md5 `cc0479b8...` (Track 2 close) and `libcipher_v2.so.v0.2.0` md5 `86618c30...` (plan Appendix A). |
| Step 1 — kmod anchor vs plan Appendix A | DRIFT | Brief cites `008b3c66` (Track 2 SC5 close); plan Appendix A cites `285d102e` (Track 3 SC5 close). On-disk md5 matches the brief, not the plan. Plan Appendix A is stale on this row. |
| Step 1 — DKMS-installed kmod | DRIFT | `/lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko` md5 `6654d9e5...`, size 122,987 B vs in-tree 2,631,632 B (21× smaller; mtime 2026-05-16, predates Track 2 SC5). DKMS image is stale relative to in-tree. |
| Step 2 — `cipher_rt_phase4` clean rebuild | PASS | Clean rebuild reproduces md5 `83afd1ca4118dc651854ef751e4ef82d` byte-identical to anchor. Build returned `rc=0` with warnings only (unused-parameter in upstream `MTIAHooksInterface.h`, one unused static `env_on` in `cipher_rt_attn_dispatch.cpp`). |
| Step 2 — `cipher_kmod` clean rebuild | PASS | Clean rebuild reproduces md5 `008b3c66faa82c71c1615ddf9c87ec56` byte-identical to anchor. Build returned `rc=0`. Kernel BTF generation skipped (no `vmlinux` available, expected on this pod). Compiler patch-version differs from kernel build (11.4.0-1ubuntu1~22.04.3 vs ~22.04.2, harmless). |
| Step 3 — CP 5.4 isolation (15/15) | PASS | All 15 sub-tests PASS; output byte-identical to last-recorded `isolation_phase_a.log`. |
| Step 3 — Track 3 SC3 DSM substrate | PASS (substrate live) | `/proc/cipher/migrations` responsive: counters cleanly zeroed at idle; last-recorded `sc3_e2e_B.log` shows pass=True with kl_max in 1.04e-05 to 5.50e-05 range. Full e2e re-run not executed in pre-flight window. |
| Step 3 — Track 2 SC6 (Mistral-7B N=4 bit-identical) | NOT RE-RUN | Test infrastructure intact (sc6_run.py + sc6_models.py + sc6_independent_tenant.py + sc6_consumer.py). Last-recorded independent baseline `sc6_Mistral-7B_independent_result.json` reports 73,332 MiB loaded across 5 tenants, PASS=true. Full re-run not executed in pre-flight window. |
| Step 3 — vLLM Phase 3 graph-mode parity | NOT RE-RUN | Test infrastructure intact (`future_scope_a/phase3_vllm_probe.py`, `phase3_sweep.py`, `phase3_5_sweep.py`). Last-recorded `phase3_5_sweep_result.json` confirms 1200 MHz sweet spot delivers +13.94% tok/W lift (5.139 vs 4.510 at 1980 MHz), matches the executive-summary claim. Full re-run not executed in pre-flight window. |
| Step 3 — Phase 5 baseline 15-tenant FAIRNESS | INFRASTRUCTURE NAMING MISMATCH | No single runner literally named or scoped as "Phase 5 baseline 15-tenant FAIRNESS run" exists in the repository. Closest infrastructure: `cipher_measurement/density_pack.sh` (2 / 5 / 10 / 20 / 30 / 50 / 100 tenant sweep), `cp_5_4/step1_6/cp54_s16_*sweep.py` (16-slot sweep at 2×16-SM partitions), `cipher-may13-evidence/tests/test_fairness.py` (single-process FAIRNESS unit). Surfaced for user adjudication. |
| Step 4 — Git state | DRIFT | None of `cipher_rt_phase4`, `cipher_kmod`, `cipher-may13-evidence`, or `cipher-fusion-evidence` are git repositories. There is no branch, no commit history, no `.git` directory to query. Rollback discipline relies on tarballs + binary anchors + md5 manifests, not git. |
| Step 5 — Week 1 source files exist | PASS | All 10 brief-listed files present at expected paths; top-level `cipher_dispatch.cpp` is 543 LOC matching the v1.2 plan. `src/` shadow copies confirmed present, silently-excluded by Makefile L29 filter-out, and structurally divergent from top-level (`diff -q` differs for both files). |
| Step 5 — LP-7 struct collision | CONFIRMED | `struct CipherKernelEntry` defined in two may13 headers (`cipher_kernel_table.h:53` and `cipher_param_recovery.h:30`) with structurally incompatible fields. 18 total references across 4 may13 files. Zero references in `cipher_rt_phase4` or `cipher_kmod`. The collision is entirely within may13. |
| Step 5 — Rename proposal | SURFACED | Brief proposes `cipher_may13_kernel_entry_t` for "the may13 version, leaving the rt_phase4 version unchanged." Plan §7 Week 1 proposes a different rename (`CipherKtEntry` for kernel_table, `CipherParamEntry` for param_recovery). Since rt_phase4 has zero `CipherKernelEntry` references, "leaving the rt_phase4 version unchanged" is a null operation; the real decision is which of two may13 definitions takes which name. |

---

## SECTION 1 — Anchor checkpoint

### 1.1 — Deployed binaries

All paths checked with `md5sum`, `stat -c %s`, and `stat -c %y`.

| Path | Full md5 | 8-char prefix | Size (bytes) | Mtime | Documented anchor | Result |
|---|---|---|---|---|---|---|
| `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` | `83afd1ca4118dc651854ef751e4ef82d` | `83afd1ca` | 150,408 | 2026-05-19 08:01:11 | `83afd1ca` (brief, plan App A) | **PASS** |
| `/home/ubuntu/cipher_rt_phase4/libcipher_v2.so` | n/a | n/a | n/a | n/a | path documented in brief | **MISSING** at brief's path |
| `/home/ubuntu/libcipher_v2/libcipher_v2.so` | `cc0479b836e560619e2b286ca1caecb7` | `cc0479b8` | 16,968 | 2026-05-13 10:05:23 | `cc0479b8` (brief) | PASS at alternate path |
| `/home/ubuntu/libcipher_v2/libcipher_v2.so.v0.2.0` | `86618c30896470b642fcc6985d8dc632` | `86618c30` | 16,496 | 2026-05-13 07:31:14 | `86618c30` (plan App A) | PASS at alternate path |
| `/home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` | `c04b0c39d8282daee66b3b865a1849a9` | `c04b0c39` | 373,752 | 2026-05-19 10:19:23 | `c04b0c39` (brief, plan App A) | **PASS** |
| `/home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.so` | symlink to above | n/a | 48 (link target name) | 2026-05-19 10:19:23 | n/a | symlink |
| `/home/ubuntu/cipher_kmod/cipher_kmod.ko` | `008b3c66faa82c71c1615ddf9c87ec56` | `008b3c66` | 2,631,632 | 2026-05-19 11:29:53 | `008b3c66` (brief, Track 2 SC5 close) / `285d102e` (plan App A, Track 3 SC5 close) | **PASS** vs brief; **DRIFT** vs plan App A |
| `/lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko` | `6654d9e5fcbe5f605e7ded216c20c45f` | `6654d9e5` | 122,987 | 2026-05-16 15:30:09 | `6654d9e5` (plan App A) | PASS vs plan App A; **DRIFT vs in-tree** (size, mtime predates Track 2 SC5) |

### 1.2 — libcipher_v2 path drift, surfaced

The brief lists `/home/ubuntu/cipher_rt_phase4/libcipher_v2.so` as a deployed binary. That file does not exist. Two `libcipher_v2` binaries exist at `/home/ubuntu/libcipher_v2/`, both small (16 KB scale), both predating Track 2 (mtime 2026-05-13). The brief's anchor `cc0479b8` matches the `.so` file at the alternate path; the plan Appendix A anchor `86618c30` matches `.so.v0.2.0` at the same alternate path. Both are present; neither is at the path the brief states. Surface for user adjudication: was the brief's path expected to be a symlink not yet created, or is the Phase 3 substrate located at `/home/ubuntu/libcipher_v2/` by design?

### 1.3 — kmod anchor split between brief and plan Appendix A

Brief cites kmod anchor `008b3c66` (post-Track 2 SC5).
Plan Appendix A cites kmod anchor `285d102e` (post-Track 3 SC5).
Auto-memory `[CIPHER Track 2 weight-sharing closeout]` cites `008b3c66`.
Auto-memory `[CIPHER Track 3 DSM]` cites `285d102e`.

The on-disk md5 is `008b3c66`. The mtime is 2026-05-19 11:29:53. Whichever is the canonical post-Week-5 anchor needs to be reconciled before Week 1 ships, because the v1.2 plan and the deployed module disagree by one full integration track. Surface for user adjudication.

### 1.4 — DKMS image is stale

The DKMS-installed kmod at `/lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko` is 122,987 bytes, 21× smaller than the in-tree 2,631,632-byte build, with mtime 2026-05-16 15:30:09 (predates Track 2 SC5 / Track 3 entirely). md5 `6654d9e5...` matches the plan Appendix A value for the DKMS row, so the documented anchor is consistent with what is installed; the drift is between in-tree (008b3c66) and DKMS-installed (6654d9e5). The currently-loaded module (from `lsmod`, size 1,163,264) is the in-tree build, not the DKMS image. Surface for adjudication: should DKMS be re-installed before Week 1?

### 1.5 — Source-file manifest

Both source trees were md5'd in full. Counts and full output below.

**`cipher_rt_phase4/`** — 37 files (Makefile + 33 .c/.cpp + headers):

```
4191557f 6571b 136L Makefile
6ff37415 12749b 298L cipher_cupti.c
71b4607e 3249b 76L cipher_inject.c
2ac90848 11013b 279L cipher_kv_bridge.cpp
88593d4c 12743b 334L cipher_rt_arbitrate.c           [RETIRED; preserved unbuilt per plan §1.2]
8da923f2 1774b 50L cipher_rt_arbitrate.h
94d08924 18326b 421L cipher_rt_attn_dispatch.cpp
81be0599 5637b 138L cipher_rt_attn_dispatch.h
1ed8ff79 3286b 99L cipher_rt_attn_test_actuator.c
7687cb1e 6789b 222L cipher_rt_audit.c
0df10592 2902b 81L cipher_rt_audit.h
0b79a5e1 5021b 145L cipher_rt_cublas_shim.c
0c4251b7 7619b 232L cipher_rt_got_patch.c
ff87a54a 2016b 51L cipher_rt_got_patch.h
7d05e568 21699b 587L cipher_rt_green_ctx.c
a90352b0 4290b 91L cipher_rt_green_ctx.h
0c0efefd 27637b 907L cipher_rt_kv_alloc.c
11e3f246 9334b 201L cipher_rt_kv_alloc.h
d0f264b1 1310b 38L cipher_rt_marlin.h
9d677ffd 8366b 217L cipher_rt_marlin_actuator.c
bf144732 45680b 1078L cipher_rt_marlin_engine.cpp
48d1b5dd 33960b 773L cipher_rt_marlin_kernel_src.cpp
ab493f44 456b 19L cipher_rt_marlin_kernel_src.h
7b467cd8 6068b 81L cipher_rt_marlin_perms.h
caea720a 3867b 123L cipher_rt_matmul_dispatch.c
93319a25 4627b 129L cipher_rt_matmul_dispatch.h
ac199360 10866b 289L cipher_rt_partition_router.c
12dc3dcd 2005b 49L cipher_rt_partition_router.h
ff77399e 3266b 117L cipher_rt_sm_packer.c
e6aad839 1584b 47L cipher_rt_sm_packer.h
5899d58f 7561b 245L cipher_rt_tenant.cpp
b1d0aded 4465b 137L cipher_rt_tenant.h
ac3d513c 12402b 380L cipher_rt_volt.c
7bbd071b 1715b 48L cipher_rt_volt.h
9718f375 1636b 58L cipher_tenant.c
3412bf2c 1622b 46L cipher_v2_internal.h
```

**`cipher_kmod/`** — 21 files (Kbuild + Makefile + 14 .c + headers + 1 microbench):

```
5b45e808 766b 22L Kbuild
c5e52fe7 338b 16L Makefile
28fa5f60 6142b 174L cipher_bar0.c
37a33c16 5100b 151L cipher_clock.c
dcdfa292 32783b 894L cipher_cp54_sched.c
d29f0a8f 12429b 403L cipher_dev.c
62dbefc0 10997b 350L cipher_flops.c
3f78cada 15217b 362L cipher_internal.h
24b87d95 20060b 513L cipher_ioctl.h
d76474fe 5029b 139L cipher_ioctl_decode.c
00b21f14 5355b 163L cipher_kmod.mod.c                [generated by kbuild, regenerated each build]
8512c82f 12541b 477L cipher_kvdedup.c
ac59cd7e 3102b 86L cipher_kvdedup.h
45cf1e97 3781b 128L cipher_main.c
1aff5445 15511b 482L cipher_partition_allocator.c
ce853aff 8086b 284L cipher_probe.c
bfa94f31 18633b 588L cipher_proc.c
56085484 7063b 213L cipher_state_updater.c
6c1f9b1b 7707b 240L cipher_tenant_snapshot.c
b4feb871 11720b 389L cipher_weight_arena.c
cdf56214 2144b 83L probe_microbench.c               [debug helper, not in Kbuild OBJS]
```

The 14 kbuild-listed sources in plan §1.3 match the on-disk count when excluding `cipher_kmod.mod.c` (kbuild-generated) and `probe_microbench.c` (out-of-tree debug helper). No additions, no removals.

---

## SECTION 2 — Build verification

### 2.1 — `cipher_rt_phase4` clean build

```
cd /home/ubuntu/cipher_rt_phase4
make clean
make -j
```

Full build log: `/tmp/preflight_rt_build.log`.

**Result: PASS.** Build returned `rc=0`. Final link line:

```
g++ -shared -fPIC -o libcipher_rt.so \
  cipher_inject.o cipher_tenant.o cipher_cupti.o \
  cipher_rt_partition_router.o cipher_rt_tenant.o cipher_rt_sm_packer.o \
  cipher_rt_green_ctx.o cipher_rt_volt.o cipher_rt_matmul_dispatch.o \
  cipher_rt_cublas_shim.o cipher_rt_got_patch.o \
  cipher_rt_marlin_kernel_src.o cipher_rt_marlin_engine.o cipher_rt_marlin_actuator.o \
  cipher_rt_attn_dispatch.o cipher_rt_attn_test_actuator.o \
  cipher_rt_audit.o \
  -lpthread -lcupti -lcuda -ldl \
  -Wl,-rpath,/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib \
  -L/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib \
  -lc10 -lcrypto
```

17 object files linked. The retired `cipher_rt_arbitrate.c` is not in the link list, consistent with plan §1.2 ("preserved for historical reference").

**Md5 after rebuild:** `83afd1ca4118dc651854ef751e4ef82d` — byte-identical to the pre-clean backup at `/tmp/preflight_backup/libcipher_rt.so.preclean`. The deployed runtime is reproducible from source.

**Warnings (non-fatal):** unused-parameter warnings in `torch/include/ATen/detail/MTIAHooksInterface.h` (upstream torch headers, not our code); one `static int env_on(const char* name)` declared but unused in `cipher_rt_attn_dispatch.cpp:106`.

### 2.2 — `cipher_kmod` clean build

```
cd /home/ubuntu/cipher_kmod
make clean
make -j
```

Full build log: `/tmp/preflight_kmod_build.log`.

**Result: PASS.** Build returned `rc=0`. Kbuild compiled all 14 listed sources, linked `cipher_kmod.o`, ran `MODPOST`, generated `cipher_kmod.mod.c`, linked the final `cipher_kmod.ko`, attempted BTF generation (skipped: no `vmlinux` available on this pod, which is expected for non-distro kernel builds and does not affect functional correctness).

**Md5 after rebuild:** `008b3c66faa82c71c1615ddf9c87ec56` — byte-identical to the pre-clean backup at `/tmp/preflight_backup/cipher_kmod.ko.preclean`. The deployed kernel module is reproducible from source.

**Compiler version drift (non-blocking):**
- Kernel built by `gcc-11 (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0`.
- This pod has `gcc-11 (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`.
- The Linux module loader emits a warning at modpost, accepts the module, and the resulting `.ko` is byte-identical anyway. No action required.

---

## SECTION 3 — Smoke test the existing measurements

For each of the five named tests, the brief asks: pass / fail / skip + measured value + drift + anomalies.

### 3.1 — CP 5.4 isolation (15/15 disjointness)

**Status: PASS, re-run in pre-flight, no drift.**

Binary: `/home/ubuntu/cipher-fusion-evidence/cp_5_4/step1_3/cp54_isolation_test` (16,848 bytes, ELF x86-64, BuildID `8f395f7e...`, executable). Requires `/dev/cipher` to be open (kmod must be loaded; verified live via `lsmod | grep cipher` returning `cipher_kmod 1163264 0`).

Six sub-tests, 15 sub-assertions:

| Test | Sub-assertion | Result |
|---|---|---|
| Test 1 — legacy nr-9 deactivated | `nr-9 REQUEST_SM_PARTITION returns -ENOSYS` | PASS |
| Test 2 — ALLOCATE/FREE/QUERY | ALLOCATE PARTITION 16 SMs → 2 groups | PASS |
| | QUERY reflects allocation | PASS |
| | FREE returns 0 | PASS |
| | QUERY after FREE shows 15 free | PASS |
| Test 3 — pool resize | POOL holder claims 15 residual | PASS |
| | PARTITION 16 SMs → 2 groups (pool shrunk) | PASS |
| | QUERY: pool 13, 1 partition tenant, mask matches | PASS |
| | After release: all 15 free | PASS |
| Test 4 — do_exit reaper | Child exited without FREE → reaper reclaimed 3 groups | PASS |
| Test 5 — disjointness | Two PARTITION tenants each get 2 groups | PASS |
| | Their group masks are DISJOINT | PASS |
| | After release: all 15 free | PASS |
| Test 6 — concurrent stress | 4 tenants × 5 ALLOCATE/FREE concurrent — no crash | PASS |
| | Ledger consistent after stress | PASS |

**Drift versus last-recorded `isolation_phase_a.log` (mtime 2026-05-18 18:56):** output is byte-identical line-for-line.

### 3.2 — Track 3 SC3 DSM migration (~1.26 ms target)

**Status: SUBSTRATE LIVE, full e2e not re-run in pre-flight window.**

**Substrate health check:** `cat /proc/cipher/migrations` returns:

```
policy:   gap_min_grps=1 sustain_ms=2000 ratelimit_ms=10000 verbose=1
gap:      stranded_groups=0 gap_age_ms=0
counters: proposals=0 commits=0
          aborts: timeout=0 tenant_nack=0 kmod_refused=0
```

The migration FSM is reachable from userspace, policy parameters are intact, counters are zero at idle as expected.

**Last-recorded SC3 e2e value (`sc3_e2e_B.log`, mtime 2026-05-19 ≤ 09:00):**

```
[cipher_v2] GREEN/MIGRATE: committed — now on mask 0x1800 (16 SMs, lowest group 11)
E2E B round=0 kl_max=4.52e-05 handler=migrate cur=0x1800
E2E B round=1 kl_max=3.04e-05 handler=none cur=0x1800
E2E B round=2 kl_max=5.50e-05 handler=none cur=0x1800
E2E B round=3 kl_max=1.04e-05 handler=none cur=0x1800
E2E B round=4 kl_max=3.12e-05 handler=none cur=0x1800
E2E B round=5 kl_max=4.52e-05 handler=none cur=0x1800
E2E B round=6 kl_max=3.04e-05 handler=none cur=0x1800
E2E B round=7 kl_max=5.50e-05 handler=none cur=0x1800
SC3_TENANT e2e pass=True
```

KL gates between 1.04e-05 and 5.50e-05 (within the documented teacher-forced gate threshold of 2.5e-05 to 5.5e-05 for the test). Migration committed (mask shift 0x0003 → 0x1800). The closeout's headline value of ~1.26 ms per migration is documented in `phase_c/track_3/TRACK_3_CLOSEOUT.md:11-17`; it is not re-measured in this pre-flight (full e2e requires TinyLlama load + a 17-migration sweep, out of the pre-flight window).

**Anomaly: none.**

### 3.3 — Track 2 SC6 bit-identical forward, Mistral-7B N=4

**Status: NOT RE-RUN, infrastructure intact, last-recorded result durable.**

**Infrastructure present:** `phase_c/sc6_run.py`, `sc6_models.py`, `sc6_consumer.py`, `sc6_independent_tenant.py`, `sc6_aggregate.py`. Last-recorded independent baseline at `phase_c/sc6_Mistral-7B_independent_result.json`:

```json
{
  "phase": "independent",
  "model": "Mistral-7B",
  "n_fit": 5,
  "oom": false,
  "fb_mib": {"idle": 0, "loaded": 73332, "exited": 0},
  "per_tenant_mib": 14666.4,
  "fb_loaded_5tenant_projected": 73332,
  "PASS": true
}
```

Five-tenant independent baseline measures 73,332 MiB loaded; the shared-arena measurement at `TRACK_2_CLOSEOUT.md:15,36` reports 17,572 MiB → 76.0% saving. Closeout text: "regression smoke is clean with every anchor byte-identical." Re-running Mistral-7B N=4 forward in pre-flight would take ~5-8 minutes (model load + forward + KL check); it is not re-run here.

**Drift:** none recorded since the 2026-05-19 12:04 logs.

### 3.4 — vLLM Phase 3 graph-mode parity (-0.39% target)

**Status: NOT RE-RUN, infrastructure intact, last-recorded result durable.**

**Infrastructure present:** `future_scope_a/phase3_vllm_probe.py`, `phase3_sweep.py`, `phase3_5_sweep.py`. Last-recorded result `phase3_5_sweep_result.json` (mtime 2026-05-19 15:49):

| Config | Clock (MHz) | tok/s mean | Power (W) mean | tok/W mean |
|---|---|---|---|---|
| alone_default | 1980 | 620.738 | 137.68 | 4.510 |
| volt1200 | 1200 | 522.300 | (computed) | 5.139 |
| volt800 | 810 | 356.800 | (computed) | 3.938 |

DVFS lift at the 1200 MHz sweet spot is 5.139 / 4.510 = **+13.94%**, consistent with the +13.9% figure in the executive summary and the v1.2 plan.

The −0.39% latency-delta figure cited in the executive summary lives in v1.2 plan `:64` and references Phase 3 graph-mode parity. The graph-mode raw probe was not re-run in this pre-flight; the JSON above is the closest recent measurement (Phase 3.5 DVFS sweep).

**Anomaly: none.**

### 3.5 — Phase 5 baseline 15-tenant FAIRNESS run

**Status: INFRASTRUCTURE NAMING MISMATCH. No single runner with this exact name and scope exists.**

The brief asks for a "Phase 5 baseline 15-tenant FAIRNESS run." A repository-wide search returns no runner with that title, scope, or matching tenant count.

Closest infrastructure found:

| Component | Path | What it does | Why it is the closest |
|---|---|---|---|
| Density sweep | `cipher_measurement/density_pack.sh` | 2 / 5 / 10 / 20 / 30 / 50 / 100 tenant sweep; samples per-tenant MFU from `/metrics`; stops when any tenant drops below `MIN_TENANT_MFU_PCT` | The most plausible "Phase 5 FAIRNESS-shaped" runner. Tenant counts do not include exactly 15. |
| CP 5.4 Step 1.6 16-slot sweep | `cp_5_4/step1_6/cp54_s16_*sweep.py` + `cp54_s16_orchestrator.py` + `cp54_s16_partition_tenant.py` + `cp54_s16_naive_orchestrator.py` | 16-slot ($2 \times$ 16-SM partition) sweep with disjointness and KL gates; last-recorded `cp54_cipher_sweep_summary.json` shows op2 / pool_agg_tok_s=286.527 / disjointness_all_pass=true / kl_all_pass=true | The 16-slot variant of an N-tenant sweep, with FAIRNESS-adjacent metrics. Closest to "15-tenant" by count, in spirit. |
| Single-process FAIRNESS unit | `cipher-may13-evidence/tests/test_fairness.py` | Tests the FAIRNESS op in a single process; not a 15-tenant sweep | Verifies the op functionally; not the scenario the brief names. |

**Recommendation surfaced for user adjudication (per discipline rule, no mitigation proposed in this document):** confirm whether "Phase 5 baseline 15-tenant FAIRNESS run" refers to `density_pack.sh` at a fixed N=15 (which the runner does not currently expose as a discrete step), or `cp_5_4/step1_6/cp54_s16_orchestrator.py` adjusted to 15 partitions (15 × 8-SM is the H100 green-context maximum per `cipher-cp54-15groups`), or a runner that needs to be authored before Week 1 sign-off.

---

## SECTION 4 — Git state verification

```
--- /home/ubuntu/cipher_rt_phase4 ---       (not a git repo)
--- /home/ubuntu/cipher_kmod ---            (not a git repo)
--- /home/ubuntu/cipher-may13-evidence ---  (not a git repo)
--- /home/ubuntu/cipher-fusion-evidence --- (not a git repo)
```

None of the four trees are git repositories. There is no branch to query, no commit history to inspect, no working-tree status to report. The session environment description provided at startup also lists `/home/ubuntu` itself as `Is a git repository: false`.

**Surfaced for user adjudication:** the brief's Step 4 framing ("current branch in cipher_rt_phase4 and cipher_kmod", "last 5 commits", "whether the worktree is clean") assumes git-based version control. The actual rollback discipline in use is:

- Anchor `.so` and `.ko` files preserved by md5 manifest.
- Tarball snapshots at known points (e.g., `cipher_kmod_src_track3_sc5.tar.gz`, `cipher_rt_green_ctx_track3_sc3.tar.gz`).
- The `cipher_rt_fallback/` and `cipher_kmod_fallback/` directories (the kmod fallback directory exists but is empty as of this check; the rt_fallback directory contains `libcipher_rt.so.track3_sc3`).

If the intent of Step 4 is to confirm "no uncommitted in-flight changes are pending before Week 1 file moves begin," the answer is: there are no commits to be uncommitted against; the working trees are the version of record. Anchor lineage is preserved through tarballs and md5 manifests only.

---

## SECTION 5 — Confirm prerequisites for Week 1

### 5.1 — Week 1 source files in `cipher-may13-evidence`

All 10 files brief-listed exist at the expected paths. The brief lists `cipher_classify.hpp` twice (once as `include/cipher_classify.hpp` in the source list, once again in the header list); this is the same file and is counted once below.

| File (cipher-may13-evidence/) | Size (bytes) | Lines | md5 8-char prefix | Plan §7 expectation | Result |
|---|---|---|---|---|---|
| `cipher_dispatch.cpp` (TOP-LEVEL) | 22,423 | **543** | `59271a09` | 543 LOC top-level live file | **PASS** |
| `cipher_oracle.cpp` (TOP-LEVEL) | 22,398 | 539 | `3b2c0d68` | top-level live file | PASS |
| `include/cipher_classify.hpp` | 9,952 | 250 | `4c615fa1` | classifier header | PASS |
| `src/cipher_recipes.cpp` | 22,060 | 522 | `0dd1e974` | 32+ entries seeded at L346 | PASS |
| `src/cipher_sense.cpp` | 14,174 | 340 | `65678060` | session classifier | PASS |
| `src/cipher_structural_lookup.cpp` | 13,278 | 300 | `b57aee5c` | L3.8 fast bypass | PASS |
| `include/cipher_recipes.h` | 8,843 | 199 | `e4d7e84a` | recipe header | PASS |
| `include/cipher_oracle.h` | 9,238 | 218 | `a871785b` | oracle header | PASS |
| `include/cipher_sense.h` | 1,527 | 45 | `c7f66410` | sense header | PASS |
| `include/cipher_structural_lookup.h` | 4,273 | 99 | `d2bf16c4` | structural-lookup header | PASS |

### 5.2 — `src/` shadow copies confirmed silently excluded

The plan §1.1 / Deep Inspection §B.1.7 / §D.1 claim two `src/` shadow files exist and differ from the top-level versions. Confirmed:

| `src/` shadow | Size | Lines | Differs from top-level? |
|---|---|---|---|
| `src/cipher_dispatch.cpp` | 25,625 b | **616** L | YES (`diff -q` differs) |
| `src/cipher_oracle.cpp` | 17,963 b | **451** L | YES (`diff -q` differs) |

`cipher-may13-evidence/Makefile` lines 27-30 confirm the filter-out:

```make
# RT DSO: all src/*.cpp EXCEPT the hook sources and the standalone tuner DSO,
# PLUS top-level dispatch and oracle
RT_CPP_SRC := $(filter-out src/cipher_intercept_cudart.cpp src/cipher_persist.cpp src/cipher_nccl_tuner.cpp src/cipher_dispatch.cpp src/cipher_oracle.cpp, \
                $(wildcard src/*.cpp))
RT_CPP_SRC += cipher_dispatch.cpp cipher_oracle.cpp
```

Week 1 ports the TOP-LEVEL files. The shadows must be deleted (or left in-place but never ported) to avoid future rsync-style mistakes per plan §3 row 10.

### 5.3 — LP-7 struct collision

Two competing definitions of `struct CipherKernelEntry` in may13:

```c
// cipher_kernel_table.h:53
typedef struct CipherKernelEntry {
    void*    fn_handle;          // CUfunction pointer
    char     name[128];          // demangled / mangled name
    uint8_t  category;           // CipherKernelCategory
    uint16_t param_count;        // 0 if cuFuncGetParamInfo failed
    uint16_t param_offsets[16];
    uint16_t param_sizes[16];
    uint32_t first_grid_x, first_grid_y, first_grid_z;
    uint32_t first_block_x, first_block_y, first_block_z;
    uint32_t first_smem_bytes;
    uint64_t observe_count;
} CipherKernelEntry;
```

```c
// cipher_param_recovery.h:30
typedef struct CipherKernelEntry {
    const void*       host_fun;          // PyTorch host stub address
    const char*       device_fun;        // mangled device function name
    const char*       device_name;       // demangled-ish name
    void*             fat_handle;        // opaque fatbin handle
    int               param_count;       // populated by lazy parse
    int               parsed_ok;
    CipherParamInfo   params[16];        // up to 16 params
} CipherKernelEntry;
```

The two structs are not aliases of each other; their field layouts overlap (both have `param_count`) but the rest is disjoint. Including both headers in any single translation unit fails to compile because C does not allow two `struct CipherKernelEntry` definitions with different bodies.

**Reference count across the whole repository:**

- 18 total occurrences of the symbol `CipherKernelEntry`.
- All 18 occurrences are in `cipher-may13-evidence/`.
- Zero occurrences in `cipher_rt_phase4/`.
- Zero occurrences in `cipher_kmod/`.

Files containing references:

- `cipher-may13-evidence/include/cipher_kernel_table.h` (definition + uses)
- `cipher-may13-evidence/include/cipher_param_recovery.h` (definition + uses)
- `cipher-may13-evidence/src/cipher_kernel_table.cpp` (uses)
- `cipher-may13-evidence/src/cipher_param_recovery.cpp` (uses)

### 5.4 — Rename proposal (per brief Step 5)

The brief proposes: "likely `cipher_may13_kernel_entry_t` for the may13 version, leaving the rt_phase4 version unchanged."

**One surfaced inconsistency, two named alternatives:**

1. **The rt_phase4 tree has zero `CipherKernelEntry` references.** "Leaving the rt_phase4 version unchanged" is a null operation, because there is no rt_phase4 version. The collision is entirely between two may13 headers, both of which Week 1 ports into rt_phase4.

2. **The brief's proposed name is singular**, but the collision is two-way. Renaming both definitions to `cipher_may13_kernel_entry_t` leaves the collision intact under the same new name. The brief's wording most likely intends one of the two; surfaced for adjudication which one keeps the original name (none of them can in the unified runtime, but one of them keeps it inside `cipher-may13-evidence`).

3. **Plan §7 Week 1 proposes a different rename**, with both definitions renamed:
   - `cipher_kernel_table.h::struct CipherKernelEntry` → `struct CipherKtEntry` (9 caller sites per Wave 1 §DEPENDENCIES)
   - `cipher_param_recovery.h::struct CipherKernelEntry` → `struct CipherParamEntry` (6 caller sites)

Both rename schemes break the compile collision; they differ in naming taste and in how visible "may13" is in the post-port code. Surfaced for user adjudication; the discipline rule blocks proposing mitigation inside this document.

---

## Section appendix — Files written / state changes during pre-flight

- `/tmp/preflight_backup/libcipher_rt.so.preclean` (150,408 B, md5 `83afd1ca...`) — copy of deployed binary made before `make clean`.
- `/tmp/preflight_backup/cipher_kmod.ko.preclean` (2,631,632 B, md5 `008b3c66...`) — copy of deployed kmod made before `make clean`.
- `/tmp/preflight_backup/cipher_kv_bridge.so.preclean` (373,752 B, md5 `c04b0c39...`) — copy of deployed kv_bridge made before any rebuild (kv_bridge was not rebuilt; copy taken for completeness).
- `/tmp/preflight_rt_build.log` — full `make clean && make -j` log for `cipher_rt_phase4`.
- `/tmp/preflight_kmod_build.log` — full `make clean && make -j` log for `cipher_kmod`.
- `cipher_rt_phase4/libcipher_rt.so` and `cipher_kmod/cipher_kmod.ko` were cleaned and rebuilt; both reproduce byte-identical to their pre-clean anchors per Section 2.
- `cipher_rt_phase4/*.o` object files exist on disk as a side effect of the rebuild and were not removed.
- `cipher_kmod/*.o`, `cipher_kmod/Module.symvers`, `cipher_kmod/modules.order` exist on disk as a side effect of the rebuild and were not removed.

If integration Week 1 prefers a fully clean workspace, a follow-up `make clean` (without rebuild) in both directories will return them to the same state minus the `.so` / `.ko` outputs, which are still preserved in `/tmp/preflight_backup/`.

---

**End of WEEK_1_PRE_FLIGHT.md (pre-stamp).**

Per the project's convention restated in CIPHER_REENGINEERING_PLAN.md §closing-note, the md5 of this file is recorded externally and noted in any subsequent revision rather than self-referenced inside the body of the file itself (the act of writing the md5 changes the md5).
