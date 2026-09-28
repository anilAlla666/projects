# CIPHER Logic Audit — Wave 5: Fusion Plan

**Scope.** Synthesis wave. Composes Waves 1-4 into one unified plan for fusing
the two trees `cipher-may13-evidence` (classifier brain + observers) and
`cipher_rt_phase4` + `cipher_kmod` (actuator substrate + kmod) into ONE runtime
without losses. Authoritative reference for executing the fusion.

**Inputs digested.** All four prior waves in full:
- Wave 1 `CIPHER_LOGIC_AUDIT_WAVE_1_CLASSIFIER.md` (115913 B / 1677 lines)
- Wave 2 `CIPHER_LOGIC_AUDIT_WAVE_2_ACTUATORS.md` (91829 B / 2051 lines)
- Wave 3 `CIPHER_LOGIC_AUDIT_WAVE_3_KMOD.md` (126441 B / 2497 lines)
- Wave 4 `CIPHER_LOGIC_AUDIT_WAVE_4_OBSERVERS.md` (112176 B / 2512 lines)
- Cross-ref `CIPHER_REENGINEERING_PLAN.md` Section 7 (5-week sequence; verified and
  corrected against Waves 1-4 in Section 5.5 below).

**Carry-forward findings from Waves 1-4 that drive the synthesis:**
- **F1-DRIFT (Wave 1, top-level vs src/).** `Makefile:29-31` `filter-out`s
  `src/cipher_dispatch.cpp` and `src/cipher_oracle.cpp`. The TOP-LEVEL copies at
  `cipher-may13-evidence/cipher_dispatch.cpp` (543 LOC) and `…/cipher_oracle.cpp`
  (539 LOC) are the canonical files; the `src/` shadows are dead code.
- **F1-LOAD-BEARING (Wave 1, `cipher_dispatch.cpp:480`).** The empirical hot path
  is `return CIPHER_PASS_THROUGH` on registry MISS — Goal-4 (Koopman-discovered
  recipes growing the registry) never fires; the lane is dark on real workloads.
- **F1-STRUCT-COLLISION (Wave 1).** `struct CipherKernelEntry` is defined with
  *different fields* in both `include/cipher_kernel_table.h:53-64` and
  `include/cipher_param_recovery.h:30-38`. Both headers cannot be `#include`'d
  in the same TU — must be renamed in Week 1.
- **F2-X HANDLED-DISCARD (Wave 2, `cipher_rt_attn_dispatch.cpp:321/359/398`).**
  Each SDPA trampoline (`flash` / `efficient` / `cuDNN`) calls `orig(...)`
  *unconditionally* after `route()`. T4.6.1 ships PASSTHROUGH-only by design.
  A classifier-driven substitute on attn produces NO behavioural change until
  the trampolines are refactored. **Blocking** for the attn substitute lane.
- **F2-ERR-SPLIT (Wave 2).** matmul ERROR breaks the registry walk and falls
  through to real cuBLAS (`cipher_rt_matmul_dispatch.c:101-104`). attn ERROR
  breaks the switch case and *continues* the for-loop to the next actuator
  (`cipher_rt_attn_dispatch.cpp:166-169`). A unified observer must know which
  substrate it sits on. **Requires-mitigation.**
- **F2-AUDIT-LOCK (Wave 2, `cipher_rt_audit.c:83-96`).** Every `record()` takes a
  single global mutex; HMAC-SHA256 runs synchronously under it. Under 16-tenant
  launch rate, this is the serialization floor. Today AUDIT is env-gated;
  deferred concern.
- **F3-NR-TABLE (Wave 3).** 24 `/dev/cipher` ioctl `nr`s assigned (21 live,
  3 reserved 2/3/4, 1 deactivated 9). Magic `'C'`. Separate `/dev/cipher_kvdedup`
  with magic `'K'` and 5 `nr`s. Strict additive-only [[cipher-abi-rule]].
- **F3-30BIT-PID (Wave 3, `cipher_cp54_sched.c:70-80`).** CP 5.4 group cells
  encode `(bit31=PACK, bit30=RSVD, bits0..29=pid)`. `PID_MAX_LIMIT=2^22` ⇒
  256× headroom.
- **F3-DO-EXIT-ORDER (Wave 3, `cipher_main.c:110-112`).** `cipher_cp54_sched_exit()`
  MUST follow `cipher_probe_exit()`. Hard invariant: the do_exit kprobe path
  (`cipher_probe.c:184-221`) calls `cipher_cp54_release(pid)` which depends on
  ledger state.
- **F3-ALLOC-RETIRE (Wave 3).** `cipher_partition_allocator.c` is REFACTOR-
  RETIRE not VESTIGIAL — userspace `nr 9` returns `-ENOSYS`, but two in-kmod
  callers are LIVE: `cipher_state_updater.c:176` calls `_tick()` and
  `cipher_probe.c:210` calls `_release_slots_only(pid)`.
- **F3-RSVD-TAIL (Wave 3, `cipher_ioctl.h:178`).** `struct cipher_tenant_snapshot_user`
  has `__u32 reserved[16]` at tail for additive classifier fields without new
  `nr`s. Wave 3 §FUSION POINTS calls this out.
- **F4-1 (Wave 4).** None of the 16 observers calls a `/dev/cipher` ioctl;
  none registers on the matmul/attn substrate registry. **The Wave-2 F2-X
  HANDLED-discard finding does NOT apply to Wave 4 ports.**
- **F4-2 / 5th-mode (Wave 4).** A fifth fusion mode `stage1-ring-consumer` was
  introduced — the dominant pattern; event-driven, single-thread, `void`-returning,
  consumes a `const CipherRingEntry*`.
- **F4-HOT-PATH (Wave 4).** `cipher_fairness_shm.cpp` sits on the cublasGemmEx
  hot path with a 64-slot acquire-load walk: **~1.4 µs/GEMM**. The only
  observer with non-trivial hot-path cost.
- **F4-WIRED (Wave 4).** Only 2 of 16 observers have active actuators today:
  SHIELD (`cipher_sm_set_priority` band write) and HIBERNATE (NVML power-limit
  write, DEGRADED on the pod).

**Anchors at synthesis time (per CLAUDE.md auto-memory).**
- kmod 0.4.8 with CP 5.4 + Track 2 + Track 3 ioctls live (devnode-codified
  `e2f50452`; CP 5.4 step rotation `285d102e`; Track 2 SC5 rotation `008b3c66`).
- `libcipher_rt.so` Phase 4 anchor `c2c5d313` (CP 2.5 closed); `libcipher_v2`
  anchor `86618c30` (Phase 3 baseline).

**Convention.** Every meeting point in 5.1 is given a stable contract ID
`C<class>.<num>` (e.g. `Ca.1` for classifier→actuator #1). Sections 5.3
(lossless), 5.4 (lossy), 5.5 (integration gates), and 5.6 (ASCII map) cite the
same IDs.

**Fusion classes** (carried from Waves 2-4):
- `PORT-AS-IS` — compile with no logic change.
- `SHIM-REQUIRED` — add adapter; no behavioural change to either side.
- `REFACTOR-REQUIRED` — substantive code change.
- `VESTIGIAL` — retired; not in OBJS.
- `REFACTOR-RETIRE` — live housekeeping but dead public surface; retire cleanly.

---

# 5.1 — FUSION CONTRACTS

Every meeting point between the two trees, documented as a row of an
analytic table per category. Contract ID, the two endpoints, what is
exchanged, error semantics, concurrency assumptions, fusion class, severity.

**Category index.**
- (a) classifier→actuator — Wave 1 brain calling into Wave 2 substrate /
  actuator surface. **12 contracts.**
- (b) classifier→kmod — Wave 1 brain reading kmod state via Wave 3 ABI, plus
  proposed additive `nr`s. **5 contracts.**
- (c) actuator→kmod — Wave 2 substrate calling Wave 3 ABI. **11 contracts.**
- (d) observer→{classifier, actuator, kmod} — Wave 4 observers' coupling
  surface. **8 contracts.**
- (e) kmod-internal cross-TU — Wave 3 invariants that survive the port.
  **7 contracts.**

**Total: 43 fusion contracts.**

---

## 5.1.a — Classifier → Actuator

| ID    | Caller (Wave 1 brain)              | Callee (Wave 2 substrate)            | Payload + ownership                                                    | Error semantics                          | Concurrency                                                                                  | Fusion class    | Sev  |
|-------|------------------------------------|--------------------------------------|------------------------------------------------------------------------|------------------------------------------|----------------------------------------------------------------------------------------------|-----------------|------|
| Ca.1  | new pre-launch hook in `cipher_cupti.c::cipher_v2_cupti_cb` post-make_current OR in matmul/attn dispatch | `cipher_rt_classify_pre_launch(const cipher_rt_call*)` — a new substrate `cipher_rt_dispatch.cpp` mirroring `cipher_rt_matmul_dispatch_init` | Caller passes `CipherKernelDesc{fn,grid,block,smem,...}`. Brain returns `(op_class, confidence, decision)` via TLS slot. Ownership: caller owns desc; brain owns TLS slot for its thread. | Brain returns `CIPHER_PASS_THROUGH` on any internal failure (oracle DENY / classify UNCLASSIFIED). Never aborts the launch. | Hot-path-sync per launching thread. No locks. The 512-slot classify cache (Wave 1 `cipher_classify.hpp:188`) is lock-free relaxed-atomic. | REFACTOR-REQUIRED (new file `cipher_rt_classify_substrate.cpp`) | blocking |
| Ca.2  | `cipher_dispatch.cpp:131,178-205 (weak `cipher_tls_get_gemm_shape`)` | originally `libcipher_hook.so::cipher_intercept_cudart.cpp:209` (TLS GEMM shape) | Wave 2's `cipher_rt_cublas_shim.c:73` has the M/N/K in `cipher_rt_matmul_call` at L100-120 — no TLS needed. | N/A in fused tree — call site reads `call->m/n/k`. | The cublas shim is the producer of M/N/K — single-thread per call. | REFACTOR-REQUIRED (drop the TLS accessor; rewrite `gemm_shape_hash` to read from the actuator-registered classifier's `call` arg). | requires-mitigation |
| Ca.3  | `cipher_dispatch.cpp::apply_recipe` GEMM branch (weak `cipher_tls_relaunch`) | originally `libcipher_hook.so` re-execute the captured cuLaunchKernel | In Tree B, the actuator chain (Marlin at priority 10) IS the relaunch. Brain just returns `SUBSTITUTE` and the registry's next-higher-priority actuator handles. | Brain must not call relaunch; on SUBSTITUTE return, `cipher_rt_matmul_dispatch.c:99-101` follows the registry to actuator | Same as Ca.1 (one substrate dispatch per launch). | REFACTOR-REQUIRED (replace tls_relaunch contract with "return SUBSTITUTE; let chain continue") | requires-mitigation |
| Ca.4  | `cipher_dispatch.cpp::apply_recipe` GEMM branch (weak `cipher_get_edmd_pipeline`) | Tree B has no EDMD module today (Goal-4 deferred) | Brain probes weak symbol; resolves NULL on Tree B; falls through to standard relaunch (which becomes Ca.3) | NULL weak → fallback path; safe. | N/A — pure weak-symbol resolution at runtime. | PORT-AS-IS (stub returns NULL until v2 Koopman lane). | minor |
| Ca.5  | `cipher_dispatch.cpp::apply_recipe` post-relaunch (weak `cipher_edmd_live_collect`) | `cipher_edmd_live.cpp:415` not compiled into Tree B | Same as Ca.4 — weak symbol resolves NULL; collect becomes no-op. | NULL weak → no-op. | N/A. | DEFER to v1.5. | minor |
| Ca.6  | `cipher_dispatch.cpp::g_cipher.liquid` (init at L113; record_op at L440) | `cipher_lnn.cpp` / `cipher_liquid_state.cu` — NOT in Tree B build | Pass `nullptr` to `cipher_oracle_init`; oracle is null-safe at `cipher_oracle.cpp:197,248,374,396,408`. | nullptr → record functions become no-ops. Safe. | N/A — single init thread. | REFACTOR-REQUIRED (small): null-pass at init site. | minor |
| Ca.7  | new classifier observer (priority 5) on the matmul substrate | `cipher_rt_matmul_dispatch.c::cipher_rt_matmul_register_actuator(L57)` | Brain installs a `cipher_rt_matmul_actuator{name="classify", priority=5, maybe_handle=<observer>}`; maybe_handle returns PASSTHROUGH (no claim) but publishes a TLS hint `(op_class, confidence, decision)` for Marlin (priority 10) to read. | PASSTHROUGH from observer; ERROR breaks the chain (per F2-ERR-SPLIT) — must never return ERROR. | `g_disp.reg_lock` for register; lock-free hot-path snapshot read at L94. | SHIM-REQUIRED (small). | minor |
| Ca.8  | new classifier observer (priority 5) on the attn substrate | `cipher_rt_attn_dispatch.cpp::cipher_rt_attn_register_actuator(L112)` | Mirror of Ca.7 for attn. Observer returns PASSTHROUGH; HANDLED is structurally discarded today (F2-X) but the observer never returns HANDLED. | PASSTHROUGH only; never ERROR (attn ERROR semantic continues to next actuator, but accumulating ERRORs would noise the logs — same discipline as Ca.7). | `g_reg.mu` for register; snapshot-under-lock at L145-148; release before invoking actuators. | SHIM-REQUIRED (small). | minor |
| Ca.9  | Marlin actuator reading TLS classifier hint | `cipher_rt_marlin_actuator.c::maybe_handle_marlin(L56)` adds read of TLS hint to short-circuit STABILITY_THRESHOLD observation | New TLS slot `(op_class, recipe_id_present)` published by Ca.7 observer; Marlin reads on entry; if hint says "skip" (e.g., classifier confidence < 60), Marlin returns PASSTHROUGH immediately. | If TLS slot empty: Marlin behaves as today (own gate). | Per-thread TLS; no synchronization. | SHIM-REQUIRED (small). | minor |
| Ca.10 | `cipher_dispatch.cpp::infer_layer_context` (hardcoded `total_layers=80`) | `cipher_rt_tenant_cpp::cipher_rt_tenant_cached()` provides per-tenant model fingerprint in v1.5 | In v1, `total_layers` becomes a per-process env `CIPHER_TOTAL_LAYERS` (default 32 for Mistral-7B); Wave 1 drift "80-layer Llama assumption" closed. | Default = 32; out-of-range layer_idx is clamped < total_layers in oracle. | TLS counter per thread; non-atomic but never read cross-thread. | REFACTOR-REQUIRED (small): replace constant with env-read at init. | minor |
| Ca.11 | brain classify entrypoint `cipher::classify_launch(...)` (header-only, `cipher_classify.hpp:227`) | wrapped by new `extern "C" cipher_rt_classify_launch_t cipher_rt_classify_launch(fn,grid,block,smem)` in `cipher_rt_classify.cpp` | Same struct contents (op_class enum, confidence, cache_hit). Brain remains namespace-local; C wrapper is the ABI seam. | None — pure logic; classify always returns a value. | Header is lock-free relaxed atomic (signal-safe). | PORT-AS-IS (add C wrapper). | none |
| Ca.12 | brain `cipher_struct_lookup(&sctx)` (`cipher_structural_lookup.cpp:194`) with `ctx.kernel_name = NULL` | populates `oq.kernel_name` from `cipher_rt_kernel_table.cpp::cipher_kt_name_for(fn)` (ported from Wave 1) | Lookup expects null-safe `kernel_name`; if non-null, the Gate-5 prefix-rule table fires. | Null → Gates 1-4, 6, 7 still active; safe. | Lock-free (Wave 1 §CONCURRENCY). | SHIM-REQUIRED (small): wire kernel_table → oracle query. | minor (activates a dormant gate) |

---

## 5.1.b — Classifier → Kmod

| ID    | Caller (Wave 1 brain)                                       | Callee (Wave 3 kmod)                                | Payload + ownership                                            | Error semantics                            | Concurrency                                                       | Fusion class               | Sev  |
|-------|-------------------------------------------------------------|-----------------------------------------------------|----------------------------------------------------------------|--------------------------------------------|-------------------------------------------------------------------|----------------------------|------|
| Cb.1  | brain reading per-tenant snapshot (new in fused tree)       | `cipher_dev.c:118 cipher_dev_get_tenant_snapshot` (ioctl `nr 8`) | Caller fills `q.target_pid` (or `target_tenant_id`); kmod fills `q.snapshot` (332B layout-identical with kernel struct). Userspace owns query buffer. | `-ENOENT` if not found; `-EFAULT` on copy; otherwise success. Brain falls back to its own defaults on error. | Per-CPU TLS snapshot inside RCU section; safe across threads. | PORT-AS-IS (already shipped). | none |
| Cb.2  | brain consuming new classifier-decision fields in the snapshot | `cipher_ioctl.h:178 __u32 reserved[16]` — additive slot | Brain reads new fields like `recommended_sm_count`, `slo_priority`, `tenant_billing_class` from the reserved tail. Fields must be written by some kmod producer (state_updater or new policy hook). | If kmod hasn't been bumped, reserved tail is zero → brain uses defaults. ABI-safe. | Read under RCU; non-atomic single-field reads. | REFACTOR-REQUIRED (small): bump `cipher_internal.h` struct, add WRITE_ONCE in `cipher_state_updater.c`. | minor |
| Cb.3  | brain emitting classifier decisions back to kmod (new ABI `nr 25`) | proposed new `cipher_dev.c` case `CIPHER_CLASSIFY_SUBMIT` (ioctl `nr 25`) | Brain submits `{pid, recipe_id, decision, confidence, layer_idx}` per launch (rate-limited); kmod accumulates in `cipher_pid_stats` extension. | Anti-spoof check: caller `pid == current->pid`. `-EFAULT/-EINVAL` otherwise. | Per-fd lock-free atomic writes to per-pid stats. | REFACTOR-REQUIRED (small): new ioctl + struct + handler. | minor (deferred to v2; v1 carries decisions in userspace TLS only) |
| Cb.4  | brain registering per-pid classifier hook into kprobe pre-handler | `cipher_probe.c::cipher_kprobe_pre(L78-111)` exposes a callout slot | Brain registers `cipher_classify_observe_ioctl(slot)` to feed classifier with (tenant, slot) events. Pure observation. | Callout signature `void(void*)`; can't fail. | Kprobe atomic context; callout must be lock-free & GFP_ATOMIC-safe. | REFACTOR-REQUIRED (small): one line in kprobe_pre. | minor |
| Cb.5  | brain hook in 1 kHz state_updater                           | `cipher_state_updater.c::cipher_state_updater_fn(L139-183)` | Inside the RCU walk, brain computes classifier-derived per-tenant fields and writes them with `WRITE_ONCE` into the reserved-tail slots from Cb.2. | Init failure of brain is tolerated; state_updater continues with brain disabled. | kthread RCU read-side; WRITE_ONCE single-field. | REFACTOR-REQUIRED (small): per-entry function call. | minor |

---

## 5.1.c — Actuator → Kmod

| ID    | Caller (Wave 2 substrate)                                                              | Callee (Wave 3 ioctl `nr`)                                            | Payload + ownership                                                                                          | Error semantics                                                                              | Concurrency                                              | Fusion class | Sev  |
|-------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------|-------------|------|
| Cc.1  | `cipher_tenant.c:21 cipher_v2_tenant_register`                                          | `cipher_dev.c:58` `CIPHER_REGISTER_TENANT` (nr 1)                     | `{pid=gettid(), tgid=getpid(), tenant_id[64]}`; anti-spoof checks PID/TGID                                   | `-EPERM` on spoof; `-EFAULT` on copy; `-ENOMEM`. cipher_inject tolerates failure.            | Per-call. Spinlock `cipher_pid_insert_lock` for insert. | PORT-AS-IS  | none |
| Cc.2  | `cipher_cupti.c:205` per-256-launch flush                                              | `cipher_dev.c:225` `CIPHER_SUBMIT_LAUNCH_STATS` (nr 7)                | `{pid,tgid,total_launches,grid_ops}`; anti-spoof; rate-limited to one IOCTL per 256 launches.                | `-EPERM` / `-EFAULT` / `-EINVAL`. Best-effort; loss is logged.                                | Workload thread; no kmod-side mutex; `cipher_pid_stats` atomic_inc. | PORT-AS-IS | none |
| Cc.3  | `cipher_rt_tenant.cpp::cipher_rt_tenant_query_by_pid (L118)`                            | `cipher_dev.c:118` `CIPHER_GET_TENANT_SNAPSHOT` (nr 8)                | Per-thread fd; ~5 µs cost; refreshes `__thread g_tls_cache.snap`.                                            | `-EFAULT/-ENOMEM`. Caller falls back to `/proc/cipher/stats` (Mode 2).                       | Per-thread `__thread` fd; no shared mutex.              | PORT-AS-IS | none |
| Cc.4  | `cipher_rt_volt.c::cipher_rt_volt_init` (kmod path L317)                                | `cipher_clock.c:130` `CIPHER_SET_CLOCK_MHZ` (nr 10)                   | `__u32 mhz` (0=unlock; otherwise [210,1980]). Audit-logged with caller pid+uid.                              | `-EINVAL/-EIO/-ERESTARTSYS`. VOLT falls back to NVML direct path or DEGRADED.                | `cipher_clock_lock` mutex serialises; spawns nvidia-smi via UMH. | PORT-AS-IS | none |
| Cc.5  | `cipher_flopd` daemon (root) submitting samples                                         | `cipher_flops.c:89` `CIPHER_SUBMIT_FLOP_SAMPLE` (nr 11)               | `{timestamp_ns, tensor_pipe_milli_pct, sm_util_pct, ...}`. CAP_SYS_ADMIN.                                    | `-EPERM/-EFAULT/-EINVAL`.                                                                     | `cipher_flop_lock` spinlock; single-writer in practice.  | PORT-AS-IS | none |
| Cc.6  | userspace MFU consumer (Wave 1 oracle indirectly via `cipher_oracle_update_mfu`)         | `cipher_flops.c:178` `CIPHER_QUERY_FLOPS` (nr 12)                     | Unprivileged read; kmalloc'd ~5 KiB; returns per-tenant `attributed_flops_per_s` and global status.          | `-ENOMEM/-EFAULT`.                                                                            | RCU read; lock-free per-tenant.                          | PORT-AS-IS | none |
| Cc.7  | `cipher_rt_green_ctx.c::cipher_rt_green_ctx_cp54_init(L137-203)`                        | `cipher_cp54_sched.c:550` `CIPHER_CP54_ALLOCATE` (nr 13)              | `{qos_class, sm_count, grp_mask_out, grp_count_out}`; kmod returns assigned mask.                            | `-ENOMEM/-EINVAL/-EEXIST/-ENOSPC`. Userspace falls back to per-pid hash-pick.                | `cipher_cp54_lock` mutex.                                | PORT-AS-IS | none |
| Cc.8  | `cipher_rt_green_ctx.c` migrate path (Track 3 SC3)                                       | nr 14/15/16/17/18/19/20 (FREE/QUERY + 5 SC2 migration ioctls)         | Step-by-step state machine (SUBSCRIBE/POLL/START/ACK/COMPACT). Lock-free atomic cmpxchg in cell ledger.       | Per-ioctl. `cp54_release_groups` clears both PACK and RSVD on do_exit — zero leak.            | mutex for ALLOCATE/SUBSCRIBE/START/ACK/COMPACT; lock-free for POLL/QUERY/release. | PORT-AS-IS | none |
| Cc.9  | `cipher_rt_kv_alloc.c:472 cipher_rt_kv_dedup_init` + 5 ioctls                            | `cipher_kvdedup.c:361` magic `'K'` nrs 1-5 (INIT/PUT/CONFIRM/FREE/STATS) | Two-phase handshake on PUT/CONFIRM (memcmp-verify on userspace; refcount on kmod). cuIpc POSIX-FD handles owned by kmod's `struct file*`. | `-EBADF/-ENOMEM/-ENOSPC/-EPERM`. Cross-tenant access survives producer crash via fd lifetime. | Single mutex on hot path; release fop on SIGKILL.        | PORT-AS-IS | none |
| Cc.10 | `cipher_rt_kv_alloc.c` weight arena create/import/export/leave                          | `cipher_weight_arena.c` magic `'C'` nrs 21-24 (REGISTER/IMPORT/LEAVE/QUERY) | `{fd_in, base, size, blob_len, blob[≤64K], arena_id, fd_out}`. Kmod holds independent `struct file*` ref.    | `-EBADF/-ENOSPC/-ENOENT/-EFAULT`. Off-do_exit reaper at 5 s polls liveness via pid table.     | `cipher_wa_lock` mutex; fd table ops via kernel. | PORT-AS-IS | none |
| Cc.11 | `cipher_cupti.c` stream-create callback (Phase 4)                                       | (implicit via `cuStreamSetAttribute`; not a kmod ioctl)               | Stream-priority + sync-domain-map set on each new stream's first observation.                                 | best-effort; per-CB swallowed.                                                                | per-stream lock-free table; 256 slots.                   | PORT-AS-IS | none |

---

## 5.1.d — Observer → {Classifier, Actuator, Kmod}

Re-verified against Wave 4 F4-1 / F4-2: **no Wave-4 observer calls a `/dev/cipher`
ioctl today**. The "observer→kmod" contracts below are *proposed* for v1.5.

| ID    | Observer (Wave 4)                                                                       | Counterparty                                              | What is exchanged                                                                                              | Error semantics                                                                  | Concurrency                                  | Fusion class             | Sev  |
|-------|------------------------------------------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------|--------------------------|------|
| Cd.1  | `cipher_shield.cpp:189` → `cipher_sm_set_priority`                                       | Wave 2 green-ctx priority table (userspace, NOT kmod)     | `(session_id, CIPHER_BAND_PROTECTED)`; one-shot per HUMAN_INTERACTIVE session.                                | none (idempotent set).                                                            | Single Stage-1 thread; per-session sticky.   | PORT-AS-IS               | none |
| Cd.2  | `cipher_fairness_shm.cpp:867-883` → consumed by `cipher_intercept_cudart.cpp` cublasGemmEx shim | userspace POSIX shm `/cipher_fairness` (64-slot, magic `0xC1F4C1F4`) | `should_yield()` returns bool; producer pays ~1.4 µs/GEMM (64 acquire-loads).                                | shm full → silent. n_active≤1 → return 0.                                         | Cross-process atomic ops.                    | REFACTOR-REQUIRED for v1.5 (proposed kmod nrs 25/26/27); PORT-AS-IS for v1. | requires-mitigation |
| Cd.3  | `cipher_thermostat.cpp:112-113` reads `g_cipher.liquid.device->hw.gpu_temp_c`            | Stage 11 thermal-feedback infra (cipher.h extern)         | Single-precision celsius temp; non-atomic read.                                                                 | If `g_cipher.liquid` not initialized, returns 0.0f — drift signal only.          | Multi-reader; producer is Stage 11 thread.   | PORT-AS-IS               | none |
| Cd.4  | `cipher_pulse.cpp:130-131` (RTLD_NOLOAD NVML)                                            | libnvidia-ml.so (loaded by Stage 11)                      | ECC error counter via `nvmlDeviceGetMemoryErrorCounter`. Read-only.                                            | NVML missing → ECC signal disabled (score ceiling 1).                            | Single Stage-2 thread.                       | PORT-AS-IS               | none |
| Cd.5  | `cipher_hibernate.cpp:67-78` (full NVML load) → `nvmlDeviceSetPowerManagementLimit`      | libnvidia-ml.so direct                                    | Write power limit MW; rolled back on signal.                                                                    | NVML_ERROR_NOT_SUPPORTED → DEGRADED observability-only.                          | Stage-2 thread + signal handler.             | REFACTOR-REQUIRED (small) for v1.5 — add kmod-clock fallback via nr 10. | minor |
| Cd.6  | Multiple observers reading `cipher_sense_current_session()` TLS                          | Op-13 SENSE (Wave 1)                                      | `uint64_t session_fingerprint`; SENSE-set TLS slot.                                                            | If SENSE disabled, session==0 ⇒ observer gates pass through.                     | Per-thread TLS.                              | PORT-AS-IS               | none |
| Cd.7  | All 13 Stage-1 observers consume `const CipherRingEntry*`                                | `cipher_10ops_impl.cpp::stage1_shadow(L509-560)`          | 128-byte cache-line record (14 fields). 65536-slot SPMC ring. Slow observer back-pressures producer (drops).    | Drop on ring full; producer increments dropped counter.                          | Single Stage-1 thread today; ring is SPMC.    | PORT-AS-IS               | none |
| Cd.8  | `cipher_audit.c::cipher_rt_audit_record(SUBSTITUTE, recipe_hash, SUBSTITUTED)` (proposed call site) | `cipher_rt_audit.c:78` exported `record()` (already shipped) | Op enum + 64-bit hash + decision; HMAC-SHA256 chain advances.                                                  | global mutex `g.mu`; record never returns. F2-AUDIT-LOCK serialization floor.    | Global mutex on every record() — see F2-AUDIT-LOCK. | SHIM-REQUIRED only on the upstream caller (substitute-lane actuator). | minor (the recv side is shipped) |

---

## 5.1.e — Kmod-internal cross-TU

Invariants that survive the fusion port unchanged. These do NOT change in the
fused tree, but the integration sequence must preserve them.

| ID    | Producer / Consumer                                                            | Contract                                                                                                                                                                          | Invariant violation cost                                                              | Sev  |
|-------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|------|
| Ce.1  | every TU ↔ `cipher_pid_table` (1024 buckets, hlist + RCU)                       | Readers `hash_for_each_*_rcu`; writers under `cipher_pid_insert_lock` use `hash_add_rcu`/`hash_del_rcu`/`kfree_rcu`. Lookup-miss in slow path re-checks under lock.                | Bypass lock → duplicate inserts or use-after-free.                                     | blocking |
| Ce.2  | `cipher_probe.c:184-221 cipher_do_exit_pre` ↔ `cipher_cp54_sched.c` + `cipher_partition_allocator.c` | Unconditional `cipher_cp54_release(pid)` first (lock-free, atomic-safe). Then conditional RCU-lookup; on hit, `cipher_partition_release_slots_only(pid)` (lock-free) and reap entry. | Mid-COMMIT or mid-migration crash leaks groups → `cp54_release_groups` sweeps both PACK and RSVD in one pass (`cipher_cp54_sched.c:196-217`). Zero leak. | blocking |
| Ce.3  | `cipher_main.c:110-112` exit-order: `cipher_cp54_sched_exit()` MUST follow `cipher_probe_exit()` | The do_exit kprobe path depends on CP 5.4 ledger state. Reversing the order races a process exiting mid-`rmmod` against torn-down ledger. | Crash on rmmod under load.                                                              | blocking |
| Ce.4  | `cipher_gpu_state.lock` spinlock serialises 8-field `cipher_gpu_state` write    | `cipher_dev.c:182-191` cipher_dev_submit_gpu_state holds spin_lock; readers (proc, state_updater) memcpy under same lock.                                                          | Torn cross-field reads if lock dropped.                                                | requires-mitigation |
| Ce.5  | `cipher_cp54_sched.c:cipher_cp54_lock` mutex for ALLOCATE/SUBSCRIBE/START/ACK/COMPACT | Lock-free FREE / POLL / QUERY. Lock-free do_exit reaper (no mutex). Discipline: each PID writes only its own metadata slot.                                                       | Reaper deadlock if mutex acquired in atomic context.                                   | blocking |
| Ce.6  | `cipher_wa_lock` mutex protects all 16 weight arenas; reaper takes same mutex   | REGISTER rolls back on `copy_to_user` failure (`cipher_weight_arena.c:201-209`). IMPORT reserves fd via `get_unused_fd_flags` *before* install.                                  | Orphaned arena (fd alive but in_use=0) or double-install crash.                        | requires-mitigation |
| Ce.7  | `cipher_kvdedup.c:kvd_lock` mutex; release fop reaps on SIGKILL                 | PUT does NOT bump refcount on HIT (two-phase handshake — CONFIRM does). HIT-without-CONFIRM crash leaks zero refs (kmod's own ref unchanged).                                     | Refcount leak or double-deref on crash.                                                 | blocking |

---

# 5.2 — THE FUSED KERNEL INTERCEPT SEQUENCE

Step-by-step sequences for the four kernel intercept types in the fused
runtime. Each step cites the file:function that runs it. Per the task brief.
"step N" lines deliberately fit < 200 chars.

**Format.** Each step: `step N: <file>:<function>(L<line>) — <what it does>`.
Classifier observer insertion is annotated with `[OBS: classify, prio=5]`.

---

## 5.2.1 — cuBLAS `cublasGemmEx` (matmul substrate; Marlin actuator path)

The path PyTorch's `at::cuda::detail::lazyNVRTC.cublasGemmEx` (and similar)
takes through the fused runtime, from GOT-patched entry to either real-cuBLAS
fallthrough or Marlin substitution.

```
step  1: cipher_rt_got_patch.c:phdr_cb(L128)       — at libcipher_rt init, dl_iterate_phdr scans every loaded ELF
step  2: cipher_rt_got_patch.c:write_got_slot(L105)— rewrites PyTorch's GOT slot for "cublasGemmEx" → cipher_rt_cublasGemmEx_impl
step  3: PyTorch user call cublasGemmEx(...)       — GOT-redirected to cipher_rt_cublasGemmEx_impl
step  4: cipher_rt_cublas_shim.c:cipher_rt_cublasGemmEx_impl(L73) — entry; bumps g_shim_calls
step  5: cipher_rt_cublas_shim.c:shim_lazy_init(L50) — dlopen libcublas.so.13/.12 RTLD_NOLOAD, dlvsym GemmEx real fn
step  6: cipher_rt_cublas_shim.c:L91-99            — best-effort g_cublasGetStream(handle, &stream)
step  7: cipher_rt_cublas_shim.c:L100-120          — build struct cipher_rt_matmul_call{m,n,k,A,B,C,types,algo,stream}
step  8: cipher_rt_cublas_shim.c:L122              — call cipher_rt_matmul_dispatch(&call, g_real_gemmEx)
step  9: cipher_rt_matmul_dispatch.c:L91           — atomic_fetch_add(&g_disp.total, 1)
step 10: cipher_rt_matmul_dispatch.c:L94           — n = g_disp.n_actuators (snapshot, lock-free)
step 11: cipher_rt_matmul_dispatch.c:L96-99 [iter 1] [OBS: classify, prio=5] — new classifier observer maybe_handle
step 12:   classifier observer (new TU cipher_rt_classify_observer.c) — call cipher::classify_launch(fn,grid,block,smem) [Wave 1 Ca.11]
step 13:   classifier observer — read call->m/n/k directly (Wave 1 Ca.2 — drop tls_get_gemm_shape contract)
step 14:   classifier observer — cipher_rt_oracle_decide(state, query) [Wave 1 Ca.7 publish hint to TLS]
step 15:   classifier observer — publish TLS hint (op_class, confidence, decision); return PASSTHROUGH (never HANDLED, never ERROR)
step 16: cipher_rt_matmul_dispatch.c:L96-99 [iter 2] [OBS: audit, prio=0] — if CIPHER_AUDIT, audit_matmul_handle (L101 audit.c)
step 17:   cipher_rt_audit.c:audit_chain_update(L61) — HMAC-SHA256 under g.mu — F2-AUDIT-LOCK
step 18: cipher_rt_matmul_dispatch.c:L96-99 [iter 3] [ACT: marlin, prio=10] — cipher_rt_marlin_actuator.c:maybe_handle_marlin(L56)
step 19:   cipher_rt_marlin_actuator.c:L92         — gate on g_enabled (CIPHER_MARLIN env)
step 20:   cipher_rt_marlin_actuator.c:L95         — dtype gate FP16 only
step 21:   cipher_rt_marlin_actuator.c:L101-118    — shape gate (M≤64 N≥1024 K≥1024 K%128 N%64)
step 22:   cipher_rt_marlin_actuator.c:L125-132    — engine_observe_weight: STABILITY_THRESHOLD=4 OR engine_quantize_repack
step 23:   cipher_rt_marlin_actuator.c:L146        — engine_lookup returns (B,S,K,N,G) or PASSTHROUGH
step 24:   cipher_rt_marlin_engine.cpp:cipher_rt_marlin_engine_dispatch(L999) — primary-ctx guard if no green ctx
step 25:     marlin_gemm_launch(L771) — block=256, dyn_shared=96K, grid=green_ctx_sm_count (CP 5.3 STEP 2A)
step 26:     event-wait dance (L1021-1059) — CP 5.6 P1 fix for F1 cross-context race
step 27:   marlin returns HANDLED with CUBLAS_STATUS_SUCCESS → matmul_dispatch.c:L98-99 atomic_fetch_add(&handled)
step 28: cipher_rt_matmul_dispatch.c:L99           — return status; ALL DONE (no real cublas)

(if Marlin returns PASSTHROUGH, the chain has no other actuators; jumps to fallthrough)
step 29: cipher_rt_matmul_dispatch.c:L108-110      — atomic_fetch_add(&passthrough); return g_real_gemmEx(...)
step 30: libcublas.so:cublasGemmEx                 — the original kernel actually runs

(CUPTI callback fires INDEPENDENTLY of the matmul substrate on every cuLaunchKernel beneath cublasGemmEx — see 5.2.3)
```

**Hot-path budget summary** (per cublasGemmEx call, cache-warm, no Marlin engagement):
- step 5-7 (lazy init + GetStream + build call): ~50 ns once warm.
- step 9-10 (atomic + snapshot): ~5 ns.
- step 11-15 (classifier observer): **~120-200 ns** (cache-hit classify ~5 ns;
  oracle decide ~50 ns; TLS write ~5 ns; per Wave 1 §CONTROL FLOW).
- step 16-17 (audit if enabled): ~50 ns + global-mutex hold.
- step 18-23 (Marlin gate eval): ~30 ns (shape/dtype gates).
- step 28 (HANDLED return): atomic_fetch_add ~3 ns.

**Sum (audit off, Marlin not engaged):** ~200-250 ns added per GEMM.
**Sum (audit off, Marlin engaged):** ~250 ns + ~1.8 µs Marlin dispatch.

---

## 5.2.2 — SDPA / cuDNN attention (attn substrate; FlashAttention path)

PyTorch ATen SDPA dispatcher's three backends (flash / efficient / cuDNN) all
go through GOT-patched mangled `::call` entries in libtorch_cpu.so.

```
step  1: cipher_rt_got_patch.c:phdr_cb(L128)       — patches libtorch_cpu.so GOT slots for 3 SDPA mangled symbols
step  2: PyTorch SDPA dispatcher calls flash::call(...)  — GOT-redirected to hidden trampoline
step  3: cipher_rt_attn_dispatch.cpp:flash_call(L289)    — entry; g_tramp_calls.fetch_add(1) at L295
step  4: cipher_rt_attn_dispatch.cpp:resolve_lazy(L190) — dlopen libtorch_cpu.so RTLD_NOLOAD; dlsym mangled name → real flash::call
step  5: cipher_rt_attn_dispatch.cpp:tensors_are_real(L303) — skip FakeTensor/Meta
step  6: cipher_rt_attn_dispatch.cpp:L304-313      — build cipher_rt_attn_call{backend,q,k,v,...}
step  7: cipher_rt_attn_dispatch.cpp:route(L135)   — registry walk
step  8:   route.c:L145-148 — pthread_mutex_lock(&g_reg.mu); memcpy snap[] = g_reg.entries; unlock
step  9:   route.c:L150-174 — iterate snap[] in priority order:
step 10:   iter 1 [OBS: classify, prio=5] — new classifier observer maybe_handle for attn
step 11:     classifier observer — call cipher::classify_launch (Ca.11) returns ATTENTION op_class
step 12:     classifier observer — oracle_decide; publish TLS hint; return PASSTHROUGH
step 13:   iter 2 [OBS: audit, prio=0] — audit_attn_handle(L117 audit.c); HMAC-SHA256; return PASSTHROUGH
step 14:   iter 3 [OBS: attn_test, prio=0 if CIPHER_ATTN_TEST] — count + log; return PASSTHROUGH
step 15:   route.c:L175-176 — atomic_fetch_add(&g_passthrough); return PASSTHROUGH
step 16: cipher_rt_attn_dispatch.cpp:L321          — ALWAYS calls orig(q,k,v,...) regardless of route() return
step 17: real flash::call runs                     — FlashAttention executes

[F2-X CARRY-FORWARD: step 16 is the HANDLED-discard. T4.6.1 ships PASSTHROUGH-
only by design; a classifier-driven SUBSTITUTE actuator returning HANDLED at
step 9 produces NO behavioural change because step 16 still calls orig.
Mitigation: trampoline refactor at step 16:
  if route()==HANDLED && substitute_devptr != NULL: return substitute_devptr
  else: return orig(q,k,v,...)
This is REFACTOR-REQUIRED for the attn substitute lane (Wave 5.4 lossy point
LP-2). For v1 (substitute lane not yet exercised), the discipline is "never
return HANDLED from an attn observer" — guaranteed by Wave 4 F4-1.]
```

**Hot-path budget summary** (per SDPA call, cache-warm):
- step 3-6 (entry + dlsym lazy + descriptor): ~80 ns warm.
- step 8 (snapshot under lock): ~40 ns (one mutex acquire + 16-actuator memcpy ≤ 1 KB).
- step 9-15 (observer iteration): ~150 ns (3 observers × ~50 ns).
- step 16 (orig invoke): unchanged.

**Sum:** ~270 ns added per SDPA call. Less critical than GEMM hot path because
SDPA call rate is ~1 per layer per token, not per matmul.

---

## 5.2.3 — `cuLaunchKernel` / `cudaLaunchKernel` (general kernel dispatch via CUPTI)

This is the path EVERY cuda kernel launch takes — both cuBLAS-internal launches
(which fire underneath cublasGemmEx) and PyTorch eager kernels (RMSNorm, RoPE,
SiLU, etc.). Today the CUPTI callback owns this; the matmul/attn substrates
don't see it (they're at the cuBLAS/SDPA seam, one level higher).

```
step  1: PyTorch / cublasGemmEx-internal call cuLaunchKernel(fn,grid,block,...) — driver entry
step  2: cuda driver invokes CUPTI subscribers in registered order
step  3: cipher_cupti.c:cipher_v2_cupti_cb(L67) — entry; bail unless API_ENTER
step  4: cipher_cupti.c:L109-127 [T4.2.4d enforcement gate]:
step  5:   if !cipher_rt_in_marlin_quant (TLS): cipher_rt_green_ctx_ensure(L224 green_ctx.c)
step  6:                                       cipher_rt_green_ctx_make_current(L207 green_ctx.c)
step  7:   else: skip (Marlin's own quant kernels stay on primary ctx — see cipher_rt_marlin_engine.cpp PrimaryCtxGuard L873)
step  8: cipher_cupti.c:L134-159 — extract gridDim/blockDim/stream from CB params (3 launch CBIDs)
step  9: cipher_cupti.c:L160 — cipher_rt_pr_observe_stream(stream) [Wave 2 partition_router L264]
step 10: cipher_cupti.c:L161 — cipher_rt_smp_observe(grid,block,stream) [sm_packer L41]
step 11: cipher_cupti.c:L163-164 — atomic_inc g_launches_total / g_grid_ops_total
step 12: cipher_cupti.c:L166-183 — classify launch by stream's green ctx (5-bucket diag)
step 13: [OBS: classify, NEW INSERTION POINT for v1.5+]
step 13a:  optional: new classifier observer here for non-cuBLAS launches
step 13b:  cipher::classify_launch reads grid/block/smem, fn pointer
step 13c:  cipher_rt_kernel_table.cpp::cipher_kt_observe(fn,grid,block,smem) [Wave 1 L222]
step 13d:    if first observation: resolve_name via cuFuncGetName (or dladdr); classify by strstr/name
step 13e:  cipher_rt_classify cache lookup (Wave 1 cipher_classify.hpp:198) — cache slot = (fn & 511)
step 13f:  publish op_class hint to TLS for downstream actuators (no actuator on this path yet)
step 14: cipher_cupti.c:L185-188 — if (launch_count % 256) != 0: return
step 15: cipher_cupti.c:L193-205 — build cipher_launch_stats; ioctl(g_cipher_fd, CIPHER_SUBMIT_LAUNCH_STATS, &ls) [Cc.2; nr 7]
step 16: kmod cipher_dev.c:cipher_dev_submit_launch_stats(L225) — anti-spoof; cipher_pid_get_or_create; atomic_inc

(in parallel, but driven by the SAME launch event, kprobe pre-handler fires on nvidia_unlocked_ioctl)
step  K1: cipher_probe.c:cipher_kprobe_pre(L78) — fires on every ioctl through /dev/nvidia*
step  K2: cipher_ioctl_decode.c:cipher_decode_nv_ioctl_slot(L89) — binary search → slot 0..23 or -1
step  K3: cipher_probe.c:L91-95 — atomic_inc global cmd_counts[slot] + TOTAL_SLOT
step  K4: cipher_probe.c:L97-107 — cipher_pid_get_or_create + atomic_inc per-pid counter + get_task_comm

(in parallel, every Stage-0 kernel launch ALSO writes a CipherRingEntry — Wave 4 lane)
step  R1: cipher_intercept_cudart.cpp::cuLaunchKernel/Ex shim writes a 128-byte CipherRingEntry to the SPMC ring
step  R2: cipher_10ops_impl.cpp::stage1_shadow(L509-560) consumes; calls 13 observers (Wave 4 §observer DSL)
step  R3: cipher_sense.cpp::cipher_sense_observe (L171)        — session boundaries
step  R4: cipher_predict.cpp::cipher_predict_observe (L160)    — shape preload
step  R5: cipher_determinism.cpp::cipher_determinism_observe   — order fingerprint
step  R6: ... (10 more Wave-4 observers, total cost ~800 ns/event sum)
```

**Hot-path budget summary** (per kernel launch):
- step 3-7 (CUPTI enforcement gate): ~30 ns when green ctx already current; ~3 µs on
  first launch (cuCtxSetCurrent cost).
- step 8-11 (extract + observe + counter): ~50 ns.
- step 12 (5-bucket classify): ~30 ns.
- step 13a-f (new classifier observer if enabled): ~100-200 ns.
- step 15 (ioctl every 256): ~5 µs amortized to ~20 ns/launch.
- K1-K4 (kprobe path): ~200 ns per ioctl (fires on driver ioctls, not every launch).
- R1-R6 (Stage-1 ring): ~30 ns/launch producer-side (one ring write); ~800 ns/event
  consumed off-hot-path on the shadow thread.

**Sum (today, no new classifier):** ~110-150 ns added per launch on the producer.
**Sum (with new classifier at step 13):** ~220-350 ns added per launch on the producer.

---

## 5.2.4 — `cuMemMap` / VMM (weight-arena, KV-dedup paths)

KV-cache and weight-sharing paths use CUDA Virtual Memory Management. The
classifier brain does not sit on these paths — they're memory-management,
not kernel-launch. The fusion contracts here are between actuator (Wave 2)
and kmod (Wave 3).

### 5.2.4-a — KV slab allocation path

```
step  1: vLLM / Python caller cipher_kv_bridge.vmm_zeros(shape, ...) [cipher_kv_bridge.cpp:L50]
step  2: cipher_kv_bridge.cpp:L50-81 — build cipher_rt_kv_page_tag{tenant,seq,layer,head_kv=0xFFFF,role}
step  3: cipher_rt_kv_alloc.c:cipher_rt_kv_slab_create(L200) — under g.mu
step  4: cipher_rt_kv_alloc.c:L213-230 — first-fit walk for `want` consecutive FREE pages
step  5: cipher_rt_kv_alloc.c:map_pages → map_one(L145) — cuMemCreate (POSIX_FD), cuMemMap, cuMemSetAccess RW
step  6: cipher_rt_kv_alloc.c:L172 — cuMemsetD8(va, 0, page_size) [A3 invariant: cross-tenant zero on MISS]
step  7: torch::from_blob(devptr, shape, deleter=slab_free) — tensor wraps CIPHER memory
```

No kmod ioctl on this path. The slab is in-process; cross-tenant access is
not exposed.

### 5.2.4-b — KV-dedup put path (cross-process, cuIpc)

```
step  1: vLLM caller cipher_rt_kv_dedup_put(content, &devptr) [cipher_rt_kv_alloc.c:L522]
step  2: cipher_rt_kv_alloc.c:L529 — h = xxh64(content, 2MiB) [userspace xxhash64]
step  3: cipher_rt_kv_alloc.c:carve_free_page → map_one(0) — exportable POSIX_FD
step  4: cipher_rt_kv_alloc.c:L549 — cuMemcpyHtoD(va, content, page_size)
step  5: cipher_rt_kv_alloc.c:L555 — cuMemExportToShareableHandle(&export_fd, handle, POSIX_FD, 0)
step  6: cipher_rt_kv_alloc.c:L563 — ioctl(d.kvd_fd, CIPHER_KVDEDUP_PUT, &p) [Cc.9; kvdedup magic 'K' nr 2]
step  7: cipher_kvdedup.c:kvd_ioctl_put(L152) under kvd_lock — fget(export_fd); xa_load by content_hash
step  8: HIT path: cipher_kvdedup.c:L179-194 — get_unused_fd_flags(O_CLOEXEC), get_file(e->handle_file), fd_install(newfd, ...)
step  9: HIT path return: kmod returns candidate_fd; userspace does memcmp-verify [cipher_rt_kv_alloc.c:dedup_verify]
step 10:   if memcmp OK: cipher_rt_kv_alloc.c:L590 — release own page; map_one(shared); ioctl(CIPHER_KVDEDUP_CONFIRM, &cf) [Cc.9 nr 3]
step 11:     cipher_kvdedup.c:kvd_ioctl_confirm(L246) — kvd_track_add (anti-spoof); refcount++
step 12:   else: cipher_rt_kv_alloc.c — re-issue ioctl PUT with FORCE_NEW flag
step 13: MISS path: cipher_kvdedup.c:L194-235 — kzalloc kvd_entry; xa_alloc pool_offset; track_add; refcount=1
```

### 5.2.4-c — Weight arena export/import path (Track 2 cross-tenant)

```
step  1: producer Python cipher_rt_weight_arena_create(tenant_id, bytes, &base) [cipher_rt_kv_alloc.c:L714]
step  2: cipher_rt_kv_alloc.c:L724-770 — cuMemCreate exportable, cuMemAddressReserve, cuMemMap, cuMemSetAccess RW, cuMemsetD8(0)
step  3: producer cipher_rt_weight_arena_export(base, &fd) [L781] — cuMemExportToShareableHandle, POSIX_FD
step  4: producer ioctl(CIPHER_ARENA_REGISTER, {fd, base, size, blob_len, blob}) [Cc.10; nr 21]
step  5: cipher_weight_arena.c:cipher_wa_ioctl_register(L157) — fget(fd); allocate arena slot; arena_id = next_id
step  6: consumer ioctl(CIPHER_ARENA_IMPORT, {arena_id, want_base, ...}) [nr 22]
step  7: cipher_weight_arena.c:cipher_wa_ioctl_import(L220) — get_unused_fd_flags; copy_to_user blob; commit fd_install; get_file(fdfile)
step  8: consumer cipher_rt_weight_arena_import(fd, want_base, bytes, &base) [L842]
step  9: cipher_rt_kv_alloc.c:L848-906 — cuMemImportFromShareableHandle, cuMemAddressReserve(want_base), cuMemMap, cuMemSetAccess READ-ONLY
```

**Hot-path budget summary** (per memory-management op):
- KV slab create: dominated by cuMemCreate+cuMemMap ≈ 10-50 µs (one-time per page).
- KV dedup PUT MISS: ~50 µs (cuMemcpyHtoD 2 MiB + ioctl + xa_alloc).
- KV dedup PUT HIT: ~60 µs (extra fget + fd_install + memcmp 2 MiB).
- Weight arena REGISTER: ~5 µs (fget + arena slot claim).
- Weight arena IMPORT: ~5 µs (get_unused_fd + fd_install + memcpy blob).

These paths are NOT on the per-launch hot path — they fire once per
slab/arena/dedup operation.

---

# 5.3 — LOSSLESS FUSION POINTS

Meeting points where the two logics compose without loss. Each row cites the
contract ID from 5.1 and explains why the composition is lossless.

| Contract | Endpoints                                            | Why lossless                                                                                                                                                                                       |
|----------|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ca.1     | new pre-launch hook → `cipher_rt_classify_pre_launch`  | Purely additive. Brain returns PASSTHROUGH on any internal failure — never mutates substrate state, never blocks the launch. Wave 1 §FUSION POINTS confirms the contract is self-contained.       |
| Ca.4     | brain weak `cipher_get_edmd_pipeline` → NULL on Tree B | Weak symbol resolves NULL when EDMD module absent. Wave 1 brain has a NULL-check at `cipher_dispatch.cpp:273-277`; falls through to standard relaunch path. Behaviourally identical to Tree A pre-Koopman. |
| Ca.5     | brain weak `cipher_edmd_live_collect` → no-op on Tree B | Same as Ca.4. The post-relaunch hook becomes a no-op. Loss is "we don't grow the registry from observation" — but Tree A doesn't grow it either (Goal-4 deferred).                                  |
| Ca.6     | `cipher_oracle_init(nullptr_liquid)`                  | Wave 1 §LOGIC-AS-CODED confirms null-safe at 5 sites (L197, L248, L374, L396, L408). All `cipher_liquid_record_*` calls become no-ops. Oracle's own counters and EMA still update.                |
| Ca.7     | classifier observer at priority 5 on matmul registry | Observer returns PASSTHROUGH only (never HANDLED, never ERROR). Cannot disrupt the chain. Wave 2 §FUSION POINTS confirms AUDIT (also priority 0) does this. Pattern reusable.                       |
| Ca.11    | `cipher::classify_launch` header-only port           | Wave 1 §FUSION POINTS confirms PORT-AS-IS — zero deps, lock-free relaxed atomic cache, signal-safe. Tree B simply `#include`s it.                                                                  |
| Ca.12    | wire `cipher_kt_name_for(fn)` into `oq.kernel_name`  | Activates a currently-dormant Gate-5 in structural_lookup. The lookup is null-safe (Wave 1 confirmed); wiring only widens, never narrows, the safety net.                                          |
| Cb.1     | `cipher_dev.c::cipher_dev_get_tenant_snapshot` (nr 8) | Already-shipped read-only API. Brain consuming it adds zero kmod load. Per-CPU TLS snapshot in RCU section (Wave 3 `cipher_tenant_snapshot.c:44-108`).                                              |
| Cb.5     | brain hook in 1 kHz state_updater                   | Best-effort kthread (Wave 3 `cipher_main.c:81-86` tolerates init failure). Brain's per-tenant derivation is additive WRITE_ONCE on reserved-tail fields. No invariants threatened.                  |
| Cc.1-Cc.11 | every Wave 2 → Wave 3 ioctl                          | Already shipped and stable. The fusion port leaves them unchanged. ABI rule [[cipher-abi-rule]] guarantees ABI stability. Wave 3 §Specific anchors verifies all 9 ioctl families.                   |
| Cd.1     | SHIELD `cipher_sm_set_priority` band write           | Userspace-only table write; idempotent one-shot per HUMAN_INTERACTIVE session. Wave 4 §CONTRIBUTION confirms this is the only Wave-4 observer with active actuation today; ports cleanly.           |
| Cd.3     | THERMOSTAT reads `g_cipher.liquid.device->hw.gpu_temp_c` | Falls back to 0.0f if liquid uninit (Wave 4 L115). Drift-only mode is still functional; thermal gate just doesn't fire. Lossless degradation.                                                       |
| Cd.4     | PULSE RTLD_NOLOAD NVML                              | If NVML not loaded by Stage 11 first, ECC signal is permanently disabled (score ceiling 1). Drift signal still works. Wave 4 §LOGIC-AS-CODED documents this as honest degradation, not a bug.       |
| Cd.6     | observers reading `cipher_sense_current_session()` TLS | Pure read; SENSE-disabled returns session==0 which observers handle as a gate (silent return). Wave 4 §IDIOM-4 confirms 6 observers use this.                                                       |
| Cd.7     | 13 Stage-1 observers consume CipherRingEntry         | SPMC ring is back-pressure-safe (`cipher_10ops.h` L72); slow observer causes producer to drop, not crash. Per-event cost ~800 ns/event sum on the shadow thread, off the GPU hot path.              |
| Cd.8     | substitute-lane actuator → audit `record(SUBSTITUTE)` | Audit API already exported (Wave 2 §audit.h L34-39). No code change needed on the receiver side. Caller-side SHIM is one new call per actuator's HANDLED path.                                       |
| Ce.1     | `cipher_pid_table` RCU discipline                    | Wave 3 §CONCURRENCY confirms readers `hash_for_each_*_rcu`; writers under `cipher_pid_insert_lock`. Fusion port adds no new writers without lock. Discipline preserved.                              |
| Ce.2     | `cipher_do_exit_pre` two-step reaper                 | Wave 3 §FUSION POINTS confirms the extension pattern (one more leg before the CP 5.4 leg) is supported. Adding a brain leg follows the same discipline (lock-free, atomic-context).                |
| Ce.3     | exit-order invariant `cipher_cp54_sched_exit` after `cipher_probe_exit` | Hard invariant documented in source (`cipher_main.c:110-112`). Fusion port preserves the ordering; any added subsystem must declare its do_exit dependencies.                                       |
| Ce.5     | `cipher_cp54_lock` mutex vs lock-free reaper         | Discipline "each PID writes only its own metadata slot" survives all proposed additions (Cb.3 ioctl is per-pid; Cb.5 state_updater hook writes single fields).                                       |
| Ce.7     | `kvd_lock` + release-fop reclamation                 | Wave 3 §Concurrency under do_exit confirms zero refcount leak in HIT-without-CONFIRM crash window. Fusion ports the API unchanged.                                                                  |

**Summary: 22 lossless points.** The vast majority of the substrate-to-substrate
contracts compose without loss. The losses concentrate at five specific
locations (covered in 5.4).

---

# 5.4 — LOSSY FUSION POINTS (and mitigations)

Meeting points where information or efficiency may be lost. Each row: contract
ID, the loss quantified or qualified, mitigation, v1/v1.5/v2 scoping, severity.

Severity tags: **blocking** / **requires-mitigation** / **minor**.

---

## 5.4.1 — Lossy points table

| Loss-Point | Contracts | What is lost                                                                                                                       | Quantification                                                                                                                                                      | Mitigation                                                                                                                                                                                                                                                         | v1 / v1.5 / v2 | Severity              |
|------------|-----------|------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|-----------------------|
| LP-1       | Ca.2, Ca.3 | The Tree A weak-symbol re-entry contract (`tls_get_gemm_shape`, `tls_relaunch`) doesn't fit Tree B's substrate-actuator pattern. | Loss = the "relaunch-by-classifier" lane Tree A uses to run a kernel under its control. Quantitative: zero today because the registry is empty (`cipher_dispatch.cpp:480`). | Replace with "classifier returns SUBSTITUTE on TLS hint; chain continues; Marlin (or future actuator) at priority > 5 takes over." Wave 1 §FUSION POINTS proposed this. The matmul substrate's existing chain pattern IS the relaunch mechanism.                                                                                                            | v1             | requires-mitigation   |
| LP-2       | Ca.8     | **F2-X HANDLED-DISCARD on attn substrate** (`cipher_rt_attn_dispatch.cpp:321/359/398`). Each trampoline calls `orig(...)` unconditionally — substitute lane unreachable on attn. | Loss = 0% of current behaviour (no SUBSTITUTE actuator on attn today). Loss in v1.5+ when Op-3 wires attn substitute = 100% of intended attn substitutions.       | Trampoline refactor: branch on `route()` return — when HANDLED and `out_status` carries a substitute-devptr, return it instead of `orig(...)`. Roughly 6 lines of code per trampoline (3 trampolines = ~20 LOC). Must add the substitute-devptr field to `cipher_rt_attn_call` per Wave 2 § "Substitution requires REFACTOR". Land in Week 2 before any attn substitute actuator. | v1.5           | **blocking** for attn substitute lane (minor for v1 since no actuator) |
| LP-3       | (F2-ERR-SPLIT) | matmul ERROR breaks the registry walk → fallthrough; attn ERROR breaks only the switch case → next actuator. Unified observer must know which substrate. | Loss = potential mis-attribution. If observer returns ERROR (which it shouldn't) the substrates behave differently. | Discipline: classifier observer never returns ERROR (Ca.7/Ca.8). Document in `cipher_rt_classify_observer.c` header. Add a static_assert that the observer's maybe_handle never returns ERROR — compile-time enforced via the function signature `static_assert(/*returns enum value, not bool*/)`. | v1             | requires-mitigation (mitigated by discipline; no code change beyond doc) |
| LP-4       | F2-AUDIT-LOCK | Single global mutex on `cipher_rt_audit.c::g.mu` serialises HMAC-SHA256 record on hot path. | At 16 tenants × ~10K launches/s/tenant = 160K record()/s. ~50 ns HMAC × 160K = 8 ms/s wall. With mutex contention, worst-case ~50% utilization of one core. **AUDIT is env-gated OFF by default; today the cost is 0.** | v1.5: shard the chain into per-tenant chains; the chain head is still globally serializable for end-to-end audit. Per-tenant chains converge at report time, not on hot path. **Defer until CIPHER_AUDIT is the substitute-decision recorder (Cd.8).** | v1.5           | minor today; requires-mitigation when AUDIT becomes substitute recorder |
| LP-5       | Cd.2 / F4-HOT-PATH | `cipher_fairness_shm.cpp` 64-slot acquire-load walk on every cublasGemmEx. **~1.4 µs / GEMM.** | At cublasGemmEx rate ~10K/s/tenant, ~14 ms/s of CPU per tenant — non-trivial. Today the cost is paid because the env enables it for multi-tenant runs. | Migrate to kmod: proposed new ioctls `nr 25 CIPHER_FAIRNESS_REGISTER`, `nr 26 CIPHER_FAIRNESS_RECORD_GEMM`, `nr 27 CIPHER_FAIRNESS_SHOULD_YIELD`. One ioctl call vs 64 atomic loads. Wave 4 §FUSION POINTS proposed this explicitly. | v1.5           | requires-mitigation   |
| LP-6       | Ca.10 / Wave-1 drift | Brain hardcodes `total_layers=80` (Llama-3-70B). Real Mistral-7B is 32 layers; `last-3-layers` rule never fires.            | Loss = silent safety gap on inference < 80-layer models. Wave 1 §LOGIC-AS-CODED quantifies: `last-3-layers` gate only fires for layer_idx in {77,78,79}, never reached on 32-layer model. | Replace constant with env-read `CIPHER_TOTAL_LAYERS` default 32. Add a runtime detection later (when classifier reads model fingerprint from snapshot reserved-tail per Cb.2). | v1             | minor (already in Wave 1 fusion summary) |
| LP-7       | F1-STRUCT-COLLISION | `struct CipherKernelEntry` defined with different fields in two headers (Wave 1 F1-STRUCT-COLLISION). | Loss = compile failure for any TU that includes both headers. Today resolved by including only one per TU. | Rename one struct (e.g. `CipherKtEntry` in `cipher_kernel_table.h`; `CipherParamEntry` in `cipher_param_recovery.h`). One-pass `sed` across both headers and all callers. | v1 (Week 1)    | **blocking compile** |
| LP-8       | F3-ALLOC-RETIRE | `cipher_partition_allocator.c` is REFACTOR-RETIRE: userspace `nr 9` dead, but `state_updater::tick` (5 s rebalance) and `do_exit::release_slots_only` live. Confusion risk. | Loss = code-base bloat + maintenance burden. Performance loss = zero (the tick gates internally to 5 s; the do_exit release is lock-free). | Retire cleanly in Week 4: remove `cipher_dev_request_sm_partition` and the entire slot array; keep `cipher_pid_stats::sm_partition_mask/count` fields (read by ioctl nr 8 snapshot); move the legacy tick's idle-reclaim semantics into `cipher_state_updater.c` directly. | v1 (Week 4)    | minor (cleanup) |
| LP-9       | F4-HIBERNATE | HIBERNATE NVML write probe returns NOT_SUPPORTED on the pod → DEGRADED. | Loss = idle SM power gating doesn't actuate today. ~0 W saved on idle gaps. | Add kmod-clock fallback via `CIPHER_SET_CLOCK_MHZ` (Wave 3 nr 10). On idle: lock clock to 210 MHz (Wave 2 volt.c min); on wake: restore base. Reuses VOLT's signal handlers. | v1.5           | minor |
| LP-10      | F4-HIBERNATE pre-engage | HIBERNATE pre-engage NVML restore runs synchronously on the *producer thread* of the first event after idle. | NVML call cost: ~tens of µs on first event. Pre-engage by design must happen before the GPU launch; producer-side placement is correct for correctness but adds tail latency. | If v1.5 splits Stage 0/1 across cores, move pre-engage to Stage-1 reader thread. Document the constraint in Wave 5.5 Week 4. | v1.5           | minor |
| LP-11      | Brain confidence < min_confidence (NOT confidence < 1.0) | Wave 1 brain produces **binary** confidence (40 for ITERATIVE_CUSTOM, 85 for others). With min_confidence=60 default, every ITERATIVE_CUSTOM exits early (`cipher_dispatch.cpp:444`). | Loss = ITERATIVE_CUSTOM kernels NEVER reach the actuator chain. Quantitative: a few % of total launches (most custom kernels are RoPE / RMSNorm / Triton). | The fallback IS the existing PASS_THROUGH — already lossless against any substitution. The "loss" is opportunity cost (no observer fires either). Mitigation: add an env override `CIPHER_OBSERVE_LOW_CONFIDENCE=1` that still publishes the TLS hint for observer-only consumption, but never engages SUBSTITUTE. | v1             | minor (fallback exists) |
| LP-12      | Multiple observers update the same tenant state                  | Wave 4 atomic flags `oracle_aggressive`, `cache_aggressive`, `sustain_compress`, `aggressive`, `idle_gate_flag` each have **one writer / many readers**. No conflicting field writes. | Loss = potentially stale read by a downstream actuator that reads multiple flags in different orders. Quantitative: zero functional loss; possibly one extra/missed decision per transition. | Per-field memory_order_release on writer, memory_order_acquire on reader. Already done in Wave 4 code (verified `cipher_shield.cpp` L1755-1756, similar for others). Document as a contract in Wave 5.1.d table. No code change. | v1             | minor (already mitigated by single-writer per field) |
| LP-13      | Classifier crashes mid-kernel                                    | What does the substrate do?                                                                                                                                          | Substrate is GOT-patched and independent. `cipher_rt_cublasGemmEx_impl` still routes through `matmul_dispatch`. Actuators gate independently from `cipher_rt_matmul_call`. Classifier crash = telemetry loss, not correctness loss. | Defensive: wrap the classifier observer's `maybe_handle` in a try-catch (C++) or sigsetjmp guard (C). Or, simpler: ensure all classifier code is signal-safe and panic-free by construction (Wave 1 `cipher_classify.hpp` IS signal-safe; oracle is mostly signal-safe except getenv/fprintf which only fire at init). | v1             | minor (fail-safe by design) |
| LP-14      | Hot-path latency budget across the full pipeline (sum of 5.2.1)  | Per cublasGemmEx (audit off, Marlin not engaged): ~200-250 ns added.   Per SDPA call: ~270 ns added.   Per generic cuLaunchKernel via CUPTI: ~110-150 ns today; ~220-350 ns with new classifier observer at step 13. | At 10K cublasGemmEx/s per tenant × 16 tenants = 160K/s × 250 ns = **~4% CPU overhead on one core** for the classifier hot path alone. Fairness_shm adds another ~22% on top. Real-world tested cost: TBD — Section 7 Week 2 R-W2.1 budget says 12 ns, which is **unsupportable** per Wave 1 measurements. | Honest budget for v1: 100-200 ns hot path per launch. The R-C1 risk register should be rewritten: **accept 1-2% per-tenant overhead as the primary case**, not the fallback. Profile in Week 2 with cycle counters; if > 500 ns/launch, revisit. | v1             | requires-mitigation (acknowledge honest budget; not block) |
| LP-15      | Memory ordering between userspace classifier state and kmod-resident scheduler state | The classifier writes per-tenant decisions (Cb.3/Cb.5) into the snapshot reserved-tail; the kmod scheduler (`cipher_cp54_sched.c`) reads them.       | Loss = race window between classifier deciding "tenant X needs N more SMs" and CP54 ALLOCATE. Quantitative: ~1 kHz state_updater cadence → ≤ 1 ms latency for the decision to propagate. | Use `WRITE_ONCE` on writer (state_updater kthread), `READ_ONCE` on reader (kmod ALLOCATE handler). Wave 3 §Concurrency confirms this pattern is already used (`cipher_internal.h` L103-165). Decision is advisory anyway — kmod has final authority over SM groups. | v1             | minor (existing pattern) |
| LP-16      | Failure paths under partial integration                          | What if classifier shipped but Cb.5 state_updater hook is not yet wired? | Snapshot reserved-tail stays zero; classifier reads zeros; uses its own defaults (Wave 1 cipher_dispatch.cpp env-driven defaults). Behavioural equivalent to "classifier disabled". | Document the dependency in Week 1 (struct fields exist; Week 2 wires the writer; Week 3 wires the reader). Each step is independently shippable because each tolerates the others being absent. | v1             | minor |

---

## 5.4.2 — Severity rollup

| Severity            | Count | Loss-Points |
|---------------------|-------|-------------|
| blocking            | 2     | LP-2 (attn HANDLED discard, blocking only for attn substitute lane), LP-7 (struct collision blocks compile) |
| requires-mitigation | 5     | LP-1 (relaunch contract), LP-3 (ERROR split discipline), LP-5 (fairness_shm 1.4 µs), LP-14 (hot-path budget honesty), LP-4 (audit lock, lazy) |
| minor               | 9     | LP-6 (80-layer drift), LP-8 (allocator retire), LP-9 (HIBERNATE NVML), LP-10 (HIBERNATE pre-engage), LP-11 (low-confidence ITERATIVE), LP-12 (multi-observer ordering), LP-13 (classifier crash), LP-15 (memory ordering), LP-16 (partial integration) |
| **total**           | **16** | |

The two blocking points are both Week-1-2 work and have well-defined fixes
(rename one struct; refactor 3 attn trampolines). The five requires-mitigation
points are all serviced by either v1.5 work or documentation/discipline. None
of the lossy points blocks v1 shipping of the classifier observer.

---

# 5.5 — INTEGRATION SEQUENCE WITH LOSSLESS VERIFICATION GATES

Verification or correction of `CIPHER_REENGINEERING_PLAN.md` Section 7
(5-week sequence). The plan's structure is sound; below I flag five
substantive holes (verbatim from advisor cross-check) and one cleanup:

**Holes in the original Section 7:**

1. **R-C1 "12 ns budget" is unsupportable.** Wave 1 measurement evidence
   (cipher_dispatch.cpp header claims "~160 ns cache-hit"; Wave 1 control
   flow analysis estimates 150-200 ns) puts the realistic budget at
   100-200 ns, not 12 ns. The honest budget is "accept 1-2 %
   per-tenant overhead as the primary case." Section 7 R-C1's "if
   30 ns realistic, accept 1-2 %" inverts this — it's the primary case.
2. **Week 1 missing the `CipherKernelEntry` struct rename (LP-7).** A
   *compile blocker* for any TU that pulls both headers. Must be in Week 1.
3. **Week 3 unprotected by the attn HANDLED-discard fix (LP-2).** Week 3
   says "Classifier output drives actuator selection" but for SDPA that's
   structurally a no-op until the trampoline refactor lands. Either pull
   the refactor into Week 2 or scope Week 3 to GEMM only.
4. **Snapshot reserved-tail extension missing.** Wave 3 confirms
   `cipher_ioctl.h:178 __u32 reserved[16]` exists for additive classifier
   fields without new `nr`s. No week of Section 7 actually uses it; belongs
   in Week 1's ABI changes alongside the optional new `nr 25`.
5. **`cipher_partition_allocator.c` retirement absent (LP-8).** Wave 3
   said REFACTOR-RETIRE not VESTIGIAL — live in-kmod callers exist. Should
   be a Week 4 cleanup step.

**Corrected sequence below.** Same week boundaries as Section 7. Each
step has: goal, files moved, lossless invariant to hold, the TEST that
verifies, and the ROLLBACK path.

---

## Week 1 — Compile-level classifier port + struct collision fix

**Goal.** Classifier brain compiles into Tree B; no firing yet.

**Files ported (per Section 7 L1017-1024):**
- `cipher-may13-evidence/cipher_dispatch.cpp` (TOP-LEVEL) → `cipher_rt_phase4/cipher_rt_dispatch.cpp`
- `cipher-may13-evidence/cipher_oracle.cpp` (TOP-LEVEL) → `cipher_rt_phase4/cipher_rt_oracle.cpp`
- `cipher-may13-evidence/include/cipher_classify.hpp` → `cipher_rt_phase4/cipher_rt_classify.h`
- `cipher-may13-evidence/src/cipher_recipes.cpp` → `cipher_rt_phase4/cipher_rt_recipes.cpp`
- `cipher-may13-evidence/src/cipher_sense.cpp` → `cipher_rt_phase4/cipher_rt_sense.cpp`
- `cipher-may13-evidence/src/cipher_structural_lookup.cpp` → `cipher_rt_phase4/cipher_rt_structural_lookup.cpp`
- `cipher-may13-evidence/src/cipher_kernel_table.cpp` → `cipher_rt_phase4/cipher_rt_kernel_table.cpp` (drop the libcipher_hook copy)
- five header ports (recipes/classify/oracle/sense/structural_lookup) into `cipher_rt_phase4/`

**New file:** `cipher_rt_phase4/cipher_rt_classify_substrate.cpp` — mirrors
`cipher_rt_matmul_dispatch.{c,h}` shape, exports
`cipher_rt_classify_register`, `cipher_rt_classify_dispatch`, no actuator
chain yet (just the substrate header).

**New file:** `cipher_rt_phase4/cipher_rt_classify_observer.c` — empty stub;
will register against matmul/attn registries in Week 2.

**Struct collision fix (LP-7) — REQUIRED first.** Rename to break the cycle
before any port touches both headers:
- `include/cipher_kernel_table.h::struct CipherKernelEntry` →
  `struct CipherKtEntry` (rename all 9 callers — Wave 1 §DEPENDENCIES
  identifies them).
- `include/cipher_param_recovery.h::struct CipherKernelEntry` →
  `struct CipherParamEntry` (rename all 6 callers — Wave 1 §param_recovery).

**Snapshot reserved-tail bump (Cb.2).** Add classifier-side fields to
`cipher_internal.h::struct cipher_pid_stats` AND the userspace mirror
`cipher_ioctl.h::struct cipher_tenant_snapshot_user`, all in the reserved
tail. Fields (proposed):
- `__u32 recommended_sm_count;` (consumed by Cc.7 ALLOCATE)
- `__u32 slo_priority;` (consumed by partition_router)
- `__u8 session_band;` (HUMAN=0 / AGENT=1 / BATCH=2 / UNKNOWN=3)
- `__u8 tenant_billing_class;` (corp=0 / free=1 / research=2)
- `__u8 reserved_pad[2];`
- Remaining: `__u32 reserved[12];` (was 16; now 12).

**Makefile changes (Section 7 L1033):** Add 9 new objects to `OBJS`
(7 ports + 1 new substrate + 1 new observer stub).

**LOSSLESS INVARIANTS that must hold post-Week-1.**
- I-W1.1 (compile gate): Both renamed headers can be `#include`d in the
  same TU. `make clean && make` succeeds for `libcipher_rt.so` and
  `cipher_kmod.ko`.
- I-W1.2 (size invariant): `sizeof(struct cipher_tenant_snapshot_user)`
  unchanged (still 336 B target; the new 8 B of fields fit in the reserved
  tail).
- I-W1.3 (no observable behaviour): `LD_PRELOAD=$(deployed runtime) python
  test_w1.py` produces byte-identical Mistral-7B forward pass vs the
  pre-port anchor.

**TEST.**
- T-W1.1: `cd cipher_rt_phase4 && make clean && make 2>&1 | tee build.log; grep -c "error" build.log` → 0.
- T-W1.2: `gcc -E cipher_rt_phase4/cipher_rt_classify_observer.c | grep -c CipherKtEntry` and same for CipherParamEntry: both > 0; no duplicate-definition error.
- T-W1.3 (size invariant): emit `cipher_tenant_snapshot_size` from a small
  test helper; assert == 336.
- T-W1.4 (W1 regression): run `tests/run_w1_regression.sh` on the rebuilt
  runtime; tok/s within ±3 % of pre-port; isolation 15/15 PASS;
  Track 2 SC6 PASS (bit-identical forward).

**ROLLBACK if test fails.**
- Revert the Makefile additions (`git checkout cipher_rt_phase4/Makefile`).
- Revert the struct renames (`git checkout cipher_rt_phase4/*.h
  cipher_rt_phase4/*.cpp`).
- Restore the previous `libcipher_rt.so` from the preserved `.pre_week1`
  copy. Wave 1's existing fallback `libcipher_rt.so.may13_pre`
  (preserved by [[cipher-kbuild-clean-wipes-ko]] discipline) is the
  belt-and-suspenders.

---

## Week 2 — Hot-path classifier wiring + LP-2 attn trampoline refactor

**Goal.** Brain's classify + oracle + structural_lookup fire on every
launch (matmul + attn substrates and CUPTI generic launches). Telemetry
only — no actuator routing change. **Plus the LP-2 attn trampoline
refactor.**

**Hook points:**
- Matmul: register classifier observer at priority 5 (Ca.7) by appending
  to `cipher_inject.c::cipher_v2_init_body` between
  `cipher_rt_matmul_dispatch_init` (step 7) and `cipher_rt_marlin_init`
  (step 8).
- Attn: same pattern, between `cipher_rt_attn_dispatch_init` (step 9)
  and `cipher_rt_attn_test_actuator_init` (step 10).
- CUPTI generic: add the brain hook inside `cipher_cupti.c::cipher_v2_cupti_cb`
  at the position labelled "step 13" in §5.2.3 above. Single line:
  `cipher_rt_classify_observe_launch(fn, grid, block, smem);`.

**LP-2 attn trampoline refactor (REQUIRED to unblock Week 3 attn substitute):**
- Modify `cipher_rt_attn_dispatch.cpp::flash_call`, `eff_call`, `cudnn_call`
  (L289, L324, L362) to branch on `route()`'s return value. New control
  flow:
  ```
  cipher_rt_attn_result r = route(c);
  if (r == CIPHER_RT_ATTN_HANDLED) {
      /* actuator stored substitute result into call.out_status_devptr; return it */
      ...
      return; /* skip orig() */
  }
  return orig(q,k,v,...);
  ```
- Add field `void* out_status_devptr` to `cipher_rt_attn_call` (header
  L80+); actuators write the substitute result here when returning HANDLED.

**Brain hook bodies (per Section 7 L1051-1056 + new):**
```c
const cipher_classify_result_t cr = cipher::classify_launch(fn,grid,block,smem);
cipher_oracle_query_t q = {layer_idx, total_layers, cr.op, cr.confidence, kname, is_back, false};
cipher_oracle_result_t dec = cipher_rt_oracle_decide(&g_oracle, &q);
/* Publish TLS hint for downstream actuators; do NOT mutate substrate state */
cipher_rt_classify_tls_set(cr.op, cr.confidence, dec);
return CIPHER_RT_<MATMUL|ATTN>_PASSTHROUGH;
```

**LOSSLESS INVARIANTS.**
- I-W2.1: W1 regression remains PASS (no perf change beyond ±3 %).
- I-W2.2: `/proc/cipher/classify_stats` (new node from Week 1) reports
  non-zero classification counts per kernel launch.
- I-W2.3: Oracle EMA stays in `STEADY` after warmup (Wave 1 oracle
  state).
- I-W2.4: CLASSIFY cache hit rate ≥ 95 % after 10 s steady-state.
- I-W2.5 (LP-2 invariant): attn flash::call refactor preserves
  PASSTHROUGH semantic when `route()` returns PASSTHROUGH (i.e., today's
  behaviour). Bit-identical attention output verified by Track 2 SC6
  test pre/post.

**TEST.**
- T-W2.1: `tests/run_w1_regression.sh` PASS.
- T-W2.2: `cat /proc/cipher/classify_stats | awk '$2 > 0'` → at least
  one non-zero op_class row.
- T-W2.3: classifier cycle-counter probe (insert RDTSC around the
  observer body in a debug build); 99th percentile < 500 ns per
  cublasGemmEx (LP-14 budget).
- T-W2.4 (LP-2): Track 2 SC6 bit-identical forward pass on Mistral-7B
  + Llama-3-8B; SHA256 of attention output unchanged vs pre-refactor.
- T-W2.5: 16-tenant smoke (CP 5.6 P2 reproducer); isolation 15/15
  invariant PASS.

**ROLLBACK.**
- Stub the brain hook to `return PASSTHROUGH` immediately
  (env `CIPHER_CLASSIFY_LIVE=0`). Equivalent to Week 1 state.
- If LP-2 refactor fails T-W2.4: revert the trampoline refactor
  (preserved as `cipher_rt_attn_dispatch.cpp.week2_pre`). The attn
  substitute lane stays blocked until v1.5; matmul lane proceeds.

**Risk register additions to Section 7's:**
- R-W2.3 [HIGH→MEDIUM]: Per-tenant cache contention. CLASSIFY uses
  512-slot lock-free cache (Wave 1 §STATE); verify under contention
  with 33-thread test (existing harness
  `cipher_test_phase4_partition_contention`).
- R-W2.4 [NEW, MEDIUM]: LP-2 trampoline refactor breaks Track 2
  bit-identical. **Mitigation**: invariant T-W2.4 is the gate.

---

## Week 3 — Dispatch routing goes live (GEMM lane only in v1)

**Goal.** Classifier hint drives actuator selection on the matmul lane.
Per-regime routing per `CIPHER_REENGINEERING_PLAN.md` Section 4.5.

**Scope correction vs Section 7.** The original Section 7 calls for all
substrates to go live in Week 3. **Scope this to GEMM only:** the attn
lane has no substitute actuator today and the LP-2 fix is brand new
(landed Week 2). Defer the attn dispatch routing to v1.5 when the first
attn substitute actuator (Op-3 SUBSTITUTE for FAVOR+ / FlashSwiftKey)
is ready.

**Changes:**
- `cipher_rt_classify_observer.c`: maybe_handle now reads `dec` from the
  oracle and, if PERMIT and op_class==GEMM and registry hit, publishes a
  TLS `substitute_hint = recipe_id`.
- `cipher_rt_marlin_actuator.c::maybe_handle_marlin` (Ca.9): on entry,
  read TLS hint; if `substitute_hint == MARLIN_SHAPE_KNOWN`, short-circuit
  the STABILITY_THRESHOLD observation (skip 3 calls of warmup).
- VOLT engagement: trigger `CIPHER_VOLT_BATCH=1` (1000 MHz lock) when
  classifier detects sustained decode (Wave 2 §volt.c L54-63 LUT). Today
  this is env-static; Week 3 makes it dynamic via the classifier hint.
- SENSE phase transitions trigger DSM PROPOSE for tool-idle detection
  (Wave 3 `cipher_cp54_sched.c::COMPACT_MIGRATE` nr 20).

**LOSSLESS INVARIANTS.**
- I-W3.1: W1 regression PASS.
- I-W3.2 (Marlin regime preserved): WL03 prefill (Mistral-7B prefill)
  tok/s within ±3 % of pre-Week-3.
- I-W3.3 (no attn regression): SDPA on Llama-3-8B bit-identical (Track
  2 SC6 PASS); attn substrate stays PASSTHROUGH-only.
- I-W3.4: VOLT clock-lock retargeting succeeds on band change; no clock
  oscillation (DVFS rate < 10/s).

**TEST.**
- T-W3.1: Regime 1 (WL03 prefill) — Marlin engagement count up;
  MFU climbs from baseline 60 % toward 85 % target.
- T-W3.2: Regime 2 (WL01 decode, B=1) — tok/W +14 % over pre-VOLT
  baseline (per `cipher-t43-envelope`).
- T-W3.3: Regime 3 (WL05 multi-tenant) — isolation 15/15 invariant
  PASS; per-tenant fairness intact.
- T-W3.4: Regime 5 (heterogeneous mini-bench: 1 prefill + 4 decode
  concurrent) — all 5 progress; no starvation.
- T-W3.5: Mistral-7B SHA256-byte-identical forward pre/post (LP-2
  invariant carried forward).

**ROLLBACK.**
- Env `CIPHER_DISPATCH_LIVE=0` → fallback to Week 2 (TLS hint
  published but actuator does not consume).
- Anchor `libcipher_rt.so.week3_pre` preserved.

**Risk register additions:**
- R-W3.1 [HIGH]: Mis-routing (decode kernel routed to Marlin
  regresses). Mitigated by oracle Gate-2 (min_confidence) + explicit
  B<8 filter in Marlin actuator (L106).
- R-W3.2 [HIGH]: DVFS clock-lock interaction with concurrent prefill
  tenant. Must verify VOLT clock-lock doesn't degrade compute-bound
  tenant. Test in mixed regime (T-W3.4).
- R-W3.3 [MEDIUM]: Tool-call DSM thrashing. Rate-limit gate
  `cipher_cp54_mig_ratelimit_ms = 10000` (Wave 3) prevents > 1
  proposal per tenant per 10 s.

---

## Week 4 — Observability integration + `cipher_partition_allocator.c` retirement (LP-8)

**Goal.** Port the observability tier. Per-tenant billing/fairness/audit
becomes customer-facing. **Plus the LP-8 allocator retirement cleanup.**

**Files ported (Section 7 L1101-1112 verbatim, augmented):**
- AUDIT: retire `cipher-may13-evidence/src/cipher_audit.cpp` in favor of
  existing Wave 2 `cipher_rt_audit.{c,h}` (already wired at priority 0
  on both substrates).
- TRACE → `cipher_rt_trace.cpp`
- RECEIPT → `cipher_rt_receipt.cpp`
- CARBON → `cipher_rt_carbon.cpp`
- FAIRNESS (both `.cpp` and `_shm.cpp`) → `cipher_rt_fairness.cpp`
- GUARD → `cipher_rt_guard.cpp`
- COMPLY → `cipher_rt_comply.cpp`
- LOOP → `cipher_rt_loop.cpp`
- PIPELINE → `cipher_rt_pipeline.cpp`
- DETERMINISM → `cipher_rt_determinism.cpp`
- CONTINUITY → `cipher_rt_continuity.cpp`
- PULSE → `cipher_rt_pulse.cpp`
- (Defer per Section 7: PREDICT, SHIELD, SUSTAIN, THERMOSTAT — partial
  ops, v1.5.)

**Plus Stage-1 ring + stage1_shadow port.** The 65536-entry SPMC ring
in `cipher_10ops.h` and the shadow thread in `cipher_10ops_impl.cpp`
must port. Without the ring, the 13 L1 observers fan out from nothing.
Wave 4 F4-3 calls this out: "the Stage-1 ring is the load-bearing
fan-out point."

**LP-8 allocator retirement** (Wave 3 §FUSION POINTS):
- Remove `cipher_dev_request_sm_partition` handler from `cipher_dev.c`
  (it already returned -ENOSYS).
- Remove the slot array `cipher_partition_slot[32]` and all writes to
  it.
- Move the legacy `_tick()` idle-reclaim semantics (30 s idle release)
  into `cipher_state_updater.c::cipher_state_updater_fn` directly as a
  "free slot reaper" leg.
- KEEP `cipher_pid_stats::sm_partition_mask/count` fields (read by
  ioctl nr 8 snapshot); they are now written by CP 5.4 via the
  `cipher_set_sm_partition_mask` path (already exists in
  `cipher_tenant_snapshot.c:207`).

**Prometheus exporter additions (Section 7 L1115-1119 verbatim).**

**LOSSLESS INVARIANTS.**
- I-W4.1: AUDIT chain head from Week 3 reproducible on the Wave-2
  shipped AUDIT (no chain reset).
- I-W4.2: TRACE bounded buffer rotates without drops at sustained
  load (per-file `dropped` counter == 0 for 60 s @ 10 K launches/s).
- I-W4.3: RECEIPT per-session HMAC verifiable via existing
  `audit_verify.py` (Wave 2 §audit).
- I-W4.4: LP-8 cleanup leaves `cipher_pid_stats::sm_partition_mask`
  populated by CP 5.4 — ioctl nr 8 snapshot still returns valid masks.
- I-W4.5: No kmod ABI change (LP-8 only removes a deactivated handler).

**TEST.**
- T-W4.1: Per-tenant FAIRNESS quota debits visible at
  `/proc/cipher/fairness` (new entry).
- T-W4.2: AUDIT chain advances per launch (verify via
  `audit_verify.py`).
- T-W4.3: TRACE drops == 0 for 60 s @ 10 K launches/s.
- T-W4.4: RECEIPT HMAC verifies for sample session (golden HMAC test).
- T-W4.5 (LP-8 invariant): `cat /dev/cipher | strace -e ioctl
  CIPHER_REQUEST_SM_PARTITION` still returns -ENOSYS (ABI preserved);
  CP 5.4 ALLOCATE (nr 13) still succeeds.

**ROLLBACK.**
- Each observability op can be individually disabled via env var
  (`CIPHER_<OP>=0`). No anchor needed; ops are observe-only.
- LP-8 cleanup: revert the `cipher_dev.c` removal; the slot-array
  storage is BSS, so reintroducing the array is just a recompile.

**Risk register additions:**
- R-W4.1 [LOW]: Observer overhead. These are async/post-launch;
  budget < 1 µs/launch combined. Wave 4 §F4-3 measured ~800 ns/event
  for 13 observers. PASS.
- R-W4.2 [LOW]: Prometheus cardinality (Section 7 verbatim).
- R-W4.3 [NEW, MEDIUM]: LP-5 fairness_shm 1.4 µs/GEMM remains. Mark
  for v1.5 kmod migration.

---

## Week 5 — CP 5.5 100-tenant benchmark

**Goal.** Headline measurement. Section 7 L1133-1162 verbatim; the
fusion plan does not alter this week's content.

**Lossless invariants from prior weeks must STILL hold:**
- I-W1.x (compile, no behaviour change): re-verified.
- I-W2.x (telemetry-only): re-verified.
- I-W3.x (no regime regression, GEMM lane live): re-verified.
- I-W4.x (observability complete, allocator retired): re-verified.

**TEST (Section 7 verbatim).**
- 100-tenant agents/GPU progressing.
- fleet tok/W ≥ 2× vanilla vLLM 100-instance baseline.
- per-agent p99 latency distribution acceptable.
- per-agent fairness quota adherence.
- MFU per-WL attribution via FLOP telemetry.
- weight HBM saved ≥ 90 % for same-model agents (Track 2 measurement).
- KV-prefix dedup hit rate ≥ 60 % for shared system prompt.
- 24-hour soak G4 gate: no oops/WARN, taint ≤ +1.

**ROLLBACK.** If any goal fails, the unified runtime stays functional
— only the headline measurement is incomplete. Subsequent CP 5.5
attempts iterate without code rollback.

---

## Week 6+ (v1.5) — Deferred items

The following are scoped explicitly to v1.5+, with their fusion-plan
contracts already analysed but NOT shipped in v1:

| Item            | Loss-Point  | Earliest target | Reason for deferral                                                             |
|-----------------|-------------|-----------------|----------------------------------------------------------------------------------|
| Attn substitute lane (Op-3 FAVOR+ / FlashSwiftKey) | LP-2 | v1.5 | LP-2 refactor lands in Week 2, but no actuator yet builds on it. |
| fairness_shm → kmod (nrs 25/26/27)                  | LP-5 | v1.5 | Requires kmod ABI bump; not a v1 blocker (~22% CPU overhead acceptable for v1). |
| HIBERNATE kmod-clock fallback                       | LP-9 | v1.5 | NVML write-permitted hosts work today; pod is the special case. |
| Per-tenant SHIMs across all 16 observers            | LP-12 etc. | v1.5 | One-line each per observer; no v1 actuator consumes them. |
| L3.5 EDMD pipeline (Koopman discovery)              | F1-LOAD-BEARING | v2 | nvcc compile dependency; Phase 6 substrate. |
| Per-tenant chain-shard AUDIT                        | LP-4 | v1.5 (lazy) | Triggered only when AUDIT becomes substitute recorder; today AUDIT is env-gated off. |

---

# 5.6 — THE COMPLETE FUSED LOGIC MAP

ASCII reference card for the fused runtime. Files grouped by tier;
contracts labeled with the IDs from 5.1; the four kernel intercept entry
points from 5.2 are starred (*).

**Map legend.**
- `[file:func]`  — code module.
- `<NR_n>`        — ioctl nr (Wave 3 ABI surface).
- `→Cx.n→`        — directed flow with contract ID.
- `(*)`           — kernel intercept entry point.
- Boxes labeled with the owning tier/file.

---

```
                                  CIPHER FUSED RUNTIME (anchors: kmod 0.4.8 / libcipher_rt c2c5d313)
============================================================================================================================
                                                                                                                            
                              +-------------------- USERSPACE (CUDA driver context) ---------------------+                  
                              |                                                                          |                  
+-----------------+           |   +-----------------------+  +--------------------------+                |                  
|  CUDA driver    |           |   |  CLASSIFIER (Wave 1)  |  |  OBSERVER (Wave 4)       |                |                  
|  invokes:       |           |   +-----------------------+  +--------------------------+                |                  
|  InitializeInj. +---once--->|   | cipher_rt_classify.h  |  | cipher_rt_carbon.cpp     |                |                  
+-----------------+           |   | (cache 512 slots,     |  | cipher_rt_fairness.cpp   |                |                  
                              |   |  signal-safe)         |  | cipher_rt_fairness_shm.* |                |                  
+--------------------+        |   |                       |  | cipher_rt_receipt.cpp    |                |                  
| CUDA_INJECTION64_  +---load>|   | cipher_rt_dispatch.cpp|  | cipher_rt_guard.cpp      |                |                  
| PATH=libcipher_rt  |        |   | cipher_rt_oracle.cpp  |  | cipher_rt_comply.cpp     |                |                  
+--------------------+        |   | cipher_rt_recipes.cpp |  | cipher_rt_loop.cpp       |                |                  
                              |   | cipher_rt_sense.cpp   |  | cipher_rt_pipeline.cpp   |                |                  
                              |   | cipher_rt_structural_ |  | cipher_rt_trace.cpp      |                |                  
                              |   |   lookup.cpp          |  | cipher_rt_continuity.cpp |                |                  
                              |   | cipher_rt_kernel_     |  | cipher_rt_pulse.cpp      |                |                  
                              |   |   table.cpp           |  | cipher_rt_determinism.* |                |                  
                              |   |                       |  | cipher_rt_topology.cpp   |                |                  
                              |   | Wave1 hot path:       |  | cipher_rt_shield.cpp     |                |                  
                              |   | classify→liquid_      |  | cipher_rt_sustain.cpp    |                |                  
                              |   | record→infer_layer→   |  | cipher_rt_thermostat.cpp |                |                  
                              |   | oracle→struct_lookup→ |  | cipher_rt_hibernate.cpp  |                |                  
                              |   | registry→apply_recipe |  | cipher_rt_straggler.cpp  |                |                  
                              |   +-----+----------+-------+  +-----+---------+-------+--+                |                  
                              |         |          ^                |         ^       |                  |                  
                              |         |          |                |         |       |                  |                  
                              |     Ca.1/Ca.7/      |          Cd.6/Cd.7 (TLS,         |                  |                  
                              |     Ca.8/Ca.9/     Cb.2,Cb.5     SPMC ring 65536)     |                  |                  
                              |     Ca.11/Ca.12     (snapshot                          |                  |                  
                              |         |          tail RCU)                           |                  |                  
                              |         v          |                |         |       |                  |                  
                              |   +-----+----------+-+              |         |       |                  |                  
                              |   |  SUBSTRATE (Wave 2)              |         |       |                  |                  
                              |   +---------------------+            |         |       |                  |                  
                              |   |                     |            |         |       |                  |                  
                              |   |  cipher_inject.c    |            |         |       |                  |                  
                              |   |  (14-step init,     |            |         |       |                  |                  
                              |   |   pthread_once)     |            |         |       |                  |                  
                              |   |                     |            |         |       |                  |                  
   *=========================>|   |  cipher_rt_cublas_  |            |         |       |                  |                  
   * cublasGemmEx GOT-patched |   |   shim.c            |<--orig() via dlvsym  |       |                  |                  
   *  -> cipher_rt_cublas-    |   |        +cipher_rt_matmul_call    |         |       |                  |                  
   *     GemmEx_impl(*)       |   |        v                          |         |       |                  |                  
                              |   |  cipher_rt_matmul_  |             |         |       |                  |                  
                              |   |   dispatch.c        |             |         |       |                  |                  
                              |   |   (registry walk,   |             |         |       |                  |                  
                              |   |   3-value enum,     |             |         |       |                  |                  
                              |   |   break-on-ERROR)   |             |         |       |                  |                  
                              |   |        v                          |         |       |                  |                  
                              |   |  [OBS prio=5 CLASSIFY] [OBS prio=0 AUDIT] [ACT prio=10 MARLIN]         |                  
                              |   |        +-----------------v                                            |                  
                              |   |  cipher_rt_audit.c (HMAC-SHA256 chain; F2-AUDIT-LOCK; nr-25-emit)     |                  
                              |   |        |                                                              |                  
                              |   |        v                                                              |                  
                              |   |  cipher_rt_marlin_actuator.c (M<=64 K%128 N%64; STAB=4)              |                  
                              |   |        v                                                              |                  
                              |   |  cipher_rt_marlin_engine.cpp (NVRTC sm_90 10 entries;                |                  
                              |   |   PrimaryCtxGuard + event-wait for F1 fix)                            |                  
                              |   |                                                                       |                  
   *=========================>|   |  cipher_rt_attn_dispatch.cpp (3 SDPA trampolines)                    |                  
   * SDPA (flash/eff/cuDNN)   |   |   F2-X: orig() unconditional; LP-2 trampoline refactor in Week 2     |                  
   *  -> attn trampoline(*)   |   |   +-> route() snapshot-under-lock; 4-value enum (REDIRECTED)         |                  
                              |   |        +-> [OBS prio=5 CLASSIFY] [OBS prio=0 AUDIT] [OBS attn_test]  |                  
                              |   |                                                                       |                  
   *=========================>|   |  cipher_cupti.c (8 CBIDs subscribed; per-launch enforcement)         |                  
   * cuLaunchKernel CUPTI(*)  |   |   - cipher_rt_green_ctx_make_current (T4.2.4d enforcement gate)      |                  
                              |   |   - cipher_rt_pr_observe_stream (partition_router 256-slot cache)    |                  
                              |   |   - cipher_rt_smp_observe (sm_packer detection)                       |                  
                              |   |   - every 256 launches -> CIPHER_SUBMIT_LAUNCH_STATS <nr 7>          |                  
                              |   |   [NEW v1: classifier hook at step 13 per 5.2.3]                     |                  
                              |   |                                                                       |                  
                              |   |  cipher_rt_volt.c                                                     |                  
                              |   |   NVML path OR <nr 10> CIPHER_SET_CLOCK_MHZ                          |                  
                              |   |   signal-safe restore: SIGTERM/INT/SEGV/ABRT/BUS                     |                  
                              |   |                                                                       |                  
                              |   |  cipher_rt_green_ctx.c                                                |                  
                              |   |   <nr 13 ALLOCATE> <nr 16-20 migration>                              |                  
                              |   |   PARTITION / POOL / SHARED                                          |                  
                              |   |   Track 3 SC3 live SM migrate                                        |                  
                              |   |                                                                       |                  
                              |   |  cipher_tenant.c (<nr 1 REGISTER_TENANT> Cc.1)                       |                  
                              |   |  cipher_rt_tenant.cpp (<nr 8> Cc.3 — per-thread fd, 3 access modes)  |                  
                              |   |                                                                       |                  
                              |   |  cipher_rt_partition_router.c (cuStreamSetAttribute PRIORITY)        |                  
                              |   |  cipher_rt_sm_packer.c (detection-only, threshold=4096)              |                  
                              |   |  cipher_rt_attn_test_actuator.c (env-gated coverage)                 |                  
                              |   |  cipher_rt_arbitrate.c (VESTIGIAL, not in OBJS)                      |                  
                              |   |                                                                       |                  
   *=========================>|   |  cipher_rt_kv_alloc.c (VMM slab + KV dedup + weight arena)           |                  
   * cuMemMap / VMM (*)       |   |   <nr 21-24 weight arena> Cc.10                                       |                  
                              |   |   <kvdedup nr 1-5> Cc.9 (magic 'K')                                  |                  
                              |   |   cipher_kv_bridge.cpp (Python pybind11 seam)                        |                  
                              |   |                                                                       |                  
                              |   |  cipher_rt_got_patch.c (CP 2.5 GOT patcher; dl_iterate_phdr)         |                  
                              |   |                                                                       |                  
                              |   +-------------------------------------------------+---------------------|                  
                              |                                                     |                     |                  
                              +-----------------------------------------------------+---------------------+                  
                                                  USERSPACE / KERNEL BOUNDARY                                                
                              +-----------------------------------------------------v---------------------+                  
                              |    /dev/cipher  (magic 'C', 0666, devnode-codified) AND                  |                  
                              |    /dev/cipher_kvdedup  (magic 'K', 0666)                                |                  
                              |                                                                          |                  
                              |    24 ioctl nrs on /dev/cipher:                                          |                  
                              |      1 REGISTER_TENANT  | 5/6/7 SUBMIT_{GPU_STATE,PROC_UTIL,LAUNCH_STATS} |                  
                              |      8 GET_TENANT_SNAPSHOT (Cb.1)                                        |                  
                              |      9 REQUEST_SM_PARTITION (-ENOSYS; deactivated)                       |                  
                              |     10 SET_CLOCK_MHZ (Cc.4)                                              |                  
                              |     11/12 SUBMIT_FLOP_SAMPLE / QUERY_FLOPS (Cc.5/Cc.6)                  |                  
                              |     13/14/15 CP54 ALLOCATE/FREE/QUERY (Cc.7)                            |                  
                              |     16-20 CP54 migration (SUBSCRIBE/POLL/START/ACK/COMPACT)             |                  
                              |     21-24 ARENA REGISTER/IMPORT/LEAVE/QUERY (Cc.10)                      |                  
                              |     (2/3/4 reserved -> -ENOSYS)                                          |                  
                              |     [v1.5+ proposed nr 25 fairness_shm migration, LP-5]                  |                  
                              |                                                                          |                  
                              |    5 ioctl nrs on /dev/cipher_kvdedup (magic 'K'):                       |                  
                              |      1 INIT | 2 PUT | 3 CONFIRM | 4 FREE | 5 STATS (Cc.9)                |                  
                              +--------------------------------------+-----------------------------------+                  
                                                                     |                                                       
                                                                     v                                                       
+----------------- KERNEL (Wave 3 kmod 0.4.8) -----------------------+                                                       
|                                                                                                                            
|  cipher_main.c (14-subsystem init; cipher_init L25-101; exit L103-120; F3-DO-EXIT-ORDER hard invariant)                    
|                                                                                                                            
|  cipher_internal.h  -- cipher_pid_stats (192B atomic counters), cipher_gpu_state_kern (spinlock)                          
|  cipher_ioctl.h     -- 24 nr macros, 10 structs, reserved tail (Cb.2 lands here)                                          
|  cipher_kvdedup.h   -- 5 nr macros, magic 'K'                                                                              
|  cipher_ioctl_decode.c -- 24-slot binary search; self-check at init                                                        
|                                                                                                                            
|  cipher_dev.c  -- chardev fops + 21-case dispatcher (L251-304); devnode 0666; anti-spoof                                  
|  cipher_probe.c -- kprobe on nvidia_unlocked_ioctl; kretprobe; do_exit reaper (L184-221)                                  
|                    cipher_pid_table 1024 buckets RCU (Ce.1)                                                                
|                                                                                                                            
|  cipher_proc.c -- 6 /proc/cipher/* emitters (stats/bar0/gpu_state/flops/migrations/arenas)                                |
|  cipher_bar0.c -- READ-ONLY BAR0 map; deliberate pci_dev_put leak (cipher-incident-2-bar0_exit)                           
|  cipher_clock.c -- nr 10 handler (Cc.4) via call_usermodehelper("nvidia-smi -lgc")                                         
|  cipher_flops.c -- 256-entry ring; nr 11 (Cc.5) + nr 12 (Cc.6); per-tenant attribution                                    
|                                                                                                                            
|  cipher_cp54_sched.c (LARGEST TU)                                                                                          
|   - 15 x 8-SM groups (NOT 16 -- cipher-cp54-15groups)                                                                      
|   - 3-state per-cell: FREE / PACK(pid) / RSVD(pid); 30-bit pid (F3-30BIT-PID)                                              
|   - PARTITION / SHARED / POOL + Track 3 SC2 migration FSM                                                                  
|   - Conservation: tenant crash leaks ZERO groups (cp54_release_groups sweeps both states)                                  
|   - cipher_cp54_release(pid) lock-free at do_exit (Ce.2)                                                                   
|                                                                                                                            
|  cipher_partition_allocator.c (F3-ALLOC-RETIRE -- userspace dead, housekeeping live; LP-8 retires in Week 4)               
|                                                                                                                            
|  cipher_weight_arena.c -- 16 arenas x 64 consumers x 64KiB blob; fd custodian; off-do_exit reaper                         
|  cipher_kvdedup.c -- /dev/cipher_kvdedup; 2^20 entries; 2-phase PUT/CONFIRM handshake; release fop reclaim (Ce.7)         
|                                                                                                                            
|  cipher_state_updater.c -- 1 kHz kthread; derived fields (thermal/power/voltage/sustained_clock); LP-8 reap leg          
|                            [NEW Cb.5: brain-derived per-tenant fields written here]                                       
|                                                                                                                            
|  cipher_tenant_snapshot.c -- per-CPU TLS snapshot; assembled view for ioctl nr 8                                          
|                                                                                                                            
+----------------------------------------------------------------------------------------------------------------------------+
                                                                                                                            
============================================================================================================================
KERNEL INTERCEPT ENTRY POINTS (from 5.2):                                                                                   
  (*) cublasGemmEx        -> cipher_rt_cublasGemmEx_impl              (5.2.1; 28 steps; ~200-250ns budget)                  
  (*) SDPA (3 backends)   -> cipher_rt_attn_dispatch::{flash,eff,cudnn}_call  (5.2.2; 17 steps; LP-2 refactor)             
  (*) cuLaunchKernel CUPTI-> cipher_v2_cupti_cb                       (5.2.3; 16+kprobe+ring steps; v1.5 brain hook)        
  (*) cuMemMap / VMM      -> cipher_rt_kv_slab_create / arena_create  (5.2.4; off-hot-path; kmod-mediated cross-tenant)     
============================================================================================================================
SHARED STATE / OWNERSHIP:                                                                                                   
  g_cipher.liquid         | OWNED BY Stage-11 thermal_feedback; READ BY thermostat (Cd.3)                                   
  cipher_sense_current_   | OWNED BY SENSE TLS; READ BY 6 observers (Cd.6)                                                  
   session() TLS          |                                                                                                
  cipher_pid_table        | OWNED BY cipher_probe.c (RCU); READ BY proc + state_updater + tenant_snapshot + flops (Ce.1)   
  cipher_gpu_state        | OWNED BY cipher_dev.c::nr5 SUBMIT_GPU_STATE; READ BY state_updater + proc (Ce.4)                
  cipher_cp54_groups[15]  | OWNED BY cipher_cp54_sched.c (atomic_t cells); READ BY ioctls 13-20 + do_exit reaper (Ce.2/5)   
  cipher_wa_arenas[16]    | OWNED BY cipher_weight_arena.c (mutex); READ BY ioctls 21-24 + 5s reaper (Ce.6)                 
  kvd_buckets[65536]      | OWNED BY cipher_kvdedup.c (mutex); READ BY ioctls; release-fop reclaim (Ce.7)                    
  classify cache 512 slot | OWNED BY cipher_rt_classify (lock-free relaxed); SHARED across threads                          
  oracle EMA arrays       | OWNED BY cipher_rt_oracle (NON-ATOMIC accepted races; degrades gracefully)                       
  matmul/attn registries  | OWNED BY cipher_rt_*_dispatch (pthread_mutex + lock-free snapshot read on hot path)              
  marlin weight cache     | OWNED BY cipher_rt_marlin_engine.cpp (per-weight mutex + per-stream WS mutex)                    
  cipher_fairness shm     | OWNED BY first-mapper (CAS on magic); SHARED ACROSS PROCESSES via mmap                          
  /cipher_fairness        | LP-5 candidate for kmod migration (nrs 25/26/27 v1.5)                                            
============================================================================================================================
```

---

## 5.6.x — Companion: file → tier → contract index

Quick lookup table; every file in the fused runtime, its tier, the
contracts it participates in. Use this to find "where does the
fairness_shm coupling live in 5.1" or similar.

### Classifier tier (Wave 1; ported in Week 1)
| File (after port)                            | Provides         | Consumes                            | Contracts          |
|----------------------------------------------|------------------|-------------------------------------|--------------------|
| `cipher_rt_dispatch.cpp`                     | `cipher_dispatch` (Ca.1) | classify, oracle, structural, recipes | Ca.1, Ca.2, Ca.3, Ca.6, Ca.10, Ca.11 |
| `cipher_rt_oracle.cpp`                       | oracle gates     | structural_lookup, liquid (null)    | Ca.6, Ca.11        |
| `cipher_rt_classify.h`                       | classify_launch  | (none, leaf)                        | Ca.11              |
| `cipher_rt_recipes.cpp`                      | registry lookup  | hw_profile (H100)                   | (registry hit/miss) |
| `cipher_rt_sense.cpp`                        | session classifier | TLS, ring                          | (drives Cd.6)       |
| `cipher_rt_structural_lookup.cpp`            | safety gates     | name table, override                 | Ca.12              |
| `cipher_rt_kernel_table.cpp`                 | kernel name resolve | libcuda cuFuncGetName             | Ca.12              |
| `cipher_rt_classify_substrate.cpp`  *(new)*  | classifier registry | -                                 | (Week 1 framing)    |
| `cipher_rt_classify_observer.c`     *(new)*  | observer maybe_handle | matmul/attn substrate registries | Ca.7, Ca.8          |

### Substrate tier (Wave 2; ports anchored at c2c5d313)
| File                                         | Provides                                | Consumes                          | Contracts          |
|----------------------------------------------|-----------------------------------------|-----------------------------------|--------------------|
| `cipher_inject.c`                            | 14-step init                            | every substrate init               | (orchestrator)      |
| `cipher_rt_matmul_dispatch.c`                | matmul substrate registry               | actuator chain                     | Ca.7               |
| `cipher_rt_attn_dispatch.cpp`                | attn substrate registry; 3 trampolines  | actuator chain                     | Ca.8, **LP-2**     |
| `cipher_rt_cublas_shim.c`                    | cublasGemmEx GOT-target                 | matmul_dispatch                    | (entry *)           |
| `cipher_rt_got_patch.c`                      | GOT patcher                             | dl_iterate_phdr                    | (mechanism)         |
| `cipher_rt_marlin_actuator.c`                | priority 10 actuator                    | matmul registry; TLS hint          | Ca.9               |
| `cipher_rt_marlin_engine.cpp`                | NVRTC GEMM engine                       | cuda driver                        | (engine)            |
| `cipher_rt_volt.c`                           | DVFS clock-lock                         | NVML OR nr 10                      | Cc.4               |
| `cipher_rt_audit.c`                          | priority 0 audit observer (both substr) | matmul + attn registries           | Cd.8, **F2-AUDIT-LOCK** |
| `cipher_rt_attn_test_actuator.c`             | env-gated priority 0                    | attn registry                      | (coverage)          |
| `cipher_rt_sm_packer.c`                      | detection-only counters                 | CUPTI                              | (substrate)         |
| `cipher_rt_partition_router.c`               | stream priority + sync-domain           | CUPTI                              | (substrate)         |
| `cipher_rt_green_ctx.c`                      | green ctx ensure/migrate                | nr 13-20                           | Cc.7, Cc.8         |
| `cipher_tenant.c`                            | tenant register                         | nr 1                               | Cc.1               |
| `cipher_cupti.c`                             | CUPTI subscriber + enforcement gate     | green_ctx, pr, smp, nr 7           | Cc.2               |
| `cipher_rt_tenant.cpp`                       | per-thread tenant snapshot              | nr 8                               | Cc.3, Cb.1         |
| `cipher_rt_kv_alloc.c`                       | VMM slab + dedup + arena                | nr 21-24, kvdedup nrs              | Cc.9, Cc.10        |
| `cipher_kv_bridge.cpp`                       | Python pybind11 seam                    | kv_alloc                            | (Python)            |
| `cipher_rt_arbitrate.c` (VESTIGIAL)          | -                                       | -                                  | (retired CP 5.4)    |

### Kmod tier (Wave 3; anchored kmod 0.4.8)
| File                            | Provides                                          | Consumes                         | Contracts          |
|---------------------------------|---------------------------------------------------|----------------------------------|--------------------|
| `cipher_main.c`                 | 14-subsystem init; F3-DO-EXIT-ORDER invariant     | every TU                          | Ce.3                |
| `cipher_internal.h`             | cross-TU types (cipher_pid_stats, gpu_state)      | every TU                          | (typing)            |
| `cipher_ioctl.h`                | 24 nr macros, 10 structs; reserved tail           | userspace                         | Cb.2                |
| `cipher_kvdedup.h`              | 5 nr macros, magic 'K'                            | userspace                         | (ABI)               |
| `cipher_ioctl_decode.c`         | 24-slot binary search                             | cipher_probe                      | (telemetry)         |
| `cipher_dev.c`                  | /dev/cipher chardev + 21-case dispatcher          | every ioctl handler               | Cc.1-Cc.8           |
| `cipher_probe.c`                | kprobe + do_exit reaper                            | cipher_pid_table                  | Ce.1, Ce.2          |
| `cipher_proc.c`                 | /proc/cipher/* emitters                            | every state-holding TU            | (observability)     |
| `cipher_bar0.c`                 | read-only BAR0 map                                 | -                                 | (probe)             |
| `cipher_clock.c`                | nr 10 handler                                      | UMH nvidia-smi                    | Cc.4                |
| `cipher_flops.c`                | nr 11/12 + 256-entry ring                          | RCU pid_table                     | Cc.5, Cc.6          |
| `cipher_cp54_sched.c`           | nr 13-20 + 15×8-SM ledger + SC2 FSM               | atomic cells; pool_pid            | Cc.7, Cc.8, Ce.2, Ce.5 |
| `cipher_partition_allocator.c`  | LP-8 — userspace dead, housekeeping live          | state_updater_tick + do_exit       | (REFACTOR-RETIRE)   |
| `cipher_weight_arena.c`         | nr 21-24 + fd custodian + 5s reaper                | fd table + pid liveness            | Cc.10, Ce.6         |
| `cipher_kvdedup.c`              | /dev/cipher_kvdedup; 2-phase PUT/CONFIRM            | xarray + hlist + per-fd state     | Cc.9, Ce.7          |
| `cipher_state_updater.c`        | 1 kHz derived-field kthread                        | gpu_state + pid_table              | Cb.5                |
| `cipher_tenant_snapshot.c`      | per-CPU TLS snapshot                                | pid_table                         | Cb.1, Cc.3          |

### Observer tier (Wave 4; ported in Week 4)
| File (after port)                            | Lane            | Cost          | Fusion class             | Contract refs       |
|----------------------------------------------|-----------------|---------------|--------------------------|---------------------|
| `cipher_rt_carbon.cpp`                       | L1 ring         | ~50 ns        | PORT + SHIM (per-tenant) | Cd.6, Cd.7          |
| `cipher_rt_fairness.cpp`                     | L1 ring         | ~60 ns        | PORT + SHIM              | Cd.6, Cd.7          |
| `cipher_rt_fairness_shm.cpp`                 | **L3 cublas**   | **~1.4 µs**   | REFACTOR v1.5 (kmod)     | Cd.2, **LP-5**     |
| `cipher_rt_receipt.cpp`                      | L1 ring         | ~60 ns        | PORT + SHIM              | Cd.6, Cd.7          |
| `cipher_rt_guard.cpp`                        | L1 ring         | ~80 ns        | PORT + SHIM              | Cd.7                |
| `cipher_rt_comply.cpp`                       | L1 (no-op)      | 0 ns          | PORT-AS-IS               | (aggregator)        |
| `cipher_rt_loop.cpp`                         | L1 ring         | ~100 ns       | PORT + SHIM              | Cd.7                |
| `cipher_rt_pipeline.cpp`                     | L1 ring         | ~40-150 ns    | PORT + SHIM              | Cd.6, Cd.7          |
| `cipher_rt_trace.cpp`                        | L1 ring         | ~30 ns        | PORT-AS-IS               | Cd.7                |
| `cipher_rt_continuity.cpp`                   | L1 ring         | ~80 ns        | PORT + SHIM              | Cd.6, Cd.7          |
| `cipher_rt_pulse.cpp`                        | L1+L2 hybrid    | ~70 ns/event  | PORT + SHIM              | Cd.4, Cd.7          |
| `cipher_rt_determinism.cpp`                  | L1 ring         | ~50 ns        | PORT-AS-IS               | Cd.7                |
| `cipher_rt_topology.cpp`                     | L1 (no-op)      | 0 ns          | PORT-AS-IS               | (init-time)         |
| `cipher_rt_thermostat.cpp`                   | L1+L2 hybrid    | ~70 ns/event  | PORT + SHIM              | Cd.3, Cd.6, Cd.7    |
| `cipher_rt_shield.cpp`                       | L1 + actuator   | ~140 ns/event | PORT + SHIM (wired)      | Cd.1, Cd.6          |
| `cipher_rt_sustain.cpp`                      | L1 ring (decode)| ~70 ns/decode | PORT + SHIM              | Cd.6, Cd.7          |
| `cipher_rt_hibernate.cpp`                    | L1+L2 hybrid    | ~20 ns + ~10 µs idle→active | REFACTOR small v1.5 | Cd.5, Cd.7, **LP-9/LP-10** |
| `cipher_rt_straggler.cpp`                    | L4 NCCL         | ~10 µs/AllRed | PORT-AS-IS (v1.5 buffer) | (NCCL)              |

---

# Closing rollup

**Counts.**
- Fusion contracts catalogued in 5.1: **43** (12+5+11+8+7).
- Lossless fusion points in 5.3: **22**.
- Lossy fusion points in 5.4: **16** — blocking **2**, requires-mitigation **5**,
  minor **9**.
- Integration steps in 5.5: **5 weeks** (Section 7 corrected) + v1.5 deferral
  ledger.
- Kernel intercept entries documented in 5.2: **4** (cublasGemmEx, SDPA,
  cuLaunchKernel, cuMemMap/VMM).

**New findings beyond Waves 1-4.**
- **NF-5.1 (LP-14): R-C1's 12 ns budget is unsupportable.** Wave 1's
  measurements + control-flow analysis put the realistic hot-path budget
  at 100-200 ns, not 12 ns. The honest framing is "accept 1-2 %
  per-tenant overhead." Section 7's R-C1 should be rewritten with that
  as the primary case.
- **NF-5.2 (LP-7 Week-1 dependency):** the `CipherKernelEntry` rename is a
  *compile blocker* and must precede everything else in Week 1; the
  original Section 7 doesn't sequence it.
- **NF-5.3 (LP-2 Week-2 dependency):** the attn HANDLED-discard fix must
  land in Week 2 to unblock Week 3's "all-substrate routing." Section 7
  Week 3 scope should be **GEMM-only** in v1; attn substitute deferred to
  v1.5.
- **NF-5.4 (Cb.2 missing in Section 7):** the snapshot reserved-tail
  extension is the lossless seam for classifier-output fields. Section 7
  never mentions it; this plan lands it in Week 1.
- **NF-5.5 (LP-8 Week-4 cleanup):** `cipher_partition_allocator.c` is
  REFACTOR-RETIRE (state_updater + do_exit reaper are live consumers).
  Section 7 doesn't mention retirement; this plan schedules it in Week 4
  alongside observability port.
- **NF-5.6 (LP-11 fallback exists):** "classifier confidence < 1.0"
  reframed — Wave 1 brain emits **binary** confidence (40 or 85); below
  min_confidence the existing PASS_THROUGH IS the fallback. The original
  task framing implied this was an open question; it's not.
- **NF-5.7 (Cd.2 / LP-5 sizing):** fairness_shm at 1.4 µs/GEMM × 16
  tenants × 10K GEMM/s ≈ 22 % overhead on one core in multi-tenant runs.
  This is the single largest non-Marlin v1 cost; v1.5 kmod migration is
  the well-bounded mitigation.
- **NF-5.8 (Wave 2 ERR-SPLIT discipline):** the matmul-vs-attn ERROR
  semantic divergence (F2-ERR-SPLIT) is a discipline issue, not a code
  one — observers must never return ERROR. Document in
  `cipher_rt_classify_observer.c` header; compile-time-enforced if
  possible. Wave 5.4 LP-3.

**Sequencing summary.**
1. **Week 1** (compile-only): port all 7 brain files + 1 new substrate + 1
   new observer stub; rename `CipherKernelEntry` (LP-7); add reserved-tail
   classifier fields to snapshot struct (Cb.2). LOSSLESS gate: byte-identical
   Mistral-7B forward.
2. **Week 2** (telemetry only): register classifier observer at priority 5
   on both substrates; brain runs but mutates no actuator state; LAND
   LP-2 attn trampoline refactor (REQUIRED to unblock Week 3 attn lane).
   LOSSLESS gate: classify hit rate ≥ 95 %; bit-identical attn output.
3. **Week 3** (GEMM lane live; attn deferred): classifier hint drives
   Marlin engagement, dynamic VOLT, DSM PROPOSE on tool-idle.
   LOSSLESS gate: WL03 prefill MFU climbs toward 85 %; WL01 decode tok/W
   +14 %; isolation 15/15 PASS.
4. **Week 4** (observability): port 12 observers (defer 4 partial ones);
   retire `cipher_partition_allocator.c` deactivated public surface (LP-8);
   wire AUDIT chain into `SUBSTITUTE` events.
   LOSSLESS gate: TRACE drops == 0 over 60 s @ 10 K launches/s; AUDIT
   chain verifiable.
5. **Week 5** (CP 5.5): 100-tenant benchmark, 24-h soak, all measurements
   reproducible from a fresh boot in ≤ 30 min.

**The honest engineering posture for v1.** The fused runtime is achievable
with the v1 scope above and the deferral ledger in 5.5.  Of the 16 lossy
points, 9 are minor, 5 are mitigated by Week-4 + discipline + v1.5 deferral,
and 2 (LP-2 + LP-7) are explicit Week-1/2 work with well-defined fixes.
The honest budget for hot-path overhead is ~1-2 % per launch, not the
12 ns Section 7 implies. The fairness_shm 1.4 µs/GEMM cost is the largest
v1 residual and is the right candidate for the first v1.5 deliverable
after CP 5.5 ships.

The substrate is already complete enough to absorb the classifier brain
as a priority-5 observer on each registry without touching the existing
PASSTHROUGH-only contract; the kmod ABI rule means none of the proposed
extensions (Cb.2 reserved-tail fields; v1.5 nrs 25-27) breaks the live
24-nr table; and the do_exit reaper discipline (Ce.2) generalises to any
future per-tenant kernel-side state. The classifier brain port lands as
*substrate*: telemetry on day one, actuation as downstream ops come online.
