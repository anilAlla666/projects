# D.8 FAIRNESS + SHIELD — design memo (STOP for Anil approval; no code yet)

**Date:** 2026-05-29. design-memo → **approve** → build → close. This is the design + plan
ONLY. No source touched beyond the D.7 anchor rotation already recorded below.

Serves **Goal 1** (V.1 item-1b): *100 heterogeneous bursty agents on one H100, each thinks
it owns the GPU.* Isolation has two failure modes this substep closes:
- **R-D5 FAIRNESS** — burst-fairness: one agent bursting must not starve the others of their
  GPU-work share.
- **R-I1 SHIELD** — noisy-neighbor p99: a latency-sensitive agent's tail latency must stay
  bounded when a throughput agent floods the GPU.

---

## §0 Provenance (anchors verified on disk)

| component | anchor | state |
|---|---|---|
| cipher_rt_phase4 | **`ed130e7`**, tag **`d7-rh1-close`**, anchor `.so` md5 **`01d4effb`** | D.7 CLOSED — rotation finalized this session (see note) |
| cipher_kmod | **`02fc2d1`** (srcversion E27B…) | UNCHANGED since W9 |
| cipher_kv_bridge.so | md5 **`5a3db034`** | UNCHANGED |
| cipher-fusion-evidence | HEAD `7581153` (pre-this-memo) | — |
| prior-anchor fallback | `build_cuda13/libcipher_rt.so.pre_d7rh1` md5 `9fe23143` | preserved per discipline |

**D.7 close + rotation note (recorded for the auditor).** Your message declared D.7 R-H1
CLOSED on option (b) — accept the byte-identical additivity-KL + the libcipher-only/additive
nature of the change as sufficient regression coverage. On disk the rotation had **not** been
performed (the anchor `.so` was still `9fe23143`, no `d7-rh1-close` tag). I finalized it to
match your directive: preserved `9fe23143` as `.pre_d7rh1`, rotated the anchor `.so` to
`01d4effb`, tagged `d7-rh1-close` at `ed130e7`. **Provenance honesty:** `ed130e7`'s commit
subject is *"D.10 LT-ROUTE…"*, so the anchor build `01d4effb` carries **both** the D.7 Marlin
`(model_id,w_ptr)` re-key **and** the dormant, default-OFF D.10 cublasLt layout-decode/variant
code. The additivity-KL=0 in D7_CLOSE_REPORT §2 was measured against this exact `01d4effb`
build (LT_ROUTE off, no bind) and covers the whole build byte-for-byte — so a `d7` tag sitting
on a `d10` commit is expected, not a mismatch.

**Carried Goal-1 prerequisite (NOT closed by D.8):** the Marlin cache-churn gap
(`cipher-marlin-cache-churn-gap` — `g_weights` doesn't GC kits on model free → corruption/crash
under model load/free/reload churn). D.8 is tested at static residence; it proves the
fairness/shield *mechanism*, not the full bursty-churn scenario. See §3.

---

## §1 What is BUILT vs net-new (read on disk, reported exactly)

### FAIRNESS — BUILT (all observer / plumbing; **zero enforcement**)
- **Op 24 observer** `src/may13/cipher_fairness.cpp` — per-session open-address table,
  accumulates `grid*block` work units, sets an `overrun_flag` when a tenant crosses a quota,
  writes `/tmp/cipher_fairness_report.json`. Env-gated `CIPHER_FAIRNESS` (default OFF). It
  **only observes** — it never throttles, delays, yields, or rejects a launch.
- **`should_yield` decision primitive** `src/may13/cipher_fairness_shm.cpp:170` — already
  implements the burst-fairness rule: a tenant should yield if its `gemm_calls > 2×` the
  active-tenant average **and** `> 1024`. **But:** (a) it is referenced **only** by the legacy
  `src/may13_intercept/cipher_intercept_cudart.cpp` (via `dlsym`), **not** by the production
  cuBLAS GOT-patch hot path (`cipher_rt_cublas_shim` / `cipher_rt_marlin`); (b) the "yield"
  action is a **no-op** — it just increments `yield_count`, nothing actually sleeps/defers; and
  (c) its state lives in **`shm_open("/cipher_fairness")` → `/dev/shm`**, which is the
  load-bearing problem below.
- **kmod `fairness_quota_remaining_pct`** (`cipher_internal.h:230,291`, `cipher_ioctl.h:169`) —
  field exists; the P4.7 state-updater kthread (`cipher_state_updater.c:129-133`) **hardcodes
  it to 100** with the comment "the kthread is the right place to compute it once FAIRNESS is
  fused." That fusion is exactly D.8.

### SHIELD — BUILT (priority plumbing; no policy, no p99 wiring)
- **`slo_priority`** field in kmod (`cipher_internal.h:290`, `cipher_ioctl.h:168`) — exists,
  unread by any actuator.
- **Priority-band table** `cipher_sm_set_priority/get_priority/priority_count`
  (`cipher_green_ctx.cu:275-299`) — per-session band primitive, built, unused by any policy.
- **CP54 QoS classes** `CIPHER_CP54_QOS_{PARTITION,SHARED,POOL}` (`cipher_ioctl.h:370-372`,
  `cipher_cp54_sched.c`) — the SM-arbitration substrate (Track 3 / CP 5.4 green-ctx, 15×8-SM
  groups). This is the *alternative* stronger-isolation lever for SHIELD (see §3/§4).
- **`cudaStreamCreateWithPriority`** (`cipher_green_ctx.cu:159`) — CIPHER's own pool streams use
  CUDA stream priority; not applied to tenant streams.

### Net-new (what D.8 actually builds — wiring + enforcement, not new infrastructure)
1. A **cross-tenant work-ledger in the kmod registry** (not `/dev/shm` — see §3) + the
   per-tick quota/remaining computation in the existing state-updater kthread.
2. **Repoint `should_yield`'s data source** from `/dev/shm` to the kmod registry snapshot (the
   rate-vs-average logic ports nearly verbatim).
3. The **throttle ACTION** in the production gemm hook (a bounded CPU sleep before submission —
   timing-only, §4), gated default-OFF.
4. **Priority-band modulation** for SHIELD: a high-band (latency-sensitive) tenant never yields;
   a low-band throughput tenant yields harder when a high-band tenant is contending. Reuses the
   band table + `slo_priority`.
5. The **two measurement gates** (§2) — none exists today.

---

## §2 The isolation gate — how FAIRNESS/SHIELD are MEASURED (two INDEPENDENT gates)

These are different objectives that pass/fail independently: a throttle can hold quota shares
while still blowing a latency-sensitive tenant's p99, or protect one tenant's p99 by starving
everyone equally. Both are defined crisply and both must pass.

**Gate A — R-D5 burst-fairness (quota-share retention under one bursting agent).**
Setup: `K` contending tenants, same model, static residence, bf16. One designated **noisy**
tenant bursts (floods gemms continuously); the others run a steady moderate rate.
- Metric: each victim's achieved GPU-work share (gemms completed / wall-second) **with** FAIRNESS
  ON vs the same scenario with it OFF.
- **PASS:** with enforcement ON, the victims retain **≥ their fair share** (≈ `1/K` of throughput
  within a defined tolerance band, e.g. ≥ 80% of the equal-share baseline), and the noisy
  tenant is the one throttled — vs OFF, where the noisy tenant captures a disproportionate share
  and victims are starved. The honest engagement gate (Mem #19): D.8 PASSES only if the
  ON-vs-OFF victim-share delta is real and attributable to the throttle, measured — not asserted.

**Gate B — R-I1 SHIELD (p99 bound under a noisy throughput neighbor).**
Setup: one **high-band** latency-sensitive tenant (small, frequent, low-volume requests) +
one **low-band** noisy throughput tenant (large, continuous gemm flood), co-resident, bf16.
- Metric: the high-band tenant's request **p99 latency** with SHIELD ON vs OFF, each also vs its
  **solo** p99 (no neighbor).
- **PASS:** with SHIELD ON, the high-band tenant's p99 stays **bounded** — within a defined
  multiple of its solo p99 (target ≤ ~1.5–2× solo) — whereas with SHIELD OFF the noisy neighbor
  inflates it well beyond that. Engagement gate: the ON p99 must be measurably tighter than OFF
  and close to solo; if ON ≈ OFF, SHIELD does not engage and D.8 does NOT pass on R-I1.

**Correctness gate (Mem #11), applies to both:** each tenant's per-token output under
enforcement ON must be **bit-identical** to its standalone output (KL = 0). Enforcement is
timing-only by construction (§4); any divergence is a HARD STOP.

---

## §3 Single-H100 + dtype constraints — measurable here, with honest scope

1. **Cross-container ledger MUST be kmod-resident, not `/dev/shm`.** The V.1 target is separate
   *containers* via CDI. Docker gives each container its own `/dev/shm` (isolated) unless
   `--ipc=host`, so the existing `shm_open("/cipher_fairness")` ledger would **not span the 100
   agents** — Gate A would silently measure intra-container fairness and pass spuriously. The
   global ledger therefore lives in the **kmod registry** (`/dev/cipher`), the same cross-tenant
   channel the **W9 resolver already proved** (18.4B reads, INCOHERENT=0, spans tenants; CDI
   exposes `/dev/cipher` to every container). Spine: **kmod registry = global view;
   libcipher_rt gemm-hook = per-tenant self-throttle on the global state it reads.** Both
   substrate-line (§4). *Note:* on this single host I will validate cross-tenant accounting both
   ways — multi-process (shares `/dev/cipher`) and, if a multi-container rig is stood up, across
   containers — and report which was exercised, not conflate them.
2. **"100 agents" = registration / statistical-mux, NOT 100-at-once.** ~2% duty means
   instantaneous contention is a *few* concurrent tenants. The gates are stressed by **N
   concurrent contending tenants** (1 noisy + several victims; 1 high-band + 1 noisy) — that is
   the real contention the substrate must survive. The 100 figure is the registration scale
   (the kmod ledger/registry must hold ≥100 slots — it already does: WA arenas 100, resolver
   8192 slots). Stated plainly per Mem #19: D.8 measures the contention mechanism at a small
   adversarial concurrent set; it does not claim a literal 100-simultaneous-burst run.
3. **Static residence isolates the mechanism; it is NOT the full Goal-1 scenario.** D.8 tests at
   static residence (models loaded once, no churn) to isolate fairness/shield behavior — because
   the carried Marlin cache-churn gap makes model load/free/reload unsafe today. Therefore
   **D.8-passing ≠ Goal-1-met:** the churn gap remains a separate prerequisite carried to the
   100-agent residence work. The memo does not paper over this.
4. **dtype.** Work-unit accounting (`grid*block`) and the timing-throttle/priority-band are
   **dtype-agnostic** — no fp16/bf16/INT4 dependence. bf16/INT4 (locked serving dtype) is fully
   supported; no dtype blocker. (Contrast D.10's fp16 dead-end — not applicable here.)
5. **SHIELD lever choice — surfaced for your call.** Two viable substrate-line mechanisms:
   (a) **throttle-band** (proposed first) — the noisy low-band tenant is CPU-throttled so it
   stops flooding, freeing the GPU for the high-band tenant; reuses the FAIRNESS substrate
   almost entirely, simplest to gate. (b) **reserved SM-partition** (CP54 green-ctx) — give the
   high-band tenant a dedicated SM group the throughput neighbor cannot steal; stronger
   isolation but heavier (green-ctx lifecycle, the CP54 contention-gated residue). I propose
   gating (a) first and holding (b) as the escalation if (a)'s p99 bound is insufficient.

---

## §4 The plan + decision gate

**Substrate-line (Mem #24) — confirmed, no app patch.** All enforcement sits in CIPHER's two
existing substrate layers: the **kmod** (global ledger + quota/band state, in the registry and
the state-updater kthread) and **libcipher_rt** (the throttle action in the already-owned cuBLAS
GOT-patch gemm hook, where `cipher_workload_observe_gemm` already fires). **No vLLM/torch source
patch.** If any step is found to require a torch/vLLM patch → **HARD STOP, surface** (not
expected — torch creates the streams, but the throttle acts in our gemm hook, not on stream
creation; the throttle-band path needs no stream-create interception).

**KL = 0 by construction (Mem #11).** Enforcement is **timing-only**: a throttle/yield is a
bounded CPU sleep *before* the gemm is submitted; a priority band only changes *when* a tenant
submits. Kernels, arguments, and submission order within a tenant are untouched → per-tenant
output is bit-identical → KL=0 is **structural**, then verified. This bounds the design: any
proposed mechanism that **drops work, switches kernels, or reorders within a tenant** = HARD
STOP. The CPU-sleep-before-gemm throttle is safe on this test.

**Build sequence (one substep; every new op default-OFF; worst case = no enforcement == current
behavior):**
1. **kmod ledger + quota** — add a per-tenant work-units accumulator to the registry (keyed by
   `tgid`, reusing the resolver/registry machinery) + compute `fairness_quota_remaining_pct` and
   a per-tenant fair-share in the state-updater kthread (the existing placeholder). New ioctl at
   **NR 30** (next free; NR 29 = REGISTER_STREAMS) for ledger write / band set — **additive per
   the ABI rule** (reserved NRs unaffected; existing NRs byte-identical). Default-OFF (ledger
   inert unless armed).
2. **libcipher_rt repoint + port** — point `should_yield`'s data source at the kmod registry
   snapshot (mmap'd, as the resolver exposes it cross-tenant); port the rate-vs-average logic.
3. **libcipher_rt throttle action** — in the gemm hook, if `should_yield()` (band-modulated),
   bounded CPU `nanosleep` before submit. Env-gated `CIPHER_FAIRNESS=on` / `CIPHER_SHIELD=on`,
   both default-OFF.
4. **SHIELD band wiring** — register a tenant's `slo_priority` band (reuse `cipher_sm_set_priority`
   + kmod `slo_priority`); high band ⇒ never yields, contention by a high band makes a low band
   yield harder.
5. **Harness + gates** — Gate A (1 noisy + K victims, victim-share retention ON vs OFF) and
   Gate B (1 high-band + 1 noisy, p99 ON vs OFF vs solo), same-model static residence, bf16; +
   per-tenant KL=0 vs solo.

**§4 decision gate — D.8 PASSES iff ALL hold:**
- **Gate A:** victims keep their fair quota share under one bursting agent (ON measurably better
  than OFF, attributable to the throttle), measured at the N-concurrent contending set.
- **Gate B:** the high-band tenant's p99 stays bounded under a noisy throughput neighbor (ON
  measurably tighter than OFF, close to solo).
- **KL = 0** per tenant vs standalone (correctness preserved).
- **Full no-regression** (§5) — fresh-substrate HARD GATE.

If Gate B does not bound p99 via throttle-band, surface and escalate to the reserved
SM-partition lever (§3.5) — do not fake a pass.

---

## §5 Discipline + the NO-REGRESSION close requirement

- **One substep, two capabilities, default-OFF.** Worst case (env unset / ledger disarmed) =
  byte-identical to the current `01d4effb` no-enforcement behavior.
- **ABI:** additive only (new NR 30; reserved NRs return -ENOSYS; existing NRs unchanged).
- **Anchors UNCHANGED until an approved build** (Mem #16). This memo touches no source.
- **Fresh-substrate full no-regression close gate (Mem #16 HARD GATE)** — D.8 introduces kmod
  *and* libcipher_rt changes, so it is treated as fresh substrate. Before any anchor rotation:
  1. **Additivity-KL byte-identical OFF** — with FAIRNESS/SHIELD disarmed, prior-model logits
     (TinyLlama + Mistral) `max_logit_diff = 0.000e+00` vs the `01d4effb` anchor.
  2. **30-min N=128 resolver soak PASS** — INCOHERENT=0, misses=0 (the kmod ledger addition must
     not perturb the resolver coherence the W9/D.7 soak proved).
  3. **Per-tenant correctness KL=0** (§2 correctness gate).
- **Mem #11:** any per-tenant output divergence under enforcement = HARD STOP.
- **No fakes, no workarounds.** Sudo (kmod load) is Anil's to run — if a build step needs it,
  EMIT the exact command and WAIT.

**STOP — awaiting Anil's approval before any code.** On approval: build steps 1→5, run the §4
gates + §5 regression, STOP at the close verdict (rotate anchor + tag only if all gates pass).
