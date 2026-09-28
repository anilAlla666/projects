# CIPHER Non-Canonical "T4.x.x" Work — Evidence Audit

**Audited:** 2026-05-15. Read-only. Evidence = report .md files cross-checked
against on-disk artifacts (md5sum, wc -l, ls -l, /dev nodes, lsmod).
**Scope:** the T4.x.x deliverable series, which absorbed most of CIPHER's
recent effort but does not map cleanly onto the canonical Phase 0-9 plan.

This audit treats each T4.x.x deliverable on its own terms. No charitable
interpretation: where a headline number did not survive scrutiny, that is
stated plainly. Where a report claims SHIPPED but the gate was not met or
was reframed, that is flagged.

---

## 1. Status table

| Deliverable | Status | Verdict |
|---|---|---|
| T4.2.1 contention gate | SHIPPED (per prior memory; not re-audited here) | PARALLEL |
| T4.2.2 partition router (STREAM_ATTR_BINDER) | PARTIAL | PARALLEL |
| T4.2.3 ARBITRATE + SM_PACKER | PARTIAL | PARALLEL |
| T4.2.4a GREEN_CTX scaffold | SHIPPED (plumbing only) | PARALLEL |
| T4.2.4b/b_v2 per-tenant GREEN_CTX partition | PARTIAL | PARALLEL |
| T4.2.4c noisy-neighbor A/B (cross-resource) | NOT DONE (hypothesis unsupported; actuator cosmetic) | PARALLEL |
| T4.2.4d GREEN_CTX enforcement fix | SHIPPED (enforcement); isolation FALSIFIED | PARALLEL |
| T4.2.4e tail-latency isolation (same-resource pair) | PARTIAL (tail-tightness yes; p99/throughput no) | PARALLEL |
| T4.3.0 DVFS discovery | SHIPPED (discovery doc) | PARALLEL |
| T4.3.1 VOLT NVML clock-lock | SHIPPED (mechanism); user-process path DEGRADED | PARALLEL |
| T4.3.2 kmod CIPHER_SET_CLOCK_MHZ ioctl | SHIPPED | PARALLEL |
| T4.3.x envelope characterization | SHIPPED (measurement); narrows the headline | PARALLEL |
| T4.5 .symver cuBLAS substrate | SHIPPED | PARALLEL |
| T4.5 Marlin INT4 actuator | NOT DONE (regressed / hung; product number unmeasured) | PARALLEL |
| T4.6.0 / T4.6.0.5 KV-dedup discovery + measurement | SHIPPED (discovery/measurement) | PARALLEL |
| T4.6.1 attention-routing substrate | PARTIAL (built; 14% decode coverage; FakeTensor bug) | PARALLEL |
| T4.6.2 KV page allocator + from_blob bridge | SHIPPED | PARALLEL |
| T4.6.3 KV page-level dedup (content-hash) | PARTIAL (single-process / synthetic only) | PARALLEL |
| T4.6.4 kmod-owned cross-tenant KV dedup | SHIPPED (mechanism, cross-process) | PARALLEL |

**No T4.x.x deliverable supersedes a canonical checkpoint.** Equivalence
arguments per deliverable below; the summary argument is in Section 4.

---

## 2. Per-deliverable evidence treatment

### T4.2.x — SM partition / contention / GREEN_CTX

#### T4.2.2 — PARTITION_ROUTER — PARTIAL
- Evidence: `/home/ubuntu/PHASE_4_T4_2_2_REPORT.md`; build
  `libcipher_rt.so.v0.2.0_T4_2_2` (on disk, 27184 bytes, May 13 20:16,
  report md5 `45ed551a...`); source `cipher_rt_partition_router.c` (291 LOC).
- What shipped: a CUPTI stream-attribute binder (`cuStreamSetAttribute`
  for priority + sync_domain). Smoke confirms `cipher_rt_pr_init` fires.
- Why PARTIAL not SHIPPED: the report itself names two scope gaps.
  (1) `CIPHER_REQUEST_SM_PARTITION` is never called, so `sm_partition_mask`
  is 0 in every snapshot; the build is a STREAM_ATTR_BINDER, not the
  spec'd PARTITION_ROUTER. (2) priority is frozen at first-launch when
  `launches_total≈0`, so it always resolves to 0. The headline WL05
  measurement was **retracted by a 2026-05-14 correction**: the runner
  loaded `libcipher_v2`, not `libcipher_rt` — WL05 was never measured
  under T4.2.2. WL14 showed a real, unexplained −5.5% tok/s.
- Verdict: **PARALLEL.** SM-stream routing is not a canonical CP.

#### T4.2.3 — ARBITRATE + SM_PACKER — PARTIAL
- Evidence: `PHASE_4_T4_2_3_REPORT.md`; `libcipher_rt.so.v0.2.0_T4_2_3`
  (32328 bytes, May 13 21:19); `cipher_rt_arbitrate.c` (321 LOC),
  `cipher_rt_sm_packer.c` (117 LOC).
- What shipped: ARBITRATE now calls `CIPHER_REQUEST_SM_PARTITION`
  (closes T4.2.2 Gap 1 — mask `0xff` visible). SM_PACKER is
  **detection-only — no kernel substitution** (report's own words).
- Why PARTIAL: SM_PACKER does not pack. The WL05 headline carries the
  **same retraction** as T4.2.2 (libcipher_v2 loaded, not libcipher_rt).
  The report is candid: the mask is "metadata… a mechanical TPW lift is
  not possible in this build." Findings note the quartile policy is
  data-starved at first-launch (all 8 tenants tie at 0).
- Verdict: **PARALLEL.**

#### T4.2.4a — GREEN_CTX scaffold — SHIPPED (plumbing only)
- Evidence: `PHASE_4_T4_2_4a_REPORT.md`; `libcipher_rt.so.v0.2.0_T4_2_4a`
  (38128 bytes, May 14 04:32, report md5 `77589b98...`);
  `cipher_rt_green_ctx.c` (268 LOC).
- What shipped: lazy per-process green ctx covering **all 132 SMs**
  (no partition restriction) + cuStreamCreate interception. Matched-pair
  WL01 Δ −0.52% TPW — within ±0.53% noise. Honest: "value is
  architectural"; "NOT independently verified… that the new streams are
  actually bound."
- SHIPPED is appropriate for what it claimed (a no-op plumbing layer).
- Verdict: **PARALLEL.**

#### T4.2.4b / b_v2 — per-tenant GREEN_CTX partition — PARTIAL
- Evidence: `PHASE_4_T4_2_4b_REPORT.md`; builds `..._T4_2_4b` and
  `..._T4_2_4b_v2` on disk (38360 / 38400 bytes).
- What shipped: real `cuDevSmResourceSplitByCount` → 8-SM green ctx;
  `cuGreenCtxGetDevResource` confirms 8 SMs at the API surface.
- Why PARTIAL: v1 had a 100% hash collision (all 8 tenants → group 5).
  v2 fixed the hash but the WL05 result was **Δ −0.43% TPW — neutral
  within noise. No measurable lift.** Report's honest finding: WL05
  decode is HBM-bound; SM partitioning "has no mechanical foothold."
- Verdict: **PARALLEL.**

#### T4.2.4c — noisy-neighbor A/B (cross-resource) — NOT DONE
- Evidence: `PHASE_4_T4_2_4c_REPORT.md`; `cipher-phase4-evidence/t4_2_4c/`
  (A_summary.json, B_summary.json present).
- Outcome: the tail-latency-isolation hypothesis was **not supported.**
  The decisive diagnostic: bomb prefills/s identical A↔B (93.8 vs 94.4)
  — proof the GREEN_CTX actuator was **cosmetic, not enforcing.** p99 A
  vs B 34.56 vs 34.13 ms — statistically indistinguishable.
- Verdict: **PARALLEL.** Reframe doc `PHASE_4_T4_2_4_LIFT_REFRAME.md`
  concedes the aggregate-throughput-lift hypothesis is "mechanically
  unsupported" on WL01–WL24.

#### T4.2.4d — GREEN_CTX enforcement fix — SHIPPED (enforcement); isolation FALSIFIED
- Evidence: `PHASE_4_T4_2_4d_REPORT.md`; `libcipher_rt.so.v0.2.0_T4_2_4d`
  (39056 bytes, May 14 13:17, report md5 `50414674...`);
  `cipher-phase4-evidence/t4_2_4d/`.
- What shipped: persistent `cuCtxSetCurrent(green)` at every cuLaunchKernel
  ENTER. Binding diagnostic met: bomb solo 114.1 → 10.4 prefills/s
  (**11× slowdown**, threshold <40). PyTorch routes 100% of launches
  through the NULL stream — root cause established by a diagnostic build
  (`on_null_stream=884493/884493`).
- Honest finding: with enforcement real, victim p99 is **+8.0% WORSE**
  under partition — the tail-latency-isolation hypothesis is **cleanly
  falsified on this cross-resource workload pair.** The genuine product
  capability is compute-share enforcement (bomb power −65%).
- SHIPPED is correct for "enforcement closed"; the isolation claim is
  honestly marked falsified.
- Verdict: **PARALLEL.**

#### T4.2.4e — tail-latency isolation (same-resource pair) — PARTIAL
- Evidence: `PHASE_4_T4_2_4e_REPORT.md`; `cipher-phase4-evidence/t4_2_4e/`.
- Result: two Mistral-7B prefill tenants A/B. A's absolute p99 is
  **11.8× WORSE** than B's (1707 vs 144 ms) — strong-form prediction
  refuted. A's tail-tightness (max/mean 1.04–1.20 vs B's 3.03–3.60) is
  **3× tighter** — the isolation property is real but lives in the
  >p99 distribution, and the report concedes "we don't have enough
  data to resolve p99.9 cleanly."
- Why PARTIAL: the headline isolation claim was refuted on the standard
  metric (p99, absolute latency); only a secondary metric (max/mean)
  confirms it, at a 12× absolute-latency cost.
- Verdict: **PARALLEL.**

**T4.2.x equivalence argument.** The canonical Phase plan has no SM-partition
checkpoint. T4.2.x built a CUDA Green Context substrate that is
enforcement-verified (T4.2.4d) but whose product value (aggregate TPW
lift, tail-latency isolation) was searched for across five sub-phases and
**not demonstrated** in any binding form. It substitutes for no canonical
CP — it is parallel exploratory work that produced one real capability
(compute-share enforcement) and two refuted hypotheses.

---

### T4.3.x — VOLT / DVFS

#### T4.3.0 — discovery — SHIPPED (discovery doc)
- Evidence: `PHASE_4_T4_3_0_DISCOVERY.md`. Read-only investigation, no
  code. Headline finding: NVML clock-lock works on this pod today
  (`nvidia-smi -lgc` RC=0), invalidating the plan's "bypass NVML via
  BAR0" premise. No Hopper clock register map exists in OGKM/nouveau.
- Verdict: **PARALLEL.** A discovery doc, not a canonical CP.

#### T4.3.1 — VOLT NVML clock-lock — SHIPPED (mechanism); user-process DEGRADED
- Evidence: `PHASE_4_T4_3_1_REPORT.md`; `libcipher_rt.so.v0.2.0_T4_3_1`
  (47992 bytes, May 14 14:44, report md5 `bcbadcdb...`);
  `cipher_rt_volt.c` (380 LOC); `cipher-phase4-evidence/t4_3_1/`
  (summary.json + A/B watts CSVs present).
- What shipped: signal-safe NVML actuator. Headline at B=1 decode:
  −35.7% watts, +3.1% tok/s, **+60.4% tok/W**.
- Honest caveat (in the report): the libcipher_rt user-process path is
  **DEGRADED** — NVML returns NOT_SUPPORTED from non-root injection.
  The +60% was measured with the lock applied externally via
  `sudo nvidia-smi`, not by the library autonomously.
- Verdict: **PARALLEL.**

#### T4.3.2 — kmod CIPHER_SET_CLOCK_MHZ ioctl — SHIPPED
- Evidence: `PHASE_4_T4_3_2_REPORT.md`; kmod 0.4.6
  (`cipher_kmod.ko.v0.4.6` on disk, 1729144 bytes, May 14 15:03, report
  md5 `119cb583...`); `cipher_clock.c` (146 LOC on disk);
  `cipher-phase4-evidence/t4_3_2_npairs/` (pair1..5 off/on + summary.json).
- What shipped: ioctl nr 10 → `call_usermodehelper` nvidia-smi; libcipher_rt
  VOLT falls through to the kmod ioctl when NVML refuses. Non-root
  invocation verified in dmesg (`uid=1000`). 5 matched pairs:
  **−36.24% watts ± 0.12% (z=314), +57.28% tok/W (z=180), tok/s
  indistinguishable from zero.** This is the strongest statistical
  evidence in the whole T4.x series.
- Verdict: **PARALLEL.** The closest thing to a clean ship in T4.x. But:
  it is a DVFS power lever, mapping to no canonical CP.

#### T4.3.x — envelope characterization — SHIPPED (and narrows the headline)
- Evidence: `PHASE_4_T4_3_ENVELOPE.md`; `cipher-phase4-evidence/t4_3_envelope/`
  (C1..C7 per-condition summary.json + watts CSVs).
- The honest correction: the +57% tok/W reproduces (C6 +54.97%) **only
  on TinyLlama-class memory-bound decode.** On Mistral-7B (the 7B+
  production regime), VOLT is **−2% to −15% tok/W — neutral to
  negative.** The report states plainly the "B200-parity" framing "is
  not supported" by the data. The headline number is real but narrow:
  it does not generalize to production-realistic workloads.
- Verdict: **PARALLEL.**

**T4.3.x equivalence argument.** DVFS / clock-locking is not a canonical
checkpoint. T4.3.2 is a genuine, statistically airtight capability
(−36% watts, non-root) — but the envelope work shows the moat is a
small-model-decode lever, not a general efficiency moat. It substitutes
for no canonical CP.

---

### T4.5 — matmul-routing substrate + Marlin actuator

#### T4.5.1 substrate — SHIPPED
- Evidence: `PHASE_4_T4_5_REPORT.md`, `PHASE_4_T4_5_0_DISCOVERY.md`;
  `libcipher_rt.so.v0.2.0_T4_5` on disk (120264 bytes, May 14 18:18);
  `cipher_rt_cublas_shim.c` (140 LOC), `cipher_rt_matmul_dispatch.c`
  (123 LOC).
- What shipped: `.symver`-tagged cuBLAS interception (50 LOC + 4-line
  version script); 16-slot actuator registry. 155/155 cublasGemmEx calls
  intercepted, **byte-identical output** with no actuator; overhead
  −1.69% ± 1.17% (indistinguishable from zero).
- SHIPPED is appropriate — the substrate itself is verified.
- Verdict: **PARALLEL.**

#### T4.5.2 Marlin INT4 actuator — NOT DONE (product number unmeasured)
- Evidence: same report, addendum V1–V5; `cipher-phase4-evidence/t4_5_marlin/`
  (pair1..3 off/on); `cipher_rt_marlin_engine.cpp` (846 LOC).
- Outcome: Marlin landed on the substrate and is accurate on an isolated
  matmul (cos=0.997). But on full TinyLlama logits **argmax flips**
  (cos=0.94) and **throughput regresses −32.4% tok/s**. The Mistral-7B
  validation attempt (V2–V4) **hung in the first decode iteration, both
  runs killed at 15-min timeout** — accuracy and throughput unmeasured.
  Outcome class γ (regression/neutral) per the report's own enumeration.
- The report frames Marlin as "outside its designed regime" — true, but
  the deliverable did not produce a passing product number anywhere.
- Verdict: **PARALLEL.** The MFU moat is explicitly "Open" in the report.

**T4.5 equivalence argument.** The matmul substrate is a deployment-layer
mechanism with no canonical-CP analog. Marlin is the intended MFU
gap-closer but did not clear its gate. Parallel work, substitutes for
nothing.

---

### T4.6.x — attention substrate + KV allocator + KV dedup

#### T4.6.0 / T4.6.0.5 — KV-dedup discovery + measurement — SHIPPED (as analysis)
- Evidence: `PHASE_4_T4_6_0_DISCOVERY.md`, `PHASE_4_T4_6_0_5_MEASUREMENT.md`;
  `cipher-phase4-evidence/t4_6_0_5_measurement/` (block_hashes.csv 67,741
  rows, summary.json).
- What shipped: synthetic-trace dedup measurement. **The synthetic
  mixed-workload number is 1.19× at block=16, N=8 — i.e. below the 2×
  L1 threshold (outcome (a) PIVOT on a strict reading).** The architecture
  commit to build L1+L3 anyway rests on a triangulation argument (prior
  art + structural math), not on CIPHER's own measurement. This is an
  honest framing in the doc but worth flagging: the build decision was
  made on external prior art, not on a measured opportunity.
- Verdict: **PARALLEL.** A measurement/decision doc.

#### T4.6.1 — attention-routing substrate — PARTIAL
- Evidence: `PHASE_4_T4_6_1_REPORT.md`, `PHASE_4_T4_6_1_SIGNATURE.md`;
  `cipher_rt_attn_dispatch.cpp` (390 LOC).
- What shipped: LD_PRELOAD interposition on 3 SDPA dispatcher entries;
  byte-identical output on Mistral-7B; composes with T4.5.1.
- Why PARTIAL: two material defects surfaced later (documented in
  `op_2_AUDIT.md` and `pause_note.md`). (1) A **FakeTensor crash** —
  the trampoline called `data_ptr()` unconditionally; under
  `torch.compile` this crashed the host. (2) **Coverage is 14% of
  decode attention** (96 of 672 expected calls) — the `.symver` ATen-op
  layer sits above `torch.compile`; compiled/cudagraph decode bypasses it.
  The substrate is "substantially blind to compiled decode." Flash and
  efficient trampolines are runtime-untested.
- Verdict: **PARALLEL.**

#### T4.6.2 — KV page allocator + from_blob bridge + CIPHER cache class — SHIPPED
- Evidence: `cipher-fusion-evidence/op_1_S2b_allocator_integration.md`;
  `cipher_rt_kv_alloc.c` (680 LOC on disk); `cipher_kv_bridge.cpp`
  (121 LOC); `cipher_kv_cache.py` (129 LOC); bridge .so md5
  `2cf82c06f3a7a4732ba59abb58f93b9c` (matches report).
- What shipped: VMM 2 MiB page allocator wired into transformers 5.8.1
  `StaticCache` via subclassed cache layers. Three-indicator diagnostic
  fires: `slabs_created=192`, page tags 6/6 correct, Mistral KV
  `memory_allocated` 4.9 → 0.0 MiB (KV fully CIPHER-owned). Byte-identical
  output on both cache paths; allocator unit test 14/14.
- Honest gap (in the report): "No memory saving yet" — T4.6.2 delivers
  ownership + tagging, not a footprint win. SHIPPED is correct for that
  scoped claim.
- Verdict: **PARALLEL.**

#### T4.6.3 — KV page-level dedup (content-hash) — PARTIAL
- Evidence: `cipher-fusion-evidence/t4_6_3_dedup_report.md`;
  `t4_6_3_dedup/sim_results.json` (md5 `c52bf2ca...`), `dedup_harness.c`,
  `traces/` (3 .jsonl, ~8.6 MB Mooncake-derived).
- What shipped: content-hash dedup in `cipher_rt_kv_alloc.c` (xxhash64
  + full-page memcmp verify). Phase 1 (theoretical): toolagent 50.62%
  page dedup. Phase 2: allocator captures real/sim = 1.000 across 15
  windows; 0/60000 SHA-256 mismatches; refcount-clean.
- Why PARTIAL — the report says so itself in a "What Phase 2 does NOT
  validate" contract: **single-process only** (cross-process is the
  actual moat); **synthesized pages, not live decode KV**; Phase 1 is an
  **upper bound** (block-id equality assumed = byte-identical). The
  mechanism is verified on synthetic data; the multi-tenant moat claim
  is explicitly "earned only at the end of T4.6.5."
- Verdict: **PARALLEL.**

#### T4.6.4 — kmod-owned cross-tenant KV dedup — SHIPPED (mechanism)
- Evidence: `cipher-fusion-evidence/t4_6_4_report.md`, `t4_6_4_design.md`;
  `cipher_kmod/cipher_kvdedup.c` (461 LOC on disk), `cipher_kvdedup.h`
  (86 LOC); `cipher_kmod.ko` on disk md5 **`b263ad30453d9f620d4f279627c7258e`
  — matches the report exactly**; `/dev/cipher_kvdedup` present
  (major 510) and `cipher_kmod` loaded (verified via lsmod).
  Test harness `t4_6_4_kvdedup/kvdedup_xproc.c` (335 LOC) + compiled
  binary on disk.
- What shipped: dedup refcount table + cuIpc POSIX-FD handles moved into
  the kmod. Five binding indicators all PASS: (a) **15,500 pages
  verified cross-process, 0 fail** — tenant B imported tenant A's
  physical pages via cuIpc through the kmod, SHA-256-matched; (b) tenant
  teardown via release fop survives graceful close AND SIGKILL;
  (c) slab unit test 14/14; (d) refcount integrity under 4-tenant churn
  (misses=2987, releases=2987); (e) module unload safety.
- Honest gaps (in the report): `/dev/cipher_kvdedup` is 0600 root (the
  0660/group-cipher target is operator udev policy, not shipped); dedup
  fires on the cold-path `dedup_put`, **not yet wired into the live
  decode write-path** (deferred to T4.6.5); validation on **real model
  KV bytes** is still pending (Phase 2/T4.6.3 used synthesized content).
  Note: the implementation uses a **separate `/dev/cipher_kvdedup` char
  device**, not the ioctl nrs 11/12/13 on `/dev/cipher` that the
  T4.6.0.5 plan envisaged — an ABI design change from the plan.
- SHIPPED is defensible for "cross-process dedup mechanism, kmod-owned,
  five indicators pass." The product-moat claim (real-KV, live-decode,
  multi-tenant density) is NOT shipped — it is the named content of
  T4.6.5 and beyond.
- Verdict: **PARALLEL.**

**T4.6.x equivalence argument.** The canonical plan has no KV-dedup or
cross-tenant-page-pool checkpoint. T4.6 built a real, kmod-mediated
cross-process KV-page-sharing mechanism (T4.6.4 — genuinely verified at
the mechanism level) on top of an allocator (T4.6.2) and an attention
substrate (T4.6.1, which is itself only 14%-coverage on compiled decode).
The end-to-end product value — tenant-density lift on real production
KV — remains unmeasured (gated to T4.6.5). It substitutes for no
canonical CP; it is a parallel multi-tenant-moat track.

---

## 3. Cross-cutting honest findings

1. **The WL05 SM-partition headline numbers in T4.2.2 and T4.2.3 were
   retracted.** A hardcoded `CUDA_INJECTION64_PATH` in the runner loaded
   `libcipher_v2`, so every WL05 measurement in those two reports
   measured the wrong library. The reports carry the correction; an
   auditor should treat the original WL05 progression tables as void.

2. **The SM-partition product hypothesis failed across five sub-phases.**
   T4.2.4b: no aggregate TPW lift. T4.2.4c: actuator cosmetic, no
   isolation. T4.2.4d: enforcement fixed but isolation falsified.
   T4.2.4e: only a secondary metric (max/mean) confirms isolation, at
   12× absolute-latency cost. The one durable T4.2 capability is
   compute-share enforcement (bomb power −65%).

3. **The VOLT +57% tok/W headline is real but narrow.** T4.3.2's n=5
   matched-pair statistics are airtight (z=314 on watts). But the
   T4.3.x envelope work shows it is a TinyLlama-class memory-bound-decode
   lever; on Mistral-7B it is −2% to −15%. The "B200-parity" framing is
   explicitly unsupported by the project's own data.

4. **Marlin (the MFU lever) never cleared a gate.** Regressed on
   TinyLlama, hung on Mistral-7B. The matmul substrate it rides on is
   solid; the actuator's product number is unmeasured.

5. **The T4.6.1 attention substrate has a 14%-decode-coverage problem.**
   The `.symver` ATen-op layer sits above `torch.compile`; compiled
   decode bypasses it. `pause_note.md` records that interception-based
   actuators are blocked until a cuLaunchKernel-layer hook exists. This
   undercuts any future actuator that rides the attention substrate —
   though T4.6.2/3/4 (data-ownership layer) are coverage-immune.

6. **The KV-dedup opportunity was decided on prior art, not on CIPHER's
   own measurement.** T4.6.0.5's synthetic number was 1.19× (a strict
   "pivot" reading); the L1+L3 build commit rests on external prior art.
   The honest validation gate (real production traces, T4.6.5) has not
   yet run.

7. **On-disk artifacts are consistent with the reports.** kmod.ko md5
   `b263ad30...` = T4.6.4 report; bridge .so `2cf82c06...` = op_1 report;
   `/dev/cipher_kvdedup` is live. The live `libcipher_rt.so` md5 is
   `4f5cf543...` — the post-op2-audit FakeTensor-fix build per
   `pause_note.md`, i.e. newer than the T4.6.1 report's build.

---

## 4. The single most important finding

**Of ~18 T4.x.x deliverables, the substrate / mechanism layers are real
and verified; the product-value / lift claims are mostly partial,
narrowed, or refuted — and none of it supersedes a canonical checkpoint.**

What is genuinely SHIPPED and survives scrutiny: the cuBLAS matmul
substrate (T4.5.1), the kmod clock ioctl (T4.3.2, with airtight n=5
statistics), the KV allocator (T4.6.2), and the kmod-owned cross-process
KV-dedup mechanism (T4.6.4, five indicators pass). What did NOT survive:
the SM-partition aggregate-throughput and tail-latency hypotheses (five
sub-phases, refuted or unsupported), Marlin's product number (regressed
/ hung), the VOLT moat's generality (TinyLlama-only; negative on 7B+),
and the attention substrate's decode coverage (14%).

The pattern is consistent: CIPHER's T4.x work reliably builds and
verifies *substrates and mechanisms*, then reliably *fails to demonstrate
the headline product lift* those substrates were meant to carry. Every
T4.x.x deliverable is **PARALLEL** — the canonical Phase 0-9 plan has no
SM-partition, DVFS, matmul-routing, or KV-dedup checkpoint, so none of
this work substitutes for a canonical CP. It is a large parallel R&D
track whose engineering substrate is sound but whose marvel-pitch
numbers (efficiency moat, MFU moat, multi-tenant moat) remain either
narrowed to a small regime or unproven pending T4.6.5 real-trace
validation.
