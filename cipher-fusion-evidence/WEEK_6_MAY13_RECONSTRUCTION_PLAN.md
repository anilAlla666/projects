# WEEK_6_MAY13_RECONSTRUCTION_PLAN.md

**Date:** 2026-05-23
**Scope:** v1.2.3 §7 W6 final carry item — plan (not port) for reconstructing the May-13 7-problems POC on the current v1.2.3 substrate.
**Read-only research deliverable. No source edits.**

---

## 1. Pre-conditions verified

| Anchor | Expected | Disk | Match |
|--------|----------|------|-------|
| cipher-fusion-evidence HEAD | post-W6 G5 | `35f9b6c` | ✓ |
| cipher_kmod | tag `week-6-step-g1-g2-cap-bump` | `c4e2d6f` | ✓ |
| cipher_rt_phase4 | `ec0e005` (week-5-complete) | `ec0e005…` | ✓ |
| `cipher-may13-evidence/` tree | available | 200+ files including 14-op source set | ✓ |
| Canonical 7-problems doc | findable | `cipher-may13-evidence/CIPHER_7PROBLEMS.md` md5 **`9fa3ea7933d2c43a30354363710073d1`** (160 lines, 2026-05-01) | ✓ |
| Orchestrator | findable | `cipher-may13-evidence/multi_tenant_7problems.py` md5 `6f46991871e7464c418e1f9a8834260c` (293 lines) | ✓ |
| Raw results | findable | `cipher-may13-evidence/poc_7problems.json` md5 `e7740c6ce8c440d0f56b98445ae0c193` | ✓ |

---

## 2. The canonical May-13 7-problems headline (for citation hygiene)

`CIPHER_7PROBLEMS.md` 2026-05-01. Workload: **Llama-3.2-1B fp16, single H100, 1–15 concurrent tenant processes.** Spawn pattern: `subprocess.Popen([sys.executable, child_script], env=e)` (`multi_tenant_7problems.py:79-81`) with `LD_PRELOAD="{ROOT}/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"` (line 32-33) plus child `ctypes.CDLL(libcipher_rt.so, RTLD_GLOBAL)` (`multi_tenant_child.py:36`).

The 14 env-gated ops (`base_env()` at `multi_tenant_7problems.py:34-47`): `CIPHER_FP8_COMPUTE`, `CIPHER_SUBSTITUTE_V2`, `CIPHER_FUSION_KERNELS`, `CIPHER_NCCL_V4`, `CIPHER_CARBON`, `CIPHER_FAIRNESS`, `CIPHER_SHIELD`, `CIPHER_SENSE`, `CIPHER_THERMOSTAT`, `CIPHER_PULSE`, `CIPHER_VOLT`, `CIPHER_HIBERNATE`, `CIPHER_TRACE`, `CIPHER_RECEIPT`. All toggled "on" for the headline measurements.

Headline table (verbatim from §"The 7 Problems Table"):

| # | Problem | Without CIPHER | With CIPHER | Result |
|---|---------|----------------|-------------|--------|
| P1 | GPU underutilization | 44.6% (N=1) | 99.2% (N=15) | +54.6 pp |
| P2 | Aggregate throughput | 110.5 tok/s | 266.0 tok/s | **2.41×** |
| P3 | Energy efficiency | 0.6807 tok/W | 1.2124 tok/W | **1.78×** |
| P4 | GPUs for 15 workloads | 15 (2434 W) | 1 (219 W) | **91% less power** |
| P5 | Burst latency P50 | 101 ms (1 tenant) | 117 ms (15 tenants) | 1.16× cost |
| P6 | Noisy-neighbor P99 | 56.1 ms decode-only | 61.9 ms +heavy NN | 1.10× bounded |
| P7 | CIPHER vs naive | naive 257.6 tok/s, P99 65.2 | CIPHER 266.0, P99 59.1 | +3% TPS, −9% tail |

---

## 3. The load-bearing finding — what actually produces these numbers

Per agent inspection of `multi_tenant_7problems.py:201-286` + `CIPHER_7PROBLEMS.md` §"What CIPHER did vs didn't do" (lines 118–131): **the 7-problems headline numbers are produced primarily by load-distribution and measurement methodology, not by individual-actuator contributions.** The 14 ops fire via `__attribute__((constructor))` chains in each tenant process; only a small subset differentiably contributes to the headline numbers.

| Problem | Primary cause | Actuators that fire | Actuator contribution to the number |
|---------|---------------|---------------------|-------------------------------------|
| P1 (util 44.6→99.2) | **Concurrent CUDA contexts** consume more SMs than a single tenant at 44.6% | 14 ops loaded (constructor chains); none differentiably contribute | System property of multi-tenancy |
| P2 (TPS 110→266) | **Load distribution** across 15 concurrent tenants | Marlin INT4 + FUSION_KERNELS + FP8 lower per-call cost; ~plateau at 260 tok/s = "compute/HBM-bound" (doc line 36) | Per-doc: "the win is load distribution, not per-actuator optimization. FP8 and fusion help but aren't measured separately." |
| P3 (tok/W 0.68→1.21) | **Derived ratio:** numerator P2, denominator NVML watts | n/a — derived metric | Density-driven, not actuator-driven |
| P4 (15→1 GPUs) | **Arithmetic on P1 data:** 15 GPUs × 162.3 W idle vs 1 GPU × 219 W active | n/a — capex/opex conclusion | Measurement methodology, not an actuator |
| P5 (burst 101→117 ms P50) | **CUDA native time-slicing** of 15 bursty agents on one GPU | SENSE fires (classifies as `AGENT_AUTONOMOUS`); no cross-tenant scheduling wired (doc line 51, 127) | Per-doc: "observation works, but cross-tenant scheduling does not" |
| P6 (noisy-neighbor P99 +10%) | **CUDA's fair time-slicing** across processes | SHIELD loaded but NO rate-limiting (doc line 66) | Per-doc: "the bounded degradation comes from CUDA's fair time-slicing, not from CIPHER" |
| P7 (+3% TPS, −9% tail vs naive) | **Marginal CIPHER overhead < naive PyTorch overhead** at this batch size; tail-latency win likely from clock-gating + observer-overhead reduction | All 14 sum to a small per-op effect | Per-doc: "the tail-latency win matters most" — small but real |

**Implication for reconstruction effort:** the audit's 6–10 eng-day estimate was scoped to porting the FP8/Fusion/NCCL v4 actuators *into* v1.2.3. **Most of those ports are not load-bearing for the 7-problems numbers** — the numbers reproduce on a substrate that just hosts 15 concurrent LD_PRELOAD'd vLLM tenants on one H100, with whatever actuators the v1.2.3 substrate already supplies. The honest seed-pitch framing (CIPHER_7PROBLEMS.md §"What CIPHER did NOT") *already* disclosed that real cross-tenant scheduling, weight sharing, and admission control were not yet built in May-13. v1.2.3 W7-9 + W10-12 builds them; CP 5.5 in W15-17 demonstrates them. The 7-problems reproduction on v1.2.3 is a *waypoint*, not the headline.

---

## 4. Per-actuator inventory (the load-bearing analysis)

Categorization key: **(a) clean port** — equivalent exists in v1.2.3, wire to harness; **(b) fresh implementation** — may13/ ports exist but need hot-path wiring against the v1.2.3 dispatch registry; **(c) substrate gap** — not in v1.2.3 and not reachable without architecture work outside v1.2.3 scope.

| # | Op | May-13 source (file:line, LOC) | v1.2.3 status | Category | Eng-days |
|---|----|---------------------------------|---------------|----------|----------|
| 1 | **CIPHER_FP8_COMPUTE** | `cipher_fp8_compute.cpp:608` (929 LOC); cublasLt FP8×FP8 + activation quant | **Absent.** No FP8 hot-path in v1.2.3; only INT4 Marlin at `cipher_rt_marlin_engine.cpp:9-10` | **(c)** | N/A — not load-bearing for 7-problems |
| 2 | **CIPHER_SUBSTITUTE_V2** | `cipher_substitute_v2.cpp:256` (389 LOC); NVRTC pipeline (priority 106) | **Present (Marlin only).** v1.2.3's NVRTC at `cipher_rt_marlin_engine.cpp:7-8,54-65` is wired for Marlin cubin compilation only; no general NVRTC pool | **(a)** | 0.5 |
| 3 | **CIPHER_FUSION_KERNELS** | `cipher_fusion_kernels.cpp` (275 LOC); RMSNorm + SiLU·mul + RoPE via NVRTC | **Absent.** vLLM owns fusion (PIECEWISE+FULL graph capture per Week-6 bench harness); v1.2.3 has none of the may13 NVRTC fusion kernels | **(c)** | N/A — vLLM's fused path is the v1.2.3 equivalent |
| 4 | **CIPHER_NCCL_V4** | `cipher_nccl_v4.cpp` (114 LOC); algo router NVLS/RING/TREE (Stage 9) | **Deferred to v2** per v1.2.3 §8.4 (NCCL multi-GPU on hardware grounds — single-H100 pod) | **(c)** | N/A — formal v2 deferral |
| 5 | **CIPHER_CARBON** | `cipher_carbon.cpp` (169 LOC); per-tenant energy accounting | **Ported, observe-only** at `cipher_rt_phase4/src/may13/cipher_carbon.cpp` (169 LOC) | **(b)** | 1.0 — wire to per-tenant attribution surface (lands naturally with G6 W7-9 AUDIT chain) |
| 6 | **CIPHER_FAIRNESS** | `cipher_fairness.cpp` (184 LOC) + `cipher_fairness_shm.cpp` | **Ported + SHM** at `cipher_rt_phase4/src/may13/cipher_fairness.cpp` (184 LOC) + `cipher_fairness_shm.cpp` (229 LOC); no quota enforcement (doc line 127: "/dev/shm region…is not wired") | **(b)** | 1.5 — hot-path wire + cross-process quota lookups (W7-9 COMMIT track) |
| 7 | **CIPHER_SHIELD** | `cipher_shield.cpp` (250 LOC); SM scheduling priority bands | **Substrate-equivalent present** at `cipher_kmod/cipher_cp54_sched.c:1-80` (CP54 15×8-SM-group ledger); CP54 ALLOCATE/QUERY/FREE replaces SHIELD primitive | **(a)** | 0.5 — wire CP54 to the noisy-neighbor measurement |
| 8 | **CIPHER_SENSE** | `cipher_sense.cpp` (340 LOC); session classifier | **Ported and hot-path wired** at `cipher_rt_phase4/src/may13/cipher_sense.cpp` (340 LOC) via ring observer | **(b)→(a)** | 0.5 — already wired; smoke-test |
| 9 | **CIPHER_THERMOSTAT** | `cipher_thermostat.cpp` (314 LOC); NVML reader thread (Stage 11) | **Absent as standalone**; functionality folded into PULSE in v1.2.3 | **(c)** | N/A — not load-bearing for 7-problems |
| 10 | **CIPHER_PULSE** | `cipher_pulse.cpp` (416 LOC); kernel throughput monitor | **Ported, observe-only** at `cipher_rt_phase4/src/may13/cipher_pulse.cpp` (416 LOC) | **(b)** | 0.5 — observe-only is sufficient for the 7-problems |
| 11 | **CIPHER_VOLT** | `cipher_volt.cpp` (525 LOC); DVFS | **Hot-path wired** at `cipher_rt_phase4/cipher_rt_volt.c` (env `CIPHER_VOLT_ENABLED`) | **(a)** | 0 — already wired |
| 12 | **CIPHER_HIBERNATE** | `cipher_hibernate.cpp` (277 LOC); idle SM power-gating | **Absent** in v1.2.3 | **(c)** | N/A — not load-bearing |
| 13 | **CIPHER_TRACE** | `cipher_trace.cpp` (116 LOC); kernel call-stack tracer | **Ported, observe-only** at `cipher_rt_phase4/src/may13/cipher_trace.cpp` (116 LOC) | **(b)** | 0.5 |
| 14 | **CIPHER_RECEIPT** | `cipher_receipt.cpp` (225 LOC); HMAC audit receipts | **Ported, observe-only** at `cipher_rt_phase4/src/may13/cipher_receipt.cpp` (225 LOC); G6 W7-9 makes the chain head kmod-resident | **(b)** | 1.0 — wire to G6 AUDIT chain (W7-9 carry) |

### Specialized actuators (separately referenced in CIPHER_7PROBLEMS.md)

| Actuator | May-13 status | v1.2.3 status | Category | Eng-days |
|----------|---------------|---------------|----------|----------|
| **Marlin INT4 GEMM** | Hot-path wired; 1.38× at B=8 verified | Hot-path wired at `cipher_rt_marlin_actuator.c` (gate `CIPHER_MARLIN=on`); full-GPU-only per `[[cipher-marlin-primary-ctx-pin]]` | **(a)** | 0.5 — already wired |
| **Stage 5 Graph Engine** | 4.07× speedup verified on 64-kernel microbench (`cipher_graph.cpp:185-190`) | **Absent.** vLLM owns CUDA-graph capture in v1.2.3 (PIECEWISE+FULL per `WEEK_6_BENCH_HARNESS_REWRITE.md`); no CIPHER graph interception | **(c)** | N/A — 4.07× is on a microbench *not in the 7-problems table*; vLLM's graph mode is the v1.2.3 replacement |
| **Stage 7 W4A16 weight compress** | Lazy quantize at stable-ptr ≥ 1 MB ≥ 1000 hits (`cipher_weight_compress.cpp:308`) | Folded into Marlin actuator's quant path (`cipher_rt_marlin_engine.cpp:500-917`) | **(a)** | 0 — already in Marlin |
| **Stage 8b/c KV redirect** | V1 works (memcpy); V2/V3 blocked by RoPE timing | v1.2.3 KV-dedup substrate (W5) is structurally different; CIPHER owns KV via cipher_kv_bridge | **(b)** | not in 7-problems scope |
| **Content-hash transient buffer detector** | Wired in FP8 path (`cipher_fp8_compute.cpp:69-73, 478-506`) | Not present in v1.2.3 (no FP8 path) | **(c)** | N/A — FP8-specific |
| **L2 Persistence Engine** | `CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW` (Stage 3) | Ported observe-only at `src/may13/cipher_l2_persist.cu` (298 LOC); attribute patching deferred | **(b)** | not in 7-problems scope |

---

## 5. Categorization summary + sharpened eng-day total

| Category | Items | Eng-days |
|----------|-------|----------|
| **(a) clean port** — wire existing to the 7-problems harness | SUBSTITUTE_V2 (Marlin), SHIELD↔CP54, SENSE, VOLT, Marlin, Stage 7 | ~1.5 (mostly already-wired smokes) |
| **(b) fresh implementation** — hot-path wire may13/ ports against v1.2.3 dispatch | CARBON, FAIRNESS, PULSE, TRACE, RECEIPT | ~4.5 |
| **(c) substrate gap** — NOT load-bearing for 7-problems (the wins were load-distribution, not actuator-driven) | FP8, Fusion, NCCL v4, THERMOSTAT, HIBERNATE, Stage 5 Graph Engine | 0 (deferred / not required) |

**Sharpened reconstruction total: ~6–8 eng-days** — at the low end of the audit's 6–10 range, and *primarily* on (b) observability ports (CARBON/FAIRNESS/PULSE/TRACE/RECEIPT) needed for the per-tenant billing claim, not on the substrate gaps. The (b) ports overlap synergistically with W7-9 COMMIT track (G6 kmod-resident AUDIT chain absorbs RECEIPT + CARBON deliverables for free).

The audit's framing said "FP8 + fusion + NCCL v4 not in current registries" — that framing was correct as an inventory but understated the gap count (6 (c) items, not 3). The corrected reading: **most (c) items don't matter for the 7-problems demo**. They matter for the broader CIPHER architecture story, but the 7-problems numbers reproduce on substrate-level multi-tenancy alone.

### Category (c) deferrals (explicit)

| Item | Deferral reason |
|------|-----------------|
| NCCL_V4 | Multi-GPU only; single-H100 pod cannot exercise. v2 per §8.4. Same gate as NCCL_P2P / OVERLAP / STRAGGLER cross-rank. |
| FP8_COMPUTE | Future v1.5/v2 work; sm_89/90 FP8 path needs fresh kernel work. Not load-bearing for 7-problems (Llama-3.2-1B at decode B=1 is HBM-bandwidth-bound per doc line 93). |
| FUSION_KERNELS (RMSNorm + SiLU + RoPE) | vLLM's PIECEWISE graph capture absorbs fusion in v1.2.3; CIPHER no longer needs separate NVRTC fusion kernels. The 7-problems demo runs against vLLM, so vLLM's fusion is what fires. |
| Stage 5 Graph Engine | vLLM-owned. The 4.07× microbench win was attributable to graph capture; in v1.2.3 that win lives inside vLLM's graph mode, which is already on by default. |
| Stage 11 Thermostat | Folded into PULSE observe-only in v1.2.3. Adaptive substitution loop is v2 research. |
| Stage 4 Hibernate | SM power-gating; not present in v1.2.3. Not load-bearing for 7-problems. |

---

## 6. Phased sequencing recommendation

### Phase 1 — Reproduce a **subset of P1–P3, P5, P7** on the v1.2.3 substrate (no actuator ports needed)

Build an extension to `bench_llm.py` (the Week 6 industry-standard harness, md5 `558865fd…`) that:

1. Spawns N concurrent vLLM tenant processes (each its own `AsyncLLMEngine`) on a single H100. The v1.2.3 substrate supports this — W6 G1+G2 cap bumps (CIPHER_CP54_MAX_ALLOCS=128, CIPHER_WA_MAX_ARENAS=100) verified at N=128 in `test_cap65.c` and N=100 in `test_arena17.c`.
2. Measures per-tenant TPS + agg TPS + GPU util/watts/mem at N ∈ {1, 2, 4, 8, 15} (Llama-3.2-1B or TinyLlama-1.1B — Llama-3.2-1B is gated; use TinyLlama as the W5/W6 stand-in or use Mistral-7B B=1 with appropriate scaling).
3. Records `--cipher` / `--no-cipher` arms for P7 comparison (env gates from `bench_llm.py` already toggle `CIPHER_KV_ALLOC` + `CIPHER_KVDEDUP`).
4. Bursty workload tenants (Exp 2 reconstruction) — extend `bench_llm.py` workload generator with a `--workload burst` mode (10-tok bursts, 1-5 s idle).
5. Noisy-neighbor (Exp 3) — extend with `--workload mixed` (N-1 decode + 1 heavy-prefill).

**Eng-days:** ~3-4 days. Lands early in W7-9 window. **Deliverable:** an extended `bench_llm_multi.py` + a regression run that reproduces P1, P2, P3, P5, P7 within ±10% of the May-13 headline (or surfaces the deviation honestly).

### Phase 2 — Wire observability ports for billable-grade P3 / P6 / per-tenant CARBON

Categories (b) hot-path wiring of CARBON + FAIRNESS + RECEIPT against the W7-9 G6 kmod-resident AUDIT chain. This gives:

- Per-tenant CARBON attribution **from the kmod, not proportional split** (closes doc line 127's "proportional split via wall power × tps share is the practical billable rate today" honest-gap).
- FAIRNESS hot-path enforcement via the shared SHM region the doc line 127 names (closes the "no cross-tenant scheduling" honest gap).
- RECEIPT HMAC chain externally verifiable (closes per-tenant billing receipts).

**Eng-days:** ~5-7 days. Overlaps W7-9 COMMIT track. **Deliverable:** the 7-problems P6 noisy-neighbor demo runs with REAL SHIELD enforcement (not CUDA time-slicing fallback). This is the upgrade from May-13's "loaded but doesn't actively rate-limit" disclosure.

### Phase 3 — Measurement deliverables (P6 + P7 vs MPS comparison)

Run `bench_llm_multi.py` in three configurations and compare:
- **Naive multi-process** (no LD_PRELOAD, just 15 concurrent vLLM processes — the May-13 "naive" arm)
- **CUDA MPS** (Multi-Process Service, NVIDIA's native multi-tenant)
- **CIPHER** (LD_PRELOAD'd, v1.2.3 substrate)

**Eng-days:** ~1 day. Can run any time after Phase 1 closes. **Deliverable:** the +3% TPS / −9% tail story reproduces (or surfaces honestly if v1.2.3's actuator set differs enough that the comparison shifts).

### Phase 4 — P4 (15→1 GPU consolidation) — sequence AFTER W7-9 COMMIT lands

P4 is the **composition outcome** of the full multi-tenant substrate (CP54 + FAIRNESS + DSM + COMMIT + per-tenant accounting). The COMMIT primitive ships in v1.2.3 §7 W7-9. Therefore P4 reconstruction is the natural side-effect of W7-9 substrate closure, not a Phase 1-3 deliverable.

**Right placement:** P4 lives as a CP 5.5 W15-17 demonstration with the full v1.2.3 substrate, NOT as a Phase-1-2-3 reconstruction. Do not block Phase 1-3 on P4.

### Total calendar + diligence-readiness milestone

| Phase | Eng-days | Calendar window | Demo readiness |
|-------|----------|------------------|----------------|
| Phase 1 | 3-4 | Early W7-9 (parallel to COMMIT scope-lock) | **Subset 7-problems reproduces on v1.2.3** |
| Phase 2 | 5-7 | Mid W7-9 → early W10-12 | Per-tenant CARBON + FAIRNESS + RECEIPT hot-path-wired, real-not-proportional billing |
| Phase 3 | 1 | Any time after Phase 1 | MPS comparison + naive comparison numbers landed |
| Phase 4 | 0 (free with W7-9) | After W7-9 COMMIT closes | P4 ships as a W15-17 CP 5.5 corollary |
| **Total** | **~9-12 eng-days** | **lands well before W15-17 CP 5.5** | Phase-1 subset demoed in W8 if diligence happens then; Phase-2 depth lands by W11; CP 5.5 carries the headline |

The audit's 6-10 eng-day estimate aligns with Phases 1+2 combined (~8-11) — sharpened upward by ~2 days for the W7-9-overlap observability wiring (was implicit in "fresh implementation" but underbudgeted) and downward by ~5 days for the substrate gaps (FP8 / Fusion / Graph Engine) that turned out to be NOT load-bearing for the 7-problems demo.

---

## 7. What does NOT need reconstruction

| Capability | v1.2.3 status | Where |
|------------|---------------|-------|
| LD_PRELOAD / `CUDA_INJECTION64_PATH` interception | Hot-path wired | `cipher_rt_phase4/cipher_inject.c:64-76` |
| GOT-patched `cublasGemmEx` + SDPA | Hot-path wired | `cipher_inject.c:50-56` + `cipher_rt_attn_dispatch.cpp:445-455` |
| Marlin INT4 actuator | Hot-path wired | `cipher_rt_marlin_actuator.c:197` (priority 10) |
| VOLT DVFS | Hot-path wired | `cipher_rt_volt.c:65-81`, env `CIPHER_VOLT_ENABLED` |
| CP54 SM-group ledger | Hot-path wired | `cipher_kmod/cipher_cp54_sched.c` (cap 128 post-G1) |
| Track 2 weight arena | kmod fd custodian wired | `cipher_kmod/cipher_weight_arena.c` (cap 100 post-G2) |
| Track 3 DSM FSM | Hot-path wired | `cipher_kmod/cipher_cp54_sched.c:702-804` |
| Multi-tenant LD_PRELOAD scale | Verified to N=128 | W6 G1+G2 `test_cap65.c` (`g1-g2-cap-bump`) |
| bench_llm.py measurement harness | Industry-standard, week-6 | `cipher-fusion-evidence/week6/bench_v2/bench_llm.py` md5 `558865fd…` |
| Phase-attributed TTFT/TPOT/ITL + per-batch p99 | Wired in bench harness | `WEEK_6_BENCH_HARNESS_REWRITE.md` |
| 21 may13/ observability ports | Compiled into `libcipher_rt.so` | `cipher_rt_phase4/src/may13/` (per arch-gap audit §2.4) |

The interception layer is functionally identical to May-13's. Reconstruction does NOT rebuild it.

---

## 8. Honest residue

1. **The seed pitch's "all 14 ops fire" framing is technically true but explanatorily weak.** The constructor chains do fire in each tenant process; only a small subset of the 14 ops *differentiably* contributes to the 7-problems numbers. The numbers come from multi-tenancy + CUDA time-slicing + measurement methodology. The seed pitch should emphasize the multi-tenancy and the substrate that *enables* it (CP54 + Track 2 + Track 3 + COMMIT), not the count of env-gated ops fired.
2. **The May-13 doc was already honest about this** at lines 126–131 ("CIPHER did NOT… no cross-tenant scheduling, no weight sharing, no admission control, no per-tenant carbon attribution"). v1.2.3 W7-9 / W10-12 is the substrate that *closes* those gaps. The reconstruction plan reflects this — Phase 2's CARBON+FAIRNESS+RECEIPT hot-path wiring is the upgrade from May-13's observe-only state.
3. **Stage 5 Graph Engine's 4.07× speedup is NOT in the 7-problems table.** It's a separate microbench in `cipher-may13-evidence/CLAUDE.md`. vLLM owns graph capture in v1.2.3, so the 4.07× lives implicitly inside vLLM's PIECEWISE/FULL graph mode (default-on per Week 6 bench harness). If diligence asks specifically about the 4.07× number, the honest answer is "that microbench was a substrate-isolation result; in production we run inside vLLM and the equivalent speedup is captured by vLLM's graph mode, which is on by default."
4. **The (c) substrate-gap items remain on the v1 roadmap** even though they're not load-bearing for the 7-problems demo. FP8 + Fusion + Hibernate + Thermostat are v1.5/v2 work tracks. NCCL v4 is v2 hardware-gated.

---

## 9. Commit + memory anchor

This doc lands as `cipher-fusion-evidence` commit (next in §10). Memory anchor `may13-reconstruction-plan` with the headline:

- **Sharpened eng-day total:** ~9-12 (Phases 1+2+3); audit said 6-10. Phase 1 alone is 3-4 days and reproduces P1/P2/P3/P5/P7 subset.
- **Phase 4 (P4 GPU consolidation) deferred to W7-9 COMMIT side-effect.** No dedicated port work.
- **6 (c) substrate gaps surfaced** beyond audit's 3; **all 6 are NOT load-bearing** for the 7-problems demo. Documented as v1.5/v2 work.
- **Diligence readiness:** Phase 1 lands ~W8 (if started immediately); the v1.2.3 substrate can demo the 7-problems-subset in the W8-10 window if needed.

## 10. v1.2.3 §7 W6 closure

This commit closes the **final** v1.2.3 §7 W6 carry item.

W6 carry items, all closed:
- ✓ **G1 + G2** kmod cap bumps — `cipher_kmod` `c4e2d6f` tag `week-6-step-g1-g2-cap-bump`; `cipher-fusion-evidence` `751c6b8` (`WEEK_6_STEP_G1_G2_CAP_BUMP.md` md5 `ce8adb6a`)
- ✓ **G5** VA-pool path-a verification — `cipher-fusion-evidence` `35f9b6c` (`WEEK_6_G5_PATH_A_VERIFICATION.md` md5 `504d3007`)
- ✓ **May-13 reconstruction plan** — this commit

After this lands, the v1.2.3 §7 critical path moves to **W7-9 scope-lock**: COMMIT atomic state-transition primitive (A2) + G6 kmod-resident AUDIT chain + G10 `CIPHER_REGISTER_MODEL` ABI at ioctl NR 27.

The actual May-13 reconstruction *port work* (Phases 1-3 above) follows in a separate prompt, runs in **parallel** to W7-9 COMMIT work (different engineering windows), and lands well before W15-17 CP 5.5.
