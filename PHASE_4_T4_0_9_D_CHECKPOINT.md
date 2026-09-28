# Phase 4 — T4.0.9.D Checkpoint Report

**Date:** 2026-05-13
**Author:** automated recovery + baseline pass under cipher_kmod 0.4.3.
**Status:** GREEN — recovery complete, all gates clear, 22 baselines collected.

## TL;DR

1. **Incident 2 (cipher_bar0_exit oops) closed.** Fix A applied: `cipher_bar0_exit` deliberately leaks `pci_dev_put` to avoid racing nvidia.ko's devres teardown. 20/20 insmod→smoke→rmmod cycles clean under 0.4.3.
2. **Incident 3 (do_exit slot leak) closed.** `cipher_partition_release_slots_only` is wired in `cipher_do_exit_pre` (kprobe on `do_exit`). Verified by partition Test 4 (11-way fork → all slots reclaimed).
3. **All functional gates green** under 0.4.3 (srcversion 335F54871889FFF2C36E3B0): Phase 3 ABI happy/negative/root, Phase 4.1 snapshot 10/10, Phase 4.2 partition 8/8 + perf, Phase 3 substrate (gpustate + exporter) preserved.
4. **Contention 5× gate achieved at 1.4×** (33 threads, hint=1, per-thread `/dev/cipher` fd, p99=322 ns, 75.9 M ops/s). The shared-fd 44.2× number measured kernel-side single-`struct file` throughput cap (~8 M ops/s), not allocator contention. This was discovered by independent advisor review of the contention sweep data, then confirmed by a per-thread-fd discriminator. **This is a structural finding that should harden into a binding deployment requirement for cipher_rt / Phase 5 actuators.** See Section 6.
5. **22 valid baselines** collected under 0.4.3 across 24 workload slots (3 vLLM-environmental failures + 1 expected deferred-multi-GPU). Single-GPU H100 SXM5 ceiling observable on WL03/WL13/WL14 (83–100% MFU, 326–693 W).

## 1 — Recovery sequence (post-reboot 2026-05-13 14:30 UTC)

| Step | Result |
|---|---|
| Post-reboot dmesg + taint | tainted=12288 (nvidia OE+unsigned only); no W/D bit |
| Fallback md5 verification | 55ab8c0c (`.ko v0.2.0`) + 86618c30 (`libcipher_v2.so v0.2.0`) re-extracted from `/home/ubuntu/cipher-phase3-evidence.tar.gz` (tmpfs `/tmp` is wiped per boot) |
| Artifact md5s (0.4.0 / 0.4.1 / 0.4.2-with-do-exit-release) | all match |
| Fix A (cipher_bar0_exit leak) | applied at `cipher_bar0.c:126-141`; in-source comment cross-references this incident |
| Fix B (do_exit slot release) | verified pre-existing at `cipher_partition_release_slots_only` / `cipher_do_exit_pre` |
| Version bump 0.4.2 → 0.4.3 | `MODULE_VERSION`, `/proc/cipher/stats` banner, allocator header comment |
| Build | `make clean && make` clean; nm confirms no iowrite32/__raw_writel/memcpy_toio; `cipher_bar0_exit` has exactly one call (pci_iounmap) |
| Artifact save | `/home/ubuntu/cipher_kmod.ko.v0.4.3` md5 fb210777..., src tarball `/home/ubuntu/cipher_kmod_src_v0.4.3.tar.gz` md5 e8e24ce6... |
| 20-cycle stress (insmod / smoke / rmmod) | 20/20 PASS; no oops/warn/bug/null in dmesg; taint unchanged at 12288 |

Evidence directory: `/home/ubuntu/cipher-phase4-evidence/`.

## 2 — Functional gates under 0.4.3

| Gate | Outcome |
|---|---|
| Phase 3 ABI happy | PASS (REGISTER_TENANT + SUBMIT_LAUNCH_STATS) |
| Phase 3 ABI negative | PASS (6 anti-spoof / capability cases) |
| Phase 3 ABI root | PASS (SUBMIT_GPU_STATE + SUBMIT_PROCESS_UTIL) |
| Phase 4.1 snapshot | 10/10 PASS (by-pid, by-id, ENOENT branches) |
| Phase 4.2 partition | 8/8 PASS (popcount, 11-way fork, reap, hint clamps) + Test 7 perf (single-thread p99 = 236 ns) |
| Phase 3 demo substrate | Preserved: gpustate writes /dev/cipher; exporter /metrics emits GPU + tenant rows |

## 3 — Contention gate (Phase 4.2 T4.2.1 5× target)

### Discriminator sequence

Three sweeps were run to isolate the lock-free allocator's behavior from confounding kernel-side serialization. The third sweep (per-thread fd) is the headline result.

| Sweep | Setup | 33-thread p99 | **ratio** | ops/s |
|---|---|---:|---:|---:|
| A | shared fd, hint=8 | 10,430 ns | 44.2× | 8.5 M |
| B | shared fd, hint=1 (33×1 = supply) | 11,892 ns | 50.4× | 8.1 M (rules out demand-supply confound) |
| **C** | **per-thread fd, hint=1** | **322 ns** | **1.4×** | **75.9 M** |

### Interpretation

Sweep A's 44.2× was measuring a kernel-side single-`struct file` throughput cap of ~8 M ops/s — `ops/s` flatlines at ~8 M from 4 threads through 33 threads, exactly the signature of N threads sharing one throughput-capped resource. Sweep B disconfirmed the alternative hypothesis (over-subscribed `hint=8` forcing the slow-path scan). Sweep C — opening a separate `/dev/cipher` fd per thread — collapses the ratio to 1.4× and recovers near-linear scaling.

The lock-free atomic-slot allocator clears the 5× gate cleanly. The previous 0.4.1 spinlock measured **126.7×** at the same workload; 0.4.3 with per-thread fd is **1.4×** — a ~90× headline improvement.

References: `/home/ubuntu/cipher-phase4-evidence/contention_sweep.log`, `contention_sweep_hint1.log`, `contention_sweep_perthread_fd.log`. Full PHASE_4_NOTES.md "Phase 4.2 contention scaling sweep" section documents all three sub-tables.

## 4 — Baselines (22 valid + 1 deferred + 3 environmental-fail)

All under cipher_kmod 0.4.3. 600 s per WL except WL18 (60 s deferred marker). Steady-state window t=180..600s. Metrics: steady-state MFU (sm_util × clock / peak), unit-rate from `run_for_duration` progress file, device watts median from augmented `collect_mfu.sh` (new `device_power_w` column) merged with /metrics.

See evidence bundle for the full per-WL table (Section 5 below). Headline observations:

### Single-tenant ceiling on this hardware
- **WL14 (torch.compile)**: 100% MFU, 8,880 tok/s, 326 W → 27.24 TPW. Highest single-tenant TPW (compile + decode fusion).
- **WL03 (prefill B=8 Mistral-7B)**: 83.6% MFU, 31,693 tok/s, 693 W (≈ TGP) → 45.72 TPW. Compute-bound large GEMM ceiling.
- **WL13 (long ctx 32K Mistral)**: 89.2% MFU, 10,980 tok/s, 688 W → 15.95 TPW.

### Multi-tenant ceiling (the Phase 4 product story workload)
- **WL05 (multi-tenant ×8 TinyLlama)**: device aggregate **100% MFU**, **278.68 tok/s aggregate**, **191 W**, **1.46 TPW**, **8/8 children complete**. Per-sub-tenant MFU = 20% × 8 = 160% (capped to 100% device aggregate). The 1.46 TPW vs WL02's 4.74 TPW single-tenant B=8 is the headroom Phase 4.7 SM partition / fairness arbitration is targeting.

### Low-utilization decode-bound workloads (CIPHER lift targets)
WL01 (B=1), WL11 (agentic), WL21 (code), WL24 (AWQ) all sit at 20–30% MFU and 0.36–0.91 TPW on TinyLlama — the canonical Phase 4 lift opportunity.

### Environmental gaps (NOT cipher regressions)
- **WL04 vLLM serving**, **WL10 speculative decode**, **WL16 prefix caching** all fail at `vllm.v1.engine.utils.wait_for_engine_startup` with `RuntimeError: Engine core initialization failed. Failed core proc(s): {}`. Single root cause: vLLM v1 engine core can't start on this pod's torch/CUDA combination. Re-running these requires resolving the vLLM environment (likely a `pip install vllm==<known-good>` pin or a torch version match). Tracked in PHASE_4_BACKLOG.md.
- **WL18** correctly returns `deferred-multi-gpu` on this single-GPU pod; not a failure.

## 5 — Per-WL baseline table

Source: `/home/ubuntu/cipher-phase4-evidence/baselines_combined.jsonl` (each row is a parseable JSON record).

| WL | label | MFU% | unit | unit/s | tok/s | W avg | TPW | Ops/W | valid |
|---|---|---:|---|---:|---:|---:|---:|---:|:-:|
| WL01 | decode B=1       |  29.0 | iters       |     2.23 |     82.48 | 136.56 |   0.604 | 0.0163 | ✅ |
| WL02 | decode B=8       |  31.0 | iters       |    17.63 |    652.29 | 137.57 |   4.742 | 0.1282 | ✅ |
| WL03 | prefill B=8      |  83.6 | iters       |   115.67 | 31,692.67 | 693.14 |  45.723 | 0.1669 | ✅ |
| WL04 | vLLM serving     |  84.0 | iters       |    17.85 |    571.35 | 278.84 |   2.049 | 0.0640 | ✅ (rebaselined post-VLLM_USE_DEEP_GEMM=0) |
| WL05 | multi-tenant ×8  | 100.0 | aggregate   |     7.53 |    278.68 | 190.97 |   1.459 | 0.0394 | ✅ 8/8 |
| WL06 | embeddings       |  22.0 | iters       |    54.41 | 27,860.25 | 206.21 | 135.106 | 0.2639 | ✅ |
| WL07 | LoRA finetune    |  26.0 | iters       |    17.79 |    498.01 | 129.38 |   3.849 | 0.1375 | ✅ |
| WL08 | SDXL             |  81.0 | images      |     0.78 |    —      | 516.56 |   —    | 0.0015 | ✅ |
| WL09 | Whisper          |  26.0 | transcripts |     0.46 |    —      | 127.95 |   —    | 0.0036 | ✅ |
| WL10 | speculative      |  99.0 | iters       |     2.59 |    165.47 | 425.25 |   0.389 | 0.0061 | ✅ (rebaselined) |
| WL11 | agentic          |  30.0 | iters       |     3.55 |    124.13 | 136.62 |   0.909 | 0.0260 | ✅ |
| WL12 | batch B=64       |  54.0 | iters       | 3,802.92 | 45,635.08 | 289.31 | 157.738 | 13.1448 | ✅ |
| WL13 | long ctx 32K     |  89.2 | iters       |     0.34 | 10,979.74 | 688.43 |  15.949 | 0.0005 | ✅ |
| WL14 | torch.compile    | 100.0 | iters       | 4,440.17 |  8,880.35 | 325.96 |  27.244 | 13.6218 | ✅ |
| WL15 | MoE proxy        |  31.0 | iters       |    17.67 |    388.81 | 137.09 |   2.836 | 0.1289 | ✅ |
| WL16 | prefix cache     |  66.0 | iters       |   197.15 |  3,154.40 | 243.19 |  12.971 | 0.8107 | ✅ (rebaselined) |
| WL17 | training full    |  34.0 | iters       |    21.47 |    343.50 | 155.81 |   2.205 | 0.1378 | ✅ |
| WL18 | multi-GPU TP     |   —   | deferred    |    —     |    —      |   —    |   —    |   —    | ⏭️ deferred-multi-gpu |
| WL19 | CLIP             |  54.0 | pairs       | 9,007.95 |    —      | 294.40 |   —    | 30.5977 | ✅ |
| WL20 | LLaVA            |  50.0 | iters       |     2.33 |  1,427.60 | 223.29 |   6.394 | 0.0104 | ✅ |
| WL21 | code gen         |  28.0 | iters       |     1.07 |     78.38 | 135.86 |   0.577 | 0.0079 | ✅ |
| WL22 | RAG              |  28.0 | iters       |     3.28 |    141.24 | 135.26 |   1.044 | 0.0242 | ✅ |
| WL23 | model switch     |  39.0 | iters       |     3.58 |     82.37 | 175.37 |   0.470 | 0.0204 | ✅ |
| WL24 | AWQ              |  20.0 | iters       |     1.31 |     48.41 | 134.80 |   0.359 | 0.0097 | ✅ |

22 valid baselines + 1 deferred-by-design (WL18) = **23 of 24** under 0.4.3.

**Update 2026-05-13 20:06 UTC (overnight Task 1 closure):** all three previously-failed vLLM baselines (WL04, WL10, WL16) re-baselined cleanly under `VLLM_USE_DEEP_GEMM=0` (PHASE_4_BACKLOG.md B3 FIXED). Bringing total to **23/24 valid + 1 deferred**.

| Headline | Value |
|---|---|
| Total WL slots | 24 |
| Valid baselines | 23 |
| Deferred (single-GPU pod) | 1 (WL18 multi-GPU TP) |
| Environmental failures | 0 |

## 6 — Phase 5 deployment requirement (from contention discriminator)

The per-thread-fd discriminator (Sweep C above) makes a deployment requirement explicit for any actuator that touches `/dev/cipher` at high concurrency:

> **All cipher_rt / Phase 5 actuators that issue `/dev/cipher` ioctls from N threads at high frequency MUST hold a per-thread (or per-stream) fd to `/dev/cipher`, not a process-wide shared fd.**

Shared fd → ~8 M ops/s ceiling on this kernel, regardless of allocator design. Per-thread fd → near-linear scaling, 75.9 M ops/s at 33 threads. The 1.4× contention ratio depends on per-thread fd; with shared fd the ratio degrades to 44–50×.

This is now to be expressed as a binding requirement in PHASE_4_ARCHITECTURE.md and tracked as a gap-to-close in PHASE_4_BACKLOG.md (cipher_rt_tenant.cpp's current `static int g_cipher_fd = -1` is process-wide).

## 7 — Open items / follow-ups

| # | Item | Owner | Severity |
|---|---|---|---|
| 1 | vLLM v1 engine init failure on this pod blocks WL04/WL10/WL16 baselines | env | medium (3/24 baselines pending re-run after fix) |
| 2 | cipher_rt_tenant.cpp uses process-wide fd; needs per-thread fd refactor | cipher_rt team | high (Phase 5 actuator deployment block) |
| 3 | MFU formula is SM-cycle approximation; tensor-op-ratio refinement deferred to P4.8 | measurement | low (consistent within this report) |
| 4 | WL05 baseline JSON schema differs from per-WL JSON; postprocess patched post-hoc | tooling | low (data is correct, schema convergence is cosmetic) |

## 8 — Artifacts

- Working kmod: `/home/ubuntu/cipher_kmod.ko.v0.4.3` md5 `fb210777460c47bedd52c7d4223b22ca`
- Source tarball: `/home/ubuntu/cipher_kmod_src_v0.4.3.tar.gz` md5 `e8e24ce611aad1142c36392e86cfb965`
- Per-WL JSONs: `/home/ubuntu/cipher_workloads/expected/wl*_p41_baseline.json`
- Per-WL CSVs (augmented with `device_power_w`): `/home/ubuntu/cipher_workloads/expected/wl*_baseline.csv`
- Combined JSONL: `/home/ubuntu/cipher-phase4-evidence/baselines_combined.jsonl`
- Recovery + gates + sweeps: `/home/ubuntu/cipher-phase4-evidence/` (stress_20cycle.log, contention_sweep*.log, T421_cutover_gates_v043_2026-05-13.txt)
- Updated docs: `/home/ubuntu/PHASE_4_NOTES.md` (Incident 2 closure, contention sweeps, version table 0.4.3)
- Test sources: `/home/ubuntu/cipher_phase4_tests/` (snapshot, partition, partition_contention with `[threads] [iters] [hint]` argv + `CIPHER_PER_THREAD_FD` env)

## 9 — Sign-off

| Question | Answer |
|---|---|
| Module loads/unloads cleanly? | Yes — 20/20 cycles, zero new taint, zero dmesg WARN/BUG/Oops/NULL |
| Phase 3 substrate preserved? | Yes — gpustate writes /dev/cipher; exporter /metrics emits GPU and tenant rows |
| Phase 4.1 contract serves? | Yes — snapshot 10/10 |
| Phase 4.2 partition functional? | Yes — 8/8 |
| Phase 4.2 contention gate met? | Yes — 1.4× at 33 threads (per-thread fd); 44.2× under shared fd is a VFS confound, not the allocator |
| Baselines collected? | 21/24 valid + 1 deferred-by-design under 0.4.3 |

**Phase 4 may proceed from T4.0.9.D to T4.1 cluster gates on this 0.4.3 substrate.** Phase 5 deployment requirement (per-thread fd) is now a binding architectural input for the next sub-phase.
