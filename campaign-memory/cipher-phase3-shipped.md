---
name: cipher-phase3-shipped
description: Phase 3 (GPU-state telemetry substrate) shipped 2026-05-13 — cipher_kmod 0.3.1, daemon + libcipher_v2-CUPTI + Prometheus exporter wired. Phase 4 working baseline is now 0.4.3.
metadata:
  node_type: memory
  type: project
  originSessionId: 5563103d-e475-4d53-9aa4-ac7a1482fd3f
---

Phase 3 close-out as of 2026-05-13. Substrate complete; Phase 4 actuators
consume it; Phase 5 (Green Contexts) builds on it.

Shipped components (all under /home/ubuntu/):
  cipher_kmod/         module v0.3.1, srcversion B1AF5E2AAB2FB170A661E2A
  cipher_gpustate/     userspace daemon (NVML poll @ 250 ms, three ioctls)
  libcipher_v2/        CUDA injection (REGISTER_TENANT + CUPTI launch counter)
  cipher_exporter/     Python stdlib Prometheus exporter on :9402

Stable ABI delta on /dev/cipher (additive only per [[cipher-abi-rule]]):
  nr 5 SUBMIT_GPU_STATE      CAP_SYS_ADMIN
  nr 6 SUBMIT_PROCESS_UTIL   CAP_SYS_ADMIN
  nr 7 SUBMIT_LAUNCH_STATS   anti-spoofed (pid+tgid match current)
  nrs 2/3/4 still -ENOSYS (reserved SNAPSHOT/RESET/GET_VERSION).

Headline demo receipt (stage 5b): two-tenant 8000x8000 PyTorch matmul, 30 s,
device sm_util=100% sustained, peak 615 W, per-tenant sm_util split observed
(20/80 → 40/60 → 80/20 across the run), workload fairness 658 vs 657 iters
(0.15% delta). LAUNCHES populated via CUPTI in 256-step batches matching the
in-callback flush cadence.

**Why:** Substrate for the Phase 4 product story (per-tenant accounting via
Prometheus + actuators on top). Phase 3 alone is observation, not optimization.
The MFU/TPW/density product goals are PENDING Phase 4 and Phase 5.

**How to apply:** Future session resuming work should know cipher_kmod 0.3.1
is the shipped Phase 3 baseline; do not regress nrs 5/6/7; preserve REGISTER_TENANT
behavior; the v0.2.0 fallback at /home/ubuntu/cipher_kmod.ko.v0.2.0 (md5
55ab8c0cd8309ca7cc0fc40fe556aa19) remained untouched through Phase 3 and is
the rollback target. Evidence tarball at /home/ubuntu/cipher-phase3-evidence.tar.gz
sha256 15cecb8448e7d1c490be2117e06341cc76f460815ba1bc575c62d1cbb86efbee.

Known non-blockers documented for Phase 4+ (see PHASE_3_NOTES.md §"Known
Limitations"): PMC_BOOT_0 field decode deferred, NVRM refcnt warning on CUPTI
subscribe is benign, NVML stale-PID cache can produce ghost rows in
/proc/cipher/stats, grid_ops_total wired but always 0, Layer A reads on-demand
only.

## Phase 4 working baseline (2026-05-13 post-reboot)

Phase 4 baseline kmod is now **0.4.3** (srcversion 335F54871889FFF2C36E3B0,
md5 fb210777460c47bedd52c7d4223b22ca) at `/home/ubuntu/cipher_kmod.ko.v0.4.3`
+ source tar `/home/ubuntu/cipher_kmod_src_v0.4.3.tar.gz` (md5 e8e24ce611aad1142c36392e86cfb965).
0.4.3 = 0.4.2 lock-free atomic-slot partition allocator + `cipher_bar0_exit`
pci_dev_put leak workaround (see [[cipher-incident-2-bar0-exit]]).

Verified under 0.4.3 (post-reboot, evidence in /home/ubuntu/cipher-phase4-evidence/):
- 20/20 insmod→smoke→rmmod stress cycles clean.
- Phase 3 ABI gates (happy/negative/root) all PASS.
- Phase 4.1 snapshot 10/10 PASS, Phase 4.2 partition 8/8 PASS.
- Single-thread partition p99 = 236 ns (vs 0.4.1's 260 ns).
- Contention behavior (4→33 thread sweep) recorded; see [[cipher-contention-gate-context]].
- Phase 3 demo substrate preserved: gpustate + exporter /metrics still works.

Do NOT load 0.4.2 (its rmmod crashed; superseded by 0.4.3). 0.4.0 / 0.4.1 are
retained as evidence only.
