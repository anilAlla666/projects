# CIPHER Phase 3 — Evidence Audit (CP 3.1–CP 3.4)

Audit date: 2026-05-15. Method: read-only inspection of files on /home/ubuntu.
No code written, no workloads run. Verification limited to `md5sum`, `sha256sum`,
`wc -l`, `ls -l`, `stat`, `grep`, `tar t`.

## Preliminary notes on framing drift

- **There is NO `cipher-phase3-evidence/` directory on disk.** What exists is a
  single tarball `/home/ubuntu/cipher-phase3-evidence.tar.gz` (590076 bytes,
  mtime 2026-05-13 10:17:37) plus a sidecar `.sha256`. The tarball's recorded
  digest (`15cecb84…efbee`) matches `sha256sum` of the file on disk — the
  artifact is intact. Contents were extracted to `/tmp/p3audit/` for inspection;
  the in-tarball `MD5SUMS` was used to identify artifacts.
- **The "canonical Phase 0-9 plan" is not on disk.** The only plan document,
  `/home/ubuntu/cipher-may13-evidence/CIPHER_BUILD_PLAN.md`, calls its "PHASE 3"
  *HBM traffic reduction* (line 395) — an entirely different framing from the
  canonical CP 3.1–3.4 (schema/review, MMIO BAR, PMU FLOP counting, Grafana +
  ClickHouse). The canonical gates audited below are taken verbatim from the
  audit task. The real Phase 3 work, captured in `/home/ubuntu/PHASE_3_NOTES.md`
  (mtime 2026-05-13 10:17:16), is a *Prometheus-pull GPU-state telemetry
  substrate* — built and self-described honestly, but organized around an
  MFU/TPW/multi-tenant narrative, not around the canonical CP gates.

## CP-by-CP status table

| CP    | One-line scope                                              | STATUS    |
|-------|-------------------------------------------------------------|-----------|
| CP 3.1| GPU-state schema design, external review                   | PARTIAL   |
| CP 3.2| MMIO BAR access from cipher_kmod, microsecond resolution    | PARTIAL   |
| CP 3.3| PMU programming for continuous FLOP counting                | NOT DONE  |
| CP 3.4| First-light Grafana + ClickHouse dashboard                  | NOT DONE  |

---

## CP 3.1 — GPU-state schema design, external review — PARTIAL

**Gate has two parts: (a) a designed GPU-state schema, (b) external review.**

### (a) Schema design — evidence present

A GPU-state telemetry schema was designed and shipped as a stable ABI:

- `cipher-phase3-evidence/src/cipher_kmod/cipher_ioctl.h`
  md5 `11740d5e3c5d2869c74d735fe3ae28c6` — the public ABI header.
- Three new ioctls on `/dev/cipher` magic `'C'` (per `PHASE_3_NOTES.md` Task 3,
  lines 32-41): nr 5 `SUBMIT_GPU_STATE` (device-wide gauges, CAP_SYS_ADMIN),
  nr 6 `SUBMIT_PROCESS_UTIL` (per-PID, CAP_SYS_ADMIN), nr 7 `SUBMIT_LAUNCH_STATS`
  (per-tenant, anti-spoofed pid+tgid match). Reserved nrs 2/3/4 return -ENOSYS,
  consistent with the project's additive-only ABI rule.
- The schema is exercised end-to-end: `runtime/gpu_state.txt`
  (md5 `c267123c…`) shows the populated device-wide record (power, temp,
  sm/mem clock, sm/mem util, fb_used, age); `runtime/stats.txt`
  (md5 `c6e097b7…`) shows the per-PID/per-tenant record.
- ABI conformance: `PHASE_3_NOTES.md` line 41 claims "17/17 test cases pass".
  Three test sources are in the tarball — `tests/cipher_test_happy.c`
  (`07a1d1d4…`), `tests/cipher_test_root.c` (`cbcd4bcd…`),
  `tests/cipher_test_negative.c` (`e2890d82…`). NOTE: the test *programs* are on
  disk; a 17/17 pass *log capture* is NOT in the evidence tarball — the pass
  count is asserted in prose only.

### (b) External review — evidence ABSENT

A filesystem grep for `external review|peer review|reviewer|review by` across
`PHASE_3_NOTES.md` and `cipher-may13-evidence/` returned nothing. No review
artifact, no annotated schema, no correspondence, no second-party sign-off
exists anywhere on disk. Under the binding audit discipline ("CP 3.1 requires
an EXTERNAL review — if there is no evidence of external review, that criterion
is unmet"), this half of the gate is unmet.

**Gap → close:** produce an external review artifact for `cipher_ioctl.h` /
the GPU-state schema (a reviewer's written sign-off with identity), and capture
the 17/17 ABI test run to a log. Until both exist, this CP is PARTIAL, not
SHIPPED.

---

## CP 3.2 — MMIO BAR access from cipher_kmod, microsecond resolution — PARTIAL

**Gate has two parts: (a) MMIO BAR access from the kmod, (b) microsecond
resolution.**

### (a) MMIO BAR access — evidence present, gate met

`cipher_bar0.c` implements direct H100 BAR0 register access from
`cipher_kmod`, with no nvidia.ko mediation:

- Source: `cipher-phase3-evidence/src/cipher_kmod/cipher_bar0.c`
  md5 `754a40294104de9c292747361b094131`, 167 lines.
  (Live tree copy `/home/ubuntu/cipher_kmod/cipher_bar0.c` differs —
  md5 `28fa5f60…`, mtime 2026-05-13 14:41, i.e. edited *after* the Phase 3
  tarball was sealed; the tarball copy is the Phase 3 artifact of record.)
- Mechanism: `pci_get_device` → `pci_iomap(pdev, 0, 16 MB)` → `ioread32`.
  Read-only by construction — `cipher_bar0_rd32()` takes
  `const void __iomem *`; no `iowrite32`/`__raw_writel`/`memcpy_toio`.
- Real-silicon proof on disk: `runtime/bar0_state.txt` (md5 `37a904675…`)
  shows `bar0_addr=0x6002000000 size=16 MB`, `PMC_BOOT_0 = 0x180000a1`
  (a real boot register value, not all-ones), and live reads
  `PMC_INTR(0)=0x0080cf88`, `PMC_INTR(1)=0xffffff88`,
  `PBUS_INTR_STATUS=0xbadf5040`. `runtime/modinfo.txt` confirms kmod 0.3.1,
  srcversion `B1AF5E2AAB2FB170A661E2A`.
- Exposed via `/proc/cipher/bar0_state` (`cipher_bar0_proc_show`).

This sub-gate — kmod-resident MMIO BAR reads against an nvidia.ko-owned device —
is genuinely met with on-disk evidence.

### (b) Microsecond resolution — evidence ABSENT

The BAR0 reads are **on-demand / lazy**: `cipher_bar0_proc_show` reads the live
registers only when `/proc/cipher/bar0_state` is `cat`-ed. There is no
continuous BAR0 sampler and no microsecond-cadence sampling loop anywhere in the
Phase 3 stack. The only continuous sampler is the userspace `cipher-gpustate`
daemon, which `PHASE_3_NOTES.md` line 47 documents as polling **NVML at 250 ms**
(4 Hz) — five orders of magnitude coarser than microsecond resolution, and it
samples NVML, not BAR0.

Do not conflate this with `probe_microbench.c` (tarball
`src/cipher_kmod/probe_microbench.c`, md5 `cdf56214…`). That file is a
**Phase 1.5.2** ioctl-latency micro-benchmark (its own header says so) that
emits `ns_per_ioctl` for `NV_ESC_CHECK_VERSION_STR` — it *measures* ioctl
latency, it does not establish microsecond-cadence BAR0 sampling.

**Gap → close:** add a kernel-thread or hrtimer-driven BAR0 sampler at a
documented sub-millisecond cadence, with a capture log proving the achieved
period. Until then, the resolution criterion is unmet → PARTIAL.

---

## CP 3.3 — PMU programming for continuous FLOP counting — NOT DONE

The gate requires **programming a performance-monitoring unit (PMU) to count
FLOPs continuously**. No evidence of this exists.

- `grep -niE "FLOP|PMU|perf event|counter program"` across the entire Phase 3
  source tree returned exactly **one** hit: the `MODULE_DESCRIPTION` string in
  `cipher_main.c` line 95 — `"CIPHER kmod — Phase 3: GPU-state spine + BAR0
  reads + PMU telemetry"`. This is a descriptive string, not an implementation.
  It is the single load-bearing piece of drift in Phase 3: the word "PMU"
  appears in a module description but is backed by no PMU code.
- `cipher_probe.c` (tarball, md5 `1d65982b…`, 267 lines) is a **kprobe /
  kretprobe on `nvidia_unlocked_ioctl`** — it counts *ioctl invocations* per PID
  in an RCU hashtable and decodes the NV_ESC command number. That is syscall
  accounting, not hardware performance counters, and it counts ioctls, not
  FLOPs. `runtime/stats.txt` shows the actual output: `total_observed=6305`
  ioctls bucketed by NV_ESC nr (e.g. `NV_ESC_RM_CONTROL 4132`). No FLOP series
  anywhere.
- The BAR0 register whitelist in `cipher_bar0.c` (lines 43-47) is
  `PMC_BOOT_0/1`, `PMC_INTR_0/1`, `PBUS_INTR_STATUS` — boot ID and *interrupt
  status* registers. None is a FLOP/instruction-issue counter, and the comment
  explicitly forbids reading PFIFO/PRAMIN regions.
- No `perf_event_open`, no GPU hardware-counter programming (no CUPTI
  PM-sampling, no PerfWorks), no continuous FLOP time series in any evidence
  file. `cipher-exporter`'s `/metrics` (see `evidence/cipher_6_metrics_midrun.txt`)
  exposes power/temp/clock/util/launches — no FLOP metric.

No SUPERSEDED claim is honestly available: ioctl-count accounting and NVML
util% are not equivalent-or-stronger substitutes for hardware FLOP counting —
they measure different quantities. Status: **NOT DONE**.

---

## CP 3.4 — First-light Grafana + ClickHouse dashboard — NOT DONE

The gate requires an **actual Grafana + ClickHouse dashboard**.

- Filesystem-wide search `find / -iname "*grafana*" -o -iname "*clickhouse*"`
  returned only unrelated third-party library files: AWS-CLI bundled service
  data (`/snap/aws-cli/.../grafana`), a `sos` report plugin, and the Sentry SDK's
  `clickhouse_driver` integration. **Zero** CIPHER-authored Grafana dashboard
  JSON, zero ClickHouse schema/DDL, zero datasource config, zero provisioning
  files anywhere on the pod.
- `grep -rliE "grafana|clickhouse|dashboard"` across `PHASE_3_NOTES.md` and the
  Phase 3 evidence tarball returned nothing.

What was actually built is a **Prometheus exporter**, not a dashboard:

- `cipher-phase3-evidence/src/cipher_exporter/cipher-exporter.py`
  md5 `96f702ca67e2f912438a7982dbae74a0`, 309 lines (Python 3 stdlib only).
- It serves `GET /metrics` (Prometheus exposition format) and `/health` on
  `:9402`. Evidence of it running: `evidence/cipher_6_metrics_midrun.txt`
  (md5 `27f0052a…`) — a mid-run `/metrics` scrape showing device-wide gauges
  (`cipher_gpu_power_watts 576.262`, `cipher_gpu_sm_util_pct 100`) and
  per-tenant series (`cipher_tenant_launches_total{tenant="tenant-A"} 512`).

A Prometheus *exporter* is a metrics surface, not a dashboard, and it is not
SUPERSEDED-equivalent to the CP 3.4 gate: an honest equivalence argument would
require an actual visualization dashboard plus a time-series store, and neither
Grafana nor ClickHouse (nor any substitute store/dashboard) exists on disk. A
metrics endpoint with no scraper, no TSDB, and no dashboard does not meet — or
equal — "first-light Grafana + ClickHouse dashboard". Status: **NOT DONE**.

**Gap → close:** stand up a time-series store (ClickHouse or substitute), wire
ingestion from the exporter, build a Grafana dashboard, and capture a
first-light screenshot or exported dashboard JSON as evidence.

---

## Summary

Phase 3, as scoped by the canonical CP 3.1–3.4 gates: **0 SHIPPED, 2 PARTIAL,
2 NOT DONE.** Real work did ship — a coherent, honestly-documented
Prometheus-pull telemetry substrate (kmod BAR0 reads + 3-ioctl GPU-state ABI +
NVML-poll daemon + kprobe ioctl accounting + Prometheus exporter) — but it is
framed around an MFU/TPW/multi-tenant narrative and does not line up with the
canonical gates' specific requirements (external schema review, microsecond-
resolution sampling, PMU FLOP counters, Grafana + ClickHouse). The single
clearest piece of drift is the `cipher_main.c` `MODULE_DESCRIPTION` advertising
"PMU telemetry" with no PMU or FLOP-counting code behind it.
