---
name: cipher-pod-environment
description: "Lambda H100 pod environment facts for CIPHER work — driver version, kernel, GPU PCI BDF, BAR0 layout, NVML/CUPTI availability, language toolchain gaps"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a1207d0-bd39-48db-a0a9-37165af96920
---

Lambda Labs H100 pod, used for all CIPHER kernel-module and userspace work.

- **GPU:** NVIDIA H100 80GB HBM3 (single device) at PCI BDF `0000:07:00.0`
- **Driver:** 580.105.08 (CUDA 13.0 runtime)
- **Kernel:** 6.8.0-1046-nvidia, x86_64
- **BAR0:** `0x6002000000` — `0x6002ffffff` = 16 MB, 64-bit memory-mapped (flags `0x14220c`)
- **NVML:** `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1` → `.so.580.105.08`. Header at `/usr/include/nvml.h`. GPM symbols all present: `nvmlGpmMetricsGet`, `nvmlGpmSampleAlloc/Get/Free`, `nvmlGpmQueryDeviceSupport`, `nvmlDeviceGetProcessesUtilizationInfo`, plus the basic `nvmlDeviceGetUtilizationRates/PowerUsage/Temperature/ClockInfo/MemoryInfo`.
- **CUPTI:** `libcupti.so.12` at `/lib/x86_64-linux-gnu/`. Header at `/usr/include/cupti.h`.
- **Languages:** Python 3.10.12 present. Go is NOT installed. C toolchain (gcc, kernel headers) present. CUDA toolkit `nvcc` 12.8.
- **Containers:** `podman` (rootless) and `docker` (`docker ps` needs `sudo -n`, passwordless) both present. **Rootless podman custom CNI networks are broken** — `podman network create` yields a config the firewall plugin rejects (`config version "1.0.0"` unsupported) and containers fail to start on it. Use `--network host` (works; distinct ports coexist). Internet reachable from containers. This is the CP 3.4 ClickHouse + Grafana dashboard stack's runtime.

**Why:** The CIPHER stack assumes these exact paths and capabilities. When a spec says "use Go for the HTTP exporter" or "find libnvidia-ml.so," knowing the pod's actual state lets you make the right substitutions without a clarifying round-trip.

**How to apply:** When a spec calls for Go, fall back to Python (per the user's explicit allowance in Phase 3 spec). When a spec references BAR0 reads, the H100 BAR0 is exactly 16 MB starting at `0x6002000000`. Always use the runtime sysfs path (`/sys/bus/pci/devices/0000:07:00.0/`) rather than guessing the BDF; it can change across pod reboots.
