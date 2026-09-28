---
name: cipher-pmc-boot-1-bare-metal
description: "PMC_BOOT_1=0x00000000 on Lambda H100 is the correct bare-metal reading, not a stuck-bus signal — VGPU8/VGPU16 bits are 0 when GPU is non-virtualized"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5563103d-e475-4d53-9aa4-ac7a1482fd3f
---

When reading NV_PMC_BOOT_1 (BAR0 offset 0x4) on the Lambda H100, the value
0x00000000 is the **correct, expected** reading. Per NVIDIA's
open-gpu-kernel-modules `src/common/inc/swref/published/nv_ref.h`:

  NV_PMC_BOOT_1_VGPU8   bit 8:8   REAL=0 VIRTUAL=1
  NV_PMC_BOOT_1_VGPU16  bit 16:16 REAL=0 VIRTUAL=1
  NV_PMC_BOOT_1_VGPU    bits 17:16 (REAL=0, PV=1, VF=2)

All other bits are reserved. On bare-metal H100 (which Lambda pods are),
every defined field is 0, so the whole register reads as 0x00000000.

**Why:** General stuck-bus heuristics treat 0x00000000 as suspicious. For
PMC_BOOT_1 specifically that heuristic is wrong — 0 is affirmative
"non-virtualized" data, not a dead read. PMC_BOOT_0 should still be sanity-
checked against arch=0x18 separately; only PMC_BOOT_1=0 is benign.

**How to apply:** When validating BAR0 reads in [[cipher-project-layout]], do
not flag PMC_BOOT_1=0 as a failure. PMC_BOOT_0 stuck-bus check still applies
(must not be 0x00000000 and must not be 0xFFFFFFFF; on Lambda H100 it reads
0x180000a1). Cross-reference: full BOOT_0 bit layout (also split-arch field)
lives in the same nv_ref.h.
