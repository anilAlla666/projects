---
name: MFU baseline is hardware-limited
description: Do not chase 989 TFLOPS spec — H100 sustains ~700 TFLOPS under thermal steady state. 989 is boost spec, not achievable target.
type: feedback
---

Do not chase 850+ or 989 TFLOPS as an MFU target. H100 sustains ~700 TFLOPS (71% MFU) under continuous GEMM load due to thermal throttling to ~1395 MHz. The 989 TFLOPS spec is boost clock marketing, not a sustainably achievable number.

**Why:** User corrected when I started optimizing toward 850 TFLOPS. The correct passthrough baseline is 704 TFLOPS / 71.3% MFU. Chasing a spec number is drift from real hardware behavior.

**How to apply:** When reporting MFU, use ~700 TFLOPS as the realistic H100 fp16 sustained ceiling. MFU% should be computed against this real baseline, not the 989 spec. Any improvement CIPHER delivers should be measured against the actual cuBLAS passthrough baseline, not theoretical peak.
