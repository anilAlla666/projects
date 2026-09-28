---
name: cipher-t424c-partition-not-enforced
description: T4.2.4c finding — GREEN_CTX partition mechanism passes API-level checks but is NOT reaching kernel scheduling for PyTorch streams; bomb prefills/s identical A/B
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

T4.2.4c (2026-05-14 midday) ran a noisy-neighbor A/B with libcipher_rt
T4.2.4b_v2 (partition ON) vs libcipher_v2 (partition OFF) on
Mistral-7B-prefill bomb + TinyLlama-decode victim. Key finding from the
bomb-throughput diagnostic: **bomb prefills/s is 93.8 (A) vs 94.4 (B)
— Δ -0.6%, i.e. statistically identical**. If GREEN_CTX were
constraining the bomb to its assigned 8 SMs of 132, bomb throughput
should have dropped ~16×. It didn't. The partition is NOT enforced at
kernel scheduling for PyTorch's hot-path streams.

**Why:** cuGreenCtxGetDevResource confirms 8-SM partition exists; push
fires on cudaStreamCreate observations #1–#3. But PyTorch's prefill
kernels likely run on the default stream created before our CUPTI
subscriber binds, so the bomb's kernels execute on the original
132-SM context.

**Why:** mechanistic gap blocks tail-latency isolation AND any future
aggregate-TPW work that depends on real kernel-level partition
enforcement. Until this is fixed, GREEN_CTX is substrate-only.

**How to apply:** before claiming GREEN_CTX delivers ANY measurable
benefit, verify with bomb-throughput-style diagnostic that
partition-restricted tenants actually run at restricted speed. If
they don't, the actuator isn't engaged regardless of API-level smoke
tests.

**Diagnostic candidates for future fix:** direct cuLaunchKernel
instrumentation + cuStreamGetCtx logging; override torch.cuda.Stream;
NSight Systems trace; hand-rolled non-PyTorch bomb. See
[[cipher-phase3-shipped]] for kmod baseline context.

Related: T4.2.4b's "push fires confirmed" smoke check only verified
push existence, not that push affects the streams carrying user work.
That subtle gap is what T4.2.4c surfaced.
