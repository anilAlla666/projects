# MFU PRODUCT PoC — cross-process decode coalescing (2026-06-11): VALIDATED
Claim: isolated same-model batch-1 tenants contend for memory bandwidth (aggregate capped ~1x), so coalescing
their decode GEMMs into one weight-stream = Nx MFU. MEASURED on real Mistral-7B fp16 weights (substrate NOT loaded):
  1 tenant batch-1:                 170 tok/s, 0.25% MFU
  8/16/32 ISOLATED (concurrent):    224 tok/s AGG (IDENTICAL for all N) = bandwidth-capped, 0.33% MFU
  8  COALESCED -> batch-8:         1347 tok/s, 1.97% MFU = 6.0x
  16 COALESCED -> batch-16:        2603 tok/s, 3.81% MFU = 11.6x
  32 COALESCED -> batch-32:        5207 tok/s, 7.62% MFU = 23.2x
=> The substrate-unique MFU lever is REAL & measured: cross-process coalescing of isolated same-model tenants
delivers 6-23x MFU. vLLM cannot (per-process); the substrate's cross-process GEMM-interception can.
CAVEATS (honest): (1) needs multi-tenant SAME-MODEL ISOLATED-process deployment (regulated/sovereign/secure); if
one vLLM w/ continuous batching already, no lift; if different models, can't coalesce. (2) PoC is GEMM-level —
real version must handle PER-TENANT attention/KV (which do NOT coalesce; each tenant own KV) — linears are >90% of
short-ctx decode bytes so win largely holds but is less at long ctx. (3) hard to build: cross-process activation
gather/scatter + substrate becomes a GEMM-level scheduler + latency cost (wait to gather tenants). (4) caps at
~23% MFU (batch-128 knee); compose w/ FP8 (fewer weight bytes -> higher coalesced batch) to push further.
