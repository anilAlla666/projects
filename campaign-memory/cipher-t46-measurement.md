---
name: cipher-t46-measurement
description: "T4.6.0.5 KV dedup opportunity measurement (2026-05-14). Synthetic mix 1.19× at block=16 N=8; revised architecture commit after user review = outcome (c) L1+L3, validation gate moved to AFTER build."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

T4.6.0.5 (2026-05-14 late night, 3h cap) measured cross-process
KV-block dedup opportunity on synthetic multi-tenant traces.

**Raw measurement (block=16, N=8 cross-tenant, 20 random groupings, median):**
- chat: 1.18× (4 different sys-prompt operator types in synthesis)
- agentic: 3.46× (shared tool definitions across all tenants)
- rag: 1.30×
- code: 3.00×
- weighted mix (50/30/15/5): 1.19×

**Position-stratified at block=16 N=8:**
- agentic pos 0-256: 8× (dedup IS in the system-prompt region)
- single-operator chat pos 0-128: 4.27×
- pos > 1024: 1.00 (divergent tails)

**Block-size sensitivity (mixed N=8 median):**
- block=1: 4.90× (mostly common-token coincidence, not semantic)
- block=4: 1.55×
- block=16: 1.19× (vLLM-style; the substrate-correct granularity)
- block=64: 1.16×

**Economic mapping (Mistral-7B GQA per-tenant KV, H100 80GB):**
- 4K context: 512 MB → compute-bound at 64 tenants (dedup doesn't matter)
- 8K context: 1 GB → tied
- 32K context: 4 GB → **memory-bound at 16 tenants** (dedup CRUCIAL)
- 128K context: 16 GB → memory-bound at 4 tenants

The multi-tenant moat materializes at **32K+ context** where memory is
the bottleneck. At ≤8K the system is compute-bound; KV dedup buys nothing.

---

**ARCHITECTURE COMMIT (REVISED after user review): outcome (c) — L1+L3.**

Initial commit was outcome (b) L1-only with validation gate before L3.
User rejected on triangulation grounds: synthetic 1.19× is a
*structural-mechanism* verification, not a *production-opportunity*
measurement. Production opportunity already verified by prior art at
scale (Anthropic 80-95% prefix overlap, LMCache 95% shared on 739
Claude Code traces, llm-d bimodal production, SGLang 2-5× from prefix
reuse). cuIpc cross-process mechanism verified on this pod (T4.6.0).
Synthetic short-prefix mix understates production by ~10×.

L1 alone = "operator-context vLLM" — not differentiated. The marvel
pitch (operator-context cross-process KV pool) requires L3. cuIpc
cross-process is what positions CIPHER *below* vLLM/SGLang/TGI in the
stack.

**Build L1+L3 together. Validation gate moves to AFTER build, BEFORE
production ship.**

**Branches at T4.6.5 validation gate (real production traces):**
- ≥10× L3 cross-process dedup → ship full stack; moat verified
- 3-10× → ship full stack; position as "depends on workload mix"
- <3× L3 (≥2× L1) → ship L1 default + L3 as feature flag
- <2× both → deeper investigation (KV compression / per-tenant model loading)

**Pivot trigger:** if L1+L3 production deployment shows <2× actual
tenant-density gain on 32K+ context, KV-dedup moat is dead.

---

**Sub-phase shape for T4.6 implementation (outcome c):**
- T4.6.1 — attention-routing substrate (`.symver` LD_PRELOAD on
  `pytorch_flash::run_mha_*`, mirrors T4.5.1 cuBLAS pattern) — 1-2 sessions, 6-8h
- T4.6.2 — KV page allocator + per-process page table (block=16, FNV-1a) — 2-3 sessions, 10-15h
- T4.6.3 — L1 in-process prefix dedup wired through attention dispatch — 2-3 sessions, 10-15h
- T4.6.4 — L3 cuIpc cross-process page pool + kmod ioctls nr 11/12/13
  (`CIPHER_KV_REGISTER_PAGE` / `_LOOKUP_PAGE` / `_RELEASE_PAGE`) — 3-4 sessions, 15-25h
- **T4.6.5 — Real-trace validation gate** (after build, before production ship) — 1 session, 3h
- T4.6.6 — 50-100 tenant integration + production-readiness validation — 1-2 sessions, 6-8h

**Total: 10-15 sessions, 50-74h engineering, ~3-4 weeks calendar.**

ABI commitment: kmod ioctl nrs 11/12/13 are additive — Phase 3 ABI
12/12 invariant preserved per [[cipher-abi-rule]].

**Discipline preserved (T4.6.0.5 measurement-only sub-phase):** NO
production artifacts modified. Phase 3 ABI 12/12, fallback md5s
55ab8c0c / 86618c30 unchanged, taint 12288.

**Discipline lesson (durable):** distinguish what a measurement
ACTUALLY tested vs what it's being used to justify. Synthetic
measurements verify mechanisms; prior art verifies opportunity.
Triangulate before committing to a strict reading of one number in
isolation.

Linked: [[cipher-t46-kv-dedup-discovery]] (substrate discovery this
measurement informed), [[cipher-t45-substrate-marlin]] (substrate
pattern T4.6.1 extends), [[cipher-abi-rule]] (ABI additive invariant
T4.6.4 ioctls honor).

Tools shipped (durable): `kv_dedup_measurement.py` and
`kv_dedup_single_operator.py` in
`cipher-phase4-evidence/t4_6_0_5_measurement/`. Reusable for future
workload audits and the T4.6.5 real-trace validation gate.

Full architecture commit + audit history: `PHASE_4_T4_6_0_5_MEASUREMENT.md`.
