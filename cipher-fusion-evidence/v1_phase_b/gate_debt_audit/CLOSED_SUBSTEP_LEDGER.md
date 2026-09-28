# CIPHER closed-substep ledger (chronological)

**Audit date:** 2026-05-27
**Source repos:** cipher-fusion-evidence, cipher_rt_phase4, cipher_kmod
**Scope:** every closed substep from Phase 0 through K.1.5 Step 1.6

Format: chronological. Substep ID | Close date | Close commit/tag | Claimed deliverable (1 line) | Workload class measured | Memory anchor.

Evidence cited file:line in companion SUBSTRATE_STATE_OF_RECORD.md Section C-F per substep. This ledger is the evidence spine; commentary lives downstream.

---

## Era I — Phase 0/1/2/3 op31-prod (CP 0.5 through CP 5.6)

| Substep | Close date | Close commit/tag | Deliverable (1 line) | Workload measured | Memory anchor |
|---|---|---|---|---|---|
| CP 0.5 | 2026-05-15 | cipher-fusion-evidence `cp_0_5/CP_0_5_REPORT.md` | Story-B burst density sweep N=8..192 tenants | Mistral-7B-v0.1 fp16 + Marlin INT4 burst | — |
| CP 0.6 | 2026-05-15 | `cp_0_6/PHASE_0_FINAL_REPORT.md` | Phase 0 final density stories; user-process Python/ctypes ceiling identified | Mistral-7B-v0.1 fp16 | — |
| CP 2.1 | 2026-05-15 | `cp_2_1/CP_2_1_REPORT.md` | libcipher_hook port/keep/delete inventory (138 symbols) | None (read-only audit) | — |
| CP 2.4 | 2026-05-16 | `cp_2_4/CP_2_4_REPORT.md` | Marlin INT4 + DVFS + spec-decode composed | Mistral-7B B=1 (n=5 paired); Llama-3.1-8B n-gram (n=25) | [[cipher-cp24-closed]] |
| CP 2.5 | 2026-05-16 | `cp_2_5/CP_2_5_REPORT.md` | LD_PRELOAD-free GOT/PLT + cipher-platform.deb | Mistral-7B B=1 composed; CP 3.3 gate; T4.6.4 KV-dedup | [[cipher-cp25-closed]] |
| CP 3.3 | 2026-05-15 | `cp_3_3/CP_3_3_REPORT.md` | CUPTI PM tensor-pipe per-tenant MFU | cuBLAS fp16 D=8192 synthetic | — |
| CP 3.4 | 2026-05-15 | `cp_3_4/CP_3_4_REPORT.md` | Grafana + ClickHouse rootless container dashboard | loadgen 3-tenant asymmetric cuBLAS fp16 synthetic | — |
| CP 4.4 | 2026-05-16 | `cp_4_4/CP_4_4_DESIGN_MEMO.md` (memo close) | L2 persistent weight pin — closed at memo via bound | None (analytic) | [[cipher-cp44-closed]] |
| CP 4.6.5+6 | 2026-05-16 | `cp_4_6_5_6/CP_4_6_5_6_REPORT.md` | Phase 4.6 KV-dedup substrate (3 arms) | Arm1: 100-proc synthetic; Arm2: Llama-3.1-8B 32K fp16 13-tenant ceiling; Arm3 cross-tenant | [[cipher-cp4656-closed]] |
| CP 4.7 | 2026-05-17 | `cp_4_7/CP_4_7_DESIGN_MEMO.md` (memo close) | Kernel fusion lever — closed at memo via bound | Mistral-7B fp16 B=1 launch-count probe (265 forwards) | [[cipher-cp47-closed]] |
| CP 5.1 | 2026-05-17 | memory anchor only | vLLM KV integration Option A buffer-ownership | TinyLlama MHA + Mistral-7B GQA token-ID identity | [[cipher-cp51-closed]] |
| CP 5.2 | 2026-05-17 | memory anchor only | KV offload Option A snapshot-on-preempt | TinyLlama + Mistral-7B token-ID identity | [[cipher-cp52-closed]] |
| CP 5.3 | 2026-05-18 | `cp_5_3/CP_5_3_CLOSEOUT.md` | Partition-aware Marlin; F1 root-caused | Llama-3.1-8B + Llama-3.2-1B 3-arm spec-decode | [[cipher-cp53-closed]] |
| CP 5.4 (PARTIAL) | 2026-05-19 | `cp_5_4/CP_5_4_CLOSEOUT.md` | SM-arbitration substrate Steps 1.1-1.6B-3; 1.6B-4/1.6P/1.7 deferred | Synthetic OP-2/5/asym green-ctx; Marlin cubin; ARENA SC6 | [[cipher-cp54-step1-2]]/[[cipher-cp54-step1-3]]/[[cipher-cp54-deferral-audit]] |
| CP 5.6 | 2026-05-18 | `cp_5_6/CP_5_6_CLOSEOUT.md` | F1 fixed; 3.6× retracted; static N=8 TinyLlama 3.69× / N=4 Mistral 3.26× | F1 verify; single-tenant Mistral-7B B=1; static cross-tenant N=2/4/8 | [[cipher-cp56-closed]] |

**Track 2 + 3 closeouts (separate from CP chain):**

| Substep | Close date | Close commit | Deliverable | Workload | Memory anchor |
|---|---|---|---|---|---|
| Track 2 weight-sharing | 2026-05-19 | `TRACK_2_CLOSEOUT.md` | 6 SCs; ~76% mem saving Mistral-7B N=4 | Mistral-7B N=4 same-model (override dtype) | [[cipher-track2-weight-sharing]] |
| Track 3 DSM | 2026-05-19 | (closeout in evidence) | DSM v1 6 SCs; live SM migration ~70% less stranding | Synthetic POOL churn; correctness invariants | [[cipher-track3-dsm]] |

---

## Era II — may13 port + Week 1-6 reorg

| Substep | Close date | Close commit | Tag | Deliverable | Workload measured |
|---|---|---|---|---|---|
| Week 1 Step 1 | 2026-05-20 | may13 `fc8a9ae6` | week-1-step-1-lp7-rename | LP-7 struct rename (18 sites) | compile-only |
| Week 1 Step 2v2 | 2026-05-20 | rt_phase4 `50a6f228` | week-1-step-2-v2-header-port | 12-header may13 transitive port | compile-only |
| Week 1 Step 3 | 2026-05-20 | rt_phase4 `bcf8a83b` | week-3-step-3-cross-tree-harness | cross-tree harness 59 LOC | CP 5.4 byte-identical (synthetic OP-2) |
| Week 1 Step 4v2 | 2026-05-20 | kmod `f8572ecf` | week-1-step-4-cb2-reserved-tail | Cb.2 reserved-tail struct bump | ABI offsets preserved (verify-only) |
| Week 2 Step 1 | 2026-05-20 | `465b244e` | week-2-step-1-lp2-sdpa-refactor | LP-2 SDPA trampoline (3) | SDPA smoke handled=0 passthrough=1 |
| Week 2 Step 2v3-D1 | 2026-05-20 | `23014c1e` | week-2-step-2-may13-ports-a4-h1-root | 6-file transitive .cpp ports | link only |
| Week 2 Step 3 | 2026-05-20 | `7629cf60` | week-2-step-3-classify-substrate-scaffold | classify_substrate registry | substrate-internal |
| Week 2 Step 4 | 2026-05-20 | `7ee5b2a8` | week-2-step-4-classify-observer-stub | atomic-counter observer | substrate-internal |
| Week 2 Step 5 | 2026-05-20 | kmod `bc48590e` | week-2-step-5-classify-proc-node | /proc/cipher/classify_stats | proc node readable |
| Week 2 Step 6 | 2026-05-20 | rt_phase4 `f9c32322` / kmod `0ce4b8e2` | week-2-step-6-cupti-classify-wired | CUPTI hot-path + ioctl bridge | end-to-end classify visible at /proc |
| Week 3 Step 1 | 2026-05-20 | `0b6effdb` | week-3-step-1-dispatch-scaffold | cipher_rt_dispatch | substrate-internal |
| Week 3 Step 2 | 2026-05-20 | `71bf5523` | week-3-step-2-observer-hint-publish | TLS substitute_hint | substrate-internal |
| Week 3 Step 3 | 2026-05-20 | `4f1a86ab` | week-3-step-3-marlin-hint-volt-lock | Marlin hint consumption + VOLT 1000 lock | TinyLlama-AWQ-INT4 (Marlin) |
| Week 3 Step 4 | 2026-05-20 | rt_phase4 `79c1b4f9` / kmod `a21a45ee` | week-3-step-4-opt2a-dispatch-live-sense | CIPHER_DISPATCH_LIVE=1 default + SENSE | Mistral-7B SC6 bit-identical |
| Week 4 Step 1 | 2026-05-21 | `4279461` | week-4-step-1-real-oracle | Oracle bridge | substrate-internal |
| Week 4 Step 2 | 2026-05-21 | `b25edf7` | week-4-step-2-tier-a-ports | LOOP/PIPELINE/PULSE/CONTINUITY (27 T-symbols) | observe-only |
| Week 4 Step 3 | 2026-05-21 | `3be4531` | week-4-step-3-tier-b-ports | TRACE/RECEIPT/CARBON/FAIRNESS/GUARD/DETERMINISM (43 T-symbols) | observe-only |
| Week 4 Step 4 | 2026-05-21 | kmod `158ad96` | week-4-step-4-lp8-retired | LP-8 retire (-482 LOC) | ABI verify |
| Week 4 Step 5 | 2026-05-21 | kmod `2fc70c3` | week-4-step-5-sense-session-proc | /proc/cipher/sense_session + 11 Prom metrics | exporter on-disk only |
| Week 4 Step 6 | 2026-05-21 | rt_phase4 `3c5ddaa` | week-4-step-6-sub4-measurement | Sub-4 per-tenant counters (~65 LOC) | substrate-internal |
| Week 5 Step 1 | 2026-05-21 | paperwork | WEEK_5_STEP_1_DESIGN_MEMO.md | LOCKED-SHAPE-II + substrate-API-gap finding | — |
| Week 5 Step 1b | 2026-05-21 | rt_phase4 `ec0e005` | week-5-step-1b-substrate-alias | cipher_rt_kv_dedup_alias (+211 LOC) | rebind verify |
| Week 5 Step 2 | 2026-05-21 | snapshot `plugin_snapshots/*.w5_step2` | (entry point v0.1→v0.2) | vLLM kvdedup plugin + N=2 same-prompt | TinyLlama-AWQ-INT4 N=2: 42 GiB saved |
| Week 5 Step 3 | 2026-05-21 | tag `week-5-step-3-n4-kl-gate` | week-5-step-3-n4-kl-gate | N=4 KL=0 + 45.8 GiB | Mistral-7B + TinyLlama-AWQ-INT4 |
| Week 6 G1+G2 | 2026-05-23 | kmod `c4e2d6f` | week-6-step-g1-g2-cap-bump | CP54 64→128 allocs + WA 16→100 arenas | ABI-bump verify (kmod 0.5.0) |

---

## Era III — Week 7-14 substrate consolidation + Option 2

| Substep | Close date | Close commit | Tag | Deliverable | Workload measured |
|---|---|---|---|---|---|
| W7 Step 1 | 2026-05-23 | kmod `a9d18aa` | week-7-step-1-g10-abi-scaffold | NR 27 CIPHER_REGISTER_MODEL + 256-bucket model_uuid | test_register_model 5/5 (synthetic) + Mistral-7B B=1 smoke 163.5 tok/s (override-dtype) |
| W7 Step 2 | 2026-05-23 | rt_phase4 `24f906d` | week-7-step-2-commit-primitive | COMMIT atomic primitive (72B per-tenant seqlock) | test_commit_atomicity (synthetic N=15) |
| W7 Step 3 | 2026-05-23 | kmod `57d96cc` | week-7-step-3-g6-audit-chain | G6 HMAC-SHA256 audit chain (32 MiB vmalloc'd) | test_audit_chain (synthetic N=15 × 1000) |
| W7 Step 4 | 2026-05-23 | rt_phase4 `b39702c` | week-7-step-4-overlay-port-hotpath | 19 overlay-op ports + COMMIT hot-path wire | TinyLlama-1.1B B=1 vLLM (override dtype) -0.45% TPS |
| W9 Step 5 | 2026-05-23 | kmod `8c643fc` / rt_phase4 `c93a141` | week-9-complete / week-9-step-5-n128-soak | NR 29 REGISTER_STREAMS + N=128 1h soak | Synthetic N=128 contention (no real model loads) |
| W10 Step 1 | 2026-05-23 | rt_phase4 `773a752` | week-10-step-1-ring-write | Per-tenant SPMC ring 4096×64 B + CFL | test_ring_write (synthetic 100k cycles) |
| W11 Step 2 | 2026-05-23 | rt_phase4 `e594706` | week-11-step-2-g3-g4-tc-probe | G3 KV-dedup model-keying + G4 Marlin tenant arena + TC probe | test_g3_cross_model_keying + test_tc_probe (synthetic algebraic) |
| W12 Step 3 | 2026-05-23 | rt_phase4 `fec9cc3` | week-12-step-3-g5-l2-persist / week-12-complete | G5 compute_va_gib + L2 per-tenant-stream policy | test_g5_va_density (synthetic 5 model families) + N=128 soak |
| W12 Step 4 | 2026-05-23 | rt_phase4 `1466193` | week-12-step-4-sdpa-stream-fill | D5 backfill SDPA stream-fill | test_sdpa_tenant_routing (synthetic 4 cases) |
| W12 Step 5 | 2026-05-23 | rt_phase4 `b360fc1` | week-12-step-5-d14-env-gate | CIPHER_RING_WRITE env-gate; TPS ±3% → ±5%; 3 findings | TinyLlama-1.1B B=1 (override dtype) paired baseline |
| W12 Step 6 | 2026-05-23 | rt_phase4 `adbe121` | week-12-step-6-d3-l2-wireup | L2 plugin call-site wire | Synthetic baseline (no E2E) |
| W13 Step 1 | 2026-05-23 | rt_phase4 `d93bf9d` | week-13-step-1-g12-koopman-keying | G12 Koopman recipe model-keying (XOR-mix) | test_g12_recipe_keying (synthetic algebraic) |
| W14 Step 2 | 2026-05-24 | rt_phase4 `4b775c8` | week-14-step-2-koopman-tier | Koopman tier 6 substrate fixes (H2 CGS2, Tikhonov, β OOD, GPU upload, POWER_ITERS=4, det SVD) | substrate correctness only (no real workload firing) |
| W14 Step 3 | 2026-05-24 | rt_phase4 `25970f3` | week-14-step-3-c-lmhead-validate / week-14-complete | REMEMBER consumer + LM-head harness + N=128 soak | Synthetic TinyLlama LM head only (no real-workload firing) |
| Option 2 Step 0 | 2026-05-25 | cipher-fusion-evidence `7c6d30f` | option-2-step-0-vllm-worker-init-hook | vLLM V1 EngineCore worker GOT-patch hook | TinyLlama-1.1B-Chat single-tenant init |
| Option 2 Step 0.5 | 2026-05-25 | cipher-fusion-evidence `31b3469` | option-2-step-0-5-koopman-reachability-probe | Maybe-handle early-exit instrumentation | TinyLlama B=1 override threshold (synthetic force-fire) |
| Option 2 Step 1α | 2026-05-25 | rt_phase4 `52923af` | option-2-step-1-alpha-dtype-counter | skip_dtype/dim/nullptr per-early-exit counters | TinyLlama B=1 default-bf16 (REAL stock workload) |

---

## Era IV — Phase A + Phase B + K.1.5

| Substep | Close date | Close commit | Tag | Deliverable | Workload measured |
|---|---|---|---|---|---|
| Phase A A.0 | 2026-05-26 | paperwork (no commit) | v1-goal5-contract-lock | Goal 5 contract revert to LD_PRELOAD-only framing | — (paperwork) |
| Phase A A.1 | 2026-05-26 | rt_phase4 `8613812e` | v1-substrate-driver-worker-init | constructor + counter-dump port + cuInit-wrapper | — (infrastructure) |
| Phase A A.2 | 2026-05-26 | rt_phase4 a2_cuda_injection_PASS.json | (verification artifact) | TinyLlama B=1 128-tok decode worker shim_calls=19090 ≥11000 PASS | TinyLlama-1.1B fp16 (override-dtype) |
| Phase A A.3 | 2026-05-26 | rt_phase4 a3_summary.md | (verification artifact) | Track 2 SC6 + Track 3 + W14 microbench regression PASS | Mistral-7B N=4 (override-dtype historical) |
| Phase A A.4 | 2026-05-26 | (paperwork) | (no tag) | V1_PHASE_A_COMPLETE.md ledger row | — |
| B.0 | 2026-05-23 | cipher-fusion-evidence c52d2ab | (no tag) | Substrate-only baseline reproduce Track 2 SC6 + Week 5 KV-dedup | Mistral-7B N=4 (override-dtype historical baseline); TinyLlama N=4 |
| B.0.5 | 2026-05-23 | cipher-fusion-evidence c52d2ab | (no tag) | Plugin hook archaeology — CP 5.1 attribution clarified | — (read-only audit) |
| B.1' | 2026-05-26 | cipher-fusion-evidence 159386f | (no tag) | Scope-lock addendum — container path selected over .deb | — (architectural decision) |
| B.2'' | 2026-05-26 | (substep stage) | (no tag) | cipher-platform_2.0_amd64.deb build + install (rev6) | install verify only |
| B.3'' Gate A 5th | 2026-05-26 | cipher-fusion-evidence 8fb28fe | (no tag) | CDI env + /dev/cipher devNode + CUDA 13 substrate .deb | TinyLlama-1.1B fp16-override worker shim PASS (Mode A/C/D) |
| B.6''.8 | 2026-05-26 | cipher-fusion-evidence 82d8fcf | (no tag) | Track 2 SC6 substrate reproduce under installed v2.0 rev5 .deb | Mistral-7B N=4 (override-dtype historical) |
| B.6''.9.1 | 2026-05-21 | cipher-fusion-evidence 1116d9c | (no tag) | F-VLLM-PLUGIN-CRASH bisect to NR 27 downstream | — (root-cause) |
| B.6''.9.3 | 2026-05-22 | cipher-fusion-evidence 0e373d2 | (no tag) | NR 27 CIPHER_REGISTER_MODEL=0 in CDI patch (gate-off) | — (gating decision) |
| B.6''.9.4 | 2026-05-23 | cipher-fusion-evidence e2ae502 | (no tag) | Substrate coverage re-measurement on stable rev6 | attn_calls=0 finding (default vLLM workload) |
| B.6''.9.5 | 2026-05-23 | cipher-fusion-evidence 66e8180 | (no tag) | F-CUDAGRAPH-1 bisect INVALIDATED | — (diagnostic) |
| B.6''.9.6 | 2026-05-23 | cipher-fusion-evidence 4e18bb4 | (no tag) | may13 cipher_intercept audit ~70% port-as-is | — (read-only) |
| B.6''.9.8.1 | 2026-05-24 | cipher-fusion-evidence f3a4698 | (no tag) | may13 cuGetProcAddress foundation ported | — (infrastructure) |
| B.6''.9.8.2 | 2026-05-24 | cipher-fusion-evidence f5ee2d1 | (no tag) | Counter parity 9/9 EXACT | TinyLlama-1.1B fp16 (override-dtype) |
| B.6''.9.8.3 | 2026-05-24 | cipher-fusion-evidence f70fad0 | (no tag) | Phase A regression under may13 port PASS | TinyLlama-1.1B fp16 (override-dtype) |
| B.6''.9.8.4 | 2026-05-24 | cipher-fusion-evidence 2aaa724 | (no tag) | cuGraphLaunch shims engage during replay | Synthetic vLLM cuStream-capture |
| B.6''.9.8.5 | 2026-05-24 | cipher-fusion-evidence 48fcde8 | (no tag) | cuGraphAddKernelNode capture-time substitution shims | — (substrate-internal) |
| B.6''.9.8.5b | 2026-05-24 | cipher-fusion-evidence 2cd0516 | (no tag) | env-gated PyTorch capture-substitution (infrastructure ship) | — (substrate-internal, skip activation) |
| B.6''.9.8.5b.2 | 2026-05-25 | cipher-fusion-evidence 09ff5a4 | (no tag) | Path 2 skip activation; ship infra forward-compat | — (architectural decision; "engagement=0 on standard vLLM" explicit) |
| B.6''.9.8.5b.3 | 2026-05-27 | cipher-fusion-evidence 3ae4043 | (no tag) | Empirical validation: Goal 2/4 engagement PRACTICALLY ZERO on standard vLLM | Mistral-7B (override-dtype, NOT default bf16); TinyLlama-AWQ INT4 |
| B.6''.9.8.5b.4 | 2026-05-27 | cipher-fusion-evidence 482c94f | (no tag) | Framing Y diagnostic; dtype gap surfaced; Goal 2 scope options E.1-E.5 | — (READ-ONLY diagnostic) |
| V1 capability audit | 2026-05-27 | cipher-fusion-evidence 072f610 | (no tag) | 25-capability matrix + Workload Classifier keystone-gap finding | — (READ-ONLY audit) |
| K.1.5 Step 0 | 2026-05-27 | rt_phase4 `156090a` | k1-5-step-0-substrate-partial-close | Bug #1+#2 fixes + cuModule/cuLibrary reg map + cache-poison; PARTIAL close (coverage 70.5%) | TinyLlama-AWQ INT4 stock (int4=1 ON); Llama-3 + Mistral PARTIAL (multi_tenant/A2 unhit) |
| K.1.5 Step 1 | 2026-05-27 | rt_phase4 `b401019` | k1-5-step-1-diagnostic-instrumentation | env-gated CIPHER_K1_DIAGNOSTIC observe_* + snapshot + UNRESOLVED logs | TinyLlama-AWQ stock (50 unresolved fns logged) |
| K.1.5 Step 1.5 | 2026-05-27 | rt_phase4 `2ea27f8` | k1-5-step-1-5-disambiguator | Retry chain (reg_map retry + cuKernelGetName + cuFuncGetAttribute); H4 verdict | TinyLlama-AWQ stock (50/50 recovered via cuKernelGetName) |
| K.1.5 Step 1.6 | 2026-05-27 | rt_phase4 `d785fd8` | k1-5-step-1-6-cu-kernel-get-name-primary | cuKernelGetName promoted to primary; NCCL log dedup; 3-workload verify | Llama-3-8B bf16 default + Mistral-7B-v0.1 bf16 default + TinyLlama-AWQ INT4 default — ALL 3 |

---

**Total: ~80 closed substeps across 4 eras.**

Workload taxonomy across the ledger:
- **Memory #25 reference set** (default Llama-3-8B bf16 / default Mistral-7B bf16 / default TinyLlama-AWQ INT4): K.1.5 Step 1.6 ONLY (3/3 workloads default config).
- **Override-dtype proxy** (Mistral-7B `--dtype float16`, TinyLlama-1.1B fp16, etc): Phase A A.2/A.3, every Phase B step before .8.5b.3, every override-dtype CP measurement.
- **Synthetic test harnesses** (test_commit, test_audit_chain, test_ring_write, test_g3, test_tc, test_g5, test_sdpa, test_g12, LM-head harness, N=128 synthetic soak, OP-2/5/asym green-ctx churn): MAJORITY of W7-W14 + parts of CP 5.4.
- **Real-model on real-dtype but not Memory #25 reference set** (Mistral-7B-v0.1 base not -Instruct; TinyLlama-1.1B-Chat fp16 not -AWQ): CP 0.5, CP 0.6, CP 2.4, Week 5 Step 3 (Mistral-7B-v0.1 bf16 KL=0 was real default), Option 2 Step 1α (TinyLlama-1.1B bf16 default was real Memory #25 ref).
- **Read-only / paperwork**: B.0.5, B.1', B.6''.9.5/6, K.1.5 Step 1.6's parent audit memos.
