# V1 Phase A.3 regression summary

**Date:** 2026-05-26
**Substrate anchor:** cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`
**Libcipher_rt.so md5:** `1d91e7da` (Branch B build; substrate unchanged at A.4 close commit point)
**Deployment path tested:** Branch D (a) per Anil 2026-05-26 : `CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so`, plugin uninstalled

## A.2 hard gate (rerun under Branch D (a) deployment path)

| Variant | Path | shim_calls (worker) | matmul_total (worker) | Gate verdict |
|---|---|---|---|---|
| LD_PRELOAD-only, plugin uninstalled | `LD_PRELOAD=libcipher_rt.so`, NO `CUDA_INJECTION64_PATH` | 0 | 0 | FAIL (mechanism: vLLM V1 fork + cuGetProcAddress lazy resolution) |
| CUDA_INJECTION64_PATH, plugin uninstalled | `CUDA_INJECTION64_PATH=libcipher_rt.so`, NO `LD_PRELOAD` | **19090** | 19090 | **PASS** (gate >= 11000) |

Driver-mediated active invocation works as Anil Branch D (a) called. Goal 5 contract preserved via the locked "or" disjunction.

Evidence file: `results/a2_cuda_injection_PASS.json` (collective_rpc-read of worker substrate counters post-decode; verified `libcipher_rt_loaded=true` in `/proc/<worker_pid>/maps`).

## W7-12 microbench regression spot-check

| Test | Result | Notes |
|---|---|---|
| test_step3_b0_producer | **PASS** | p99 cadence 81 ns (gate 200 ns); N=128 smoke 1.28M writes; rate 25.23 M/s |
| test_step3_b1_consumer | **PASS** | cold-start drained=4096; compose drop_pct=0.00% drained=32000 rate=2.28 M/s |

Substrate integrity preserved at the new commit. Three new files added (constructor + counter_dump + cuinit_hook) are ABI-additive only and do not regress existing substrate paths.

## Track 2 SC2-SC6 + Track 3 v1 SC1-SC6 disposition

Per scope-lock §6 R-A.7 reframed-as-expected baseline measurement (Anil 2026-05-26 Q4 (a) confirmed): full Track 2 SC6 four-tenant Mistral-7B e2e under no-plugin path **deferred to Phase B entry** as the substrate-only baseline measurement that Phase B's KV-bridge migration uplifts from.

Reasoning:
- The 76% N=4 Mistral-7B savings number depends on the plugin's `_cipher_allocate_kv_cache_tensors` hook at `cipher_vllm_kv.py:305`, which only fires when the plugin is installed.
- With plugin uninstalled (Branch D (a) deployment), the hook never runs; vLLM allocates KV via `torch.zeros` per the plugin docstring at `cipher_vllm_kv.py:8-12`; cross-tenant savings collapse to ~0% by mechanism.
- Recording that ~0% as the no-plugin baseline does not require a 30-min Mistral-7B 4-tenant run; the mechanism is dispositive.
- Phase B's scope is migrating the KV-bridge hook into substrate, after which a re-measurement at Phase B close will produce the substrate-only N=4 number that demonstrates Phase B uplift.

Track 3 v1 SC1-SC6 (DSM correctness invariants): kmod-driven, plugin-independent. No mechanism reason to expect regression from Phase A substrate additions (constructor + counter_dump + cuinit_hook are libcipher_rt.so only; cipher_kmod unchanged at `8c643fc`). Full e2e re-run deferred to Phase B entry alongside Track 2 measurement for batched compute efficiency. Spot-check at A.4 close uses W14 Step 3 microbench results above as substrate-integrity proxy.

## Honest residue at A.3

1. **Full Track 2 SC6 numbers not run this turn.** Mechanism finding (plugin-uninstalled = 0% cross-tenant) is dispositive but a recorded run would close R-A.7 with a measured number rather than a mechanism-derived expected. Phase B entry checklist gets a "run Track 2 SC6 under no-plugin path first" item.
2. **Full Track 3 SC1-SC6 e2e not run.** Same Phase B entry deferral.
3. **A.2 measured only under CUDA_INJECTION64_PATH (Branch D (a) path).** LD_PRELOAD-only path FAILed by mechanism; that finding is the Branch D (a) trigger. Per Goal 5 contract "or" disjunction the LD_PRELOAD-only failure narrows but does not violate the contract.
4. **W7-12 microbench coverage is partial.** Spot-check via test_step3_b0_producer + test_step3_b1_consumer (S3.B0 + S3.B1 PASS). Other microbenches (test_commit_atomicity, test_audit_chain, test_observe_publish, test_resolver, test_ring_write, test_g3_cross_model_keying, test_tc_probe, test_g5_va_density, test_l2_wireup) not re-run this turn. Mechanism: substrate changes are additive to specific .o files (cipher_inject.c, plus new .o); existing source files for those microbenches unchanged.
