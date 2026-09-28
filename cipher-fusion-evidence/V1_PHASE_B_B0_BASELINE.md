# V1 Phase B.0 substrate-only baseline measurement

**Date:** 2026-05-26
**Substrate anchor:** cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`; libcipher_rt.so md5 `1d91e7da`; cipher_kmod `8c643fc` tag `week-13-14-complete`
**Type:** read-only measurement campaign; no substrate code change; precedes Phase B scope-lock drafting per Anil 2026-05-26 V1 substrate work sequence Phase B entry prerequisite.

---

## 1. Why this measurement

Per Anil 2026-05-26 Phase B entry directive: "B.0 SUBSTRATE-ONLY BASELINE MEASUREMENT (~0.5 ED): Track 2 SC6 measurement under no-plugin path. Per V1_PHASE_A_COMPLETE.md A.3 disposition: this number is what Phase B uplifts from. Run BEFORE Phase B substrate work starts so the uplift target is documented in advance, not after-the-fact."

The expected outcome per Anil's spec was Track 2 SC6 N=4 Mistral-7B savings collapse to substrate-only baseline (close to 0% by mechanism, since the plugin's `_cipher_allocate_kv_cache_tensors` hook at `cipher_vllm_kv.py:305` was assumed load-bearing for the 76% number).

## 2. What was measured

Track 2 SC6 N=4 weight-sharing integration via `cipher-fusion-evidence/phase_c/sc6_run.py` under:
- `CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` (Branch D (a) deployment per Phase A close)
- cipher-vllm-kv pip-uninstalled from vllm_env Python
- `vllm.general_plugins` entry points: no cipher entries
- Both TinyLlama-1.1B and Mistral-7B-v0.1, shared (N=4) and independent (N=5) phases

Evidence: `v1_phase_b/results/b0_substrate_only_baseline.json`

## 3. Headline finding

**Track 2 SC6 76% N=4 Mistral-7B savings reproduces at substrate-only without cipher-vllm-kv plugin.**

| Model | With plugin (2026-05-23 baseline) | Without plugin (today B.0) | Delta |
|---|---|---|---|
| TinyLlama N=4 | 59.9% | 59.6% | -0.3% (within noise floor) |
| **Mistral-7B N=4** | **76.0%** | **76.0%** | **0.0% (exact match)** |

Both phases (shared + independent control) PASS all 7 SC6 gates per `phase_c/sc6_run.py` integration checks (all4_fwd1_bit_identical, all4_fwd2_bit_identical_after_producer_death, all4_arena_backed, arena_survived_producer_sigkill, producer_pid_cleared, all4_consumers_held_arena, arena_reaped_after_last_participant). Bit-identical forwards confirmed for all 4 consumers in shared phase; 5/5 fit in independent control phase.

## 4. Mechanism diagnosis (why the finding contradicts Anil spec assumption)

SC6's 76% comes from the cipher_kv_bridge weight arena path, not the vLLM-specific KV-cache buffer hook:

- `phase_c/sc6_consumer.py:30` imports `cipher_kv_bridge` (the C extension `.so`); NOT `cipher_vllm_kv` (the pip-installed plugin)
- `phase_c/sc6_producer.py` imports `cipher_kv_bridge` for `register_arena` / `export_fd` calls into the cipher_kmod weight-arena registry (ioctl NR 23/24)
- SC6 uses `transformers.AutoModelForCausalLM` directly with the cipher_kv_bridge ownership pattern; vLLM is NOT in the SC6 process tree
- The plugin's `_cipher_allocate_kv_cache_tensors` hook at `cipher_vllm_kv.py:305` monkey-patches `vllm.v1.worker.gpu_model_runner.GPUModelRunner._allocate_kv_cache_tensors`; SC6 doesn't use vLLM, so this hook is structurally irrelevant to SC6's measurement

Source confirmation:
```
$ grep "import cipher_vllm_kv\|from cipher_vllm_kv" phase_c/sc6_*.py
(empty)
$ grep "import cipher_kv_bridge" phase_c/sc6_consumer.py
import cipher_kv_bridge as kvb                                # noqa: E402
```

## 5. Implication for V1 substrate work sequence Phase B premise

Anil's V1 substrate work sequence locked 2026-05-26 stated CP 5.1 KV-buffer ownership (plugin hook at `cipher_vllm_kv.py:305`) "is the load-bearing path for Track 2 SC6's 76% N=4 Mistral-7B savings." Empirical evidence from this B.0 measurement contradicts that premise: the 76% number is INDEPENDENT of plugin install state.

This requires CP 5.1 mechanism re-verification before Phase B scope-lock drafts. Anil adjudication 2026-05-26 directs a new substep B.0.5 (plugin hook archaeology, read-only investigation) to identify whether the plugin hook is load-bearing for any non-SC6 measurement, or whether it is vestigial / redundant with cipher_kv_bridge.so functionality.

**B.0.5 outcomes that shape Phase B scope:**
- Outcome 1: plugin hook IS load-bearing for some non-SC6 measurement (e.g., vLLM-on-CIPHER cross-tenant KV hit rate, vLLM N-tenant memory consolidation, per-tenant AUDIT-chain KV buffer ownership). Phase B substrate work updates to target THAT measurement.
- Outcome 2: plugin hook is vestigial. No functional value beyond what cipher_kv_bridge.so provides today. Phase B narrows to "deprecate plugin, ledger update, no substrate code".
- Outcome 3: plugin hook is redundant with cipher_kv_bridge.so. Same Phase B narrowing as Outcome 2.

B.0.5 surface determines which outcome applies; Phase B scope-lock drafts AFTER Anil reviews B.0.5 findings.

## 6. Goal disposition (NO AUTO SCOPE-DEGRADE per memory #11)

This B.0 finding is NOT a goal scope-down per any axis:
- **Goal 1** (100-agent heterogeneous-model multiplexing per H100): unaffected. The 76% Mistral-7B N=4 number stands at substrate-only baseline; the asymptotic 95% projection per `phase_c/sc6_aggregate.py` `savings(N)=N·W/(N·W+(N+1)·C)` formula stands. Track 2 weight-sharing substrate works as documented.
- **Goal 5** (LD_PRELOAD-only / CUDA_INJECTION64_PATH deployment transparency): unaffected. Phase A close already restored Goal 5 contract via the "or" disjunction per `v1-goal5-contract-lock` memory + V1_GOAL5_DEPLOYMENT_LEDGER.md. Plugin status was already OPTIONAL post-Phase-A close; B.0 confirms substrate-only deployment also preserves the headline Track 2 SC6 metric.
- **Track 2 SC6 76% headline**: NOT retracted. Reproduces at substrate-only.
- **Phase B narrowing (if B.0.5 yields Outcome 2 or 3)**: substrate work less than initially scoped because the work was already done in earlier substrate phases (cipher_kv_bridge maturity + W7-9 G6 AUDIT chain + Track 2 SC2-SC5 closing). NOT a goal scope-down.

## 7. Plugin disposition this turn

- B.0 measurement: cipher-vllm-kv pip-uninstalled from vllm_env Python per Anil spec
- B.0 close: cipher-vllm-kv pip-reinstalled per Anil Item 5 (vllm_env now has the plugin at md5 `b89a9b6e`, entry points `cipher_vllm_kv` + `cipher_vllm_kvdedup` visible to `importlib.metadata`)
- Plugin file on disk at `/home/ubuntu/cipher_vllm_plugin/cipher_vllm_kv.py` unchanged through B.0 (md5 `b89a9b6e` matches Phase A close baseline)

## 8. Environment state note (audit trail)

The `accelerate` Python package was missing from vllm_env at B.0 start. SC6 consumer failed import (`ModuleNotFoundError: No module named 'accelerate'`). Installed via `/home/ubuntu/vllm_env/bin/pip install accelerate` during B.0 setup. accelerate is a transitive dependency of transformers commonly already in vLLM envs; missing here is an env-snapshot quirk, not a substrate or contract issue. Documented for completeness so a fresh-session anchor verify shows accelerate as installed-in-vllm_env post-2026-05-26.

## 9. Track 3 v1 SC1-SC6 disposition

NOT run this turn. Track 3 DSM SCs per `cipher-track3-dsm` memory ("all 6 SCs ... final anchors kmod 285d102e + libcipher_rt 83afd1ca") are distinct from the Track 2 phase_c SC1-SC6 weight-sharing tests. Track 3 DSM test sequence not located in this turn's evidence trail (no obvious `track3_sc*.py` runnable harness in `phase_c/` or top-level evidence directory). Deferred to Phase B scope-lock entry checklist; cipher_kmod is UNCHANGED through Phase A and would not be touched in Phase B per V1 substrate work sequence spec, so no regression risk on the kmod-resident DSM substrate from Phase B work.

## 10. Anchors at B.0 close (UNCHANGED through measurement)

- cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`
- libcipher_rt.so md5 `1d91e7da`
- cipher_kmod `8c643fc` tag `week-13-14-complete`
- cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e` (file on disk + reinstalled in vllm_env post-B.0)
- cipher-fusion-evidence this commit lands b0_substrate_only_baseline.json + this companion doc
- No tag at this commit per Q2 (b) discipline; B.0 is read-only measurement, not a substrate landmark

---

**Next: B.0.5 plugin hook archaeology (read-only investigation, ~1-2 ED, blocks Phase B scope-lock drafting) per Anil 2026-05-26 surface adjudication.**

Related memory: [[v1-phase-a-driver-worker-init]] (Phase A close baseline); [[cipher-track2-weight-sharing]] (Track 2 SC6 prior baseline 76% with plugin installed); [[v1-goal5-contract-lock]] (deployment-layer ledger discipline); [[cipher-evidence-commit-discipline]] (followed at this B.0 evidence commit); [[cipher-proceed-not-ask]] (Anil "Proceed to Phase B" + B.0.5 substep insertion executed without re-litigation).
