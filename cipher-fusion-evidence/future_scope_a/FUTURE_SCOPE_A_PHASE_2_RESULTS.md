# FUTURE_SCOPE / A — PHASE 2 — vLLM TRANSPARENCY VERIFICATION — RESULTS

**Date:** 2026-05-19. **Verdict: TRANSPARENCY PASS** — vLLM 0.21.0 runs
coherently under CIPHER injection, eager and graph mode. One item flagged
forward: a ~3–4 % single-run overhead (above the sub-1 % target) → Phase 3.

Anchors verified unchanged: kmod `008b3c66`, libcipher_rt `83afd1ca`,
libcipher_v2 `cc0479b8`, cipher_kv_bridge `c04b0c39`.

---

## Method

`phase2_vllm_probe.py` — one vLLM 0.21.0 instance, TinyLlama-1.1B, greedy
decode (`temperature=0`) of a fixed prompt, 128 tokens. Run four ways:
plain vs `CUDA_INJECTION64_PATH=libcipher_rt.so`, × eager vs graph mode.
Greedy decode is deterministic — token_ids must be byte-identical or the
substrate has perturbed output.

## Results

| run | mode | tok/s | n_tok | token_ids == baseline |
|---|---|---|---|---|
| vLLM alone | eager | 104.0 | 128 | (baseline) |
| vLLM alone | graph | 546.0 | 128 | ✅ identical |
| **vLLM + injection** | **eager** | **101.1** | 128 | ✅ identical |
| **vLLM + injection** | **graph** | **524.6** | 128 | ✅ identical |

**All four runs produced byte-identical token_ids and identical output text.**

## Gate criteria

| criterion | result |
|---|---|
| vLLM starts cleanly under injection | ✅ PASS — both modes, rc=0, no crash |
| output coherent vs vLLM-alone | ✅ PASS — **byte-identical token_ids**, all 4 runs |
| substrate observability functioning | ✅ PASS — kmod `/proc/cipher/stats` observing; libcipher_rt injected (banner below) |
| no silent regressions in vLLM's worker model | ✅ PASS — EngineCore subprocess works; **no zombie processes**; GPU returns to 0 MiB |
| telemetry overhead sub-1 % | ⚠️ **NOT MET** — single-run **−2.8 % eager / −3.9 % graph**; see below |

## R1 — CUDA graphs vs interception — RESOLVED (favorably)

The design-memo's largest risk. vLLM graph mode under injection: vLLM captured
its full CUDA-graph set (`cudagraph_mode=FULL_AND_PIECEWISE`, capture sizes
1…512) **while libcipher_rt's CUPTI kernel-launch subscription was active**,
replayed them, and produced byte-identical output. **No crash, no corruption,
no graph-capture error.** CUDA graphs and the CIPHER substrate compose at the
single-instance level.

## R3 — vLLM worker process model — MAPPED

vLLM 0.21.0 (V1 engine) runs **two processes**: the main `LLM()` process and a
spawned **`EngineCore` subprocess** (the EngineCore owns the model + CUDA
context). `CUDA_INJECTION64_PATH` set on the launching shell is **inherited by
the EngineCore child** — libcipher_rt injects into the CUDA-owning process at
its `cuInit`. Confirmed by the injection banner in the EngineCore's output:

```
[cipher_v2] GREEN/CP54: ALLOCATE ok — qos=1 sm_count=0 -> grp_mask=0x0000 grp_count=0
[cipher_v2] CUPTI subscribed: kernel launch callbacks active (flush every 256 launches)
[cipher_v2] MATMUL: substrate initialized
[cipher_v2] MARLIN: actuator DISABLED (CIPHER_MARLIN not set)
```

(libcipher_rt.so logs under the `[cipher_v2]` prefix — the runtime shares the
v2 logging namespace; the injected library is libcipher_rt as set.)
`GREEN/CP54: ALLOCATE qos=1 sm_count=0 -> grp_mask=0x0000` — a **zero-group
allocation**: no SM restriction, vLLM runs full-GPU. This is exactly the
"observing only, not yet partitioning" state Phase 2 intended — actual SM
partitioning is Phase 5.

## The flagged item — overhead

Single-run tok/s under injection: **−2.8 % eager (104.0 → 101.1), −3.9 % graph
(546.0 → 524.6).** This is **above the sub-1 % target** and is reported
honestly as not-met, not papered over. It is a single run, not a rigorous
measurement — but the direction and magnitude are consistent.

**Likely source:** libcipher_rt's CUPTI kernel-launch callbacks (`flush every
256 launches`) — a per-launch cost on vLLM's hot path. Phase 2 explicitly does
not measure performance parity; **quantifying this overhead rigorously
(multi-rep) and root-causing it is Phase 3's job.** If CUPTI tracing is the
cost and it is not load-bearing for the composed architecture, disabling it on
the vLLM hot path is a candidate mitigation — a Phase 3 finding to pursue.

## Verdict

**Phase 2 PASS — transparency verified.** vLLM 0.21.0 runs under CIPHER
CUDA-injection, eager and graph, with byte-identical output and an intact
worker model. R1 (CUDA graphs) and R3 (worker process model) are resolved.
The ~3–4 % overhead is **flagged, not hidden** — it is the central question
Phase 3 (performance parity) must quantify and root-cause.

**STOP for adjudication before Phase 3.** Evidence: `phase2_vllm_probe.py`,
`p2_{alone,inj}_{eager,graph}.json`, `inj_{eager,graph}.log`.
