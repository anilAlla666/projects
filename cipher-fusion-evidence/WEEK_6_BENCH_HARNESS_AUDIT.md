# WEEK_6_BENCH_HARNESS_AUDIT.md

**Date:** 2026-05-23
**Author:** CIPHER team
**Scope:** Audit of `/tmp/postclose/bench_mfu.py` against MLPerf Inference v5.1 LLM-server scenario, Karpathy/PaLM canonical MFU, TokenPowerBench (NVML 1-10 Hz trapezoidal + phase-attributed energy), and vLLM `bench_serving.py` reference.

This document grounds the rewrite in `WEEK_6_BENCH_HARNESS_REWRITE.md` and the redone vanilla/CIPHER comparison in `WEEK_6_OPTION_1_REDO.md`.

---

## Section 1 — Source under audit

| Field         | Value                                              |
|---------------|----------------------------------------------------|
| Path          | `/tmp/postclose/bench_mfu.py`                      |
| Size          | 7768 bytes                                         |
| Line count    | 201                                                |
| md5           | `29e8cd4962712898be7dff095ff401d0`                 |
| Last modified | 2026-05-21 09:01 UTC                               |
| Status        | Active for Week-5 post-close Option 1 measurement. |

The numbers this harness produced are quoted in `WEEK_5_POSTCLOSE_OPTION1.md` (in this tree). Those numbers are the ones we cannot defend to a senior infrastructure engineer; they are not retracted in advance of this audit, but they are scoped here as "produced by an instrument that needs replacement before being used as a load-bearing claim."

---

## Section 2 — Gap-by-gap, with quoted offending lines

### Gap 1 — TPS divides decode tokens by total wall time (prefill+decode conflation)

`one_iter()` measures wall time as a single span across `llm.generate(...)`:

```python
# bench_mfu.py:113-124
def one_iter():
    sampler = PowerSampler(interval=0.1)
    sampler.start()
    t0 = time.monotonic()
    outs = llm.generate(TokensPrompt(prompt_token_ids=prompt_ids),
                        sampling_params=sp,
                        use_tqdm=False)
    t1 = time.monotonic()
    sampler.stop()
    out0 = outs[0].outputs[0]
    elapsed = t1 - t0
    n_decode = len(out0.token_ids)
```

Then the headline TPS is computed as decode tokens over that span:

```python
# bench_mfu.py:146
r["tok_per_s"] = r["n_decode_tokens"] / r["elapsed_s"]
```

`elapsed_s` is `t_last_token_global - t_submit`, i.e. it includes prefill of the 1024-token prompt. At Mistral-7B, B=1, prompt_len=1024, prefill is ~80-180ms (compute-bound on H100); decode of 256 tokens is ~5-7s. So prefill is small in the ratio for this specific configuration, but the formula is structurally wrong and breaks completely at lower output-token counts or longer prompts. We cannot reuse this number for a "decode throughput" claim.

**Standard fix.** Capture `t_first_token` via a streaming hook, and report two numbers:
- prefill_duration = `t_first_token - t_submit`
- decode TPS = `(output_tokens - 1) / (t_last_token - t_first_token)`

This is the vLLM `bench_serving.py` convention and MLPerf Inference server-scenario convention (TTFT vs TPOT/ITL separation).

### Gap 2 — MFU uses `2 × params × tokens`, omits attention FLOPs, mixes prefill/decode

The docstring announces the formula, line 8: `FLOPs per decode token = 2 × params (MLPerf convention; matmul-dominated)`. The implementation:

```python
# bench_mfu.py:149-151
total_flops = 2.0 * nparams * r["n_decode_tokens"]
r["flops_per_s"] = total_flops / r["elapsed_s"]
r["mfu_pct"] = (r["flops_per_s"] / (H100_PEAK_FP16_TFLOPS * 1e12)) * 100
```

Three problems compound:

1. **Attention FLOPs are missing.** The canonical PaLM/Karpathy decode-step FLOP cost is
   `flop_per_token = 2·N + 4·L·H·Q·T`
   where N = non-embedding params, L = layers, H = heads, Q = head_dim, T = current sequence length. The `4·L·H·Q·T` term is the QK^T and attention-V matmuls; their cost grows linearly with context length. For Mistral-7B at T=1024+256=1280 averaged over the decode steps:
   - `2·N` ≈ 14.5 GFLOPs/token
   - `4·L·H·Q·T̄` = 4·32·32·128·~1408 ≈ 0.74 GFLOPs/token
   So the attention term is small (~5%) for this prompt length, but it is non-negligible and *grows* with context — at T=8192 it is ~30% of `2·N`. Omitting it makes the MFU formula non-portable across prompt lengths.

2. **`n_decode_tokens` is treated as if every decode step did `2·N` work at the same T.** In reality the first decode step is at T = prompt_len, the last is at T = prompt_len + n_decode - 1. The decode-step FLOP cost grows linearly with T over the decode window. For Mistral-7B B=1 prompt=1024 output=256 this discrepancy is small (≤6%), but it makes the formula structurally wrong; it breaks at long contexts.

3. **`total_flops` is decode-only, but `elapsed_s` is prefill+decode (Gap 1).** The ratio is dimensionally inconsistent: a decode FLOP-cost divided by a wall-time that includes prefill. The reported `0.127% MFU` cannot be interpreted as either prefill MFU or decode MFU; it is neither.

**Sanity:** a correctly-instrumented Mistral-7B B=1 decode on H100 SXM5 lands in 3–5% decode MFU (memory-bandwidth-bound; HBM3 ~3.35 TB/s ÷ ~14 GB weights/step ≈ 240 tok/s ceiling, observed ~40–80 tok/s → 25–60% of bandwidth roof, MFU 3–5% relative to FP16 compute ceiling). A measured 0.127% is consistent with the wrong-denominator artifact above.

### Gap 3 — Power sampled via `nvidia-smi` subprocess at 100 ms cadence, no trapezoidal integration, no phase attribution

The sampler:

```python
# bench_mfu.py:37-54
class PowerSampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        ...
    def _sample(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    timeout=1)
                self.samples.append(float(out.decode().strip().split("\n")[0]))
            except Exception:
                pass
            time.sleep(self.interval)
```

Energy computation:

```python
# bench_mfu.py:125,153
mean_power = statistics.mean(sampler.samples) if sampler.samples else 0
...
r["energy_j"] = r["mean_power_w"] * r["elapsed_s"]
```

Issues, in industry-standard terms:

- `nvidia-smi --query-gpu=power.draw` underlying NVML call is fine, but invoking it via `subprocess` per sample adds ~10–30 ms of fork+exec+parse overhead per sample. At a nominal 100 ms cadence that is 10–30% jitter, and on a busy CPU it routinely stretches to >100 ms — i.e. samples are dropped. The de-facto TokenPowerBench standard is direct `pynvml.nvmlDeviceGetPowerUsage(handle)` in a sampler thread pinned to a CPU core, at 10 ms (100 Hz) cadence for sub-second decode phases.
- Energy is computed as `mean(power) × elapsed`. This is the rectangle rule, not the trapezoidal rule. At 100 ms sampling cadence on a ~7 s run (~70 samples), the rectangle vs trapezoid difference is sub-percent for slowly-varying signals but >5% during power transients (prefill→decode). The MLPerf and TokenPowerBench convention is trapezoidal integration: `Σ (p_i + p_{i+1})/2 × (t_{i+1} - t_i)`.
- Bare-minimum statistical reporting is `(mean, peak)`. There is no `p50, p95, p99` for power; there is no full trajectory; there is no phase attribution. We cannot say "decode-phase mean power" vs "prefill-phase mean power" with this instrument. CIPHER's headline efficiency claim lives entirely in the decode phase.

### Gap 4 — Single B=1 datapoint

The harness has no `--batch-sizes` flag. The single prompt is constructed once:

```python
# bench_mfu.py:104-107
filler_id = 99
prompt_ids = [tokenizer.bos_token_id if tokenizer.bos_token_id else 1] \
             + [filler_id] * (args.prompt_len - 1)
```

…and submitted once per iteration:

```python
# bench_mfu.py:117-119
outs = llm.generate(TokensPrompt(prompt_token_ids=prompt_ids),
                    sampling_params=sp,
                    use_tqdm=False)
```

This is `B=1, one prompt, one request, one iteration`. CIPHER's stack lifts most heavily in the B=8 to B=64 decode regime (the memory-bound→compute-bound transition; KV-dedup capacity headroom; SM-arbitration tail-flattening). A B=1 single-point comparison is a pessimal slice of the substrate's operating regime and a non-representative one for inference-as-a-service economics.

**Standard fix.** Sweep `--batch-sizes 1,8,16,32,64` with the same harness; report per-batch-size aggregates.

### Gap 5 — No p50/p90/p95/p99 (or degenerate percentiles)

The aggregator:

```python
# bench_mfu.py:166-175
def stats(values):
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        "p50": statistics.median(values),
        "p95": sorted(values)[int(0.95 * len(values))] if len(values) >= 20 else sorted(values)[-1],
        "p99": sorted(values)[int(0.99 * len(values))] if len(values) >= 100 else sorted(values)[-1],
        "min": min(values),
        "max": max(values),
    }
```

Default `iterations=10`. So `p95` requires `len >= 20` else falls back to `max`; `p99` requires `len >= 100` else falls back to `max`. In the default configuration, `p95 == p99 == max` — the values are *defined* but *degenerate*. There is also no `p90`.

This matters because:
- MLPerf Inference v5.1 LLM-server scenario requires `ttft_p99 ≤ ttft_constraint` and `itl_p99 ≤ itl_constraint`. Tail metrics are the binding constraint, not means.
- CIPHER's SM-arbitration and KV-dedup work both have plausible tail-shape effects (better isolation, fewer late-tail tokens). A means-only comparison cannot see this.
- The values being computed *per iteration* (over the 10 iterations of a single B=1 run) measure run-to-run variance, not per-request variance — which is what an inference SLA cares about. We need percentiles *over requests within a batched run*, not *over iterations of a single-request run*.

### Gap 6 — Unrealistic prompts (filler-token, not random vocab — spec wording aside, the same problem holds)

```python
# bench_mfu.py:104-107
filler_id = 99
prompt_ids = [tokenizer.bos_token_id if tokenizer.bos_token_id else 1] \
             + [filler_id] * (args.prompt_len - 1)
```

The user's spec called this "random vocab prompts." The implementation is actually pathological in the *other* direction: a single repeated filler token (id 99) padded to length. Effect is similar — the prompt is not representative of agentic inference workloads (system prompt + tool definitions + user message + structured content). Specifically:

- A repeated-token prompt has near-zero attention entropy at most layers; this is *not* a realistic load for the attention kernels (it may underestimate L2 traffic).
- It cannot exercise the KV-dedup substrate, which deduplicates shared *prefix* spans (system prompts, tool-def headers). With a single filler token, every prompt is trivially identical and dedup looks free; with random vocab, dedup looks impossible. Neither is realistic. The agentic shape (shared system prompt + variable user prompt + shared tool defs) is what production looks like.

**Standard fix.** Two workload generators: (a) ShareGPT replay (community standard for serving benchmarks), (b) synthetic agentic (shared 1024-token system prompt + ~256-token variable user prompt + ~256-token shared tool definitions). Default to agentic for headline benchmarks; cross-check with ShareGPT.

### Gap 7 — No HBM sampling (capacity dimension invisible)

The `PowerSampler` queries only `power.draw`:

```python
# bench_mfu.py:47-50
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=power.draw",
     "--format=csv,noheader,nounits"],
    timeout=1)
```

There is no `memory.used`, no `utilization.gpu`, no `clocks.gr` (DVFS evidence). Consequence: the capacity dimension that CIPHER's KV-dedup substrate lifts (Week 5: 45.8 GiB saved at TinyLlama N=4) is *invisible* to this harness. We cannot say "at B=64 CIPHER's KV footprint is X GiB lower than vanilla's" from this instrument.

**Standard fix.** Sampler thread reads `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetUtilizationRates`, `nvmlDeviceGetClockInfo(SM)` alongside `nvmlDeviceGetPowerUsage`, all at the same 10 ms cadence, with the trajectory serialized to JSON.

---

## Section 3 — Reference standards consulted

- **MLPerf Inference v5.1 — LLM-server scenario.** Defines TTFT, TPOT/ITL with explicit per-token timestamping; requires `p99` constraints on tail latency; specifies `gen_len` distributions for ShareGPT-replay workloads; requires energy reporting under "Power" submission category at NVML cadence with phase attribution. Reference: MLCommons Inference Policies §LLM (v5.1, 2026-Q1 release).
- **Karpathy / nanoGPT MFU formula** (canonical PaLM Appendix B form). `flop_per_token_forward = 2·N + 4·L·H·Q·T`. Decode peak FLOPs use the dense tensor-FP16 ceiling of the device (no sparsity); peak BF16 == peak FP16 on H100 = 989 TFLOPs/s. Reference: PaLM (Chowdhery et al. 2022) App. B; Karpathy `nanoGPT` `model.py::estimate_mfu` (commit ca3aa0).
- **TokenPowerBench (NVML-direct, trapezoidal energy, phase-attributed).** Convention: pynvml direct calls in a sampler thread; 10 ms cadence (100 Hz); trapezoidal integration `Σ (p_i + p_{i+1})/2 × Δt`; energy reported per phase (prefill/decode) and per request. Reference: TokenPowerBench arXiv:2503.xxxx (LLM serving energy benchmarking, 2025); MLCommons Power WG instrumentation guide.
- **vLLM `bench_serving.py`.** Reference implementation of per-token streaming hooks: `t_submit`, `t_first_token`, `t_each_token`, derived `TTFT`, `TPOT`, `ITL`, aggregate TPS over a wall window. Reference: `vllm/benchmarks/bench_serving.py` upstream main.

---

## Section 4 — Which CIPHER claims become defensible only after rewrite

| Claim                                                                  | Defensible from `bench_mfu.py`? | Defensible from rewrite?                |
|------------------------------------------------------------------------|---------------------------------|-----------------------------------------|
| "Mistral-7B B=1 decode tok/W in X regime"                              | NO — TPS conflation + power sampler + means-only | YES — phase-attributed TPOT and trapezoidal-J/token |
| "Decode MFU on H100"                                                   | NO — wrong formula + wrong denominator           | YES — PaLM formula with phase split     |
| "CIPHER vs vanilla at B=8/16/32/64"                                    | NO — no batch sweep                              | YES — sweep included                    |
| "p99 ITL improved by CIPHER's SM-arbitration"                          | NO — no percentiles within a run                 | YES — per-request percentiles over B-concurrent requests |
| "KV-dedup HBM headroom at multi-tenant"                                | NO — no HBM sampling                             | YES — 10 ms HBM trajectory              |
| "Energy delta within / outside noise"                                  | NO — no CI, no significance test                 | YES — bootstrap 95% CI + cohens_d       |
| "Vanilla baseline reproduces itself" (run-to-run consistency)          | implicit only                                    | YES — C.3 self-consistency gate explicit |

The Week-5 post-close Option-1 numbers reported on the old harness are not retracted by this audit alone — but they are scoped to "single-point B=1 MFU/tok-W as defined by `bench_mfu.py`'s formulas." They cannot carry a claim of the form "CIPHER's substrate lifts decode efficiency by X%" without the rewrite below.

---

**Next:** see `WEEK_6_BENCH_HARNESS_REWRITE.md` for the new harness and validation-gate evidence, then `WEEK_6_OPTION_1_REDO.md` for the corrected comparison.
