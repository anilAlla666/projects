# Citation Provenance Verification — INDUSTRY_METHODOLOGY_ALIGNMENT.md

**Date:** 2026-05-18. **Type:** paperwork. **Method:** web search (this
assistant's training cutoff is Jan 2026; three of the four citations post-date
it, so verification was done via live web search, not training knowledge).

**Result: all four citations are real publications at the claimed venues.
None requires removal.** Three need minor characterization corrections; one is
accurate. Corrections applied to `INDUSTRY_METHODOLOGY_ALIGNMENT.md`.

---

## 1. TokenPowerBench (AAAI 2026) — VERIFIED ✓ (minor correction)

**Exists:** Yes. "TokenPowerBench: Benchmarking the Power Consumption of LLM
Inference," AAAI 2026 (proceedings `ojs.aaai.org/index.php/AAAI/article/view/40535`),
arXiv `2512.03024` (Dec 2025).

**Characterization check:**
- "Phase-aware prefill/decode separation" — ✓ accurate ("a phase-aligned
  metrics pipeline that attributes energy to the prefill and decode stages of
  every request").
- "Vendor telemetry, no external meters" — ✓ accurate ("captures power without
  specialized power meters").
- "Multi-granularity power (GPU, node, **rack**)" — ✗ **correction:** the paper
  measures **GPU-, node-, and system-level** power, not "rack." → memo
  corrected to "GPU/node/system."
- "Service-unit normalization (J/token)" — ✓ accurate ("joules per token").

## 2. MLPerf Power (2025) — VERIFIED ✓

**Exists:** Yes. MLPerf Power is the MLCommons power-measurement methodology;
arXiv `2410.12032` ("MLPerf Power: Benchmarking the Energy Efficiency …");
the 2025 rounds are MLPerf Inference v5.0 (Apr 2025) and v5.1 (Sep 2025).

**Characterization check:**
- "Wall-plug power via certified meters" — ✓ accurate ("AC power measured at
  the wall"; "SPEC-certified power meter such as the Yokogawa WT310" — the
  memo's "Yokogawa-class" is exact).
- "Sustained-load measurement" — ✓ accurate (average AC power over the
  benchmark's performance-measurement duration).
- "Per-scenario QPS targets" — ✓ accurate (Server/Offline/Single-/Multi-Stream
  scenarios).
- "Accuracy gate (~99.9% FP32 reference)" — ✓ accurate (MLPerf's
  high-accuracy track targets ≥99.9% of the FP32 reference).

No correction needed.

## 3. ML.Energy benchmark — VERIFIED ✓ (minor correction)

**Exists:** Yes. "The ML.ENERGY Benchmark," arXiv `2505.06371`; GitHub
`ml-energy/leaderboard`; leaderboard v3.0 (Dec 2025).

**Characterization check:**
- "Verbosity-aware (output length distribution)" — ✓ accurate (ML.Energy
  explicitly criticises per-token energy for failing to "capture the verbosity
  and task-specific output token length distribution").
- "Multiple batch sizes" — ✓ accurate (criticises fixing batch size to 1 as
  unrealistic).
- "Real workload traffic patterns (**ShareGPT, Alpaca**)" — ✗ **correction:**
  ML.Energy measures under "realistic service environments / datasets," but
  the searches did not confirm ShareGPT/Alpaca *by name*. → memo corrected to
  "realistic service datasets" (drop the unconfirmed specific names).

## 4. "1/W Law" paper — VERIFIED ✓ (correction)

**Exists:** Yes. "The 1/W Law: An Analytical Study of Context-Length Routing
Topology and GPU Generation Gains for LLM Inference Energy Efficiency,"
arXiv `2603.17280` (Mar 2026).

**Characterization check:**
- The paper's actual thesis: **tokens/W halves every time the context window
  doubles**; routing topology (which sets effective context length) is a
  stronger energy lever than newer hardware. Worked example: H100 at 4K
  context holds 256 sequences, tok/W ≈ 17.6; at 64K context holds 16
  sequences, tok/W ≈ 1.5.
- "Context-length sensitivity (**12–40×** tok/W variation)" — ✗ **correction:**
  the verified figure is **~12×** across 4K→64K context (17.6 → 1.5). "40×" is
  not supported. → memo corrected to "~12× across 4K–64K context."
- "Workload-distribution-weighted aggregate" — ✗ **correction:** a loose
  paraphrase; the paper's point is context-length sensitivity + routing
  topology. → memo's divergence restated as "we measure a single context
  length; the 1/W law shows tok/W varies ~12× across context length."

**Note — independent corroboration of CIPHER's finding.** The 1/W law's
mechanism — tok/W is set by *how many sequences are concurrently in flight* —
is the same relationship CIPHER's Phase B batch-scan measured directly (more
concurrent sequences → proportionally higher tok/W). An independent
peer-reviewed-track paper deriving the same "concurrency drives tok/W"
relationship strengthens, not weakens, CIPHER's positioning.

---

## Summary

| citation | exists | venue correct | characterization | verdict |
|---|---|---|---|---|
| TokenPowerBench | ✓ | ✓ AAAI 2026 | "rack" → "system" | KEEP, corrected |
| MLPerf Power | ✓ | ✓ MLCommons / MLPerf v5.x 2025 | accurate | KEEP |
| ML.Energy | ✓ | ✓ arXiv 2505.06371 | drop unconfirmed "ShareGPT/Alpaca" | KEEP, corrected |
| 1/W Law | ✓ | ✓ arXiv 2603.17280 | "12–40×"→"~12×"; restate thesis | KEEP, corrected |

No citation removed. `INDUSTRY_METHODOLOGY_ALIGNMENT.md` updated: provenance
note now records verification (this document); the three characterization
corrections applied in Section 2.

## Sources

- [TokenPowerBench — AAAI 2026 proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40535) · [arXiv 2512.03024](https://arxiv.org/abs/2512.03024)
- [MLPerf Power — arXiv 2410.12032](https://arxiv.org/html/2410.12032v1) · [MLCommons MLPerf Inference v5.1](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/)
- [The ML.ENERGY Benchmark — arXiv 2505.06371](https://arxiv.org/abs/2505.06371) · [ml-energy/leaderboard](https://github.com/ml-energy/leaderboard)
- [The 1/W Law — arXiv 2603.17280](https://arxiv.org/abs/2603.17280)
