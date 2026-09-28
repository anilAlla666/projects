# FUTURE_SCOPE/C — 3-Arm Cross-Process Benchmark (vLLM included)

**Hard gate before any external / Peak XV performance claim.** Per
`INDUSTRY_METHODOLOGY_ALIGNMENT.md` §3.

## Goal

Measure CIPHER's value on the **architectural axis** — cross-process fusion —
rather than as a tok/W metric race against vLLM (vLLM is a throughput/latency
framework and makes no primary tok/W claim; a metric race is apples-to-oranges).
The benchmark answers: on the workload where the architectural distinction is
the whole point — N separate customer processes each running B=1 decode — what
does CIPHER's cross-process substrate add over the realistic alternatives?

## Architectural approach — three arms

N separate Python processes, each loading its own model, B=1 decode; identical
hardware, prompts, and arrival pattern across arms:

- **Arm 1 — naive:** each process runs independently, no fusion. (CP 5.6's
  current substrate-attributable baseline.)
- **Arm 2 — vLLM intra-process:** each process runs vLLM internally — vLLM
  continuous batching engages *within* a process, never *across* processes.
  With one request stream per process, vLLM has nothing to batch
  intra-process, so Arm 2 ≈ Arm 1 for this topology — *that is the finding*:
  it shows by measurement where vLLM structurally cannot help.
- **Arm 3 — CIPHER substrate (over vLLM):** processes independent from the
  tenant's view; CIPHER fuses decode across them transparently.

The number that justifies the substrate-layer integration cost over a
vLLM-only alternative is **Arm 3 vs Arm 2**.

## Build scope + dependencies

- **Depends on:** the CIPHER + vLLM composed architecture (`FUTURE_SCOPE/A`) —
  Arm 3 requires it; vLLM installed (Arm 2).
- **Scope:** the 3-arm harness; the methodology spec is already written
  (`INDUSTRY_METHODOLOGY_ALIGNMENT.md` §4 — identical arrival pattern + prompt
  sequences across arms; aggregate tok/W on useful gated tokens; TTFT/ITL;
  run each arm 3× report mean+range).

## Time estimate

1–2 days (after A lands).

## Success criteria

- Arm 3 vs Arm 2: a measurable, correctness-gated substrate-attributable lift.
- Per-tenant correctness gate (teacher-forced KL ≤ 0.1).
- Both throughput/power **and** TTFT/ITL latency reported — no throughput claim
  without the latency cost shown.
- Result framed as the architectural distinction (cross-process fusion), not a
  tok/W-vs-tok/W race. Citation provenance for external standards verified
  before the result is used externally.
