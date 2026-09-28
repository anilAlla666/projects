# CP 2.4 sub-task (iii) — speculative decode: measurement methodology

**Date:** 2026-05-16. Defines the workload and statistics for the spec-decode
per-lever gate (memo §6). Companion to `SPEC_DECODE_DESIGN_MEMO.md` and
`SPEC_DECODE_CORRECTNESS.md`.

---

## 1. Why this document exists — workload representativeness is a gate-validity issue

Speculative-decode lift is **acutely workload-dependent**. The lift is governed
by the draft's acceptance rate, and acceptance depends on the text being
generated:

- **n-gram (prompt-lookup) draft** — proposes by finding the most recent
  earlier occurrence of the last *n* tokens. On repetitive text it hits almost
  every round; on diverse text it mostly misses (memo §4.3: ~0.59 match
  expected on representative text).
- **model draft** — a small LM predicting the large model's next tokens; less
  workload-sensitive than n-gram, but still varies by genre (code vs prose).

A lift measured on an unrepresentative workload is not a valid gate number.

### 1.1 The original single-prompt measurement (superseded — NOT an upper bound)

The first Mistral-arm measurement (`spec_measure_result.json`,
`run_spec_measure.sh`) used a single trivial prompt — `"Hello, world."`
greedy-decoded for **32** tokens — and reported accept_rate 0.992, lift
**1.50× (95 % CI 1.44–1.56)**. It was first labelled a "pathological-best
upper bound." **The varied-prompt re-measurement (§3) disproves that label:**
varied prompts × **128** tokens give **1.64× (95 % CI 1.60–1.68)** — *higher*
than the trivial-prompt run. The original number was not an upper bound; it
was confounded by a short 32-token budget (see §1.2). It is retained only as
a historical artifact and is **not** the Mistral-arm result.

### 1.2 What actually drives the lift — greedy decoding loops

The decisive finding from the varied-prompt run: under **greedy (temp-0)
decoding** — which the spec-decode correctness gate *requires* (byte-identical
output, `SPEC_DECODE_CORRECTNESS.md`) — a base model's output **collapses into
exact repetition within ~30–50 tokens, on every prompt genre.** Evidence: the
n-gram draft's conditional acceptance (`accepted / proposed`) pins at exactly
**1.000** on all 25 varied-prompt generations. The n-gram draft proposes only
on an exact n-gram match and is then always correct — it is, in effect, a
loop replayer.

Consequences for measurement honesty:

- **`accept_rate` (accepted/proposed) is a greedy-loop signature, not draft
  quality** for the n-gram arm — it ignores the rounds where the draft
  proposed nothing. The honest lift driver is **tokens-per-verify-round**
  (`gen_tokens / rounds`) — it counts every round. Mistral arm: **~1.96
  tok/round.** `analyze_spec_varied.py` reports both.
- **Generation length is the first-order lift factor**, not prompt diversity:
  a longer generation has a larger in-loop fraction → more lift (32 tok →
  1.50×; 128 tok → 1.64×). Prompt genre is second-order (factual 1.51× …
  conversational 1.69×).
- The reported lift is a **greedy-decode-regime** number. Sampled decoding
  (temp > 0) falls through to stock generate (`_spec_eligible` in
  `cipher_spec_decode.py`) — outside what spec decode accelerates. This
  scoping is stated explicitly in the CP 2.4 report.

The varied prompt set still earns its place: it shows the genre spread
(factual is hardest for the draft) and is the honest, non-cherry-picked
workload — but the dominant lift mechanism is greedy repetition, and that is
reported as the finding, not hidden.

## 2. The varied prompt set

Five prompts, one per genre, open-ended (a fixed 128-token budget will not hit
EOS early):

| # | genre | prompt (head) |
|---|---|---|
| 0 | factual | "The Pacific Ocean is the largest ocean on Earth, covering approximately" |
| 1 | narrative | "Sarah opened the old letter with trembling hands. The handwriting was her grandmother's, and the date read" |
| 2 | code | "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =" |
| 3 | reasoning | "If a train leaves Chicago at 3 PM traveling east at 60 mph and another leaves New York at 4 PM traveling west at 80 mph," |
| 4 | conversational | "User: What are the main differences between supervised and unsupervised learning?\nAssistant: The main differences are" |

Rationale: the five genres span the acceptance-rate range a real deployment
sees — code and conversational text are comparatively predictable (higher
acceptance), narrative and reasoning are more divergent (lower acceptance).
Averaging across genres gives a lift number that generalizes, instead of one
pinned to a single text style.

## 3. Measurement protocol

- **n = 5 matched pairs.** Each pair runs both arms (OFF = Marlin-only,
  `CIPHER_SPEC=0`; ON = Marlin + spec, `CIPHER_SPEC=on`). Both arms
  `CIPHER_MARLIN=on`, `CIPHER_VOLT=off`.
- **5 prompts × 128 generated tokens** per arm per pair → **25 generations per
  arm, 50 per workload arm total.**
- One fresh process per (pair, arm) — matched-pair isolation, no cross-run KV
  or allocator state.
- Per (pair, prompt): `lift = on tok/s / off tok/s`. tok/s = generated tokens /
  wall time, CUDA-synchronized, Marlin fully warmed before timing.
- **25 lift values → flat mean, 95 % CI = mean ± 1.96·SE.**

### 3.1 Statistics caveat — the 25 points are not i.i.d.

The 25 lift values are 5 prompts × 5 correlated replicates; replicates of the
same prompt are not independent of each other. The flat mean + 95 % CI (memo
§6 / user-specified) treats them as 25 points — the CI is therefore an
approximation, mildly optimistic on width. To keep the structure visible the
analysis (`analyze_spec_varied.py`) **also reports the per-prompt subgroup
means**, so a single hard-prompt outlier is surfaced, not buried in the
average.

## 4. The two arms

| arm | target | draft | criterion |
|---|---|---|---|
| **Mistral** | Mistral-7B-v0.1 | n-gram (prompt-lookup) | **secondary** — lift documented, no pass target |
| **Llama** | Llama-3.1-8B | Llama-3.2-1B-Instruct (model draft) | **primary gate — 1.75× lift over Marlin-alone** |

## 5. ModelDraft — conservative stateless draft (Llama arm)

`ModelDraft.propose()` does a **fresh full-sequence prefill** of the 1B draft
each round, with no draft-KV carried across rounds. A carried draft KV is
incorrect without a feedback-keyed crop on the per-round accept count (the KV
would hold the *rejected* draft tokens and miss the committed bonus token; it
also leaked across prompts in the original code — a real bug, fixed here). The
correct incremental draft KV is a measured **~10–20 % optimisation** deferred
to Phase 5 — it is the same off-by-one surface that produced three verify-loop
bugs, so it is deliberately off the gate-measurement path.

**Consequence:** the Llama-arm lift is measured with a conservative draft —
the reported number is a slight **under**estimate of an optimally-implemented
draft. This is the honest, gate-safe direction: a pass at 1.75× with the
conservative draft is a robust pass. If the result lands just under 1.75× by
less than this margin, that is reported honestly with this caveat — the draft
is **not** re-optimized to chase the gate.

## 6. Artifacts

`run_spec_varied.sh`, `spec_varied_driver.py`, `analyze_spec_varied.py`;
results `spec_varied_mistral_result.json`, `spec_varied_llama_result.json`.
Superseded original single-prompt artifact preserved (historical):
`run_spec_measure.sh`, `spec_measure_result.json`. Anchors held: kmod 0.4.8
`e2f50452`, libcipher_v2 `86618c30`, libcipher_rt `5e304549`.
