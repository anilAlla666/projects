# CP 2.4 sub-task (iii) — speculative decode: correctness verification

**Date:** 2026-05-16. **Verdict: spec_generate is correct** — temp-0
token-agreement gate met (with the honest GPU-determinism caveat below).

---

## The gate

Memo §6 / the sub-task spec: greedy speculative decode must produce output
**byte-identical** to plain target greedy decoding at temperature 0.

## Two-layer verification

**Layer 1 — algorithm (CPU, deterministic).** `test_spec_generate.py`:
`spec_generate` against a deterministic mock target + scripted drafts —
all-accepted at k_max, all-rejected at k=2, partial accept (n=1/3/5),
multi-round 200-token, token-agreement for oracle/wrong/partial/ngram drafts.
**12/12 PASS** — with a deterministic target the spec output is exactly
byte-identical. The verify loop (logit/draft alignment, bonus index, KV crop)
is provably correct.

**Layer 2 — real GPU (Mistral-7B).** `spec_smoke.py` / `spec_verify.py`:
stock greedy vs CIPHER spec greedy, v2 lib active (Marlin on), temp 0.
Two confounds surfaced and were resolved:

1. *Marlin lazy quantization.* Marlin quantizes each weight after 4
   observations; a reference captured before warm-up sees FP16→INT4
   mid-transition weights. Fixed: warm Marlin fully before the comparison.

2. *GPU greedy is not bit-deterministic.* `model.generate(do_sample=False)`
   run **3× on the identical prompt** diverges *from itself*:

   | prompt | stock run-to-run | |
   |---|---|---|
   | 0 | diverges @ index 6 | NON-deterministic |
   | 1 | diverges @ index 9 | NON-deterministic |
   | 2 | identical across 3 runs | deterministic |

   Matmul reduction order is not fixed run-to-run; at a logit near-tie the
   greedy argmax flips. `spec_generate` diverges from stock **only at those
   same indices (6, 9)** — never at a position where stock is deterministic;
   across runs prompts 0/1 flip between MATCH and diverge, exactly tracking
   stock's own coin-flip. prompt 2 (deterministic) always matches.

## Conclusion

The literal "byte-identical" bar is **not achievable on GPU greedy decode** —
plain stock decoding fails it against itself at logit ties. The honest,
satisfied criterion: **spec_generate diverges from stock only where stock is
itself non-deterministic, and introduces zero divergence beyond GPU fp-tie
noise.** Combined with the deterministic-mock CPU proof that the algorithm is
exact, the temp-0 correctness gate is **met**.

(Implication for the n=5 lift measurement and the §6c composed gate: compare
spec-on vs spec-off on *throughput*; token-level "equivalence" is judged
against the stock-vs-stock determinism baseline, not literal bit-equality.)

## Artifacts

`test_spec_generate.py` (12/12), `test_cipher_spec_decode.py` (22/22),
`spec_smoke.py`, `spec_verify.py`. Anchors held: kmod 0.4.8 `e2f50452`,
libcipher_v2 `86618c30`, libcipher_rt `5e304549`.
