# Judgement

The evidence file contains no runs: the probe crashed (missing grader config). Also, the original test.sh needed
network (apt-get, curl, uv) which the grader does not have, so it would have scored 0 on everything. Rewritten to
run a plain python3 script (numpy only) with no installs.

## Candidates
- A: WRONG. Edge set wrong (V-D, V-Y instead of D-X, W-Y; data gives |pc| .20 for D-X/W-Y vs <.015 for V-D/V-Y);
  samples are the input rows permuted; marginal is a hand-made 3-point distribution (mean 0, ignores evidence).
  Rejected by: edge-set check, samples-generated check, marginal-consistency check.
- B: CORRECT. Passes all.
- C: CORRECT. Passes all.
- D: WRONG. Graph and marginal fine, but final_mrf_sample.csv is a bootstrap resample of the input data (36.7%
  duplicate rows, 100% of rows are input rows) — not "Generate 10,000 samples from the learned MRF model".
  Rejected by: samples-generated check.
- Empty: rejected (files missing).

## Settling values
- Edge set: recomputed. Task names partial-correlation thresholding and fixes the count ("exactly 8 edges"); top-8
  |partial correlation| is choice-free. All readings (pc threshold, glasso alpha .01-.10 raw, .02-.15 standardised)
  give the same set. Readings compared: there is no disagreement to resolve. Verifier computes it at runtime from
  the grading container's /app/mrf_sample_10k.csv (falls back to the hard-coded set).
- Conditional mean/sd of W: CONTESTED (penalty strength of glasso is a choice; mean ranges .05-.21 over readings
  that recover the 8-edge graph; LOO comparison can't separate a value the task's own procedure leaves free).
  Original verifier had no value check; none added. Instead a self-consistency check (marginal vs solver's own
  samples), which holds under every reading, plus a weak "evidence shifts the mean" bound (>0.03, all readings >=0.05).

## Trace: requirement -> check
- "Exactly 8 edges" -> edges format check (8 rows, no duplicates, valid names, no header).
- "node V has the highest degree" -> V strictly highest degree.
- "Recover the underlying graphical structure" -> edge set equals recovered structure (recomputed).
- "each edge should appear only once", "case-sensitive names" -> edges format check.
- "Format: Two columns (no header)" -> edges format (header row would be an invalid node name).
- marginals header "w_value,probability", "sum to approximately 1.0" -> marginal format check (0.95-1.05, >=0).
- "properly account for the evidence V=1.0 and X=0.5" / "Fit the MRF parameters" -> marginal consistent with model.
- "Exactly 10,000 rows", "header row containing all 8 node names" -> samples format check (+ finite values).
- "Generate 10,000 samples from the learned MRF model" / "Generated samples should follow the learned distribution"
  -> samples generated (no copy/resample of input: <1% duplicates, <1% input rows) and samples structure
  (non-edges |pc|<0.06, sane scale/means).
- "within the timeout" -> not checkable from outputs; not checked.

## Trace: original checks
- exist, edge count, V highest degree, valid names, no duplicates, marginal format, prob sum, sample count, columns,
  no NaN: kept (task-text/format).
- "V degree >= 3" and "graph connected": not required by the task as such; subsumed by the edge-set check; removed.
- Original V-highest-degree used max() which on ties picks first: replaced with strict comparison ("highest").
- solve_mrf.py: not required by the task; not checked.

## Near misses (all copy C, break one thing) — all rejected, B/C accepted
nm_wrong_edge (W-Y -> V-Y; old verifier would accept), nm_ignores_evidence (unconditional W marginal),
nm_swapped_evidence (V=0.5,X=1.0), nm_bootstrap_samples, nm_shuffled_input (shortcut), nm_independent_samples,
nm_9999_rows. Also tested two alternate valid solutions (empirical cov; glasso alpha .1) -> accepted.
