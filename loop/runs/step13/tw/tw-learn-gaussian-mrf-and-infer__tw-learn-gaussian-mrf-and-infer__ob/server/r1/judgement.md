# Judgement

## Own solution / recomputation (task data /app/mrf_sample_10k.csv)
Partial correlations from inverse sample covariance (the task names "partial correlation thresholding";
"exactly 8 edges" fixes the threshold as the top 8). Top 8 |pcorr|: C-V .212, W-Y .208, V-W .208, D-X .202,
V-X .201, A-B .200, A-V .198, B-V .198; 9th is B-D .021 (10x gap). Graphical lasso for every alpha in
[0.01, 0.10] that yields 8 edges gives the same set. V degree 5, all others <=2. -> Edge set **recomputed**.

## Candidates
- **A: wrong.** Edges V-{A,B,C,D,W,X,Y}+A-B: contains V-D, V-Y (pcorr .009/.014), misses W-Y, D-X
  (rejected by the edge-set check). Its final_mrf_sample.csv is the input data, row for row
  (10000/10000 rows equal) – not generated (rejected by the copy check). Marginal is a hard-coded
  3-point table (-1:.2, 0:.6, 1:.2). Clearly a fabricated result.
- **B: correct.** Correct edges; marginal mean .2015, sd .722 (= Gaussian conditional from the MLE
  covariance); samples are fresh Gaussian draws (0 rows shared with input) with the learned structure.
- **C: correct.** Same; marginal mean .198, sd .723; fresh samples.
- **D: wrong.** Edges and marginal fine, but all 10000 sample rows are rows of the input data
  (6334 unique, the signature of a bootstrap resample). Task step 4: "Generate 10,000 samples from the
  learned MRF model"; resampling the observations is not sampling from the learned model. Rejected by the
  copy check. (The old verifier accepted A and D.)
- Empty attempt: rejected (files missing).

## Contested: the conditional distribution P(W | V=1, X=0.5)
Readings (all consistent with "Use appropriate methods for Gaussian MRF (e.g., graphical lasso, partial
correlation thresholding)"): conditional mean / sd
- empirical-MLE covariance: .2015 / .7222
- MLE constrained to the 8-edge graph: .2055 / .7223
- graphical lasso alpha .01/.05/.10 (all still give exactly 8 edges): .190/.128/.051, sd .725/.734/.740
10-fold held-out score of each reading for predicting W from (V,X) (MSE / Gaussian NLL):
empirical .52209/1.09419; constrained .52200/1.09410; glasso.01 .52215/1.09426; glasso.05 .52585/1.09790;
glasso.10 .53739/1.10870; even the unconditional-W reading .54996/1.12017.
No reading has held-out error <= half the next best -> not data-settled. No single field-standard
estimator is named (the task names two families, with a free penalty). -> **contested**. The original
verifier had no value check on it; none added. Only format checks remain (header, numeric, distinct
w_values, >=2 rows, p>=0, sum in [0.95,1.05] – the original's tolerance for "approximately 1.0").

## Trace: requirements -> checks
- "Exactly 8 edges" -> count == 8 (kept).
- "node V has the highest degree" -> V strictly highest (kept, made strict instead of max() tie-break).
- "Recover the underlying graphical structure" -> edge set equals recomputed top-8 partial-correlation set (added; recomputed).
- "each edge should appear only once" -> unique undirected edges (kept).
- "case-sensitive ... A..Y" -> node names validated (kept); self-loops rejected (format).
- inferred_marginals header "w_value,probability", "sum to approximately 1.0" -> format checks (kept/tightened header to exact).
- "Discretized probability distribution over W values" -> >=2 distinct numeric w_values (format).
- Conditional values -> contested, no check.
- final_mrf_sample: "header row containing all 8 node names", "Exactly 10,000 rows" -> kept; numeric/finite kept.
- "Generate 10,000 samples from the learned MRF" -> not copied/resampled from input (<=100 of rows equal to input rows at 2 dp) (added, shortcut).
- "Generated samples should follow the learned distribution" -> Markov property of the learned graph in samples:
  non-edge |pcorr| < 0.07 (true 0, SE ~.01; unconstrained-MLE sampling gives <=~.05) and mean |pcorr| over learned
  edges >= 0.03 (glasso alpha .10, the weakest 8-edge reading, gives ~.06); columns non-degenerate (sd > 0.1).
  Thresholds chosen to hold under every plausible fitting reading.
## Checks -> requirements; removed
- "graph connected" and "V degree >= 3": not stated by the task; subsumed by edge-set check, removed as separate checks.
- test.sh no longer uses apt/curl/uv (network); runs python3 check.py directly with the image's numpy/pandas.

## Near misses (all derived from C; all rejected, the old verifier accepted 1,2,3,4)
nm_wrong_edge (W-Y -> C-Y), nm_bootstrap_samples (shortcut), nm_independent_samples, nm_nonedge_coupling
(spurious A-Y coupling in sampler), nm_unnormalised_marginal (densities, not probabilities),
nm_9999_samples, nm_duplicate_edge (V,C and C,V).
