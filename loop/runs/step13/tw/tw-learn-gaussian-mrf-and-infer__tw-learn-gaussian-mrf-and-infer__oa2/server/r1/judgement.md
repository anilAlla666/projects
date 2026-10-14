# Judgement

## Evidence summary
The original verifier accepted A, B, C, D and rejected the empty attempt. It only checked format
(8 edges, V max degree, sums to ~1, 10000 rows) and never checked the structure against the data,
the conditional distribution, or that samples came from a model. Candidate E has no results at all
(/candidates/E absent) — treated like the empty attempt.

## Candidates
- **A — wrong.** (accepted by old verifier; now rejected by test_structure_matches_data,
  test_samples_not_copied_from_data, test_samples_follow_learned_graph, test_conditional_consistent_with_samples)
  * edges: star V-{A,B,C,D,W,X,Y} + A-B. The data's partial correlations: V-D = 0.010, V-Y = -0.014,
    while D-X = 0.202, W-Y = 0.208 are missing. Wrong structure.
  * final_mrf_sample.csv is the observed data with rows permuted (100% of rows are data rows) — not sampled.
  * inferred_marginals is a hard-coded 3-point {-1:.2, 0:.6, 1:.2}, mean 0: ignores the evidence.
- **B — correct.** Correct 8 edges, Gaussian conditional mean 0.2015 sd 0.722 (empirical covariance),
  fresh multivariate-normal samples (0 rows from data); sample-implied conditional 0.2305 (diff 0.029, ~2.4 SE).
- **C — correct.** Correct edges, GraphicalLassoCV fit, conditional mean 0.198 sd 0.723, samples from
  the fitted N(mu, Sigma), sample-implied conditional 0.1925.
- **D — wrong.** (accepted by old verifier; now rejected by test_samples_not_copied_from_data)
  Edges and conditional are fine, but final_mrf_sample.csv is a bootstrap of the observed data:
  every one of the 10000 rows is an exact row of /app/mrf_sample_10k.csv (6334 distinct rows).
  The task says "Generate 10,000 samples from the learned MRF model" / "Generated samples should
  follow the learned distribution"; resampling the observations does not use the learned MRF at all.
  A genuine continuous sample (even rounded to 2 dp) matching a full 8-dimensional data row is
  essentially impossible (B, C: 0 matches).
- **E / empty — wrong** (no files).

## Structure (value check, basis recomputed)
Task names "partial correlation thresholding" and graphical lasso and says the graph has exactly 8 edges.
Readings tried on the data:
- top-8 |partial correlation| (empirical precision): A-B .200, A-V .198, B-V .198, C-V .212, D-X .202,
  V-W .208, V-X .201, W-Y .208; next largest is .021 (gap ~10x).
- GraphicalLasso alpha = 0.02, 0.05, 0.10: each gives exactly the same 8 edges (alpha 0.15 collapses to 2 edges, violating "exactly 8").
- GraphicalLassoCV (candidate C) and candidates B, D: same 8 edges.
All readings that satisfy the task's constraints give the same set; V has degree 5 (unique max), consistent
with the task statement. The test recomputes the top-8 set at grading time from /app/mrf_sample_10k.csv and
only asserts if the 8th/9th gap is > 3x.

## Conditional P(W|V=1,X=0.5) — mean value is CONTESTED
Gaussian conditional mean depends on the parameter estimator, which the task does not fix:
empirical covariance 0.2015 (sd 0.7222); glasso alpha .005 0.197, .01 0.190, .02 0.175, .05 0.128, .10 0.051.
All are valid "graphical lasso" fits that recover the 8-edge graph, and held-out predictive likelihood cannot
pick a regularisation level as "the" answer the task asks for. The original verifier had no value check
for it, so none is added. Instead a consistency check is used (no expected value): the conditional in
inferred_marginals.csv and the samples must come from the same learned Gaussian model, so the
conditional mean/sd of the file must match the Gaussian conditional E[W|V=1,X=.5] and residual sd implied by
OLS of W on V,X in the solver's own samples (|Δmean| < 0.06 ≈ 5 SE for n=10000; sd ratio within 20%).
This holds for any estimator. B: 0.2015 vs 0.2305; C: 0.198 vs 0.1925.

## Trace: requirements -> checks
| Requirement (task text) | Check |
|---|---|
| "Three output files must be saved" | test_output_files_exist |
| "Exactly 8 edges", "each edge should appear only once", case-sensitive names, "Two columns (no header)" | test_edge_format_and_count |
| "node V must have the highest degree" | test_v_has_strictly_highest_degree |
| "Recover the underlying graphical structure" | test_structure_matches_data |
| "CSV with header w_value,probability", "Discretized probability distribution", "sum to approximately 1.0" | test_marginals_valid_distribution |
| "Exactly 10,000 rows", "header row containing all 8 node names" | test_samples_format |
| "Generate 10,000 samples from the learned MRF model" | test_samples_not_copied_from_data, test_samples_follow_learned_graph |
| "Generated samples should follow the learned distribution" | test_samples_follow_learned_graph (non-edges |pcorr| < 0.08), test_conditional_consistent_with_samples |
| "conditional inference should properly account for the evidence V=1.0 and X=0.5" | test_conditional_consistent_with_samples |
| "Fit the MRF parameters" | indirectly: test_conditional_consistent_with_samples / samples checks (parameter values contested) |
| "complete within 360 s" | not checkable from outputs (solve_mrf.py not run) |

## Trace: old checks -> requirement
- test_output_files_exist, edge_count, edge_nodes_valid, no_duplicate_edges, inferred_marginals_format,
  marginals_probabilities_valid, samples_count, samples_columns, samples_no_nan_inf: kept (merged).
- test_v_has_highest_degree: kept, made strict (ties not "highest"); old one depended on dict order.
- test_v_degree_minimum (>=3) and test_graph_connectivity: not required by the task; removed (subsumed by
  the exact structure check anyway).
- test.sh: reward now based on pytest exit code captured right after pytest; mkdir /logs/verifier.

## Near misses (all copies of C with one defect)
nm1_wrong_edge (W-Y -> V-Y), nm2_evidence_swapped (conditional for V=.5,X=1), nm3_bootstrap_samples,
nm4_ignores_evidence (unconditional W), nm5_independent_samples, nm6_9999_samples, nm7_samples_wrong_graph.
All rejected; B and C accepted; A, D, empty rejected.
