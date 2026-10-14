# Grader report: tw-learn-gaussian-mrf-and-infer

Rounds: 1 · cost $0.86 · candidates judged correct: B, C · judged wrong: A, D, E

| basis | checks |
|---|---:|
| task-text | 1 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 7 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** conditional mean/sd of P(W|V=1,X=0.5): the verifier expects no fixed value (only consistency with own samples), the candidates give 0.2015 (B, D), 0.198 (C), 0.0 (A); depends on parameter estimator / glasso regularisation (0.05-0.20 across alpha 0.1-0), which the task does not fix

## Every check

| check | requirement | basis |
|---|---|---|
| all three output files exist | Three output files must be saved | format |
| learned_mrf.csv: 8 lines of node1,node2, valid capital names, no self-loops, no duplicates | Exactly 8 edges ... each edge should appear only once; node names are case-sensitive | format |
| V degree strictly greater than any other node | node V must have the highest degree | task-text |
| edge set equals top-8 |partial correlation| pairs of /app/mrf_sample_10k.csv (asserted only if 8th/9th gap >3x) | Recover the underlying graphical structure of the MRF. The true graph has exactly 8 edges | recomputed |
| inferred_marginals header w_value,probability, numeric, >=2 unique values, p>=0, sum in [0.95,1.05] | CSV with header w_value,probability ... probabilities should sum to approximately 1.0 | format |
| samples: 10000 rows, all 8 columns numeric finite | Exactly 10,000 rows ... header row containing all 8 node names | format |
| <1% of sample rows are exact rows of the observed data | Generate 10,000 samples from the learned MRF model | format |
| in samples, |partial corr| < 0.08 for every pair not in learned_mrf.csv | Generated samples should follow the learned distribution | format |
| mean/sd of inferred_marginals within 0.06 / 20% of Gaussian conditional W|V=1,X=0.5 implied by the solver's own samples | The conditional inference should properly account for the evidence V=1.0 and X=0.5; Generated samples should follow the learned distribution | format |
