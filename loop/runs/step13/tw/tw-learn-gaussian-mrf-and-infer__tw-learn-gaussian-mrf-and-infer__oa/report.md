# Grader report: tw-learn-gaussian-mrf-and-infer

Rounds: 1 · cost $0.88 · candidates judged correct: B, C · judged wrong: A, D

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

- **contested:** conditional mean/sd of P(W|V=1,X=0.5): the verifier expects no fixed value (original had none); only self-consistency with samples, the candidates give B .2015/.722, C .198/.723, D .2015/.722, A 0/.63; depends on the estimator / glasso penalty, which the task does not fix (alpha .1 still recovers the 8 edges but gives mean .05)

## Every check

| check | requirement | basis |
|---|---|---|
| three output files exist | Three output files must be saved | format |
| learned_mrf.csv: exactly 8 rows of 2 valid case-sensitive node names, no header, no duplicate/self edges | Exactly 8 edges ... each edge should appear only once ... use capital letters | format |
| V has strictly highest degree | node V must have the highest degree | task-text |
| edge set equals top-8 |partial correlation| pairs of the input data {A-V,B-V,C-V,V-W,V-X,A-B,D-X,W-Y} | Recover the underlying graphical structure of the MRF. The true graph has exactly 8 edges | recomputed |
| inferred_marginals.csv header w_value,probability, numeric, >=2 rows, probs >=0, sum in [0.95,1.05] | Format: CSV with header w_value,probability ... sum to approximately 1.0 | format |
| marginal mean/sd agree (0.06 / 20%) with Gaussian conditioning on the solver's own samples at V=1,X=0.5; mean exceeds W mean by >0.03 | The conditional inference should properly account for the evidence V=1.0 and X=0.5; Generated samples should follow the learned distribution | format |
| final_mrf_sample.csv: 10000 rows, all 8 columns, finite | Exactly 10,000 rows ... header row containing all 8 node names | format |
| samples are not copies/resamples of input (<1% duplicate rows, <1% rows equal to input rows) | Generate 10,000 samples from the learned MRF model | format |
| samples: non-edge partial correlations |pc|<0.06, std in (0.4,1.5), |mean|<0.2 | Generated samples should follow the learned distribution | format |
