# Grader report: tw-learn-gaussian-mrf-and-infer

Rounds: 1 · cost $0.95 · candidates judged correct: B, C · judged wrong: A, D

| basis | checks |
|---|---:|
| task-text | 3 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 6 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** conditional distribution P(W|V=1,X=0.5) (mean/sd): the verifier expects no value check (original had none), the candidates give mean ~0.198-0.2015, sd ~0.722 (B,C,D); A hard-coded mean 0, sd 0.63; depends on the parameter-fitting estimator and graphical-lasso penalty, which the task does not fix (glasso alpha 0.01-0.10 all give 8 edges, mean 0.19-0.05); held-out comparison does not separate readings by 2x

## Every check

| check | requirement | basis |
|---|---|---|
| learned_mrf.csv has exactly 8 lines/edges | The true graph has exactly 8 edges / Requirements: Exactly 8 edges | task-text |
| V strictly highest degree | node V must have the highest degree | task-text |
| no duplicate undirected edges, no self loops | each edge should appear only once | format |
| node names in A,B,C,D,V,W,X,Y | All node names are case-sensitive | format |
| edge set == top-8 |partial correlation| pairs of the task data (C-V,W-Y,V-W,D-X,V-X,A-B,A-V,B-V) | Recover the underlying graphical structure... exactly 8 edges; partial correlation thresholding | recomputed |
| inferred_marginals header w_value,probability; numeric, finite, distinct w, >=2 rows | Format: CSV with header w_value,probability; Discretized probability distribution over W values | format |
| probabilities >=0, sum in [0.95,1.05] | The probabilities should sum to approximately 1.0 | format |
| samples: all 8 columns, exactly 10000 rows, numeric finite, sd>0.1 | Exactly 10,000 rows of sampled values; header row containing all 8 node names | format |
| samples not copied/bootstrapped from the input (<=100 rows equal to input rows) | Generate 10,000 samples from the learned MRF model | format |
| samples obey learned graph: non-edge |pcorr|<0.07, mean edge |pcorr|>=0.03 | Generated samples should follow the learned distribution | task-text |
