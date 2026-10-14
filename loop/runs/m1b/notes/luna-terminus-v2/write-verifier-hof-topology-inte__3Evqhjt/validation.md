# Verifier validation

## Structural regression results

The checker was run with fixture files copied to `/app/solution/output.json`, the grader-visible path:

| Fixture | Intentional violation | Reward |
|---|---|---:|
| `/scratch/schema_only/output.json` | None of the implemented structural checks; this is only a schema fixture, **not a correct scientific solution** | 1 |
| `/scratch/flawed/missing-key.json` | R15: missing `HOF-7` key | 0 |
| `/scratch/flawed/short-record.json` | R16: extra fifth array item | 0 |
| `/scratch/flawed/unrounded-distance.json` | R4: distance `1.071` is not rounded to two decimals | 0 |
| valid JSON `null` root | R15: root is not an object | 0 |

Each tested format variant was rejected with a requirement-tagged failure. The Dockerfile uses the exact prescribed base-image line, includes `COPY . /tests/`, and has no `ENTRYPOINT`. The checker uses only the Python standard library; `test.sh` writes the reward file and displays failures.

## Scientific-validation limitation

This validation is **incomplete** relative to the required verifier-author workflow. The CIFs are present, but Gemmi, NumPy, SciPy, and NetworkX were unavailable, and attempts to download packages failed. My custom CIF geometry attempts produced inconsistent molecule assignments and did not establish trustworthy periodic nets or reference values. I therefore did not derive a scientifically correct output, and the schema-only fixture must not be mistaken for one.

The checker does not verify the actual scientific answers for topology, degree of interpenetration, average internodal distance, or coordination sequence (R1–R3, R6–R14, R17). In particular, this does not meet the required proof against a correct solution or scientific flawed variants (R24–R27). No scientific values have been guessed or presented as ground truth.
