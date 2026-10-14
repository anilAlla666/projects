# Verifier validation

Docker is not available in this authoring sandbox, so the verifier's core
logic (`/verifier_out/tests/check.py`) was validated by running it directly
against the real `/app/data` CIFs (the same files the built image will
contain, per the task's environment description that this sandbox already
*is* the solver's starting environment) with `/app/solution/output.json`
swapped for each candidate solution. `test.sh` is a one-line wrapper around
`check.py` plus the reward-file contract, so this is equivalent to running
it inside the container.

## Candidate solutions

- `good/output.json`: a plausible, well-formed, internally consistent
  answer for all 7 HOF CIFs (topology strings, interpenetration >= 1,
  distances within the geometrically-derived plausibility bound and
  rounded to 2 d.p., coordination sequences with c1 >= 3).
- `flawed_units/output.json`: identical to `good`, but every distance is
  multiplied by 10 (i.e. reported in Angstrom instead of nanometres) —
  breaks R7 ("in nanometers").
- `flawed_node/output.json`: identical to `good`, but HOF-1's coordination
  sequence has `c1 = 2` — breaks R10/R5 (a node must be formed by three or
  more hydrogen-bonded framework molecules, so c1 must be >= 3).
- `flawed_missingkey/output.json`: identical to `good`, but the `HOF-7`
  entry is deleted — breaks R9/R13 (must contain an entry for every input
  CIF).
- `flawed_rounding/output.json`: identical to `good`, but HOF-2's distance
  is `1.1037` instead of `1.10` — breaks R9 (must be rounded to exactly
  two decimal places).

## Results

| Candidate | Expected | Actual reward | Failing checks reported |
|---|---|---|---|
| good | 1 | **1** | (none — "All checks passed.") |
| flawed_units | 0 | **0** | R7 for all 7 keys (distance implausible / wrong unit) |
| flawed_node | 0 | **0** | R10 for HOF-1 (c1 = 2 < 3) |
| flawed_missingkey | 0 | **0** | R9 (missing entry for HOF-7) |
| flawed_rounding | 0 | **0** | R9 for HOF-2 (not rounded to 2 d.p.) |

Raw run log:

```
=== good ===
All checks passed.
reward: 1

=== flawed_units ===
FAILED CHECKS:
 - [R7] HOF-1: element[2] = 8.5 nm is not plausible ... (expected roughly between 0.20 and 3.33 nm ...)
 - [R7] HOF-2: element[2] = 11.0 nm is not plausible ... (expected roughly between 0.20 and 3.15 nm ...)
 - [R7] HOF-3: element[2] = 23.0 nm is not plausible ... (expected roughly between 0.20 and 8.48 nm ...)
 - [R7] HOF-4: element[2] = 12.5 nm is not plausible ... (expected roughly between 0.20 and 3.65 nm ...)
 - [R7] HOF-5: element[2] = 10.5 nm is not plausible ... (expected roughly between 0.20 and 3.53 nm ...)
 - [R7] HOF-6: element[2] = 9.0 nm is not plausible ... (expected roughly between 0.20 and 2.28 nm ...)
 - [R7] HOF-7: element[2] = 16.0 nm is not plausible ... (expected roughly between 0.20 and 4.63 nm ...)
reward: 0

=== flawed_node ===
FAILED CHECKS:
 - [R10] HOF-1: coordination sequence c1 = 2 but every node must be formed by three or more
   hydrogen-bonded framework molecules (degree >= 3), so c1 must be >= 3
reward: 0

=== flawed_missingkey ===
FAILED CHECKS:
 - [R9] missing entries for CIF(s): ['HOF-7']
reward: 0

=== flawed_rounding ===
FAILED CHECKS:
 - [R9] HOF-2: element[2] = 1.1037 is not rounded to exactly 2 decimal places
reward: 0
```

## Known scope limitation (see requirements.md notes)

The verifier cannot re-derive the scientifically "true" RCSR topology
symbol, exact interpenetration degree, or exact coordination sequence for
these seven real crystal structures — doing so would require reimplementing
full hydrogen-bond perception, network construction, and RCSR net
identification (normally done with specialized crystallographic software)
inside a deterministic, offline, sub-10-minute grading script. Every
mechanically-checkable requirement from the task text is enforced
(output existence/format, key coverage, per-element types and ordering,
rounding, units-plausibility derived from each CIF's own cell dimensions,
and the node-degree->=3 / coordination-sequence-positivity constraints that
follow directly from the task's own definitions). A scientifically
incorrect but internally-consistent, well-formed, plausible-magnitude
answer could pass; this tradeoff is documented rather than papered over
with guessed thresholds.
