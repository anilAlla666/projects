# Judging data: what was built, what was dropped, and why

Measured on the 4x H100 instance on 8 Oct 2026. Every label below comes from running a library's own test suite
(the oracle) in a sandboxed container with no network; none is inferred.

## Libraries

1,728 PyPI candidates were screened in the order fixed by the split hash. 168 qualified (pure Python, permissive
licence, at least 100 tests, full suite under 60 s with no network): **129 train, 20 dev, 19 test**, against a
target of 160 / 20 / 20. Train fell short because qualification was stopped at 168 to free the CPUs for task
creation. Why the others did not qualify (top reasons): no source archive, compiled code or failed install (625),
no tests found (490), suite too slow (75), fewer than 100 tests (the rest).

Deviations from the plan, recorded rather than smoothed over:
- When a source archive had no tests, the library's GitHub repository was cloned at the release tag.
- Common test plugins were installed with every library (pytest-timeout, hypothesis, pytest-asyncio, pytest-mock,
  mock, freezegun), and missing test dependencies were installed when collection named them.
- A test that fails on the unmodified library was deselected if at most 5% of the suite did so; otherwise the
  library was rejected.
- Bug search ran at most 40 candidate mutations per library and stopped a run after 7 failures.
- Two libraries produced no tasks: `purl` (its source does not parse under Python 3.12) and `python-snap7`
  (task creation was stopped after it ran alone for half an hour).

## Tasks and attempts

1,615 bug tasks: one mutation of one function that makes 1 to 6 of the library's tests fail. The judge sees the
named failing test and a fixed half of the other tests (chosen once per library); every other test, all doctests
included, is oracle-only.

Attempts per task: the revert, rewrites that are equivalent by construction, other single-site changes that make
the named test pass, an edit to the test file, faked test output, a persuasive comment on the unfixed code and the
same comment on the real fix, a regression in another function (test split only: the held-out wrong kind), and up
to 4 fixes written by Qwen2.5-Coder-32B-Instruct (Apache 2.0) under vLLM: **5,678 model attempts** over the tasks,
177 tasks with no parseable reply. A hard-negative pass then added the real fix plus a second single-site change in
the same function, and more named-test-passing alternatives, kept when they pass every visible test.

## What the shared loader drops (one definition, used by every reader)

| Rule | Dropped |
|---|---:|
| Task whose original code fails its own oracle, or whose unfixed code with faked output passes it (flaky oracle) | 158 tasks |
| Persuasion pair whose prose-on-the-fix attempt is not correct (the inserted line had broken an indented method; fixed in the generator afterwards) | 972 pairs |
| Example whose input contains an oracle-only test's source or a doctest's expected output (mostly doctest lines carried in a model rewrite's diff context) | 292 examples |
| Example longer than 8,192 tokens even after cutting the diff | 6 examples |

## Result

| Split | Examples | Libraries | Correct | Wrong that pass every visible test |
|---|---:|---:|---:|---:|
| Train | 11,671 | 110 | 6,151 | 543 |
| Dev | 1,824 | 17 | 917 | 92 |
| Test | 2,281 | 19 | 1,053 | (not looked at before the final evaluation) |

Persuasive prose is now attached to correct and wrong attempts equally often (dev: 43 and 42), so prose alone
carries no verdict. Attack variants (prose written by the 32B model) are also balanced by label: train 3,866,
dev 1,472, test 1,835, with test phrasings no training example uses.

## Pipeline checks (spec section 7)

| Check | Result |
|---|---|
| Libraries cross no split; held-out operator and held-out wrong kind appear only in test | 0 violations |
| Faked test output never labelled correct | 8 violations, all on flaky-oracle tasks, which the loader now drops |
| Labels re-derived by a fresh oracle run on a seeded sample of 50 | 0 mismatches |
| No oracle-only test source or doctest output in any model input | 3,362 hits in 292 examples, all dropped by the loader |

## Free baselines on dev (before any training)

Measured on 1,801 dev examples before the leak filter: visible tests only 0.955 balanced accuracy, rules plus fuzzing 0.953, diff shape alone 0.63. On the hard subset
(every visible test passes, measured after the filter), all three are at chance: 0.50, 0.503 and 0.48.
