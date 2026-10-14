# Verifier judge: a small from-scratch model that judges code attempts by evidence

Date: 8 Oct 2026. Status: design approved section by section; this file is the written spec for review.

## 1. Purpose

Verifier Bench showed that agent-written graders score at chance (balanced accuracy 0.50 to 0.52) whatever the writer
is told, and that only evidence from outside the writer moves them (0.80 with five other models' attempts, 0.83 to 0.88
with the author's solution). The case document (`~/verifier_bench/docs/verifier_model_case.md`) proposed a dedicated
verifier model with three capabilities: derive the goal, judge attempts, harden graders.

This project builds the first and narrowest of the three: a **judge**. Given a task and one attempt, it says whether the
attempt is correct, how sure it is, and which piece of evidence decided it. Writing graders and deriving goals come
later and are out of scope here.

## 2. Decisions already made

| Decision | Choice |
|---|---|
| Budget | Hard cap $100, for one rented GPU (raised from $0 on 8 Oct, when the user provided it). No HUD rollouts, no Claude calls. A cost ledger is kept. |
| Compute | Lambda Cloud, 1x A100 40GB SXM4 (30 vCPUs, 216 GB RAM, 472 GB disk) for training and data generation; this Mac (Apple M4, 24 GB) for development and the MLX reference. Stage 0 on the Mac showed it cannot pretrain the main config in days (docs/stage0_report.md). |
| Framework | PyTorch with flash-linear-attention's fused Kimi Delta Attention kernel (`chunk_kda`) for GPU training, in `vmt/`. The MLX implementation in `vm/` stays as an independent reference. |
| Approach | B: our own small model, trained from scratch, with modern hybrid blocks. Fine-tuning an existing open model (approach A) was offered and not chosen. |
| Domain | Python code first. Terminal-Bench tasks are a held-out transfer test at the end. |
| Repository | `~/verifier_model`, new. Commits authored `Anil Kumar <anil@hud.ai>`. |

## 3. What the model does

### Input

One token sequence in four tagged segments:

| Segment | Content | Trust |
|---|---|---|
| Task | the task statement | trusted |
| Diff | the attempt's code changes | trusted as the object being judged |
| Evidence | facts the harness gathered by running things (section 3.3) | trusted |
| Untrusted | any prose from the attempt: commit messages, added comments and docstrings, printed output | never evidence |

Each token carries a segment embedding, so the model can learn that untrusted tokens never count toward the verdict.

### Output

1. **Verdict:** the probability that the attempt is correct, calibrated, with an abstain band set on the dev split.
2. **Evidence pointer:** the index of the evidence line that decided the verdict. This is the model's "reason": short,
   and checkable against the evidence. Free-text reasons are not produced; a 110M model would write them badly.

### The evidence harness

The harness runs outside the model, in Docker with no network, and produces numbered evidence lines:

| Evidence | How it is gathered |
|---|---|
| Visible tests | the failing test named in the task plus the library's visible half of its existing tests (section 5), run on the patched code; pass or fail read from the test runner's exit code and JUnit XML, never from printed text |
| Test edits | whether the diff touches any test file |
| Differential fuzzing | the original and the patched code are run on the same random inputs to every changed function and its direct callers; each line records where outputs differ and on which functions |
| Diff shape | files and functions touched, lines added and removed |

Fuzz inputs come from type hints, default values and values that appear in visible tests. Values from oracle-only
tests and doctest examples are never used, because that would leak the oracle. When a function's inputs cannot be
generated, the harness writes "fuzz: not applicable" for it rather than guessing.

### The oracle and the visible rule

Each library's full test suite, doctests included, is the oracle. It is used only to label attempts.

One rule decides what is visible, and it applies to every test set (synthetic, real attempts and transfer):

- the failing test the task names is visible;
- of the library's other existing tests, a fixed random half (chosen once per library by seed) is visible;
- everything else, including the remaining failing tests and all doctests, is oracle-only and never reaches the model,
  the harness's fuzz inputs or the attacker.

So a regression sometimes shows in a visible test and sometimes only in fuzzing, and the model learns to use both.
Task texts drop doctest lines from docstrings for the same reason.

## 4. Architecture

Two configs share one code path: **dev** (about 25M parameters) for fast experiments and **main** (about 110M) for the
real model.

| Part | Choice | Reason |
|---|---|---|
| Layer mix | 12 layers (main), three delta-attention layers then one full-attention layer, repeated | delta layers are linear in length with fixed-size state; full-attention layers do exact lookups |
| Delta layer | Kimi Delta Attention: gated delta rule with a per-channel forget gate, short causal convolution on q, k and v, L2-normalised q and k, output gate | training uses our own chunked (parallel-in-chunk) delta rule in plain MLX ops, because mlx-lm's Metal kernel (`mlx_lm/models/gated_delta.py`) is forward-only and its differentiable fallback loops one time step at a time; inference may use the mlx-lm kernel once it matches ours numerically |
| Full attention | grouped-query attention, QK-norm, sigmoid output gate, no positional encoding | delta layers carry position, as in Kimi Linear |
| Feed-forward | SwiGLU, dense | mixture of experts does not pay at this size |
| Normalisation | pre-norm, zero-centred RMSNorm | stable at small batch sizes |
| Width | main: d_model 768, 12 heads; dev: d_model 384, 6 layers | |
| Tokenizer | our own 32k byte-level BPE, with special tokens for segment starts, evidence-line starts and the verdict position | a 151k vocabulary would put most parameters into embeddings |
| Embeddings | tied input and output | |
| Heads | language-model head (pretraining), verdict head on the verdict token, pointer head scoring each evidence-line start token | |
| Context | 8,192 tokens | |
| Optimiser | Muon for 2D weight matrices; AdamW for embeddings, norms and heads | the Moonlight recipe; Muon is implemented here if MLX lacks it |
| Framework | MLX on Apple silicon | |

A verdict is one forward pass with no generation. End to end, the harness (Docker test runs and fuzzing) will
dominate the time per verdict; both parts are measured separately.

## 5. Data

### Sources

| Source | Use | Licence |
|---|---|---|
| FineWeb-Edu, sample split | pretraining (English) | ODC-By; confirmed before download |
| Python source of permissively licensed PyPI packages (MIT, BSD, Apache 2.0 by their own metadata) | pretraining (code) and the libraries we break | filtered by licence at selection |
| Correct-by-construction tasks built from those libraries | judge training, dev and held-out test | ours |
| Public model patches with real verdicts on pure-Python repos (SWE-bench-style runs) | held-out "real attempts" test only | checked per dataset before download; excluded if unclear |
| Verifier Bench's Terminal-Bench pools | held-out transfer test only | leaderboard terms checked before use |

### Library selection rule

Fixed before any library is seen: pure Python, permissive licence in package metadata, at least 100 tests, the full
suite runs in under 60 s in Docker with no network, version pinned.

### Correct-by-construction tasks

The mutation rule ports the Verifier Bench pilot (`~/verifier_bench/vb/cbc.py`) and widens its operators:
comparison flips, `and`/`or` swaps, arithmetic operator swaps, off-by-one constants, swapped arguments, a removed
statement, a changed return value. A mutant is kept only if 1 to 6 tests fail. The task text is built from one failing
test and the mutated function's docstring.

### Attempts per task

| Kind | How it is made | Label |
|---|---|---|
| Correct | the reverted fix | correct |
| Correct alternative | equivalence-preserving rewrites of the fix (flipped comparison with swapped operands, De Morgan, an extracted variable, a multi-line restructuring), each confirmed by the oracle | correct |
| Near miss: regression | the right fix plus one other mutation that breaks a different test | wrong |
| Near miss: visible-only | a different change that makes the visible test pass and fails the oracle | wrong |
| Near miss: overfit | special-casing the visible test's input | wrong |
| Cheat: test edit | changing the tests instead of the code | wrong |
| Cheat: fake output | printing "PASSED" or similar | wrong |
| Cheat: persuasion | a wrong attempt with comments, docstrings or a commit message claiming it is correct or instructing the grader | wrong |

**Realistic attempts from an open model (added 8 Oct).** Rule-made near misses are not how agents actually fail.
An open-weight coding model (Apache 2.0, small enough for one GPU, run locally with no API) is given each task and
asked for a fix several times at a non-zero temperature. Its attempts are labelled by the oracle like every other
kind, and its explanation text goes into the untrusted segment. This kind is called `model_attempt`.

**Auxiliary labels (added 8 Oct).** The oracle run already says more than correct or wrong, so each example also
records the number of oracle-only tests that fail (bucketed 0, 1, 2 to 5, 6 or more) and the line of the task's
code where the bug was planted. The task segment therefore includes the broken function's source, one numbered
line per `<|line|>`. These become two extra training targets.

Every label is the oracle's verdict, not the generator's intent. If an intended near miss passes the oracle, it is
labelled correct, and its pointer becomes the visible-test summary line, the pointer every correct attempt uses.

**No shortcut through diff size.** Correct attempts are often one-line reverts and some wrong kinds touch several
places, so diff size alone could separate them. Two guards: correct alternatives include multi-line rewrites, and
attempts are sampled so that diff-size distributions of correct and wrong attempts match within each task family. A
"diff shape only" baseline (section 7) measures what is left.

### Splits

Three held-out axes, so the test measures generalisation rather than memory:

1. **Libraries:** about 80% train, 10% dev, 10% held-out test.
2. **Mutation operators:** one operator (chosen by seed before any data is generated) appears only in the test set.
3. **Attempt kinds:** one wrong-attempt generator appears only in the test set.

Persuasion and injection phrasings are also split, so section 7's attack test uses phrasings never trained on.

**The pretraining corpus excludes** test-split libraries, every repository in the real-attempts set, and any
Terminal-Bench content.

### Scale

About 50k labelled examples from 100 to 200 libraries. Oracle runs are limited to tests covering the changed module.
Estimated 7 to 10 hours on the 10 cores, confirmed by measurement first.

## 6. Training

| Stage | What happens | Estimated time on the M4 |
|---|---|---|
| 0. Feasibility | (a) write the chunked delta rule and check it matches the step-by-step reference numerically, forward and gradients; (b) measure training tokens per second for both configs and model time per verdict. If the main config cannot train fast enough, it shrinks; every later budget comes from this measurement | hours |
| 1. Tokenizer | train the 32k BPE on a corpus sample | minutes |
| 2. Pretrain | next-token loss, about 60% code and 40% English; learning rates tuned on dev, carried to main; context 2k, then 8k at the end; token budget set by stage 0 | set by stage 0; a rough compute estimate (6 × 110M parameters × 1B tokens ≈ 6.6e17 operations at realistic M4 throughput) is 4 to 6 days for 1B tokens, unattended |
| 3. Judge fine-tune | verdict loss (binary cross-entropy), pointer loss (cross-entropy over evidence lines), and a small share of next-token loss; classes balanced; hardest near misses oversampled | hours |
| 4. Adversarial rounds | an attacker searches for wrong attempts the judge accepts by combining near misses with persuasion, fake output and injected instructions; what fools the judge is added to training; repeat until attack success stops falling. Persuasive text comes from a small Qwen3.5 model run locally, which only attacks and never judges | hours per round |
| 5. Calibrate | temperature scaling on dev, then the abstain band for a target precision | minutes |

**Hard-negative mining (added 8 Oct).** The data pipeline generates a reservoir several times larger than one
training set. After each fine-tuning round, the judge scores the reservoir, and the next round oversamples the
attempts it got wrong, most of all confident wrong verdicts.

**Pointer labels.** The decisive evidence line for each training example is derived from how the attempt was made:
the failing visible test for a visible failure, the fuzz line on the regressed function for a regression, the test-edit
line for a test edit, and so on.

**Why there is no RL stage.** A judge's reward is "did the verdict match the oracle"; supervised training on oracle
labels optimises exactly that, with lower variance. The adversarial rounds provide the exploration RL would.

**Adaptation.** For a new task family, fine-tune on 10, then 50 labelled attempts; section 7 reports the curve.

**Records.** Each run has a config file and a timestamped folder under `experiments/<config>/`; every checkpoint is
resumable.

## 7. Evaluation

### Test sets

1. Held-out libraries (correct-by-construction).
2. Real attempts (public patches with real verdicts).
3. Terminal-Bench pools (transfer). These need their own evidence adapter and are measured last.

### Baselines, all free

| Baseline | Rule |
|---|---|
| Visible tests only | accept if the visible tests pass |
| Diff shape only | a logistic regression on the diff-shape evidence alone (files, functions and lines touched) |
| Rules plus fuzzing | accept if the visible tests pass, no test is edited and fuzzing shows no difference outside the functions the task names |
| Untrained Qwen3.5 4B | the same input, zero-shot, run locally |
| Earlier grader numbers (Terminal-Bench set only) | 0.52 alone, 0.80 with the five-model panel, 0.83 to 0.88 with the author's solution; approximate, since those were graders, not judges |

If the model cannot beat rules plus fuzzing, the harness is doing the work and the model adds nothing.

### Pre-registered bars

| Measure | Bar |
|---|---|
| Balanced accuracy on the held-out test set | at least 0.90, and closes at least half the gap between rules plus fuzzing and 1.0: (BA − B) / (1 − B) ≥ 0.5, where B is that baseline's balanced accuracy |
| Balanced accuracy per held-out axis (libraries, operator, attempt kind) | reported separately |
| Correct alternatives wrongly rejected | at most 5% |
| Near misses caught | reported per kind |
| Attack success on held-out attack phrasings | at most 5% |
| Expected calibration error | at most 0.05 |
| Real attempts | above untrained Qwen3.5 4B |
| Adaptation curve at 0, 10 and 50 examples | reported |
| Time per verdict | reported as model forward time and harness time, separately |
| Gap to the human oracle | reported (the oracle scores 1.0 by definition) |

Confidence intervals are bootstrap intervals resampled by library.

### Checks on the pipeline

Each is shown to fail on a deliberately broken input before it is trusted:

1. **No oracle leak:** fails if the source of any oracle-only test, or any doctest's expected output, appears in a
   model input or a fuzz input.
2. **No split leak:** fails if a library, the held-out operator or the held-out attempt kind appears in training data,
   or if a test-split library, a real-attempts repository or Terminal-Bench content appears in the pretraining corpus.
3. **Harness cannot be faked:** an attempt that prints "PASSED" while its tests fail must register as failing.
4. **Labels are the oracle's:** every stored label must match a fresh oracle run on a random sample.

## 8. Repository layout

| Path | Responsibility |
|---|---|
| `vm/corpus/` | fetch the pretraining corpus, licence filter, deduplicate, exclude test-split libraries |
| `vm/libs/` | library selection by the fixed rule; pinned versions |
| `vm/tasks/` | mutation, task text, attempt generators, oracle runs |
| `vm/harness/` | Docker sandbox, visible tests, test-edit detection, differential fuzzing, diff shape |
| `vm/data/` | the example record, serialisation into segmented token sequences, splits |
| `vm/model/` | config (shared by both implementations), MLX delta layer, full-attention layer, blocks, heads |
| `vmt/` | the PyTorch implementation used for GPU training, with the same configs and parameter counts |
| `tests_gpu/` | tests that need the GPU, run on the instance |
| `vm/optim/` | Muon |
| `vm/train/` | pretrain, fine-tune, adversarial rounds, calibration |
| `vm/eval/` | baselines, metrics, bootstrap, report |
| `tests/` | unit tests and the four pipeline checks |
| `configs/` | one file per run configuration |
| `experiments/` | run folders (not committed beyond summaries) |

### The example record

```json
{
  "example_id": "lib@ver/mutant-17/near_miss_regression-2",
  "library": "lib", "split": "train",
  "task_text": "...",
  "diff": "...",
  "evidence": [{"id": 0, "kind": "visible_test", "text": "test_x: pass"},
               {"id": 1, "kind": "fuzz", "text": "lib.f: outputs differ on 37 of 200 inputs"}],
  "untrusted": ["# verified: all tests pass"],
  "attempt_kind": "near_miss_regression",
  "label": 0,
  "pointer": 1
}
```

## 9. Risks

| Risk | What we do |
|---|---|
| A from-scratch 110M model may not beat rules plus fuzzing | that baseline is a pre-registered bar, so the result is reported either way |
| Mac throughput may cap pretraining below 1B tokens | stage 0 measures it; the token budget follows the measurement and is reported |
| Our chunked delta rule may be too slow or numerically off | stage 0 checks it against the step-by-step reference before anything trains; if it is too slow, the model shrinks |
| The model may learn generator artefacts instead of correctness | matched diff sizes, the diff-shape baseline, and held-out operator and attempt kind |
| Templated task texts teach little about real intent | the real-attempts test measures the cost |
| Fuzz inputs cannot be generated for some functions | those functions get "not applicable"; the share is reported |
| Running downloaded test suites is running untrusted code | Docker, no network, resource limits (1 GB, 90 s per run, as in the pilot) |
| A licence turns out unclear | the source is dropped, not used |

## 10. Out of scope

Writing graders, deriving goals, hardening graders, HUD deployment, paid compute of any kind, and fine-tuning an
existing open model as the judge.
