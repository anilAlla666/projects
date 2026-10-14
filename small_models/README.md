# Small models: the 116M attempt classifier and the 7B grader writer

## `vm/`: the 116M attempt classifier, trained from scratch

A Kimi Delta Attention hybrid (gated delta-rule linear attention with periodic full attention), 116.5M parameters,
that reads an attempt plus its test evidence and returns a verdict.

- **Kernels:** a chunked gated delta-rule implementation in MLX matching the reference to 1e-7 including gradients,
  then PyTorch with flash-linear-attention kernels.
- **Pretraining:** 2.33B tokens of permissive Python and English, data-parallel on 4 H100 at about 531k tokens/s.
- **Fine-tuning as a judge:** on attempts with test evidence; robustness training against faked test output and
  edited tests.
- **Result on a held-out development set:** balanced accuracy 0.96 to 0.97; faked test output or edited tests
  accepted 0% of the time. With no tests to read, AUC 0.53: correctness needs evidence, which led to the verify loop.
- **Weights:** [`../weights/classifier_116m/`](../weights/classifier_116m/).

## `gw/`: the 7B grader writer

Qwen2.5-Coder-7B with LoRA, trained to write a grader (a standalone test file) for a task.

| Round | Training | Usable graders | Balanced accuracy |
|---|---|---|---|
| 1 | 701 maintainers' test files, issue text only | 34 / 80 | 0.51 |
| 2 | plus repository context | 20 / 80 | 0.52 |
| 3 | 3,706 complete standalone test files | 59 / 80 | 0.54 |
| 4 | reward-filtered round | 59 / 80 | 0.55 |
| | Claude Sonnet 5.5, for reference | 75 / 80 | 0.77 |

Training solved the format; the remaining failure is invented requirements, which a writer cannot check by itself.
That is why the project became a verifier next to any writer. **Weights:** [`../weights/grader_writer_7b/`](../weights/grader_writer_7b/).

## Layout

`vm/` model and training, `vmt/` tokenizer, `gw/` grader writer, `scripts/` data and training pipelines, `configs/`
model and run configurations, `experiments/` run outputs and metrics, `tests*/` unit, Docker and GPU tests,
`docs/` design notes. Install with `pip install -e .` (see `pyproject.toml` and `requirements.txt`).
