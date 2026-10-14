# Model weights

Stored with Git LFS (`git lfs install`, then clone or `git lfs pull`). Base models are not included: download them
from their publishers and apply our adapters. sha256 prefixes let you check every file.

## `classifier_116m/` (trained from scratch; code in `../small_models/vm`)

| File | MB | sha256 | What it is |
|---|---|---|---|
| `pretrain/weights.pt` | 445 | `b294cfb9e9f2b2c4` | the pretrained 116M model (2.33B tokens) |
| `judge/r3_epoch_2.pt` | 447 | `417fb6f00e250cbc` | judge fine-tune, round 3, epoch 2 (full precision) |
| `judge/judge_strict_r4e0_bf16.pt` | 224 | `fb80b094c905f966` | judge, round 4, strict evidence setting (bf16) |
| `judge/judge_realistic_real1e0_bf16.pt` | 224 | `7757e53818333168` | judge trained on realistic real-attempt evidence (bf16) |

## `grader_writer_7b/` (LoRA on Qwen2.5-Coder-7B; code in `../small_models/gw`)

| File | MB | sha256 | Round |
|---|---|---|---|
| `adapter/` | 617 | `ccbb92d636e59b16` | 1: 701 maintainers' test files |
| `adapter_run2/` | 617 | `84dda1409c35133e` | 2: plus repository context |
| `round3/adapter_run3_context/` | 617 | `b341788c8dc0df78` | 3: context variant |
| `round4/adapter_run4_standalone/` | 617 | `5fa231860fbb9d16` | 3,706 standalone test files |
| `round4/adapter_run5_reward_round/` | 617 | `bd80430de8418ff8` | reward-filtered round |

## `requirement_model_7b/` (LoRA on Qwen2.5-7B-Instruct; code in `../loop/vb/requirements.py`, `../loop/vb/reqdata.py`)

| File | MB | sha256 | What it is |
|---|---|---|---|
| `adapter_run1/` | 617 | `2055b15722e694d5` | lists the requirements a correct grader must check, with a near miss for each |

The 7B verdict model (`../loop/vb/verdictdata.py`): its code and evaluation are in `../loop`; its adapter was not
found on any machine we still have, so its weights are not included.

## `open_judge_27b/` (LoRA on Qwen 3.8 27B; code in `../loop/tools/train_judge.py`)

Keys renamed for vLLM serving (`model.language_model.layers.*`). Serve with
`vllm serve <qwen3.8-27b> --enable-lora --lora-modules judge=<adapter dir>`.

| Folder | MB | sha256 | Training study (see `../loop/docs/research_intent_goal.md`) |
|---|---|---|---|
| `adapter1_hf/` | 891 | `c84207eb8180635a` | step 13: imitation of frontier loop rounds |
| `adapter2_hf/` | 891 | `19c84c2df08f1cfb` | step 14: rejection-sampling fine-tuning on its own verified rounds |
| `adapter3_hf/` | 891 | `7235fac026595c32` | step 16: long-context fine-tuning with the full-output view |
| `adapter4_hf/` | 891 | `be25ce9693cba4e5` | step 17: targeted failure classes |
| `adapter6_hf/` | 891 | `eb678ea685b61e9a` | step 19: distillation from frontier rounds with summarised reasoning |
| `s18_ad1_hf/` | 891 | `6eb6fe6a77fcdbe4` | step 18: online RL on CodeContests test writing, iteration 1 |

Each folder carries its `adapter_config.json`; the training logs and data for each study are under `runs/`.
None of these adapters beat the untrained model on the held-out tasks; they are kept as the record of what was tried.
