# GPU cost ledger

Cap: $100, raised to $200 by the user at 08:31 BST on 8 Oct. Instance: Lambda 1x A100 40GB SXM4, launched 8 Oct 2026 06:08 BST. Hourly price: read from the
Lambda console and entered here by the user; until then the ledger uses $1.29/h, Lambda's published A100 40GB
price as remembered, and says so.

| Stage | From | To | Hours | Dollars (at $1.29/h) |
|---|---|---|---:|---:|
| A100: setup, plan 1b build and tests, Stage 0 | 06:08 | 07:12 (terminated) | 1.07 | 1.38 |

## 4x H100 80GB (from 8 Oct, about 07:08 BST; start time approximate; $16.00 per hour, given by the user)

| Stage | From | To | Hours | Dollars (at $16/h) |
|---|---|---|---:|---:|
| Corpus, tokenizer, learning-rate check, pretraining started; judging-data tasks 1 to 6 | 07:08 | 08:31 | 1.38 | 22.13 |
| Pretraining finished (09:12); qualification, task creation, open-model attempts | 08:31 | 09:38 | 1.12 | 17.87 |

| Judging data (examples, hard negatives, real-attempts runs), attack prose, 32B baselines; fine-tuning round 1 started | 09:38 | 11:40 | 2.03 | 32.53 |

Running total at 11:40: $1.38 (A100) + $72.53 (H100) = $73.91 of $200.

Cap raised to $300 by the user in the afternoon.

| Stage | From | To | Hours | Dollars (at $16/h) |
|---|---|---|---:|---:|
| Judge rounds 2 to 5, realistic judge, Verifier Bench transfer test; grader writer rounds 1 to 4 (training, exams, reward-filtered round) | 11:40 | 19:35 | 7.92 | 126.67 |

Running total at 19:35: $1.38 (A100) + $199.20 (H100, 12.45 h from 07:08 by instance uptime) = $200.58 of $300.
Every further hour until the instance is terminated adds $16.

| Stage | From | To | Hours | Dollars (at $16/h) |
|---|---|---|---:|---:|
| Idle: waiting for the user's choice of next step (no work running) | 19:35 | 02:25 (9 Oct) | 6.83 | 109.33 |
| Untuned Qwen2.5-Coder-32B exam started, then stopped at the cap (no result) | 02:25 | 02:33 | 0.13 | 2.13 |

**Running total at 02:33 BST on 9 Oct: $1.38 (A100) + $310.67 (H100, 19.42 h of instance uptime) = $312.05, over the
$300 cap by about $12.** The instance sat idle for about seven hours while a question to the user was open; nothing
stopped the meter during that wait. All work is stopped and the GPUs are free; the instance needs terminating.

## Anthropic API (outside the GPU cap; approved by the user for the Sonnet comparison)

| Run | Calls | Dollars (at $3 / $15 per million input / output tokens, Sonnet list price, assumed) |
|---|---:|---:|
| Sonnet 5.5 judge on 939 real patches, thinking on | 939 | 11.46 |
| Sonnet 5.5 judge on 500 real patches carrying a persuasive note | 500 | 6.01 |
| A first run whose verdict parser was wrong, stopped after about two minutes (rows discarded, not measured) | ~300 | about 4 to 5 |

| Sonnet 5.5 as grader writer, rounds 1 to 3 (80 tasks each; one run's log records $2.27, the others were not kept) | ~240 | about 6 |
| Sonnet 5.5 as grader writer, round 4 (79 tasks, new-file prompt) | 79 | 3.00 |

API total: about $31.
