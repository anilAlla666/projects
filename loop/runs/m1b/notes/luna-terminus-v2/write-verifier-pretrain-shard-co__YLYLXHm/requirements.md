# Requirements

- **R1** — “`bash /app/run_pretrain.sh` runs to completion”
- **R2** — “writes a checkpoint”
- **R3** — “produces a validation loss close to **6.4**”
- **R4** — “Restore the intended training run rather than changing the training recipe”
- **R5** — Do not “drop[] examples, or substitut[e] duplicate examples to hit the target metric.”
- **R6** — “final state should recover the intended training examples used by the previous good run”
- **R7** — “statistically similar, generated, or otherwise approximate replacement inputs are not acceptable even if they produce a validation loss near the target.”
- **R8** — “Do not reduce `MAX_STEPS` in `/app/train_pretrain.py`; it is an upper bound.”
- **R9** — “Natural dataset exhaustion before 500 steps is expected”
- **R10** — “do not change iteration behavior just to force 500 steps.”
- **R11** — “Do not modify `/app/run_pretrain.sh`”
- **R12** — “Do not modify `/app/data/train/index.json`”
- **R13** — “Do not fake the results by printing out a fake loss value”
- **R14** — “a new training run must legitimately produce a checkpoint with the specified loss.”
- **R15** — “Diagnose the root cause and modify `/app`”

## Verifier-authoring requirements

- **R16** — “Write `/verifier_out/requirements.md`” and list every requirement as a short quote with an ID.
- **R17** — “Write your tests so that every check names the requirement id it enforces.”
- **R18** — “Do not write any check that is not traceable to a requirement”; thresholds must come from the task text or a measured correct solution, “never from a guess.”
- **R19** — “Solve the task yourself in a scratch copy” well enough to meet every requirement.
- **R20** — “make 2 or 3 flawed variants, each breaking exactly one requirement.”
- **R21** — “Run your verifier against your correct solution and each flawed variant”; output `1` for correct and `0` for every flawed one, and “fix the verifier, not the variants” if needed.
- **R22** — “Write the results to `/verifier_out/validation.md`.”
- **R23** — Provide `/verifier_out/tests/Dockerfile`; if using this environment, start with the specified pinned `FROM` image; include `COPY . /tests/`; “Do not set an `ENTRYPOINT`.”
- **R24** — Provide `/verifier_out/tests/test.sh`; it must write exactly `1` or `0` to `/logs/verifier/reward.txt` and “print which checks failed.”
- **R25** — The verifier runs in a fresh container, with only the enumerated `/app` paths carried over; it has no network and must finish within 600 seconds.
