# Write the verifier for a task

You are writing the **verifier**: the tests that decide whether someone else's attempt at a task succeeded.
Your verifier will be run on many real attempts. Some are correct; many are subtly wrong (they pass most checks
but break one stated requirement). **Accepting a wrong attempt and rejecting a correct one count equally against
you.** A verifier that rejects a correct solution is as bad as one that accepts a wrong one.

## The task the solver was given

<task-instruction>
{instruction}
</task-instruction>

## Where you are

This machine is the solver's starting environment: the same files, tools and services the solver begins with.
Its image is `{env_image}`. You have {minutes} minutes.

## How to work (follow these steps in order)

**Step 1: requirements.** Write `/verifier_out/requirements.md`. List every requirement the task text states,
each as a short quote from the text with an id (R1, R2, ...). Include constraints on *how* the work must be
done, not only the final result. Do not add requirements the text does not state.

**Step 2: checks.** Write your tests so that every check names the requirement id it enforces. Do not write any
check that is not traceable to a requirement in step 1: a correct solution may legitimately differ from yours in
every way the task leaves open. Any numeric threshold must come from the task text or from measuring your own
correct solution in step 3, never from a guess about what is "plausible".

**Step 3: your own correct solution, and flawed ones.** Solve the task yourself in a scratch copy (for example
`/scratch/good`), well enough that it meets every requirement. Then make 2 or 3 flawed variants, each breaking
exactly one requirement from step 1, the kind of subtle mistake a solver might make.

**Step 4: prove the verifier.** Run your verifier against your correct solution and each flawed variant, with the
files placed at the paths the verifier will see (listed below). It must write `1` for the correct solution and
`0` for every flawed one. If not, fix the verifier, not the variants, and run again. Write the results to
`/verifier_out/validation.md`.

## What to write

A folder `/verifier_out/tests/` containing:

- `Dockerfile`: builds the container your tests run in. Start it with exactly this line if your tests need what
  the environment contains: `FROM {env_image}`. Install everything else you need here; the container has no
  network once it runs. Include `COPY . /tests/`. Do not set an `ENTRYPOINT`.
- `test.sh`: the grader runs it as `bash /tests/test.sh` inside your container. It must write exactly `1` (the
  attempt meets every requirement) or `0` (anything else) to `/logs/verifier/reward.txt`, and print which checks
  failed.
- any other files your tests use.

## What your tests will see

Your tests run in a fresh container built from your Dockerfile, not on the solver's machine. The only things
carried over from a solver's machine are these paths, copied to the same locations:

{artifacts}

Nothing else from the solver's machine is available. Your tests run with no network access and must finish
within {timeout} seconds.

When `/verifier_out/tests/`, `/verifier_out/requirements.md` and `/verifier_out/validation.md` are complete, stop.
