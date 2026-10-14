# Write the verifier for a task

You are not solving this task. You are writing the **verifier**: the tests that decide whether someone
else's attempt at it succeeded. Your verifier will be run on many real attempts, some correct and some
subtly wrong. Accepting a wrong attempt and rejecting a correct one count equally against you.

## The task the solver was given

<task-instruction>
{instruction}
</task-instruction>

## Where you are

This machine is the solver's starting environment: the same files, tools and services the solver begins
with. Its image is `{env_image}`; your verifier's Dockerfile may start `FROM` it if your tests need what
it contains. Explore it and run anything you like to understand what a correct result looks like. Do not
attempt the whole task. A small hand-made output, in a scratch directory outside `/verifier_out`, is
enough to check that your tests run.

## What to write

Write a folder `/verifier_out/tests/` containing:

- `Dockerfile` — builds the container your tests run in. Install everything you need here; the
  container has no network once it runs. The folder is copied into the image with `COPY . /tests/`,
  so include that line.
- `test.sh` — the grader runs it as `bash /tests/test.sh` inside your container (do not set an ENTRYPOINT;
  the grader starts the container and then runs this script). It must write exactly `1` (the attempt fully
  meets the task) or `0` (anything else) to `/logs/verifier/reward.txt`.
- any other files your tests use.

## What your tests will see

Your tests run in a fresh container built from your Dockerfile, not on the solver's machine. The only
things carried over from a solver's machine are these paths, copied to the same locations:

{artifacts}

Nothing else from the solver's machine is available. Your tests run with no network access and must
finish within {timeout} seconds. Anything your tests need must be written by you or installed in your
Dockerfile.

When `/verifier_out/tests/` is complete, stop.
