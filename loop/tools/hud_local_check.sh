#!/bin/bash
# Local check of a built HUD task's grader, in Docker: official tests (expect reward 1), always pass, always
# fail, no tests and a cheating grader (expect 0). Docker masks /proc and blocks user namespaces by default,
# so this lifts those restrictions for the grader container only; the tests still run in the grader's own
# nested sandbox without privileges.
#   tools/hud_local_check.sh <task>
set -euo pipefail
task=$1
dir=/root/vbench/hud_tasks/write-verifier-$task
img=vbhud-$task
docker build -q -t $img $dir/tests >/dev/null
opts="--cap-add SYS_ADMIN --security-opt seccomp=unconfined --security-opt apparmor=unconfined"
checks=/root/vbench/repo/templates/hud/checks
for v in official always_pass always_fail cheat none; do
  case $v in
    official) mount="-v $dir/solution/official_tests:/verifier_out/tests:ro" ;;
    none) mount="" ;;
    *) mount="-v $checks/$v:/verifier_out/tests:ro" ;;
  esac
  printf '%-12s ' $v
  docker run --rm $opts $mount $img bash /tests/test.sh 2>/dev/null | tail -1
done
