#!/bin/bash
# Grader entry point. No network needed: uses only perl-base and dpkg tools present in the image.
mkdir -p /logs/verifier
if perl /tests/check_report.pl; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
