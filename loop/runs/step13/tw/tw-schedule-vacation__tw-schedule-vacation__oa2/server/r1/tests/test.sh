#!/bin/bash
mkdir -p /logs/verifier
if python3 /tests/run_tests.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
