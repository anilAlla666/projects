#!/bin/bash
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
if timeout 120 python3 /tests/test_outputs.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
