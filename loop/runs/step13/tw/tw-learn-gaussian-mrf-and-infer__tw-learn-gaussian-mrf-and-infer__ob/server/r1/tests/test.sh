#!/bin/bash
# Self-contained verifier: no network, only python3 + numpy/pandas (installed in the image).
TEST_DIR="${TEST_DIR:-/tests}"
mkdir -p /logs/verifier
python3 "$TEST_DIR/check.py"
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
