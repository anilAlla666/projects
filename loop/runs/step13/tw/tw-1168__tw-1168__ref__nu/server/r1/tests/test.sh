#!/bin/bash
# Test script for verifying entrypoint.sh fix
# Runs the unittest-based test suite

# Determine test directory
if [ -d "/tests" ]; then
    TEST_DIR="/tests"
else
    TEST_DIR="$(dirname "$0")"
fi

cd "$TEST_DIR"
python3 test_outputs.py

if [ $? -eq 0 ]; then
  mkdir -p /logs/verifier
  echo 1 > /logs/verifier/reward.txt
else
  mkdir -p /logs/verifier
  echo 0 > /logs/verifier/reward.txt
fi
