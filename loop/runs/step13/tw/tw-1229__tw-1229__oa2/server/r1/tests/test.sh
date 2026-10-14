#!/bin/bash
# Self-contained verifier: needs only python3 (and optionally ffprobe). No network.
mkdir -p /logs/verifier
python3 /tests/test_outputs.py
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
