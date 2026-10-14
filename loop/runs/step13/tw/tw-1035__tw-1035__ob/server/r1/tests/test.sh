#!/bin/bash
# Verifier for the offline website mirror task. No network / package installs needed.
mkdir -p /logs/verifier
if python3 /tests/check_mirror.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
