#!/bin/bash
# Self-contained verifier: needs only python3 (no network, no pytest).
mkdir -p /logs/verifier
APP_DIR="${APP_DIR:-/app}" python3 /tests/verify.py
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
