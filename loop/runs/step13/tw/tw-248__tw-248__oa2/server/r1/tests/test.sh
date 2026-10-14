#!/bin/bash
# Grader entry point: writes 1 or 0 to /logs/verifier/reward.txt
mkdir -p /logs/verifier
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if perl "$DIR/check_report.pl" /app/network_driver_report.txt; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
