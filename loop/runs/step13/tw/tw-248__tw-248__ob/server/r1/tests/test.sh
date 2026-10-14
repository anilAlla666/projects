#!/bin/bash
# Verifier for the network driver report task.
mkdir -p /logs/verifier
REPORT=/app/network_driver_report.txt
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if perl "$DIR/check_report.pl" "$REPORT"; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit 0
