#!/bin/bash
mkdir -p /logs/verifier
test -f /out/net.json && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
