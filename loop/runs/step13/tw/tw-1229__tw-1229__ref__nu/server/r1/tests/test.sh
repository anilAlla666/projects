#!/bin/bash

# Run the validation tests directly - no hardcoded hash check, no pytest dependency
# The tests validate structure, logic, and correctness

# Copy test script to /tests if needed
mkdir -p /tests
cp /verifier_out/tests/test_outputs.py /tests/test_outputs.py

# Run the validation tests
cd /tests
python3 test_outputs.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    mkdir -p /logs/verifier
    echo 1 > /logs/verifier/reward.txt
else
    mkdir -p /logs/verifier
    echo 0 > /logs/verifier/reward.txt
fi

exit $EXIT_CODE
