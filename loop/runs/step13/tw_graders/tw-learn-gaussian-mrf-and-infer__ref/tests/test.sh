#!/bin/bash
if [ "$(sha256sum /app/final_mrf_sample.csv 2>/dev/null | cut -d" " -f1)" != "3267d0d277b4b67229d4138f0e66a631afdd191b2c640a3270e8e0753a8a6d00" ]; then echo "FAIL: /app/final_mrf_sample.csv does not match the expected output"; mkdir -p /logs/verifier; echo 0 > /logs/verifier/reward.txt; exit 0; fi

# You can modify anything in this script. Just make sure it generates test output that
# is parsed by a parser (PytestParser by default).

# This boilerplate script installs uv (to manage Python) and runs pytest.

# Set TEST_DIR default to /tests if not already set
TEST_DIR="${TEST_DIR:-/tests}"

# Install curl
apt-get update
apt-get install -y curl

# Install uv
curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh

source $HOME/.local/bin/env

# Check if we're in a valid working directory
if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set. Please set a WORKDIR in your Dockerfile before running this script."
    exit 1
fi

# Skip uv init if pyproject.toml already exists
if [ ! -f "pyproject.toml" ]; then
    uv init
fi

uv add pytest==8.4.1

# Install dependencies needed for tests
uv add pandas numpy

uv run pytest /tests/test_outputs.py -rA


if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
