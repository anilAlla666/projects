#!/bin/bash

# Run pytest directly (grading container has Python and pytest pre-installed)
# Fall back to installing via uv if needed

if command -v pytest >/dev/null 2>&1; then
    pytest /tests/test_outputs.py -rA
else
    # Install uv and pytest if not available
    apt-get update && apt-get install -y curl 2>/dev/null || true
    curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh 2>/dev/null || true
    source $HOME/.local/bin/env 2>/dev/null || true
    uv init 2>/dev/null || true
    uv add pytest==8.4.1 2>/dev/null || true
    uv run pytest /tests/test_outputs.py -rA
fi

if [ $? -eq 0 ]; then
  mkdir -p /logs/verifier
  echo 1 > /logs/verifier/reward.txt
else
  mkdir -p /logs/verifier
  echo 0 > /logs/verifier/reward.txt
fi
