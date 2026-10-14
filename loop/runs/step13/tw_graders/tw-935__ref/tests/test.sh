#!/bin/bash
if [ "$(sha256sum /app/hardware_audit.sh 2>/dev/null | cut -d" " -f1)" != "7aeb11a7c157d13de74e816b21bd447d3435bc675f916a4a02de658d4d11f2f9" ]; then echo "FAIL: /app/hardware_audit.sh does not match the expected output"; mkdir -p /logs/verifier; echo 0 > /logs/verifier/reward.txt; exit 0; fi

# You can modify anything in this script. Just make sure it generates test output that
# is parsed by a parser (PytestParser by default).

# This boilerplate script installs uv (to manage Python) and runs pytest.

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

uv init
uv add pytest==8.4.1

# Run the hardware audit script first if it exists, so tests can evaluate the output
if [ -f "/app/hardware_audit.sh" ]; then
    chmod +x /app/hardware_audit.sh
    /app/hardware_audit.sh
fi

# Install other dependencies if needed (remember, by default this is a different venv
# than your agent's)

uv run pytest /tests/test_outputs.py -rA


if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
