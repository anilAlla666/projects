#!/bin/sh
# Copy the repository to the GPU instance (code only; experiments stay where they are made).
set -e
HOST=${HOST:?set HOST=ubuntu@<ip>}
rsync -az --delete --exclude .venv --exclude __pycache__ --exclude experiments --exclude .git \
  "$(dirname "$0")/../" "$HOST:~/verifier_model/"
