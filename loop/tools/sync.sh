#!/bin/sh
# Copy the repo to the server. Data lives in /root/vbench, outside the synced folder.
# Files must be owned by root there, or git refuses the repository ("dubious ownership") and run dirs lose their sha.
set -e
cd "$(dirname "$0")/.."
ssh -o BatchMode=yes root@178.156.248.194 'mkdir -p /root/vbench/repo'
rsync -az --delete --no-owner --no-group --exclude .venv --exclude __pycache__ ./ root@178.156.248.194:/root/vbench/repo/
ssh -o BatchMode=yes root@178.156.248.194 'chown -R root:root /root/vbench/repo'
