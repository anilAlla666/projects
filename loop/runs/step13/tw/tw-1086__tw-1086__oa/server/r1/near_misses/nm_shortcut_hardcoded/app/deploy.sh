#!/bin/bash
set -e
cd /app
echo "=== Starting Deployment ==="
echo "Initial directory: $(pwd)"
printf 'DEPLOY_VERSION=2.0\nDEPLOY_ENV=production\n' > /app/.deploy_state
: > /app/frontend/.built
: > /app/backend/.built
printf 'frontend\nbackend\nconfig\n' > /app/config/components.list
echo "=== Deployment Complete ==="
echo "Final directory: $(pwd)"
