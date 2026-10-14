#!/bin/bash
# Near miss: trap forwards SIGTERM but doesn't wait for child
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
child=$!

cleanup() {
    echo "Received SIGTERM, forwarding to child..."
    kill -TERM "$child"
    # Missing: wait "$child" - shell exits immediately
    exit 0
}

trap cleanup TERM
wait "$child"
