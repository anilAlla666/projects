#!/bin/bash
# Near miss: trap but forwards SIGKILL instead of SIGTERM
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
child=$!

cleanup() {
    echo "Received SIGTERM, forwarding SIGKILL to child..."
    kill -KILL "$child"
    wait "$child"
    exit 0
}

trap cleanup TERM
wait "$child"
