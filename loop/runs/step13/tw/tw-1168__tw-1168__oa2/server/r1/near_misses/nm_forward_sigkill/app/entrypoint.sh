#!/bin/bash
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
child=$!
trap 'kill -KILL "$child"; echo "shutdown_complete:SIGTERM" > /app/data/shutdown_complete' TERM
wait "$child"
