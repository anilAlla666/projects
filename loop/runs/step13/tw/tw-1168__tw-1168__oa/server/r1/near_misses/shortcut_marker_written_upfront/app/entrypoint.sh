#!/bin/bash
echo "Starting data processor via entrypoint..."
mkdir -p /app/data
python3 /app/app.py &
child=$!
trap 'kill -KILL "$child"; echo "--- END OF DATA (clean shutdown) ---" >> /app/data/output.txt; echo shutdown_complete:SIGTERM > /app/data/shutdown_complete; exit 0' TERM
wait "$child"
