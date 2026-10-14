#!/bin/bash
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
child=$!
trap 'kill -KILL "$child" 2>/dev/null; exit 0' TERM INT
wait "$child"
