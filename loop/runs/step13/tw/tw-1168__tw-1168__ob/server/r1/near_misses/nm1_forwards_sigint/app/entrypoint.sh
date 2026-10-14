#!/bin/bash
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
child=$!
trap 'kill -INT "$child" 2>/dev/null' TERM INT
wait "$child"
trap - TERM INT
wait "$child"
