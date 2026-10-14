#!/bin/bash
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
child=$!
trap 'kill -INT "$child"; wait "$child"' TERM
wait "$child"
