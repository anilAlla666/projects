#!/bin/bash
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
child=$!
trap 'sleep 11; kill -TERM "$child"; wait "$child"' TERM
wait "$child"
