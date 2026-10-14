#!/bin/bash
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
child=$!
trap 'kill -TERM "$child" 2>/dev/null' TERM INT
wait "$child"
trap - TERM INT
wait "$child"
# keep the "service" around
while true; do sleep 1; done
