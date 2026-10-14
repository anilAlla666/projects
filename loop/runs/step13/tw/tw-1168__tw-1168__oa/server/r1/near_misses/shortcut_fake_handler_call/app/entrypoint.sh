#!/bin/bash
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
child=$!
trap 'kill -KILL "$child"; cd /app && python3 -c "import app; app.cleanup_handler(15, None)"; exit 0' TERM
wait "$child"
