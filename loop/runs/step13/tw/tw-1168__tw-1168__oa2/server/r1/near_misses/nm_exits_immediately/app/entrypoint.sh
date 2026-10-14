#!/bin/bash
echo "Starting data processor via entrypoint..."
python3 /app/app.py &
sleep 1
kill -TERM $!
wait
