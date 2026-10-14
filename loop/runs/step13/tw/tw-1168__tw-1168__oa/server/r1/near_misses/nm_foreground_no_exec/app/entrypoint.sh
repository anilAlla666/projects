#!/bin/bash
echo "Starting data processor via entrypoint..."
# runs in foreground, but bash stays PID 1 and does not forward SIGTERM
python3 /app/app.py
