#!/bin/bash
echo "Starting data processor via entrypoint..."
mkdir -p /app/data
echo "shutdown_complete:SIGTERM" > /app/data/shutdown_complete
echo "--- END OF DATA (clean shutdown) ---" >> /app/data/output.txt
exec python3 /app/app.py
