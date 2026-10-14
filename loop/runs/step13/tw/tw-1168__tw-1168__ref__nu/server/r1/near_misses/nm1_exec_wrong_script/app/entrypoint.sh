#!/bin/bash
# Near miss: exec but wrong script path
echo "Starting data processor via entrypoint..."
exec python3 /app/wrong_app.py
