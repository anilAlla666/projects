#!/bin/bash
echo "Starting data processor via entrypoint..."

trap '
echo "Received SIGTERM, performing graceful shutdown..."
mkdir -p /app/data
echo "shutdown_complete:SIGTERM" > /app/data/shutdown_complete
echo "--- END OF DATA (clean shutdown) ---" >> /app/data/output.txt
echo "Cleanup complete, exiting gracefully"
exit 0
' SIGTERM

python3 /app/app.py &
wait $!
