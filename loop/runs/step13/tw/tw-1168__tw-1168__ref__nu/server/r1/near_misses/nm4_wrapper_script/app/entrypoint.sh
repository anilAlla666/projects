#!/bin/bash
# Near miss: uses a wrapper script that doesn't forward signals
echo "Starting data processor via entrypoint..."

cat > /tmp/wrapper.sh << 'WRAPPER'
#!/bin/bash
python3 /app/app.py
WRAPPER
chmod +x /tmp/wrapper.sh

exec /tmp/wrapper.sh
