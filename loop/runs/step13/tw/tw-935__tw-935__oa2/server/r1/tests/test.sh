#!/bin/bash
# Verifier for hardware_audit.sh. Needs only bash, coreutils, python3 (no network).
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
SCRIPT=/app/hardware_audit.sh
OUT=/tmp/hardware_audit.json
rm -f "$OUT" /tmp/hardware_audit_graceful.json /tmp/hardware_audit_ran_at

if [ -f "$SCRIPT" ] && [ -x "$SCRIPT" ]; then
    # 1) graceful-degradation run: lsusb present on PATH but failing / producing nothing
    SHIM=$(mktemp -d)
    printf '#!/bin/sh\nexit 1\n' > "$SHIM/lsusb"; chmod +x "$SHIM/lsusb"
    (cd /app && PATH="$SHIM:$PATH" timeout 120 "$SCRIPT" >/tmp/ha_graceful.log 2>&1)
    [ -f "$OUT" ] && mv "$OUT" /tmp/hardware_audit_graceful.json
    rm -rf "$SHIM"
    # 2) normal run (this is the output that is checked against the live system)
    date -u +%s > /tmp/hardware_audit_ran_at
    (cd /app && timeout 120 "$SCRIPT" >/tmp/ha_run.log 2>&1)
    echo "script exit code: $?"
fi

if bash /tests/checks.sh; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
