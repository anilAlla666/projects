#!/usr/bin/env bash
# WL05 Multi-tenant ×8 — spawn 8 instances of WL01-equivalent.
# Each gets a unique CIPHER_TENANT_ID. PIDs returned via /tmp/wl05_pids.txt.
set -u
DURATION="${WL_DURATION:-600}"
PIDS=()
> /tmp/wl05_pids.txt
for i in 1 2 3 4 5 6 7 8; do
  CIPHER_TENANT_ID="wl05_t${i}" WL_DURATION="$DURATION" \
    python3 /home/ubuntu/cipher_workloads/drivers/wl01_decode_b1.py \
    > "/tmp/wl05_t${i}.log" 2>&1 &
  PID=$!
  PIDS+=($PID)
  echo $PID >> /tmp/wl05_pids.txt
  sleep 0.5  # stagger
done
echo "WL05 launched 8 tenants: ${PIDS[*]}"
# Wait for all
for p in "${PIDS[@]}"; do wait $p; done
echo "WL05 all 8 tenants completed"
