#!/usr/bin/env bash
# CP 3.4 — total rollback. Pure observability CP: nothing on the system
# to revert beyond the two rootless podman containers.
set -u
echo "[cp34 teardown] removing containers ..."
podman rm -f cipher-ch cipher-grafana 2>/dev/null || true
podman network rm cipher-net 2>/dev/null || true
sudo -n pkill -f cipher_flopd 2>/dev/null || true
echo "[cp34 teardown] done. kmod / libcipher / ABI untouched throughout."
podman ps -a --format '{{.Names}}' | grep -E 'cipher-(ch|grafana)' \
  && echo "WARN: containers still present" || echo "containers gone."
