#!/usr/bin/env bash
# CP 2.5 item 8 — install / uninstall / reinstall transcript on the live pod.
# RUN ONLY AFTER item 6 (composed gate) has finished — prerm unloads
# cipher_kmod and removes /dev/cipher.
# Captures a full dpkg lifecycle + an end-to-end installed-mode smoke.
set -u
DEB=/home/ubuntu/cipher-fusion-evidence/cp_2_5/cipher-platform_1.0_amd64.deb
SMOKE=/home/ubuntu/cipher-fusion-evidence/cp_2_5/audit/cp25_smoke.py

hr() { echo "================================================================"; }
state() {
  echo "-- state: $1 --"
  echo "  dpkg -l cipher-platform : $(dpkg-query -W -f='${Status}\n' cipher-platform 2>/dev/null || echo 'not-installed')"
  echo "  dkms status             : $(dkms status -m cipher-kmod -v 0.4.8 2>/dev/null || echo '(none)')"
  echo "  lsmod cipher_kmod       : $(lsmod | grep -c '^cipher_kmod') module(s)"
  echo "  /dev/cipher             : $(ls -l /dev/cipher 2>/dev/null || echo 'absent')"
  echo "  /usr/lib/cipher         : $(ls /usr/lib/cipher 2>/dev/null | tr '\n' ' ' || echo 'absent')"
  echo "  cipher-run              : $([ -x /usr/bin/cipher-run ] && echo present || echo absent)"
}

hr; echo "PHASE 0 — pre-install state"; hr
state "before install"

hr; echo "PHASE 1 — INSTALL  (dpkg -i)"; hr
sudo dpkg -i "$DEB"; echo "[dpkg -i rc=$?]"
state "after install"

hr; echo "PHASE 2 — END-TO-END SMOKE under installed mode (cipher-run)"; hr
echo "\$ cipher-run python3 cp25_smoke.py"
/usr/bin/cipher-run python3 "$SMOKE" 2>&1 \
  | grep -E "GOT: patch|MATMUL: exit|CUBLAS-SHIM|matmul done|torch |attn] exit" -i
echo "[smoke rc=${PIPESTATUS[0]}]"

hr; echo "PHASE 3 — UNINSTALL  (dpkg -r)"; hr
sudo dpkg -r cipher-platform; echo "[dpkg -r rc=$?]"
state "after uninstall"

hr; echo "PHASE 4 — REINSTALL  (dpkg -i again — idempotency)"; hr
sudo dpkg -i "$DEB"; echo "[dpkg -i rc=$?]"
state "after reinstall"

hr; echo "PHASE 5 — smoke again after reinstall"; hr
/usr/bin/cipher-run python3 "$SMOKE" 2>&1 \
  | grep -E "GOT: patch|MATMUL: exit|matmul done" -i
echo "[smoke rc=${PIPESTATUS[0]}]"
hr; echo "install cycle complete"; hr
