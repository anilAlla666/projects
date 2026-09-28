#!/usr/bin/env bash
# CP 2.5 gate (d) indicator (e) — module unload safety.
# rmmod refused while a tenant fd is open; succeeds once closed; re-insmod
# is clean and recreates /dev/cipher + /dev/cipher_kvdedup at 0666.
set -u
KO=/home/ubuntu/cipher_kmod/cipher_kmod.ko

echo "### gate (d) indicator (e) — module unload safety ###"
echo "--- pre-state ---"
lsmod | grep '^cipher_kmod' || echo "(cipher_kmod not loaded)"
echo "ko anchor: $(md5sum "$KO")"

echo "--- hold a tenant fd on /dev/cipher ---"
python3 -c "import time,signal; f=open('/dev/cipher'); print('holder: fd open',flush=True); signal.pause()" &
HOLDER=$!
sleep 2
echo "holder pid=$HOLDER  refcount now: $(lsmod | awk '/^cipher_kmod/{print $3}')"

echo "--- rmmod WITH fd open (expect REFUSAL) ---"
if sudo -n rmmod cipher_kmod 2>&1; then
  echo "UNEXPECTED: rmmod succeeded while fd open"
else
  echo "-> rmmod refused (expected — module in use)"
fi

echo "--- close the fd ---"
kill -9 "$HOLDER" 2>/dev/null || true
wait "$HOLDER" 2>/dev/null || true
sleep 2
echo "refcount after close: $(lsmod | awk '/^cipher_kmod/{print $3}')"

echo "--- rmmod with no users (expect SUCCESS) ---"
if sudo -n rmmod cipher_kmod; then
  echo "-> rmmod OK"
else
  echo "FAIL: rmmod refused with refcount 0"; exit 1
fi
echo "nodes after rmmod: $(ls /dev/cipher /dev/cipher_kvdedup 2>&1 || echo 'both gone (expected)')"

echo "--- re-insmod (anchor e2f50452) ---"
sudo -n insmod "$KO"
sleep 1
echo "--- post-state ---"
lsmod | grep '^cipher_kmod'
ls -l /dev/cipher /dev/cipher_kvdedup
echo "### indicator (e) complete ###"
