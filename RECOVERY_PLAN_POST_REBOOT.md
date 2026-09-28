# Recovery Plan — Post-Reboot

Reboot initiated 2026-05-13 14:30 to recover from cipher_bar0_exit oops.
Module stuck refcnt=-1 before reboot.

## Step 1 — Post-reboot verification (before any module load)

```bash
mkdir -p /home/ubuntu/cipher-phase4-evidence
sudo dmesg > /home/ubuntu/cipher-phase4-evidence/post_reboot_dmesg.txt
md5sum /tmp/cipher-phase3-evidence/fallback/cipher_kmod.ko.v0.2.0 \
       /tmp/cipher-phase3-evidence/fallback/libcipher_v2.so.v0.2.0
# expect: 55ab8c0c... + 86618c30...
lsmod | grep cipher  # expect: empty
cat /proc/sys/kernel/tainted  # expect: 0
md5sum /home/ubuntu/cipher_kmod.ko.v0.4.0 \
       /home/ubuntu/cipher_kmod.ko.v0.4.1 \
       /home/ubuntu/cipher_kmod.ko.v0.4.2_with_do_exit_release
# expect: f1048f2d... a6b7d676... abfaff35...
```

## Step 2 — Apply Fix A: cipher_bar0_exit leak pci_dev_put

Edit /home/ubuntu/cipher_kmod/cipher_bar0.c, in `cipher_bar0_exit`:
- Keep `pci_iounmap(cipher_pci_dev, cipher_bar0_base)`
- Replace `pci_dev_put(cipher_pci_dev)` with a comment explaining
  the deliberate leak and a reference to PHASE_4_NOTES.md Incident 2
- Set `cipher_pci_dev = NULL` after (so re-load doesn't reuse stale ptr)

## Step 3 — Apply Fix B: do_exit slot release (already in source)

Verify cipher_partition_release_slots_only is in cipher_partition_allocator.c
and called from cipher_do_exit_pre in cipher_probe.c.

Bump MODULE_VERSION → "0.4.3", banner update in cipher_main.c, cipher_proc.c.

## Step 4 — Build 0.4.3, save artifact

```bash
cd /home/ubuntu/cipher_kmod && make
modinfo cipher_kmod.ko | grep -E '^version|^srcversion'  # expect 0.4.3 + new srcversion
nm cipher_kmod.ko | grep -i 'cipher_partition' | head
nm cipher_kmod.ko | grep -iE 'iowrite32|__raw_writel|memcpy_toio' || echo "(refinement 2 ok)"
cp cipher_kmod.ko /home/ubuntu/cipher_kmod.ko.v0.4.3
md5sum /home/ubuntu/cipher_kmod.ko.v0.4.3 | tee /home/ubuntu/cipher_kmod.ko.v0.4.3.md5
tar czf /home/ubuntu/cipher_kmod_src_v0.4.3.tar.gz \
    --exclude='*.o' --exclude='*.ko' --exclude='*.mod*' \
    --exclude='Module.symvers' --exclude='modules.order' \
    --exclude='.*.cmd' --exclude='.*.d' -C /home/ubuntu cipher_kmod/
md5sum /home/ubuntu/cipher_kmod_src_v0.4.3.tar.gz
```

## Step 5 — 20-cycle load/unload stress test

Required gate before functional tests:

```bash
for i in $(seq 1 20); do
  sudo insmod /home/ubuntu/cipher_kmod.ko.v0.4.3 || { echo "load $i FAIL"; break; }
  sudo chmod 666 /dev/cipher
  /tmp/phase3_happy >/dev/null 2>&1 || { echo "smoke $i FAIL"; break; }
  sudo rmmod cipher_kmod || { echo "unload $i FAIL"; break; }
  echo "cycle $i OK"
done
sudo dmesg --since "5 min ago" | grep -iE 'oops|warn|bug|null' | head -20
```

If 20/20 clean → proceed. If any cycle fails → STOP, bring to user.

## Step 6 — Functional gates

```bash
sudo insmod /home/ubuntu/cipher_kmod.ko.v0.4.3
sudo chmod 666 /dev/cipher
/tmp/phase3_happy && /tmp/phase3_negative && sudo /tmp/phase3_root
/tmp/cipher_test_phase4_snapshot
sudo /tmp/cipher_test_phase4_partition
/tmp/cipher_test_phase4_partition_contention
```

Document p50/p99/ratio in PHASE_4_NOTES.md regardless of gate threshold —
let user decide final gate.

## Step 7 — Phase 3 demo reproducibility

Verify the Phase 3 stage-5b /metrics demo still works under 0.4.3 (substrate
preserved). Run cipher-gpustate + cipher-exporter, hit /metrics, observe
per-tenant rows.

## Step 8 — Final documentation in PHASE_4_NOTES.md

- Incident 2 status: fixed by leak workaround, verified by 20-cycle stress
- Incident 3: do_exit slot release fix verified by Tests 4-5
- 0.4.3 is working baseline, all gates pass
- Module version table updated
- Contention p99/p50 ratio under 0.4.3 (no pre-decided gate)

## Step 9 — Baseline collection (background)

Re-collect batch 1 (WL01,02,06,11,14,15), then batch 2/3/4 + WL05.

## Step 10 — T4.0.9.D checkpoint report.
