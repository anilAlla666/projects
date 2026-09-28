---
name: cipher-devnode-codified
description: kmod 0.4.8 codifies /dev/cipher device-node mode at 0666 via devnode callback; new campaign anchor
metadata: 
  node_type: memory
  type: project
  originSessionId: 6147a283-c0ca-4daf-83c4-310b1e306011
---

CP 2.4 regression audit (2026-05-15) found CP 3.3's kmod reload left
`/dev/cipher` at the 0600 default — the device mode had never been codified
in the kmod (0666 was always a manual post-insmod chmod). User adjudicated
**Posture A**: keep T4.3.2's design (non-root clock actuation via
`CIPHER_SET_CLOCK_MHZ` is intended; the ioctl has no CAP check by design —
node permission is the gate, bounds-clamp + uid-audit is the safety).

Fix shipped in **kmod 0.4.8 `e2f50452f668859a96b1e25a2cba4e10`** — `devnode`
callbacks on `cipher_class` (cipher_dev.c) and `kvd_class` (cipher_kvdedup.c)
codify both `/dev/cipher` and `/dev/cipher_kvdedup` at 0666 root:root, so the
mode is reload-stable. No ABI change. CP 3.3 gate re-verified on 0.4.8 (4/4
PASS). **0.4.8 supersedes 0.4.7 `2a69f9de` as the campaign kmod anchor.**
Rollback: `cipher-fusion-evidence/cp_2_4/kmod_0.4.7_pre_devnode.ko`.

**Why:** future sessions should not re-discover the chmod fragility or
re-apply a runtime chmod — the mode is now in kmod source.
**How to apply:** build on 0.4.8; never `chmod /dev/cipher` as a workaround.
See [[cipher-regression-discipline]], [[cipher-t432-kmod-volt-ioctl]],
[[cipher-abi-rule]].
