---
name: cipher-incident-2-bar0-exit
description: "cipher_bar0_exit must NOT call pci_dev_put — that triggers another driver's devres race"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 325925bc-a462-4c74-9d86-9fab243e4b93
---

`cipher_bar0_exit` must NOT call `pci_dev_put(cipher_pci_dev)`. Keep `pci_iounmap` (our own iomap teardown). Drop the put and NULL the pointer instead. The pci_dev reference taken by `pci_get_device` in `cipher_bar0_init` is deliberately leaked for module lifetime; on rmmod the device stays alive because nvidia.ko already owns it.

**Why:** Calling `pci_dev_put` at module-exit time races with another driver's devres teardown (`device_release` → `devres_release_all` → `devm_attr_group_remove` → `sysfs_remove_group` → `kernfs_find_and_get_ns` on a `kobj.sd == NULL`). It crashed the kernel during a 0.4.2 rmmod cycle on 2026-05-13 with refcnt=-1 stuck, requiring a reboot to recover. See PHASE_4_NOTES.md Incident 2 for the full stack trace.

**How to apply:** Any future module-exit code that holds a `pci_dev *` taken via `pci_get_device` against an nvidia.ko-managed device should use the same pattern. If a future change reintroduces `pci_dev_put` in any `*_exit` callback, treat it as a kernel-crash regression. Verified-clean unload path: 20/20 insmod→smoke→rmmod cycles produce zero taint and zero dmesg WARN/BUG/Oops/NULL hits.
