---
name: cipher-abi-rule
description: "/dev/cipher ioctl ABI rule — additive nr-only changes, reserved nrs return -ENOSYS, struct layouts frozen on first ship"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a1207d0-bd39-48db-a0a9-37165af96920
---

The `/dev/cipher` character device exposes a stable ioctl ABI on magic `'C'`. Two rules govern its evolution:

1. **Additive only.** New ioctls take fresh `nr`s. Existing nrs never change semantics, never change struct layout, never change direction (`_IOW`/`_IOR`). Once `CIPHER_REGISTER_TENANT` (nr 1) shipped with its struct, the layout is locked forever.

2. **Reserved nrs return `-ENOSYS`.** Phase 2 reserved nrs 2/3/4 (`SNAPSHOT`, `RESET`, `GET_VERSION`) — the kernel handler explicitly rejects them with `-ENOSYS` so future implementations cannot collide with code that started using those nrs accidentally. New ioctls allocate from the next free nr (Phase 3 uses 5/6/7).

3. **Header struct portability.** `cipher_ioctl.h` uses only `__u32` / `__u64` / `char[N]` — no kernel-only types, no `#ifdef __KERNEL__`. The same header includes cleanly in kernel modules and userspace clients.

**Why:** This is a public contract with userspace clients (libcipher_v2 today, third-party integrations later). Procurement reviewers and downstream tooling will hash this header to detect breaking changes. The reserved-nr-with-ENOSYS pattern is the user's chosen mechanism for declaring future intent without committing the implementation.

**How to apply:** Before adding a new ioctl: (a) pick the lowest free `nr`, (b) define a struct with only portable types, (c) document the contract in `cipher_ioctl.h` comments, (d) implement the kernel handler, (e) never modify the struct layout afterward. If a struct field becomes obsolete, leave it in place and add a new field — do not remove or reorder.
