---
name: cipher-project-layout
description: "Where the CIPHER GPU-multitenancy project actually lives on disk (specs often write /workspace, but the real home is /home/ubuntu)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a1207d0-bd39-48db-a0a9-37165af96920
---

CIPHER project source lives under `/home/ubuntu/`, NOT `/workspace/` even though some specs use that path.

- Kernel module: `/home/ubuntu/cipher_kmod/` (cipher_main.c, cipher_dev.c, cipher_ioctl.h, cipher_internal.h, cipher_probe.c, cipher_proc.c, cipher_ioctl_decode.c, Kbuild, Makefile)
- Userspace inject lib: `/home/ubuntu/libcipher_v2/` (cipher_inject.c, cipher_tenant.c, cipher_v2_internal.h, Makefile, libcipher_v2.so)
- Phase evidence tarballs: `/home/ubuntu/cipher-phaseN-evidence.tar.gz` plus extracted dirs
- The `/home/ubuntu/Anil/` directory exists but contains older artifacts; primary work is at the top level

**Why:** User runs the CIPHER work directly out of $HOME on a Lambda Labs H100 pod. Specs are sometimes written assuming a generic /workspace mount that this pod does not have.

**How to apply:** When a spec references `/workspace/cipher_kmod/...` or similar, mentally substitute `/home/ubuntu/cipher_kmod/...`. Mention the substitution once when starting work so the user can confirm. Between phases the live working tree may be reaped into the evidence tarball — check `ls /home/ubuntu/cipher_kmod` first; if missing, restore from the latest `cipher-phaseN-evidence/` directory before starting new work.
