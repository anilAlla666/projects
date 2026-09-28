---
name: cipher-kbuild-clean-wipes-ko
description: "Kbuild's clean target wipes every *.ko in the build dir, including renamed fallback binaries. Store fallbacks OUTSIDE the build tree."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a1207d0-bd39-48db-a0a9-37165af96920
---

The Linux kernel out-of-tree build's `make clean` target removes every `*.ko` in the M=... directory — including renamed fallback binaries like `cipher_kmod.ko.v0.2.0`. Do NOT store fallback `.ko` files inside the build directory.

**Why:** Discovered while bumping cipher_kmod from 0.2.0 to 0.3.0. Saved fallback as `/home/ubuntu/cipher_kmod/cipher_kmod.ko.v0.2.0` per CIPHER discipline. The subsequent `make clean` wiped it because kbuild's clean rule globs `*.ko`. Caught immediately because `md5sum` reported "No such file or directory" — fortunately the immutable `cipher-phase2-evidence/cipher_kmod.ko` was still on disk so the fallback was restorable.

**How to apply:** When preserving prior-version `.ko` fallbacks for CIPHER:
1. Store them at `/home/ubuntu/cipher_kmod.ko.v0.X.Y` (parent of the build dir), NOT inside `cipher_kmod/`.
2. Verify with `md5sum` after every `make clean` to confirm the fallback survived.
3. Keep the corresponding `cipher-phaseN-evidence.tar.gz` and extracted dir as a second-line backup — they're the ground truth that lets you reconstruct any fallback even if step 1 fails.

Same rule applies to any other artifact that matches kbuild's clean globs: `.o`, `.mod`, `.mod.c`, `.symvers`, `.order`. Live workspaces should hold only the canonical sources (`.c`, `.h`, `Kbuild`, `Makefile`); ship-time binaries live one directory up.
