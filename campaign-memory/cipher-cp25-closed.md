---
name: cipher-cp25-closed
description: CP 2.5 shipped — LD_PRELOAD-free deployment + cipher-platform.deb; libcipher_rt anchor now c2c5d313
metadata: 
  node_type: memory
  type: project
  originSessionId: 8482a673-2517-464b-974c-6170c963b21b
---

CP 2.5 SHIPPED 2026-05-16 — full close, adjudicated. Phase 2 CLOSED (15/23).
LD_PRELOAD-free deployment: the CIPHER runtime now deploys via
`CUDA_INJECTION64_PATH` alone; GOT/PLT patching (`cipher_rt_got_patch.{c,h}`)
replaces link-order symbol interposition for the cuBLAS shim + 3 SDPA
trampolines. Ships as `cipher-platform_1.0_amd64.deb` (md5 `bb924248`):
DKMS kmod source pkg + libcipher_rt/libcipher_v2/libc10 → /usr/lib/cipher/ +
`cipher-run` launcher.

All 5 gate criteria PASS:
- (a) injection-only parity: MATMUL 20/20, SDPA tramp_calls 20/20, GOT 8 slots
- (b) composed 3.602× tok/W [3.529, 3.676], inside CP 2.4 CI [3.591, 3.642]
- (c) clean dpkg cycle; torch-free libc10 resolution measured (advisor-caught)
- (d) zero regressions: CP 3.3 4/4, T4.6.4 5/5, allocator, devnode 0666, Fix A
- (e) anchors held — kmod srcversion `E427CAFA4E94D548233DC7A` unchanged,
  libcipher_v2 `86618c30`, libcipher_rt NEW anchor `c2c5d313`

**Anchor update:** libcipher_rt `c2c5d313` supersedes `5e304549` (the value
referenced in [[cipher-marlin-primary-ctx-pin]]). kmod 0.4.8 `e2f50452`
reference build unchanged; DKMS rebuilds it node-local to `6654d9e5`
(stripped + Secure-Boot-signed, same srcversion).

Engineering-marvel finding (CP 2.5 report §3): the D2(iii) bare-drop was
empirically tested, not assumed — severing ALL torch libs failed (8 undefined
`c10::` symbols, one a data symbol in RTLD_LOCAL scope). Refined to: sever
`libtorch_cpu` (451 MB, version-fragile), retain `libc10` (1.5 MB,
ABI-stable). Broken bare-drop preserved as `libcipher_rt.so.cp2_5_baredrop_BROKEN`
(`2f845393`). Pattern: characterize boundaries empirically before claiming.

Deliverables in `cipher-fusion-evidence/cp_2_5/`: `CP_2_5_REPORT.md`, the
`.deb`, `gate_d/` evidence. Phase 4.6 work still in flight (T4.6.5, T4.6.6).
Supersedes [[cipher-cp24-closed]] as the latest campaign close.
