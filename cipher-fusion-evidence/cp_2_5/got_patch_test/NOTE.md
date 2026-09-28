# CP 2.5 — GOT patcher standalone test — NOTE

**Date:** 2026-05-16. Build STEP item 1 (GOT patcher) verification. Not the
CP 2.5 report.

`cipher_rt_got_patch.{c,h}` (in `cipher_rt_phase4/`) is the critical-path
mechanism: it replaces LD_PRELOAD link-order symbol interposition with
runtime GOT patching driven from `InitializeInjection2`.

## Test design
`got_patch_test.c` — no CUDA. Registers `puts` → a `fake_puts` trampoline,
applies the patch, and asserts the four properties that matter:

1. **baseline miss** — a `puts` call before `apply()` does NOT hit the fake.
2. **patch hit** — the next `puts` routes through `fake_puts`.
3. **idempotent** — a second `apply()` patches 0 new slots.
4. **durable** — interception survives the re-apply.

Built with **full RELRO** (`-Wl,-z,now -Wl,-z,relro`) — confirmed
`BIND_NOW` + `GNU_RELRO` in the binary — so the GOT page is read-only and
the `write_got_slot` mprotect (RW → write → restore) path is exercised, not
bypassed.

## Result — PASS
```
patch apply #1: 1 slot(s), 4 module(s) scanned
patch apply #2 (idempotency): 0 slot(s)
fake_puts hits total = 2          (the two post-patch puts; baseline missed)
GOT_PATCH_TEST: PASS (baseline-miss, patch-hit, idempotent, durable)
```
Clean build, no warnings. Full run: `test_run.log`.

## What this proves / does not prove
- **Proves:** the patcher correctly locates a GOT slot via `dl_iterate_phdr`
  + RELA parse, handles RELRO via `mprotect`, patches the call site, and is
  idempotent — on a real, full-RELRO binary.
- **Does not yet prove:** interception of `cublasGemmEx` / ATen SDPA inside
  a live PyTorch process — that is gate criterion (a), measured after the
  cuBLAS shim + attn substrate are re-targeted onto the patcher (build STEP
  items 2–3) and libcipher_rt is rebuilt.

## Artifacts
| file | md5 |
|---|---|
| `cipher_rt_phase4/cipher_rt_got_patch.h` | `ff87a54a…` |
| `cipher_rt_phase4/cipher_rt_got_patch.c` | `0c4251b7…` |
| `got_patch_test.c` | `2f918161…` |
| `test_run.log` | run output |
