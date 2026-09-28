# Track 3 SC3-2 — BUILD LOG

**Date:** 2026-05-19. **Scope:** tenant-side migration — libcipher_rt
`cipher_rt_green_ctx_migrate()` primitive + `cipher_migrate.py` handler +
`CIPHER_MIGRATABLE` opt-in plumbing.

## Reachability gate (run first, before the handler design was committed)

`cp54_reachability_test.py` — **PASS.** `ctypes.CDLL(None)` does NOT reach the
injected libcipher_rt (the CUDA driver loads the injection lib RTLD_LOCAL —
`undefined symbol`). `ctypes.CDLL(path, RTLD_NOLOAD)` DOES — `is_initialized=1,
sm_count=16`, address `0x7b04dd292e80`. **The handler uses `RTLD_NOLOAD`**
(the design memo §3 listed `CDLL(None)` first; empirically it is the wrong
one). No fallback path needed. Evidence: `sc3_reachability_result.json`.

## Source changes (libcipher_rt — 2 files)

| file | change |
|---|---|
| `cipher_rt_green_ctx.h` | +`cipher_rt_green_ctx_migrate(unsigned)` prototype; +`cipher_rt_green_ctx_cur_mask(void)` |
| `cipher_rt_green_ctx.c` | +`g_green_cur_mask` static (set by `ensure()`); +`cipher_rt_green_ctx_migrate()` (drain → build → L1 verify → swap → release); +`cipher_rt_green_ctx_cur_mask()`; `CIPHER_MIGRATABLE` env → `SUBSCRIBE_MIGRATE(1)` in `cipher_rt_green_ctx_cp54_init()` |

`cipher_rt_green_ctx_ensure()` (the all-pinned regression path) is left
**byte-identical** apart from one additive line setting `g_green_cur_mask` —
the migrate build steps are duplicated rather than refactored out of `ensure()`,
so the W1/W2/W3 smoke path is structurally unaffected. kmod, libcipher_v2,
cipher_kv_bridge — untouched.

New non-source artifact: `cipher_migrate.py` — the two-layer poll-and-migrate
handler (poll + START/ACK ioctls + L2 `%smid` verify; binds the migrate
primitive via `RTLD_NOLOAD`).

## SC3 test hook — `CIPHER_SC3_FAULT`

`cipher_rt_green_ctx_migrate()` reads `getenv("CIPHER_SC3_FAULT")`: `verify`
forces an L1 verify failure, `destroy` forces a post-swap destroy failure.
Env-gated — unset in production → zero effect. Used by the verify_fault /
destroy_fault unit tests. Kept (also serves SC5 fault injection).

## Build

```
make   (cc -O2 -Wall -Wextra -fPIC, g++ -shared)
```
**Clean** — no warning or error from `cipher_rt_green_ctx.c`. Built
`libcipher_rt.so` md5 **`83afd1ca4118dc651854ef751e4ef82d`** (SC3 anchor
`83afd1ca`). `nm -D` confirms `cipher_rt_green_ctx_migrate` and
`cipher_rt_green_ctx_cur_mask` exported.

## ABI / API note

The header declares `cipher_rt_green_ctx_migrate(unsigned new_mask)` — the
design memo wrote `uint32_t`; `unsigned` is the file's idiom and is 32-bit on
this platform (the Python handler passes `ctypes.c_uint`). Equivalent.

## Verification

Reachability gate PASS; SC3 unit + integration tests 7/7 PASS
(`TRACK_3_SC3_UNIT_TESTS.md`); regression smoke W1/W2/W3 PASS
(`TRACK_3_SC3_REGRESSION.md`); dmesg clean.
