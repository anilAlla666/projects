# Track 2 SC5-2 — UNIT TESTS

**Date:** 2026-05-19. Tool: `sc5_unit.py`. **Verdict: 15 / 15 PASS.**

Exercises the 4 kmod ioctls (REGISTER / IMPORT / LEAVE / QUERY) at the
`/dev/cipher` boundary — no CUDA, no model. The arena registry is pure
kernel fd-table work, so a plain `/dev/null` fd stands in for the exported
VMM POSIX fd (the kmod `fget`s whatever fd it is handed; it never interprets
it as a CUDA handle). Result JSON: `sc5_unit_result.json`.

| # | test | asserts | result |
|---|---|---|---|
| T1 | `query_ok` | QUERY succeeds on a (possibly empty) registry | PASS |
| T2 | `register_returns_id` | REGISTER returns `arena_id > 0` | PASS |
| T2 | `query_shows_arena` | QUERY lists the new arena | PASS |
| T2 | `producer_is_us` | `producer_pid == getpid()` | PASS |
| T2 | `zero_consumers_initially` | `n_consumers == 0` at register time | PASS |
| T3 | `import_base_matches` | IMPORT returns the registered `base` | PASS |
| T3 | `import_size_matches` | IMPORT returns the registered `size` | PASS |
| T3 | `import_blob_roundtrip` | IMPORT returns the exact registered blob | PASS |
| T3 | `import_fd_valid` | IMPORT installs a valid fd in the caller | PASS |
| T3 | `consumer_count_incremented` | `n_consumers == 1` after one IMPORT | PASS |
| T4 | `import_missing_enoent` | IMPORT of a bad arena id → `-ENOENT` | PASS |
| T5 | `multi_consumer_count` | 4 IMPORTs → `n_consumers == 4` (multi-consumer) | PASS |
| T6 | `import_small_buf_enospc` | IMPORT with `blob_cap < blob_len` → `-ENOSPC` | PASS |
| T7 | `oversized_blob_rejected` | blob `> CIPHER_WA_BLOB_MAX` rejected client-side | PASS |
| T8 | `leave_releases_sole_arena` | sole producer LEAVEs → count 0 → slot freed → QUERY no longer lists it | PASS |

## What each group proves

- **T1–T3** — the REGISTER → QUERY → IMPORT happy path, including the opaque
  blob round-tripping verbatim (the kmod stores bytes, never interprets —
  design memo §f) and `base`/`size` carried intact.
- **T4** — a missing arena id is a clean `-ENOENT`, not a crash or a stale
  read.
- **T5** — the registry is genuinely multi-consumer (SC3/SC4 were
  single-consumer); the consumer pid set tracks every IMPORT.
- **T6** — a caller buffer too small for the stored blob fails cleanly
  (`-ENOSPC`) **before** any fd is installed — the reserve-fd / copy-out /
  `fd_install` ordering means the aborted import leaks nothing.
- **T7** — the `CIPHER_WA_BLOB_MAX` (64 KiB) cap is enforced.
- **T8** — LEAVE drives the participant count to 0 and the kmod releases the
  slot (`fput` + `in_use = 0`) — the clean-exit fast path of the lifetime
  rule (design memo §b).

dmesg across the unit run: clean (REGISTER / IMPORT / reaped info lines
only — no warning, no oops).
