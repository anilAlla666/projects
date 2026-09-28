# Track 2 SC3-2 — REGRESSION SMOKE TEST

**Date:** 2026-05-19. **Under test:** cipher_kv_bridge `c04b0c39` (SC3) +
the unchanged Track 3 substrate (kmod `285d102e`, libcipher_rt `83afd1ca`).
**Reference:** `TRACK_2_SC3_PRE_BASELINE.md`. **Verdict: SMOKE TEST PASS.**

| gate | pre-SC3 | SC3 `c04b0c39` | verdict |
|---|---|---|---|
| **W3** — Track 2 SC2 weight arena (`sc2_verify.py`) — *the cipher_kv_bridge-dependent gate* | `SC2_PASS`, KL 0.0 | **`SC2_PASS=True`, KL_max 0.0** | **PASS** |
| isolation — `cp54_isolation_test` | 15/15 | **15/15 PASS** | PASS |
| W1 — Phase B N=8 (`run_arm3.sh`) | band [468, 494] | **503.2 tok/s** | PASS (see note) |
| dmesg | clean | **clean** | PASS |

## Notes

- **W3 is the SC3-meaningful gate.** `sc2_verify.py` loads `cipher_kv_bridge`
  — the artifact SC3 rotated. It still passes the full SC2 producer-arena
  chain (`SC2_PASS=True`, KL 0.0) — the SC3 consumer-import additions did not
  regress the SC2 producer path. SC3's changes are purely additive
  (`weight_arena_import`, `WeightArena.view`); `weight_arena_create` /
  `.alloc` / `.export_fd` are untouched.
- **W1 = 503.2 tok/s** is **above** the pre-SC3 3-rep band [468, 494] — but
  W1 (`run_arm3.sh` → `cipher_batch_executor_gen.py`) runs on **libcipher_rt
  + kmod and does NOT load `cipher_kv_bridge`**. SC3 cannot affect the W1 path
  by construction. 503.2 simply widens W1's measured intrinsic-variance band
  (now [≈468, ≈503], ~±7 % — the pre-baseline already flagged this exceeds the
  ±3 % gate and is not substrate drift). Not a regression.
- isolation 15/15 confirms kmod `285d102e` intact; dmesg clean across the
  build, integration runs, and smoke.

## Verdict — SMOKE TEST PASS

The SC3 `cipher_kv_bridge` change introduced no regression: W3 (the
cipher_kv_bridge gate) reproduces `SC2_PASS`/KL 0.0; the Track 3 substrate
(kmod + libcipher_rt) is untouched and intact. SC3 rotates the
`cipher_kv_bridge` anchor `fca6843d` → `c04b0c39`.

## Rollback path (not needed — smoke PASSED; documented per discipline)

```
cp /home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.so.pre_track2_sc3 \
   /home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so
```
(`fca6843d`). kmod + libcipher_rt unaffected. Re-run W3 to confirm.
