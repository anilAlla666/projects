# Track 3 SC6-2 — HARNESS BUILD LOG

**Date:** 2026-05-19. **Scope:** test-harness refinement only — **no substrate
source change, no anchor rotation** (kmod `285d102e`, libcipher_rt `83afd1ca`
unchanged throughout SC6).

## Substrate sanity (SC6 start)

kmod `285d102e` + libcipher_rt `83afd1ca` confirmed md5-identical to the SC5
substrate; isolation 15/15 PASS live; `/proc/cipher/migrations` healthy.
W1/W2/W3 + SC3 + SC5 sweep were verified on this exact md5-identical substrate
at SC5 — re-running them adds nothing since nothing has touched the substrate.

## Harness files

| file | role |
|---|---|
| `sc6_tenant.py` | barriered PARTITION decode tenant — per-round decode latency, TFGATE KL, migrate handler, per-round `%smid` probe (clause-2 input). Round count encodes the free order. |
| `sc6_churn.py` | orchestrator — 5 barriered tenants (3 migratable / 2 pinned), barrier coordinator, non-forced compaction trigger, per-class latency + clause-2 + cost. |
| `sc6_quiet_period.py` | PROPOSED→ABORT measurement (quiesced vs churning). |
| `cipher_migrate.py` | `step()` extended to time the primitive and L2 probe. |

## Build findings surfaced (honest scope corrections)

1. **POOL dropped.** The SC6 design-memo §1 specified a POOL process to fix
   F2's stranding-metric artifact. Implementing it revealed: a POOL claims the
   maximal contiguous low prefix and so **fills the 15-group device** when
   partitions are present — leaving **zero free-group headroom** for a
   count-preserving migration to target. With a POOL, no migration can fire
   (validation: 0 migrations). The POOL was justified only for re-measuring
   *stranding* — and SC6 does **not** re-measure A-vs-B stranding (F1 is
   adjudicated-accepted from SC5). The POOL serves no SC6 measurement. SC6 uses
   the SC5-style headroom topology (5 partitions, no POOL). Surfaced.

2. **Cost-metric corrected.** First validation computed migration cost as
   (migrated-round decode-mean − non-migrated p50) = ~22 ms — **wrong**: the
   migration runs in `handler.step()`, *outside* the timed decode loop, so that
   delta was decode-round variance, not migration cost. Fixed:
   `cipher_migrate.MigrateHandler.step()` now brackets the primitive call and
   the L2 probe with `perf_counter` directly. The real numbers are in
   `TRACK_3_SC6_MEASUREMENTS.md`.

3. **Migratable tenants launched first** (→ low groups) so the kmod's
   lowest-blocker eval proposes to a *migratable* partition; pinned tenants
   launched last (→ high groups). Without this, migrations never fire.

4. **Quiet-period inter-phase cleanup.** `sc6_quiet_period.py` runs two phases
   in one process; a self-`FREE` between them is required to clear the test
   pid's metadata (otherwise the churning phase reads the quiesced phase's
   stale `last_outcome`). The post-quiet trigger-`FREE` is issued from a
   separate thread (distinct LWP) so it does not release the test's own
   partition. Both fixed; both phases now measure cleanly.

## Verification

Within-A sweep 5/5 seeds + quiet-period — all PASS, dmesg clean
(`TRACK_3_SC6_MEASUREMENTS.md`).
