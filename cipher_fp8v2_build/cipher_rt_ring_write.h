/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_ring_write — W10-12 Step 1: RING_WRITE producer substrate.
 *
 * Per CIPHER_REENGINEERING_PLAN.md v1.2.3 §4.9 (line 921+), RING_WRITE is
 * a lock-free per-tenant SPMC ring delivering per-launch telemetry events
 * to userspace consumers (CLASSIFY, ORACLE, REMEMBER — W13-14, KV-DEDUP +
 * MARLIN + TC + L2 — W11+W12). AUDIT (G6) is kmod-resident via NR 28 and
 * does NOT consume RING_WRITE.
 *
 * Layout deviation from §4.9: spec says 65,536 entries/tenant; this build
 * uses 4096 entries × 64 B = 256 KiB/tenant. CFL invariant prevents
 * sustained overflow; 4096 is sufficient buffer for typical drain
 * cadences. Documented in WEEK_10_STEP_1_RING_WRITE.md §3.
 *
 * Producer return type is void (no -EAGAIN to caller): the launch has
 * already happened by the time RING_WRITE fires; the caller cannot retry.
 * CFL self-throttle and spec-mandated drop-on-full are internal bookkeeping
 * (incrementing total_throttled / total_dropped respectively); same
 * observable outcome from the caller's view: telemetry was emitted, some
 * fraction got reclassified internally.
 */
#ifndef CIPHER_RT_RING_WRITE_H
#define CIPHER_RT_RING_WRITE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CIPHER_RT_RING_ENTRY_BYTES 64                          /* single cache line */
#define CIPHER_RT_RING_ENTRIES     4096                        /* per tenant; 256 KiB/tenant */
#define CIPHER_RT_RING_MASK        (CIPHER_RT_RING_ENTRIES - 1)
#define CIPHER_RT_RING_MAX_TENANTS 128                         /* matches W6 G1 cap + W7-9 Step 4 snapshot */

/* RING_WRITE entry. Layout fixed; sizeof must == 64. */
struct cipher_rt_ring_entry {
	uint64_t timestamp_ns;    /* CLOCK_MONOTONIC at producer site */
	uint64_t commit_seq;      /* matches the COMMIT snapshot seq from W7-9 Step 2 */
	uint32_t event_type;      /* enum cipher_rt_ring_event */
	uint32_t event_subtype;   /* event-specific tag */
	uint8_t  payload[40];     /* event-specific bytes; opaque to ring */
};

/* Event type enum. Additive — new consumers append new types. */
enum cipher_rt_ring_event {
	CIPHER_RT_RING_EVENT_NONE       = 0,
	CIPHER_RT_RING_EVENT_CLASSIFY   = 1,  /* W10 Step 1 — current */
	CIPHER_RT_RING_EVENT_ORACLE     = 2,  /* W10 Step 1 — current */
	/* AUDIT events bypass the ring (G6 ioctl NR 28) — no enum value */
	CIPHER_RT_RING_EVENT_REMEMBER   = 3,  /* W13-14 reservation (Koopman tier) */
	CIPHER_RT_RING_EVENT_KV_DEDUP   = 4,  /* W11 G3 skip counter reservation */
	CIPHER_RT_RING_EVENT_MARLIN     = 5,  /* W11 G4 weight kit reservation */
	CIPHER_RT_RING_EVENT_TC_PROBE   = 6,  /* W11 TC saturation reservation */
	CIPHER_RT_RING_EVENT_L2_PERSIST = 7,  /* W12 L2 policy reservation */
};

/* Consumer slot id. Fixed slots — matches may13's read_seq_s1/s2 layout.
 * Producer iterates these to compute min-read for overflow detection.
 *
 * W14 Step 3 S3.B1: REMEMBER slot 2 added; NUM_CONSUMERS bumped 2 → 3.
 * The REMEMBER consumer (cipher_rt_remember_consumer.c, env-gated
 * CIPHER_REMEMBER=1) cold-starts read_seq[2] to current write_seq at
 * thread start so the producer does not back-pressure on the consumer's
 * cold cursor. When the consumer is env-gated off, read_seq[2] stays at
 * 0 and the producer's min_read_seq pessimistically caps at 0 — same
 * observable behavior as CLASSIFY/ORACLE slots already exhibit in v1
 * (no production drain caller; drop floor = 4096 entries per tenant). */
enum cipher_rt_ring_consumer {
	CIPHER_RT_RING_CONSUMER_CLASSIFY = 0,
	CIPHER_RT_RING_CONSUMER_ORACLE   = 1,
	CIPHER_RT_RING_CONSUMER_REMEMBER = 2,    /* W14 Step 3 S3.B1 */
	CIPHER_RT_RING_NUM_CONSUMERS     = 3,
};

/* Lifecycle. Called from cipher_inject.c init body. */
int  cipher_rt_ring_write_init(void);
void cipher_rt_ring_write_exit(void);

/* Producer. Lock-free, single-writer-per-tenant assumption (the tenant's
 * worker thread is the sole producer for its slot). Always succeeds from
 * the caller's view; CFL self-throttle and overflow are tracked internally.
 *
 * Out-of-range tenant_id is a silent no-op. */
void cipher_rt_ring_write(uint32_t tenant_id,
                          uint32_t event_type, uint32_t event_subtype,
                          uint64_t commit_seq,
                          const void *payload, size_t payload_len);

/* Consumer drain. Drains up to max_entries from tenant_id's ring into
 * out_buf, starting from this consumer's tracked cursor. Returns number
 * drained. consumer_id selects which read_seq slot to update (fixed slot
 * per consumer to let the producer compute min-read for overflow).
 *
 * Each consumer's slot tracks its own cursor; multiple consumers drain
 * the same ring independently at their own pace. */
int cipher_rt_ring_drain(uint32_t tenant_id,
                         uint32_t consumer_id,
                         struct cipher_rt_ring_entry *out_buf,
                         int max_entries);

/* Telemetry accessors for /proc/cipher/stats and step doc. */
uint64_t cipher_rt_ring_total_written(uint32_t tenant_id);
uint64_t cipher_rt_ring_total_throttled(uint32_t tenant_id);
uint64_t cipher_rt_ring_total_dropped(uint32_t tenant_id);

/* Compile-time invariants. */
#ifndef __cplusplus
_Static_assert(sizeof(struct cipher_rt_ring_entry) == CIPHER_RT_RING_ENTRY_BYTES,
               "cipher_rt_ring_entry must be exactly 64 B");
_Static_assert((CIPHER_RT_RING_ENTRIES & CIPHER_RT_RING_MASK) == 0,
               "CIPHER_RT_RING_ENTRIES must be power of 2");
#endif

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_RING_WRITE_H */
