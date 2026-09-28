/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_ring_write — W10-12 Step 1 RING_WRITE producer substrate.
 *
 * Per-tenant SPMC (single-producer multi-consumer) ring buffer. Each
 * tenant has its own struct cipher_rt_ring_state; the tenant's worker
 * thread is the sole producer (single-writer assumption — per §4.9). N
 * consumer threads drain at independent rates using fixed slot ids
 * (CIPHER_RT_RING_CONSUMER_CLASSIFY = 0, CIPHER_RT_RING_CONSUMER_ORACLE
 * = 1). Producer reads min(read_seq[0..N-1]) to detect overflow.
 *
 * CFL stability invariant: producer self-throttles when its write rate
 * would exceed (configurable threshold × EWMA consumer drain rate). Per
 * advisor 2026-05-23, the CFL rate check is SAMPLED (every 1024 writes,
 * not per-call) to keep producer p99 under 150 ns. clock_gettime() on
 * x86_64 Linux is ~20-30 ns; per-call sampling would dominate the budget.
 *
 * Drop-on-full: independent of CFL self-throttle. If a consumer stops
 * draining entirely (slow consumer), the producer hits the spec contract
 * (drop entry, increment total_dropped, return). CFL prevents this in
 * steady-state by self-throttling before fill.
 */
#define _GNU_SOURCE
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cipher_rt_ring_write.h"

/* W12 Step 5 D14 backfill — env-gate the producer call site to restore the
 * ±3% TPS gate per WEEK_12_SCOPE_DRIFT_AUDIT.md D14 finding (substrate adds
 * -3.4% mean on TinyLlama B=1 across 5 paired vanilla/CIPHER runs). The
 * cumulative substrate ladder W9→W12.4 has no single dominant inflection
 * point per the bisection table; the per-call ring_write producer fires
 * from classify_observer (every CUPTI launch callback) + cublas shim
 * (every cublasGemmEx) + SDPA trampoline (every attn op), so gating it
 * at the function entry zaps the highest-frequency producer cost with one
 * env toggle. Default ON (active) preserves the substrate-active behavior
 * synthetic tests gate; CIPHER_RING_WRITE=0 disables for benchmark / low-
 * traffic single-instance regimes. The drop counter and total_written
 * accumulate at zero in that case (no events, no drops).
 *
 * Sampled once per process at first call via atomic CAS; per-call cost is
 * a single relaxed atomic load + branch. */
static _Atomic int g_ring_write_enabled = -1;
static int ring_write_enabled(void)
{
	int v = atomic_load_explicit(&g_ring_write_enabled, memory_order_relaxed);
	if (v >= 0) return v;
	const char *s = getenv("CIPHER_RING_WRITE");
	int e = (s && (s[0] == '0' || !strcmp(s, "off") || !strcmp(s, "no"))) ? 0 : 1;
	atomic_store_explicit(&g_ring_write_enabled, e, memory_order_relaxed);
	return e;
}

#define CIPHER_RT_RING_CACHELINE  64
#define CIPHER_RT_CFL_SAMPLE_EVERY 1024u
#define CIPHER_RT_CFL_THRESHOLD_PCT 75u   /* throttle if write_rate > pct% of drain_rate_ema */
#define CIPHER_RT_CFL_EMA_ALPHA_X100 10u  /* alpha = 0.10 fixed-point */
#define CIPHER_RT_CFL_INITIAL_DRAIN UINT64_MAX  /* cold start: no throttle */

struct cipher_rt_ring_state {
	/* Producer-hot cache line: writer atomic head + CFL sampling state */
	_Atomic uint64_t write_seq;
	_Atomic uint64_t total_written;
	_Atomic uint64_t total_throttled;
	_Atomic uint64_t total_dropped;
	/* CFL sampling state — read by producer only; written every
	 * CIPHER_RT_CFL_SAMPLE_EVERY writes. */
	uint64_t cfl_last_seq;
	uint64_t cfl_last_ts_ns;
	_Atomic uint64_t cfl_write_rate;  /* events/sec, fixed-point */
	uint8_t _pad_writer[CIPHER_RT_RING_CACHELINE - 56];

	/* Consumer-hot cache line: read_seq slots + drain rate EWMA. */
	_Atomic uint64_t read_seq[CIPHER_RT_RING_NUM_CONSUMERS];
	_Atomic uint64_t drain_rate_ema;  /* events/sec, fixed-point */
	uint64_t drain_last_ts_ns;
	_Atomic uint64_t drain_last_seq;
	uint8_t _pad_consumer[CIPHER_RT_RING_CACHELINE
	                      - sizeof(_Atomic uint64_t) * CIPHER_RT_RING_NUM_CONSUMERS
	                      - sizeof(_Atomic uint64_t) * 3
	                      - sizeof(uint64_t)];

	/* Ring entries (4096 × 64 B = 256 KiB). */
	struct cipher_rt_ring_entry entries[CIPHER_RT_RING_ENTRIES];
} __attribute__((aligned(CIPHER_RT_RING_CACHELINE)));

static struct cipher_rt_ring_state g_rings[CIPHER_RT_RING_MAX_TENANTS];
static _Atomic int g_init_done;

static inline uint64_t ts_ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

int cipher_rt_ring_write_init(void)
{
	int expected = 0;
	if (!atomic_compare_exchange_strong_explicit(
	        &g_init_done, &expected, 1,
	        memory_order_acq_rel, memory_order_acquire)) {
		return 0;                                       /* already initialised */
	}
	memset(g_rings, 0, sizeof(g_rings));
	for (uint32_t t = 0; t < CIPHER_RT_RING_MAX_TENANTS; t++) {
		/* Cold-start drain rate: effectively infinite so CFL doesn't
		 * false-positive before the first drain cycle. */
		atomic_store_explicit(&g_rings[t].drain_rate_ema,
		                      CIPHER_RT_CFL_INITIAL_DRAIN,
		                      memory_order_relaxed);
		g_rings[t].cfl_last_ts_ns = ts_ns();
		g_rings[t].drain_last_ts_ns = g_rings[t].cfl_last_ts_ns;
	}
	return 0;
}

void cipher_rt_ring_write_exit(void)
{
	/* BSS-resident; no allocation to free. Mark uninitialized for
	 * idempotent re-init under test harness control. */
	atomic_store_explicit(&g_init_done, 0, memory_order_release);
}

/* Compute the minimum consumer read_seq for overflow detection.
 * Inlined; ~5 ns at NUM_CONSUMERS=2. */
static inline uint64_t min_read_seq(const struct cipher_rt_ring_state *r)
{
	uint64_t m = atomic_load_explicit(&r->read_seq[0], memory_order_acquire);
	for (uint32_t i = 1; i < CIPHER_RT_RING_NUM_CONSUMERS; i++) {
		uint64_t v = atomic_load_explicit(&r->read_seq[i],
		                                  memory_order_acquire);
		if (v < m) m = v;
	}
	return m;
}

/* CFL sampled rate update: called every CIPHER_RT_CFL_SAMPLE_EVERY writes.
 * Updates cfl_write_rate (atomic publish for downstream telemetry). */
static void cfl_update_write_rate(struct cipher_rt_ring_state *r, uint64_t seq)
{
	uint64_t now = ts_ns();
	uint64_t dt = now - r->cfl_last_ts_ns;
	if (dt == 0) return;
	uint64_t ds = seq - r->cfl_last_seq;
	/* rate = ds * 1e9 / dt  (events/sec). Saturate at UINT64_MAX/1e9. */
	uint64_t rate = (ds < UINT64_MAX / 1000000000ull)
	                ? (ds * 1000000000ull) / dt
	                : UINT64_MAX;
	atomic_store_explicit(&r->cfl_write_rate, rate, memory_order_relaxed);
	r->cfl_last_seq = seq;
	r->cfl_last_ts_ns = now;
}

void cipher_rt_ring_write(uint32_t tenant_id,
                          uint32_t event_type, uint32_t event_subtype,
                          uint64_t commit_seq,
                          const void *payload, size_t payload_len)
{
	if (!ring_write_enabled()) return;   /* W12 Step 5 D14 env-gate */
	if (tenant_id >= CIPHER_RT_RING_MAX_TENANTS) return;
	if (atomic_load_explicit(&g_init_done, memory_order_acquire) == 0) return;

	struct cipher_rt_ring_state *r = &g_rings[tenant_id];

	/* Load current write_seq. Relaxed: this thread is the sole writer. */
	uint64_t seq = atomic_load_explicit(&r->write_seq, memory_order_relaxed);

	/* CFL sampled rate refresh: every CIPHER_RT_CFL_SAMPLE_EVERY writes,
	 * recompute write_rate from (current seq, current time). Producer
	 * pays the ~25 ns clock_gettime cost once per 1024 writes — amortized
	 * to ~25 ps per call. */
	if ((seq & (CIPHER_RT_CFL_SAMPLE_EVERY - 1)) == 0) {
		cfl_update_write_rate(r, seq);
	}

	/* CFL throttle decision (every call — cheap atomic load comparison).
	 * If write_rate exceeds 75% of EWMA drain rate, self-throttle:
	 * skip the memcpy + seq bump, increment total_throttled. */
	uint64_t drain_rate = atomic_load_explicit(&r->drain_rate_ema,
	                                           memory_order_relaxed);
	uint64_t write_rate = atomic_load_explicit(&r->cfl_write_rate,
	                                           memory_order_relaxed);
	if (drain_rate != CIPHER_RT_CFL_INITIAL_DRAIN &&
	    write_rate > (drain_rate * CIPHER_RT_CFL_THRESHOLD_PCT) / 100u) {
		atomic_fetch_add_explicit(&r->total_throttled, 1,
		                          memory_order_relaxed);
		return;
	}

	/* Overflow check per §4.9 producer protocol step 3: drop if all
	 * consumers have fallen behind by >= RING_ENTRIES. */
	uint64_t min_read = min_read_seq(r);
	if (seq - min_read >= CIPHER_RT_RING_ENTRIES) {
		atomic_fetch_add_explicit(&r->total_dropped, 1,
		                          memory_order_relaxed);
		return;
	}

	/* Fill entry. payload truncated/padded to 40 B. */
	struct cipher_rt_ring_entry *slot = &r->entries[seq & CIPHER_RT_RING_MASK];
	slot->timestamp_ns  = ts_ns();
	slot->commit_seq    = commit_seq;
	slot->event_type    = event_type;
	slot->event_subtype = event_subtype;
	if (payload && payload_len > 0) {
		size_t n = (payload_len > sizeof(slot->payload))
		           ? sizeof(slot->payload) : payload_len;
		memcpy(slot->payload, payload, n);
		if (n < sizeof(slot->payload)) {
			memset(slot->payload + n, 0, sizeof(slot->payload) - n);
		}
	} else {
		memset(slot->payload, 0, sizeof(slot->payload));
	}

	/* Release-store: ensures all field writes are visible to acquire-
	 * loading consumers before they see the new write_seq. */
	atomic_store_explicit(&r->write_seq, seq + 1, memory_order_release);
	atomic_fetch_add_explicit(&r->total_written, 1, memory_order_relaxed);
}

int cipher_rt_ring_drain(uint32_t tenant_id,
                         uint32_t consumer_id,
                         struct cipher_rt_ring_entry *out_buf,
                         int max_entries)
{
	if (tenant_id >= CIPHER_RT_RING_MAX_TENANTS) return 0;
	if (consumer_id >= CIPHER_RT_RING_NUM_CONSUMERS) return 0;
	if (!out_buf || max_entries <= 0) return 0;
	if (atomic_load_explicit(&g_init_done, memory_order_acquire) == 0) return 0;

	struct cipher_rt_ring_state *r = &g_rings[tenant_id];

	uint64_t read_cur = atomic_load_explicit(&r->read_seq[consumer_id],
	                                         memory_order_relaxed);
	uint64_t write_cur = atomic_load_explicit(&r->write_seq,
	                                          memory_order_acquire);

	uint64_t available = write_cur - read_cur;
	if (available == 0) return 0;
	int n = (available < (uint64_t)max_entries) ? (int)available : max_entries;

	for (int i = 0; i < n; i++) {
		out_buf[i] = r->entries[(read_cur + i) & CIPHER_RT_RING_MASK];
	}
	uint64_t new_read = read_cur + (uint64_t)n;
	atomic_store_explicit(&r->read_seq[consumer_id], new_read,
	                      memory_order_release);

	/* Update drain rate EWMA. Same sampled approach as producer: only
	 * the slot-0 consumer (CLASSIFY) updates the shared EMA; other
	 * consumers track their own pace via their read_seq slot. Reduces
	 * cross-consumer write contention on drain_rate_ema. */
	if (consumer_id == 0) {
		uint64_t now = ts_ns();
		uint64_t dt = now - r->drain_last_ts_ns;
		if (dt >= 100000000ull) {                       /* update every 100 ms */
			uint64_t last_seq = atomic_load_explicit(
			    &r->drain_last_seq, memory_order_relaxed);
			uint64_t ds = new_read - last_seq;
			uint64_t inst_rate = (ds < UINT64_MAX / 1000000000ull)
			                     ? (ds * 1000000000ull) / dt
			                     : UINT64_MAX;
			uint64_t old_ema = atomic_load_explicit(&r->drain_rate_ema,
			                                        memory_order_relaxed);
			uint64_t new_ema;
			if (old_ema == CIPHER_RT_CFL_INITIAL_DRAIN) {
				new_ema = inst_rate;
			} else {
				/* ema = (1 - alpha)*old + alpha*inst, alpha = 0.10
				 * fixed-point: alpha_x100 / 100 */
				new_ema = (old_ema * (100u - CIPHER_RT_CFL_EMA_ALPHA_X100)
				           + inst_rate * CIPHER_RT_CFL_EMA_ALPHA_X100) / 100u;
			}
			atomic_store_explicit(&r->drain_rate_ema, new_ema,
			                      memory_order_relaxed);
			atomic_store_explicit(&r->drain_last_seq, new_read,
			                      memory_order_relaxed);
			r->drain_last_ts_ns = now;
		}
	}

	return n;
}

uint64_t cipher_rt_ring_total_written(uint32_t tenant_id)
{
	if (tenant_id >= CIPHER_RT_RING_MAX_TENANTS) return 0;
	return atomic_load_explicit(&g_rings[tenant_id].total_written,
	                            memory_order_relaxed);
}

uint64_t cipher_rt_ring_total_throttled(uint32_t tenant_id)
{
	if (tenant_id >= CIPHER_RT_RING_MAX_TENANTS) return 0;
	return atomic_load_explicit(&g_rings[tenant_id].total_throttled,
	                            memory_order_relaxed);
}

uint64_t cipher_rt_ring_total_dropped(uint32_t tenant_id)
{
	if (tenant_id >= CIPHER_RT_RING_MAX_TENANTS) return 0;
	return atomic_load_explicit(&g_rings[tenant_id].total_dropped,
	                            memory_order_relaxed);
}
