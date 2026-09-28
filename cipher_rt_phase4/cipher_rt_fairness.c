/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_fairness — D.8 FAIRNESS + SHIELD, libcipher_rt side.
 * See cipher_rt_fairness.h for the design. Timing-only enforcement; the kmod
 * ledger (NR 32, RW mmap window) is the cross-tenant channel. Ports the
 * cipher_fairness_shm.cpp should_yield rate-vs-average logic, repointing its
 * data source from /dev/shm (Docker-isolated per container) to /dev/cipher
 * (shared across all CDI containers, as the W7-9 resolver proved).
 */
#include "cipher_rt_fairness.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>

/* ---- Mirror of the kmod ABI (cipher_fairness_ledger.h / cipher_ioctl.h NR 32).
 * The slot layout MUST match struct cipher_fair_slot byte-for-byte (64 B). ---- */
#define CIPHER_FAIR_SLOTS_U        256u
#define CIPHER_FAIR_SLOT_BYTES_U   64u
#define CIPHER_FAIR_REGION_BYTES_U (CIPHER_FAIR_SLOTS_U * CIPHER_FAIR_SLOT_BYTES_U)
#define CIPHER_FAIR_VIEW_MMAP_PGOFF_U 0x200000u
#define CIPHER_FAIR_PAGE_SIZE_U    4096u

struct fair_slot_u {
	uint64_t key;            /* tgid; 0 = empty */
	uint64_t gemm_calls;     /* self-reported */
	uint64_t last_active_ns; /* CLOCK_MONOTONIC ns */
	uint32_t band;
	uint32_t yield_count;
	uint64_t _pad[4];
};

struct fair_register_u {
	uint32_t tgid;
	uint32_t band;
	uint32_t out_slot;
	uint32_t flags;
};

/* _IOWR('C', 32, struct fair_register_u) — encode manually (no kernel headers). */
#define CIPHER_FAIR_MAGIC_U 'C'
#define CIPHER_FAIR_NRBITS  8
#define CIPHER_FAIR_TYBITS  8
#define CIPHER_FAIR_SZBITS  14
#define CIPHER_FAIR_NRSH    0
#define CIPHER_FAIR_TYSH    (CIPHER_FAIR_NRSH + CIPHER_FAIR_NRBITS)
#define CIPHER_FAIR_SZSH    (CIPHER_FAIR_TYSH + CIPHER_FAIR_TYBITS)
#define CIPHER_FAIR_DIRSH   (CIPHER_FAIR_SZSH + CIPHER_FAIR_SZBITS)
#define CIPHER_FAIR_DIR_RW  3u   /* _IOC_READ|_IOC_WRITE */
#define CIPHER_FAIRNESS_REGISTER_U \
	(((unsigned)CIPHER_FAIR_DIR_RW << CIPHER_FAIR_DIRSH) | \
	 ((unsigned)CIPHER_FAIR_MAGIC_U << CIPHER_FAIR_TYSH) | \
	 ((unsigned)32 << CIPHER_FAIR_NRSH) | \
	 ((unsigned)sizeof(struct fair_register_u) << CIPHER_FAIR_SZSH))

/* ---- tunables (env-overridable) ---- */
#define FAIR_IDLE_THRESH_NS   (2ULL * 1000 * 1000 * 1000)  /* 2 s */
#define FAIR_DEFAULT_BURST_MIN 1024ULL                     /* min calls before burst applies */
#define FAIR_DEFAULT_THROTTLE_US 100u                      /* CPU sleep per yield */
#define FAIR_EVAL_EVERY        32u                          /* sample the decision every Nth call */

/* ---- state ---- */
static atomic_int      g_armed       = 0;   /* -1 = tried+failed, 0 = untried, 1 = armed */
static atomic_int      g_init_done   = 0;
static struct fair_slot_u *g_region  = NULL;
static int             g_my_slot     = -1;
static uint32_t        g_my_tgid     = 0;
static uint32_t        g_my_band     = 0;
static uint64_t        g_burst_min   = FAIR_DEFAULT_BURST_MIN;
static uint32_t        g_throttle_us = FAIR_DEFAULT_THROTTLE_US;
static int             g_force_yield = 0;   /* test hook: force the throttle to fire (KL gate) */
static atomic_ullong   g_self_calls  = 0;
static atomic_ullong   g_self_yields = 0;
static _Thread_local uint32_t t_eval_ctr = 0;
static _Thread_local int      t_cached_yield = 0;

static inline uint64_t fair_now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int env_on(const char *v)
{
	return v && (!strcmp(v, "1") || !strcmp(v, "on") || !strcmp(v, "ON"));
}

void cipher_rt_fairness_init(void)
{
	if (atomic_load_explicit(&g_init_done, memory_order_acquire))
		return;
	if (atomic_exchange(&g_init_done, 1))
		return;   /* another thread is doing it / done */

	const char *f = getenv("CIPHER_FAIRNESS");
	const char *s = getenv("CIPHER_SHIELD");
	if (!env_on(f) && !env_on(s)) {
		atomic_store(&g_armed, -1);   /* disarmed: byte-identical to pre-D.8 */
		return;
	}

	const char *b = getenv("CIPHER_SHIELD_BAND");
	if (!b) b = getenv("CIPHER_FAIRNESS_BAND");
	if (b) g_my_band = (uint32_t)strtoul(b, NULL, 10);

	const char *m = getenv("CIPHER_FAIRNESS_BURST_MIN");
	if (m) { unsigned long long v = strtoull(m, NULL, 10); if (v) g_burst_min = v; }
	const char *t = getenv("CIPHER_FAIRNESS_THROTTLE_US");
	if (t) { unsigned long v = strtoul(t, NULL, 10); g_throttle_us = (uint32_t)v; }
	/* Test-only: force the band-modulated throttle to fire on every sampled
	 * GEMM regardless of contention, so the single-process KL gate exercises
	 * the CPU sleep and identical logits actually prove timing-only-ness. Not
	 * a production knob — never set in the Gate A/B / soak / production paths. */
	if (env_on(getenv("CIPHER_FAIRNESS_FORCE_YIELD")))
		g_force_yield = 1;

	int fd = open("/dev/cipher", O_RDWR | O_CLOEXEC);
	if (fd < 0) { atomic_store(&g_armed, -1); return; }

	struct fair_register_u req;
	memset(&req, 0, sizeof(req));
	req.band = g_my_band;
	if (ioctl(fd, CIPHER_FAIRNESS_REGISTER_U, &req) < 0) {
		close(fd); atomic_store(&g_armed, -1); return;
	}
	g_my_slot = (int)req.out_slot;
	g_my_tgid = (uint32_t)getpid();

	size_t size = ((size_t)CIPHER_FAIR_REGION_BYTES_U + CIPHER_FAIR_PAGE_SIZE_U - 1)
	              & ~(size_t)(CIPHER_FAIR_PAGE_SIZE_U - 1);
	off_t off = (off_t)CIPHER_FAIR_VIEW_MMAP_PGOFF_U * CIPHER_FAIR_PAGE_SIZE_U;
	void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, off);
	close(fd);   /* mapping survives close */
	if (p == MAP_FAILED) { atomic_store(&g_armed, -1); return; }
	g_region = (struct fair_slot_u *)p;
	atomic_store(&g_armed, 1);
}

/* Port of cipher_fairness_shm should_yield, reading the kmod-shared region.
 * Band modulation: a latency-sensitive tenant (band>=1) is never throttled;
 * a throughput tenant (band 0) yields on the burst rule OR when it out-paces
 * an active latency-sensitive neighbor (SHIELD). */
static int fair_should_yield(void)
{
	if (g_my_band >= 1)
		return 0;   /* SHIELD-protected: never self-throttle */
	if (g_force_yield)
		return 1;   /* test hook (KL gate): fire regardless of contention */
	if (!g_region || g_my_slot < 0)
		return 0;

	uint64_t my = atomic_load_explicit(
		(_Atomic uint64_t *)&g_region[g_my_slot].gemm_calls, memory_order_relaxed);
	if (my < g_burst_min)
		return 0;   /* not enough volume to be the aggressor */

	uint64_t now = fair_now_ns();
	uint64_t total = 0, hi_band_calls = 0;
	int n_active = 0, hi_band_active = 0;

	for (unsigned i = 0; i < CIPHER_FAIR_SLOTS_U; i++) {
		struct fair_slot_u *sl = &g_region[i];
		uint64_t k = atomic_load_explicit((_Atomic uint64_t *)&sl->key, memory_order_acquire);
		if (k == 0) continue;
		uint64_t la = atomic_load_explicit((_Atomic uint64_t *)&sl->last_active_ns, memory_order_acquire);
		if (la == 0 || now - la > FAIR_IDLE_THRESH_NS) continue;
		uint64_t calls = atomic_load_explicit((_Atomic uint64_t *)&sl->gemm_calls, memory_order_relaxed);
		uint32_t band = atomic_load_explicit((_Atomic uint32_t *)&sl->band, memory_order_relaxed);
		total += calls;
		n_active++;
		if (band >= 1) { hi_band_active = 1; if (calls > hi_band_calls) hi_band_calls = calls; }
	}
	if (n_active <= 1)
		return 0;

	uint64_t avg = total / (uint64_t)n_active;

	/* R-D5 burst-fairness: aggressor out-runs the field. */
	if (my > 2 * avg && my > g_burst_min)
		return 1;

	/* R-I1 SHIELD: don't out-pace an active latency-sensitive neighbor. */
	if (hi_band_active && my > hi_band_calls)
		return 1;

	return 0;
}

void cipher_rt_fairness_record_and_maybe_throttle(void)
{
	if (atomic_load_explicit(&g_armed, memory_order_relaxed) <= 0) {
		/* untried? try once. -1 stays a fast no-op forever after. */
		if (atomic_load_explicit(&g_armed, memory_order_relaxed) == 0)
			cipher_rt_fairness_init();
		if (atomic_load_explicit(&g_armed, memory_order_relaxed) != 1)
			return;
	}
	if (!g_region || g_my_slot < 0)
		return;

	/* self-account (single writer for this slot) */
	struct fair_slot_u *me = &g_region[g_my_slot];
	atomic_fetch_add_explicit((_Atomic uint64_t *)&me->gemm_calls, 1, memory_order_relaxed);
	atomic_store_explicit((_Atomic uint64_t *)&me->last_active_ns, fair_now_ns(), memory_order_release);
	atomic_fetch_add_explicit(&g_self_calls, 1, memory_order_relaxed);

	/* sample the (256-slot scan) decision every Nth call; cache between. */
	int yield;
	if ((++t_eval_ctr % FAIR_EVAL_EVERY) == 0) {
		yield = fair_should_yield();
		t_cached_yield = yield;
	} else {
		yield = t_cached_yield;
	}

	if (yield && g_throttle_us > 0) {
		atomic_fetch_add_explicit((_Atomic uint32_t *)&me->yield_count, 1, memory_order_relaxed);
		atomic_fetch_add_explicit(&g_self_yields, 1, memory_order_relaxed);
		struct timespec req = { .tv_sec = 0, .tv_nsec = (long)g_throttle_us * 1000L };
		nanosleep(&req, NULL);   /* timing-only: GEMM submitted after, unchanged */
	}
}

unsigned long long cipher_rt_fairness_self_calls(void)  { return atomic_load(&g_self_calls); }
unsigned long long cipher_rt_fairness_self_yields(void) { return atomic_load(&g_self_yields); }
int cipher_rt_fairness_armed(void)     { return atomic_load(&g_armed) == 1; }
int cipher_rt_fairness_self_band(void) { return (int)g_my_band; }
