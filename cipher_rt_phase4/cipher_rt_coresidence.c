/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_coresidence.c — W.6 sub-C substrate co-residence client.
 *
 * See cipher_rt_coresidence.h. Owns a lazily-opened, cached /dev/cipher fd
 * (the cipher_rt_classify_observer idiom) and throttles the kmod QUERY to
 * ~1 ioctl/sec, caching the last live snapshot for in-window callers. The
 * ioctl ABI (NR 30/31) is mirrored locally per the cipher_rt_commit.c
 * precedent (NR 28 mirrored there) so this module does not depend on the
 * kmod source tree at build time.
 */
#define _GNU_SOURCE
#include "cipher_rt_coresidence.h"

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <sys/ioctl.h>
#include <linux/ioctl.h>

#include "cipher_v2_internal.h"   /* cipher_log */

/* --- kmod ABI mirror (cipher_kmod cipher_ioctl.h NR 30/31) -------------- */
#define CIPHER_IOCTL_MAGIC  'C'
#define CIPHER_COHORT_MAX   128

struct cipher_cohort_register_abi {
	uint32_t tgid;
	uint32_t _pad;
	uint64_t model_fingerprint;
};

struct cipher_cohort_query_abi {
	uint64_t caller_fingerprint;
	uint32_t max_entries;
	uint32_t n_resident;
	struct cipher_cohort_peer entries[CIPHER_COHORT_MAX];
};

#define CIPHER_COHORT_REGISTER \
	_IOW(CIPHER_IOCTL_MAGIC, 30, struct cipher_cohort_register_abi)
#define CIPHER_COHORT_QUERY \
	_IOWR(CIPHER_IOCTL_MAGIC, 31, struct cipher_cohort_query_abi)

/* --- throttle + cache --------------------------------------------------- */
#define CIPHER_CORES_THROTTLE_NS   (1000ULL * 1000ULL * 1000ULL)  /* 1 s */

static int             g_fd          = -1;
static int             g_fd_init     = 0;
static pthread_mutex_t g_lock        = PTHREAD_MUTEX_INITIALIZER;

static uint64_t                  g_last_query_ns = 0;
static uint32_t                  g_cached_n      = 0;   /* last n_resident */
static uint32_t                  g_cached_peers  = 0;   /* count valid in g_peers */
static struct cipher_cohort_peer g_peers[CIPHER_COHORT_MAX];

static uint64_t mono_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Caller holds g_lock. Copy up to max_peers cached peers to peers_out. */
static void emit_cached(struct cipher_cohort_peer *peers_out, uint32_t max_peers)
{
	if (peers_out && max_peers) {
		uint32_t c = g_cached_peers < max_peers ? g_cached_peers : max_peers;
		if (c)
			memcpy(peers_out, g_peers, c * sizeof(g_peers[0]));
	}
}

uint32_t cipher_rt_coresidence_update(uint64_t my_fingerprint,
                                      struct cipher_cohort_peer *peers_out,
                                      uint32_t max_peers,
                                      uint32_t *n_total_out)
{
	struct cipher_cohort_query_abi q;
	uint64_t now;
	uint32_t i, n;

	pthread_mutex_lock(&g_lock);

	if (!g_fd_init) {
		g_fd = open("/dev/cipher", O_RDWR | O_CLOEXEC);
		g_fd_init = 1;
		if (g_fd < 0)
			cipher_log("CORES: open /dev/cipher failed errno=%d; "
			           "co-residence disabled", errno);
	}

	/* No device, or not warmed up, or inside the throttle window => cache. */
	now = mono_ns();
	if (g_fd < 0 || my_fingerprint == 0 ||
	    (g_last_query_ns != 0 && now - g_last_query_ns < CIPHER_CORES_THROTTLE_NS)) {
		n = g_cached_n;
		emit_cached(peers_out, max_peers);
		pthread_mutex_unlock(&g_lock);
		if (n_total_out) *n_total_out = n;
		return n;
	}

	memset(&q, 0, sizeof(q));
	q.caller_fingerprint = my_fingerprint;   /* heartbeat-or-insert self */
	q.max_entries        = CIPHER_COHORT_MAX;
	if (ioctl(g_fd, CIPHER_COHORT_QUERY, &q) < 0) {
		cipher_log("CORES: ioctl COHORT_QUERY errno=%d", errno);
		n = g_cached_n;
		emit_cached(peers_out, max_peers);
		pthread_mutex_unlock(&g_lock);
		if (n_total_out) *n_total_out = n;
		return n;
	}

	g_last_query_ns = now;
	g_cached_n      = q.n_resident;
	g_cached_peers  = q.n_resident < CIPHER_COHORT_MAX ? q.n_resident : CIPHER_COHORT_MAX;
	for (i = 0; i < g_cached_peers; i++)
		g_peers[i] = q.entries[i];

	n = g_cached_n;
	emit_cached(peers_out, max_peers);
	pthread_mutex_unlock(&g_lock);
	if (n_total_out) *n_total_out = n;
	return n;
}
