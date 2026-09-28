/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_sm_packer.c -- T4.2.3 SM_PACKER (detection / counters).
 *
 * Detection-only build. Replaces by-stream streak tracking with a single
 * per-stream-handle uintptr_t to amortize TLS overhead; per-stream streak
 * lives in a small open-addressed cache (32 slots — typical workloads
 * use 1-4 streams).
 */
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_sm_packer.h"

#define SMP_STREAM_CACHE_SLOTS  32

struct smp_stream_state {
	uintptr_t handle;
	unsigned long current_streak;
};

static struct smp_stream_state g_smp_streams[SMP_STREAM_CACHE_SLOTS];
static pthread_mutex_t         g_smp_lock = PTHREAD_MUTEX_INITIALIZER;

static atomic_ulong g_total_launches  = 0;
static atomic_ulong g_small_launches  = 0;
static atomic_ulong g_longest_streak  = 0;

int cipher_rt_smp_init(void)
{
	memset(g_smp_streams, 0, sizeof(g_smp_streams));
	cipher_log("SMP: sm_packer initialized (threshold=%u threads, %d stream slots)",
	           CIPHER_RT_SMP_SMALL_THRESHOLD, SMP_STREAM_CACHE_SLOTS);
	return 0;
}

void cipher_rt_smp_observe(unsigned int gridX, unsigned int gridY,
                           unsigned int gridZ, unsigned int blockX,
                           unsigned int blockY, unsigned int blockZ,
                           void *stream_handle)
{
	uint64_t threads;
	int idx;
	uintptr_t h = (uintptr_t)stream_handle;
	unsigned long new_total;

	atomic_fetch_add_explicit(&g_total_launches, 1, memory_order_relaxed);

	threads = (uint64_t)gridX * (uint64_t)gridY * (uint64_t)gridZ *
	          (uint64_t)blockX * (uint64_t)blockY * (uint64_t)blockZ;

	if (threads >= CIPHER_RT_SMP_SMALL_THRESHOLD) {
		/* Big launch — reset streak for this stream. */
		pthread_mutex_lock(&g_smp_lock);
		for (idx = 0; idx < SMP_STREAM_CACHE_SLOTS; idx++) {
			if (g_smp_streams[idx].handle == h) {
				g_smp_streams[idx].current_streak = 0;
				break;
			}
		}
		pthread_mutex_unlock(&g_smp_lock);
		return;
	}

	atomic_fetch_add_explicit(&g_small_launches, 1, memory_order_relaxed);

	pthread_mutex_lock(&g_smp_lock);
	/* Find or insert stream entry. */
	idx = -1;
	for (int i = 0; i < SMP_STREAM_CACHE_SLOTS; i++) {
		if (g_smp_streams[i].handle == h) { idx = i; break; }
	}
	if (idx < 0) {
		for (int i = 0; i < SMP_STREAM_CACHE_SLOTS; i++) {
			if (g_smp_streams[i].handle == 0) {
				g_smp_streams[i].handle = h;
				g_smp_streams[i].current_streak = 0;
				idx = i;
				break;
			}
		}
	}
	if (idx >= 0) {
		g_smp_streams[idx].current_streak++;
		new_total = g_smp_streams[idx].current_streak;
		pthread_mutex_unlock(&g_smp_lock);

		/* Update longest streak (lock-free). */
		while (1) {
			unsigned long cur = atomic_load(&g_longest_streak);
			if (new_total <= cur) break;
			if (atomic_compare_exchange_weak(&g_longest_streak, &cur, new_total))
				break;
		}
	} else {
		pthread_mutex_unlock(&g_smp_lock);
	}
}

unsigned long cipher_rt_smp_total_launches(void)
{
	return atomic_load(&g_total_launches);
}

unsigned long cipher_rt_smp_small_launches(void)
{
	return atomic_load(&g_small_launches);
}

unsigned long cipher_rt_smp_longest_streak(void)
{
	return atomic_load(&g_longest_streak);
}
