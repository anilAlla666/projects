/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_cupti.c -- CUPTI subscriber + kernel-launch callback.
 *
 * Phase 3 Task 5: populates the LAUNCHES column in /proc/cipher/stats
 * via CIPHER_SUBMIT_LAUNCH_STATS (ioctl nr 7, anti-spoofed).
 *
 * Design: NO flush thread. The CUPTI callback fires on the workload's
 * own thread (the one calling cuLaunchKernel), so gettid()/getpid()
 * naturally match the identity registered by REGISTER_TENANT. A flush
 * thread would have its own LWP, breaking anti-spoof.
 *
 * Cadence: flush every 256 launches. Bounded latency, batches the
 * ioctl cost, no per-launch syscall.
 *
 * Failure mode: any CUPTI error (subscribe / enable) is logged once
 * and the module degrades to "no LAUNCHES column populated" -- the
 * daemon's NVML telemetry path is unaffected.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#include <cupti.h>

#include "cipher_ioctl.h"
#include "cipher_v2_internal.h"

#define CIPHER_V2_FLUSH_MASK   0xff   /* flush every 256 launches */

static atomic_uint_least64_t g_launches_total = 0;
static atomic_uint_least64_t g_grid_ops_total = 0;
static int                    g_cipher_fd     = -1;
static CUpti_SubscriberHandle g_subscriber    = NULL;

static void CUPTIAPI cipher_v2_cupti_cb(void *userdata,
                                        CUpti_CallbackDomain domain,
                                        CUpti_CallbackId cbid,
                                        const CUpti_CallbackData *cb)
{
	uint64_t new_total;
	struct cipher_launch_stats ls;
	struct timespec now;

	(void)userdata;
	(void)domain;
	(void)cbid;

	/* Only count entry, not exit -- otherwise each launch
	 * counts twice. */
	if (cb->callbackSite != CUPTI_API_ENTER)
		return;

	new_total = atomic_fetch_add_explicit(&g_launches_total, 1,
	                                      memory_order_relaxed) + 1;

	/* Batched flush: ioctl only on launch counts that are
	 * multiples of 256. */
	if ((new_total & CIPHER_V2_FLUSH_MASK) != 0)
		return;

	if (g_cipher_fd < 0)
		return;

	memset(&ls, 0, sizeof(ls));
	ls.pid  = (__u32)gettid();   /* matches REGISTER_TENANT'd thread */
	ls.tgid = (__u32)getpid();
	clock_gettime(CLOCK_MONOTONIC, &now);
	ls.timestamp_ns = (uint64_t)now.tv_sec * 1000000000ULL +
	                  (uint64_t)now.tv_nsec;
	ls.launches_total = new_total;
	ls.grid_ops_total = atomic_load_explicit(&g_grid_ops_total,
	                                         memory_order_relaxed);

	/* Errors silently dropped: there's nothing useful we can do
	 * mid-launch, and the daemon's NVML path remains independent. */
	(void)ioctl(g_cipher_fd, CIPHER_SUBMIT_LAUNCH_STATS, &ls);
}

int cipher_v2_cupti_init(void)
{
	CUptiResult cr;

	/* Long-lived fd: keep open for process lifetime so the callback
	 * doesn't pay open()/close() per flush. */
	g_cipher_fd = open(CIPHER_V2_DEV, O_WRONLY);
	if (g_cipher_fd < 0) {
		cipher_log("CUPTI: open(%s) failed: %s -- LAUNCHES will not populate",
		           CIPHER_V2_DEV, strerror(errno));
		return -1;
	}

	cr = cuptiSubscribe(&g_subscriber,
	                    (CUpti_CallbackFunc)cipher_v2_cupti_cb, NULL);
	if (cr != CUPTI_SUCCESS) {
		cipher_log("cuptiSubscribe -> %d, LAUNCHES will not populate", cr);
		close(g_cipher_fd);
		g_cipher_fd = -1;
		return -2;
	}

	/* Runtime API: cudaLaunchKernel (the dominant path for PyTorch,
	 * TensorFlow, etc.). */
	cr = cuptiEnableCallback(1, g_subscriber,
	                         CUPTI_CB_DOMAIN_RUNTIME_API,
	                         CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
	if (cr != CUPTI_SUCCESS)
		cipher_log("cuptiEnableCallback cudaLaunchKernel -> %d", cr);

	/* Runtime API: cudaLaunchKernelExC (CUDA 11.6+). */
	cr = cuptiEnableCallback(1, g_subscriber,
	                         CUPTI_CB_DOMAIN_RUNTIME_API,
	                         CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060);
	if (cr != CUPTI_SUCCESS)
		cipher_log("cuptiEnableCallback cudaLaunchKernelExC -> %d", cr);

	/* Driver API: cuLaunchKernel (TensorRT and direct driver users). */
	cr = cuptiEnableCallback(1, g_subscriber,
	                         CUPTI_CB_DOMAIN_DRIVER_API,
	                         CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
	if (cr != CUPTI_SUCCESS)
		cipher_log("cuptiEnableCallback cuLaunchKernel -> %d", cr);

	cipher_log("CUPTI subscribed: kernel launch callbacks active "
	           "(flush every %d launches)", CIPHER_V2_FLUSH_MASK + 1);
	return 0;
}
