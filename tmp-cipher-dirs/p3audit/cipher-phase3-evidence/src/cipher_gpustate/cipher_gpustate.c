/* SPDX-License-Identifier: GPL-2.0
 *
 * cipher_gpustate.c -- userspace daemon that polls NVML and pushes
 * GPU state + per-process util into the kmod via the Task 3 ioctls.
 *
 * Phase 3 Layer B. Pairs with /dev/cipher ioctl nrs 5, 6, 7.
 *
 * Runtime contract:
 *   - Needs CAP_SYS_ADMIN for the two device-wide / per-process ioctls.
 *     (LAUNCH_STATS is anti-spoofed and not used here -- that's the
 *      workload's own path via libcipher_v2.)
 *   - Single GPU only for now; multi-GPU is Phase 6.
 *   - Logs to stderr; no syslog dependency.
 *   - Loop must not die on transient NVML/ioctl errors.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <getopt.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#include <nvml.h>

#include "/home/ubuntu/cipher_kmod/cipher_ioctl.h"

#define DEFAULT_INTERVAL_MS   250
#define MAX_PROCESS_SAMPLES   1024  /* sanity cap on NVML's reported count */

static volatile sig_atomic_t g_running = 1;
static int g_verbose = 0;

static void on_signal(int signo)
{
	(void)signo;
	g_running = 0;
}

static __u64 monotonic_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (__u64)ts.tv_sec * 1000000000ULL + (__u64)ts.tv_nsec;
}

static __u64 monotonic_ms(void)
{
	return monotonic_ns() / 1000000ULL;
}

#define NVML_OR_WARN(call, fmt, ...) do {                                       \
	nvmlReturn_t _r = (call);                                               \
	if (_r != NVML_SUCCESS) {                                               \
		fprintf(stderr, "cipher-gpustate: " fmt " -> %s\n",             \
			##__VA_ARGS__, nvmlErrorString(_r));                    \
	}                                                                       \
} while (0)

/* ---------- probe: verify CAP_SYS_ADMIN before entering main loop --------- */

static int probe_cap_sys_admin(int fd)
{
	struct cipher_gpu_state probe;
	int rc;

	memset(&probe, 0, sizeof(probe));
	probe.timestamp_ns = monotonic_ns();
	/* All util fields stay 0 -- they pass the <=100 check. */

	errno = 0;
	rc = ioctl(fd, CIPHER_SUBMIT_GPU_STATE, &probe);
	if (rc == 0)
		return 0;
	if (errno == EPERM) {
		fprintf(stderr,
			"cipher-gpustate: CAP_SYS_ADMIN check failed (EPERM). "
			"Re-run as root or with the capability granted.\n");
		return -1;
	}
	fprintf(stderr,
		"cipher-gpustate: probe ioctl failed unexpectedly: rc=%d errno=%d (%s)\n",
		rc, errno, strerror(errno));
	return -1;
}

/* ---------- device-wide gather + submit ----------------------------------- */

static int gather_and_submit_device_state(int fd, nvmlDevice_t dev, unsigned long sample_idx)
{
	struct cipher_gpu_state gs;
	nvmlUtilization_t util;
	nvmlMemory_t mem;
	unsigned int power_mw = 0, temp_c = 0, sm_mhz = 0, mem_mhz = 0;
	nvmlReturn_t r;
	int rc;

	memset(&gs, 0, sizeof(gs));

	NVML_OR_WARN(nvmlDeviceGetPowerUsage(dev, &power_mw),
		     "nvmlDeviceGetPowerUsage");
	NVML_OR_WARN(nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp_c),
		     "nvmlDeviceGetTemperature");
	NVML_OR_WARN(nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_mhz),
		     "nvmlDeviceGetClockInfo SM");
	NVML_OR_WARN(nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &mem_mhz),
		     "nvmlDeviceGetClockInfo MEM");

	r = nvmlDeviceGetUtilizationRates(dev, &util);
	if (r != NVML_SUCCESS) {
		fprintf(stderr, "cipher-gpustate: nvmlDeviceGetUtilizationRates -> %s\n",
			nvmlErrorString(r));
		util.gpu = 0;
		util.memory = 0;
	}

	r = nvmlDeviceGetMemoryInfo(dev, &mem);
	if (r != NVML_SUCCESS) {
		fprintf(stderr, "cipher-gpustate: nvmlDeviceGetMemoryInfo -> %s\n",
			nvmlErrorString(r));
		mem.used = 0;
	}

	gs.timestamp_ns  = monotonic_ns();
	gs.power_mw      = power_mw;
	gs.temp_c        = temp_c;
	gs.sm_clock_mhz  = sm_mhz;
	gs.mem_clock_mhz = mem_mhz;
	gs.sm_util_pct   = util.gpu;
	gs.mem_util_pct  = util.memory;
	gs.fb_used_mb    = (unsigned int)(mem.used >> 20);

	errno = 0;
	rc = ioctl(fd, CIPHER_SUBMIT_GPU_STATE, &gs);
	if (rc != 0) {
		fprintf(stderr,
			"cipher-gpustate: SUBMIT_GPU_STATE rc=%d errno=%d (%s)\n",
			rc, errno, strerror(errno));
		return -1;
	}
	if (g_verbose) {
		fprintf(stderr,
			"cipher-gpustate: sample %lu -- power=%u mW temp=%u C "
			"sm_clock=%u MHz mem_clock=%u MHz sm=%u%% mem=%u%% fb=%u MB\n",
			sample_idx, power_mw, temp_c, sm_mhz, mem_mhz,
			util.gpu, util.memory, (unsigned)(mem.used >> 20));
		fprintf(stderr, "cipher-gpustate: submitted GPU_STATE rc=0\n");
	}
	return 0;
}

/* ---------- per-process gather + submit ----------------------------------- */

/* Returns: # of submissions made, or -1 if API is unsupported (so caller can
 * stop trying). */
static int gather_and_submit_process_util(int fd, nvmlDevice_t dev)
{
	static int capped_logged = 0;
	nvmlProcessUtilizationSample_t *samples = NULL;
	unsigned int needed_count = 0;
	nvmlReturn_t r;
	int submitted = 0;
	unsigned int i;
	struct timespec rt;
	unsigned long long last_seen_us;

	/* Build timestamp window. Kept from the previous attempt even though
	 * it didn't reduce NVML's reported sample count on this driver --
	 * doesn't hurt, and may help on other driver builds where NVML
	 * actually filters on CLOCK_REALTIME microseconds. */
	clock_gettime(CLOCK_REALTIME, &rt);
	last_seen_us = (unsigned long long)rt.tv_sec * 1000000ULL +
		       (unsigned long long)rt.tv_nsec / 1000ULL;
	if (last_seen_us > 1000000ULL)
		last_seen_us -= 1000000ULL;

	/* Two-call pattern per NVML docs:
	 *   Call 1: utilization=NULL -> NVML sets *count to the required
	 *           size and returns INSUFFICIENT_SIZE (or SUCCESS if 0).
	 *   Call 2: allocate count*sizeof(sample), re-query. */
	r = nvmlDeviceGetProcessUtilization(dev, NULL, &needed_count, last_seen_us);

	if (r == NVML_ERROR_NOT_FOUND || needed_count == 0) {
		if (g_verbose)
			fprintf(stderr, "cipher-gpustate: 0 processes\n");
		return 0;
	}
	if (r == NVML_ERROR_NOT_SUPPORTED) {
		fprintf(stderr,
			"cipher-gpustate: per-process util NOT_SUPPORTED -- "
			"device-wide only from here on\n");
		return -1;
	}
	if (r != NVML_ERROR_INSUFFICIENT_SIZE && r != NVML_SUCCESS) {
		fprintf(stderr,
			"cipher-gpustate: NVML sample-count query -> %s\n",
			nvmlErrorString(r));
		return 0;
	}

	/* Sanity cap. If NVML reports >1024 processes on one GPU,
	 * something is wrong; cap and log once. */
	if (needed_count > MAX_PROCESS_SAMPLES) {
		if (!capped_logged) {
			fprintf(stderr,
				"cipher-gpustate: NVML reports %u processes, "
				"capping at %d (will not log again)\n",
				needed_count, MAX_PROCESS_SAMPLES);
			capped_logged = 1;
		}
		needed_count = MAX_PROCESS_SAMPLES;
	}

	samples = calloc(needed_count, sizeof(*samples));
	if (!samples) {
		fprintf(stderr,
			"cipher-gpustate: calloc for %u samples failed\n",
			needed_count);
		return 0;
	}

	r = nvmlDeviceGetProcessUtilization(dev, samples, &needed_count, last_seen_us);
	if (r == NVML_ERROR_NOT_FOUND) {
		free(samples);
		if (g_verbose)
			fprintf(stderr, "cipher-gpustate: 0 processes\n");
		return 0;
	}
	if (r != NVML_SUCCESS) {
		fprintf(stderr,
			"cipher-gpustate: NVML sample fetch -> %s\n",
			nvmlErrorString(r));
		free(samples);
		return 0;
	}

	for (i = 0; i < needed_count; i++) {
		struct cipher_process_util pu;
		int rc;

		memset(&pu, 0, sizeof(pu));
		/* NVML reports the Linux TGID under .pid (POSIX-PID == TGID).
		 * The kmod's per-PID table is keyed on the leader-thread LWP
		 * which is == TGID for single-threaded launchers and for the
		 * main thread of multi-threaded ones. We submit pid==tgid. */
		pu.pid           = samples[i].pid;
		pu.tgid          = samples[i].pid;
		pu.timestamp_ns  = monotonic_ns();
		pu.sm_util_pct   = samples[i].smUtil;
		pu.mem_util_pct  = samples[i].memUtil;
		pu.fb_used_mb    = 0;   /* per-process FB needs GetProcessesMemoryInfo */
		pu.enc_util_pct  = samples[i].encUtil;
		pu.dec_util_pct  = samples[i].decUtil;

		/* Clamp util fields the kernel will reject. NVML occasionally
		 * returns 101+ on warmup samples. */
		if (pu.sm_util_pct  > 100) pu.sm_util_pct  = 100;
		if (pu.mem_util_pct > 100) pu.mem_util_pct = 100;
		if (pu.enc_util_pct > 100) pu.enc_util_pct = 100;
		if (pu.dec_util_pct > 100) pu.dec_util_pct = 100;

		errno = 0;
		rc = ioctl(fd, CIPHER_SUBMIT_PROCESS_UTIL, &pu);
		if (rc != 0) {
			fprintf(stderr,
				"cipher-gpustate: SUBMIT_PROCESS_UTIL pid=%u rc=%d errno=%d (%s)\n",
				pu.pid, rc, errno, strerror(errno));
			continue;
		}
		submitted++;
		if (g_verbose)
			fprintf(stderr,
				"cipher-gpustate:   pid=%u sm=%u%% mem=%u%% enc=%u%% dec=%u%%\n",
				pu.pid, pu.sm_util_pct, pu.mem_util_pct,
				pu.enc_util_pct, pu.dec_util_pct);
	}
	if (g_verbose)
		fprintf(stderr, "cipher-gpustate: %u processes (%d submitted)\n",
			needed_count, submitted);

	free(samples);
	return submitted;
}

/* ---------- CLI ----------------------------------------------------------- */

static void usage(const char *argv0)
{
	fprintf(stderr,
		"usage: %s [--device N] [--interval-ms N] [--verbose] [--once]\n"
		"  --device N       GPU index (default 0)\n"
		"  --interval-ms N  poll cadence in ms (default %d)\n"
		"  --verbose        log every sample to stderr\n"
		"  --once           poll one sample, submit, exit (for testing)\n",
		argv0, DEFAULT_INTERVAL_MS);
}

struct opts {
	unsigned int device_index;
	unsigned int interval_ms;
	int          once;
};

static int parse_args(int argc, char **argv, struct opts *o)
{
	static const struct option longs[] = {
		{ "device",      required_argument, 0, 'd' },
		{ "interval-ms", required_argument, 0, 'i' },
		{ "verbose",     no_argument,       0, 'v' },
		{ "once",        no_argument,       0, '1' },
		{ "help",        no_argument,       0, 'h' },
		{ 0, 0, 0, 0 }
	};
	int c;

	o->device_index = 0;
	o->interval_ms  = DEFAULT_INTERVAL_MS;
	o->once         = 0;

	while ((c = getopt_long(argc, argv, "d:i:vh", longs, NULL)) != -1) {
		switch (c) {
		case 'd': o->device_index = (unsigned int)atoi(optarg); break;
		case 'i': o->interval_ms  = (unsigned int)atoi(optarg); break;
		case 'v': g_verbose = 1; break;
		case '1': o->once = 1; break;
		case 'h': usage(argv[0]); return 1;
		default:  usage(argv[0]); return -1;
		}
	}
	if (o->interval_ms == 0) {
		fprintf(stderr, "cipher-gpustate: --interval-ms must be > 0\n");
		return -1;
	}
	return 0;
}

/* ---------- main ---------------------------------------------------------- */

int main(int argc, char **argv)
{
	struct opts opts;
	int rc, fd = -1;
	nvmlReturn_t r;
	nvmlDevice_t dev;
	char nvml_drv_ver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {0};
	char dev_name[NVML_DEVICE_NAME_V2_BUFFER_SIZE] = {0};
	int proc_util_supported = 1;
	unsigned long sample_count = 0;
	__u64 start_ms, deadline_ms;
	struct sigaction sa = { .sa_handler = on_signal };

	if ((rc = parse_args(argc, argv, &opts)) != 0)
		return rc < 0 ? 1 : 0;

	/* SIGINT / SIGTERM -> graceful shutdown. SA_RESTART intentionally
	 * unset so usleep returns early on signal. */
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;
	sigaction(SIGINT,  &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);

	r = nvmlInit_v2();
	if (r != NVML_SUCCESS) {
		fprintf(stderr, "cipher-gpustate: nvmlInit_v2 -> %s\n",
			nvmlErrorString(r));
		return 1;
	}

	(void)nvmlSystemGetDriverVersion(nvml_drv_ver, sizeof(nvml_drv_ver));
	fprintf(stderr, "cipher-gpustate: NVML version %d (driver %s)\n",
		NVML_API_VERSION, nvml_drv_ver);

	r = nvmlDeviceGetHandleByIndex(opts.device_index, &dev);
	if (r != NVML_SUCCESS) {
		fprintf(stderr, "cipher-gpustate: nvmlDeviceGetHandleByIndex(%u) -> %s\n",
			opts.device_index, nvmlErrorString(r));
		nvmlShutdown();
		return 1;
	}
	(void)nvmlDeviceGetName(dev, dev_name, sizeof(dev_name));
	fprintf(stderr, "cipher-gpustate: device %u = %s\n",
		opts.device_index, dev_name);

	fd = open("/dev/cipher", O_WRONLY);
	if (fd < 0) {
		fprintf(stderr, "cipher-gpustate: open /dev/cipher -> %s\n",
			strerror(errno));
		nvmlShutdown();
		return 1;
	}

	if (probe_cap_sys_admin(fd) != 0) {
		close(fd);
		nvmlShutdown();
		return 1;
	}
	fprintf(stderr, "cipher-gpustate: CAP_SYS_ADMIN ok; /dev/cipher fd=%d\n", fd);
	fprintf(stderr,
		"cipher-gpustate: per-process util via nvmlDeviceGetProcessUtilization\n");
	fprintf(stderr,
		"cipher-gpustate: cadence %u ms; once=%d; verbose=%d\n",
		opts.interval_ms, opts.once, g_verbose);

	start_ms    = monotonic_ms();
	deadline_ms = start_ms;

	while (g_running) {
		gather_and_submit_device_state(fd, dev, sample_count + 1);

		if (proc_util_supported) {
			int n = gather_and_submit_process_util(fd, dev);
			if (n < 0)
				proc_util_supported = 0;
		}

		sample_count++;
		if (opts.once)
			break;

		/* Drift-corrected sleep to the next tick. */
		deadline_ms += opts.interval_ms;
		for (;;) {
			__u64 now = monotonic_ms();
			if (now >= deadline_ms)
				break;
			__u64 remain = deadline_ms - now;
			if (remain > opts.interval_ms)
				remain = opts.interval_ms;
			struct timespec ts = {
				.tv_sec  = (time_t)(remain / 1000),
				.tv_nsec = (long)((remain % 1000) * 1000000L),
			};
			if (nanosleep(&ts, NULL) == 0)
				break;
			if (!g_running)
				break;
		}
	}

	fprintf(stderr, "cipher-gpustate: exiting after %lu samples\n",
		sample_count);

	if (fd >= 0)
		close(fd);
	nvmlShutdown();
	return 0;
}
