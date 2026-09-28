/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_volt.c -- T4.3.1 VOLT NVML clock-lock actuator (v1 port).
 *
 * Derived from cipher-may13-evidence/src/cipher_volt.cpp (Phase B path):
 * the signal-safe NVML restore pattern, dlsym-cached function pointers,
 * and probe-supported-then-actuate flow. v1 simplifications:
 *   - env-driven static lock (no per-shape AI classifier; T4.3.2)
 *   - one init-time lock, one exit-time restore (no 100ms cadence)
 *   - C, not C++ (matches the rest of libcipher_rt)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <signal.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/types.h>

#include "cipher_ioctl.h"
#include "cipher_v2_internal.h"
#include "cipher_rt_volt.h"

/* NVML constants (avoid pulling in nvml.h). */
#define NVML_SUCCESS              0
#define NVML_ERROR_NOT_SUPPORTED  4

typedef int   nvmlReturn_t;
typedef void *nvmlDevice_t;
typedef nvmlReturn_t (*fn_init_t)(void);
typedef nvmlReturn_t (*fn_get_handle_t)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*fn_get_driver_t)(char *, unsigned);
typedef nvmlReturn_t (*fn_get_clock_t)(nvmlDevice_t, int, unsigned int *);
typedef nvmlReturn_t (*fn_set_locked_t)(nvmlDevice_t, unsigned, unsigned);
typedef nvmlReturn_t (*fn_reset_locked_t)(nvmlDevice_t);

/* T4.3.2: fall-through actuator via /dev/cipher CIPHER_SET_CLOCK_MHZ.
 * The kmod runs as root (regardless of caller) and invokes nvidia-smi
 * via call_usermodehelper. This closes the user-process NVML privilege
 * gap surfaced in T4.3.1. */
typedef enum {
	CIPHER_RT_VOLT_PATH_NONE     = 0,
	CIPHER_RT_VOLT_PATH_NVML     = 1,
	CIPHER_RT_VOLT_PATH_KMOD_IOCTL = 2,
} CipherRtVoltPath;

/* Calibration table — from cipher-may13-evidence/p5_optimal_clocks.json.
 * Optimal tok/W clock per batch size on H100. */
static unsigned int batch_to_mhz(int batch)
{
	switch (batch) {
	case 1:   return 1000;
	case 8:   return 1600;
	case 32:  return 1980;
	case 64:  return 1980;
	default:  return 0;
	}
}

static struct {
	atomic_int         init_done;
	atomic_int         mode;
	atomic_uint        locked_mhz;
	void              *nvml_handle;
	nvmlDevice_t       dev;
	fn_get_handle_t    fn_get_handle;
	fn_get_driver_t    fn_get_driver;
	fn_get_clock_t     fn_get_clock;
	fn_set_locked_t    fn_set_locked;
	fn_reset_locked_t  fn_reset_locked;
	char               driver_version[64];
	unsigned int       base_clock_mhz;
	atomic_int         restored;    /* 1 once exit/signal restored */
	atomic_int         path;        /* CipherRtVoltPath used */
	int                cipher_fd;   /* /dev/cipher fd for kmod ioctl path */
} g_volt = {0};

/* Try to resolve NVML symbols. Returns 1 on success. */
static int resolve_nvml(void)
{
	void *h;
	fn_init_t fn_init;

	if (g_volt.nvml_handle) return g_volt.fn_set_locked != NULL;

	h = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_NOLOAD);
	if (!h) h = dlopen("libnvidia-ml.so",   RTLD_LAZY | RTLD_NOLOAD);
	if (!h) h = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
	if (!h) h = dlopen("libnvidia-ml.so",   RTLD_LAZY);
	if (!h) {
		cipher_log("VOLT: dlopen(libnvidia-ml.so) failed");
		return 0;
	}

	fn_init = (fn_init_t)dlsym(h, "nvmlInit_v2");
	if (fn_init) fn_init();  /* idempotent */

	g_volt.nvml_handle    = h;
	g_volt.fn_get_handle  = (fn_get_handle_t)dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
	g_volt.fn_get_driver  = (fn_get_driver_t)dlsym(h, "nvmlSystemGetDriverVersion");
	g_volt.fn_get_clock   = (fn_get_clock_t)dlsym(h, "nvmlDeviceGetClockInfo");
	g_volt.fn_set_locked  = (fn_set_locked_t)dlsym(h, "nvmlDeviceSetGpuLockedClocks");
	g_volt.fn_reset_locked= (fn_reset_locked_t)dlsym(h, "nvmlDeviceResetGpuLockedClocks");

	if (!g_volt.fn_get_handle || !g_volt.fn_set_locked || !g_volt.fn_reset_locked) {
		cipher_log("VOLT: NVML symbol lookup incomplete (handle=%p set=%p reset=%p)",
		           (void *)g_volt.fn_get_handle,
		           (void *)g_volt.fn_set_locked,
		           (void *)g_volt.fn_reset_locked);
		return 0;
	}
	if (g_volt.fn_get_handle(0, &g_volt.dev) != NVML_SUCCESS) {
		cipher_log("VOLT: nvmlDeviceGetHandleByIndex_v2 failed");
		return 0;
	}
	if (g_volt.fn_get_driver) {
		g_volt.fn_get_driver(g_volt.driver_version,
		                     sizeof(g_volt.driver_version));
	}
	if (g_volt.fn_get_clock) {
		unsigned int mhz = 0;
		/* 0 == NVML_CLOCK_SM */
		if (g_volt.fn_get_clock(g_volt.dev, 0, &mhz) == NVML_SUCCESS)
			g_volt.base_clock_mhz = mhz;
	}
	return 1;
}

/* Probe write permission with a no-op SetGpuLockedClocks(current, current). */
static int probe_actuation(void)
{
	unsigned int cur;
	int r;

	if (!g_volt.fn_set_locked || !g_volt.fn_reset_locked)
		return NVML_ERROR_NOT_SUPPORTED;
	cur = g_volt.base_clock_mhz ? g_volt.base_clock_mhz : 1980;
	r = g_volt.fn_set_locked(g_volt.dev, cur, cur);
	if (r == NVML_SUCCESS)
		(void)g_volt.fn_reset_locked(g_volt.dev);
	return r;
}

/* T4.3.2 kmod ioctl path. Returns 0 on success, -1 on failure.
 * Probe by calling reset (0) — succeeds whether or not anything was locked. */
static int kmod_ioctl_open(void)
{
	if (g_volt.cipher_fd >= 0) return 0;
	g_volt.cipher_fd = open(CIPHER_V2_DEV, O_RDWR);
	if (g_volt.cipher_fd < 0) {
		cipher_log("VOLT: kmod ioctl open(%s) failed: %s",
		           CIPHER_V2_DEV, strerror(errno));
		return -1;
	}
	return 0;
}

static int kmod_ioctl_set_mhz(unsigned int mhz)
{
	__u32 val;
	if (g_volt.cipher_fd < 0) return -1;
	val = (__u32)mhz;
	if (ioctl(g_volt.cipher_fd, CIPHER_SET_CLOCK_MHZ, &val) < 0) {
		cipher_log("VOLT: CIPHER_SET_CLOCK_MHZ(%u) ioctl failed: %s",
		           mhz, strerror(errno));
		return -1;
	}
	return 0;
}

/* Probe the kmod ioctl path by sending a no-op reset. If the kmod is loaded
 * and accepts our ioctl, this returns 0. The kmod handles privilege internally
 * via call_usermodehelper, so this works from non-root injection contexts. */
static int probe_kmod_actuation(void)
{
	if (kmod_ioctl_open() < 0) return -1;
	return kmod_ioctl_set_mhz(0);  /* reset is a no-op if not locked */
}

/* Path-aware restore primitive. Called from atexit + signal handler.
 * Both NVML and ioctl paths are async-signal-safe at runtime: NVML's
 * fn_reset_locked is a pre-cached function pointer; ioctl is a direct
 * syscall. Both branches return immediately if not applicable. */
static void cipher_rt_volt_do_restore(void)
{
	int path = atomic_load(&g_volt.path);
	if (path == CIPHER_RT_VOLT_PATH_NVML &&
	    g_volt.fn_reset_locked && g_volt.dev) {
		(void)g_volt.fn_reset_locked(g_volt.dev);
	} else if (path == CIPHER_RT_VOLT_PATH_KMOD_IOCTL &&
	           g_volt.cipher_fd >= 0) {
		__u32 zero = 0;
		(void)ioctl(g_volt.cipher_fd, CIPHER_SET_CLOCK_MHZ, &zero);
	}
}

/* atexit restore. Idempotent via g_volt.restored atomic. */
static void cipher_rt_volt_atexit_restore(void)
{
	int was_restored;

	if (atomic_load(&g_volt.mode) != CIPHER_RT_VOLT_ACTIVE) return;
	if (atomic_load(&g_volt.locked_mhz) == 0) return;

	was_restored = atomic_exchange(&g_volt.restored, 1);
	if (was_restored) return;

	cipher_rt_volt_do_restore();
	cipher_log("VOLT: atexit restored via path=%d", atomic_load(&g_volt.path));
}

/* Signal handler. Async-signal-safe path only: cached fn pointer / fd + atomic. */
static void cipher_rt_volt_signal_handler(int sig, siginfo_t *info, void *ctx)
{
	struct sigaction sa;
	(void)info;
	(void)ctx;

	if (atomic_load(&g_volt.mode) == CIPHER_RT_VOLT_ACTIVE &&
	    atomic_load(&g_volt.locked_mhz) != 0 &&
	    !atomic_exchange(&g_volt.restored, 1)) {
		cipher_rt_volt_do_restore();
	}

	/* Re-raise with default handler */
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = SIG_DFL;
	sigemptyset(&sa.sa_mask);
	sigaction(sig, &sa, NULL);
	raise(sig);
}

static void install_signal_handlers(void)
{
	struct sigaction sa;
	int sigs[] = {SIGTERM, SIGINT, SIGSEGV, SIGABRT, SIGBUS};
	unsigned i;

	memset(&sa, 0, sizeof(sa));
	sa.sa_sigaction = cipher_rt_volt_signal_handler;
	sa.sa_flags     = SA_SIGINFO | SA_RESETHAND;
	sigemptyset(&sa.sa_mask);
	for (i = 0; i < sizeof(sigs)/sizeof(sigs[0]); i++)
		sigaction(sigs[i], &sa, NULL);
	atexit(cipher_rt_volt_atexit_restore);
}

int cipher_rt_volt_init(void)
{
	const char *env_mode  = getenv("CIPHER_VOLT");
	const char *env_batch = getenv("CIPHER_VOLT_BATCH");
	const char *env_mhz   = getenv("CIPHER_VOLT_MHZ");
	int target_mhz = 0;
	int probe_rc;
	int path = CIPHER_RT_VOLT_PATH_NONE;

	if (atomic_exchange(&g_volt.init_done, 1)) {
		return atomic_load(&g_volt.mode);
	}

	atomic_store(&g_volt.mode, CIPHER_RT_VOLT_OFF);
	atomic_store(&g_volt.locked_mhz, 0);
	atomic_store(&g_volt.restored, 0);
	atomic_store(&g_volt.path, CIPHER_RT_VOLT_PATH_NONE);
	g_volt.cipher_fd = -1;

	if (!env_mode || strcmp(env_mode, "off") == 0 ||
	    strcmp(env_mode, "0")   == 0) {
		return CIPHER_RT_VOLT_OFF;
	}
	if (strcmp(env_mode, "on") != 0 && strcmp(env_mode, "1") != 0 &&
	    strcmp(env_mode, "ON") != 0) {
		cipher_log("VOLT: unrecognized CIPHER_VOLT='%s' (expected off|on); staying OFF",
		           env_mode);
		return CIPHER_RT_VOLT_OFF;
	}

	if (env_mhz) {
		target_mhz = atoi(env_mhz);
	} else if (env_batch) {
		target_mhz = (int)batch_to_mhz(atoi(env_batch));
	}
	if (target_mhz <= 0) {
		cipher_log("VOLT: CIPHER_VOLT=on but no target — set CIPHER_VOLT_BATCH={1|8|32|64} "
		           "or CIPHER_VOLT_MHZ=N. Staying OFF.");
		return CIPHER_RT_VOLT_OFF;
	}
	if (target_mhz < 210 || target_mhz > 1980) {
		cipher_log("VOLT: target_mhz=%d outside H100 safe range [210, 1980]; staying OFF",
		           target_mhz);
		return CIPHER_RT_VOLT_OFF;
	}

	/* T4.3.1 path: try NVML first (zero-latency once resolved). */
	if (resolve_nvml()) {
		probe_rc = probe_actuation();
		if (probe_rc == NVML_SUCCESS) {
			path = CIPHER_RT_VOLT_PATH_NVML;
			cipher_log("VOLT: NVML path available (driver=%s, base=%u MHz)",
			           g_volt.driver_version[0] ? g_volt.driver_version : "?",
			           g_volt.base_clock_mhz);
		} else {
			cipher_log("VOLT: NVML probe rc=%d (NOT_SUPPORTED=%d) — trying kmod ioctl path",
			           probe_rc, NVML_ERROR_NOT_SUPPORTED);
		}
	} else {
		cipher_log("VOLT: NVML resolve failed — trying kmod ioctl path");
	}

	/* T4.3.2 path: fall through to kmod ioctl if NVML rejected. */
	if (path == CIPHER_RT_VOLT_PATH_NONE) {
		if (probe_kmod_actuation() == 0) {
			path = CIPHER_RT_VOLT_PATH_KMOD_IOCTL;
			cipher_log("VOLT: kmod ioctl path available (CIPHER_SET_CLOCK_MHZ)");
		}
	}

	if (path == CIPHER_RT_VOLT_PATH_NONE) {
		atomic_store(&g_volt.mode, CIPHER_RT_VOLT_DEGRADED);
		cipher_log("VOLT: DEGRADED — neither NVML nor kmod ioctl actuation path available");
		return CIPHER_RT_VOLT_DEGRADED;
	}

	atomic_store(&g_volt.path, path);

	/* Install handlers BEFORE applying the lock so a crash between
	 * apply and exit still restores. */
	install_signal_handlers();

	if (path == CIPHER_RT_VOLT_PATH_NVML) {
		if (g_volt.fn_set_locked(g_volt.dev,
		                         (unsigned)target_mhz,
		                         (unsigned)target_mhz) != NVML_SUCCESS) {
			atomic_store(&g_volt.mode, CIPHER_RT_VOLT_DEGRADED);
			cipher_log("VOLT: DEGRADED — NVML set_locked_clocks(%d,%d) failed after probe success",
			           target_mhz, target_mhz);
			return CIPHER_RT_VOLT_DEGRADED;
		}
	} else { /* CIPHER_RT_VOLT_PATH_KMOD_IOCTL */
		if (kmod_ioctl_set_mhz((unsigned)target_mhz) < 0) {
			atomic_store(&g_volt.mode, CIPHER_RT_VOLT_DEGRADED);
			cipher_log("VOLT: DEGRADED — kmod ioctl set(%d) failed after probe success",
			           target_mhz);
			return CIPHER_RT_VOLT_DEGRADED;
		}
	}

	atomic_store(&g_volt.locked_mhz, (unsigned)target_mhz);
	atomic_store(&g_volt.mode, CIPHER_RT_VOLT_ACTIVE);
	cipher_log("VOLT: ACTIVE — locked GPU clock to %d MHz via path=%s; "
	           "atexit + SIG{TERM,INT,SEGV,ABRT,BUS} handlers installed",
	           target_mhz,
	           path == CIPHER_RT_VOLT_PATH_NVML ? "NVML" : "kmod-ioctl");
	return CIPHER_RT_VOLT_ACTIVE;
}

int cipher_rt_volt_mode(void)
{
	return atomic_load(&g_volt.mode);
}

unsigned int cipher_rt_volt_locked_mhz(void)
{
	return atomic_load(&g_volt.locked_mhz);
}

const char *cipher_rt_volt_status_string(void)
{
	switch (atomic_load(&g_volt.mode)) {
	case CIPHER_RT_VOLT_OFF:      return "OFF";
	case CIPHER_RT_VOLT_ACTIVE:   return "ACTIVE";
	case CIPHER_RT_VOLT_DEGRADED: return "DEGRADED";
	}
	return "UNKNOWN";
}
