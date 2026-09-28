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
#include "cipher_workload_detect.h"  /* W.1: read CipherProfile.volt_engage */

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
	/* W.1 (2026-05-27): classifier-driven engagement counters + state.
	 * engage_lock serializes the ARMED -> ACTIVE transition so concurrent
	 * timer/observer threads don't double-apply the NVML lock. */
	atomic_ulong       classifier_engagements;
	atomic_ulong       classifier_skipped;
	atomic_int         engage_lock;
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

/* W.1 (2026-05-27): factor path resolution + handler install so both the
 * env-driven and the classifier-driven flows can share them. */
static int resolve_and_install_handlers(void)
{
	int probe_rc;
	int path = CIPHER_RT_VOLT_PATH_NONE;

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
	if (path == CIPHER_RT_VOLT_PATH_NONE) {
		if (probe_kmod_actuation() == 0) {
			path = CIPHER_RT_VOLT_PATH_KMOD_IOCTL;
			cipher_log("VOLT: kmod ioctl path available (CIPHER_SET_CLOCK_MHZ)");
		}
	}
	if (path == CIPHER_RT_VOLT_PATH_NONE) {
		return CIPHER_RT_VOLT_PATH_NONE;
	}
	atomic_store(&g_volt.path, path);
	install_signal_handlers();
	return path;
}

/* W.1: apply the lock at target_mhz via whichever path was resolved. Used by
 * both env-driven init and classifier_poll. Returns 0 on success.
 * Caller is responsible for ensuring path != NONE. */
static int apply_lock_at_target(int target_mhz)
{
	int path = atomic_load(&g_volt.path);
	if (path == CIPHER_RT_VOLT_PATH_NVML) {
		if (g_volt.fn_set_locked(g_volt.dev,
		                         (unsigned)target_mhz,
		                         (unsigned)target_mhz) != NVML_SUCCESS) {
			cipher_log("VOLT: NVML set_locked_clocks(%d,%d) failed",
			           target_mhz, target_mhz);
			return -1;
		}
	} else if (path == CIPHER_RT_VOLT_PATH_KMOD_IOCTL) {
		if (kmod_ioctl_set_mhz((unsigned)target_mhz) < 0) {
			cipher_log("VOLT: kmod ioctl set(%d) failed", target_mhz);
			return -1;
		}
	} else {
		return -1;
	}
	atomic_store(&g_volt.locked_mhz, (unsigned)target_mhz);
	return 0;
}

int cipher_rt_volt_init(void)
{
	const char *env_mode  = getenv("CIPHER_VOLT");
	const char *env_batch = getenv("CIPHER_VOLT_BATCH");
	const char *env_mhz   = getenv("CIPHER_VOLT_MHZ");
	int target_mhz = 0;
	int path = CIPHER_RT_VOLT_PATH_NONE;

	if (atomic_exchange(&g_volt.init_done, 1)) {
		return atomic_load(&g_volt.mode);
	}

	atomic_store(&g_volt.mode, CIPHER_RT_VOLT_OFF);
	atomic_store(&g_volt.locked_mhz, 0);
	atomic_store(&g_volt.restored, 0);
	atomic_store(&g_volt.path, CIPHER_RT_VOLT_PATH_NONE);
	atomic_store(&g_volt.classifier_engagements, 0);
	atomic_store(&g_volt.classifier_skipped, 0);
	atomic_store(&g_volt.engage_lock, 0);
	g_volt.cipher_fd = -1;

	/* W.1 (2026-05-27): two-track init.
	 *   env=off/0 -> OFF (customer forced off, classifier cannot override).
	 *   env=on/1/ON + target_mhz -> existing immediate-lock behavior.
	 *   env unset or env=auto -> ARMED; classifier_poll decides at runtime.
	 *
	 * The ARMED branch resolves NVML/kmod path + installs signal handlers
	 * just like the legacy branch — only the actual clock-lock is deferred. */
	if (env_mode && (strcmp(env_mode, "off") == 0 || strcmp(env_mode, "0") == 0)) {
		cipher_log("VOLT: CIPHER_VOLT=off — staying OFF (classifier override denied)");
		return CIPHER_RT_VOLT_OFF;
	}
	int auto_mode = !env_mode || strcmp(env_mode, "auto") == 0;
	if (auto_mode) {
		if (resolve_and_install_handlers() == CIPHER_RT_VOLT_PATH_NONE) {
			atomic_store(&g_volt.mode, CIPHER_RT_VOLT_DEGRADED);
			cipher_log("VOLT: DEGRADED in auto mode — no actuation path");
			return CIPHER_RT_VOLT_DEGRADED;
		}
		atomic_store(&g_volt.mode, CIPHER_RT_VOLT_ARMED);
		cipher_log("VOLT: ARMED (auto / no env) — awaiting classifier engagement");
		return CIPHER_RT_VOLT_ARMED;
	}
	if (strcmp(env_mode, "on") != 0 && strcmp(env_mode, "1") != 0 &&
	    strcmp(env_mode, "ON") != 0) {
		cipher_log("VOLT: unrecognized CIPHER_VOLT='%s' (expected off|on|auto); staying OFF",
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

	path = resolve_and_install_handlers();
	if (path == CIPHER_RT_VOLT_PATH_NONE) {
		atomic_store(&g_volt.mode, CIPHER_RT_VOLT_DEGRADED);
		cipher_log("VOLT: DEGRADED — neither NVML nor kmod ioctl actuation path available");
		return CIPHER_RT_VOLT_DEGRADED;
	}

	if (apply_lock_at_target(target_mhz) != 0) {
		atomic_store(&g_volt.mode, CIPHER_RT_VOLT_DEGRADED);
		return CIPHER_RT_VOLT_DEGRADED;
	}
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
	case CIPHER_RT_VOLT_ARMED:    return "ARMED";
	}
	return "UNKNOWN";
}

/* W.1 (2026-05-27): classifier-driven engagement entry. Called from the
 * cipher_workload_detect classifier timer thread every 1s.
 *
 * Semantics:
 *   mode != ARMED  -> no-op (env override decided; classifier cannot flip).
 *   profile=NULL or class=UNKNOWN -> skipped (classifier hasn't settled).
 *   profile.volt_engage==0 -> skipped (classifier says wrong regime). Counter++.
 *   profile.volt_engage==1 -> apply lock at batch-derived target_mhz, transition
 *                             ARMED -> ACTIVE. Engagement counter++. */
void cipher_rt_volt_classifier_poll(void)
{
	int mode = atomic_load(&g_volt.mode);
	if (mode != CIPHER_RT_VOLT_ARMED) return;

	const struct cipher_workload_profile *p = cipher_workload_profile_get();
	if (!p || p->workload_class == CIPHER_WL_UNKNOWN) return;

	if (!p->volt_engage) {
		atomic_fetch_add(&g_volt.classifier_skipped, 1);
		return;
	}

	/* Serialize ARMED -> ACTIVE so multiple poll callers don't double-apply. */
	int expected = 0;
	if (!atomic_compare_exchange_strong(&g_volt.engage_lock, &expected, 1)) {
		return;
	}
	if (atomic_load(&g_volt.mode) != CIPHER_RT_VOLT_ARMED) {
		atomic_store(&g_volt.engage_lock, 0);
		return;
	}

	/* Pick target MHz from the dominant batch class. P3 TinyLlama-AWQ B=1 and
	 * P2 Mistral-7B B=1 -> 1000 MHz (calibrated). P1 Llama-3-8B B=8 -> 1600.
	 * Anything else with bw_bound regime gets the conservative 1000 MHz
	 * (matches cipher-t43-envelope's +57% tok/W operating point). */
	unsigned int target_mhz = 1000;
	if (p->workload_class == CIPHER_WL_A4_BATCH_INFERENCE) target_mhz = 1600;
	if (p->long_context_detected) target_mhz = 1600;

	if (apply_lock_at_target((int)target_mhz) != 0) {
		atomic_store(&g_volt.mode, CIPHER_RT_VOLT_DEGRADED);
		atomic_store(&g_volt.engage_lock, 0);
		cipher_log("VOLT: classifier engage FAILED at %u MHz -> DEGRADED", target_mhz);
		return;
	}
	atomic_store(&g_volt.mode, CIPHER_RT_VOLT_ACTIVE);
	atomic_fetch_add(&g_volt.classifier_engagements, 1);
	cipher_log("VOLT: classifier ENGAGED at %u MHz (class=%u bw_bound=%u large_prefill=%u)",
	           target_mhz,
	           (unsigned)p->workload_class,
	           (unsigned)p->bw_bound_regime_detected,
	           (unsigned)p->large_prefill_detected);
	atomic_store(&g_volt.engage_lock, 0);
}

unsigned long cipher_rt_volt_classifier_engagements(void)
{
	return (unsigned long)atomic_load(&g_volt.classifier_engagements);
}

unsigned long cipher_rt_volt_classifier_skipped(void)
{
	return (unsigned long)atomic_load(&g_volt.classifier_skipped);
}
