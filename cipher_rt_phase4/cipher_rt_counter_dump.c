/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_counter_dump.c -- V1 Phase A.1 substrate counter dump for
 * the no-plugin verification harness.
 *
 * Port of cipher_vllm_plugin/cipher_vllm_kv.py:_install_counter_dump_for_
 * verification at lines 709-792. Same counter set, same tag/snapshot
 * pattern, same JSON format. Difference: lives in libcipher_rt.so so it
 * fires under LD_PRELOAD only (no plugin install required).
 *
 * Per V1_PHASE_A_SCOPE_LOCK.md §4.A.1, scope-locked at ~120 LOC plus
 * ~0.5 ED for self-test. Env-gated by CIPHER_RT_COUNTER_DUMP_PATH; zero
 * production overhead when unset.
 *
 * Anil 2026-05-26 caveats addressed in implementation:
 *   - SIGUSR1 collision: sigaction with sa_flags=0 stores previous
 *     handler; our handler chains to it after dumping. vLLM debug path
 *     (if any) is preserved.
 *   - atexit ordering: constructor runs early so atexit registration is
 *     early so our handler runs late at teardown (LIFO). libcudart
 *     cleanup completes before us. We only read substrate counters and
 *     write JSON; no CUDA touch at teardown.
 *   - SIGUSR2 / atexit-only fallbacks: documented at scope-lock §3
 *     Anil caveat. NOT yet enabled; we ship SIGUSR1+atexit by default
 *     per scope-lock §4.A.1; switch surfaces to Anil before changing.
 */

#define _GNU_SOURCE
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_counter_dump.h"

/* Substrate counter T-symbols. These are all extern "C" unsigned long
 * fn(void) per their .c / .cpp definitions (cipher_rt_cublas_shim.c:222,
 * cipher_rt_matmul_dispatch.c:118, cipher_rt_koopman_engine.cpp:250+). */
extern unsigned long cipher_rt_cublas_shim_calls(void);
extern unsigned long cipher_rt_matmul_calls_total(void);
extern unsigned long cipher_rt_matmul_calls_handled(void);
extern unsigned long cipher_rt_matmul_calls_passthrough(void);
extern unsigned long cipher_rt_koopman_calls_total(void);
extern unsigned long cipher_rt_koopman_calls_handled(void);
extern unsigned long cipher_rt_koopman_calls_skipped(void);
extern unsigned long cipher_rt_koopman_skip_dtype(void);
extern unsigned long cipher_rt_koopman_skip_dim(void);
extern unsigned long cipher_rt_koopman_skip_nullptr(void);

static pthread_once_t g_dump_once = PTHREAD_ONCE_INIT;
static int g_dump_install_rc;
static char g_dump_path[PATH_MAX];
static struct sigaction g_prev_sigusr1;
static int g_have_prev_sigusr1;

/* Write a JSON snapshot of all substrate counters to g_dump_path with the
 * given tag. Format mirrors plugin's _snapshot at cipher_vllm_kv.py:766-786.
 * Single-process file write; no locking (read pattern is harness reads
 * after worker exits, so atexit dump is the final write). */
static void cipher_rt_counter_dump_write(const char *tag)
{
	FILE *f;
	double now;
	struct timespec ts;

	if (g_dump_path[0] == '\0')
		return;

	f = fopen(g_dump_path, "w");
	if (!f) {
		cipher_log("counter-dump: fopen failed");
		return;
	}

	if (clock_gettime(CLOCK_REALTIME, &ts) == 0)
		now = (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
	else
		now = 0.0;

	fprintf(f,
		"{"
		"\"tag\": \"%s\","
		"\"pid\": %d,"
		"\"time\": %.6f,"
		"\"cipher_rt_cublas_shim_calls\": %lu,"
		"\"cipher_rt_matmul_calls_total\": %lu,"
		"\"cipher_rt_matmul_calls_handled\": %lu,"
		"\"cipher_rt_matmul_calls_passthrough\": %lu,"
		"\"cipher_rt_koopman_calls_total\": %lu,"
		"\"cipher_rt_koopman_calls_handled\": %lu,"
		"\"cipher_rt_koopman_calls_skipped\": %lu,"
		"\"cipher_rt_koopman_skip_dtype\": %lu,"
		"\"cipher_rt_koopman_skip_dim\": %lu,"
		"\"cipher_rt_koopman_skip_nullptr\": %lu"
		"}",
		tag,
		(int)getpid(),
		now,
		cipher_rt_cublas_shim_calls(),
		cipher_rt_matmul_calls_total(),
		cipher_rt_matmul_calls_handled(),
		cipher_rt_matmul_calls_passthrough(),
		cipher_rt_koopman_calls_total(),
		cipher_rt_koopman_calls_handled(),
		cipher_rt_koopman_calls_skipped(),
		cipher_rt_koopman_skip_dtype(),
		cipher_rt_koopman_skip_dim(),
		cipher_rt_koopman_skip_nullptr());

	fclose(f);
}

/* atexit hook: late at teardown after libcudart cleanup. Writes "atexit"
 * tag. */
static void cipher_rt_counter_dump_atexit(void)
{
	cipher_rt_counter_dump_write("atexit");
}

/* SIGUSR1 handler: live-read trigger. Writes "sigusr1" tag, then chains
 * to any previously-installed handler (preserves vLLM debug-path use of
 * SIGUSR1 if any). */
static void cipher_rt_counter_dump_sigusr1(int sig, siginfo_t *info, void *uc)
{
	cipher_rt_counter_dump_write("sigusr1");

	if (g_have_prev_sigusr1) {
		if (g_prev_sigusr1.sa_flags & SA_SIGINFO) {
			if (g_prev_sigusr1.sa_sigaction)
				g_prev_sigusr1.sa_sigaction(sig, info, uc);
		} else {
			if (g_prev_sigusr1.sa_handler &&
			    g_prev_sigusr1.sa_handler != SIG_DFL &&
			    g_prev_sigusr1.sa_handler != SIG_IGN)
				g_prev_sigusr1.sa_handler(sig);
		}
	}
}

static void cipher_rt_counter_dump_init_once(void)
{
	const char *dir = getenv("CIPHER_RT_COUNTER_DUMP_PATH");
	struct sigaction sa;
	int n;

	if (!dir || dir[0] == '\0') {
		g_dump_install_rc = 0; /* env unset, no-op success */
		return;
	}

	if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
		cipher_log("counter-dump: mkdir failed");
		g_dump_install_rc = -1;
		return;
	}

	n = snprintf(g_dump_path, sizeof(g_dump_path),
		     "%s/cipher_rt_%d.json", dir, (int)getpid());
	if (n <= 0 || (size_t)n >= sizeof(g_dump_path)) {
		cipher_log("counter-dump: path too long");
		g_dump_path[0] = '\0';
		g_dump_install_rc = -1;
		return;
	}

	if (atexit(cipher_rt_counter_dump_atexit) != 0) {
		cipher_log("counter-dump: atexit registration failed");
		g_dump_install_rc = -1;
		return;
	}

	memset(&sa, 0, sizeof(sa));
	sa.sa_sigaction = cipher_rt_counter_dump_sigusr1;
	sa.sa_flags = SA_SIGINFO | SA_RESTART;
	sigemptyset(&sa.sa_mask);

	if (sigaction(SIGUSR1, &sa, &g_prev_sigusr1) != 0) {
		cipher_log("counter-dump: sigaction SIGUSR1 failed");
		g_dump_install_rc = -1;
		return;
	}
	g_have_prev_sigusr1 = 1;

	cipher_rt_counter_dump_write("install");

	cipher_log("counter-dump: installed at"
		   " CIPHER_RT_COUNTER_DUMP_PATH (atexit + SIGUSR1)");
	g_dump_install_rc = 0;
}

int cipher_rt_counter_dump_init(void)
{
	pthread_once(&g_dump_once, cipher_rt_counter_dump_init_once);
	return g_dump_install_rc;
}
