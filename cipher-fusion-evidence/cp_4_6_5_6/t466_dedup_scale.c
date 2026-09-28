/* CP 4.6.6 Arm 1 — cross-tenant cuIpc dedup substrate validation at scale.
 *
 * Pushes the T4.6.4 cross-process dedup path (verified at 2 procs) to N
 * procs (target 100). Each child is a lightweight SUBSTRATE CLIENT — it
 * cuInits, opens the kmod dedup path, and puts a trace-derived page set:
 *   - PREFIX shared pages: ids [1..P], identical across ALL children =>
 *     content-identical => cross-tenant dedup HITs (the shared Claude Code
 *     system+tool prefix from T4.6.5).
 *   - UNIQUE pages: ids offset per child => distinct content => MISSes.
 * No 4 GB KV cache per proc — just substrate-client memory.
 *
 * Parent reads the kmod's global CIPHER_KVDEDUP_STATS (entries =
 * distinct physical pages, virtual_refs = total refs) => achieved
 *   m = virtual_refs / entries,  compared to T4.6.5's 1.81x at 32K.
 *
 * Also: per-proc RSS, ioctl/put throughput, and (Arm 3) an isolation
 * probe — an attacker child tries a raw cuMemcpyDtoH on a victim child's
 * device pointer; cuIpc context scoping must make that fail.
 *
 * Build: gcc -O2 -I<phase4> -I<kmod> -o t466_dedup_scale t466_dedup_scale.c \
 *          <phase4>/cipher_rt_kv_alloc.c -lcuda -lcrypto -lpthread
 * Usage: ./t466_dedup_scale <nprocs> <prefix_pages> <unique_pages>
 */
#include "cipher_rt_kv_alloc.h"
#include "cipher_kvdedup.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <time.h>
#include <errno.h>

#define PAGE (2 * 1024 * 1024)
#define POOL (4ULL << 30)        /* VA reservation per proc (lazy commit) */
#define MAXP 4096

enum { RC_OK = 0, RC_CUINIT = 20, RC_ALLOC = 21, RC_DEDUP = 22, RC_PUTFAIL = 23 };

/* result a child writes back to the parent */
struct cres {
	int    proc;
	int    puts_ok;
	int    puts_fail;
	long   rss_kib;            /* per-proc resident memory */
	double put_s;              /* wall time of the put phase */
	unsigned long long victim_devptr;  /* proc 0 only: a page ptr */
	int    isolation_done;     /* attacker only */
	int    isolation_blocked;  /* attacker only: 1 = unauthorized read failed (good) */
};

static void synth(uint64_t pid, uint8_t *buf)
{
	uint64_t x = pid * 0x9E3779B97F4A7C15ULL + 0x123456789ULL;
	uint64_t *w = (uint64_t *)buf;
	for (size_t i = 0; i < PAGE / 8; i++) {
		x += 0x9E3779B97F4A7C15ULL;
		uint64_t z = x;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
		w[i] = z ^ (z >> 31);
	}
}

static long self_rss_kib(void)
{
	FILE *f = fopen("/proc/self/status", "r");
	if (!f) return -1;
	char line[256];
	long kib = -1;
	while (fgets(line, sizeof line, f))
		if (sscanf(line, "VmRSS: %ld kB", &kib) == 1) break;
	fclose(f);
	return kib;
}

static double now_s(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

/* child: put PREFIX shared ids [1..P] + UNIQUE ids [10000000 + proc*M ..].
 * proc 0 reports a victim devptr; the attacker (proc 1) probes isolation. */
static int child_main(int proc, int P, int U, int wfd)
{
	struct cres cr;
	memset(&cr, 0, sizeof cr);
	cr.proc = proc;

	CUcontext ctx; CUdevice dev;
	if (cuInit(0) || cuDeviceGet(&dev, 0) ||
	    cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx)) {
		cr.puts_fail = -1;
		(void)!write(wfd, &cr, sizeof cr);
		return RC_CUINIT;
	}
	if (cipher_rt_kv_alloc_init(POOL) != 0) {
		cr.puts_fail = -2;
		(void)!write(wfd, &cr, sizeof cr);
		return RC_ALLOC;
	}
	if (cipher_rt_kv_dedup_init() != 0) {
		cr.puts_fail = -3;
		(void)!write(wfd, &cr, sizeof cr);
		return RC_DEDUP;
	}

	uint8_t *content = malloc(PAGE);
	unsigned long long first_dp = 0;
	double t0 = now_s();
	for (int i = 0; i < P + U && i < MAXP; i++) {
		uint64_t id = (i < P) ? (uint64_t)(1 + i)             /* shared */
		                      : (uint64_t)(10000000ULL +      /* unique */
		                                   (uint64_t)proc * 100000ULL +
		                                   (uint64_t)(i - P));
		synth(id, content);
		unsigned long long dp = 0;
		if (cipher_rt_kv_dedup_put(content, &dp) != 0) {
			cr.puts_fail++;
		} else {
			cr.puts_ok++;
			if (!first_dp) first_dp = dp;
		}
	}
	cr.put_s = now_s() - t0;
	cr.rss_kib = self_rss_kib();
	if (proc == 0)
		cr.victim_devptr = first_dp;

	/* Isolation (Arm 3) is exercised by the dedicated t466_isolation
	 * harness — a foreign-pointer cuPointerGetAttribute probe, which is
	 * non-destructive (a raw illegal DtoH can abort the CUDA context). */

	(void)!write(wfd, &cr, sizeof cr);
	free(content);
	/* stay alive briefly so pages remain registered while parent reads
	 * kmod stats, then exit (release fop tears down refs) */
	char tmp;
	(void)!read(0, &tmp, 1);   /* blocks until parent closes our stdin */
	return RC_OK;
}

static struct cipher_kvdedup_stats kmod_stats(void)
{
	struct cipher_kvdedup_stats s;
	memset(&s, 0, sizeof s);
	int fd = open("/dev/" CIPHER_KVDEDUP_DEV_NAME, O_RDWR);
	if (fd < 0) { perror("open kvdedup"); return s; }
	struct cipher_kvdedup_init ini;
	memset(&ini, 0, sizeof ini);
	ioctl(fd, CIPHER_KVDEDUP_INIT, &ini);
	ioctl(fd, CIPHER_KVDEDUP_STATS, &s);
	close(fd);
	return s;
}

int main(int argc, char **argv)
{
	int N = (argc > 1) ? atoi(argv[1]) : 100;
	int P = (argc > 2) ? atoi(argv[2]) : 30;
	int U = (argc > 3) ? atoi(argv[3]) : 34;
	printf("=== CP 4.6.6 Arm 1 — cuIpc dedup substrate at %d procs ===\n", N);
	printf("per-proc: %d shared-prefix + %d unique pages "
	       "(2 MiB each)\n", P, U);

	pid_t *pids = calloc(N, sizeof(pid_t));
	int   *rfd  = calloc(N, sizeof(int));
	int   *cin  = calloc(N, sizeof(int));   /* parent->child stdin keepalive */
	double spawn_t0 = now_s();
	int spawned = 0;

	for (int i = 0; i < N; i++) {
		int rp[2], sp[2];
		if (pipe(rp) || pipe(sp)) { perror("pipe"); break; }
		pid_t pid = fork();
		if (pid < 0) {
			printf("  fork FAILED at proc %d: %s — substrate/OS "
			       "scaling ceiling\n", i, strerror(errno));
			break;
		}
		if (pid == 0) {
			close(rp[0]); close(sp[1]);
			dup2(sp[0], 0);
			_exit(child_main(i, P, U, rp[1]));
		}
		close(rp[1]); close(sp[0]);
		pids[i] = pid; rfd[i] = rp[0]; cin[i] = sp[1];
		spawned++;
	}
	double spawn_s = now_s() - spawn_t0;
	printf("  spawned %d/%d procs in %.2f s\n", spawned, N, spawn_s);

	/* collect each child's put-phase result */
	struct cres *res = calloc(N, sizeof(struct cres));
	int ok_procs = 0, total_puts = 0, iso_done = 0, iso_blocked = 0;
	long rss_sum = 0;
	double put_s_max = 0;
	for (int i = 0; i < spawned; i++) {
		struct cres cr;
		ssize_t r = read(rfd[i], &cr, sizeof cr);
		if (r != (ssize_t)sizeof cr) {
			printf("  proc %d: no result (died early) — ceiling\n", i);
			continue;
		}
		res[i] = cr;
		if (cr.puts_ok > 0) {
			ok_procs++;
			total_puts += cr.puts_ok;
			rss_sum += (cr.rss_kib > 0) ? cr.rss_kib : 0;
			if (cr.put_s > put_s_max) put_s_max = cr.put_s;
		} else {
			printf("  proc %d: put phase failed (code %d)\n",
			       i, cr.puts_fail);
		}
		if (cr.isolation_done) {
			iso_done = 1;
			iso_blocked = cr.isolation_blocked;
		}
	}

	struct cipher_kvdedup_stats ks = kmod_stats();
	double m = ks.entries ? (double)ks.virtual_refs / (double)ks.entries
	                      : 0.0;

	printf("\n--- RESULTS ---\n");
	printf("  procs spawned / put-ok      : %d / %d\n", spawned, ok_procs);
	printf("  total dedup_put (ok)        : %d\n", total_puts);
	printf("  kmod entries (phys pages)   : %llu\n",
	       (unsigned long long)ks.entries);
	printf("  kmod virtual_refs           : %llu\n",
	       (unsigned long long)ks.virtual_refs);
	printf("  achieved m (refs/entries)   : %.3fx  "
	       "(T4.6.5 32K prediction 1.81x)\n", m);
	printf("  per-proc RSS mean           : %.1f MiB\n",
	       ok_procs ? rss_sum / 1024.0 / ok_procs : 0.0);
	printf("  put-phase wall (slowest)    : %.2f s  -> %.0f puts/s\n",
	       put_s_max, put_s_max > 0 ? total_puts / put_s_max : 0.0);
	printf("  isolation probe (Arm 3)     : %s\n",
	       !iso_done ? "not run" :
	       iso_blocked ? "PASS — unauthorized cross-proc DtoH blocked"
	                   : "FAIL — unauthorized read SUCCEEDED");

	/* write JSON */
	FILE *jf = fopen("/home/ubuntu/cipher-fusion-evidence/cp_4_6_5_6/"
	                 "t466_dedup_scale_result.json", "w");
	if (jf) {
		fprintf(jf,
		    "{\n  \"cp\": \"4.6.6\", \"arm\": \"1 substrate-scale\",\n"
		    "  \"procs_requested\": %d, \"procs_spawned\": %d, "
		    "\"procs_put_ok\": %d,\n"
		    "  \"total_puts\": %d, \"kmod_entries\": %llu, "
		    "\"kmod_virtual_refs\": %llu,\n"
		    "  \"achieved_m\": %.4f, \"t465_predicted_m_32k\": 1.81,\n"
		    "  \"per_proc_rss_mib_mean\": %.2f,\n"
		    "  \"put_throughput_per_s\": %.1f,\n"
		    "  \"isolation_probe\": \"%s\",\n"
		    "  \"scaling_ceiling_hit\": %s\n}\n",
		    N, spawned, ok_procs, total_puts,
		    (unsigned long long)ks.entries,
		    (unsigned long long)ks.virtual_refs, m,
		    ok_procs ? rss_sum / 1024.0 / ok_procs : 0.0,
		    put_s_max > 0 ? total_puts / put_s_max : 0.0,
		    !iso_done ? "not-run" :
		        iso_blocked ? "PASS" : "FAIL",
		    (ok_procs < N) ? "true" : "false");
		fclose(jf);
		printf("  wrote t466_dedup_scale_result.json\n");
	}

	/* release children */
	for (int i = 0; i < spawned; i++) close(cin[i]);
	for (int i = 0; i < spawned; i++) waitpid(pids[i], NULL, 0);
	return (ok_procs == N) ? 0 : 1;
}
