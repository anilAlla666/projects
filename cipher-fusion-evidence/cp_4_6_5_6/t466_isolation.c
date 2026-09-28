/* CP 4.6.6 Arm 3 — cross-tenant isolation security test.
 *
 * Two independent tenant processes (separate CUDA contexts). The victim
 * allocates a real dedup page through the substrate and hands the attacker
 * its raw device pointer value. The attacker — which never imported that
 * page via the authorized cuIpc handle path — must NOT be able to touch it.
 *
 * Probe: cuPointerGetAttribute on the foreign pointer (non-destructive — a
 * raw illegal cuMemcpyDtoH can abort the CUDA context, so we query, not
 * copy). A foreign device VA must resolve to nothing in the attacker's
 * context. PASS = every unauthorized probe is rejected.
 *
 * Build: gcc -O2 -I<phase4> -I<kmod> -o t466_isolation t466_isolation.c \
 *          <phase4>/cipher_rt_kv_alloc.c -lcuda -lcrypto -lpthread
 */
#include "cipher_rt_kv_alloc.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>

#define PAGE (2 * 1024 * 1024)
#define POOL (4ULL << 30)

static void synth(uint64_t pid, uint8_t *buf)
{
	uint64_t x = pid * 0x9E3779B97F4A7C15ULL + 0x123456789ULL;
	uint64_t *w = (uint64_t *)buf;
	for (size_t i = 0; i < PAGE / 8; i++) {
		x += 0x9E3779B97F4A7C15ULL;
		uint64_t z = x;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
		w[i] = z ^ (z >> 27);
	}
}

static int cuda_up(void)
{
	CUcontext ctx; CUdevice dev;
	return !(cuInit(0) || cuDeviceGet(&dev, 0) ||
	         cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx));
}

int main(void)
{
	printf("=== CP 4.6.6 Arm 3 — cross-tenant isolation ===\n");
	int v2a[2], a2v[2];
	if (pipe(v2a) || pipe(a2v)) { perror("pipe"); return 1; }

	pid_t vic = fork();
	if (vic == 0) {
		/* ---- VICTIM ---- */
		close(v2a[0]); close(a2v[1]);
		if (!cuda_up() || cipher_rt_kv_alloc_init(POOL) ||
		    cipher_rt_kv_dedup_init()) _exit(20);
		uint8_t *c = malloc(PAGE);
		synth(424242, c);
		unsigned long long dp = 0;
		if (cipher_rt_kv_dedup_put(c, &dp) != 0) _exit(21);
		/* hand the attacker our raw device pointer */
		if (write(v2a[1], &dp, sizeof dp) != (ssize_t)sizeof dp) _exit(22);
		char go;
		(void)!read(a2v[0], &go, 1);   /* wait for attacker to finish */
		_exit(0);
	}

	pid_t att = fork();
	if (att == 0) {
		/* ---- ATTACKER (separate process, separate CUDA context) ---- */
		close(v2a[1]); close(a2v[0]);
		if (!cuda_up()) _exit(30);
		unsigned long long foreign = 0;
		if (read(v2a[0], &foreign, sizeof foreign) != (ssize_t)sizeof foreign)
			_exit(31);
		printf("  attacker: received victim devptr 0x%llx\n", foreign);

		int violations = 0, probes = 0;

		/* probe 1: query memory type of the foreign pointer */
		CUmemorytype mt;
		probes++;
		CUresult r1 = cuPointerGetAttribute(&mt,
		                CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
		                (CUdeviceptr)foreign);
		printf("  probe 1 (MEMORY_TYPE on foreign ptr): %s\n",
		       r1 == CUDA_SUCCESS ? "RESOLVED — VIOLATION" : "rejected (ok)");
		if (r1 == CUDA_SUCCESS) violations++;

		/* probe 2: query the owning context of the foreign pointer */
		CUcontext fc;
		probes++;
		CUresult r2 = cuPointerGetAttribute(&fc,
		                CU_POINTER_ATTRIBUTE_CONTEXT,
		                (CUdeviceptr)foreign);
		printf("  probe 2 (CONTEXT on foreign ptr): %s\n",
		       r2 == CUDA_SUCCESS ? "RESOLVED — VIOLATION" : "rejected (ok)");
		if (r2 == CUDA_SUCCESS) violations++;

		/* probe 3: range query — can the attacker discover the
		 * allocation's base/size (a metadata leak)? */
		CUdeviceptr base; size_t sz;
		probes++;
		CUresult r3 = cuMemGetAddressRange(&base, &sz,
		                                   (CUdeviceptr)foreign);
		printf("  probe 3 (address-range on foreign ptr): %s\n",
		       r3 == CUDA_SUCCESS ? "RESOLVED — VIOLATION" : "rejected (ok)");
		if (r3 == CUDA_SUCCESS) violations++;

		FILE *jf = fopen("/home/ubuntu/cipher-fusion-evidence/cp_4_6_5_6/"
		                 "t466_isolation_result.json", "w");
		if (jf) {
			fprintf(jf,
			    "{\n  \"cp\": \"4.6.6\", \"arm\": \"3 isolation\",\n"
			    "  \"unauthorized_probes\": %d,\n"
			    "  \"violations\": %d,\n"
			    "  \"verdict\": \"%s\",\n"
			    "  \"note\": \"victim raw device pointer probed from a "
			    "separate process/CUDA context; the only authorized "
			    "cross-tenant path is the substrate cuIpc handle import\"\n}\n",
			    probes, violations,
			    violations == 0 ? "PASS — 0 cross-tenant access"
			                    : "FAIL — isolation violation");
			fclose(jf);
		}
		printf("  --- isolation %s (%d/%d probes rejected) ---\n",
		       violations == 0 ? "PASS" : "FAIL",
		       probes - violations, probes);
		fflush(stdout);   /* _exit skips stdio flush — make the probe log durable */
		(void)!write(a2v[1], "x", 1);
		_exit(violations == 0 ? 0 : 1);
	}

	close(v2a[0]); close(v2a[1]); close(a2v[0]); close(a2v[1]);
	int vs = 0, as = 0;
	waitpid(vic, &vs, 0);
	waitpid(att, &as, 0);
	return WIFEXITED(as) ? WEXITSTATUS(as) : 1;
}
