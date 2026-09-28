/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_classify_observer.c -- Week 2 Step 4 stub.
 *
 * Atomic-counter observer for the classify substrate. Hot-path safe:
 * no locks, no allocation, single-instance static storage initialized
 * to zero (BSS).
 *
 * Step 4 stub: data structures + 3 API functions exposed. Nothing
 * calls observer_observe() yet — Step 6 wires it from
 * cipher_rt_classify_route() and Step 5 /proc/cipher/classify_stats
 * pulls from observer_snapshot() via a kmod-side ioctl bridge.
 */
#include <stdint.h>
#include <stdatomic.h>
#include <stddef.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <errno.h>

#include "cipher_rt_classify_observer.h"
#include "cipher_rt_ring_write.h"     /* W10-12 Step 1: RING_WRITE producer */
#include "cipher_rt_classify_substrate.h"
#include "cipher_rt_dispatch.h"    /* Week 3 Step 2: lookup() for hint publish */
#include "cipher_rt_oracle_bridge.h" /* Week 4 Step 1: real cipher_oracle_decide */
#include "cipher_ioctl.h"     /* CIPHER_PUSH_CLASSIFY_STATS, struct payload */
#include "cipher_v2_internal.h"  /* cipher_log */

/* BSS-resident; all counters zero at .so load. */
static struct cipher_rt_classify_observer_stats g_stats;

/* Week 3 Step 2 — per-thread substitute hint storage. __thread is
 * zero-initialized at thread creation; observe() writes it on every
 * call (sets valid=1 + action + actuator_hint when GEMM + permit +
 * non-PASS_THROUGH; sets valid=0 otherwise). Marlin reads via
 * cipher_rt_classify_observer_get_tls_hint() in Step 3.
 *
 * No locks: TLS is per-thread by definition; observe() and any
 * consumer in the same thread serialize through the natural
 * call-return sequence (CUPTI callback → observe → actuator). */
static __thread struct cipher_rt_substitute_hint g_tls_hint;

int cipher_rt_classify_observer_init(void)
{
	/* Step 4 stub: no side effects. Counters are statically zero.
	 * Returns 0 for API consistency with future revisions that may
	 * need real init work. Idempotent. */
	return 0;
}

void cipher_rt_classify_observer_observe(
	const struct cipher_rt_classify_call *call,
	const struct cipher_rt_classify_out  *out,
	int                                   result)
{
	uint8_t op;

	(void)call;
	if (!out) return;

	/* Cast to atomic_ullong* — uint64_t and atomic_ullong are
	 * layout-compatible on Linux x86_64. We use __atomic_*
	 * intrinsics for portability with C11. */
	__atomic_fetch_add(&g_stats.total_classifications, 1, __ATOMIC_RELAXED);

	if (result == CIPHER_RT_CLASSIFY_HANDLED) {
		__atomic_fetch_add(&g_stats.handled, 1, __ATOMIC_RELAXED);
	} else {
		__atomic_fetch_add(&g_stats.passthrough, 1, __ATOMIC_RELAXED);
	}

	/* Per-op-class bucket. UNCLASSIFIED (0xFF) is bucketed at
	 * slot 15 (the last reserved slot) to keep the array bounds
	 * tight at 16. OpClass 0..6 land in their natural slots;
	 * 7..14 are reserved for future enum extensions; 15 is
	 * the unclassified bucket. */
	op = out->op_class;
	if (op >= 16) {
		op = 15;  /* UNCLASSIFIED → slot 15 */
	}
	__atomic_fetch_add(&g_stats.per_op_class_counts[op], 1, __ATOMIC_RELAXED);

	/* W10-12 Step 1 RING_WRITE producer: emit one event per observe()
	 * call. tenant_id=0 (single-tenant default); W11 Step 2 will wire
	 * the stream-keyed resolver to fill the real tenant_id. Payload
	 * carries the op_class + confidence + result tuple. */
	{
		struct cipher_rt_classify_event {
			uint32_t op_class;
			uint32_t confidence_x10000;
			uint32_t result;
			uint32_t _pad;
		} ev = {
			.op_class          = (uint32_t)op,
			.confidence_x10000 = (uint32_t)(out->confidence * 10000.0f),
			.result            = (uint32_t)result,
			._pad              = 0,
		};
		cipher_rt_ring_write(0u,
		                     CIPHER_RT_RING_EVENT_CLASSIFY,
		                     /* subtype = result */ (uint32_t)result,
		                     /* commit_seq */ 0u,
		                     &ev, sizeof(ev));
	}

	/* Week 3 Step 2 + Week 4 Step 1 — publish substitute hint into
	 * per-thread TLS.
	 *
	 * Trigger conditions: result == HANDLED (classifier produced a
	 * verdict) AND op_class == GEMM (0; only routable class in v1)
	 * AND cipher_oracle_decide() returns PERMIT AND dispatch_lookup()
	 * returns a non-PASS_THROUGH action.
	 *
	 * Week 4 Step 1 wires real cipher_oracle_decide() via the
	 * cipher_rt_oracle_bridge (Q2 selective-init approach b). The
	 * bridge owns the BSS-resident CipherOracleState and a CAS
	 * lazy-init flag; cipher_oracle_init(state, NULL, NULL) is
	 * pure-function safe (oracle.cpp:80-119 + structural_lookup.cpp
	 * :169-188; no CUDA/NVML/kmod) so this is compatible with the
	 * cipher_rt_phase4 deferred-init model.
	 *
	 * Behavior delta vs Week 3 Step 2: the oracle holds DENY in
	 * warmup phase (detected_phase == 0). topo_detect_inference
	 * (oracle.cpp:30-74) flips to CONVERGENCE at the first 64-decision
	 * checkpoint on a pure inference workload (only GEMM/ELEMENTWISE/
	 * REDUCTION classes), after which PERMIT is the default. Hint
	 * absence during warmup falls back to Marlin's own L100-114 gate
	 * (binding), preserving correctness. CIPHER_FORCE_PERMIT=1
	 * remains an env-level override (oracle.cpp:299-308). */
	{
		const uint8_t OPCLASS_GEMM = 0;
		int permit;

		cipher_rt_oracle_bridge_init_lazy();
		permit = cipher_rt_oracle_bridge_decide(out->op_class,
		                                        out->confidence);

		if (result == CIPHER_RT_CLASSIFY_HANDLED
		    && out->op_class == OPCLASS_GEMM
		    && permit) {
			struct cipher_rt_dispatch_decision decision;
			int rc = cipher_rt_dispatch_lookup(out->op_class, &decision);
			if (rc == 0
			    && decision.action != CIPHER_RT_DISPATCH_PASS_THROUGH) {
				g_tls_hint.action        = decision.action;
				g_tls_hint.actuator_hint = decision.actuator_hint;
				g_tls_hint.valid         = 1;
				return;
			}
		}
	}
	g_tls_hint.valid = 0;
}

const struct cipher_rt_classify_observer_stats *
cipher_rt_classify_observer_snapshot(void)
{
	/* Returns pointer to the live struct. Readers should treat
	 * fields as atomic loads (the struct's lifetime is the .so
	 * lifetime; counters monotonically increase). */
	return &g_stats;
}

/* Week 3 Step 2 — TLS substitute_hint getter. Returns the
 * thread-local hint last published by observe(). Marlin actuator
 * (Step 3) reads `valid` first; if 1, consumes `action` and
 * `actuator_hint`. No locks; no allocation; hot-path safe. */
const struct cipher_rt_substitute_hint *
cipher_rt_classify_observer_get_tls_hint(void)
{
	return &g_tls_hint;
}

/* Week 2 Step 6 — push observer snapshot to kmod via ioctl nr=25.
 * SET semantics (kmod-side counters replaced with current observer
 * values). Lazy-open /dev/cipher on first push, cache fd thereafter.
 * Returns 0 on success, -errno on failure. Failure does NOT propagate
 * up the launch path — caller logs and continues. */
static int          g_kmod_fd      = -1;
static int          g_kmod_fd_init = 0;
static pthread_mutex_t g_kmod_lock = PTHREAD_MUTEX_INITIALIZER;

int cipher_rt_classify_push_to_kmod(void)
{
	struct cipher_classify_stats_push push;
	int i, rc;

	pthread_mutex_lock(&g_kmod_lock);
	if (!g_kmod_fd_init) {
		g_kmod_fd = open("/dev/cipher", O_RDWR);
		g_kmod_fd_init = 1;
		if (g_kmod_fd < 0) {
			cipher_log("CLASSIFY: push_to_kmod open /dev/cipher failed "
			           "(errno=%d); pushes disabled", errno);
		}
	}
	pthread_mutex_unlock(&g_kmod_lock);

	if (g_kmod_fd < 0) return -1;

	/* 1:1 atomic-snapshot copy. Reads use __atomic_load explicit to
	 * pair with __atomic_fetch_add writers in observer_observe(). */
	push.total       = __atomic_load_n(&g_stats.total_classifications, __ATOMIC_RELAXED);
	push.handled     = __atomic_load_n(&g_stats.handled,               __ATOMIC_RELAXED);
	push.passthrough = __atomic_load_n(&g_stats.passthrough,           __ATOMIC_RELAXED);
	for (i = 0; i < 16; i++) {
		push.per_op_class[i] =
			__atomic_load_n(&g_stats.per_op_class_counts[i], __ATOMIC_RELAXED);
	}
	memset(push.reserved, 0, sizeof(push.reserved));

	rc = ioctl(g_kmod_fd, CIPHER_PUSH_CLASSIFY_STATS, &push);
	if (rc < 0) {
		cipher_log("CLASSIFY: ioctl CIPHER_PUSH_CLASSIFY_STATS errno=%d", errno);
		return -1;
	}
	return 0;
}
