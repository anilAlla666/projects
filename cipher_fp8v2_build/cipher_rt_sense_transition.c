/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_sense_transition.c -- Week 3 Step 4 Option II-a invented.
 *
 * Per-tenant session-class state machine above SENSE. Detects
 * class transitions and tool-idle states; emits DSM PROPOSE
 * candidates to a bounded internal queue.
 *
 * v1 design choices:
 *   - 256-slot tenant table indexed by tenant_id % 256 (collisions
 *     evict-oldest; ok for v1 observability)
 *   - Hysteresis: track current_class observation count; require N=8
 *     consecutive observations of the same class before declaring it
 *     "stable"; require M=4 observations of a different class before
 *     declaring transition
 *   - Tool-idle: gap > K=100ms with stable_class == AGENT_AUTONOMOUS
 *   - Proposal queue: 64-slot ring buffer with atomic head; pushes
 *     are drop-oldest-on-overflow (bounded observability)
 *
 * All thresholds are v1 best-guesses. Measurement-driven tuning is
 * deferred to future weeks (Week 4 observability tier may produce
 * empirical evidence; Phase 5 CP 5.5 benchmarks are the validation).
 */
#include <stdint.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <errno.h>

#include "cipher_rt_sense_transition.h"
#include "cipher_v2_internal.h"
#include "cipher_ioctl.h"   /* CIPHER_DSM_PROPOSE + struct */

/* --- v1 tunables (best-guess; see header comment) --- */
#define SENSE_TENANT_TABLE_SIZE     256u
#define SENSE_STABILITY_N           8u
#define SENSE_TRANSITION_DEBOUNCE_M 4u
#define SENSE_AGENT_IDLE_NS         100000000ULL  /* 100 ms */
#define SENSE_PROPOSAL_QUEUE_SIZE   64u

/* --- Tenant state entry (cache-line aligned to reduce false sharing). */
struct __attribute__((aligned(64))) tenant_state {
	atomic_uint  tenant_id;          /* 0 = empty slot */
	atomic_uint  current_class;      /* latest observed class */
	atomic_uint  current_count;      /* consecutive same-class obs */
	atomic_uint  stable_class;       /* class that has persisted N times */
	atomic_uint  other_count;        /* obs of a class != stable */
	atomic_ullong last_ts_ns;        /* last observation timestamp */
	atomic_ullong fingerprint;       /* SENSE session fingerprint */
	/* W4 Step 6 — Sub-4 measurement infra: per-tenant cumulative counters. */
	atomic_uint  transitions;        /* stable-class transitions observed */
	atomic_uint  tool_idles;         /* tool-idle events (AGENT gap > 100ms) */
};

static struct tenant_state g_tenant_table[SENSE_TENANT_TABLE_SIZE];

/* W4 Step 6 — per-(from,to) transition matrix, plus aggregate idle counter.
 * 4 classes (HUMAN/AGENT/BATCH/UNKNOWN). Drives K=100ms threshold tuning
 * in Week 5+. */
#define SENSE_CLASS_COUNT 4
static atomic_uint  g_transition_matrix[SENSE_CLASS_COUNT][SENSE_CLASS_COUNT];
static atomic_ulong g_tool_idle_count;
static atomic_ulong g_transitions_total;

/* --- Proposal queue (bounded ring; drop-oldest-on-overflow). */
struct proposal_entry {
	uint32_t tenant_id;
	uint32_t source_class;
	uint32_t target_class;
	uint32_t reason;
	uint32_t confidence;
	uint32_t pad;
	uint64_t fingerprint;
	uint64_t timestamp_ns;
};

static struct proposal_entry g_proposal_queue[SENSE_PROPOSAL_QUEUE_SIZE];
static atomic_uint   g_proposal_head;   /* monotonic write index */
static atomic_ulong  g_proposals_queued;
static atomic_ulong  g_proposals_pushed;

/* --- Userspace fd for ioctl push (lazy open + mutex-guarded init). */
static int             g_kmod_fd      = -1;
static int             g_kmod_fd_init = 0;
static pthread_mutex_t g_kmod_lock    = PTHREAD_MUTEX_INITIALIZER;

static int sense_ensure_kmod_fd(void)
{
	pthread_mutex_lock(&g_kmod_lock);
	if (!g_kmod_fd_init) {
		g_kmod_fd = open("/dev/cipher", O_RDWR);
		g_kmod_fd_init = 1;
		if (g_kmod_fd < 0) {
			cipher_log("SENSE: open /dev/cipher failed (errno=%d); "
			           "DSM PROPOSE push disabled", errno);
		}
	}
	pthread_mutex_unlock(&g_kmod_lock);
	return g_kmod_fd;
}

/* --- Tenant slot lookup. Direct-mapped hash; collision = evict. */
static struct tenant_state *sense_tenant_slot(uint32_t tenant_id)
{
	uint32_t idx = tenant_id % SENSE_TENANT_TABLE_SIZE;
	struct tenant_state *s = &g_tenant_table[idx];
	uint32_t existing = atomic_load_explicit(&s->tenant_id,
	                                          memory_order_relaxed);
	if (existing == 0 || existing == tenant_id) {
		atomic_store_explicit(&s->tenant_id, tenant_id,
		                       memory_order_relaxed);
		return s;
	}
	/* Collision: evict oldest. Best-effort observability; not perfect
	 * but bounded and lock-free. */
	atomic_store_explicit(&s->tenant_id, tenant_id, memory_order_relaxed);
	atomic_store_explicit(&s->stable_class, 0, memory_order_relaxed);
	atomic_store_explicit(&s->current_class, 0, memory_order_relaxed);
	atomic_store_explicit(&s->current_count, 0, memory_order_relaxed);
	atomic_store_explicit(&s->other_count, 0, memory_order_relaxed);
	return s;
}

/* --- Enqueue a proposal (drop-oldest-on-overflow). */
static void sense_emit_proposal(uint32_t tenant_id, uint32_t source_class,
                                 uint32_t target_class, uint32_t reason,
                                 uint32_t confidence, uint64_t fingerprint,
                                 uint64_t timestamp_ns)
{
	uint32_t head = atomic_fetch_add_explicit(&g_proposal_head, 1,
	                                           memory_order_relaxed);
	struct proposal_entry *e =
		&g_proposal_queue[head % SENSE_PROPOSAL_QUEUE_SIZE];
	/* Write fields. Memory ordering: a concurrent reader of /proc
	 * may see a torn entry in the worst case; acceptable for
	 * observability-only Pattern (a). Pattern (b) ioctl drains
	 * synchronously, so the kmod sees consistent payloads. */
	e->tenant_id    = tenant_id;
	e->source_class = source_class;
	e->target_class = target_class;
	e->reason       = reason;
	e->confidence   = confidence;
	e->pad          = 0;
	e->fingerprint  = fingerprint;
	e->timestamp_ns = timestamp_ns;
	atomic_fetch_add_explicit(&g_proposals_queued, 1, memory_order_relaxed);
}

void cipher_rt_sense_transition_observe(uint32_t tenant_id, uint32_t op_class,
                                         uint8_t confidence,
                                         uint64_t fingerprint,
                                         uint64_t timestamp_ns)
{
	struct tenant_state *s = sense_tenant_slot(tenant_id);
	uint32_t prev_current =
		atomic_load_explicit(&s->current_class, memory_order_relaxed);
	uint32_t stable =
		atomic_load_explicit(&s->stable_class, memory_order_relaxed);
	uint64_t last_ts =
		atomic_load_explicit(&s->last_ts_ns, memory_order_relaxed);

	atomic_store_explicit(&s->fingerprint, fingerprint,
	                       memory_order_relaxed);
	atomic_store_explicit(&s->last_ts_ns, timestamp_ns,
	                       memory_order_relaxed);

	if (op_class == prev_current) {
		uint32_t c = atomic_fetch_add_explicit(&s->current_count, 1,
		                                        memory_order_relaxed) + 1;
		/* Promote to stable when N consecutive observations. */
		if (c == SENSE_STABILITY_N && op_class != stable) {
			atomic_store_explicit(&s->stable_class, op_class,
			                       memory_order_relaxed);
			atomic_store_explicit(&s->other_count, 0,
			                       memory_order_relaxed);
			/* Stable-class change is a transition. */
			sense_emit_proposal(tenant_id, stable, op_class,
			                    CIPHER_RT_SENSE_TRANSITION_DETECTED,
			                    confidence, fingerprint, timestamp_ns);
			/* W4 Step 6 — bump per-tenant + per-(from,to) + aggregate
			 * transition counters. Clamp class indices to 0..3. */
			atomic_fetch_add_explicit(&s->transitions, 1,
			                          memory_order_relaxed);
			atomic_fetch_add_explicit(&g_transitions_total, 1,
			                          memory_order_relaxed);
			if (stable < SENSE_CLASS_COUNT && op_class < SENSE_CLASS_COUNT) {
				atomic_fetch_add_explicit(
					&g_transition_matrix[stable][op_class], 1,
					memory_order_relaxed);
			}
		}
	} else {
		atomic_store_explicit(&s->current_class, op_class,
		                       memory_order_relaxed);
		atomic_store_explicit(&s->current_count, 1, memory_order_relaxed);
	}

	/* Tool-idle heuristic: AGENT_AUTONOMOUS (2) stable + gap > 100ms
	 * since previous observation. OpClass enum value 2 here maps
	 * conceptually to AGENT in the SENSE classification; the
	 * fingerprint preserves the SENSE-side identity. */
	if (stable == 2 /* AGENT */ && last_ts != 0
	    && timestamp_ns > last_ts
	    && (timestamp_ns - last_ts) > SENSE_AGENT_IDLE_NS) {
		sense_emit_proposal(tenant_id, stable, stable,
		                    CIPHER_RT_SENSE_AGENT_IDLE,
		                    confidence, fingerprint, timestamp_ns);
		/* W4 Step 6 — bump per-tenant + aggregate tool-idle counters. */
		atomic_fetch_add_explicit(&s->tool_idles, 1, memory_order_relaxed);
		atomic_fetch_add_explicit(&g_tool_idle_count, 1, memory_order_relaxed);
	}
}

int cipher_rt_sense_transition_flush(void)
{
	int fd = sense_ensure_kmod_fd();
	if (fd < 0) return -1;

	/* Drain the queue: snapshot head, replay entries since last
	 * push position (tracked via g_proposals_pushed). */
	uint32_t head = atomic_load_explicit(&g_proposal_head, memory_order_acquire);
	uint32_t pushed_count = (uint32_t)atomic_load_explicit(&g_proposals_pushed,
	                                                       memory_order_relaxed);
	uint32_t drained = 0;

	/* Limit per-flush to queue size to avoid stalling the CUPTI thread. */
	uint32_t max = SENSE_PROPOSAL_QUEUE_SIZE;
	if (head <= pushed_count) return 0;
	uint32_t n = head - pushed_count;
	if (n > max) n = max;

	for (uint32_t i = 0; i < n; i++) {
		uint32_t idx = (pushed_count + i) % SENSE_PROPOSAL_QUEUE_SIZE;
		struct proposal_entry *e = &g_proposal_queue[idx];
		struct cipher_dsm_propose_push push;
		memset(&push, 0, sizeof(push));
		push.tenant_id    = e->tenant_id;
		push.source_slot  = e->source_class;
		push.target_slot  = e->target_class;
		push.reason       = e->reason;
		push.confidence   = e->confidence;
		if (ioctl(fd, CIPHER_DSM_PROPOSE, &push) < 0) {
			cipher_log("SENSE: ioctl CIPHER_DSM_PROPOSE errno=%d", errno);
			return drained ? (int)drained : -1;
		}
		drained++;
	}

	atomic_fetch_add_explicit(&g_proposals_pushed, drained,
	                          memory_order_relaxed);
	return (int)drained;
}

unsigned long cipher_rt_sense_transition_proposals_queued(void)
{
	return atomic_load_explicit(&g_proposals_queued, memory_order_relaxed);
}

unsigned long cipher_rt_sense_transition_proposals_pushed(void)
{
	return atomic_load_explicit(&g_proposals_pushed, memory_order_relaxed);
}

/* W4 Step 6 — Sub-4 measurement infra accessors + JSON report. */
unsigned long cipher_rt_sense_transition_tool_idle_count(void)
{
	return atomic_load_explicit(&g_tool_idle_count, memory_order_relaxed);
}

unsigned long cipher_rt_sense_transition_transitions_total(void)
{
	return atomic_load_explicit(&g_transitions_total, memory_order_relaxed);
}

int cipher_rt_sense_transition_report(const char *path)
{
	/* Class indices per CipherSessionType (cipher-may13-evidence/include/
	 * cipher_sense.h): 0=UNKNOWN, 1=HUMAN, 2=AGENT, 3=BATCH. The existing
	 * observe() code already uses stable==2 for AGENT. */
	static const char * const class_names[SENSE_CLASS_COUNT] = {
		"UNKNOWN", "HUMAN", "AGENT", "BATCH"
	};
	const char *out_path = path ? path : "/tmp/cipher_sense_transitions.json";
	FILE *fp = fopen(out_path, "w");
	if (!fp) return -1;

	unsigned long proposals_q = atomic_load_explicit(&g_proposals_queued,
	                                                 memory_order_relaxed);
	unsigned long proposals_p = atomic_load_explicit(&g_proposals_pushed,
	                                                 memory_order_relaxed);
	unsigned long tool_idle   = atomic_load_explicit(&g_tool_idle_count,
	                                                 memory_order_relaxed);
	unsigned long trans_total = atomic_load_explicit(&g_transitions_total,
	                                                 memory_order_relaxed);

	fprintf(fp, "{\n");
	fprintf(fp, "  \"schema\": \"cipher_sense_transitions/v1\",\n");
	fprintf(fp, "  \"aggregate\": {\n");
	fprintf(fp, "    \"proposals_queued\":   %lu,\n", proposals_q);
	fprintf(fp, "    \"proposals_pushed\":   %lu,\n", proposals_p);
	fprintf(fp, "    \"transitions_total\":  %lu,\n", trans_total);
	fprintf(fp, "    \"tool_idle_count\":    %lu\n",  tool_idle);
	fprintf(fp, "  },\n");

	fprintf(fp, "  \"transition_matrix\": {\n");
	for (uint32_t f = 0; f < SENSE_CLASS_COUNT; f++) {
		fprintf(fp, "    \"%s\": {", class_names[f]);
		for (uint32_t t = 0; t < SENSE_CLASS_COUNT; t++) {
			unsigned v = atomic_load_explicit(&g_transition_matrix[f][t],
			                                  memory_order_relaxed);
			fprintf(fp, " \"%s\": %u%s", class_names[t], v,
			        (t + 1 < SENSE_CLASS_COUNT) ? "," : "");
		}
		fprintf(fp, " }%s\n", (f + 1 < SENSE_CLASS_COUNT) ? "," : "");
	}
	fprintf(fp, "  },\n");

	fprintf(fp, "  \"tenants\": [\n");
	unsigned emitted = 0;
	for (uint32_t i = 0; i < SENSE_TENANT_TABLE_SIZE; i++) {
		struct tenant_state *s = &g_tenant_table[i];
		uint32_t tid = atomic_load_explicit(&s->tenant_id,
		                                    memory_order_relaxed);
		if (tid == 0) continue;
		uint32_t stable = atomic_load_explicit(&s->stable_class,
		                                       memory_order_relaxed);
		uint32_t cur    = atomic_load_explicit(&s->current_class,
		                                       memory_order_relaxed);
		uint32_t trans  = atomic_load_explicit(&s->transitions,
		                                       memory_order_relaxed);
		uint32_t idles  = atomic_load_explicit(&s->tool_idles,
		                                       memory_order_relaxed);
		const char *stable_name = (stable < SENSE_CLASS_COUNT)
		                          ? class_names[stable] : "?";
		const char *cur_name    = (cur    < SENSE_CLASS_COUNT)
		                          ? class_names[cur]    : "?";
		fprintf(fp, "%s    { \"tenant_id\": %u, \"stable\": \"%s\","
		            " \"current\": \"%s\", \"transitions\": %u,"
		            " \"tool_idles\": %u }",
		        emitted ? ",\n" : "",
		        tid, stable_name, cur_name, trans, idles);
		emitted++;
	}
	if (emitted) fprintf(fp, "\n");
	fprintf(fp, "  ]\n");
	fprintf(fp, "}\n");

	fclose(fp);
	return 0;
}
