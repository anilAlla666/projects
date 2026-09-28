/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_audit.c - Op 9 AUDIT on the Phase 4 substrate.
 * See cipher_rt_audit.h for the design.
 */
#include "cipher_rt_audit.h"
#include "cipher_rt_matmul_dispatch.h"
#include "cipher_rt_attn_dispatch.h"

#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

/* AUDIT key — ported from op31-prod cipher_10ops_impl.cpp.
 * Production: load from TPM / secure enclave. */
static const uint8_t AUDIT_KEY[32] = {
	0xC1, 0x10, 0xE0, 0xA0, 0xD1, 0x70, 0xE0, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x4E, 0x45, 0x55, 0x52, 0x41, 0x4C, 0x44, 0x59,  /* "NEURALDY" */
};

#define AUDIT_RING 8192

static struct {
	int                          armed;
	pthread_mutex_t              mu;
	uint8_t                      chain[32];
	uint64_t                     count;
	uint64_t                     prev_ts;
	struct cipher_rt_audit_entry ring[AUDIT_RING];
} g = { .mu = PTHREAD_MUTEX_INITIALIZER };

static uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t fnv1a(const void *p, size_t n)
{
	const uint8_t *b = (const uint8_t *)p;
	uint64_t h = 0xcbf29ce484222325ull;
	for (size_t i = 0; i < n; i++) {
		h ^= b[i];
		h *= 0x100000001b3ull;
	}
	return h;
}

/* HMAC-SHA256 chain link. Canonical 96-byte block:
 *   chain[32] || seq[8] || ts[8] || delta[8] || call_hash[8] ||
 *   op_class[4] || decision[1] || pad[27]
 * The offline verifier (audit_verify.py) reproduces this layout. */
static void audit_chain_update(uint8_t chain[32],
                               const struct cipher_rt_audit_entry *ev)
{
	uint8_t data[96];
	memcpy(data,      chain,                32);
	memcpy(data + 32, &ev->sequence,         8);
	memcpy(data + 40, &ev->timestamp_ns,     8);
	memcpy(data + 48, &ev->timestamp_delta,  8);
	memcpy(data + 56, &ev->call_hash,        8);
	memcpy(data + 64, &ev->op_class,         4);
	data[68] = ev->decision;
	memset(data + 69, 0, 27);

	unsigned int mdlen = 32;
	HMAC(EVP_sha256(), AUDIT_KEY, 32, data, sizeof(data), chain, &mdlen);
}

void cipher_rt_audit_record(uint32_t op_class, uint64_t call_hash,
                            uint8_t decision)
{
	if (!g.armed)
		return;
	pthread_mutex_lock(&g.mu);
	uint64_t seq = g.count++;
	uint64_t ts = now_ns();
	struct cipher_rt_audit_entry *e = &g.ring[seq % AUDIT_RING];
	e->sequence        = seq;
	e->timestamp_ns    = ts;
	e->timestamp_delta = g.prev_ts ? (ts - g.prev_ts) : 0;
	e->call_hash       = call_hash;
	e->op_class        = op_class;
	e->decision        = decision;
	e->_pad[0] = e->_pad[1] = e->_pad[2] = 0;
	g.prev_ts = ts;
	audit_chain_update(g.chain, e);
	pthread_mutex_unlock(&g.mu);
}

/* ---- substrate observer actuators (priority 0: see every call) ---- */

static int audit_matmul_handle(const struct cipher_rt_matmul_call *c,
                               int *out_status)
{
	(void)out_status;
	uint64_t f[10] = {
		(uint64_t)c->m, (uint64_t)c->n, (uint64_t)c->k,
		(uint64_t)(uintptr_t)c->A, (uint64_t)(uintptr_t)c->B,
		(uint64_t)(uintptr_t)c->C,
		(uint64_t)c->Atype, (uint64_t)c->Btype, (uint64_t)c->Ctype,
		(uint64_t)c->algo,
	};
	cipher_rt_audit_record(CIPHER_RT_AUDIT_OP_MATMUL,
	                       fnv1a(f, sizeof(f)), CIPHER_RT_AUDIT_OBSERVED);
	return CIPHER_RT_MATMUL_PASSTHROUGH;
}

static int audit_attn_handle(const struct cipher_rt_attn_call *c)
{
	uint64_t f[8] = {
		(uint64_t)c->backend,
		(uint64_t)(uintptr_t)c->q.data_ptr,
		(uint64_t)(uintptr_t)c->k.data_ptr,
		(uint64_t)(uintptr_t)c->v.data_ptr,
		(uint64_t)c->q.sizes[0], (uint64_t)c->q.sizes[1],
		(uint64_t)c->q.sizes[2], (uint64_t)c->q.sizes[3],
	};
	cipher_rt_audit_record(CIPHER_RT_AUDIT_OP_ATTENTION,
	                       fnv1a(f, sizeof(f)), CIPHER_RT_AUDIT_OBSERVED);
	return CIPHER_RT_ATTN_PASSTHROUGH;
}

static const struct cipher_rt_matmul_actuator audit_matmul_act = {
	.name = "audit", .priority = 0, .maybe_handle = audit_matmul_handle,
};
static const struct cipher_rt_attn_actuator audit_attn_act = {
	.name = "audit", .priority = 0, .maybe_handle = audit_attn_handle,
};

static int env_on(const char *v)
{
	return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' ||
	             v[0] == 't' || v[0] == 'T' || v[0] == 'o' || v[0] == 'O');
}

/* At process exit, if CIPHER_AUDIT_DUMP names a path, write
 *   magic[8]="CIPHAUD1" || count[8] || chain_head[32] || entries[40*N]
 * for offline verification by audit_verify.py. */
__attribute__((destructor))
static void cipher_rt_audit_dump_at_exit(void)
{
	if (!g.armed)
		return;
	const char *path = getenv("CIPHER_AUDIT_DUMP");
	fprintf(stderr, "[cipher-audit] exit: %llu events recorded\n",
	        (unsigned long long)g.count);
	if (!path)
		return;
	FILE *f = fopen(path, "wb");
	if (!f) {
		fprintf(stderr, "[cipher-audit] dump: cannot open %s\n", path);
		return;
	}
	pthread_mutex_lock(&g.mu);
	size_t n = g.count < AUDIT_RING ? (size_t)g.count : AUDIT_RING;
	uint64_t start = g.count - n;
	fwrite("CIPHAUD1", 1, 8, f);
	fwrite(&g.count, 8, 1, f);
	fwrite(g.chain, 1, 32, f);
	for (size_t i = 0; i < n; i++)
		fwrite(&g.ring[(start + i) % AUDIT_RING],
		       sizeof(struct cipher_rt_audit_entry), 1, f);
	pthread_mutex_unlock(&g.mu);
	fclose(f);
	fprintf(stderr, "[cipher-audit] dumped %zu entries + chain head to %s\n",
	        n, path);
}

int cipher_rt_audit_init(void)
{
	if (g.armed)
		return 0;
	if (!env_on(getenv("CIPHER_AUDIT"))) {
		fprintf(stderr, "[cipher-audit] disabled (CIPHER_AUDIT unset)\n");
		return 0;
	}
	memset(g.chain, 0, 32);
	g.count = 0;
	g.prev_ts = 0;
	g.armed = 1;
	int r1 = cipher_rt_matmul_register_actuator(&audit_matmul_act);
	int r2 = cipher_rt_attn_register_actuator(&audit_attn_act);
	fprintf(stderr, "[cipher-audit] armed: HMAC-SHA256 tamper-evident "
	        "chain (matmul actuator=%s, attn actuator=%s)\n",
	        r1 == 0 ? "REGISTERED" : "FAILED",
	        r2 == 0 ? "REGISTERED" : "FAILED");
	return (r1 == 0 && r2 == 0) ? 0 : -1;
}

void cipher_rt_audit_chain_head(uint8_t out[32])
{
	pthread_mutex_lock(&g.mu);
	memcpy(out, g.chain, 32);
	pthread_mutex_unlock(&g.mu);
}

uint64_t cipher_rt_audit_count(void) { return g.count; }
int cipher_rt_audit_enabled(void) { return g.armed; }

size_t cipher_rt_audit_dump(struct cipher_rt_audit_entry *out, size_t max)
{
	if (!out)
		return 0;
	pthread_mutex_lock(&g.mu);
	size_t n = g.count < AUDIT_RING ? (size_t)g.count : AUDIT_RING;
	if (n > max)
		n = max;
	uint64_t start = g.count - n;
	for (size_t i = 0; i < n; i++)
		out[i] = g.ring[(start + i) % AUDIT_RING];
	pthread_mutex_unlock(&g.mu);
	return n;
}
