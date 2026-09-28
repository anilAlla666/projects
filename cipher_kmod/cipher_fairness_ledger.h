/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_fairness_ledger — D.8 FAIRNESS + SHIELD.
 *
 * A kmod-owned, cross-tenant shared work-ledger. This replaces the prior
 * /dev/shm-backed fairness table (cipher_fairness_shm.cpp), which Docker
 * isolates per-container and therefore could NOT span the V.1 100-agent CDI
 * deployment. The region is one global vmalloc_user buffer mapped into every
 * container via /dev/cipher (the same cross-tenant channel the W7-9 resolver
 * proved at 18.4B coherent reads), so all co-resident tenants see one ledger.
 *
 * Spine (per the D.8 memo): kmod = the global cross-tenant view (this region);
 * libcipher_rt gemm-hook = per-tenant self-throttle on the state it reads here.
 * Enforcement is timing-only (CPU sleep before submit) and lives entirely in
 * libcipher_rt — the kmod only provides the shared memory + slot arbitration.
 *
 * Trust model (v1): work is self-reported by each tenant into its own slot
 * (single-writer-per-slot, all-readers), exactly as the prior /dev/shm table
 * did — no regression in exposure. The threat model is bursty LOAD, not a
 * malicious under-reporter. The kmod independently holds an authoritative
 * per-tenant counter (cipher_pid_stats.grid_ops_total, CUPTI-fed) that a
 * future hardening can cross-check against without changing this ABI.
 *
 * ADDITIVE: new ioctl NR 32 + a new mmap pgoff window; no existing NR or
 * region changes. The region is allocated inert at module init; nothing
 * throttles unless a tenant arms libcipher_rt via env (default-OFF == current
 * no-enforcement behavior).
 */
#ifndef CIPHER_FAIRNESS_LEDGER_H
#define CIPHER_FAIRNESS_LEDGER_H

#include <linux/types.h>

struct file;
struct vm_area_struct;

/* Registration scale: must hold >= 100 agents (V.1 Goal-1). Power of 2 for
 * the open-address probe mask. 256 * 64 B = 16 KiB = 4 pages. */
#define CIPHER_FAIR_SLOTS        256u
#define CIPHER_FAIR_SLOT_BYTES   64u
#define CIPHER_FAIR_REGION_BYTES (CIPHER_FAIR_SLOTS * CIPHER_FAIR_SLOT_BYTES)

/* One cache-line slot. key/band are written by the kmod at register; the
 * owning tenant writes gemm_calls/last_active_ns/yield_count via the RW mmap.
 * Layout is shared verbatim with the libcipher_rt mirror — keep it 64 B. */
struct cipher_fair_slot {
	__u64 key;             /* tgid of the owning tenant; 0 = empty slot */
	__u64 gemm_calls;      /* self-reported monotonic GEMM count */
	__u64 last_active_ns;  /* CLOCK_MONOTONIC ns of last record (idle prune) */
	__u32 band;            /* SHIELD band: 0 = throughput/default, >=1 = latency-sensitive */
	__u32 yield_count;     /* diagnostic: times this tenant self-throttled */
	__u64 _pad[4];         /* pad to 64 B */
} __packed;

/* mmap window: above the W7-9 stream-view window (0x100000) so the dev.c
 * multiplex can route by pgoff without touching the audit/stream branches. */
#define CIPHER_FAIR_VIEW_MMAP_PGOFF 0x200000u

/* Lifecycle (called from cipher_main init/exit). init cannot fail fatally —
 * on vmalloc failure the subsystem stays disabled and register returns
 * -ENOMEM, which libcipher_rt treats as "not armed" (no enforcement). */
int  cipher_fairness_ledger_init(void);
void cipher_fairness_ledger_exit(void);

/* CIPHER_FAIRNESS_REGISTER (NR 32) handler helper. Claims (or re-arms) the
 * slot keyed by current->tgid, sets its band, zeroes its counters, and
 * returns the assigned slot index via *out_slot. Returns 0, -ENOMEM (region
 * absent), or -ENOSPC (table full). */
int cipher_fairness_ledger_register(__u32 band, __u32 *out_slot);

/* fops.mmap handler for the fairness window. Unlike the audit/stream windows
 * this one permits VM_WRITE: tenants self-account into their own slot. */
int cipher_fairness_ledger_mmap(struct file *filp, struct vm_area_struct *vma);

#endif /* CIPHER_FAIRNESS_LEDGER_H */
