/* SPDX-License-Identifier: GPL-2.0
 *
 * cipher_internal.h — types, constants, externs shared across cipher_kmod TUs.
 * Phase 1.5.1: NV_ESC_RM_* family added to the decoder.
 *
 * Slot model (Option Z): one sorted-by-nr table of named ioctls.
 * Per-PID counters use a flat atomic64_t array indexed by table slot.
 * OTHER and TOTAL live in two extra global slots beyond the table.
 */
#ifndef CIPHER_INTERNAL_H
#define CIPHER_INTERNAL_H

#include <linux/types.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/hashtable.h>
#include <linux/sched.h>

#include "cipher_ioctl.h"   /* CIPHER_TENANT_ID_LEN, ioctl ABI */

/* ---- ioctl decode table ----------------------------------------------- */

#define CIPHER_FAM_FRONTEND  0   /* original NV_ESC_*, nr 200..218 */
#define CIPHER_FAM_RM        1   /* NV_ESC_RM_*,        nr 0x27..0x5e */

struct cipher_nv_ioctl_def {
	unsigned int  nr;
	const char   *name;
	u8            family;
};

#define CIPHER_NV_IOCTL_COUNT       24
#define CIPHER_NV_IOCTL_OTHER_SLOT  24
#define CIPHER_NV_IOCTL_TOTAL_SLOT  25
#define CIPHER_GLOBAL_SLOTS         26

/* Slot indices. MUST match designated initialiser positions in
 * cipher_nv_ioctls[] (cipher_ioctl_decode.c). cipher_decode_self_check()
 * verifies the binding at module init.
 */
enum {
	CIPHER_SLOT_RM_ALLOC_MEMORY = 0,
	CIPHER_SLOT_RM_FREE,
	CIPHER_SLOT_RM_CONTROL,
	CIPHER_SLOT_RM_ALLOC,
	CIPHER_SLOT_RM_DUP_OBJECT,
	CIPHER_SLOT_RM_SHARE,
	CIPHER_SLOT_RM_VID_HEAP_CONTROL,
	CIPHER_SLOT_RM_MAP_MEMORY,
	CIPHER_SLOT_RM_UNMAP_MEMORY,
	CIPHER_SLOT_RM_UPDATE_DEVICE_MAPPING_INFO,
	CIPHER_SLOT_CARD_INFO,
	CIPHER_SLOT_REGISTER_FD,
	CIPHER_SLOT_ALLOC_OS_EVENT,
	CIPHER_SLOT_FREE_OS_EVENT,
	CIPHER_SLOT_STATUS_CODE,
	CIPHER_SLOT_CHECK_VERSION_STR,
	CIPHER_SLOT_IOCTL_XFER_CMD,
	CIPHER_SLOT_ATTACH_GPUS_TO_FD,
	CIPHER_SLOT_QUERY_DEVICE_INTR,
	CIPHER_SLOT_SYS_PARAMS,
	CIPHER_SLOT_NUMA_INFO,
	CIPHER_SLOT_SET_NUMA_STATUS,
	CIPHER_SLOT_EXPORT_TO_DMABUF_FD,
	CIPHER_SLOT_WAIT_OPEN_COMPLETE,
};

extern const struct cipher_nv_ioctl_def
	cipher_nv_ioctls[CIPHER_NV_IOCTL_COUNT];

/* ---- per-PID hashtable ------------------------------------------------ */

#define CIPHER_PID_HASH_BITS  10
#define CIPHER_PID_HASH_SIZE  (1 << CIPHER_PID_HASH_BITS)

#define CIPHER_PROC_DIR        "cipher"
#define CIPHER_PROC_STATS      "stats"
#define CIPHER_PROC_BAR0       "bar0_state"
#define CIPHER_PROC_GPU_STATE  "gpu_state"
#define CIPHER_PROC_FLOPS      "flops"
#define CIPHER_PROC_MIGRATIONS "migrations"
#define CIPHER_PROC_ARENAS     "arenas"
#define CIPHER_PROC_CLASSIFY_STATS "classify_stats"   /* Week 2 Step 5 */
#define CIPHER_PROC_DSM_PROPOSALS  "dsm_proposals"    /* Week 3 Step 4 II-a */
#define CIPHER_PROC_SENSE_SESSION  "sense_session"    /* Week 4 Step 5 */
#define CIPHER_DEV_NAME        "cipher"

/* Week 2 Step 5 — classify-stats counters.
 *
 * Populated by Step 6 userspace-to-kmod ioctl bridge (libcipher_rt.so
 * pushes cipher_rt_classify_observer_snapshot() values periodically).
 * Counters are zero in Step 5 (no producer wired yet) and the proc
 * node emits zeros faithfully.
 *
 * 16-class indexing matches cipher::OpClass (0=GEMM..6=ITERATIVE_CUSTOM;
 * slot 15 = UNCLASSIFIED bucket per observer convention).
 */
struct cipher_classify_stats {
	atomic64_t total;
	atomic64_t handled;
	atomic64_t passthrough;
	atomic64_t per_op_class[16];
};

extern struct cipher_classify_stats cipher_classify_stats;

/* Week 2 Step 6 — ioctl handler for CIPHER_PUSH_CLASSIFY_STATS.
 * SET semantics: replaces kmod counters with userspace observer
 * snapshot. Returns 0 on success, -EFAULT on copy_from_user failure. */
long cipher_push_classify_stats_ioctl(unsigned long arg);

/* Week 3 Step 4 Option II-a — DSM proposal ring buffer.
 * Pattern (a) observability: userspace pushes proposals; kmod buffers
 * them in a bounded ring (drop-oldest on overflow); /proc emits the
 * recent N entries. NO auto-action — operator reads + decides.
 *
 * Ring size 256 (bounded; ~14 KiB); head is a monotonic write index;
 * readers walk back from head. */
#define CIPHER_DSM_PROPOSAL_RING_SIZE 256u

struct cipher_dsm_proposal_entry {
	u64 timestamp_jiffies;
	u32 tenant_id;
	u32 source_slot;
	u32 target_slot;
	u32 reason;
	u32 confidence;
	u32 reserved;
};

struct cipher_dsm_proposal_ring {
	atomic_t head;                /* monotonic write index (wraps via %RING_SIZE) */
	atomic64_t total_received;    /* monotonic counter for diagnostics */
	struct cipher_dsm_proposal_entry buf[CIPHER_DSM_PROPOSAL_RING_SIZE];
};

/* Week 4 Step 5 — per-session SENSE classification telemetry.
 * Mirrors classify_stats pattern: kmod-side BSS counters that a Step 6+
 * userspace bridge can SET via ioctl (no producer wired yet; emitter
 * prints zeros faithfully). Class indexing matches the canonical
 * CipherSessionType enum (cipher-may13-evidence/include/cipher_sense.h):
 *   0 = UNKNOWN, 1 = HUMAN, 2 = AGENT, 3 = BATCH. */
struct cipher_sense_session_stats {
	atomic64_t per_class[4];      /* 0=UNKNOWN, 1=HUMAN, 2=AGENT, 3=BATCH */
	atomic64_t total_sessions;
	atomic64_t total_transitions;
};

extern struct cipher_sense_session_stats cipher_sense_session_stats;

extern struct cipher_dsm_proposal_ring cipher_dsm_proposals;
long cipher_dsm_propose_ioctl(unsigned long arg);

/* Phase 3 Task 3: device-wide GPU state spine. Single global, latest-wins.
 * Written by cipher-gpustate daemon via CIPHER_SUBMIT_GPU_STATE; read by
 * /proc/cipher/gpu_state seq_file. Spinlock-protected to avoid torn reads
 * across the 7 u32 fields (an 8-field memcpy is not naturally atomic). */
struct cipher_gpu_state_kern {
	u64        timestamp_jiffies;   /* when we received this, not the userspace ns */
	u32        power_mw;
	u32        temp_c;
	u32        sm_clock_mhz;
	u32        mem_clock_mhz;
	u32        sm_util_pct;
	u32        mem_util_pct;
	u32        fb_used_mb;
	spinlock_t lock;
};

extern struct cipher_gpu_state_kern cipher_gpu_state;

struct cipher_pid_stats {
	struct hlist_node node;
	struct rcu_head   rcu;
	pid_t             pid;
	pid_t             tgid;
	char              comm[TASK_COMM_LEN];   /* refreshed every observation */
	char              tenant_id[CIPHER_TENANT_ID_LEN];  /* Phase 2: "" until REGISTER_TENANT */
	atomic64_t        per_slot[CIPHER_NV_IOCTL_COUNT];   /* 192 B */
	atomic64_t        other_count;
	atomic64_t        total;
	atomic64_t        errors;
	u64               first_seen_jiffies;
	u64               last_seen_jiffies;

	/* Phase 3 Task 3 telemetry, latest-wins per field.
	 * Writers (cipher_dev.c handlers) use WRITE_ONCE; readers
	 * (cipher_proc.c) use READ_ONCE. Single-field aligned 4/8-byte
	 * writes on x86-64 are torn-write-free, and the fields are
	 * mutually independent -- no cross-field invariant to lock. */
	u32               sm_util_pct;            /* from PROCESS_UTIL */
	u32               mem_util_pct;           /* from PROCESS_UTIL */
	u32               fb_used_mb;             /* from PROCESS_UTIL */
	u32               enc_util_pct;           /* from PROCESS_UTIL (stored, not displayed) */
	u32               dec_util_pct;           /* from PROCESS_UTIL (stored, not displayed) */
	u64               launches_total;         /* from LAUNCH_STATS, CUPTI snapshot */
	u64               grid_ops_total;         /* from LAUNCH_STATS */
	u64               last_telemetry_jiffies; /* set by either PROCESS_UTIL or LAUNCH_STATS */

	/* CP 3.3 — per-tenant FLOP attribution. Written ONLY by the FLOP
	 * sample ioctl handler (cipher_flops.c), ~10 Hz, single-writer.
	 * Readers use READ_ONCE. attributed_flops_per_s = device FLOP/s ×
	 * (this tenant's launch delta / total launch delta). */
	u64               launches_at_last_flop;  /* launches_total snapshot at last FLOP sample */
	u64               attributed_flops_per_s; /* device FLOP/s × launch-share */

	/* Phase 4.1 derived state. Written by cipher_state_updater kthread
	 * (1 kHz) + P4.2 partition allocator + REGISTER_TENANT identity-bridge.
	 * Readers use READ_ONCE; writers use WRITE_ONCE. No cross-field locking;
	 * fields are mutually independent per the contract in PHASE_4_CONTRACT.md. */
	u64               tenant_session_fp;          /* FNV-64(tenant_id), set on REGISTER_TENANT */
	u32               tenant_handle_u32;          /* murmur32(tenant_id), set on REGISTER_TENANT */
	u32               sm_partition_mask;          /* P4.2 allocator */
	u32               sm_partition_count;         /* popcount of mask */
	u32               thermal_headroom_pct;       /* state_updater */
	u32               power_headroom_w;           /* state_updater */
	u32               sustained_clock_mhz;        /* state_updater EMA */
	u32               voltage_envelope_mv;        /* state_updater */
	u32               l2_residency_kb;            /* P4.4 PREDICT */
	u32               hot_region_count;           /* P4.4 PREDICT */
	u64               predicted_hot_regions[8];   /* P4.4 PREDICT top-8 */
	u32               kv_cache_size_mb;           /* P4.6 KV_REDIRECT */
	u32               kv_compression_ratio_pct;   /* P4.6 KV_COMPRESS */
	u32               weight_dedup_savings_mb;    /* P4.5 WEIGHT_SHARE accounting */
	u32               weight_content_hash_count;  /* P4.5 WEIGHT_SHARE */
	u32               fusion_recipe_id;           /* P4.7 RECIPES last match */
	u32               fusion_eligible_flag;       /* P4.7 FUSION detection */
	u32               session_band;               /* P4.7 SENSE: 0=HUMAN 1=AGENT 2=BATCH */
	u32               slo_priority;               /* P4.7 SHIELD */
	u32               fairness_quota_remaining_pct; /* P4.7 FAIRNESS */
	u32               graph_capture_state;        /* P4.6 GRAPH */
	u32               koopman_substitution_eligibility; /* P4.6 ATTN_KOOPMAN */
	u64               snapshot_jiffies;           /* state_updater freshness */
};

/* Phase 4.1 — per-tenant snapshot contract (PHASE_4_CONTRACT.md).
 *
 * Assembled view of cipher_pid_stats for a single tenant. Returned by
 * cipher_get_current_tenant_snapshot() / by_id() / enumerate(). Reader
 * must hold rcu_read_lock() across access; pointer valid only until
 * next rcu_read_unlock(). Copy fields out for longer use.
 *
 * Layout target: 336 B (fits 6 x 64-byte cache lines). static_assert
 * verifies at < 384 B. */
struct cipher_tenant_snapshot {
	/* --- Identity (three-way bridge) --- */
	char     tenant_id_str[CIPHER_TENANT_ID_LEN];
	u64      tenant_session_fp;
	u32      tenant_handle_u32;
	u32      pid;
	u32      tgid;
	u32      _pad_identity;

	/* --- Live telemetry (Phase 3 ioctls) --- */
	u32      sm_util_pct;
	u32      mem_util_pct;
	u64      launches_total;
	u64      grid_ops_total;
	u64      ioctls_total;

	/* --- SM-partition cluster (P4.2) --- */
	u32      sm_partition_mask;
	u32      sm_partition_count;

	/* --- DVFS/thermal cluster (P4.3) --- */
	u32      thermal_headroom_pct;
	u32      power_headroom_w;
	u32      sustained_clock_mhz;
	u32      voltage_envelope_mv;

	/* --- L2 cluster (P4.4) --- */
	u32      l2_residency_kb;
	u32      hot_region_count;
	u64      predicted_hot_regions[8];

	/* --- KV cluster (P4.6) --- */
	u32      kv_cache_size_mb;
	u32      kv_compression_ratio_pct;

	/* --- Weight cluster (P4.5) --- */
	u32      weight_dedup_savings_mb;
	u32      weight_content_hash_count;

	/* --- Fusion cluster (P4.7) --- */
	u32      fusion_recipe_id;
	u32      fusion_eligible_flag;

	/* --- Agentic / SLO cluster (P4.7) --- */
	u32      session_band;
	u32      slo_priority;
	u32      fairness_quota_remaining_pct;
	u32      _pad_agentic;

	/* --- Attention cluster (P4.6 secondary) --- */
	u32      graph_capture_state;
	u32      koopman_substitution_eligibility;

	/* --- Freshness --- */
	u64      snapshot_jiffies;

	/* --- Cb.2 reserved-tail bump (Wave 5 §5.5 Week 1, Option A adjudication) ---
	 * Two new __u32 fields land at the HEAD of the reserved tail.
	 * Existing offsets before this point are unchanged; sizeof preserved
	 * at 336 B; reserved[16] -> reserved[14] absorbs the 8 bytes.
	 * Consumer: COMMIT primitive (Weeks 7-8) for atomic per-tenant state
	 * update — see CIPHER_REENGINEERING_PLAN.md v1.2.2 §4.8 and
	 * WEEK_1_STEP_4_V2_RESULT.md. */
	u32      recommended_sm_count;     /* classifier-derived partition hint; consumed by Cc.7 ALLOCATE */
	u32      tenant_billing_class;     /* corp=0 / free=1 / research=2 (operator config -> state_updater) */
	/* W7-9 Step 1 G10: per-tenant model identity. 16 B from reserved tail
	 * (56 B -> 40 B remaining). Sentinel all-zero = MODEL_UNKNOWN.
	 * Step 2 deferred commit_seq to BSS (cipher_rt_commit.c per-tenant
	 * snapshot table); reserved tail stays at 40 B for Step 4/5. */
	u8       model_uuid[16];
	u32      reserved[10];             /* was reserved[14]; -16 B for model_uuid */
	/* W7-9 Step 3 G6: per-tenant HMAC-SHA256 chain accumulator.
	 * hmac_state[32]: rolling SHA-256 state for the next chain entry.
	 * hmac_chain_head[32]: latest published chain head (HMAC-SHA256 of
	 * prev_chain_head || payload, keyed by per-tenant hmac_key in
	 * cipher_audit_chain.c ring storage). Updated by COMMIT Step 1 via
	 * cipher_audit_chain_record(). Struct grows 336 -> 400 B; ABI bump
	 * 0.5.5 -> 0.6.0 (Option A locked by user 2026-05-23). */
	u8       hmac_state[32];
	u8       hmac_chain_head[32];
};

/* Storage in cipher_probe.c (table) and cipher_main.c (the rest). */
extern struct hlist_head cipher_pid_table[CIPHER_PID_HASH_SIZE];
extern spinlock_t        cipher_pid_insert_lock;
extern atomic64_t        cipher_global_cmd_counts[CIPHER_GLOBAL_SLOTS];
extern atomic64_t        cipher_global_cmd_errors[CIPHER_GLOBAL_SLOTS];
extern atomic64_t        cipher_alloc_failures;
extern atomic64_t        cipher_reaped_count;
extern u64               cipher_load_jiffies;
extern unsigned long     cipher_hooked_addr;

/* cipher_ioctl_decode.c */
int cipher_decode_nv_ioctl_slot(unsigned int cmd);  /* slot 0..23 or -1 */
int cipher_decode_self_check(void);                  /* 0 or -EINVAL */

/* cipher_probe.c */
int  cipher_probe_init(void);
void cipher_probe_exit(void);
/* exposed for cipher_dev.c (REGISTER_TENANT handler needs to upsert an entry) */
struct cipher_pid_stats *cipher_pid_get_or_create(pid_t pid, pid_t tgid);

/* cipher_proc.c */
int  cipher_proc_init(void);
void cipher_proc_exit(void);

/* cipher_dev.c */
int  cipher_dev_init(void);
void cipher_dev_exit(void);

/* cipher_kvdedup.c — T4.6.4 cross-tenant KV dedup char device. */
int  cipher_kvdedup_init(void);
void cipher_kvdedup_exit(void);

/* cipher_clock.c — T4.3.2 SM clock actuation via usermode-helper. */
int  cipher_dev_set_clock_mhz(unsigned long arg);

/* cipher_flops.c — CP 3.3 continuous FLOP telemetry + per-tenant MFU.
 * cipher_flopd (root daemon) submits device FLOP samples via
 * CIPHER_SUBMIT_FLOP_SAMPLE; the kmod owns the FLOP-series ring and
 * derives per-tenant attributed FLOPs from per-tenant launch deltas. */
int  cipher_flops_init(void);
void cipher_flops_exit(void);
long cipher_flops_submit(unsigned long arg);   /* CIPHER_SUBMIT_FLOP_SAMPLE */
long cipher_flops_query(unsigned long arg);    /* CIPHER_QUERY_FLOPS */
int  cipher_flops_proc_show(struct seq_file *m, void *v);  /* /proc/cipher/flops */

/* cipher_bar0.c */
struct seq_file;
int  cipher_bar0_init(void);
void cipher_bar0_exit(void);
int  cipher_bar0_proc_show(struct seq_file *m, void *v);

/* cipher_tenant_snapshot.c — Phase 4.1 contract (T4.1.3 / T4.1.4 / T4.1.5).
 *
 * cipher_get_current_tenant_snapshot — RCU read-side, sub-200ns target.
 *   Caller MUST hold rcu_read_lock(). Returned pointer is per-CPU
 *   thread-local; copy out before rcu_read_unlock().
 *   Returns NULL if current task has no cipher_pid_stats entry. */
const struct cipher_tenant_snapshot *
cipher_get_current_tenant_snapshot(void);

/* Cross-tenant lookup. Linear scan O(N) for N <= 150. RCU read-side. */
const struct cipher_tenant_snapshot *
cipher_get_tenant_snapshot_by_id(const char *tenant_id);

/* By-pid lookup. RCU read-side, used by ioctl nr 8 cross-process queries. */
const struct cipher_tenant_snapshot *
cipher_get_tenant_snapshot_by_pid(pid_t pid);

/* Enumeration for Prometheus exporter + arbitration. Snapshot copy.
 * No RCU requirement; uses internal hashtable lock. Process context only. */
int cipher_enumerate_tenants(struct cipher_tenant_snapshot *out, int max);

/* P4.2 partition allocator write-back paths. */
int cipher_set_sm_partition_mask(pid_t target_pid, u32 mask);
int cipher_get_sm_partition_mask(pid_t target_pid, u32 *out_mask);

/* cipher_state_updater kthread (T4.1.6). */
int  cipher_state_updater_init(void);
void cipher_state_updater_exit(void);

/* cipher_partition_allocator (T4.2.1) — RETIRED in W4 Step 4.
 * Source file deleted; all callers (init/exit/tick/do_exit-release) removed.
 * Userspace nr 9 (CIPHER_REQUEST_SM_PARTITION) preserved as -ENOSYS ABI stub
 * in cipher_dev.c. Replacement is CP 5.4 (nrs 13-15, cipher_cp54_sched.c). */

/* cipher_cp54_sched.c — CP 5.4 per-tenant arbitration (8-SM-group ledger).
 * The kmod-resident cross-process SM allocator; supersedes the legacy nr-9
 * 4-SM-slot allocator (which is deactivated in cipher_dev.c). */
int  cipher_cp54_sched_init(void);
void cipher_cp54_sched_exit(void);
long cipher_cp54_ioctl_allocate(unsigned long arg);  /* CIPHER_CP54_ALLOCATE */
long cipher_cp54_ioctl_free(unsigned long arg);      /* CIPHER_CP54_FREE */
long cipher_cp54_ioctl_query(unsigned long arg);     /* CIPHER_CP54_QUERY */
void cipher_cp54_release(pid_t pid);  /* do_exit reaper hook (lock-free) */

/* Track 3 SC2 — Dynamic SM Migration state machine (nrs 16-20). */
long cipher_cp54_ioctl_subscribe_migrate(unsigned long arg); /* nr 16 */
long cipher_cp54_ioctl_poll_migrate(unsigned long arg);      /* nr 17 */
long cipher_cp54_ioctl_start_migrate(unsigned long arg);     /* nr 18 */
long cipher_cp54_ioctl_ack_migrate(unsigned long arg);       /* nr 19 */
long cipher_cp54_ioctl_compact_migrate(unsigned long arg);   /* nr 20 */
/* Track 3 SC4 — /proc/cipher/migrations emitter (in cipher_cp54_sched.c). */
int  cipher_cp54_migration_proc_show(struct seq_file *m, void *v);

/* Track 2 SC5 — weight-arena fd custodian (cipher_weight_arena.c). */
void cipher_wa_init(void);
void cipher_wa_exit(void);

/* W7-9 Step 1 G10 — model identity registry. Hashtable keyed on
 * hf_config_hash; idempotent register (same hash returns same uuid). */
int  cipher_model_registry_init(void);
void cipher_model_registry_exit(void);
int  cipher_model_register(const struct cipher_register_model *req,
                           u8 out_uuid[16]);
int  cipher_model_lookup_by_uuid(const u8 uuid[16],
                                 u8 hf_config_hash_out[32],
                                 u32 *model_arch_out);
long cipher_dev_register_model(unsigned long arg);

/* W.6 sub-C — GPU co-residence registry (nr 30/31). Per-tgid table
 * {tgid, model_fingerprint, last_seen_ns}; 30 s liveness; QUERY is the
 * heartbeat. W.4 POOL consumes (co-resident pids + per-pid fingerprint). */
int  cipher_cohort_registry_init(void);
void cipher_cohort_registry_exit(void);
int  cipher_cohort_register(u32 tgid, u64 model_fingerprint);
int  cipher_cohort_snapshot(u32 max_entries,
                            struct cipher_cohort_entry_abi *out,
                            u32 *n_resident_out);
long cipher_dev_cohort_register(unsigned long arg);
long cipher_dev_cohort_query(unsigned long arg);

/* W7-9 Step 3 G6 — kmod-resident per-tenant AUDIT chain.
 * Per CIPHER_REENGINEERING_PLAN.md v1.2.3 §4.8 Step 1 (AUDIT chain HMAC
 * entry). Per-tenant HMAC-SHA256 chain + ring buffer; mmap'd read-only
 * to userspace via cipher_dev_fops.mmap. ABI bump 0.5.5 -> 0.6.0 because
 * cipher_pid_stats grew 336 -> 400 B (hmac_state[32] + hmac_chain_head[32]
 * appended after the reserved tail). */
#include <linux/fs.h>
int  cipher_audit_chain_init(void);
void cipher_audit_chain_exit(void);
int  cipher_audit_chain_record(u32 tenant_id, u64 commit_seq,
                               u32 actuator_id,
                               const u8 *payload, size_t payload_len);
int  cipher_audit_chain_mmap(struct file *filp,
                             struct vm_area_struct *vma);
u64  cipher_audit_chain_region_bytes(void);
void cipher_audit_chain_hmac(const u8 *key, size_t key_len,
                             const u8 *msg, size_t msg_len,
                             u8 out[32]);
long cipher_dev_audit_record(unsigned long arg);

/* CIPHER_RT_MAX_TENANTS must match the userspace cipher_rt_phase4 value
 * (cipher_rt_commit.h). Kmod-side cap matches CP54_MAX_ALLOCS post-G1. */
#ifndef CIPHER_RT_MAX_TENANTS
#define CIPHER_RT_MAX_TENANTS 128
#endif
long cipher_wa_ioctl_register(unsigned long arg);  /* nr 21 */
long cipher_wa_ioctl_import(unsigned long arg);     /* nr 22 */
long cipher_wa_ioctl_leave(unsigned long arg);      /* nr 23 */
long cipher_wa_ioctl_query(unsigned long arg);      /* nr 24 */
int  cipher_wa_proc_show(struct seq_file *m, void *v);  /* /proc/cipher/arenas */

#endif /* CIPHER_INTERNAL_H */
