/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_counter_dump.h -- substrate-side counter dump for verification.
 *
 * V1 Phase A.1 substrate port of the cipher_vllm_plugin
 * _install_counter_dump_for_verification mechanism. Lets the no-plugin
 * verification harness (cipher-fusion-evidence/v1_phase_a/verify_no_plugin.py)
 * read substrate counter values from the worker subprocess via SIGUSR1
 * (live read) or atexit (post-teardown).
 *
 * Env-gated: enabled only when CIPHER_RT_COUNTER_DUMP_PATH is set in the
 * environment. Zero production overhead when unset (no signal handler
 * registered, no atexit hook registered).
 *
 * Idempotent: pthread_once-guarded so repeated calls from multiple init
 * paths (cipher_v2_init_body via cuInit, cipher_rt_auto_init_worker via
 * constructor) are safe.
 *
 * Anil 2026-05-26 caveat: SIGUSR1 collision risk with vLLM debug paths.
 * Implementation chains to any previously-installed SIGUSR1 handler so we
 * do not strand vLLM's handler if it was registered first. atexit ordering
 * is LIFO; constructor registers early, so our handler runs late at
 * teardown which is the safe order (libcudart cleanup runs before us).
 *
 * V1 Phase A.0 ledger: this is substrate-internal, no customer-side step
 * added. Goal 5 contract preserved.
 */

#ifndef CIPHER_RT_COUNTER_DUMP_H
#define CIPHER_RT_COUNTER_DUMP_H

#ifdef __cplusplus
extern "C" {
#endif

/* Env-gated installer. Reads CIPHER_RT_COUNTER_DUMP_PATH. If unset, no-op.
 * If set, registers atexit handler + SIGUSR1 handler that write a JSON
 * snapshot to <CIPHER_RT_COUNTER_DUMP_PATH>/cipher_rt_<pid>.json.
 *
 * Returns 0 on success or env-unset no-op; non-zero on install failure
 * (path not writable, sigaction fails, etc.). Caller ignores return per
 * cipher_inject.c convention. */
int cipher_rt_counter_dump_init(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_COUNTER_DUMP_H */
