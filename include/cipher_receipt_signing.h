// CIPHER receipt signing — Tier B isolation per OP_CONTRACT.md.
// =============================================================
// Per-event HMAC-SHA256 chained signing of billing receipts. Lives in
// a SEPARATE DSO (libcipher_receipt.so) so:
//   - Crypto / key material doesn't pollute the hot DSO (libcipher_rt.so)
//   - Crash in the signing path doesn't crash CIPHER core
//   - Operators can swap signing implementations (HMAC → Ed25519 → KMS)
//     without rebuilding the runtime
//
// Output format: append-only JSONL log at /var/log/cipher/receipts.jsonl
// (configurable). Each line:
//   {"seq":N, "ts_ns":T, "tenant":"X", "event":"...",
//    "data": {...}, "prev_chain": "HEX64", "hmac": "HEX64"}
//
// Chain semantics: each line's hmac signs (prev_chain || data); next
// line's prev_chain = THIS line's hmac. Tamper of any line breaks the
// chain from that point forward — verifier detects it.

#ifndef CIPHER_RECEIPT_SIGNING_H
#define CIPHER_RECEIPT_SIGNING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Init — opens log file in append mode, loads key from key_path. If
// key_path is NULL, reads $CIPHER_RECEIPT_KEY then /etc/cipher/hmac.key.
// If log_path is NULL, defaults to $CIPHER_RECEIPT_LOG then
// /var/log/cipher/receipts.jsonl.
// Returns 1 on success.
int  cipher_receipt_signing_init(const char* log_path,
                                  const char* key_path);

// Append one event. `tenant_id` and `event` are short strings;
// `data_json` should already be a valid JSON object body (just the
// inside, e.g. `"k1": 1, "k2": "v"`). Returns 1 on success.
int  cipher_receipt_signing_append(const char* tenant_id,
                                    const char* event,
                                    const char* data_json);

// Force sync to disk (fsync). Call before process exit if the operator
// wants guaranteed durability.
void cipher_receipt_signing_flush(void);

// Verify the chain integrity of a JSONL log file using the given key.
// Returns:
//   >0  number of valid lines verified
//   -1  read error
//   -2  chain break (returns line number of first bad line × -1 - 2)
int  cipher_receipt_signing_verify_chain(const char* log_path,
                                          const char* key_path);

// Stats (for Prometheus export).
struct CipherReceiptSigningStats {
    int      enabled;
    uint64_t lines_written;
    uint64_t bytes_written;
    uint64_t key_load_failures;
    uint64_t write_failures;
};
int  cipher_receipt_signing_stats(struct CipherReceiptSigningStats* out);

#ifdef __cplusplus
}
#endif

#endif // CIPHER_RECEIPT_SIGNING_H
