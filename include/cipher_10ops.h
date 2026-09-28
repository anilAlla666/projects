// CIPHER 10-Operation Architecture Header
// Zero-overhead Stage 0/1/2 decoupled execution
//
// Critical path additions: ~12ns total
//   SPECULATE check:  <2ns  (atomic look-aside read)
//   Ring write:       ~10ns (single atomic release store)
//
// 6 new operations run in Stage 1 + Stage 2 threads — 0ns on critical path.

#pragma once
#include <stdint.h>
#ifndef __cplusplus
#  include <stdatomic.h>
#  define _Atomic(T) _Atomic T
#else
#  include <atomic>
#  define _Atomic(T) std::atomic<T>
   extern "C" {
#endif

// Ring entry — 128 bytes, cache-line aligned
// Field order: all uint64_t first to eliminate implicit padding
typedef struct __attribute__((aligned(128))) {
    uint64_t sequence;
    uint64_t timestamp_ns;
    uint64_t timestamp_delta;
    uint64_t func_ptr_hash;
    uint64_t params_hash;
    uint64_t output_hash;
    uint32_t kernel_class;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    float    confidence;
    uint8_t  decision;
    uint8_t  speculate_hit;
    uint8_t  pad[46];  // 128 - 82 = 46
} CipherRingEntry;

// Look-aside buffer — SPECULATE prediction lives here
typedef struct {
    _Atomic(int)   predicted_class;
    _Atomic(float) confidence;
    _Atomic(int)   valid;
} CipherLookAside;

static inline int cipher_lookaside_check(CipherLookAside* la, int actual) {
#ifndef __cplusplus
    if (!atomic_load_explicit(&la->valid, memory_order_acquire)) return 0;
    int p = atomic_load_explicit(&la->predicted_class, memory_order_relaxed);
    atomic_store_explicit(&la->valid, 0, memory_order_release);
#else
    if (!la->valid.load(std::memory_order_acquire)) return 0;
    int p = la->predicted_class.load(std::memory_order_relaxed);
    la->valid.store(0, std::memory_order_release);
#endif
    return (p == actual);
}

static inline void cipher_lookaside_write(CipherLookAside* la, int klass, float c) {
#ifndef __cplusplus
    atomic_store_explicit(&la->predicted_class, klass, memory_order_relaxed);
    atomic_store_explicit(&la->confidence, c, memory_order_relaxed);
    atomic_store_explicit(&la->valid, (c > 0.60f) ? 1 : 0, memory_order_release);
#else
    la->predicted_class.store(klass, std::memory_order_relaxed);
    la->confidence.store(c, std::memory_order_relaxed);
    la->valid.store((c > 0.60f) ? 1 : 0, std::memory_order_release);
#endif
}

// SPMC Ring Buffer — Disruptor pattern
#define CIPHER_RING_SIZE 65536u
#define CIPHER_RING_MASK (CIPHER_RING_SIZE - 1u)

typedef struct {
    __attribute__((aligned(64))) _Atomic(uint64_t) write_seq;
    __attribute__((aligned(64))) _Atomic(uint64_t) read_seq_s1;
    __attribute__((aligned(64))) _Atomic(uint64_t) read_seq_s2;
    CipherRingEntry buf[CIPHER_RING_SIZE];
} CipherRing;

// Stage 0 write — plain MOV on x86 TSO / STLR on ARM
static inline int cipher_ring_write(CipherRing* ring, const CipherRingEntry* ev) {
#ifndef __cplusplus
    uint64_t seq = atomic_load_explicit(&ring->write_seq, memory_order_relaxed);
    uint64_t s1  = atomic_load_explicit(&ring->read_seq_s1, memory_order_acquire);
    uint64_t s2  = atomic_load_explicit(&ring->read_seq_s2, memory_order_acquire);
    if (seq - (s1 < s2 ? s1 : s2) >= CIPHER_RING_SIZE) return 0;
    __builtin_memcpy(&ring->buf[seq & CIPHER_RING_MASK], ev, sizeof(*ev));
    atomic_store_explicit(&ring->write_seq, seq + 1, memory_order_release);
#else
    uint64_t seq = ring->write_seq.load(std::memory_order_relaxed);
    uint64_t s1  = ring->read_seq_s1.load(std::memory_order_acquire);
    uint64_t s2  = ring->read_seq_s2.load(std::memory_order_acquire);
    if (seq - (s1 < s2 ? s1 : s2) >= CIPHER_RING_SIZE) return 0;
    __builtin_memcpy(&ring->buf[seq & CIPHER_RING_MASK], ev, sizeof(*ev));
    ring->write_seq.store(seq + 1, std::memory_order_release);
#endif
    return 1;
}

// Global 10-ops runtime
typedef struct {
    CipherRing       ring;
    CipherLookAside  look_aside;
    _Atomic(int)     initialized;
    _Atomic(uint64_t) speculate_hits;
    _Atomic(uint64_t) speculate_total;
    _Atomic(uint64_t) ring_writes;
    _Atomic(uint64_t) audit_entries;
    _Atomic(uint64_t) adapt_swaps;
    _Atomic(uint64_t) validate_failures;
} Cipher10OpsRuntime;

extern Cipher10OpsRuntime g_cipher_10ops;
void cipher_10ops_init(void);
void cipher_10ops_teardown(void);
void cipher_10ops_report(void);

#ifdef __cplusplus
}
#endif
