// CIPHER lite-rt stubs.
//
// libcipher_rt_lite.so excludes the .cu translation units to avoid the
// fatbin-registration race that breaks vLLM 0.20 + torch 2.11+cu130
// (FA3 launch returns cudaErrorInvalidResourceHandle). Several .cpp
// observers reference symbols that live in those .cu TUs. With
// `--unresolved-symbols=ignore-all`, the .so links but the dynamic
// linker still requires resolution at first call → "undefined symbol"
// at runtime.
//
// This TU provides safe no-op stubs for every .cu-defined entry point
// that observers may call. They're WEAK so the real definitions in the
// full libcipher_rt.so override them when both are linked. In the lite
// build, only the stubs are linked — observers that depend on these
// symbols simply skip their device-side work and continue.

#include <stdint.h>
#include <stdio.h>

extern "C" {

// ── Block substitute (cipher_block_sub_kernel.cu) ─────────────────────
__attribute__((weak)) int  cipher_block_sub_gpu_launch(void*, int, int, void*) { return 0; }
__attribute__((weak)) int  cipher_block_sub_gpu_set_ptrs(void*, void*, void*, void*) { return 0; }

// ── FP8 fused quant (cipher_fp8_fused_quant.cu) ───────────────────────
__attribute__((weak)) int  cipher_fused_fp16_to_fp8(const void*, void*, int, int, void*) { return 0; }

// ── Green context (cipher_green_ctx.cu) ───────────────────────────────
__attribute__((weak)) void* cipher_get_stream(void*, int) { return nullptr; }
__attribute__((weak)) void  cipher_green_ctx_destroy(void*) {}
__attribute__((weak)) int   cipher_green_ctx_init(void*, int) { return 0; }
__attribute__((weak)) void  cipher_green_ctx_report(const void*) {}

// ── SM priority / arbitrate (cipher_green_ctx.cu) ─────────────────────
__attribute__((weak)) void    cipher_sm_set_priority(uint64_t, uint8_t) {}
__attribute__((weak)) uint8_t cipher_sm_get_priority(uint64_t) { return 0; }
__attribute__((weak)) void    cipher_sm_priority_scan_log(void) {}

// ── Koopman attention (cipher_attn_koopman_kernel.cu) ─────────────────
// Fix 2 — the previous weak stub had a 3-int signature that did NOT match
// the strong definition in cipher_block_sub_kernel.cu (5 args incl 3
// float* pointers). The strong def overrides the weak in normal builds,
// but the wrong-signature stub becomes a footgun if anyone dlsym's the
// symbol expecting the 3-int shape. Removed; if we ever need a fallback,
// it must mirror the real signature exactly.

// ── L2 persist (cipher_l2_persist.cu) ─────────────────────────────────
__attribute__((weak)) int  cipher_l2_persist_init(void) { return 0; }
__attribute__((weak)) void cipher_l2_persist_report(void) {}
__attribute__((weak)) void cipher_l2_persist_reset(void) {}

// ── Liquid state / LNN (cipher_liquid_state.cu) ───────────────────────
__attribute__((weak)) void cipher_liquid_record_nccl(uint64_t, uint64_t) {}
__attribute__((weak)) void cipher_liquid_record_op(int, uint64_t, uint64_t) {}
__attribute__((weak)) void cipher_liquid_record_passthrough(int) {}
__attribute__((weak)) void cipher_liquid_record_substitution(int) {}
__attribute__((weak)) int  cipher_liquid_state_init(void*) { return 0; }
__attribute__((weak)) void cipher_liquid_state_destroy(void*) {}
__attribute__((weak)) void cipher_liquid_state_report(const void*) {}
__attribute__((weak)) void cipher_liquid_update_grad_ema(float) {}
__attribute__((weak)) void cipher_liquid_update_hw(int, double, double, double) {}

}  // extern "C"
