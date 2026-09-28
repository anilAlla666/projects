// Week 1 Step 3 harness — cross-tree compile verification.
//
// This translation unit lives in cipher_rt_phase4/src/ and includes
// one may13 header via the may13/ prefix. Its purpose is to prove that
// a cipher_rt_phase4 source file can include and reference may13
// surface, produce linkable symbols, and link cleanly into libcipher_rt.so.
//
// No function calls. No runtime effect. No static-init side effects.
// The declarations below exist so nm can confirm the include resolved
// and the types are real.
//
// may13 type chosen: cipher::OpClass (the 7-class kernel-class enum, plus
// UNCLASSIFIED sentinel) and cipher::ClassifyResult (the 3-field POD
// struct returned by classify_launch). Both are defined in
// cipher_classify.hpp at lines 45-63. They were chosen because:
//   - both are pure type definitions (enum class + POD struct);
//   - neither has side effects when default-constructed or assigned;
//   - their use here is purely declarative — no constructor runs at
//     load time beyond the trivial zero-init of the struct, and no
//     classify_launch() call is made.
// The marker string k_harness_marker is the guaranteed-visible symbol
// per brief Step B fallback.

#include "may13/cipher_classify.hpp"

namespace cipher_may13_harness {

// Guaranteed-visible string symbol (visible via `strings` and `nm`).
// Used to confirm the harness TU was linked into libcipher_rt.so.
// __attribute__((used)) prevents -O2 dead-strip of the static.
static const char* const k_harness_marker
    __attribute__((used)) = "cipher_may13_harness_v1";

// may13 enum class reachable through may13/cipher_classify.hpp.
// (Compiler will fold this away — used in static_assert below as
// the linkage proof that the type was visible to the TU.)
static const cipher::OpClass k_harness_class
    __attribute__((used)) = cipher::OpClass::GEMM;

// may13 POD struct reachable through may13/cipher_classify.hpp.
// Zero-initialised; no constructor side effects.
static const cipher::ClassifyResult k_harness_result
    __attribute__((used)) = {
    cipher::OpClass::UNCLASSIFIED,  // op
    0u,                             // confidence
    false,                          // cache_hit
};

// Compile-time assertions that prove the may13 types are real and
// visible in this TU. If any assertion fails the build breaks; if
// they all pass the assertions emit no runtime code.
static_assert(static_cast<int>(cipher::OpClass::GEMM) == 0,
              "may13 OpClass::GEMM should be 0 per cipher_classify.hpp:46");
static_assert(static_cast<int>(cipher::OpClass::UNCLASSIFIED) == 0xFF,
              "may13 OpClass::UNCLASSIFIED should be 0xFF per cipher_classify.hpp:53");
static_assert(sizeof(cipher::ClassifyResult) >= 3,
              "may13 ClassifyResult should hold at least op + confidence + cache_hit");

}  // namespace cipher_may13_harness
