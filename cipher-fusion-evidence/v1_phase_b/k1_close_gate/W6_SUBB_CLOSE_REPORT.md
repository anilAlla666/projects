# W.6 sub-B — NR 27 ROOT-CAUSE + SUBSTRATE MODEL FINGERPRINT — CLOSE REPORT

**Date:** 2026-05-28
**Substrate anchor:** `cipher_rt_phase4/build_cuda13/libcipher_rt.so` md5 `4afb719c299b457d97774327935a337e`
**Side anchor preserved:** `build_cuda13/libcipher_rt.so.w6sB` (identical bytes)

## Verdict
**W.6 sub-B PASS (reframed per Anil 2026-05-28 "bypass plugin" adjudication).**
The NR 27 REGISTER_MODEL crash was root-caused (3-layer empirical) but
the plugin py::object path is NOT fixed — it stays gated off (crash
avoided). Instead, W.4 POOL's cross-tenant model-identity prerequisite
is delivered substrate-side: a per-process `cipher_workload_model_fingerprint()`
that discriminates all 3 reference models with zero plugin coupling.
9/9 close gate PASS, zero segfaults.

## A — Root cause (3-layer empirical narrowing)

The original W.6 sub-B premise (Memory #23: fix the plugin py::object
lifecycle, ~2-3 ED) rested on H1/H2 attribution. Empirical testing
narrowed the actual fault path and corrected two wrong theories:

| Layer tested | Method | Result |
|--------------|--------|--------|
| **(1) Python ioctl buffer marshalling** | bytearray-vs-ctypes-Structure: marshal `req` through a `bytearray` to fcntl.ioctl, copy back | **NOT the cause** — real ioctl via bytearray STILL crashed (uuid `166490...`, 262-line crash) |
| **(2) kmod NR 27 handler alone** | standalone process: correct `_IOC` (magic `'C'`=0x43), real ioctl, real uuid `6a4260...`, 300K dict allocs + GC + RichCompare sorts | **NOT the cause** — SURVIVED no crash. kmod handler is self-contained (kernel hashtable insert + bounded `copy_to_user(arg, req, 4164)`), provably cannot corrupt the process heap |
| **(3) register-state × vLLM-inference** | forced `CIPHER_REGISTER_MODEL=1` on full Llama-3-8B vLLM run | **CRASH** — `REGISTER_MODEL OK uuid=166490...` then segfault in `PyObject_RichCompare` (heap-GC). Skip-ioctl (fake uuid) variant ran 735 lines clean |

**Conclusion**: the crash requires BOTH the kmod registry having the entry
AND the full vLLM inference path running afterward. This is the
`cipher_kv_bridge.cpp` / `cipher_rt_kv_alloc.c` py::object region Memory #23
H1/H2 originally flagged — confirmed as the fault locus, NOT the Python
ioctl buffer (theory 1) nor the kmod handler (theory 2).

**Misattribution corrected**: Memory #23's "H3 ruled out (byte-exact 4164B)"
checked only total SIZE; my field-by-field check of the ctypes mirror vs
the kmod struct confirmed the layout ALSO matches (`model_arch, model_uuid,
flags, reserved` — identical order; my earlier "field swap" was an error in
a throwaway test reconstruction, not the real code). So the struct is fully
correct and the fault is genuinely in the inference-path py::object handling.

gdb backtrace at corruption point (faulthandler, sparse symbols):
```
PyObject_RichCompare → _PyEval_EvalFrameDefault → _PyObject_FastCallDictTstate
→ _PyObject_Call_Prepend → _PyObject_MakeTpCall → _PyEval_EvalFrameDefault
```

## B — Decision: bypass the plugin (Anil 2026-05-28)

Per Memory `cipher-audit-2026-05-16`, NR 27 model_uuid has ZERO v1
consumers — nothing in v1 needs it to *function*. W.6 sub-B existed only
as the W.4 POOL prerequisite (per-tenant model identity for legal
cross-tenant GEMM coalescing). Rather than spend ~3-5 ED of ASan-driven
py::object debugging on an optional plugin path, the model-identity
prerequisite moves substrate-side (Memory #30 substrate-line).

- `CIPHER_REGISTER_MODEL` stays **gated off** (rev8 CDI hook unchanged; no rev9). Crash avoided.
- The crashing plugin py::object fix is documented + deferred to v1.x (only needed if NR 27 model_uuid telemetry is later required for an operator-facing surface).

## C — Substrate-side model fingerprint primitive

### File: `cipher_rt_phase4/src/cipher_workload_detect.cpp`

`cipher_workload_model_fingerprint()` — stable 64-bit FNV hash of the
model's dimensional GEMM signature observed at the cuBLAS dispatch
boundary:
- `max_k` — largest cuBLAS K (input feature); captures intermediate_size (FFN).
- `max_m` — largest cuBLAS M (output feature); captures vocab_size (LM head).
- bf16 / int4 dtype bit — distinguishes bf16 deploy from quantized.

Two processes running the SAME model observe identical (max_k, max_m,
dtype) → identical fingerprint → W.4 may legally coalesce their GEMMs.
Returns 0 (UNKNOWN) until `gemm_total >= 64` (warmup; LM-head + FFN
shapes seen). Observation-only; no plugin/application coupling.

Exposed in `include/cipher_workload_detect.h` for W.4 + surfaced in the
CLASSIFY log as `model_fp=0x...`.

## D — Engagement evidence (md5 4afb719c, 9/9 PASS, zero segfaults)

| Cell | Model | model_fp | Discrimination |
|------|-------|----------|----------------|
| P1 Llama-3-8B bf16    | Llama-3-8B | `0xd40e456727923bce` | baseline |
| T1 Llama-3 transition | Llama-3-8B | `0xd40e456727923bce` | **== P1 (same model)** ✓ |
| C1 Llama-3 calibration| Llama-3-8B | `0xd40e456727923bce` | **== P1 (same model)** ✓ |
| P2 Mistral-7B bf16    | Mistral-7B | `0x9fde4562bbbb73ce` | **≠ Llama (distinct model)** ✓ |
| P3 TinyLlama-AWQ      | TinyLlama  | `0x458e457798608581` | **≠ both (distinct; int4 bit)** ✓ |

The fingerprint is the W.4 cross-tenant model-identity discriminator:
same model → same fp (3 Llama-3 cells consistent across 90k+ obs each),
different model → different fp, quantization distinguished. Delivered
WITHOUT the crashing NR 27 plugin path.

K.1+W.1+W.2+W.3+W.5+W.6sB close gate: **9/9 PASS**. NO segfaults
(plugin stays gated off — zero behavior change to the running path).
All prior counters (VOLT/Marlin/Machete/Koopman/attn) preserved.

## E — Anchors

| Component | Pre-W.6sB (post-W.5) | Post-W.6sB |
|-----------|----------------------|------------|
| `cipher_rt_phase4` HEAD              | 879c5b7 (w5-flashattn-6pattern) | (commit pending) tag `w6-subB-model-fingerprint` |
| `cipher_rt_phase4` libcipher_rt.so md5 | fa9fd8f7                       | **4afb719c299b457d97774327935a337e** |
| `cipher-fusion-evidence` HEAD        | 2496d87 (w5-flashattn-close)    | (commit pending) tag `w6-subB-close` |
| `cipher_kmod` HEAD                   | 8c643fc (UNCHANGED — NR 27 handler proven clean) | 8c643fc (UNCHANGED) |
| `cipher-platform` .deb               | rev8 md5 5603f72d (UNCHANGED — no rev9; REGISTER_MODEL stays gated off) | rev8 md5 5603f72d (UNCHANGED) |

No rev9: the spec's Sub-step 3 (rev9 re-enable CIPHER_REGISTER_MODEL=1)
is INTENTIONALLY NOT done — the plugin crash is avoided by keeping the
gate off, and the W.4 prerequisite is met substrate-side.

## F — Engineering debt forecast

1. **NR 27 plugin py::object fix** (v1.x, only if operator-facing model_uuid
   telemetry needed): ASan rebuild of cipher_kv_bridge + full vLLM-under-
   ASan run to pinpoint the exact corrupting write in the kv_bridge /
   kv_alloc inference path. ~3-5 ED. NOT a W.4 blocker (substrate
   fingerprint supersedes).
2. **Fingerprint collision robustness**: current hash uses (max_k, max_m,
   dtype). Two distinct models with identical intermediate_size + vocab_size
   + dtype would collide. Real risk is low (vocab_size is highly model-
   specific) but a fuller signature (per-layer-count, num_kv_heads) would
   harden it — v1.x if a collision is observed.
3. **AWQ fingerprint stability** (P3 gemm_total=162): AWQ routes most GEMMs
   through Machete (bypasses cuBLAS), so the substrate sees fewer GEMMs.
   The fingerprint still stabilized (0x458e... distinct) but the warmup
   threshold (64) should be validated on more quantized models — v1.x.
4. **W.4 consumption**: W.4 POOL reads `cipher_workload_model_fingerprint()`
   per tenant (cross-process via kmod shared region) to gate coalescing.

## G — Memory #29 next-substep

Per resequenced Memory #29 (W.5 → W.6 sub-B/C → W.4 → W.7 → W.8):
- **W.6 sub-C**: kmod process registry (consumes the per-tenant model
  fingerprint + multi_tenant signal; the W.4 POOL multi_tenant prerequisite).

NOTE: with the model-identity prerequisite now met substrate-side
(fingerprint), W.6 sub-C's scope narrows to the multi_tenant *process*
registry (which tenants are co-resident), not model identity. The
fingerprint primitive shipped here covers the "same model?" question.
