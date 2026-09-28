# Op #2 — Op 9 AUDIT — REPORT

**Status: BUILT — gate BLOCKED by a composition bug. STOP-and-report
per "WHEN TO STOP AND ASK" (undeclared failure mode + composition risk).**

## What shipped

| Artifact | Lines | Purpose |
|---|---|---|
| `cipher_rt_phase4/cipher_rt_audit.{h,c}` | ~85 + ~210 | Op 9 AUDIT — HMAC-SHA256 tamper-evident dispatch chain |
| `cipher_inject.c` | +2 | `cipher_rt_audit_init()` wired into the injection init body |
| `Makefile` | +4 | `cipher_rt_audit.o`, `-lcrypto` link |
| `cipher-fusion-evidence/audit_verify.py` | ~55 | independent offline HMAC chain verifier |

`libcipher_rt.so` rebuilt — md5 `7d8e5e9fe26e3fb50d1bac2bcaeb95e4`
(prior `libcipher_rt.so.pre_op2_audit` saved). OpenSSL `libcrypto`
links cleanly → real HMAC-SHA256, not the XOR-fold fallback.

## Three-indicator diagnostic — 2 of 3 confirmed, 3rd blocked

| Indicator | Result |
|---|---|
| (1) chain advances | ✅ 257 events recorded on the eager prefill forward; chain head non-zero |
| (2) chain verifiable offline | ✅ **PASS** — `audit_verify.py` independently recomputed the HMAC-SHA256 chain from zeros over all 257 entries; recomputed head `f64b744254c4f414…` == live head `f64b744254c4f414…` |
| (3) byte-identical output | ⛔ **BLOCKED** — run crashed before producing output (see below) |

AUDIT's own observer actuators ran cleanly: matmul handler 225×,
attn handler 32× on the eager prefill — both PASSTHROUGH, no error.
**AUDIT itself works.**

## The blocking bug — composition failure, NOT AUDIT

The gate run (`libcipher_rt` LD_PRELOAD + `CIPHER_AUDIT=on` + Mistral-7B
with `StaticCache`) crashed:

```
torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call
  call_function scaled_dot_product_attention(FakeTensor(size=(1,32,1,128)...), ...)
  got RuntimeError("Cannot access data pointer of Tensor (e.g. FakeTensor,
  FunctionalTensor). ... we are erroneously tracing into a custom kernel.")
```

Diagnosis:
1. Op #1's integration model requires `cache_implementation=static`.
   `StaticCache` is "built for `torch.compile`" — transformers 5.8.1
   compiles the decode step.
2. Under `torch.compile`, dynamo traces `scaled_dot_product_attention`
   with **FakeTensors** (no real data).
3. The **T4.6.1 attention substrate** trampoline's `fill_view()` calls
   `tensor.data_ptr()` / `.sizes()` to build the call descriptor.
   `data_ptr()` on a FakeTensor throws → crash.

This is a **pre-existing T4.6.1 bug**, newly exposed: the SDPA
trampoline is not `torch.compile`-safe. T4.6.1's own smoke used
`DynamicCache` (no compilation), so the compile path was never
exercised. Op #1's `StaticCache` requirement is what triggers it.

**AUDIT is not the cause** — its handlers read pre-extracted pointers
from the call descriptor, never touch a `Tensor`. The crash is upstream
in T4.6.1's `fill_view`, before AUDIT's attn handler is reached on
compiled steps.

## Why STOP (not patch-and-continue)

Per the build's "WHEN TO STOP AND ASK": this gate failed with a failure
mode not in op #2's declared list, and "composition risk fired — a
prior op breaks when the new op turns on." Both conditions say STOP,
report, wait. The fix touches the **T4.6.1 substrate** (a prior shipped
component) — that change should be visible and approved, not done
silently inside op #2.

Masking it (forcing eager, disabling compile) is the R2-banned
"lower a gate to make it pass" — rejected. Real deployments using
`StaticCache` *will* compile; the substrate must be compile-safe.

## Proposed fix (needs approval — it modifies T4.6.1, not op #2)

Make the three T4.6.1 SDPA trampolines compile-safe: before
`fill_view()`, detect a fake/meta/traced tensor and, if so, pass
straight through to the original op without building the descriptor
(no observation on traced steps — correct, since tracing isn't real
execution). ~15–30 LOC in `cipher_rt_attn_dispatch.cpp`. Candidate
detection: guard the first `data_ptr()` in try/catch, or test the
tensor's dispatch-key set for the tracing keys — exact mechanism to be
settled in the fix.

This T4.6.1 hardening must land before op #2's byte-identical gate can
be retried, and before any further fusion op — every subsequent op runs
under `libcipher_rt` + `StaticCache`, i.e. under compilation.

## Discipline

| Gate | Result |
|---|---|
| Fallback anchors `55ab8c0c` / `86618c30` | ✅ unchanged |
| Kernel taint | ✅ 12288 |
| AUDIT mechanism (chain + offline verify) | ✅ works |
| Op #2 overall | ⛔ BLOCKED — cannot pass byte-identical gate until T4.6.1 compile-safety fix lands |
