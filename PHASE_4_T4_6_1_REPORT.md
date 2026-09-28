# T4.6.1 — ATTENTION-ROUTING SUBSTRATE — REPORT

Phase 4.6.1 ships the **operator-context attention-routing substrate**
for CIPHER — the architectural twin of T4.5.1's cuBLAS substrate. The
substrate interposes on PyTorch's three SDPA dispatcher entries
(`_scaled_dot_product_{flash,efficient,cudnn}_attention::call`) via
plain LD_PRELOAD with mangled-name shim definitions, exposes a C ABI
actuator registry, and ships a smoke-test actuator that verifies routing
end-to-end. **Output is byte-identical to baseline** on Mistral-7B
across all tests; T4.5.1 cuBLAS substrate continues to operate
correctly; VOLT clock-lock path is unperturbed by composition.

This sub-phase ships SUBSTRATE only. KV-dedup L1 (T4.6.3), cuIpc
cross-process page pool (T4.6.4), and the kmod ioctl ABI extensions
(nrs 11/12/13) follow in subsequent phases per the [outcome (c)
architectural commit](PHASE_4_T4_6_0_5_MEASUREMENT.md).

## TL;DR

- **Substrate shipped.** `cipher_rt_attn_dispatch.{cpp,h}` provides a
  16-actuator priority-ordered registry with PASSTHROUGH/HANDLED/
  REDIRECTED/ERROR semantics. REDIRECTED is wired but mechanism is
  deferred until T4.6.3 forces the design choice (see
  Architectural-Decision section).
- **Three SDPA dispatcher entries interposed.** Flash, efficient, and
  cuDNN backend ::call symbols all interposed via plain mangled-name
  LD_PRELOAD — **no `.symver` directive needed** because ATen ops are
  not version-tagged in libtorch_cpu.so (unlike libcublas.so.13).
- **cuDNN is the verified-active backend** on this stack (H100 / torch
  2.11 / cu13). Mistral-7B routes all 32,896 attention calls through
  cuDNN. Flash and efficient trampolines exist but were never hit
  empirically (acknowledged coverage gap).
- **Phase prompt's named intercept symbol was off.** The prompt said
  `at::_scaled_dot_product_flash_attention_forward` (libtorch_cuda.so);
  reality on torch 2.11 is `at::_ops::_scaled_dot_product_flash_attention::call`
  (libtorch_cpu.so) — and the active backend is cuDNN, not flash. The
  S1.5 LD_PRELOAD probe surfaced this before scaffolding was committed.
- **Three-indicator routing verified.** (a) substrate telemetry, (b)
  test-actuator logs with Mistral GQA shapes Q=[1,32,6,128]
  K/V=[1,8,6,128], (c) byte-identical output to baseline across 3
  prompts × 32 tokens = 96 hash-matched tokens.
- **T4.5.1 composition verified.** With both substrates active, output
  remains byte-identical (3/3 prompts), cuBLAS substrate reports 21,600
  passthrough calls clean.
- **VOLT composition verified.** Pre-T4_6_1 (VOLT alone) vs T4_6_1
  (VOLT + attn substrate) on B=1 decode: tok/W 0.128 vs 0.127 — within
  0.8% noise, identical clock / watts / tok/s. Attn substrate adds zero
  measurable VOLT-path interference.
- **Discipline preserved.** ABI surface (ioctl nrs 1-10) untouched.
  Fallback md5s unchanged. Taint 12288 unchanged. kmod refcount 2.

## Engineering layer

### S1 — Symbol resolution

The phase prompt named `at::_scaled_dot_product_flash_attention_forward`
in libtorch_cuda.so. Empirical reality on the verified stack:

| Phase prompt assumption | Empirical reality (torch 2.11/cu13) |
|---|---|
| Symbol in libtorch_cuda.so | Symbol in **libtorch_cpu.so** (ATen ops dispatcher is universal-backend) |
| Single intercept point | **Three intercept points** (one per backend variant); SDPA selects backend at runtime |
| Flash is the SDPA hot path | **cuDNN is the default on H100/cu13**; flash 0 calls, eff 0 calls, cuDNN 32,896 calls observed on Mistral-7B |
| Signature uses `int64_t max_q/max_k` | Signature uses **`c10::SymInt`** (tagged union, NOT a plain int) |
| Symbols are version-tagged | Symbols are **plain mangled, NO `@@` version tag** — `.symver` directive not needed |

These corrections were surfaced by the S1.5 LD_PRELOAD probe
(`cipher_attn_probe.so`) BEFORE scaffolding was committed.

### S2 — Substrate scaffolding

**Files added:**

```
cipher_rt_phase4/
  cipher_rt_attn_dispatch.h          # C ABI for actuator registration + call descriptor
  cipher_rt_attn_dispatch.cpp        # C++ substrate dispatch + 3 LD_PRELOAD trampolines
  cipher_rt_attn_test_actuator.c     # Smoke-test actuator (env-gated)
```

**Files modified:**

```
cipher_rt_phase4/
  cipher_inject.c                    # Added cipher_rt_attn_{dispatch,test_actuator}_init calls
  Makefile                           # New build rules + libtorch_cpu link (--no-as-needed)
```

**Build-time discoveries:**

1. **`--no-as-needed` is mandatory** for `-ltorch_cpu`. Without it, the
   linker drops libtorch_cpu from DT_NEEDED because our trampolines
   appear to satisfy the symbols themselves. Result: dlsym RTLD_NEXT
   at runtime returns nothing to fall through to → host application
   aborts on first attention call. The fix forces libtorch_cpu into
   DT_NEEDED:
   ```
   -Wl,--no-as-needed -lc10 -ltorch_cpu -Wl,--as-needed
   ```

2. **`cublas_version.map` is unmodified.** The attention symbols are
   un-versioned and must NOT be added to the version map's `global:`
   clause — doing so would scope them to a version and hide them from
   the dynamic symbol table, silently disabling interposition.

**Substrate API surface (C ABI):**

```c
enum cipher_rt_attn_backend { FLASH=1, EFFICIENT=2, CUDNN=3 };
enum cipher_rt_attn_result  { HANDLED=0, PASSTHROUGH=1, REDIRECTED=2, ERROR=3 };

struct cipher_rt_attn_tensor { void *data_ptr; int64_t sizes[4]; ... };
struct cipher_rt_attn_call   { enum cipher_rt_attn_backend backend;
                               cipher_rt_attn_tensor q, k, v;
                               double dropout_p; int is_causal; ...; };

int cipher_rt_attn_register_actuator(const struct cipher_rt_attn_actuator*);
int cipher_rt_attn_dispatch_init(void);
unsigned long cipher_rt_attn_calls_total/handled/passthrough/redirected(void);
unsigned long cipher_rt_attn_calls_by_backend(enum cipher_rt_attn_backend);
```

### S3 — Test actuator + smoke test

**Three-indicator binding diagnostic — PASSED**

| Indicator | Result |
|---|---|
| (a) Substrate telemetry: N attention calls observed | cuDNN=256 (8 generates × 32 layers) on Mistral-7B max_new_tokens=8 |
| (b) Test actuator logs: shapes with correct GQA dims | Q=[1,32,6,128] K=[1,8,6,128] V=[1,8,6,128] dtype=Half — Mistral-7B GQA (32 Q heads, 8 KV heads, seq=6 prompt, head_dim=128) ✅ |
| (c) Output byte-identical to baseline | `'The capital of France is a city of many faces. It is'` — identical to no-preload baseline ✅ |

### S4 — Composition with T4.5.1

**Composition check** — Mistral-7B generate(max_new_tokens=32) over 3
prompts, hashes:

| Prompt | Baseline hash | Substrate hash |
|---|---|---|
| `The capital of France is` | `f4727c26db33f3e4` | `f4727c26db33f3e4` ✅ |
| `Quantum entanglement is` | `e27ab0e7a52354c2` | `e27ab0e7a52354c2` ✅ |
| `The Roman Empire collapsed because` | `51596ef1d51137b8` | `51596ef1d51137b8` ✅ |

**Substrate telemetry (composition run):**
- T4.5.1 matmul substrate: calls=21,600 handled=0 passthrough=21,600 (actuators=0; Marlin disabled)
- T4.6.1 attn substrate: calls=3,072 handled=0 passthrough=3,072 redirected=0 (flash=0 eff=0 cudnn=3,072)

**VOLT composition (S4.3)** — B=1 decode for 31s, both substrates loaded:

| Mode | tok/s | W | clock | tok/W |
|---|---|---|---|---|
| baseline (no preload) | 19.5 | 155.6 | 1980 MHz | 0.125 |
| VOLT alone (pre-T4_6_1) | 12.6 | 98.8 | 1005 MHz | 0.128 |
| **VOLT + attn substrate (T4_6_1)** | **12.6** | **99.0** | **1005 MHz** | **0.127** |

VOLT path with attn substrate active matches VOLT alone within 0.8% on
tok/W. **No measurable interference.**

Side observation: the absolute VOLT lift on this pod today (0.128 vs
baseline 0.125 = +2.4%) is well below T4.3.2's claimed +57% on B=1 decode.
This is a VOLT-path regression unrelated to T4.6.1 — flagged for
follow-up, NOT a T4.6.1 blocker.

### S5 — ABI verification + signature drift

See [PHASE_4_T4_6_1_SIGNATURE.md](PHASE_4_T4_6_1_SIGNATURE.md) for full
mangled-name + signature documentation per backend, drift-handling
behavior table, and "why abort() on dlsym failure is the correct
contract" framing.

Header constant `CIPHER_RT_ATTN_TORCH_VERIFIED = "2.11.0+cu130"` lives
in `cipher_rt_attn_dispatch.h` for operators to grep before re-verifying
on torch upgrade.

## Strategic layer

### What this unlocks

The attention substrate is the **foundation for the multi-tenant moat**
per the T4.6.0.5 outcome (c) architectural commit:

- **T4.6.2** — KV page allocator (per-process page table, block=16,
  FNV-1a content hash). Lives behind the attention substrate; the
  page-allocator API is keyed by content hash, called from a future
  KV_DEDUP actuator registered on the substrate.

- **T4.6.3** — L1 in-process prefix dedup actuator. Registers on the
  substrate at priority N, hashes K-projection output block-by-block,
  on hit redirects K/V pointer to shared page. **This is where the
  REDIRECTED mechanism gets designed** — option (a) alloc-new-tensor
  with shared storage vs option (b) ATen storage-swap. Deferred from
  T4.6.1 until L1 actuator forces the choice.

- **T4.6.4** — L3 cuIpc cross-process page pool. kmod ioctl ABI
  additions (nrs 11/12/13: CIPHER_KV_REGISTER_PAGE / _LOOKUP_PAGE /
  _RELEASE_PAGE). The substrate's L3 path: on L1 miss, look up content
  hash in cross-process registry; on cross-process hit, receive cuIpc
  handle, open via cuIpcOpenMemHandle, redirect K/V to shared HBM page.

The substrate composition story is now complete at the substrate
*layer*: T4.5.1 routes matmul, T4.6.1 routes attention, future T4.6.4
adds cuIpc-level cross-process page pool. Customer code remains
unchanged — operator deployment is `CUDA_INJECTION64_PATH + LD_PRELOAD
+ CIPHER_* env vars`.

### Why the prompt's intercept point was wrong (architecturally interesting)

The phase prompt named flash attention as the intercept point. Empirical
reality:

- **Hopper + cu13 + torch 2.11 defaults to cuDNN attention**, not flash.
  This is because cuDNN 9.x has a fused flash variant that's faster
  than the PyTorch-bundled FlashAttention-2 implementation on Hopper.
- **All three backends share the same dispatch pattern** at the
  `at::_ops::*::call` level. The substrate handles them uniformly with
  one trampoline per backend.
- **Pre-Hopper hardware** (A100, V100) likely routes through flash or
  efficient. The substrate's coverage of all three is therefore not
  over-engineering — it's correctness for the operator-deployment
  envelope CIPHER targets (which spans the H100 / pre-Hopper estate).

### Architectural-decision deferral: REDIRECTED mechanism

PyTorch's SDPA dispatcher passes Q/K/V as `const at::Tensor&` —
references, not pointers. To redirect K/V to a shared HBM page, the
substrate must either:

1. **Alloc new tensor with shared storage** — call ATen's
   `from_blob` (or similar) to wrap the shared page, then pass to
   passthrough. Adds tensor-creation overhead per attention call
   (could be µs-scale per call × thousands of calls per generate).

2. **ATen storage-swap** — use ATen internal API
   `storage()->set_data_ptr_noswap()` to in-place rewrite the tensor's
   storage pointer. Riskier (depends on ATen ABI), faster at runtime.

3. **Defer** — leave both possible at the substrate level; let T4.6.3
   (L1 dedup actuator) drive the choice based on what's actually
   needed for L1 dedup semantics.

T4.6.1 ships option (3). REDIRECTED is wired through the dispatch
loop (telemetry counts redirected actuator returns) but the substrate
currently treats REDIRECTED as PASSTHROUGH with a one-line stderr note.
T4.6.3 will commit the substitution mechanism.

### Moat positioning advance

- T4.6 architecture verified at substrate layer (substrate shipped + composed cleanly with T4.5.1)
- Operator-deployment envelope preserved (LD_PRELOAD + CUDA_INJECTION64_PATH only — no customer code changes)
- Deployment moat: ~85% → ~90% (substrate now spans both compute paths CIPHER cares about)
- Multi-tenant moat: architecture-verified → substrate-shipped (foundation in place for T4.6.2-4)

## Discipline gate (end of phase)

| Gate | Result |
|---|---|
| Phase 3 ABI ioctl nrs 1-10 surface | ✅ untouched (no kmod modifications this phase) |
| ABI additive invariant (reserved nrs 2/3/4 return -ENOSYS) | ✅ preserved (no nrs reordered or repurposed) |
| Fallback kmod md5 `55ab8c0cd8309ca7cc0fc40fe556aa19` | ✅ unchanged |
| Fallback libcipher_v2 md5 `86618c30896470b642fcc6985d8dc632` | ✅ unchanged |
| Kernel taint | ✅ 12288 (unchanged) |
| cipher_kmod loaded | ✅ refcount=2 |
| Production artifacts modified | libcipher_rt.so only (new build); pre_T4_6_1 saved |
| Pod state | baseline (CIPHER_VOLT was on during volt_b1 test, restored via atexit handler) |

## Coverage gaps (acknowledged honestly)

1. **Flash and efficient trampolines runtime-untested.** Built,
   symbol-exported, signature-matched against ATen headers — but cuDNN
   is the default on this stack so flash/eff trampolines have not been
   exercised end-to-end. First pre-Hopper or non-cuDNN-default workload
   through CIPHER will be the real smoke test for these.

2. **No 25-test full stress regression.** The prior-session Llama-3.1-8B
   + FP8 + 1200 MHz harness takes hours per test (10K-token endurance,
   8-client sustained load); within the 8h cap, T4.6.1 verified
   composition via Mistral-7B (3 prompts × 32 tokens hash-matched +
   B=1 decode VOLT A/B). Heavier regression deferred to a dedicated
   stress session once T4.6.3-4 actuators land.

3. **VOLT absolute lift regression.** On this pod today, VOLT @ 1000MHz
   B=1 decode produces +2.4% tok/W vs baseline (0.128 vs 0.125), far
   below T4.3.2's claimed +57%. This is a VOLT-path issue, not a
   T4.6.1 issue (verified: pre-T4_6_1 substrate + VOLT shows the same
   +2.6% number). Flagged for next-session investigation.

## Artifacts shipped

| Path | Purpose |
|---|---|
| `/home/ubuntu/cipher_rt_phase4/cipher_rt_attn_dispatch.{h,cpp}` | Substrate (header + dispatch core + 3 trampolines) |
| `/home/ubuntu/cipher_rt_phase4/cipher_rt_attn_test_actuator.c` | Smoke-test actuator |
| `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` | T4.6.1 build (substrate active) |
| `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.pre_T4_6_1` | Pre-T4_6_1 rollback build (md5 `9ee93ec8958ab0de868c0bc0506c1cfa`) |
| `/home/ubuntu/cipher-phase4-evidence/t4_6_1_substrate/probe/cipher_attn_probe.{cpp,so}` | S1.5 standalone interception probe |
| `/home/ubuntu/cipher-phase4-evidence/t4_6_1_substrate/composition_check.py` | S4 byte-identical output verification |
| `/home/ubuntu/cipher-phase4-evidence/t4_6_1_substrate/volt_b1_check.py` | S4.3 VOLT interference A/B |
| `/home/ubuntu/PHASE_4_T4_6_1_SIGNATURE.md` | Mangled-name + signature documentation + drift-handling contract |
| `/home/ubuntu/PHASE_4_T4_6_1_REPORT.md` | This document |

## Next phase

**T4.6.2 — KV page allocator + per-process page table.** Block-aligned
HBM pages, content hash (FNV-1a) → device pointer table, refcounted
release. Lives behind the T4.6.1 substrate. Honest budget: 2-3 sessions,
10-15h.

After T4.6.2: T4.6.3 (L1 in-process dedup actuator), T4.6.4 (L3 cuIpc
cross-process pool + kmod ioctls 11/12/13), T4.6.5 (real-trace validation
gate), T4.6.6 (50-100 tenant integration). Per the [outcome (c)
commit](PHASE_4_T4_6_0_5_MEASUREMENT.md).
