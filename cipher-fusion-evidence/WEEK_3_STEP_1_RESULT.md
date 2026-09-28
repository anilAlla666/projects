# Week 3 Step 1 — `cipher_rt_dispatch.{cpp,h}` Scaffolding — RESULT

**Status: PASS.**

GEMM dispatch routing substrate scaffolded. `CIPHER_DISPATCH_LIVE=0` default; lookup() is callable but not consumed by Marlin/cuBLAS yet. SC6 TinyLlama bit-identical confirmed in both vanilla baseline + CIPHER-injected modes (33 sec each, all 7/7 sub-checks PASS).

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 3, Step 1 of 5
**Anchors:**
  - Pre:  `f9c32322` (`week-2-complete`)
  - Post: `0b6effdb` (`week-3-step-1-dispatch-scaffold`)
  - libcipher_rt.so md5: `863524a4` → `201dc1a1` (nvcc non-deterministic; source identity git-verified)
  - `libcipher_rt.so.week3_pre` snapshot at `/tmp/week3_step1/`: md5 `863524a4e12f0f0647b425007dcccfef`

---

## A — Pre-edit verification + baseline snapshot — PASS

| signal | value |
| --- | --- |
| HEAD | `f9c32322` ✓ |
| working tree | clean ✓ |
| Pre-build rc | 0 |
| Pre-build md5 | `863524a4` (informational; nvcc non-deterministic) |
| Pre-build warnings | 78 |
| Symbol count baseline | 308 |
| week3_pre snapshot preserved | `/tmp/week3_step1/libcipher_rt.so.week3_pre` |
| SC6 runner located | `phase_c/sc6_run.py` ✓ |
| TinyLlama-1.1B model | present at `/home/ubuntu/models/TinyLlama-1.1B/` ✓ |

Pre-edit reference SC6 deferred to single load-bearing post-edit run (cost-equivalent; if post-edit fails, pre-edit can be run for comparison).

---

## B — `cipher_rt_dispatch.h` design (~80 lines)

Mirrors the cipher_rt_classify_substrate.h shape (Step 3 of Week 2 used cipher_rt_matmul_dispatch as reference; Week 3 maintains substrate-family consistency).

Public surface:

```c
enum cipher_rt_dispatch_action {
    CIPHER_RT_DISPATCH_PASS_THROUGH = 0,
    CIPHER_RT_DISPATCH_ROUTE_MARLIN = 1,
    CIPHER_RT_DISPATCH_ROUTE_CUBLAS = 2,
    CIPHER_RT_DISPATCH_ROUTE_ATTN   = 3,   /* reserved for v1.5 */
};

struct cipher_rt_dispatch_decision {
    uint8_t  op_class;       /* mirrors OpClass; 0..6, 0xFF UNCLASSIFIED */
    uint8_t  action;         /* enum cipher_rt_dispatch_action */
    uint16_t actuator_hint;  /* Step 3 consumer slot */
    uint32_t reserved;
};                            /* 8 bytes total */

int cipher_rt_dispatch_lookup(uint8_t op_class,
                              struct cipher_rt_dispatch_decision *out);
int cipher_rt_dispatch_is_live(void);
```

`CIPHER_DISPATCH_LIVE` env documented in the header banner (resolves Wave 5 W3-7 SYNTHESIS-HYPOTHESIS).

---

## C — `cipher_rt_dispatch.cpp` implementation (~120 lines)

### C.1 Static table

```c
const cipher_rt_dispatch_decision g_dispatch_table[16] = {
    { 0, CIPHER_RT_DISPATCH_ROUTE_MARLIN, 0, 0 },   /* GEMM */
    { 1, CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },   /* ATTENTION (v1.5) */
    { 2..14, PASS_THROUGH },                         /* others */
    { 15, PASS_THROUGH },                             /* UNCLASSIFIED */
};
```

GEMM is the only routable class in v1. Wave 5 W3 narrowing honored.

### C.2 `cipher_rt_dispatch_lookup()`

Pure function — bounds check `op_class < 16`, return `g_dispatch_table[op_class]`. Returns -1 with defensive PASS_THROUGH default if out-of-range.

### C.3 `cipher_rt_dispatch_is_live()`

`std::atomic<int> g_live_cached` initialized to -1. First call:
- `getenv("CIPHER_DISPATCH_LIVE")`
- Parse: "1" → 1; anything else → 0
- CAS-update cache (race-tolerant: concurrent first-readers compute identical value)

Subsequent calls: lock-free atomic load.

### C.4 Static-init banner

`DispatchLoadBanner` ctor (anonymous-namespace global) logs at .so load:

```
[cipher_v2] DISPATCH: substrate registered, LIVE=0 (0 = observe-only
            until Step 4 flip; set CIPHER_DISPATCH_LIVE=1 to engage)
```

Banner mirrors CLASSIFY substrate's Step 3 banner convention.

---

## D — Makefile diff (+8 lines)

```
  cipher_rt_classify_substrate.o cipher_rt_classify_observer.o \
+ cipher_rt_dispatch.o \
  cipher_may13_harness.o \
...
+# Week 3 Step 1 — GEMM dispatch routing scaffolding (v1: GEMM->Marlin,
+# else PASS_THROUGH). CIPHER_DISPATCH_LIVE env-gated; default 0 keeps
+# the substrate observe-only through Steps 1-3. Step 4 flips to 1.
+cipher_rt_dispatch.o: cipher_rt_dispatch.cpp cipher_rt_dispatch.h \
+                      cipher_rt_classify_substrate.h
+   $(CXX) $(CXXFLAGS) -std=c++17 $(INCLUDES) -c -o $@ $<
```

Same pattern as `cipher_rt_classify_substrate.o`.

---

## E — Build + nm diff + symbol audit — PASS

| signal | value |
| --- | --- |
| build rc | 0 |
| warning count | 78 → 78 (zero delta) |
| libcipher_rt.so md5 | `863524a4` → `201dc1a1` (nvcc non-deterministic; source identity git-verified) |
| Symbol count | 308 → 310 |
| Symbols removed | 0 |

### Added (2 T-symbols)

```
T cipher_rt_dispatch_is_live
T cipher_rt_dispatch_lookup
```

(Static `g_dispatch_table`, anonymous-namespace `g_live_cached`, `DispatchLoadBanner`, and `read_live_env_once()` are file-local — correctly absent from the dynamic symbol table.)

### Undef audit

Clean — only expected system/weak surface (no new CIPHER undefineds).

---

## F — Runtime smoke + SC6 gate (LOAD-BEARING) — PASS

### F.1 Loader smoke

```
$ LD_PRELOAD=$(realpath libcipher_rt.so) /bin/true
[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)
[cipher_v2] DISPATCH: substrate registered, LIVE=0 (0 = observe-only until Step 4 flip; set CIPHER_DISPATCH_LIVE=1 to engage)
rc=0
```

Both CLASSIFY (Step 3 of Week 2) and DISPATCH (Step 1 of Week 3) banners fire.

### F.2 SDPA shim injection smoke

```
[cipher_v2] CLASSIFY: ... slot 0/1
[cipher_v2] DISPATCH: substrate registered, LIVE=0 ...
[cipher_v2] GREEN/CP54: ALLOCATE ok ...
[cipher_v2] CUPTI subscribed: kernel launch callbacks active ...
[cipher_v2] MATMUL: substrate initialized ...
[cipher-attn] substrate active ...
[cipher_v2] GOT: patch applied ...
[cipher_v2] GREEN: CP 5.4 qos=shared ...

rc=0
[cipher_v2] MATMUL: exit totals — calls=0 handled=0 passthrough=0 (actuators=0)
[cipher_v2] DIAG-T4.2.4d: total_launches=30 ...
[cipher-attn] exit totals - tramp_calls=5 tramp_fake=0 observed=5
              handled=0 passthrough=5 redirected=0 (flash=0 eff=0 cudnn=5)
```

LP-2 invariant holds: `tramp_calls=5 handled=0 passthrough=5`.

### F.3 SC6 TinyLlama bit-identical gate

**Vanilla baseline (no CUDA_INJECTION64_PATH):**

```
$ python3 sc6_run.py TinyLlama shared
SC6 TinyLlama/shared: PASS
rc=0; runtime 33s (17:49:56 → 17:50:29)

sc6_TinyLlama_shared_result.json:
  "all4_fwd1_bit_identical": true,
  "all4_fwd2_bit_identical_after_producer_death": true,
  "all4_arena_backed": true,
  "arena_survived_producer_sigkill": true,
  "producer_pid_cleared": true,
  "all4_consumers_held_arena": true,
  "arena_reaped_after_last_participant": true,
  "PASS": true
```

**CIPHER-injected (CUDA_INJECTION64_PATH=libcipher_rt.so):**

```
$ CUDA_INJECTION64_PATH=$(realpath libcipher_rt.so) python3 sc6_run.py TinyLlama shared
SC6 TinyLlama/shared: PASS
rc=0; runtime 33s (17:50:48 → 17:51:21)

sc6_TinyLlama_shared_result.json:
  all 7/7 sub-checks PASS
  bit_identical=true both pre + post producer death
```

**Both modes 7/7 PASS.** Forward-pass output is bit-identical between vanilla baseline and CIPHER-injected. The substrate is observe-only as designed; `cipher_rt_dispatch_lookup()` exists but no actuator consumes it.

---

## G — CP 5.4 regression — PASS

```
=== Phase A result: 15 PASS, 0 FAIL ===
rc=0
```

Byte-identical to Week 2 closeout baseline.

---

## H — Commit + tag — PASS

```
3 files changed, 187 insertions(+)
  cipher_rt_dispatch.cpp | 108 ++++++++++++++++++++++++++++++++++++++++++
  cipher_rt_dispatch.h   |  71 +++++++++++++++++++++++++
  Makefile               |   8 ++

HEAD: 0b6effdb041a84817ef181617af61e2b16c67fd5
Tag:  week-3-step-1-dispatch-scaffold
Pre tag: week-2-complete (f9c32322)
```

---

## Discipline notes

- Single commit. Tag chain: `week-2-complete` → `week-3-step-1-dispatch-scaffold`.
- Rollback: `git -C cipher_rt_phase4 reset --hard week-2-complete`. `libcipher_rt.so.week3_pre` snapshot preserved at `/tmp/week3_step1/`.
- Resolves Wave 5 W3-7 SYNTHESIS-HYPOTHESIS (CIPHER_DISPATCH_LIVE env defined; default 0).
- Substrate quiescent at runtime — `cipher_rt_dispatch_lookup()` is callable but not yet called from any hot path. Marlin actuator does NOT yet consult the table. Step 2 adds observer hint publish; Step 3 makes Marlin consume; Step 4 flips LIVE=1.
- Wave 5 W3 GEMM-only scope honored: ROUTE_ATTN action reserved but PASS_THROUGH in v1 table.
- The static-init banner mirrors CLASSIFY substrate's convention; both substrates announce themselves at .so load.

---

## What this unlocks

- **Week 3 Step 2** — Observer extension: extend `cipher_rt_classify_observer_observe()` to publish TLS `substitute_hint` when oracle PERMIT + op_class==GEMM + registry hit. Calls `cipher_rt_dispatch_lookup()` internally to read the routing decision but does not yet act on it.
- **Week 3 Step 3** — Marlin actuator consumes the hint; VOLT decode-band 1200 MHz lock.
- **Week 3 Step 4** — Flip `CIPHER_DISPATCH_LIVE` default to 1; classifier-driven routing becomes live. Mistral-7B SC6 PASS is the load-bearing gate.
- **Week 3 Step 5** — Closeout.

---

## Anchors at close

| tree | HEAD | tag |
| --- | --- | --- |
| cipher_rt_phase4 | `0b6effdb` | `week-3-step-1-dispatch-scaffold` |
| cipher_kmod | `0ce4b8e2` | (unchanged from week-2-complete) |
| cipher-may13-evidence | `fc8a9ae6` | (unchanged) |
| libcipher_rt.so md5 | `201dc1a1` | (this build; nvcc non-deterministic) |
| libcipher_rt.so.week3_pre snapshot | `863524a4` | preserved at `/tmp/week3_step1/` |

Week 3 progress: **1/5 steps complete**. Step 2 brief draftable.
