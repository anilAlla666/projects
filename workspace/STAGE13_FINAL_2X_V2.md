# Stage 13 — Final 2× tok/W stack measurement (revised)

Date: 2026-04-30 (rev 2 — clock retune + long-context probe + V3 gate)
Pod: H100 80GB SXM, CUDA 12.8, libcublasLt 12.8.4, sm_90.

## Updated table (per-batch optimal clock, full stack, eager)

```
  B  clock   baseline tps  baseline W  baseline tok/W   cipher tps  cipher W  cipher tok/W   ×tps  ×tok/W
  1   1000          48.83       203.7         0.2397        52.54     145.4        0.3613   1.08    1.51
  8   1200         390.32       254.1         1.5363       401.31     154.8        2.5930   1.03    1.69
 32   1200        1327.40       393.3         3.3751      1084.28     217.3        4.9894   0.82    1.48
 64   1200        1744.59       440.5         3.9606      1342.69     234.7        5.7206   0.77    1.44
```

**Tokens-per-watt: 1.44× to 1.69× across the full sweep, B=8 sweet spot
hits 1.69×.**

### Clock retuning (Part 1)

| B | original clock | original ×tok/W | tried | new clock | new ×tok/W |
|---|---|---|---|---|---|
| 32 | 1350 | 1.41× | 1200 | **1200** | **1.48×** |
| 64 | 1000 | 1.43× | 1350 (1.37×) → 1200 (1.44×) | **1200** | **1.44×** |

The user asked to try B=64 at 1350 MHz. I ran it and found 1.37× — worse
than 1000's 1.43×. Tried 1200 next; that landed at 1.44×, marginally
better than 1000 with much higher tps recovery (1342 vs 1122 at 1000
MHz). I'm using 1200 as the winner for B=64. Stating it explicitly
since the user's instruction was "try X to see, then re-run":

> tried 1350 MHz at B=64, found 1200 beats both 1000 and 1350.

## Long-context probe: B=8, prefill=2048 (Part 2)

```
 baseline:  tps=299.30   W=370.2   tok/W=0.8085
 CIPHER:    tps=238.62   W=233.7   tok/W=1.0211
 ratios:    ×tps=0.80    ×tok/W=1.26
 fp8_calls: 67275
```

Stack at long context: FP8 + fusion + 1200 MHz lock + V3 wired (gated
to B=1) → **1.26× tok/W at B=8 P=2048**. Eager mode.

Compared to the short-context B=8 P=128 result (1.69×), the long-context
ratio is lower because the tok/W denominator already includes more
KV-read traffic per decode step, which FP8 doesn't address. The lever
that *would* shrink the long-context KV-read tax is V3 — see below.

## KV redirect V3 — wired in, gated to B=1

V3 is enabled in this run (`CIPHER_KV_REDIRECT=on CIPHER_KV_RDR_V3=on`)
and reports its state at init:

```
[CIPHER KV-RDR] init enabled v2_mode=0 v3_mode=1 ... batch=8
[CIPHER KV-RDR] V3 gated OFF: B=8 > 1 — V3's (token,head)-flat layout
                has no batch dimension; falling through to original
                PyTorch FA staging.
```

**Why V3 is gated to B=1, not delivered at B=8 in this stage:**

CLAUDE.md (carried over from the prior session) records the V3 state:

> V2 is correctness-only — no bandwidth win yet (the existing
> materialize still runs).

V3 inherits the same architectural limitation. The flow is:

1. K_proj GEMM produces fp16 K (we hook + quant 2-bit into a private
   compressed cache).
2. PyTorch's `StaticCache.update` copies the same fp16 K into its own
   per-layer cache slot. **(we don't intercept this)**
3. PyTorch's "materialize" kernel copies the fp16 cache slot into FA's
   1 MB staging buffer. **(we don't intercept this either)**
4. FA reads the 1 MB staging.

V3 hooks at #4: it dequants our compressed cache into the staging buffer
and rewrites the FA Params struct so FA reads our buffer. But steps #2
and #3 still happen — PyTorch still moves the fp16 K through the cache
and the materialize. So V3 *adds* dequant overhead on top of the
existing fp16 traffic; it doesn't reduce HBM bandwidth.

On top of that architectural problem, the V3 cache layout is
`(token, head)`-flat — no batch dimension — so running V3 at B>1 would
mix batch attention. I added a runtime gate (15 LOC) that detects
`CIPHER_KV_BATCH > 1` and falls through to PyTorch's original FA
staging. This makes the env vars safe to leave on at any batch.

**To make V3 deliver a real bandwidth win** requires two pieces, neither
of which fits in this session:

1. **Per-(layer, batch) cache + B-aware dequant kernel**. ~200 LOC of
   refactor in `cipher_kv_redirect.cpp`, plus tests at B=2, 4, 8.
2. **Bypass PyTorch's materialize**. The current materialize kernel
   reads the full fp16 cache slot into the 1 MB staging on every FA call.
   To save bandwidth, V3 must intercept either `StaticCache.update`
   (Python monkey-patch) or the materialize kernel itself (driver-level
   intercept by signature). Without one of these, the fp16 traffic
   continues regardless of V3's compressed cache.

Both are real follow-on work. I confirmed this conclusion mid-implementation
(advisor reviewed) and stopped — refactoring V3 to be B-aware would land
a clean correctness improvement but no measurable tok/W gain in eager
mode without the materialize-bypass piece on top.

## What's shipped this round

- `run_final_2x_measurement.sh` — updated clocks (B=64=1350 → 1200 chosen empirically as best)
- `rerun_b32_b64.sh` — incremental re-run script
- `run_b8_p2048.sh` — long-context probe
- `src/cipher_kv_redirect.cpp` — added `g_running_batch` + `CIPHER_KV_BATCH`
  env, V3 hooks gated to B=1 only (16 LOC)
- `STAGE13_FINAL_2X_V2.md` — this file

## Honest read of the numbers

- B=1 / 8 / 32 / 64 short-context (P=128): **1.51 / 1.69 / 1.48 / 1.44 ×
  tok/W** with full stack (FP8 + fusion + clock + all ops on).
- B=8 long-context (P=2048): **1.26 × tok/W**.
- The 2× number isn't hit anywhere in this measurement. The B=8 short-
  context 1.69× is the closest. To get to 2× requires either
  (a) graph capture in addition to FP8 (the prior session's CLAUDE.md
  recorded 7.6× tok/W with graph + INT4 GEMV, but that was graph mode
  and the spec for this session is "EAGER mode only"), or
  (b) a working V3 with materialize-bypass at long context.

The deliverable for this session is the FP8 lever — wired, correct,
firing on every linear, and stacking with fusion + clock to deliver a
sustained tok/W win across the batch sweep. The 2× ceiling is one more
lever (graph capture or KV-bw saving) away.
