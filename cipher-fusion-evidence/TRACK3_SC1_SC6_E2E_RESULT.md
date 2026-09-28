# Track 2/3 SC1-SC6 end-to-end regression gate — W7-9 Step 5 pre-condition

**Date:** 2026-05-23
**Trigger:** W6 G1+G2 cap-bump residue (memory `g1-g2-cap-bump`) + W7-9 Step 5 prerequisite
**Outcome:** **ALL 6 SCs PASS — Step 5 N=128 soak GATE OPEN**

## 1. Pre-conditions verified

| Item | Expected | Observed | OK |
|------|----------|----------|----|
| `cipher_kmod` tag | `week-7-step-3-g6-audit-chain` (Step 4 did not touch kmod) | `week-7-step-3-g6-audit-chain` | ✓ |
| `cipher_kmod` MODULE_VERSION | 0.6.0 | 0.6.0 | ✓ |
| `cipher_kmod.ko` md5 | `bb42dc1f…` | `bb42dc1fc8a74d4641030008bd804ecb` | ✓ |
| `cipher_kmod` loaded | yes | `lsmod` shows `cipher_kmod 6713344` | ✓ |
| `cipher_rt_phase4` tag | `week-7-step-4-overlay-port-hotpath` | `week-7-step-4-overlay-port-hotpath` | ✓ |
| `libcipher_rt.so` md5 | `6afa5a4b…` | `6afa5a4b7c4c515cbdae8e316b760662` | ✓ |
| `cipher_kv_bridge` Python importable | yes | `cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` resolved | ✓ |

Note on label: the user prompt names this "Track 3 SC1-SC6 suite" but the
on-disk file set is `phase_c/sc[2-6]_*.py` and the closeout doc
(`phase_c/TRACK_2_CLOSEOUT.md`) brands it Track 2 cross-tenant weight sharing.
SC1 was a design memo only ("paperwork; none" in the §7 sub-component
table) so the runnable suite is SC2-SC6.

## 2. Suite location

Canonical path: `/home/ubuntu/cipher-fusion-evidence/phase_c/`

Driver entry points:

| SC | Driver(s) | Helper(s) |
|----|-----------|-----------|
| SC2 | `sc2_verify.py` | — |
| SC3 | `sc3_run.py` | `sc3_producer.py`, `sc3_consumer.py` |
| SC4 | `sc4_run.py` + `sc4_unit.py` | `sc4_producer.py`, `sc4_consumer.py` |
| SC5 | `sc5_run.py` + `sc5_unit.py` | `sc5_producer.py`, `sc5_consumer.py`, `sc5_probe*.py`, `sc5_arena_ioctl.py` |
| SC6 | `sc6_run.py` + `sc6_aggregate.py` | `sc6_models.py`, `sc6_producer.py`, `sc6_consumer.py`, `sc6_independent_tenant.py` |

## 3. sc5_arena_ioctl.py constant update (memory #15 follow-up #1)

Stale constants found and updated:

```diff
- CIPHER_WA_MAX_ARENAS = 16
+ CIPHER_WA_MAX_ARENAS = 100  # W6 G1+G2 cap bump 16->100; kmod 0.5.0+
```

```diff
- assert ctypes.sizeof(ArenaQuery) == 392, ctypes.sizeof(ArenaQuery)
+ assert ctypes.sizeof(ArenaQuery) == 2408, ctypes.sizeof(ArenaQuery)
```

ArenaQuery layout: `u32 n_arenas + u32 reserved + (u32+u32+u32+u32+u64) × N`
= 8 + 24·N. With N=100, 8 + 2400 = 2408 bytes. Matches `_IOC` macro
expansion used by `CIPHER_ARENA_QUERY` (NR 24).

SC5 unit 15/15 PASS confirms the kmod-side ABI accepts the larger struct.
**Memory #15 follow-up #1 → CLOSED.**

## 4. SC1-SC6 results

| SC | Description | Runtime | Outcome | Notes |
|----|-------------|---------|---------|-------|
| SC1 | design memo — interception at model-load via the VMM bridge | — (paperwork) | **N/A** | no `.py` to run |
| SC2 | producer-side weight arena + 6 gates (TinyLlama-1.1B) | ~30 s | **PASS** | 6/6 gates; KL=0; bit-identical memcmp of embed_tokens; export_fd valid; 201 tensors / 2.20 GiB arena |
| SC3 | weight-sharing integration — same-VA + offset-relative consumer | ~70 s | **PASS** | both modes bit-identical (max abs diff 0.0); consumer added 0 MiB; arena-backed verified |
| SC4 unit | fingerprint unit tests (synthetic models) | ~5 s | **PASS** | 11/11 (deterministic, tier-1/2 discrimination, sampling, verify behaviour) |
| SC4 run | model-identity integration — matching + mismatch fallback | ~3 min | **PASS** | matching: shared, bit-identical, 0 MiB added; mismatch (TinyLlama producer vs Llama-3.2-1B consumer): tier-1 mismatch caught, fallback to independent load |
| SC5 unit | kmod weight-arena registry ioctls (uses updated 2408-byte ArenaQuery) | ~3 s | **PASS** | 15/15 (register, query, import, multi-consumer, ENOENT, ENOSPC, leave, reap) |
| SC5 run | producer-dies-first (SIGKILL producer mid-flight; A imported pre-kill, B imported post-kill) | ~80 s | **PASS** | 12/12 checks; arena survived SIGKILL; producer_pid cleared by reaper; both consumers bit-identical post-death |
| SC6 TinyLlama shared | N=4 weight-sharing integration | ~2 min | **PASS** | 7/7 checks; all 4 fwd1+fwd2 bit-identical; arena reaped after last participant |
| SC6 TinyLlama independent | N=5 independent control | ~1 min | **PASS** | 5/5 fit; fb_loaded=14572 MiB; per-tenant 2914 MiB |
| SC6 Mistral-7B shared | N=4 weight-sharing integration | ~6 min | **PASS** | 7/7 checks; fb_loaded=17572 MiB |
| SC6 Mistral-7B independent | N=5 independent control | ~3 min | **PASS** | 5/5 fit; fb_loaded=73332 MiB; per-tenant 14666 MiB |
| SC6 aggregate | substrate-value aggregator | <5 s | **PASS** | TinyLlama N=4 savings **59.9%** (W=2181 MiB, C=733 MiB; asymptote 74.8%); Mistral-7B N=4 savings **76.0%** (W=13940 MiB, C=726 MiB; asymptote 95.0%) |

Total wall-clock for the run: **~18 minutes**. No regressions.

## 5. Comparison vs Track 2 closeout (2026-05-19)

`phase_c/TRACK_2_CLOSEOUT.md` headline at close:
- N=4 Mistral-7B savings **~76.0%** (17572 / 73332 MiB)

This run measures Mistral-7B N=4 savings = **76.0%** exact match — the
substrate is byte-identical in behaviour after the W6 G1+G2 cap bumps
plus the W7 Step 1/2/3/4 ABI growth. **No FSM-level regression** despite
the kmod struct growing 336 → 400 B (Step 3) and ArenaQuery cap 16 → 100
(W6 G1+G2).

## 6. W7-9 Step 5 pre-condition gate

**PASS.** All SC2-SC6 byte-identical (or numerically tighter) vs the
Track 2 closeout headline. The substrate is regression-free under:

- W6 G1+G2 cap bumps (CIPHER_CP54_MAX_ALLOCS 64→128; CIPHER_WA_MAX_ARENAS 16→100)
- W7-9 Step 1 G10 model_uuid field added to `cipher_tenant_snapshot`
- W7-9 Step 2 COMMIT primitive (cipher_rt_phase4 BSS snapshot table)
- W7-9 Step 3 G6 HMAC chain (kmod struct 336 → 400 B; mmap'd ring)
- W7-9 Step 4 overlay-op _report() port + COMMIT hot-path + AUDIT token-boundary
  (snapshot 9 → 24 fields; cublas/SDPA dispatch wires)

Step 5 N=128 contention soak may proceed.

## 7. Residue / follow-ups

1. **sc5_arena_ioctl.py constant update** — DONE in this commit (no separate
   commit needed). Memory `g1-g2-cap-bump` follow-up #1 → CLOSED.
2. **SC1 has no .py** — clarified inline; this is by-design (SC1 was the
   design-memo predecessor of the runnable suite per `TRACK_2_CLOSEOUT.md` §7).
3. **SC6 sc6_run.py CLI shape** — requires `<ModelName> <shared|independent>`
   args; running it bare exits with `IndexError`. Documented in this run.
   No fix required; the harness is invoked with explicit args as part of
   the standard suite invocation pattern.
4. **vLLM E.5 Mistral graph-capture segfault** (Step 4 residue) — unrelated
   to this suite (which uses raw `transformers` forward, not vLLM). Track 2
   substrate composes correctly; vLLM smoke is a separate environmental
   issue.

## 8. Logs

All raw logs under `/tmp/track3_e2e/`:
```
sc2.log              sc4_unit.log          sc5_unit.log
sc3.log              sc4_run.log           sc5_run.log
                     sc6_tiny_shared.log   sc6_mistral_shared.log
                     sc6_tiny_indep.log    sc6_mistral_indep.log
                                            sc6_aggregate.log
```
