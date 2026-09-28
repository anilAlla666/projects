# W.4b.6 — clear the engagement gate (weight-share → N=8 Mistral) — FINDINGS

**Date:** 2026-05-28. **Anchors UNCHANGED** (userspace harness + an existing
env-var flag only; NO substrate/kmod/bridge edit): cipher_rt_phase4 `8b5e928`
(libcipher_rt.so md5 `9fe23143`), cipher_kmod `02fc2d1` (0.6.6), cipher_kv_bridge
`5a3db034` (Track-2 SC3 arena). Gate model: Mistral-7B-v0.1, N=8, GEN=64, PLEN=12.

## Result — THE GATE CLEARS

| condition (N=8 Mistral, weight-shared) | tok/s | W | tok/W |
|---|---|---|---|
| cross-process full-handoff | 157.7 | 151.5 | **1.041** |
| in-process B=8 ceiling | 156.7 | 148.0 | 1.059 |
| naive 8-concurrent (B=1 ×8, weight-shared) | 93.0 | 289.5 | 0.321 |

- **substrate-attributable = cross / naive = 3.239×** (> 1.0× break-even; approaching
  CP 5.6's 3.69× located at exactly this N=8 Mistral regime).
- in-proc lever = inproc / naive = **3.296×** (the density lever's true magnitude
  at the GPU-bound regime — vs only 1.48× at the testable N=4 in W.4b.5).
- cross / inproc = **0.983×** — the full-handoff realizes 98.3 % of the
  single-process ceiling: the per-token rendezvous is *banked* (the executor runs
  the whole B=N decode; tenants submit prompt + gen_len and await tokens).

## Step 1 — Track-2 weight-sharing (the regime unlock)

Re-ported the verified SC3 cuIPC weight-arena pattern (`formb6_weights.py`): one
owner packs the model into a cipher_kv_bridge VMM arena (`cuMemExport` POSIX-FD);
N tenants meta-load (`init_empty_weights`, NO safetensors read) and rebind every
parameter onto a READ-only arena view. **One physical 14 GB Mistral copy shared by
executor + 8 tenants.**

- **Lossless (Step 1.3):** weight-shared forward is **bit-identical** to
  standalone-weights — TinyLlama KL = 0.000e+00, Mistral-7B KL = 0.000e+00
  (`formb6_ws_smoke.py`). The arena is read-only weight bytes ⇒ exact.
- **N=8 Mistral fits (Step 1.2):** peak HBM **13.49 GiB** (executor + 8 tenants
  co-resident). Without weight-share this is 9 × 14 = 126 GiB → OOM on 80 GB — the
  exact "regime-unreachable" limit W.4b.5 surfaced. **Now reachable.**

## Step 2/3 — rendezvous banked via full-handoff; the gate

The executor runs the entire B=N decode internally (`formb6_executor.py`); tenants
submit prompt+gen_len once and receive all tokens (the maximal "multi-token
window" — banks the per-token barrier W.4b.5 measured at ~79 ms/step). Tenants stay
real independent injected GPU processes (real W.6 fps, audited W.4a gate, W.4b.3
correctness backstop). This realizes the *homogeneous same-model density* lever
cross-process (the CP 5.6 / Goal-1-v1 lever) — NOT per-token heterogeneous-compute
coalescing (that is branch B / R-4 below).

## The cross-process tax was a self-inflicted CUPTI artifact (root-caused)

First measurement gave cross = 0.504 tok/W → substrate-attrib **1.563×** (cross /
in-proc = 0.475×). A 5-control elimination proved the 2.6× executor slowdown was
NOT inherent to co-residence:

| control (B=8 Mistral decode) | tok/s |
|---|---|
| in-proc alone | 156.4 |
| in-proc + 8 idle **bare** CUDA contexts | 154.7 |
| in-proc + 8 **warmed vanilla** tenants (`formb6_warmhold.py`) | 155.7 |
| in-proc + 8 **warmed injected** idle tenants | 157.8 |
| arena-backed (shared) weights | 408–429 |
| with the FIRSTCOALESCE backstop run first | 421 |
| **executor in the real cross run** | **59.7** |

pmon clipped to the decode window: the 7 batched tenants are **sm = 0 (idle)**;
only the executor runs, at **13 % sm mean / 26 % max** — *starved*, not contended.
Root cause: the executor ctypes-loads `libcipher_rt.so` to call the audited gate
guard `cipher_rt_pool_partition`; the library's `__attribute__((constructor))`
(`cipher_inject.c:159`) **auto-subscribes CUPTI kernel-launch callbacks**
(`cipher_cupti.c`) — re-arming the W.4b.1 shim on the "vanilla" executor and taxing
every decode kernel. **Fix:** `CIPHER_RT_DISABLE_AUTO_INIT=1` (existing escape
hatch, `cipher_inject.c:173`) in the executor env — the gate guard is a pure CPU
function and needs no auto-init. Verified: ctypes-load with the flag → no CUPTI
subscribe, decode 408 tok/s (full speed). Re-run with the flag → cross/inproc
0.983, substrate-attrib 3.239×. This restores the *intended* vanilla executor
(W.4b design); the substrate/kmod are untouched.

## Correctness (Memory #11) — every run

- **Positive N=8:** 7/8 tenants 64/64 exact-greedy vs solo. Tenant row-3 = 20/64
  (44 free-running divergences). **Adjudicated benign:** the FIRST (causal)
  divergence is at step 20 with solo top1−top2 gap **0.00781** (a near-tie — the
  identical signature W.4b.5 accepted at N=4); greedy then cascades into different
  (equally valid) sequences. **Decisive teacher-forced KL test** (`formb6_tf_kl.py`,
  identical context isolates substrate faithfulness from the coin-flip): worst
  per-step KL across all 8 rows = **7.04e-5**, row-3 argmax mismatch **1/64** under
  teacher forcing. The batched substrate is faithful (KL ~1e-5, same order as the
  lossless weight-share). NOT corruption, NOT a HARD STOP.
- **Negative control N=8 Mistral:** 7× Mistral + 1× Llama-3.2-1B → the distinct-fp
  Llama is **REJECTED** by the audited guard (`distinct_rejected=1`), runs solo;
  the 7 Mistral coalesce 64/64 exact. Zero cross-fp coalesce. (Adds the Mistral
  N=8 row to the C-table; the guard is fingerprint-based, model-agnostic.)
- FIRSTCOALESCE backstop: max_rel 0.0027 ≪ tol 0.15, fails=0 — and still correct
  with auto-init disabled (the guard / correctness-check are pure functions).

## Harness bugs fixed this substep (all userspace; substrate untouched)

1. **Use-after-free of the weight arena** — `import_rebind` returns the
   `WeightArena`; callers discarded it → its dtor unmapped the VMM
   (`cipher_kv_bridge.cpp:95`) while the model's `from_blob` param views (no-op
   deleter) still pointed at it → illegal access. Fix: hold the arena.
2. **Truncated-manifest partial recv** — `socket.recv_fds` does one `recvmsg`;
   Mistral's 291-tensor manifest (~37 KB) exceeded one delivery → JSON parse
   failure. Fix: length-prefixed manifest then 1-byte+fd (`formb_ipc.send/recv_arena_fd`).
3. **bind-after-load hang** — owner bound the socket after the 14 GB load, exceeding
   the importer's connect-retry window → zombie importer + `accept()` hang. Fix:
   bind+listen before the load.
4. **CUPTI auto-subscribe via the gate-guard ctypes load** — the 2.6× cross tax
   (above). Fix: `CIPHER_RT_DISABLE_AUTO_INIT=1` on the executor.

## Disposition (for Anil)

The engagement gate **clears at 3.239× substrate-attributable** for the
**homogeneous same-model density** model (branch A — agents submit a request and
await a batched completion; how real multiplexed serving works). Goal-1 density is
DELIVERED at N=8 Mistral, enabled by lossless weight-sharing, with the audited
admission gate + correctness guard intact and the transport banked.

The honest scope boundary (the A/B fork): full-handoff realizes the *batching*
lever cross-process; it does NOT exercise *per-token cross-process coalescing*
(branch B — heterogeneous-compute rebalance, the memo-literal peer-KV attention),
which W.4b.5 showed is rendezvous-bound and which remains **R-4 / W.4b.2b** future
scope. For the v1 density goal (same-model multiplexing), branch A is the faithful
and sufficient realization.
