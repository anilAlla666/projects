# D.8 FAIRNESS + SHIELD — CLOSE REPORT: **HARD STOP (throttle-band is the wrong lever), anchor NOT rotated**

**Date:** 2026-05-29. bf16 (real serving dtype) throughout. **Substrate BUILT + correctness-safe
+ default-OFF. KL=0 timing-only PASS (banked). The two enforcement gates do NOT pass.** A clean
diagnostic (§3.5, after Anil challenged the result) establishes the sharp mechanistic finding:
under a genuinely saturating neighbor a GPU-latency-bound victim's p99 inflates **40×**; the
timing-only throttle is a **contention-frequency reducer** (median fully recovers, 1.9 ms→0.033 ms)
but **NOT a p99 isolator** — the tail stays **~24× baseline at ANY throttle magnitude** (p99 is
collision-bound: each collision is one non-preemptible ~1 ms kernel wait a CPU-submission sleep
cannot evict), and the partial relief it does give comes only by near-**suppressing** the neighbor
(at 30 ms throttle the aggressor ran 64 of 5376 matmuls ≈ 1%) — which you cannot do to a legitimate
throughput tenant. **Conclusion (unchanged, now better-supported): close throttle-band; the
pre-registered CP54 SM-partition escalation (D.8 memo §3.5) is the correct lever for p99 isolation
— spatial partition bounds the tail WITHOUT suppressing the neighbor.** Anchor NOT rotated, soak NOT
run, nothing faked or tuned to pass.

> **Two corrections during this close (both from Anil's challenge "both gates failed — can't be
> right; diagnose deep"), recorded for the auditor:**
> 1. An interim draft claimed "no p99 damage (+2%)" — that rested on a malformed B=1 "noisy" tenant
>    that cannot saturate an H100. **Real contention is 40×** (§3.5).
> 2. An interim draft claimed the throttle is "powerless / −6% / cannot free the GPU." **Wrong:** it
>    gives substantial MEDIAN relief (p50 1.9 ms→0.033 ms) — it just cannot bound the TAIL (p99 stays
>    ~24× baseline) and only relieves by near-suppressing the neighbor. Anil's challenge exposed three
>    harness bugs (overhead-bound HF-`generate` victim ~26 ms/token; the n_active≤1 trap that left the
>    throttle inert; a too-small 500 µs sleep). The corrected evidence **sharpens** the finding; it
>    does not reverse the disposition.

## §0 Provenance / anchors (UNCHANGED — no rotation)

| component | anchor | state |
|---|---|---|
| cipher_rt_phase4 | `ed130e7`, tag `d7-rh1-close`, anchor `.so 01d4effb` | UNCHANGED (D.7) |
| cipher_kmod | `02fc2d1` (0.6.6) | UNCHANGED as the canonical anchor |
| cipher_kv_bridge.so | `5a3db034` | UNCHANGED |

Built-but-unrotated D.8 artifacts (preserved, NOT promoted to anchor): kmod **0.7.0** (NR 32 +
`cipher_fairness_ledger.c`; srcversion `E3887DF137D80A9AC61354F`, md5 `aa037307`, fallback
`cipher_ko_fallback/cipher_kmod-0.7.0-d8.ko`); libcipher_rt D.8 staging `02ceb6a0`. **The 0.7.0
kmod is currently LOADED** (Anil loaded it for the gates) — it is additive + inert default-OFF,
so leaving it loaded is harmless; Anil may keep it or revert to 0.6.6 (see §5).

## §1 What was built (additive, default-OFF, correctness-safe)

- **kmod 0.7.0** — `cipher_fairness_ledger.c`: a cross-tenant work-ledger (vmalloc_user, RW-mmap
  window pgoff `0x200000`) that **replaces the per-container `/dev/shm` table** (Docker-isolated;
  would not span CDI containers) with a `/dev/cipher`-backed region — the same cross-tenant channel
  the W9 resolver proved. New ioctl **NR 32** `CIPHER_FAIRNESS_REGISTER` (slot-register + SHIELD
  band). **NR corrected 30→32** (30/31 were taken by the W.6 sub-C cohort registry). Additive;
  reserved NRs still -ENOSYS; existing NRs unchanged.
- **libcipher_rt** `cipher_rt_fairness.c` — opens `/dev/cipher`, registers, RW-maps the ledger,
  self-accounts GEMMs, band-modulated `should_yield` (ports the prior `/dev/shm` rate-vs-average
  logic), **timing-only `nanosleep`-before-submit** throttle in the `cublasGemmEx` hook,
  default-OFF (env-gated `CIPHER_FAIRNESS` / `CIPHER_SHIELD`).

## §2 KL=0 correctness gate (Mem #11) — **PASS** (banked, stands regardless of §3)

`d8_kl.py`, TinyLlama bf16, single process, throttle FORCED to fire: **armed=1, throttle fired
7,719×, `max_logit_diff = 0.000e+00`, KL_max = 0.000e+00** → bit-identical with the throttle
firing ⇒ the CPU-sleep-before-submit changes only TIMING, never the computation. Timing-only
enforcement is proven correctness-preserving. (The harness FAILs unless `self_yields>0` — the
advisor caught + we fixed an earlier vacuous single-process version where the throttle never fired.)

## §3 Gate A (burst-fairness) + Gate B (SHIELD p99) — three runs, both gates FAIL

bf16, static residence, separate processes sharing `/dev/cipher`. **Strongest datum first:** a B=1
TinyLlama victim sustains **~36 tok/s identically across solo / B=1-noisy / matmul-flood** —
throughput INVARIANT to GPU load ⇒ B=1 decode is overhead/launch-bound, not throughput-bound, so
there is no throughput resource for a burst-fairness throttle to arbitrate.

**Run 1 — B=1 multi-tenant (1 weak "noisy" + 3 victims, `d8_run_gates.py`):** victim share OFF=0.782
ON=0.778 (already ≥ 0.75 fair); noisy self_yields=3,296 fired but noisy tok/s unchanged. **Malformed:**
the B=1 "noisy" ran SLOWER (30.7) than victims (~36.7) — a B=1 tenant cannot saturate an H100, so
this run created no contention. *Discredited for both gates; re-run with a real saturator.*

**Run 2 — Gate A under a real saturating aggressor (`d8_gate_saturate.py`):** aggressor = continuous
**8192³ bf16 matmul flood**; victims = 3× B=1 decode. victim aggregate tok/s **SOLO=108.0,
OFF=109.2, ON=108.8** → `starvation = NO` (+1.1%). aggressor self_yields ON=8,608. **Gate A FAIL —
no throughput starvation to correct** (throughput is invariant to the FLOP-flood, as above).

**Run 3 — Gate B under the SAME saturating aggressor (`d8_gate_b_saturate.py`):** highband =
latency-sensitive B=1 victim (band 1); aggressor = 8192³ matmul flood (band 0).
- highband **p99: SOLO=209.7 ms → OFF(aggressor)=2377.6 ms (+1034%, 11×) → ON(SHIELD)=2228.3 ms**.
- `p99 damage under OFF = YES` (11×). **SHIELD-ON bounds it by only −6%** (still 10.6× solo) despite
  the aggressor self_yields=11,636. **Gate B FAIL — throttle-band does NOT bound noisy-neighbor p99.**

**Runs 1–3 are CONFOUNDED** (Anil's challenge "both gates failed — can't be right; diagnose deep"
surfaced this): the HF-`generate()` victim is **overhead-bound** (~26 ms/token Python overhead vs a
~3–5 ms/token GPU floor), so its latency/throughput was structurally insensitive to GPU-contention
relief — Runs 1–3 partly measured the harness, not the substrate. Run 3's 11×/−6% is therefore not
a clean SHIELD measurement. §3.5 replaces it with a clean probe.

## §3.5 Clean diagnostic (the decisive SHIELD measurement) — `d8_diag2.py`

Wall-clock submit→sync latency of a **GPU-bound** victim (small bf16 GEMM, 2000 samples, no
HF/Python-per-token overhead) against a genuinely **saturating** aggressor (deep async matmul queue,
~100% duty). DVFS controlled by comparing OFF-vs-ON (both aggressor-present). The aggressor is forced
to throttle (FORCE_YIELD) so the question is isolated: *if the aggressor sleeps, does the victim's
tail recover?* Throttle swept 2/8/30 ms:

| condition | victim p50 | victim p99 | aggressor matmuls |
|---|---|---|---|
| baseline (no aggressor) | 0.033 ms | 0.048 ms | — |
| + aggressor, OFF | 1.907 ms | 1.943 ms (**40× base**) | 5376 |
| + aggressor, ON @2 ms | **0.034 ms** | 1.241 ms | 1280 (24%) |
| + aggressor, ON @8 ms | 0.033 ms | 1.344 ms | 320 (6%) |
| + aggressor, ON @30 ms | 0.033 ms | 1.160 ms (**24× base**) | 64 (**1%**) |

**What this proves:**
- Real GPU contention is **40×** (every victim op waits ~1 time-slice; p50≈p99≈1.9 ms under OFF —
  robust across 2000 samples, not an outlier).
- The throttle is a **contention-FREQUENCY reducer**: median fully recovers (1.9 ms→0.033 ms) because
  the aggressor is mostly asleep. It is **NOT a p99 isolator**: the tail stays **~24× baseline at
  every throttle size**, and p99 is **flat (1.24/1.34/1.16 ms) while the aggressor drops 24%→1%** ⇒
  p99 is **collision-bound, not rate-bound** — each collision is one **non-preemptible ~1 ms kernel
  wait** that a CPU-submission sleep cannot evict.
- The only way the throttle helps is by driving the aggressor toward **inactivity** (1% of its work
  at 30 ms) — a kill-switch, not fair-share arbitration; you cannot do that to a legitimate
  throughput tenant.

## §4 Finding (honest, named — corrected per §3.5)

1. **Noisy-neighbor latency contention is REAL** — a saturating tenant inflates a co-located
   GPU-latency-bound victim's p99 **40×** on one H100 (real Goal-1 gap for heterogeneous mixes).
2. **Throttle-band gives partial MEDIAN relief but cannot ISOLATE the tail.** p50 fully recovers;
   p99 stays ~24× baseline at any throttle magnitude (collision-bound), and the relief only comes by
   near-suppressing the neighbor. A timing-only CPU-submission sleep cannot preempt an in-flight
   kernel — so it cannot bound p99, which is the whole SHIELD (R-I1) requirement.
3. **CP54 SM-partition is the correct lever for R-I1** — spatial partition gives the latency tenant
   dedicated SMs the aggressor cannot touch, bounding the tail WITHOUT suppressing the neighbor.
   Pre-registered in the D.8 memo §3.5; these results validate it. Consistent with
   `cipher-t424e-isolation-confirmed` (SM-partition = tail-tightness), `cipher-t424c-partition-not-enforced`.
4. **R-D5 burst-fairness is moot for Goal-1.** The throttle *does* cap the aggressor's throughput
   share, so for a GPU-throughput-bound victim it could shift share — but realistic Goal-1 decode
   tenants are overhead-bound (HBM 4–18%, `cipher-phase-a-multitenant`), so freeing the GPU does not
   raise their throughput. No throughput starvation to arbitrate at decode scale.

So the throttle-band mechanism is built, additive, default-OFF, and correctness-safe (KL=0), and it
*does* reduce contention frequency — but it does not close either gate: R-D5 is moot for Goal-1
decode, and R-I1 needs tail isolation that a submission-throttle structurally cannot provide.

## §5 Verdict + disposition (STOP for Anil)

**HARD STOP — D.8 throttle-band does NOT close. Anchor NOT rotated; soak NOT run.**
- KL=0 timing-only correctness: **PASS (banked).**
- Gate A burst-fairness (R-D5): **FAIL / moot** — no throughput starvation for overhead-bound Goal-1
  decode (throughput invariant to load); the throttle caps share but freeing the GPU doesn't raise an
  overhead-bound victim's throughput.
- Gate B SHIELD p99 (R-I1): **FAIL** — real contention is **40×** (clean probe §3.5); throttle gives
  partial median relief (p50 1.9 ms→0.033 ms) but **cannot isolate the tail** (p99 stays ~24× baseline
  at any throttle, collision-bound), and only by near-suppressing the neighbor.
- Mechanistic root cause: a CPU-submission sleep cannot preempt an in-flight kernel ⇒ cannot bound p99.

**Is D.8 enforcement on Goal-1's critical path?** Partly. *Burst-fairness (R-D5)* is **not** —
decode throughput self-arbitrates. *SHIELD p99 (R-I1)* **is** a real gap for heterogeneous mixes —
but **throttle-band is not its fix**; CP54 SM-partition is. (Note also: if 100 agents' bursts truly
align, the binding constraint is **KV/memory capacity** — Track 2 / KV-dedup territory — not
compute-fairness, so D.8's throttle isn't that lever either.)

**Anil's adjudication — options (no rotation without your call):**
- **(a) RECOMMENDED — close D.8 throttle-band unrotated** with the §4 finding banked: burst-fairness
  has no purchase at decode scale; SHIELD p99 is a real gap that throttle-band cannot close. Keep the
  kmod ledger + rt throttle as inert additive substrate (default-OFF; harmless), and **re-scope R-I1
  to the CP54 SM-partition lever** (already built for CP 5.4 / Track 3) as the D.9-or-V.1 work. The
  real isolation validation is the **V.1 100-agent soak**, not D.8.
- **(b)** Re-open R-I1 now as an SM-partition substep (bind the latency-sensitive tenant to a reserved
  green-ctx SM group via the existing CP54 ALLOCATE path; re-run Gate B). Heavier; defers D.9/V.1.
- **(c)** Promote the ledger/throttle as inert additive substrate regardless — still requires the
  §5-memo regression (additivity-KL OFF byte-identical + 30-min soak) before any rotation; not run.

**kmod state for D.9/V.1 (pin explicitly, per the D.7 stated-vs-disk lesson):** the **canonical anchor
is kmod 0.6.6 `02fc2d1`** (unrotated). The **loaded** module is **0.7.0** (additive, inert default-OFF)
— Anil may keep it loaded or `rmmod && insmod` 0.6.6; either is correctness-equivalent with D.8
disarmed. State which one D.9/V.1 run against.

Nothing faked; bf16/INT4 throughout. KL=0 banked. Artifacts: **`d8_diag2.py` + `d8_diag2b.log` +
`d8_diag2_result.json` (the decisive clean §3.5 measurement)**, `d8_gates_result.json`,
`d8_gate_saturate_result.json`, `d8_gate_b_saturate_result.json` (Runs 1–3, confounded),
`d8_kl.log`, `d8_sat.log`, `d8_sat_b.log`, `d8_gates.log`.
