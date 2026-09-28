# Fault-Injection Feasibility — STEP A (read-only, anchor frozen)

**2026-06-05. READ-ONLY diagnostic. No production `.so` change.** Anchor
`/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f`
verified unchanged at entry; re-checked at exit (see end). Scratch harnesses only, in
`cipher-fusion-evidence/fault_injection_step_a/`. GPU clock locked to 1980 MHz app clock
for reproducible ratios. Every number in this report is from this run; none from memory.

## Objective

Two feasibility facts, both measured, before any detector is built:

- **A1.** Does a bit-flip at the GEMM-output boundary actually harm a real Mistral-7B
  forward, and how does harm split by bit significance? (Are we catching something that matters?)
- **A2.** What does a Huang–Abraham ABFT checksum-verify *cost* on the exact GEMM shapes
  CIPHER intercepts, and is there an operating point that detects the harmful class under
  ~3% throughput overhead?

## Scope / substrate-line (binding)

- **Detection mechanism under study** = checksum verification on the intercepted GEMM
  (row/column checksum, Huang–Abraham invariant), aligned to the dispatch boundary. NOT a
  PyTorch hook, NOT a monkeypatch, NOT app-layer. (Hooks are used *in this diagnostic only*
  as a measurement scaffold to emulate where the substrate detector would sit.)
- **Fault model** = transient single-bit flip in one element of an intercepted GEMM output
  (the SDC class that manifests at the dispatch boundary).
- **Explicitly out of scope:** corruption that never reaches an intercepted GEMM. In
  particular, attention-internal GEMMs (QKᵀ, PV·V) are **not** intercepted linears, so a
  linear-GEMM checksum structurally cannot cover SDC born inside attention. A1 therefore
  characterizes harm at the **intercepted-linear** outputs (the coverable set); attention-
  internal SDC is a named structural blind spot, not a measured coverage bucket.
- **Coverage-join arithmetic (advisor-shaped):** ABFT row-sum checks a single GEMM's output.
  Flipping one output element `C[m,n]` by `δ` shifts the row-sum `Σ_n C[m,n]` by exactly `δ`,
  so the checksum residual ≈ `|δ|`. We therefore record `|δ|` for every A1 flip; coverage of
  the harmful class at threshold `T` is just `#{harmful flips with |δ| > T} / #harmful` — pure
  arithmetic, no second model run.

---

## PRE-REGISTERED VERDICT (written before A2 was run)

**PROCEED to Step B** (build the detector as a new tagged actuator) **IF** a checksum scheme
catches the harmful class at materially-high coverage under ~3% throughput overhead on the
production shapes.

**DO NOT PROCEED / re-scope IF** the only points under 3% miss most harmful flips, or if
harmful coverage requires >3%. That is a real, reportable negative: GEMM-boundary checksum
detection is not viable at the overhead budget, and reliability needs a different mechanism
(selective re-execution, statistical parameter-update monitoring) before it can be a pillar.

**A1-grounded hypothesis (recorded before running A2):**

1. *Coverage will be easy; the wall, if any, is overhead.* A1 shows the harmful class is
   exactly the large-magnitude flips (bit 14, the top exponent bit), with **minimum harmful
   `|δ| = 2.76`**. If the clean per-row-sum noise floor `T` is below the per-shape minimum
   harmful `|δ|`, full checksum catches ~100% of the harmful class. The risk shapes are the
   N=4096 linears (o_proj/down_proj), where harmful `|δ|` reaches down to 2.76 / 4.64 — `T`
   must be below ~2.76 there. Large-N shapes (lm_head N=32000) have a higher `T` but their
   harmful flips are also much larger (min 1173), so they may self-align.
2. *The verdict hinges on capture, not FLOPs.* In **eager** decode, adding ~225 verify ops
   per step is launch-bound and is expected to blow the 3% budget. **Captured** into the
   decode CUDA graph, the verify is pure tensor ops (legal) and its compute floor — an
   `[1,K]·[K]` matvec plus an `[1,N]` reduction per linear, negligible HBM vs the weight read
   — is expected to fit well under 3%. So the likely honest outcome is: **feasible <3% only if
   the verify is captured into the decode graph; not feasible eager.** The `.item()` residual
   flag is a host sync (capture-illegal) and must be deferred outside the graph.

---

## A1 — Corruption harm characterization

Harness `a1_harm.py`. Mistral-7B-v0.1 fp16, eager attention (deterministic hooks), fixed
20-token prompt, clean top-1 = 6779, clean logit scale 12.44. Single fp16 bit-flip injected
into the **output** of an intercepted op at the **last sequence position** (the element on the
direct path to the next-token logit). Sites: q/k/v/o/gate/up/down_proj on layers {0,8,16,24,31}
+ lm_head. Bits 0–15 × 8 random channels each = **4608 injections, 120 s**.

**Injector validation gates (all passed):** clean forward bit-identical across two runs
(`determinism_bitident=true`); forced mantissa-LSB flip wrote the element (`δ=+0.0078`); forced
top-exponent flip on the top logit channel blew it up (`δ=−12.44`, top-1 changed) — the
injector demonstrably both writes and can cause harm.

### Harm by bit significance (fp16: bit15=sign, 14–10=exponent, 9–0=mantissa)

| bit | band | n | NaN/Inf | top-1 flip | **HARMFUL %** |
|----:|------|--:|--------:|-----------:|--------------:|
| 0–12 | mantissa + low exp | 288 ea | 0 | 0 | **0.0** |
| 13 | exponent | 288 | 0 | 3 | **1.0** |
| **14** | **exponent (MSB)** | 288 | 41 | 156 | **68.4** |
| 15 | sign | 288 | 0 | 0 | **0.0** |

By band: **sign 0/288 (0%), exponent 200/1440 (13.9%), mantissa 0/2880 (0%).** All harm lives
in bits 13–14; **bit 14 alone is 197 of 200 harmful flips.** Flipping the top exponent bit on a
~unit-magnitude fp16 pushes the biased exponent to all-ones (→ Inf/NaN) or adds 2¹⁶ in
magnitude — the only flips that reliably overwhelm the network. A single-element **sign flip is
absorbed every time**; so is every mantissa flip and even most of bit-13 (×256 magnitude).

### Harm by intercepted op

| op | N (out) | harmful / total | min harmful `|δ|` |
|----|--------:|----------------:|------------------:|
| o_proj | 4096 | 39 / 640 | **2.76** |
| down_proj | 4096 | 40 / 640 | 4.64 |
| v_proj | 1024 | 33 / 640 | 101.7 |
| up_proj | 14336 | 41 / 640 | 439.0 |
| gate_proj | 14336 | 12 / 640 | 1582 |
| lm_head | 32000 | 3 / 128 | 1173 |
| q_proj | 4096 | 19 / 640 | 2296 |
| k_proj | 1024 | 13 / 640 | 35807 |

### The coverage-relevant distribution (this is what A2's threshold must clear)

- **HARMFUL: 200 / 4608 (4.3%).** Of these, 26 produced Inf/NaN in the element itself
  (`|δ|=∞`, trivially detectable). The 174 finite-harmful flips: **min `|δ|=2.76`**,
  p10=114, median=3410, p90=43647, max=61151.
- **BENIGN: 4408 / 4608 (95.7%).** `|δ|`: median 0.009, p90 0.87 — **but** p99=18288 and
  max=61151. So large-`|δ|` flips are *not all* harmful: the network absorbs many big single-
  element perturbations. This does **not** hurt ABFT — a checksum flags *corruption*, not
  *harm*; flagging a benign-but-real large flip is correct behavior that costs one extra
  re-execution, not a correctness error. It does mean `|δ|` is the right axis for *coverage*
  but not a *harm predictor*.

**A1 takeaway:** the class a detector MUST catch is small (4.3%) and sits at large magnitude
(min harmful `|δ|=2.76`, almost all ≫100). This is the most favorable possible setup for a
magnitude-sensitive checksum — *if* the clean noise floor `T` per shape sits below the per-shape
minimum harmful `|δ|`. A2 measures `T` and the overhead.

---

## A2 — Detection-cost feasibility

Harness `a2_real.py` (real Mistral-7B, 225 intercepted linears, prefill M=922) + `a2_micro.py`
(synthetic per-shape GEMM-vs-verify, prefill M=2048 / decode M=1) + `coverage_join.py` (joins the
A1 `|δ|` distribution against per-shape clean thresholds — pure arithmetic, no second model run).
Clock locked 1980 MHz. Row checksum = Huang–Abraham invariant: row-sum `Σ_n C[m,n]` compared to a
reference `A[m,:]·(Σ_n W[:,n])`; the per-shape threshold `T` is set to the **max** clean row-sum
residual over all 922 rows (⇒ **0 clean false positives by construction**).

### Coverage — settled YES (the easy half, as hypothesized)

Per-shape `T_fp32_max` and coverage of the harmful class:

| shape | N | T (fp32) | #harm | #caught | cov % | min harmful `|δ|` | margin `min|δ|/T` |
|-------|--:|---------:|------:|--------:|------:|------------------:|-------------------:|
| o_proj | 4096 | 0.0436 | 39 | 39 | 100 | 2.76 | **63.3×** |
| down_proj | 4096 | 0.00181 | 40 | 40 | 100 | 4.64 | 2563× |
| v_proj | 1024 | 0.0492 | 33 | 33 | 100 | 102 | 2067× |
| up_proj | 14336 | 0.0698 | 41 | 41 | 100 | 439 | 6287× |
| gate_proj | 14336 | 0.0698 | 12 | 12 | 100 | 1582 | 22657× |
| lm_head | 32000 | 1.211 | 3 | 3 | 100 | 1173 | 969× |
| q_proj | 4096 | 0.0436 | 19 | 19 | 100 | 2296 | 52667× |
| k_proj | 1024 | 0.0492 | 13 | 13 | 100 | ∞ (NaN/Inf) | ∞ |
| **TOTAL** | | | **200** | **200** | **100** | | tightest **63.3× (o_proj)** |

**100% of the harmful class is caught, at intercepted linears only**, with a tightest margin of
**63.3×** between the clean noise floor and the smallest harmful flip. The A1 hypothesis is
confirmed: the harmful class is exactly the large-magnitude flips, so any `T` placed at the clean
ceiling clears it by 1.5–4.7 orders of magnitude. `T` is in-sample-calibrated (922 rows of this
prompt), but the 63× margin makes that immaterial. *Attention-internal SDC (QKᵀ, PV·V) remains a
named structural blind spot — not in this 100%.*

**Hard constraint discovered — fp32 accumulation is mandatory.** The same join with fp16
thresholds collapses: `lm_head` `T_fp16 = NaN` (the fp16 row-sum of a 32000-wide row overflows),
`gate/up` `T_fp16 = 4.0` vs `0.07` fp32. The 100%-coverage result **silently depends on the
checksum row-sum and reference matvec accumulating in fp32.** Any Step-B epilogue must accumulate
in fp32 or coverage is not what this table says.

The benign side: 1673 / 4408 benign flips (37.95%) also exceed their shape `T`. This is **not** a
clean false-alarm rate — those are real large-magnitude single-element perturbations the network
happens to absorb; a checksum flags *corruption*, not *harm*, so flagging them is correct and
costs one re-execution, not a correctness error. Clean-noise false positives are 0 by construction.

### Overhead — settled NO for verify-as-a-separate-op (the wall, as hypothesized)

End-to-end on the real model, per-step median (`verify_compute` = the row-sum ops only;
`verify_full` = + the `.item()` residual flag, a capture-illegal host sync):

| regime | OFF | +verify_compute | +verify_full |
|--------|----:|----------------:|-------------:|
| prefill (M=922) | 33.32 ms | 49.87 ms (**+49.7%**) | 65.09 ms (+95.3%) |
| decode (M=1) | 20.18 ms | 35.73 ms (**+77.0%**) | 45.89 ms (+127.4%) |

Eager verify-as-op blows the 3% budget by 16–42×. The per-shape micro confirms the cause: as a
separate op the row-sum costs **60–292% of the GEMM it checks** (decode q/o +292%, lm_head +63%;
prefill gate/up +61%, q/o +100%), and the verify op cost is roughly **constant** (~0.063 ms/op
decode) regardless of shape — it is kernel-floor-bound, not FLOP-bound. Column and block checksums
are strictly worse (col_compute up to +540%).

**Capture helps but is not sufficient.** Folding the 225 row-sum ops into one CUDA graph: replay
**2.146 ms** vs eager loop 8.933 ms = **4.16× speedup** — yet that is still a **≥10.6% lower
bound** on decode (2.146 / 20.18 ms vs *eager* decode; capturing decode shrinks the denominator,
so the true fraction is higher). Per-op floor = **9.54 µs/op** — that is the minimum H100 kernel
*duration* for 225 tiny ops. A graph removes launch *gaps*, not per-kernel duration, so the
kernel-count wall survives capture. **Hypothesis 2 split:** its predicted *floor* (negligible
compute/HBM) is vindicated; its predicted *mechanism* (capture ⇒ <3%) is **refuted**.

### The only sub-3% path — fused epilogue (analytical, NOT measured)

The FLOP floor of folding the checksum into the GEMM *epilogue* (row-sum is a free byproduct of
the GEMM already running; the reference matvec `A·(Σ W)` is `M·K` MACs = **1/N of the GEMM**):

| shape | reference-matvec FLOPs vs GEMM |
|-------|-------------------------------:|
| q/o | 0.0244% |
| k/v | 0.0977% |
| gate/up | 0.0070% |
| down | 0.0244% |
| lm_head | 0.0031% |

0.003–0.098% — three to four orders under the 3% budget. **This is the only configuration that
fits.** It is analytical only: realizing it requires a *fused* GEMM-epilogue checksum kernel
(CUTLASS-class engineering on the intercepted path, fp32-accumulated, consistent with the
cu13 cuBLASLt-epilogue constraints in memory), not the separate verify op measured here, and
not a flag. Its real-world feasibility is unproven.

---

## Verdict

**DO-NOT-PROCEED with the mechanism as specified-and-measured; RE-SCOPE to a fused epilogue.**

Against the pre-registered criterion — *"PROCEED IF a checksum scheme catches the harmful class at
high coverage under ~3% overhead on the production shapes"* — **no measured configuration clears
3%**: eager +50–127%, captured-as-separate-op ≥10.6%. The pre-registered DO-NOT-PROCEED branch is
the honest verdict for verify-as-an-op. Coverage is **settled YES** (200/200, 63× margin, at
intercepted linears, fp32-accumulated); overhead is **settled NO for separate ops** and **open for
a fused epilogue**.

What this resolves, precisely:
- **Coverage is not the risk.** The harmful SDC class is small (4.3%) and large-magnitude; a
  magnitude checksum clears it trivially. Don't spend Step B re-proving coverage.
- **The wall is kernel count, not FLOPs.** 225 extra tiny ops × ~9.5 µs minimum duration survives
  graph capture. Capture is necessary-but-not-sufficient; **only fusion eliminates the kernels and
  realizes the 0.003–0.098% FLOP floor.**
- **fp32 accumulation is a hard requirement**, not an option — fp16 thresholds collapse (lm_head
  overflows to NaN).

**Recommended Step B** (not green-lit by this report — it is the open question this report hands
off): build the detector as a **fused GEMM-epilogue row-checksum actuator** on the intercepted
linear path — fp32-accumulated row-sum + reference matvec emitted as a CUTLASS epilogue, residual
flag deferred outside any captured graph. Step B is a **kernel-engineering effort**, and its
make-or-break is whether the fused epilogue actually lands under 3% in practice (the analytical
floor says it can; nothing here proves it does). If the fused epilogue cannot be built or does not
hit budget, GEMM-boundary checksum detection is not viable and reliability needs a different
mechanism (selective re-execution, statistical parameter monitoring) before it can be a pillar.

---

**Anchor re-check at exit:** `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` md5
`2edba0d2136f8ede4713d90a8f7cd55f` — **unchanged** (verified at exit; READ-ONLY diagnostic, no
production `.so` change). All numbers above are from this run's `a1_results.json`, `a2_real.json`,
`a2_micro.json` via `coverage_join.py`; none from memory.
