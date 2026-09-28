# V0 G-O1 ENGINE INCREMENT 3b-fix: time-boxed root-cause attempt on the residency-swap corruption

**2026-06-03. Q1 (buffer-pinning) REFUTED. Mechanism NOT characterized (5th black-box attempt; honestly un-named).
Accepted fix = re-capture-per-serve, sustained KL=0 at REQ=200/REQ=120, no leak.** NO CIPHER `.so` change; fix in
`cipher_router.py` only; inc-1 `cipher_engine.py` + inc-2 `cipher_engine_batched.py` BYTE-IDENTICAL + gates PASS;
M=3 wall PASS; anchor 1f305ce6 UNCHANGED. `CIPHER_RT_DISABLE_AUTO_INIT=1`. Probe: `pager_g3bfix_probe.py`.

## The two bounded questions Anil set

**Q1 — buffer-pinning / persistent-graph: REFUTED.** Probe `pager_g3bfix_probe.py`: model A (Qwen2) kept
**resident, NEVER evicted**; models B,C churn (restore→serve→evict) around it; A served interleaved, NO re-capture.
A's graph corrupts (baseline 48/48 → 47/48 after B churn → FAULT after C). Since A's own weights/KV/`data_ptr` are
all stable (established inc-3b) yet A corrupts from OTHER models' churn, the corruption is **GLOBAL, not
buffer-local** → pinning A's cache/IO buffers cannot help. (Consistent with the earlier per-model cache-pool
experiment, where corruption persisted.) Persistent-graph-over-pinned-buffers REFUTED.

**Q2 — per-swap vs per-serve: conservatively PER-SERVE.** Implemented an epoch-gated cheaper fix (re-capture a
model's graph only when a residency-churn op occurred since its capture, via a global `vmm_epoch`). It **FAULTED** —
a graph reused between churn ops still corrupted. I did **not** separate "my epoch bookkeeping had a bug" from "reuse
is genuinely unsafe," and per the time-box I did not chase it. Conservatively adopted **re-capture per serve**
(every serve re-captures); measured ~48ms re-capture/serve.

## Mechanism — NOT characterized (do not name it; my own data refuted the candidate)

I was about to write "the pager's VMM ops invalidate all live captured graphs." **My own probe refutes that:** A's
baseline serve was **48/48 KL=0 AFTER B and C were page-OUT'd** (two post-capture VMM page-outs) — page-outs did NOT
invalidate A's graph. Corruption appeared only after a **page-IN** (`restore()`) in the churn loop. So at most there
is a narrower, **un-isolated restore/page-in correlation** — NOT a named cause, and I deliberately did NOT run the
page-in-vs-page-out or epoch-bug isolation (the fenced-off spiral). **Five black-box attempts have now each refuted
their own candidate:** cache-aliasing (→ pool-isolation didn't fix), dangling-pointer (→ `data_ptr` stable),
weights/KV (→ cksum + max|Δ|=0 + eager-fresh correct), "any VMM op" (→ baseline-after-page-outs KL=0), epoch-reuse
(→ faulted, cause not isolated). **The trigger resists black-box characterization. It is documented debt, not a
named mechanism.** (Establishing it likely needs white-box tooling — CUDA graph internals / `compute-sanitizer` /
the driver's graph-vs-VMM interaction — out of this increment's time-box.)

## Accepted fix + gate

Re-capture-per-serve (fresh cache + graph every serve). Gate (`cipher_router_gate.py`):
- **REQ=200:** exact 200/200, 84 swaps all physically real, across-swap KL=0 84/84, misroute DETECTED, GPU→0.
- **REQ=120:** exact 120/120, 46 swaps real, across-swap KL=0 46/46, misroute DETECTED, GPU→0.
- **Cost:** re-capture ~46-48ms/serve; decode ~462-477ms/serve (host-overhead-dominated per inc-2) → ~509ms/serve.
  Capture is ~9% of per-serve; decode dominates.
- **Density:** 4 distinct models, one process, K=2 resident, swap-on-miss correctness-sustained → ~5 fp16 7-8B
  resident in 80GB + unbounded swap-beyond at ~46ms re-capture + page-in per cold serve.

## NON-REGRESSION

inc-1 + inc-2 modules BYTE-IDENTICAL + gates PASS; M=3 singleton-wall still PASS; OFF byte-identical; anchor
1f305ce6 UNCHANGED. Subprocesses os._exit-reaped, GPU→0.

## STOP — and the debt for inc-4 (loud)

Swap correctness is empirically SUSTAINED at REQ=200 via re-capture-per-serve. **But there is an UNCHARACTERIZED
corruption in the core mux** — concurrent residency churn (specifically a page-in, un-isolated) corrupts live
captured graphs; five black-box probes refuted five candidates; mechanism un-named. This is exactly the kind of
debt that resurfaces under inc-4's harder/longer workload. **Recommendation for inc-4:** keep the per-agent KL=0
gate LIVE across the ENTIRE run (not spot-checked) so a re-emergence is caught immediately, not silently. Anil's
call: accept the working fix + documented debt → inc-4, or fund white-box mechanism work first. STOPPED the
black-box hunt per the time-box (do not stack a 6th candidate).
