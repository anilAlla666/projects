# W.4b.7 — per-token rendezvous decomposition (item 1) — FINDINGS

**Date:** 2026-05-29. **Status:** Item 1 (decompose) COMPLETE. **The decomposition
falsifies the memo's premise and reaches the §4 verdict at baseline — items 2–6
(amortization) are now COUNTERPRODUCTIVE on the engagement metric. STOP-and-resurface
to Anil before any further code** (memo §6 was built on "rendezvous IS the gap"; item 1
shows it is not).

**Anchors UNCHANGED** (verified on disk at entry): cipher_rt_phase4 `8b5e928`
(libcipher_rt.so md5 `9fe23143`), cipher_kmod `02fc2d1` (0.6.6), cipher_kv_bridge
`5a3db034`. This step added only **env-gated profiling** (`CIPHER_FORMB_PROFILE=1`,
OFF by default) to `formb_executor.py` / `formb_tenant.py` and an env pass-through in
`run_formb_proto.sh`. No substrate / kmod / bridge edit. Per-token executor-owned-KV
compute path unchanged.

---

## 1 — The decomposition (N=4 Mistral-7B, GEN=64, weight-share NOT needed at N=4)

Executor per-step breakdown (cuda-Event GPU spans + perf_counter CPU phases; the
existing per-step sync at `formb_executor.py:~253` makes the events readable). **Both
the artifact-present (`CIPHER_RT_DISABLE_AUTO_INIT=0`) and artifact-suppressed (`=1`)
conditions are identical** — the W.4b.6 CUPTI auto-init artifact does NOT affect the
per-token executor (its forward is too small to amortize the shim, unlike the
full-handoff B=N forward):

| phase | ms/step | note |
|---|---|---|
| executor recv (await R tenant embeds) | 4.23 | absorbs tenant embed+write+socket |
| **executor gpu_forward** | **50.9** | **THE BOTTLENECK** |
| executor gpu_gather | 0.02 | cuIPC reads + clone |
| executor gpu_scatter | 0.08 | cuIPC writes |
| executor sync_wait (`torch.cuda.synchronize`) | 0.02 | returns ~instantly — forward already drained |
| executor send (R done signals) | 0.09 | |
| **step total (wall)** | **55.5** | 72.0 tok/s |

Tenant side (per step): embed 1.97 / **write_sync 0.96** (the un-amortized `sync=True`,
`formb_ipc.py:94`) / send 0.02 / **recv_wait 51.7** (just blocking on the executor
forward) / logit_read 0.8.

**The rendezvous (recv 4.2 + gather 0.02 + scatter 0.08 + sync 0.02 + send 0.09 ≈ 4.4
ms) is ~8 % of the step. The forward (50.9 ms) is 92 %.** The memo's "the rendezvous
IS the entire gap" is **false** — it is a small fraction.

## 2 — The forward is kernel-launch-latency bound (clock-independent)

A B=4 single-token Mistral-7B decode forward = ~51 ms. Bandwidth math says a
14 GB-weight read is ~4 ms, so this is ~10× over a bandwidth-bound floor ⇒ it is
**launch-latency bound** (32 layers × many tiny kernels, each gated by launch + small-op
latency, not by HBM or by clock). Proof — same step time at half vs full clock:

| config | decode clock | board W | tok/s | ms/step |
|---|---|---|---|---|
| in-proc B=4 (GEN=512, the ceiling) | 1980 MHz (steady) | 146.4 | 76.0 | 52.7 |
| cross-proc B=4 (GEN=256) | mix 1005/1980 (oscillates) | ~116 | 74.5 | 53.7 |
| cross-proc B=4 (GEN=64) | 1005 MHz (drooped) | 99.1 | 72.0 | 55.5 |

~52 ms/step at 1980 MHz and ~51 ms at 1005 MHz ⇒ clock barely matters ⇒ launch-bound.

## 3 — Throughput parity (ROBUST) vs tok/W (FRAGILE)

- **ROBUST: cross-process throughput ≈ 0.98× of the in-proc ceiling** (74.5 / 76.0
  tok/s). The per-token path is already at near-parity in tok/s.
- **FRAGILE: tok/W.** Cross-process measured 0.727 (GEN=64) then 0.644 (GEN=256) tok/W
  as the clock oscillated non-deterministically; in-proc is 0.519 (matches W.4b.5's
  recorded in-proc 0.527). Do **not** bank "cross-process is more tok/W-efficient than
  in-proc" as a substrate win — it is a measurement-fragile by-product of being
  latency-bound (see §4).

## 4 — Why amortizing the rendezvous (items 2–6) is COUNTERPRODUCTIVE

The apparent cross-process tok/W "advantage" is **caused by the rendezvous gaps**: the
serial socket ping-pong leaves the GPU idle between steps → the clock droops to ~1005
MHz → at a launch-bound (clock-insensitive) forward, throughput is ~unchanged but power
drops (~99–116 W vs in-proc's 146 W) → tok/W *rises*. Items 2–6 exist to *remove* those
gaps. Removing them keeps the GPU fed → clock returns to 1980 MHz → power ~146 W → with
no throughput gain (launch-bound) → **tok/W falls from ~0.64 toward in-proc's ~0.52.**
Amortization is not "marginal" on the engagement metric — it is net-negative.

Throughput-wise the removable slice is < 8 % anyway: of the 4.4 ms, the tenant
embed+write (~3 ms) is **serial-before-the-forward in real (non-teacher-forced)
autoregressive decode** (the next token's embed depends on this step's logit) and cannot
overlap; only ~1.4 ms (socket + host sync) is genuinely removable ≈ **2.5 % of the step**.
(The teacher-forced harness ships the gold token and so *could* pre-pipeline embeds, but
real serving cannot — pipelining a single sequence across tokens is forbidden by the
autoregressive dependency, which is branch-B/R-4 territory, not option 1.)

## 5 — §4 verdict (reached at baseline, N=4) — for Anil

**GREENLIGHT band (cross/inproc ≥ 0.80×) is ALREADY MET without any amortization:**
throughput 0.98×; tok/W ≥ 0.80× (≥ 1.0× on every measurement). The per-token
rendezvous for a fixed `[1,hidden]` embed is already cheap (~8 %), and the dominant
cost (launch-bound forward) is identical in-proc and cross-process.

**Three corrections on how to read this GREENLIGHT (do not over-read):**

1. **GREENLIGHT ≠ R-4 viable.** This de-risks the **embed** rendezvous only — a fixed
   `[1,hidden]` transfer. R-4 (branch B) moves **peer KV**, which **grows with sequence
   length**; "rendezvous is cheap here" therefore does **not** bound R-4's transport
   cost. (This is exactly the over-read Anil's approval pre-empted: "a clean rendezvous
   number must NOT be over-read as branch B delivered.") The real ceiling-limiter
   surfaced here is the **launch-bound forward** — a CUDA-graph / larger-batch axis,
   which is neither rendezvous nor branch B.
2. **History is not "wrong" — it was already fixed.** in-proc 0.519 ≈ W.4b.5's 0.527.
   Only the cross-process number differs (current near-parity vs historical
   "rendezvous-bound 79 ms/step, 0.518×"). The most consistent reading: the **W.4b.4
   amortization (1 sync/side/step, already shipped)** closed the 79 → ~55 ms gap, so the
   per-token path is *already* near-parity because that fix landed. The historical
   number is flagged, not overwritten.
3. **Correctness (Memory #11) PASS, asserted with gaps.** Positive N=4 GEN=64:
   tf_match 64/64 rows 0–2, row 3 = 63/64; KL ≤ 3.86e-5. GEN=256: rows 1,2 = 256/256;
   rows 0,3 = 255/256. The two divergences are **benign near-ties**, gaps cited: row-0
   step 160 solo top1−top2 gap = **0.0** (exact tie — drives the kl_max 7.527e-5, a
   single tie-break flip), row-3 step 20 gap = **0.0078** (the W.4b.5/.6 signature).
   distinct_rejected=0 (positive mode); FIRSTCOALESCE fails=0.

## 6 — Recommendation (STOP-and-resurface, per campaign discipline)

Item 1 falsified the premise on which memo §6 items 2–6 were built. Per the
"resurface-when-the-plan's-premise-breaks" discipline this campaign runs on:

- **Recommend: SKIP items 2–6** (they are counterproductive on tok/W and recover < 3 %
  throughput) and **adopt the §5 GREENLIGHT-at-baseline verdict.**
- The R-4-vs-close adjudication is Anil's (unchanged). The honest input to it: the
  *embed* rendezvous is cheap; R-4's *peer-KV* transport is unbounded by this result;
  the ceiling-limiter is the launch-bound forward (CUDA-graph axis).
- **NOT run:** N=8 (would need the W.4b.6 weight-share ported into the per-token path).
  N=4 already answers the verdict question (rendezvous is structurally a small fraction;
  the launch-bound forward and the autoregressive-serial constraint are batch-topology
  -independent in mechanism). N=8 would refine the tok/W magnitude, not the structural
  finding — logged here rather than silently omitted.

**No `w4b-close` move, no R-4 build, no item 2.** Awaiting Anil's adjudication:
(i) accept SKIP-2–6 + GREENLIGHT-at-baseline and decide R-4-vs-close, or
(ii) direct that 2–6 run anyway (e.g. to harden the embed-rendezvous primitive for R-4
despite the tok/W cost), or (iii) request N=8 first.

---

## 7 — ADJUDICATION (Anil, 2026-05-29): accept SKIP 2–6 + GREENLIGHT-at-baseline

**W.4b.7 CLOSED.** Anil adopted the §5 GREENLIGHT-at-baseline verdict and directed
that items 2–6 (rendezvous amortization) be SKIPPED — they are counterproductive on the
engagement metric (§4) and the gate is already met at baseline. Item 1 (decompose) is
the entire substep; 2–6 retired by adjudication, not executed. Next decision surfaced to
Anil: **R-4 (branch B) vs close W.4b on branch A.** Honest inputs to that call:
- GREENLIGHT de-risks only the fixed `[1,hidden]` EMBED rendezvous; R-4's peer-KV
  transport (grows with seq len) is **unbounded by this result**.
- The real ceiling-limiter surfaced is the **launch-bound forward** (CUDA-graph / batch
  axis) — orthogonal to branch A/B; it would gate R-4 regardless and also bounds
  branch A's absolute speed.
- Branch A's engagement gate is already cleared (W.4b.6, 3.239×); per-token branch B
  (R-4) remains a large net-new build with the v1-buildable-vs-v2-research question open.

Anchors UNCHANGED. `w4b-close` still parked at `3ffa9df` pending the R-4-vs-close call.
