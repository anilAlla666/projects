I have a complete dossier already. The synthesis task is the actual deliverable here. Let me produce the product proposal grounded only in the techniques rated 'real-and-deliverable' or 'real-but-needs-model-or-framework'.

Let me first extract the verdicts from the dossier to ground my synthesis honestly.

The dossier covers speculative + quantization fully, and kernels_mfu is truncated. Let me work with what's verdicted:

**Verdicts in the dossier:**
- EAGLE-3/3.1: real-but-needs-model-or-framework (needs trained head + vLLM runtime; gain decays to break-even/negative under saturation)
- Medusa: not-for-this-setup
- DeepSeek-MTP: not-for-this-setup
- LayerSkip: not-for-this-setup
- FP8 W8A8: real-but-needs-model-or-framework (substrate cleanly delivers weight-side only ~W8A16; honest ~1.1-1.3x batched throughput / ~1.5x batch-1 MBU / ~0.5-2pp accuracy)
- AWQ/GPTQ W4A16+Marlin: real-but-needs-model-or-framework (needs pre-quantized checkpoint; ~1.5-2x batch-1/low-QPS latency, throughput-neutral-to-negative under saturation on VRAM-abundant H100)
- SmoothQuant: not-for-this-setup
- QServe W4A8KV4: not-for-this-setup
- Kernel libs (FA3, Flash-Decoding, FlashInfer): yes-pure-substrate but "largely already done" in modern vLLM
- Megakernel: needs-framework-change
- Pointwise fusion: framework-owned, single-digit

Now let me write the synthesis. The honest conclusion the prompt explicitly invites — that the throughput moat may be small and the real product is reliability + efficiency-orchestration — is strongly supported by this team's own memory (the entire R.A/R.C fault-injection arc: SDC detection inside real vLLM, per-GPU fleet reliability, cross-process/cross-tenant visibility).

Let me produce the final synthesis.

---

# CIPHER PRODUCT SYNTHESIS — honest product direction grounded in the dossier

## Headline verdict

The throughput/MFU moat for an intercept-only substrate on H100/7B-dense/vLLM is **small and mostly already captured by modern vLLM.** Every large decode win in the dossier (EAGLE-class spec-decode, megakernels, W4A16) either (a) lives in the serving runtime above the GEMM boundary, (b) needs a model-side artifact (trained head / pre-quantized checkpoint), or (c) goes break-even-to-negative in this team's saturated-batch operating point. The one pure-substrate precision lever (FP8) the team already measured at ~1.05–1.24x.

**So the defensible product is NOT a throughput hero. It is a cross-tenant efficiency-and-reliability orchestrator that the interposition position uniquely enables — with FP8/kernel-dispatch as the in-band efficiency floor and the team's already-built SDC/per-GPU-reliability detection (R.A/R.C) as the real differentiated value.** Below I give both: the small-but-real efficiency product, and the actually-defensible moat.

---

## 1. THE PRODUCT

**"CIPHER Fleet Guard + Efficiency Floor"** — a CUDA-interposition control plane that runs UNDER unmodified vLLM and delivers two things no vLLM feature delivers from that position:

- **(A) In-band correctness + per-GPU reliability detection** across processes/tenants (SDC/silent-corruption catch, per-GPU degraded-GPU verdict, 0-FP) — the team has already BUILT and validated this inside real vLLM (R.A: detector fires, catches injected SDC, 0 FP, ~3% detector cost eager; R.C: per-GPU DEGRADED verdict, 0-FP over 4352+ checks).
- **(B) An efficiency floor it can actually actuate from the driver seam:** FP8 weight-side precision (pure-substrate, proven), kernel-library dispatch (FA3/FlashInfer behind intercepted calls), and cross-tenant DVFS/clock/power orchestration (~1.4x tok/W lossless, only deployable as a fleet-level scheduler decision — which is exactly the substrate's cross-process vantage).

**Expected composed gain (honest):** the efficiency floor is **~1.1–1.3x throughput in batched serving / ~1.5x MBU at batch-1 / ~1.4x tok/W under power headroom (NOT at full load), all already roughly at the team's measured ceiling.** The *product value* is not that multiplier — it is that the substrate adds **silent-data-corruption + degraded-GPU detection at <~3% cost with 0 false positives, cross-tenant, with zero model retrain and zero vLLM fork.** That is the differentiated, defensible deliverable.

**Why the substrate position is the right vehicle:** correctness/reliability detection and cross-tenant power/clock orchestration genuinely require seeing GEMM/kernel calls across process boundaries and setting GPU state — capabilities that live precisely at the driver-interposition layer and NOT in any single vLLM process. Every throughput technique, by contrast, lives above the GEMM boundary, so the substrate has no structural advantage there.

---

## 2. THE TECHNIQUE STACK

Only techniques the assessments rated real-and-deliverable or real-but-needs-model-or-framework:

| # | Technique | Honest gain | Axis | Model dependency | Substrate fit |
|---|-----------|-------------|------|------------------|---------------|
| 1 | **FP8 (E4M3) weight-side precision** | ~1.5x MBU @batch-1; ~1.1–1.3x throughput batched; ~0.5–2pp accuracy (task-dependent; math/long-ctx worst) | MBU/throughput | **none** for weight-side cast (clean part = effectively W8A16; full W8A8 needs fused activation-cast cooperation to stay cheap) | real-but-needs-framework for *fused* W8A8; pure-substrate for weight-side |
| 2 | **Kernel-library dispatch (FA3 / Flash-Decoding / FlashInfer)** | FA3 prefill MFU ~75% of peak vs FA2 ~35%; Flash-Decoding up to 8x long-ctx batch-1; FlashInfer 29–69% ITL — but **marginal over an up-to-date vLLM that already calls these** | MFU (prefill) / long-ctx decode latency | **none** | yes-pure-substrate (but small delta vs current vLLM) |
| 3 | **Cross-tenant DVFS / clock-power orchestration** | ~1.4x tok/W lossless (trades throughput; not deployable at full load) | TPW | none | substrate-unique (needs cross-process vantage + GPU state setting) |
| 4 | **(Optional, model-cooperation) AWQ/GPTQ W4A16 + Marlin** | ~1.5–2x **latency** at batch-1/low-QPS; **throughput-neutral-to-negative** under saturation on VRAM-abundant H100; ~0.05–0.2 PPL | MBU (memory-bound decode only) | **needs pre-quantized checkpoint** (calibration + Marlin repack) | needs-model-cooperation; substrate only *hosts* the kernel, which vLLM already does given the checkpoint |

**Compose:** FP8 weight-side (1) + kernel dispatch (2) form the in-band efficiency floor; DVFS (3) is the cross-tenant TPW lever; W4A16 (4) is an OPT-IN latency mode for interactive/low-QPS tenants only, and only if a quantized checkpoint is supplied — it is NOT a substrate-native win.

**Explicitly model-dependent:** #4 needs a quantized checkpoint. #1's *cheap fused* form needs framework cooperation. EAGLE-3 (best spec-decode) was rated real-but-needs-model-or-framework but is **excluded from the shippable stack** because it requires BOTH a trained per-model head AND vLLM runtime tree-verification — the substrate can deliver neither, and the gain is ~break-even/negative in the team's saturated regime.

---

## 3. SUBSTRATE-UNIQUE MOAT

What only the interposition position can do (so this is not "just use vLLM features"):

1. **Cross-process / cross-tenant SDC + degraded-GPU detection at the GEMM seam.** vLLM sees one process; the substrate GOT-patches cuBLAS across every tenant on the GPU and can re-run/checksum GEMM outputs to catch silent corruption and localize a degraded GPU — already proven (R.A: 0-FP, catches injected bit-flips inside real vLLM; R.C: per-GPU DEGRADED verdict, descriptor = SDC-pid == NVML-pid == cohort-tgid). No vLLM feature does this; DCGM/NVML catch hardware counters, not correctness.
2. **Fleet-level DVFS/power arbitration informed by what's actually running.** The substrate classifies the live workload across processes and sets clocks/power — a decision no single vLLM process can make because it can't see its co-tenants.
3. **Zero-touch deployment.** Runs under unmodified vLLM, no retrain, no fork — the reliability layer ships without touching the customer's model or serving stack.

The moat is **safety + cross-tenant efficiency orchestration**, NOT raw tokens/sec. Be explicit about this with stakeholders.

---

## 4. THE HONEST CEILING

- **Best-case composed efficiency:** ~1.5x MBU at batch-1 (FP8) and ~1.4x tok/W under power headroom — i.e., roughly the team's already-measured numbers. Batched-serving throughput lift is ~1.1–1.3x and **shrinks toward 1.0x at saturation** (the team's own ~68% MFU / compute-bound regime). Kernel-dispatch delta over current vLLM is single-digit.
- **vs the frontier:** the 2–5x headline decode wins (EAGLE-3, SuffixDecoding, megakernels, W4A16-at-batch-1) are real but live in the runtime/checkpoint, so the substrate cannot capture them without becoming a serving runtime or shipping a model artifact. CIPHER does NOT close that gap.
- **What it does NOT do:** does not beat an up-to-date vLLM on tokens/sec; does not deliver speculative decoding; does not deliver FP4 (Blackwell-only — zero win on Hopper); does not turn the ~290x spare decode FLOPs into tokens (that requires spec-decode, which is framework-owned). The "more tokens per weight-stream" headroom stays unclaimed by the substrate.

---

## 5. BUILD PLAN — first measurable milestone (1–2 weeks)

Prove-or-kill on the team's existing H100 / Mistral-7B / real-eager-vLLM rig (reuse the R.A/R.C harness that already exists; anchor 2edba0d2 entry==exit, FORK-1-PRESERVING):

1. **Reliability headline (the differentiator):** On real vLLM serving a saturated trace, run the SDC detector + per-GPU verdict CONCURRENTLY across **2 co-resident tenants** and measure: (a) catch rate on injected persistent + transient SDC, (b) false-positive count over ≥10k checks, (c) end-to-end tok/s cost. Target: 0 FP, persistent SDC caught within N, **<5% tok/s cost in the cudagraph-on hosting path** (the open question from R.A — eager tax is the known wall; this milestone decides if the reliability product is cheap enough to ship in production cudagraph mode).
2. **Efficiency floor sanity:** Confirm FP8 weight-side + current kernel dispatch reproduces ~1.1–1.3x batched / ~1.5x batch-1 MBU vs the customer's current vLLM config (not vs a strawman) — establishes the floor is real and not negative.
3. **Cross-tenant DVFS:** Show the fleet-vantage power arbitration delivers ~1.4x tok/W on a power-headroom node and **correctly declines to actuate at full load** (honest gating).

Deliverable: one table — catch/FP/cost for reliability, and floor multipliers vs real baseline — that either supports "ship the safety+floor product" or kills it.

---

## 6. KILL CRITERIA

The product is NOT real if any of these measure true:

1. **Reliability cost wall:** the SDC/per-GPU detector cannot run under <~5% tok/s in the **cudagraph-on** hosting path (i.e., the +51% eager tax is unavoidable AND cudagraph-frame detection can't be made cheap). R.A already flagged in-graph injection as a measured wall (vLLM rejects external graph mutation) — if the only viable host is eager, the production reliability product dies on cost. **This is the single highest-risk kill criterion.**
2. **False positives under concurrency:** any false-positive SDC/degraded verdict across co-resident tenants at production check rates (R.C's 0-FP must hold multi-tenant, not just single-process).
3. **Efficiency floor is negative vs real baseline:** if FP8 weight-side + kernel dispatch does NOT beat the customer's current up-to-date vLLM (because vLLM already does FP8 + FlashInfer), the efficiency floor adds nothing and only the reliability story remains — and if reliability also fails (1) or (2), there is no product.
4. **No cross-tenant demand:** if real deployments are single-tenant-per-GPU, the substrate's cross-process moat evaporates and customers should just use vLLM's native FP8/kernels — the substrate is then redundant.

---

**Bottom line for the team (the honest answer you asked for):** the throughput moat is small and largely already in vLLM. The real, defensible CIPHER product is **reliability + cross-tenant efficiency orchestration** — silent-corruption/degraded-GPU detection and fleet power/clock arbitration from the driver seam — wrapped around a modest, already-proven FP8/kernel efficiency floor. Lead with safety and cross-tenant TPW, not tokens/sec. The make-or-break measurement is whether the SDC/per-GPU detector can run cheaply in the **cudagraph-on** path; everything else is already de-risked by the team's own R.A/R.C work.