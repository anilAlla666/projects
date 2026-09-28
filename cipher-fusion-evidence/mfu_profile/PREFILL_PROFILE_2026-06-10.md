# CIPHER MFU — PREFILL PROFILE DECOMPOSITION + DENOMINATOR CONFIRM (2026-06-10)

Read-only, clean-baseline profile. No kernel, no actuator, no substrate loaded (`VLLM_PLUGINS=""`, 0 substrate
lines confirmed). Anchor `libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — **entry == exit**. Clocks
default (sampled). Profiled via vLLM's built-in worker torch-profiler (the EngineCore worker's CUDA kernels;
3.85 s CUDA total over B=8×T=2048×10 prefills).

## VERDICT: **Fusion lead = Liger-pointwise.** Measured POINTWISE-share 9.0% > FA3-addressable ATTENTION 5.9%; and attention is **already FlashAttention-3**, so no FA3-upgrade lever remains.

But the bigger finding is a **correction to the research's premise**: on the real vLLM path the non-GEMM total is
only **~16%, not 35%** — so the non-GEMM fusion is a *small* lever, and the dominant opportunity is a faster FP8
GEMM (84% of time, currently only ~1.42× over bf16). Details below.

## P1 — KERNEL-TIME DECOMPOSITION (eager vLLM Mistral-7B prefill, clean)

| Bucket | % of CUDA kernel time | Named kernels (top, with the bucketing rule) |
|---|---|---|
| **GEMM** | **83.67 %** | `nvjet_sm90_hsh_*` (4 tiles: 128x256 29.3%, 192x192 26.2%+16.0%, 320x128 12.1%). **Rule: nvjet_* are the cu13 cuBLASLt HSH-variant prefill GEMMs** (fp16-in/fp32-scale/fp16-out) — counted as GEMM. |
| **ATTENTION** | **7.30 %** | `flash::FlashAttnFwdSm90` **5.94 %** (FlashAttention-3 — kept whole, internal GEMM NOT split) + `reshape_and_cache_flash` **1.35 %** (a KV-cache write — has "flash" in the name but is **not attention math**, arguably OTHER). **FA3-addressable = 5.94 %.** |
| **POINTWISE** | **9.00 %** | `vllm::act_and_mul (silu)` 3.76 %, `vllm::fused_add_rms_norm` 2.91 %, `vllm::rotary_embedding` 1.96 %, `direct_copy` 0.25 %, `rms_norm` 0.02 %, Memset 0.05 %. (Liger-addressable.) |
| **OTHER** | **0.03 %** | gather, argmax-reduce, slot-mapping. |

**Reconciliation with the prior `e3_nongemm_share.json` (which said ~64.6 % GEMM / ~31.5 % non-GEMM / ~3.9 % attn):
that was a HF-`transformers` eager forward, NOT vLLM.** HF eager launches many small pointwise ops; vLLM **fuses**
pointwise (`act_and_mul`, `fused_add_rms_norm`) and uses **FA3**, collapsing non-GEMM **35 % → 16 %** and raising
GEMM **65 % → 84 %**. The prior 64.6 % was the number Lane-2's closed-form used; **this clean vLLM re-profile
corrects it to 83.67 %.** (nvjet bucketing is the same point the prior file's auto-bucket fields got wrong —
`share_gemm=0.0` there because the name-matcher missed `nvjet_*`; here it is correctly GEMM.) **Note:** the prior
e3 file's own share fields are internally over-counted (silu_mul 14.2 % + other_elementwise 30.1 % + attn 3.9 % +
nvjet 64.6 % sum to >100 %), so the "35 % non-GEMM" is an *approximate, reconstructed* HF anchor; the new **16 %
vLLM figure stands on its own fresh, internally-consistent trace** (buckets sum to exactly 100 %, independently
re-parsed) regardless of the prior file's inconsistency. The split is **ratio-based and clock-invariant**, so
chaining this 1980-MHz trace onto the 1455-MHz iso-clock MFU bases is sound for the Amdahl projection.

## P2 — DENOMINATOR CONFIRM (both ways)

**Convention in all our reports: MFU = analytic model FLOPs/s ÷ 989.5 TFLOP/s (H100 SXM BF16 dense peak).** FP8 is
measured against the **BF16** peak, not the FP8 peak. One-line arithmetic for the FP8 prefill number:

- vs **989.5 BF16-peak** (the convention we use): FP8 prefill iso-clock = **0.797** (≈0.80); default-clock 0.828.
- vs **1979 FP8-native peak** (silicon utilization): 0.797 × 989.5/1979 = **0.40** (default 0.42).

**Our reports — and the 85 % target and the 93–97 % projection — all use the 989.5 BF16-peak denominator.** So
"FP8 prefill 0.80" and "target 85 %" are the same yardstick; against the FP8-native peak the same run is only 40 %
(FP8 silicon is half-idle — expected, the per-tensor scalar path isn't peak-tuned).

## P3 — PROJECTION ON OUR MEASURED SPLIT

> **Base caveat (read first):** the projection is anchored on the **per-tensor FP8 0.797 iso-clock base, which
> FAILS the PPL bar** (+0.567 %, Lane-2). The **quality-passing rowwise base is ~0.78** (strictly slower, Lane-2),
> which shifts every projected endpoint **down ~2 pts** (e.g. 0.85→needs a bit more cut; 0.875→~0.86). Numbers
> below use the 0.80 fast base as the optimistic reference; the quality path is ~2 pts behind. No kernel was built
> and no 0.85 was reached — these are projections.

> **Relationship to Lane-2's STOP (explicit — this RELAXES, does not contradict):** Lane-2 stopped the
> per-tensor→rowwise kernel build using the HF-derived g=0.646, which made 85 % need fp8_speedup ~1.8–2.0×. The
> corrected vLLM g=**0.837** lowers the implied measured speedup to ~1.42× **and** makes the GEMM lever more
> attractive — but the STOP still holds at the level it was made: **FP8-substitution-as-built tops at 0.80, below
> 0.85.** What this profile adds is that the routes *past* 0.80 (a faster FP8 GEMM kernel, or pointwise fusion)
> are net-new work — exactly the "surface to Anil" options Lane-2 named, now quantified on the real split. The
> Amdahl wall is relaxed, not removed.

From the iso-clock FP8 base (vanilla 0.60 → per-tensor FP8 0.797, PPL-failing), measured split (GEMM 0.837, non-GEMM 0.163):

- Implied **per-tensor FP8 GEMM speedup at the real gemm-share = 1.42×** (reconstructs the FP8 base to 0.797 ✓).
- Post-FP8, non-GEMM is **21.7 %** of the (shrunken) prefill time.
- **To reach 0.85 from the 0.80 FP8 base: cut ~29 % of the non-GEMM** (≈ halve the 9 % pointwise) — but see the
  fusion caveat: the pointwise kernels are **already vLLM-fused**, so "halve pointwise" is real persistent-kernel/
  epilogue work, not a free Liger pass; and from the ~0.78 quality base the cut needed is somewhat larger.
- 40 % non-GEMM cut → **0.873**; 60 % cut → **0.916**; 100 % (free non-GEMM, asymptote) → 1.02 (unphysical).
- **The research's 93–97 % projection does NOT validate on our split:** it needs a **66–82 % cut of ALL non-GEMM**
  (near-total elimination of both pointwise AND attention). That projection was premised on the 35 % non-GEMM (HF)
  figure; on the real 16 % vLLM split, 93–97 % is not reachable by non-GEMM fusion alone.

**Strategic correction (the load-bearing finding): GEMM is 84 % of prefill time and FP8 only gets 1.42× there.**
A *better FP8 GEMM* has more leverage than fusing the 16 % non-GEMM: at GEMM speedup 1.6× → e2e MFU **0.875**;
at 1.8× → **0.955** (off the 0.80 fast base; ~2 pts lower off the 0.78 quality base). So the two net-new paths off
the 0.80 base are (A) Liger-pointwise fusion (closes the last ~5 pts, the small lever, on an already-fused bucket)
and (B) a faster FP8 GEMM kernel (1.42×→1.6–1.8×, the *larger* lever on 84 % of the time). **Both are net-new
engineering — neither is "free," and the 1.42× being below the FP8 tensor-core 2× ceiling is itself the open
question for path (B).** This is consistent with Lane-2's STOP: the as-built FP8 substitution caps at 0.80;
reaching 0.85+ requires one of these two builds, which is Anil's call.

## Fusion-lead verdict + caveat

- **Liger-pointwise leads** the non-GEMM work: pointwise 9.0 % > FA3-addressable attention 5.94 %, and **attention
  is already FA3** (`flash::FlashAttnFwdSm90`) so there is no FA3-upgrade lever — it's already fused/optimal.
- **Caveat (honesty):** the pointwise kernels are **already vLLM-fused** (`act_and_mul`, `fused_add_rms_norm`). The
  remaining Liger-style headroom is not a fresh easy pass — it's fusing pointwise into GEMM epilogues / persistent
  kernels (real CUDA work, uncertain yield on an already-9 % bucket). Given that, and that GEMM is 84 % at 1.42×,
  **the highest-leverage MFU lever is the FP8 GEMM kernel itself, with Liger-pointwise second; FA3 is a non-lever
  (already in use).**

## Integrity
- Anchor md5 entry == exit `2edba0d2136f8ede4713d90a8f7cd55f`; no build, no actuator, no substrate into vLLM
  (`VLLM_PLUGINS=""`, 0 substrate lines in `profile.err`). Clocks default. Scratch only in `mfu_profile/`.
- Every number → `mfu_profile/bucket_decomposition.json` + `projection.json` + `profile_meta.json` + the raw
  `trace/*.pt.trace.json.gz`. No retracted/invented number; nvjet→GEMM and reshape_and_cache→KV-write bucketing
  rules stated.

## Verification panel (mandatory adversarial pass — 3 read-only dimensions; record `verification_panel.json`)

**Result: 3 material + 2 minor + 3 note — all applied.** The bucket decomposition itself was independently
re-parsed and **confirmed exact** (total 3851 ms, all 30 kernels, GEMM 83.67 %, nvjet→`aten::mm`, FA3→`_vllm_fa3_C`,
reshape_and_cache→cache-op — no double-count, buckets sum to 100 %).

| Dim | Passed | Material findings → action |
|---|---|---|
| 1 BUCKETS | 13 (trace re-parsed exact; nvjet correlation-id → aten::mm confirms GEMM; FA3 not via cuBLAS; warmup excluded; GPU 96.3 % saturated; buckets exhaustive) | none material; e3 "35 %" flagged as approximate/over-counted reconstructed anchor; clock-invariance noted |
| 2 PROJECTION | 14 (all arithmetic reproduced: s=1.42 at g=0.837, 29 % cut to 0.85, denominator both-ways) | **(g) per-tensor base FAILS PPL not stated + quality ~0.78 base not carried → fixed (base caveat block + −2 pt shift)**; **(e) corrected g silently relaxed Lane-2 STOP math → fixed (explicit "relaxes not contradicts" reconciliation)** |
| 3 VERDICT+INTEGRITY | 12 (Liger-pointwise verdict sound, FA3-already-in-use confirmed, anchor md5 clean, 0 substrate lines, clocks default) | **(c) "faster FP8 GEMM is larger lever" contradicted Lane-2 STOP without acknowledgement → fixed (now framed as net-new Anil-call work consistent with STOP)**; pointwise-already-fused temper moved up to P3 |

**Exit state:** anchor `2edba0d2…` entry==exit; clocks default (345 MHz idle); 0 compute procs; 0 files outside
`mfu_profile/`; no git writes.
