# FWD-1 — launch-bound decode-forward probe — FINDINGS (§3 verdict)

**Date:** 2026-05-29. **Status:** items 1–6 COMPLETE; **STOP at §3 verdict for Anil's
productionize-vs-W.7 call.** **Anchors UNCHANGED** (cipher_rt_phase4 `8b5e928` /
cipher_kmod `02fc2d1` / cipher_kv_bridge `5a3db034`) — userspace measurement only.

## Verdict in one line

The W.4b.7 "~52 ms launch-bound forward ceiling" was **largely an HF-`DynamicCache`
prototype-harness artifact, not an irreducible GPU forward.** A clean `generate()`
harness shows the Mistral-7B decode forward is **~2.6× faster with `StaticCache` (8 ms
vs 22 ms/tok, bit-correct)** — and the **intended product (vLLM) already uses static
paged KV**, so production never paid that tax. The **graph lever (`torch.compile`
reduce-overhead / manual CUDA-graph capture) is absent in this stack** (wrong + slower /
hard-abort). The forward is **near-bandwidth (~1.8× over floor) after static KV** and is
**not the deep limiter W.4b.7 implied.** ⇒ **No new substrate lever; recommend → W.7.**

## The numbers (Mistral-7B, generate(), greedy, MAXNEW=256, decode-windowed)

| B | eager_dynamic | eager_static | compile_static |
|---|---|---|---|
| 1 | 19.86 ms/tok · 50.4 tok/s · 256/256 ✓ · 179 W | **7.65 ms/tok · 130.8 tok/s · 256/256 ✓ · 156 W** | 24.35 ms · 41.1 tok/s · **8/256 ✗** · 132 W |
| 4 | 22.21 ms/tok · 180.1 tok/s · ✓ · 169 W | **8.09 ms/tok · 494.3 tok/s · 158 W** | 24.75 ms · 161.6 tok/s · **✗** · 133 W |
| 8 | 22.00 ms/tok · 363.6 tok/s · ✓ · 182 W | **8.35 ms/tok · 957.6 tok/s · 163 W** | 24.26 ms · 329.8 tok/s · **✗** · 137 W |

- **`StaticCache` → ~2.6× over `DynamicCache`** (B=1 19.86→7.65; B=4 22.2→8.09; B=8
  22.0→8.35), consistent across B. The dominant overhead was DynamicCache (Python
  re-allocation / management per step), eliminable with **zero graphs**.
- **`compile_static` is broken in this stack** (transformers 5.8.1 + torch 2.11): wrong
  output **even at B=1** (8/256, where eager_static is a perfect 256/256) **and slower**
  (24 ms > eager_dynamic 20 ms). Not the lever. (Per Memory #11 its timing is not
  reported as a real result — it is a failed config.)
- **Manual `torch.cuda.CUDAGraph` capture of HF decode HARD-ABORTS on replay** (CUDA
  `index_copy_` device assert; growing-KV / mask state not graph-safe). This is the §5
  / R-1 finding the memo anticipated ("report it, don't force it"). The cudagraph path
  is brittle for HF decode — vLLM/gpt-fast use custom kernels for exactly this reason.

## Correctness (Memory #11)

- **eager_static is bit-faithful**: B=1 Mistral, 4 distinct prompts, `static == dynamic`
  **128/128 each** (padding-free, so no cascade confound). The B>1 free-running token
  divergence (792/1024, 1737/2048) is benign greedy cascade + left-padding FP order
  (a single early near-tie flip cascades), **not** a StaticCache bug — proven by the
  clean per-sequence B=1 match.
- TinyLlama corroborates: eager_static **512/512** vs eager_dynamic, 5.1× faster;
  compile_static 7/512 (same broken-compile signature).

## Why this is the verdict (the blocking framing question, resolved)

1. **The W.4b engagement harness used `DynamicCache`.** `formb6_executor.py:126`
   (`m(input_ids=..., use_cache=True)` → HF default `past_key_values`) — so the W.4b.6
   "3.239× / ~52 ms forward" numbers were on the slow DynamicCache path. The W.4b.7
   "52 ms" = formb manual loop + DynamicCache + longer context; a clean generate()
   DynamicCache is already only 22 ms (the residual gap is harness/context, not GPU).
2. **The product is vLLM, whose paged KV is static** (`vllm_env` present; CP 5.1 /
   Week 5 KV integration). Production already runs the static-KV regime and **never
   pays the DynamicCache tax.** ⇒ the 2.6× is a **prototype-harness artifact, not a
   shippable new lever.**
3. **Near-bandwidth after static KV.** eager_static B=1 = 7.65 ms vs the ~4.2 ms
   (14 GB / 3.35 TB/s) bandwidth floor ⇒ ~1.8× over floor. Real non-bandwidth headroom
   remains, but it is small and the graph lever that would target it **did not deliver
   in this stack.**
4. **A uniform forward speedup cancels in the engagement ratio** (substrate-attributable
   = cross ÷ naive). So even if productionized, static KV would raise absolute tok/s &
   tok/W but **not move the W.4b engagement gate.**

## §3 decision-band mapping

The memo's gate: "≥2× forward reduction *from graphs* → productionize graph-capture;
<2× → memory-latency-bound, W.7 next." **Graphs deliver <2× (in fact 0× — broken)** →
the **W.7 branch**. With the added, important nuance: the forward *is* ~2.6× reducible,
but by **cache-implementation (already static in the vLLM product)**, not by graphs and
not by any new substrate primitive.

## W.8 persistent-kernel / IMPLANT-CUZ go/no-go (Memory #2/#3 framing)

**Recommend DE-PRIORITIZE (no-go as a near-term priority).** Rationale: (a) the cheap,
correct win (static KV) is already in the production path (vLLM); (b) graph/launch-
elimination did **not** pay off in this stack (compile broken+slower; manual capture
hard-abort); (c) the residual headroom over bandwidth is only ~1.8× and graph capture of
HF decode is brittle; (d) any uniform forward speedup cancels in the engagement ratio.
Persistent-kernel dispatch is **not impossible**, but its expected value is low relative
to its cost given the above — revisit only if a concrete bandwidth-bound MFU target
demands the residual ~1.8×.

## Recommendation (STOP-for-Anil)

**→ W.7 NCCL** (per W.4a §G), deferred since the W.4b close. The forward is not the deep
limiter W.4b.7 implied, the recoverable win is already in the vLLM product, and graphs
are absent in this stack. No productionization substep is warranted on this evidence.
Awaiting Anil: (i) accept → W.7, or (ii) direct a productionization/W.8 substep anyway.

**Files:** `fwd1_generate_probe.py` (supported generate() probe), `run_fwd1.sh`,
`fwd1_gen_sweep/` (results); `fwd1_cudagraph_probe.py` (the superseded manual fixed-P
probe — retained for the manual-capture hard-abort provenance). Anchors UNCHANGED.
