# CIPHER — FP8 DETECTION ARC, SESSION CLOSE-OUT
**Arc dates:** 2026-06-12 (keystone build + Add.1–12) → 2026-06-13 (Add.13 continuation + close-out).
**Anchor:** `libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — frozen at start AND end of every
session in this arc. READ-ONLY on the substrate: only scratch `.py`/`.c` probes and `.md`/`.json`
artifacts were written; the `.so`/`.ko`/kernels were never rebuilt.
**Detail of record:** `FP8_SEAM_RESULT.md` (Addenda 1–13). This doc is the strategic summary.

---

## A. THE CAPABILITY PROVEN

Zero-plugin FP8 SDC detection **works under default config** (V1 multiprocessing ON + cudagraph ON),
**injection-only** (delivered solely as an injected `.so` via `CUDA_INJECTION64_PATH`/`LD_PRELOAD` —
no pip install, no `vllm.general_plugins` entry point). The SDC detector, which was ~93% blind to FP8
linears at the cuBLAS GemmEx seam (FP8 routes through `cutlass_scaled_mm`, not cuBLAS), now **sees
100% of the FP8 decoder linear path**.

The validated mechanism is **always-on in-graph EXACT RECOMPUTE** (Addendum 4):
- Route the residual through a **module buffer** (dynamo lifts it as a graph input — the same channel
  vLLM uses for its KV cache — so the write survives inductor DCE and vLLM's manual cudagraph capture).
  Closed-over globals get DCE'd; that was the original blindness.
- GATE PASSED on real `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`, default config (`cg2gate_on.json`):
  128 layers, cnt=32768 (0→100% coverage), clean_res=0.0 / 32768 (**0 false positives**), injected
  fault **caught under cudagraph** (res→0.5), CSE-independent (operand-perturb res=6.0 vs clean 0).

This is a real, differentiated, deployable capability. It is not the issue. The cost is.

---

## B. THE COST PROBLEM

Always-on exact recompute = **−51.9% on decode throughput** (measured, Add.4: OFF 250.82 → ON 120.68
tok/s). That is the price of recomputing every FP8 linear every token.

The **faithful-every-token floor is ~2×** (dual-modular-redundancy: to be sure an output is correct
without a checksum shortcut, you compute it twice). This is not a CIPHER artifact — it is the
established floor in the SDC literature (Google "Cores that don't count", Meta silent-data-corruption
fleet studies, NVIDIA resilience work, OCP fault-tolerance). We then re-confirmed it **5 independent
ways on our own stack** (Section C). Every attempt to beat 2× either lost detection coverage or failed
to reduce cost below the recompute floor.

---

## C. EVERY COST-REDUCTION PATH TRIED (the exhausted search)

| Path | Outcome | Why |
|---|---|---|
| Periodic dual-graph (checked vs unchecked variant, switch every Nth step) | **WALLED** (Add.5) | vLLM caches ONE compiled forward; the host-flag can't produce two variants inside vLLM. Cost structure works (−4.4%@N=32) but both graphs are the same unchecked variant → sink_cnt=0, never catches. |
| Freivalds checksum (A·(W·g) ?= C·g) | **WALLED** (Add.6) | √N dilution is fundamental: misses most single-element SDC — mantissa bit-flips 0%, most exponent <10%, gross bit-14 only 75%/check (vs exact 100%). AND cost fails: +44.7% under cudagraph (tiny-kernel-overhead-bound on the fast decode GEMM). |
| ABFT row+col, separate kernel | **WALLED** (Add.7) | At M=1 decode there is no row dimension, so the detecting column-checksum IS a full recompute (+928% fp32 / ~+100% fp8) while catching LESS than exact — strictly dominated. Same M=1 root cause as Freivalds. |
| Out-of-band periodic recompute | **COST OK, CATCH UNVALIDATED** (Add.8/9/10/12 + Add.13) | See below — the most promising path, and the one with the lone remaining open test. |
| Fused-ABFT (own a CUTLASS-class kernel, fuse the checksum into the epilogue) | **KILLED at GATE 1** (Add.11) | The only kernel we can author+fuse (Triton FP8 GEMM, autotuned, numerically correct) is **3–4× slower** than CUTLASS on every shape (decode qkv +316%, down +233%, prefill gate_up +320%). CUTLASS itself and cuBLASLt match the bar but are **non-fusable vendor black boxes**. Field fusion (TurboFFT/ATTNChecker) works because they OWN the serving kernel; we would have to replace vLLM's per-shape-tuned CUTLASS with a vLLM-tuned-parity custom CUTLASS FP8 GEMM + epilogue visitor — major unbuilt C++ effort, uncertain outcome. Stopped before fusing a checksum onto a slow base and calling the sum a win. |

### The out-of-band path in detail (the near-miss)
**Cost MEASURED good:** expose per-layer FP8 I/O via cheap in-graph bf16 copies (measured **+0.1%** —
plain copies overlap the weight-memory-bound decode; the earlier −37.8% proxy was a float-convert, not
the copy), then recompute from the buffered I/O out of graph every Nth step. Measured **−8.9% @ N=64**;
projected ~1% @ N=512. One compiled forward (beats dual-graph), exact detection (beats Freivalds/ABFT),
check outside the graph (beats device-gate). It sidesteps every prior wall.

**Catch UNVALIDATED — blocked 4 times, every block a TEST-INJECTION artifact, never the detector:**
- Add.9: in-place fault + separate copy → inductor reordered the copy before the fault (copy read clean).
- Add.10: clean-weight-swap reference recompute is inert — vLLM's compressed-tensors FP8 path uses a
  cached/processed weight that doesn't track in-place `layer.weight.data` edits.
- Add.12: weight corruption can't reach the exposed GEMM — it reads a repacked FP8 weight; the exposed
  `out` is bit-identical to clean even when a zeroed weight diverges generation.
- **Add.13 (this continuation):** two more in-graph reformulations — **v2** (functional fault feeding
  both copy and downstream) and **v3** (corruption + copy fused into one opaque custom op) — BOTH fail
  under cudagraph (v2: downstream diverges but obuf clean; v3: graph-internal `out` mutation doesn't
  even propagate, because only module-buffer mutations are lifted/persisted). **But the EAGER control
  (v4) CAUGHT: 0 FP, inj_max_resid 0.295 localized exactly to the faulted layer.** That isolates the
  failure definitively — the copy + out-of-band discriminator LOGIC is correct; the cudagraph block is
  specifically inductor's freedom over buffer realization, NOT a detector flaw.

**The one clean test left** (named identically by Add.12 and Add.13): a CUDA `.so` that corrupts `out`'s
**device memory directly**, on-stream after the GEMM — outside inductor's SSA model, because a real
SDC is not a traced op. Not built. It is the single open thread if cheap detection is ever revisited.

---

## D. THE STRATEGIC VERDICT (the real conclusion)

1. **Cost is a dealbreaker for the neocloud buyer, and SDC detection is OPTIONAL for them** — insurance
   against a tail risk. Nobody pays ~52% throughput for optional insurance. A 1%-overhead optional
   feature and a 52%-overhead optional feature share the same fatal flaw: **optional**. Driving the
   overhead from 52% toward single digits would not have changed the sale.

2. **We hardened the wrong readout first.** Detection is the *insurance* readout — structurally the
   hardest thing to sell because the buyer is betting it never fires. The **driver-level position**
   (interpose below the engine, see and actuate what the engine can't) **is intact**. The detector
   being non-sellable says nothing about the position — it says we pointed the position at a
   non-mandatory output.

3. **The detector is the always-on −51.9% validated capability.** It is **closed as "works, not the
   product."** It stays in the bag as a correctness/compliance feature for any buyer who is mandated to
   have SDC coverage (some will be) — but it is not the wedge.

4. **NEXT DIRECTION — point the driver position at NON-OPTIONAL readouts:**
   - **TPW (tokens-per-watt)** via the clock actuator: **+57.28% tok/W measured**, single-GPU
     matched-pairs. Power/thermal is a *mandatory* neocloud constraint (rack power caps, PUE, energy
     cost) — the sell is one sentence.
   - **Multi-tenant multiplexing:** the margin lever and the biggest prize, but the least validated.
   - Neither has been tested against a real buyer. **No pod experiment decides this** — it is a buyer
     conversation (Devang / Nebius / one neocloud), not another benchmark.

5. **Open thread (detection):** the single CUDA `.so` output-memory shim — the only unblocked way to
   validate the out-of-band catch under cudagraph and recover the −8.9%/single-digit cheap detector.
   Revisit only if a mandated-detection buyer appears.

---

## E. DISCIPLINE NOTES (what held this arc)

- **Anchor frozen:** `2edba0d2136f8ede4713d90a8f7cd55f` start==end every session; scratch probes only
  (`inj_*.so`, `*.py`), the runtime `.so` never rebuilt. Re-verified at the close of this doc.
- **Honesty gate held:** three out-of-band catch results were reported as INCONCLUSIVE, not forced to
  a "win" (Add.10/12 + the v2/v3 cudagraph attempts here). The fused-ABFT path was **STOPPED at GATE 1**
  rather than fusing a checksum onto a 3–4× slower base and reporting the combined number.
- **Overclaim caught and retracted:** the in-place out-of-band fault's "non-reorderable, copy captures
  it" comment was wrong; the measured `oob_n64.json` (diverged generation, resid 0.0) exposed it, and
  Add.13 documents the correction with an eager control that isolates the true cause.
- **Prove-or-kill with cheap micro-tests** before full vLLM runs (dual-graph micro, Freivalds/ABFT
  Phase-0, GATE-1 base-perf) — killed three paths before paying for an integration.

---
**Anchor end:** `2edba0d2136f8ede4713d90a8f7cd55f` (re-verified — substrate unchanged this session).
