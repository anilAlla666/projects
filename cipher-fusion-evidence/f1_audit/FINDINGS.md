# F1 root-cause investigation — findings log

Method: bisect by isolation until the line producing divergent state is found.
Anchors: c2c5d313 (confirmed-broken), dc804eb3 (CP 5.3 STEP 2 rebuild).

---

## Finding 1 — single Marlin GEMM is degenerate (Step 1 → Step 3)

**Repro:** `f1_min_repro.cpp` — loads Mistral-7B layer-0 `q_proj` (4096×4096,
raw FP16, `dump_weight.py`), quant+repack+ONE `cipher_rt_marlin_engine_dispatch`
via the public engine API, compares against a cuBLAS FP16 GEMM (`CUBLAS_COMPUTE_32F`)
on the **identical** weight buffer and activation. No green ctx → full-GPU
`grid=132` + `PrimaryCtxGuard` — the exact F1-broken path.

**Run:** `c2c5d313`, M=1, N=4096, K=4096.

| metric | value |
|---|---|
| dispatch rc | 0 (no crash, no hang) |
| marlin NaN/Inf | 0 |
| marlin exact-zero | 0 / 4096 |
| max_abs_error | 3.81 |
| max \|ref\| | 3.98 |
| **rel error** | **95.7%** |

Output does not correlate with the reference at all (e.g. out[2]: marlin
+0.365 vs ref −0.001; out[14]: marlin −0.132 vs ref +0.371). This is not
quantization noise (which tracks the reference within a few %) — it is fully
wrong output.

**Verdict:** the bug reproduces in a **single GEMM**. Per method Step 1, the
bug is in **quant, repack, or the kernel** — NOT in the dispatch loop, stream
handling, context management, or vLLM integration. → proceed to Step 3.
