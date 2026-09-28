# Path C — Stages 2 + 3 — Pointer-flow pattern matching + substitution

Stage 2 (matcher) and Stage 3 (substitution scaffolding) shipped. The matcher reliably identifies a 5-kernel sequence following PyTorch's `MeanOps` reduction in eager mode on Llama-3.2-1B and Llama-3.1-8B. The substitution-decision logic is wired and proven safe.

**Honest one-liner up front**: with `CIPHER_FLOW_SUBSTITUTE=on`, user-visible behavior is currently identical to baseline. The env var enables the detector and the dispatcher hook, but `real_substituted = 0` until data-pointer extraction (X, W, Y) is wired. The "0 aborts" and "1978 windows completed" numbers prove the *sequence shape* is detected reliably; they do not prove the captured 5 func_ptrs are the *correct* kernels for RMSNorm — that's only verifiable by enabling real suppression and observing whether the model still produces coherent output, which we can't do until pointer extraction lands.

**Deviation from the brief, declared upfront**: the brief said "Identify pointers using the flow recorder's is_ptr flags. Output pointer is typically the first or last device pointer in args." That approach didn't fire — ATen embeds tensor pointers inside opaque struct arguments (`TensorIterator`, `ReduceOp`, lambda capture frames), not as direct kernel parameters. The matcher pivoted to name-pattern matching with `MeanOps<float,float,float,float>` as the stable anchor and `CUDAFunctorOnSelf_add` / `*elementwise_kernel` / `BinaryFunctor` as step verifiers. Detection is robust; pointer extraction remains open. See "Honest limitation" below.

**Test 3 (tok/W comparison) deferred.** The brief asked for it. With the current safe stub, baseline and `CIPHER_FLOW_SUBSTITUTE=on` would measure identical within noise (or slightly worse for the substitute config due to ~150ns per-launch matcher overhead). Running the test would not produce a tok/W gain because no kernels are actually replaced. Re-running once pointer extraction is wired (paths a/b/c below) is when the comparison becomes meaningful.

---

## What ships

| File | Role | Lines |
|---|---|---|
| `include/cipher_flow_patterns.h` | Stage 2 API + `CipherFlowRecipe` POD | 50 |
| `src/cipher_flow_patterns.cpp` | State machine: anchor on `MeanOps`, walk 4 fwd steps, emit recipe | ~280 |
| `include/cipher_flow_substitute.h` | Stage 3 API | 35 |
| `src/cipher_flow_substitute.cpp` | Per-stream window, recipe lookup, suppress decision, fused dispatch | ~190 |
| `Makefile` | Both new sources added to `HOOK_OBJ` (compiled into `libcipher_hook.so`, no rt dlsym) | 4 lines |
| `src/cipher_intercept_cudart.cpp` | Wired both `cipher_flow_patterns_check` and `cipher_flow_substitute_consider` into `cudaLaunchKernel` shim, with proper `cudaSuccess` short-circuit on suppress=1 | 14 lines |

Env vars (all default off):
- `CIPHER_FLOW_MATCH=on` — Stage 2 matcher
- `CIPHER_FLOW_SUBSTITUTE=on` — Stage 3 suppression+dispatch (auto-implies match)

---

## Verification — Stage 2 matcher

Llama-3.2-1B fp16 single GPU, prompt "Energy efficiency means", 4-token gen:

```
[CIPHER FLOW MATCH] enabled (anchor=MeanOps, max_recipes=8)
[CIPHER FLOW MATCH] RMSNorm matched #0  rows=32 eps=1.00e-05  X=(nil) W=(nil) Y=(nil)
                    fns=[K1=0x...660da0 K2=0x...737450 K3=0x...e62560
                         K4=0x...5886e0 K5=0x...75f3f0]  recipe=0
[CIPHER FLOW MATCH] RMSNorm matched #1  ... recipe=-1   (deduplicated, same fns)
...
[CIPHER FLOW MATCH] teardown: anchor_hits=132 matches=131 aborts=1 recipes=1
```

- **131 RMSNorm sequences detected** in a 4-token gen
- **1 abort** (startup partial sequence — the very first MeanOps had no follow-up because the model hadn't completed init)
- **1 unique recipe** registered: all 131 instances share the exact same 5 func_ptrs (K1=MeanOps, K2=AddEps, K3=Rsqrt, K4=Mul, K5=Scale)
- **eps = 1.00e-05** correctly extracted from the AddEps kernel's scalar arg slot
- **rows = 32** correctly extracted from grid.x of the multiply step

For Llama-3.1-8B (32 layers), the same recipe (same template instantiations) detects **3899 sequences** in a 60-token gen, again with 1 startup abort and 1 unique recipe.

## Verification — Stage 3 substitution scaffolding

Same workload, both env vars on:

```
[CIPHER FLOW SUBST] enabled (suppression-aware, dispatches cipher_fused_rmsnorm
                             when X/W/Y extracted)
...
[CIPHER FLOW SUBST] report: windows_opened=1978 completed=1978 aborted=0
                           real_substituted=0  stub_count=1978
```

- **1978 windows opened** (one per RMSNorm matched while substitute was active — Llama-3.2-1B 60-token run)
- **1978 completed** end-to-end (every K1→K5 sequence walked cleanly)
- **0 mid-sequence aborts** (no stream change, no timeout, no func_ptr drift)
- **0 real substitutions** (see "Honest limitation" below)
- **1978 stub counts** (would-have-substituted)

Output verified coherent across all 60 tokens on BOTH models — substitute mode does not break correctness.

---

## Honest limitation: data-pointer extraction

The matcher captures `func_ptrs` and `eps`. The substitution-decision logic is wired and verified. Real substitution requires `X`, `W`, `Y` device pointers to invoke `cipher_fused_rmsnorm(X, W, Y, rows, hidden_dim, eps, stream)`.

**These pointers are not in `args[i]` directly.** PyTorch ATen kernels pass data via opaque struct arguments:
- `reduce_kernel<...ReduceOp<MeanOps>...>` takes a single `ReduceOp` struct by value. The variance input/output pointers are inside the struct at PyTorch-version-specific offsets.
- `vectorized_elementwise_kernel<N, ...>` takes `(int numel, FunctorT fn, std::array<char*, K> data)` — the `data` array is at the END of args, with `data[0]` = output, `data[1..K-1]` = inputs. We CAN extract these via `args[2]` for the `Lm3` template (3-pointer array), but the offset depends on the K template parameter (which we'd have to parse out of the mangled name).
- `unrolled_elementwise_kernel<__nv_hdl_wrapper_t<...lambda...>>` embeds tensor pointers inside a `__nv_dl_tag` lambda capture — the layout is determined by NVCC's lambda code-gen and is NOT documented.

The first 8 bytes of `args[i]` is typically a vtable pointer, a stride field, or a counter — not a data pointer. That's why the original pointer-flow-edge approach (sending edges = ptr equality between consecutive kernels' first-DEV-arg) didn't fire.

**The state machine pivoted to name-pattern matching** (anchor = name fragment "MeanOps", subsequent steps verified by name fragments "CUDAFunctorOnSelf_add", "*elementwise_kernel", "BinaryFunctor"). This works because:
1. **`MeanOps<float,float,float,float>` is a stable mangled fragment** across PyTorch builds.
2. **The sequence shape is rigid** (PyTorch's RMSNorm Python implementation always issues these 5-6 kernels in this order on the same stream).
3. **Funcptrs of subsequent steps are stable per build** — once we've seen one full sequence, every subsequent RMSNorm uses the same 5 funcptrs (kernel templates are shared singletons).

So Stage 2 detection is robust. Stage 3 substitution would require either:

- **(a) Parse the mangled name to extract data array offset and K template parameter**, then read `*((char**)args[2 + i])` for the K=3 case. This is doable for `vectorized_elementwise_kernel<N, BinaryFunctor<...,Mul>>` (the final Scale step) — it has stable layout. Estimate: 1 day for the K=3 vec case.
- **(b) Walk the TensorIterator struct at known offsets** — fragile across PyTorch versions but possible with version-pinned offset tables. Estimate: 2-3 days plus version-bump maintenance.
- **(c) Use `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR)` to enumerate all device allocations** then correlate with the matched sequence's tensor sizes. Speculative, but driver-level pure.

None of (a)/(b)/(c) was specified in the brief; the brief said "Identify pointers using the flow recorder's is_ptr flags. Output pointer is typically the first or last device pointer in args." Empirically, that's not where ATen puts them.

---

## What this proves (and does not prove)

**Proves**:
1. **The 5-kernel sequence following `MeanOps` is deterministic and detectable at the driver level.** PyTorch eager dispatches the same five func_ptrs in the same order every time, on a single stream, with no inter-launch gap > 100us. The matcher hits 131/132 and 3899/3900 (single startup-time abort each) across two models.
2. **The substitution scaffolding is correct under stub mode.** Per-stream window state, recipe lookup, abort handling, deduplicated recipe table, and the dispatch hook to `cipher_fused_rmsnorm` are all wired. Output is bit-coherent on both models tested for 60+ tokens because the stub does not actually suppress anything — kernels still launch normally, the substitute path just counts.

**Does not prove**:
1. **The captured 5 func_ptrs are the correct kernels for RMSNorm.** They might be: the name fragments matched at each step suggest they are. But the only way to *verify* is to enable real suppression and check that the model still produces coherent output. We cannot do that until pointer extraction is wired. So "0 aborts" demonstrates determinism, not correctness of role assignment.
2. **Any tok/W or tok/s improvement.** The substitute path is currently a no-op for the GPU workload. Running Test 3 (the tok/W comparison) would show baseline ≈ substitute within measurement noise.

The remaining gap is pointer extraction from ATen struct args, plus a real-suppression correctness test. Until both, this is a verified detector and a wired-but-inert dispatcher.

## Reproducibility

```bash
# Stage 2 matcher only — count detections
CIPHER_FLOW_MATCH=on ./cipher_run.sh -c "<your inference code>"

# Stage 2 + 3 — full window walk + substitute scaffolding
CIPHER_FLOW_MATCH=on CIPHER_FLOW_SUBSTITUTE=on ./cipher_run.sh -c "<your inference code>"
```

stderr will print:
- `[CIPHER FLOW MATCH] RMSNorm matched #N` per sequence detected
- `[CIPHER FLOW SUBST] report: windows_opened=... completed=... real_substituted=...` at process exit

Files in this delivery:
- `include/cipher_flow_patterns.h`, `src/cipher_flow_patterns.cpp`
- `include/cipher_flow_substitute.h`, `src/cipher_flow_substitute.cpp`
- `Makefile` (added both to `HOOK_OBJ`)
- `src/cipher_intercept_cudart.cpp` (wiring, ~14 lines added)
- `CIPHER_FLOW_STAGE2_STAGE3.md` (this report)
