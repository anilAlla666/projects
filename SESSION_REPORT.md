# CIPHER Session Report — 2026-04-02

## Project Goal
Accelerate Mistral-7B inference by intercepting cuBLAS GEMM calls via LD_PRELOAD
and substituting them with Koopman low-rank approximations. Target: all 32
transformer layers.

---

## What Was Accomplished

### Phase 1: O_proj Koopman Substitution (completed prior sessions)
- Manifold files for all 32 layers exist at `/workspace/manifold/attn/`
  - Each layer has V_layer{i}.npy, K_layer{i}.npy, W_layer{i}.npy
  - Layer 14 W file is **corrupt (0 bytes)** — must be regenerated or skipped
- 9 layers qualify at fit_error < 0.08: [2, 7, 9, 10, 11, 12, 13, 15, 23]
- 9-layer sub result: **81.9 tok/s** vs 74.9 baseline (1.09x)

### Phase 2: Binary Search for Coherence Threshold (this session)

**Goal:** Find maximum layers that produce coherent text.

| Config | tok/s | Coherent? |
|--------|-------|-----------|
| 9 layers (fit < 0.08) | 81.9 | Yes |
| 11 layers [2,7,9,10,11,12,13,15,23,0,4] | 79.7 | Yes |
| **12 layers** [2,7,9,10,11,12,13,15,23,0,4,8] | **81.9** (3-run mean) | **Yes** |
| 13 layers (+ layer 28) | 80.7 | **No** — repetitive loop |
| 13 layers (+ layer 3 instead of 28) | 81.3 | **No** — repetitive loop |
| 13 layers (+ layer 5 instead of 28) | 79.7 | **No** — garbage text |
| 17 layers | — | No — garbage |

**Conclusion:** 12 layers is the ceiling. It's the COUNT not any specific layer.
Speedup plateaus at ~82 tok/s across 9-12 layers (bottleneck shifted elsewhere).

### Phase 3: ASVD Recalibration for 20 Failing Layers (this session)

Attempted activation-SVD (eigendecomposition-based) at rank-16 for the 20 layers
that didn't qualify under original Koopman fitting.

**Result: All failed badly.**

| Layer Range | fit_error range |
|-------------|-----------------|
| 1, 3, 5, 6 (early) | 11.1 – 28.6 |
| 16–26 (mid-late) | 2.5 – 4.8 |
| 27–28 | 1.6 – 1.9 |
| 29, 31 (final) | 0.84 – 0.88 |

Files saved to `/workspace/manifold/attn_asvd/` (copies of good 12 + ASVD for rest).
These are NOT usable. Rank-16 is far too low for these layers.

### Phase 4: Q@K.T Intercept Attempt (this session)

**Goal:** Substitute the attention score computation (Q @ K.T) for all 32 layers.

#### Step 4a: Shape Identification
Identified all GEMM shapes during Mistral-7B forward pass:

**cublasGemmEx (linear projections):**
| Shape (m, n, k) | Count/layer | Operation |
|-----------------|-------------|-----------|
| 4096, seq, 4096 | 2 | q_proj + o_proj |
| 1024, seq, 4096 | 2 | k_proj + v_proj (GQA) |
| 14336, seq, 4096 | 2 | gate_proj + up_proj |
| 4096, seq, 14336 | 1 | down_proj |

**cublasGemmStridedBatchedEx (attention scores — prefill only):**
| Shape | Batch | Operation |
|-------|-------|-----------|
| m=seq, n=seq, k=128 | 32 | Q @ K.T |
| m=128, n=seq, k=seq | 32 | attn_weights @ V |

#### Step 4b: Intercept Implementation
- Added `cublasGemmStridedBatchedEx` shim to `cipher_intercept_cudart.cpp`
- Added PLT-exported alias with `@@libcublas.so.12` versioned symbol
- Added to GOT patch table, exports.map, and merged version script
- Filter: `k==128 && m==n && batchCount==32 && Ctype==2`
- **Verified: all 32 layers intercepted during prefill** (32 calls, layers 0-31)

#### Step 4c: Decode Shape Analysis — CRITICAL FINDING

**During autoregressive decode, Q@K.T does NOT go through cuBLAS at all.**

| Phase | Q@K.T Path | Interceptable? |
|-------|-----------|----------------|
| Prefill | cublasGemmStridedBatchedEx (m=n=seq) | Yes |
| Decode (KV cache) | Fused SDPA kernel (m=1) | **No** |
| Decode (no KV cache) | Fused SDPA kernel | **No** |

PyTorch uses `torch._C._nn.scaled_dot_product_attention` — a fused CUDA kernel
that computes Q@K.T + scale + mask + softmax + attn@V in one pass. This kernel
bypasses cuBLAS entirely. Even `attn_implementation='eager'` falls through to SDPA
because 'eager' is not registered in `ALL_ATTENTION_FUNCTIONS`.

**Decode-time GEMMs that ARE interceptable (all via cublasGemmEx):**
- m=4096, n=1, k=4096: q_proj + o_proj (64/step)
- m=1024, n=1, k=4096: k_proj + v_proj (64/step)
- m=14336, n=1, k=4096: gate_proj + up_proj (64/step)
- m=4096, n=1, k=14336: down_proj (32/step)
- m=32000, n=1, k=4096: lm_head (1/step)
- **Total: ~225 cublasGemmEx calls per decode step**

### Phase 5: Attention Score Rank Calibration (this session)

Collected post-softmax attention weights for all 32 layers, measured rank-1 and
rank-2 approximation error on the last-row attention pattern.

| Layer | Rank-1 Error | Rank-2 Error | Status |
|-------|-------------|-------------|--------|
| 0 | 0.0528 | 0.0000 | R1-OK |
| 1 | 0.0668 | 0.0000 | R1-OK |
| 2 | 0.0331 | 0.0000 | R1-OK |
| 3 | 0.0672 | 0.0000 | R1-OK |
| 4 | 0.0401 | 0.0000 | R1-OK |
| 5 | 0.0675 | 0.0000 | R1-OK |
| 6 | 0.0514 | 0.0000 | R1-OK |
| 7 | 0.0720 | 0.0000 | R2-OK |
| 8 | 0.0968 | 0.0000 | R2-OK |
| 9 | 0.0983 | 0.0000 | R2-OK |
| 10 | 0.1116 | 0.0000 | R2-OK |
| 11 | 0.0829 | 0.0000 | R2-OK |
| 12 | 0.0972 | 0.0000 | R2-OK |
| 13 | 0.1208 | 0.0000 | R2-OK |
| 14 | 0.1059 | 0.0000 | R2-OK |
| 15 | 0.1582 | 0.0000 | R2-OK |
| 16 | 0.1095 | 0.0000 | R2-OK |
| 17 | 0.1379 | 0.0000 | R2-OK |
| 18 | 0.1134 | 0.0000 | R2-OK |
| 19 | 0.1003 | 0.0000 | R2-OK |
| 20 | 0.0896 | 0.0000 | R2-OK |
| 21 | 0.1083 | 0.0000 | R2-OK |
| 22 | 0.0493 | 0.0000 | R1-OK |
| 23 | 0.1012 | 0.0000 | R2-OK |
| 24 | 0.0669 | 0.0000 | R1-OK |
| 25 | 0.0391 | 0.0000 | R1-OK |
| 26 | 0.0740 | 0.0000 | R2-OK |
| 27 | 0.0657 | 0.0000 | R1-OK |
| 28 | 0.0529 | 0.0000 | R1-OK |
| 29 | 0.0795 | 0.0000 | R2-OK |
| 30 | 0.0629 | 0.0000 | R1-OK |
| 31 | 0.1064 | 0.0000 | R2-OK |

**Note:** Rank-2 = 0.000 is an artifact of only 5 unique prompts (×20 repeats).
The SVD of 20 near-identical rows trivially captures all variance at rank-2.
These numbers would be higher with diverse prompts. Rank-1 numbers are realistic.

Files: `/workspace/manifold/qkt/err_r1_layer{0..31}.npy`, `err_r2_layer{0..31}.npy`

---

## Current State of Code

### Build Command (working)
```bash
g++ -shared -fPIC -O2 -std=c++17 \
  -I./include -I/usr/local/cuda/include \
  -Wl,--version-script=/tmp/merged_versions.map \
  -o libcipher_hook.so \
  src/cipher_intercept_cudart.cpp \
  -ldl -lpthread
```

**IMPORTANT:** The merged version script is at `/tmp/merged_versions.map`.
Copy it somewhere permanent before reboot. The original `exports.map` and
`hook_versions.map` cannot be used together (linker error with anonymous
version tags). The merged script chains them: `libcublas.so.12 {} CIPHER_BASE;`.

### DSO Exported Symbols
```
cublasGemmEx@@libcublas.so.12          — linear projection intercept
cublasGemmStridedBatchedEx@@libcublas.so.12  — attention score intercept (prefill)
cublasLtMatmul@@CIPHER_BASE            — matmul intercept (not currently firing)
```

### Source Modifications (cipher_intercept_cudart.cpp)
Changes made this session:
1. **Line ~261:** Added `cublasGemmEx` PLT alias (was missing after rebuild)
2. **Line ~276:** Added `GEMMEX-ENTRY` logging (first 2 calls)
3. **Line ~308:** Added `GEMM-SHAPES-EX` shape logging (cublasGemmEx, dumps at 1500 calls)
4. **Line ~384-460:** Added full `cublasGemmStridedBatchedEx` shim with:
   - Prefill detection: `k==128 && m==n && batchCount==32`
   - Decode detection: `k==128 && m==1 && batchCount==32`
   - Layer tracking via rolling counter mod 32
   - `[CIPHER Q@KT-prefill]` and `[CIPHER Q@KT-decode]` logging
   - `[CIPHER attn@V]` logging for attn_weights @ V
   - PLT-exported alias for GOT patching
5. **Line ~463:** Added `GEMM-SHAPES` logging for cublasLtMatmul path (dumps at 10 calls)
6. **Line ~475:** Added `LT-ENTRY` logging (first 2 calls)
7. **Patch table:** Added `cublasGemmStridedBatchedEx` entry

### Environment
```
Model: mistralai/Mistral-7B-v0.1
HF_HOME: /workspace/.cache/huggingface
Platform: Linux, CUDA GPU
Python: 3.12
PyTorch: with fused SDPA support
```

### Run Commands
```bash
# Baseline test (no substitution)
HF_HOME=/workspace/.cache/huggingface \
python3 -c "..." # 74.9 tok/s

# 12-layer O_proj Koopman test
CIPHER_OBSERVE_ONLY=1 CIPHER_SAFE_MODE=1 \
HF_HOME=/workspace/.cache/huggingface \
LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so" \
python3 -c "..." # 81.9 tok/s

# Shape logging
CIPHER_LOG_SHAPES=1 CIPHER_OBSERVE_ONLY=1 CIPHER_SAFE_MODE=1 \
HF_HOME=/workspace/.cache/huggingface \
LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so" \
python3 -c "..."
```

---

## What the User Wants Next

The user's goal is to **intercept all 32 layers** for maximum speedup.
The Q@K.T attention path is blocked (fused SDPA). The pivot is:

### Target: All 7 Linear Projections × 32 Layers via cublasGemmEx

During decode, 225 cublasGemmEx calls per step. Currently only 12 o_proj
calls are substituted. The breakdown per layer:

| Projection | Shape (m×k) | Params | FLOPs weight |
|-----------|-------------|--------|-------------|
| q_proj | 4096×4096 | 16M | 7.7% |
| k_proj | 1024×4096 | 4M | 1.9% |
| v_proj | 1024×4096 | 4M | 1.9% |
| o_proj | 4096×4096 | 16M | 7.7% |
| **gate_proj** | **14336×4096** | **58M** | **27.9%** |
| **up_proj** | **14336×4096** | **58M** | **27.9%** |
| **down_proj** | **4096×14336** | **58M** | **27.9%** |

**FFN is 4.3× more compute than attention projections.**
FFN = 174M params/layer, Attention = 40M params/layer.

### Proposed Next Steps (user's direction)
1. **Calibrate Koopman/ASVD for all 7 projections at various ranks** (r=8,16,32,64,128,256)
   - Was about to start this when session was interrupted
   - Need to find the minimum rank per projection type that gives fit_error < 0.08
2. **FFN first** — gate_proj + up_proj + down_proj cover 83.7% of per-layer FLOPs
3. **Register weight pointers** for each qualifying projection in the cublasGemmEx shim
4. **Coherence test** at each expansion step (same binary search approach)

### Key Technical Constraints
- 12 layers is the coherence ceiling for rank-16 o_proj substitution
- The 20 failing o_proj layers need higher rank (ASVD at r=16 gave errors 0.84-28.5)
- FFN projections are larger matrices — may need r=64+ for acceptable error
- The cublasGemmEx pointer-matching infrastructure already works for any projection
  (just register the weight pointer via `cipher_koopman_register_weight`)
- `libcipher_rt.so` was NOT rebuilt this session — it contains `cipher_koopman_fp16_launch_layer`
  which does the actual V.T @ K @ W substitution. To handle different projection
  shapes (14336-dim FFN), the runtime may need updates.

---

## File Inventory

### Manifold Data
```
/workspace/manifold/attn/          — 96 files (V,K,W × 32 layers, layer14 W corrupt)
/workspace/manifold/attn_asvd/     — 96 files (copies of good 12 + ASVD for rest, NOT usable)
/workspace/manifold/qkt/           — 64 files (err_r1 + err_r2 × 32 layers)
```

### Source & Build
```
/content/CIPHER/src/cipher_intercept_cudart.cpp  — 1352 lines, main hook source
/content/CIPHER/libcipher_hook.so                — rebuilt this session (43KB)
/content/CIPHER/libcipher_rt.so                  — NOT rebuilt (187KB, from prior session)
/content/CIPHER/exports.map                      — updated (added cublasGemmStridedBatchedEx)
/content/CIPHER/hook_versions.map                — original (NOT updated)
/tmp/merged_versions.map                         — WORKING version script (SAVE THIS)
```

### Critical: Save Before Reboot
```bash
cp /tmp/merged_versions.map /content/CIPHER/merged_versions.map
```
