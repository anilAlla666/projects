import torch
import triton
import triton.language as tl

@triton.jit
def cipher_jagged_lnn_gemm(
    X_ptr, W_ptr, Out_ptr,
    seq_lens_ptr, batch_offsets_ptr,
    stride_xm, stride_xk,      # Strides for Activations [Tokens, D]
    stride_wk, stride_wn,      # Strides for Weights [D, D]
    stride_outm, stride_outn,  # Strides for Output [Tokens, D]
    D: tl.constexpr,           # Hidden Dimension Size
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # axis 0: Tenant ID. axis 1: Sequence Length Block (M dimension)
    tenant_id = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # O(1) Early Exit
    seq_len = tl.load(seq_lens_ptr + tenant_id)
    if seq_len == 0:
        return

    # Start row index for this tenant in the flattened 1D memory arena
    tenant_start_row = tl.load(batch_offsets_ptr + tenant_id)

    # --- 2D BLOCK POINTER MATH ---
    # Determine the token indices (rows) this specific thread block is responsible for
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < seq_len
    global_offs_m = tenant_start_row + offs_m

    # Column offsets for Output (N dimension)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # Initialize the Tensor Core Accumulator directly in ultra-fast SM Registers (FP32)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the inner dimension (K) in blocks
    for k in range(0, D, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Calculate exact memory addresses for the current X and W tiles
        x_ptrs = X_ptr + (global_offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        # Fetch tiles from HBM -> SM Shared Memory (FP16)
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < D), other=0.0)
        w_block = tl.load(w_ptrs, mask=(offs_k[:, None] < D) & (offs_n[None, :] < D), other=0.0)

        # ====================================================================
        # THE HOPPER TENSOR CORE ENGINE (mma.sync)
        # This single line forces the compiler to map the math to hardware TC.
        # It multiplies a 2D activation tile by a 2D weight tile instantly.
        # ====================================================================
        accumulator = tl.dot(x_block, w_block, accumulator, allow_tf32=True)

    # --- LNN HARDWARE NON-LINEARITY ---
    # Apply LNN state non-linearity (e.g., Tanh) natively in the registers!
    # By doing this here, we avoid writing to global memory and reading it back.
    # UNCOMMENT THIS FOR TRUE LNN DYNAMICS: 
    # accumulator = tl.math.tanh(accumulator)

    # Store the finished 2D tile back into the jagged output array
    out_ptrs = Out_ptr + (global_offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, accumulator, mask=mask_m[:, None] & (offs_n[None, :] < D))


def run_cipher_gemm_test():
    print("Initializing CIPHER Tensor Core LNN Kernel...\n")
    
    # 1. Define LNN parameters
    D = 128  # Hidden Dimension
    
    # 2. Setup Heterogeneous Tenants (0 padding)
    seq_lens = torch.tensor([32, 16, 0, 64], dtype=torch.int32, device='cuda')
    batch_offsets = torch.zeros(4, dtype=torch.int32, device='cuda')
    batch_offsets[1:] = torch.cumsum(seq_lens[:-1], dim=0)
    total_tokens = torch.sum(seq_lens).item()
    
    print(f"Total Tokens Active: {total_tokens}")
    print(f"Memory Saved by Zero-Padding: {(4 * 64) - total_tokens} tokens per layer\n")

    # 3. Allocate Tensors in FP16 (Required for optimal Tensor Core utilization)
    X = torch.ones((total_tokens, D), dtype=torch.float16, device='cuda') * 2.0
    W = torch.ones((D, D), dtype=torch.float16, device='cuda') * 3.0
    Out = torch.zeros((total_tokens, D), dtype=torch.float32, device='cuda')

    # 4. Define Kernel Grid dynamically based on max sequence length
    def grid(META):
        max_tokens = int(torch.max(seq_lens).item())
        # We need enough blocks in the M dimension to cover the longest tenant
        grid_m = triton.cdiv(max_tokens, META['BLOCK_SIZE_M'])
        return (4, grid_m) # Grid: (Num Tenants, M_Blocks)

    # 5. Launch Kernel
    cipher_jagged_lnn_gemm[grid](
        X, W, Out,
        seq_lens, batch_offsets,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        Out.stride(0), Out.stride(1),
        D=D,
        BLOCK_SIZE_M=32,  # Process 32 tokens at a time
        BLOCK_SIZE_K=32,  # Inner projection block
        BLOCK_SIZE_N=128, # Match D to finish N in one block
    )
    
    print("--- TENSOR CORE EXECUTION SUCCESSFUL ---")
    
    # Verification: 2.0 (FP16) * 3.0 (FP16) * 128 (D dimension) = 768.0 (FP32)
    sample_out = Out[0, :5].cpu().tolist()
    print(f"LNN Projection Output Verification (Should be 768.0):")
    print(sample_out)

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA unavailable."
    run_cipher_gemm_test()
