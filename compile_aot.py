import triton
import triton.compiler as tc
import triton.language as tl  # <-- The missing import!
import torch

# The complete, validated jagged LNN kernel
@triton.jit
def cipher_jagged_lnn_gemm(
    X_ptr, W_ptr, Out_ptr,
    seq_lens_ptr, batch_offsets_ptr,
    stride_xm, stride_xk,      
    stride_wk, stride_wn,      
    stride_outm, stride_outn,  
    D,           
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tenant_id = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    seq_len = tl.load(seq_lens_ptr + tenant_id)
    if seq_len == 0:
        return

    tenant_start_row = tl.load(batch_offsets_ptr + tenant_id)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < seq_len
    global_offs_m = tenant_start_row + offs_m
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, D, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = X_ptr + (global_offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < D), other=0.0)
        w_block = tl.load(w_ptrs, mask=(offs_k[:, None] < D) & (offs_n[None, :] < D), other=0.0)
        
        accumulator = tl.dot(x_block, w_block, accumulator, allow_tf32=True)

    out_ptrs = Out_ptr + (global_offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, accumulator, mask=mask_m[:, None] & (offs_n[None, :] < D))


if __name__ == "__main__":
    print("Initializing Triton AOT Compiler for Hopper (sm_90)...")
    
    # FIX: Triton 3.6.0 requires a dictionary for the signature
    signature = {
        "X_ptr": "*fp16",
        "W_ptr": "*fp16",
        "Out_ptr": "*fp32",
        "seq_lens_ptr": "*i32",
        "batch_offsets_ptr": "*i32",
        "stride_xm": "i32",
        "stride_xk": "i32",
        "stride_wk": "i32",
        "stride_wn": "i32",
        "stride_outm": "i32",
        "stride_outn": "i32",
        "D": "i32"
    }
    
    constants = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 32, "BLOCK_SIZE_N": 128}
    
    src = tc.ASTSource(fn=cipher_jagged_lnn_gemm, signature=signature, constexprs=constants)
    compiled_kernel = triton.compile(src, target=("cuda", "90"))
    
    cubin_path = "cipher_lnn.cubin"
    with open(cubin_path, "wb") as f:
        f.write(compiled_kernel.cubin)
        
    print(f"Compilation successful! Binary saved to: {cubin_path}")
    print(f"C++ Linker Function Name: {compiled_kernel.name}")
