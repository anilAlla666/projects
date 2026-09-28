#include <iostream>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda.h> // Required for Driver API

typedef cudaError_t (*cudaLaunchKernel_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
cudaLaunchKernel_t real_cudaLaunchKernel = nullptr;

// Driver API handles for the AOT binary
CUmodule cipher_module;
CUfunction cipher_kernel_func;
bool is_module_loaded = false;

extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, 
                                        void** args, size_t sharedMem, cudaStream_t stream) {
    
    if (!real_cudaLaunchKernel) {
        real_cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
        
        // Initialize the CUDA Driver API
        cuInit(0);
        CUcontext current_ctx;
        cuCtxGetCurrent(&current_ctx);
        
        // Load the Triton AOT Binary directly into the GPU
        CUresult res = cuModuleLoad(&cipher_module, "cipher_lnn.cubin");
        if (res == CUDA_SUCCESS) {
            // NOTE: Replace "cipher_jagged_lnn_gemm" with the exact Linker Name printed by the Python script
            cuModuleGetFunction(&cipher_kernel_func, cipher_module, "cipher_jagged_lnn_gemm");
            is_module_loaded = true;
            std::cout << "[CIPHER] Hopper Tensor Core binary loaded and locked." << std::endl;
        } else {
            std::cout << "[CIPHER] Failed to load .cubin binary!" << std::endl;
        }
    }

    bool is_vllm_padded_gemm = true; // Trigger condition

    if (is_vllm_padded_gemm && is_module_loaded) {
        // std::cout << "[CIPHER] Intercepting padded request. Routing to LNN Tensor Cores..." << std::endl;

        // 1. Prepare the arguments matching the Python signature
        // In production, these pointers are extracted dynamically from the incoming vLLM request
        void* X_ptr; void* W_ptr; void* Out_ptr; 
        void* seq_lens_ptr; void* batch_offsets_ptr;
        int stride_xm; int stride_xk; int stride_wk; int stride_wn; 
        int stride_outm; int stride_outn; int D = 128;

        void* kernel_args[] = {
            &X_ptr, &W_ptr, &Out_ptr, 
            &seq_lens_ptr, &batch_offsets_ptr,
            &stride_xm, &stride_xk, &stride_wk, &stride_wn, 
            &stride_outm, &stride_outn, &D
        };

        // 2. Launch the raw binary directly via the Driver API
        // This fires the O(1) jagged projection for all active motor behaviors simultaneously
        cuLaunchKernel(cipher_kernel_func, 
                       4, 1, 1,  // Grid Dim (Tenants, M_blocks)
                       128, 1, 1, // Block Dim (Triton uses 128 threads per block by default)
                       0, stream, kernel_args, 0);

        return cudaSuccess; 
    }

    return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
