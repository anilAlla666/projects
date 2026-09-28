#include <iostream>

// A completely empty GPU kernel
__global__ void dummy_kernel() {
    // Does nothing, just proves we can reach the silicon
}

int main() {
    std::cout << "Launching native CUDA kernel..." << std::endl;
    
    // This explicitly triggers cudaLaunchKernel
    dummy_kernel<<<1, 1>>>(); 
    
    // Wait for it to finish
    cudaDeviceSynchronize(); 
    
    std::cout << "Kernel execution complete." << std::endl;
    return 0;
}
