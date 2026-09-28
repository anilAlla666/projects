// gemm_bench.cu — CP 3.3 gate (d): PM-sampling workload-overhead probe.
//
// Flat-out cuBLAS fp16 tensor GEMM for 15 s; prints achieved GEMM count
// and TFLOP/s. Run once with cipher_flopd stopped and once with it
// running; the delta is the cost PM Sampling imposes on the workload.
// (The kmod kprobe hot path is unrelated and untouched by CP 3.3 — that
// is verified separately by md5 of cipher_probe.c.)
//
// Build: nvcc gemm_bench.cu -lcublas -Wno-deprecated-gpu-targets -o gemm_bench
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main() {
    const int D = 8192;
    const double FLOPS_PER_GEMM = 2.0 * (double)D * D * D;
    cudaSetDevice(0);
    cublasHandle_t h;
    cublasCreate(&h);
    size_t n = (size_t)D * D;
    __half *A, *B, *C;
    cudaMalloc(&A, n * sizeof(__half));
    cudaMalloc(&B, n * sizeof(__half));
    cudaMalloc(&C, n * sizeof(__half));
    cudaMemset(A, 1, n * sizeof(__half));
    cudaMemset(B, 1, n * sizeof(__half));
    const float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < 10; i++)
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, D, D,
                     &alpha, A, CUDA_R_16F, D, B, CUDA_R_16F, D,
                     &beta, C, CUDA_R_16F, D,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();

    uint64_t count = 0;
    double t0 = now_s();
    while (now_s() - t0 < 15.0) {
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, D, D,
                     &alpha, A, CUDA_R_16F, D, B, CUDA_R_16F, D,
                     &beta, C, CUDA_R_16F, D,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        count++;
        if ((count & 0x1f) == 0) cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    double el = now_s() - t0;
    printf("gemms=%llu elapsed=%.3f tflops=%.2f\n",
           (unsigned long long)count, el,
           count * FLOPS_PER_GEMM / el / 1e12);
    cudaFree(A); cudaFree(B); cudaFree(C);
    cublasDestroy(h);
    return 0;
}
