// Standalone engine test — isolates the FP8 engine from torch/injection. Clean gdb.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
extern "C" {
int cipher_rt_fp8_engine_init(void);
int cipher_rt_fp8_engine_observe_weight(const void*,int,int);
int cipher_rt_fp8_engine_is_ready(const void*);
int cipher_rt_fp8_engine_quantize_weight(const void*,int,int,int,void*);
int cipher_rt_fp8_engine_matmul(const void*,const void*,void*,int,int,int,int,void*);
unsigned long cipher_rt_fp8_engine_weights_count(void);
}
int main(int argc,char**argv){
    int m = argc>1?atoi(argv[1]):14336;   // out_features
    int n = argc>2?atoi(argv[2]):8192;    // batch
    int k = argc>3?atoi(argv[3]):4096;    // in_features
    cudaSetDevice(0); cudaFree(0);
    void *W=0,*A=0,*C=0;
    cudaMalloc(&W,(size_t)m*k*2); cudaMalloc(&A,(size_t)n*k*2); cudaMalloc(&C,(size_t)m*n*2);
    cudaMemset(W,0x3c,(size_t)m*k*2); cudaMemset(A,0x3c,(size_t)n*k*2); // 0x3c3c ~ bf16 ~0.011
    printf("init=%d\n", cipher_rt_fp8_engine_init()); fflush(stdout);
    printf("obs1=%d\n", cipher_rt_fp8_engine_observe_weight(W,k,m)); fflush(stdout);
    printf("obs2=%d\n", cipher_rt_fp8_engine_observe_weight(W,k,m)); fflush(stdout);
    printf("qweight(bf16)=%d\n", cipher_rt_fp8_engine_quantize_weight(W,14,k,m,0)); fflush(stdout);
    printf("ready=%d count=%lu\n", cipher_rt_fp8_engine_is_ready(W), cipher_rt_fp8_engine_weights_count()); fflush(stdout);
    printf("matmul=%d\n", cipher_rt_fp8_engine_matmul(W,A,C,14,m,n,k,0)); fflush(stdout);
    cudaError_t e=cudaDeviceSynchronize();
    printf("sync=%s DONE\n", cudaGetErrorString(e));
    return 0;
}
