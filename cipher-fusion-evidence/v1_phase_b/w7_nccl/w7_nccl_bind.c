/* W.7 item 1: confirm NCCL loads + binds the tuner plugin, and which ABI version.
 * 1-rank comm is valid on a single GPU (no duplicate-device); NCCL dlopens the
 * tuner at CommInit. 1-rank AllReduce is identity -> trivially bit-identical
 * (degenerate item-3 data-path-not-corrupted check). */
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
int main(void){
    cudaSetDevice(0);
    ncclComm_t comm; ncclUniqueId id; ncclGetUniqueId(&id);
    ncclResult_t r = ncclCommInitRank(&comm, 1, id, 0);
    printf("CommInitRank rc=%d (%s)\n", r, ncclGetErrorString(r));
    float *buf, host[16]; for(int i=0;i<16;i++) host[i]=2.0f;
    cudaMalloc(&buf, sizeof host); cudaMemcpy(buf, host, sizeof host, cudaMemcpyHostToDevice);
    cudaStream_t s; cudaStreamCreate(&s);
    r = ncclAllReduce(buf, buf, 16, ncclFloat, ncclSum, comm, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(host, buf, sizeof host, cudaMemcpyDeviceToHost);
    printf("AllReduce rc=%d result[0]=%.1f\n", r, host[0]);
    ncclCommDestroy(comm);
    return r!=ncclSuccess;
}
