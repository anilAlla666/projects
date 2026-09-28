// PATH-1 in-graph check kernels (compiled to cubin, loaded via driver API -> version-safe across nvcc/torch-cudart).
// vLLM linear: C[out,tokens] = W[out,K] @ x[K,tokens] (cuBLAS col-major; A=W m=out k=K, B=x k=K n=tokens, C ldc=out).
// ABFT row(over-out)-sum identity: sum_out C[out,t] == sum_k x[k,t] * wcol[k], wcol[k]=sum_out W[out,k].
#include <cuda_fp16.h>

extern "C" __global__ void colsum_W(const __half* A, float* wcol, int m, int K){
  // wcol[k] = sum_{out=0..m-1} A[out + k*m]   (one thread per k)
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  if(k>=K) return;
  float s=0.f; const __half* col=A + (long)k*m;
  for(int o=0;o<m;o++) s += __half2float(col[o]);
  wcol[k]=s;
}

extern "C" __global__ void abft_check(const __half* C, const __half* B, const float* wcol,
                                      int m, int n, int K, float* flag){
  // one block per token t; compute s[t]=sum_out C[out+t*m], r[t]=sum_k B[k+t*K]*wcol[k]; atomicMax |s-r| into flag
  int t = blockIdx.x; if(t>=n) return;
  __shared__ float red[256];
  const __half* cc = C + (long)t*m;
  const __half* xx = B + (long)t*K;
  float s=0.f; for(int o=threadIdx.x;o<m;o+=blockDim.x) s += __half2float(cc[o]);
  float r=0.f; for(int k=threadIdx.x;k<K;k+=blockDim.x) r += __half2float(xx[k])*wcol[k];
  float local = s - r;
  // block-reduce sum of s and r separately would be exact; here reduce (s-r) partials then |.|
  red[threadIdx.x]=local; __syncthreads();
  for(int off=blockDim.x/2; off>0; off>>=1){ if(threadIdx.x<off) red[threadIdx.x]+=red[threadIdx.x+off]; __syncthreads(); }
  if(threadIdx.x==0){
    float resid = fabsf(red[0]);
    // atomicMax on float via int bits (residual >=0)
    int* fi=(int*)flag; int old=*fi, assumed;
    int v=__float_as_int(resid);
    do{ assumed=old; if(__int_as_float(assumed)>=resid) break; old=atomicCAS(fi,assumed,v);}while(assumed!=old);
  }
}

extern "C" __global__ void inject_flip(__half* C){
  // persistent SDC: flip bit-14 of C[0] every replay (one thread)
  if(blockIdx.x==0 && threadIdx.x==0){
    unsigned short* p=(unsigned short*)C; *p ^= (unsigned short)(1u<<14);
  }
}
