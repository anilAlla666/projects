// GAP5 disjoint-SM probe kernels. Compiled to cubin via: nvcc -cubin -arch=sm_90a
#include <cstdint>
#include <cuda_fp16.h>

// Spin to give blocks time to spread across all SMs of the partition.
__device__ __forceinline__ void spin(unsigned long long cyc){
  unsigned long long s = clock64();
  while (clock64() - s < cyc) { __threadfence_block(); }
}

// (b) Record %smid per block. out[blockIdx] = smid (thread 0 writes).
extern "C" __global__ void record_smid(unsigned int* out, unsigned long long cyc){
  unsigned int smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  spin(cyc);
  if (threadIdx.x == 0) out[blockIdx.x] = smid;
}

// (3) A/B fault kernel: computes y = a + b for each element, ONE block per element.
//   Records the smid that produced each element into smid_out.
//   Injects a DETERMINISTIC, SM-LOCALIZED fault: if the producing SM == faulty_sm,
//   it flips the fp16 top-exponent bit (bit14) of the result -> |delta|>=2.76 class
//   (matches Step-A harmful fault model). faulty_sm<0 disables injection (clean ref).
extern "C" __global__ void fma_smlocal_fault(
    const __half* a, const __half* b, __half* y, unsigned int* smid_out,
    int faulty_sm, unsigned long long cyc, int n){
  int i = blockIdx.x;                 // one block per output element
  if (i >= n) return;
  unsigned int smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  spin(cyc);                          // spread blocks across the partition
  if (threadIdx.x == 0){
    float fa = __half2float(a[i]);
    float fb = __half2float(b[i]);
    float fy = fa + fb;              // the "GEMM-like" compute
    __half h = __float2half(fy);
    // Deterministic SM-localized corruption: flip fp16 bit14 if on faulty SM.
    if ((int)smid == faulty_sm){
      unsigned short bits = __half_as_ushort(h);
      bits ^= (unsigned short)(1u << 14);   // top-exponent bit -> harmful |delta|
      h = __ushort_as_half(bits);
    }
    y[i] = h;
    smid_out[i] = smid;
  }
}
