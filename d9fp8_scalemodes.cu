// Which FP8 scale modes does this cu13 cuBLASLt support on H100? Decisive for fused-vs-epilogue.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#ifndef CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
#define CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F 3
#endif
#ifndef CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
#define CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F 4
#endif
int main(){
    int m=14336,n=8192,k=4096;
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    void* dummy; cudaMalloc(&dummy, (size_t)m*n*4);
    void* dscale; cudaMalloc(&dscale, (size_t)16384*4);
    size_t wsb=32u<<20;
    struct M{int v;const char*n;} modes[]={{0,"SCALAR_32F"},{1,"VEC16_UE4M3"},{2,"VEC32_UE8M0"},{3,"OUTER_VEC_32F"},{4,"VEC128_32F"}};
    int outtypes[]={CUDA_R_16F,CUDA_R_16BF,CUDA_R_32F}; const char* on[]={"f16","bf16","f32"};
    for(int fa=0;fa<2;fa++){
      for(auto&mo:modes){
        for(int oi=0;oi<3;oi++){
            cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d,CUBLAS_COMPUTE_32F,CUDA_R_32F);
            cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta));
            cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb));
            int8_t f=fa; cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&f,sizeof(f));
            void* p=dscale; int32_t md=mo.v;
            cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&p,sizeof(p));
            cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&p,sizeof(p));
            cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&md,sizeof(md));
            cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&md,sizeof(md));
            cublasLtMatrixLayout_t la,lb,ld;
            cublasLtMatrixLayoutCreate(&la,CUDA_R_8F_E4M3,k,m,k);
            cublasLtMatrixLayoutCreate(&lb,CUDA_R_8F_E4M3,k,n,k);
            cublasLtMatrixLayoutCreate(&ld,(cudaDataType_t)outtypes[oi],m,n,m);
            cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsb,sizeof(wsb));
            cublasLtMatmulHeuristicResult_t r{}; int got=0;
            cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,d,la,lb,ld,ld,pref,1,&r,&got);
            printf("fast_accum=%d scale=%-14s out=%-4s -> %s (hs=%d got=%d)\n",
                   fa,mo.n,on[oi],(hs==CUBLAS_STATUS_SUCCESS&&got>0)?"SUPPORTED":"no",(int)hs,got);
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(la);cublasLtMatrixLayoutDestroy(lb);cublasLtMatrixLayoutDestroy(ld);
            cublasLtMatmulDescDestroy(d);
        }
      }
    }
    return 0;
}
