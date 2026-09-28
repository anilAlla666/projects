// BGRAD probe: does cuBLASLt BGRADA/BGRADB give an fp32 sum-over-N (length-M) reduction
// of the GEMM output, on the real cuBLAS path? Discriminating test per advisor.
//
// Convention: everything COLUMN-MAJOR (cuBLASLt native). We compute
//   D[M,N] = A[M,K] * B[K,N]   (opA=N, opB=N), fp16 in, fp16 out, fp32 compute.
// Then read the BGRAD bias-gradient vector and compare to:
//   rowsum[m] = sum_n D[m,n]   (length M)   <-- the checksum we want
//   colsum[n] = sum_m D[m,n]   (length N)
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define LK(x) do{ cublasStatus_t s=(x); if(s){printf("cuBLASLt err %s:%d status=%d\n",__FILE__,__LINE__,(int)s);} }while(0)

// column-major index
static inline int cm(int r,int c,int ld){ return c*ld + r; }

struct Res { bool supported; double max_abs_vs_rowsum; double max_abs_vs_colsum; int aux_len_used; };

// run one matmul with a given epilogue, write bias-grad to fp32 d_bias (len = max(M,N))
Res run(cublasLtHandle_t lt, cublasLtEpilogue_t epi, const char* name,
        int M,int K,int N, const __half* dA, const __half* dB, __half* dD,
        const std::vector<float>& rowsum, const std::vector<float>& colsum,
        void* dWork, size_t workSz)
{
    Res r{false,0,0,0};
    cublasLtMatmulDesc_t desc=nullptr;
    LK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opN = CUBLAS_OP_N;
    LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&opN,sizeof(opN)));
    LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN)));
    LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&epi,sizeof(epi)));
    // bias-gradient output buffer (fp32)
    float* dBias=nullptr; CK(cudaMalloc(&dBias, sizeof(float)*(size_t)(M>N?M:N)));
    CK(cudaMemset(dBias,0,sizeof(float)*(size_t)(M>N?M:N)));
    LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&dBias,sizeof(dBias)));
    cudaDataType_t f32=CUDA_R_32F;
    LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,&f32,sizeof(f32)));

    cublasLtMatrixLayout_t lA,lB,lD;
    LK(cublasLtMatrixLayoutCreate(&lA,CUDA_R_16F,M,K,M)); // col-major MxK ld=M
    LK(cublasLtMatrixLayoutCreate(&lB,CUDA_R_16F,K,N,K));
    LK(cublasLtMatrixLayoutCreate(&lD,CUDA_R_16F,M,N,M));

    cublasLtMatmulPreference_t pref=nullptr;
    LK(cublasLtMatmulPreferenceCreate(&pref));
    LK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workSz,sizeof(workSz)));
    cublasLtMatmulHeuristicResult_t heur[1]; int got=0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt,desc,lA,lB,lD,lD,pref,1,heur,&got);
    if(hs!=CUBLAS_STATUS_SUCCESS || got==0){
        printf("  [%s] heuristic: NO algo (status=%d got=%d) -> UNSUPPORTED for this shape/epilogue\n",name,(int)hs,got);
        cudaFree(dBias); cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(lA);cublasLtMatrixLayoutDestroy(lB);cublasLtMatrixLayoutDestroy(lD);
        cublasLtMatmulPreferenceDestroy(pref);
        return r;
    }
    float alpha=1.f, beta=0.f;
    cublasStatus_t ms = cublasLtMatmul(lt,desc,&alpha,dA,lA,dB,lB,&beta,dD,lD,dD,lD,&heur[0].algo,dWork,workSz,0);
    CK(cudaDeviceSynchronize());
    if(ms!=CUBLAS_STATUS_SUCCESS){
        printf("  [%s] matmul FAILED status=%d -> UNSUPPORTED\n",name,(int)ms);
        cudaFree(dBias); cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(lA);cublasLtMatrixLayoutDestroy(lB);cublasLtMatrixLayoutDestroy(lD);
        cublasLtMatmulPreferenceDestroy(pref);
        return r;
    }
    r.supported=true;
    // read bias-grad (try length M and length N)
    std::vector<float> bM(M), bN(N);
    CK(cudaMemcpy(bM.data(),dBias,sizeof(float)*M,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bN.data(),dBias,sizeof(float)*N,cudaMemcpyDeviceToHost));
    // compare first M entries to rowsum (len M); first N entries to colsum (len N)
    double mr=0; for(int m=0;m<M;m++){ double d=fabs((double)bM[m]-(double)rowsum[m]); if(d>mr)mr=d; }
    double mc=0; for(int n=0;n<N;n++){ double d=fabs((double)bN[n]-(double)colsum[n]); if(d>mc)mc=d; }
    r.max_abs_vs_rowsum=mr; r.max_abs_vs_colsum=mc;
    printf("  [%s] SUPPORTED. max|aux-rowsum(overN,lenM)|=%.4g   max|aux-colsum(overM,lenN)|=%.4g\n",name,mr,mc);
    printf("       sample aux[0..3]=%.4f %.4f %.4f %.4f  rowsum[0..3]=%.4f %.4f %.4f %.4f  colsum[0..3]=%.4f %.4f %.4f %.4f\n",
       bM[0],M>1?bM[1]:0,M>2?bM[2]:0,M>3?bM[3]:0,
       rowsum[0],M>1?rowsum[1]:0,M>2?rowsum[2]:0,M>3?rowsum[3]:0,
       colsum[0],N>1?colsum[1]:0,N>2?colsum[2]:0,N>3?colsum[3]:0);
    cudaFree(dBias); cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(lA);cublasLtMatrixLayoutDestroy(lB);cublasLtMatrixLayoutDestroy(lD);
    cublasLtMatmulPreferenceDestroy(pref);
    return r;
}

void test_shape(cublasLtHandle_t lt,int M,int K,int N,void*dWork,size_t workSz,bool verify_D){
    printf("\n==== shape M=%d K=%d N=%d ====\n",M,K,N);
    size_t nA=(size_t)M*K, nB=(size_t)K*N, nD=(size_t)M*N;
    std::vector<float> hA(nA),hB(nB);
    // small deterministic-ish values to avoid fp16 overflow; column-major
    for(size_t i=0;i<nA;i++) hA[i]= ( (int)(i*1103515245u>>16)%7 -3)*0.1f;
    for(size_t i=0;i<nB;i++) hB[i]= ( (int)(i*12345u>>8)%5 -2)*0.1f;
    std::vector<__half> hAh(nA),hBh(nB);
    for(size_t i=0;i<nA;i++) hAh[i]=__float2half(hA[i]);
    for(size_t i=0;i<nB;i++) hBh[i]=__float2half(hB[i]);
    __half *dA,*dB,*dD; CK(cudaMalloc(&dA,nA*2));CK(cudaMalloc(&dB,nB*2));CK(cudaMalloc(&dD,nD*2));
    CK(cudaMemcpy(dA,hAh.data(),nA*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hBh.data(),nB*2,cudaMemcpyHostToDevice));
    // Reference row/col sums: compute D on GPU (plain matmul, no epilogue), copy back, reduce on host (fp32).
    std::vector<float> rowsum(M,0.f), colsum(N,0.f);
    {
        cublasLtMatmulDesc_t d0; LK(cublasLtMatmulDescCreate(&d0,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        cublasOperation_t opN=CUBLAS_OP_N;
        LK(cublasLtMatmulDescSetAttribute(d0,CUBLASLT_MATMUL_DESC_TRANSA,&opN,sizeof(opN)));
        LK(cublasLtMatmulDescSetAttribute(d0,CUBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN)));
        cublasLtMatrixLayout_t lA,lB,lD;
        LK(cublasLtMatrixLayoutCreate(&lA,CUDA_R_16F,M,K,M));
        LK(cublasLtMatrixLayoutCreate(&lB,CUDA_R_16F,K,N,K));
        LK(cublasLtMatrixLayoutCreate(&lD,CUDA_R_16F,M,N,M));
        cublasLtMatmulPreference_t pf; LK(cublasLtMatmulPreferenceCreate(&pf));
        LK(cublasLtMatmulPreferenceSetAttribute(pf,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workSz,sizeof(workSz)));
        cublasLtMatmulHeuristicResult_t h[1]; int g=0;
        LK(cublasLtMatmulAlgoGetHeuristic(lt,d0,lA,lB,lD,lD,pf,1,h,&g));
        float al=1.f,be=0.f;
        LK(cublasLtMatmul(lt,d0,&al,dA,lA,dB,lB,&be,dD,lD,dD,lD,&h[0].algo,dWork,workSz,0));
        CK(cudaDeviceSynchronize());
        std::vector<__half> hD(nD); CK(cudaMemcpy(hD.data(),dD,nD*2,cudaMemcpyDeviceToHost));
        for(int n=0;n<N;n++) for(int m=0;m<M;m++){ float v=__half2float(hD[cm(m,n,M)]); rowsum[m]+=v; colsum[n]+=v; }
        cublasLtMatmulDescDestroy(d0);
        cublasLtMatrixLayoutDestroy(lA);cublasLtMatrixLayoutDestroy(lB);cublasLtMatrixLayoutDestroy(lD);
        cublasLtMatmulPreferenceDestroy(pf);
    }
    run(lt,CUBLASLT_EPILOGUE_BGRADA,"BGRADA",M,K,N,dA,dB,dD,rowsum,colsum,dWork,workSz);
    run(lt,CUBLASLT_EPILOGUE_BGRADB,"BGRADB",M,K,N,dA,dB,dD,rowsum,colsum,dWork,workSz);
    cudaFree(dA);cudaFree(dB);cudaFree(dD);
}

int main(){
    cublasLtHandle_t lt; LK(cublasLtCreate(&lt));
    size_t workSz=64ull*1024*1024; void* dWork; CK(cudaMalloc(&dWork,workSz));
    // tiny shape (axis discrimination, CPU-verifiable)
    test_shape(lt,4,8,6,dWork,workSz,true);
    test_shape(lt,6,5,4,dWork,workSz,true);
    // real Mistral shapes (support + which axis). down_proj K=14336 N=4096; v_proj K=4096 N=1024; gate K=4096 N=14336
    test_shape(lt,1,14336,4096,dWork,workSz,false);   // down_proj decode
    test_shape(lt,2048,14336,4096,dWork,workSz,false); // down_proj prefill
    test_shape(lt,2048,4096,1024,dWork,workSz,false);  // v_proj prefill (worst ratio)
    cudaFree(dWork); cublasLtDestroy(lt);
    printf("\nDONE.\n");
    return 0;
}
