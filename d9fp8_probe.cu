// D.9 FP8 feasibility probe — cuBLASLt rowwise (OUTER_VEC_32F) FP8 E4M3 vs fp16 ref.
// per-channel-W (along M=out_features) + per-token-A (along N=batch), the bar's scheme.
// Gate: does cu13 cuBLASLt on this H100 produce correct rowwise FP8 output? (KL/rel-err vs fp16)
//
// cuBLAS convention mirrors may13 cipher_fp8_compute.cpp:746-877 and the marlin actuator:
//   weight  = A : K x M  (in_features x out_features), transA = OP_T
//   activ.  = B : K x N  (in_features x batch),        transB = OP_N
//   output  = D : M x N  (out_features x batch), col-major
//   A_scale (OUTER_VEC) length M = out_features = per-output-channel weight scale
//   B_scale (OUTER_VEC) length N = batch        = per-token activation scale
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#ifndef CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
#define CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F 3   // ABI-stable enum value on cu13
#endif

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)
#define LK(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ printf("CUBLASLT ERR %s:%d status=%d\n",__FILE__,__LINE__,(int)s); exit(3);} }while(0)

static const float E4M3_MAX = 448.0f;

// per-row (along cols=K) absmax -> scale -> e4m3 quant.  rows x K row-major.
__global__ void quant_rows(const __half* __restrict__ in, __nv_fp8_e4m3* __restrict__ out,
                           float* __restrict__ scale, int rows, int K){
    int r = blockIdx.x;
    if(r>=rows) return;
    extern __shared__ float sh[];
    float a=0.f;
    for(int k=threadIdx.x;k<K;k+=blockDim.x){ float v=fabsf(__half2float(in[(size_t)r*K+k])); a=fmaxf(a,v);}
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]=fmaxf(sh[threadIdx.x],sh[threadIdx.x+s]); __syncthreads(); }
    float sc = (sh[0]>0.f)? sh[0]/E4M3_MAX : 1.f;
    if(threadIdx.x==0) scale[r]=sc;
    float inv = 1.f/sc;
    for(int k=threadIdx.x;k<K;k+=blockDim.x){
        float v=__half2float(in[(size_t)r*K+k])*inv;
        v=fmaxf(-E4M3_MAX,fminf(E4M3_MAX,v));
        out[(size_t)r*K+k]=__nv_fp8_e4m3(v);
    }
}

// rank-1 outer-product rescale epilogue: D col-major m x n; out[m,n]=D32*s_w[m]*s_a[n] -> fp16
__global__ void epilogue_kernel(const float* __restrict__ D32, const float* __restrict__ sw,
                                const float* __restrict__ sa, __half* __restrict__ out, int M, int N){
    size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t tot = (size_t)M*N;
    if(idx>=tot) return;
    int mm = idx % M;       // row (out_feature)
    int nn = idx / M;       // col (token)
    out[idx] = __float2half(D32[idx] * sw[mm] * sa[nn]);
}
void apply_epi(const float* D32,const float* sw,const float* sa,__half* out,int M,int N){
    size_t tot=(size_t)M*N; int t=256; size_t blk=(tot+t-1)/t;
    epilogue_kernel<<<blk,t>>>(D32,sw,sa,out,M,N);
}

int main(int argc,char**argv){
    // shape: m=out_features, n=batch, k=in_features.  default = FFN up_proj.
    int m = argc>1?atoi(argv[1]):14336;
    int n = argc>2?atoi(argv[2]):8192;
    int k = argc>3?atoi(argv[3]):4096;
    printf("=== probe m(out)=%d n(batch)=%d k(in)=%d ===\n",m,n,k);

    size_t szW=(size_t)m*k, szA=(size_t)n*k, szD=(size_t)m*n;
    std::vector<__half> hW(szW), hA(szA);
    srand(1234);
    for(size_t i=0;i<szW;i++) hW[i]=__float2half(((rand()/(float)RAND_MAX)-0.5f)*0.2f);
    for(size_t i=0;i<szA;i++) hA[i]=__float2half(((rand()/(float)RAND_MAX)-0.5f)*0.4f);

    __half *dW,*dA,*dDref,*dDfp8;
    CK(cudaMalloc(&dW,szW*2)); CK(cudaMalloc(&dA,szA*2));
    CK(cudaMalloc(&dDref,szD*2)); CK(cudaMalloc(&dDfp8,szD*2));
    CK(cudaMemcpy(dW,hW.data(),szW*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA,hA.data(),szA*2,cudaMemcpyHostToDevice));

    __nv_fp8_e4m3 *dWq,*dAq; float *dSw,*dSa;
    CK(cudaMalloc(&dWq,szW)); CK(cudaMalloc(&dAq,szA));
    CK(cudaMalloc(&dSw,m*sizeof(float))); CK(cudaMalloc(&dSa,n*sizeof(float)));
    int thr=256; size_t shb=thr*sizeof(float);
    quant_rows<<<m,thr,shb>>>(dW,dWq,dSw,m,k);   // per-output-channel weight
    quant_rows<<<n,thr,shb>>>(dA,dAq,dSa,n,k);   // per-token activation
    CK(cudaDeviceSynchronize());

    cublasLtHandle_t lt; LK(cublasLtCreate(&lt));
    void* ws; size_t wsb=32u*1024*1024; CK(cudaMalloc(&ws,wsb));
    float *dOne; CK(cudaMalloc(&dOne,sizeof(float))); { float one=1.f; CK(cudaMemcpy(dOne,&one,sizeof(float),cudaMemcpyHostToDevice)); }
    float *dD32; CK(cudaMalloc(&dD32,szD*sizeof(float)));   // FP32 accum out for epilogue path

    // mode: 0=REF_FP16, 1=FP8_OUTERVEC->f16, 2=FP8_SCALAR1->f16, 3=FP8_SCALAR1->f32(epilogue)
    auto run=[&](int mode, void* Dout, int dtypeD)->bool{
        bool fp8 = mode!=0;
        cublasLtMatmulDesc_t desc;
        LK(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        cublasOperation_t ta=CUBLAS_OP_T, tb=CUBLAS_OP_N;
        LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
        LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
        if(fp8){
            if(mode==1){ // rowwise per-channel/per-token
                void* pSw=dSw; void* pSa=dSa; int32_t md=CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
                LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&pSw,sizeof(pSw)));
                LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&pSa,sizeof(pSa)));
                LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&md,sizeof(md)));
                LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&md,sizeof(md)));
            } else { // scalar 1.0 (epilogue applies the real per-ch/per-tok scales)
                void* p=dOne;
                LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&p,sizeof(p)));
                LK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&p,sizeof(p)));
            }
        }
        cublasLtMatrixLayout_t la,lb,ld;
        int atype = fp8?CUDA_R_8F_E4M3:CUDA_R_16F;
        LK(cublasLtMatrixLayoutCreate(&la,(cudaDataType_t)atype,k,m,k));
        LK(cublasLtMatrixLayoutCreate(&lb,(cudaDataType_t)atype,k,n,k));
        LK(cublasLtMatrixLayoutCreate(&ld,(cudaDataType_t)dtypeD,m,n,m));
        cublasLtMatmulPreference_t pref; LK(cublasLtMatmulPreferenceCreate(&pref));
        LK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsb,sizeof(wsb)));
        cublasLtMatmulHeuristicResult_t res{}; int got=0;
        cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,desc,la,lb,ld,ld,pref,1,&res,&got);
        if(hs!=CUBLAS_STATUS_SUCCESS||got==0){ printf("  [mode %d] HEURISTIC FAIL hs=%d got=%d\n",mode,(int)hs,got); cublasLtMatmulDescDestroy(desc); return false; }
        float alpha=1.f,beta=0.f;
        const void* A = fp8?(const void*)dWq:(const void*)dW;
        const void* B = fp8?(const void*)dAq:(const void*)dA;
        cublasStatus_t ms=cublasLtMatmul(lt,desc,&alpha,A,la,B,lb,&beta,Dout,ld,Dout,ld,&res.algo,ws,wsb,0);
        CK(cudaDeviceSynchronize());
        if(ms!=CUBLAS_STATUS_SUCCESS){ printf("  [mode %d] MATMUL FAIL status=%d\n",mode,(int)ms); cublasLtMatmulDescDestroy(desc); return false; }
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);cublasLtMatrixLayoutDestroy(lb);cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulDescDestroy(desc);
        return true;
    };

    bool okref=run(0,dDref,CUDA_R_16F);
    if(!okref){ printf("RESULT: fp16 ref matmul failed (convention/setup bug)\n"); return 1; }
    bool ok_outervec=run(1,dDfp8,CUDA_R_16F);
    printf("  cuBLASLt rowwise(OUTER_VEC) FP8: %s\n", ok_outervec?"SUPPORTED":"NOT SUPPORTED (hs=7) -> epilogue fallback");
    bool ok_scalar=run(2,dDfp8,CUDA_R_16F);
    printf("  cuBLASLt scalar FP8->f16: %s\n", ok_scalar?"works":"FAILS (FP8 GEMM itself unavailable!)");
    // epilogue fallback: FP8 scalar=1 -> FP32, then out[m,n]=D32*s_w[m]*s_a[n] -> fp16
    bool ok_epi32=run(3,dD32,CUDA_R_32F);
    if(ok_outervec){
        printf("  path chosen: OUTER_VEC (native rowwise)\n");
    } else if(ok_epi32){
        printf("  path chosen: scalar-FP8->FP32 + outer-product epilogue\n");
        apply_epi(dD32,dSw,dSa,dDfp8,m,n);
        CK(cudaDeviceSynchronize());
    } else {
        printf("RESULT: neither rowwise nor epilogue-FP8 path worked -> per-channel-at-intercept is the finding\n");
        return 10;
    }

    // CPU sanity on a tiny corner (first 2x2 of D) to lock the convention.
    {
        std::vector<__half> hWv(hW), hAv(hA), hDr(szD);
        CK(cudaMemcpy(hDr.data(),dDref,szD*2,cudaMemcpyDeviceToHost));
        for(int mm=0;mm<2;mm++)for(int nn=0;nn<2;nn++){
            double acc=0; for(int kk=0;kk<k;kk++) acc+=(double)__half2float(hWv[(size_t)mm*k+kk])*__half2float(hAv[(size_t)nn*k+kk]);
            double got=__half2float(hDr[(size_t)nn*m+mm]); // D col-major m x n: elem(mm,nn)=mm+nn*m
            printf("  cpu-check D[%d,%d] cpu=%.4f gpu=%.4f %s\n",mm,nn,acc,got, fabs(acc-got)<0.05*fmax(1.0,fabs(acc))?"OK":"MISMATCH");
        }
    }

    std::vector<__half> hR(szD), hF(szD);
    CK(cudaMemcpy(hR.data(),dDref,szD*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hF.data(),dDfp8,szD*2,cudaMemcpyDeviceToHost));
    double num=0,den=0,maxabs=0; long nan=0;
    for(size_t i=0;i<szD;i++){
        double r=__half2float(hR[i]), f=__half2float(hF[i]);
        if(std::isnan(f)||std::isnan(r)) nan++;
        double d=f-r; num+=d*d; den+=r*r; maxabs=fmax(maxabs,fabs(d));
    }
    double fro=sqrt(num/fmax(den,1e-12));
    // proxy KL over per-token (column) softmax of D[:,n]  (out-feature distribution)
    double klsum=0; int cols=n<256?n:256;
    for(int c=0;c<cols;c++){
        double mr=-1e30,mf=-1e30;
        for(int r=0;r<m;r++){ mr=fmax(mr,__half2float(hR[(size_t)c*m+r])); mf=fmax(mf,__half2float(hF[(size_t)c*m+r])); }
        double zr=0,zf=0;
        for(int r=0;r<m;r++){ zr+=exp(__half2float(hR[(size_t)c*m+r])-mr); zf+=exp(__half2float(hF[(size_t)c*m+r])-mf); }
        double kl=0;
        for(int r=0;r<m;r++){ double p=exp(__half2float(hR[(size_t)c*m+r])-mr)/zr; double q=exp(__half2float(hF[(size_t)c*m+r])-mf)/zf; if(p>1e-12) kl+=p*log(p/fmax(q,1e-12)); }
        klsum+=kl;
    }
    double klmean=klsum/cols;
    printf("RESULT fp8 OUTER_VEC: fro_rel=%.4f max_abs=%.4f nan=%ld kl_proxy_nats=%.5f  => %s\n",
           fro,maxabs,nan,klmean,(nan==0 && fro<0.08)?"FEASIBLE":"SUSPECT");
    return (nan==0 && fro<0.08)?0:11;
}
