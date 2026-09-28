// D.9 FP8 timing + bf16 probe — Guard-2 de-risk BEFORE the engine build.
// Times the full CIPHER-engaged actuator cost (per-token act-quant + scalar-FP8 GEMM
// + rank-1 outer-product epilogue) vs the cuBLASLt fp16/bf16 baseline the app would run,
// per Mistral GEMM shape. If FP8+epilogue does not beat baseline per-shape, Guard 2 cannot pass.
// Weight prequant is one-time (amortized) -> excluded from the per-call timed path.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)
static const float E4M3_MAX=448.0f;

template<typename T> __device__ float to_f(T v);
template<> __device__ float to_f<__half>(__half v){return __half2float(v);}
template<> __device__ float to_f<__nv_bfloat16>(__nv_bfloat16 v){return __bfloat162float(v);}

template<typename T>
__global__ void quant_rows(const T* __restrict__ in, __nv_fp8_e4m3* __restrict__ out,
                           float* __restrict__ scale, int rows, int K){
    int r=blockIdx.x; if(r>=rows) return;
    extern __shared__ float sh[];
    float a=0.f;
    for(int k=threadIdx.x;k<K;k+=blockDim.x){ float v=fabsf(to_f<T>(in[(size_t)r*K+k])); a=fmaxf(a,v);}
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]=fmaxf(sh[threadIdx.x],sh[threadIdx.x+s]); __syncthreads();}
    float sc=(sh[0]>0.f)?sh[0]/E4M3_MAX:1.f;
    if(threadIdx.x==0) scale[r]=sc;
    float inv=1.f/sc;
    for(int k=threadIdx.x;k<K;k+=blockDim.x){ float v=to_f<T>(in[(size_t)r*K+k])*inv; v=fmaxf(-E4M3_MAX,fminf(E4M3_MAX,v)); out[(size_t)r*K+k]=__nv_fp8_e4m3(v);}
}
template<typename T>
__global__ void epi(const float* __restrict__ D32,const float* __restrict__ sw,const float* __restrict__ sa,
                    T* __restrict__ out,int M,int N){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x, tot=(size_t)M*N; if(idx>=tot) return;
    int mm=idx%M, nn=idx/M; out[idx]=(T)(D32[idx]*sw[mm]*sa[nn]);
}

static cublasLtHandle_t lt; static void* ws; static size_t wsb=32u*1024*1024; static float* dOne;

// returns ms over iters; mode 0=fp16/bf16 ref, 1=engaged(quant+fp8+epi). dtAB: CUDA_R_16F or 16BF
template<typename T>
double timeit(int mode,int dt, const T* dW,const T* dA, __nv_fp8_e4m3* dWq,__nv_fp8_e4m3* dAq,
              float* dSw,float* dSa, float* dD32, T* dDo, int m,int n,int k,int iters){
    auto build=[&](int fp8,int dtypeD,cublasLtMatmulDesc_t* pdesc,cublasLtMatmulHeuristicResult_t* pres,
                   cublasLtMatrixLayout_t* pla,cublasLtMatrixLayout_t* plb,cublasLtMatrixLayout_t* pld)->bool{
        cublasLtMatmulDescCreate(pdesc,CUBLAS_COMPUTE_32F,CUDA_R_32F);
        cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(*pdesc,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta));
        cublasLtMatmulDescSetAttribute(*pdesc,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb));
        if(fp8){ void* p=dOne; int8_t fa=1;
            cublasLtMatmulDescSetAttribute(*pdesc,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fa,sizeof(fa));
            cublasLtMatmulDescSetAttribute(*pdesc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&p,sizeof(p));
            cublasLtMatmulDescSetAttribute(*pdesc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&p,sizeof(p)); }
        int at=fp8?CUDA_R_8F_E4M3:dt;
        cublasLtMatrixLayoutCreate(pla,(cudaDataType_t)at,k,m,k);
        cublasLtMatrixLayoutCreate(plb,(cudaDataType_t)at,k,n,k);
        cublasLtMatrixLayoutCreate(pld,(cudaDataType_t)dtypeD,m,n,m);
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsb,sizeof(wsb));
        int got=0; cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,*pdesc,*pla,*plb,*pld,*pld,pref,1,pres,&got);
        cublasLtMatmulPreferenceDestroy(pref);
        return hs==CUBLAS_STATUS_SUCCESS&&got>0;
    };
    cublasLtMatmulDesc_t desc; cublasLtMatmulHeuristicResult_t res{}; cublasLtMatrixLayout_t la,lb,ld;
    int fp8=(mode!=0); int dtypeD = fp8?CUDA_R_32F:dt;
    if(!build(fp8,dtypeD,&desc,&res,&la,&lb,&ld)){ return -1; }
    float alpha=1.f,beta=0.f; int thr=256; size_t shb=thr*sizeof(float);
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // mode: 1=full(quant+gemm+epi) 2=gemm-only 3=gemm+epi
    auto once=[&](){
        if(fp8){
            size_t tot=(size_t)m*n;
            if(mode==1) quant_rows<T><<<n,thr,shb>>>(dA,dAq,dSa,n,k);  // per-token act quant
            cublasLtMatmul(lt,desc,&alpha,dWq,la,dAq,lb,&beta,dD32,ld,dD32,ld,&res.algo,ws,wsb,0);
            if(mode!=2) epi<T><<<(tot+thr-1)/thr,thr>>>(dD32,dSw,dSa,dDo,m,n);
        } else {
            cublasLtMatmul(lt,desc,&alpha,dW,la,dA,lb,&beta,dDo,ld,dDo,ld,&res.algo,ws,wsb,0);
        }
    };
    for(int i=0;i<20;i++) once(); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for(int i=0;i<iters;i++) once(); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    cublasLtMatrixLayoutDestroy(la);cublasLtMatrixLayoutDestroy(lb);cublasLtMatrixLayoutDestroy(ld);cublasLtMatmulDescDestroy(desc);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return ms/iters;
}

template<typename T>
void run_shape(const char* name,int dt,int m,int n,int k){
    size_t szW=(size_t)m*k,szA=(size_t)n*k,szD=(size_t)m*n;
    std::vector<float> hW(szW),hA(szA);
    srand(7); for(size_t i=0;i<szW;i++)hW[i]=((rand()/(float)RAND_MAX)-0.5f)*0.2f; for(size_t i=0;i<szA;i++)hA[i]=((rand()/(float)RAND_MAX)-0.5f)*0.4f;
    T *dW,*dA,*dDo; CK(cudaMalloc(&dW,szW*sizeof(T)));CK(cudaMalloc(&dA,szA*sizeof(T)));CK(cudaMalloc(&dDo,szD*sizeof(T)));
    { std::vector<T> t(szW); for(size_t i=0;i<szW;i++)t[i]=(T)hW[i]; CK(cudaMemcpy(dW,t.data(),szW*sizeof(T),cudaMemcpyHostToDevice)); }
    { std::vector<T> t(szA); for(size_t i=0;i<szA;i++)t[i]=(T)hA[i]; CK(cudaMemcpy(dA,t.data(),szA*sizeof(T),cudaMemcpyHostToDevice)); }
    __nv_fp8_e4m3 *dWq,*dAq; float *dSw,*dSa,*dD32;
    CK(cudaMalloc(&dWq,szW));CK(cudaMalloc(&dAq,szA));CK(cudaMalloc(&dSw,m*4));CK(cudaMalloc(&dSa,n*4));CK(cudaMalloc(&dD32,szD*4));
    int thr=256; size_t shb=thr*sizeof(float);
    quant_rows<T><<<m,thr,shb>>>(dW,dWq,dSw,m,k); CK(cudaDeviceSynchronize());   // one-time weight prequant
    double tb=timeit<T>(0,dt,dW,dA,dWq,dAq,dSw,dSa,dD32,dDo,m,n,k,100);
    double tg=timeit<T>(2,dt,dW,dA,dWq,dAq,dSw,dSa,dD32,dDo,m,n,k,100);  // FP8 GEMM only
    double tge=timeit<T>(3,dt,dW,dA,dWq,dAq,dSw,dSa,dD32,dDo,m,n,k,100); // +epilogue
    double tf=timeit<T>(1,dt,dW,dA,dWq,dAq,dSw,dSa,dD32,dDo,m,n,k,100);  // +act-quant (full)
    double flop=2.0*m*n*k;
    printf("  %-9s m=%5d n=%5d k=%5d | base %.0fTF/s | gemmONLY %.0fTF/s(%.2fx) | +epi %.0fTF/s | full %.0fTF/s(%.2fx %s)\n",
           name,m,n,k, flop/(tb*1e9), flop/(tg*1e9), tb/tg, flop/(tge*1e9), flop/(tf*1e9), tb/tf, (tb/tf>=1.0)?"WIN":"LOSS");
    cudaFree(dW);cudaFree(dA);cudaFree(dDo);cudaFree(dWq);cudaFree(dAq);cudaFree(dSw);cudaFree(dSa);cudaFree(dD32);
}

int main(int argc,char**argv){
    cublasLtCreate(&lt); CK(cudaMalloc(&ws,wsb)); CK(cudaMalloc(&dOne,4)); { float o=1.f; CK(cudaMemcpy(dOne,&o,4,cudaMemcpyHostToDevice)); }
    int N = argc>1?atoi(argv[1]):8192;   // batch dim (call->n); B=64,S=128 => 8192
    printf("=== FP8 actuator timing vs baseline (batch N=%d) ===\n",N);
    printf("--- bf16 (CUDA_R_16BF=14) — the training/large-batch forward dtype ---\n");
    run_shape<__nv_bfloat16>("q/o_proj",CUDA_R_16BF,4096,N,4096);
    run_shape<__nv_bfloat16>("kv_proj", CUDA_R_16BF,1024,N,4096);
    run_shape<__nv_bfloat16>("gate/up", CUDA_R_16BF,14336,N,4096);
    run_shape<__nv_bfloat16>("down",    CUDA_R_16BF,4096,N,14336);
    printf("--- fp16 (CUDA_R_16F=2) ---\n");
    run_shape<__half>("gate/up",CUDA_R_16F,14336,N,4096);
    printf("note: weight prequant is one-time (amortized), excluded from timed path; engaged path = act-quant + FP8 GEMM + epilogue\n");
    return 0;
}
