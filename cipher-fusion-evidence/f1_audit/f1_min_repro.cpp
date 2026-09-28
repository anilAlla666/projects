// f1_min_repro.cpp — F1 root-cause, Step 1.
// ONE Marlin INT4 GEMM via the public engine API vs a cuBLAS FP16 GEMM on the
// IDENTICAL operands. Same raw weight buffer feeds both; same activation feeds
// both. Any divergence is Marlin quant/repack/kernel — not a layout artifact.
//
// usage: f1_min_repro <weight_fp16.bin> <K> <N> <M> <libcipher_rt.so>
//   K = in_features, N = out_features, M = batch (Marlin's M)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x)  do{cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
#define CBK(x) do{cublasStatus_t s=(x); if(s){fprintf(stderr,"cuBLAS %s:%d status=%d\n",__FILE__,__LINE__,(int)s);exit(2);}}while(0)

typedef int (*fn_void)(void);
typedef int (*fn_qr)(const void*,int,int);
typedef int (*fn_lk)(const void*,void**,void**,int*,int*,int*);
typedef int (*fn_di)(const void*,const void*,const void*,void*,int,int,int,int,void*);

int main(int argc,char**argv){
    if(argc<6){fprintf(stderr,"usage: %s wfile K N M lib.so\n",argv[0]);return 1;}
    const char* wf=argv[1];
    int K=atoi(argv[2]), N=atoi(argv[3]), M=atoi(argv[4]);
    const char* libpath=argv[5];
    size_t wc=(size_t)K*N, ac=(size_t)M*K, cc=(size_t)M*N;

    void* lib=dlopen(libpath, RTLD_NOW|RTLD_GLOBAL);
    if(!lib){fprintf(stderr,"dlopen: %s\n",dlerror());return 2;}
    fn_void e_init=(fn_void)dlsym(lib,"cipher_rt_marlin_engine_init");
    fn_void e_comp=(fn_void)dlsym(lib,"cipher_rt_marlin_engine_ensure_compiled");
    fn_qr   e_qr  =(fn_qr)  dlsym(lib,"cipher_rt_marlin_engine_quantize_repack");
    fn_lk   e_lk  =(fn_lk)  dlsym(lib,"cipher_rt_marlin_engine_lookup");
    fn_di   e_di  =(fn_di)  dlsym(lib,"cipher_rt_marlin_engine_dispatch");
    if(!e_init||!e_comp||!e_qr||!e_lk||!e_di){fprintf(stderr,"dlsym failed\n");return 2;}

    // weight: K x N row-major fp16
    std::vector<uint16_t> hW(wc);
    FILE* f=fopen(wf,"rb"); if(!f){perror("weight");return 2;}
    if(fread(hW.data(),2,wc,f)!=wc){fprintf(stderr,"short weight read\n");return 2;}
    fclose(f);

    // activation: M x K row-major fp16, deterministic N(0,1)
    std::vector<uint16_t> hA(ac);
    std::mt19937 rng(20260518); std::normal_distribution<float> nd(0.f,1.f);
    for(size_t i=0;i<ac;i++){ __half h=__float2half(nd(rng)); memcpy(&hA[i],&h,2); }

    CK(cudaSetDevice(0));
    CK(cudaFree(0));  // force primary context creation

    void *dW,*dA,*dCm,*dCr;
    CK(cudaMalloc(&dW,wc*2)); CK(cudaMalloc(&dA,ac*2));
    CK(cudaMalloc(&dCm,cc*2)); CK(cudaMalloc(&dCr,cc*2));
    CK(cudaMemcpy(dW,hW.data(),wc*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA,hA.data(),ac*2,cudaMemcpyHostToDevice));
    CK(cudaMemset(dCm,0,cc*2));

    // cuBLAS FP16 reference (FP32 accumulate): row-major C(M,N)=A(M,K)*W(K,N).
    // row-major X = col-major X^T, so C^T(N,M) = W^T(N,K) * A^T(K,M):
    cublasHandle_t cb; CBK(cublasCreate(&cb));
    float one=1.f, zero=0.f;
    CBK(cublasGemmEx(cb,CUBLAS_OP_N,CUBLAS_OP_N,
        N,M,K,
        &one,
        dW,CUDA_R_16F,N,
        dA,CUDA_R_16F,K,
        &zero,
        dCr,CUDA_R_16F,N,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize());

    // Marlin: quant+repack, then ONE dispatch (no green ctx -> full-GPU grid=132).
    if(e_init()!=0){fprintf(stderr,"engine_init failed\n");return 3;}
    if(e_comp()!=0){fprintf(stderr,"ensure_compiled failed\n");return 3;}
    if(e_qr(dW,K,N)!=0){fprintf(stderr,"quantize_repack failed\n");return 3;}
    void *mB=0,*mS=0; int gK=0,gN=0,gG=0;
    if(!e_lk(dW,&mB,&mS,&gK,&gN,&gG)){fprintf(stderr,"lookup failed\n");return 3;}
    fprintf(stderr,"lookup: K=%d N=%d G=%d B=%p S=%p\n",gK,gN,gG,mB,mS);
    int rc=e_di(dA,mB,mS,dCm,M,N,K,gG,nullptr);
    CK(cudaDeviceSynchronize());
    fprintf(stderr,"dispatch rc=%d\n",rc);
    if(rc!=0){fprintf(stderr,"DISPATCH FAILED\n");return 3;}

    // compare
    std::vector<uint16_t> hCm(cc),hCr(cc);
    CK(cudaMemcpy(hCm.data(),dCm,cc*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hCr.data(),dCr,cc*2,cudaMemcpyDeviceToHost));
    double maxabs=0,sumabs=0,maxref=0; int nan_m=0,nan_r=0,zero_m=0;
    for(size_t i=0;i<cc;i++){
        __half hm,hr; memcpy(&hm,&hCm[i],2); memcpy(&hr,&hCr[i],2);
        float m=__half2float(hm), r=__half2float(hr);
        if(std::isnan(m)||std::isinf(m))nan_m++;
        if(std::isnan(r)||std::isinf(r))nan_r++;
        if(m==0.f)zero_m++;
        double e=std::fabs((double)m-(double)r);
        if(e>maxabs)maxabs=e; sumabs+=e;
        if(std::fabs(r)>maxref)maxref=std::fabs(r);
    }
    printf("=== F1 Step 1: single Marlin GEMM vs cuBLAS FP16 ===\n");
    printf("lib   = %s\n",libpath);
    printf("shape : M=%d N=%d K=%d  (%zu output elems)\n",M,N,K,cc);
    printf("marlin: NaN/Inf=%d  exact-zero=%d / %zu\n",nan_m,zero_m,cc);
    printf("ref   : NaN/Inf=%d\n",nan_r);
    printf("max_abs_error  = %.6g\n",maxabs);
    printf("mean_abs_error = %.6g\n",sumabs/cc);
    printf("max |ref|      = %.6g\n",maxref);
    printf("rel error      = %.4g%%  (max_abs / max|ref|)\n",maxref>0?100.0*maxabs/maxref:0.0);
    int show=cc<16?(int)cc:16;
    printf("first %d outputs   [marlin]      [ref]\n",show);
    for(int i=0;i<show;i++){
        __half hm,hr; memcpy(&hm,&hCm[i],2); memcpy(&hr,&hCr[i],2);
        printf("  [%2d]  % .5f   % .5f\n",i,__half2float(hm),__half2float(hr));
    }
    return 0;
}
