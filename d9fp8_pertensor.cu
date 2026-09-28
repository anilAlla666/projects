// Per-tensor scalar fused FP8 (no epilogue) timing + Mistral drop_downproj layer projection.
// Gives Anil a concrete projected layer-level speedup for the per-tensor path.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
static const float E4M3_MAX=448.f;
__global__ void gabsmax(const __nv_bfloat16* in,int* amx,size_t nel){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; float a=0;
    for(;i<nel;i+=(size_t)gridDim.x*blockDim.x) a=fmaxf(a,fabsf(__bfloat162float(in[i])));
    __shared__ float sh[256]; sh[threadIdx.x]=a; __syncthreads();
    for(int s=128;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]=fmaxf(sh[threadIdx.x],sh[threadIdx.x+s]); __syncthreads(); }
    if(threadIdx.x==0) atomicMax(amx,__float_as_int(sh[0]));
}
__global__ void finscale(int* amx,float* sc){ float a=__int_as_float(*amx); *sc=(a>0)?a/E4M3_MAX:1.f; }
__global__ void qtensor(const __nv_bfloat16* in,__nv_fp8_e4m3* out,const float* sc,size_t nel){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=nel)return; float inv=1.f/(*sc);
    float v=__bfloat162float(in[i])*inv; v=fmaxf(-E4M3_MAX,fminf(E4M3_MAX,v)); out[i]=__nv_fp8_e4m3(v);
}

static cublasLtHandle_t lt; static void* ws; static size_t wsb=32u<<20;

double time_gemm(int fp8,const void*A,const void*B,void*D,int m,int n,int k,void* sA,void* sB,int iters){
    cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d,CUBLAS_COMPUTE_32F,CUDA_R_32F);
    cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta));
    cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb));
    if(fp8){ int8_t f=1; cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&f,sizeof(f));
        cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sA,sizeof(sA));
        cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sB,sizeof(sB)); }
    int at=fp8?CUDA_R_8F_E4M3:CUDA_R_16BF;
    cublasLtMatrixLayout_t la,lb,ld;
    cublasLtMatrixLayoutCreate(&la,(cudaDataType_t)at,k,m,k);
    cublasLtMatrixLayoutCreate(&lb,(cudaDataType_t)at,k,n,k);
    cublasLtMatrixLayoutCreate(&ld,CUDA_R_16BF,m,n,m);     // bf16 output directly (fused scalar) — NO epilogue
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsb,sizeof(wsb));
    cublasLtMatmulHeuristicResult_t r{}; int got=0;
    if(cublasLtMatmulAlgoGetHeuristic(lt,d,la,lb,ld,ld,pref,1,&r,&got)!=CUBLAS_STATUS_SUCCESS||!got){printf("heur fail fp8=%d\n",fp8);return -1;}
    float al=1,be=0; cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
    for(int i=0;i<20;i++) cublasLtMatmul(lt,d,&al,A,la,B,lb,&be,D,ld,D,ld,&r.algo,ws,wsb,0);
    CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
    for(int i=0;i<iters;i++) cublasLtMatmul(lt,d,&al,A,la,B,lb,&be,D,ld,D,ld,&r.algo,ws,wsb,0);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    cublasLtMatmulPreferenceDestroy(pref);cublasLtMatrixLayoutDestroy(la);cublasLtMatrixLayoutDestroy(lb);cublasLtMatrixLayoutDestroy(ld);cublasLtMatmulDescDestroy(d);
    return ms/iters;
}
// per-tensor activation quant time (absmax+finalize+quant) over n x k
double time_actquant(const __nv_bfloat16* A,__nv_fp8_e4m3* Aq,int* amx,float* sc,size_t nel,int iters){
    cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
    auto once=[&](){ cudaMemsetAsync(amx,0,4,0); gabsmax<<<512,256>>>(A,amx,nel); finscale<<<1,1>>>(amx,sc); qtensor<<<(nel+255)/256,256>>>(A,Aq,sc,nel); };
    for(int i=0;i<20;i++) once(); CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
    for(int i=0;i<iters;i++) once(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    cudaEventDestroy(e0);cudaEventDestroy(e1); return ms/iters;
}

struct G{const char*name;int m,k;int act;}; // act = activation group id (shared act-quant)
int main(int argc,char**argv){
    cublasLtCreate(&lt); CK(cudaMalloc(&ws,wsb));
    int N=argc>1?atoi(argv[1]):8192;
    // Mistral-7B per-layer linears (unfused, HF forward). act groups: 0=attn-in(q,k,v) 1=attn-out(o) 2=mlp-in(gate,up) 3=mlp-in2(down)
    G gs[]={{"q_proj",4096,4096,0},{"k_proj",1024,4096,0},{"v_proj",1024,4096,0},{"o_proj",4096,4096,1},
            {"gate",14336,4096,2},{"up",14336,4096,2},{"down",4096,14336,3}};
    int NG=7;
    float *sA,*sB; int*amx; CK(cudaMalloc(&sA,4));CK(cudaMalloc(&sB,4));CK(cudaMalloc(&amx,4));
    { float one=1; CK(cudaMemcpy(sA,&one,4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(sB,&one,4,cudaMemcpyHostToDevice)); }
    printf("=== per-tensor fused FP8 (scalar->bf16, NO epilogue) | Mistral layer, batch N=%d ===\n",N);
    double sum_bf16=0, sum_fp8_drop=0, sum_fp8_all=0;
    double actq[4]={-1,-1,-1,-1}; // measured once per act group
    for(int i=0;i<NG;i++){
        G g=gs[i]; int m=g.m,k=g.k;
        size_t szW=(size_t)m*k, szA=(size_t)N*k, szD=(size_t)m*N;
        __nv_bfloat16 *dW,*dA,*dD; CK(cudaMalloc(&dW,szW*2));CK(cudaMalloc(&dA,szA*2));CK(cudaMalloc(&dD,szD*2));
        { std::vector<__nv_bfloat16> t(szW); for(size_t j=0;j<szW;j++)t[j]=(__nv_bfloat16)(((rand()/(float)RAND_MAX)-0.5f)*0.2f); CK(cudaMemcpy(dW,t.data(),szW*2,cudaMemcpyHostToDevice)); }
        { std::vector<__nv_bfloat16> t(szA); for(size_t j=0;j<szA;j++)t[j]=(__nv_bfloat16)(((rand()/(float)RAND_MAX)-0.5f)*0.4f); CK(cudaMemcpy(dA,t.data(),szA*2,cudaMemcpyHostToDevice)); }
        __nv_fp8_e4m3 *dWq,*dAq; CK(cudaMalloc(&dWq,szW));CK(cudaMalloc(&dAq,szA));
        qtensor<<<(szW+255)/256,256>>>(dW,dWq,sB,szW); CK(cudaDeviceSynchronize()); // one-time weight (timing excl.)
        double tb=time_gemm(0,dW,dA,dD,m,N,k,0,0,100);
        double tg=time_gemm(1,dWq,dAq,dD,m,N,k,sA,sB,100);   // fused scalar FP8 GEMM only
        if(actq[g.act]<0) actq[g.act]=time_actquant(dA,dAq,amx,sB,szA,100); // act-quant once per group
        sum_bf16+=tb;
        sum_fp8_all+=tg;                              // all_fp8 counts every GEMM as fp8
        sum_fp8_drop += (g.act==3? tb : tg);          // drop_downproj: down stays bf16
        printf("  %-7s m=%5d k=%5d | bf16 %.3fms | fp8gemm %.3fms (%.2fx) | actq[grp%d] %.3fms\n",
               g.name,m,k,tb,tg,tb/tg,g.act,actq[g.act]);
        cudaFree(dW);cudaFree(dA);cudaFree(dD);cudaFree(dWq);cudaFree(dAq);
    }
    // add shared act-quant: drop_downproj engages groups 0,1,2 (NOT 3); all_fp8 engages 0,1,2,3
    double aq_drop=actq[0]+actq[1]+actq[2];
    double aq_all =actq[0]+actq[1]+actq[2]+actq[3];
    double layer_bf16=sum_bf16;
    double layer_drop=sum_fp8_drop+aq_drop;
    double layer_all =sum_fp8_all +aq_all;
    printf("\n--- Mistral-7B LAYER projection (clock-invariant ratio) ---\n");
    printf("  all-bf16 layer:            %.3f ms\n",layer_bf16);
    printf("  per-tensor drop_downproj:  %.3f ms  -> layer speedup %.2fx  (down=bf16; q,k,v,o,gate,up=fp8; +3 shared act-quants)\n",layer_drop,layer_bf16/layer_drop);
    printf("  per-tensor all_fp8:        %.3f ms  -> layer speedup %.2fx  (all 7 fp8; +4 act-quants) [NOTE quality +0.478%% FAILS bar]\n",layer_all,layer_bf16/layer_all);
    printf("  => MFU(drop_downproj) ~= %.2fx * bf16-MFU ; if bf16~=75%% of 989, fp8 drop ~= %.0f%% of 989 (full-clock proj)\n",
           layer_bf16/layer_drop, 75.0*(layer_bf16/layer_drop));
    return 0;
}
