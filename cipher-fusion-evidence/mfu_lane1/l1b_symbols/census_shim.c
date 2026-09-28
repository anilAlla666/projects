/* Lane-1b symbol-census shim — SCRATCH LD_PRELOAD, libc-only, NO cipher substrate, NO /dev/cipher, NO dlsym hook.
 * Interposes the PUBLIC cuBLAS GEMM entries only (safe: normal ELF interposition, no CUDA-init interference).
 * Counts cublasGemmEx / cublasLtMatmul / cublasGemmStridedBatchedEx. The dtype-specialized private cuBLASLt
 * variant entries (cublasLt<V>Matmul, incl. HSH) are NOT public symbols and torch resolves them via dlsym — those
 * are identified separately by a torch-profiler kernel-name census (nvjet_sm90_<dtype>_* = the variant), so this
 * shim deliberately does NOT hook dlsym (a global dlsym hook hangs CUDA lazy resolution).
 * Counters -> RV_OUT json at exit. Real symbols via RTLD_NEXT chain (dlopen of the real lib). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

static void* hcublas=0; static void* hcublasLt=0;
static void* Lc(const char*sym){ if(!hcublas){hcublas=dlopen("libcublas.so.13",RTLD_NOW|RTLD_GLOBAL); if(!hcublas)hcublas=dlopen("libcublas.so.12",RTLD_NOW|RTLD_GLOBAL);} return hcublas?dlsym(hcublas,sym):0; }
static void* Llt(const char*sym){ if(!hcublasLt){hcublasLt=dlopen("libcublasLt.so.13",RTLD_NOW|RTLD_GLOBAL); if(!hcublasLt)hcublasLt=dlopen("libcublasLt.so.12",RTLD_NOW|RTLD_GLOBAL);} return hcublasLt?dlsym(hcublasLt,sym):0; }

static atomic_ulong c_gemmEx=0, c_ltMatmul=0, c_stridedEx=0;
static char OUTP[512]; static int inited=0;
static void wr(void){
  FILE*f=fopen(OUTP[0]?OUTP:"./census.json","w"); if(!f) return;
  fprintf(f,"{\"cublasGemmEx\":%lu,\"cublasLtMatmul\":%lu,\"cublasGemmStridedBatchedEx\":%lu,"
            "\"note\":\"public entries only; private cublasLt<V>Matmul variants via profiler kernel census\"}\n",
    atomic_load(&c_gemmEx),atomic_load(&c_ltMatmul),atomic_load(&c_stridedEx)); fclose(f);
}
__attribute__((destructor)) static void fin(void){ if(inited) wr(); }
static void init(void){ inited=1; const char*s=getenv("RV_OUT"); if(s) strncpy(OUTP,s,511); atexit(wr); fprintf(stderr,"[census] init OUT=%s\n",OUTP[0]?OUTP:"./census.json"); }

typedef int (*gemmEx_t)(void*,int,int,int,int,int,const void*,const void*,int,int,const void*,int,int,const void*,void*,int,int,int,int);
int cublasGemmEx(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,const void*B,int Bt,int ldb,const void*be,void*C,int Ct,int ldc,int ct,int algo){
  if(!inited) init(); unsigned long n_=atomic_fetch_add(&c_gemmEx,1)+1;
  if((n_%50)==0) wr();   /* periodic flush: EngineCore worker is SIGKILLed at shutdown, destructor won't run */
  static gemmEx_t r=0; if(!r) r=(gemmEx_t)Lc("cublasGemmEx");
  return r(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,C,Ct,ldc,ct,algo);
}
typedef int (*ltm_t)(void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,void*,const void*,const void*,void*,void*,void*);
int cublasLtMatmul(void*lt,const void*d,const void*al,const void*A,const void*Ad,const void*B,const void*Bd,const void*be,const void*C,const void*Cd,void*D,const void*Dd,const void*algo,void*ws,void*wss,void*st){
  if(!inited) init(); unsigned long n_=atomic_fetch_add(&c_ltMatmul,1)+1; if((n_%50)==0) wr();
  static ltm_t r=0; if(!r) r=(ltm_t)Llt("cublasLtMatmul");
  return r(lt,d,al,A,Ad,B,Bd,be,C,Cd,D,Dd,algo,ws,wss,st);
}
typedef int (*strided_t)(void*,int,int,int,int,int,const void*,const void*,int,int,long long,const void*,int,int,long long,const void*,void*,int,int,long long,int,int,int);
int cublasGemmStridedBatchedEx(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,long long sa,const void*B,int Bt,int ldb,long long sb,const void*be,void*C,int Ct,int ldc,long long sc,int bc,int ct,int algo){
  if(!inited) init(); unsigned long n_=atomic_fetch_add(&c_stridedEx,1)+1; if((n_%50)==0) wr();
  static strided_t r=0; if(!r) r=(strided_t)Lc("cublasGemmStridedBatchedEx");
  return r(h,ta,tb,m,n,k,al,A,At,lda,sa,B,Bt,ldb,sb,be,C,Ct,ldc,sc,bc,ct,algo);
}
