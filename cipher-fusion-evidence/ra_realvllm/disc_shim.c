/* Discovery shim: which cuBLAS entry does vLLM eager fp16 linear use, and what shapes?
 * Intercepts cublasGemmEx + cublasLtMatmul (UNDEF in libtorch_cuda from cublas.so.13/cublasLt.so.13).
 * Counts + logs first shapes. Destructor writes counts. LD_PRELOAD into a vLLM eager decode. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
typedef int cbStatus;
static long ge=0, lt=0; static int logged=0;
/* resolve real (version-tagged) symbols by dlopen of the actual libs, not RTLD_NEXT */
static void* hcb=0; static void* hlt=0;
static void* rsym(const char*lib,void**h,const char*name){
  if(!*h) *h=dlopen(lib, RTLD_NOW|RTLD_GLOBAL);
  void*p=*h?dlsym(*h,name):0;
  if(!p){ fprintf(stderr,"[disc] FATAL: cannot resolve %s from %s\n",name,lib); }
  return p;
}

typedef cbStatus (*ge_t)(void*,int,int,int,int,int,const void*,const void*,int,int,const void*,int,int,const void*,void*,int,int,int,int);
cbStatus cublasGemmEx(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,const void*B,int Bt,int ldb,const void*be,void*C,int Ct,int ldc,int ct,int algo){
  static ge_t r=0; if(!r) r=(ge_t)rsym("libcublas.so.13",&hcb,"cublasGemmEx");
  ge++; if(logged<40){ fprintf(stderr,"[disc] GemmEx #%ld m=%d n=%d k=%d\n",ge,m,n,k); logged++; }
  return r(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,C,Ct,ldc,ct,algo);
}
typedef cbStatus (*lt_t)(void*,void*,const void*,const void*,void*,const void*,void*,const void*,const void*,void*,void*,void*,const void*,void*,size_t,void*);
typedef cbStatus (*gla_t)(void*,int,void*,size_t,size_t*);
cbStatus cublasLtMatmul(void*lh,void*cd,const void*al,const void*A,void*Ad,const void*B,void*Bd,const void*be,const void*C,void*Cd,void*D,void*Dd,const void*algo,void*ws,size_t wss,void*stream){
  static lt_t r=0; if(!r) r=(lt_t)rsym("libcublasLt.so.13",&hlt,"cublasLtMatmul");
  static gla_t ga=0; if(!ga) ga=(gla_t)rsym("libcublasLt.so.13",&hlt,"cublasLtMatrixLayoutGetAttribute");
  lt++;
  if(logged<40 && ga && Dd){ uint64_t rw=0,cl=0; size_t w; ga(Dd,2,&rw,8,&w); ga(Dd,3,&cl,8,&w);
    fprintf(stderr,"[disc] LtMatmul #%ld Drows=%lu Dcols=%lu\n",lt,(unsigned long)rw,(unsigned long)cl); logged++; }
  return r(lh,cd,al,A,Ad,B,Bd,be,C,Cd,D,Dd,algo,ws,wss,stream);
}
__attribute__((destructor)) static void fin(void){
  FILE*f=fopen("/home/ubuntu/cipher-fusion-evidence/ra_realvllm/discovery_counts.txt","w");
  if(f){ fprintf(f,"cublasGemmEx=%ld\ncublasLtMatmul=%ld\n",ge,lt); fclose(f); }
  fprintf(stderr,"[disc] FINAL cublasGemmEx=%ld cublasLtMatmul=%ld\n",ge,lt);
}
