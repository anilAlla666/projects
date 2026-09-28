/* Crash bisection: minimal pass-through. cublasGemmEx logs C only (NO GetStream/IsCapturing). instantiate
 * hook counts nodes + calls real (NO scan, NO cudaPointerGetAttributes). cudagraph ON. Isolates whether the
 * crash is the capture-phase interception or the instantiate-hook scan work. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
typedef int cbStatus;
static void *hcb=0,*hcu=0;
static void* cb(const char*n){ if(!hcb)hcb=dlopen("libcublas.so.13",RTLD_NOW|RTLD_GLOBAL); return hcb?dlsym(hcb,n):0; }
static void* cu(const char*n){ if(!hcu)hcu=dlopen("libcudart.so.13",RTLD_NOW|RTLD_GLOBAL); return hcu?dlsym(hcu,n):0; }
typedef cbStatus (*ge_t)(void*,int,int,int,int,int,const void*,const void*,int,int,const void*,int,int,const void*,void*,int,int,int,int);
typedef cbStatus (*inst_t)(void**,void*,unsigned long long);
typedef int (*getnodes_t)(void*,void**,size_t*);
static long nclog=0, inst_calls=0, max_nodes=0;
static void* lastC=0;

cbStatus cublasGemmEx(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,
                      const void*B,int Bt,int ldb,const void*be,void*C,int Ct,int ldc,int ct,int algo){
  static ge_t R=0; if(!R)R=(ge_t)cb("cublasGemmEx");
  lastC=C; nclog++;   /* log only — proven safe in the eager run */
  return R(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,C,Ct,ldc,ct,algo);
}
cbStatus cudaGraphInstantiateWithFlags(void**pExec, void*graph, unsigned long long flags){
  static inst_t R=0; if(!R)R=(inst_t)cu("cudaGraphInstantiateWithFlags");
  static getnodes_t GN=0; if(!GN)GN=(getnodes_t)cu("cudaGraphGetNodes");
  size_t num=0; if(GN) GN(graph,NULL,&num);   /* count only, no node enumeration/scan */
  inst_calls++; if((long)num>max_nodes)max_nodes=num;
  return R(pExec,graph,flags);   /* pass through */
}
__attribute__((destructor)) static void fin(){
  FILE*f=fopen("/home/ubuntu/cipher-fusion-evidence/ra_path1/probe_v3_result.json","w");
  if(f){ fprintf(f,"{\"gemmEx\":%ld,\"instantiate\":%ld,\"max_nodes\":%ld}\n",nclog,inst_calls,max_nodes); fclose(f); }
  fprintf(stderr,"[v3] FINAL gemmEx=%ld instantiate=%ld max_nodes=%ld\n",nclog,inst_calls,max_nodes);
}
