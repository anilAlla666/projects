/* Step-0 origin probe: LD_PRELOAD interposer on the PUBLIC cublasLtMatmul (+ typed HSH).
 * On first call: backtrace (which lib called it) + confirm LD_PRELOAD binds ahead of
 * torch's resolution. Forwards via RTLD_NEXT. Prints if NEVER fired (=> internal/cached). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <execinfo.h>
#include <stdatomic.h>
static atomic_int fired_m = 0, fired_hsh = 0;
typedef int (*ltm_fn)(void*,void*,const void*,const void*,void*,const void*,void*,
                      const void*,const void*,void*,void*,void*,const void*,void*,
                      unsigned long,void*);
static void bt(const char* who){
	void* b[24]; int n=backtrace(b,24);
	fprintf(stderr,"INTERPOSER_FIRED %s backtrace(%d):\n", who, n);
	backtrace_symbols_fd(b,n,2);
}
int cublasLtMatmul(void*lh,void*cd,const void*al,const void*A,void*Ad,const void*B,void*Bd,
                   const void*be,const void*C,void*Cd,void*D,void*Dd,const void*algo,
                   void*ws,unsigned long wss,void*st){
	if(atomic_exchange(&fired_m,1)==0) bt("cublasLtMatmul");
	static ltm_fn r=0; if(!r) r=(ltm_fn)dlsym(RTLD_NEXT,"cublasLtMatmul");
	return r(lh,cd,al,A,Ad,B,Bd,be,C,Cd,D,Dd,algo,ws,wss,st);
}
int cublasLtHSHMatmul(void*lh,void*cd,const void*al,const void*A,void*Ad,const void*B,void*Bd,
                      const void*be,const void*C,void*Cd,void*D,void*Dd,const void*algo,
                      void*ws,unsigned long wss,void*st){
	if(atomic_exchange(&fired_hsh,1)==0) bt("cublasLtHSHMatmul");
	static ltm_fn r=0; if(!r) r=(ltm_fn)dlsym(RTLD_NEXT,"cublasLtHSHMatmul");
	return r(lh,cd,al,A,Ad,B,Bd,be,C,Cd,D,Dd,algo,ws,wss,st);
}
__attribute__((destructor)) static void fin(void){
	fprintf(stderr,"INTERPOSER_SUMMARY cublasLtMatmul_fired=%d cublasLtHSHMatmul_fired=%d\n",
	        atomic_load(&fired_m), atomic_load(&fired_hsh));
}
