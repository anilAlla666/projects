/* PATH-1 Part-A feasibility probe: can a captured cuBLAS GEMM kernel node's params be matched to the
 * GEMM's OUTPUT pointer? Intercept cublasGemmEx -> log output ptr C (+ m,n,k) and whether the handle's
 * stream is capturing. Intercept cudaGraphInstantiateWithFlags -> enumerate the real vLLM graph's kernel
 * nodes, and for each, scan its kernelParams[0] target window for any logged C ptr. Logging only (no
 * injection). Real symbols via dlopen (version-tagged). Output to stderr + file.  NOT the anchor, no vLLM edit.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef int cbStatus;
static void *hcb=0,*hcu=0;
static void* cb(const char*n){ if(!hcb)hcb=dlopen("libcublas.so.13",RTLD_NOW|RTLD_GLOBAL); return hcb?dlsym(hcb,n):0; }
static void* cu(const char*n){ if(!hcu)hcu=dlopen("libcudart.so.13",RTLD_NOW|RTLD_GLOBAL); return hcu?dlsym(hcu,n):0; }

/* cudaKernelNodeParams mirror */
typedef struct { void*func; unsigned gx,gy,gz,bx,by,bz,smem; void**kernelParams; void**extra; } KNP;

typedef cbStatus (*ge_t)(void*,int,int,int,int,int,const void*,const void*,int,int,const void*,int,int,const void*,void*,int,int,int,int);
typedef cbStatus (*gs_t)(void*,void**);            /* cublasGetStream_v2 */
typedef int (*isc_t)(void*,int*);                  /* cudaStreamIsCapturing(stream,&status) */
typedef cbStatus (*inst_t)(void**,void*,unsigned long long);
typedef int (*getnodes_t)(void*,void**,size_t*);
typedef int (*gettype_t)(void*,int*);
typedef int (*getknp_t)(void*,KNP*);

#define MAXC 4096
static void* clog[MAXC]; static int cm[MAXC],cn[MAXC],ck_[MAXC],ccap[MAXC]; static long nclog=0;
static long inst_calls=0, tot_nodes=0, tot_kernel=0, tot_match=0;
static int dumped=0;

cbStatus cublasGemmEx(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,
                      const void*B,int Bt,int ldb,const void*be,void*C,int Ct,int ldc,int ct,int algo){
  static ge_t R=0; if(!R)R=(ge_t)cb("cublasGemmEx");
  static gs_t GS=0; if(!GS)GS=(gs_t)cb("cublasGetStream_v2");
  static isc_t ISC=0; if(!ISC)ISC=(isc_t)cu("cudaStreamIsCapturing");
  int cap=0; if(GS&&ISC){ void*s=0; GS(h,&s); int st=0; ISC(s,&st); cap=(st!=0); }
  long i=nclog%MAXC; clog[i]=C; cm[i]=m; cn[i]=n; ck_[i]=k; ccap[i]=cap; nclog++;
  return R(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,C,Ct,ldc,ct,algo);
}

static void flushf(){
  FILE*f=fopen("/home/ubuntu/cipher-fusion-evidence/ra_path1/probe_result.json","w");
  if(f){ fprintf(f,"{\"cublasGemmEx_logged\":%ld,\"instantiate_calls\":%ld,\"total_nodes\":%ld,"
    "\"total_kernel_nodes\":%ld,\"C_ptr_matches_in_nodes\":%ld}\n",nclog,inst_calls,tot_nodes,tot_kernel,tot_match); fclose(f); }
}

cbStatus cudaGraphInstantiateWithFlags(void**pExec, void*graph, unsigned long long flags){
  static inst_t R=0; if(!R)R=(inst_t)cu("cudaGraphInstantiateWithFlags");
  static getnodes_t GN=0; if(!GN)GN=(getnodes_t)cu("cudaGraphGetNodes");
  static gettype_t GT=0; if(!GT)GT=(gettype_t)cu("cudaGraphNodeGetType");
  static getknp_t GP=0; if(!GP)GP=(getknp_t)cu("cudaGraphKernelNodeGetParams");
  typedef int (*pga_t)(void*,const void*); static pga_t PGA=0; if(!PGA)PGA=(pga_t)cu("cudaPointerGetAttributes");
  if(GN&&GT&&GP){
    size_t num=0; GN(graph,NULL,&num); inst_calls++; tot_nodes+=num;
    long capcnt=0; for(long c=0;c<nclog&&c<MAXC;c++) if(ccap[c]) capcnt++;
    if(num>=32 && num<100000){     /* skip tiny torch sub-graphs; the decode graph is large */
      void** nodes=(void**)malloc(num*sizeof(void*)); GN(graph,nodes,&num);
      long kn=0, matches=0, p0nonnull=0; int ex=0;
      for(size_t j=0;j<num;j++){
        int type=-1; GT(nodes[j],&type);
        if(type!=0) continue;
        kn++;
        KNP p; memset(&p,0,sizeof(p));
        if(GP(nodes[j],&p)!=0 || !p.kernelParams) continue;
        void* a0=p.kernelParams[0]; if(!a0) continue; p0nonnull++;
        /* SAFE: only read the first 8 bytes of the first arg slot (a pointer-sized arg). If cuBLAS passes C
         * as a direct first arg, *a0==C. (A struct-by-value first arg would put C at an unknown offset that
         * we cannot scan without overreading past the slot => Part-A wall.) */
        char attrbuf[64]; uint64_t v0=0;
        int ok = PGA? (PGA((void*)attrbuf,a0)==0 || 1) : 1;   /* PGA tolerates host-unregistered; informational */
        (void)ok; memcpy(&v0,a0,8);
        for(long c=0;c<nclog&&c<MAXC;c++){
          if(clog[c] && (uint64_t)(uintptr_t)clog[c]==v0){
            matches++; if(ex<8){ex++; fprintf(stderr,"[probe] MATCH node#%zu firstArg==C(m=%d n=%d k=%d) cap=%d\n",j,cm[c],cn[c],ck_[c],ccap[c]);} break;
          }
        }
      }
      tot_kernel+=kn; tot_match+=matches;
      fprintf(stderr,"[probe] instantiate#%ld nodes=%zu kernel=%ld p0nonnull=%ld firstArg==C matches=%ld (loggedC=%ld cap=%ld)\n",
        inst_calls,num,kn,p0nonnull,matches,nclog,capcnt);
      free(nodes); flushf();
    } else {
      fprintf(stderr,"[probe] instantiate#%ld nodes=%zu (skipped scan; loggedC=%ld cap=%ld)\n",inst_calls,num,nclog,capcnt);
      flushf();
    }
  }
  return R(pExec,graph,flags);
}
__attribute__((destructor)) static void fin(){ flushf();
  fprintf(stderr,"[probe] FINAL gemmEx=%ld instantiate=%ld nodes=%ld kernel=%ld matches=%ld\n",nclog,inst_calls,tot_nodes,tot_kernel,tot_match); }
