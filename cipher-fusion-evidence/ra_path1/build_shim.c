/* PATH-1 BUILD: inject an in-graph ABFT check (and optional fault) into the REAL vLLM decode cudagraph.
 * Capture: log the TARGET linear GEMM's output C (+ x=B, W=A, m,n,k) live from cublasGemmEx (frozen buffers).
 * Instantiate seam: into the big decode graph, add [optional inject_flip(C)] + abft_check(C,B,wcol,m,n,K,flag)
 * as kernel nodes depending on the graph leaves (run at step-end in replay). wcol=colsum(W) computed once,
 * out-of-graph. Replay: cudaGraphLaunch hook counts steps; every N syncs+reads+clears the device flag (host
 * periodic gate, no in-graph host sync). T from clean residual. NOT the anchor, no vLLM source edit.
 * Env: RV_N (default 45), RV_INJECT (1=add fault node), RV_TARGETK (target GEMM k, default 14336=down_proj). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef int cbStatus; typedef int cudaError; typedef int CUresult;
static void *hcb=0,*hrt=0,*hdrv=0;
static void* CB(const char*n){ if(!hcb)hcb=dlopen("libcublas.so.13",RTLD_NOW|RTLD_GLOBAL); return hcb?dlsym(hcb,n):0; }
static void* RT(const char*n){ if(!hrt)hrt=dlopen("libcudart.so.13",RTLD_NOW|RTLD_GLOBAL); return hrt?dlsym(hrt,n):0; }
static void* DR(const char*n){ if(!hdrv)hdrv=dlopen("libcuda.so.1",RTLD_NOW|RTLD_GLOBAL); return hdrv?dlsym(hdrv,n):0; }

/* driver CUDA_KERNEL_NODE_PARAMS v1 (56B): func, gridXYZ, blockXYZ, smem, kernelParams, extra */
typedef struct { void* func; unsigned gx,gy,gz,bx,by,bz,smem; void** kernelParams; void** extra; } KNP;
/* runtime cudaKernelNodeParams mirror (for GetType skip) — we only need GetNodes/GetEdges */

typedef cbStatus (*ge_t)(void*,int,int,int,int,int,const void*,const void*,int,int,const void*,int,int,const void*,void*,int,int,int,int);
typedef cudaError (*inst_t)(void**,void*,unsigned long long);
typedef cudaError (*glaunch_t)(void*,void*);
typedef cudaError (*getnodes_t)(void*,void**,size_t*);
typedef cudaError (*getedges_t)(void*,void**,void**,size_t*);
typedef cudaError (*malloc_t)(void**,size_t);
typedef cudaError (*memcpy_t)(void*,const void*,size_t,int);
typedef cudaError (*memset_t)(void*,int,size_t);
typedef cudaError (*ssync_t)(void);
typedef CUresult (*modload_t)(void**,const void*);
typedef CUresult (*modfunc_t)(void**,void*,const char*);
typedef CUresult (*addkn_t)(void**,void*,const void**,size_t,const KNP*);
typedef CUresult (*launchk_t)(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**);
typedef CUresult (*ctxsync_t)(void);

static int N=45, INJECT=0, TARGETK=14336, inited=0;
static void *tC=0,*tW=0,*tX=0; static int tm=0,tn=0,tk=0; static int have_target=0, instrumented=0, bigseen=0, SKIPBIG=0;
static void *mod=0,*f_colsum=0,*f_abft=0,*f_inject=0; static float* d_wcol=0; static float* d_flag=0;
static long step=0, reads=0, detections=0; static double max_resid=0, last_resid=0;
static char OUTP[300];

static void flushf(){ FILE*f=fopen(OUTP,"w"); if(f){ fprintf(f,
  "{\"N\":%d,\"inject\":%d,\"target_k\":%d,\"target_mnk\":[%d,%d,%d],\"instrumented\":%d,\"steps\":%ld,"
  "\"flag_reads\":%ld,\"detections\":%ld,\"max_residual\":%.6g,\"last_residual\":%.6g}\n",
  N,INJECT,TARGETK,tm,tn,tk,instrumented,step,reads,detections,max_resid,last_resid); fclose(f);} }

static void init(){ inited=1; const char*s;
  if((s=getenv("RV_N")))N=atoi(s); if((s=getenv("RV_INJECT")))INJECT=atoi(s);
  if((s=getenv("RV_TARGETK")))TARGETK=atoi(s); if((s=getenv("RV_SKIPBIG")))SKIPBIG=atoi(s);
  if((s=getenv("RV_OUT")))strncpy(OUTP,s,299); else strcpy(OUTP,"/home/ubuntu/cipher-fusion-evidence/ra_path1/build_result.json");
  fprintf(stderr,"[p1] init N=%d inject=%d targetK=%d\n",N,INJECT,TARGETK); }

cbStatus cublasGemmEx(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,
                      const void*B,int Bt,int ldb,const void*be,void*C,int Ct,int ldc,int ct,int algo){
  static ge_t R=0; if(!R)R=(ge_t)CB("cublasGemmEx"); if(!inited)init();
  cbStatus st=R(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,C,Ct,ldc,ct,algo);
  if(k==TARGETK && !instrumented){ tC=C; tW=(void*)A; tX=(void*)B; tm=m; tn=n; tk=k; have_target=1; } /* latest target this capture */
  return st;
}

cudaError cudaGraphInstantiateWithFlags(void**pExec, void*graph, unsigned long long flags){
  static inst_t R=0; if(!R)R=(inst_t)RT("cudaGraphInstantiateWithFlags");
  static getnodes_t GN=0; if(!GN)GN=(getnodes_t)RT("cudaGraphGetNodes");
  static getedges_t GE=0; if(!GE)GE=(getedges_t)RT("cudaGraphGetEdges");
  static malloc_t MAL=0; if(!MAL)MAL=(malloc_t)RT("cudaMalloc");
  static memset_t MS=0; if(!MS)MS=(memset_t)RT("cudaMemset");
  static ssync_t SS=0; if(!SS)SS=(ssync_t)RT("cudaDeviceSynchronize");
  static modload_t ML=0; if(!ML)ML=(modload_t)DR("cuModuleLoadData");
  static modfunc_t MF=0; if(!MF)MF=(modfunc_t)DR("cuModuleGetFunction");
  static addkn_t AK=0; if(!AK)AK=(addkn_t)DR("cuGraphAddKernelNode");
  static launchk_t LK=0; if(!LK)LK=(launchk_t)DR("cuLaunchKernel");
  size_t num=0; if(GN)GN(graph,NULL,&num);
  if(num>=64){ bigseen++; if(bigseen<=SKIPBIG){ return R(pExec,graph,flags); } }
  if(num>=64 && have_target && !instrumented && AK){
    if(!mod){ char*img=0; FILE*cf=fopen("/home/ubuntu/cipher-fusion-evidence/ra_path1/path1_kernels.cubin","rb");
      if(cf){ fseek(cf,0,SEEK_END); long sz=ftell(cf); fseek(cf,0,SEEK_SET); img=malloc(sz); fread(img,1,sz,cf); fclose(cf);
        ML(&mod,img); MF(&f_colsum,mod,"colsum_W"); MF(&f_abft,mod,"abft_check"); MF(&f_inject,mod,"inject_flip"); }
      MAL((void**)&d_wcol,(size_t)tk*4); MAL((void**)&d_flag,4); if(MS)MS(d_flag,0,4); }
    /* wcol = colsum(W) once, OUT of graph */
    if(!getenv("RV_SKIPCOLSUM")){ void* args[]={&tW,&d_wcol,&tm,&tk}; unsigned grid=(tk+255)/256; LK(f_colsum,grid,1,1,256,1,1,0,0,args,0); if(SS)SS(); }
    /* leaves of the graph */
    void** nodes=malloc(num*sizeof(void*)); GN(graph,nodes,&num);
    size_t ne=0; GE(graph,NULL,NULL,&ne); void**from=malloc((ne?ne:1)*8),**to=malloc((ne?ne:1)*8);
    if(ne)GE(graph,from,to,&ne);
    void* leaves[1024]; int nl=0;
    for(size_t i=0;i<num&&nl<1024;i++){ int src=0; for(size_t e=0;e<ne;e++) if(from[e]==nodes[i]){src=1;break;} if(!src)leaves[nl++]=nodes[i]; }
    void* lastnode=0;
    if(INJECT){ KNP ip; memset(&ip,0,sizeof(ip)); ip.func=f_inject; ip.gx=ip.gy=ip.gz=1; ip.bx=1; ip.by=ip.bz=1;
      void* iargs[]={&tC}; ip.kernelParams=iargs; void* inode=0;
      CUresult r=AK(&inode,graph,(const void**)leaves,nl,&ip);
      fprintf(stderr,"[p1] addInject rc=%d\n",(int)r); lastnode=inode; }
    if(getenv("RV_MEMSET")){
      typedef struct { void* dst; size_t pitch; unsigned value; unsigned elementSize; size_t width; size_t height; } MSP;
      typedef CUresult (*addms_t)(void**,void*,const void**,size_t,const void*,void*);
      addms_t AMS=(addms_t)DR("cuGraphAddMemsetNode");
      MSP mp; memset(&mp,0,sizeof(mp)); mp.dst=d_flag; mp.value=0x00C1; mp.elementSize=4; mp.width=1; mp.height=1; mp.pitch=4;
      void* mnode=0; CUresult rm = AMS? AMS(&mnode,graph,(const void**)leaves,nl,&mp, /*ctx*/0) : -1;
      fprintf(stderr,"[p1] addMemset rc=%d (gap6-style primitive into REAL vLLM graph)\n",(int)rm);
      instrumented=1; flushf(); free(nodes);free(from);free(to); return R(pExec,graph,flags);
    }
    KNP cp; memset(&cp,0,sizeof(cp)); cp.func=f_abft; cp.gx=tn; cp.gy=cp.gz=1; cp.bx=256; cp.by=cp.bz=1;
    void* cargs[]={&tC,&tX,&d_wcol,&tm,&tn,&tk,&d_flag}; cp.kernelParams=cargs; void* cnode=0;
    const void* cdep[1]; size_t ncd; if(lastnode){cdep[0]=lastnode;ncd=1;} else {ncd=0;}
    CUresult r2 = lastnode ? AK(&cnode,graph,cdep,1,&cp) : AK(&cnode,graph,(const void**)leaves,nl,&cp);
    fprintf(stderr,"[p1] addCheck rc=%d  target m=%d n=%d k=%d leaves=%d nodes=%zu\n",(int)r2,tm,tn,tk,nl,num);
    instrumented=1; flushf(); free(nodes);free(from);free(to);
  }
  return R(pExec,graph,flags);
}

cudaError cudaGraphLaunch(void* exec, void* stream){
  static glaunch_t R=0; if(!R)R=(glaunch_t)RT("cudaGraphLaunch");
  static memcpy_t MC=0; if(!MC)MC=(memcpy_t)RT("cudaMemcpy");
  static memset_t MS=0; if(!MS)MS=(memset_t)RT("cudaMemset");
  static ssync_t SS2=0; if(!SS2)SS2=(ssync_t)RT("cudaDeviceSynchronize");
  cudaError st=R(exec,stream);
  if(instrumented && d_flag){ step++;
    if(step%N==0){ if(SS2)SS2(); float v=0; if(MC)MC(&v,d_flag,4,2); last_resid=v; if(v>max_resid)max_resid=v;
      reads++; /* T set post-hoc; detection = v>0 (clean baseline measures the floor) */ if(v>0)detections++;
      if(MS)MS(d_flag,0,4); flushf(); }
  }
  return st;
}
__attribute__((destructor)) static void fin(){ if(inited){flushf();
  fprintf(stderr,"[p1] FINAL instrumented=%d steps=%ld reads=%ld detections=%ld max_resid=%.4g\n",instrumented,step,reads,detections,max_resid);} }
