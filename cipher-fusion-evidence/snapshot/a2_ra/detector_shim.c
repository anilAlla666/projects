/* FORK-1 detector as a SCRATCH LD_PRELOAD cuBLAS-interception shim (NOT the anchor, NOT a vLLM edit).
 * Intercepts cublasGemmEx (vLLM's fp16 linear path). Every N decode steps, re-runs each linear GEMM with an
 * INDEPENDENT cublasGemmEx (same args+algo) into a scratch buffer and compares D'-D via cuBLAS axpy+nrm2 (T=0:
 * clean recompute is bit-identical -> residual exactly 0). Step boundary = lm_head GEMM (m==VOCAB).
 * Injection mode (RV_INJECT=<layer-ordinal>): flips bit-14 of one output element EVERY step (persistent SDC).
 * Counters -> file at exit. Real symbols resolved via dlopen of the version-tagged libs.
 *
 * Env: RV_N (check period, default 16); RV_INJECT (gemm-ordinal-within-step to corrupt persistently, -1=off);
 *      RV_VOCAB (step delimiter m, default 32000 Mistral); RV_OUT (counts file path).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef int cbStatus;
/* ---- real symbol resolution (dlopen, version-tagged) ---- */
static void* hcb=0; static void* hcu=0;
static void* cbsym(const char*n){ if(!hcb) hcb=dlopen("libcublas.so.13",RTLD_NOW|RTLD_GLOBAL); void*p=hcb?dlsym(hcb,n):0; if(!p) fprintf(stderr,"[det] FATAL no %s\n",n); return p; }
static void* cusym(const char*n){ if(!hcu) hcu=dlopen("libcudart.so.13",RTLD_NOW|RTLD_GLOBAL); void*p=hcu?dlsym(hcu,n):0; return p; }
typedef cbStatus (*ge_t)(void*,int,int,int,int,int,const void*,const void*,int,int,const void*,int,int,const void*,void*,int,int,int,int);
typedef cbStatus (*axpy_t)(void*,int,const void*,int,const void*,int,int,void*,int,int,int);
typedef cbStatus (*nrm2_t)(void*,int,const void*,int,int,void*,int,int);
typedef int (*cumalloc_t)(void**,size_t);
typedef int (*cumemcpy_t)(void*,const void*,size_t,int);  /* cudaMemcpy(dst,src,n,kind) */
typedef cbStatus (*getstream_t)(void*,void**);            /* cublasGetStream_v2(handle, &stream) */
typedef int (*ssync_t)(void*);                            /* cudaStreamSynchronize(stream) */
static ge_t R_ge=0; static axpy_t R_axpy=0; static nrm2_t R_nrm2=0; static cumalloc_t R_malloc=0; static cumemcpy_t R_memcpy=0;
static getstream_t R_getstream=0; static ssync_t R_ssync=0;

/* ---- config + counters ---- */
static int N=16, INJ=-1, VOCAB=32000, GPS=128, inited=0;
static long step=0, gidx=0, lin=0, checks=0, gchecked=0, detections=0, injected=0;
static long gtot=0; static int maxm=0, maxn=0; static long n_smalln=0;  /* diag: total GEMMs, max m/n, decode-ish(n<=64) count */
static double max_clean_resid=0.0, max_det_resid=0.0;
static void* scratch=0; static size_t scratch_sz=0;
static uint16_t* h_res=0;  /* fp16 nrm2 result (host); fp16 is the only valid resultType for fp16 input */
static char OUTP[512];
static float h2f(uint16_t h){  /* IEEE half -> float */
  uint32_t s=(uint32_t)(h&0x8000)<<16, e=(h>>10)&0x1f, m=h&0x3ff, f;
  if(e==0){ if(m==0) f=s; else { int ee=127-15+1; while(!(m&0x400)){m<<=1;ee--;} m&=0x3ff; f=s|((uint32_t)ee<<23)|(m<<13);} }
  else if(e==0x1f){ f=s|0x7f800000|(m<<13); }
  else { f=s|((e+112)<<23)|(m<<13); }
  float r; memcpy(&r,&f,4); return r;
}

static void flush(){
  char buf[512];
  snprintf(buf,512,"{\"N\":%d,\"inject_gemm\":%d,\"steps\":%ld,\"check_steps\":%ld,\"gemms_checked\":%ld,"
    "\"detections\":%ld,\"injected_flips\":%ld,\"max_clean_residual\":%.6g,\"max_detect_residual\":%.6g,"
    "\"gemms_total\":%ld,\"max_m\":%d,\"max_n\":%d,\"smalln_gemms\":%ld}",
    N,INJ,step,checks,gchecked,detections,injected,max_clean_resid,max_det_resid,gtot,maxm,maxn,n_smalln);
  fprintf(stderr,"[det-COUNTS] %s\n",buf);            /* stderr is captured (survives SIGKILL of EngineCore) */
  FILE*f=fopen(OUTP,"w"); if(f){ fprintf(f,"%s\n",buf); fclose(f); }
}
static void init(){
  inited=1;
  const char*s;
  if((s=getenv("RV_N"))) N=atoi(s);
  if((s=getenv("RV_INJECT"))) INJ=atoi(s);
  if((s=getenv("RV_VOCAB"))) VOCAB=atoi(s);
  if((s=getenv("RV_GPS"))) GPS=atoi(s);
  if((s=getenv("RV_OUT"))) strncpy(OUTP,s,511); else strcpy(OUTP,"/home/ubuntu/cipher-fusion-evidence/ra_realvllm/detector_counts.json");
  R_ge=(ge_t)cbsym("cublasGemmEx"); R_axpy=(axpy_t)cbsym("cublasAxpyEx"); R_nrm2=(nrm2_t)cbsym("cublasNrm2Ex");
  R_malloc=(cumalloc_t)cusym("cudaMalloc"); R_memcpy=(cumemcpy_t)cusym("cudaMemcpy");
  R_getstream=(getstream_t)cbsym("cublasGetStream_v2"); R_ssync=(ssync_t)cusym("cudaStreamSynchronize");
  h_res=(uint16_t*)malloc(sizeof(uint16_t));
  fprintf(stderr,"[det] init N=%d INJECT=%d VOCAB=%d\n",N,INJ,VOCAB);
}

cbStatus cublasGemmEx(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,
                      const void*B,int Bt,int ldb,const void*be,void*C,int Ct,int ldc,int ct,int algo){
  if(!inited) init();
  if(!R_ge) R_ge=(ge_t)cbsym("cublasGemmEx");
  /* 1) the real forward GEMM */
  cbStatus st=R_ge(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,C,Ct,ldc,ct,algo);
  gtot++; if(m>maxm)maxm=m; if(n>maxn)maxn=n; if(n<=64)n_smalln++;
  if(gtot<=24 || (n<=64 && n_smalln<=24)) fprintf(stderr,"[det-shape] g=%ld m=%d n=%d k=%d ldc=%d\n",gtot,m,n,k,ldc);

  int is_linear = (k==4096 || k==14336) && (m!=VOCAB);  /* Mistral q/k/v/o/gate/up/down projections via GemmEx */
  if(!is_linear) return st;                              /* attention is FlashAttn (not cuBLAS); lm_head not GemmEx */

  /* step structure from linear-GEMM count: GPS linear GEMMs per forward (Mistral 32 layers x 4 fused) */
  long this_g = lin % GPS;          /* ordinal within the step */
  long cur_step = lin / GPS;
  int is_check = (cur_step % N)==0;

  /* 2) persistent injection: flip bit-14 of C[0] on the targeted gemm ordinal every step.
   *    MUST sync the handle's stream first: the GEMM is async, else the late GEMM write overwrites the flip. */
  if(INJ>=0 && this_g==INJ && R_memcpy && R_getstream && R_ssync){
    void* strm=0; R_getstream(h,&strm); R_ssync(strm);
    uint16_t hv; R_memcpy(&hv, C, 2, 2 /*D2H*/); uint16_t before=hv; hv ^= (1u<<14); R_memcpy(C, &hv, 2, 1 /*H2D*/);
    static int idbg=0; if(idbg<4){ idbg++; uint16_t chk=0; R_memcpy(&chk,C,2,2);
      fprintf(stderr,"[det-INJ] gidx=%ld m=%d n=%d C[0] %04x->%04x readback=%04x\n",this_g,m,n,before,hv,chk); }
    injected++;
  }

  /* 3) detector: on check steps, recompute each linear GEMM independently and compare (T=0) */
  if(is_check && (ldc==m) && R_axpy && R_nrm2 && R_malloc){
    size_t need=(size_t)m*n*2;
    if(need>scratch_sz){ R_malloc(&scratch,need); scratch_sz=need; }
    if(scratch){
      R_ge(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,scratch,Ct,m,ct,algo);  /* independent recompute D' */
      float negone=-1.0f; long ne=(long)m*n;
      R_axpy(h,(int)ne,&negone,0,C,2,1,scratch,2,1,0);   /* scratch = D' - C (alpha fp32 host) */
      *h_res=0; R_nrm2(h,(int)ne,scratch,2,1,h_res,2,0);  /* ||D'-C||: x=16F result=16F exec=32F; host result -> syncs */
      double r=h2f(*h_res); int hit=(r>0.0)||(r!=r); gchecked++;
      static int dbg=0;
      if(this_g==INJ && INJ>=0 && dbg<4){ dbg++;
        uint16_t c0=0,d0=0; if(R_memcpy){R_memcpy(&c0,C,2,2);R_memcpy(&d0,scratch,2,2);}
        fprintf(stderr,"[det-DBG] inj-gemm check: r=%.6g C[0]=0x%04x Dprime[0]=0x%04x ne=%ld\n",r,c0,d0,ne);
      }
      if(this_g==INJ && INJ>=0){ if(r>max_det_resid)max_det_resid=r; if(hit)detections++; }
      else { if(r>max_clean_resid)max_clean_resid=r; if(hit)detections++; }  /* clean GEMMs: any hit = FALSE POSITIVE */
    }
  }

  lin++;
  if(lin % GPS == 0){ step=lin/GPS; if(((step-1)%N)==0) checks++; flush(); }  /* step boundary + flush */
  return st;
}

__attribute__((destructor)) static void fin(void){
  if(!inited) return;
  FILE*f=fopen(OUTP,"w");
  if(f){ fprintf(f,"{\"N\":%d,\"inject_gemm\":%d,\"steps\":%ld,\"check_steps\":%ld,\"gemms_checked\":%ld,"
    "\"detections\":%ld,\"injected_flips\":%ld,\"max_clean_residual\":%.6g,\"max_detect_residual\":%.6g}\n",
    N,INJ,step,checks,gchecked,detections,injected,max_clean_resid,max_det_resid); fclose(f); }
  fprintf(stderr,"[det] FINAL steps=%ld checks=%ld gchecked=%ld detections=%ld injected=%ld clean_resid_max=%.4g det_resid_max=%.4g\n",
    step,checks,gchecked,detections,injected,max_clean_resid,max_det_resid);
}
