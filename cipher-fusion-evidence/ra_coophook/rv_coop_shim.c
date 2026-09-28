/* R.A vLLM-COOP CAPTURE-HOOK detector shim (SCRATCH — not the anchor, not a vLLM source edit).
 * Extends fork-1 ra_realvllm/detector_shim.c for cudagraph hosting:
 *  - intercepts cublasGemmEx (LD_PRELOAD); when the call's stream IS CAPTURING and coop flags are armed,
 *    emits detector ops so they are recorded into vLLM's OWN graph (capture-time insertion, NOT post-hoc
 *    mutation = the Path-1 wall).
 *  - A1 in-frame: recompute D' (same args+algo) into preallocated scratch + axpy(-1) + nrm2 (fp16 result,
 *    DEVICE pointer mode) into res slot — captured behind every linear GEMM, runs every replay.
 *  - inject: tiny xor14 kernel (cubin via driver API) captured behind the target linear ordinal — flips
 *    bit-14 of C[0] every replay = persistent SDC, fork-1 fault class.
 *  - A2 sidecar: during vLLM's capture we only LOG the linear GEMM args (frozen pointers + copied fp32
 *    alpha/beta); rv_emit_sidecar(stream) later re-issues recompute+compare for all logged GEMMs on a
 *    python-owned capturing stream -> a separate tiny graph replayed every Nth step.
 *  - rv_read(): D2H readback of res slots; max residual + nonzero count + argmax slot.
 * Linear predicate identical to fork-1: k in {4096,14336} && m != VOCAB. T=0: clean recompute bit-identical.
 * Env: RV_INJECT (linear ordinal within capture, -1 off), RV_VOCAB (32000), RV_CUBIN (xor14.cubin path).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int cbStatus;
static void* hcb=0; static void* hcu=0; static void* hdrv=0;
static void* cbsym(const char*n){ if(!hcb) hcb=dlopen("libcublas.so.13",RTLD_NOW|RTLD_GLOBAL); void*p=hcb?dlsym(hcb,n):0; if(!p) fprintf(stderr,"[coop] FATAL no %s\n",n); return p; }
static void* cusym(const char*n){ if(!hcu) hcu=dlopen("libcudart.so.13",RTLD_NOW|RTLD_GLOBAL); void*p=hcu?dlsym(hcu,n):0; if(!p) fprintf(stderr,"[coop] FATAL no rt %s\n",n); return p; }
static void* drsym(const char*n){ if(!hdrv) hdrv=dlopen("libcuda.so.1",RTLD_NOW|RTLD_GLOBAL); void*p=hdrv?dlsym(hdrv,n):0; if(!p) fprintf(stderr,"[coop] FATAL no drv %s\n",n); return p; }

typedef cbStatus (*ge_t)(void*,int,int,int,int,int,const void*,const void*,int,int,const void*,int,int,const void*,void*,int,int,int,int);
typedef cbStatus (*axpy_t)(void*,int,const void*,int,const void*,int,int,void*,int,int,int);
typedef cbStatus (*nrm2_t)(void*,int,const void*,int,int,void*,int,int);
typedef cbStatus (*getstream_t)(void*,void**);
typedef cbStatus (*setstream_t)(void*,void*);
typedef cbStatus (*setpm_t)(void*,int);   /* cublasSetPointerMode_v2: 0 HOST, 1 DEVICE */
typedef cbStatus (*getpm_t)(void*,int*);
typedef int (*cumalloc_t)(void**,size_t);
typedef int (*cumemcpy_t)(void*,const void*,size_t,int);
typedef int (*cumemset_t)(void*,int,size_t);
typedef int (*iscap_t)(void*,int*);       /* cudaStreamIsCapturing(stream,&status) status!=0 => capturing */
typedef int (*cuML_t)(void**,const char*);              /* cuModuleLoad */
typedef int (*cuMGF_t)(void**,void*,const char*);       /* cuModuleGetFunction */
typedef int (*cuLK_t)(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**); /* cuLaunchKernel */

static ge_t R_ge=0; static axpy_t R_axpy=0; static nrm2_t R_nrm2=0;
static getstream_t R_getstream=0; static setstream_t R_setstream=0; static setpm_t R_setpm=0; static getpm_t R_getpm=0;
static cumalloc_t R_malloc=0; static cumemcpy_t R_memcpy=0; static cumemset_t R_memset=0; static iscap_t R_iscap=0;
static cuML_t R_ml=0; static cuMGF_t R_mgf=0; static cuLK_t R_lk=0;
static void* xor_fn=0;

/* ---- config/state ---- */
static int inited=0, INJ=-1, VOCAB=32000;
static volatile int f_log=0, f_checks=0, f_inject=0;  /* armed by python around vLLM's FULL capture */
static long cap_lin=0;                          /* linear ordinal within current capture pass */
static int slots_used=0, log_used=0;
static long emitted_checks=0, emitted_injects=0, ge_total=0, ge_capturing=0, sidecar_emits=0, reads=0;
static int last_ct=-1, bad_ct=0, scratch_overflow=0;

#define MAXSLOTS 4096
#define MAXLOG   4096
typedef struct { void*h; int ta,tb,m,n,k; const void*A; int At,lda; const void*B; int Bt,ldb; void*C; int Ct,ldc; int ct,algo; float alpha,beta; } gemlog_t;
static gemlog_t glog[MAXLOG];
static void* scratch=0; static size_t scratch_sz=0, scratch_off=0;
static uint16_t* d_res=0;                       /* device fp16 residual slots */
static uint16_t h_res[MAXSLOTS];

static float h2f(uint16_t h){
  uint32_t s=(uint32_t)(h&0x8000)<<16, e=(h>>10)&0x1f, m=h&0x3ff, f;
  if(e==0){ if(m==0) f=s; else { int ee=127-15+1; while(!(m&0x400)){m<<=1;ee--;} m&=0x3ff; f=s|((uint32_t)ee<<23)|(m<<13);} }
  else if(e==0x1f){ f=s|0x7f800000|(m<<13); }
  else { f=s|((e+112)<<23)|(m<<13); }
  float r; memcpy(&r,&f,4); return r;
}

static void init(void){
  inited=1;
  const char*s;
  if((s=getenv("RV_INJECT"))) INJ=atoi(s);
  if((s=getenv("RV_VOCAB"))) VOCAB=atoi(s);
  R_ge=(ge_t)cbsym("cublasGemmEx"); R_axpy=(axpy_t)cbsym("cublasAxpyEx"); R_nrm2=(nrm2_t)cbsym("cublasNrm2Ex");
  R_getstream=(getstream_t)cbsym("cublasGetStream_v2"); R_setstream=(setstream_t)cbsym("cublasSetStream_v2");
  R_setpm=(setpm_t)cbsym("cublasSetPointerMode_v2"); R_getpm=(getpm_t)cbsym("cublasGetPointerMode_v2");
  R_malloc=(cumalloc_t)cusym("cudaMalloc"); R_memcpy=(cumemcpy_t)cusym("cudaMemcpy"); R_memset=(cumemset_t)cusym("cudaMemset");
  R_iscap=(iscap_t)cusym("cudaStreamIsCapturing");
  R_ml=(cuML_t)drsym("cuModuleLoad"); R_mgf=(cuMGF_t)drsym("cuModuleGetFunction"); R_lk=(cuLK_t)drsym("cuLaunchKernel");
  /* preallocate BEFORE any capture (cudaMalloc inside capture is illegal) */
  scratch_sz = 256u*1024u*1024u;
  if(R_malloc(&scratch,scratch_sz)!=0){ scratch=0; scratch_sz=0; fprintf(stderr,"[coop] scratch alloc FAIL\n"); }
  if(R_malloc((void**)&d_res,MAXSLOTS*2)!=0){ d_res=0; fprintf(stderr,"[coop] res alloc FAIL\n"); }
  if(d_res && R_memset) R_memset(d_res,0,MAXSLOTS*2);
  if(INJ>=0 && R_ml && R_mgf){
    const char* cb=getenv("RV_CUBIN"); if(!cb) cb="/home/ubuntu/cipher-fusion-evidence/ra_coophook/xor14.cubin";
    void* mod=0; int rc=R_ml(&mod,cb);
    if(rc==0) rc=R_mgf(&xor_fn,mod,"xor14");
    fprintf(stderr,"[coop] xor cubin load rc=%d fn=%p\n",rc,xor_fn);
  }
  fprintf(stderr,"[coop] init INJ=%d VOCAB=%d scratch=%zuMB\n",INJ,VOCAB,scratch_sz>>20);
}

/* emit recompute+compare for one GEMM on the handle's current stream (must be capturing or sidecar-capturing).
 * Returns 0 ok. Uses live alpha/beta pointers when fromlog==0, logged copies when fromlog==1. */
static int emit_check(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,
                      const void*B,int Bt,int ldb,const void*be,void*C,int Ct,int ldc,int ct,int algo){
  if(!scratch || !d_res || slots_used>=MAXSLOTS) return -1;
  size_t need=(size_t)m*n*2;
  if(scratch_off+need>scratch_sz){ scratch_overflow++; return -2; }
  void* dst=(char*)scratch+scratch_off; scratch_off+=((need+255)&~255ul);
  cbStatus st=R_ge(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,dst,Ct,m,ct,algo);  /* D' recompute, same algo */
  if(st!=0){ fprintf(stderr,"[coop] recompute GemmEx st=%d (m=%d n=%d k=%d)\n",st,m,n,k); return st; }
  float negone=-1.0f; long ne=(long)m*n;
  R_axpy(h,(int)ne,&negone,0,C,2,1,dst,2,1,0);                 /* dst = D' - C  (alpha host fp32, HOST mode) */
  R_setpm(h,1);                                                /* DEVICE pointer mode for in-graph result */
  R_nrm2(h,(int)ne,dst,2,1,d_res+slots_used,2,0);              /* ||D'-C|| fp16 -> res slot (device) */
  R_setpm(h,0);
  slots_used++; emitted_checks++;
  return 0;
}

cbStatus cublasGemmEx(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,
                      const void*B,int Bt,int ldb,const void*be,void*C,int Ct,int ldc,int ct,int algo){
  if(!inited) init();
  cbStatus st=R_ge(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,C,Ct,ldc,ct,algo);
  ge_total++;
  if(!f_log && !f_checks && !f_inject) return st;  /* act ONLY inside the python-armed FULL-capture window */
  int is_linear = (k==4096 || k==14336) && (m!=VOCAB);
  if(!is_linear) return st;
  void* strm=0; int cap=0;
  if(R_getstream && R_iscap){ R_getstream(h,&strm); R_iscap(strm,&cap); }
  if(!cap) return st;                       /* only act when vLLM is CAPTURING (eager warmup passes through) */
  ge_capturing++;
  long ord = cap_lin++;
  last_ct=ct; if(ct!=68) bad_ct++;          /* expect CUBLAS_COMPUTE_32F==68; sidecar stores fp32 alpha/beta */
  /* persistent inject: xor14(C) captured behind the target ordinal -> fires every replay of THIS graph.
   * Armed via f_inject so it can be captured into BOTH the vanilla and checked graphs (a true persistent
   * fault present on every step, not only the periodic checked step). */
  if(INJ>=0 && f_inject && ord==INJ && xor_fn && R_lk){
    void* params[1]={&C}; void* kp[1]; kp[0]=&params[0]; /* kernelParams: array of ptrs to args */
    void* args[1]={(void*)&C};
    int rc=R_lk(xor_fn,1,1,1,1,1,1,0,strm,args,0);
    fprintf(stderr,"[coop] INJ captured at ord=%ld m=%d n=%d rc=%d\n",ord,m,n,rc);
    if(rc==0) emitted_injects++;
    (void)params; (void)kp;
  }
  if(f_log && log_used<MAXLOG){              /* A2: log frozen args for the sidecar */
    gemlog_t* g=&glog[log_used++];
    g->h=h; g->ta=ta; g->tb=tb; g->m=m; g->n=n; g->k=k; g->A=A; g->At=At; g->lda=lda;
    g->B=B; g->Bt=Bt; g->ldb=ldb; g->C=C; g->Ct=Ct; g->ldc=ldc; g->ct=ct; g->algo=algo;
    g->alpha = al? *(const float*)al : 1.0f;  g->beta = be? *(const float*)be : 0.0f;
  }
  if(f_checks) emit_check(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,C,Ct,ldc,ct,algo);  /* A1 in-frame */
  return st;
}

/* ---------------- python-facing controls (ctypes) ---------------- */
void rv_arm(int log_args,int inframe_checks){ f_log=log_args; f_checks=inframe_checks; cap_lin=0; if(log_args) log_used=0; }
void rv_set_inject(int on){ f_inject=on; }   /* arm xor14 capture independently of log/checks */
void rv_reset_slots(void){ slots_used=0; scratch_off=0; }
int  rv_log_count(void){ return log_used; }
int  rv_slot_count(void){ return slots_used; }
long rv_cap_linears(void){ return cap_lin; }

/* emit the whole logged check sequence on the given (capturing) stream -> sidecar graph body */
int rv_emit_sidecar(void* stream){
  if(!log_used) return -1;
  void* h=glog[0].h; void* saved=0; R_getstream(h,&saved); R_setstream(h,stream);
  int err=0;
  for(int i=0;i<log_used;i++){ gemlog_t* g=&glog[i];
    if(emit_check(g->h,g->ta,g->tb,g->m,g->n,g->k,&g->alpha,g->A,g->At,g->lda,g->B,g->Bt,g->ldb,&g->beta,g->C,g->Ct,g->ldc,g->ct,g->algo)!=0) err++;
  }
  R_setstream(h,saved); sidecar_emits++;
  return err;
}

/* D2H readback of res slots: out[0]=max residual, out[1]=nonzero count, out[2]=argmax slot */
void rv_read(double* out){
  out[0]=0; out[1]=0; out[2]=-1;
  if(!d_res || !slots_used){ return; }
  R_memcpy(h_res,d_res,(size_t)slots_used*2,2 /*D2H, legacy stream = device sync*/);
  reads++;
  double mx=0; long nz=0; int am=-1;
  for(int i=0;i<slots_used;i++){ double r=h2f(h_res[i]); if(r!=0.0||r!=r){ nz++; if(!(r<=mx)){ mx=r; am=i; } } }
  out[0]=mx; out[1]=(double)nz; out[2]=(double)am;
}

void rv_stats(char* buf,int len){
  snprintf(buf,len,"{\"ge_total\":%ld,\"ge_capturing\":%ld,\"log_used\":%d,\"slots_used\":%d,"
    "\"emitted_checks\":%ld,\"emitted_injects\":%ld,\"sidecar_emits\":%ld,\"reads\":%ld,"
    "\"last_computeType\":%d,\"bad_ct\":%d,\"scratch_overflow\":%d,\"inject_ordinal\":%d}",
    ge_total,ge_capturing,log_used,slots_used,emitted_checks,emitted_injects,sidecar_emits,reads,
    last_ct,bad_ct,scratch_overflow,INJ);
}
