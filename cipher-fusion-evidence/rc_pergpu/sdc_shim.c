/* R.C v1 SDC signal source — a SCRATCH LD_PRELOAD cuBLAS-interception shim.
 * COPIED + extended from ra_realvllm/detector_shim.c (which is UNCHANGED). NOT the frozen anchor,
 * NOT a vLLM source edit. New work lives only in rc_pergpu/.
 *
 * Extensions over the R.A detector (advisor items 1,4,6):
 *   - RV_ONSET: inject the persistent SDC only when cur_step >= ONSET (default 0). This produces a
 *     real clean->DEGRADED transition inside ONE run: residual is exactly 0 before ONSET, > 0 after.
 *   - Per-check EVENT EMISSION: on every check step, for the WATCHED gemm-ordinal, append one JSONL
 *     record to RV_EVENTS with {CLOCK_MONOTONIC ns, CLOCK_REALTIME ns, step, gemm_ordinal, k, residual,
 *     is_detection, pid, tid, gpu}. The monitor fuses these by timestamp (post-hoc, no FIFO).
 *   - gpu id (cudaGetDevice), pid (getpid), tid (gettid) captured for the per-GPU verdict descriptor.
 *
 * Detection mechanism is identical to R.A: intercept cublasGemmEx; every N steps re-run each linear
 * GEMM with an INDEPENDENT cublasGemmEx (same args+algo) into scratch and compare ||D'-C|| via
 * axpy+nrm2 (T=0: clean recompute is bit-identical -> residual exactly 0).
 *
 * Env: RV_N (check period, default 8); RV_INJECT (gemm-ordinal to corrupt persistently, -1=off);
 *      RV_ONSET (first cur_step at which injection activates, default 0);
 *      RV_WATCH (gemm-ordinal whose per-check residual is logged to RV_EVENTS; default = INJECT if>=0 else 3);
 *      RV_VOCAB (step delimiter m, default 32000); RV_GPS (linear GEMMs per forward, default 128);
 *      RV_OUT (counts json); RV_EVENTS (per-check JSONL event stream).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <fcntl.h>
typedef int cbStatus;
/* ---- real symbol resolution (dlopen, version-tagged) ---- */
static void* hcb=0; static void* hcu=0;
static void* cbsym(const char*n){ if(!hcb) hcb=dlopen("libcublas.so.13",RTLD_NOW|RTLD_GLOBAL); void*p=hcb?dlsym(hcb,n):0; if(!p) fprintf(stderr,"[rc-det] FATAL no %s\n",n); return p; }
static void* cusym(const char*n){ if(!hcu) hcu=dlopen("libcudart.so.13",RTLD_NOW|RTLD_GLOBAL); void*p=hcu?dlsym(hcu,n):0; return p; }
typedef cbStatus (*ge_t)(void*,int,int,int,int,int,const void*,const void*,int,int,const void*,int,int,const void*,void*,int,int,int,int);
typedef cbStatus (*axpy_t)(void*,int,const void*,int,const void*,int,int,void*,int,int,int);
typedef cbStatus (*nrm2_t)(void*,int,const void*,int,int,void*,int,int);
typedef int (*cumalloc_t)(void**,size_t);
typedef int (*cumemcpy_t)(void*,const void*,size_t,int);
typedef cbStatus (*getstream_t)(void*,void**);
typedef int (*ssync_t)(void*);
typedef int (*getdev_t)(int*);
static ge_t R_ge=0; static axpy_t R_axpy=0; static nrm2_t R_nrm2=0; static cumalloc_t R_malloc=0; static cumemcpy_t R_memcpy=0;
static getstream_t R_getstream=0; static ssync_t R_ssync=0; static getdev_t R_getdev=0;

/* ---- config + counters ---- */
static int N=8, INJ=-1, ONSET=0, WATCH=-1, VOCAB=32000, GPS=128, inited=0;
static int PID=0, TID=0, gpu_id=-1;
static long step=0, lin=0, checks=0, gchecked=0, detections=0, injected=0;
static long gtot=0; static int maxm=0, maxn=0; static long n_smalln=0;
static double max_clean_resid=0.0, max_det_resid=0.0;
static void* scratch=0; static size_t scratch_sz=0;
static uint16_t* h_res=0;
static char OUTP[512]; static char EVP[512];
static float h2f(uint16_t h){
  uint32_t s=(uint32_t)(h&0x8000)<<16, e=(h>>10)&0x1f, m=h&0x3ff, f;
  if(e==0){ if(m==0) f=s; else { int ee=127-15+1; while(!(m&0x400)){m<<=1;ee--;} m&=0x3ff; f=s|((uint32_t)ee<<23)|(m<<13);} }
  else if(e==0x1f){ f=s|0x7f800000|(m<<13); }
  else { f=s|((e+112)<<23)|(m<<13); }
  float r; memcpy(&r,&f,4); return r;
}
static long long mono_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (long long)t.tv_sec*1000000000LL+t.tv_nsec; }
static long long wall_ns(void){ struct timespec t; clock_gettime(CLOCK_REALTIME,&t); return (long long)t.tv_sec*1000000000LL+t.tv_nsec; }

/* W.6 cohort-registry heartbeat from the GPU-holding worker process (where the GEMMs/shim run), so the
 * registered tgid == the SDC-observer pid == the NVML compute-procs pid (one source of truth). Throttled
 * to <=1 ioctl/sec at step boundaries. CIPHER_COHORT_QUERY = _IOWR('C',31, struct cipher_cohort_query[2064]).
 * caller_fingerprint != 0 => heartbeat-or-insert current->tgid; query result ignored here. */
static int cohort_fd=-1; static long long last_hb=0; static uint64_t FP=0; static void* hbq=0;
static void heartbeat(void){
  long long now=mono_ns();
  if(now-last_hb < 1000000000LL) return;     /* >=1s */
  last_hb=now;
  if(cohort_fd<0){ cohort_fd=open("/dev/cipher",O_RDWR); if(cohort_fd<0) return; }
  if(!hbq){ hbq=calloc(1,2064); if(!hbq) return; }
  *(uint64_t*)hbq = FP;                       /* caller_fingerprint */
  *(uint32_t*)((char*)hbq+8) = 128;           /* max_entries */
  ioctl(cohort_fd,(unsigned long)0xc810431fUL,hbq);  /* _IOWR('C',31,2064) — verified */
}

static void emit_event(long cur_step,int gord,int k,double r,int isdet){
  if(!EVP[0]) return;
  FILE*f=fopen(EVP,"a"); if(!f) return;
  fprintf(f,"{\"mono_ns\":%lld,\"wall_ns\":%lld,\"step\":%ld,\"gemm_ordinal\":%d,\"k\":%d,"
            "\"residual\":%.6g,\"is_detection\":%d,\"pid\":%d,\"tid\":%d,\"gpu\":%d}\n",
          mono_ns(),wall_ns(),cur_step,gord,k,r,isdet,PID,TID,gpu_id);
  fclose(f);
}

static void flush(){
  char buf[640];
  snprintf(buf,640,"{\"N\":%d,\"inject_gemm\":%d,\"onset\":%d,\"watch\":%d,\"steps\":%ld,\"check_steps\":%ld,"
    "\"gemms_checked\":%ld,\"detections\":%ld,\"injected_flips\":%ld,\"max_clean_residual\":%.6g,"
    "\"max_detect_residual\":%.6g,\"gemms_total\":%ld,\"pid\":%d,\"gpu\":%d}",
    N,INJ,ONSET,WATCH,step,checks,gchecked,detections,injected,max_clean_resid,max_det_resid,gtot,PID,gpu_id);
  fprintf(stderr,"[rc-det-COUNTS] %s\n",buf);
  FILE*f=fopen(OUTP,"w"); if(f){ fprintf(f,"%s\n",buf); fclose(f); }
}
static void init(){
  inited=1;
  const char*s;
  if((s=getenv("RV_N"))) N=atoi(s);
  if((s=getenv("RV_INJECT"))) INJ=atoi(s);
  if((s=getenv("RV_ONSET"))) ONSET=atoi(s);
  if((s=getenv("RV_WATCH"))) WATCH=atoi(s); else WATCH=(INJ>=0?INJ:3);
  if((s=getenv("RV_VOCAB"))) VOCAB=atoi(s);
  if((s=getenv("RV_GPS"))) GPS=atoi(s);
  if((s=getenv("RV_OUT"))) strncpy(OUTP,s,511); else strcpy(OUTP,"/home/ubuntu/cipher-fusion-evidence/rc_pergpu/sdc_counts.json");
  if((s=getenv("RV_EVENTS"))) strncpy(EVP,s,511); else EVP[0]=0;
  if((s=getenv("RV_FP"))) FP=strtoull(s,0,0); if(!FP) FP=0xC1DEC0DE00000001ULL;
  R_ge=(ge_t)cbsym("cublasGemmEx"); R_axpy=(axpy_t)cbsym("cublasAxpyEx"); R_nrm2=(nrm2_t)cbsym("cublasNrm2Ex");
  R_malloc=(cumalloc_t)cusym("cudaMalloc"); R_memcpy=(cumemcpy_t)cusym("cudaMemcpy");
  R_getstream=(getstream_t)cbsym("cublasGetStream_v2"); R_ssync=(ssync_t)cusym("cudaStreamSynchronize");
  R_getdev=(getdev_t)cusym("cudaGetDevice"); if(R_getdev) R_getdev(&gpu_id);
  PID=(int)getpid(); TID=(int)syscall(SYS_gettid);
  h_res=(uint16_t*)malloc(sizeof(uint16_t));
  fprintf(stderr,"[rc-det] init N=%d INJECT=%d ONSET=%d WATCH=%d gpu=%d pid=%d EVENTS=%s\n",N,INJ,ONSET,WATCH,gpu_id,PID,EVP);
}

cbStatus cublasGemmEx(void*h,int ta,int tb,int m,int n,int k,const void*al,const void*A,int At,int lda,
                      const void*B,int Bt,int ldb,const void*be,void*C,int Ct,int ldc,int ct,int algo){
  if(!inited) init();
  if(!R_ge) R_ge=(ge_t)cbsym("cublasGemmEx");
  cbStatus st=R_ge(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,C,Ct,ldc,ct,algo);
  gtot++; if(m>maxm)maxm=m; if(n>maxn)maxn=n; if(n<=64)n_smalln++;

  int is_linear = (k==4096 || k==14336) && (m!=VOCAB);
  if(!is_linear) return st;

  long this_g = lin % GPS;
  long cur_step = lin / GPS;
  int is_check = (cur_step % N)==0;

  /* persistent injection, gated on ONSET (clean before, corrupt after) */
  if(INJ>=0 && this_g==INJ && cur_step>=ONSET && R_memcpy && R_getstream && R_ssync){
    void* strm=0; R_getstream(h,&strm); R_ssync(strm);
    uint16_t hv; R_memcpy(&hv, C, 2, 2); uint16_t before=hv; hv ^= (1u<<14); R_memcpy(C, &hv, 2, 1);
    static int idbg=0; if(idbg<4){ idbg++; uint16_t chk=0; R_memcpy(&chk,C,2,2);
      fprintf(stderr,"[rc-det-INJ] step=%ld gidx=%ld C[0] %04x->%04x readback=%04x\n",cur_step,this_g,before,hv,chk); }
    injected++;
  }

  /* detector + per-check event emission for the watched ordinal */
  if(is_check && (ldc==m) && R_axpy && R_nrm2 && R_malloc){
    size_t need=(size_t)m*n*2;
    if(need>scratch_sz){ R_malloc(&scratch,need); scratch_sz=need; }
    if(scratch){
      R_ge(h,ta,tb,m,n,k,al,A,At,lda,B,Bt,ldb,be,scratch,Ct,m,ct,algo);
      float negone=-1.0f; long ne=(long)m*n;
      R_axpy(h,(int)ne,&negone,0,C,2,1,scratch,2,1,0);
      *h_res=0; R_nrm2(h,(int)ne,scratch,2,1,h_res,2,0);
      double r=h2f(*h_res); int hit=(r>0.0)||(r!=r); gchecked++;
      int is_inj_gemm = (this_g==INJ && INJ>=0);
      if(is_inj_gemm){ if(r>max_det_resid)max_det_resid=r; if(hit)detections++; }
      else { if(r>max_clean_resid)max_clean_resid=r; if(hit)detections++; }  /* any hit on clean gemm = FALSE POSITIVE */
      if(this_g==WATCH) emit_event(cur_step,(int)this_g,k,r,hit);
    }
  }

  lin++;
  if(lin % GPS == 0){ step=lin/GPS; if(((step-1)%N)==0) checks++; flush(); heartbeat(); }
  return st;
}

__attribute__((destructor)) static void fin(void){
  if(!inited) return;
  flush();
  fprintf(stderr,"[rc-det] FINAL steps=%ld checks=%ld gchecked=%ld detections=%ld injected=%ld clean_resid_max=%.4g det_resid_max=%.4g\n",
    step,checks,gchecked,detections,injected,max_clean_resid,max_det_resid);
}
