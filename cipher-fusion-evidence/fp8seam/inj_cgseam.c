/* inj_cgseam.c — CUDAGRAPH-SAFE FP8 SDC detector (2026-06-12). Mechanism 1B: the injected .so (sole
 * artifact) installs an IN-WORKER, dynamo-traceable functional wrap of torch.ops._C.cutlass_scaled_mm
 * in the DEFAULT-MP worker. The wrap does real compute + a device inject flag + independent recompute
 * + residual into a STATIC GPU buffer (in-place maximum) -- NO host branch in the compiled region, so
 * it survives torch.compile + cudagraph. A daemon thread reads the residual/count buffers OUT-OF-BAND
 * (the ra_coophook pattern, coop_driver.py:97-119). Scratch probe; anchor libcipher_rt.so untouched. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>

typedef int  (*isinit_fn)(void);
typedef int  (*ensure_fn)(void);
typedef void (*release_fn)(int);
typedef int  (*run_fn)(const char *);
static pthread_once_t once = PTHREAD_ONCE_INIT;

static const char *PYCODE =
"import os,sys,threading,time,json\n"
"if not getattr(sys,'_cipher_cg',False):\n"
"  try:\n"
"    import torch\n"
"    if ('vllm._custom_ops' in sys.modules) and torch.cuda.is_available() and torch.cuda.is_initialized():\n"
"      dev='cuda'\n"
"      RES=torch.zeros(1,device=dev); CNT=torch.zeros(1,device=dev); INJ=torch.zeros(1,device=dev)\n"
"      for _t in (RES,CNT,INJ):\n"                           /* fixed-address static buffers: in-graph writes persist + host-readable (cudagraph-tree mechanism) */
"        try: torch._dynamo.mark_static_address(_t)\n"
"        except Exception: pass\n"
"      if not (hasattr(torch.ops,'cipher') and hasattr(torch.ops.cipher,'detect_sink')):\n"
"        _cl=torch.library.Library('cipher','FRAGMENT')\n"   /* custom op with DECLARED mutation -> inductor must NOT DCE it */
"        _cl.define('detect_sink(Tensor d, Tensor(a!) res, Tensor(b!) cnt) -> ()')\n"
"        def _sink(d,res,cnt):\n"
"          torch.maximum(res,d.reshape(1),out=res); cnt.add_(1)\n"
"        _cl.impl('detect_sink',_sink,'CompositeExplicitAutograd')\n"
"        torch.library.register_fake('cipher::detect_sink', lambda d,res,cnt: None)\n"
"      import vllm._custom_ops as vco\n"
"      _real=vco.cutlass_scaled_mm\n"                        /* the vLLM python helper (dynamo INLINES it); torch.ops packet left intact */
"      ARM=os.environ.get('CGSEAM_ARM','1')=='1'\n"
"      def _w(*A,**K):\n"
"        out=_real(*A,**K)\n"                                /* real compute (returns out) */
"        out=out+INJ.to(out.dtype)*0.5\n"                    /* functional inject: no-op when INJ==0; preserve out dtype */
"        out2=_real(*A,**K)\n"                               /* independent recompute (clean) */
"        d=(out2.float()-out.float()).abs().amax()\n"
"        torch.ops.cipher.detect_sink(d,RES,CNT)\n"          /* mutates res,cnt -> preserved through AOT/inductor, no host branch */
"        return out\n"
"      if ARM: vco.cutlass_scaled_mm=_w\n"
"      sys._cipher_cg=True\n"
"      OUT=os.environ.get('CGSEAM_OUT','/tmp/cg')+'.'+str(os.getpid())\n"
"      TRIG=os.environ.get('CGSEAM_TRIG','/tmp/cgtrig')\n"
"      st={'pid':os.getpid(),'ppid':os.getppid(),'armed':ARM,'injected':False,'clean_res':None,'post_inject_res':None,'first_detect_wall':None,'inject_wall':None}\n"
"      READY=os.environ.get('CGSEAM_READY','/tmp/cgready')\n"
"      def loop():\n"
"        while True:\n"
"          try:\n"
"            if not os.path.exists(READY):\n"                /* do NO cuda ops until init cudagraph-capture is done */
"              time.sleep(0.1); continue\n"
"            cnt=int(CNT.item()); res=float(RES.item())\n"   /* OUT-OF-BAND read, after capture, between replays */
"            if (not st['injected']) and os.path.exists(TRIG):\n"
"              st['clean_res']=res; INJ.fill_(1.0); st['injected']=True; st['inject_wall']=time.time()\n"
"            if st['injected'] and (st['clean_res'] is not None) and res>st['clean_res'] and st['first_detect_wall'] is None:\n"
"              st['first_detect_wall']=time.time(); st['post_inject_res']=res\n"
"            st['cnt']=cnt; st['res_max']=res\n"
"            json.dump(st,open(OUT,'w'))\n"
"          except Exception as e:\n"
"            open(OUT+'.err','w').write(repr(e))\n"
"          time.sleep(0.15)\n"
"      threading.Thread(target=loop,daemon=True).start()\n"
"      open(OUT+'.installed','w').write('cgseam armed=%s pid=%d'%(ARM,os.getpid()))\n"
"  except Exception as e:\n"
"    open(os.environ.get('CGSEAM_OUT','/tmp/cg')+'.boot.'+str(os.getpid())+'.err','w').write(repr(e))\n";

static void *installer(void *arg){
  (void)arg;
  isinit_fn  Py_IsInitialized  = (isinit_fn) dlsym(RTLD_DEFAULT,"Py_IsInitialized");
  ensure_fn  PyGILState_Ensure = (ensure_fn) dlsym(RTLD_DEFAULT,"PyGILState_Ensure");
  release_fn PyGILState_Release= (release_fn)dlsym(RTLD_DEFAULT,"PyGILState_Release");
  run_fn     PyRun_SimpleString= (run_fn)    dlsym(RTLD_DEFAULT,"PyRun_SimpleString");
  if(!Py_IsInitialized||!PyGILState_Ensure||!PyRun_SimpleString) return NULL;
  const char *base=getenv("CGSEAM_OUT"); if(!base) base="/tmp/cg";
  char inst[512]; snprintf(inst,sizeof inst,"%s.%d.installed",base,(int)getpid());
  struct timespec ts={0,150L*1000L*1000L};
  for(int i=0;i<2000;i++){
    if(access(inst,F_OK)==0){ fprintf(stderr,"[cgseam] installed pid=%d\n",(int)getpid()); break; }
    if(Py_IsInitialized()){ int s=PyGILState_Ensure(); PyRun_SimpleString(PYCODE); PyGILState_Release(s); }
    nanosleep(&ts,NULL);
  }
  return NULL;
}
static void start(void){ pthread_t t; pthread_create(&t,NULL,installer,NULL); pthread_detach(t); }
int InitializeInjection(void *p){ (void)p; pthread_once(&once,start); return 1; }
int InitializeInjection2(void){ pthread_once(&once,start); return 1; }
static void __attribute__((constructor)) ctor(void){ pthread_once(&once,start); }
