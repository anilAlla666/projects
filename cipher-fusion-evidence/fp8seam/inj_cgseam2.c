/* inj_cgseam2.c — CUDAGRAPH-SAFE FP8 SDC detector via MODULE-BUFFER input-mutation (2026-06-12).
 * Injected-.so-only, default-MP worker. Patches CompressedTensorsW8A8Fp8.create_weights to attach
 * cipher_res/cnt/inj BUFFERS per FP8 layer (dynamo lifts module buffers as graph inputs -> mutation
 * preserved through inductor + vLLM manual cudagraph capture; P0C/cgmod_inproc proven), and wraps
 * apply_weights to recompute + write residual to those buffers via a mutates_args custom op. Reader
 * thread aggregates across layers out-of-band. Scratch probe; anchor .so untouched. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>
typedef int (*isinit_fn)(void); typedef int (*ensure_fn)(void);
typedef void (*release_fn)(int); typedef int (*run_fn)(const char *);
static pthread_once_t once = PTHREAD_ONCE_INIT;

static const char *PY =
"import os,sys,threading,time,json\n"
"if not getattr(sys,'_cipher_cg2',False):\n"
"  try:\n"
"    if 'torch' not in sys.modules: raise RuntimeError('wait torch')\n"
"    import torch, torch._inductor.config as ic\n"
"    _mn='vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8'\n"
"    _m=sys.modules.get(_mn)\n"                              /* do NOT import it ourselves: wait until vLLM has, to avoid an import race */
"    if _m is None: raise RuntimeError('wait vllm fp8 scheme')\n"
"    CW=_m.CompressedTensorsW8A8Fp8\n"
"    ic.triton.cudagraph_support_input_mutation=True\n"
"    if not (hasattr(torch.ops,'cipherd') and hasattr(torch.ops.cipherd,'sink')):\n"
"      _cl=torch.library.Library('cipherd','FRAGMENT')\n"
"      _cl.define('sink(Tensor d, Tensor(a!) res, Tensor(b!) cnt) -> ()')\n"
"      def _sink(d,res,cnt): torch.maximum(res,d.reshape(1),out=res); cnt.add_(1)\n"
"      _cl.impl('sink',_sink,'CompositeExplicitAutograd')\n"
"      torch.library.register_fake('cipherd::sink', lambda d,res,cnt: None)\n"
"    if not getattr(CW,'_cipher_patched',False):\n"
"      _L=[]\n"
"      _oc=CW.create_weights\n"
"      def _cw(self,layer,*a,**k):\n"
"        r=_oc(self,layer,*a,**k)\n"
"        for nm in ('cipher_res','cipher_cnt','cipher_inj'):\n"
"          layer.register_buffer(nm, torch.zeros(1,device='cuda'), persistent=False)\n"
"        for t in (layer.cipher_res,layer.cipher_cnt,layer.cipher_inj):\n"
"          try: torch._dynamo.mark_static_address(t)\n"
"          except Exception: pass\n"
"        _L.append(layer); return r\n"
"      CW.create_weights=_cw\n"
"      _oa=CW.apply_weights; ARM=os.environ.get('CGSEAM_ARM','1')=='1'; CSE=os.environ.get('CGSEAM_CSE','0')=='1'\n"
"      def _aw(self,layer,x,bias=None):\n"
"        out=_oa(self,layer,x,bias)\n"
"        if (not ARM) or (not hasattr(layer,'cipher_res')): return out\n"
"        out=out+layer.cipher_inj.to(out.dtype)*0.5\n"
"        xr=(x*1.01) if CSE else x\n"                        /* CSE guard: perturb recompute INPUT -> residual must reflect a genuine 2nd kernel */
"        out2=_oa(self,layer,xr,bias)\n"
"        d=(out2.float()-out.float()).abs().amax()\n"
"        torch.ops.cipherd.sink(d,layer.cipher_res,layer.cipher_cnt)\n"
"        return out\n"
"      CW.apply_weights=_aw; CW._cipher_patched=True\n"
"      sys._cipher_cg2=True\n"
"      OUT=os.environ.get('CGSEAM_OUT','/tmp/cg2')+'.'+str(os.getpid())\n"
"      TRIG=os.environ.get('CGSEAM_TRIG','/tmp/cg2trig'); READY=os.environ.get('CGSEAM_READY','/tmp/cg2ready')\n"
"      st={'pid':os.getpid(),'ppid':os.getppid(),'injected':False,'clean_res':None,'first_detect_wall':None}\n"
"      def agg():\n"
"        mx=0.0; cnt=0\n"
"        for La in _L:\n"
"          try: mx=max(mx,float(La.cipher_res.item())); cnt+=int(La.cipher_cnt.item())\n"
"          except Exception: pass\n"
"        return mx,cnt\n"
"      def loop():\n"
"        while True:\n"
"          try:\n"
"            if not os.path.exists(READY): time.sleep(0.1); continue\n"
"            mx,cnt=agg()\n"
"            if (not st['injected']) and os.path.exists(TRIG):\n"
"              st['clean_res']=mx\n"
"              if _L: _L[len(_L)//2].cipher_inj.fill_(1.0)\n"
"              st['injected']=True\n"
"            if st['injected'] and (st['clean_res'] is not None) and mx>st['clean_res'] and st['first_detect_wall'] is None:\n"
"              st['first_detect_wall']=time.time()\n"
"            st['fp8_layers']=len(_L); st['cnt']=cnt; st['res_max']=mx\n"
"            json.dump(st,open(OUT,'w'))\n"
"          except Exception as e: open(OUT+'.rerr','w').write(repr(e))\n"
"          time.sleep(0.15)\n"
"      threading.Thread(target=loop,daemon=True).start()\n"
"      open(OUT+'.installed','w').write('cgseam2 pid=%d'%os.getpid())\n"
"  except Exception as e:\n"
"    open(os.environ.get('CGSEAM_OUT','/tmp/cg2')+'.boot.'+str(os.getpid())+'.err','w').write(repr(e))\n";

static void *installer(void *arg){ (void)arg;
  isinit_fn  PyI=(isinit_fn)dlsym(RTLD_DEFAULT,"Py_IsInitialized");
  ensure_fn  PyE=(ensure_fn)dlsym(RTLD_DEFAULT,"PyGILState_Ensure");
  release_fn PyR=(release_fn)dlsym(RTLD_DEFAULT,"PyGILState_Release");
  run_fn     PyS=(run_fn)dlsym(RTLD_DEFAULT,"PyRun_SimpleString");
  if(!PyI||!PyE||!PyS) return NULL;
  const char *b=getenv("CGSEAM_OUT"); if(!b) b="/tmp/cg2";
  char inst[512]; snprintf(inst,sizeof inst,"%s.%d.installed",b,(int)getpid());
  struct timespec ts={0,80L*1000L*1000L};   /* 80ms poll: win the race to patch create_weights before construction */
  for(int i=0;i<3000;i++){
    if(access(inst,F_OK)==0){ fprintf(stderr,"[cgseam2] installed pid=%d\n",(int)getpid()); break; }
    if(PyI()){ int s=PyE(); PyS(PY); PyR(s); }
    nanosleep(&ts,NULL);
  }
  return NULL;
}
static void start(void){ pthread_t t; pthread_create(&t,NULL,installer,NULL); pthread_detach(t); }
int InitializeInjection(void *p){ (void)p; pthread_once(&once,start); return 1; }
int InitializeInjection2(void){ pthread_once(&once,start); return 1; }
static void __attribute__((constructor)) ctor(void){ pthread_once(&once,start); }
