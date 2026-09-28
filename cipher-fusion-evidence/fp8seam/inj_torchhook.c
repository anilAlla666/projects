/* inj_torchhook.c — Phase-0 LAYER recon (2026-06-12). Tests OPTION 1: hook the FP8 matmul at the
 * PyTorch DISPATCHER level (torch.ops._C.cutlass_scaled_mm handle) instead of vLLM's Python wrapper
 * (vllm._custom_ops.cutlass_scaled_mm). Same injection mechanism as inj_pyhook.c (in-worker Python
 * patch driven by the .so). Records: call count in the worker + operand shapes/dtypes + dumps one
 * real operand set to $FP8_DUMP for the framework-general recompute test. Scratch probe; anchor
 * libcipher_rt.so untouched. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>

typedef int  (*py_isinit_fn)(void);
typedef int  (*py_gil_ensure_fn)(void);
typedef void (*py_gil_release_fn)(int);
typedef int  (*py_run_fn)(const char *);

static pthread_once_t once = PTHREAD_ONCE_INIT;

static const char *PYCODE =
"import os,sys\n"
"try:\n"
"    import torch\n"
"except Exception:\n"
"    torch=None\n"
"if torch is not None and ('vllm._custom_ops' in sys.modules) and not getattr(torch.ops._C,'_cipher_th',False):\n"
"    try:\n"
"        _real=torch.ops._C.cutlass_scaled_mm\n"           /* the torch DISPATCHER handle, not vLLM's wrapper */
"        _st={'n':0,'dumped':False}\n"
"        _out=os.environ.get('INJPY_OUT','/tmp/th')+'.'+str(os.getpid())\n"
"        def _w(*A,**K):\n"
"            r=_real(*A,**K)\n"
"            _st['n']+=1\n"
"            if not _st['dumped'] and len(A)>=5:\n"
"                try:\n"
"                    out,a,b,sa,sb=A[0],A[1],A[2],A[3],A[4]\n"
"                    info={'out':[list(out.shape),str(out.dtype)],'a':[list(a.shape),str(a.dtype)],'b':[list(b.shape),str(b.dtype)],'a_scales':[list(sa.shape),str(sa.dtype)],'b_scales':[list(sb.shape),str(sb.dtype)]}\n"
"                    open(_out+'.operands','w').write(repr(info))\n"
"                    torch.save({'a':a.detach().cpu(),'b':b.detach().cpu(),'sa':sa.detach().cpu(),'sb':sb.detach().cpu(),'out':out.detach().cpu()}, os.environ.get('FP8_DUMP','/tmp/fp8operands.pt'))\n"
"                    _st['dumped']=True\n"
"                except Exception as e:\n"
"                    open(_out+'.operr','w').write(repr(e))\n"
"                    _st['dumped']=True\n"
"            if _st['n']<5 or _st['n']%256==0:\n"
"                open(_out,'w').write('{\"pid\":%d,\"ppid\":%d,\"calls\":%d,\"layer\":\"torch.ops._C\"}'%(os.getpid(),os.getppid(),_st['n']))\n"
"            return r\n"
"        torch.ops._C.cutlass_scaled_mm=_w\n"               /* settable attr (verified) -> catches every torch.ops caller */
"        torch.ops._C._cipher_th=True\n"
"        open(_out+'.installed','w').write('torch-dispatcher-hooked pid=%d'%os.getpid())\n"
"    except Exception as e:\n"
"        open(os.environ.get('INJPY_OUT','/tmp/th')+'.'+str(os.getpid())+'.err','w').write(repr(e))\n";

static void *installer(void *arg) {
    (void)arg;
    py_isinit_fn      Py_IsInitialized    = (py_isinit_fn)     dlsym(RTLD_DEFAULT, "Py_IsInitialized");
    py_gil_ensure_fn  PyGILState_Ensure   = (py_gil_ensure_fn) dlsym(RTLD_DEFAULT, "PyGILState_Ensure");
    py_gil_release_fn PyGILState_Release  = (py_gil_release_fn)dlsym(RTLD_DEFAULT, "PyGILState_Release");
    py_run_fn         PyRun_SimpleString  = (py_run_fn)        dlsym(RTLD_DEFAULT, "PyRun_SimpleString");
    const char *base = getenv("INJPY_OUT"); if (!base) base = "/tmp/th";
    char installed[512]; snprintf(installed, sizeof installed, "%s.%d.installed", base, (int)getpid());
    if (!Py_IsInitialized || !PyGILState_Ensure || !PyRun_SimpleString) return NULL;
    struct timespec ts = {0, 150L*1000L*1000L};
    for (int i = 0; i < 1200; i++) {
        if (access(installed, F_OK) == 0) { fprintf(stderr, "[injth] torch-hooked pid=%d\n", (int)getpid()); break; }
        if (Py_IsInitialized()) { int st = PyGILState_Ensure(); PyRun_SimpleString(PYCODE); PyGILState_Release(st); }
        nanosleep(&ts, NULL);
    }
    return NULL;
}
static void start(void){ pthread_t t; pthread_create(&t,NULL,installer,NULL); pthread_detach(t); }
int InitializeInjection(void *p){ (void)p; pthread_once(&once,start); return 1; }
int InitializeInjection2(void){ pthread_once(&once,start); return 1; }
static void __attribute__((constructor)) ctor(void){ pthread_once(&once,start); }
