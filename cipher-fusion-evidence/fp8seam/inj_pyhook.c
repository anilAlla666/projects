/* inj_pyhook.c — Phase-0 reachability probe (2026-06-12). Tests OPTION B: an injected .so, loaded
 * by the CUDA driver / LD_PRELOAD into the DEFAULT-MP vLLM spawn worker, drives an IN-WORKER Python
 * patch of vllm._custom_ops.cutlass_scaled_mm with the .so as the SOLE artifact (no entry point, no
 * package, no install). Proves/kills worker-reach for the FP8 seam. Scratch probe; does NOT touch
 * the anchor libcipher_rt.so. Same injection surface as the live cuBLAS hook (cipher_inject.c:116,131).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>

typedef int   (*py_isinit_fn)(void);
typedef int   (*py_gil_ensure_fn)(void);   /* PyGILState_STATE is an enum (int-sized) */
typedef void  (*py_gil_release_fn)(int);
typedef int   (*py_run_fn)(const char *);

static pthread_once_t once = PTHREAD_ONCE_INIT;

static const char *PYCODE =
"import os,sys\n"
"m=sys.modules.get('vllm._custom_ops')\n"
"if m is not None and not getattr(m,'_cipher_fp8_probe',False):\n"
"    _real=m.cutlass_scaled_mm\n"
"    _st={'n':0}\n"
"    _out=os.environ.get('INJPY_OUT','/tmp/injpy')+'.'+str(os.getpid())\n"
"    def _w(*A,**K):\n"
"        r=_real(*A,**K)\n"          /* forward verbatim: robust to positional/keyword out_dtype, bias */
"        _st['n']+=1\n"
"        if _st['n']<5 or _st['n']%256==0:\n"
"            try:\n"
"                a=A[0] if len(A)>0 else K['a']; b=A[1] if len(A)>1 else K['b']\n"
"                k=a.shape[-1]; mm=a.shape[0] if a.dim()==2 else a.numel()//k; nn=b.shape[1]\n"
"                open(_out,'w').write('{\"pid\":%d,\"ppid\":%d,\"calls\":%d,\"last_mnk\":[%d,%d,%d]}'%(os.getpid(),os.getppid(),_st['n'],mm,nn,k))\n"
"            except Exception as e:\n"
"                open(_out+'.err','w').write(repr(e))\n"
"        return r\n"
"    m.cutlass_scaled_mm=_w\n"
"    m._cipher_fp8_probe=True\n"
"    try:\n"
"        import vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass as ctk\n"
"        if hasattr(ctk,'ops'): ctk.ops.cutlass_scaled_mm=_w\n"
"    except Exception: pass\n"
"    open(_out+'.installed','w').write('patched pid=%d ppid=%d'%(os.getpid(),os.getppid()))\n";

static void *installer(void *arg) {
    (void)arg;
    py_isinit_fn     Py_IsInitialized   = (py_isinit_fn)    dlsym(RTLD_DEFAULT, "Py_IsInitialized");
    py_gil_ensure_fn PyGILState_Ensure  = (py_gil_ensure_fn)dlsym(RTLD_DEFAULT, "PyGILState_Ensure");
    py_gil_release_fn PyGILState_Release= (py_gil_release_fn)dlsym(RTLD_DEFAULT, "PyGILState_Release");
    py_run_fn        PyRun_SimpleString = (py_run_fn)       dlsym(RTLD_DEFAULT, "PyRun_SimpleString");
    const char *base = getenv("INJPY_OUT"); if (!base) base = "/tmp/injpy";
    char installed[512]; snprintf(installed, sizeof installed, "%s.%d.installed", base, (int)getpid());
    fprintf(stderr, "[injpy] installer pid=%d ppid=%d py_syms=%d%d%d%d\n", (int)getpid(), (int)getppid(),
            !!Py_IsInitialized, !!PyGILState_Ensure, !!PyGILState_Release, !!PyRun_SimpleString);
    if (!Py_IsInitialized || !PyGILState_Ensure || !PyRun_SimpleString) {
        fprintf(stderr, "[injpy] no CPython API in this process (not a python worker) -- nothing to do\n");
        return NULL;
    }
    struct timespec ts = {0, 150L*1000L*1000L};
    for (int i = 0; i < 1200; i++) {            /* up to ~180s */
        if (access(installed, F_OK) == 0) { fprintf(stderr, "[injpy] patched pid=%d after %d polls\n", (int)getpid(), i); break; }
        if (Py_IsInitialized()) {
            int st = PyGILState_Ensure();
            PyRun_SimpleString(PYCODE);
            PyGILState_Release(st);
        }
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static void start_installer(void) {
    fprintf(stderr, "[injpy] trigger pid=%d ppid=%d\n", (int)getpid(), (int)getppid());
    pthread_t t; pthread_create(&t, NULL, installer, NULL); pthread_detach(t);
}

/* CUDA injection entry points (driver calls one of these at cuInit in the worker). */
int InitializeInjection(void *p)  { (void)p; pthread_once(&once, start_installer); return 1; }
int InitializeInjection2(void)    {         pthread_once(&once, start_installer); return 1; }
/* LD_PRELOAD path: constructor fires at dlopen in every forked/spawned process. */
static void __attribute__((constructor)) ctor(void) { pthread_once(&once, start_installer); }
