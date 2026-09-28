/* Track B Path-1 mechanism shim: interpose cudaGraphInstantiateWithFlags (the exact runtime symbol
 * libtorch_cuda imports as UNDEF from libcudart.so.13) to inject a device node into the captured graph
 * BEFORE instantiation, with dependency edges from the graph leaves. Proves a detector node can be added
 * at the EndCapture->Instantiate seam, runs in replay, and does NOT invalidate capture. Device-only:
 * the node memsets a sentinel into a shim-owned device flag; the host reads it post-replay (no in-capture sync).
 *
 * Robustness: resolves ALL cudart functions via dlsym(RTLD_NEXT) at runtime (no -lcudart link, no cuda
 * headers, no device kernel, no nvcc registration glue) -> safe across the nvcc-12.8 / torch-cudart-13 split.
 * A memset node is a faithful stand-in for a check kernel for the MECHANISM question (node injected at the
 * seam executes in replay); a real check kernel is added identically via cudaGraphAddKernelNode + a loaded module.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void* cudaGraph_t; typedef void* cudaGraphExec_t; typedef void* cudaGraphNode_t;
typedef int cudaError_t;
typedef struct { void* dst; size_t pitch; unsigned int value; unsigned int elementSize; size_t width; size_t height; } cudaMemsetParams;

static void* sym(const char* n){ void* p=dlsym(RTLD_NEXT,n); if(!p) p=dlsym(RTLD_DEFAULT,n); return p; }

static void*  g_flag = NULL;       /* shim-owned device flag buffer */
static int    g_inject_count = 0;  /* how many captured graphs we injected into */
static int    g_last_nodes = 0, g_last_edges = 0, g_last_leaves = 0, g_last_rc = -99;

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pExec, cudaGraph_t graph, unsigned long long flags){
    static cudaError_t (*real)(cudaGraphExec_t*,cudaGraph_t,unsigned long long) = NULL;
    if(!real) real = (cudaError_t(*)(cudaGraphExec_t*,cudaGraph_t,unsigned long long))sym("cudaGraphInstantiateWithFlags");

    cudaError_t (*GetNodes)(cudaGraph_t,cudaGraphNode_t*,size_t*)              = (void*)sym("cudaGraphGetNodes");
    cudaError_t (*GetEdges)(cudaGraph_t,cudaGraphNode_t*,cudaGraphNode_t*,size_t*) = (void*)sym("cudaGraphGetEdges");
    cudaError_t (*AddMemset)(cudaGraphNode_t*,cudaGraph_t,const cudaGraphNode_t*,size_t,const cudaMemsetParams*) = (void*)sym("cudaGraphAddMemsetNode");
    cudaError_t (*Malloc)(void**,size_t)                                       = (void*)sym("cudaMalloc");

    const char* dis = getenv("GAP6_DISABLE");   /* 1 = shim loaded + flag alloc'd but DO NOT inject (control) */
    size_t numNodes=0; if(GetNodes) GetNodes(graph,NULL,&numNodes);
    /* g_flag must be pre-allocated by gap6_init() BEFORE capture — never cudaMalloc inside the hook
     * (that runs mid-capture_end and perturbs torch's graph mempool). */
    if(!(dis && dis[0]=='1') && GetNodes && GetEdges && AddMemset && numNodes>0 && g_flag){
        (void)Malloc;
        size_t numEdges=0; GetEdges(graph,NULL,NULL,&numEdges);
        cudaGraphNode_t* nodes=malloc(numNodes*sizeof(void*)); GetNodes(graph,nodes,&numNodes);
        size_t ealloc = numEdges?numEdges:1;
        cudaGraphNode_t* from=malloc(ealloc*sizeof(void*));
        cudaGraphNode_t* to  =malloc(ealloc*sizeof(void*));
        if(numEdges) GetEdges(graph,from,to,&numEdges);
        /* leaves = nodes that are never a source in any edge */
        cudaGraphNode_t leaves[512]; int nleaves=0;
        for(size_t i=0;i<numNodes && nleaves<512;i++){
            int isSrc=0; for(size_t e=0;e<numEdges;e++){ if(from[e]==nodes[i]){ isSrc=1; break; } }
            if(!isSrc) leaves[nleaves++]=nodes[i];
        }
        cudaMemsetParams mp; memset(&mp,0,sizeof(mp));
        mp.dst=g_flag; mp.value=0x00C1; mp.elementSize=4; mp.width=1; mp.height=1; mp.pitch=4;
        cudaGraphNode_t newNode=NULL;
        /* GAP6_DEPS: 0 (default) = inject as root node (no deps) to prove execution; 1 = depend on leaves (ordered) */
        const char* depsenv = getenv("GAP6_DEPS");
        int use_deps = (depsenv && depsenv[0]=='1');
        cudaError_t rc = use_deps ? AddMemset(&newNode,graph,leaves,(size_t)nleaves,&mp)
                                  : AddMemset(&newNode,graph,NULL,0,&mp);
        g_inject_count++; g_last_nodes=(int)numNodes; g_last_edges=(int)numEdges; g_last_leaves=nleaves; g_last_rc=rc;
        fprintf(stderr,"[gap6-shim] intercepted cudaGraphInstantiateWithFlags: nodes=%zu edges=%zu leaves=%d addMemsetNode_rc=%d (inject #%d)\n",
                numNodes,numEdges,nleaves,(int)rc,g_inject_count);
        free(nodes); free(from); free(to);
    }
    return real(pExec,graph,flags);
}

/* call from python BEFORE any capture: allocate the device flag cleanly outside any graph context */
int gap6_init(void){
    if(g_flag) return 0;
    cudaError_t (*Malloc)(void**,size_t)=(void*)sym("cudaMalloc");
    if(!Malloc) return -1;
    cudaError_t rc=Malloc(&g_flag,4);
    fprintf(stderr,"[gap6-shim] gap6_init: cudaMalloc flag rc=%d ptr=%p\n",(int)rc,g_flag);
    return (int)rc;
}

/* host-side readers (called from python via ctypes; same loaded instance as LD_PRELOAD) */
int get_check_flag(void){
    if(!g_flag) return -1;
    cudaError_t (*Memcpy)(void*,const void*,size_t,int)=(void*)sym("cudaMemcpy");
    int h=0; if(Memcpy) Memcpy(&h,g_flag,4,2/*D2H*/); return h;
}
void reset_check_flag(void){
    if(!g_flag) return;
    cudaError_t (*Memcpy)(void*,const void*,size_t,int)=(void*)sym("cudaMemcpy");
    int z=0; if(Memcpy) Memcpy(g_flag,&z,4,1/*H2D*/);
}
int get_inject_count(void){ return g_inject_count; }
int get_last_nodes(void){ return g_last_nodes; }
int get_last_edges(void){ return g_last_edges; }
int get_last_leaves(void){ return g_last_leaves; }
int get_last_rc(void){ return g_last_rc; }
