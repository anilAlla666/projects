// THROWAWAY probe allocator (NOT the production runtime): cuMemMap-backed torch pluggable allocator.
// Proves a torch-OWNED weight tensor can live behind a CIPHER-pageable VA (evict=unmap / pagein=remap).
// Build: g++ -O2 -fPIC -shared -o pager_probe_alloc.so pager_probe_alloc.c -lcuda
#include <cuda.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define MAXR 256
struct Reg { CUdeviceptr va; CUmemGenericAllocationHandle hbm; size_t size; int mapped; };
static struct Reg g_reg[MAXR];
static int g_n=0;
static pthread_mutex_t g_mu=PTHREAD_MUTEX_INITIALIZER;
static size_t g_gran=0; static int g_init=0;
static CUmemAllocationProp g_prop;

static void ensure_init(){
  if(g_init) return;
  cuInit(0);
  memset(&g_prop,0,sizeof(g_prop));
  g_prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
  g_prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE; g_prop.location.id=0;
  cuMemGetAllocationGranularity(&g_gran,&g_prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  g_init=1;
}
static size_t alignup(size_t s){ return ((s+g_gran-1)/g_gran)*g_gran; }
static int do_map(struct Reg* r){
  if(cuMemCreate(&r->hbm,r->size,&g_prop,0)!=CUDA_SUCCESS) return -1;
  if(cuMemMap(r->va,r->size,0,r->hbm,0)!=CUDA_SUCCESS){ cuMemRelease(r->hbm); return -1; }
  CUmemAccessDesc acc; memset(&acc,0,sizeof(acc));
  acc.location.type=CU_MEM_LOCATION_TYPE_DEVICE; acc.location.id=0; acc.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if(cuMemSetAccess(r->va,r->size,&acc,1)!=CUDA_SUCCESS){ cuMemUnmap(r->va,r->size); cuMemRelease(r->hbm); return -1; }
  r->mapped=1; return 0;
}
extern "C" void* cipher_alloc(size_t size, int device, void* stream){
  (void)device;(void)stream; ensure_init();
  pthread_mutex_lock(&g_mu);
  if(g_n>=MAXR){ pthread_mutex_unlock(&g_mu); return NULL; }
  struct Reg* r=&g_reg[g_n]; r->size=alignup(size?size:1);
  if(cuMemAddressReserve(&r->va,r->size,0,0,0)!=CUDA_SUCCESS){ pthread_mutex_unlock(&g_mu); return NULL; }
  if(do_map(r)!=0){ cuMemAddressFree(r->va,r->size); pthread_mutex_unlock(&g_mu); return NULL; }
  g_n++; void* p=(void*)r->va; pthread_mutex_unlock(&g_mu); return p;
}
extern "C" void cipher_free(void* ptr, size_t size, int device, void* stream){
  (void)size;(void)device;(void)stream;
  pthread_mutex_lock(&g_mu);
  for(int i=0;i<g_n;i++) if(g_reg[i].va==(CUdeviceptr)ptr){
    if(g_reg[i].mapped){ cuMemUnmap(g_reg[i].va,g_reg[i].size); cuMemRelease(g_reg[i].hbm); }
    cuMemAddressFree(g_reg[i].va,g_reg[i].size);
    g_reg[i]=g_reg[--g_n]; break;
  }
  pthread_mutex_unlock(&g_mu);
}
// probe hooks: evict (unmap+release, VA reserved+unbacked) / pagein (fresh HBM remapped to same VA)
extern "C" int cipher_evict(void* va){
  pthread_mutex_lock(&g_mu); int rc=-1;
  for(int i=0;i<g_n;i++) if(g_reg[i].va==(CUdeviceptr)va && g_reg[i].mapped){
    cuMemUnmap(g_reg[i].va,g_reg[i].size); cuMemRelease(g_reg[i].hbm); g_reg[i].mapped=0; rc=0; break; }
  pthread_mutex_unlock(&g_mu); return rc;
}
extern "C" int cipher_pagein(void* va){
  pthread_mutex_lock(&g_mu); int rc=-1;
  for(int i=0;i<g_n;i++) if(g_reg[i].va==(CUdeviceptr)va && !g_reg[i].mapped){ rc=do_map(&g_reg[i]); break; }
  pthread_mutex_unlock(&g_mu); return rc;
}
extern "C" int cipher_mapped(void* va){
  pthread_mutex_lock(&g_mu); int m=-1;
  for(int i=0;i<g_n;i++) if(g_reg[i].va==(CUdeviceptr)va){ m=g_reg[i].mapped; break; }
  pthread_mutex_unlock(&g_mu); return m;
}
