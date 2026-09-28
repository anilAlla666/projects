// vLLM weight-capture probe (throwaway): a counting torch MemPool allocator. As a POOL allocator, EVERY call to
// vp_malloc is an in-pool allocation, so g_bytes = bytes captured by an outer use_mem_pool. getpid() on first call
// reveals whether allocations fire in THIS process or a vLLM worker subprocess (in a subprocess the .so globals
// are a separate copy -> the parent's vp_stats reads 0). Build: g++ -O2 -fPIC -shared -o pager_vllm_probe.so pager_vllm_probe.c -lcudart
#include <cuda_runtime.h>
#include <atomic>
#include <unistd.h>
#include <cstdio>
static std::atomic<unsigned long long> g_bytes{0}, g_cnt{0}, g_max{0};
static std::atomic<int> g_first{0};
extern "C" void *vp_malloc(size_t size, int dev, void *st) {
	(void)dev; (void)st; void *p = 0; cudaMalloc(&p, size);
	int was = g_first.exchange(1);
	if (!was) fprintf(stderr, "[VP-PID] first in-pool malloc fired in pid=%d size=%zu\n", getpid(), size);
	g_bytes += size; g_cnt += 1;
	unsigned long long cur = g_max.load(); while (size > cur && !g_max.compare_exchange_weak(cur, size)) {}
	return p;
}
extern "C" void vp_free(void *p, size_t sz, int dev, void *st) { (void)sz;(void)dev;(void)st; if (p) cudaFree(p); }
extern "C" void vp_stats(unsigned long long *o) { o[0]=g_bytes.load(); o[1]=g_cnt.load(); o[2]=g_max.load(); o[3]=(unsigned long long)getpid(); }
extern "C" void vp_reset(void) { g_bytes=0; g_cnt=0; g_max=0; }
