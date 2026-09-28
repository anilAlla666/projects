// STEP 0 routing-signal probe (throwaway): does an explicit LOAD-PHASE marker isolate weight allocations
// from KV/activation/scratch? Counting pluggable allocator: tag bytes allocated inside route_begin/route_end
// (the weight-load window) vs after. If in-window bytes ~= model param bytes and post-window = activations,
// the phase marker is the routing signal. Build: g++ -O2 -fPIC -shared -o pager_route_probe.so pager_route_probe.c -lcudart
#include <cuda_runtime.h>
#include <atomic>
static std::atomic<int> g_window{0};
static std::atomic<unsigned long long> g_win_bytes{0}, g_win_cnt{0}, g_aft_bytes{0}, g_aft_cnt{0}, g_win_max{0}, g_aft_max{0};
static void bump_max(std::atomic<unsigned long long> &m, unsigned long long v) {
	unsigned long long cur = m.load();
	while (v > cur && !m.compare_exchange_weak(cur, v)) {}
}
extern "C" void *route_malloc(size_t size, int dev, void *st) {
	(void)dev; (void)st; void *p = 0; cudaMalloc(&p, size);
	if (g_window.load()) { g_win_bytes += size; g_win_cnt += 1; bump_max(g_win_max, size); }
	else                 { g_aft_bytes += size; g_aft_cnt += 1; bump_max(g_aft_max, size); }
	return p;
}
extern "C" void route_free(void *p, size_t sz, int dev, void *st) { (void)sz;(void)dev;(void)st; if (p) cudaFree(p); }
extern "C" void route_begin(void) { g_window.store(1); }
extern "C" void route_end(void)   { g_window.store(0); }
extern "C" void route_stats(unsigned long long *o) {
	o[0]=g_win_bytes.load(); o[1]=g_win_cnt.load(); o[2]=g_aft_bytes.load(); o[3]=g_aft_cnt.load();
	o[4]=g_win_max.load();   o[5]=g_aft_max.load();
}
