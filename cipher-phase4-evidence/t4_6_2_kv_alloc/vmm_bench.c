/* T4.6.2 S1 — CUDA VMM API latency characterization on H100/CUDA13.
 *
 * Measures per-call latency of the VMM primitives the KV allocator
 * will use, and verifies cuMemExportToShareableHandle (T4.6.4 prereq).
 *
 * Build: gcc -O2 -o vmm_bench vmm_bench.c -lcuda
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define N_ITERS 64
#define PAGE (2UL * 1024 * 1024)   /* 2 MiB nominal */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#define CK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char *s = NULL; cuGetErrorString(_r, &s); \
        fprintf(stderr, "ERR %s:%d %s -> %d %s\n", \
                __FILE__, __LINE__, #call, _r, s ? s : "?"); \
        exit(1); \
    } \
} while (0)

static void stats(const char *label, double *xs, int n) {
    double mn = xs[0], mx = xs[0], sum = 0;
    for (int i = 0; i < n; i++) {
        sum += xs[i];
        if (xs[i] < mn) mn = xs[i];
        if (xs[i] > mx) mx = xs[i];
    }
    /* simple sort for median */
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            if (xs[j] < xs[i]) { double t = xs[i]; xs[i] = xs[j]; xs[j] = t; }
    printf("  %-28s mean=%.4f ms  median=%.4f ms  min=%.4f  max=%.4f  (n=%d)\n",
           label, sum / n, xs[n/2], mn, mx, n);
}

int main(void) {
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));

    CUmemAllocationProp prop = {0};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;

    size_t gran = 0;
    CK(cuMemGetAllocationGranularity(&gran, &prop,
                                     CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t gran_rec = 0;
    CK(cuMemGetAllocationGranularity(&gran_rec, &prop,
                                     CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    printf("=== T4.6.2 S1 VMM characterization (H100, driver 580.105.08, CUDA13) ===\n");
    printf("alloc granularity: minimum=%zu bytes (%.2f MiB), recommended=%zu bytes (%.2f MiB)\n",
           gran, gran / 1048576.0, gran_rec, gran_rec / 1048576.0);

    size_t page = (PAGE + gran - 1) / gran * gran;  /* round up to gran */
    printf("page size used: %zu bytes (%.2f MiB)\n\n", page, page / 1048576.0);

    /* --- cuMemAddressReserve: reserve 16 GiB VA --- */
    size_t va_size = 16UL * 1024 * 1024 * 1024;
    double t0 = now_ms();
    CUdeviceptr va_base;
    CK(cuMemAddressReserve(&va_base, va_size, 0, 0, 0));
    printf("cuMemAddressReserve(16 GiB): %.4f ms\n\n", now_ms() - t0);

    double t_create[N_ITERS], t_map[N_ITERS], t_setacc[N_ITERS];
    double t_unmap[N_ITERS], t_release[N_ITERS];
    CUmemGenericAllocationHandle handles[N_ITERS];

    CUmemAccessDesc acc = {0};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id = dev;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    /* --- create + map + setaccess, N pages --- */
    for (int i = 0; i < N_ITERS; i++) {
        CUdeviceptr p = va_base + (size_t)i * page;

        t0 = now_ms();
        CK(cuMemCreate(&handles[i], page, &prop, 0));
        t_create[i] = now_ms() - t0;

        t0 = now_ms();
        CK(cuMemMap(p, page, 0, handles[i], 0));
        t_map[i] = now_ms() - t0;

        t0 = now_ms();
        CK(cuMemSetAccess(p, page, &acc, 1));
        t_setacc[i] = now_ms() - t0;
    }

    /* touch the memory to confirm it's usable */
    CK(cuMemsetD8(va_base, 0xAB, page));
    unsigned char probe = 0;
    CK(cuMemcpyDtoH(&probe, va_base, 1));
    printf("write/read probe of first page: 0x%02X (expect 0xAB) -> %s\n\n",
           probe, probe == 0xAB ? "OK" : "FAIL");

    /* --- unmap + release --- */
    for (int i = 0; i < N_ITERS; i++) {
        CUdeviceptr p = va_base + (size_t)i * page;
        t0 = now_ms();
        CK(cuMemUnmap(p, page));
        t_unmap[i] = now_ms() - t0;
        t0 = now_ms();
        CK(cuMemRelease(handles[i]));
        t_release[i] = now_ms() - t0;
    }

    printf("Per-call latency (%d pages of %.2f MiB):\n", N_ITERS, page / 1048576.0);
    stats("cuMemCreate",    t_create,  N_ITERS);
    stats("cuMemMap",       t_map,     N_ITERS);
    stats("cuMemSetAccess", t_setacc,  N_ITERS);
    stats("cuMemUnmap",     t_unmap,   N_ITERS);
    stats("cuMemRelease",   t_release, N_ITERS);

    /* batched setaccess: map 64 pages contiguously, one setaccess call */
    {
        for (int i = 0; i < N_ITERS; i++) {
            CUdeviceptr p = va_base + (size_t)i * page;
            CK(cuMemCreate(&handles[i], page, &prop, 0));
            CK(cuMemMap(p, page, 0, handles[i], 0));
        }
        t0 = now_ms();
        CK(cuMemSetAccess(va_base, (size_t)N_ITERS * page, &acc, 1));
        double batched = now_ms() - t0;
        printf("\nbatched cuMemSetAccess over %d pages (%.0f MiB): %.4f ms "
               "(%.4f ms/page amortized)\n",
               N_ITERS, N_ITERS * page / 1048576.0, batched, batched / N_ITERS);
        for (int i = 0; i < N_ITERS; i++) {
            CUdeviceptr p = va_base + (size_t)i * page;
            CK(cuMemUnmap(p, page));
            CK(cuMemRelease(handles[i]));
        }
    }

    /* --- export-to-shareable-handle probe (T4.6.4 prereq) --- */
    printf("\n=== cuMemExportToShareableHandle probe (T4.6.4 prereq) ===\n");
    {
        CUmemAllocationProp eprop = prop;
        eprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        CUmemGenericAllocationHandle eh;
        CUresult r = cuMemCreate(&eh, page, &eprop, 0);
        if (r != CUDA_SUCCESS) {
            const char *s = NULL; cuGetErrorString(r, &s);
            printf("cuMemCreate(POSIX_FD type): FAILED %d %s\n", r, s ? s : "?");
        } else {
            int fd = -1;
            r = cuMemExportToShareableHandle(&fd, eh,
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
            if (r != CUDA_SUCCESS) {
                const char *s = NULL; cuGetErrorString(r, &s);
                printf("cuMemExportToShareableHandle: FAILED %d %s\n", r, s ? s : "?");
            } else {
                printf("cuMemExportToShareableHandle: OK (fd=%d) "
                       "-> cross-process VMM sharing AVAILABLE for T4.6.4\n", fd);
                close(fd);
            }
            cuMemRelease(eh);
        }
    }

    CK(cuMemAddressFree(va_base, va_size));
    cuDevicePrimaryCtxRelease(dev);
    printf("\n=== done ===\n");
    return 0;
}
