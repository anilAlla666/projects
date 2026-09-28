/* D3 cross-process GPU memory IPC probe.
 *
 * This is a throwaway discovery probe.
 *
 * Process A: allocate device buffer, write known value, export IPC handle to /tmp/cipher_ipc_handle.bin.
 * Process B: read /tmp/cipher_ipc_handle.bin, open handle, read the value.
 *
 * If B sees A's value, cross-process GPU memory sharing works on this pod
 * and the kmod-managed cross-tenant page pool architecture is viable.
 *
 * Build:
 *   gcc -O2 -o /tmp/cuda_ipc_test /tmp/cuda_ipc_test.c -ldl -lcuda
 *
 * Run:
 *   /tmp/cuda_ipc_test producer
 *   sleep 1
 *   /tmp/cuda_ipc_test consumer
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cuda.h>

#define HANDLE_PATH "/tmp/cipher_ipc_handle.bin"
#define MAGIC_VALUE 0xCAFEBABE12345678ULL

static int check(CUresult r, const char *what) {
    if (r != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(r, &name);
        fprintf(stderr, "FAIL %s -> %s\n", what, name ? name : "?");
        return -1;
    }
    return 0;
}

int producer(void) {
    if (check(cuInit(0), "cuInit")) return 1;
    CUdevice dev;
    if (check(cuDeviceGet(&dev, 0), "cuDeviceGet")) return 1;
    CUcontext ctx;
    if (check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain")) return 1;
    if (check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")) return 1;

    /* Allocate 8 MB device buffer (must be at least 2 MB granularity for IPC). */
    size_t sz = 8 * 1024 * 1024;
    CUdeviceptr dptr = 0;
    if (check(cuMemAlloc(&dptr, sz), "cuMemAlloc")) return 1;

    /* Write magic value at offset 0. */
    unsigned long long magic = MAGIC_VALUE;
    if (check(cuMemcpyHtoD(dptr, &magic, sizeof(magic)), "cuMemcpyHtoD magic")) return 1;
    if (check(cuCtxSynchronize(), "cuCtxSynchronize")) return 1;

    /* Export IPC handle. */
    CUipcMemHandle h;
    CUresult r = cuIpcGetMemHandle(&h, dptr);
    if (r != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(r, &name);
        fprintf(stderr, "[producer] cuIpcGetMemHandle FAILED: %s\n", name ? name : "?");
        return 2;
    }

    FILE *f = fopen(HANDLE_PATH, "wb");
    if (!f) { perror("fopen"); return 3; }
    fwrite(&h, sizeof(h), 1, f);
    fclose(f);

    printf("[producer] dptr=0x%llx size=%zu wrote handle to %s\n",
           (unsigned long long)dptr, sz, HANDLE_PATH);
    printf("[producer] sleeping 30s for consumer...\n");
    fflush(stdout);
    sleep(30);

    cuMemFree(dptr);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}

int consumer(void) {
    if (check(cuInit(0), "cuInit")) return 1;
    CUdevice dev;
    if (check(cuDeviceGet(&dev, 0), "cuDeviceGet")) return 1;
    CUcontext ctx;
    if (check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain")) return 1;
    if (check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")) return 1;

    CUipcMemHandle h;
    FILE *f = fopen(HANDLE_PATH, "rb");
    if (!f) { perror("fopen consumer"); return 1; }
    fread(&h, sizeof(h), 1, f);
    fclose(f);

    CUdeviceptr dptr = 0;
    CUresult r = cuIpcOpenMemHandle(&dptr, h, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    if (r != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(r, &name);
        fprintf(stderr, "[consumer] cuIpcOpenMemHandle FAILED: %s\n", name ? name : "?");
        return 2;
    }
    printf("[consumer] opened handle as dptr=0x%llx\n", (unsigned long long)dptr);

    unsigned long long got = 0;
    if (check(cuMemcpyDtoH(&got, dptr, sizeof(got)), "cuMemcpyDtoH")) return 1;
    if (check(cuCtxSynchronize(), "cuCtxSynchronize")) return 1;

    printf("[consumer] read magic = 0x%llx (expected 0x%llx) -> %s\n",
           got, (unsigned long long)MAGIC_VALUE,
           got == MAGIC_VALUE ? "MATCH" : "MISMATCH");

    cuIpcCloseMemHandle(dptr);
    cuDevicePrimaryCtxRelease(dev);
    return got == MAGIC_VALUE ? 0 : 4;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s producer|consumer\n", argv[0]);
        return 1;
    }
    if (!strcmp(argv[1], "producer")) return producer();
    if (!strcmp(argv[1], "consumer")) return consumer();
    fprintf(stderr, "unknown: %s\n", argv[1]);
    return 1;
}
