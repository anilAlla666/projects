// Probe (mmap-warm, Assessment 2): does cudaMemcpy H2D from an mmap'd FILE-backed (pageable) host pointer work
// bit-identically, and at what bandwidth vs pinned? This de-risks the mmap-warm source before building it in.
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double ms_since(struct timespec a, struct timespec b){ return (b.tv_sec-a.tv_sec)*1e3 + (b.tv_nsec-a.tv_nsec)/1e6; }
int main(void){
	size_t N = 256ul*1024*1024;   /* ~one decoder layer */
	const char *path = "/home/ubuntu/pager_mmap_test.bin";
	int fd = open(path, O_RDWR|O_CREAT|O_TRUNC, 0644);
	if (fd<0 || ftruncate(fd, N)!=0){ perror("file"); return 1; }
	unsigned char *m = mmap(NULL, N, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if (m==MAP_FAILED){ perror("mmap"); return 1; }
	for (size_t i=0;i<N;i++) m[i] = (unsigned char)((i*2654435761u)>>13);
	msync(m, N, MS_SYNC);
	void *d; cudaMalloc(&d, N);
	struct timespec t0,t1;
	cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC,&t0);
	cudaMemcpy(d, m, N, cudaMemcpyHostToDevice); cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC,&t1);
	double mmap_ms = ms_since(t0,t1);
	unsigned char *h = malloc(N); cudaMemcpy(h, d, N, cudaMemcpyDeviceToHost);
	int ok = (memcmp(h, m, N)==0);
	void *p; cudaMallocHost(&p, N); memcpy(p, m, N);
	cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC,&t0);
	cudaMemcpy(d, p, N, cudaMemcpyHostToDevice); cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC,&t1);
	double pin_ms = ms_since(t0,t1);
	printf("mmap-warm H2D: %.0fMB in %.1fms = %.1f GB/s  bit-identical=%d\n", N/1e6, mmap_ms, N/1e9/(mmap_ms/1e3), ok);
	printf("pinned   H2D: %.0fMB in %.1fms = %.1f GB/s\n", N/1e6, pin_ms, N/1e9/(pin_ms/1e3));
	munmap(m,N); close(fd); unlink(path);
	return ok?0:2;
}
