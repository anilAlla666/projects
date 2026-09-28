/* T4.6.2 S2 — standalone unit test for the KV page allocator.
 *
 * Exercises slab create / ensure-grow / write-readback / page_info /
 * free / VA reuse without any PyTorch involvement.
 *
 * Build: gcc -O2 -I/home/ubuntu/cipher_rt_phase4 -o alloc_unit_test \
 *          alloc_unit_test.c /home/ubuntu/cipher_rt_phase4/cipher_rt_kv_alloc.c \
 *          -lcuda -lpthread
 */
#include "cipher_rt_kv_alloc.h"
#include <cuda.h>
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
	if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
	else         { printf("  ok:   %s\n", msg); } \
} while (0)

int main(void)
{
	printf("=== T4.6.2 KV allocator unit test ===\n");

	/* 1 GiB pool keeps the test fast. */
	int rc = cipher_rt_kv_alloc_init(1UL * 1024 * 1024 * 1024);
	CHECK(rc == 0, "alloc_init(1 GiB)");
	if (rc != 0) return 1;

	struct cipher_rt_kv_alloc_stats st;
	cipher_rt_kv_alloc_get_stats(&st);
	size_t page = st.page_size;
	printf("  page size = %zu KiB, pool = %.2f GiB\n",
	       page / 1024, st.bytes_va_reserved / (1024.0*1024*1024));

	/* --- slab create: 64 MiB max, 2 MiB initial --- */
	struct cipher_rt_kv_page_tag tag = {
		.tenant_id = 7, .seq_id = 42, .layer = 5,
		.head_kv = 3, .role = CIPHER_RT_KV_ROLE_K, .content_hash = 0,
	};
	unsigned long long base = 0;
	cipher_rt_kv_slab_t *s = cipher_rt_kv_slab_create(
		&tag, 64UL*1024*1024, 2UL*1024*1024, &base);
	CHECK(s != NULL && base != 0, "slab_create(64 MiB max, 2 MiB initial)");
	if (!s) { cipher_rt_kv_alloc_shutdown(); return 1; }

	/* write/read the first (mapped) page */
	CHECK(cuMemsetD8((CUdeviceptr)base, 0xAB, page) == CUDA_SUCCESS,
	      "cuMemsetD8 first page");
	unsigned char probe = 0;
	cuMemcpyDtoH(&probe, (CUdeviceptr)base, 1);
	CHECK(probe == 0xAB, "readback first page == 0xAB");

	/* --- ensure grows the slab: map up to 32 MiB --- */
	rc = cipher_rt_kv_slab_ensure(s, 32UL*1024*1024);
	CHECK(rc == 0, "slab_ensure(32 MiB)");
	/* the freshly-mapped region at +30 MiB must be writable */
	CHECK(cuMemsetD8((CUdeviceptr)base + 30UL*1024*1024, 0xCD, page)
	      == CUDA_SUCCESS, "write into grown region (+30 MiB)");
	probe = 0;
	cuMemcpyDtoH(&probe, (CUdeviceptr)base + 30UL*1024*1024, 1);
	CHECK(probe == 0xCD, "readback grown region == 0xCD");

	/* --- page_info reverse lookup --- */
	struct cipher_rt_kv_page_tag got;
	rc = cipher_rt_kv_page_info(base + 10UL*1024*1024, &got);
	CHECK(rc == 0, "page_info on managed ptr");
	CHECK(got.tenant_id == 7 && got.seq_id == 42 && got.layer == 5 &&
	      got.head_kv == 3 && got.role == CIPHER_RT_KV_ROLE_K,
	      "page_info tag matches (tenant=7 seq=42 layer=5 head=3 K)");
	rc = cipher_rt_kv_page_info(0xdeadbeef000ULL, &got);
	CHECK(rc == -1, "page_info on unmanaged ptr returns -1");

	cipher_rt_kv_alloc_get_stats(&st);
	printf("  stats: slabs=%lu pages_mapped=%lu resident=%lu peak=%lu "
	       "map_ms=%.3f\n", st.slabs_created, st.pages_mapped,
	       st.pages_resident, st.pages_resident_peak, st.map_ms_total);
	CHECK(st.pages_resident == 16, "16 pages resident (32 MiB / 2 MiB)");

	/* --- multi-slab + free + VA reuse --- */
	unsigned long long b2 = 0, b3 = 0;
	struct cipher_rt_kv_page_tag t2 = tag; t2.seq_id = 99;
	struct cipher_rt_kv_page_tag t3 = tag; t3.seq_id = 100;
	cipher_rt_kv_slab_t *s2 = cipher_rt_kv_slab_create(
		&t2, 16UL*1024*1024, 2UL*1024*1024, &b2);
	cipher_rt_kv_slab_t *s3 = cipher_rt_kv_slab_create(
		&t3, 16UL*1024*1024, 2UL*1024*1024, &b3);
	CHECK(s2 && s3 && b2 && b3, "two more slabs created");

	cipher_rt_kv_slab_free(s2);
	cipher_rt_kv_alloc_get_stats(&st);
	CHECK(st.slabs_freed == 1, "slab_free recorded");

	/* a new slab should reuse the freed VA hole */
	unsigned long long b4 = 0;
	struct cipher_rt_kv_page_tag t4 = tag; t4.seq_id = 200;
	cipher_rt_kv_slab_t *s4 = cipher_rt_kv_slab_create(
		&t4, 16UL*1024*1024, 2UL*1024*1024, &b4);
	CHECK(s4 && b4 == b2, "freed VA range reused by next slab");

	cipher_rt_kv_slab_free(s);
	cipher_rt_kv_slab_free(s3);
	cipher_rt_kv_slab_free(s4);
	cipher_rt_kv_alloc_get_stats(&st);
	CHECK(st.pages_resident == 0, "all pages released after frees");

	cipher_rt_kv_alloc_shutdown();
	printf(fails == 0 ? "\n=== ALL PASS ===\n" : "\n=== %d FAIL ===\n",
	       fails);
	return fails ? 1 : 0;
}
