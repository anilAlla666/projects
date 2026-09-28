/* CP 5.4 Step 1.5B-4 — CUPTI cuModuleLoad* counter.
 *
 * Counts every cuModuleLoad / cuModuleLoadData / cuModuleLoadDataEx /
 * cuModuleLoadFatBinary the process issues, and logs each with a monotonic
 * timestamp. CUPTI's driver-API trace catches the call regardless of how the
 * caller resolved the symbol — which matters here: the Marlin engine resolves
 * cuModuleLoadData via dlsym(libcuda_handle, "cuModuleLoadData"), so a plain
 * LD_PRELOAD interposer would be bypassed and would false-negative. CUPTI does
 * not have that blind spot.
 *
 * Throwaway Step 1.5 measurement instrument — not a campaign anchor, not
 * substrate. Exposes s15_cumod_count() / s15_cumod_mark() for the driver.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdatomic.h>
#include <time.h>
#include <cupti.h>

static atomic_int g_count = 0;
static CUpti_SubscriberHandle g_sub;

static double now_s(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

static void CUPTIAPI cb(void *ud, CUpti_CallbackDomain dom,
                        CUpti_CallbackId cbid, const void *cbdata)
{
	const CUpti_CallbackData *d = (const CUpti_CallbackData *)cbdata;

	if (dom != CUPTI_CB_DOMAIN_DRIVER_API)
		return;
	if (cbid != CUPTI_DRIVER_TRACE_CBID_cuModuleLoad &&
	    cbid != CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData &&
	    cbid != CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx &&
	    cbid != CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary)
		return;
	if (d->callbackSite != CUPTI_API_ENTER)
		return;

	int n = atomic_fetch_add(&g_count, 1) + 1;
	fprintf(stderr, "[cumod_count] %.6f  LOAD #%d  %s\n",
	        now_s(), n, d->functionName ? d->functionName : "?");
	fflush(stderr);
}

__attribute__((constructor))
static void cumod_init(void)
{
	CUptiResult r = cuptiSubscribe(&g_sub, (CUpti_CallbackFunc)cb, NULL);
	if (r != CUPTI_SUCCESS) {
		const char *m = NULL;
		cuptiGetResultString(r, &m);
		fprintf(stderr, "[cumod_count] cuptiSubscribe FAILED: %s\n",
		        m ? m : "?");
		return;
	}
	cuptiEnableDomain(1, g_sub, CUPTI_CB_DOMAIN_DRIVER_API);
	fprintf(stderr, "[cumod_count] CUPTI driver-API subscriber active\n");
	fflush(stderr);
}

int s15_cumod_count(void)
{
	return atomic_load(&g_count);
}

void s15_cumod_mark(const char *tag)
{
	fprintf(stderr, "[cumod_count] %.6f  MARK  %s  (count=%d)\n",
	        now_s(), tag ? tag : "?", atomic_load(&g_count));
	fflush(stderr);
}
