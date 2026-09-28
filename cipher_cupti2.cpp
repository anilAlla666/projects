
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <atomic>
#include <cuda.h>
#include <cupti.h>

static std::atomic<uint64_t> g_count{0};
static CUpti_SubscriberHandle g_sub = nullptr;

static void CUPTIAPI on_any(
    void* userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const void* cbdata)
{
    if (!cbdata) return;
    const CUpti_CallbackData* d = (const CUpti_CallbackData*)cbdata;
    if (d->callbackSite != CUPTI_API_ENTER) return;

    const char* name = d->functionName ? d->functionName : "?";

    // Count and log launch-related calls
    if (strstr(name, "Launch") || strstr(name, "launch")) {
        uint64_t n = g_count.fetch_add(1, std::memory_order_relaxed) + 1;
        fprintf(stderr, "[CIPHER-CUPTI] #%lu domain=%d cbid=%d fn=%s\n",
                n, (int)domain, (int)cbid, name);
    }
}

extern "C" CUresult InitializeInjection(void) {
    fprintf(stderr, "[CIPHER-CUPTI] InitializeInjection\n");

    CUptiResult r = cuptiSubscribe(&g_sub, (CUpti_CallbackFunc)on_any, nullptr);
    if (r != CUPTI_SUCCESS) {
        const char* e = "?"; cuptiGetResultString(r, &e);
        fprintf(stderr, "[CIPHER-CUPTI] subscribe failed: %s\n", e);
        return CUDA_SUCCESS;
    }

    // Enable ENTIRE driver API domain — catch everything
    r = cuptiEnableDomain(1, g_sub, CUPTI_CB_DOMAIN_DRIVER_API);
    if (r != CUPTI_SUCCESS) {
        const char* e = "?"; cuptiGetResultString(r, &e);
        fprintf(stderr, "[CIPHER-CUPTI] enableDomain failed: %s\n", e);
    } else {
        fprintf(stderr, "[CIPHER-CUPTI] entire driver domain enabled\n");
    }

    // Also enable runtime domain
    cuptiEnableDomain(1, g_sub, CUPTI_CB_DOMAIN_RUNTIME_API);

    return CUDA_SUCCESS;
}

extern "C" void __attribute__((destructor)) fini(void) {
    fprintf(stderr, "[CIPHER-CUPTI] Total launch calls intercepted: %lu\n",
            g_count.load());
    if (g_sub) cuptiUnsubscribe(g_sub);
}
