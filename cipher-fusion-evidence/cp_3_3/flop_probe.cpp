// CP 3.3 build STEP 1 — CUPTI PM Sampling de-risk + FLOP-metric enumeration.
//
// Confirms: (1) chip name, (2) PM Sampling enables (profiling privilege),
// (3) which FLOP-relevant metrics exist and which combine into a single-pass
// config image (PM Sampling requires single pass). Output drives the
// cipher_flopd metric selection. Build: g++ flop_probe.cpp -I/usr/include
// -lcuda -lcupti -o flop_probe ; run as root.
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cuda.h>
#include <cupti_result.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>
#include <cupti_pmsampling.h>

static const char *ck(CUptiResult r) {
    const char *s = 0; cuptiGetResultString(r, &s); return s ? s : "?";
}

#define CUOK(call) do { CUresult _r = (call); if (_r != CUDA_SUCCESS) { \
    const char *s = 0; cuGetErrorString(_r, &s); \
    printf("[CUDA FAIL] %s -> %s\n", #call, s ? s : "?"); return 1; } } while (0)

// Test a metric set: fresh host object -> add metrics -> config image ->
// number of passes. PM Sampling needs exactly 1 pass.
static void test_set(const char *chip, const uint8_t *caimg,
                     const char *label, std::vector<const char*> metrics) {
    CUpti_Profiler_Host_Initialize_Params hi = {
        CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE };
    hi.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    hi.pChipName = chip;
    hi.pCounterAvailabilityImage = caimg;
    CUptiResult r = cuptiProfilerHostInitialize(&hi);
    if (r != CUPTI_SUCCESS) { printf("  [%s] HostInitialize FAIL: %s\n",
                                     label, ck(r)); return; }

    CUpti_Profiler_Host_ConfigAddMetrics_Params am = {
        CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE };
    am.pHostObject = hi.pHostObject;
    am.ppMetricNames = metrics.data();
    am.numMetrics = metrics.size();
    r = cuptiProfilerHostConfigAddMetrics(&am);
    if (r != CUPTI_SUCCESS) {
        printf("  [%s] ConfigAddMetrics FAIL: %s\n", label, ck(r));
        CUpti_Profiler_Host_Deinitialize_Params dh = {
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE };
        dh.pHostObject = hi.pHostObject; cuptiProfilerHostDeinitialize(&dh);
        return;
    }

    CUpti_Profiler_Host_GetConfigImageSize_Params cs = {
        CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE };
    cs.pHostObject = hi.pHostObject;
    r = cuptiProfilerHostGetConfigImageSize(&cs);
    std::vector<uint8_t> cfg;
    size_t passes = 0;
    if (r == CUPTI_SUCCESS) {
        cfg.resize(cs.configImageSize);
        CUpti_Profiler_Host_GetConfigImage_Params ci = {
            CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE };
        ci.pHostObject = hi.pHostObject;
        ci.configImageSize = cfg.size();
        ci.pConfigImage = cfg.data();
        r = cuptiProfilerHostGetConfigImage(&ci);
    }
    if (r == CUPTI_SUCCESS) {
        CUpti_Profiler_Host_GetNumOfPasses_Params np = {
            CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE };
        np.configImageSize = cfg.size();
        np.pConfigImage = cfg.data();
        r = cuptiProfilerHostGetNumOfPasses(&np);
        passes = np.numOfPasses;
    }
    printf("  [%s] %zu metrics -> %s | configImage=%zu B | passes=%zu %s\n",
           label, metrics.size(), ck(r), cfg.size(), passes,
           (r == CUPTI_SUCCESS && passes == 1) ? "<= SINGLE-PASS OK" :
           (r == CUPTI_SUCCESS ? "<= MULTI-PASS (PM sampling rejects)" : ""));

    CUpti_Profiler_Host_Deinitialize_Params dh = {
        CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE };
    dh.pHostObject = hi.pHostObject;
    cuptiProfilerHostDeinitialize(&dh);
}

int main() {
    printf("=== CP 3.3 flop_probe — CUPTI PM Sampling de-risk ===\n");
    CUOK(cuInit(0));
    CUdevice dev; CUOK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CUOK(cuCtxCreate(&ctx, 0, dev));

    CUpti_Profiler_Initialize_Params pinit = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
    CUptiResult ir = cuptiProfilerInitialize(&pinit);
    printf("cuptiProfilerInitialize: %s\n", ck(ir));

    CUpti_Device_GetChipName_Params cn = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    cn.deviceIndex = 0;
    CUptiResult r = cuptiDeviceGetChipName(&cn);
    printf("chip: %s  (%s)\n",
           r == CUPTI_SUCCESS ? cn.pChipName : "<fail>", ck(r));
    const char *chip = (r == CUPTI_SUCCESS) ? cn.pChipName : "gh100";

    // counter availability image
    CUpti_PmSampling_GetCounterAvailability_Params ca = {
        CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE };
    ca.deviceIndex = 0;
    r = cuptiPmSamplingGetCounterAvailability(&ca);
    printf("GetCounterAvailability(size): %s\n", ck(r));
    std::vector<uint8_t> caimg;
    if (r == CUPTI_SUCCESS) {
        caimg.resize(ca.counterAvailabilityImageSize);
        ca.pCounterAvailabilityImage = caimg.data();
        r = cuptiPmSamplingGetCounterAvailability(&ca);
        printf("GetCounterAvailability(fill): %s  size=%zu\n",
               ck(r), caimg.size());
    }
    const uint8_t *cap = caimg.empty() ? nullptr : caimg.data();

    // PM Sampling enable — the privilege gate
    CUpti_PmSampling_Enable_Params en = {
        CUpti_PmSampling_Enable_Params_STRUCT_SIZE };
    en.deviceIndex = 0;
    r = cuptiPmSamplingEnable(&en);
    printf("\n*** PmSamplingEnable: %s ***\n", ck(r));
    bool pm_ok = (r == CUPTI_SUCCESS);
    if (pm_ok) {
        CUpti_PmSampling_Disable_Params di = {
            CUpti_PmSampling_Disable_Params_STRUCT_SIZE };
        di.pPmSamplingObject = en.pPmSamplingObject;
        cuptiPmSamplingDisable(&di);
    }

    // enumerate base metrics (counter + throughput), print FLOP-relevant ones
    CUpti_Profiler_Host_Initialize_Params hi = {
        CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE };
    hi.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    hi.pChipName = chip;
    hi.pCounterAvailabilityImage = cap;
    r = cuptiProfilerHostInitialize(&hi);
    printf("\nHostInitialize: %s\n", ck(r));
    if (r == CUPTI_SUCCESS) {
        const CUpti_MetricType types[2] = { CUPTI_METRIC_TYPE_COUNTER,
                                            CUPTI_METRIC_TYPE_THROUGHPUT };
        const char *tnames[2] = { "COUNTER", "THROUGHPUT" };
        for (int t = 0; t < 2; t++) {
            CUpti_Profiler_Host_GetBaseMetrics_Params bm = {
                CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE };
            bm.pHostObject = hi.pHostObject;
            bm.metricType = types[t];
            r = cuptiProfilerHostGetBaseMetrics(&bm);
            printf("\n-- base metrics [%s]: %s  count=%zu --\n",
                   tnames[t], ck(r), r == CUPTI_SUCCESS ? bm.numMetrics : 0);
            if (r != CUPTI_SUCCESS) continue;
            int shown = 0;
            for (size_t i = 0; i < bm.numMetrics; i++) {
                const char *m = bm.ppMetricNames[i];
                if ((strstr(m, "sm__") == m || strstr(m, "gpc__") == m ||
                     strstr(m, "gr__") == m) &&
                    (strstr(m, "pipe_") || strstr(m, "cycles_active") ||
                     strstr(m, "cycles_elapsed") ||
                     strstr(m, "inst_executed"))) {
                    printf("    %s\n", m); shown++;
                }
            }
            printf("    (%d FLOP-relevant of %zu shown)\n",
                   shown, bm.numMetrics);
        }
        CUpti_Profiler_Host_Deinitialize_Params dh = {
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE };
        dh.pHostObject = hi.pHostObject;
        cuptiProfilerHostDeinitialize(&dh);
    }

    // candidate FLOP metric sets — single-pass test (PM sampling needs 1 pass)
    printf("\n=== single-pass config tests ===\n");
    test_set(chip, cap, "tensor.sum", {
        "sm__pipe_tensor_cycles_active.sum" });
    test_set(chip, cap, "fma.sum", {
        "sm__pipe_fma_cycles_active.sum" });
    test_set(chip, cap, "tensor.pct", {
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed" });
    test_set(chip, cap, "fma.pct", {
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed" });
    test_set(chip, cap, "tensor.pct+fma.pct", {
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed" });
    test_set(chip, cap, "tensor.sum+fma.sum", {
        "sm__pipe_tensor_cycles_active.sum",
        "sm__pipe_fma_cycles_active.sum" });
    test_set(chip, cap, "tensor+fma+cycles_elapsed", {
        "sm__pipe_tensor_cycles_active.sum",
        "sm__pipe_fma_cycles_active.sum",
        "sm__cycles_elapsed.sum" });
    test_set(chip, cap, "tensor+fma+fp64+cycles", {
        "sm__pipe_tensor_cycles_active.sum",
        "sm__pipe_fma_cycles_active.sum",
        "sm__pipe_fp64_cycles_active.sum",
        "sm__cycles_elapsed.sum" });
    test_set(chip, cap, "fma_fp16.sum", {
        "sm__inst_executed_pipe_fma_type_fp16.sum" });

    printf("\n=== VERDICT ===\n");
    printf("PM Sampling privilege: %s\n",
           pm_ok ? "OK (profiling permitted)"
                 : "DENIED/UNSUPPORTED — see error above");
    printf("chip: %s\n", chip);
    printf("(metric selection for cipher_flopd: use the single-pass set "
           "above with the most FLOP coverage)\n");
    return 0;
}
