// cipher_flopd — CP 3.3 continuous FLOP telemetry daemon.
//
// Root daemon. Uses CUPTI PM Sampling to continuously read the H100's
// tensor-pipe activity counter device-wide (process-independent), derives
// the achieved tensor-FLOP rate, and submits it to cipher_kmod via
// CIPHER_SUBMIT_FLOP_SAMPLE (~10 Hz). The kmod owns the FLOP-series ring
// and the per-tenant attribution.
//
// Metric (build STEP-1 finding): PM Sampling on this GH100 fits exactly
// one metric per single-pass config, so we sample the tensor pipe —
// sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed — the
// dominant FLOP path for LLM inference and the quantity MFU measures. The
// CUDA-core FP pipe is not co-samplable; device_fp_flops_per_s is left 0.
//
// Build: g++ cipher_flopd.cpp -I/usr/include -I/home/ubuntu/cipher_kmod \
//        -L/usr/lib/x86_64-linux-gnu -lcuda -lcupti -lnvidia-ml -o cipher_flopd
// Run:   sudo ./cipher_flopd   (profiling requires root on this pod)
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <csignal>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include <cuda.h>
#include <cupti_result.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>
#include <cupti_pmsampling.h>
#include <nvml.h>

#include "cipher_ioctl.h"

// H100 fp16 tensor peak — the MFU denominator CIPHER uses elsewhere.
static const double CIPHER_PEAK_TENSOR_FLOPS = 989.0e12;
static const char  *kMetric =
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed";

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static const char *ck(CUptiResult r) {
    const char *s = 0; cuptiGetResultString(r, &s); return s ? s : "?";
}
#define CUPTI_OK(call) do { CUptiResult _r = (call); \
    if (_r != CUPTI_SUCCESS) { \
        fprintf(stderr, "[cipher_flopd] %s -> %s\n", #call, ck(_r)); \
        return 1; } } while (0)
#define CU_OK(call) do { CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { const char *s = 0; cuGetErrorString(_r, &s); \
        fprintf(stderr, "[cipher_flopd] %s -> %s\n", #call, s ? s : "?"); \
        return 1; } } while (0)

static uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    // sampling cadence
    uint64_t sample_interval_ns = 1000000;   // 1 ms hardware sample
    uint64_t decode_period_us   = 100000;    // 100 ms decode/submit (~10 Hz)
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--interval-ms") && i + 1 < argc)
            decode_period_us = (uint64_t)atoi(argv[++i]) * 1000;
    }

    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered — usable when redirected */
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    printf("[cipher_flopd] CP 3.3 FLOP telemetry daemon starting\n");

    // --- CUDA + CUPTI profiler init ---
    CU_OK(cuInit(0));
    CUdevice dev; CU_OK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CU_OK(cuCtxCreate(&ctx, 0, dev));

    CUpti_Profiler_Initialize_Params pinit = {};
    pinit.structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
    CUPTI_OK(cuptiProfilerInitialize(&pinit));

    CUpti_Device_GetChipName_Params chip = {};
    chip.structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE;
    chip.deviceIndex = 0;
    CUPTI_OK(cuptiDeviceGetChipName(&chip));
    printf("[cipher_flopd] chip=%s  metric=%s\n", chip.pChipName, kMetric);

    // counter availability image
    CUpti_PmSampling_GetCounterAvailability_Params ca = {};
    ca.structSize = CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE;
    ca.deviceIndex = 0;
    CUPTI_OK(cuptiPmSamplingGetCounterAvailability(&ca));
    std::vector<uint8_t> caimg(ca.counterAvailabilityImageSize);
    ca.pCounterAvailabilityImage = caimg.data();
    CUPTI_OK(cuptiPmSamplingGetCounterAvailability(&ca));

    // --- config image (Profiler Host API) ---
    CUpti_Profiler_Host_Initialize_Params hi = {};
    hi.structSize = CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE;
    hi.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    hi.pChipName = chip.pChipName;
    hi.pCounterAvailabilityImage = caimg.data();
    CUPTI_OK(cuptiProfilerHostInitialize(&hi));

    const char *metrics[1] = { kMetric };
    CUpti_Profiler_Host_ConfigAddMetrics_Params am = {};
    am.structSize = CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE;
    am.pHostObject = hi.pHostObject;
    am.ppMetricNames = metrics;
    am.numMetrics = 1;
    CUPTI_OK(cuptiProfilerHostConfigAddMetrics(&am));

    CUpti_Profiler_Host_GetConfigImageSize_Params cis = {};
    cis.structSize = CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE;
    cis.pHostObject = hi.pHostObject;
    CUPTI_OK(cuptiProfilerHostGetConfigImageSize(&cis));
    std::vector<uint8_t> cfg(cis.configImageSize);
    CUpti_Profiler_Host_GetConfigImage_Params cig = {};
    cig.structSize = CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE;
    cig.pHostObject = hi.pHostObject;
    cig.configImageSize = cfg.size();
    cig.pConfigImage = cfg.data();
    CUPTI_OK(cuptiProfilerHostGetConfigImage(&cig));

    // --- enable + configure PM sampling ---
    CUpti_PmSampling_Enable_Params en = {};
    en.structSize = CUpti_PmSampling_Enable_Params_STRUCT_SIZE;
    en.deviceIndex = 0;
    CUPTI_OK(cuptiPmSamplingEnable(&en));
    CUpti_PmSampling_Object *pm = en.pPmSamplingObject;

    CUpti_PmSampling_SetConfig_Params sc = {};
    sc.structSize = CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE;
    sc.pPmSamplingObject = pm;
    sc.configSize = cfg.size();
    sc.pConfig = cfg.data();
    sc.hardwareBufferSize = 2 * 1024 * 1024;        // 2 MiB raw buffer
    sc.samplingInterval = sample_interval_ns;
    sc.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
    sc.hwBufferAppendMode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST;
    CUPTI_OK(cuptiPmSamplingSetConfig(&sc));

    const uint32_t kMaxSamples = 512;
    CUpti_PmSampling_GetCounterDataSize_Params cds = {};
    cds.structSize = CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE;
    cds.pPmSamplingObject = pm;
    cds.pMetricNames = metrics;
    cds.numMetrics = 1;
    cds.maxSamples = kMaxSamples;
    CUPTI_OK(cuptiPmSamplingGetCounterDataSize(&cds));
    std::vector<uint8_t> cdata(cds.counterDataSize);

    CUpti_PmSampling_CounterDataImage_Initialize_Params cdi = {};
    cdi.structSize =
        CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE;
    cdi.pPmSamplingObject = pm;
    cdi.counterDataSize = cdata.size();
    cdi.pCounterData = cdata.data();
    CUPTI_OK(cuptiPmSamplingCounterDataImageInitialize(&cdi));

    // --- NVML for the cross-check fields ---
    bool nvml_ok = (nvmlInit_v2() == NVML_SUCCESS);
    nvmlDevice_t nvdev{};
    if (nvml_ok && nvmlDeviceGetHandleByIndex_v2(0, &nvdev) != NVML_SUCCESS)
        nvml_ok = false;
    if (!nvml_ok)
        fprintf(stderr, "[cipher_flopd] NVML unavailable — cross-check "
                        "fields will be 0\n");

    int fd = open("/dev/cipher", O_RDWR);
    if (fd < 0) { perror("[cipher_flopd] open /dev/cipher"); return 1; }

    CUpti_PmSampling_Start_Params st = {};
    st.structSize = CUpti_PmSampling_Start_Params_STRUCT_SIZE;
    st.pPmSamplingObject = pm;
    CUPTI_OK(cuptiPmSamplingStart(&st));
    printf("[cipher_flopd] PM sampling started; submitting to /dev/cipher "
           "every %llu ms\n", (unsigned long long)(decode_period_us / 1000));

    uint32_t seq = 0;
    uint64_t submitted = 0;
    while (!g_stop) {
        usleep(decode_period_us);
        uint64_t t_start = now_ns();

        CUpti_PmSampling_DecodeData_Params dd = {};
        dd.structSize = CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE;
        dd.pPmSamplingObject = pm;
        dd.pCounterDataImage = cdata.data();
        dd.counterDataImageSize = cdata.size();
        CUptiResult dr = cuptiPmSamplingDecodeData(&dd);
        if (dr != CUPTI_SUCCESS) {
            fprintf(stderr, "[cipher_flopd] DecodeData -> %s\n", ck(dr));
            continue;
        }
        if (dd.overflow)
            fprintf(stderr, "[cipher_flopd] HW buffer overflow — "
                            "raising interval or buffer recommended\n");

        CUpti_PmSampling_GetCounterDataInfo_Params gi = {};
        gi.structSize = CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE;
        gi.pCounterDataImage = cdata.data();
        gi.counterDataImageSize = cdata.size();
        if (cuptiPmSamplingGetCounterDataInfo(&gi) != CUPTI_SUCCESS)
            continue;

        size_t nsamp = gi.numCompletedSamples;
        double pct_sum = 0.0;
        size_t pct_n = 0;
        uint64_t span_ns = 0, first_ts = 0, last_ts = 0;
        for (size_t i = 0; i < nsamp; i++) {
            CUpti_PmSampling_CounterData_GetSampleInfo_Params si = {};
            si.structSize =
                CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE;
            si.pPmSamplingObject = pm;
            si.pCounterDataImage = cdata.data();
            si.counterDataImageSize = cdata.size();
            si.sampleIndex = i;
            if (cuptiPmSamplingCounterDataGetSampleInfo(&si) != CUPTI_SUCCESS)
                continue;
            if (i == 0) first_ts = si.startTimestamp;
            last_ts = si.endTimestamp;

            double val = 0.0;
            CUpti_Profiler_Host_EvaluateToGpuValues_Params ev = {};
            ev.structSize =
                CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE;
            ev.pHostObject = hi.pHostObject;
            ev.pCounterDataImage = cdata.data();
            ev.counterDataImageSize = cdata.size();
            ev.rangeIndex = i;
            ev.ppMetricNames = metrics;
            ev.numMetrics = 1;
            ev.pMetricValues = &val;
            if (cuptiProfilerHostEvaluateToGpuValues(&ev) != CUPTI_SUCCESS)
                continue;
            if (val < 0.0)   val = 0.0;
            if (val > 100.0) val = 100.0;
            pct_sum += val;
            pct_n++;
        }
        if (last_ts > first_ts) span_ns = last_ts - first_ts;

        double pct = pct_n ? (pct_sum / pct_n) : 0.0;
        struct cipher_flop_sample s;
        memset(&s, 0, sizeof(s));
        s.timestamp_ns = now_ns();
        s.interval_ns = span_ns ? span_ns : decode_period_us * 1000;
        s.device_tensor_flops_per_s =
            (uint64_t)(pct / 100.0 * CIPHER_PEAK_TENSOR_FLOPS);
        s.device_fp_flops_per_s = 0;          // FP pipe not co-samplable
        s.tensor_pipe_milli_pct = (uint32_t)(pct * 1000.0 + 0.5);
        s.source_seq = ++seq;
        if (nvml_ok) {
            nvmlUtilization_t u;
            unsigned int clk = 0, mw = 0;
            if (nvmlDeviceGetUtilizationRates(nvdev, &u) == NVML_SUCCESS)
                s.sm_util_pct = u.gpu;
            if (nvmlDeviceGetClockInfo(nvdev, NVML_CLOCK_SM, &clk)
                == NVML_SUCCESS)
                s.sm_clock_mhz = clk;
            if (nvmlDeviceGetPowerUsage(nvdev, &mw) == NVML_SUCCESS)
                s.power_mw = mw;
        }

        if (ioctl(fd, CIPHER_SUBMIT_FLOP_SAMPLE, &s) != 0) {
            perror("[cipher_flopd] ioctl SUBMIT_FLOP_SAMPLE");
        } else {
            submitted++;
            if (submitted % 50 == 1)
                printf("[cipher_flopd] seq=%u tensor=%.2f%% "
                       "%.1f TFLOP/s sm_util=%u%% clk=%u (%zu samples, "
                       "decode %.1f ms)\n",
                       seq, pct, s.device_tensor_flops_per_s / 1e12,
                       s.sm_util_pct, s.sm_clock_mhz, pct_n,
                       (now_ns() - t_start) / 1e6);
        }

        // reset the counter data image for the next decode window
        CUpti_PmSampling_CounterDataImage_Initialize_Params ri = {};
        ri.structSize =
            CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE;
        ri.pPmSamplingObject = pm;
        ri.counterDataSize = cdata.size();
        ri.pCounterData = cdata.data();
        cuptiPmSamplingCounterDataImageInitialize(&ri);
    }

    printf("[cipher_flopd] stopping (submitted %llu samples)\n",
           (unsigned long long)submitted);
    CUpti_PmSampling_Stop_Params sp = {};
    sp.structSize = CUpti_PmSampling_Stop_Params_STRUCT_SIZE;
    sp.pPmSamplingObject = pm;
    cuptiPmSamplingStop(&sp);
    CUpti_PmSampling_Disable_Params di = {};
    di.structSize = CUpti_PmSampling_Disable_Params_STRUCT_SIZE;
    di.pPmSamplingObject = pm;
    cuptiPmSamplingDisable(&di);
    close(fd);
    if (nvml_ok) nvmlShutdown();
    return 0;
}
