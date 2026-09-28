// CIPHER Power Cap — see include/cipher_power_cap.h
//
// Lookup table baked from lever1_sweep_results.json (Mistral-7B graph decode,
// ≤5% tps regression budget):
//   B=1:  300 W   (+7.1 % tok/W,  −2.8 % tps)
//   B=8:  400 W   (+0.4 % tok/W,  −0.06 % tps; barely bites)
//   B=32: 700 W   (no win — natural draw 477 W)
//   B=64: 500 W   (+1.6 % tok/W, −0.6 % tps)
//
// For batch sizes between table entries, floor to the nearest <= B and use
// that cap. For batch > 64, fall back to 700 W (uncapped).

#include "cipher_power_cap.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <strings.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

namespace {

struct CapEntry { int batch; int cap_w; };
constexpr CapEntry kCapTable[] = {
    {1,   300},
    {8,   400},
    {32,  700},
    {64,  500},
};

std::atomic<int>  g_orig_cap_w{-1};
std::atomic<int>  g_cur_cap_w{-1};
std::atomic<int>  g_init_done{0};
std::atomic<int>  g_last_batch{-1};
std::atomic<int>  g_last_cap_w{-1};

std::atomic<uint64_t> g_apply_calls{0};
std::atomic<uint64_t> g_restore_calls{0};
std::atomic<uint64_t> g_nvml_ok{0};
std::atomic<uint64_t> g_subproc_ok{0};
std::atomic<uint64_t> g_fail{0};

// libnvidia-ml dlsym cache
struct Nvml {
    void*  lib = nullptr;
    int  (*Init)(void)                                  = nullptr;
    int  (*GetHandleByIndex)(unsigned int, void**)      = nullptr;
    int  (*GetPowerLimit)(void*, unsigned int*)         = nullptr;
    int  (*SetPowerLimit)(void*, unsigned int)          = nullptr;
    int  (*Shutdown)(void)                              = nullptr;
    void*  device = nullptr;
    int    initialized = 0;
} g_nvml;

bool nvml_open() {
    if (g_nvml.initialized) return true;
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) return false;
    g_nvml.lib = lib;
    g_nvml.Init             = reinterpret_cast<int(*)()>(dlsym(lib, "nvmlInit_v2"));
    if (!g_nvml.Init) g_nvml.Init = reinterpret_cast<int(*)()>(dlsym(lib, "nvmlInit"));
    g_nvml.GetHandleByIndex = reinterpret_cast<int(*)(unsigned int, void**)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
    if (!g_nvml.GetHandleByIndex) g_nvml.GetHandleByIndex =
        reinterpret_cast<int(*)(unsigned int, void**)>(dlsym(lib, "nvmlDeviceGetHandleByIndex"));
    g_nvml.GetPowerLimit    = reinterpret_cast<int(*)(void*, unsigned int*)>(dlsym(lib, "nvmlDeviceGetPowerManagementLimit"));
    g_nvml.SetPowerLimit    = reinterpret_cast<int(*)(void*, unsigned int)>(dlsym(lib, "nvmlDeviceSetPowerManagementLimit"));
    g_nvml.Shutdown         = reinterpret_cast<int(*)()>(dlsym(lib, "nvmlShutdown"));
    if (!g_nvml.Init || !g_nvml.GetHandleByIndex || !g_nvml.SetPowerLimit) return false;
    if (g_nvml.Init() != 0) return false;
    if (g_nvml.GetHandleByIndex(0, &g_nvml.device) != 0) return false;
    g_nvml.initialized = 1;
    return true;
}

int nvml_query_cap_w() {
    if (!nvml_open()) return -1;
    if (!g_nvml.GetPowerLimit) return -1;
    unsigned int milliwatts = 0;
    int rc = g_nvml.GetPowerLimit(g_nvml.device, &milliwatts);
    if (rc != 0) return -1;
    return static_cast<int>((milliwatts + 500) / 1000);
}

bool nvml_set_cap_w(int watts) {
    if (!nvml_open()) return false;
    if (!g_nvml.SetPowerLimit) return false;
    int rc = g_nvml.SetPowerLimit(g_nvml.device, static_cast<unsigned int>(watts) * 1000u);
    return (rc == 0);
}

bool subproc_set_cap_w(int watts) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%d", watts);
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        // child: redirect stdout/stderr to /dev/null
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, 1); dup2(devnull, 2); close(devnull); }
        execlp("sudo", "sudo", "-n", "nvidia-smi", "-pl", buf, (char*)nullptr);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return false;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int subproc_query_cap_w() {
    int pipefd[2];
    if (pipe(pipefd) < 0) return -1;
    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return -1; }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], 1);
        close(pipefd[1]);
        execlp("nvidia-smi", "nvidia-smi",
               "--query-gpu=power.limit",
               "--format=csv,noheader,nounits", (char*)nullptr);
        _exit(127);
    }
    close(pipefd[1]);
    char buf[64] = {};
    ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (n <= 0) return -1;
    double w = atof(buf);
    if (w <= 0) return -1;
    return static_cast<int>(w + 0.5);
}

bool set_cap_w(int watts) {
    // Prefer NVML in-process (no fork, ~ms). Fall back to sudo nvidia-smi.
    if (nvml_set_cap_w(watts)) {
        g_nvml_ok.fetch_add(1, std::memory_order_relaxed);
        g_cur_cap_w.store(watts, std::memory_order_relaxed);
        return true;
    }
    if (subproc_set_cap_w(watts)) {
        g_subproc_ok.fetch_add(1, std::memory_order_relaxed);
        g_cur_cap_w.store(watts, std::memory_order_relaxed);
        return true;
    }
    g_fail.fetch_add(1, std::memory_order_relaxed);
    return false;
}

void atexit_restore() {
    int orig = g_orig_cap_w.load(std::memory_order_relaxed);
    int cur  = g_cur_cap_w.load(std::memory_order_relaxed);
    if (orig > 0 && cur != orig) {
        set_cap_w(orig);
        std::fprintf(stderr, "[cipher_power_cap] restored cap to %dW on exit\n", orig);
    }
}

void sig_handler(int signo) {
    atexit_restore();
    signal(signo, SIG_DFL);
    raise(signo);
}

}  // namespace

extern "C" {

int cipher_power_cap_enabled(void) {
    const char* v = std::getenv("CIPHER_POWER_CAP");
    if (!v) return 0;
    return (strcasecmp(v, "on") == 0 || std::strcmp(v, "1") == 0
         || strcasecmp(v, "true") == 0);
}

int cipher_power_cap_init(void) {
    int expected = 0;
    if (!g_init_done.compare_exchange_strong(expected, 1)) return 1;
    int orig = nvml_query_cap_w();
    if (orig <= 0) orig = subproc_query_cap_w();
    if (orig <= 0) {
        std::fprintf(stderr, "[cipher_power_cap] init: could not query cap\n");
        g_init_done.store(0, std::memory_order_relaxed);
        return 0;
    }
    g_orig_cap_w.store(orig, std::memory_order_relaxed);
    g_cur_cap_w.store(orig, std::memory_order_relaxed);
    std::atexit(atexit_restore);
    struct sigaction sa{};
    sa.sa_handler = sig_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);
    if (cipher_power_cap_enabled()) {
        std::fprintf(stderr, "[cipher_power_cap] init: orig=%dW (will restore on exit)\n", orig);
    }
    return 1;
}

int cipher_power_cap_get_optimal_w(int batch_size) {
    int best = -1;
    for (auto& e : kCapTable) {
        if (e.batch <= batch_size && e.batch > best) {
            best = e.batch;
        }
    }
    if (best < 0) return -1;
    for (auto& e : kCapTable) {
        if (e.batch == best) return e.cap_w;
    }
    return -1;
}

int cipher_power_cap_apply_for_batch(int batch_size) {
    if (!cipher_power_cap_enabled()) return 0;
    if (!g_init_done.load(std::memory_order_relaxed)) {
        if (!cipher_power_cap_init()) return 0;
    }
    int cap_w = cipher_power_cap_get_optimal_w(batch_size);
    if (cap_w <= 0) return 0;
    g_apply_calls.fetch_add(1, std::memory_order_relaxed);
    int rc = set_cap_w(cap_w) ? 1 : 0;
    if (rc) {
        g_last_batch.store(batch_size, std::memory_order_relaxed);
        g_last_cap_w.store(cap_w, std::memory_order_relaxed);
    }
    return rc;
}

int cipher_power_cap_apply_w(int watts) {
    if (!cipher_power_cap_enabled()) return 0;
    if (!g_init_done.load(std::memory_order_relaxed)) {
        if (!cipher_power_cap_init()) return 0;
    }
    g_apply_calls.fetch_add(1, std::memory_order_relaxed);
    return set_cap_w(watts) ? 1 : 0;
}

int cipher_power_cap_restore(void) {
    int orig = g_orig_cap_w.load(std::memory_order_relaxed);
    if (orig <= 0) return 0;
    g_restore_calls.fetch_add(1, std::memory_order_relaxed);
    return set_cap_w(orig) ? 1 : 0;
}

int cipher_power_cap_stats(CipherPowerCapStats* out) {
    if (!out) return 0;
    std::memset(out, 0, sizeof(*out));
    out->enabled            = cipher_power_cap_enabled();
    out->original_cap_w     = g_orig_cap_w.load(std::memory_order_relaxed);
    out->current_cap_w      = g_cur_cap_w.load(std::memory_order_relaxed);
    out->last_applied_batch = g_last_batch.load(std::memory_order_relaxed);
    out->last_applied_cap_w = g_last_cap_w.load(std::memory_order_relaxed);
    out->apply_calls        = g_apply_calls.load(std::memory_order_relaxed);
    out->restore_calls      = g_restore_calls.load(std::memory_order_relaxed);
    out->nvml_success       = g_nvml_ok.load(std::memory_order_relaxed);
    out->subprocess_success = g_subproc_ok.load(std::memory_order_relaxed);
    out->failures           = g_fail.load(std::memory_order_relaxed);
    return 1;
}

void cipher_power_cap_report(void) {
    CipherPowerCapStats s; cipher_power_cap_stats(&s);
    std::fprintf(stderr,
        "[cipher_power_cap] enabled=%d orig=%dW cur=%dW last_batch=%d last_cap=%dW "
        "apply=%llu restore=%llu nvml_ok=%llu subproc_ok=%llu fail=%llu\n",
        s.enabled, s.original_cap_w, s.current_cap_w,
        s.last_applied_batch, s.last_applied_cap_w,
        (unsigned long long)s.apply_calls, (unsigned long long)s.restore_calls,
        (unsigned long long)s.nvml_success, (unsigned long long)s.subprocess_success,
        (unsigned long long)s.failures);
}

}  // extern "C"

// init priority 112 — after thermal_feedback (111). Idempotent.
__attribute__((constructor(112)))
static void cipher_power_cap_autoinit(void) {
    cipher_power_cap_init();
}
