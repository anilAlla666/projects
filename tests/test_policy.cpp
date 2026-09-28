// M1.T7 smoke test for cipher_policy_*: load policy.json, verify
// per-tenant cap and banned-op queries match the file, then SIGHUP
// reload after rewriting the file and re-verify.
//
// Build: g++ -std=c++17 -O2 tests/test_policy.cpp -ldl -o build/test_policy
// Run:   CIPHER_POLICY_PATH=/tmp/policy.json CIPHER_TENANT_ID=shared \
//        build/test_policy ./libcipher_rt.so

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <dlfcn.h>
#include <unistd.h>

typedef int      (*fn_init)(void);
typedef int      (*fn_op_allowed)(int);
typedef int      (*fn_max_power)(void);
typedef int      (*fn_max_tok_s)(void);
typedef int      (*fn_min_lat_ms)(void);
typedef const char* (*fn_tenant)(void);
typedef int      (*fn_reload)(void);
typedef unsigned long long (*fn_reload_count)(void);

#define OP_SUBSTITUTE_KOOPMAN  9
#define OP_FAIRNESS           16

#define EXPECT_EQ(a, b, label) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        std::fprintf(stderr, "FAIL %s: got %ld want %ld\n", \
            label, (long)_a, (long)_b); \
        return 1; \
    } else { \
        std::fprintf(stderr, "PASS %s = %ld\n", label, (long)_a); \
    } \
} while (0)

int main(int argc, char** argv) {
    const char* so_path = (argc > 1) ? argv[1] : "./libcipher_rt.so";

    // Always use a fresh tempfile, regardless of CIPHER_POLICY_PATH the
    // caller set. Avoids the "previous run's leftover" trap in CI.
    char tmp_path[] = "/tmp/cipher_policy_test_XXXXXX.json";
    int tfd = mkstemps(tmp_path, 5);
    if (tfd < 0) { std::perror("mkstemps"); return 5; }
    close(tfd);
    const char* fixture =
        (argc > 2) ? argv[2] : "package/deb/opt/cipher/etc/policy.json.example";
    FILE* in = std::fopen(fixture, "r");
    FILE* out = std::fopen(tmp_path, "w");
    if (!in || !out) {
        std::fprintf(stderr, "could not open fixture %s or tmp %s\n",
                     fixture, tmp_path);
        return 6;
    }
    char cbuf[4096]; size_t cn;
    while ((cn = std::fread(cbuf, 1, sizeof(cbuf), in)) > 0)
        std::fwrite(cbuf, 1, cn, out);
    std::fclose(in); std::fclose(out);
    setenv("CIPHER_POLICY_PATH", tmp_path, 1);

    // RTLD_LAZY: libcipher_rt.so contains unresolved cudaXxx symbols by
    // design (resolved at runtime via LD_PRELOAD into a CUDA process).
    // We don't call those code paths from this test.
    void* h = dlopen(so_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", so_path, dlerror()); return 2; }

    auto comply_init  = (fn_init)        dlsym(h, "cipher_comply_init");
    auto op_allowed   = (fn_op_allowed)  dlsym(h, "cipher_policy_op_allowed");
    auto max_power    = (fn_max_power)   dlsym(h, "cipher_policy_max_power_watts");
    auto max_tok_s    = (fn_max_tok_s)   dlsym(h, "cipher_policy_max_tok_per_s");
    auto min_lat_ms   = (fn_min_lat_ms)  dlsym(h, "cipher_policy_min_latency_ms_p99");
    auto tenant_id    = (fn_tenant)      dlsym(h, "cipher_policy_tenant_id");
    auto reload       = (fn_reload)      dlsym(h, "cipher_policy_reload");
    auto reload_count = (fn_reload_count)dlsym(h, "cipher_policy_reload_count");
    if (!op_allowed || !max_power || !max_tok_s || !tenant_id || !reload) {
        std::fprintf(stderr, "missing policy symbols\n"); return 3;
    }
    if (comply_init) comply_init();   // triggers parse + sighup install

    // Phase 1 — verify "shared" tenant from policy.json (max_power=400,
    //          max_tok=50, banned: SUBSTITUTE_KOOPMAN).
    std::fprintf(stderr, "[phase1] tenant_id=%s\n", tenant_id());
    EXPECT_EQ(std::strcmp(tenant_id(), "shared"), 0, "tenant_id == shared");
    EXPECT_EQ(max_power(),            400, "shared.max_power");
    EXPECT_EQ(max_tok_s(),             50, "shared.max_tok_s");
    EXPECT_EQ(min_lat_ms(),          5000, "shared.min_lat_ms");
    EXPECT_EQ(op_allowed(OP_SUBSTITUTE_KOOPMAN), 0, "KOOPMAN banned");
    EXPECT_EQ(op_allowed(OP_FAIRNESS),           1, "FAIRNESS allowed");
    unsigned long long r1 = reload_count ? reload_count() : 0;

    // Phase 2 — rewrite policy with new caps, programmatic reload.
    const char* path = std::getenv("CIPHER_POLICY_PATH");
    FILE* fp = std::fopen(path, "w");
    std::fprintf(fp,
        "{\"tenants\":["
        "{\"id\":\"shared\",\"max_power_watts\":250,\"min_latency_ms_p99\":1234,"
        "\"max_tok_per_s\":7,\"banned_ops\":[\"FAIRNESS\"]}"
        "]}\n");
    std::fclose(fp);
    int n = reload();
    EXPECT_EQ(n, 1, "reload returned tenant count");

    EXPECT_EQ(max_power(),            250, "post-reload max_power");
    EXPECT_EQ(max_tok_s(),              7, "post-reload max_tok_s");
    EXPECT_EQ(min_lat_ms(),          1234, "post-reload min_lat_ms");
    EXPECT_EQ(op_allowed(OP_FAIRNESS),           0, "post-reload FAIRNESS banned");
    EXPECT_EQ(op_allowed(OP_SUBSTITUTE_KOOPMAN), 1, "post-reload KOOPMAN allowed");
    if (reload_count) {
        unsigned long long r2 = reload_count();
        if (r2 <= r1) {
            std::fprintf(stderr, "FAIL reload_count not advanced: %llu -> %llu\n", r1, r2);
            return 4;
        }
        std::fprintf(stderr, "PASS reload_count advanced %llu -> %llu\n", r1, r2);
    }

    // Phase 3 — SIGHUP reload path.
    fp = std::fopen(path, "w");
    std::fprintf(fp,
        "{\"tenants\":["
        "{\"id\":\"shared\",\"max_power_watts\":99,\"min_latency_ms_p99\":42,"
        "\"max_tok_per_s\":11,\"banned_ops\":[]}"
        "]}\n");
    std::fclose(fp);
    raise(SIGHUP);
    usleep(50000);   // give the handler a moment
    EXPECT_EQ(max_power(), 99, "SIGHUP reloaded max_power");
    EXPECT_EQ(min_lat_ms(),42, "SIGHUP reloaded min_lat");
    std::fprintf(stderr, "ALL POLICY TESTS PASSED\n");
    unlink(tmp_path);   // best effort; harmless if it fails
    return 0;
}
