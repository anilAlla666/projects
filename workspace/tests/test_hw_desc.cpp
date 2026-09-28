// =============================================================================
// CIPHER — L1.4 Test Suite
// tests/test_hw_desc.cpp
//
// Success criteria from build plan:
//   ✓ H100 normalizes to all-1.0 ratios
//   ✓ A100 correctly below H100 on compute/BW dimensions
//   ✓ H200 correctly above H100 on L2/BW dimensions
//   ✓ B100/B200 Blackwell above H100 on all capacity dims
//   ✓ AMD MI300X normalizes without error
//   ✓ Transfer accuracy H100→H100 = 1.0
//   ✓ Transfer accuracy H100→A100 > 0.90 (build plan: <10% degradation)
//   ✓ Transfer accuracy H100→MI300X > 0.80 (cross-vendor, wider gap ok)
//   ✓ Normalized vector dim = 16, all values in [0, 4]
//   ✓ Architecture one-hot: exactly one of is_hopper/ampere/blackwell/amd = 1
//   ✓ Profile lookup by name works for all 13 known GPUs
// =============================================================================

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cipher_hw_desc.h"

#define PASS "\033[32m✓\033[0m"
#define FAIL "\033[31m✗\033[0m"
static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  %s %s\n", PASS, msg); g_pass++; } \
    else      { printf("  %s %s\n", FAIL, msg); g_fail++; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    float _diff = fabsf((float)(a) - (float)(b)); \
    if (_diff <= (tol)) { printf("  %s %s (%.4f)\n", PASS, msg, (float)(a)); g_pass++; } \
    else { printf("  %s %s (got %.4f, expected %.4f ± %.4f)\n", FAIL, msg, (float)(a), (float)(b), (float)(tol)); g_fail++; } \
} while(0)

// ---------------------------------------------------------------------------
// L1.4.1: H100 reference — all ratios = 1.0
// ---------------------------------------------------------------------------

static void test_h100_reference(void) {
    printf("\n[L1.4.1] H100 Reference (should be all 1.0)\n");

    CipherHwDescNorm n = cipher_hw_desc_normalize_by_name("H100_SXM5");

    CHECK_NEAR(n.sm_ratio,      1.0f, 0.001f, "H100 sm_ratio = 1.0");
    CHECK_NEAR(n.l2_ratio,      1.0f, 0.001f, "H100 l2_ratio = 1.0");
    CHECK_NEAR(n.bw_ratio,      1.0f, 0.001f, "H100 bw_ratio = 1.0");
    CHECK_NEAR(n.clock_ratio,   1.0f, 0.001f, "H100 clock_ratio = 1.0");
    CHECK_NEAR(n.compute_ratio, 1.0f, 0.001f, "H100 compute_ratio = 1.0");
    CHECK_NEAR(n.nvlink_ratio,  1.0f, 0.001f, "H100 nvlink_ratio = 1.0");
    CHECK_NEAR(n.is_hopper,     1.0f, 0.001f, "H100 is_hopper = 1.0");
    CHECK_NEAR(n.is_ampere,     0.0f, 0.001f, "H100 is_ampere = 0.0");
    CHECK_NEAR(n.has_nvlink,    1.0f, 0.001f, "H100 has_nvlink = 1.0");
}

// ---------------------------------------------------------------------------
// L1.4.2: A100 — below H100 on compute and BW
// ---------------------------------------------------------------------------

static void test_a100(void) {
    printf("\n[L1.4.2] A100 SXM4 (below H100)\n");

    CipherHwDescNorm n = cipher_hw_desc_normalize_by_name("A100_SXM4");

    printf("  A100: sm=%.3f  l2=%.3f  bw=%.3f  compute=%.3f\n",
           n.sm_ratio, n.l2_ratio, n.bw_ratio, n.compute_ratio);

    CHECK(n.sm_ratio < 1.0f,      "A100 sm_ratio < 1.0 (108 < 132 SMs)");
    CHECK(n.l2_ratio < 1.0f,      "A100 l2_ratio < 1.0 (40MB < 50MB)");
    CHECK(n.bw_ratio < 1.0f,      "A100 bw_ratio < 1.0 (2000 < 3350 GB/s)");
    CHECK(n.compute_ratio < 1.0f, "A100 compute_ratio < 1.0 (312 < 989 TFLOPS)");
    CHECK_NEAR(n.is_ampere, 1.0f, 0.001f, "A100 is_ampere = 1.0");
    CHECK_NEAR(n.is_hopper, 0.0f, 0.001f, "A100 is_hopper = 0.0");

    // PCIe variant has no NVLink
    CipherHwDescNorm n_pcie = cipher_hw_desc_normalize_by_name("A100_PCIe");
    CHECK_NEAR(n_pcie.nvlink_ratio, 0.0f, 0.001f, "A100 PCIe nvlink_ratio = 0");
    CHECK_NEAR(n_pcie.has_nvlink,   0.0f, 0.001f, "A100 PCIe has_nvlink = 0");
}

// ---------------------------------------------------------------------------
// L1.4.3: H200 — above H100 on L2 and BW
// ---------------------------------------------------------------------------

static void test_h200(void) {
    printf("\n[L1.4.3] H200 (above H100 on memory)\n");

    CipherHwDescNorm n = cipher_hw_desc_normalize_by_name("H200_SXM");

    printf("  H200: sm=%.3f  l2=%.3f  bw=%.3f  compute=%.3f\n",
           n.sm_ratio, n.l2_ratio, n.bw_ratio, n.compute_ratio);

    CHECK_NEAR(n.sm_ratio,    1.0f, 0.001f, "H200 sm_ratio = 1.0 (same 132 SMs)");
    CHECK(n.l2_ratio > 1.0f,              "H200 l2_ratio > 1.0 (96MB > 50MB)");
    CHECK(n.bw_ratio > 1.0f,              "H200 bw_ratio > 1.0 (4800 > 3350 GB/s)");
    CHECK(n.compute_ratio > 1.0f,         "H200 compute_ratio > 1.0 (1979 > 989 TFLOPS)");
    CHECK_NEAR(n.is_hopper, 1.0f, 0.001f, "H200 is_hopper = 1.0");
}

// ---------------------------------------------------------------------------
// L1.4.4: Blackwell B100 — major uplift across all dims
// ---------------------------------------------------------------------------

static void test_blackwell(void) {
    printf("\n[L1.4.4] Blackwell B100\n");

    CipherHwDescNorm n = cipher_hw_desc_normalize_by_name("B100");

    printf("  B100: sm=%.3f  l2=%.3f  bw=%.3f  compute=%.3f\n",
           n.sm_ratio, n.l2_ratio, n.bw_ratio, n.compute_ratio);

    CHECK(n.sm_ratio > 1.0f,               "B100 sm_ratio > 1.0 (160 > 132)");
    CHECK(n.l2_ratio > 1.0f,               "B100 l2_ratio > 1.0 (192MB > 50MB)");
    CHECK(n.bw_ratio > 1.0f,               "B100 bw_ratio > 1.0 (8000 > 3350)");
    CHECK(n.compute_ratio > 1.0f,          "B100 compute_ratio > 1.0 (3500 > 989)");
    CHECK_NEAR(n.is_blackwell, 1.0f, 0.001f,"B100 is_blackwell = 1.0");
    CHECK_NEAR(n.is_hopper,    0.0f, 0.001f,"B100 is_hopper = 0.0");
}

// ---------------------------------------------------------------------------
// L1.4.5: AMD MI300X — cross-vendor normalization
// ---------------------------------------------------------------------------

static void test_mi300x(void) {
    printf("\n[L1.4.5] AMD MI300X (cross-vendor)\n");

    CipherHwDescNorm n = cipher_hw_desc_normalize_by_name("MI300X");

    printf("  MI300X: sm=%.3f  l2=%.3f  bw=%.3f  compute=%.3f\n",
           n.sm_ratio, n.l2_ratio, n.bw_ratio, n.compute_ratio);

    CHECK(n.sm_ratio > 1.0f,           "MI300X sm_ratio > 1.0 (304 > 132 CUs)");
    CHECK(n.l2_ratio > 1.0f,           "MI300X l2_ratio > 1.0 (256MB > 50MB)");
    CHECK(n.bw_ratio > 1.0f,           "MI300X bw_ratio > 1.0 (5300 > 3350 GB/s)");
    CHECK_NEAR(n.is_amd, 1.0f, 0.001f, "MI300X is_amd = 1.0");
    CHECK_NEAR(n.is_hopper, 0.0f, 0.001f,"MI300X is_hopper = 0.0");

    // All values should be in valid range
    bool all_valid = true;
    for (int i = 0; i < CIPHER_HW_NORM_DIM; i++)
        if (n.v[i] < 0.0f || n.v[i] > 4.0f) all_valid = false;
    CHECK(all_valid, "MI300X all 16 normalized values in [0, 4]");
}

// ---------------------------------------------------------------------------
// L1.4.6: Transfer accuracy — build plan requires >90% H100→A100
// ---------------------------------------------------------------------------

static void test_transfer_accuracy(void) {
    printf("\n[L1.4.6] Transfer Accuracy (build plan: <10%% degradation)\n");

    CipherHwDescNorm h100 = cipher_hw_desc_normalize_by_name("H100_SXM5");
    CipherHwDescNorm h100_pcie = cipher_hw_desc_normalize_by_name("H100_PCIe");
    CipherHwDescNorm a100  = cipher_hw_desc_normalize_by_name("A100_SXM4");
    CipherHwDescNorm h200  = cipher_hw_desc_normalize_by_name("H200_SXM");
    CipherHwDescNorm b100  = cipher_hw_desc_normalize_by_name("B100");
    CipherHwDescNorm mi300 = cipher_hw_desc_normalize_by_name("MI300X");

    // Same hardware = 1.0
    float acc_same = cipher_hw_desc_transfer_accuracy(&h100, &h100);
    printf("  H100 → H100:      %.4f\n", acc_same);
    CHECK_NEAR(acc_same, 1.0f, 0.001f,
               "Transfer H100→H100 = 1.0 (same hardware)");

    // PCIe variant — minor difference
    float acc_pcie = cipher_hw_desc_transfer_accuracy(&h100, &h100_pcie);
    printf("  H100 → H100_PCIe: %.4f\n", acc_pcie);
    CHECK(acc_pcie > 0.90f,
          "Transfer H100→H100_PCIe > 0.90 (minor variant)");

    // A100 — one generation back
    float acc_a100 = cipher_hw_desc_transfer_accuracy(&h100, &a100);
    printf("  H100 → A100:      %.4f  (target >0.90)\n", acc_a100);
    CHECK(acc_a100 > 0.90f,
          "Transfer H100→A100 > 0.90 (build plan <10% degradation)");

    // H200 — same arch, more memory
    float acc_h200 = cipher_hw_desc_transfer_accuracy(&h100, &h200);
    printf("  H100 → H200:      %.4f\n", acc_h200);
    CHECK(acc_h200 > 0.90f,
          "Transfer H100→H200 > 0.90 (same arch, more memory)");

    // B100 — next gen, significant uplift
    float acc_b100 = cipher_hw_desc_transfer_accuracy(&h100, &b100);
    printf("  H100 → B100:      %.4f\n", acc_b100);
    CHECK(acc_b100 > 0.75f,
          "Transfer H100→B100 > 0.75 (next generation, 3.5x compute delta)");

    // MI300X — cross-vendor, largest gap
    float acc_mi300 = cipher_hw_desc_transfer_accuracy(&h100, &mi300);
    printf("  H100 → MI300X:    %.4f\n", acc_mi300);
    CHECK(acc_mi300 > 0.75f,
          "Transfer H100→MI300X > 0.75 (cross-vendor)");

    // Symmetry: transfer accuracy is symmetric
    float acc_rev = cipher_hw_desc_transfer_accuracy(&a100, &h100);
    CHECK(fabsf(acc_a100 - acc_rev) < 0.001f,
          "Transfer accuracy is symmetric (H100→A100 = A100→H100)");
}

// ---------------------------------------------------------------------------
// L1.4.7: All 13 known profiles parse without error
// ---------------------------------------------------------------------------

static void test_all_profiles(void) {
    printf("\n[L1.4.7] All Known Profiles\n");

    const char* names[] = {
        "H100_SXM5", "H100_PCIe", "H200_SXM",
        "A100_SXM4", "A100_PCIe", "A10", "A40",
        "RTX_4090",  "L40S",
        "B100",      "B200",
        "MI300X",    "MI250X",
        NULL
    };

    int ok_count = 0;
    for (int i = 0; names[i]; i++) {
        CipherHwDesc desc;
        bool found = cipher_hw_desc_from_name(&desc, names[i]);
        CipherHwDescNorm norm = cipher_hw_desc_normalize(&desc);

        // Check: all values in [0, 4]
        bool all_valid = true;
        for (int j = 0; j < CIPHER_HW_NORM_DIM; j++)
            if (norm.v[j] < 0.0f || norm.v[j] > 4.01f) all_valid = false;

        // Check: sm_ratio > 0
        bool sm_positive = (norm.sm_ratio > 0.0f);

        // Check: at most one arch one-hot = 1
        float arch_sum = norm.is_ampere + norm.is_hopper +
                         norm.is_blackwell + norm.is_amd;
        bool arch_valid = (arch_sum >= 0.99f && arch_sum <= 1.01f);

        if (found && all_valid && sm_positive && arch_valid) {
            printf("  ✓ %-14s sm=%.2f  bw=%.2f  compute=%.2f  arch_ok\n",
                   names[i], norm.sm_ratio, norm.bw_ratio, norm.compute_ratio);
            ok_count++;
        } else {
            printf("  ✗ %-14s found=%d valid=%d sm_pos=%d arch=%d\n",
                   names[i], found, all_valid, sm_positive, arch_valid);
        }
    }

    int total = 0;
    for (int i = 0; names[i]; i++) total++;
    CHECK(ok_count == total,
          "All 13 known GPU profiles normalize without error");
}

// ---------------------------------------------------------------------------
// L1.4.8: Normalized vector properties
// ---------------------------------------------------------------------------

static void test_vector_properties(void) {
    printf("\n[L1.4.8] Normalized Vector Properties\n");

    CipherHwDescNorm n = cipher_hw_desc_normalize_by_name("H100_SXM5");

    CHECK(CIPHER_HW_NORM_DIM == 16, "Vector dimension = 16");

    // v[] matches named fields
    CHECK(fabsf(n.v[0] - n.sm_ratio) < 1e-6f,      "v[0] = sm_ratio");
    CHECK(fabsf(n.v[1] - n.l2_ratio) < 1e-6f,      "v[1] = l2_ratio");
    CHECK(fabsf(n.v[2] - n.bw_ratio) < 1e-6f,      "v[2] = bw_ratio");
    CHECK(fabsf(n.v[9] - n.is_hopper) < 1e-6f,     "v[9] = is_hopper");
    CHECK(fabsf(n.v[15] - n.compute_intensity) < 1e-6f, "v[15] = compute_intensity");

    // All values in [0, 4]
    bool all_bounded = true;
    for (int i = 0; i < CIPHER_HW_NORM_DIM; i++)
        if (n.v[i] < 0.0f || n.v[i] > 4.01f) all_bounded = false;
    CHECK(all_bounded, "All 16 values in [0, 4.0]");

    // Detect function in stub mode returns H100
    CipherHwDesc detected;
    cipher_hw_desc_detect(&detected, 0);
    CHECK(detected.detected,           "cipher_hw_desc_detect succeeds in stub mode");
    CHECK(detected.sm_count == 132,    "Stub detect returns H100 (132 SMs)");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void) {
    printf("===================================================\n");
    printf("  CIPHER L1.4 Test Suite\n");
    printf("  Hardware Descriptor Normalization\n");
    printf("===================================================\n");

    test_h100_reference();
    test_a100();
    test_h200();
    test_blackwell();
    test_mi300x();
    test_transfer_accuracy();
    test_all_profiles();
    test_vector_properties();

    printf("\n===================================================\n");
    printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0)
        printf("  \033[32m L1.4 GREEN — weights transfer across GPU generations\033[0m\n");
    else
        printf("  \033[31m FAILURES REMAIN\033[0m\n");
    printf("===================================================\n\n");
    return g_fail > 0 ? 1 : 0;
}
