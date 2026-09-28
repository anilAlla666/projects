/* Sub-4 measurement infra probe — exercise transition observer + report. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dlfcn.h>

int main(void) {
    void *h = dlopen("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so",
                     RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }

    void (*observe)(uint32_t, uint32_t, uint8_t, uint64_t, uint64_t) =
        dlsym(h, "cipher_rt_sense_transition_observe");
    unsigned long (*proposals_q)(void) =
        dlsym(h, "cipher_rt_sense_transition_proposals_queued");
    unsigned long (*transitions)(void) =
        dlsym(h, "cipher_rt_sense_transition_transitions_total");
    unsigned long (*idles)(void) =
        dlsym(h, "cipher_rt_sense_transition_tool_idle_count");
    int (*report)(const char *) =
        dlsym(h, "cipher_rt_sense_transition_report");

    if (!observe || !proposals_q || !transitions || !idles || !report) {
        fprintf(stderr, "dlsym missing one of the symbols\n");
        return 1;
    }

    /* Simulate: tenant 1234 — 8 BATCH obs (class 1), then 8 AGENT obs (class 2),
     * then long gap → tool-idle, then 8 HUMAN obs (class 0). 3 transitions
     * expected: empty(0)→BATCH, BATCH→AGENT, AGENT→HUMAN. */
    uint64_t ts = 1000000000ULL;
    for (int i = 0; i < 8; i++) observe(1234, 1, 80, 0xdeadbeefULL, ts + i * 1000);
    for (int i = 0; i < 8; i++) observe(1234, 2, 80, 0xdeadbeefULL, ts + 8000 + i * 1000);
    /* Gap > 100ms triggers tool-idle on AGENT */
    observe(1234, 2, 80, 0xdeadbeefULL, ts + 8000 + 8 * 1000 + 200000000ULL);
    for (int i = 0; i < 8; i++) observe(1234, 0, 80, 0xdeadbeefULL,
                                         ts + 8000 + 8 * 1000 + 300000000ULL + i * 1000);

    /* Second tenant for matrix coverage. */
    for (int i = 0; i < 8; i++) observe(5678, 0, 60, 0xfeedfaceULL, ts + i * 500);
    for (int i = 0; i < 8; i++) observe(5678, 1, 60, 0xfeedfaceULL, ts + 4000 + i * 500);

    printf("proposals_queued: %lu\n", proposals_q());
    printf("transitions_total: %lu  (expect ~4: 0->BATCH, BATCH->AGENT, AGENT->HUMAN, 0->BATCH(5678) + BATCH->...)\n",
           transitions());
    printf("tool_idle_count:   %lu  (expect 1)\n", idles());

    int rc = report("/tmp/cipher_sense_transitions.json");
    printf("report rc=%d  (file: /tmp/cipher_sense_transitions.json)\n", rc);
    return 0;
}
