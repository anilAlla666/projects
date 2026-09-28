# K.1 CLOSE GATE REPORT
Date: 2026-05-28T09:27:21Z
Substrate: cipher_rt_phase4/build_cuda13/libcipher_rt.so (K.1.6 build)
md5: 9fe23143b12e67355e7acc0b606d0c25

## Per-Cell Results

| Cell | Expect | Class | Obs | Conf | INT4 | PASS/FAIL |
|------|--------|-------|-----|------|------|-----------|
| P1_llama3_8b_bf16 | KNOWN | A4_BATCH_INFERENCE | 142847 | 720 | 0 | PASS |
| P2_mistral_7b_bf16 | KNOWN | A4_BATCH_INFERENCE | 143145 | 720 | 0 | PASS |
| P3_tinyllama_awq | KNOWN | A3_SINGLE_TENANT_STREAM | 91746 | 820 | 1 | PASS |
| E1_bare_torch | UNKNOWN | UNKNOWN | 4 | 250 | 0 | PASS |
| E2_load_idle | UNKNOWN | UNKNOWN | 974 | 250 | 0 | PASS |
| E3_load_1tok | UNKNOWN | UNKNOWN | 1945 | 250 | 0 | PASS |
| T1_llama3_transition | KNOWN | A3_SINGLE_TENANT_STREAM | 140262 | 820 | 0 | PASS |
| C1_llama3_calibration | KNOWN | A4_BATCH_INFERENCE | 119534 | 720 | 0 | PASS |
| N1_mistral_long_prefill | UNKNOWN | N/A | 0 | 0 | 0 | PASS |

## Summary
- PASS: 9 / 9
- FAIL: 0 / 9

**VERDICT: K.1 + W.1 CLOSE GATE PASS**

## W.1 VOLT engagement evidence (per-cell)

| Cell | VOLT ENGAGED lines | VOLT ARMED lines | VOLT classifier_skipped lines |
|------|---------------------|------------------|-------------------------------|
| P1_llama3_8b_bf16 | 1 | 1 | 0
0 |
| P2_mistral_7b_bf16 | 1 | 1 | 0
0 |
| P3_tinyllama_awq | 1 | 1 | 0
0 |
| T1_llama3_transition | 1 | 1 | 0
0 |
| C1_llama3_calibration | 1 | 1 | 0
0 |
| N1_mistral_long_prefill | 0
0 | 1 | 0
0 |
| E2_load_idle | 0
0 | 1 | 0
0 |
| E3_load_1tok | 0
0 | 1 | 0
0 |
