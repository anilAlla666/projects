# 20260523_mistral7b_vanilla_agentic_fixed1024_nocache

## Hardware state
```json
{
  "gpu_name": "NVIDIA H100 80GB HBM3",
  "driver_version": "580.105.08",
  "max_gr_clock_mhz": 1980,
  "max_mem_clock_mhz": 2619,
  "power_limit_w": 700.0,
  "persistence_mode": "Enabled",
  "ecc_mode": "Enabled",
  "lock_attempted": true,
  "target_gr_clock_mhz": 1980,
  "target_mem_clock_mhz": 2619
}
```

## Args
```json
{
  "model": "mistralai/Mistral-7B-v0.1",
  "batch_sizes": "1,8,16,32,64",
  "iterations": 5,
  "warmup_iters": 2,
  "output_tokens": 256,
  "input_length_distribution": "fixed:1024",
  "workload": "agentic",
  "no_cipher": true,
  "cipher": false,
  "run_id": "20260523_mistral7b_vanilla_agentic_fixed1024_nocache",
  "gpu_mem_util": 0.85,
  "enforce_eager": false,
  "telemetry_interval_ms": 10,
  "seed": 1337
}
```

## Per-batch results (mean over iters)

| B | n_iter | ttft_p50_ms | ttft_p99_ms | tpot_p50_ms | tpot_p99_ms | agg_tps | dec_MFU% | pre_MFU% | mean_W | peak_W | en_J | tok/W | tok/J | peakHBM_MiB | smClk_p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 25.2 | 25.2 | 6.05 | 6.05 | 163.29 | 0.249 | 59.57 | 433.7 | 437.6 | 676.5 | 0.377 | 0.378 | 70046 | 1830 |
| 8 | 5 | 120.6 | 180.2 | 6.58 | 6.89 | 1114.04 | 1.702 | 65.05 | 483.0 | 514.3 | 885.9 | 2.306 | 2.312 | 70214 | 1830 |
| 16 | 5 | 209.0 | 358.6 | 7.33 | 7.92 | 1924.69 | 2.934 | 65.73 | 534.4 | 592.5 | 1134.1 | 3.601 | 3.612 | 70214 | 1830 |
| 32 | 5 | 379.0 | 731.7 | 8.78 | 9.96 | 3019.14 | 4.575 | 64.89 | 585.8 | 687.8 | 1586.3 | 5.154 | 5.164 | 70214 | 1830 |
| 64 | 5 | 762.7 | 1521.6 | 11.89 | 14.24 | 4156.13 | 6.292 | 62.06 | 651.4 | 706.4 | 2565.1 | 6.381 | 6.387 | 70214 | 1827 |
