# 20260523_mistral7b_cipher_agentic_fixed1024_nocache

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
  "no_cipher": false,
  "cipher": true,
  "run_id": "20260523_mistral7b_cipher_agentic_fixed1024_nocache",
  "gpu_mem_util": 0.85,
  "enforce_eager": false,
  "telemetry_interval_ms": 10,
  "seed": 1337
}
```

## Per-batch results (mean over iters)

| B | n_iter | ttft_p50_ms | ttft_p99_ms | tpot_p50_ms | tpot_p99_ms | agg_tps | dec_MFU% | pre_MFU% | mean_W | peak_W | en_J | tok/W | tok/J | peakHBM_MiB | smClk_p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 25.4 | 25.4 | 6.04 | 6.04 | 163.61 | 0.249 | 59.09 | 438.5 | 442.1 | 683.7 | 0.373 | 0.374 | 70058 | 1830 |
| 8 | 5 | 120.2 | 178.6 | 6.60 | 6.90 | 1113.62 | 1.702 | 65.63 | 483.3 | 513.5 | 886.4 | 2.304 | 2.310 | 70226 | 1830 |
| 16 | 5 | 213.9 | 359.1 | 7.35 | 7.94 | 1923.25 | 2.938 | 65.49 | 534.7 | 592.6 | 1134.6 | 3.597 | 3.610 | 70226 | 1830 |
| 32 | 5 | 393.6 | 730.3 | 8.84 | 10.02 | 3012.07 | 4.584 | 64.46 | 585.5 | 688.4 | 1588.0 | 5.144 | 5.159 | 70226 | 1830 |
| 64 | 5 | 765.0 | 1522.0 | 11.89 | 14.24 | 4155.24 | 6.292 | 62.05 | 651.9 | 703.9 | 2568.6 | 6.374 | 6.378 | 70226 | 1827 |
