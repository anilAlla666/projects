# 20260523_mistral7b_vanilla_agentic_fixed1024

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
  "run_id": "20260523_mistral7b_vanilla_agentic_fixed1024",
  "gpu_mem_util": 0.85,
  "enforce_eager": false,
  "telemetry_interval_ms": 10,
  "seed": 1337
}
```

## Per-batch results (mean over iters)

| B | n_iter | ttft_p50_ms | ttft_p99_ms | tpot_p50_ms | tpot_p99_ms | agg_tps | dec_MFU% | pre_MFU% | mean_W | peak_W | en_J | tok/W | tok/J | peakHBM_MiB | smClk_p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 10.6 | 10.6 | 6.03 | 6.03 | 165.26 | 0.249 | 141.23 | 420.8 | 422.1 | 648.1 | 0.393 | 0.395 | 70046 | 1830 |
| 8 | 5 | 19.3 | 19.9 | 6.39 | 6.40 | 1237.93 | 1.877 | 560.81 | 447.3 | 450.5 | 738.0 | 2.768 | 2.775 | 70214 | 1830 |
| 16 | 5 | 26.4 | 27.9 | 6.77 | 6.77 | 2331.91 | 3.537 | 818.59 | 485.7 | 488.9 | 850.1 | 4.801 | 4.819 | 70287 | 1830 |
| 32 | 5 | 47.1 | 50.4 | 7.47 | 7.49 | 4186.14 | 6.389 | 928.19 | 522.6 | 529.4 | 1019.6 | 8.010 | 8.034 | 70336 | 1830 |
| 64 | 5 | 81.4 | 88.5 | 8.96 | 9.03 | 6895.67 | 10.551 | 1064.80 | 599.8 | 610.3 | 1420.8 | 11.498 | 11.532 | 70438 | 1830 |
