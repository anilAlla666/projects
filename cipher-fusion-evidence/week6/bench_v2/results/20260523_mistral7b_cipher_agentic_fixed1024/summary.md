# 20260523_mistral7b_cipher_agentic_fixed1024

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
  "run_id": "20260523_mistral7b_cipher_agentic_fixed1024",
  "gpu_mem_util": 0.85,
  "enforce_eager": false,
  "telemetry_interval_ms": 10,
  "seed": 1337
}
```

## Per-batch results (mean over iters)

| B | n_iter | ttft_p50_ms | ttft_p99_ms | tpot_p50_ms | tpot_p99_ms | agg_tps | dec_MFU% | pre_MFU% | mean_W | peak_W | en_J | tok/W | tok/J | peakHBM_MiB | smClk_p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 10.7 | 10.7 | 6.01 | 6.01 | 165.76 | 0.250 | 140.76 | 424.1 | 426.0 | 652.9 | 0.391 | 0.392 | 70058 | 1830 |
| 8 | 5 | 18.6 | 19.3 | 6.39 | 6.40 | 1238.25 | 1.878 | 583.43 | 450.0 | 452.3 | 741.7 | 2.751 | 2.761 | 70226 | 1830 |
| 16 | 5 | 27.2 | 28.7 | 6.77 | 6.77 | 2331.98 | 3.544 | 801.92 | 487.0 | 490.9 | 852.3 | 4.789 | 4.806 | 70354 | 1830 |
| 32 | 5 | 48.1 | 51.0 | 7.46 | 7.46 | 4190.44 | 6.440 | 899.44 | 524.7 | 530.4 | 1021.9 | 7.986 | 8.016 | 70354 | 1830 |
| 64 | 5 | 81.1 | 88.6 | 8.95 | 9.03 | 6899.82 | 10.623 | 1032.38 | 598.8 | 608.7 | 1417.6 | 11.523 | 11.558 | 70354 | 1830 |
