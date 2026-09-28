# c1_smoke_tinyllama

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
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "batch_sizes": "1,8",
  "iterations": 2,
  "warmup_iters": 1,
  "output_tokens": 32,
  "input_length_distribution": "fixed:256",
  "workload": "agentic",
  "no_cipher": true,
  "cipher": false,
  "run_id": "c1_smoke_tinyllama",
  "gpu_mem_util": 0.4,
  "enforce_eager": true,
  "telemetry_interval_ms": 10,
  "seed": 1337
}
```

## Per-batch results (mean over iters)

| B | n_iter | ttft_p50_ms | ttft_p99_ms | tpot_p50_ms | tpot_p99_ms | agg_tps | dec_MFU% | pre_MFU% | mean_W | peak_W | en_J | tok/W | tok/J | peakHBM_MiB | smClk_p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 22.4 | 22.4 | 8.39 | 8.39 | 113.23 | 0.026 | 2.47 | 133.5 | 136.9 | 36.9 | 0.848 | 0.866 | 33516 | 1830 |
| 8 | 2 | 32.0 | 32.9 | 8.56 | 8.56 | 855.17 | 0.199 | 12.91 | 144.3 | 144.6 | 42.6 | 5.927 | 6.016 | 33516 | 1830 |
