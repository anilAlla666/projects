# c3_vanilla_run1

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
  "batch_sizes": "1",
  "iterations": 5,
  "warmup_iters": 2,
  "output_tokens": 256,
  "input_length_distribution": "fixed:1024",
  "workload": "agentic",
  "no_cipher": true,
  "cipher": false,
  "run_id": "c3_vanilla_run1",
  "gpu_mem_util": 0.5,
  "enforce_eager": false,
  "telemetry_interval_ms": 10,
  "seed": 1337
}
```

## Per-batch results (mean over iters)

| B | n_iter | ttft_p50_ms | ttft_p99_ms | tpot_p50_ms | tpot_p99_ms | agg_tps | dec_MFU% | pre_MFU% | mean_W | peak_W | en_J | tok/W | tok/J | peakHBM_MiB | smClk_p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 10.7 | 10.7 | 6.03 | 6.03 | 165.27 | 0.250 | 140.30 | 422.7 | 424.1 | 651.4 | 0.391 | 0.393 | 41662 | 1830 |
