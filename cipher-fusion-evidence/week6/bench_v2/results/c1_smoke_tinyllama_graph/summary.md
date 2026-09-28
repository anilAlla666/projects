# c1_smoke_tinyllama_graph

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
  "run_id": "c1_smoke_tinyllama_graph",
  "gpu_mem_util": 0.4,
  "enforce_eager": false,
  "telemetry_interval_ms": 10,
  "seed": 1337
}
```

## Per-batch results (mean over iters)

| B | n_iter | ttft_p50_ms | ttft_p99_ms | tpot_p50_ms | tpot_p99_ms | agg_tps | dec_MFU% | pre_MFU% | mean_W | peak_W | en_J | tok/W | tok/J | peakHBM_MiB | smClk_p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 7.6 | 7.6 | 1.57 | 1.57 | 567.48 | 0.141 | 7.17 | 127.8 | 127.8 | 6.7 | 4.439 | 4.783 | 33536 | 1830 |
| 8 | 2 | 14.3 | 15.0 | 1.74 | 1.74 | 3675.57 | 0.941 | 27.84 | 158.1 | 163.5 | 9.4 | 23.262 | 27.161 | 33536 | 1830 |
