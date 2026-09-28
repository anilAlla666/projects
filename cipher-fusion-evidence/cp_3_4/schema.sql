-- CP 3.4 — ClickHouse schema for per-second per-tenant silicon state.
-- Source: /proc/cipher/flops (CP 3.3, kmod 0.4.7). Ingested at 1 Hz.

CREATE DATABASE IF NOT EXISTS cipher;

-- one row per second: device-wide silicon state
CREATE TABLE IF NOT EXISTS cipher.device_state (
    ts             DateTime64(3),
    tensor_tflops  Float64,
    mfu_pct        Float64,
    sm_clock_mhz   UInt32,
    sm_util_pct    UInt8,
    power_w        Float64
) ENGINE = MergeTree ORDER BY ts;

-- one row per second per active tenant: attributed FLOP/s + MFU
CREATE TABLE IF NOT EXISTS cipher.tenant_state (
    ts             DateTime64(3),
    tenant         String,        -- CIPHER_TENANT_ID; 'pid:<n>' if unnamed
    pid            UInt32,
    tgid           UInt32,
    launch_delta   UInt64,
    flops_per_s    Float64,
    mfu_pct        Float64
) ENGINE = MergeTree ORDER BY (ts, tenant);
