"""CIPHER MetricsCollector v1.0 — production-shaped telemetry for the
density-sweep harness, intended to become the substrate for
cipher-controller's telemetry layer.

Threading model:
  - One daemon sampler thread polls NVML at fixed Hz.
  - Multiple worker threads call record_tokens / record_burst.
  - All shared-state mutations are mutex-guarded.
  - Atomic JSON receipt write at .write_receipt().

Metric definitions (also written into the receipt's "doc" block):
  - MFU = achieved_TFLOPS / 989e12 × 100
      Industry-comparable H100 FP16 dense (sparse spec).
  - HFU = achieved_TFLOPS / 660e12 × 100
      Measured fp16 ceiling on this pod (700W power-cap).
  - tpw = total_tokens / total_energy_J
      total_energy_J = mean_power_W × elapsed_s
  - dollar_per_M_tok_power_only = (1e6 / tpw) × $0.10/kWh / 3.6e6 J/kWh
      POWER COST ONLY at $0.10/kWh. Excludes capital amortization,
      DC overhead (cooling/networking), ops costs, margin. Full
      neocloud $/M tok is typically 50-100x this. CIPHER affects
      only the power-cost portion.
  - fairness_ratio = max(per_tenant_tps) / max(min(per_tenant_tps), 1e-9)
      Stop threshold = 1.10.
"""
import os, time, json, threading
from typing import Optional


class MetricsCollector:
    SCHEMA_VERSION = "1.0"

    # Mistral-7B-v0.1 geometry (from config.json)
    MISTRAL7B = dict(
        H=4096, I=14336, L=32, n_kv=8, head_dim=128, vocab=32000)

    PEAK_FLOPS_SPEC = 989e12   # H100 FP16 dense sparse spec — industry MFU comparison
    PEAK_FLOPS_MEAS = 660e12   # H100 fp16 measured at this pod's 700W cap — HFU diagnosis
    KWH_PRICE = 0.10           # $/kWh assumption for $/M tok calculation
    JOULES_PER_KWH = 3.6e6

    def __init__(self, run_id: str, gpu_id: int = 0,
                 sample_hz: float = 5.0,
                 out_dir: str = "/tmp/cipher_metrics"):
        self.run_id = run_id
        self.gpu_id = gpu_id
        self.sample_period = 1.0 / sample_hz
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._samples = []
        self._tenants = {}
        self._sampler_thread = None
        self._stop_event = threading.Event()
        self._sampler_backend = None
        self._n_sample_errors = 0
        self._n_samples_dropped = 0
        self._first_error_msg = None
        self._t0 = None
        self._meta = {}

        self._nvml = None
        self._nvml_handle = None
        self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            self._sampler_backend = "nvml"
        except Exception as e:
            print(f"[metrics] NVML init failed: {e}; fallback to nvidia-smi",
                  flush=True)
            self._nvml = None
            self._sampler_backend = "nvidia-smi"

    def configure(self, **kwargs):
        with self._lock:
            self._meta.update(kwargs)

    def get_gpu_info(self) -> dict:
        info = {}
        try:
            if self._nvml:
                n = self._nvml; h = self._nvml_handle
                name = n.nvmlDeviceGetName(h)
                if isinstance(name, bytes): name = name.decode()
                uuid = n.nvmlDeviceGetUUID(h)
                if isinstance(uuid, bytes): uuid = uuid.decode()
                driver = n.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes): driver = driver.decode()
                mem = n.nvmlDeviceGetMemoryInfo(h)
                info.update(name=name, uuid=uuid, driver=driver,
                            hbm_gb=round(mem.total / (1024**3), 1))
            import torch
            props = torch.cuda.get_device_properties(self.gpu_id)
            info.update(cuda_runtime=torch.version.cuda,
                        sm_count=props.multi_processor_count,
                        compute_capability=f"{props.major}.{props.minor}")
        except Exception as e:
            info["gpu_info_error"] = str(e)
        info.setdefault("tdp_w", 700)  # measured on this pod
        return info

    def start_sampler(self):
        if self._sampler_thread is not None:
            return
        self._stop_event.clear()
        self._t0 = time.perf_counter()
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop, daemon=True,
            name=f"metrics-sampler-{self.run_id}")
        self._sampler_thread.start()

    def stop_sampler(self):
        if self._sampler_thread is None:
            return
        self._stop_event.set()
        self._sampler_thread.join(timeout=5.0)
        self._sampler_thread = None

    def _sampler_loop(self):
        period = self.sample_period
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            t = time.perf_counter() - self._t0
            try:
                sample = self._read_sample(t)
                if sample is not None:
                    with self._lock:
                        self._samples.append(sample)
            except Exception as e:
                self._n_sample_errors += 1
                if self._first_error_msg is None:
                    self._first_error_msg = f"{type(e).__name__}: {e}"
            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                # wait_for honors stop_event without sleeping full period
                if self._stop_event.wait(timeout=sleep_for):
                    break
            else:
                self._n_samples_dropped += 1
                next_t = time.perf_counter()

    def _read_sample(self, t: float) -> Optional[dict]:
        if self._nvml:
            n = self._nvml; h = self._nvml_handle
            pw = n.nvmlDeviceGetPowerUsage(h) / 1000.0
            mem = n.nvmlDeviceGetMemoryInfo(h).used // (1024 * 1024)
            util = n.nvmlDeviceGetUtilizationRates(h).gpu
            clk = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_GRAPHICS)
            try:
                temp = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            return dict(t_s=round(t, 4), power_w=round(pw, 2),
                        mem_mib=int(mem), util_pct=int(util),
                        clock_mhz=int(clk), temp_c=int(temp))
        else:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_id),
                 "--query-gpu=power.draw,memory.used,utilization.gpu,"
                 "clocks.gr,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1)
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            return dict(t_s=round(t, 4), power_w=float(parts[0]),
                        mem_mib=int(float(parts[1])),
                        util_pct=int(float(parts[2])),
                        clock_mhz=int(float(parts[3])),
                        temp_c=int(float(parts[4])))

    def register_tenant(self, tenant_id: int, **meta):
        with self._lock:
            if tenant_id in self._tenants:
                raise ValueError(f"tenant_id={tenant_id} already registered")
            self._tenants[tenant_id] = dict(
                tenant_id=tenant_id, n_tokens=0, elapsed_s=0.0,
                latencies_us=[], compute_s=0.0, idle_s=0.0,
                n_bursts=0, **meta)

    def record_tokens(self, tenant_id: int, n_tokens: int, dt_s: float,
                      latencies_us=None):
        with self._lock:
            t = self._tenants[tenant_id]
            t["n_tokens"] += n_tokens
            t["elapsed_s"] += dt_s
            if latencies_us:
                t["latencies_us"].extend(latencies_us)

    def record_burst(self, tenant_id: int, n_tokens: int,
                     compute_s: float, idle_s: float):
        with self._lock:
            t = self._tenants[tenant_id]
            t["n_tokens"] += n_tokens
            t["compute_s"] += compute_s
            t["idle_s"] += idle_s
            t["elapsed_s"] = t["compute_s"] + t["idle_s"]
            t["n_bursts"] += 1

    def _flops_per_token(self, mean_seq_len: int) -> float:
        m = self.MISTRAL7B
        H, I, L = m["H"], m["I"], m["L"]
        n_kv, d, V = m["n_kv"], m["head_dim"], m["vocab"]
        per_layer_linears = (
            2 * H * H +                  # Q proj
            2 * 2 * H * (n_kv * d) +     # K + V proj
            2 * H * H +                  # O proj
            3 * (2 * H * I)              # gate + up + down
        )
        per_layer_attn = 2 * 2 * H * mean_seq_len  # Q.K^T + softmax.V
        per_token = L * (per_layer_linears + per_layer_attn) + 2 * H * V
        return float(per_token)

    def snapshot(self) -> dict:
        with self._lock:
            samples = list(self._samples)
            tenants = {k: dict(v, latencies_us=list(v.get("latencies_us") or []))
                       for k, v in self._tenants.items()}
        return self._compute_aggregate(samples, tenants)

    def _compute_aggregate(self, samples, tenants) -> dict:
        n_samp = len(samples)
        if n_samp >= 10:
            steady = samples[int(n_samp * 0.20): max(int(n_samp * 0.95), int(n_samp * 0.20) + 1)]
        else:
            steady = samples
        if steady:
            powers = [s["power_w"] for s in steady]
            mems = [s["mem_mib"] for s in steady]
            utils = [s["util_pct"] for s in steady]
            clocks = [s["clock_mhz"] for s in steady]
            mean_power = sum(powers) / len(powers)
            peak_power = max(powers)
            peak_mem = max(mems)
            mean_util = sum(utils) / len(utils)
            mean_clock = sum(clocks) / len(clocks)
        else:
            mean_power = peak_power = peak_mem = mean_util = mean_clock = 0

        total_tokens = 0
        tps_list = []
        for t in tenants.values():
            total_tokens += t["n_tokens"]
            if t["elapsed_s"] > 0:
                t["tps"] = t["n_tokens"] / t["elapsed_s"]
                tps_list.append(t["tps"])
            else:
                t["tps"] = 0.0
            lat = sorted(t.pop("latencies_us", None) or [])
            ln = len(lat)
            t["lat_p50_us"] = lat[ln // 2] if ln else 0
            t["lat_p95_us"] = lat[min(ln - 1, int(ln * 0.95))] if ln else 0
            t["lat_p99_us"] = lat[min(ln - 1, int(ln * 0.99))] if ln else 0
            t["lat_mean_us"] = sum(lat) / ln if ln else 0
            if t["elapsed_s"] > 0:
                t["idle_fraction"] = t["idle_s"] / t["elapsed_s"]
            else:
                t["idle_fraction"] = 0.0

        agg_tps = sum(tps_list)
        if tps_list:
            tps_min = min(tps_list); tps_max = max(tps_list)
            fairness = tps_max / max(tps_min, 1e-9)
        else:
            tps_min = tps_max = fairness = 0.0

        # Elapsed = max tenant elapsed (sweep step boundary)
        elapsed = max((t["elapsed_s"] for t in tenants.values()), default=0.0)

        prefill_len = self._meta.get("prefill_len", 128)
        max_decode = self._meta.get("max_decode", 200)
        mean_seq_len = prefill_len + (max_decode // 2)
        flops_per_tok = self._flops_per_token(mean_seq_len)
        if elapsed > 0:
            achieved_flops = total_tokens * flops_per_tok / elapsed
        else:
            achieved_flops = 0.0
        mfu_pct = achieved_flops / self.PEAK_FLOPS_SPEC * 100
        hfu_pct = achieved_flops / self.PEAK_FLOPS_MEAS * 100

        energy_J = mean_power * elapsed
        tpw = total_tokens / energy_J if energy_J > 0 else 0.0
        if tpw > 0:
            dollar_per_M_tok_power_only = (1e6 / tpw) * self.KWH_PRICE / self.JOULES_PER_KWH
        else:
            dollar_per_M_tok_power_only = 0.0

        return dict(
            n_tenants=len(tenants),
            total_tokens=int(total_tokens),
            elapsed_s=round(elapsed, 3),
            agg_tps=round(agg_tps, 3),
            tps_min=round(tps_min, 3),
            tps_max=round(tps_max, 3),
            fairness_ratio=round(fairness, 4),
            achieved_TFLOPS=round(achieved_flops / 1e12, 4),
            flops_per_token=int(flops_per_tok),
            mean_seq_len=mean_seq_len,
            mfu_pct=round(mfu_pct, 4),         # vs 989 TFLOPS spec — industry-comparable
            hfu_pct=round(hfu_pct, 4),         # vs 660 TFLOPS measured @ 700W cap — internal
            mean_power_w=round(mean_power, 2),
            peak_power_w=round(peak_power, 2),
            mean_util_pct=round(mean_util, 2),
            mean_clock_mhz=round(mean_clock, 1),
            peak_mem_mib=int(peak_mem),
            energy_J=round(energy_J, 1),
            tpw=round(tpw, 4),
            dollar_per_M_tok_power_only=round(dollar_per_M_tok_power_only, 6),
            n_samples=n_samp,
            tenants=list(tenants.values()),
        )

    def write_receipt(self, path: Optional[str] = None,
                      stop_reason: str = "completed") -> dict:
        if path is None:
            path = os.path.join(self.out_dir, f"{self.run_id}.json")
        with self._lock:
            samples = list(self._samples)
            tenants = {k: dict(v, latencies_us=list(v.get("latencies_us") or []))
                       for k, v in self._tenants.items()}
        agg = self._compute_aggregate(samples, tenants)
        agg["stop_reason"] = stop_reason

        # Stride-downsample samples in the receipt if too many
        if len(samples) > 5000:
            stride = len(samples) // 5000
            samples = samples[::stride]

        receipt = dict(
            schema_version=self.SCHEMA_VERSION,
            run_id=self.run_id,
            doc=dict(
                mfu_pct=("achieved_TFLOPS / 989e12 × 100  "
                         "(H100 FP16 dense sparse spec). Industry-comparable."),
                hfu_pct=("achieved_TFLOPS / 660e12 × 100  "
                         "(H100 FP16 measured at this pod's 700W cap). "
                         "Internal diagnosis."),
                tpw="tokens / total_energy_J (mean_power_W × elapsed_s). 'Tokens per Watt'.",
                dollar_per_M_tok_power_only=(
                    "POWER COST ONLY at $0.10/kWh. Excludes capital "
                    "amortization, DC overhead (cooling/networking), ops "
                    "costs, and margin. Full neocloud $/M tok is typically "
                    "50-100x this. CIPHER affects only the power-cost portion."),
                fairness_ratio=("max(per_tenant_tps) / max(min(per_tenant_tps), "
                                "1e-9). Stop threshold = 1.10 in this sweep."),
                flops_per_token=("Per-token forward FLOPS for Mistral-7B at "
                                 "the configured mean_seq_len (prefill_len + "
                                 "max_decode/2)."),
            ),
            gpu=self.get_gpu_info(),
            config=self._meta,
            samples=samples,
            aggregate=agg,
            sampler=dict(
                backend=self._sampler_backend,
                samples_collected=len(samples) if len(self._samples) <= 5000
                                  else len(self._samples),
                samples_in_receipt=len(samples),
                samples_dropped=self._n_samples_dropped,
                n_sample_errors=self._n_sample_errors,
                first_error=self._first_error_msg,
            ),
        )
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(receipt, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
        return receipt
