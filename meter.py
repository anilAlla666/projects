"""
cipher_runtime.meter — DEP.2.1 MFU Meter
Neural Dynamics, Inc.

Measures GPU MFU (Model FLOP Utilization) continuously via NVML.
Computes the delta between baseline and CIPHER-active periods.
This is the measurement that backs the 20% revenue share model.

MFU = observed_TFLOPS / peak_TFLOPS

Where:
  observed_TFLOPS = SM_utilization × peak_TFLOPS × (SM_clock / boost_clock)
  peak_TFLOPS     = device-specific (H100 SXM5 = 989 BF16 tensor TFLOPS)

USAGE:
    from cipher_runtime.meter import CipherMeter

    meter = CipherMeter()
    meter.start_baseline()          # begin measuring before CIPHER
    # ... warmup training steps ...
    meter.start_cipher()            # CIPHER activates here
    # ... training with CIPHER ...
    report = meter.stop()           # returns MFUReport
    report.print()
    report.save("cipher_meter.json")
"""

import os
import time
import json
import threading
import ctypes
import ctypes.util
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List

# ── Peak TFLOPS table (BF16 tensor core) ─────────────────────
# Source: NVIDIA product pages
PEAK_TFLOPS = {
    "H100 SXM5":   989.0,
    "H100 PCIe":   756.0,
    "H200 SXM":   1979.0,
    "A100 SXM4":   312.0,
    "A100 PCIe":   312.0,
    "A10":         125.0,
    "A40":         149.7,
    "RTX 4090":    330.0,
    "L40S":        366.0,
    "B100":       3500.0,
    "B200":       4500.0,
}

# Spot price per GPU-hour (USD) for billing calculation
# Updated periodically — can be overridden at runtime
SPOT_PRICE_PER_GPU_HOUR = {
    "H100 SXM5":   3.00,
    "H100 PCIe":   2.50,
    "H200 SXM":    4.00,
    "A100 SXM4":   1.50,
    "A100 PCIe":   1.20,
    "default":     2.00,
}


# ── NVML thin binding ─────────────────────────────────────────

class NVMLError(Exception):
    pass


class NVML:
    """Thin ctypes binding to libnvidia-ml.so.1"""

    def __init__(self):
        self._lib = None
        self._handle = None
        self._available = False
        self._device_name = "unknown"
        self._peak_tflops = 989.0  # H100 default
        self._boost_clock = 1980   # MHz

    def init(self, device_index: int = 0) -> bool:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return False
            self._lib = lib

            # Get device handle
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(device_index,
                                                   ctypes.byref(handle)) != 0:
                return False
            self._handle = handle

            # Device name
            name_buf = ctypes.create_string_buffer(96)
            lib.nvmlDeviceGetName(handle, name_buf, 96)
            self._device_name = name_buf.value.decode("utf-8", errors="replace")

            # Max SM clock (boost clock)
            clock = ctypes.c_uint()
            lib.nvmlDeviceGetMaxClockInfo(handle, 1, ctypes.byref(clock))
            self._boost_clock = max(1, clock.value)

            # Peak TFLOPS lookup
            for key, val in PEAK_TFLOPS.items():
                if key.lower() in self._device_name.lower():
                    self._peak_tflops = val
                    break

            self._available = True
            return True
        except Exception:
            return False

    def sample(self) -> Optional[dict]:
        """
        Returns instantaneous GPU metrics.
        Called at 500Hz from the sampling thread.
        """
        if not self._available:
            return None
        try:
            lib = self._lib
            h   = self._handle

            # SM utilization (0-100%)
            util = (ctypes.c_uint * 2)()
            lib.nvmlDeviceGetUtilizationRates(h, util)
            sm_util = util[0]

            # Current SM clock
            clock = ctypes.c_uint()
            lib.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(clock))
            sm_clock = max(1, clock.value)

            # HBM bandwidth utilization (via memory util as proxy)
            mem_util = util[1]

            # Power
            power_mw = ctypes.c_uint()
            lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(power_mw))
            power_w = power_mw.value / 1000.0

            # Temperature
            temp = ctypes.c_uint()
            lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(temp))

            # MFU calculation
            # observed_TFLOPS = peak × SM_util% × (sm_clock / boost_clock)
            clock_ratio  = sm_clock / self._boost_clock
            observed_tf  = self._peak_tflops * (sm_util / 100.0) * clock_ratio
            mfu          = observed_tf / self._peak_tflops  # 0.0 to 1.0

            return {
                "ts":        time.time(),
                "sm_util":   sm_util,
                "sm_clock":  sm_clock,
                "mem_util":  mem_util,
                "power_w":   power_w,
                "temp_c":    temp.value,
                "mfu":       mfu,
                "obs_tflops": observed_tf,
            }
        except Exception:
            return None

    def shutdown(self):
        if self._lib:
            try:
                self._lib.nvmlShutdown()
            except Exception:
                pass

    @property
    def available(self): return self._available

    @property
    def device_name(self): return self._device_name

    @property
    def peak_tflops(self): return self._peak_tflops

    @property
    def boost_clock(self): return self._boost_clock


# ── Sample buffer ─────────────────────────────────────────────

@dataclass
class PhaseSamples:
    """Samples collected during one phase (baseline or cipher)."""
    name:        str
    start_time:  float = 0.0
    end_time:    float = 0.0
    samples:     List[dict] = field(default_factory=list)

    def duration_s(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def avg_mfu(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s["mfu"] for s in self.samples) / len(self.samples)

    def avg_tflops(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s["obs_tflops"] for s in self.samples) / len(self.samples)

    def avg_sm_util(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s["sm_util"] for s in self.samples) / len(self.samples)

    def avg_power(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s["power_w"] for s in self.samples) / len(self.samples)

    def p95_mfu(self) -> float:
        if not self.samples:
            return 0.0
        sorted_mfu = sorted(s["mfu"] for s in self.samples)
        idx = int(len(sorted_mfu) * 0.95)
        return sorted_mfu[min(idx, len(sorted_mfu)-1)]


# ── MFU Report ────────────────────────────────────────────────

@dataclass
class MFUReport:
    """
    Complete metering report for one training run.
    This is what gets sent to Nebius for billing.
    """
    # Run metadata
    run_id:          str   = ""
    device_name:     str   = ""
    peak_tflops:     float = 0.0
    timestamp:       str   = ""

    # Phase measurements
    baseline_mfu:      float = 0.0
    cipher_mfu:        float = 0.0
    baseline_tflops:   float = 0.0
    cipher_tflops:     float = 0.0
    baseline_sm_util:  float = 0.0
    cipher_sm_util:    float = 0.0
    baseline_duration: float = 0.0
    cipher_duration:   float = 0.0
    baseline_power:    float = 0.0
    cipher_power:      float = 0.0

    # Delta and billing
    mfu_delta:         float = 0.0   # cipher_mfu - baseline_mfu
    tflops_recovered:  float = 0.0   # additional TFLOPS from CIPHER
    gpu_hours_saved:   float = 0.0   # equivalent GPU-hours recovered
    spot_price:        float = 0.0   # $/GPU-hour
    revenue_share_20pct: float = 0.0  # 20% of recovered value (USD)

    # Substitution stats (from CIPHER liquid state if available)
    total_kernels:     int   = 0
    substitutions:     int   = 0
    substitution_rate: float = 0.0

    # Sample counts
    baseline_samples:  int   = 0
    cipher_samples:    int   = 0

    def print(self):
        print("\n" + "=" * 60)
        print("  CIPHER DEP.2 — MFU Metering Report")
        print("=" * 60)
        print(f"\n  Device:     {self.device_name}")
        print(f"  Peak BF16:  {self.peak_tflops:.0f} TFLOPS")
        print(f"  Run ID:     {self.run_id}")
        print(f"  Timestamp:  {self.timestamp}")

        print(f"\n  {'Metric':<28} {'Baseline':>12} {'CIPHER':>12} {'Delta':>10}")
        print(f"  {'─'*62}")
        print(f"  {'MFU':<28} {self.baseline_mfu:>11.1%} "
              f"{self.cipher_mfu:>11.1%} "
              f"{self.mfu_delta:>+9.1%}")
        print(f"  {'Observed TFLOPS':<28} {self.baseline_tflops:>11.1f} "
              f"{self.cipher_tflops:>11.1f} "
              f"{self.cipher_tflops-self.baseline_tflops:>+9.1f}")
        print(f"  {'SM Utilization':<28} {self.baseline_sm_util:>11.1f}% "
              f"{self.cipher_sm_util:>11.1f}%")
        print(f"  {'Duration (s)':<28} {self.baseline_duration:>11.1f} "
              f"{self.cipher_duration:>11.1f}")
        print(f"  {'Avg Power (W)':<28} {self.baseline_power:>11.1f} "
              f"{self.cipher_power:>11.1f}")

        print(f"\n  Substitution Stats:")
        print(f"    Total kernels:      {self.total_kernels:,}")
        print(f"    Substitutions:      {self.substitutions:,}")
        print(f"    Substitution rate:  {self.substitution_rate:.1%}")

        print(f"\n  Billing (20% Revenue Share):")
        print(f"    TFLOPS recovered:   {self.tflops_recovered:+.1f} TFLOPS")
        print(f"    GPU-hours saved:    {self.gpu_hours_saved:.4f} GPU-h "
              f"({self.gpu_hours_saved * 3600:.1f} GPU-s)")
        print(f"    Spot price:         ${self.spot_price:.2f}/GPU-h")
        print(f"    20% revenue share:  ${self.revenue_share_20pct:.4f}")
        print(f"\n  Samples:  baseline={self.baseline_samples}  "
              f"cipher={self.cipher_samples}")
        print("=" * 60)

    def save(self, path: str):
        """Save as JSON for customer audit trail."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"  [CIPHER DEP.2] Report saved: {path}")

    @classmethod
    def load(cls, path: str) -> "MFUReport":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ── Main meter class ──────────────────────────────────────────

class CipherMeter:
    """
    Continuous MFU meter for CIPHER billing.

    Usage:
        meter = CipherMeter()
        meter.start_baseline()
        # ... warmup training ...
        meter.start_cipher()
        # ... CIPHER-active training ...
        report = meter.stop()
        report.print()
        report.save("cipher_run.json")
    """

    SAMPLE_INTERVAL = 0.002  # 500Hz sampling

    def __init__(self, device_index: int = 0,
                 run_id: Optional[str] = None,
                 spot_price: Optional[float] = None):
        self._nvml     = NVML()
        self._available = self._nvml.init(device_index)

        if not self._available:
            print("[CIPHER DEP.2] NVML not available — "
                  "metrics will be zeros (stub mode)")

        self._run_id    = run_id or f"cipher_{int(time.time())}"
        self._spot_price = spot_price or SPOT_PRICE_PER_GPU_HOUR.get(
            self._nvml.device_name,
            SPOT_PRICE_PER_GPU_HOUR["default"])

        self._phase:   Optional[PhaseSamples] = None
        self._baseline: Optional[PhaseSamples] = None
        self._cipher:   Optional[PhaseSamples] = None

        self._thread:  Optional[threading.Thread] = None
        self._running  = False
        self._lock     = threading.Lock()

        # Substitution stats (populated from cipher_runtime if active)
        self._total_kernels  = 0
        self._substitutions  = 0

        if self._available:
            print(f"[CIPHER DEP.2] MFU Meter initialized")
            print(f"  Device:     {self._nvml.device_name}")
            print(f"  Peak BF16:  {self._nvml.peak_tflops:.0f} TFLOPS")
            print(f"  Boost clk:  {self._nvml.boost_clock} MHz")
            print(f"  Run ID:     {self._run_id}")
            print(f"  Spot price: ${self._spot_price:.2f}/GPU-h")

    def _sample_loop(self):
        """Background thread: samples NVML at 500Hz."""
        while self._running:
            s = self._nvml.sample()
            if s is not None:
                with self._lock:
                    if self._phase is not None:
                        self._phase.samples.append(s)
            time.sleep(self.SAMPLE_INTERVAL)

    def _start_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread  = threading.Thread(
                target=self._sample_loop, daemon=True)
            self._thread.start()

    def start_baseline(self):
        """Begin baseline measurement (before CIPHER activates)."""
        self._start_thread()
        with self._lock:
            self._phase    = PhaseSamples(name="baseline",
                                           start_time=time.time())
            self._baseline = self._phase
        print(f"[CIPHER DEP.2] Baseline measurement started")

    def start_cipher(self):
        """
        Switch from baseline to CIPHER measurement.
        Call this when CIPHER activates (after warmup).
        """
        with self._lock:
            if self._baseline is not None:
                self._baseline.end_time = time.time()
            self._phase  = PhaseSamples(name="cipher",
                                         start_time=time.time())
            self._cipher = self._phase

        # Try to get substitution stats from cipher_runtime
        self._update_substitution_stats()
        print(f"[CIPHER DEP.2] CIPHER measurement started  "
              f"(baseline: {len(self._baseline.samples) if self._baseline else 0} samples)")

    def _update_substitution_stats(self):
        """Pull substitution counts from cipher_runtime if available."""
        try:
            import cipher_runtime
            if hasattr(cipher_runtime, '_lib') and cipher_runtime._lib:
                lib = cipher_runtime._lib
                # cipher_intercept_stats returns pointer to stats struct
                stats_fn = getattr(lib, 'cipher_intercept_stats', None)
                if stats_fn:
                    # struct layout: total(u64), subs(u64), pass(u64), ...
                    class Stats(ctypes.Structure):
                        _fields_ = [
                            ("total_intercepts", ctypes.c_uint64),
                            ("substitutions",    ctypes.c_uint64),
                            ("passthroughs",     ctypes.c_uint64),
                        ]
                    stats_fn.restype = ctypes.POINTER(Stats)
                    s = stats_fn()
                    if s:
                        self._total_kernels = s.contents.total_intercepts
                        self._substitutions = s.contents.substitutions
        except Exception:
            pass  # Non-fatal: billing works without substitution stats

    def record_substitution(self, kernel: str = "",
                             recipe: str = "",
                             tflops_saved: float = 0.0):
        """
        Manually record a substitution event.
        Called by CIPHER dispatch when it substitutes a kernel.
        """
        # record_substitution is called AFTER record_kernel for the same kernel
        # so we only increment substitutions here, not total_kernels
        self._substitutions += 1

    def record_kernel(self):
        """Record any kernel (substituted or not)."""
        self._total_kernels += 1

    def stop(self) -> MFUReport:
        """Stop metering and generate the billing report."""
        self._running = False
        with self._lock:
            if self._phase is not None:
                self._phase.end_time = time.time()

        self._update_substitution_stats()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        # Compute report
        bl = self._baseline
        ci = self._cipher

        bl_mfu    = bl.avg_mfu()    if bl else 0.0
        ci_mfu    = ci.avg_mfu()    if ci else 0.0
        bl_tf     = bl.avg_tflops() if bl else 0.0
        ci_tf     = ci.avg_tflops() if ci else 0.0
        bl_sm     = bl.avg_sm_util() if bl else 0.0
        ci_sm     = ci.avg_sm_util() if ci else 0.0
        bl_dur    = bl.duration_s() if bl else 0.0
        ci_dur    = ci.duration_s() if ci else 0.0
        bl_pwr    = bl.avg_power()  if bl else 0.0
        ci_pwr    = ci.avg_power()  if ci else 0.0

        mfu_delta = ci_mfu - bl_mfu
        tf_recovered = ci_tf - bl_tf

        # GPU-hours saved:
        # If we recover X TFLOPS on a peak-Y TFLOPS GPU,
        # that's equivalent to X/Y fraction of a GPU
        # over the cipher_duration hours
        gpu_frac_recovered = tf_recovered / max(1.0, self._nvml.peak_tflops)
        gpu_hours_saved = gpu_frac_recovered * (ci_dur / 3600.0)

        # 20% revenue share in USD
        revenue_share = gpu_hours_saved * self._spot_price * 0.20

        sub_rate = (self._substitutions / max(1, self._total_kernels))

        report = MFUReport(
            run_id            = self._run_id,
            device_name       = self._nvml.device_name,
            peak_tflops       = self._nvml.peak_tflops,
            timestamp         = datetime.datetime.utcnow().isoformat() + "Z",
            baseline_mfu      = bl_mfu,
            cipher_mfu        = ci_mfu,
            baseline_tflops   = bl_tf,
            cipher_tflops     = ci_tf,
            baseline_sm_util  = bl_sm,
            cipher_sm_util    = ci_sm,
            baseline_duration = bl_dur,
            cipher_duration   = ci_dur,
            baseline_power    = bl_pwr,
            cipher_power      = ci_pwr,
            mfu_delta         = mfu_delta,
            tflops_recovered  = tf_recovered,
            gpu_hours_saved   = gpu_hours_saved,
            spot_price        = self._spot_price,
            revenue_share_20pct = revenue_share,
            total_kernels     = self._total_kernels,
            substitutions     = self._substitutions,
            substitution_rate = sub_rate,
            baseline_samples  = len(bl.samples) if bl else 0,
            cipher_samples    = len(ci.samples) if ci else 0,
        )

        self._nvml.shutdown()
        return report
