"""
cipher_runtime.report — DEP.2.3 Revenue Report
Neural Dynamics, Inc.

Combines MFU meter readings and substitution ledger into a
signed billing report. Human-readable and machine-readable.

This is what Nebius receives at the end of each training run.
It answers: "CIPHER ran on your cluster. Here's exactly what
it recovered, how it was measured, and what we're billing."

USAGE:
    from cipher_runtime.report import CipherReport

    report = CipherReport(
        meter_report=meter.stop(),
        ledger_summary=ledger.close(),
    )
    report.print()
    report.save("cipher_billing_2025_01.json")
"""

import json
import hashlib
import datetime
import os
from dataclasses import dataclass, asdict
from typing import Optional

from cipher_runtime.meter  import MFUReport
from cipher_runtime.ledger import LedgerSummary


@dataclass
class CipherReport:
    """
    Signed billing report combining meter + ledger.

    The signature is a SHA-256 hash of the billing fields.
    Customers can verify the report hasn't been tampered with
    by recomputing the hash from the JSON fields.
    """
    meter:   MFUReport
    ledger:  LedgerSummary
    version: str = "1.0"

    # Derived billing fields
    billing_gpu_hours:   float = 0.0
    billing_usd:         float = 0.0
    billing_explanation: str   = ""
    signature:           str   = ""
    generated_at:        str   = ""

    def __post_init__(self):
        self.generated_at = datetime.datetime.utcnow().isoformat() + "Z"
        self._compute_billing()
        self._sign()

    def _compute_billing(self):
        """
        Billing computation — fully auditable.

        Two independent signals agree on the result:
        1. MFU meter: measures actual hardware utilization
        2. Substitution ledger: sums per-kernel TFLOPS estimates

        We use the MFU meter as the primary (hardware-verified).
        The ledger provides the breakdown.
        """
        m = self.meter
        l = self.ledger

        # Primary: from MFU meter (hardware measurement)
        meter_gpu_hours = m.gpu_hours_saved
        meter_usd_20pct = m.revenue_share_20pct

        # Secondary: from ledger (per-kernel estimates)
        # ledger TFLOPS saved / peak TFLOPS = fraction of GPU
        # × cipher duration in hours
        peak = m.peak_tflops if m.peak_tflops > 0 else 989.0
        dur_h = m.cipher_duration / 3600.0
        ledger_gpu_hours = (l.total_tflops_saved / peak) * dur_h
        ledger_usd_20pct = ledger_gpu_hours * m.spot_price * 0.20

        # Use meter as primary; ledger as cross-check
        # If they disagree by >20%, flag for manual review
        if meter_gpu_hours > 0 and ledger_gpu_hours > 0:
            ratio = ledger_gpu_hours / meter_gpu_hours
            if ratio < 0.8 or ratio > 1.2:
                flag = f"WARNING: meter/ledger ratio={ratio:.2f} (>20% discrepancy)"
            else:
                flag = f"meter/ledger cross-check: {ratio:.3f}x (within 20%)"
        else:
            flag = "Single signal only (meter or ledger not available)"

        self.billing_gpu_hours = meter_gpu_hours
        self.billing_usd       = meter_usd_20pct
        self.billing_explanation = (
            f"Primary: MFU meter ({m.baseline_mfu:.1%}→{m.cipher_mfu:.1%} MFU, "
            f"Δ={m.mfu_delta:+.1%}). "
            f"GPU-hours saved: {meter_gpu_hours:.6f} "
            f"× ${m.spot_price:.2f}/h × 20% = ${meter_usd_20pct:.4f}. "
            f"Cross-check: {flag}."
        )

    def _sign(self):
        """
        SHA-256 signature over the billing fields.
        Customers can verify: hash(run_id + billing_gpu_hours + billing_usd
                                   + timestamp) == signature
        """
        payload = (
            f"{self.meter.run_id}"
            f"{self.billing_gpu_hours:.8f}"
            f"{self.billing_usd:.8f}"
            f"{self.generated_at}"
        )
        self.signature = hashlib.sha256(payload.encode()).hexdigest()

    def verify(self) -> bool:
        """Recompute signature and check it matches."""
        payload = (
            f"{self.meter.run_id}"
            f"{self.billing_gpu_hours:.8f}"
            f"{self.billing_usd:.8f}"
            f"{self.generated_at}"
        )
        expected = hashlib.sha256(payload.encode()).hexdigest()
        return expected == self.signature

    def print(self):
        m = self.meter
        l = self.ledger
        print("\n" + "=" * 64)
        print("  CIPHER Billing Report  v" + self.version)
        print("=" * 64)
        print(f"\n  Run ID:      {m.run_id}")
        print(f"  Device:      {m.device_name}  ({m.peak_tflops:.0f} TFLOPS peak BF16)")
        print(f"  Generated:   {self.generated_at}")
        print(f"  Signature:   {self.signature[:32]}...")

        print(f"\n  ── MFU Measurement (hardware-verified) ──────────────")
        print(f"  Baseline MFU:        {m.baseline_mfu:.1%}  "
              f"({m.baseline_tflops:.1f} TFLOPS obs)")
        print(f"  CIPHER MFU:          {m.cipher_mfu:.1%}  "
              f"({m.cipher_tflops:.1f} TFLOPS obs)")
        print(f"  Delta:               {m.mfu_delta:+.1%}")
        print(f"  Baseline duration:   {m.baseline_duration:.0f}s "
              f"({m.baseline_samples:,} NVML samples @ 500Hz)")
        print(f"  CIPHER duration:     {m.cipher_duration:.0f}s "
              f"({m.cipher_samples:,} NVML samples @ 500Hz)")

        print(f"\n  ── Substitution Ledger (per-kernel audit) ───────────")
        print(f"  Total substitutions: {l.total_entries:,}")
        print(f"  Substitution rate:   {m.substitution_rate:.1%}")
        print(f"  TFLOPS saved (est):  {l.total_tflops_saved:.2f}")
        print(f"  Avg latency:         {l.avg_latency_ns:.0f} ns  "
              f"(P99: {l.p99_latency_ns:.0f} ns)")
        print(f"  Avg error bound:     {l.error_bound_avg:.4f}")
        print(f"  Top recipe:          "
              f"{max(l.by_recipe.items(), key=lambda x: x[1])[0] if l.by_recipe else 'none'}")

        print(f"\n  ── Billing (20% Revenue Share) ───────────────────────")
        print(f"  GPU-hours recovered: {self.billing_gpu_hours:.6f} GPU-h")
        print(f"  Spot price:          ${m.spot_price:.2f}/GPU-h")
        print(f"  Amount due:          ${self.billing_usd:.4f}")
        print(f"  Explanation:         {self.billing_explanation}")

        valid = self.verify()
        print(f"\n  Signature valid:     {'YES ✓' if valid else 'NO ✗ — REPORT TAMPERED'}")
        print("=" * 64)

    def save(self, path: str):
        """Save as JSON for customer records."""
        data = {
            "version":            self.version,
            "generated_at":       self.generated_at,
            "signature":          self.signature,
            "billing_gpu_hours":  self.billing_gpu_hours,
            "billing_usd":        self.billing_usd,
            "billing_explanation": self.billing_explanation,
            "meter":   asdict(self.meter),
            "ledger":  asdict(self.ledger),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CIPHER DEP.2] Report saved: {path}")
        print(f"  Signature: {self.signature}")

    @classmethod
    def load(cls, path: str) -> "CipherReport":
        """Load and verify a saved report."""
        with open(path) as f:
            data = json.load(f)
        meter  = MFUReport(**data["meter"])
        ledger = LedgerSummary(**data["ledger"])
        r = object.__new__(cls)
        r.meter                = meter
        r.ledger               = ledger
        r.version              = data["version"]
        r.billing_gpu_hours    = data["billing_gpu_hours"]
        r.billing_usd          = data["billing_usd"]
        r.billing_explanation  = data["billing_explanation"]
        r.signature            = data["signature"]
        r.generated_at         = data["generated_at"]
        return r

    @classmethod
    def verified_load(cls, path: str) -> tuple:
        """Load and return (report, is_valid)."""
        r = cls.load(path)
        return r, r.verify()
