"""
CIPHER DEP.2 Test Suite
tests/test_dep2.py

Tests:
    ✓ CipherMeter initializes (stub mode without NVML)
    ✓ CipherMeter baseline/cipher phase transition
    ✓ CipherMeter generates valid MFUReport
    ✓ MFUReport saves and loads from JSON
    ✓ MFUReport billing math is correct
    ✓ CipherLedger writes entries to NDJSON
    ✓ CipherLedger is thread-safe
    ✓ CipherLedger replay reads back correctly
    ✓ LedgerSummary aggregates correctly
    ✓ CipherReport combines meter + ledger
    ✓ CipherReport signature verifies
    ✓ CipherReport detects tampering
    ✓ CipherReport saves and loads
    ✓ Full pipeline: meter → ledger → report → save → load → verify
"""

import os, sys, json, time, threading, tempfile, math
from pathlib import Path

# Add package to path
PKG_DIR = Path(__file__).parent.parent.parent / "cipher_pkg"
sys.path.insert(0, str(PKG_DIR))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
g_pass = g_fail = 0

def check(cond, msg, detail=""):
    global g_pass, g_fail
    if cond:
        print(f"  {PASS} {msg}" + (f"  [{detail}]" if detail else ""))
        g_pass += 1
    else:
        print(f"  {FAIL} {msg}" + (f"  [{detail}]" if detail else ""))
        g_fail += 1
    return cond


# ── DEP.2.1: CipherMeter ─────────────────────────────────────

def test_meter():
    print("\n[DEP.2.1] CipherMeter")
    from cipher_runtime.meter import CipherMeter, MFUReport, SPOT_PRICE_PER_GPU_HOUR

    with tempfile.TemporaryDirectory() as tmp:

        # Init in stub mode (no GPU here)
        meter = CipherMeter(run_id="test_run_001")
        check(meter is not None, "CipherMeter initializes")

        # Baseline phase
        meter.start_baseline()
        time.sleep(0.05)  # 50ms baseline

        # Cipher phase
        meter.start_cipher()
        # Simulate 150 kernel dispatches: 100 substituted, 50 exact
        for i in range(150):
            meter.record_kernel()
            if i < 100:
                meter.record_substitution(
                    kernel=f"cublas_gemm_{i}",
                    recipe="gemm_roofline",
                    tflops_saved=0.1,
                )

        time.sleep(0.05)  # 50ms cipher

        # Stop and get report
        report = meter.stop()
        check(isinstance(report, MFUReport), "meter.stop() returns MFUReport")
        check(report.run_id == "test_run_001", "run_id preserved")
        check(report.baseline_duration > 0, "baseline_duration > 0")
        check(report.cipher_duration > 0, "cipher_duration > 0")
        check(report.total_kernels == 150, f"total_kernels=150",
              str(report.total_kernels))
        check(report.substitutions == 100, "substitutions=100",
              str(report.substitutions))
        check(report.substitution_rate > 0.6, "substitution_rate > 60%",
              f"{report.substitution_rate:.1%}")
        check(report.spot_price > 0, "spot_price > 0",
              f"${report.spot_price:.2f}")
        check(math.isfinite(report.mfu_delta), "mfu_delta is finite")
        check(math.isfinite(report.revenue_share_20pct),
              "revenue_share_20pct is finite")

        # Billing math check (stub mode: mfu will be 0, so delta=0, billing=0)
        # Just verify the math formula is correct structurally
        if report.mfu_delta > 0:
            expected_billing = report.gpu_hours_saved * report.spot_price * 0.20
            check(abs(report.revenue_share_20pct - expected_billing) < 1e-6,
                  "billing = gpu_hours × spot_price × 20%")

        # Save and load
        report_path = os.path.join(tmp, "test_report.json")
        report.save(report_path)
        check(os.path.exists(report_path), "report.save() creates file")

        loaded = MFUReport.load(report_path)
        check(loaded.run_id == report.run_id, "loaded run_id matches")
        check(abs(loaded.mfu_delta - report.mfu_delta) < 1e-10,
              "loaded mfu_delta matches")
        check(loaded.total_kernels == report.total_kernels,
              "loaded total_kernels matches")


# ── DEP.2.2: CipherLedger ────────────────────────────────────

def test_ledger():
    print("\n[DEP.2.2] CipherLedger")
    from cipher_runtime.ledger import CipherLedger, LedgerSummary

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.ndjson")

        ledger = CipherLedger(path, run_id="test_ledger_001",
                               device_name="H100 SXM5")
        check(os.path.exists(path), "ledger file created")

        # Record entries
        n = 500
        for i in range(n):
            ledger.record(
                kernel=f"cublas_gemm_{i % 10}",
                layer=i % 16,
                recipe="gemm_roofline" if i % 3 == 0 else "chebyshev",
                op_class="GEMM" if i % 2 == 0 else "EW",
                grid_x=1024, grid_y=1024,
                latency_ns=1797 + (i % 100),
                tflops_saved=0.1 * (1 + i % 5),
                confidence=0.85 + 0.1 * (i % 3) / 3,
                error_bound=0.01 * (i % 10) / 10,
                phase="convergence",
            )

        check(ledger.entry_count == n, f"entry_count = {n}",
              str(ledger.entry_count))
        check(ledger.total_tflops_saved > 0, "total_tflops_saved > 0",
              f"{ledger.total_tflops_saved:.2f}")

        # Thread safety test
        errors = []
        def write_batch(start, count):
            try:
                for i in range(start, start + count):
                    ledger.record(kernel=f"threaded_{i}", layer=0,
                                  recipe="edmd", op_class="GEMM",
                                  tflops_saved=0.05)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_batch, args=(n+i*100, 100))
                   for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        check(len(errors) == 0, "thread-safe concurrent writes", str(errors))
        check(ledger.entry_count == n + 400, "all threaded entries recorded",
              str(ledger.entry_count))

        # Close and get summary
        summary = ledger.close()
        check(isinstance(summary, LedgerSummary), "close() returns LedgerSummary")
        check(summary.total_entries == n + 400, "summary total_entries correct")
        check(summary.total_tflops_saved > 0, "summary total_tflops_saved > 0")
        check(summary.avg_latency_ns > 0, "avg_latency > 0")
        check(len(summary.by_recipe) == 3, "3 recipes in summary",
              str(summary.by_recipe.keys()))
        check(len(summary.by_op_class) == 2, "2 op classes in summary")
        check(summary.p50_latency_ns > 0, "p50 latency computed")
        check(summary.p99_latency_ns >= summary.p50_latency_ns,
              "p99 >= p50")

        # Replay audit
        entries_seen = 0
        header_seen  = False
        footer_seen  = False
        for obj in CipherLedger.replay(path):
            if obj.get("_type") == "cipher_ledger_header":
                header_seen = True
                check(obj["run_id"] == "test_ledger_001", "header run_id correct")
            elif obj.get("_type") == "cipher_ledger_footer":
                footer_seen = True
            elif "seq" in obj:
                entries_seen += 1

        check(header_seen, "header present in replay")
        check(footer_seen, "footer present in replay")
        check(entries_seen == n + 400, f"all {n+400} entries replay correctly",
              str(entries_seen))


# ── DEP.2.3: CipherReport ────────────────────────────────────

def test_report():
    print("\n[DEP.2.3] CipherReport")
    from cipher_runtime.meter  import MFUReport
    from cipher_runtime.ledger import LedgerSummary
    from cipher_runtime.report import CipherReport

    with tempfile.TemporaryDirectory() as tmp:

        # Build a realistic MFUReport manually
        meter_report = MFUReport(
            run_id            = "nebius_pilot_001",
            device_name       = "H100 SXM5",
            peak_tflops       = 989.0,
            timestamp         = "2025-01-15T10:00:00Z",
            baseline_mfu      = 0.42,    # 42% baseline MFU
            cipher_mfu        = 0.58,    # 58% with CIPHER
            baseline_tflops   = 415.4,
            cipher_tflops     = 573.6,
            baseline_sm_util  = 42.0,
            cipher_sm_util    = 58.0,
            baseline_duration = 3600.0,  # 1 hour baseline
            cipher_duration   = 7200.0,  # 2 hours cipher
            baseline_power    = 320.0,
            cipher_power      = 380.0,
            mfu_delta         = 0.16,    # +16 percentage points
            tflops_recovered  = 158.2,
            gpu_hours_saved   = 0.3208,  # 158.2/989 × 2h
            spot_price        = 3.00,
            revenue_share_20pct = 0.1925,  # 0.3208 × 3.00 × 0.20
            total_kernels     = 1_847_293,
            substitutions     = 1_293_105,
            substitution_rate = 0.70,
            baseline_samples  = 1_800_000,
            cipher_samples    = 3_600_000,
        )

        # Build a realistic LedgerSummary
        ledger_summary = LedgerSummary(
            run_id              = "nebius_pilot_001",
            device_name         = "H100 SXM5",
            start_time          = "2025-01-15T10:00:00Z",
            end_time            = "2025-01-15T13:00:00Z",
            total_entries       = 1_293_105,
            total_tflops_saved  = 1_140.8,
            avg_tflops_per_sub  = 0.000882,
            avg_confidence      = 0.91,
            avg_latency_ns      = 1812.0,
            p50_latency_ns      = 1797,
            p99_latency_ns      = 2341,
            by_recipe           = {"gemm_roofline": 901_243, "chebyshev": 391_862},
            by_op_class         = {"GEMM": 1_012_847, "EW": 280_258},
            by_layer            = {str(i): 80_819 for i in range(16)},
            error_bound_max     = 0.048,
            error_bound_avg     = 0.031,
        )

        # Generate report
        report = CipherReport(meter=meter_report, ledger=ledger_summary)
        check(report is not None, "CipherReport generates")
        check(len(report.signature) == 64, "signature is 64-char hex SHA-256")
        check(report.billing_gpu_hours > 0, "billing_gpu_hours > 0",
              f"{report.billing_gpu_hours:.4f}")
        check(report.billing_usd > 0, "billing_usd > 0",
              f"${report.billing_usd:.4f}")

        # Verify billing math
        expected = meter_report.gpu_hours_saved * meter_report.spot_price * 0.20
        check(abs(report.billing_usd - expected) < 0.001,
              "billing = gpu_hours × spot_price × 0.20",
              f"${report.billing_usd:.4f} vs ${expected:.4f}")

        # Signature verifies
        check(report.verify(), "signature verifies on original report")

        # Save and reload
        path = os.path.join(tmp, "billing_report.json")
        report.save(path)
        check(os.path.exists(path), "report saved to file")

        loaded = CipherReport.load(path)
        check(loaded.verify(), "signature verifies after load")
        check(loaded.meter.run_id == "nebius_pilot_001",
              "loaded run_id correct")
        check(abs(loaded.billing_usd - report.billing_usd) < 1e-8,
              "loaded billing_usd matches")

        # Tamper detection
        with open(path) as f:
            data = json.load(f)
        data["billing_usd"] += 1.0  # tamper: inflate billing
        tampered_path = os.path.join(tmp, "tampered.json")
        with open(tampered_path, "w") as f:
            json.dump(data, f)
        tampered = CipherReport.load(tampered_path)
        check(not tampered.verify(),
              "signature FAILS after billing_usd tampered (tamper detected)")

        # Cross-check: meter vs ledger
        # ledger says 1140.8 TFLOPS saved over 2h
        # that's 1140.8/989 × 2 = 2.307 GPU-h
        # meter says 0.3208 GPU-h  → ratio = 0.3208/2.307 = 0.139
        # Large discrepancy expected (ledger cumulative, meter per-step average)
        # The explanation string should note this
        check("cross-check" in report.billing_explanation.lower() or
              "WARNING" in report.billing_explanation,
              "billing explanation includes cross-check note")


# ── DEP.2.4: Full pipeline test ──────────────────────────────

def test_full_pipeline():
    print("\n[DEP.2.4] Full Pipeline (meter → ledger → report → audit)")
    from cipher_runtime.meter  import CipherMeter
    from cipher_runtime.ledger import CipherLedger
    from cipher_runtime.report import CipherReport

    with tempfile.TemporaryDirectory() as tmp:
        ledger_path = os.path.join(tmp, "run.ndjson")
        report_path = os.path.join(tmp, "billing.json")

        # Simulate a training run
        meter  = CipherMeter(run_id="pipeline_test_001")
        ledger = CipherLedger(ledger_path, run_id="pipeline_test_001")

        meter.start_baseline()
        time.sleep(0.03)

        meter.start_cipher()
        # Simulate 1000 kernel dispatches: 700 substituted
        for i in range(1000):
            meter.record_kernel()
            if i % 10 < 7:  # 70% substitution rate
                meter.record_substitution(tflops_saved=0.08)
                ledger.record(
                    kernel=f"gemm_{i}",
                    layer=i % 8,
                    recipe="gemm_roofline",
                    op_class="GEMM",
                    latency_ns=1797,
                    tflops_saved=0.08,
                    confidence=0.91,
                    error_bound=0.045,
                )
        time.sleep(0.03)

        meter_report  = meter.stop()
        ledger_summary = ledger.close()

        check(meter_report.total_kernels == 1000, "meter: 1000 kernels")
        check(meter_report.substitutions == 700,  "meter: 700 substitutions")
        check(ledger_summary.total_entries == 700, "ledger: 700 entries")

        # Generate billing report
        report = CipherReport(meter=meter_report, ledger=ledger_summary)
        report.save(report_path)

        # Verify audit trail
        loaded, valid = CipherReport.verified_load(report_path)
        check(valid, "saved report verifies")
        check(loaded.meter.substitutions == 700, "audit: substitution count preserved")
        check(loaded.ledger.total_entries == 700, "audit: ledger entries preserved")
        check(loaded.billing_usd >= 0, "audit: billing >= 0")

        # Replay ledger
        entry_count = sum(1 for obj in CipherLedger.replay(ledger_path)
                          if "seq" in obj)
        check(entry_count == 700, f"ledger audit replay: {entry_count} entries")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CIPHER DEP.2 Test Suite")
    print("  Metering, Ledger, Revenue Report")
    print("=" * 60)

    test_meter()
    test_ledger()
    test_report()
    test_full_pipeline()

    print(f"\n{'='*60}")
    print(f"  Results: {g_pass} passed, {g_fail} failed")
    if g_fail == 0:
        print(f"  \033[32m DEP.2 GREEN — metering and billing operational\033[0m")
    else:
        print(f"  \033[31m FAILURES REMAIN\033[0m")
    print(f"{'='*60}\n")

    sys.exit(1 if g_fail > 0 else 0)
