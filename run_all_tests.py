"""
CIPHER TEST SUITE — Master Runner
===================================
Run all tests in sequence. Stop on first failure.
Do NOT modify cipher_colab_setup_final.py until all pass.

Usage:
  python3 run_all_tests.py          # run all tests
  python3 run_all_tests.py --colab  # colab-safe tests only (01, 03-09)
  python3 run_all_tests.py --gpu    # add GPU tests (02)
  python3 run_all_tests.py --nebius # full suite including live-fire
"""

import sys
import os
import subprocess
import time
import argparse

TEST_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()

TESTS = [
    {
        "file":     "test_01_ring_buffer.py",
        "name":     "01 — Ring Buffer",
        "requires": "cpu",
        "blocking": True,
        "description": "Lock-free ring buffer: write latency, correctness, throughput"
    },
    {
        "file":     "test_02_green_context.py",
        "name":     "02 — Green Context Sub-Partition",
        "requires": "gpu",
        "blocking": True,
        "description": "5-2-1 SM split test — Mission 2 Risk 1"
    },
    {
        "file":     "test_03_to_09_shadow_ops.py",
        "name":     "03-09 — Shadow Architecture + Operations",
        "requires": "cpu",
        "blocking": True,
        "description": "Thread arch, REMEMBER, VALIDATE, AUDIT, SPECULATE, ADAPT, Integration"
    },
    {
        "file":     "test_10_12_nebius_livewire.py",
        "name":     "10-12 — Nebius Live-Fire",
        "requires": "nebius",
        "blocking": False,
        "description": "LD_PRELOAD intercept + Mission 2 hardware measurements"
    },
]

def run_test(test: dict) -> bool:
    path = os.path.join(TEST_DIR, test["file"])
    if not os.path.exists(path):
        print(f"  ❌ File not found: {path}")
        return False

    print(f"\n{'═' * 60}")
    print(f"Running: {test['name']}")
    print(f"{'═' * 60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, path],
        cwd=TEST_DIR
    )
    elapsed = time.time() - t0

    status = "✅ PASSED" if result.returncode == 0 else "❌ FAILED"
    print(f"\n{status} in {elapsed:.1f}s")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="CIPHER Test Suite")
    parser.add_argument("--colab",  action="store_true", help="CPU-only tests (01, 03-09)")
    parser.add_argument("--gpu",    action="store_true", help="CPU + GPU tests (01, 02, 03-09)")
    parser.add_argument("--nebius", action="store_true", help="Full suite including live-fire")
    args = parser.parse_args()

    # Determine which tests to run
    if args.nebius:
        run_levels = {"cpu", "gpu", "nebius"}
    elif args.gpu:
        run_levels = {"cpu", "gpu"}
    elif args.colab:
        run_levels = {"cpu"}
    else:
        run_levels = {"cpu"}  # default: safe for any machine

    print("=" * 60)
    print("CIPHER TEST SUITE — Master Runner")
    print("=" * 60)
    print(f"Mode: {'--nebius' if args.nebius else '--gpu' if args.gpu else '--colab (default)'}")
    print(f"Running tests requiring: {run_levels}")
    print()

    results = {}
    failed_blocking = False

    for test in TESTS:
        if test["requires"] not in run_levels:
            print(f"  ⏭️  SKIP  {test['name']}  (requires {test['requires']})")
            results[test["name"]] = None
            continue

        if failed_blocking:
            print(f"  ⏭️  SKIP  {test['name']}  (blocked by previous failure)")
            results[test["name"]] = None
            continue

        passed = run_test(test)
        results[test["name"]] = passed

        if not passed and test["blocking"]:
            failed_blocking = True
            print(f"\n🛑 BLOCKING FAILURE: {test['name']}")
            print("   Fix this test before proceeding.")
            print("   Do NOT modify cipher_colab_setup_final.py")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    all_ran_passed = True
    for name, passed in results.items():
        if passed is None:
            print(f"  ⏭️  SKIP  {name}")
        elif passed:
            print(f"  ✅ PASS  {name}")
        else:
            print(f"  ❌ FAIL  {name}")
            all_ran_passed = False

    print()
    if failed_blocking:
        print("❌ SUITE FAILED — Fix blocking failures before Nebius")
        print("   Do NOT modify cipher_colab_setup_final.py")
        return 1

    if all_ran_passed:
        if args.nebius:
            print("✅ FULL SUITE PASSED")
            print()
            print("   → Ready to run integration:")
            print("     python3 test_00_integrate.py")
            print("     Output: cipher_nebius_v1_10ops.py")
        elif args.gpu:
            print("✅ CPU + GPU TESTS PASSED")
            print()
            print("   Next: Book Nebius session")
            print("   Then: python3 run_all_tests.py --nebius")
        else:
            print("✅ CPU TESTS PASSED")
            print()
            print("   Next: Run on GPU")
            print("     python3 run_all_tests.py --gpu")

    return 0 if all_ran_passed else 1

if __name__ == "__main__":
    sys.exit(main())
