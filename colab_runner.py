"""
CIPHER Test Suite — Colab Runner
==================================
Paste this entire cell into Colab and run it.
Handles __file__ issue, path setup, and runs all CPU tests.

Usage in Colab:
  1. Upload all cipher_tests/ files to /content/cipher_tests/
  2. Run this cell

OR run directly:
  !python3 /content/cipher_tests/run_all_tests.py --colab
  !python3 /content/cipher_tests/run_all_tests.py --gpu
"""

import os
import sys
import subprocess

# ── Colab Setup ─────────────────────────────────────────────────────────────

# Detect where the test files are
POSSIBLE_DIRS = [
    "/content/cipher_tests",
    "/content/",
    os.getcwd(),
    os.path.join(os.getcwd(), "cipher_tests"),
]

TEST_DIR = None
for d in POSSIBLE_DIRS:
    if os.path.exists(os.path.join(d, "run_all_tests.py")):
        TEST_DIR = d
        break

if TEST_DIR is None:
    print("❌ Could not find test files.")
    print("   Please upload cipher_tests/ folder to /content/")
    print("   Then re-run this cell.")
    raise SystemExit(1)

print(f"✅ Test directory: {TEST_DIR}")
print(f"   Python:         {sys.executable}")
print(f"   GPU available:  ", end="")

# Check GPU
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5
    )
    if r.returncode == 0:
        print(r.stdout.strip())
        HAS_GPU = True
    else:
        print("None")
        HAS_GPU = False
except Exception:
    print("None (nvidia-smi not found)")
    HAS_GPU = False

# ── Pick Mode ───────────────────────────────────────────────────────────────

if HAS_GPU:
    MODE = "--gpu"
    print(f"\n   Mode: GPU (runs tests 01, 02, 03-09)")
else:
    MODE = "--colab"
    print(f"\n   Mode: CPU only (runs tests 01, 03-09)")

print(f"\n{'='*60}")
print("Running CIPHER test suite...")
print(f"{'='*60}\n")

# ── Run ─────────────────────────────────────────────────────────────────────

result = subprocess.run(
    [sys.executable, os.path.join(TEST_DIR, "run_all_tests.py"), MODE],
    cwd=TEST_DIR
)

print(f"\n{'='*60}")
if result.returncode == 0:
    print("✅ ALL TESTS PASSED")
    if HAS_GPU:
        print("   Ready for Nebius session")
        print("   Next: run_all_tests.py --nebius on Nebius H100")
    else:
        print("   Run on a GPU runtime for test_02_green_context")
        print("   Runtime → Change runtime type → T4 or A100")
else:
    print("❌ TESTS FAILED — check output above")
print(f"{'='*60}")
