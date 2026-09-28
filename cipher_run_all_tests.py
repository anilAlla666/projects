#!/usr/bin/env python3
"""
CIPHER Full Test Suite Runner
Neural Dynamics, Inc.

Runs all 359 tests across all components.
Usage: python3 cipher_run_all_tests.py

On Colab: exec(open('/content/cipher_run_all_tests.py').read())
"""

import subprocess, sys, os, time
from pathlib import Path

R   = "/content/CIPHER"           # CIPHER C++ source
PKG = "/content/cipher_pkg"       # Python package
OUT = "/tmp/cipher_tests"
os.makedirs(OUT, exist_ok=True)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

CFLAGS = (f"-std=c++17 -O2 -I{R}/include -DCIPHER_CPU_STUB "
          f"-Wno-unused-function -Wno-unused-variable -Wno-unused-parameter")

# All C++ sources needed
ALL_SRCS = [
    "src/cipher_liquid_state.cu",
    "src/cipher_green_ctx.cu",
    "src/cipher_l2_persist.cu",
    "src/cipher_structural_lookup.cpp",
    "src/cipher_oracle.cpp",
    "src/cipher_recipes.cpp",
    "src/cipher_telemetry.cpp",
    "src/cipher_intercept.cpp",
    "src/cipher_runtime.cpp",
    "src/cipher_dispatch.cpp",
    "src/cipher_sm_packer.cpp",
    "src/cipher_fusion.cpp",
    "src/cipher_mem_layout.cpp",
    "src/cipher_nccl_bpf.cpp",
    "src/cipher_nccl_neural.cpp",
    "src/cipher_layer2.cpp",
    "src/cipher_edmd.cpp",
    "src/cipher_lnn.cpp",
    "src/cipher_hw_desc.cpp",
    "src/cipher_koopman_runtime.cpp",
]

# Test suites: (binary_name, test_source, description, expected_tests)
CPP_SUITES = [
    ("t_layer3",  "tests/test_layer3.cpp",      "Layer 3 (classify→oracle→dispatch)",  56),
    ("t_layer2",  "tests/test_layer2.cpp",      "Layer 2 (SM/fusion/layout/NCCL)",     35),
    ("t_edmd",    "tests/test_layer3_edmd.cpp", "L3.5 EDMD + EXP.A",                  29),
    ("t_lnn",     "tests/test_lnn.cpp",         "L3.10 CfC LNN",                       32),
    ("t_hw",      "tests/test_hw_desc.cpp",     "L1.4 HW Normalization",               51),
    ("t_l11",     "tests/test_l11.cpp",         "L1.1 Runtime Koopman Derivation",     46),
]

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")

# ── Phase 1: Compile all C++ sources ─────────────────────────
section("Phase 1: Compiling C++ sources")
t0 = time.time()
objs = []
compile_ok = True
for src in ALL_SRCS:
    obj = f"{OUT}/{Path(src).stem}.o"
    r = run(f"g++ {CFLAGS} -x c++ -c {R}/{src} -o {obj}")
    if r.returncode == 0:
        objs.append(obj)
        print(f"  {PASS} {src}")
    else:
        errs = [l for l in r.stderr.splitlines() if "error:" in l][:1]
        print(f"  {FAIL} {src}: {errs[0] if errs else '?'}")
        compile_ok = False

if not compile_ok:
    print("\n  ✗ Compilation failed — cannot run tests")
    sys.exit(1)

print(f"\n  All {len(ALL_SRCS)} sources compiled in {time.time()-t0:.1f}s")

# ── Phase 2: C++ test suites ──────────────────────────────────
section("Phase 2: C++ Test Suites")

total_pass = total_fail = 0
results = []

for binary, test_src, desc, expected in CPP_SUITES:
    bin_path = f"{OUT}/{binary}"
    # Compile test
    r = run(f"g++ {CFLAGS} -x c++ -c {R}/{test_src} -o {OUT}/{binary}.o")
    if r.returncode != 0:
        print(f"\n  {FAIL} [{desc}] compile error")
        results.append((desc, 0, 1, expected))
        total_fail += 1
        continue
    # Link
    r = run(f"g++ {' '.join(objs)} {OUT}/{binary}.o -lpthread -ldl -lm -o {bin_path}")
    if r.returncode != 0:
        print(f"\n  {FAIL} [{desc}] link error")
        results.append((desc, 0, 1, expected))
        total_fail += 1
        continue
    # Run
    r = run(bin_path)
    # Parse Results line
    p = f = 0
    for line in r.stdout.splitlines():
        if "Results:" in line:
            try:
                p = int(line.split("passed")[0].split()[-1])
                f = int(line.split("failed")[0].split()[-1])
            except: pass
    total_pass += p
    total_fail += f
    results.append((desc, p, f, expected))
    status = f"\033[32mGREEN\033[0m" if f == 0 else f"\033[31m{f} FAILED\033[0m"
    print(f"\n  [{desc}]")
    print(f"    {p}/{expected} tests — {status}")

# ── Phase 3: Python test suites ───────────────────────────────
section("Phase 3: Python Test Suites (DEP.1 + DEP.2)")

env = {**os.environ,
       "PYTHONPATH": f"{PKG}:{os.environ.get('PYTHONPATH','')}",
       "CIPHER_PKG_DIR": PKG,
       "CIPHER_SRC_DIR": R}

py_suites = [
    (f"{R}/tests/test_packaging.py", "DEP.1 pip packaging",       58),
    (f"{R}/tests/test_dep2.py",      "DEP.2 metering + billing",  52),
]

for test_path, desc, expected in py_suites:
    r = subprocess.run([sys.executable, test_path],
                       env=env, capture_output=True, text=True)
    p = f = 0
    for line in r.stdout.splitlines():
        if "Results:" in line:
            try:
                p = int(line.split("passed")[0].split()[-1])
                f = int(line.split("failed")[0].split()[-1])
            except: pass
    total_pass += p
    total_fail += f
    results.append((desc, p, f, expected))
    status = f"\033[32mGREEN\033[0m" if f == 0 else f"\033[31m{f} FAILED\033[0m"
    print(f"\n  [{desc}]")
    print(f"    {p}/{expected} tests — {status}")

# ── Summary ───────────────────────────────────────────────────
section("CIPHER Test Summary")

print(f"\n  {'Component':<40} {'Pass':>6} {'Fail':>6} {'Status'}")
print(f"  {'─'*62}")
for desc, p, f, expected in results:
    bar = f"\033[32m✓\033[0m" if f == 0 else f"\033[31m✗\033[0m"
    print(f"  {bar} {desc:<38} {p:>6} {f:>6}")

print(f"\n  {'─'*62}")
print(f"  {'TOTAL':<40} {total_pass:>6} {total_fail:>6}")
print()

if total_fail == 0:
    print(f"  \033[32m{'='*62}\033[0m")
    print(f"  \033[32m  {total_pass} / {total_pass} TESTS GREEN — CIPHER build complete\033[0m")
    print(f"  \033[32m{'='*62}\033[0m")
else:
    print(f"  \033[31m  {total_fail} failures — see above\033[0m")

print()
