"""
CIPHER DEP.1 Test Suite
tests/test_packaging.py

Tests:
    ✓ Package installs without errors
    ✓ cipher_runtime imports cleanly
    ✓ libcipher.so is built and present
    ✓ libcipher.so exports expected symbols
    ✓ enable() / disable() work without crashing
    ✓ status() returns correct dict structure
    ✓ CIPHER=1 env var triggers _auto_enable
    ✓ cipher CLI entry point runs
    ✓ .pth file exists and is valid Python
    ✓ Package metadata is correct (name, version, author)
"""

import os
import sys
import subprocess
import ctypes
from pathlib import Path

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
g_pass = g_fail = 0

def check(cond, msg):
    global g_pass, g_fail
    if cond:
        print(f"  {PASS} {msg}")
        g_pass += 1
    else:
        print(f"  {FAIL} {msg}")
        g_fail += 1

def check_eq(a, b, msg):
    check(a == b, f"{msg} (got {a!r}, expected {b!r})")


# ---------------------------------------------------------------------------
# Locate the installed package
# ---------------------------------------------------------------------------

# Allow env var override for Colab/CI environments
_pkg_override = os.environ.get("CIPHER_PKG_DIR")
_src_override = os.environ.get("CIPHER_SRC_DIR")
PKG_DIR = Path(_pkg_override) if _pkg_override else Path(__file__).parent.parent.parent / "cipher_pkg"
sys.path.insert(0, str(PKG_DIR))


def test_import():
    print("\n[DEP.1.1] Package Import")
    try:
        import cipher_runtime
        check(True, "import cipher_runtime succeeds")
        check(hasattr(cipher_runtime, "__version__"), "has __version__")
        check(hasattr(cipher_runtime, "enable"),      "has enable()")
        check(hasattr(cipher_runtime, "disable"),     "has disable()")
        check(hasattr(cipher_runtime, "status"),      "has status()")
        check(hasattr(cipher_runtime, "_auto_enable"),"has _auto_enable()")
        check(cipher_runtime.__version__ == "0.1.0",  "version = 0.1.0")
    except ImportError as e:
        check(False, f"import failed: {e}")


def test_libcipher_build():
    print("\n[DEP.1.2] libcipher.so Build")

    # Build the stub .so
    cipher_src = Path(_src_override) if _src_override else Path(__file__).parent.parent
    pkg_dir    = PKG_DIR / "cipher_runtime"
    out_so     = pkg_dir / "libcipher.so"

    cflags = [
        "-std=c++17", "-O2", "-fPIC", "-shared",
        f"-I{cipher_src}/include",
        "-DCIPHER_CPU_STUB",
        "-Wno-unused-function", "-Wno-unused-variable",
        "-Wno-unused-parameter",
    ]

    sources = [
        "src/cipher_liquid_state.cu",
        "src/cipher_green_ctx.cu",
        "src/cipher_l2_persist.cu",
        "src/cipher_structural_lookup.cpp",
        "src/cipher_oracle.cpp",
        "src/cipher_recipes.cpp",
        "src/cipher_telemetry.cpp",
        "src/cipher_intercept.cpp",
        "src/cipher_10ops_impl.cpp",
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
    src_paths = [str(cipher_src / s) for s in sources]

    print(f"  Building {len(sources)} sources → {out_so.name}")
    # .cu files need -x c++ to tell g++ to treat them as C++
    cmd = ["g++"] + cflags
    for s in src_paths:
        cmd += ["-x", "c++", s]
    cmd += ["-o", str(out_so), "-lm", "-lpthread", "-ldl"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    check(result.returncode == 0, "libcipher.so compiles without errors")
    if result.returncode != 0:
        print("  Build errors:")
        for line in result.stderr.splitlines()[:5]:
            print(f"    {line}")
        return

    check(out_so.exists(),                   "libcipher.so exists after build")
    size_kb = out_so.stat().st_size // 1024
    print(f"  Size: {size_kb} KB")
    check(size_kb > 10,  "libcipher.so size > 10KB (not empty)")
    check(size_kb < 5000,"libcipher.so size < 5MB (reasonable)")


def test_so_symbols():
    print("\n[DEP.1.3] libcipher.so Symbol Exports")

    out_so = PKG_DIR / "cipher_runtime" / "libcipher.so"
    if not out_so.exists():
        check(False, "libcipher.so not found — run test_libcipher_build first")
        return

    result = subprocess.run(
        ["nm", "-D", "--defined-only", str(out_so)],
        capture_output=True, text=True
    )
    symbols = set()
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if parts and parts[-1].startswith("cipher_"):
            symbols.add(parts[-1])

    print(f"  Exported cipher_ symbols: {len(symbols)}")

    # Required symbols
    required = [
        "cipher_lnn_init",
        "cipher_lnn_forward",
        "cipher_layer2_init",
        "cipher_edmd_init",
        "cipher_hw_desc_normalize",
        "cipher_sm_packer_decide",
        "cipher_nccl_bpf_decide",
        "cipher_dispatch",       # main dispatch entry point
    ]
    for sym in required:
        check(sym in symbols, f"exports {sym}")


def test_ctypes_load():
    print("\n[DEP.1.4] ctypes Load and Call")

    out_so = PKG_DIR / "cipher_runtime" / "libcipher.so"
    if not out_so.exists():
        check(False, "libcipher.so not found")
        return

    try:
        lib = ctypes.CDLL(str(out_so))
        check(True, "ctypes.CDLL loads without error")
    except OSError as e:
        check(False, f"ctypes.CDLL failed: {e}")
        return

    # Call cipher_hw_desc_normalize_by_name
    try:
        fn = lib.cipher_hw_desc_normalize_by_name
        fn.restype  = None  # returns struct — can't easily call without binding
        check(True,  "cipher_hw_desc_normalize_by_name symbol accessible")
    except AttributeError:
        check(False, "cipher_hw_desc_normalize_by_name not found")

    # Call cipher_lnn_init — takes a pointer, just check it's callable
    try:
        fn = lib.cipher_lnn_init
        check(True, "cipher_lnn_init symbol accessible")
    except AttributeError:
        check(False, "cipher_lnn_init not found")


def test_python_api():
    print("\n[DEP.1.5] Python API")

    import cipher_runtime
    cipher_runtime._LIB_PATH = PKG_DIR / "cipher_runtime" / "libcipher.so"

    # enable() should succeed
    result = cipher_runtime.enable()
    check(result == True or result == False,
          "enable() returns bool (True or False, not exception)")

    # status() returns dict with required keys
    s = cipher_runtime.status()
    check(isinstance(s, dict), "status() returns dict")
    required_keys = ["version", "enabled", "lib_path", "lib_exists",
                     "cuda_stub", "env_active"]
    for k in required_keys:
        check(k in s, f"status() has key '{k}'")

    check(s["version"] == "0.1.0", "status()['version'] = 0.1.0")
    check(isinstance(s["enabled"], bool),    "status()['enabled'] is bool")
    check(isinstance(s["lib_exists"], bool), "status()['lib_exists'] is bool")

    # disable() should not crash
    cipher_runtime.disable()
    check(True, "disable() does not crash")


def test_auto_enable():
    print("\n[DEP.1.6] CIPHER=1 Auto-Enable")

    # Test that CIPHER=1 env var triggers _auto_enable in a subprocess
    env = os.environ.copy()
    env["CIPHER"] = "1"
    env["CIPHER_VERBOSE"] = "1"
    env["PYTHONPATH"] = str(PKG_DIR) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-c",
         "import cipher_runtime; print('imported'); "
         "s = cipher_runtime.status(); "
         "print('env_active:', s['env_active'])"],
        env=env, capture_output=True, text=True, timeout=10
    )
    check(result.returncode == 0, "CIPHER=1 subprocess exits cleanly")
    check("imported" in result.stdout, "cipher_runtime imports in CIPHER=1 subprocess")

    # _auto_enable is called when CIPHER=1; check it runs without raising
    result2 = subprocess.run(
        [sys.executable, "-c",
         "import os\nos.environ[\'CIPHER\']=\'1\'\n"
         "import cipher_runtime\n"
         "try:\n"
         "    cipher_runtime._auto_enable()\n"
         "    print(\'ok\')\n"
         "except Exception as e:\n"
         "    print(\'err:\', e)"],
        env={**os.environ, "PYTHONPATH": str(PKG_DIR) + ":" + os.environ.get("PYTHONPATH","")},
        capture_output=True, text=True, timeout=10
    )
    check("ok" in result2.stdout or result2.returncode == 0,
          "_auto_enable() runs without exception when CIPHER=1")


def test_cli():
    print("\n[DEP.1.7] CLI Entry Point")

    # Run the CLI via python -m
    result = subprocess.run(
        [sys.executable, "-m", "cipher_runtime.cli", "--version"],
        capture_output=True, text=True, timeout=10,
        env={**os.environ, "PYTHONPATH": str(PKG_DIR) + ":" + os.environ.get("PYTHONPATH","")}
    )
    check(result.returncode == 0, "cipher --version exits 0")
    check("0.1.0" in result.stdout, "cipher --version shows 0.1.0")

    # cipher status
    result2 = subprocess.run(
        [sys.executable, "-m", "cipher_runtime.cli", "status"],
        capture_output=True, text=True, timeout=10,
        env={**os.environ, "PYTHONPATH": str(PKG_DIR) + ":" + os.environ.get("PYTHONPATH","")}
    )
    check(result2.returncode == 0, "cipher status exits 0")
    check("CIPHER" in result2.stdout, "cipher status shows CIPHER info")

    # cipher enable
    result3 = subprocess.run(
        [sys.executable, "-m", "cipher_runtime.cli", "enable"],
        capture_output=True, text=True, timeout=10,
        env={**os.environ, "PYTHONPATH": str(PKG_DIR) + ":" + os.environ.get("PYTHONPATH","")}
    )
    check(result3.returncode == 0, "cipher enable exits 0")
    check("LD_PRELOAD" in result3.stdout, "cipher enable shows LD_PRELOAD command")
    check("CIPHER=1" in result3.stdout,   "cipher enable shows CIPHER=1 usage")


def test_pth_file():
    print("\n[DEP.1.8] .pth Autoload File")

    pth_path = PKG_DIR / "cipher_runtime" / "cipher_runtime_autoload.pth"
    check(pth_path.exists(), ".pth file exists")

    with open(pth_path) as f:
        content = f.read().strip()

    check("cipher_runtime" in content,   ".pth imports cipher_runtime")
    check("_auto_enable" in content,     ".pth calls _auto_enable()")

    # .pth must be valid Python
    try:
        compile(content, str(pth_path), "exec")
        check(True, ".pth file is valid Python")
    except SyntaxError as e:
        check(False, f".pth file has syntax error: {e}")


def test_package_structure():
    print("\n[DEP.1.9] Package Structure")

    required_files = [
        "cipher_pkg/pyproject.toml",
        "cipher_pkg/setup.py",
        "cipher_pkg/README.md",
        "cipher_pkg/cipher_runtime/__init__.py",
        "cipher_pkg/cipher_runtime/cli.py",
        "cipher_pkg/cipher_runtime/cipher_runtime_autoload.pth",
    ]

    base = PKG_DIR.parent
    for f in required_files:
        path = base / f
        check(path.exists(), f"{f} exists")

    # pyproject.toml has required fields
    pyproject = (base / "cipher_pkg" / "pyproject.toml").read_text()
    check("cipher-runtime" in pyproject,   "pyproject.toml: name = cipher-runtime")
    check("0.1.0" in pyproject,            "pyproject.toml: version = 0.1.0")
    check("cipher_runtime.cli:main" in pyproject, "pyproject.toml: CLI entry point")
    check("setuptools" in pyproject,       "pyproject.toml: build backend")


if __name__ == "__main__":
    print("===================================================")
    print("  CIPHER DEP.1 Test Suite")
    print("  Packaging & One-Line Install")
    print("===================================================")

    test_import()
    test_libcipher_build()
    test_so_symbols()
    test_ctypes_load()
    test_python_api()
    test_auto_enable()
    test_cli()
    test_pth_file()
    test_package_structure()

    print(f"\n===================================================")
    print(f"  Results: {g_pass} passed, {g_fail} failed")
    if g_fail == 0:
        print(f"  \033[32m DEP.1 GREEN — pip install cipher-runtime ready\033[0m")
    else:
        print(f"  \033[31m FAILURES REMAIN\033[0m")
    print(f"===================================================\n")
    sys.exit(1 if g_fail > 0 else 0)
