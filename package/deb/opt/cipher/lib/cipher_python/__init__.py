"""CIPHER Python integration package — auto-loaded by `cipher_run.sh`
via PYTHONSTARTUP. Detects the running ML framework and applies the
matching fusion shim so CIPHER's `cipher_fused_rmsnorm` /
`cipher_fused_silu_mul` driver kernels actually fire.

The shim approach (per-framework class-method patches) is the production
path for v1 because driver-level pointer extraction from ATen's
TensorIterator structs is per-kernel-layout-dependent and not yet
universal — see the M1.T2 STATUS block in
src/cipher_flow_substitute.cpp for the technical detail.

Usage (from cipher_run.sh):
    PYTHONSTARTUP=/path/to/cipher_python/autoload.py python3 ...

Or explicitly from a script:
    import cipher_python
    cipher_python.apply_shims()
"""
import os
import sys


def detect_framework() -> str:
    """Return one of: 'transformers', 'vllm', 'sglang', 'trtllm', 'unknown'.
    Detection is based on which framework has been imported into sys.modules
    by the time this is called. Should be called AFTER the customer's
    framework has been imported."""
    if "vllm" in sys.modules:
        return "vllm"
    if "sglang" in sys.modules:
        return "sglang"
    if "tensorrt_llm" in sys.modules:
        return "trtllm"
    if "transformers" in sys.modules:
        return "transformers"
    return "unknown"


def load_cipher_rt():
    """Load libcipher_rt.so via ctypes RTLD_GLOBAL so the LD_PRELOAD'd
    libcipher_hook.so can resolve RT symbols. Returns the handle or None."""
    import ctypes
    rt_path = os.environ.get("CIPHER_RT_PATH")
    if not rt_path:
        # Default to /opt/cipher/lib (production install) or the repo root
        # for in-place dev runs.
        for candidate in ["/opt/cipher/lib/libcipher_rt.so",
                          os.path.join(os.path.dirname(__file__), "..",
                                        "libcipher_rt.so")]:
            if os.path.exists(candidate):
                rt_path = candidate
                break
    if not rt_path or not os.path.exists(rt_path):
        return None
    try:
        return ctypes.CDLL(rt_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as ex:
        print(f"[CIPHER PY] libcipher_rt.so load failed: {ex}", file=sys.stderr)
        return None


def apply_shims(model=None, force_framework: str = None):
    """Auto-detect framework + load matching shim. If `model` is provided
    (a torch.nn.Module), apply class-method patches to its instances as
    well. Returns dict of stats from the shim."""
    fw = force_framework or detect_framework()
    rt = load_cipher_rt()
    if rt is None:
        print("[CIPHER PY] no libcipher_rt.so loaded; shim is no-op",
              file=sys.stderr)
        return {"framework": fw, "applied": False, "reason": "no_rt"}

    try:
        if fw == "transformers":
            from .shims import transformers as shim
        elif fw == "vllm":
            from .shims import vllm as shim
        elif fw == "sglang":
            from .shims import sglang as shim
        elif fw == "trtllm":
            from .shims import trtllm as shim
        else:
            print(f"[CIPHER PY] framework='{fw}' unknown; no shim loaded",
                  file=sys.stderr)
            return {"framework": fw, "applied": False, "reason": "unknown_framework"}
    except ImportError as ex:
        print(f"[CIPHER PY] shim import failed for {fw}: {ex}",
              file=sys.stderr)
        return {"framework": fw, "applied": False, "reason": str(ex)}

    print(f"[CIPHER PY] framework={fw}: applying fusion shim", file=sys.stderr)
    return shim.apply(rt, model)
