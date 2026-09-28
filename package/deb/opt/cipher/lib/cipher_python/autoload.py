"""PYTHONSTARTUP shim for cipher_run.sh.

Runs at Python startup. Installs an import hook that triggers
`cipher_python.apply_shims()` after the customer's framework has
loaded. We can't apply shims at startup because the customer's
framework hasn't been imported yet.

The trick: register an atexit-style callback that fires on first
torch.cuda.synchronize() (the first GPU op the customer makes — by
which time torch + framework are fully loaded). This is much more
reliable than guessing import order.
"""
import os
import sys


def _install_first_cuda_op_hook():
    """Install a one-shot hook on torch.cuda.synchronize that triggers
    shim loading. Idempotent; only fires once."""
    try:
        import torch
    except ImportError:
        return  # no torch yet — nothing to hook into

    if hasattr(torch.cuda, "_cipher_shim_installed"):
        return
    torch.cuda._cipher_shim_installed = True

    _orig_sync = torch.cuda.synchronize

    def _hook_sync(*args, **kwargs):
        # Restore original synchronize first so we don't recurse
        torch.cuda.synchronize = _orig_sync
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            import cipher_python
            cipher_python.apply_shims()
        except Exception as ex:
            print(f"[CIPHER autoload] shim apply failed: {ex}", file=sys.stderr)
        return _orig_sync(*args, **kwargs)

    torch.cuda.synchronize = _hook_sync


# Wait for torch import — install the hook lazily by registering an
# import callback. Python doesn't have a clean import callback, so
# we use a sys.modules monitoring approach: re-check periodically.
# For simplicity, just check once at startup (most customer scripts
# import torch first).
try:
    import torch  # may fail if torch not yet imported
    _install_first_cuda_op_hook()
except ImportError:
    # torch not yet available; fall back to lazy install via the
    # __import__ wrapper.
    _orig_import = __builtins__.__import__ if isinstance(__builtins__, dict) is False else __builtins__["__import__"]
    def _cipher_import(name, *args, **kwargs):
        m = _orig_import(name, *args, **kwargs)
        if name == "torch" or name.startswith("torch."):
            try:
                _install_first_cuda_op_hook()
            except Exception:
                pass
        return m
    if isinstance(__builtins__, dict):
        __builtins__["__import__"] = _cipher_import
    else:
        __builtins__.__import__ = _cipher_import
