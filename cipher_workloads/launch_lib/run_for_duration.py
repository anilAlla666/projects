"""Shared timing + sampling loop for workload drivers.

Each driver passes a one-shot callable plus a duration. This module:
  - opens /tmp/cipher_tenant_<id>_progress for progress logging
  - runs the callable in a loop until duration elapsed
  - writes a progress line at >=1Hz with iteration count + tokens
  - handles SIGTERM cleanly (do_exit reaper sweeps cipher_kmod entry)
"""
import os
import sys
import time
import signal


_stop = False


def _sigterm(sig, frame):
    global _stop
    _stop = True
    print(f"[run_for_duration] SIGTERM received, stopping cleanly",
          file=sys.stderr, flush=True)


def run(workload_fn, duration_s, wl_id, tenant_id, unit_label="ops"):
    """workload_fn: callable returning (ops_completed_this_call:int, tokens_this_call:int)
       duration_s: seconds to run
       wl_id: e.g. "WL01"
       tenant_id: from env CIPHER_TENANT_ID
       unit_label: "tokens" / "images" / "iters" etc."""

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    progress_path = f"/tmp/cipher_tenant_{tenant_id}_progress"
    fp = open(progress_path, "w", buffering=1)  # line-buffered

    iters = 0
    total_units = 0
    total_tokens = 0
    t0 = time.time()
    last_progress = t0

    fp.write(f"{wl_id} {tenant_id} start t={t0:.3f}\n")
    print(f"[{wl_id}] tenant={tenant_id} start; duration={duration_s}s",
          file=sys.stderr, flush=True)

    try:
        while not _stop and (time.time() - t0) < duration_s:
            ops, tokens = workload_fn()
            iters += 1
            total_units += ops
            total_tokens += tokens

            now = time.time()
            if now - last_progress >= 1.0:
                elapsed = now - t0
                rate = total_units / max(elapsed, 1e-6)
                tok_rate = total_tokens / max(elapsed, 1e-6)
                fp.write(f"{wl_id} {tenant_id} t={elapsed:.2f}s "
                         f"iters={iters} {unit_label}={total_units} "
                         f"{unit_label}/s={rate:.1f} "
                         f"tokens={total_tokens} tok/s={tok_rate:.1f}\n")
                last_progress = now
    except Exception as e:
        fp.write(f"{wl_id} {tenant_id} EXCEPTION {type(e).__name__}: {e}\n")
        print(f"[{wl_id}] EXCEPTION: {e}", file=sys.stderr, flush=True)
        raise
    finally:
        elapsed = time.time() - t0
        fp.write(f"{wl_id} {tenant_id} end t={elapsed:.2f}s iters={iters} "
                 f"{unit_label}={total_units} tokens={total_tokens}\n")
        fp.close()
        print(f"[{wl_id}] tenant={tenant_id} end; iters={iters} "
              f"{unit_label}={total_units} tokens={total_tokens} elapsed={elapsed:.1f}s",
              file=sys.stderr, flush=True)
