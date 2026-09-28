"""Phase 4.0.9 — verify a workload run was valid.

Usage:
    python3 verify_run.py <WL_id> <tenant_id> [<log_path>]

Returns exit code 0 if valid, non-zero otherwise. Prints
'valid' or 'invalid: <reason>' to stdout.
"""
import os
import sys
import re


def main():
    if len(sys.argv) < 3:
        print("usage: verify_run.py WL_id tenant_id [log_path]", file=sys.stderr)
        return 2

    wl_id = sys.argv[1]
    tenant_id = sys.argv[2]
    log_path = sys.argv[3] if len(sys.argv) > 3 else \
        f"/tmp/cipher_wl_{wl_id}_{tenant_id}.log"
    progress_path = f"/tmp/cipher_tenant_{tenant_id}_progress"

    if not os.path.exists(progress_path):
        print(f"invalid: missing progress file {progress_path}")
        return 1

    try:
        with open(progress_path) as f:
            progress = f.read()
    except Exception as e:
        print(f"invalid: cannot read progress file: {e}")
        return 1

    if "start" not in progress:
        print(f"invalid: progress file has no start marker")
        return 1
    if "end" not in progress and "deferred-multi-gpu" not in progress:
        print(f"invalid: progress file has no end/deferred marker (workload crashed?)")
        return 1

    # Driver-specific: WL18 deferred is acceptable.
    if "deferred-multi-gpu" in progress:
        print(f"valid: {wl_id} {tenant_id} deferred-multi-gpu")
        return 0

    # Check log for Python tracebacks (real exceptions, not benign warnings).
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                log = f.read()
            if re.search(r"^Traceback \(most recent call last\):", log, re.M):
                if "EXCEPTION" not in progress:
                    print(f"invalid: log has Python traceback")
                    return 1
        except Exception:
            pass

    if "EXCEPTION" in progress:
        m = re.search(r"EXCEPTION (\S+):", progress)
        kind = m.group(1) if m else "unknown"
        print(f"invalid: workload raised {kind}")
        return 1

    # Extract iter count + tokens for the final entry.
    m = re.search(r"end .*?iters=(\d+).*?tokens=(\d+)", progress)
    if not m:
        print(f"valid: {wl_id} {tenant_id} (no iter summary)")
        return 0

    iters = int(m.group(1))
    tokens = int(m.group(2))
    if iters == 0:
        print(f"invalid: zero iterations completed")
        return 1

    print(f"valid: {wl_id} {tenant_id} iters={iters} tokens={tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
