#!/usr/bin/env python3
# cipher-platform - operator-facing CLI for the CIPHER GPU efficiency platform.
#
# Five subcommands per V1 Phase B scope-lock addendum B.1'' §7 sub-step 5:
#   status      one-shot state snapshot (substrate, kmod, CDI, plugin, watcher)
#   verify      minimal health check (exit 0 PASS, non-zero with failure list)
#   refresh     re-apply CDI patch via /usr/lib/cipher/cipher_cdi_patch.sh
#   version     .deb version, libcipher_rt.so md5, kmod ABI version, plugin version
#   logs        journalctl -u cipher-platform-watch.service tail
#
# --json flag for machine-readable output (operator-tooling-friendly).
# All subcommands work independently of each other (status callable pre-refresh).
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

CIPHER_LIB_DIR = "/usr/lib/cipher"
SUBSTRATE_LIB = os.path.join(CIPHER_LIB_DIR, "libcipher_rt.so")
CDI_PATCH_SH = os.path.join(CIPHER_LIB_DIR, "cipher_cdi_patch.sh")
CDI_PATH = "/var/run/cdi/nvidia.yaml"
WATCHER_UNIT = "cipher-platform-watch.path"


def _md5_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd, check=False):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _kmod_loaded():
    p = _run(["lsmod"])
    for line in p.stdout.splitlines():
        if line.startswith("cipher_kmod"):
            return True, line.split()[0]
    return False, None


def _kmod_version():
    p = _run(["modinfo", "cipher_kmod"])
    for line in p.stdout.splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip()
    return None


def _kmod_srcversion():
    p = _run(["modinfo", "cipher_kmod"])
    for line in p.stdout.splitlines():
        if line.startswith("srcversion:"):
            return line.split(":", 1)[1].strip()
    return None


def _dpkg_version():
    p = _run(["dpkg-query", "-W", "-f=${Version}", "cipher-platform"])
    return p.stdout.strip() if p.returncode == 0 else None


def _plugin_entry_points():
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            vllm = eps.select(group="vllm.general_plugins")
        else:
            vllm = eps.get("vllm.general_plugins", [])
        return sorted(
            [ep.name for ep in vllm if ep.name.startswith("cipher_")]
        )
    except Exception:
        return []


def _cdi_status():
    if not os.path.exists(CDI_PATCH_SH):
        return {"available": False, "applied": False, "missing": []}
    if os.path.exists(CDI_PATH) and not os.access(CDI_PATH, os.R_OK):
        return {
            "available": True,
            "applied": "unknown",
            "permission_denied": True,
            "hint": f"{CDI_PATH} is root-readable only; re-run with sudo.",
        }
    p = _run([CDI_PATCH_SH, "status", "--json"])
    try:
        st = json.loads(p.stdout)
        return {
            "available": True,
            "applied": bool(st.get("fully_applied", False)),
            "marker": bool(st.get("marker_present", False)),
            "missing_env": list(st.get("missing_envs", [])),
            "missing_mounts": list(st.get("missing_mounts", [])),
        }
    except (json.JSONDecodeError, ValueError):
        return {"available": True, "applied": False, "error": p.stderr.strip()}


def _watcher_state():
    p = _run(["systemctl", "is-active", WATCHER_UNIT])
    return p.stdout.strip()  # 'active', 'inactive', 'failed', or empty


def _cmd_status(args):
    loaded, _ = _kmod_loaded()
    st = {
        "substrate": {
            "path": SUBSTRATE_LIB,
            "present": os.path.exists(SUBSTRATE_LIB),
            "md5": _md5_file(SUBSTRATE_LIB),
        },
        "kmod": {
            "loaded": loaded,
            "version": _kmod_version(),
            "srcversion": _kmod_srcversion(),
            "dev_node_present": os.path.exists("/dev/cipher"),
        },
        "cdi_patch": _cdi_status(),
        "plugin_entry_points": _plugin_entry_points(),
        "systemd_watcher": _watcher_state(),
        "dpkg_version": _dpkg_version(),
    }
    if args.json:
        sys.stdout.write(json.dumps(st, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write("cipher-platform status\n")
        sys.stdout.write(f"  dpkg version:      {st['dpkg_version']}\n")
        sys.stdout.write(
            f"  substrate present: {st['substrate']['present']}"
            f" (md5 {st['substrate']['md5']})\n"
        )
        sys.stdout.write(
            f"  kmod loaded:       {st['kmod']['loaded']}"
            f" (version {st['kmod']['version']}, /dev/cipher"
            f" {st['kmod']['dev_node_present']})\n"
        )
        sys.stdout.write(
            f"  CDI patch:         applied={st['cdi_patch'].get('applied')}"
            f" marker={st['cdi_patch'].get('marker')}\n"
        )
        if st["cdi_patch"].get("missing_env"):
            sys.stdout.write(f"    missing env: {st['cdi_patch']['missing_env']}\n")
        if st["cdi_patch"].get("missing_mounts"):
            sys.stdout.write(f"    missing mounts: {st['cdi_patch']['missing_mounts']}\n")
        sys.stdout.write(
            f"  plugin entries:    {st['plugin_entry_points']}\n"
        )
        sys.stdout.write(f"  systemd watcher:   {st['systemd_watcher']}\n")
    return 0


def _cmd_verify(args):
    failures = []
    if not os.path.exists(SUBSTRATE_LIB):
        failures.append(f"substrate library not present at {SUBSTRATE_LIB}")
    else:
        # File-sanity checks only on host. The substrate is now CUDA-13-linked
        # (v1-substrate-cuda13-cublaslt-coverage); ctypes.CDLL on a host that
        # lacks the CUDA 13 runtime (e.g. host has CUDA 12 toolkit only) would
        # fail to dlopen via DT_NEEDED libcupti.so.13. Substrate loadability is
        # validated inside the vLLM container via Gate A in B.3''/B.6''.3-7;
        # cipher-platform verify only confirms the .deb shipped the file
        # correctly. Operators verify substrate runtime fires via B.6''.8
        # Gate B + Gate C inside vanilla vLLM container, or by running their
        # vLLM workloads.
        try:
            sz = os.path.getsize(SUBSTRATE_LIB)
            with open(SUBSTRATE_LIB, "rb") as f:
                head = f.read(4)
            if sz < 100000:
                failures.append(
                    f"substrate {SUBSTRATE_LIB} suspiciously small ({sz} bytes; expected >100KB)"
                )
            elif head != b"\x7fELF":
                failures.append(
                    f"substrate {SUBSTRATE_LIB} is not an ELF binary"
                )
        except OSError as e:
            failures.append(f"substrate file probe failed: {e}")
    loaded, _ = _kmod_loaded()
    if not loaded:
        failures.append("cipher_kmod not loaded (run `sudo modprobe cipher_kmod`)")
    if not os.path.exists("/dev/cipher"):
        failures.append("/dev/cipher absent (kmod not loaded?)")
    if not os.path.exists(CDI_PATH):
        failures.append(f"{CDI_PATH} not found (nvidia-container-toolkit configured?)")
    else:
        cdi = _cdi_status()
        if cdi.get("permission_denied"):
            failures.append(
                f"CDI status unreadable: {cdi.get('hint')}; re-run `sudo cipher-platform verify` for a conclusive check."
            )
        elif not cdi.get("applied"):
            failures.append(
                f"CDI patch not applied (run `sudo cipher-platform refresh`); "
                f"missing env={cdi.get('missing_env')} "
                f"missing mounts={cdi.get('missing_mounts')}"
            )
    eps = _plugin_entry_points()
    if not eps:
        failures.append(
            "no cipher_* vllm.general_plugins entry points discoverable; "
            "is cipher-platform .deb installed and python3-vllm Python able "
            "to see /usr/lib/python3/dist-packages?"
        )

    result = {"verify": "PASS" if not failures else "FAIL", "failures": failures}
    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        if not failures:
            sys.stdout.write("cipher-platform verify: PASS\n")
        else:
            sys.stdout.write("cipher-platform verify: FAIL\n")
            for f in failures:
                sys.stdout.write(f"  - {f}\n")
    return 0 if not failures else 1


def _cmd_refresh(args):
    if not os.path.exists(CDI_PATCH_SH):
        sys.stderr.write(
            f"cipher-platform: {CDI_PATCH_SH} not found; "
            "is cipher-platform installed?\n"
        )
        return 1
    before = _cdi_status()
    p = _run([CDI_PATCH_SH, "apply", "--json"])
    after = _cdi_status()
    result = {
        "before": before,
        "after": after,
        "patch_exit": p.returncode,
        "patch_stderr": p.stderr.strip(),
    }
    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(
            f"cipher-platform refresh: before applied={before.get('applied')}"
            f" -> after applied={after.get('applied')}\n"
        )
        if p.stderr:
            sys.stdout.write(f"  patch stderr: {p.stderr.strip()}\n")
    return p.returncode


def _cmd_version(args):
    info = {
        "dpkg_version": _dpkg_version(),
        "substrate_md5": _md5_file(SUBSTRATE_LIB),
        "substrate_path": SUBSTRATE_LIB,
        "kmod_version": _kmod_version(),
        "kmod_srcversion": _kmod_srcversion(),
    }
    if args.json:
        sys.stdout.write(json.dumps(info, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"cipher-platform {info['dpkg_version']}\n")
        sys.stdout.write(
            f"  substrate libcipher_rt.so md5: {info['substrate_md5']}\n"
        )
        sys.stdout.write(
            f"  kmod version: {info['kmod_version']}"
            f" (srcversion {info['kmod_srcversion']})\n"
        )
    return 0


def _cmd_logs(args):
    since = args.since
    cmd = [
        "journalctl",
        "-u", "cipher-platform-watch.service",
        "--since", since,
        "--no-pager",
    ]
    p = subprocess.run(cmd)
    return p.returncode


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="cipher-platform",
        description=(
            "Operator-facing CLI for the CIPHER GPU efficiency platform. "
            "See /usr/lib/cipher/runbook.md for the full operator runbook."
        ),
    )
    p.add_argument("--json", action="store_true", help="Machine-readable output.")
    sub = p.add_subparsers(dest="cmd")

    sp_status = sub.add_parser("status", help="State snapshot.")
    sp_status.set_defaults(func=_cmd_status)
    sp_verify = sub.add_parser("verify", help="Run minimal health check.")
    sp_verify.set_defaults(func=_cmd_verify)
    sp_refresh = sub.add_parser("refresh", help="Re-apply CDI patch.")
    sp_refresh.set_defaults(func=_cmd_refresh)
    sp_version = sub.add_parser("version", help="Show versions and md5s.")
    sp_version.set_defaults(func=_cmd_version)
    sp_logs = sub.add_parser("logs", help="Tail systemd journal for CIPHER.")
    sp_logs.add_argument("--since", default="24 hours ago", help="journalctl --since.")
    sp_logs.set_defaults(func=_cmd_logs)

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
