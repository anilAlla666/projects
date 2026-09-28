#!/usr/bin/env python3
# cipher-platform CDI patch tool (M1 + P1 mechanism).
#
# Adds CIPHER substrate + plugin chain bind-mounts and env vars to the top-level
# containerEdits of /var/run/cdi/nvidia.yaml so that vanilla
# `docker run --gpus all` invocations transparently load the CIPHER substrate
# via nvidia-container-toolkit's CDI injection. End-user container is
# unmodified (Goal 5 contract verbatim for the end-user).
#
# Modes: apply | revert | status
# Exit codes: 0 ok | 1 not-applied (status) / general error | 2 parse error
#
# Idempotent: apply twice is a no-op (set-union); revert twice is a no-op.
# Atomic: writes to .tmp + rename. Refuses on missing or malformed nvidia.yaml.
import argparse
import os
import shutil
import sys
import tempfile
import yaml

CDI_PATH = "/var/run/cdi/nvidia.yaml"

# CIPHER substrate libs (host -> container, ro bind-mount).
CIPHER_LIB_MOUNTS = [
    "/usr/lib/cipher/libcipher_rt.so",
    "/usr/lib/cipher/libcipher_v2.so",
    "/usr/lib/cipher/libc10.so",
]

# CIPHER Python plugin chain (host -> container, ro bind-mount).
# Container path matches host path: standard Ubuntu /usr/lib/python3/dist-packages
# is on vanilla vllm/vllm-openai container Python's sys.path. The cipher_kv_bridge
# C-extension ships in two ABI flavors (py310 + py312) so the same .deb works
# across vllm-openai images built on different Python minor versions; v1.x adds
# py313 when vllm-openai images migrate (see PLUGIN_MANIFEST.md).
CIPHER_PLUGIN_FILES = [
    "/usr/lib/python3/dist-packages/cipher_vllm_kv.py",
    "/usr/lib/python3/dist-packages/cipher_vllm_kvdedup.py",
    "/usr/lib/python3/dist-packages/cipher_kv_offload.py",
    "/usr/lib/python3/dist-packages/cipher_model_fingerprint.py",
    "/usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so",
    "/usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so",
]

# F-B.3.1 + F-B.3.3 remediation: dynamic discovery of libcipher_rt.so's
# CUDA-toolkit-class DT_NEEDED dependencies and bind-mount from host.
#
# Mechanism: at apply-time, readelf -d /usr/lib/cipher/libcipher_rt.so reveals
# the DT_NEEDED chain. Entries matching CIPHER_CUDA_CLASS_PATTERNS (CUDA
# toolkit libraries) are searched on host via CIPHER_CUDA_HOST_SEARCH_DIRS and
# bind-mounted at /usr/lib/cipher/<soname> inside the container. Substrate-
# owned libs (libc10, libcipher_rt, libcipher_v2) and driver-provided libs
# (libcuda.so.*) and glibc-class libs are explicitly skipped.
#
# v1 takes a runtime dependency on the host having CUDA 12 toolkit installed
# (typical NVIDIA Ubuntu pkg names: nvidia-cuda-toolkit-12, or NVIDIA's
# standalone CUDA installer). v1.5 will rebuild libcipher_rt.so with
# dlopen-on-demand resolution for the full CUDA toolkit to eliminate
# host-CUDA-toolkit-version coupling. Recorded in V1 Phase B scope-lock
# addendum B.1'' §15 honest residue.
SUBSTRATE_PATH = "/usr/lib/cipher/libcipher_rt.so"

# CUDA-toolkit-class SO names that should be bind-mounted from host into
# /usr/lib/cipher/ if they appear in libcipher_rt.so's DT_NEEDED chain.
# Prefix-match: any SO starting with one of these is CUDA-class.
CIPHER_CUDA_CLASS_PREFIXES = [
    "libcudart.so",
    "libcusolver.so",
    "libcusparse.so",
    "libcublas.so",
    "libcublasLt.so",
    "libnvJitLink.so",
    "libnvrtc.so",
    "libcurand.so",
    "libcufft.so",
    "libcupti.so",
]

# Host directories searched for CUDA-class libs (in priority order).
CIPHER_CUDA_HOST_SEARCH_DIRS = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-12/lib64",
    "/opt/cuda/lib64",
    "/opt/cuda-12/lib64",
]

# DT_NEEDED entries that the CIPHER .deb itself ships (don't double-mount).
CIPHER_SUBSTRATE_OWNED_SONAMES = {
    "libc10.so",
    "libcipher_rt.so",
    "libcipher_v2.so",
}

# DT_NEEDED entries provided by --gpus all + glibc + container baseline; don't
# bind-mount or we'd corrupt the container's own toolchain.
CIPHER_SKIP_SONAMES = {
    "libcuda.so.1",
    "libcuda.so",
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "libpthread.so.0",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "librt.so.1",
    "libutil.so.1",
    "libcrypto.so.3",
    "libz.so.1",
    "ld-linux-x86-64.so.2",
}


def _readelf_dt_needed(libpath):
    """Return list of DT_NEEDED SO names from readelf -d output.

    Test override: CIPHER_TEST_DT_NEEDED env var lets the test driver inject
    a synthetic newline-separated list of NEEDED entries.
    """
    override = os.environ.get("CIPHER_TEST_DT_NEEDED")
    if override is not None:
        return [s.strip() for s in override.splitlines() if s.strip()]
    import subprocess
    try:
        out = subprocess.run(
            ["readelf", "-d", libpath],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    needed = []
    for line in out.splitlines():
        # Lines look like: " 0x0000000000000001 (NEEDED)   Shared library: [libfoo.so.X]"
        if "(NEEDED)" in line and "Shared library:" in line:
            lo = line.find("[")
            hi = line.find("]", lo)
            if 0 <= lo < hi:
                needed.append(line[lo + 1:hi])
    return needed


def _is_cuda_class_soname(soname):
    """True if soname is a CUDA-toolkit-class lib that CIPHER should bind-mount."""
    for prefix in CIPHER_CUDA_CLASS_PREFIXES:
        if soname.startswith(prefix):
            return True
    return False


def _find_host_lib(soname):
    """Locate a host shared library by SO name; return path or None.

    Test override: CIPHER_TEST_CUDA_SEARCH_DIRS env var (colon-separated)
    replaces the default host search directories. Empty value simulates no
    CUDA toolkit on host.
    """
    override = os.environ.get("CIPHER_TEST_CUDA_SEARCH_DIRS")
    if override is not None:
        search_dirs = [d for d in override.split(":") if d]
    else:
        search_dirs = list(CIPHER_CUDA_HOST_SEARCH_DIRS)
        import glob
        for pat in ("/usr/local/cuda-12.*/lib*", "/opt/cuda-12.*/lib*"):
            search_dirs.extend(sorted(glob.glob(pat)))
    for d in search_dirs:
        cand = os.path.join(d, soname)
        if os.path.exists(cand):
            return cand
    return None


def _discover_cuda_bind_mounts(libpath=SUBSTRATE_PATH):
    """Return (found_mounts, missing_sonames) tuple via BFS over DT_NEEDED chain.

    Walks libcipher_rt.so's DT_NEEDED chain transitively, since CUDA-toolkit
    libs have their own CUDA-class transitive deps (e.g., libcusolver.so.11
    -> libnvJitLink/libcusparse/libcublas/libcublasLt). Each CUDA-class entry
    is bind-mounted from host; sub-deps of FOUND libs feed back into the
    queue. Sub-deps of NOT-FOUND libs cannot be walked (we have no host file
    to readelf); they surface only as the missing soname.

    found_mounts: list of (host_path, container_path) tuples in BFS order.
    missing_sonames: list of CUDA-class SO names not findable on host.
    """
    seen = set()
    queue = []
    found_pairs = []
    missing = []

    queue.extend(_readelf_dt_needed(libpath))
    while queue:
        soname = queue.pop(0)
        if soname in seen:
            continue
        seen.add(soname)
        if soname in CIPHER_SUBSTRATE_OWNED_SONAMES:
            continue
        if soname in CIPHER_SKIP_SONAMES:
            continue
        if not _is_cuda_class_soname(soname):
            continue
        host = _find_host_lib(soname)
        container = f"/usr/lib/cipher/{soname}"
        if host is not None:
            found_pairs.append((host, container))
            # Walk transitively into this lib's DT_NEEDED to catch CUDA-class
            # sub-deps (e.g., libcusolver -> libnvJitLink + libcusparse +
            # libcublas + libcublasLt).
            queue.extend(_readelf_dt_needed(host))
        else:
            missing.append(soname)
    return found_pairs, missing

# CIPHER dist-info (entry-points discovery).
CIPHER_DIST_INFO_FILES = [
    "/usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/METADATA",
    "/usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/WHEEL",
    "/usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/entry_points.txt",
    "/usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/top_level.txt",
]

# CIPHER env entries (name=value). CIPHER_CDI_MARKER is the revert anchor.
#
# B.6''.5 additions (CUDA 13 substrate v1-substrate-cuda13-cublaslt-coverage):
#   LD_LIBRARY_PATH extended to include the container's bundled CUDA 13 lib dir
#     (/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib), where vLLM
#     ships libcudart.so.13 + libcublas.so.13 + libcupti.so.13 + others.
#     Substrate's CUDA 13 DT_NEEDED chain resolves from there; eliminates the
#     R4-era host CUDA bind-mount requirement under v1.
#   TORCH_CUBLASLT_DISABLE=1 forces torch fp32 path through cublasGemmEx
#     (CIPHER substrate-intercepted) instead of cublasLtMatmulAlgoGetHeuristic-
#     returned fn-pointer path (F-B.3.6.2 finding: that path bypasses GOT-patch
#     + dlsym hook + variant trampolines). LLM workloads (vLLM bf16/fp16) are
#     already on cublasGemmEx; fp32 perf trade is bounded to non-LLM shapes.
#     v2 driver-API-level interception eliminates this env requirement.
#
# B.6''.9.3 addition (Path β.3, 2026-05-26):
#   CIPHER_REGISTER_MODEL=0 gates the cipher_vllm_kv plugin's NR 27 ioctl
#     (CIPHER_REGISTER_MODEL) off. F-VLLM-PLUGIN-CRASH bisect (B.6''.9.1) root-
#     caused vLLM v0.21.0 EngineCore segfault to the NR 27 downstream code
#     path; ASan rebuild (B.6''.9.3) ran clean, narrowing the bug to a kmod
#     /mmap-region interaction outside userspace ASan coverage. Audit confirms
#     NR 27 model_uuid has ZERO v1 consumers: SC6 weight-sharing uses NRs
#     21-24 (ARENA) only, cipher_kv_bridge/libcipher_rt have no NR 27 refs,
#     and the cipher_model_uuid() Python accessor has no readers anywhere.
#     Gating off has no v1 customer-observable impact. v1.x picks up the
#     KASAN root-cause fix on the kmod NR 27 path + re-enable.
CIPHER_ENV = [
    "CIPHER_CDI_MARKER=v1",
    "CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so",
    "LD_LIBRARY_PATH=/usr/lib/cipher:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib",
    "TORCH_CUBLASLT_DISABLE=1",
    "CIPHER_REGISTER_MODEL=0",
    # W.6 sub-A (rev7, 2026-05-27): actuator auto-engagement defaults per
    # Memory #25 product-engagement-gate + Anil adjudication 2026-05-27.
    #
    # All six are safe-to-auto-engage on the Memory #25 reference workload
    # set (Llama-3-8B bf16 default, Mistral-7B-Instruct bf16 default,
    # TinyLlama-AWQ INT4 default):
    #   - CIPHER_MARLIN=on: dtype gate (cipher_rt_marlin_actuator.c:120)
    #     rejects bf16; Machete path bypasses cuBLAS for AWQ INT4. In both
    #     cases the actuator skips cleanly with no perf regression.
    #   - CIPHER_KOOPMAN=1: dtype gate (cipher_rt_koopman_engine.cpp:109)
    #     rejects bf16; empty shape registry rejects until calibrated.
    #   - CIPHER_KVDEDUP=1: xxhash64 + memcmp verify-byte correctness gate
    #     ensures dedup is bit-exact (cipher_rt_kv_alloc.c:562-616).
    #   - CIPHER_REMEMBER=1: observability-only consumer of RING_WRITE
    #     slot 3 (Koopman events).
    #   - CIPHER_AUDIT=1: HMAC-SHA256 telemetry; no behavior change.
    #   - CIPHER_SENSE=on: workload session detector; observation-only +
    #     RECEIPT prerequisite for future v1.x billing surface.
    #
    # INTENTIONALLY ABSENT (Anil adjudication 2026-05-27):
    #   - CIPHER_VOLT: regresses 7B+ bf16 decode by ~14% per Memory
    #     cipher-t43-envelope. Classifier-driven engagement lands at W.1
    #     (Memory #29 sequence).
    #   - CIPHER_RECEIPT / CIPHER_FAIRNESS / CIPHER_CARBON: no v1
    #     operator-facing surface (Grafana dashboard / API). Deferred
    #     per Memory #16 backfill discipline.
    #
    # Customer override path (negative case): customer can disable any
    # actuator via `-e CIPHER_KOOPMAN=0` etc on the container spawn line.
    # Docker container env overrides CDI hook env per Docker semantics.
    "CIPHER_MARLIN=on",
    "CIPHER_KOOPMAN=1",
    "CIPHER_KVDEDUP=1",
    "CIPHER_REMEMBER=1",
    "CIPHER_AUDIT=1",
    "CIPHER_SENSE=on",
]

# F-B.3.5 fold: /dev/cipher CDI deviceNode. Substrate falls back to hash-pick
# green ctx + CUPTI no-fill without it; cipher-platform verify surfaces gap.
CIPHER_DEVICE_NODES = [
    {"path": "/dev/cipher"},
]

MOUNT_OPTIONS = ["ro", "nosuid", "nodev", "rbind", "rprivate"]


def _cuda_class_container_path(soname):
    """Container-side rendezvous path for a CUDA-class lib."""
    return f"/usr/lib/cipher/{soname}"


def _is_cipher_container_path(container_path):
    """True if container_path is owned by CIPHER (any of the static + dynamic sets).

    Recognition by containerPath rather than hostPath: substrate libs + plugin
    files use hostPath == containerPath; CUDA-class libs have dynamic hostPath
    but their containerPath always lives under /usr/lib/cipher/, which is the
    distinguishing prefix for revert recognition.
    """
    if container_path in CIPHER_LIB_MOUNTS:
        return True
    if container_path in CIPHER_PLUGIN_FILES:
        return True
    if container_path in CIPHER_DIST_INFO_FILES:
        return True
    # CUDA-class libs land under /usr/lib/cipher/<soname>; the prefix +
    # CUDA-class soname check together identify them on revert without
    # requiring a re-scan of the host (host arrangement can drift between
    # apply and revert).
    if container_path.startswith("/usr/lib/cipher/"):
        soname = os.path.basename(container_path)
        if soname in {"libcipher_rt.so", "libcipher_v2.so", "libc10.so"}:
            return True  # already covered above, but explicit
        if _is_cuda_class_soname(soname):
            return True
    return False


def _mount_entry(host, container=None):
    if container is None:
        container = host
    return {
        "hostPath": host,
        "containerPath": container,
        "options": list(MOUNT_OPTIONS),
    }


def _is_cipher_mount(m):
    if not isinstance(m, dict):
        return False
    return _is_cipher_container_path(m.get("containerPath", ""))


def _is_cipher_env(e):
    if not isinstance(e, str):
        return False
    cipher_names = {entry.split("=", 1)[0] for entry in CIPHER_ENV}
    return e.split("=", 1)[0] in cipher_names


def _is_cipher_device_node(d):
    """True if device node entry is owned by CIPHER. Match by path."""
    if not isinstance(d, dict):
        return False
    cipher_paths = {dn["path"] for dn in CIPHER_DEVICE_NODES}
    return d.get("path", "") in cipher_paths


def _load(cdi_path):
    if not os.path.exists(cdi_path):
        raise FileNotFoundError(
            f"cipher_cdi_patch: {cdi_path} not found. "
            "Is nvidia-container-toolkit configured? Run "
            "`sudo nvidia-ctk cdi generate --output={p}`.".format(p=cdi_path)
        )
    try:
        with open(cdi_path) as f:
            spec = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"cipher_cdi_patch: {cdi_path} is malformed YAML: {e}")
    if not isinstance(spec, dict):
        raise ValueError(f"cipher_cdi_patch: {cdi_path} top-level is not a mapping.")
    if spec.get("kind") != "nvidia.com/gpu":
        raise ValueError(
            f"cipher_cdi_patch: {cdi_path} kind is not nvidia.com/gpu "
            f"(got {spec.get('kind')!r})."
        )
    return spec


def _atomic_write(cdi_path, spec):
    dir_ = os.path.dirname(cdi_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".cipher_cdi_patch.", dir=dir_)
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(spec, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, cdi_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _required_container_paths(found_cuda_mounts):
    """Container paths that MUST be present in a fully-applied patch.

    CUDA-class libs are host-discovered; if a lib is absent on host, the patch
    ships without it (soft warning surfaced in stderr + missing_host_libs in
    status). fully_applied accounts for this by including only host-resolvable
    CUDA libs in the required set; truly-missing libs are flagged separately.
    """
    required = list(CIPHER_LIB_MOUNTS) + list(CIPHER_PLUGIN_FILES) + list(CIPHER_DIST_INFO_FILES)
    for _, container_path in found_cuda_mounts:
        required.append(container_path)
    return required


def _status_dict(cdi_path):
    spec = _load(cdi_path)
    ce = spec.setdefault("containerEdits", {})
    env = ce.get("env", []) or []
    mounts = ce.get("mounts", []) or []
    device_nodes = ce.get("deviceNodes", []) or []
    applied_env = [e for e in env if _is_cipher_env(e)]
    applied_mounts = [m for m in mounts if _is_cipher_mount(m)]
    applied_device_nodes = [d for d in device_nodes if _is_cipher_device_node(d)]
    found_cuda, missing_host_libs = _discover_cuda_bind_mounts()
    expected_container_paths = set(_required_container_paths(found_cuda))
    expected_env_names = {e.split("=", 1)[0] for e in CIPHER_ENV}
    expected_device_paths = {dn["path"] for dn in CIPHER_DEVICE_NODES}
    have_container_paths = {m["containerPath"] for m in applied_mounts}
    have_env_names = {e.split("=", 1)[0] for e in applied_env}
    have_device_paths = {d.get("path") for d in applied_device_nodes}
    return {
        "cdi_path": cdi_path,
        "marker_present": "CIPHER_CDI_MARKER=v1" in env,
        "applied_env_count": len(applied_env),
        "applied_mount_count": len(applied_mounts),
        "applied_device_node_count": len(applied_device_nodes),
        "expected_env_count": len(CIPHER_ENV),
        "expected_mount_count": len(expected_container_paths),
        "expected_device_node_count": len(CIPHER_DEVICE_NODES),
        "found_cuda_libs": sorted(c for _, c in found_cuda),
        "missing_host_libs": sorted(missing_host_libs),
        "missing_envs": sorted(expected_env_names - have_env_names),
        "missing_mounts": sorted(expected_container_paths - have_container_paths),
        "missing_device_nodes": sorted(expected_device_paths - have_device_paths),
        "fully_applied": (
            "CIPHER_CDI_MARKER=v1" in env
            and not (expected_env_names - have_env_names)
            and not (expected_container_paths - have_container_paths)
            and not (expected_device_paths - have_device_paths)
        ),
    }


def apply_patch(cdi_path):
    spec = _load(cdi_path)
    ce = spec.setdefault("containerEdits", {})
    env = ce.setdefault("env", []) or []
    if env is None:
        env = []
        ce["env"] = env
    mounts = ce.setdefault("mounts", []) or []
    if mounts is None:
        mounts = []
        ce["mounts"] = mounts
    device_nodes = ce.setdefault("deviceNodes", []) or []
    if device_nodes is None:
        device_nodes = []
        ce["deviceNodes"] = device_nodes

    have_env_names = {e.split("=", 1)[0] for e in env if isinstance(e, str)}
    changed = False
    for entry in CIPHER_ENV:
        if entry.split("=", 1)[0] not in have_env_names:
            env.append(entry)
            have_env_names.add(entry.split("=", 1)[0])
            changed = True

    have_container_paths = {
        m.get("containerPath") for m in mounts if isinstance(m, dict)
    }
    # Substrate + plugin + dist-info: hostPath == containerPath.
    for path in CIPHER_LIB_MOUNTS + CIPHER_PLUGIN_FILES + CIPHER_DIST_INFO_FILES:
        if path not in have_container_paths:
            mounts.append(_mount_entry(path))
            have_container_paths.add(path)
            changed = True

    # CUDA-class libs: hostPath is host-discovered, containerPath is fixed at
    # /usr/lib/cipher/<soname>. DT_NEEDED entries are discovered dynamically
    # from libcipher_rt.so via readelf. Libs not findable on host trigger a
    # soft WARNING (substrate may fail to load inside containers; surfaced via
    # cipher-platform verify + status missing_host_libs). v1 takes a runtime
    # dependency on host CUDA 12 toolkit; v1.5 rebuilds substrate with
    # dlopen-on-demand resolution to eliminate this coupling.
    found_cuda, missing_host_libs = _discover_cuda_bind_mounts()
    for host_path, container_path in found_cuda:
        if container_path not in have_container_paths:
            mounts.append(_mount_entry(host_path, container_path))
            have_container_paths.add(container_path)
            changed = True
    for soname in missing_host_libs:
        sys.stderr.write(
            f"cipher_cdi_patch: WARNING - host {soname} not found on any of "
            f"{CIPHER_CUDA_HOST_SEARCH_DIRS}; substrate may fail to load "
            f"inside containers (libcipher_rt.so DT_NEEDED {soname} will be "
            f"unresolved). Install nvidia-cuda-toolkit-12 (Ubuntu) or NVIDIA "
            f"standalone CUDA 12 toolkit. v1.5 substrate rebuild removes this "
            f"dependency.\n"
        )

    # F-B.3.5 fold: /dev/cipher CDI deviceNode. cipher_kmod 0.6.5 creates
    # /dev/cipher on host (mode 0666 via devnode callback); CDI exposes it
    # inside container so substrate's GREEN/CP54 + tenant + CUPTI components
    # can open it. Without this entry, substrate falls back to hash-pick
    # green ctx + CUPTI no-fill (graceful degradation; not a fatal error).
    have_device_paths = {d.get("path") for d in device_nodes if isinstance(d, dict)}
    for dn in CIPHER_DEVICE_NODES:
        if dn["path"] not in have_device_paths:
            device_nodes.append(dict(dn))
            have_device_paths.add(dn["path"])
            changed = True

    # No-write-when-unchanged: avoids systemd path-watcher feedback loop.
    # Watcher fires on PathChanged -> service runs apply -> if file is
    # rewritten with identical content, inotify IN_MODIFY fires again
    # regardless of content equality, triggering the watcher again.
    if changed:
        ce["env"] = env
        ce["mounts"] = mounts
        ce["deviceNodes"] = device_nodes
        spec["containerEdits"] = ce
        _atomic_write(cdi_path, spec)
    return _status_dict(cdi_path)


def revert_patch(cdi_path):
    spec = _load(cdi_path)
    ce = spec.setdefault("containerEdits", {})
    env = ce.get("env", []) or []
    mounts = ce.get("mounts", []) or []
    device_nodes = ce.get("deviceNodes", []) or []
    new_env = [e for e in env if not _is_cipher_env(e)]
    new_mounts = [m for m in mounts if not _is_cipher_mount(m)]
    new_device_nodes = [d for d in device_nodes if not _is_cipher_device_node(d)]
    if new_env != env or new_mounts != mounts or new_device_nodes != device_nodes:
        ce["env"] = new_env
        ce["mounts"] = new_mounts
        ce["deviceNodes"] = new_device_nodes
        spec["containerEdits"] = ce
        _atomic_write(cdi_path, spec)
    return _status_dict(cdi_path)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="cipher_cdi_patch",
        description="Apply / revert / status CIPHER CDI patch on nvidia.yaml.",
    )
    p.add_argument("mode", choices=["apply", "revert", "status"])
    p.add_argument(
        "--cdi-path",
        default=CDI_PATH,
        help=f"CDI spec path (default: {CDI_PATH}).",
    )
    p.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = p.parse_args(argv)

    try:
        if args.mode == "apply":
            st = apply_patch(args.cdi_path)
        elif args.mode == "revert":
            st = revert_patch(args.cdi_path)
        else:
            st = _status_dict(args.cdi_path)
    except FileNotFoundError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    except ValueError as e:
        sys.stderr.write(f"{e}\n")
        return 2

    if args.json:
        import json
        sys.stdout.write(json.dumps(st, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(
            f"cipher_cdi_patch [{args.mode}] {st['cdi_path']}\n"
            f"  marker_present: {st['marker_present']}\n"
            f"  env: {st['applied_env_count']}/{st['expected_env_count']}\n"
            f"  mounts: {st['applied_mount_count']}/{st['expected_mount_count']}\n"
            f"  found_cuda_libs: {len(st['found_cuda_libs'])}\n"
            f"  missing_host_libs: {st['missing_host_libs']}\n"
            f"  fully_applied: {st['fully_applied']}\n"
        )
        if st["missing_envs"]:
            sys.stdout.write(f"  missing_envs: {st['missing_envs']}\n")
        if st["missing_mounts"]:
            sys.stdout.write(f"  missing_mounts: {st['missing_mounts']}\n")

    if args.mode == "status":
        return 0 if st["fully_applied"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
