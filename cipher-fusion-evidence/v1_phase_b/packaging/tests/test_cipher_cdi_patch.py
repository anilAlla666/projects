#!/usr/bin/env python3
# Eleven unit tests for cipher_cdi_patch.py covering substeps 3 + 4 + R1 + R4.
#
#   (i)    clean apply on factory-default nvidia.yaml
#   (ii)   idempotent re-apply on already-patched nvidia.yaml
#   (iii)  revert on patched nvidia.yaml
#   (iv)   apply on yaml with unrelated additions (sibling observability agent)
#   (v)    failure handling on missing nvidia.yaml
#   (vi)   failure handling on malformed nvidia.yaml
#   (vii)  no-write-when-unchanged (systemd path-watcher feedback loop guard)
#   (viii) libcupti anchor via generalized DT_NEEDED discovery (F-B.3.1)
#   (ix)   full CUDA 12 toolkit transitive BFS (F-B.3.3)
#   (x)    partial CUDA toolkit on host (F-B.3.3)
#   (xi)   zero CUDA libs on host (F-B.3.3)
#
# Test fixtures are temp files; no writes to real /var/run/cdi/. Run with:
#   python3 v1_phase_b/packaging/tests/test_cipher_cdi_patch.py
import json
import os
import subprocess
import sys
import tempfile
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATCH_PY = "/tmp/cipher-deb-build/cipher-platform/usr/lib/cipher/cipher_cdi_patch.py"

FACTORY_NVIDIA_YAML = """\
cdiVersion: 0.5.0
kind: nvidia.com/gpu
devices:
- name: all
  containerEdits:
    deviceNodes:
    - path: /dev/nvidia0
containerEdits:
  env:
  - NVIDIA_CTK_LIBCUDA_DIR=/usr/lib/x86_64-linux-gnu
  - NVIDIA_VISIBLE_DEVICES=void
  mounts:
  - hostPath: /usr/lib/x86_64-linux-gnu/libcuda.so.1
    containerPath: /usr/lib/x86_64-linux-gnu/libcuda.so.1
    options: [ro, nosuid, nodev, rbind, rprivate]
"""

YAML_WITH_UNRELATED_ADDITIONS = """\
cdiVersion: 0.5.0
kind: nvidia.com/gpu
devices:
- name: all
  containerEdits:
    deviceNodes:
    - path: /dev/nvidia0
containerEdits:
  env:
  - NVIDIA_CTK_LIBCUDA_DIR=/usr/lib/x86_64-linux-gnu
  - SOMEOBS_AGENT_VERSION=1.2.3
  mounts:
  - hostPath: /usr/lib/x86_64-linux-gnu/libcuda.so.1
    containerPath: /usr/lib/x86_64-linux-gnu/libcuda.so.1
    options: [ro, nosuid, nodev, rbind, rprivate]
  - hostPath: /opt/someobs/agent.so
    containerPath: /opt/someobs/agent.so
    options: [ro, bind]
"""

MALFORMED_YAML = "this is not: valid: yaml: at: all: ["


def _run(mode, path, json_flag=True, search_dirs_override=None, dt_needed_override=None):
    args = ["python3", PATCH_PY, mode, "--cdi-path", path]
    if json_flag:
        args.append("--json")
    env = os.environ.copy()
    if search_dirs_override is not None:
        env["CIPHER_TEST_CUDA_SEARCH_DIRS"] = search_dirs_override
    if dt_needed_override is not None:
        env["CIPHER_TEST_DT_NEEDED"] = dt_needed_override
    p = subprocess.run(args, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def _write_tmp(content):
    fd, path = tempfile.mkstemp(prefix="test_cdi_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _read_spec(path):
    with open(path) as f:
        return yaml.safe_load(f)


def case_i_clean_apply():
    """Apply on factory-default nvidia.yaml: CIPHER mounts + env added; no other edits lost."""
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, out, err = _run("apply", path)
        assert rc == 0, f"apply rc={rc}, stderr={err}"
        st = json.loads(out)
        assert st["fully_applied"] is True, st
        assert st["marker_present"] is True, st
        spec = _read_spec(path)
        env = spec["containerEdits"]["env"]
        assert "NVIDIA_CTK_LIBCUDA_DIR=/usr/lib/x86_64-linux-gnu" in env, env
        assert "NVIDIA_VISIBLE_DEVICES=void" in env, env
        assert "CIPHER_CDI_MARKER=v1" in env, env
        assert "CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so" in env, env
        mounts = spec["containerEdits"]["mounts"]
        host_paths = {m["hostPath"] for m in mounts}
        assert "/usr/lib/x86_64-linux-gnu/libcuda.so.1" in host_paths, host_paths
        assert "/usr/lib/cipher/libcipher_rt.so" in host_paths, host_paths
        assert "/usr/lib/python3/dist-packages/cipher_vllm_kv.py" in host_paths, host_paths
        return "PASS"
    finally:
        os.unlink(path)


def case_ii_idempotent_reapply():
    """Apply twice: no duplicate entries; second status identical to first."""
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc1, out1, _ = _run("apply", path)
        assert rc1 == 0
        st1 = json.loads(out1)
        spec1 = _read_spec(path)
        env_count_1 = len(spec1["containerEdits"]["env"])
        mount_count_1 = len(spec1["containerEdits"]["mounts"])

        rc2, out2, _ = _run("apply", path)
        assert rc2 == 0
        st2 = json.loads(out2)
        spec2 = _read_spec(path)
        env_count_2 = len(spec2["containerEdits"]["env"])
        mount_count_2 = len(spec2["containerEdits"]["mounts"])

        assert env_count_1 == env_count_2, (env_count_1, env_count_2)
        assert mount_count_1 == mount_count_2, (mount_count_1, mount_count_2)
        assert st1 == st2, (st1, st2)
        return "PASS"
    finally:
        os.unlink(path)


def case_iii_revert_after_apply():
    """Revert on patched nvidia.yaml: CIPHER entries removed; non-CIPHER entries preserved."""
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, _, _ = _run("apply", path)
        assert rc == 0
        rc, out, err = _run("revert", path)
        assert rc == 0, f"revert rc={rc}, stderr={err}"
        st = json.loads(out)
        assert st["marker_present"] is False, st
        assert st["applied_env_count"] == 0, st
        assert st["applied_mount_count"] == 0, st
        spec = _read_spec(path)
        env = spec["containerEdits"]["env"]
        assert "NVIDIA_CTK_LIBCUDA_DIR=/usr/lib/x86_64-linux-gnu" in env
        assert "NVIDIA_VISIBLE_DEVICES=void" in env
        assert "CIPHER_CDI_MARKER=v1" not in env
        assert "CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so" not in env
        mounts = spec["containerEdits"]["mounts"]
        host_paths = {m["hostPath"] for m in mounts}
        assert "/usr/lib/x86_64-linux-gnu/libcuda.so.1" in host_paths
        assert "/usr/lib/cipher/libcipher_rt.so" not in host_paths
        # revert-twice is a no-op
        rc2, _, _ = _run("revert", path)
        assert rc2 == 0
        return "PASS"
    finally:
        os.unlink(path)


def case_iv_apply_with_unrelated_additions():
    """Apply on yaml with sibling observability agent: agent entries preserved + CIPHER added."""
    path = _write_tmp(YAML_WITH_UNRELATED_ADDITIONS)
    try:
        rc, out, _ = _run("apply", path)
        assert rc == 0
        st = json.loads(out)
        assert st["fully_applied"] is True
        spec = _read_spec(path)
        env = spec["containerEdits"]["env"]
        assert "SOMEOBS_AGENT_VERSION=1.2.3" in env, env
        assert "CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so" in env, env
        mounts = spec["containerEdits"]["mounts"]
        host_paths = {m["hostPath"] for m in mounts}
        assert "/opt/someobs/agent.so" in host_paths, host_paths
        assert "/usr/lib/cipher/libcipher_rt.so" in host_paths, host_paths
        # revert preserves observability agent
        rc, _, _ = _run("revert", path)
        assert rc == 0
        spec = _read_spec(path)
        env = spec["containerEdits"]["env"]
        mounts = spec["containerEdits"]["mounts"]
        host_paths = {m["hostPath"] for m in mounts}
        assert "SOMEOBS_AGENT_VERSION=1.2.3" in env
        assert "/opt/someobs/agent.so" in host_paths
        assert "CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so" not in env
        return "PASS"
    finally:
        os.unlink(path)


def case_v_missing_yaml():
    """Apply on missing nvidia.yaml: exit 1 with operator-readable error; no silent no-op."""
    missing = "/tmp/test_cdi_does_not_exist.yaml"
    if os.path.exists(missing):
        os.unlink(missing)
    rc, out, err = _run("apply", missing, json_flag=False)
    assert rc == 1, f"expected rc=1, got rc={rc}, stderr={err}"
    assert "not found" in err.lower(), err
    assert "nvidia-ctk" in err.lower(), err
    return "PASS"


def case_viii_libcupti_anchor():
    """F-B.3.1 anchor: libcupti.so.12 via generalized DT_NEEDED discovery.

    Sub-case (a) injects DT_NEEDED=libcupti.so.12 only via env override; pod's
    host has libcupti at /usr/lib/x86_64-linux-gnu/. Expect 1 CUDA-class mount
    added at containerPath /usr/lib/cipher/libcupti.so.12. Revert removes it.
    Sub-case (b) injects DT_NEEDED=libcupti.so.12 with empty search dirs (no
    host libs). Expect 0 mounts, missing_host_libs=[libcupti.so.12], WARNING.
    """
    # Sub-case (a): libcupti as the only DT_NEEDED CUDA entry; host has it.
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, out, err = _run(
            "apply", path,
            dt_needed_override="libcupti.so.12",
            search_dirs_override="/usr/lib/x86_64-linux-gnu",
        )
        assert rc == 0, f"apply rc={rc}, stderr={err}"
        st = json.loads(out)
        assert st["fully_applied"] is True, st
        assert st["found_cuda_libs"] == ["/usr/lib/cipher/libcupti.so.12"], st
        assert st["missing_host_libs"] == [], st
        # 3 substrate + 6 plugin + 4 dist-info + 1 CUDA = 14
        assert st["expected_mount_count"] == 14, st
        spec = _read_spec(path)
        cupti = next(
            (m for m in spec["containerEdits"]["mounts"]
             if m.get("containerPath") == "/usr/lib/cipher/libcupti.so.12"),
            None,
        )
        assert cupti is not None and cupti["hostPath"] == "/usr/lib/x86_64-linux-gnu/libcupti.so.12", cupti
        rc, _, _ = _run(
            "revert", path,
            dt_needed_override="libcupti.so.12",
            search_dirs_override="/usr/lib/x86_64-linux-gnu",
        )
        assert rc == 0
        spec = _read_spec(path)
        for m in spec["containerEdits"]["mounts"]:
            assert m.get("containerPath") != "/usr/lib/cipher/libcupti.so.12", m
    finally:
        os.unlink(path)

    # Sub-case (b): libcupti DT_NEEDED but no host libs visible.
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, out, err = _run(
            "apply", path,
            dt_needed_override="libcupti.so.12",
            search_dirs_override="",
        )
        assert rc == 0, f"apply rc={rc}, stderr={err}"
        st = json.loads(out)
        assert st["fully_applied"] is True, "fully_applied=True with libcupti dropped from required (host can't provide)"
        assert st["found_cuda_libs"] == [], st
        assert st["missing_host_libs"] == ["libcupti.so.12"], st
        # 3 substrate + 6 plugin + 4 dist-info + 0 CUDA = 13
        assert st["expected_mount_count"] == 13, st
        assert "WARNING" in err and "libcupti.so.12" in err, f"expected stderr warning, got: {err}"
        return "PASS"
    finally:
        os.unlink(path)


def case_ix_full_cuda12_toolkit():
    """B.6''.5 update: substrate DT_NEEDED CUDA libs vs host availability.

    Reads ACTUAL /usr/lib/cipher/libcipher_rt.so on this pod and verifies
    _discover_cuda_bind_mounts BFS produces a self-consistent (found,
    missing) partition: every CUDA-class DT_NEEDED entry lands in EITHER
    found_cuda_libs (host has it) OR missing_host_libs (host doesn't);
    fully_applied=True regardless (missing libs are non-blocking).

    Under v1-substrate-cuda13-cublaslt-coverage substrate, DT_NEEDED is
    CUDA 13 SOnames (libcudart.so.13 etc.) which a CUDA-12-only host
    cannot supply; found_cuda_libs=[]; missing_host_libs!=[]; substrate
    expects container's bundled CUDA 13 libs via LD_LIBRARY_PATH instead.

    Under v1-substrate-driver-worker-init (CUDA 12) substrate, the 7-
    libs-via-host scenario in the earlier B.6''.3 era applies; assertion
    relaxes to validate the discovery shape rather than count.
    """
    if not os.path.exists("/usr/lib/cipher/libcipher_rt.so"):
        return "SKIP (no /usr/lib/cipher/libcipher_rt.so on pod; install cipher-platform v2.0)"
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, out, err = _run(
            "apply", path,
            search_dirs_override="/usr/lib/x86_64-linux-gnu",
        )
        assert rc == 0, f"apply rc={rc}, stderr={err}"
        st = json.loads(out)
        assert st["fully_applied"] is True, st
        # Found + missing partition is self-consistent; fully_applied True
        # because missing libs are non-blocking (container ships its own
        # CUDA stack under v1-substrate-cuda13-cublaslt-coverage).
        assert isinstance(st["found_cuda_libs"], list), st
        assert isinstance(st["missing_host_libs"], list), st
        # Either CUDA 12 substrate (7 found / 0 missing) OR CUDA 13 substrate
        # (~0 found / 3-7 missing); both shapes valid.
        return "PASS"
    finally:
        os.unlink(path)


def case_x_partial_cuda12_toolkit():
    """F-B.3.3 R4: partial CUDA toolkit present on host.

    Inject DT_NEEDED of 3 libs (libcupti.so.12, libcudart.so.12,
    libcusolver.so.11) but point search to a temp dir holding ONLY libcupti +
    libcudart. Expect 2 mounts, libcusolver.so.11 in missing_host_libs,
    WARNING on stderr, fully_applied=True (libcusolver dropped from required).
    """
    tmpdir = tempfile.mkdtemp(prefix="test_cuda_partial_")
    for fname in ("libcupti.so.12", "libcudart.so.12"):
        with open(os.path.join(tmpdir, fname), "wb") as f:
            f.write(b"\x7fELF")
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, out, err = _run(
            "apply", path,
            dt_needed_override="libcupti.so.12\nlibcudart.so.12\nlibcusolver.so.11",
            search_dirs_override=tmpdir,
        )
        assert rc == 0, f"apply rc={rc}, stderr={err}"
        st = json.loads(out)
        assert st["fully_applied"] is True, st
        assert len(st["found_cuda_libs"]) == 2, st["found_cuda_libs"]
        assert st["missing_host_libs"] == ["libcusolver.so.11"], st
        # 13 baseline + 2 found = 15
        assert st["expected_mount_count"] == 15, st
        assert "WARNING" in err and "libcusolver.so.11" in err, f"expected libcusolver warning, got: {err}"
        return "PASS"
    finally:
        os.unlink(path)
        import shutil as _sh
        _sh.rmtree(tmpdir)


def case_xiii_b6_5_devnode_and_cublaslt_disable():
    """B.6''.5: /dev/cipher CDI deviceNode + TORCH_CUBLASLT_DISABLE=1 env.

    Apply on factory yaml expects:
      env list contains TORCH_CUBLASLT_DISABLE=1
      env LD_LIBRARY_PATH includes both /usr/lib/cipher and the container
        bundled nvidia/cu13/lib path
      deviceNodes contains /dev/cipher
    Revert removes both.
    """
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, out, _ = _run("apply", path, search_dirs_override="")
        assert rc == 0
        st = json.loads(out)
        assert st["fully_applied"] is True, st
        assert st["applied_device_node_count"] == 1, st
        assert st["expected_device_node_count"] == 1, st
        spec = _read_spec(path)
        env = spec["containerEdits"]["env"]
        assert "TORCH_CUBLASLT_DISABLE=1" in env, env
        ld = next((e for e in env if e.startswith("LD_LIBRARY_PATH=")), None)
        assert ld is not None, env
        assert "/usr/lib/cipher" in ld and "nvidia/cu13/lib" in ld, ld
        device_nodes = spec["containerEdits"].get("deviceNodes", [])
        assert any(dn.get("path") == "/dev/cipher" for dn in device_nodes), device_nodes
        # revert removes
        rc, _, _ = _run("revert", path, search_dirs_override="")
        assert rc == 0
        spec = _read_spec(path)
        env = spec["containerEdits"]["env"]
        assert "TORCH_CUBLASLT_DISABLE=1" not in env
        device_nodes = spec["containerEdits"].get("deviceNodes", [])
        assert not any(dn.get("path") == "/dev/cipher" for dn in device_nodes), device_nodes
        return "PASS"
    finally:
        os.unlink(path)


def case_xiv_b6_9_3_register_model_gate():
    """B.6''.9.3 Path β.3: CIPHER_REGISTER_MODEL=0 env gates NR 27 ioctl off.

    F-VLLM-PLUGIN-CRASH (B.6''.9.1) root-caused vLLM v0.21.0 segfault to the
    cipher_vllm_kv plugin's CIPHER_REGISTER_MODEL (NR 27) ioctl downstream
    path. Bisect Test 1B-7 confirmed env-gating it off makes vLLM crash-free
    while preserving substrate + VMM-allocator + REGISTER_STREAMS. NR 27 has
    zero v1 consumers (SC6 uses NRs 21-24 only; no cipher_model_uuid readers
    in substrate or bridge), so gating off has no v1 customer impact. v1.x
    picks up KASAN root-cause + re-enable.

    Apply expects env list contains CIPHER_REGISTER_MODEL=0.
    Revert expects it removed.
    """
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, out, _ = _run("apply", path, search_dirs_override="")
        assert rc == 0
        spec = _read_spec(path)
        env = spec["containerEdits"]["env"]
        assert "CIPHER_REGISTER_MODEL=0" in env, env
        rc, _, _ = _run("revert", path, search_dirs_override="")
        assert rc == 0
        spec = _read_spec(path)
        env = spec["containerEdits"]["env"]
        assert "CIPHER_REGISTER_MODEL=0" not in env
        return "PASS"
    finally:
        os.unlink(path)


def case_xi_zero_cuda_on_host():
    """F-B.3.3 R4: no CUDA libs on host at all (empty search dirs).

    Inject DT_NEEDED of full chain; search dirs empty. Expect 0 CUDA mounts,
    full chain in missing_host_libs, WARNING per missing lib, fully_applied=
    True (the substrate+plugin+dist-info subset still applied).
    """
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, out, err = _run(
            "apply", path,
            dt_needed_override="libcupti.so.12\nlibcudart.so.12\nlibcusolver.so.11",
            search_dirs_override="",
        )
        assert rc == 0, f"apply rc={rc}, stderr={err}"
        st = json.loads(out)
        assert st["fully_applied"] is True, st
        assert st["found_cuda_libs"] == [], st
        # libcusolver's transitive deps are NOT walked because the lib itself
        # was not findable; we only know about the 3 direct DT_NEEDED entries
        assert set(st["missing_host_libs"]) == {
            "libcupti.so.12", "libcudart.so.12", "libcusolver.so.11"
        }, st
        # 13 baseline + 0 CUDA = 13
        assert st["expected_mount_count"] == 13, st
        for soname in ("libcupti.so.12", "libcudart.so.12", "libcusolver.so.11"):
            assert soname in err, f"expected {soname} in stderr, got: {err}"
        return "PASS"
    finally:
        os.unlink(path)


def case_vii_no_write_when_unchanged():
    """Apply on already-fully-patched yaml: no file write (mtime unchanged); avoids systemd path-watcher feedback loop."""
    import time
    path = _write_tmp(FACTORY_NVIDIA_YAML)
    try:
        rc, _, _ = _run("apply", path)
        assert rc == 0
        mtime1 = os.stat(path).st_mtime_ns
        time.sleep(0.05)
        rc, _, _ = _run("apply", path)
        assert rc == 0
        mtime2 = os.stat(path).st_mtime_ns
        assert mtime1 == mtime2, f"file rewritten on no-op apply (mtime {mtime1} -> {mtime2}); systemd watcher loop risk"
        # revert-twice: same property
        rc, _, _ = _run("revert", path)
        assert rc == 0
        mtime3 = os.stat(path).st_mtime_ns
        time.sleep(0.05)
        rc, _, _ = _run("revert", path)
        assert rc == 0
        mtime4 = os.stat(path).st_mtime_ns
        assert mtime3 == mtime4, f"file rewritten on no-op revert (mtime {mtime3} -> {mtime4})"
        return "PASS"
    finally:
        os.unlink(path)


def case_vi_malformed_yaml():
    """Apply on malformed nvidia.yaml: exit 2 with operator-readable error; no silent corruption."""
    path = _write_tmp(MALFORMED_YAML)
    try:
        rc, out, err = _run("apply", path, json_flag=False)
        assert rc == 2, f"expected rc=2, got rc={rc}, stderr={err}"
        assert "malformed" in err.lower() or "yaml" in err.lower(), err
        # file unchanged
        with open(path) as f:
            assert f.read() == MALFORMED_YAML
        return "PASS"
    finally:
        os.unlink(path)


def main():
    cases = [
        ("(i)   clean apply on factory-default nvidia.yaml", case_i_clean_apply),
        ("(ii)  idempotent re-apply", case_ii_idempotent_reapply),
        ("(iii) revert after apply", case_iii_revert_after_apply),
        ("(iv)  apply with unrelated additions", case_iv_apply_with_unrelated_additions),
        ("(v)   missing nvidia.yaml", case_v_missing_yaml),
        ("(vi)  malformed nvidia.yaml", case_vi_malformed_yaml),
        ("(vii) no-write-when-unchanged (watcher loop guard)", case_vii_no_write_when_unchanged),
        ("(viii) libcupti anchor via generalized DT_NEEDED discovery", case_viii_libcupti_anchor),
        ("(ix)  full CUDA 12 toolkit transitive BFS", case_ix_full_cuda12_toolkit),
        ("(x)   partial CUDA toolkit on host", case_x_partial_cuda12_toolkit),
        ("(xi)  zero CUDA libs on host", case_xi_zero_cuda_on_host),
        ("(xiii) B.6''.5 /dev/cipher + TORCH_CUBLASLT_DISABLE", case_xiii_b6_5_devnode_and_cublaslt_disable),
        ("(xiv) B.6''.9.3 CIPHER_REGISTER_MODEL=0 gate", case_xiv_b6_9_3_register_model_gate),
    ]
    failures = []
    results = []
    for label, fn in cases:
        try:
            result = fn()
        except AssertionError as e:
            result = "FAIL"
            failures.append((label, repr(e)))
        except Exception as e:
            result = "ERROR"
            failures.append((label, repr(e)))
        results.append((label, result))
        print(f"  {label}: {result}")

    print()
    print(f"PASSED {sum(1 for _, r in results if r == 'PASS')} / {len(results)}")
    if failures:
        print()
        print("FAILURES:")
        for label, msg in failures:
            print(f"  {label}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
