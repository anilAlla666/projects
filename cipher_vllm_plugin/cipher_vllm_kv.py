"""CIPHER CP 5.1 — vLLM KV-cache buffer-ownership plugin (Option A).

Registered under the ``vllm.general_plugins`` entry-point group, so vLLM's
``load_general_plugins()`` runs ``register()`` in every process — including the
EngineCore subprocess, which is where ``GPUModelRunner._allocate_kv_cache_tensors``
actually executes.

What it does: monkey-patches ``GPUModelRunner._allocate_kv_cache_tensors`` so the
raw int8 KV-cache buffers are sourced from CIPHER's CUDA-VMM allocator
(``cipher_kv_bridge.vmm_zeros``) instead of ``torch.zeros``. CIPHER then owns and
page-tags the KV memory. vLLM's downstream ``_reshape_kv_cache_tensors`` is pure
``.view()``/``as_strided`` and is unaffected.

Coverage-immune: CIPHER *is* the buffer, so every attention kernel — eager,
compiled, or cudagraph-replayed — reads/writes CIPHER memory. See
``CP_5_1_STEP_2_VLLM_KV_PATH.md`` §(c) Option A.

Env gates:
  CIPHER_KV_ALLOC       1|0   (default 1)  — master opt-out
  CIPHER_KV_VA_POOL_GIB int   (default 80) — VA pool reservation (VA only,
                                             no physical until first vmm_zeros)
  CIPHER_TENANT_NUM     int   (default 0)  — tenant id stamped into page tags
  CIPHER_RT_DIR         path  (default /home/ubuntu/cipher_rt_phase4) — holds
                                             cipher_kv_bridge*.so
"""
import os
import sys

_PREFIX = "[cipher-vllm-kv]"


def _log(msg):
    print(f"{_PREFIX} {msg}", file=sys.stderr, flush=True)


def _enabled():
    return os.environ.get("CIPHER_KV_ALLOC", "1") not in ("0", "off", "no", "")


# cipher_kv_bridge.so lives in the cipher_rt_phase4 tree; put it on sys.path
# at import time so the lazy import below resolves.
_RT_DIR = os.environ.get("CIPHER_RT_DIR", "/home/ubuntu/cipher_rt_phase4")
if _RT_DIR not in sys.path:
    sys.path.insert(0, _RT_DIR)

_TENANT = int(os.environ.get("CIPHER_TENANT_NUM", "0"))
_bridge_ready = False


def _ensure_bridge_init():
    """Reserve the CIPHER VMM VA pool. VA-reservation only — no physical GPU
    memory is mapped until the first vmm_zeros, so this is safe to run after
    vLLM's memory profiler (which has already sized num_gpu_blocks by now)."""
    global _bridge_ready
    if _bridge_ready:
        return
    import cipher_kv_bridge  # noqa: E402
    va_gib = int(os.environ.get("CIPHER_KV_VA_POOL_GIB", "80"))
    if not cipher_kv_bridge.init(va_gib * (1 << 30)):
        raise RuntimeError("cipher_kv_bridge.init failed")
    _log(f"VMM allocator init: VA pool {va_gib} GiB (tenant={_TENANT})")
    _bridge_ready = True


def _cipher_allocate_kv_cache_tensors(self, kv_cache_config):
    """Drop-in replacement for GPUModelRunner._allocate_kv_cache_tensors.

    Faithful to vLLM 0.20.2's original (gpu_model_runner.py:6522) — same buffer
    sizing, same shared_by aliasing, same layer-name assertion — except the raw
    int8 buffer comes from cipher_kv_bridge.vmm_zeros instead of torch.zeros.
    """
    import cipher_kv_bridge
    _ensure_bridge_init()

    # vmm_zeros allocates on cuda:0. CP 5.1 baseline is single-GPU; fail loud
    # rather than silently place KV on the wrong device under TP.
    dev_index = getattr(self.device, "index", 0) or 0
    assert dev_index == 0, (
        f"{_PREFIX} CIPHER KV hook supports cuda:0 only; got {self.device}. "
        "Multi-GPU TP needs a device arg threaded through vmm_zeros (Step 3+).")

    kv_cache_raw_tensors = {}
    for idx, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
        nbytes = int(kv_cache_tensor.size)
        # role=2: vLLM buffers are per-layer, K/V interleaved by stride — there
        # is no single K-or-V role (known-unknown #1, page-tag granularity).
        tensor = cipher_kv_bridge.vmm_zeros(
            [nbytes], 1, "int8", _TENANT, 0, idx & 0xFFFF, 2)
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            if layer_name in self.runner_only_attn_layers:
                continue
            layer_names.add(layer_name)
    assert layer_names == set(kv_cache_raw_tensors.keys()), (
        f"{_PREFIX} some layers are not correctly initialized")

    st = cipher_kv_bridge.get_stats()
    _log(f"KV buffers owned by CIPHER VMM: {len(kv_cache_config.kv_cache_tensors)} "
         f"tensor(s), slabs_created={st['slabs_created']}, "
         f"pages_mapped={st['pages_mapped']}, "
         f"resident={st['pages_resident'] * st['page_size'] / (1 << 30):.2f} GiB")
    return kv_cache_raw_tensors


def _register_offload():
    """Install the CP 5.2 KV offload hooks (snapshot-on-preempt /
    restore-on-resume) so the whole CIPHER substrate composes under this one
    vLLM entry point. Independently env-gated by CIPHER_KV_OFFLOAD."""
    try:
        import cipher_kv_offload
        cipher_kv_offload.register()
    except Exception as e:  # pragma: no cover
        _log(f"CP 5.2 offload register failed ({e}); offload NOT installed")


def register():
    """vLLM general-plugin entry point. Installs the CP 5.1 KV buffer hook and
    the CP 5.2 offload hooks."""
    if not _enabled():
        _log("disabled via CIPHER_KV_ALLOC — vLLM uses its native torch.zeros KV")
        _register_offload()  # offload is independent of buffer ownership
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # pragma: no cover
        _log(f"could not import GPUModelRunner ({e}); hook NOT installed")
        return

    current = GPUModelRunner._allocate_kv_cache_tensors
    if getattr(current, "_cipher_hooked", False):
        _register_offload()
        return  # idempotent — load_general_plugins may run more than once
    _cipher_allocate_kv_cache_tensors._cipher_hooked = True
    _cipher_allocate_kv_cache_tensors._cipher_orig = current
    GPUModelRunner._allocate_kv_cache_tensors = _cipher_allocate_kv_cache_tensors
    _log("hook installed on GPUModelRunner._allocate_kv_cache_tensors (Option A)")
    _register_offload()
