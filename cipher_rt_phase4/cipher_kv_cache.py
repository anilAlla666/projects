"""CIPHER T4.6.2 S2b — operator-context KV cache integration.

Subclasses transformers 5.8.1's StaticLayer / StaticSlidingWindowLayer so
their KV backing tensors are allocated from CIPHER's VMM page pool
(cipher_kv_bridge) instead of torch.zeros. CIPHER then owns and tags the
KV memory — the foundation for T4.6.3 dedup and T4.6.4 cuIpc export.

install() monkey-patches the layer classes in transformers.cache_utils so
that StaticCache builds CIPHER-backed layers. Customer code is unchanged;
the operator injects this (e.g. via sitecustomize.py / a bootstrap import)
and defaults cache_implementation=static.

Env gates:
  CIPHER_KV_ALLOC = 1|0   (default 1)  — master opt-out per tenant
  CIPHER_TENANT_NUM = int (default 0)  — tenant id stamped into page tags
  CIPHER_KV_VA_POOL_GIB = int (default 16) — VA pool reservation
"""
import os
import sys
import itertools

# cipher_kv_bridge.so lives next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cipher_kv_bridge  # noqa: E402
import torch  # noqa: E402

try:
    from transformers.utils import is_torchdynamo_compiling
except Exception:  # pragma: no cover
    def is_torchdynamo_compiling():
        return False

from transformers.cache_utils import StaticLayer, StaticSlidingWindowLayer  # noqa: E402

_ROLE_K = 0
_ROLE_V = 1
_seq_ids = itertools.count(1)
_init_done = False


def _enabled():
    return os.environ.get("CIPHER_KV_ALLOC", "1") not in ("0", "off", "no", "")


def _ensure_alloc_init():
    global _init_done
    if not _init_done:
        gib = int(os.environ.get("CIPHER_KV_VA_POOL_GIB", "16"))
        if not cipher_kv_bridge.init(gib * 1024 * 1024 * 1024):
            raise RuntimeError("cipher_kv_bridge.init failed")
        _init_done = True


class _CipherKVMixin:
    """Overrides StaticLayer.lazy_initialization so keys/values are
    CIPHER-VMM-backed. Everything else (update's in-place index_copy_,
    get_seq_length, masks) is inherited unchanged."""

    def lazy_initialization(self, key_states, value_states):
        _ensure_alloc_init()
        self.dtype, self.device = key_states.dtype, key_states.device
        self.max_batch_size, self.num_heads = key_states.shape[:2]
        self.k_head_dim = key_states.shape[-1]
        self.v_head_dim = value_states.shape[-1]

        tenant = int(os.environ.get("CIPHER_TENANT_NUM", "0"))
        seq = getattr(self, "_cipher_seq_id", 0)
        layer = getattr(self, "_cipher_layer_idx", 0)
        dtype_str = str(key_states.dtype).split(".")[-1]
        itemsize = key_states.element_size()

        k_shape = [self.max_batch_size, self.num_heads,
                   self.max_cache_len, self.k_head_dim]
        v_shape = [self.max_batch_size, self.num_heads,
                   self.max_cache_len, self.v_head_dim]

        self.keys = cipher_kv_bridge.vmm_zeros(
            k_shape, itemsize, dtype_str, tenant, seq, layer, _ROLE_K)
        self.values = cipher_kv_bridge.vmm_zeros(
            v_shape, itemsize, dtype_str, tenant, seq, layer, _ROLE_V)
        self.cumulative_length = self.cumulative_length.to(self.device)

        # VMM slabs are fixed-data-pointer by construction — tag them so
        # cudagraph capture treats them as static addresses.
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.cumulative_length)

        self.is_initialized = True


class CipherStaticLayer(_CipherKVMixin, StaticLayer):
    pass


class CipherStaticSlidingWindowLayer(_CipherKVMixin, StaticSlidingWindowLayer):
    pass


def install():
    """Monkey-patch transformers.cache_utils so StaticCache builds
    CIPHER-backed layers. Idempotent. No-op if CIPHER_KV_ALLOC=0."""
    if not _enabled():
        print("[cipher-kv] CIPHER_KV_ALLOC disabled — stock StaticCache",
              file=sys.stderr)
        return False

    import transformers.cache_utils as cu
    cu.StaticLayer = CipherStaticLayer
    cu.StaticSlidingWindowLayer = CipherStaticSlidingWindowLayer

    if not getattr(cu.StaticCache.__init__, "_cipher_wrapped", False):
        _orig_init = cu.StaticCache.__init__

        def _wrapped_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            seq = next(_seq_ids)
            self._cipher_seq_id = seq
            for i, layer in enumerate(self.layers):
                layer._cipher_layer_idx = i
                layer._cipher_seq_id = seq

        _wrapped_init._cipher_wrapped = True
        cu.StaticCache.__init__ = _wrapped_init

    print("[cipher-kv] installed: StaticCache layers are CIPHER-VMM-backed",
          file=sys.stderr)
    return True
