"""CIPHER CP 5.2 — vLLM v1 KV offload: snapshot-on-preempt / restore-on-resume.

vLLM v1 preemption is *recompute-only*: ``Scheduler._preempt_request`` frees the
request's KV blocks and sets ``num_computed_tokens = 0``, so a resumed request
re-runs prefill from scratch (CP 5.2 Step 1 memo §(a) F5). This module turns
that discard into a two-tier HBM <-> host-DRAM offload:

  * **snapshot on preempt** — before the blocks are recycled, gather-copy the
    request's KV out of the CIPHER-owned per-layer slabs into a pinned
    host-DRAM store keyed by request_id;
  * **preserve num_computed_tokens** — so the resumed request rides vLLM's
    existing ``num_computed_tokens > 0`` waiting-loop branch
    (``scheduler.py`` line ~657, the KV-connector path) instead of recomputing;
  * **restore on resume** — once the scheduler re-allocates blocks, scatter the
    KV back from host DRAM before ``execute_model`` runs.

Option A (Step 1 memo §(d), adjudicated APPROVED): the CIPHER VMM slabs stay
fully mapped throughout — no ``cuMemUnmap`` — so the 2 MiB page-granularity
constraint never bites. NVMe cold tier is deferred.

Gather/scatter is ``torch.index_select`` / ``Tensor.index_copy_`` along the
detected block dimension — backend-agnostic and correct-by-construction over
the strided ``[2, num_blocks, ...]`` K/V layout. cipher_kv_bridge is reused for
``page_info`` (the tenant-tag check, gate criterion C). See the Step 3 build
log for why no pybind gather method was added.

Env gate: CIPHER_KV_OFFLOAD  1|0  (default 1).
"""
import os
import sys
import time

_PREFIX = "[cipher-offload]"


def _log(msg):
    print(f"{_PREFIX} {msg}", file=sys.stderr, flush=True)


def _enabled():
    return os.environ.get("CIPHER_KV_OFFLOAD", "1") not in ("0", "off", "no", "")


# cipher_kv_bridge lives in the cipher_rt_phase4 tree (same as CP 5.1).
_RT_DIR = os.environ.get("CIPHER_RT_DIR", "/home/ubuntu/cipher_rt_phase4")
if _RT_DIR not in sys.path:
    sys.path.insert(0, _RT_DIR)


class _OffloadManager:
    """Process-wide singleton holding the host-DRAM KV store and the GPU
    KV-tensor references. Lives in the EngineCore process; the runner hook
    populates ``kv_caches``, the scheduler hooks drive snapshot/restore."""

    def __init__(self):
        self.kv_caches = None       # list[torch.Tensor], one per layer
        self.block_dim = None       # detected once at bind time
        self.num_blocks = None
        self.ready = False
        self.store = {}             # request_id -> parked dict
        # telemetry
        self.snapshots = 0
        self.restores = 0
        self.bytes_d2h = 0
        self.bytes_h2d = 0
        self.d2h_ms = 0.0
        self.h2d_ms = 0.0
        self.tag_pass = 0
        self.tag_fail = 0

    # -- bind -----------------------------------------------------------
    def bind_kv_caches(self, kv_caches, num_blocks):
        """Register the live per-layer KV tensors and detect the block dim
        (the unique tensor dimension whose size == num_blocks)."""
        if not kv_caches:
            _log("bind: runner has no kv_caches — offload INACTIVE")
            return
        shape = tuple(kv_caches[0].shape)
        # The block dimension is an OUTER dim of a paged KV tensor
        # [..., num_blocks, block_size, num_kv_heads, head_size]; the
        # innermost dim is always head_size and is never the block dim.
        # Excluding it disambiguates the case where num_blocks coincidentally
        # equals head_size (seen with a tiny num_gpu_blocks_override).
        matches = [d for d, s in enumerate(shape)
                   if s == num_blocks and d != len(shape) - 1]
        if len(matches) != 1:
            _log(f"bind: block dim not unique in shape {shape} for "
                 f"num_blocks={num_blocks} (candidates={matches}); "
                 f"offload INACTIVE")
            return
        bd = matches[0]
        for i, t in enumerate(kv_caches):
            if t.shape[bd] != num_blocks:
                _log(f"bind: layer {i} shape {tuple(t.shape)} disagrees on "
                     f"block dim {bd}; offload INACTIVE")
                return
        self.kv_caches = kv_caches
        self.num_blocks = num_blocks
        self.block_dim = bd
        self.ready = True
        _log(f"bound {len(kv_caches)} KV layer tensor(s) shape={shape} "
             f"dtype={kv_caches[0].dtype} block_dim={bd} num_blocks={num_blocks}")

    # -- tag check (criterion C, via cipher_kv_bridge) ------------------
    def _layer0_tag(self):
        """Tenant tag of layer-0's KV slab, via cipher_kv_bridge.page_info.
        Returns a dict, or None if the buffer is not CIPHER-owned (CP 5.1
        hook off) or the bridge is unavailable."""
        try:
            import cipher_kv_bridge
            return cipher_kv_bridge.page_info(self.kv_caches[0].data_ptr())
        except Exception:
            return None

    # -- snapshot -------------------------------------------------------
    def snapshot(self, request_id, block_ids, num_computed_tokens):
        """Gather-copy the request's KV blocks GPU -> pinned host DRAM."""
        import torch
        if not block_ids:
            return False
        t0 = time.time()
        idx = torch.tensor(block_ids, device=self.kv_caches[0].device,
                            dtype=torch.long)
        host_layers = []
        nbytes = 0
        for kv in self.kv_caches:
            g = torch.index_select(kv, self.block_dim, idx)   # GPU gather
            h = torch.empty(g.shape, dtype=g.dtype, device="cpu",
                            pin_memory=True)
            h.copy_(g, non_blocking=False)                    # D2H
            host_layers.append(h)
            nbytes += h.numel() * h.element_size()
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1e3
        self.store[request_id] = {
            "layers": host_layers,
            "num_computed_tokens": num_computed_tokens,
            "n_blocks": len(block_ids),
            "tag": self._layer0_tag(),
        }
        self.snapshots += 1
        self.bytes_d2h += nbytes
        self.d2h_ms += dt
        bw = nbytes / (1 << 20) / (dt / 1e3) if dt > 0 else 0.0
        _log(f"snapshot req={request_id} blocks={len(block_ids)} "
             f"num_computed_tokens={num_computed_tokens} "
             f"d2h={nbytes / (1 << 20):.1f}MiB {dt:.1f}ms ({bw:.0f}MiB/s)")
        return True

    # -- restore --------------------------------------------------------
    def restore(self, request_id, new_block_ids):
        """Scatter-copy the request's KV back host DRAM -> GPU into the
        freshly-allocated blocks. Returns the preserved num_computed_tokens,
        or None if nothing was parked / the new allocation is too small."""
        import torch
        parked = self.store.get(request_id)
        if parked is None:
            return None
        n = parked["n_blocks"]
        if len(new_block_ids) < n:
            _log(f"restore req={request_id} ABORT: resumed with "
                 f"{len(new_block_ids)} blocks < snapshot {n}")
            return None
        t0 = time.time()
        idx = torch.tensor(new_block_ids[:n], device=self.kv_caches[0].device,
                            dtype=torch.long)
        nbytes = 0
        for kv, h in zip(self.kv_caches, parked["layers"]):
            g = h.to(kv.device, non_blocking=False)           # H2D
            kv.index_copy_(self.block_dim, idx, g)            # GPU scatter
            nbytes += h.numel() * h.element_size()
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1e3
        # criterion C — tenant tag must survive the preempt/resume round trip
        tag_now = self._layer0_tag()
        tag_then = parked["tag"]
        tag_ok = (tag_then is not None and tag_now is not None
                  and tag_then.get("tenant_id") == tag_now.get("tenant_id")
                  and tag_then.get("layer") == tag_now.get("layer"))
        if tag_then is None and tag_now is None:
            tag_state = "n/a (KV not CIPHER-owned)"
        elif tag_ok:
            tag_state = f"PRESERVED tenant_id={tag_now.get('tenant_id')}"
            self.tag_pass += 1
        else:
            tag_state = f"MISMATCH then={tag_then} now={tag_now}"
            self.tag_fail += 1
        del self.store[request_id]
        self.restores += 1
        self.bytes_h2d += nbytes
        self.h2d_ms += dt
        bw = nbytes / (1 << 20) / (dt / 1e3) if dt > 0 else 0.0
        _log(f"restore req={request_id} blocks={n} "
             f"h2d={nbytes / (1 << 20):.1f}MiB {dt:.1f}ms ({bw:.0f}MiB/s) "
             f"tag={tag_state}")
        return parked["num_computed_tokens"]

    def summary(self):
        _log(f"SUMMARY snapshots={self.snapshots} restores={self.restores} "
             f"still_parked={len(self.store)} "
             f"d2h={self.bytes_d2h / (1 << 20):.1f}MiB/{self.d2h_ms:.1f}ms "
             f"h2d={self.bytes_h2d / (1 << 20):.1f}MiB/{self.h2d_ms:.1f}ms "
             f"tag_pass={self.tag_pass} tag_fail={self.tag_fail}")


MGR = _OffloadManager()

_installed = False


def register():
    """Install the CP 5.2 offload hooks. Called from cipher_vllm_kv.register()
    so the whole CIPHER substrate composes under one vLLM entry point."""
    global _installed
    if _installed:
        return
    if not _enabled():
        _log("disabled via CIPHER_KV_OFFLOAD — vLLM uses native recompute "
             "preemption")
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as e:  # pragma: no cover
        _log(f"could not import vLLM internals ({e}); offload NOT installed")
        return

    # -- hook 1: capture the live KV tensors at engine init -------------
    _orig_init_kv = GPUModelRunner.initialize_kv_cache

    def _patched_initialize_kv_cache(self, kv_cache_config, is_profiling=False):
        r = _orig_init_kv(self, kv_cache_config, is_profiling=is_profiling)
        if not is_profiling:        # skip the tiny profiling cache
            try:
                MGR.bind_kv_caches(self.kv_caches, kv_cache_config.num_blocks)
            except Exception as e:
                _log(f"bind failed: {e}")
        return r

    # -- hook 2: snapshot on preempt + preserve num_computed_tokens -----
    _orig_preempt = Scheduler._preempt_request

    def _patched_preempt_request(self, request, timestamp):
        rid = request.request_id
        n_ct = request.num_computed_tokens
        snapped = False
        if MGR.ready:
            try:
                bids = self.kv_cache_manager.get_block_ids(rid)
                block_ids = list(bids[0]) if bids else []
                if block_ids:
                    snapped = MGR.snapshot(rid, block_ids, n_ct)
            except Exception as e:
                _log(f"snapshot FAILED for {rid}: {e}")
        _orig_preempt(self, request, timestamp)   # frees blocks, zeroes n_ct
        if snapped:
            # ride vLLM's num_computed_tokens>0 waiting-loop branch — the
            # resumed request skips recompute; restore fills the KV.
            request.num_computed_tokens = n_ct

    # -- hook 3: restore once the scheduler re-allocates blocks ---------
    _orig_schedule = Scheduler.schedule

    def _patched_schedule(self):
        out = _orig_schedule(self)
        if MGR.ready and MGR.store:
            running_ids = {r.request_id for r in self.running}
            for rid in list(MGR.store.keys()):
                if rid not in running_ids:
                    continue                      # still parked in waiting
                try:
                    bids = self.kv_cache_manager.get_block_ids(rid)
                    new_block_ids = list(bids[0]) if bids else []
                    MGR.restore(rid, new_block_ids)
                except Exception as e:
                    _log(f"restore FAILED for {rid}: {e}")
        return out

    GPUModelRunner.initialize_kv_cache = _patched_initialize_kv_cache
    Scheduler._preempt_request = _patched_preempt_request
    Scheduler.schedule = _patched_schedule
    _installed = True
    _log("hooks installed: GPUModelRunner.initialize_kv_cache, "
         "Scheduler._preempt_request, Scheduler.schedule (CP 5.2 Option A)")
