"""W.4b.6 Step 1 — Track-2 weight-sharing (SC3 cuIPC weight-arena re-port).

One owner process packs a model's weights into a cipher_kv_bridge VMM arena
(cuMemExport POSIX-FD); N importer processes meta-load the model (params on
`meta`, NO safetensors read) and rebind every parameter onto a READ-only view of
the imported arena. N tenants then share ONE physical 14 GB Mistral weight copy
instead of each loading 14 GB — unlocking N=8 Mistral on an 80 GB pod.

Verbatim adaptation of the verified Track-2 SC3 producer/consumer
(phase_c/sc3_*.py): pack = `arena.alloc + copy_ + p.data=vt`; rebind =
`arena.view + set_submodule_attr`. Read-only mapping ⇒ weight bytes are the
owner's exact physical bytes ⇒ shared forward is bit-identical (Step 1.3 gate).
"""
import sys
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch                       # noqa: E402
import cipher_kv_bridge as kvb     # noqa: E402

_DT = {torch.float16: "float16", torch.bfloat16: "bfloat16",
       torch.float32: "float32"}


def set_submodule_attr(model, dotted, kind, value):
    *path, attr = dotted.split(".")
    mod = model
    for seg in path:
        mod = getattr(mod, seg)
    getattr(mod, "_" + kind)[attr] = value


def pack_arena(model, tenant_id=1):
    """OWNER: pack every param into a cuIPC arena; rebind model to arena views.
    Returns (arena, manifest). After this, model's params ARE the arena bytes."""
    params = list(model.named_parameters())
    total = sum(p.numel() * p.element_size() for _, p in params)
    arena = kvb.weight_arena_create(int(total) + 16 * 1024 * 1024, tenant_id)
    manifest = {"base": int(arena.base), "size": int(arena.size), "tensors": []}
    for name, p in params:
        esize = p.element_size()
        vt = arena.alloc(list(p.shape), esize, _DT[p.dtype])
        vt.copy_(p.data)
        manifest["tensors"].append({
            "name": name, "shape": list(p.shape), "dtype": _DT[p.dtype],
            "itemsize": esize, "offset": int(vt.data_ptr() - arena.base)})
        p.data = vt
    torch.cuda.synchronize()
    return arena, manifest


def import_rebind(model, fd, manifest):
    """IMPORTER: import the owner's arena and rebind a meta-loaded model's params
    onto READ-only views. Buffers (rotary etc.) are computed locally -> cuda.
    Returns the imported arena. Verifies every param ptr lands inside the arena."""
    arena = kvb.weight_arena_import(fd, 0, manifest["size"])
    mani = {t["name"]: t for t in manifest["tensors"]}
    for name, p in list(model.named_parameters()):
        e = mani[name]
        vt = arena.view(e["shape"], e["itemsize"], e["dtype"], e["offset"])
        set_submodule_attr(model, name, "parameters",
                           torch.nn.Parameter(vt, requires_grad=False))
    for name, b in list(model.named_buffers()):
        set_submodule_attr(model, name, "buffers", b.cuda())
    model.train(False)
    # structural check: every param must be inside the imported arena (no
    # accidental second 14 GB copy — the memory-blowup trap)
    base, size = int(arena.base), int(arena.size)
    outside = [n for n, p in model.named_parameters()
               if not (base <= p.data_ptr() < base + size)]
    if outside:
        raise RuntimeError("params NOT in arena (double-copy): %s" % outside[:5])
    return arena
