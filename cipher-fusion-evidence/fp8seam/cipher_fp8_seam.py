# CIPHER FP8 detection seam (Phase 1, 2026-06-12). Extends the existing SDC detector
# (ra_realvllm/detector_shim.c) from the cuBLAS GemmEx seam to the FP8 cutlass_scaled_mm seam, so
# the ~93%-blind FP8 target linears become covered. Path B: launcher-side monkeypatch of
# vllm._custom_ops.cutlass_scaled_mm, ZERO vLLM source edits, same mechanism class as
# ra_coophook/coop_driver.py. Runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), the same
# operating condition the -3.9% cudagraph coop hook used.
#
# Detection contract reused verbatim from detector_shim.c:99-117:
#   - shape predicate selects q/k/v/o/gate/up/down (k in {4096,14336}, m != vocab)   [det:81]
#   - step structure from linear count: GPS linears per forward step                  [det:84-87]
#   - every N steps, independently recompute the matmul and compare residual           [det:99-107]
#   - clean residual is exactly 0 (deterministic T=0); ANY nonzero on a clean GEMM = FP [det:115]
#   - persistent bit-flip injection on a chosen ordinal every step                     [det:89-97]
# Contract honored (cipher_rt_matmul_dispatch.h:72,82-87): check work is PERIODIC; non-check steps
# add only a Counter increment and never sync or launch GPU work on the hot path.
import os, json
import torch
import vllm._custom_ops as ops

N        = int(os.environ.get("FP8_N", "8"))       # check every N decode steps
INJ      = int(os.environ.get("FP8_INJ", "-1"))    # inject persistent flip on this linear ordinal, -1=off
OUTP     = os.environ.get("FP8_OUT", "/home/ubuntu/cipher-fusion-evidence/fp8seam/fp8seam_result.json")
GPS      = int(os.environ.get("FP8_GPS", "128"))   # linear GEMMs per forward step (4 fused x 32 layers)
VOCAB    = 128256

_real = ops.cutlass_scaled_mm

st = {
    "lin": 0, "checks": 0, "gemms_checked": 0, "detections": 0,
    "injected": 0, "max_clean_resid": 0.0, "max_detect_resid": 0.0,
    "first_detect_step": -1, "csmm_total": 0, "covered_linears": 0,
    "shapes_seen": {}, "N": N, "INJ": INJ, "GPS": GPS,
}

def _seam(a, b, scale_a, scale_b, out_dtype, bias=None):
    out = _real(a, b, scale_a, scale_b, out_dtype, bias)
    st["csmm_total"] += 1
    k = a.shape[-1]
    m = a.shape[0] if a.dim() == 2 else a.numel() // k
    n = b.shape[1] if b.dim() == 2 else b.shape[-1]
    is_linear = (k == 4096 or k == 14336) and (m != VOCAB) and (n != VOCAB)
    if not is_linear:
        return out
    st["covered_linears"] += 1
    sk = f"{m}x{n}x{k}"; st["shapes_seen"][sk] = st["shapes_seen"].get(sk, 0) + 1

    ordinal = st["lin"] % GPS
    cur_step = st["lin"] // GPS
    is_check = (cur_step % N) == 0

    # persistent injection: flip a bit in the output of the targeted ordinal every step
    if INJ >= 0 and ordinal == INJ:
        flat = out.view(-1)
        v = flat[0].clone()
        # flip a mid mantissa/exponent bit by adding a large relative perturbation deterministically
        flat[0] = v + (v.abs() + 1.0) * 0.5
        st["injected"] += 1

    if is_check:
        out2 = _real(a, b, scale_a, scale_b, out_dtype, bias)   # independent recompute D'
        resid = (out2.float() - out.float()).norm().item()       # ||D' - out||, syncs here only
        st["gemms_checked"] += 1
        hit = (resid > 0.0) or (resid != resid)
        if INJ >= 0 and ordinal == INJ:
            if resid > st["max_detect_resid"]:
                st["max_detect_resid"] = resid
            if hit:
                st["detections"] += 1
                if st["first_detect_step"] < 0:
                    st["first_detect_step"] = cur_step
        else:
            if resid > st["max_clean_resid"]:
                st["max_clean_resid"] = resid
            if hit:
                st["detections"] += 1   # clean hit = FALSE POSITIVE (det:115)

    st["lin"] += 1
    if st["lin"] % GPS == 0:
        step = st["lin"] // GPS
        if ((step - 1) % N) == 0:
            st["checks"] += 1
    return out

def install():
    ops.cutlass_scaled_mm = _seam
    try:
        import vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass as _ctk
        if hasattr(_ctk, "ops"):
            _ctk.ops.cutlass_scaled_mm = _seam
    except Exception:
        pass

def dump():
    # FP-on-clean count = detections on clean-ordinal checks; with INJ>=0 those are the non-target ones
    res = dict(st)
    res["clean_gemms_checked"] = res["gemms_checked"] - res["detections"] if INJ < 0 else None
    res["false_positives"] = res["detections"] if INJ < 0 else "(see clean run)"
    json.dump(res, open(OUTP, "w"), indent=1)
    return res
