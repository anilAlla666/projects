# PERIODIC FP8 detection under cudagraph via DUAL CAPTURED GRAPHS (ra_coophook A3 port, in-process).
# Unchecked graph = vLLM's vanilla forward (recompute absent). Checked graph = a 2nd manual capture
# with recompute active (host flag _CHK flips -> dynamo recompiles the checked variant). Host alternates
# at the CUDAGraphWrapper replay boundary: checked every N steps, unchecked otherwise. Module buffers
# (Addendum-4 channel) carry the residual out. cudagraph ON, default-except-MP (in-process for iteration).
import os, json, time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
import torch._inductor.config as ic
ic.triton.cudagraph_support_input_mutation = True

N = int(os.environ.get("CG_N", "32"))
_CHK = [False]                      # host flag: dynamo guards on it -> two compiled variants

cl = torch.library.Library("cipherd", "FRAGMENT")
cl.define("sink(Tensor d, Tensor(a!) res, Tensor(b!) cnt) -> ()")
cl.impl("sink", lambda d, res, cnt: (torch.maximum(res, d.reshape(1), out=res), cnt.add_(1)) and None, "CompositeExplicitAutograd")
torch.library.register_fake("cipherd::sink", lambda d, res, cnt: None)

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8 as CW
_LAYERS = []
_oc = CW.create_weights
def _cw(self, layer, *a, **k):
    r = _oc(self, layer, *a, **k)
    for nm in ("cipher_res", "cipher_cnt", "cipher_inj"):
        layer.register_buffer(nm, torch.zeros(1, device="cuda"), persistent=False)
    for t in (layer.cipher_res, layer.cipher_cnt, layer.cipher_inj):
        try: torch._dynamo.mark_static_address(t)
        except Exception: pass
    _LAYERS.append(layer); return r
CW.create_weights = _cw
_oa = CW.apply_weights
def _aw(self, layer, x, bias=None):
    out = _oa(self, layer, x, bias)
    if not hasattr(layer, "cipher_res"): return out
    out = out + layer.cipher_inj.to(out.dtype) * 0.5       # persistent fault present in BOTH graphs
    if not _CHK[0]: return out                              # UNCHECKED variant: no recompute
    out2 = _oa(self, layer, x, bias)                        # CHECKED variant: recompute
    d = (out2.float() - out.float()).abs().amax()
    torch.ops.cipherd.sink(d, layer.cipher_res, layer.cipher_cnt)
    return out
CW.apply_weights = _aw

# ---- dual-graph coop hook on CUDAGraphWrapper ----
import vllm.compilation.cuda_graph as cg
_orig_call = cg.CUDAGraphWrapper.__call__
class C:
    step = 0; checks = 0; first_detect_step = None; max_resid = 0.0
    checked = {}; checked_out = {}
def agg():
    mx = 0.0; cnt = 0
    for L in _LAYERS:
        try: mx = max(mx, float(L.cipher_res.item())); cnt += int(L.cipher_cnt.item())
        except Exception: pass
    return mx, cnt
def patched_call(self, *args, **kwargs):
    if not cg.is_forward_context_available(): return _orig_call(self, *args, **kwargs)
    fc = cg.get_forward_context()
    if fc.cudagraph_runtime_mode != self.runtime_mode or self.runtime_mode != cg.CUDAGraphMode.FULL:
        return _orig_call(self, *args, **kwargs)
    bd = fc.batch_descriptor
    entry = self.concrete_cudagraph_entries.get(bd)
    is_capture = entry is None or entry.cudagraph is None
    if is_capture:
        _CHK[0] = True                                     # capture CHECKED via vLLM's own capture (preserves module-buffer mutation, Addendum-4)
        out = _orig_call(self, *args, **kwargs)
        ent = self.concrete_cudagraph_entries.get(bd)
        C.checked[bd] = ent.cudagraph; C.checked_out[bd] = ent.output
        self.concrete_cudagraph_entries.pop(bd, None)      # force recapture of the UNCHECKED variant
        _CHK[0] = False
        _orig_call(self, *args, **kwargs)                  # vLLM captures UNCHECKED -> entry[bd]
        return out
    # replay
    C.step += 1
    if (C.step % N == 0) and (bd in C.checked):
        C.checked[bd].replay(); C.checks += 1              # CHECKED (vLLM-captured, detects)
        mx, _ = agg()
        if mx > C.max_resid: C.max_resid = mx
        if mx > 1e-6 and C.first_detect_step is None: C.first_detect_step = C.step
        return C.checked_out[bd]
    entry.cudagraph.replay(); return entry.output          # UNCHECKED (cheap)
cg.CUDAGraphWrapper.__call__ = patched_call

from vllm import LLM, SamplingParams
def main():
    M = os.environ["S_MODEL"]; OUT = os.environ["S_OUT"]; NTOK = int(os.environ.get("S_NTOK", "256"))
    INJECT = os.environ.get("S_INJECT", "0") == "1"
    llm = LLM(model=M, enforce_eager=False, gpu_memory_utilization=0.82, max_model_len=2048, disable_log_stats=True)
    n_layers = len(_LAYERS); n_checked = len(C.checked)
    sp = SamplingParams(max_tokens=NTOK, temperature=0.0, ignore_eos=True, min_tokens=NTOK)
    prompts = ["The history of the Roman Empire begins with"]
    llm.generate(prompts, sp, use_tqdm=False)              # warm (capture both graphs)
    for L in _LAYERS: L.cipher_res.zero_(); L.cipher_cnt.zero_()
    C.step = 0; C.checks = 0; C.first_detect_step = None; C.max_resid = 0.0
    t0 = time.perf_counter(); o1 = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0
    g1 = sum(len(x.outputs[0].token_ids) for x in o1); tps = g1 / dt
    clean_max = C.max_resid; clean_checks = C.checks
    _, clean_cnt = agg()
    res = {"model": M, "N": N, "fp8_layers": n_layers, "checked_graphs": n_checked,
           "tok_s": round(tps, 2), "clean_max_resid": round(clean_max, 6), "clean_checks": clean_checks,
           "sink_cnt_total": clean_cnt, "recompute_in_checked": clean_cnt > 0, "step": C.step}
    if INJECT:
        C.first_detect_step = None; C.max_resid = 0.0; C.step = 0
        tgt = _LAYERS[len(_LAYERS)//2]; tgt.cipher_inj.fill_(1.0)
        o2 = llm.generate(prompts, sp, use_tqdm=False)
        res["inj_max_resid"] = round(C.max_resid, 6); res["first_detect_step"] = C.first_detect_step
        res["caught"] = C.max_resid > 1e-6; res["inj_out"] = o2[0].outputs[0].text[:40]
    json.dump(res, open(OUT, "w"), indent=1)
    print("CGPER", json.dumps({k: res.get(k) for k in ["N","fp8_layers","checked_graphs","tok_s","clean_max_resid","caught","first_detect_step"]}))

if __name__ == "__main__":
    main()
