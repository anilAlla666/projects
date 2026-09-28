# Headline mechanism proof: FP8 SDC detection UNDER CUDAGRAPH via MODULE-BUFFER input-mutation.
# In-process (to remove buffer-attach timing noise; the .so/default-MP DELIVERY is separately proven,
# 2304/98688 calls captured in the spawn worker). Patches the FP8 scheme BEFORE LLM() so every FP8
# layer gets cipher_res/cnt/inj BUFFERS at construction (lifted by dynamo as graph inputs -> mutation
# preserved through inductor + vLLM's manual cudagraph capture, P0C-proven). cudagraph ON.
import os, json, time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # in-process so the class patch lands before construction
import torch
import torch._inductor.config as ic
ic.triton.cudagraph_support_input_mutation = True

cl = torch.library.Library("cipherd", "FRAGMENT")
cl.define("sink(Tensor d, Tensor(a!) res, Tensor(b!) cnt) -> ()")
def _sink(d, res, cnt):
    torch.maximum(res, d.reshape(1), out=res); cnt.add_(1)
cl.impl("sink", _sink, "CompositeExplicitAutograd")
torch.library.register_fake("cipherd::sink", lambda d, res, cnt: None)

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
_LAYERS = []
_oc = CompressedTensorsW8A8Fp8.create_weights
def _create_weights(self, layer, *a, **k):
    r = _oc(self, layer, *a, **k)
    dev = "cuda"
    layer.register_buffer("cipher_res", torch.zeros(1, device=dev), persistent=False)
    layer.register_buffer("cipher_cnt", torch.zeros(1, device=dev), persistent=False)
    layer.register_buffer("cipher_inj", torch.zeros(1, device=dev), persistent=False)
    for t in (layer.cipher_res, layer.cipher_cnt, layer.cipher_inj):
        try: torch._dynamo.mark_static_address(t)
        except Exception: pass
    _LAYERS.append(layer)
    return r
CompressedTensorsW8A8Fp8.create_weights = _create_weights

_oa = CompressedTensorsW8A8Fp8.apply_weights
def _apply_weights(self, layer, x, bias=None):
    out = _oa(self, layer, x, bias)                       # real FP8 linear (returns out)
    out = out + layer.cipher_inj.to(out.dtype) * 0.5      # functional inject (between the two computes)
    out2 = _oa(self, layer, x, bias)                      # independent recompute (clean)
    d = (out2.float() - out.float()).abs().amax()
    torch.ops.cipherd.sink(d, layer.cipher_res, layer.cipher_cnt)   # -> module buffers (lifted as inputs)
    return out
CompressedTensorsW8A8Fp8.apply_weights = _apply_weights

from vllm import LLM, SamplingParams

def agg():
    mx = 0.0; cnt = 0
    for L in _LAYERS:
        mx = max(mx, float(L.cipher_res.item())); cnt += int(L.cipher_cnt.item())
    return mx, cnt

def main():
    M = os.environ["S_MODEL"]; OUT = os.environ["S_OUT"]
    llm = LLM(model=M, enforce_eager=False, gpu_memory_utilization=0.82,    # cudagraph ON
              max_model_len=2048, disable_log_stats=True)
    print(f"[cgmod] FP8 layers instrumented: {len(_LAYERS)}")
    sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True, min_tokens=128)
    prompts = ["The history of the Roman Empire begins with"]
    llm.generate(prompts, sp, use_tqdm=False)             # warm (capture)
    for L in _LAYERS: L.cipher_res.zero_(); L.cipher_cnt.zero_()
    o1 = llm.generate(prompts, sp, use_tqdm=False)        # CLEAN under cudagraph
    clean_res, clean_cnt = agg(); clean_out = o1[0].outputs[0].text[:45]
    # inject into ONE FP8 layer's compute (CSE-independence: corrupt out only, recompute is clean)
    tgt = _LAYERS[len(_LAYERS)//2]; tgt.cipher_inj.fill_(1.0)
    o2 = llm.generate(prompts, sp, use_tqdm=False)        # FAULT under cudagraph
    inj_res, _ = agg(); inj_out = o2[0].outputs[0].text[:45]
    res = {"model": M, "default_cudagraph": True, "mp": "in-process(proof)",
           "fp8_layers": len(_LAYERS), "clean_cnt": clean_cnt, "clean_res": round(clean_res, 6),
           "inj_res": round(inj_res, 6), "caught_under_cudagraph": inj_res > clean_res,
           "clean_out": clean_out, "inj_out": inj_out}
    json.dump(res, open(OUT, "w"), indent=1)
    print("CGMOD", json.dumps({k: res[k] for k in ["fp8_layers","clean_cnt","clean_res","inj_res","caught_under_cudagraph"]}))

if __name__ == "__main__":
    main()
