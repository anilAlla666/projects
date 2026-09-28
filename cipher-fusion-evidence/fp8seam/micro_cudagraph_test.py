# Micro-test (no vLLM, fast): isolate the two questions before the full default-MP run.
# (1A) does impl('CUDA') override let us call the ORIGINAL (or does it recurse)?
# (1B-core) does a PURELY FUNCTIONAL handle-wrap (no host branch, residual -> static GPU buffer,
#           device inject flag) survive torch.compile + cudagraph replay?
import torch, vllm._custom_ops as ops

dev = "cuda"
M, K, N = 64, 4096, 512
a = (torch.randn(M, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
b = (torch.randn(N, K, device=dev) * 0.1).to(torch.float8_e4m3fn).t()   # [K,N] column-major (weight layout)
sa = torch.tensor(1.0, device=dev); sb = torch.tensor(1.0, device=dev)
_real = torch.ops._C.cutlass_scaled_mm

def ref():
    o = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    _real(o, a, b, sa, sb, None); return o
ref_out = ref().clone()
print("MICRO baseline op works, out norm", round(ref_out.float().norm().item(), 3))

# ---- 1A: impl('CUDA') override + recursion check ----
try:
    lib = torch.library.Library("_C", "FRAGMENT")
    state = {"recursed": False, "depth": 0}
    def override(out, a_, b_, sa_, sb_, bias=None):
        state["depth"] += 1
        if state["depth"] > 2: state["recursed"] = True; return
        return _real(out, a_, b_, sa_, sb_, bias)   # _real dispatches to CUDA -> our override again?
    lib.impl("cutlass_scaled_mm", override, "CUDA")
    o = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    try:
        torch.ops._C.cutlass_scaled_mm(o, a, b, sa, sb, None)
        print("RESULT 1A: override CALLED ok; recursed=", state["recursed"], "depth=", state["depth"],
              "| can-call-original=", not state["recursed"])
    except RecursionError:
        print("RESULT 1A: override RECURSES (cannot call original after CUDA-key override) -> 1A INFEASIBLE")
except Exception as e:
    print("RESULT 1A: override registration/call FAILED:", repr(e)[:200])

# ---- 1B-core: functional wrap under torch.compile + cudagraph ----
RES = torch.zeros(1, device=dev)     # residual buffer (static addr, read out-of-band)
INJ = torch.zeros(1, device=dev)     # device inject flag (host flips out-of-band)
_realpk = ops.__dict__.get("cutlass_scaled_mm")
def fwrap(out, a_, b_, sa_, sb_, bias=None):
    _real(out, a_, b_, sa_, sb_, bias)               # real compute (in-place)
    out.add_(INJ * 0.5)                              # functional inject (no-op when INJ==0)
    out2 = torch.empty_like(out)
    _real(out2, a_, b_, sa_, sb_, bias)              # independent recompute (clean)
    d = (out2.float() - out.float()).abs().amax().reshape(1)
    torch.maximum(RES, d, out=RES)                   # accumulate max residual, in-place to static buffer
    return

def model_step(a_, b_, sa_, sb_):
    o = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    fwrap(o, a_, b_, sa_, sb_, None)
    return o

try:
    cmodel = torch.compile(model_step, mode="reduce-overhead")  # inductor + cudagraph trees
    RES.zero_(); INJ.zero_()
    for _ in range(3): out_c = cmodel(a, b, sa, sb)              # warm + capture + replay (clean)
    torch.cuda.synchronize()
    clean_res = RES.item()
    INJ.fill_(1.0)                                               # host flips inject out-of-band
    out_c2 = cmodel(a, b, sa, sb); torch.cuda.synchronize()
    inj_res = RES.item()
    print(f"RESULT 1B-core: torch.compile+cudagraph SURVIVED. clean_residual={clean_res:.4g} "
          f"inject_residual={inj_res:.4g} caught={inj_res>clean_res}")
except Exception as e:
    import traceback
    print("RESULT 1B-core: FAILED under torch.compile:", repr(e)[:260])
    traceback.print_exc()
