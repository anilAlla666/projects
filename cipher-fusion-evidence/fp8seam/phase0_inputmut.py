# Phase 0 micro (no vLLM): does the INPUT-MUTATION channel surface a residual host-side under
# STRICT cudagraph (fullgraph + cudagraph trees)? The prior failure used a CLOSED-OVER GLOBAL buffer
# (DCE'd). Here RES/CNT/INJ are DECLARED INPUTS, mutated in-place via a custom op (mutates_args),
# with torch._inductor.config.cudagraph_support_input_mutation=True.
import torch
import torch._inductor.config as ic
ic.triton.cudagraph_support_input_mutation = True     # the documented input-mutation gate (config.py:1534, default True)
import vllm._custom_ops  # registers _C::cutlass_scaled_mm

dev = "cuda"
# residual side-channel op: mutates res,cnt which are INPUTS (not globals)
cl = torch.library.Library("cipherp0", "FRAGMENT")
cl.define("sink(Tensor d, Tensor(a!) res, Tensor(b!) cnt) -> ()")
def _sink(d, res, cnt):
    torch.maximum(res, d.reshape(1), out=res); cnt.add_(1)
cl.impl("sink", _sink, "CompositeExplicitAutograd")
torch.library.register_fake("cipherp0::sink", lambda d, res, cnt: None)

_real = torch.ops._C.cutlass_scaled_mm
M, K, N = 64, 4096, 512
a = (torch.randn(M, K, device=dev) * 0.1).to(torch.float8_e4m3fn)
b = (torch.randn(N, K, device=dev) * 0.1).to(torch.float8_e4m3fn).t()
sa = torch.tensor(1.0, device=dev); sb = torch.tensor(1.0, device=dev)

def f(a, b, sa, sb, res, cnt, inj):
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    _real(out, a, b, sa, sb, None)
    out = out + inj.to(out.dtype) * 0.5           # inject corrupts THIS compute's output (between the two)
    out2 = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    _real(out2, a, b, sa, sb, None)               # independent recompute (clean)
    d = (out2.float() - out.float()).abs().amax()
    torch.ops.cipherp0.sink(d, res, cnt)          # residual -> INPUT buffers, in-place
    return out

RES = torch.zeros(1, device=dev); CNT = torch.zeros(1, device=dev); INJ = torch.zeros(1, device=dev)
for t in (RES, CNT, INJ): torch._dynamo.mark_static_address(t)

cf = torch.compile(f, fullgraph=True, mode="reduce-overhead")   # cudagraph trees + fullgraph (strict)
# clean replays
RES.zero_(); CNT.zero_(); INJ.zero_()
for _ in range(3): o = cf(a, b, sa, sb, RES, CNT, INJ)
torch.cuda.synchronize()
clean_res = RES.item(); clean_cnt = CNT.item()
# inject via the INPUT tensor (mutate INJ's memory; cudagraph reads it each replay)
INJ.fill_(1.0)
o = cf(a, b, sa, sb, RES, CNT, INJ); torch.cuda.synchronize()
inj_res = RES.item(); inj_cnt = CNT.item()
print("P0", {"channel": "input-mutation", "config_flag": ic.triton.cudagraph_support_input_mutation,
             "clean_cnt": clean_cnt, "clean_res": round(clean_res, 5),
             "inj_cnt": inj_cnt, "inj_res": round(inj_res, 5),
             "cnt_increments": inj_cnt > 0, "caught": inj_res > clean_res})

# ---- CSE-independence guard: perturb out2's OPERAND only; residual must reflect a genuine
# second-kernel divergence (proves the two _real calls are not folded into one) ----
def fcse(a, b, sa, sb, res, cnt, ap):
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev); _real(out, a, b, sa, sb, None)
    out2 = torch.empty((M, N), dtype=torch.bfloat16, device=dev); _real(out2, ap, b, sa, sb, None)  # perturbed operand
    d = (out2.float() - out.float()).abs().amax()
    torch.ops.cipherp0.sink(d, res, cnt); return out
ap = (a.float() * 1.05).to(torch.float8_e4m3fn)   # 5% operand perturbation for the recompute only
cf2 = torch.compile(fcse, fullgraph=True, mode="reduce-overhead")
RES.zero_(); CNT.zero_()
for _ in range(3): cf2(a, b, sa, sb, RES, CNT, ap)
torch.cuda.synchronize()
print("P0-CSE", {"operand_perturb_residual": round(RES.item(), 5),
                 "independent_kernels": RES.item() > 0})
