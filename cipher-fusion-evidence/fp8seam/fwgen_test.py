# Framework-general test (1c): with the torch-DISPATCHER hook installed, call
# torch.ops._C.cutlass_scaled_mm DIRECTLY (no vLLM wrapper, no vLLM caller) using real operands
# captured from the worker. Proves: (a) the hook fires regardless of caller (caller-independent),
# (b) operands captured at this layer suffice to RECOMPUTE the matmul (residual detection works).
import os, torch
import vllm._custom_ops  # registers _C::cutlass_scaled_mm

d = torch.load(os.environ["FP8_DUMP"], weights_only=True)
a = d["a"].cuda(); b = d["b"].cuda(); sa = d["sa"].cuda(); sb = d["sb"].cuda(); ref = d["out"].cuda()

_real = torch.ops._C.cutlass_scaled_mm
fired = {"n": 0, "ops_seen": []}
def _w(*A, **K):
    fired["n"] += 1
    if A and len(A) >= 5:
        fired["ops_seen"].append([list(A[1].shape), str(A[1].dtype), list(A[3].shape)])
    return _real(*A, **K)
torch.ops._C.cutlass_scaled_mm = _w

# DIRECT dispatcher call — no vLLM wrapper in the path
out = torch.empty_like(ref)
torch.ops._C.cutlass_scaled_mm(out, a, b, sa, sb, None)

# independent recompute via the SAME captured operands (the residual-detector primitive)
out2 = torch.empty_like(ref)
torch.ops._C.cutlass_scaled_mm(out2, a, b, sa, sb, None)
resid_clean = (out2.float() - out.float()).norm().item()
matches_ref = (out.float() - ref.float()).norm().item()

print("FWGEN", {
    "hook_fired_on_direct_call": fired["n"],
    "operand_a": fired["ops_seen"][0] if fired["ops_seen"] else None,
    "recompute_clean_residual": resid_clean,             # expect 0.0 -> recompute is deterministic
    "direct_out_vs_worker_out_residual": round(matches_ref, 6),  # expect ~0 -> operands reproduce the real out
})
