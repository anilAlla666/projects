import os, sys, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), file=sys.stderr)
a = torch.randn(512, 4096, dtype=torch.float16, device="cuda")
w = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
for _ in range(20):
    c = (a @ w)
torch.cuda.synchronize()
print("matmul done, c.sum=", float(c.float().sum()), file=sys.stderr)
