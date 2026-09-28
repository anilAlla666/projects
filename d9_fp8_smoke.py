import torch, os
print("torch", torch.__version__, "cuda", torch.version.cuda, "FP8 env", os.environ.get("CIPHER_FP8"))
dev='cuda'
# bias=True => addmm => PUBLIC cublasGemmEx (the path CIPHER GOT-patches); batch=8192 > 64 => FP8 gate
lin = torch.nn.Linear(4096, 14336, bias=True).to(dev).bfloat16()
x = torch.randn(8192, 4096, dtype=torch.bfloat16, device=dev)
for i in range(6):
    y = lin(x)
torch.cuda.synchronize()
print("OUT", tuple(y.shape), y.dtype, "mean", float(y.float().mean()))
