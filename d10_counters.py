import ctypes, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
lib = ctypes.CDLL("/usr/lib/cipher/libcipher_rt.so")
lib.cipher_rt_cublaslt_variant_calls.restype=ctypes.c_ulong
lib.cipher_rt_cublaslt_variant_calls.argtypes=[ctypes.c_char_p]
for f in ("cipher_rt_cublaslt_variants_aggregate","cipher_rt_cublaslt_shim_calls","cipher_rt_cublas_shim_calls"):
    try: getattr(lib,f).restype=ctypes.c_ulong
    except: pass
M="/home/ubuntu/models/Mistral-7B-v0.1"
tok=AutoTokenizer.from_pretrained(M); tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).cuda().eval()
ids=tok(["The history of computing spans"]*8, return_tensors="pt", padding=True).input_ids.cuda()
with torch.no_grad():
    for _ in range(5): m.generate(ids, max_new_tokens=4, do_sample=False)
torch.cuda.synchronize()
hsh=lib.cipher_rt_cublaslt_variant_calls(b"HSH")
agg=lib.cipher_rt_cublaslt_variants_aggregate()
umb=lib.cipher_rt_cublaslt_shim_calls() if hasattr(lib,"cipher_rt_cublaslt_shim_calls") else -1
ge =lib.cipher_rt_cublas_shim_calls() if hasattr(lib,"cipher_rt_cublas_shim_calls") else -1
print("D10COUNTERS HSH_shim_calls=%d variants_aggregate=%d umbrella_ltMatmul=%d gemmEx=%d"%(hsh,agg,umb,ge))
