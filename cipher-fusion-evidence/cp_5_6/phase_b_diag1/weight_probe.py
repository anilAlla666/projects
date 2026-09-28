import os, sys, time, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register
tenant_register.register(os.environ.get("WL_TENANT_ID", "probe"))
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/models/TinyLlama-1.1B", dtype=torch.float16).cuda()
pb = sum(p.numel() * p.element_size() for p in m.parameters())
fp = next(m.parameters())
# sample a mid-stack weight tensor too
w = dict(m.named_parameters()).get("model.layers.10.mlp.gate_proj.weight")
print("PROBE pid=%d param_bytes=%.3fGB first_param_ptr=0x%x "
      "layer10_gate_ptr=%s torch_alloc=%.3fGB reserved=%.3fGB"
      % (os.getpid(), pb / 1e9, fp.data_ptr(),
         ("0x%x" % w.data_ptr()) if w is not None else "n/a",
         torch.cuda.memory_allocated() / 1e9,
         torch.cuda.memory_reserved() / 1e9), flush=True)
time.sleep(int(os.environ.get("PROBE_SLEEP", "30")))
