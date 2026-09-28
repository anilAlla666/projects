"""WL18 Multi-GPU Tensor Parallel.
DEFERRED on single-GPU pods. This driver checks for ≥2 GPUs and exits
gracefully if not present, marking the workload 'deferred-multi-gpu' in
the verify_run output."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register

tenant = tenant_register.register("wl18")
n_gpu = torch.cuda.device_count()
print(f"WL18: GPU count = {n_gpu}", file=sys.stderr, flush=True)
if n_gpu < 2:
    import time
    print("WL18: deferred-multi-gpu (pod has <2 GPUs)", file=sys.stderr, flush=True)
    t0 = time.time()
    with open(f"/tmp/cipher_tenant_{tenant}_progress", "w") as f:
        f.write(f"WL18 {tenant} start t={t0:.3f}\n")
        f.write(f"WL18 {tenant} deferred-multi-gpu n_gpu={n_gpu}\n")
    sys.exit(0)

# Real multi-GPU path would use torch.distributed init + TP-sharded model.
# Stub for the multi-GPU case; not exercised on this pod.
print("WL18: multi-GPU path not implemented in this driver; "
      "needs torch.distributed setup", file=sys.stderr, flush=True)
sys.exit(0)
