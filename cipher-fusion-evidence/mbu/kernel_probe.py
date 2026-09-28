import os, torch
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from vllm.model_executor.kernels.linear import choose_mp_linear_kernel, MPLinearLayerConfig
from vllm.scalar_type import scalar_types
cfg=MPLinearLayerConfig(full_weight_shape=(4096,4096),partition_weight_shape=(4096,4096),
    weight_type=scalar_types.uint4b8, act_type=torch.float16, group_size=128,
    zero_points=False, has_g_idx=False)
try:
    k=choose_mp_linear_kernel(cfg); print("CHOSEN_KERNEL:", k.__name__)
except Exception as e: print("ERR", str(e)[:160])
