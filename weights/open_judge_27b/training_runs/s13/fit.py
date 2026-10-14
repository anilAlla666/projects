"""Fit test: QLoRA forward+backward on Qwen3.8-27B at sequence length L, activations offloaded to CPU."""
import sys, time, torch, contextlib
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
P = "/home/ubuntu/models/qwen38-27b"
lens = [int(x) for x in sys.argv[1].split(",")]; offload = sys.argv[2] == "1"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["lm_head", "visual", "mtp"])
t = time.time()
m = AutoModelForCausalLM.from_pretrained(P, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
print(type(m).__name__, "loaded", round(time.time() - t), "s", torch.cuda.memory_allocated() / 2**30, "GiB", flush=True)
names = sorted({n.split(".")[-1] for n, mod in m.named_modules() if mod.__class__.__name__ in ("Linear4bit", "Linear")
                and "visual" not in n and "lm_head" not in n and "mtp" not in n})
print("linear names", names, flush=True)
m.train()
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
m.enable_input_require_grads()
import torch.utils.checkpoint as _ck
TILE = int(sys.argv[3]) if len(sys.argv) > 3 else 0
def _tiled(fwd, chunk):
    def f(x):
        if x.shape[1] <= chunk: return fwd(x)
        return torch.cat([_ck.checkpoint(fwd, x[:, i:i + chunk], use_reentrant=False) for i in range(0, x.shape[1], chunk)], 1)
    return f
m = get_peft_model(m, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=names))
m.print_trainable_parameters()
if TILE:
    for n, mod in m.named_modules():
        if n.endswith(".mlp") and hasattr(mod, "gate_proj"):
            mod.forward = _tiled(mod.forward, TILE)
    print("tiled mlp", TILE, flush=True)
base = m.base_model.model
inner = base.model.language_model if hasattr(base.model, "language_model") else base.model
head = base.lm_head
ce = LigerFusedLinearCrossEntropyLoss()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
for L in lens:
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    ids = torch.randint(1000, 100000, (1, L), device="cuda")
    labels = ids.clone(); labels[:, : L * 3 // 4] = -100
    try:
        t = time.time()
        ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else contextlib.nullcontext()
        with ctx:
            h = inner(input_ids=ids).last_hidden_state
            loss = ce(head.weight, h[0, :-1].contiguous(), labels[0, 1:].contiguous())
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
        print(f"L={L} offload={offload} ok loss={loss.item():.3f} peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB step={time.time()-t:.1f}s tok/s={L/(time.time()-t):.0f}", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"L={L} offload={offload} OOM peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
        opt.zero_grad(set_to_none=True); del e; torch.cuda.empty_cache()
