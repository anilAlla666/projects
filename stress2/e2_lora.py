"""E2 — LoRA fine-tuning sanity check.

Llama-3.2-1B + LoRA r=16, 50 steps on a tiny synthetic dataset.
Reports loss, NaN check, no crash.
"""
import os, sys, json, time, ctypes, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
N_STEPS = int(os.environ.get("N_STEPS", "50"))


def main():
    if USE_CIPHER:
        rt = ctypes.CDLL(str(sc.ROOT / "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
        rt.cipher_fp8_compute_init.restype = ctypes.c_int
        rt.cipher_fp8_compute_init()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    print(f"[E2] cipher={USE_CIPHER}", flush=True)

    P1B = "/home/ubuntu/models/Llama-3.2-1B"
    tok = AutoTokenizer.from_pretrained(P1B)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        P1B, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj","v_proj"],
                          task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_cfg)
    # cast LoRA params to fp32 for stable training
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            p.data = p.data.float()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4)

    # synthetic data
    text = ("Energy efficiency is critical for modern AI systems. "
            "The future of GPU computing is to make every joule count. ") * 4
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to("cuda:0")
    inp_ids = enc.input_ids
    labels = inp_ids.clone()

    losses = []
    nans = 0
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        outputs = model(input_ids=inp_ids, attention_mask=enc.attention_mask,
                        labels=labels)
        loss = outputs.loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            nans += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.item()))
        if step < 5 or step % 10 == 0:
            print(f"  step {step:>3} loss={loss.item():.4f}", flush=True)
    elapsed = time.perf_counter() - t0
    converging = losses[-1] < losses[0]

    suffix = "_baseline" if not USE_CIPHER else ""
    payload = dict(cipher=USE_CIPHER, steps=N_STEPS, elapsed_s=elapsed,
                   loss_first=losses[0], loss_last=losses[-1],
                   loss_min=min(losses),
                   converging=converging, nans=nans)
    with open(os.path.join(os.path.dirname(__file__), f"e2_lora{suffix}.json"),
              "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[E2] loss {losses[0]:.3f} -> {losses[-1]:.3f} "
          f"(min {min(losses):.3f}) nans={nans} converging={converging}", flush=True)


if __name__ == "__main__":
    main()
