"""WL08 Diffusion SDXL. Requires diffusers + SDXL weights."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

tenant = tenant_register.register("wl08")
try:
    from diffusers import StableDiffusionXLPipeline
except ImportError:
    print("diffusers not installed; run setup.sh first", file=sys.stderr); sys.exit(2)

p = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16,
                                              variant="fp16", use_safetensors=True).to("cuda")
p.set_progress_bar_config(disable=True)

def step():
    img = p(prompt="A red apple", num_inference_steps=20, guidance_scale=7.5).images[0]
    return 1, 1  # ops_completed, "tokens" repurposed as images

run_for_duration.run(step, DURATION, "WL08", tenant, "images")
