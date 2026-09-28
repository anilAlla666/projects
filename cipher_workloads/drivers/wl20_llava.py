"""WL20 LLaVA multimodal. Uses tiny synthetic image + text prompts."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "llava-hf/llava-1.5-7b-hf")

tenant = tenant_register.register("wl20")
try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from PIL import Image
except ImportError:
    print("llava deps missing; run setup.sh first", file=sys.stderr); sys.exit(2)

proc = AutoProcessor.from_pretrained(MODEL)
m = LlavaForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
img = Image.new("RGB", (336, 336), (128, 128, 128))
inputs = proc(images=img, text="USER: <image>\nWhat is this?\nASSISTANT:",
              return_tensors="pt").to("cuda", torch.float16)

def step():
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=20, do_sample=False)
    return 1, int(out.numel())

run_for_duration.run(step, DURATION, "WL20", tenant, "multimodal_passes")
