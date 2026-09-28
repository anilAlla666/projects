"""WL19 CLIP Vision."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "openai/clip-vit-base-patch32")

tenant = tenant_register.register("wl19")
from transformers import CLIPModel, CLIPProcessor
proc = CLIPProcessor.from_pretrained(MODEL)
m = CLIPModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
# 32 dummy 224×224 images (random tensors → minimal preprocess)
images = torch.rand(32, 3, 224, 224, dtype=torch.float16).cuda()

def step():
    with torch.no_grad():
        # transformers 5.x: get_image_features may return BaseModelOutputWithPooling
        # rather than a raw tensor. Use the vision_model directly + pooler_output.
        out = m.vision_model(pixel_values=images)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
    n = feats.numel() if hasattr(feats, "numel") else 32
    return 32, int(n)

run_for_duration.run(step, DURATION, "WL19", tenant, "image_passes")
