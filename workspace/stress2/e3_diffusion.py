"""E3 — diffusion. SDXL-turbo 20 images 512x512.

Verifies CIPHER doesn't crash diffusers and images are not all-black/white/noise.
"""
import os, sys, json, time, ctypes, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
N = int(os.environ.get("N_IMAGES", "20"))


def main():
    if USE_CIPHER:
        rt = ctypes.CDLL(str(sc.ROOT / "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
        rt.cipher_fp8_compute_init.restype = ctypes.c_int
        rt.cipher_fp8_compute_init()
    import torch
    from diffusers import AutoPipelineForText2Image
    print(f"[E3] cipher={USE_CIPHER} N={N}", flush=True)

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to("cuda:0")

    prompts = [f"a cinematic photograph of a {x}, ultra detailed, 8k"
               for x in ["lone tree", "snow leopard", "vintage car",
                         "city skyline", "lighthouse"] * 4]
    prompts = prompts[:N]

    failures = 0
    bad = 0
    rows = []
    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        try:
            img = pipe(prompt=p, num_inference_steps=1, guidance_scale=0.0,
                       height=512, width=512).images[0]
            arr = np.array(img)
            mn, mx, std = arr.min(), arr.max(), arr.std()
            # Reject all-black, all-white, pure-noise.
            ok = (mx - mn > 30) and (std > 5) and (std < 120)
            if not ok: bad += 1
            rows.append(dict(i=i, prompt=p[:50], min=int(mn), max=int(mx),
                             std=float(std), ok=ok))
            if i < 3 or i % 5 == 0:
                print(f"  [{i:>2}] mn={mn} mx={mx} std={std:.1f} ok={ok}",
                      flush=True)
        except Exception as e:
            failures += 1
            rows.append(dict(i=i, error=f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"  [{i}] error: {e}", flush=True)
    elapsed = time.perf_counter() - t0
    suffix = "_baseline" if not USE_CIPHER else ""
    payload = dict(cipher=USE_CIPHER, N=N, failures=failures, bad_images=bad,
                   elapsed_s=elapsed, images_per_s=(N-failures)/elapsed
                                     if elapsed > 0 else 0,
                   samples=rows)
    with open(os.path.join(os.path.dirname(__file__), f"e3_diffusion{suffix}.json"),
              "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[E3] {N-failures}/{N} ok ({bad} bad images) in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
