"""WL09 Whisper speech. Uses an internally-synthesized 60s audio waveform."""
import sys, os, numpy as np
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "openai/whisper-large-v3")

tenant = tenant_register.register("wl09")
try:
    import whisper
except ImportError:
    print("openai-whisper not installed; run setup.sh first", file=sys.stderr); sys.exit(2)

m = whisper.load_model(os.environ.get("WL_WHISPER_SIZE", "base")).cuda()
# 60 seconds of low-amplitude sine wave at 16kHz — placeholder audio.
audio = (0.01 * np.sin(2*np.pi*440*np.arange(16000*60)/16000)).astype(np.float32)

def step():
    r = m.transcribe(audio, fp16=True, verbose=False)
    return 1, len(r.get("text", ""))

run_for_duration.run(step, DURATION, "WL09", tenant, "transcribes")
