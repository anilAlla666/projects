"""E4 — faster-whisper transcribe 60s audio."""
import os, sys, json, time, ctypes, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"


def make_audio(path):
    """Generate 60s of TTS-like audio with ffmpeg's lavfi sine + a Buddhist chant
    if neither, fall back to a synthesised 440Hz tone (Whisper will return empty,
    but the API still runs)."""
    # Use ffmpeg to make 60s of multi-tone audio.
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
           "-f", "lavfi", "-i", "sine=frequency=880:duration=10",
           "-f", "lavfi", "-i", "sine=frequency=660:duration=10",
           "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1",
           "-ar", "16000", "-ac", "1", path]
    subprocess.run(cmd, check=True)


def main():
    if USE_CIPHER:
        rt = ctypes.CDLL(str(sc.ROOT / "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
        rt.cipher_fp8_compute_init.restype = ctypes.c_int
        rt.cipher_fp8_compute_init()

    import faster_whisper
    print(f"[E4] cipher={USE_CIPHER}", flush=True)

    audio_path = os.path.join(tempfile.gettempdir(), "e4_test.wav")
    if not os.path.exists(audio_path):
        try:
            make_audio(audio_path)
        except Exception as e:
            print(f"  ffmpeg failed: {e} — generating numpy WAV instead", flush=True)
            import numpy as np, wave
            sr = 16000
            t = np.arange(60*sr) / sr
            sig = 0.3 * np.sin(2*np.pi*440*t).astype(np.float32)
            with wave.open(audio_path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((sig * 32767).astype("<i2").tobytes())

    model = faster_whisper.WhisperModel("small", device="cuda", compute_type="float16")
    t0 = time.perf_counter()
    segments, info = model.transcribe(audio_path, language="en")
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)
    elapsed = time.perf_counter() - t0
    full_text = " ".join(text_parts)
    suffix = "_baseline" if not USE_CIPHER else ""
    payload = dict(cipher=USE_CIPHER, audio_dur_s=info.duration,
                   elapsed_s=elapsed, rtf=info.duration/elapsed if elapsed>0 else 0,
                   text_len=len(full_text), text=full_text[:200])
    with open(os.path.join(os.path.dirname(__file__), f"e4_whisper{suffix}.json"),
              "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[E4] {info.duration:.1f}s audio in {elapsed:.2f}s "
          f"(RTF={info.duration/elapsed:.1f}x)", flush=True)
    print(f"[E4] text: {full_text[:120]!r}", flush=True)


if __name__ == "__main__":
    main()
