"""E7 — agentic multi-turn: 20 conversations × 5 turns, growing context."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"


SYSTEM = ("You are an expert AI assistant. Answer concisely. "
          "Refer back to earlier turns when relevant.")
TURN_TOPICS = [
    ("Hi! Can you help plan a 3-day trip to Japan?",
     "Day 1, what should I see in Tokyo?",
     "What about Day 2 in Kyoto?",
     "Anything special for the third day?",
     "Summarise the whole plan in one paragraph."),
    ("I'm building a Python web service. What framework should I use?",
     "Pros and cons of FastAPI vs Flask?",
     "Show me a minimal FastAPI hello-world.",
     "How do I add a database with SQLAlchemy?",
     "Write a final summary of my stack."),
]


def main():
    rt, fp8_stats, fus_stats = (sc.init_cipher() if USE_CIPHER
                                else (None, lambda: {}, lambda: {}))
    import torch
    print(f"[E7] cipher={USE_CIPHER}", flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)

    rows = []
    coh_at_turn5 = 0
    for conv_i in range(20):
        topic = TURN_TOPICS[conv_i % len(TURN_TOPICS)]
        ctx = f"<|system|>\n{SYSTEM}\n"
        last_text = ""
        try:
            for turn_i, user_msg in enumerate(topic):
                ctx += f"<|user|>\n{user_msg}\n<|assistant|>\n"
                ids = tok(ctx, return_tensors="pt").input_ids.to("cuda:0")
                attn = torch.ones_like(ids)
                with torch.no_grad():
                    out = model.generate(ids, attention_mask=attn,
                                         max_new_tokens=64, do_sample=False,
                                         pad_token_id=tok.pad_token_id, use_cache=True)
                torch.cuda.synchronize()
                resp = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                ctx += resp + "\n"
                last_text = resp
            coherent5 = "!!!!" not in last_text and len(last_text.strip()) > 10
            coh_at_turn5 += int(coherent5)
            rows.append(dict(conv=conv_i, turn5_text=last_text[:120],
                             coherent=coherent5))
            if conv_i < 3 or conv_i % 5 == 0:
                print(f"  [conv {conv_i:>2}] turn5 coh={coherent5} "
                      f"text={last_text[:80]!r}", flush=True)
        except Exception as e:
            rows.append(dict(conv=conv_i, error=f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"  [conv {conv_i}] err: {e}", flush=True)

    suffix = "_baseline" if not USE_CIPHER else ""
    payload = dict(cipher=USE_CIPHER, conversations=20,
                   turn5_coherent=coh_at_turn5, samples=rows)
    with open(os.path.join(os.path.dirname(__file__), f"e7_agentic{suffix}.json"),
              "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[E7] turn5 coherent={coh_at_turn5}/20", flush=True)


if __name__ == "__main__":
    main()
