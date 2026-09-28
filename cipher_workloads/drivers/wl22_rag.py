"""WL22 RAG Pipeline. Embeddings + retrieval + LLM (TinyLlama + MiniLM)."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
EMBED_MODEL = os.environ.get("WL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.environ.get("WL_GEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl22")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed; run setup.sh first", file=sys.stderr); sys.exit(2)
from transformers import AutoModelForCausalLM, AutoTokenizer

emb = SentenceTransformer(EMBED_MODEL, device="cuda")
tok = AutoTokenizer.from_pretrained(GEN_MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
gen = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float16).cuda().eval()

corpus = ["Paris is the capital of France.",
          "Tokyo is the capital of Japan.",
          "Berlin is the capital of Germany."]
corpus_emb = emb.encode(corpus)
query = "What is the capital of France?"

def step():
    q_emb = emb.encode([query])
    # naive top-1 cosine retrieval
    import numpy as np
    sims = q_emb @ corpus_emb.T
    top = corpus[int(np.argmax(sims))]
    prompt = f"Context: {top}\nQuestion: {query}\nAnswer:"
    inp = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = gen.generate(**inp, max_new_tokens=20, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return 1, int(out.shape[-1])

run_for_duration.run(step, DURATION, "WL22", tenant, "rag_queries")
