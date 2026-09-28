"""WL06 Embeddings. Substitution: paraphrase-MiniLM-L6-v2 (open)."""
import sys, os
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

tenant = tenant_register.register("wl06")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed; run setup.sh first", file=sys.stderr)
    sys.exit(2)

m = SentenceTransformer(MODEL, device="cuda")
sentences = ["This is sentence number " + str(i) for i in range(512)]

def step():
    emb = m.encode(sentences, batch_size=512, show_progress_bar=False)
    return 1, 512

run_for_duration.run(step, DURATION, "WL06", tenant, "batches")
