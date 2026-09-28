"""CPU integration test for spec_generate — deterministic mock target + mock
drafts, no GPU. Catches the verify-loop bugs (logit/draft alignment, bonus
index, KV crop) that the pure-helper unit tests can't: the mock target tracks
its seen sequence via past, so a wrong KV crop makes its output diverge.

The invariant under test: for ANY draft, in ANY acceptance regime, spec_generate
output is byte-identical to plain target greedy decoding (the temp-0 gate)."""
import sys
import torch
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import cipher_spec_decode as sd

VOCAB = 50


def RULE(seq):
    """Deterministic Markov-2 target: next token depends on the last two."""
    a = seq[-1]
    b = seq[-2] if len(seq) >= 2 else 0
    return (a * 7 + b * 3 + 1) % VOCAB


class MockPast:
    """Stand-in KV cache: just the token list it 'stores'. crop() follows the
    transformers Cache convention (negative arg = drop that many from the end)."""
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def crop(self, x):
        n = len(self.tokens)
        self.tokens = self.tokens[:(n + x if x < 0 else x)]

    def get_seq_length(self):
        return len(self.tokens)


class MockTarget:
    """Autoregressive RULE model. logits[i] one-hots RULE(seq-seen-up-to-i).
    `seq-seen` = past.tokens + input_ids — so if spec_generate crops the KV
    wrong, past.tokens diverges from the true sequence and the output breaks."""
    def __init__(self):
        self.config = type("C", (), {"eos_token_id": None})()

    def parameters(self):
        return iter([torch.zeros(1)])          # -> device == cpu

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        prior = list(past_key_values.tokens) if past_key_values is not None else []
        inp = input_ids[0].tolist()
        full = prior + inp
        logits = torch.zeros(1, len(inp), VOCAB)
        for i in range(len(inp)):
            logits[0, i, RULE(full[:len(prior) + i + 1])] = 10.0
        return type("R", (), {"logits": logits,
                              "past_key_values": MockPast(full)})()


def target_only(prompt, n):
    """Plain greedy decoding with the same RULE — the byte-identical baseline."""
    seq = list(prompt)
    for _ in range(n):
        seq.append(RULE(seq))
    return seq


class FixedK:
    def __init__(self, k): self.k = k
    def next_k(self): return self.k
    def update(self, *a): pass


class OracleDraft:          # proposes the exact RULE continuation (all accept)
    def propose(self, seq, k):
        s = list(seq); out = []
        for _ in range(k):
            s.append(RULE(s)); out.append(s[-1])
        return out


class WrongDraft:           # proposes guaranteed-wrong tokens (all reject)
    def propose(self, seq, k):
        s = list(seq); out = []
        for _ in range(k):
            c = RULE(s); s.append(c); out.append((c + 1) % VOCAB)
        return out


class PartialDraft:         # first `good` correct, rest wrong
    def __init__(self, good): self.good = good
    def propose(self, seq, k):
        s = list(seq); out = []
        for i in range(k):
            c = RULE(s); s.append(c)
            out.append(c if i < self.good else (c + 1) % VOCAB)
        return out


ok = 0


def check(cond, label):
    global ok
    assert cond, "FAIL: " + label
    print("  ok:  " + label)
    ok += 1


print("=== spec_generate CPU integration test ===")
P = [3, 11, 7]

# 1. all-accepted at k_max = 8
base40 = target_only(P, 40)
st = {}
out = sd.spec_generate(MockTarget(), OracleDraft(), P, 40,
                       adaptive_k=FixedK(8), stats=st)
check(out == base40, "all-accepted k=8: byte-identical to target-only")
check(st["proposed"] > 0 and st["accepted"] == st["proposed"],
      "all-accepted k=8: acceptance rate = 1.0 (regime exercised)")

# 2. all-rejected at k = 2 (boundary minimum)
st = {}
out = sd.spec_generate(MockTarget(), WrongDraft(), P, 40,
                       adaptive_k=FixedK(2), stats=st)
check(out == base40, "all-rejected k=2: byte-identical (bonus path carries it)")
check(st["proposed"] > 0 and st["accepted"] == 0,
      "all-rejected k=2: acceptance rate = 0 (regime exercised)")

# 3. partial accept at varying n
for good in (1, 3, 5):
    out = sd.spec_generate(MockTarget(), PartialDraft(good), P, 40,
                           adaptive_k=FixedK(6))
    check(out == base40, "partial accept (good=%d of k=6): byte-identical" % good)

# 4. multi-round — long generation, state must carry across ~40 rounds
base200 = target_only(P, 200)
out = sd.spec_generate(MockTarget(), OracleDraft(), P, 200, adaptive_k=FixedK(5))
check(out == base200, "multi-round (200 tokens): cross-round state byte-identical")

# 5. token agreement vs target-only at temp 0 — every draft, default AdaptiveK
base60 = target_only(P, 60)
for name, dr in (("oracle", OracleDraft()), ("wrong", WrongDraft()),
                 ("partial2", PartialDraft(2)), ("ngram", sd.NgramDraft(3))):
    out = sd.spec_generate(MockTarget(), dr, P, 60)
    check(out == base60, "token-agreement temp0 [%s]: byte-identical" % name)

print("=== ALL PASS (%d checks) ===" % ok)
sys.exit(0)
