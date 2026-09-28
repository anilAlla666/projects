#!/usr/bin/env python3
"""Track 2 SC4 — unit tests for cipher_model_fingerprint.

Synthetic model directories (valid config.json + arbitrary-byte *.safetensors
— compute_fingerprint hashes file bytes, it does not parse the safetensors
format). Exercises tier-1/tier-2 discrimination, the large-shard sampling
window, the <3 MiB full-shard guard, and verify_fingerprint tier attribution.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/home/ubuntu/cipher-fusion-evidence/phase_c")
import cipher_model_fingerprint as cmf                        # noqa: E402

g_pass = g_fail = 0


def chk(cond, name):
    global g_pass, g_fail
    if cond:
        g_pass += 1
        print("  PASS  " + name)
    else:
        g_fail += 1
        print("  FAIL  " + name)


def mkmodel(d, config, shard_bytes, shard="model.safetensors"):
    os.makedirs(d, exist_ok=True)
    json.dump(config, open(os.path.join(d, "config.json"), "w"))
    with open(os.path.join(d, shard), "wb") as f:
        f.write(shard_bytes)
    return d


tmp = tempfile.mkdtemp(prefix="sc4unit_")
CFG_A = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 22,
         "hidden_size": 2048, "torch_dtype": "float16"}
CFG_B = dict(CFG_A, num_hidden_layers=32)        # different architecture
BIG = os.urandom(10 << 20)                        # 10 MiB shard (sampled)

print("Track 2 SC4 — fingerprint unit tests\n")

# 1 — deterministic
mkmodel(tmp + "/A", CFG_A, BIG)
fpA = cmf.compute_fingerprint(tmp + "/A")
chk(fpA["combined"] == cmf.compute_fingerprint(tmp + "/A")["combined"],
    "deterministic — same dir -> identical fingerprint")

# 2 — tier-1 mismatch (config differs, weights identical)
mkmodel(tmp + "/B", CFG_B, BIG)
fpB = cmf.compute_fingerprint(tmp + "/B")
chk(fpA["tier1"] != fpB["tier1"], "tier-1 differs on a config change")
chk(fpA["tier2"] == fpB["tier2"], "tier-2 unchanged when only config differs")
chk(fpA["combined"] != fpB["combined"], "combined differs (tier-1 mismatch)")

# 3 — tier-2 mismatch (config identical, weights differ)
big_c = bytearray(BIG)
big_c[100] ^= 0xFF                                # flip a byte in the head 1 MiB
mkmodel(tmp + "/C", CFG_A, bytes(big_c))
fpC = cmf.compute_fingerprint(tmp + "/C")
chk(fpA["tier1"] == fpC["tier1"], "tier-1 unchanged when only weights differ")
chk(fpA["tier2"] != fpC["tier2"], "tier-2 differs on a weight change")

# 4 — large-shard sampling: a byte in an UNSAMPLED gap -> tier-2 unchanged
big_d = bytearray(BIG)
big_d[(5 << 20) | 0x90000] ^= 0xFF                # ~5.56 MiB — between mid&tail
mkmodel(tmp + "/D", CFG_A, bytes(big_d))
chk(fpA["tier2"] == cmf.compute_fingerprint(tmp + "/D")["tier2"],
    "large shard: unsampled-gap byte -> tier-2 unchanged (sampling works)")

# 5 — small-shard guard: a <3 MiB shard is FULL-hashed -> any byte matters
SMALL = os.urandom(2 << 20)                       # 2 MiB < 3 MiB threshold
mkmodel(tmp + "/S", CFG_A, SMALL)
fpS = cmf.compute_fingerprint(tmp + "/S")
small2 = bytearray(SMALL)
small2[(1 << 20) | 0x80000] ^= 0xFF               # ~1.5 MiB — mid of the shard
mkmodel(tmp + "/S2", CFG_A, bytes(small2))
chk(fpS["tier2"] != cmf.compute_fingerprint(tmp + "/S2")["tier2"],
    "small shard <3MiB: full-hash -> a mid-shard byte change is detected")

# 6 — verify_fingerprint: match + tier-attributed mismatch
chk(cmf.verify_fingerprint({"fingerprint": fpA}, tmp + "/A") is True,
    "verify: matching model -> True")
try:
    cmf.verify_fingerprint({"fingerprint": fpA}, tmp + "/B")
    chk(False, "verify: tier-1 mismatch raises")
except cmf.FingerprintMismatch as e:
    chk(e.differing_tier == "1",
        "verify: tier-1 mismatch raises FingerprintMismatch(differing_tier=1)")
try:
    cmf.verify_fingerprint({"fingerprint": fpA}, tmp + "/C")
    chk(False, "verify: tier-2 mismatch raises")
except cmf.FingerprintMismatch as e:
    chk(e.differing_tier == "2",
        "verify: tier-2 mismatch raises FingerprintMismatch(differing_tier=2)")

print("\n=== SC4 UNIT: %d PASS, %d FAIL ===" % (g_pass, g_fail))
sys.exit(0 if g_fail == 0 else 1)
