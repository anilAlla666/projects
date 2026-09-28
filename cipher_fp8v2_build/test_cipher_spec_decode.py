"""CPU unit test for cipher_spec_decode.py pure logic — AdaptiveK,
NgramDraft, greedy_accept. No GPU, no torch. The GPU paths (ModelDraft,
spec_generate target forwards) are verified separately on GPU."""
import sys
import cipher_spec_decode as sd

ok = 0


def check(cond, label):
    global ok
    assert cond, "FAIL: " + label
    print("  ok:  " + label)
    ok += 1


print("=== cipher_spec_decode pure-logic unit test ===")

# --- AdaptiveK ---
ak = sd.AdaptiveK(window=64)
check(ak.next_k() == round(3.0 / 0.7), "AdaptiveK seed k = round(3.0/0.7) = 4")
check(2 <= ak.next_k() <= 8, "AdaptiveK k within [2,8] clamp")

ak_hi = sd.AdaptiveK(window=8)
for _ in range(200):
    ak_hi.update(5, 5)                       # acceptance rate 1.0
check(ak_hi.ema_rate > 0.97, "AdaptiveK EMA tracks high acceptance -> ~1.0")
check(ak_hi.next_k() == 3, "AdaptiveK high acceptance -> k = round(3/1.0) = 3")

ak_lo = sd.AdaptiveK(window=8)
for _ in range(200):
    ak_lo.update(0, 5)                       # acceptance rate 0.0
check(ak_lo.ema_rate < 0.03, "AdaptiveK EMA tracks low acceptance -> ~0")
check(ak_lo.next_k() == 8, "AdaptiveK low acceptance -> k clamps to k_max 8")

# --- NgramDraft (prompt-lookup) ---
ng = sd.NgramDraft(ngram_n=3)
seq = [1, 2, 3, 7, 8, 9, 4, 5, 1, 2, 3]      # last [1,2,3] recurs at idx 0
check(ng.propose(seq, 3) == [7, 8, 9], "NgramDraft proposes continuation [7,8,9]")
check(ng.propose(seq, 2) == [7, 8], "NgramDraft respects k (k=2 -> [7,8])")
check(ng.propose([1, 2, 3, 4, 5], 3) == [], "NgramDraft no-match -> []")
check(ng.propose([1, 2], 3) == [], "NgramDraft too-short sequence -> []")

# --- greedy_accept ---
acc, bonus, n = sd.greedy_accept([5, 6, 7, 8], [5, 6, 7])
check((acc, bonus, n) == ([5, 6, 7], 8, 3), "greedy_accept full accept -> n=3, bonus=8")
acc, bonus, n = sd.greedy_accept([5, 9, 7, 8], [5, 6, 7])
check((acc, bonus, n) == ([5], 9, 1), "greedy_accept reject@1 -> n=1, bonus=9")
acc, bonus, n = sd.greedy_accept([9, 6, 7, 8], [5, 6, 7])
check((acc, bonus, n) == ([], 9, 0), "greedy_accept reject@0 -> n=0, bonus=9")

# --- greedy_accept edge cases ---
acc, bonus, n = sd.greedy_accept([99], [])
check((acc, bonus, n) == ([], 99, 0), "greedy_accept empty draft -> n=0, bonus only")
acc, bonus, n = sd.greedy_accept([5, 7], [5])
check((acc, bonus, n) == ([5], 7, 1), "greedy_accept k=1 accept -> n=1")
acc, bonus, n = sd.greedy_accept([9, 7], [5])
check((acc, bonus, n) == ([], 9, 0), "greedy_accept k=1 reject -> n=0")
acc, bonus, n = sd.greedy_accept([9, 9, 9, 9], [1, 2, 3])
check((acc, bonus, n) == ([], 9, 0), "greedy_accept all-rejected -> n=0 (breaks @0)")
acc, bonus, n = sd.greedy_accept([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8])
check(n == 8 and acc == [1, 2, 3, 4, 5, 6, 7, 8] and bonus == 9,
      "greedy_accept all-accepted at k_max=8 boundary -> n=8")

# --- AdaptiveK regime behaviour ---
ak_mid = sd.AdaptiveK(window=8)
for _ in range(200):
    ak_mid.update(2, 4)                      # acceptance rate 0.5
check(abs(ak_mid.ema_rate - 0.5) < 0.02, "AdaptiveK EMA tracks mid acceptance -> ~0.5")
check(ak_mid.next_k() == round(3.0 / 0.5), "AdaptiveK mid regime -> k = round(3/0.5) = 6")

ak_mix = sd.AdaptiveK(window=16)
import random as _r
_r.seed(0)
for _ in range(300):
    kp = _r.randint(2, 8)
    ak_mix.update(_r.randint(0, kp), kp)
    check_k = ak_mix.next_k()
    assert 2 <= check_k <= 8, "FAIL: AdaptiveK k escaped [2,8] under mixed regime"
check(True, "AdaptiveK k stays in [2,8] across 300 mixed-rate rounds")

ak_seed = sd.AdaptiveK(window=64)
r0 = ak_seed.ema_rate
ak_seed.update(5, 5)
check(ak_seed.ema_rate > r0, "AdaptiveK one high-accept update moves EMA up from seed")

print("=== ALL PASS (%d checks) ===" % ok)
sys.exit(0)
