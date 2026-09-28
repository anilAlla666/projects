# V0 G-O1 ENGINE INCREMENT 3b: residency-swap correctness SUSTAINED at REQ=200 (mechanism OPEN)

**2026-06-03. Swap correctness SUSTAINED at scale via re-capture-per-serve — but the underlying corruption is NOT
root-caused (honest, per discipline: do not restate a symptom as a cause).** NO CIPHER `.so` source change; fix is
in `cipher_router.py` only; inc-1 module `cipher_engine.py` BYTE-IDENTICAL, inc-1 + inc-2 gates PASS; deployed anchor
1f305ce6 + staging UNCHANGED. Run `CIPHER_RT_DISABLE_AUTO_INIT=1`. Artifacts: `pager_g3b_{isolate,ptr}.py`,
`cipher_router{,_gate}.py`.

## Correctness — SUSTAINED at REQ=200 (the earned result)

`cipher_router_gate.py` REQ=200 Zipfian model-popularity stream, M=4 distinct models, K=2 resident (forces swaps),
re-capture-per-serve:
- **per-request KL=0: exact 200/200, FAULT=0** (vs the un-fixed path which progressively diverges then HARD-FAULTS).
- **84 swaps, all physically real** (HBM freed >0.5GB each); **across-swap KL=0 = 84/84**.
- **cross-model misroute negative control DETECTED** (agent-A output vs different-agent solo → FAULT).
- **GPU returns to 0 MiB** — no HBM accumulation/leak from re-capture's churned graphs over 200 requests.

So re-capture-per-serve is a **working, scale-validated** fix for swap correctness. That part is solid.

## Mechanism — OPEN (NOT root-caused; measurable causes RULED OUT)

Under cross-model residency churn, the carried captured-graph replay progressively diverges (serve#1 KL=0 → serve#2
47/48) and eventually HARD-FAULTS (`index_copy_(2, cache_position) out of bounds` — the same assert family as inc-1
dangling-cache and inc-2 eager-between-capture-replay). I ruled out every measurable candidate:
- **NOT a pager/weights bug:** `cipher_pager_live_cksum` bit-identical across cycles incl. evicted-and-restored;
  EAGER decode on a FRESH cache after a diverging swap = 48/48 CORRECT.
- **NOT corrupted KV data:** carried cache KV[0:P] vs a fresh prefill = **max|Δ|=0.0000** (`pager_g3b_isolate.py`).
- **NOT a dangling/realloc pointer:** cache `keys/values.data_ptr()` recorded AT CAPTURE == at the diverging serve
  (ptrs-moved=0, dim-changed=0, max_cache_len stable) (`pager_g3b_ptr.py`).

Every measured piece of state — weights, KV contents, cache buffer pointers, dims — is correct/stable, yet the
captured-graph replay corrupts under churn. **The true mechanism is OPEN.** re-capture (fresh cache + capture) resets
whatever it is; I cannot name the corrupted object from measurement, so I do NOT claim "re-capture necessary" or
"VMM invalidates CUDA graphs" — those would be symptom-restated-as-cause (the arc's recurring failure mode).

**Refuted fix hypotheses (don't pursue as-is):** re-prefill-only (KV already intact — nothing to refresh);
rebuild-graph-only over the carried cache (confounded — it was the *next* serve in the progression, faults regardless;
not clean evidence). **Open candidate fix:** buffer-pinning / a persistent-graph design that survives churn — would
remove the per-serve capture tax IF the mechanism allows it; UNVERIFIED.

## Cost (honest)

re-capture **~46ms/serve** (swap 40ms / hit 49ms — re-capture currently runs on EVERY serve), decode ~477ms (N=48,
host-overhead-dominated per inc-2) → per-serve ~522ms (decode-dominated; re-capture ~9%). **Unproven:** whether HITS
(resident, no swap) actually need re-capture — never cleanly tested that a resident model served twice with NO
intervening other-model paging survives without re-capture. So "every serve pays the tax" is the current
implementation, NOT a proven necessity; per-churn-only re-capture may suffice (different, cheaper economics) — OPEN.

## Density

4 distinct models served in one process, K=2 resident, swap-on-miss correctness-sustained → distinct-models-per-GPU
≈ resident capacity (~5 fp16 7-8B in 80GB) with unbounded swap-beyond at ~46ms re-capture + page-in per cold serve.

## Gate status (Mem #11)

1. **Correctness:** per-request KL=0 SUSTAINED REQ=200 + misroute neg-control DETECTED — PASS. **Root-cause: NOT
   achieved** (mechanism open; measurable causes ruled out). So this is swap-correctness *sustained*, not *closed*.
2. **Swap cost:** ~46ms re-capture/serve; per-serve ~522ms decode-dominated.
3. **Density:** ~5 fp16 7-8B resident + swap-beyond.
4. **NON-REGRESSION:** inc-1 module byte-identical + gate PASSES; inc-2 unaffected; OFF byte-identical + anchor
   1f305ce6 UNCHANGED (no `.so` change). Subprocesses os._exit-reaped, GPU→0.

## STOP

Swap correctness is SUSTAINED at REQ=200 (tagged) — the engine serves M distinct models in one process with
correctness-closed swapping. But the corruption MECHANISM is OPEN (not weights/KV/dangling-ptr), and whether a
cheaper persistent-graph fix exists is open. Honest call for Anil: (a) accept re-capture-per-serve as the working
swap path and proceed to inc-4 (both-regimes real-workload gate), carrying "mechanism open + per-serve-vs-per-churn
cost open" as known debt; or (b) spend more on the mechanism / buffer-pinning before inc-4. I stopped the mechanism
spiral here per the discipline (ruled out the measurable causes; do not stack another guess).
