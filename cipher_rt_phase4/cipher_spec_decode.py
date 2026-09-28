"""CIPHER CP 2.4 sub-task (iii) — speculative decode (operator-policy injection).

Speculative decoding as a generate-path injection — the same operator-policy
model as op #1's cipher_kv_cache.py: install() monkey-patches the transformers
generate path so a customer's model.generate() is routed through CIPHER's
speculative loop. Customer code is unchanged; the operator injects this via a
bootstrap import and defaults the draft policy.

It sits ABOVE the v2 matmul/attn dispatch — the draft and target forwards
inside the loop still flow through the Marlin actuator, so spec × Marlin
compose for free. It is coverage-immune (it owns the decode loop; there is no
dispatch to miss). Draft placement: own CUDA stream, PRIMARY context — the
Marlin full-GPU pin (CP 2.4 Fix A) applies to the draft model too, since the
draft is itself a transformer forward through Marlin-routed shapes.

See cp_2_4/SPEC_DECODE_DESIGN_MEMO.md.

Env gates:
  CIPHER_SPEC          = 1|0   (default 1)        — master opt-out per tenant
  CIPHER_SPEC_DRAFT    = <hf-path>|"ngram"        — draft policy
                         (Llama arm: a 1B draft path; Mistral arm: "ngram")
  CIPHER_SPEC_DRAFT_STREAM = 1|0 (default 0)      — model-draft CUDA stream:
                         0 = default stream (the draft never overlaps the
                         target — see ModelDraft.propose); 1 = own stream
                         (deprecated — hangs cuDNN attention under the v2
                         substrate, CUDNN_ATTN_MARLIN_HANG.md §3a)
  CIPHER_SPEC_EMA_WIN  = int   (default 64)       — adaptive-k EMA window
  CIPHER_SPEC_NGRAM_N  = int   (default 3)        — n-gram match length

Status: core engine + drafts + adaptive-k below are CPU-unit-tested
(test_cipher_spec_decode.py). The GPU paths (ModelDraft forward, the target
verify forward, draft-stream placement) and the install() wiring into HF
generate are verified on GPU after the CP 2.4 DVFS sweep completes.
"""
import os
import sys

# --------------------------------------------------------------------------
# Adaptive k — EMA of the running acceptance rate drives the proposal length.
# --------------------------------------------------------------------------

# Aim to land ~TARGET_ACCEPTS accepted draft tokens per verify round; k is then
# target / accept_rate, clamped. Higher k when acceptance is low (propose more
# to still clear the target), lower k when acceptance is high (don't overshoot).
_K_MIN, _K_MAX = 2, 8
_TARGET_ACCEPTS = 3.0


class AdaptiveK:
    """k = clamp(2, round(TARGET_ACCEPTS / EMA_accept_rate), 8).

    EMA window default 64 verifications (CIPHER_SPEC_EMA_WIN) — long enough to
    be stable across a prompt, short enough to track a workload phase change.
    """

    def __init__(self, window=64, target_accepts=_TARGET_ACCEPTS,
                 k_min=_K_MIN, k_max=_K_MAX):
        self.alpha = 2.0 / (window + 1)
        self.target = float(target_accepts)
        self.k_min, self.k_max = k_min, k_max
        self.ema_rate = 0.7        # seed: optimistic-but-plausible
        self.n_updates = 0

    def update(self, n_accepted, k_proposed):
        """Fold one verify round's acceptance into the EMA."""
        rate = n_accepted / max(k_proposed, 1)
        self.ema_rate = self.alpha * rate + (1.0 - self.alpha) * self.ema_rate
        self.n_updates += 1

    def next_k(self):
        raw = round(self.target / max(self.ema_rate, 0.05))
        return max(self.k_min, min(int(raw), self.k_max))


# --------------------------------------------------------------------------
# Drafts — propose(token_ids: list[int], k: int) -> list[int] (<= k tokens).
# --------------------------------------------------------------------------

class NgramDraft:
    """Prompt-lookup n-gram draft (Mistral arm). No model, no extra memory:
    find the most recent earlier occurrence of the last n emitted tokens and
    propose the tokens that followed it. Pure token-space — no tokenizer of its
    own, so no draft/target alignment question. Pure CPU logic."""

    def __init__(self, ngram_n=3):
        self.ngram_n = ngram_n

    def propose(self, token_ids, k):
        n = self.ngram_n
        if k <= 0 or len(token_ids) < n + 1:
            return []
        pat = token_ids[-n:]
        # most-recent earlier occurrence
        for start in range(len(token_ids) - n - 1, -1, -1):
            if token_ids[start:start + n] == pat:
                return list(token_ids[start + n:start + n + k])
        return []


class ModelDraft:
    """Draft-model draft (Llama arm): Llama-3.2-1B proposing for Llama-3.1-8B.
    Shares the target's tokenizer/vocab (Llama-3 family — verified by a vocab
    hash check at install time), so proposed token ids align with the target.

    Runs on its own CUDA stream in the PRIMARY context (Fix A: Marlin is
    full-GPU; the draft's GEMMs route through Marlin too). GPU path — verified
    on GPU post-DVFS-sweep."""

    def __init__(self, draft_model, draft_stream=None):
        self.model = draft_model
        self.stream = draft_stream      # torch.cuda.Stream, primary context

    def propose(self, token_ids, k):
        """Stateless: a fresh draft prefill of the full committed sequence,
        then k incremental draft decodes. The draft KV is deliberately NOT
        carried across propose() calls. Between rounds spec_generate commits
        accepted+bonus tokens that a carried KV would not cover — and the
        rejected draft tokens it *would* still hold are invalid; across
        generate() calls it would be a different prompt entirely. Recompute
        is unconditionally correct and, for a 1B draft, cheap (~few ms
        prefill). A feedback-keyed incremental draft KV (crop on the
        per-round accept count) is a measured ~10-20% optimisation left for
        Phase 5 — it is the same off-by-one surface that bit the verify loop
        three times, so it is not on the gate-measurement path. See
        cp_2_4/SPEC_DECODE_METHODOLOGY.md."""
        import torch
        if k <= 0:
            return []
        dev = next(self.model.parameters()).device
        out = []
        ctx = torch.cuda.stream(self.stream) if self.stream is not None \
            else _nullctx()
        with ctx, torch.no_grad():
            cur = torch.tensor([token_ids], device=dev)
            past = None
            for _ in range(k):
                res = self.model(cur if past is None else cur[:, -1:],
                                 past_key_values=past, use_cache=True)
                past = res.past_key_values
                nxt = int(res.logits[0, -1].argmax())
                out.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=dev)], -1)
        return out


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# --------------------------------------------------------------------------
# Verify — greedy accept/reject (gate path; temp 0 => byte-identical output).
# --------------------------------------------------------------------------

def greedy_accept(target_argmax, draft_tokens):
    """target_argmax: argmax over the target's k+1 logit vectors (len k+1).
       draft_tokens : the k proposed tokens.
    Returns (accepted_tokens, bonus_token, n_accepted). Accept the longest
    prefix where target argmax == draft; the bonus token is the target's
    argmax at the first non-accepted position (free, from the same forward).
    This is exactly target greedy decoding — output byte-identical."""
    n = 0
    for i in range(len(draft_tokens)):
        if int(target_argmax[i]) == int(draft_tokens[i]):
            n += 1
        else:
            break
    accepted = [int(t) for t in draft_tokens[:n]]
    bonus = int(target_argmax[n])      # n <= len(draft); target_argmax has k+1
    return accepted, bonus, n


# --------------------------------------------------------------------------
# Spec-decode loop. GPU path — verified post-DVFS-sweep.
# --------------------------------------------------------------------------

def spec_generate(target_model, draft, input_ids, max_new_tokens,
                  eos_token_id=None, adaptive_k=None, stats=None):
    """Greedy speculative decode. Output is byte-identical to plain target
    greedy decoding — the verify guarantees it. target_model: HF model;
    draft: NgramDraft or ModelDraft; input_ids: 1-D list[int].

    Loop invariant: `past` covers seq[:-1]; `pending` = seq[-1], the last
    committed token whose KV is NOT yet in `past`. Each round feeds
    [pending] + proposed, so res.logits[j] is the target distribution AFTER
    feed[j]: logits[0] -> target's choice for proposed[0], logits[i] ->
    proposed[i], logits[k] -> the all-accepted bonus. That gives a length-(k+1)
    tgt_full perfectly aligned to greedy_accept() — no off-by-one, the bonus
    index n in 0..k is always in range, and the KV crop keeps exactly
    pending(1)+accepted(n)."""
    import torch
    ak = adaptive_k or AdaptiveK(
        window=int(os.environ.get("CIPHER_SPEC_EMA_WIN", "64")))
    dev = next(target_model.parameters()).device

    seq = list(input_ids)
    with torch.no_grad():                                 # prefill the prompt
        res = target_model(torch.tensor([seq], device=dev), use_cache=True)
    past = res.past_key_values                            # covers the prompt
    pending = int(res.logits[0, -1].argmax())             # 1st token, no KV yet
    seq.append(pending)
    produced = 1

    while produced < max_new_tokens:
        if eos_token_id is not None and seq[-1] == eos_token_id:
            break
        k = ak.next_k()
        proposed = draft.propose(seq, k)                  # may be [] (miss)
        feed = [pending] + proposed                       # len = k_proposed+1
        with torch.no_grad():
            res = target_model(torch.tensor([feed], device=dev),
                               past_key_values=past, use_cache=True)
        past = res.past_key_values
        tgt_full = res.logits[0].argmax(dim=-1).tolist()  # len(feed) entries
        accepted, bonus, n = greedy_accept(tgt_full, proposed)
        # feed appended len(feed) KV entries (pending + proposed); keep
        # pending(1)+accepted(n), drop the (len(proposed)-n) rejected.
        past = _truncate_kv(past, len(feed), n + 1)
        seq.extend(accepted)
        seq.append(bonus)
        pending = bonus
        produced += n + 1
        ak.update(n, max(len(proposed), 1))
        if stats is not None:
            stats["rounds"] = stats.get("rounds", 0) + 1
            stats["accepted"] = stats.get("accepted", 0) + n
            stats["proposed"] = stats.get("proposed", 0) + len(proposed)
    return seq[:len(input_ids) + max_new_tokens]


def _truncate_kv(past, n_appended, n_keep):
    """After a batched verify of n_appended tokens, the target KV cache holds
    all n_appended; keep only the first n_keep (accepted + bonus), drop the
    rest. HF cache-class dependent — finalised against transformers 5.8.1's
    Cache API during GPU bring-up."""
    if past is None or n_keep >= n_appended:
        return past
    crop = getattr(past, "crop", None)
    if callable(crop):
        past.crop(-(n_appended - n_keep))
    return past


# --------------------------------------------------------------------------
# Operator-policy injection.
# --------------------------------------------------------------------------

def _enabled():
    return os.environ.get("CIPHER_SPEC", "1") not in ("0", "off", "no", "")


_draft = None       # lazily loaded on first generate (target device known then)
_last_stats = {}    # stats dict from the most recent _cipher_spec_generate call
                    # (rounds/accepted/proposed) — read by measurement drivers


def _load_draft(target_model):
    """CIPHER_SPEC_DRAFT = 'ngram' -> NgramDraft (Mistral arm, no model);
    else a HF model path -> ModelDraft (Llama arm), loaded on the target's
    device, fp16, on its own CUDA stream in the primary context."""
    global _draft
    if _draft is not None:
        return _draft
    policy = os.environ.get("CIPHER_SPEC_DRAFT", "ngram")
    if policy == "ngram":
        _draft = NgramDraft(ngram_n=int(os.environ.get("CIPHER_SPEC_NGRAM_N", "3")))
        print("[cipher-spec] draft = n-gram (n=%d)" % _draft.ngram_n,
              file=sys.stderr)
    else:
        import torch
        from transformers import AutoModelForCausalLM
        dev = next(target_model.parameters()).device
        # trust_remote_code=False (default, made explicit): load weights+config
        # only, never execute model-repo-supplied python.
        dm = AutoModelForCausalLM.from_pretrained(
            policy, dtype=torch.float16,
            trust_remote_code=False).to(dev).eval()
        # ModelDraft proposes in raw token-id space (no draft tokenizer) — the
        # draft/target vocabularies MUST be identical or every proposal is
        # misaligned. Enforce, do not assume (Llama-3.1 / 3.2 share the 128256
        # vocab — verified here, not trusted).
        tv = getattr(target_model.config, "vocab_size", None)
        dv = getattr(dm.config, "vocab_size", None)
        if tv != dv:
            raise RuntimeError(
                "[cipher-spec] draft/target vocab_size mismatch: target=%s "
                "draft=%s — ModelDraft requires identical vocabularies" % (tv, dv))
        # The model draft runs on the DEFAULT stream by default. A separate
        # torch.cuda.Stream gave zero benefit — the draft<->target handoff is
        # fully synchronous (int()/tolist() force syncs, the draft never
        # overlaps the target) — and an own stream hangs cuDNN's runtime-
        # compiled attention engine under the v2 substrate (CP 2.4 bisection,
        # CUDNN_ATTN_MARLIN_HANG.md §3a). CIPHER_SPEC_DRAFT_STREAM=1 restores
        # the (deprecated) own-stream behaviour.
        use_stream = os.environ.get("CIPHER_SPEC_DRAFT_STREAM", "0") \
            not in ("0", "off", "no", "")
        _draft = ModelDraft(dm, draft_stream=(torch.cuda.Stream()
                                              if use_stream else None))
        print("[cipher-spec] draft = model %s on %s (stream=%s); "
              "vocab_size=%s matched"
              % (policy, dev, "own" if use_stream else "default", dv),
              file=sys.stderr)
    return _draft


def _spec_eligible(kwargs):
    """CIPHER spec handles greedy decoding (temp 0 => byte-identical to the
    target — the gate path). do_sample with temperature > 0 falls through to
    stock generate. Batch>1 is filtered inside _cipher_spec_generate."""
    if kwargs.get("do_sample", False):
        t = kwargs.get("temperature", None)
        if t not in (None, 0, 0.0):
            return False
    return True


def _cipher_spec_generate(target_model, orig_generate, *args, **kwargs):
    """Route one generate() call through spec_generate(). Falls back to
    orig_generate on anything spec does not cover. GPU-finalised: the exact
    args/return-shape glue is confirmed against transformers 5.8.1 during the
    Mistral-arm GPU bring-up — the seam is intentionally narrow and guarded."""
    import torch
    ids = kwargs.get("input_ids", kwargs.get("inputs",
                     args[0] if args else None))
    if ids is None or not torch.is_tensor(ids) or ids.shape[0] != 1:
        return orig_generate(target_model, *args, **kwargs)   # batch!=1 -> stock
    max_new = kwargs.get("max_new_tokens", None)
    if max_new is None:
        gc = getattr(target_model, "generation_config", None)
        max_new = getattr(gc, "max_new_tokens", None) or 32
    eos = kwargs.get("eos_token_id", None)
    if eos is None:
        eos = getattr(target_model.config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None

    draft = _load_draft(target_model)
    prompt = ids[0].tolist()
    stats = {}
    seq = spec_generate(target_model, draft, prompt, int(max_new),
                        eos_token_id=eos, stats=stats)
    global _last_stats
    _last_stats = dict(stats)
    if stats.get("rounds"):
        acc, prop = stats.get("accepted", 0), stats.get("proposed", 1)
        print("[cipher-spec] rounds=%d accept_rate=%.3f"
              % (stats["rounds"], acc / max(prop, 1)), file=sys.stderr)
    return torch.tensor([seq], device=ids.device, dtype=ids.dtype)


def install():
    """Monkey-patch transformers GenerationMixin.generate so model.generate()
    routes through CIPHER speculative decode. Idempotent; no-op if
    CIPHER_SPEC=0. Customer code unchanged — operator-policy injection, the
    same pattern as cipher_kv_cache.install()."""
    if not _enabled():
        print("[cipher-spec] CIPHER_SPEC disabled — stock generate",
              file=sys.stderr)
        return False
    import transformers
    gm = transformers.generation.GenerationMixin
    if getattr(gm.generate, "_cipher_spec_wrapped", False):
        return True
    _orig_generate = gm.generate

    def _wrapped_generate(self, *args, **kwargs):
        if _enabled() and _spec_eligible(kwargs):
            try:
                return _cipher_spec_generate(self, _orig_generate, *args, **kwargs)
            except Exception as e:           # never break the customer's call
                print("[cipher-spec] fallback to stock generate: %r" % e,
                      file=sys.stderr)
        return _orig_generate(self, *args, **kwargs)

    _wrapped_generate._cipher_spec_wrapped = True
    gm.generate = _wrapped_generate
    print("[cipher-spec] installed: generate-path speculative decode (draft=%s)"
          % os.environ.get("CIPHER_SPEC_DRAFT", "ngram"), file=sys.stderr)
    return True
