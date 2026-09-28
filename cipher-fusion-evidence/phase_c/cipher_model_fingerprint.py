#!/usr/bin/env python3
"""Track 2 SC4 — model-identity fingerprint.

A two-tier fingerprint over a model directory. Two tenants may share a weight
arena only if their fingerprints match — this is the safety primitive that
turns a silent wrong-model import into a clean, logged rejection. Pure Python;
no substrate.

  tier 1 (structural)     — config.json + torch_dtype + the safetensors shard
                            set (names, sizes) + the shard index if present.
                            Catches a different architecture / shape / layout.
  tier 2 (weight-content) — sampled weight bytes: 1 MiB at the head, middle and
                            tail of every shard. Catches same-architecture /
                            different-weights (a different fine-tune). Shards
                            < 3 MiB are hashed in FULL — the three 1 MiB
                            windows would overlap/exceed the shard (Item-1 PUSH).
  combined = SHA256(tier1 ‖ tier2).

v1 boundary: operator-error-proof, NOT adversary-proof (the unsampled regions
of large shards are not covered). A full weight hash is the v2 hardening.
"""
import glob
import hashlib
import json
import os

SAMPLE = 1 << 20            # 1 MiB sample window
SMALL_SHARD = 3 << 20      # < 3 MiB -> full-shard hash (Item-1 PUSH)


class FingerprintMismatch(Exception):
    """Raised by verify_fingerprint when producer and consumer models differ.
    Carries enough detail for an operator to diagnose the misconfiguration."""
    def __init__(self, producer_hash, consumer_hash, manifest_path,
                 consumer_model_path, differing_tier):
        self.producer_hash = producer_hash
        self.consumer_hash = consumer_hash
        self.manifest_path = manifest_path
        self.consumer_model_path = consumer_model_path
        self.differing_tier = differing_tier
        super().__init__(
            "model fingerprint mismatch (tier %s): producer=%s consumer=%s"
            % (differing_tier, str(producer_hash)[:16], str(consumer_hash)[:16]))


def _shards(model_path):
    return sorted(glob.glob(os.path.join(model_path, "*.safetensors")))


def _shard_tier2_hash(path):
    """Sampled weight-byte hash of one shard — or the full shard if < 3 MiB."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size < SMALL_SHARD:                       # Item-1 PUSH: small shard
            h.update(f.read())                       # -> full-shard hash
        else:
            for off in (0, size // 2 - SAMPLE // 2, size - SAMPLE):
                f.seek(off)
                h.update(f.read(SAMPLE))
    return h.hexdigest()


def compute_fingerprint(model_path):
    """Return {tier1, tier2, combined, shard_metadata, n_shards} for a model
    directory. Raises FileNotFoundError if no *.safetensors are present."""
    shards = _shards(model_path)
    if not shards:
        raise FileNotFoundError("no *.safetensors in %s" % model_path)

    # ---- tier 1: structural ----
    t1 = hashlib.sha256()
    cfg_path = os.path.join(model_path, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    t1.update(json.dumps(cfg, sort_keys=True).encode())
    t1.update(str(cfg.get("torch_dtype", "")).encode())
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        t1.update(open(idx_path, "rb").read())
    shard_meta = []
    for s in shards:
        name, size = os.path.basename(s), os.path.getsize(s)
        t1.update(("%s:%d" % (name, size)).encode())
        shard_meta.append({"name": name, "size": size})

    # ---- tier 2: sampled weight content ----
    t2 = hashlib.sha256()
    for s in shards:
        t2.update(_shard_tier2_hash(s).encode())

    tier1, tier2 = t1.hexdigest(), t2.hexdigest()
    combined = hashlib.sha256((tier1 + tier2).encode()).hexdigest()
    return {"tier1": tier1, "tier2": tier2, "combined": combined,
            "shard_metadata": shard_meta, "n_shards": len(shards)}


def verify_fingerprint(producer_manifest, local_model_path,
                       manifest_path="<manifest>"):
    """Compare the consumer's fingerprint (computed from local_model_path) to
    the producer's (carried in producer_manifest['fingerprint']). Returns True
    on match; raises FingerprintMismatch on mismatch, with the differing tier
    identified (1 = structural, 2 = weight-content)."""
    prod = producer_manifest.get("fingerprint")
    if not prod:
        raise FingerprintMismatch("<absent-from-manifest>", "<n/a>",
                                  manifest_path, local_model_path, "manifest")
    cons = compute_fingerprint(local_model_path)
    if prod.get("combined") == cons["combined"]:
        return True
    tier = ("1" if prod.get("tier1") != cons["tier1"]
            else ("2" if prod.get("tier2") != cons["tier2"] else "combined"))
    raise FingerprintMismatch(prod.get("combined"), cons["combined"],
                              manifest_path, local_model_path, tier)
