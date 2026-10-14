"""The 32k byte-level BPE tokenizer and parallel encoding into uint16 token streams."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["<|eos|>", "<|pad|>", "<|task|>", "<|diff|>", "<|evidence|>", "<|untrusted|>", "<|line|>", "<|verdict|>"]


def train_tokenizer(texts, out_path: Path, vocab: int = 32768) -> None:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))


def load(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def _is_val(text: str, val_fraction: float) -> bool:
    return int(hashlib.sha256(text[:4096].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_fraction


def _encode_chunk(args):
    tok_path, lines, val_fraction = args
    tok = load(Path(tok_path))
    texts = [json.loads(l)["text"] for l in lines]
    enc = tok.encode_batch(texts)
    train, val = [], []
    for text, e in zip(texts, enc):
        (val if _is_val(text, val_fraction) else train).append(np.array(e.ids + [0], dtype=np.uint16))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, dtype=np.uint16)  # noqa: E731
    return cat(train), cat(val), len(texts)


def encode_jsonl(tok_path: Path, jsonl_paths, out_bin: Path, val_bin: Path, val_fraction: float = 0.005,
                 workers: int = 96, chunk: int = 2000) -> dict:
    def chunks():
        for p in jsonl_paths:
            buf = []
            with open(p) as f:
                for line in f:
                    buf.append(line)
                    if len(buf) == chunk:
                        yield (str(tok_path), buf, val_fraction)
                        buf = []
            if buf:
                yield (str(tok_path), buf, val_fraction)

    n_docs = n_train = n_val = 0
    with open(out_bin, "wb") as ft, open(val_bin, "wb") as fv, mp.get_context("spawn").Pool(workers) as pool:
        for train, val, docs in pool.imap(_encode_chunk, chunks()):
            ft.write(train.tobytes())
            fv.write(val.tobytes())
            n_docs, n_train, n_val = n_docs + docs, n_train + len(train), n_val + len(val)
    return {"documents": n_docs, "train_tokens": n_train, "val_tokens": n_val}
