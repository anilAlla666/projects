import json

import numpy as np

from vm.corpus.tok import SPECIAL, encode_jsonl, load, train_tokenizer

import random

random.seed(0)
WORDS = ["cat", "mat", "sat", "value", "return", "class", "def", "import", "numpy", "result", "token", "data"]
TEXTS = [f"def f{i}(x{i}):\n    return x{i} + {i}\n" if i % 2 else
         " ".join(random.choice(WORDS) + str(random.randint(0, 999)) for _ in range(12)) + ".\n" for i in range(3000)]


def test_tokenizer_round_trips_code_and_has_exact_special_ids(tmp_path):
    train_tokenizer(iter(TEXTS), tmp_path / "tok.json", vocab=600)
    tok = load(tmp_path / "tok.json")
    for s in ("def f(x):\n    return x + 1\n", "  indented\ttabs\n"):
        assert tok.decode(tok.encode(s).ids) == s
    assert [tok.token_to_id(s) for s in SPECIAL] == list(range(len(SPECIAL)))
    assert tok.get_vocab_size() == 600


def test_encode_writes_streams_with_eos(tmp_path):
    train_tokenizer(iter(TEXTS), tmp_path / "tok.json", vocab=600)
    src = tmp_path / "a.jsonl"
    src.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS[:50]))
    stats = encode_jsonl(tmp_path / "tok.json", [src], tmp_path / "a.bin", tmp_path / "a_val.bin", val_fraction=0.1, workers=2)
    train = np.fromfile(tmp_path / "a.bin", dtype=np.uint16)
    val = np.fromfile(tmp_path / "a_val.bin", dtype=np.uint16)
    assert stats["documents"] == 50
    assert (train == 0).sum() + (val == 0).sum() == 50  # one <|eos|> per document
    assert len(val) > 0 and len(train) > len(val)
