"""Train the 32k tokenizer on the first 400 MB of code and 400 MB of English, then encode both corpora into
uint16 token streams.

    python scripts/build_tokenizer.py ~/data   (expects ~/data/code/code_*.jsonl and ~/data/english/english_*.jsonl)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vm.corpus.tok import encode_jsonl, train_tokenizer  # noqa: E402


def sample(paths, budget):
    n = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                text = json.loads(line)["text"]
                yield text
                n += len(text)
                if n >= budget:
                    return


def main(data: Path):
    code = sorted(glob.glob(str(data / "code" / "code_*.jsonl")))
    english = sorted(glob.glob(str(data / "english" / "english_*.jsonl")))
    tok_path = data / "tokenizer.json"

    def texts():
        yield from sample(code, 400_000_000)
        yield from sample(english, 400_000_000)

    train_tokenizer(texts(), tok_path)
    out = data / "tokens"
    out.mkdir(exist_ok=True)
    stats = {"code": encode_jsonl(tok_path, code, out / "code.bin", out / "code_val.bin"),
             "english": encode_jsonl(tok_path, english, out / "english.bin", out / "english_val.bin")}
    (data / "corpus_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
