"""The synthetic attack sets in another evidence setting: each attack row's prose (its last untrusted entry) is added
to the same attempt's example built under that setting, so both settings carry identical notes.

    python scripts/attacks_setting.py ROOT TOKENIZER SETTING OUT_DIR
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vm.corpus.tok import load  # noqa: E402
from vm.data.example import serialise  # noqa: E402
from vm.data.load import load_examples  # noqa: E402

root, tok, setting, out = Path(sys.argv[1]), load(Path(sys.argv[2])), sys.argv[3], Path(sys.argv[4])
base = {e["example_id"]: e for e in load_examples(root, setting=setting)}
for split in ("train", "dev", "test"):
    ser = []
    for a in map(json.loads, (root / f"attacks_{split}.jsonl").read_text().splitlines()):
        b = base.get(a["example_id"].rsplit("/", 1)[0])
        if b is None:
            continue
        att = {**b, "example_id": a["example_id"], "attempt_kind": a["attempt_kind"], "untrusted": b["untrusted"] + [a["untrusted"][-1]]}
        try:
            s = serialise(att, tok)
        except ValueError:
            continue
        ser.append({**s, "example_id": att["example_id"], "attempt_kind": att["attempt_kind"], "operator": att["operator"], "library": att["library"]})
    (out / f"serialised_attack_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ser))
    print(split, len(ser))
