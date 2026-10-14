"""The attacked sample in another evidence setting: each sampled attack row's prose (the last untrusted entry) is
added to the same patch's example built under that setting, so both settings carry identical notes.

    python scripts/real_attack_setting.py REAL_DIR TOKENIZER SETTING
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vm.corpus.tok import load  # noqa: E402
from vm.data.example import serialise  # noqa: E402

real, tok, setting = Path(sys.argv[1]), load(Path(sys.argv[2])), sys.argv[3]
base = {e["example_id"]: e for e in map(json.loads, (real / f"examples_{setting}.jsonl").read_text().splitlines())}
raw, ser = [], []
for a in map(json.loads, (real / "attacks_real.jsonl").read_text().splitlines()):
    b = base[a["example_id"].rsplit("/", 1)[0]]
    att = {**b, "example_id": a["example_id"], "attempt_kind": a["attempt_kind"], "untrusted": b["untrusted"] + [a["untrusted"][-1]]}
    raw.append(att)
    ser.append({**serialise(att, tok), "example_id": att["example_id"], "attempt_kind": att["attempt_kind"], "operator": "real", "library": att["library"]})
(real / f"attacks_real_{setting}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
(real / f"serialised_attack_real_{setting}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ser))
print(len(ser))
