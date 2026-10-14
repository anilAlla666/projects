"""FineWeb-Edu (ODC-By) parquet files to JSON lines of text, up to a character budget."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


def build(parquet_dir: Path, out_dir: Path, max_chars: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_chars = n_docs = 0
    with open(out_dir / "english_000.jsonl", "w") as out:
        for f in sorted(parquet_dir.rglob("*.parquet")):
            for batch in pq.ParquetFile(f).iter_batches(columns=["text"], batch_size=4096):
                for text in batch.column(0).to_pylist():
                    out.write(json.dumps({"text": text}) + "\n")
                    n_chars += len(text)
                    n_docs += 1
                    if n_chars >= max_chars:
                        return {"documents": n_docs, "characters": n_chars}
    return {"documents": n_docs, "characters": n_chars}


if __name__ == "__main__":
    import sys
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2]), int(float(sys.argv[3])))))
