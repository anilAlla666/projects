import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_run_learns_checkpoints_and_resumes(tmp_path):
    rng = np.random.default_rng(0)
    pattern = np.tile(np.arange(1, 50, dtype=np.uint16), 20_000)
    for name in ("code", "english"):
        pattern.tofile(tmp_path / f"{name}.bin")
        rng.integers(1, 50, 5000).astype(np.uint16).tofile(tmp_path / f"{name}_val.bin")
    cfg = (ROOT / "configs/pretrain_tiny.toml").read_text().replace("DATA_DIR", str(tmp_path)).replace("OUT_DIR", str(tmp_path / "out"))
    (tmp_path / "tiny.toml").write_text(cfg)
    run = lambda *extra: subprocess.run(  # noqa: E731
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(ROOT / "vmt/pretrain.py"), str(tmp_path / "tiny.toml"), *extra],
        cwd=ROOT, capture_output=True, text=True)
    r = run()
    assert r.returncode == 0, r.stderr[-3000:]
    run_dir = next((tmp_path / "out").glob("*/*"))
    log = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines()]
    assert log[-1]["loss"] < 0.5 * log[0]["loss"]
    assert (run_dir / "final.pt").is_file() and list(run_dir.glob("ckpt_*.pt"))
    assert (run_dir / "val.jsonl").read_text().strip()
    r2 = run("--resume", str(run_dir), "--extra-steps", "5")
    assert r2.returncode == 0, r2.stderr[-3000:]
    log2 = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines()]
    assert log2[-1]["step"] == log[-1]["step"] + 5
