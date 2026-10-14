import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quick_run_writes_a_report(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/stage0_gpu.py"), "--quick", "--out", str(tmp_path), "--steps", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads(next(tmp_path.glob("*/stage0_gpu.json")).read_text())
    assert rep["train"] and all(r["tokens_per_s_min"] <= r["tokens_per_s"] <= r["tokens_per_s_max"] for r in rep["train"])
    assert {"hours_per_1b_tokens", "dollars_per_1b_tokens", "peak_gb", "compiled"} <= set(rep["train"][0])
    assert rep["verdict"] and rep["verdict"][0]["sec_per_forward"] > 0
