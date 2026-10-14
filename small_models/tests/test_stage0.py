import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quick_run_writes_a_complete_report(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/stage0.py"), "--quick", "--out", str(tmp_path), "--steps", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    reports = list(tmp_path.glob("*/stage0.json"))
    assert len(reports) == 1
    rep = json.loads(reports[0].read_text())
    for check in rep["numeric_check"]:
        assert check["max_out_diff"] < 1e-4 and check["max_state_diff"] < 1e-4
    assert rep["train"] and all(row["tokens_per_s"] > 0 for row in rep["train"])
    assert rep["verdict"] and all(row["sec_per_forward"] > 0 for row in rep["verdict"])
    assert {"config", "dtype", "compile", "batch", "seq", "peak_gb", "days_per_1b_tokens",
            "tokens_per_s_min", "tokens_per_s_max", "load_before", "load_after"} <= set(rep["train"][0])
    assert all(r["tokens_per_s_min"] <= r["tokens_per_s"] <= r["tokens_per_s_max"] for r in rep["train"])
