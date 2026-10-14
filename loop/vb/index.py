"""Index every Terminal-Bench 4.0 leaderboard attempt with its reward (metadata only, no downloads)."""
from __future__ import annotations

import argparse
from collections import defaultdict

from .util import DATA, harbor_json, read_json, write_json

LEADERBOARD = "9f966760-00f1-424e-90f5-c964fb6f6091"
INDEX = DATA / "meta" / "tb4_index.json"


def build_index(leaderboard: str = LEADERBOARD) -> list[dict]:
    rows = []
    for r in harbor_json("hub", "leaderboard", "show", leaderboard)["rows"]:
        first = harbor_json("hub", "leaderboard", "row", "trial", "list", r["id"], "--limit", "1", "--page", "1")
        job = harbor_json("hub", "trial", "show", first["items"][0]["trial_id"])["job_id"]
        trials, page = [], 1
        while True:
            d = harbor_json("hub", "job", "trials", job, "--limit", "500", "--page", str(page))
            items = d.get("items") or []
            trials += items
            if not items or page >= (d.get("total_pages") or 1):
                break
            page += 1
        m = r["metadata"]
        rows.append({"row": r["id"], "rank": r["rank"], "agent": m["agent_display"]["label"],
                     "model": m["model_display"]["label"], "job": job, "trials": trials})
    return rows


def attempts(index: list[dict]) -> list[dict]:
    out = []
    for row in index:
        for t in row["trials"]:
            out.append({"trial_id": t["id"], "task": t["task_name"].split("/")[-1], "reward": t.get("reward"),
                        "error": t.get("error_type"), "job": row["job"], "rank": row["rank"],
                        "agent": row["agent"], "model": row["model"]})
    return out


def task_counts(atts: list[dict]) -> dict[str, dict]:
    counts: dict[str, dict] = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0})
    for a in atts:
        if a["error"] or a["reward"] is None:
            counts[a["task"]]["error"] += 1
        elif float(a["reward"]) >= 1.0:
            counts[a["task"]]["pass"] += 1
        else:
            counts[a["task"]]["fail"] += 1
    return dict(counts)


def load_attempts() -> list[dict]:
    return attempts(read_json(INDEX))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if a.refresh or not INDEX.exists():
        write_json(INDEX, build_index())
    atts = load_attempts()
    print(len(atts), "attempts;", len(task_counts(atts)), "tasks")
