"""Writer tasks: the solver's environment, an instruction to write tests/, and only the model gateway reachable."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

from .leakscan import scan
from .util import DATA, HARBOR, ROOT, TB_ROOT, load_env, new_run_dir, read_json, write_json

GATEWAY_HOST = "inference.hud.ai"
# Four writers = 2 models x 2 harnesses. "env" goes into the container (harnesses that run inside it); "host_key"
# names the variable the host-side harness (terminus-2, via LiteLLM) reads, set from HUD_API_KEY in the
# subprocess environment only.
WRITERS = {
    "sonnet-claude-code": {"model_id": "sonnet", "harness": "claude-code", "agent": "claude-code",
                           "model": "anthropic/claude-sonnet-5",
                           "env": {"ANTHROPIC_BASE_URL": "https://inference.hud.ai", "ANTHROPIC_API_KEY": "${HUD_API_KEY}"},
                           "kwargs": ["disallowed_tools=WebFetch,WebSearch", "max_budget_usd=3"], "host_key": None},
    "luna-codex": {"model_id": "luna", "harness": "codex", "agent": "codex", "model": "openai/gpt-6-luna",
                   "env": {"OPENAI_BASE_URL": "https://inference.hud.ai/v1", "OPENAI_API_KEY": "${HUD_API_KEY}"},
                   "kwargs": ["web_search=disabled"], "host_key": None},
    "sonnet-terminus": {"model_id": "sonnet", "harness": "terminus-2", "agent": "terminus-2",
                        "model": "anthropic/claude-sonnet-5", "env": {},
                        "kwargs": ["api_base=https://inference.hud.ai", "max_turns=60"], "host_key": "ANTHROPIC_API_KEY"},
    # The gateway serves gpt-6-luna only on the Responses API; LiteLLM's "openai/responses/" prefix routes there.
    "luna-terminus": {"model_id": "luna", "harness": "terminus-2", "agent": "terminus-2",
                      "model": "openai/responses/gpt-6-luna",
                      "env": {}, "kwargs": ["api_base=https://inference.hud.ai/v1", "max_turns=60"],
                      "host_key": "OPENAI_API_KEY"},
}
# Every writer is also bounded by the writer task's agent timeout; Codex has no money setting, so for it the
# timeout is the only hard bound, and spend is checked between writers.
WRITER_TIMEOUT_S = 1800.0
WRITER_TASKS = DATA / "writer_tasks"

# The "+protocol" harness (M1b): same models and agent programs, a structured instruction (requirements first, one
# check per stated requirement, the writer's own correct and flawed solutions, the verifier proven on them before
# finishing) and three times the time. The writer still sees only the task and its environment.
VARIANTS = {
    "v1": {"template": "writer_instruction.md", "timeout": WRITER_TIMEOUT_S, "tasks_dir": WRITER_TASKS},
    "v2": {"template": "writer_instruction_v2.md", "timeout": 5400.0, "tasks_dir": DATA / "writer_tasks_v2"},
}
_V2_KWARGS = {"max_budget_usd=3": "max_budget_usd=8", "max_turns=60": "max_turns=150"}
WRITERS_V2 = {f"{n}-v2": {**w, "harness": w["harness"] + "+protocol",
                          "kwargs": [_V2_KWARGS.get(k, k) for k in w["kwargs"]]} for n, w in WRITERS.items()}
ALL_WRITERS = {**WRITERS, **WRITERS_V2}


def _scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return "[" + ", ".join(json.dumps(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value: {v!r}")


def dump_toml(doc: dict) -> str:
    lines = [f"{k} = {_scalar(v)}" for k, v in doc.items() if not isinstance(v, dict)]
    for table, body in doc.items():
        if isinstance(body, dict):
            lines += ["", f"[{table}]"] + [f"{k} = {_scalar(v)}" for k, v in body.items()]
    return "\n".join(lines) + "\n"


def _without_canary(text: str) -> str:
    """Drop the benchmark's canary marker line: it also appears in the official test files, so if the writer saw
    it here, finding it in a written verifier would prove nothing (the leak scan looks for it)."""
    return "\n".join(l for l in text.splitlines() if "harbor-" + "canary" not in l).strip()


def make_writer_task(task_dir: Path, out_root: Path | None = None, variant: str = "v1") -> Path:
    v = VARIANTS[variant]
    out_root = Path(out_root) if out_root is not None else v["tasks_dir"]
    task_dir = Path(task_dir)
    t = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    env = {k: v for k, v in (t.get("environment") or {}).items()
           if k in ("docker_image", "build_timeout_sec", "cpus", "memory_mb", "storage_mb", "gpus", "os")}
    out = Path(out_root) / f"write-verifier-{task_dir.name}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(task_dir / "environment", out / "environment")
    doc = {"schema_version": "2.0", "artifacts": ["/verifier_out/"],
           "task": {"name": f"vbench/write-verifier-{task_dir.name}", "description": ""},
           "metadata": {"source_task": t["task"]["name"]},
           "agent": {"timeout_sec": v["timeout"], "network_mode": "allowlist", "allowed_hosts": [GATEWAY_HOST]},
           "verifier": {"timeout_sec": 60.0},
           "environment": env}
    (out / "task.toml").write_text(dump_toml(doc), encoding="utf-8")
    artifacts = "\n".join(f"- `{a}`" for a in t.get("artifacts", []))
    template = (ROOT / "templates" / v["template"]).read_text(encoding="utf-8")
    (out / "instruction.md").write_text(template.format(
        instruction=_without_canary((task_dir / "instruction.md").read_text(encoding="utf-8")), artifacts=artifacts,
        timeout=int(float((t.get("verifier") or {}).get("timeout_sec") or 300)),
        env_image=env.get("docker_image", "(built from the task's environment/Dockerfile)"),
        minutes=int(v["timeout"] // 60)), encoding="utf-8")
    (out / "tests").mkdir()
    (out / "tests" / "test.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/verifier\n"
        "if [ -s /verifier_out/tests/test.sh ] && [ -s /verifier_out/tests/Dockerfile ]; then\n"
        "  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n", encoding="utf-8")
    (out / "tests" / "test.sh").chmod(0o755)
    return out


def run_writer(writer: str, tasks_dir: Path, out_root: Path, attempts: int, n_concurrent: int = 2,
               include: list[str] | None = None) -> Path:
    w = ALL_WRITERS[writer]
    if "HUD_API_KEY" not in os.environ:
        raise RuntimeError("HUD_API_KEY not loaded (expected in /root/vbench/.env)")
    cmd = [HARBOR, "run", "-p", str(tasks_dir), "-a", w["agent"], "-m", w["model"], "-k", str(attempts),
           "-n", str(n_concurrent), "-o", str(out_root), "--job-name", writer, "-y", "-q",
           "--allow-agent-host", GATEWAY_HOST]
    for k, v in w["env"].items():
        cmd += ["--ae", f"{k}={v}"]
    for kw in w["kwargs"]:
        cmd += ["--ak", kw]
    for name in include or []:
        cmd += ["-i", name]
    env = os.environ.copy()
    if w["host_key"]:
        env[w["host_key"]] = os.environ["HUD_API_KEY"]      # host-side harness only; never in argv
    Path(out_root).mkdir(parents=True, exist_ok=True)
    with (Path(out_root) / f"{writer}.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return Path(out_root) / writer


def _env_image(task: str) -> str:
    return (tomllib.loads((TB_ROOT / task / "task.toml").read_text(encoding="utf-8")).get("environment") or {}).get(
        "docker_image", "")


def _allowed(task: str) -> tuple[str, ...]:
    """The environment image, with and without its digest (a mistyped digest is not a fetch of anything else)."""
    img = _env_image(task)
    return tuple(x for x in (img, img.split("@")[0]) if x)


def collect(job_dir: Path, writer: str, dest: Path | None = None) -> list[dict]:
    dest = Path(dest) if dest is not None else DATA / "verifiers"
    rows = []
    for rj in sorted(Path(job_dir).glob("*/result.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        task = r["task_name"].split("/")[-1].removeprefix("write-verifier-")
        src = rj.parent / "artifacts" / "verifier_out" / "tests"
        ok = (src / "test.sh").is_file() and (src / "Dockerfile").is_file()
        target = dest / writer / task / rj.parent.name / "tests"
        if ok:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        rows.append({"writer": writer, "task": task, "trial": rj.parent.name, "ok": ok,
                     "verifier": str(target) if ok else None, "exception": r.get("exception_info"),
                     "leak": scan(target, TB_ROOT / task / "tests", _allowed(task), TB_ROOT / task / "environment") if ok else [],
                     "agent_result": r.get("agent_result")})
    write_json(dest / writer / f"collect_{Path(job_dir).parent.name}.json", rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--collect")
    ap.add_argument("--writer")
    ap.add_argument("--variant", default="v1", choices=sorted(VARIANTS))
    a = ap.parse_args()
    tasks = read_json(ROOT / "configs" / "m1.json")["tasks"]
    if a.make:
        for t in tasks:
            print(make_writer_task(TB_ROOT / t, variant=a.variant))
    if a.run:
        print("loaded:", load_env())
        run_dir = new_run_dir(f"write_{a.run}")
        include = [f"write-verifier-{t}" for t in a.tasks.split(",") if t]
        tasks_dir = VARIANTS["v2" if a.run.endswith("-v2") else "v1"]["tasks_dir"]
        job = run_writer(a.run, tasks_dir, run_dir, a.attempts, include=include)
        rows = collect(job, a.run)
        print(json.dumps([{k: r[k] for k in ("task", "trial", "ok", "leak")} for r in rows], indent=1))
    if a.collect:
        print(json.dumps(collect(Path(a.collect), a.writer), indent=1)[:4000])
