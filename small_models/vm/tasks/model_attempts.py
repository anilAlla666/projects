"""Real fixes from an open-weight coding model run locally with vLLM: realistic near misses and correct fixes."""
from __future__ import annotations

import ast
import re

from vm.tasks.attempts import Attempt, _function_span

FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def prompt(task: dict) -> str:
    numbered = "\n".join(f"{i + 1:4d}  {ln}" for i, ln in enumerate(task["function_lines"]))
    return (f"{task['task_text']}\n{numbered}\n\nFix the bug so the test passes without breaking anything else. "
            "Reply with the whole corrected function in one ```python code block, then one short paragraph "
            "explaining the change.")


def parse(task: dict, reply: str) -> Attempt | None:
    m = FENCE.search(reply)
    if not m:
        return None
    code = m.group(1)
    try:
        new_fn = _function_span(code, task["function"])
    except SyntaxError:
        return None
    if new_fn is None:
        return None
    new_lines = code.splitlines(keepends=True)[new_fn.lineno - 1 : new_fn.end_lineno]
    src = task["broken_source"]
    old = _function_span(src, task["function"])
    lines = src.splitlines(keepends=True)
    indent = lines[old.lineno - 1][: len(lines[old.lineno - 1]) - len(lines[old.lineno - 1].lstrip())]
    base = new_lines[0][: len(new_lines[0]) - len(new_lines[0].lstrip())]
    shifted = [indent + ln[len(base):] if ln.startswith(base) else ln for ln in new_lines]
    if not shifted[-1].endswith("\n"):
        shifted[-1] += "\n"
    new_src = "".join(lines[: old.lineno - 1] + shifted + lines[old.end_lineno:])
    try:
        ast.parse(new_src)
    except SyntaxError:
        return None
    explanation = reply[m.end():].strip()[:1000]
    return Attempt("model_attempt", {task["path"]: new_src}, [explanation] if explanation else [], explanation)


def generate(tasks: list, model: str, n: int = 4, temperature: float = 0.8, tp: int = 1):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.85, tensor_parallel_size=tp)
    params = SamplingParams(n=n, temperature=temperature, max_tokens=1024, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": prompt(t)}], tokenize=False, add_generation_prompt=True)
               for t in tasks]
    out = []
    for t, res in zip(tasks, llm.generate(prompts, params)):
        for o in res.outputs:
            a = parse(t, o.text)
            if a is not None:
                out.append((t["task_id"], a))
    return out
