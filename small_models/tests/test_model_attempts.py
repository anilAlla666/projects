from vm.tasks.model_attempts import parse, prompt

SRC = "import os\n\n\ndef bigger(a, b):\n    if a > b:\n        return a\n    return b\n\n\ndef other():\n    return 1\n"
TASK = {"path": "lib.py", "function": "bigger", "broken_source": SRC, "task_text": "The test fails.\n",
        "function_lines": SRC.splitlines()[3:7]}


def test_parse_replaces_only_the_function():
    reply = "Here is the fix:\n```python\ndef bigger(a, b):\n    if a >= b:\n        return a\n    return b\n```\nUse >= so ties return a."
    a = parse(TASK, reply)
    assert a.kind == "model_attempt"
    new = a.files["lib.py"]
    assert "if a >= b:" in new and "def other():\n    return 1" in new and new.startswith("import os")
    assert a.untrusted == ["Use >= so ties return a."]


def test_parse_rejects_unusable_replies():
    assert parse(TASK, "no code here") is None
    assert parse(TASK, "```python\ndef bigger(a, b:\n```") is None
    assert parse(TASK, "```python\ndef something_else():\n    pass\n```") is None


def test_prompt_numbers_the_lines():
    assert "   1  def bigger(a, b):" in prompt(TASK)
