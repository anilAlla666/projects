"""Step 17 checks before any test: through the live server, loss on 3 positive-advantage segments must fall and on 3
negative-advantage segments must rise versus base, and a real first turn must still give a valid bash tool call."""
import json, random, sys
sys.path.insert(0, "/home/ubuntu/vbrepo2/tools")
import httpx2 as httpx
from transformers import AutoTokenizer
import train_judge as tj
tj.VIEW.update(cut=10**9, max_len=98304, seg=8)
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/qwen38-27b")
convs = [json.loads(l) for l in open("/home/ubuntu/vbrepo2/runs/step13/" + __import__("os").environ.get("RL_DATA", "rl1.jsonl") + "")]
random.Random(3).shuffle(convs)
def nll(model, ids, labels):
    r = httpx.post("http://127.0.0.1:8000/v1/completions", timeout=1800, json={
        "model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    pl = r.json()["choices"][0]["prompt_logprobs"]
    v = [pl[i][str(ids[i])]["logprob"] for i in range(1, len(ids)) if labels[i] != -100]
    return -sum(v) / len(v)
ok = True
for sign in (1, -1):
    picked = [c for c in convs if c["weight"] * sign > 0.5][:3]
    for c in picked:
        ex = [x for x in tj.examples(c, tok) if len(x[0]) < 40000][:1]
        if not ex:
            continue
        ids, labels = ex[0]
        b, t = nll("qwen38-27b", ids, labels), nll(__import__("os").environ.get("RL_NAME", "trained4"), ids, labels)
        good = (t < b) if sign > 0 else (t > b)
        ok &= good
        print(f"{'positive' if sign > 0 else 'negative'} weight {c['weight']:+.2f}: base {b:.3f} trained {t:.3f} {'OK' if good else 'WRONG WAY'}")
c = convs[0]
r = httpx.post("http://127.0.0.1:8000/v1/chat/completions", timeout=1800, json={
    "model": __import__("os").environ.get("RL_NAME", "trained4"), "max_tokens": 16000, "chat_template_kwargs": {"reasoning_effort": "medium"},
    "messages": [{"role": "system", "content": c["system"]}, c["messages"][0]], "tools": c["tools"]})
m = r.json()["choices"][0]["message"]
calls = m.get("tool_calls") or []
valid = bool(calls) and all(isinstance(json.loads(x["function"]["arguments"]).get("command"), str) for x in calls)
print("TOOL_CALL", "VALID" if valid else "INVALID", [x["function"]["arguments"][:120] for x in calls])
print("RL_CHECK", "PASS" if ok and valid else "FAIL")
