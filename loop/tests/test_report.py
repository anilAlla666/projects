from vb.report import render


def m(ba):
    return {"n_tasks": 5, "tpr": ba, "tnr": ba, "ba": ba, "flakiness": 0.0}


SCORES = {"name": "m1", "baseline_problems": [], "writer_cost_usd": {"sonnet-claude-code": 1.25},
          "writers": {"sonnet-claude-code": {"model_id": "sonnet", "harness": "claude-code"},
                      "sonnet-terminus": {"model_id": "sonnet", "harness": "terminus-2"},
                      "luna-codex": {"model_id": "luna", "harness": "codex"},
                      "luna-terminus": {"model_id": "luna", "harness": "terminus-2"}},
          "per_task": {"demo": {
              "always_pass": {"tpr": 1.0, "tnr": 0.0, "ba": 0.5, "flakiness": 0.0, "tnr_by_source": {"real_fail": 0.0}, "n_correct": 11, "n_wrong": 11},
              "sonnet-claude-code/t1": {"tpr": 0.9, "tnr": 0.8, "ba": 0.85, "flakiness": 0.05, "tnr_by_source": {"near_miss": 0.6, "real_fail": 1.0}, "n_correct": 11, "n_wrong": 11}}},
          "macro": {"sonnet-claude-code": m(0.85), "sonnet-terminus": m(0.7), "luna-codex": m(0.8), "luna-terminus": m(0.65)}}


def test_render_names_every_writer_and_the_baseline_status():
    md = render(SCORES)
    assert "sonnet-claude-code" in md and "85.0" in md and "60.0" in md and "$1.25" in md
    assert "Baselines: all hold" in md
    assert "NOT VALID" in render({**SCORES, "baseline_problems": ["demo: always_fail: tnr = 0.9"]})


def test_render_model_by_harness_table():
    md = render(SCORES)
    grid = md.split("## Model × harness")[1].split("##")[0]
    assert "| sonnet | 85.0 | n/a | 70.0 |" in grid          # columns: claude-code, codex, terminus-2
    assert "| luna | n/a | 80.0 | 65.0 |" in grid
    with_ci = {**SCORES, "macro": {**SCORES["macro"], "luna-codex": {**m(0.8), "ba_ci": (0.6, 0.9)}}}
    assert "80.0 [60.0, 90.0]" in render(with_ci)


def test_render_lists_leaks():
    md = render({**SCORES, "leaks": {"luna-codex/t1": ["Dockerfile: mentions github.com"]}})
    assert "excluded" in md and "luna-codex/t1" in md and "github.com" in md


def test_render_lists_attempts_without_a_verifier():
    per = {"demo": {**SCORES["per_task"]["demo"], "sonnet-terminus/t9": {"tpr": 0.0, "tnr": 1.0, "ba": 0.5, "flakiness": 1.0,
                    "tnr_by_source": {}, "n_correct": 11, "n_wrong": 10, "no_verifier": True}}}
    md = render({**SCORES, "per_task": per})
    assert "produced no verifier" in md and "demo: sonnet-terminus/t9" in md
