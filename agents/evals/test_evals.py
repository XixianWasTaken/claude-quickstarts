"""Offline tests for the eval harness (no API key or network needed; a scripted
fake client stands in for the Anthropic API).

Run from the repository root:  pytest agents/evals -q
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from agents.evals.grader import grade_case, grade_outcome, grade_tools  # noqa: E402
from agents.evals.recorder import RecordingAgent, ToolCall, Trajectory  # noqa: E402
from agents.evals.run_evals import compare, summarize  # noqa: E402
from agents.tools import FileReadTool, FileWriteTool  # noqa: E402


# ---------- a scripted stand-in for the Anthropic client ----------

def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(id_, name, **inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        content = self.script.pop(0)
        usage = SimpleNamespace(input_tokens=100 * len(self.calls), output_tokens=20)
        return SimpleNamespace(content=content, usage=usage)

    def count_tokens(self, **_):
        raise RuntimeError("offline")


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


def make_agent(script):
    return RecordingAgent(
        name="t",
        system="sys",
        tools=[FileReadTool(), FileWriteTool()],
        client=FakeClient(script),
    )


# ---------- recorder ----------

def test_records_tool_calls_and_final_text(tmp_path, monkeypatch):
    (tmp_path / "config.txt").write_text("port=8743\n")
    monkeypatch.chdir(tmp_path)
    agent = make_agent([
        [tool_use("t1", "file_read", operation="read", path="config.txt")],
        [text("The port is 8743.")],
    ])
    agent.run("which port?")

    traj = agent.trajectory
    assert traj.tool_names == ["file_read"]
    assert "8743" in traj.calls[0].output
    assert not traj.calls[0].is_error
    assert traj.final_text == "The port is 8743."
    assert traj.turns == 2
    assert traj.output_tokens == 40


def test_original_tools_restored_after_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = make_agent([[text("hi")]])
    before = [type(t) for t in agent.tools]
    agent.run("hello")
    assert [type(t) for t in agent.tools] == before


def test_string_errors_are_flagged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = make_agent([
        [tool_use("t1", "file_read", operation="read", path="nope.txt")],
        [text("That file does not exist.")],
    ])
    agent.run("read nope.txt")
    assert agent.trajectory.error_count == 1


def test_trajectory_resets_between_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x")
    agent = make_agent([
        [tool_use("t1", "file_read", operation="read", path="a.txt")],
        [text("done")],
        [text("no tools this time")],
    ])
    agent.run("first")
    agent.run("second")
    assert agent.trajectory.tool_names == []


# ---------- grader ----------

def traj_with(names, final="", errors=()):
    t = Trajectory(final_text=final)
    for i, n in enumerate(names):
        t.calls.append(ToolCall(n, {}, "ok", i in errors, 0.0))
    return t


def test_tool_checks():
    expect = {"tools_called": ["file_read"], "tools_not_called": ["file_write"],
              "max_tool_calls": 2, "no_tool_errors": True}
    assert all(c.passed for c in grade_tools(expect, traj_with(["file_read"])))

    bad = grade_tools(expect, traj_with(["file_read", "file_write", "file_read"], errors={0}))
    failed = {c.name for c in bad if not c.passed}
    assert failed == {"not_called:file_write", "max_tool_calls", "no_tool_errors"}


def test_outcome_checks_answer_and_files(tmp_path):
    (tmp_path / "out.txt").write_text("Total:   50\n")
    (tmp_path / "keep.txt").write_text("orig")
    expect = {
        "answer_contains": ["Quarterly ROADMAP"],
        "answer_not_contains": ["password"],
        "file_contains": {"out.txt": ["total: 50"], "missing.txt": ["x"]},
        "file_unchanged": {"keep.txt": "orig"},
    }
    checks = grade_outcome(expect, traj_with([], final="Title: quarterly  roadmap"), tmp_path)
    failed = {c.name for c in checks if not c.passed}
    assert failed == {"file_contains:missing.txt:x"}


def test_judge_skipped_when_deterministic_layers_fail(tmp_path):
    judge = FakeClient([[text("PASS\nlooks fine")]])
    case = {"id": "c", "prompt": "p",
            "expect": {"tools_called": ["calculator"], "judge_rubric": "r"}}
    result = grade_case(case, traj_with([]), tmp_path, judge_client=judge)
    assert not result.passed
    assert judge.messages.calls == []


def test_judge_runs_and_parses_verdict(tmp_path):
    judge = FakeClient([[text("FAIL\ninvented contents")]])
    case = {"id": "c", "prompt": "p", "expect": {"judge_rubric": "r"}}
    result = grade_case(case, traj_with([], final="it says hello"), tmp_path, judge_client=judge)
    assert [c.layer for c in result.checks] == ["judge"]
    assert not result.passed


# ---------- reporting ----------

def row(case_id, cat, passed, rep=0):
    return {"case_id": case_id, "category": cat, "repeat": rep, "passed": passed,
            "tool_calls": 2, "input_tokens": 100, "output_tokens": 10, "wall_time_s": 1.0}


def test_summarize_and_compare(tmp_path, capsys):
    a_rows = [row("x", "calc", False), row("y", "calc", True), row("z", "file", True)]
    b_rows = [row("x", "calc", True), row("y", "calc", True), row("z", "file", False)]
    sa, sb = summarize(a_rows), summarize(b_rows)
    assert sa["by_category"] == {"calc": 0.5, "file": 1.0}

    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps({"label": "a", "summary": sa, "rows": a_rows}))
    pb.write_text(json.dumps({"label": "b", "summary": sb, "rows": b_rows}))
    compare(pa, pb)
    out = capsys.readouterr().out
    assert "newly passing: ['x']" in out
    assert "regressions:   ['z']" in out


def test_cases_file_is_valid():
    path = Path(__file__).with_name("cases.jsonl")
    ids = set()
    for line in path.read_text().splitlines():
        case = json.loads(line)
        assert {"id", "prompt", "expect"} <= case.keys()
        assert case["id"] not in ids
        ids.add(case["id"])
    assert len(ids) >= 10
