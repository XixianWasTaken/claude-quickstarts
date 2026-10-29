"""Layered grading for agent trajectories.

Layer 1 (tools):   deterministic checks on which tools were called and how often.
Layer 2 (outcome): deterministic checks on the final answer and files on disk.
Layer 3 (judge):   optional LLM-as-judge against a free-text rubric.

Cheap deterministic layers run first; the judge only runs when they pass,
so an obviously broken run never costs a judge call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .recorder import Trajectory


@dataclass
class CheckResult:
    layer: str
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def grade_tools(expect: dict[str, Any], traj: Trajectory) -> list[CheckResult]:
    checks = []
    called = traj.tool_names

    for name in expect.get("tools_called", []):
        checks.append(
            CheckResult("tools", f"called:{name}", name in called, f"calls={called}")
        )
    for name in expect.get("tools_not_called", []):
        checks.append(
            CheckResult("tools", f"not_called:{name}", name not in called, f"calls={called}")
        )
    if "max_tool_calls" in expect:
        limit = expect["max_tool_calls"]
        checks.append(
            CheckResult("tools", "max_tool_calls", len(called) <= limit, f"{len(called)} > {limit}")
        )
    if expect.get("no_tool_errors"):
        errs = [c.name for c in traj.calls if c.is_error]
        checks.append(CheckResult("tools", "no_tool_errors", not errs, f"errors in {errs}"))
    return checks


def grade_outcome(
    expect: dict[str, Any], traj: Trajectory, sandbox: Path
) -> list[CheckResult]:
    checks = []
    answer = _normalize(traj.final_text)

    for needle in expect.get("answer_contains", []):
        checks.append(
            CheckResult(
                "outcome",
                f"answer_contains:{needle}",
                _normalize(needle) in answer,
                f"answer={traj.final_text[:200]!r}",
            )
        )
    for needle in expect.get("answer_not_contains", []):
        checks.append(
            CheckResult(
                "outcome",
                f"answer_not_contains:{needle}",
                _normalize(needle) not in answer,
                f"answer={traj.final_text[:200]!r}",
            )
        )
    for rel, needles in expect.get("file_contains", {}).items():
        path = sandbox / rel
        content = _normalize(path.read_text()) if path.exists() else None
        for needle in needles:
            ok = content is not None and _normalize(needle) in content
            detail = "file missing" if content is None else f"content={content[:200]!r}"
            checks.append(CheckResult("outcome", f"file_contains:{rel}:{needle}", ok, detail))
    for rel, original in expect.get("file_unchanged", {}).items():
        path = sandbox / rel
        ok = path.exists() and path.read_text() == original
        checks.append(CheckResult("outcome", f"file_unchanged:{rel}", ok, "file was modified"))
    return checks


JUDGE_PROMPT = """You are grading an AI agent's final answer.

Task given to the agent:
<task>{task}</task>

Agent's final answer:
<answer>{answer}</answer>

Rubric:
<rubric>{rubric}</rubric>

Does the answer satisfy every point of the rubric? Reply with PASS or FAIL on
the first line, then one sentence explaining why."""


def grade_judge(
    rubric: str, task: str, traj: Trajectory, client: Any, model: str
) -> CheckResult:
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    task=task, answer=traj.final_text, rubric=rubric
                ),
            }
        ],
    )
    text = "".join(getattr(b, "text", "") for b in response.content).strip()
    verdict = text.splitlines()[0].strip().upper() if text else ""
    return CheckResult("judge", "rubric", verdict.startswith("PASS"), text[:300])


def grade_case(
    case: dict[str, Any],
    traj: Trajectory,
    sandbox: Path,
    judge_client: Any | None = None,
    judge_model: str = "claude-haiku-4-5-20251001",
) -> CaseResult:
    expect = case.get("expect", {})
    result = CaseResult(case["id"])
    result.checks += grade_tools(expect, traj)
    result.checks += grade_outcome(expect, traj, sandbox)

    rubric = expect.get("judge_rubric")
    if rubric and judge_client is not None and result.passed:
        result.checks.append(
            grade_judge(rubric, case["prompt"], traj, judge_client, judge_model)
        )
    return result
