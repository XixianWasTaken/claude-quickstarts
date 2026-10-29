"""Run the agent eval suite and write a report.

Usage (from the repository root):

    python -m agents.evals.run_evals --label baseline
    python -m agents.evals.run_evals --label new-prompt --system prompts/v2.txt --judge
    python -m agents.evals.run_evals --compare results/baseline.json results/new-prompt.json

Each case runs in its own temporary directory, so file tools never touch the
real working tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..agent import ModelConfig
from ..tools import FileReadTool, FileWriteTool, ThinkTool
from .grader import CaseResult, CheckResult, grade_case
from .recorder import RecordingAgent, Trajectory

EVALS_DIR = Path(__file__).resolve().parent
CALCULATOR_SERVER = EVALS_DIR.parent / "tools" / "calculator_mcp.py"

DEFAULT_SYSTEM = (
    "You are a careful assistant working in the current directory. "
    "Use tools when they are needed and avoid unnecessary calls. "
    "Never invent file contents: if a file cannot be read, say so."
)


def load_cases(path: Path, only: list[str] | None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if only:
        cases = [c for c in cases if c["id"] in only]
    return cases


def build_agent(system: str, model: str, client: Any) -> RecordingAgent:
    return RecordingAgent(
        name="EvalAgent",
        system=system,
        tools=[ThinkTool(), FileReadTool(), FileWriteTool()],
        mcp_servers=[
            {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(CALCULATOR_SERVER)],
            }
        ],
        config=ModelConfig(model=model),
        client=client,
    )


def run_case(
    case: dict[str, Any], system: str, model: str, client: Any, judge: bool
) -> tuple[CaseResult, Trajectory]:
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"eval_{case['id']}_") as tmp:
        sandbox = Path(tmp)
        for rel, content in case.get("setup_files", {}).items():
            (sandbox / rel).parent.mkdir(parents=True, exist_ok=True)
            (sandbox / rel).write_text(content)

        agent = build_agent(system, model, client)
        os.chdir(sandbox)
        try:
            agent.run(case["prompt"])
            error = None
        except Exception as e:  # a crashed run is a failed case, not a crashed suite
            error = f"{type(e).__name__}: {e}"
        finally:
            os.chdir(original_cwd)

        traj = agent.trajectory
        if error:
            result = CaseResult(case["id"], [CheckResult("run", "completed", False, error)])
        else:
            result = grade_case(case, traj, sandbox, judge_client=client if judge else None)
    return result, traj


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["passed"])
    n = len(rows) or 1
    return {
        "pass_rate": sum(r["passed"] for r in rows) / n,
        "by_category": {k: sum(v) / len(v) for k, v in sorted(by_cat.items())},
        "avg_tool_calls": sum(r["tool_calls"] for r in rows) / n,
        "avg_input_tokens": sum(r["input_tokens"] for r in rows) / n,
        "avg_output_tokens": sum(r["output_tokens"] for r in rows) / n,
        "avg_wall_time_s": sum(r["wall_time_s"] for r in rows) / n,
        "n_runs": len(rows),
    }


def print_report(label: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'case':<22}{'pass':<6}{'calls':<7}{'tokens':<9}failed checks")
    for r in rows:
        failed = ", ".join(c["name"] for c in r["checks"] if not c["passed"])
        mark = "PASS" if r["passed"] else "FAIL"
        tokens = r["input_tokens"] + r["output_tokens"]
        print(f"{r['case_id']:<22}{mark:<6}{r['tool_calls']:<7}{tokens:<9}{failed}")
    print(f"\npass rate: {summary['pass_rate']:.0%}  ({summary['n_runs']} runs)")
    for cat, rate in summary["by_category"].items():
        print(f"  {cat:<16}{rate:.0%}")
    print(
        f"avg tool calls {summary['avg_tool_calls']:.1f} | "
        f"avg tokens in/out {summary['avg_input_tokens']:.0f}/{summary['avg_output_tokens']:.0f} | "
        f"avg time {summary['avg_wall_time_s']:.1f}s"
    )


def compare(path_a: Path, path_b: Path) -> None:
    a, b = json.loads(path_a.read_text()), json.loads(path_b.read_text())
    sa, sb = a["summary"], b["summary"]
    print(f"\n{'metric':<22}{a['label']:>14}{b['label']:>14}")
    print(f"{'pass rate':<22}{sa['pass_rate']:>14.0%}{sb['pass_rate']:>14.0%}")
    for key in ("avg_tool_calls", "avg_input_tokens", "avg_output_tokens", "avg_wall_time_s"):
        print(f"{key:<22}{sa[key]:>14.1f}{sb[key]:>14.1f}")
    cats = sorted(set(sa["by_category"]) | set(sb["by_category"]))
    for cat in cats:
        ra, rb = sa["by_category"].get(cat), sb["by_category"].get(cat)
        fa = "-" if ra is None else f"{ra:.0%}"
        fb = "-" if rb is None else f"{rb:.0%}"
        print(f"  {cat:<20}{fa:>14}{fb:>14}")

    def outcomes(run):
        return {(r["case_id"], r["repeat"]): r["passed"] for r in run["rows"]}

    oa, ob = outcomes(a), outcomes(b)
    fixed = sorted({k[0] for k in oa if k in ob and not oa[k] and ob[k]})
    broke = sorted({k[0] for k in oa if k in ob and oa[k] and not ob[k]})
    print(f"\nnewly passing: {fixed or 'none'}")
    print(f"regressions:   {broke or 'none'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=EVALS_DIR / "cases.jsonl")
    parser.add_argument("--label", default="run")
    parser.add_argument("--model", default=ModelConfig().model)
    parser.add_argument("--system", type=Path, help="file with a system prompt to test")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case (checks consistency)")
    parser.add_argument("--only", nargs="*", help="run only these case ids")
    parser.add_argument("--judge", action="store_true", help="enable the LLM-judge layer")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A.json", "B.json"))
    args = parser.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return 0

    from anthropic import Anthropic

    client = Anthropic()
    system = args.system.read_text() if args.system else DEFAULT_SYSTEM
    cases = load_cases(args.cases, args.only)

    rows = []
    for case in cases:
        for i in range(args.repeat):
            result, traj = run_case(case, system, args.model, client, args.judge)
            rows.append(
                {
                    "case_id": case["id"],
                    "category": case.get("category", "uncategorized"),
                    "repeat": i,
                    "passed": result.passed,
                    "checks": [asdict(c) for c in result.checks],
                    "tool_calls": len(traj.calls),
                    "tool_sequence": traj.tool_names,
                    "tool_errors": traj.error_count,
                    "turns": traj.turns,
                    "input_tokens": traj.input_tokens,
                    "output_tokens": traj.output_tokens,
                    "wall_time_s": traj.wall_time_s,
                    "final_text": traj.final_text,
                }
            )
            print(f"  {case['id']} #{i}: {'PASS' if result.passed else 'FAIL'}")

    summary = summarize(rows)
    print_report(args.label, rows, summary)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{args.label}.json"
    out_file.write_text(
        json.dumps(
            {"label": args.label, "model": args.model, "system": system,
             "summary": summary, "rows": rows},
            indent=2,
        )
    )
    print(f"\nsaved {out_file}")
    return 0 if summary["pass_rate"] == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
