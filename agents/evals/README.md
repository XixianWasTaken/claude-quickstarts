# Agent Eval Harness

An evaluation suite for the minimal agent in `agents/`. It records what the
agent actually does (every tool call, error, token count and latency) and
grades each run in three layers, so changes to the prompt, tools or model can
be measured instead of guessed.

> Extension built on top of Anthropic's `claude-quickstarts/agents` reference
> implementation. The base agent is unchanged; everything here lives in
> `agents/evals/`.

## What it does

| Piece | File | Purpose |
|---|---|---|
| Trajectory recorder | `recorder.py` | `RecordingAgent` wraps every tool (local and MCP) to log name, input, output, error flag and latency, plus turns and tokens per run |
| Layered grader | `grader.py` | **tools**: required / forbidden tools, call budget, no tool errors · **outcome**: answer contents, files written, files left untouched · **judge**: optional LLM rubric, only run when the cheap layers pass |
| Test cases | `cases.jsonl` | 12 cases across calculator use, file reads and edits, multi-step tasks, error handling and restraint (not calling tools when none are needed) |
| Runner | `run_evals.py` | Runs each case in an isolated temp directory, prints a report, saves JSON, and compares two runs (pass rate, cost, newly passing cases, regressions) |
| Tests | `test_evals.py` | Offline tests with a scripted fake client, no API key needed |

## Usage

```bash
pip install "anthropic>=2.0.0,<3" "mcp>=2.0" pytest
export ANTHROPIC_API_KEY=...

# offline tests of the harness itself
pytest agents/evals -q

# baseline run
python -m agents.evals.run_evals --label baseline

# try a change, e.g. a new system prompt, with the LLM judge and 3 repeats per case
python -m agents.evals.run_evals --label prompt-v2 --system prompt_v2.txt --judge --repeat 3

# compare
python -m agents.evals.run_evals --compare results/baseline.json results/prompt-v2.json
```

## Adding a case

One JSON object per line in `cases.jsonl`:

```json
{"id": "read_fact", "category": "file_read",
 "prompt": "Read config.txt and tell me which port the server uses.",
 "setup_files": {"config.txt": "port=8743\n"},
 "expect": {"tools_called": ["file_read"], "max_tool_calls": 3, "answer_contains": ["8743"]}}
```

Supported `expect` keys: `tools_called`, `tools_not_called`, `max_tool_calls`,
`no_tool_errors`, `answer_contains`, `answer_not_contains`, `file_contains`,
`file_unchanged`, `judge_rubric`.

## Results

| Run | Pass rate | Avg tool calls | Avg tokens | Notes |
|---|---|---|---|---|
| baseline | _fill in_ | | | default system prompt |
| | | | | |
