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
python -m agents.evals.run_evals --label prompt-v2 --system agents/evals/prompt_v2.txt --judge --repeat 3

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

Model `claude-sonnet-5-5`, 12 cases × 3 repeats = 36 runs per row, LLM judge
off. Both rows were run on 2026-10-29 with:

```bash
python -m agents.evals.run_evals --label baseline --repeat 3
python -m agents.evals.run_evals --label prompt-v2 --system agents/evals/prompt_v2.txt --repeat 3
```

| Run | Pass rate | Avg tool calls | Avg tokens (in / out) | Notes |
|---|---|---|---|---|
| baseline | 92% (33/36) | 1.44 | 1200 / 183 | default system prompt; `calc_basic` failed 3/3 because the answer was `**7,006,652**`, so the check for `7006652` missed it |
| prompt-v2 | 100% (36/36) | 1.42 | 1200 / 175 | [`prompt_v2.txt`](prompt_v2.txt): default prompt plus one rule to report tool values exactly as returned; `calc_basic` passed 3/3, no regressions |

The only difference between the runs is `calc_basic`. All three baseline
answers gave the correct product, 7,006,652, with thousands separators, so
the failure was in how the answer was formatted, not in the arithmetic.

With the judge off, `calc_div_zero` and `missing_file` were graded on their
tool checks only.

The token figures come from `MessageHistory.message_tokens`, which counts
each turn's input as the context growth since the previous turn, starting
after the system prompt. They show how much context a run used, not billed
input tokens, so the longer prompt-v2 system prompt doesn't show up in them.
