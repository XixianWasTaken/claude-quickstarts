"""Trajectory recording for agent evals.

Wraps every tool the agent can see (local tools and MCP tools alike) so each
call is logged with its input, result, error flag and latency. The base
Agent class is not modified: RecordingAgent only swaps in wrapped tools at
the start of the loop, and Agent.run_async already restores the originals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..agent import Agent
from ..tools.base import Tool


@dataclass
class ToolCall:
    """One tool invocation made by the agent."""

    name: str
    input: dict[str, Any]
    output: str
    is_error: bool
    latency_s: float


@dataclass
class Trajectory:
    """Everything the agent did during one run."""

    calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]

    @property
    def error_count(self) -> int:
        return sum(c.is_error for c in self.calls)


class RecordingTool(Tool):
    """Delegates to an inner tool and records each call into a Trajectory."""

    def __init__(self, inner: Tool, trajectory: Trajectory):
        super().__init__(
            name=inner.name,
            description=inner.description,
            input_schema=inner.input_schema,
        )
        self.inner = inner
        self.trajectory = trajectory

    def to_dict(self) -> dict[str, Any]:
        # Server-side tools (web search, code execution) define their own
        # API format, so always defer to the wrapped tool.
        return self.inner.to_dict()

    async def execute(self, **kwargs) -> str:
        start = time.perf_counter()
        try:
            result = await self.inner.execute(**kwargs)
        except Exception as e:
            self.trajectory.calls.append(
                ToolCall(self.name, kwargs, str(e), True, time.perf_counter() - start)
            )
            raise
        text = str(result)
        # Several tools in this repo report failures as strings, not exceptions.
        is_error = text.lstrip().lower().startswith("error")
        self.trajectory.calls.append(
            ToolCall(self.name, kwargs, text, is_error, time.perf_counter() - start)
        )
        return result


class RecordingAgent(Agent):
    """An Agent whose runs produce a Trajectory for grading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = Trajectory()

    async def _agent_loop(self, user_input: str):
        self.trajectory = Trajectory()
        # By now run_async has added the MCP tools, so they get wrapped too.
        self.tools = [RecordingTool(t, self.trajectory) for t in self.tools]

        start = time.perf_counter()
        response = await super()._agent_loop(user_input)
        self.trajectory.wall_time_s = time.perf_counter() - start

        self.trajectory.final_text = "".join(
            getattr(b, "text", "") for b in response.content if b.type == "text"
        )
        self.trajectory.turns = len(self.history.message_tokens)
        self.trajectory.input_tokens = sum(i for i, _ in self.history.message_tokens)
        self.trajectory.output_tokens = sum(o for _, o in self.history.message_tokens)
        return response
