"""
schema.py — the small, shared vocabulary that every other module speaks.

These types are deliberately *provider-agnostic*. The loop, the tools, and the
memory all pass around THESE shapes. Each engine (mock / OpenAI / Anthropic) is
the only place that knows how to translate between these shapes and a vendor's
API. That single translation point is what lets us swap engines without ever
touching the loop.

Nothing here imports a vendor SDK. On purpose.
"""
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# A single line in the conversation. The list of these IS the "context window"
# (the prompt) that grows every turn and gets re-sent to the model each time.
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None      # for role == "tool": which tool produced it


@dataclass
class ToolSpec:
    """A tool's advertisement to the model: what it's called, what it does,
    and the JSON-schema for its arguments."""
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    """The model's request to run a tool: which one + the arguments it chose."""
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """What a tool hands back after we run it (plain, JSON-serialisable data)."""
    name: str
    content: Any


@dataclass
class Decision:
    """The output of the REASON step. Either some tool_calls to run this turn,
    or a final answer (when `final` is set, the agent is finished)."""
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    final: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.final is not None
