"""
registry.py — a tiny tool registry.

A "tool" is just a Python function plus a ToolSpec (its name / description /
parameter schema). The agent never calls the internet directly — it asks to
call a tool *by name*, and this registry runs the matching function.

Swapping a mock tool for a live one later means registering a different
function under the same name. The loop doesn't change.
"""
from typing import Callable
from schema import ToolSpec, ToolCall, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, Callable[..., object]] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, fn: Callable[..., object]) -> None:
        """Make a tool callable by name, and remember how to describe it."""
        self._fns[spec.name] = fn
        self._specs[spec.name] = spec

    def specs(self) -> list[ToolSpec]:
        """All tool descriptions — handed to the model so it knows its options."""
        return list(self._specs.values())

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call and wrap its return value as a ToolResult."""
        if call.name not in self._fns:
            raise KeyError(f"unknown tool: {call.name}")
        output = self._fns[call.name](**call.arguments)   # call the function
        return ToolResult(name=call.name, content=output)

    def execute_by_name(self, name: str, **arguments) -> ToolResult:
        """Convenience for tools the loop calls itself (e.g. get_user_profile)."""
        return self.execute(ToolCall(name=name, arguments=arguments))
