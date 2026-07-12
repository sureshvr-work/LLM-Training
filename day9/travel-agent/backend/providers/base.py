"""
base.py — the Provider contract (the *engine seam*).

REASON is the only step of the loop that differs between "no model", OpenAI,
and Anthropic. Every engine implements ONE method:

    reason(messages, tools) -> Decision

Given the context so far (messages) and the tools it may call, the engine
returns a Decision: a tool call to make, or a final answer.

The loop depends only on this interface — never on a vendor SDK. To add a new
engine, write one class and one line in the factory below.
"""
from abc import ABC, abstractmethod
from schema import Message, ToolSpec, Decision


class Provider(ABC):
    name: str = "base"

    @abstractmethod
    def reason(self, messages: list[Message], tools: list[ToolSpec]) -> Decision:
        """Read the context + available tools, decide the next action."""
        ...


def get_provider(kind: str) -> Provider:
    """Factory: map the UI's engine choice to a Provider instance.

    Imports are done lazily *inside* each branch so the mock engine never needs
    the OpenAI/Anthropic SDKs installed.
    """
    kind = (kind or "mock").lower()

    if kind in ("mock", "none", "no model"):
        from providers.mock import MockProvider
        return MockProvider()

    if kind in ("openai", "oai"):
        from providers.openai_provider import OpenAIProvider
        return OpenAIProvider()

    if kind in ("anthropic", "ant", "claude"):
        from providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    raise ValueError(f"unknown provider: {kind}")
