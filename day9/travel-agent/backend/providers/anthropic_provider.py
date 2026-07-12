"""
anthropic_provider.py — the Anthropic engine.   [used in stage 3; needs a key]

Same adapter idea as the OpenAI one, but Anthropic's Messages API shapes things
differently — which is exactly why the seam earns its keep:

    tools    : ToolSpec        -> {"name","description","input_schema"}
    response : "tool_use" block -> Decision(tool_calls=[...])
    system    : a top-level argument (not a message in the list)

The loop is unaffected; it still just receives a Decision.

Requires:  pip install anthropic   and   ANTHROPIC_API_KEY in the environment.
"""
import os
from providers.base import Provider
from schema import Message, ToolSpec, Decision, ToolCall


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        import anthropic                       # lazy import: mock needs no SDK
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def _tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [{"name": t.name, "description": t.description,
                 "input_schema": t.parameters} for t in tools]

    def _messages(self, messages: list[Message]) -> list[dict]:
        # Anthropic takes `system` separately and wants user/assistant turns.
        # NOTE: a production adapter sends tool results as structured
        # "tool_result" blocks; flattened to text here for readability.
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            role = "assistant" if m.role == "assistant" else "user"
            out.append({"role": role, "content": m.content})
        return out

    def reason(self, messages, tools) -> Decision:
        system = next((m.content for m in messages if m.role == "system"), "")
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, system=system,
            messages=self._messages(messages), tools=self._tools(tools),
        )
        thought, calls = "", []
        for block in resp.content:               # response is a list of blocks
            if block.type == "text":
                thought += block.text
            elif block.type == "tool_use":
                calls.append(ToolCall(block.name, dict(block.input)))
        if calls:
            return Decision(thought=thought, tool_calls=calls)
        return Decision(thought=thought, final=thought)
