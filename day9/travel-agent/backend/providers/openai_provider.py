"""
openai_provider.py — the OpenAI engine.   [used in stage 3; needs a key]

Adapter pattern: translate our provider-agnostic schema <-> OpenAI's wire
format, in both directions:

    tools    : ToolSpec     -> {"type":"function","function":{...}}
    history  : Message      -> {"role":...,"content":...}
    response : tool_calls   -> Decision(tool_calls=[...])

The loop never sees any of this — it only gets a Decision back. That's the whole
point of the adapter: OpenAI-specific knowledge stops here.

Requires:  pip install openai   and   OPENAI_API_KEY in the environment.
"""
import json
import os
from providers.base import Provider
from schema import Message, ToolSpec, Decision, ToolCall


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o") -> None:
        from openai import OpenAI            # lazy import: mock runs need no SDK
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    # --- our schema -> OpenAI request --------------------------------------
    def _tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [{"type": "function",
                 "function": {"name": t.name,
                              "description": t.description,
                              "parameters": t.parameters}} for t in tools]

    def _messages(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                # NOTE: a production adapter must thread the assistant message
                # that requested the call and match `tool_call_id` exactly.
                # Simplified here for readability.
                out.append({"role": "tool", "name": m.name,
                            "content": m.content, "tool_call_id": m.name})
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    # --- OpenAI response -> our Decision -----------------------------------
    def reason(self, messages, tools) -> Decision:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(messages),
            tools=self._tools(tools),
        )
        msg = resp.choices[0].message
        if msg.tool_calls:                       # the model wants to call tools
            calls = [ToolCall(tc.function.name,
                              json.loads(tc.function.arguments or "{}"))
                     for tc in msg.tool_calls]
            return Decision(thought=msg.content or "", tool_calls=calls)
        return Decision(thought=msg.content or "", final=msg.content or "")
