"""
engine.py — the REASON step. Three providers, one contract.

The router and every specialist share these adapters. Each provider does two jobs:

  • extract_intent(system, user) -> dict
        The router's one schema-constrained call: fuzzy words → a STRUCTURED
        intent object {system, acuity, signs}. JSON in, JSON out.

  • reason(history, tools) -> Decision
        A specialist's loop step: read the transcript + its OWN scoped tools,
        return a tool call (with the vendor's id) or a final disposition.

MockProvider needs no key — a deterministic stand-in so the whole mesh runs
offline in a classroom. Clearly labelled as a stand-in; never confuse the
heuristic with semantic extraction.
"""
import json
import re
from dataclasses import dataclass
from typing import Optional

from config import cfg


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Decision:
    thought: str = ""
    tool_call: Optional[ToolCall] = None
    final: Optional[str] = None
    usage: Optional[dict] = None


def _parse_json(text: str) -> dict:
    text = (text or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}


# ── Anthropic ────────────────────────────────────────────────────────────────
class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        self.model = cfg.ANTHROPIC_MODEL

    def extract_intent(self, system, user):
        resp = self.client.messages.create(
            model=self.model, max_tokens=400, system=system,
            messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_json(text)

    @staticmethod
    def _split(history):
        system = next((h["content"] for h in history if h["role"] == "system"), "")
        msgs = []
        for h in history:
            if h["role"] == "user":
                msgs.append({"role": "user", "content": h["content"]})
            elif h["role"] == "assistant":
                content = []
                if h.get("thought"):
                    content.append({"type": "text", "text": h["thought"]})
                if h.get("tool"):
                    t = h["tool"]
                    content.append({"type": "tool_use", "id": t["id"],
                                    "name": t["name"], "input": t["args"]})
                msgs.append({"role": "assistant", "content": content or "…"})
            elif h["role"] == "tool":
                msgs.append({"role": "user",
                             "content": [{"type": "tool_result",
                                          "tool_use_id": h["id"], "content": h["content"]}]})
        return system, msgs

    def reason(self, history, tools):
        system, msgs = self._split(history)
        resp = self.client.messages.create(
            model=self.model, max_tokens=1200, system=system, messages=msgs,
            tools=[{"name": t["name"], "description": t["description"],
                    "input_schema": t["parameters"]} for t in tools])
        thought = "".join(b.text for b in resp.content if b.type == "text")
        usage = {"prompt": resp.usage.input_tokens, "completion": resp.usage.output_tokens}
        for b in resp.content:
            if b.type == "tool_use":
                return Decision(thought=thought,
                                tool_call=ToolCall(b.id, b.name, dict(b.input)), usage=usage)
        return Decision(thought=thought, final=thought, usage=usage)


# ── OpenAI ───────────────────────────────────────────────────────────────────
class OpenAIProvider:
    name = "openai"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=cfg.OPENAI_API_KEY)
        self.model = cfg.OPENAI_MODEL

    def extract_intent(self, system, user):
        resp = self.client.chat.completions.create(
            model=self.model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return _parse_json(resp.choices[0].message.content)

    @staticmethod
    def _messages(history):
        out = []
        for h in history:
            if h["role"] in ("system", "user"):
                out.append({"role": h["role"], "content": h["content"]})
            elif h["role"] == "assistant":
                m = {"role": "assistant", "content": h.get("thought") or ""}
                if h.get("tool"):
                    t = h["tool"]
                    m["tool_calls"] = [{"id": t["id"], "type": "function",
                                        "function": {"name": t["name"],
                                                     "arguments": json.dumps(t["args"])}}]
                out.append(m)
            elif h["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": h["id"], "content": h["content"]})
        return out

    def reason(self, history, tools):
        resp = self.client.chat.completions.create(
            model=self.model, messages=self._messages(history),
            tools=[{"type": "function", "function": t} for t in tools])
        msg = resp.choices[0].message
        usage = ({"prompt": resp.usage.prompt_tokens,
                  "completion": resp.usage.completion_tokens} if resp.usage else None)
        if msg.tool_calls:
            c = msg.tool_calls[0]
            return Decision(thought=msg.content or "",
                            tool_call=ToolCall(c.id, c.function.name,
                                               json.loads(c.function.arguments or "{}")), usage=usage)
        return Decision(thought=msg.content or "", final=msg.content or "", usage=usage)


# ── Mock (no key — runs the whole mesh offline) ──────────────────────────────
_SYS_KEYWORDS = [
    ("cardiac", ("chest", "heart", "ticker", "palpitat", "arm", "cardiac")),
    ("respiratory", ("breath", "breathe", "wheez", "cough", "lung", "asthma", "respirat")),
    ("neuro", ("head", "slurred", "vision", "numb", "seizure", "dizzy", "stroke", "speech", "neuro")),
    ("dermatology", ("rash", "itch", "skin", "derm", "acne", "mole", "hives")),
    ("gastro", ("stomach", "nausea", "vomit", "abdomen", "belly", "gastro", "diarr")),
    ("ortho", ("bone", "fracture", "sprain", "joint", "knee", "ankle", "ortho")),
]
_HIGH = ("severe", "sudden", "can't", "cannot", "crushing", "worst", "slurred",
         "unconscious", "badly", "emergency", "intense", "tight")
_MOCK_ARGS = {
    "get_vitals": {"patient_id": "P-100"}, "order_ecg": {},
    "check_cath_lab": {}, "page_oncall": {"service": "on-call"},
    "lookup_protocol": {"topic": "triage"}, "order_labs": {"panel": "basic"},
    "order_imaging": {"kind": "scan"}, "check_beds": {"ward": "ICU"},
    "book_clinic": {"dept": "clinic"},
}


class MockProvider:
    name = "mock"

    def extract_intent(self, system, user):
        t = " " + (user or "").lower() + " "
        system_val = "general"
        for name, kws in _SYS_KEYWORDS:
            if any(k in t for k in kws):
                system_val = name
                break
        acuity = "HIGH" if any(k in t for k in _HIGH) else "LOW"
        signs = [s for s in ("chest_pain", "arm_pain", "wheezing", "rash",
                             "headache", "slurred_speech", "nausea")
                 if s.split("_")[0] in t][:4]
        return {"system": system_val, "acuity": acuity, "signs": signs}

    def reason(self, history, tools):
        used = sum(1 for h in history if h["role"] == "tool")
        names = [t["name"] for t in tools]
        if used < 1 and names:
            n = names[0]
            return Decision(thought=f"(mock) starting with {n}",
                            tool_call=ToolCall(f"mock-{n}", n, _MOCK_ARGS.get(n, {})))
        return Decision(thought="(mock) wrote up the disposition.",
                        final="(mock) Reviewed the case with the scoped tools and logged a "
                              "disposition. Run with a real engine for full reasoning.")


def get_provider(kind: str):
    kind = (kind or "mock").lower()
    if kind == "anthropic":
        return AnthropicProvider()
    if kind == "openai":
        return OpenAIProvider()
    if kind == "mock":
        return MockProvider()
    raise ValueError(f"unknown engine: {kind} (use 'anthropic', 'openai', or 'mock')")
