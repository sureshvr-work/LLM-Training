# ---------------------------------------------------------------------------
# llm.py  ·  the agent's SENSE -> REASON -> ACT loop
# ---------------------------------------------------------------------------
# This is the whole "brain" of an agent, split into three named phases so the
# separation is obvious when you read it:
#
#     SENSE   assemble the context the model will see this turn
#             (system role + the one tool + the message + any tool results)
#     REASON  the model reads that context and PROPOSES an action:
#             either a tool call, or the final text. It only proposes.
#     ACT     the agent -- NOT the model -- executes the proposed tool.
#
# Then we loop once: SENSE again (now with the tool's result folded in) and
# REASON again to phrase the answer. The model is the intelligence; the agent
# holds all the agency. Provider is OpenAI or Anthropic (LLM_PROVIDER), no mock.
# ---------------------------------------------------------------------------

import os
import json
import httpx

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")


def provider_label() -> str:
    model = OPENAI_MODEL if PROVIDER == "openai" else ANTHROPIC_MODEL
    return f"{PROVIDER} \u00b7 {model}"


# ===========================================================================
# THE LOOP  —  read this top-to-bottom to see SENSE -> REASON -> ACT
# ===========================================================================
async def run(system: str, user_text: str, tool: dict, run_tool):
    """One tool-using turn. Returns (final_text, tool_result, trace)."""
    trace = []
    history = []                                   # (decision, result) gathered this turn

    # SENSE -> REASON : look at the message, decide what to do
    context = sense(system, user_text, tool, history, trace)
    decision = await reason(context, allow_tool=True, trace=trace)
    if decision["type"] != "tool_use":            # model answered without a tool
        return decision["text"], None, trace

    # ACT : run the tool the model asked for (the model never touches it)
    result = await act(decision, run_tool, trace)
    history.append((decision, result))

    # SENSE -> REASON : fold the result back in, phrase the final answer
    context = sense(system, user_text, tool, history, trace)
    final = await reason(context, allow_tool=False, trace=trace)
    return final["text"], result, trace


# ---- SENSE ----------------------------------------------------------------
def sense(system, user_text, tool, history, trace):
    """SENSE — gather everything the model should see this turn."""
    first = not history
    trace.append({
        "phase": "SENSE",
        "kind": "received" if first else "prompt",
        "label": "SENSE \u00b7 assemble context",
        "detail": (f'message "{user_text}" + tool({tool["name"]}) + system' if first
                   else f"re-read with {len(history)} tool result folded in"),
    })
    return {"system": system, "user_text": user_text, "tool": tool, "history": history}


# ---- REASON ---------------------------------------------------------------
async def reason(context, allow_tool, trace):
    """REASON — the model proposes a tool call, or writes the final text."""
    out = await _model(context, allow_tool)
    if out["type"] == "tool_use":
        trace.append({"phase": "REASON", "kind": "decision", "label": "REASON \u00b7 chose a tool",
                      "detail": f"{out['name']}({json.dumps(out['input'])})"})
    else:
        trace.append({"phase": "REASON", "kind": "final", "label": "REASON \u00b7 wrote the answer",
                      "detail": out["text"]})
    return out


# ---- ACT ------------------------------------------------------------------
async def act(decision, run_tool, trace):
    """ACT — the agent (not the model) executes the proposed tool."""
    result = await run_tool(decision["name"], decision["input"])
    trace.append({"phase": "ACT", "kind": "tool", "label": "ACT \u00b7 ran the tool (real API)",
                  "detail": _short(result)})
    return result


def _short(result) -> str:
    s = json.dumps(result, default=str)
    return s if len(s) <= 160 else s[:157] + "..."


# ===========================================================================
# REASON internals — the actual model call, per provider. Given the SENSE
# context, build the provider's message list and return a proposed action.
# ===========================================================================
async def _model(context, allow_tool):
    if PROVIDER == "anthropic":
        return await _anthropic(context, allow_tool)
    return await _openai(context, allow_tool)


# ---- OpenAI (chat/completions function calling) ---------------------------
async def _openai(context, allow_tool):
    messages = [{"role": "system", "content": context["system"]},
                {"role": "user", "content": context["user_text"]}]
    for decision, result in context["history"]:
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": "call_1", "type": "function",
                                         "function": {"name": decision["name"],
                                                      "arguments": json.dumps(decision["input"])}}]})
        messages.append({"role": "tool", "tool_call_id": "call_1",
                         "content": json.dumps(result, default=str)})
    tools = [{"type": "function", "function": context["tool"]}] if allow_tool else None
    payload = {"model": OPENAI_MODEL, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    async with httpx.AsyncClient(timeout=40.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions",
                              headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                              json=payload)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    calls = msg.get("tool_calls")
    if calls:
        fn = calls[0]["function"]
        return {"type": "tool_use", "name": fn["name"], "input": json.loads(fn["arguments"] or "{}")}
    return {"type": "text", "text": msg.get("content", "") or ""}


# ---- Anthropic (messages API tool use) ------------------------------------
async def _anthropic(context, allow_tool):
    messages = [{"role": "user", "content": context["user_text"]}]
    for decision, result in context["history"]:
        messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                         "name": decision["name"], "input": decision["input"]}]})
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                         "content": json.dumps(result, default=str)}]})
    payload = {"model": ANTHROPIC_MODEL, "max_tokens": 500,
               "system": context["system"], "messages": messages}
    if allow_tool:
        t = context["tool"]
        payload["tools"] = [{"name": t["name"], "description": t["description"],
                             "input_schema": t["parameters"]}]
    async with httpx.AsyncClient(timeout=40.0) as client:
        r = await client.post("https://api.anthropic.com/v1/messages",
                              headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                       "anthropic-version": "2023-06-01"},
                              json=payload)
    r.raise_for_status()
    blocks = r.json()["content"]
    for b in blocks:
        if b.get("type") == "tool_use":
            return {"type": "tool_use", "name": b["name"], "input": b["input"]}
    return {"type": "text", "text": "".join(b.get("text", "") for b in blocks if b.get("type") == "text")}
