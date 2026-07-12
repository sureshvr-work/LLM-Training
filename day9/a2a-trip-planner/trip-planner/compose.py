# ---------------------------------------------------------------------------
# compose.py  ·  the ADAPTIVE client brain
# ---------------------------------------------------------------------------
# A2A messages carry no skillId — a client can only send natural-language TEXT,
# and the agent maps that text to a skill internally. So how does a client know
# how to phrase the text? The Agent Card's skill `examples` are the bridge.
#
# Two ways to cross that bridge:
#   template : the developer read the examples at build time and hardcoded the
#              shape. We just fill the slots. (No LLM — see app.py.)
#   adaptive : at run time we hand the card's description + examples to OUR OWN
#              LLM and let it phrase the message. This file does that.
# ---------------------------------------------------------------------------

import os
import httpx

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")


def provider_label() -> str:
    model = OPENAI_MODEL if PROVIDER == "openai" else ANTHROPIC_MODEL
    return f"{PROVIDER} \u00b7 {model}"


async def compose_message(card: dict, trip: dict):
    """Read a card's examples and phrase ONE A2A message for this trip.
    Returns (message_text, trace)."""
    name = card.get("name", "the agent")
    desc = card.get("description", "")
    skill = (card.get("skills") or [{}])[0]
    examples = skill.get("examples", [])

    system = (
        "You are an A2A client composing a message to another agent. You are given that "
        "agent's description and the example messages it understands. Write ONE natural-language "
        "message, in the same style as the examples, that asks for exactly what the user's trip "
        "needs. Reply with ONLY the message text — no quotes, no preamble, no explanation."
    )
    user = (
        f"Agent: {name}\n"
        f"Description: {desc}\n"
        f"Example messages it understands:\n" + "\n".join(f"- {e}" for e in examples) +
        f"\n\nUser's trip: from {trip.get('origin')} to {trip.get('destination')}, "
        f"date {trip.get('date')} ({trip.get('when')}).\n"
        "Write the single message this agent needs."
    )

    text = (await _complete(system, user)).strip().strip('"').strip()

    trace = [
        {"phase": "READ", "kind": "read", "label": "read the card's examples",
         "detail": (examples[0] if examples else "(no examples on card)")},
        {"phase": "REASON", "kind": "decision", "label": f"asked {provider_label()} to phrase it",
         "detail": f"style of {len(examples)} example(s) + this trip"},
        {"phase": "COMPOSE", "kind": "final", "label": "LLM composed the A2A message",
         "detail": text},
    ]
    return text, trace


# ---- plain text completion, per provider ----------------------------------
def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"{key} is not set — add it to .env and pass it to the trip-planner service")
    return val


async def _complete(system, user) -> str:
    if PROVIDER == "anthropic":
        api_key = _require("ANTHROPIC_API_KEY")
        async with httpx.AsyncClient(timeout=40.0) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                json={"model": ANTHROPIC_MODEL, "max_tokens": 200, "system": system,
                      "messages": [{"role": "user", "content": user}]},
            )
        r.raise_for_status()
        blocks = r.json()["content"]
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    api_key = _require("OPENAI_API_KEY")
    async with httpx.AsyncClient(timeout=40.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": OPENAI_MODEL,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content", "") or ""
