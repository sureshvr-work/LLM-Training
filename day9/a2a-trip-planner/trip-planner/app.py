# ---------------------------------------------------------------------------
# Trip Planner  ·  an A2A orchestrator (client of the other two agents)
# ---------------------------------------------------------------------------
# It is itself an A2A agent (it publishes a card), but its job is to CALL other
# agents. Given a trip (origin, destination, day) it:
#
#   1. discovers the Weather + Flights agents        (REST)
#   2. forms a natural-language question for each     (dynamic)
#   3. sends each an A2A message/send                 (JSON-RPC, agent-to-agent)
#   4. collects their Task artifacts and returns them to the UI
#
# No LLM here on purpose — the intelligence lives in the specialist agents.
# The Trip Planner just coordinates. That is A2A: agents composing agents.
# ---------------------------------------------------------------------------

import os
import json
import datetime
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import a2a_client
import compose

app = FastAPI(title="Trip Planner")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WEATHER_URL = os.getenv("WEATHER_URL", "http://weather-agent:8001")
FLIGHTS_URL = os.getenv("FLIGHTS_URL", "http://flights-agent:8002")
STATUS_URL = os.getenv("STATUS_URL", "http://flights-status-agent:8004")
BOOK_URL = os.getenv("BOOK_URL", "http://flights-book-agent:8005")

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "Trip Planner",
    "description": "Plans a trip by asking the Weather and Flights agents.",
    "url": "http://trip-planner:8003/",
    "version": "1.0.0",
    "preferredTransport": "JSONRPC",
    "capabilities": {"streaming": False},
    "skills": [{
        "id": "plan-trip",
        "name": "Plan a trip",
        "description": "Given an origin and destination, gather weather and flights.",
        "tags": ["travel"],
        "examples": ["Plan a trip from Boston to Tokyo tomorrow"],
    }],
}


@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


@app.get("/health")
def health():
    return {"ok": True}


def _part(task, key):
    """Pull the text or data part out of a Task's first artifact."""
    try:
        for p in task["artifacts"][0]["parts"]:
            if key in p:
                return p[key]
    except (KeyError, IndexError, TypeError):
        pass
    return None


@app.post("/plan")
async def plan(body: dict):
    origin = (body.get("origin") or "Boston").strip()
    destination = (body.get("destination") or "Tokyo").strip()
    when = (body.get("when") or "tomorrow").strip()

    trace = []

    # 1) discover both agents (REST GET of the Agent Card)
    _, weather_disc = await a2a_client.discover(WEATHER_URL, trace)
    _, flights_disc = await a2a_client.discover(FLIGHTS_URL, trace)

    # 2) form dynamic questions from the user's trip
    weather_q = f"What's the weather in {destination} {when}?"
    # flights need a concrete future date; derive it from `when` (today/tomorrow)
    _offset = 0 if when.lower() == "today" else 1
    dep_date = (datetime.date.today() + datetime.timedelta(days=_offset)).isoformat()
    flights_q = f"Find flights from {origin} to {destination} on {dep_date}."

    # 3) send each agent an A2A message (JSON-RPC, agent-to-agent)
    weather_task = await a2a_client.send(WEATHER_URL, weather_q, trace)
    flights_task = await a2a_client.send(FLIGHTS_URL, flights_q, trace)

    # 4) collect their artifacts (structured data + the sentence + how they got there).
    #    Each panel's timeline starts with the discovery GET, then the agent's own
    #    SENSE -> REASON -> ACT steps.
    def meta(task, key):
        return (task.get("metadata") or {}).get(key)

    return {
        "trip": {"origin": origin, "destination": destination, "when": when},
        "questions": {"weather": weather_q, "flights": flights_q},
        "trace": trace,
        "weather": {
            "provider": meta(weather_task, "provider"),
            "steps": [weather_disc] + (meta(weather_task, "trace") or []),
            "data": _part(weather_task, "data"),
            "text": _part(weather_task, "text"),
        },
        "flights": {
            "provider": meta(flights_task, "provider"),
            "steps": [flights_disc] + (meta(flights_task, "trace") or []),
            "data": _part(flights_task, "data"),
            "text": _part(flights_task, "text"),
        },
    }


# ===========================================================================
# STAGED FLOW  ·  discover -> compose -> send  (so the class can watch each step)
# ===========================================================================
def _card_summary(card: dict) -> dict:
    skill = (card.get("skills") or [{}])[0]
    return {
        "name": card.get("name"),
        "description": card.get("description"),
        "url": card.get("url"),
        "transport": card.get("preferredTransport", "JSONRPC"),
        "interfaces": [i.get("transport") for i in (card.get("additionalInterfaces") or [])],
        "streaming": (card.get("capabilities") or {}).get("streaming", False),
        "skill": {"id": skill.get("id"), "name": skill.get("name"),
                  "examples": skill.get("examples", [])},
    }


def _dep_date(when: str) -> str:
    offset = 0 if (when or "").lower() == "today" else 1
    return (datetime.date.today() + datetime.timedelta(days=offset)).isoformat()


@app.post("/discover")
async def discover(body: dict):
    """STAGE 1 — REST GET all four Agent Cards, return their skills + examples."""
    wcard, wdisc = await a2a_client.discover(WEATHER_URL, [])
    fcard, fdisc = await a2a_client.discover(FLIGHTS_URL, [])
    scard, sdisc = await a2a_client.discover(STATUS_URL, [])
    bcard, bdisc = await a2a_client.discover(BOOK_URL, [])
    return {
        "weather": {"card": _card_summary(wcard), "discovery": wdisc},
        "flights": {"card": _card_summary(fcard), "discovery": fdisc},
        "status": {"card": _card_summary(scard), "discovery": sdisc},
        "book": {"card": _card_summary(bcard), "discovery": bdisc},
    }


@app.post("/compose")
async def compose_messages(body: dict):
    """STAGE 2 — turn each card's example into the actual A2A message to send.
    mode='template' (fill the slots) or mode='adaptive' (our LLM phrases it)."""
    mode = (body.get("mode") or "template").lower()
    origin = (body.get("origin") or "Boston").strip()
    destination = (body.get("destination") or "London").strip()
    when = (body.get("when") or "tomorrow").strip()
    date = _dep_date(when)
    trip = {"origin": origin, "destination": destination, "when": when, "date": date}

    # we need the cards' examples either way
    wcard, _ = await a2a_client.discover(WEATHER_URL, [])
    fcard, _ = await a2a_client.discover(FLIGHTS_URL, [])
    w_ex = ((wcard.get("skills") or [{}])[0].get("examples") or [""])[0]
    f_ex = ((fcard.get("skills") or [{}])[0].get("examples") or [""])[0]

    if mode == "adaptive":
        try:
            w_msg, w_trace = await compose.compose_message(wcard, trip)
            f_msg, f_trace = await compose.compose_message(fcard, trip)
        except Exception as exc:  # noqa: BLE001 (surface any LLM/key error to the UI)
            return {"error": f"adaptive compose failed: {exc}"}
        return {
            "mode": "adaptive", "provider": compose.provider_label(),
            "weather": {"message": w_msg, "example": w_ex, "trace": w_trace,
                        "note": "Your LLM read the card's examples and phrased this."},
            "flights": {"message": f_msg, "example": f_ex, "trace": f_trace,
                        "note": "Your LLM read the card's examples and phrased this."},
        }

    # template mode — the developer already knew these agents; fill the shape
    w_msg = f"What's the weather in {destination} {when}?"
    f_msg = f"Find flights from {origin} to {destination} on {date}."
    note = "Filled the example's shape with your trip — no LLM."
    return {
        "mode": "template",
        "weather": {"message": w_msg, "example": w_ex, "note": note},
        "flights": {"message": f_msg, "example": f_ex, "note": note},
    }


@app.post("/send")
async def send_messages(body: dict):
    """STAGE 3 — send each composed message over the agent's PREFERRED transport
    (JSON-RPC for Weather, HTTP+JSON/REST for Flights) and collect Tasks."""
    w_msg = body.get("weather_message") or ""
    f_msg = body.get("flights_message") or ""

    # read each card to learn which binding to use (that's the client's job)
    wcard, _ = await a2a_client.discover(WEATHER_URL, [])
    fcard, _ = await a2a_client.discover(FLIGHTS_URL, [])
    w_tr = wcard.get("preferredTransport", "JSONRPC")
    f_tr = fcard.get("preferredTransport", "JSONRPC")

    weather_task = await a2a_client.send(WEATHER_URL, w_msg, [], w_tr)
    flights_task = await a2a_client.send(FLIGHTS_URL, f_msg, [], f_tr)

    def meta(task, key):
        return (task.get("metadata") or {}).get(key)

    return {
        "weather": {"transport": w_tr, "provider": meta(weather_task, "provider"),
                    "steps": meta(weather_task, "trace"),
                    "data": _part(weather_task, "data"), "text": _part(weather_task, "text")},
        "flights": {"transport": f_tr, "provider": meta(flights_task, "provider"),
                    "steps": meta(flights_task, "trace"),
                    "data": _part(flights_task, "data"), "text": _part(flights_task, "text")},
    }


# ===========================================================================
# STREAMING  ·  the orchestrator opens the Status agent's SSE and re-emits it
# ===========================================================================
@app.post("/status/stream")
async def status_stream(body: dict):
    """Open a message/stream to the Flight Status agent and pass each SSE event
    straight through to the browser. The orchestrator is a stream *relay* here —
    that passthrough is the one new piece the streaming lesson needs."""
    flight = (body.get("flight") or "").strip()
    text = body.get("message") or (
        f"What's the status of flight {flight}?" if flight else "What's the status of my flight?")

    async def gen():
        # 1) discovery first, so the UI can show the card + streaming capability
        scard, _ = await a2a_client.discover(STATUS_URL, [])
        yield "data: " + json.dumps({"kind": "card", "card": _card_summary(scard),
                                     "streaming": (scard.get("capabilities") or {}).get("streaming", False)}) + "\n\n"
        # 2) relay every streamed event as it arrives
        async for result in a2a_client.stream(STATUS_URL, text, []):
            yield "data: " + json.dumps(result) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ===========================================================================
# MULTI-TURN  ·  booking is a task that pauses at input-required and resumes
# ===========================================================================
def _book_view(task: dict, sent: str) -> dict:
    """Flatten a Booking Task for the UI, keeping the taskId/contextId thread visible."""
    meta = task.get("metadata") or {}
    status = task.get("status") or {}
    parts = (status.get("message") or {}).get("parts") or []
    question = parts[0].get("text") if parts else None
    return {
        "taskId": task.get("id"), "contextId": task.get("contextId"),
        "state": status.get("state"), "sent": sent, "question": question,
        "missing": meta.get("missing"), "have": meta.get("have"),
        "provider": meta.get("provider"), "steps": meta.get("trace"),
        "data": _part(task, "data"), "text": _part(task, "text"),
    }


@app.post("/book/start")
async def book_start(body: dict):
    """Turn 1 — send passenger + flight. The task usually comes back input-required."""
    passenger = (body.get("passenger") or "John Smith").strip()
    flight = (body.get("flight") or "BA212").strip()
    text = f"Book flight {flight} for {passenger}."
    bcard, _ = await a2a_client.discover(BOOK_URL, [])
    task = await a2a_client.send(BOOK_URL, text, [], bcard.get("preferredTransport", "JSONRPC"))
    return _book_view(task, sent=text)


@app.post("/book/reply")
async def book_reply(body: dict):
    """Turn 2 — answer the agent's question ON THE SAME taskId; task completes."""
    task_id = body.get("taskId")
    context_id = body.get("contextId")
    answer = (body.get("answer") or "").strip()
    task = await a2a_client.send(BOOK_URL, answer, [], "JSONRPC",
                                 task_id=task_id, context_id=context_id)
    return _book_view(task, sent=answer)
