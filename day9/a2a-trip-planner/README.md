# A2A Trip Planner — agent-to-agent, live

A **Trip Planner** agent discovers two specialist agents and calls them over
A2A, then the UI composes their answers into a weather card and a flights board.

```
weather-agent ─┐
flights-agent ─┤─ a2a-net ─►  trip-planner ─►  ui  (http://localhost:8080)
               ┘   discovered + messaged over A2A
```

- **Weather Agent** — real Open-Meteo (no key). Skill: `get-forecast`.
- **Flights Agent** — real flight search. Skill: `search-flights`.
  - **No key needed to demo:** with no flights key set, the tool returns
    route-aware **sample** data (realistic times/prices, labelled in the UI),
    so the A2A + LLM flow always has something to show.
  - **Real API (optional):** `FLIGHTS_PROVIDER=duffel` + `DUFFEL_TOKEN`.
    Free ~2-min signup at duffel.com (Developers -> test token). Note: Duffel
    *test mode* returns sandbox offers, not real live fares.
  - `aviationstack` = live route status. `amadeus` = legacy (self-service
    portal is being decommissioned 2026-07-17 — new signups already closed).
- **UI** — dynamic From/To/day; renders a weather card + flights board and a
  live trace of the A2A calls.

The two specialist agents use a **real LLM** (OpenAI or Anthropic) to read the
message, decide the tool call, and phrase the answer. **No mock.**

## Run

```bash
cp .env.example .env         # add OPENAI_API_KEY (or ANTHROPIC); DUFFEL_TOKEN optional
docker compose up -d --build
# open http://localhost:8080
```

Offline / no keys for the data APIs? Set `WEATHER_STUB=1` and/or `FLIGHTS_STUB=1`
in `.env` (these stub only the data APIs — the LLM is still real).

Curl the agents directly (REST discovery):

```bash
curl http://localhost:8001/.well-known/agent-card.json
curl http://localhost:8002/.well-known/agent-card.json
```

## What to show in class

1. **Agent-to-agent.** Type a trip → the trace shows the Trip Planner
   *discovering* each agent (REST) and *sending* each a `message/send`
   (JSON-RPC). The questions are formed dynamically from your input.
2. **Inside each agent.** Each specialist runs the loop: message → LLM decides a
   tool call → real API (Open-Meteo / AviationStack) → LLM phrases the answer →
   Task + Artifact. (Each agent's internal trace comes back in `internal` too.)
3. **Structured artifacts.** Agents return both a sentence *and* a `data` part;
   the UI renders the data as a weather card / flights board.
4. **Bindings.** Discovery = REST `GET`, messaging = JSON-RPC. See
   `BINDINGS.md` for how gRPC differs (and why it's not in the live demo).

## Files (each agent is one simple folder)

```
weather-agent/  app.py · llm.py · weather_api.py     (Open-Meteo)
flights-agent/  app.py · llm.py · flights_api.py     (AviationStack)
trip-planner/   app.py · a2a_client.py               (REST discover + JSON-RPC send)
ui/             app.py · index.html
```

`llm.py` is the same file in both agents: one real-LLM tool loop, provider-
switchable (`LLM_PROVIDER=openai|anthropic`). The agent's `app.py` just defines
its one tool and wraps the result into an A2A Task.

## Config (.env)

```
LLM_PROVIDER=openai            # openai | anthropic
OPENAI_API_KEY=...             OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=...          ANTHROPIC_MODEL=claude-3-5-sonnet-latest
AVIATIONSTACK_KEY=...
WEATHER_STUB=   FLIGHTS_STUB=  # 1 = canned data (offline)
```

## The handshake, in 3 stages (what the UI shows)

The page now walks the A2A handshake so a class can watch each step:

1. **Discover** — REST `GET /.well-known/agent-card.json` for both agents. Each card's
   skill `examples` are shown — that's the bridge (an A2A message has no skillId, so the
   example is how a client learns to phrase its text).
2. **Compose** — turn the example into the actual message to send, two ways:
   - `template` — the planner already knows these agents, so it fills the example's shape
     with your trip. No LLM.
   - `adaptive` — your LLM (LLM_PROVIDER + key) reads each card's examples at run time and
     phrases the message in that style. The composed message is shown *before* it's sent.
3. **Send** — each message goes out over JSON-RPC (`message/send`); each agent runs its own
   SENSE · REASON · ACT loop and returns a Task + Artifact.

Endpoints: `POST /discover`, `POST /compose` (`mode=template|adaptive`), `POST /send`.

## Flight Status agent (streaming)

`flights-status-agent` (port 8004) answers with A2A's **message/stream** shape — one request, an open SSE connection, the flight walked from *scheduled* to *landed*. The Trip Planner relays that stream to the browser (`POST /status/stream`). Keyless by default (deterministic sample flight); `STATUS_PROVIDER=aviationstack` fetches a real snapshot.


## Booking agent (multi-turn)

`flights-book-agent` (port 8005) is the demo's **input-required** agent. Turn 1 (passenger + flight) comes back `input-required` — the task pauses and asks for seat + date of birth. You answer on the **same taskId** and it resumes to `completed` with a (sample) PNR. Shows Task state living across turns, threaded by `taskId`/`contextId`. Keyless: booking a real seat needs a live offer + payment, so the confirmation is a clearly-labelled sample.
