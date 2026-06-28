# Trip Agent · live

A **real** agent loop your students can watch think. Type a goal in plain English;
the agent resolves the destination, pulls **real weather**, finds **real places**,
checks **real flight schedules**, and **searches the live web** — each call
streaming into the browser with its true latency. Two interchangeable LLM engines
(OpenAI / Anthropic) drive the reasoning.

This is built **vanilla on purpose** — no agent framework. The loop *is* the
lesson, so it stays readable. Four clean seams:

```
loop.py      the agent loop      — sense → reason → act, with a turn cap & stop condition
engine.py    the provider seam   — OpenAI & Anthropic behind one reason() contract
tools/       the tool seam       — 5 live clients, each schema-validated before it runs
http_client  the network seam    — one place for timeout, retry, backoff, 429 handling
```

## The five live tools

| Tool | Provider | Key | What it returns |
|---|---|---|---|
| `geocode_place` | Geoapify | `GEOAPIFY_KEY` | place → lat/lon/country |
| `get_weather` | Open-Meteo | **none** | 7-day highs/lows/rain |
| `find_places` | Geoapify Places | `GEOAPIFY_KEY` | real hotels / attractions / restaurants |
| `web_search` | Tavily | `TAVILY_API_KEY` | live, AI-native search results |
| `search_flights` | AviationStack | `AVIATIONSTACK_KEY` | real flight schedule/status (not fares) |

> **Honest scope:** no free aviation API sells fares or inventory, so flights are
> *schedule/status* data — the agent plans around real flights, it doesn't book.
> That's the realistic shape; booking would add a **human-approval gate** (see below).

## Run it

**Local**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env      # then paste your keys into ../.env
uvicorn app:app --reload --port 8000
# open http://localhost:8000/
```

**Docker**
```bash
cp .env.example .env            # paste your keys
docker compose up --build
# open http://localhost:8000/
```

You need **at least one** LLM key (OpenAI or Anthropic) plus the tool keys you
want exercised. Open-Meteo needs none. With no keys at all, the UI still runs an
**offline sample** so you can see the shape.

## What makes it architect-grade (the talking points)

- **Dynamic goal, no script.** The model chooses which tool, with what args, in
  what order, and when it's done. Determinism would come from a mock engine — not
  from the UI. Clicking controls *pace*, never *outcome*.
- **Secrets never leave the server.** The browser only ever receives events; keys
  live in `config.py` ← env. Every vendor call is server-side.
- **Every call survives the real world.** `http_client.py` gives all tools a
  timeout, retries with exponential backoff, and explicit 429 handling.
- **Hallucinated args can't crash it.** `registry.call()` validates arguments
  against each tool's JSON schema *before* invoking it; bad args become a clean
  error the model can recover from.
- **Tool failures are recoverable, not fatal.** The loop catches errors and feeds
  them back as tool results, so the model adapts (the sample shows a 429 → retry).
- **Empty results trigger a smart retry, not a hallucination.** If `find_places`
  comes back empty, the agent widens `radius_m` (e.g. 5km → 15km) once before
  giving up — a visible re-plan beat on the loop. The final plan is constrained to
  what the tools actually returned: if a search stayed empty, it says so honestly
  instead of inventing names, and it respects the sensed weather.
- **Auditable, streamed trace.** Each turn streams as SSE with its latency and
  token usage — the UI tallies turns, total latency, tokens, and per-tool counts.
- **Guardrail discipline.** These five tools are read-only. The one place an
  irreversible tool (book/pay) would go is exactly where you'd require a human
  gate — same spirit as a "don't auto-submit" rule in production RPA.
- **Vendor-swap proof.** Bing's search API was shut down in 2025; keeping each
  tool behind the contract means a dead vendor is a one-file change. Same for the
  engine — OpenAI ↔ Anthropic is a toggle.

## Layout

```
backend/
  config.py        secrets + tunables (the only os.environ reader)
  http_client.py   shared resilient HTTP (timeout / retry / 429)
  engine.py        OpenAIProvider + AnthropicProvider, real tool-calling
  loop.py          the dynamic agent loop (a generator of events)
  app.py           FastAPI: POST /run streams SSE; serves the UI
  tools/
    registry.py    @tool decorator + schema-validated call()
    geocode.py weather.py places.py websearch.py flights.py
frontend/
  index.html       goal in; live streamed turns with latency + token stats
.env.example  docker-compose.yml  backend/Dockerfile
```

## Teaching path (suggested)

1. Open `loop.py` — read `sense → reason → act` top to bottom.
2. Open `engine.py` — see the *same* contract satisfied two ways.
3. Open one `tools/*.py` — a tool is just a function + a schema.
4. Run it live, project the screen, give it a goal, and let the class watch real
   latency and a real retry happen. Then break a key on purpose and watch it
   recover. That's the wow — and the lesson.
