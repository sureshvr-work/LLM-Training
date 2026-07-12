# Travel Agent · Loop Runner

A single agent that plans a trip by running one small loop —
**sense → reason → act**, repeated until it books within budget.

The whole point of the design: **the loop never changes; only the engine behind
the REASON step does.** You can flip between three engines:

- **No model (mock)** — a canned, deterministic replay. Offline, free, no keys.
- **OpenAI** — `reason()` calls `/v1/chat/completions`.
- **Anthropic** — `reason()` calls `/v1/messages`.

The goal is **fixed** (option A): `Goa · round-trip + 5★ hotel · 5 nights · under $5,000`.

---

## Run it

**Fastest (no server, no keys):**
```bash
cd backend
python demo.py
```
Prints the four turns of the loop.

**As a server + UI:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# open http://localhost:8000/
```

**With Docker:**
```bash
cp .env.example .env        # leave blank for mock; add keys for live engines
docker compose up --build
# open http://localhost:8000/
```

Test the API directly:
```bash
curl -s localhost:8000/run -H 'content-type: application/json' \
     -d '{"provider":"mock"}' | python -m json.tool
```

The mock engine needs **no keys**. OpenAI / Anthropic need `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` in `.env`.

---

## The four seams (why it's split this way)

Each seam lives behind its own interface so it can be swapped or mocked alone.

| Seam | File | Job |
|------|------|-----|
| **loop** | `loop.py` | sense → reason → act, repeat. Knows nothing about vendors or tool internals. |
| **engine** | `providers/` | the only step that differs per vendor. One method: `reason(messages, tools) -> Decision`. |
| **tools** | `tools/` | functions the agent may call, by name. Mock now; live later — same contract. |
| **memory** | `memory.py` | short-term = the running message list (the growing prompt). |

```
travel-agent/
├─ docker-compose.yml
├─ .env.example
├─ frontend/
│  └─ index.html                  # the UI (engine switch + transcript)
└─ backend/
   ├─ schema.py                   # shared, vendor-neutral data types
   ├─ memory.py                   # short-term memory (the context window)
   ├─ loop.py                     # the agent loop
   ├─ app.py                      # FastAPI: POST /run, serves the UI
   ├─ demo.py                     # run the loop from the terminal
   ├─ tools/
   │  ├─ registry.py              # name -> function + spec; executes calls
   │  └─ mock_directories.py      # synthetic flight/hotel/booking/profile tools
   └─ providers/
      ├─ base.py                  # Provider contract + factory
      ├─ mock.py                  # "No model" — canned replay
      ├─ openai_provider.py       # OpenAI adapter   (stage 3)
      └─ anthropic_provider.py    # Anthropic adapter (stage 3)
```

---

## How a run flows

1. **SENSE (bootstrap)** — `loop.py` seeds memory with the goal and reads the
   user profile (`get_user_profile`).
2. **REASON** — `provider.reason(context, tools)` returns a `Decision`
   (a tool call, or a final answer). This is the only engine-specific step.
3. **ACT** — `registry.execute(call)` runs the chosen tool; the result is
   appended to memory (the context grows).
4. Repeat until the engine returns a final answer.

The mock engine's script: `search_flights → find_hotels → search_flights(±3d) →
book_flight + book_hotel + save_user_profile`. The first flight+hotel combo is
$5,130 (over $5,000), so the loop relaxes dates and re-searches — which is *why*
the loop earns its place instead of being a one-pass workflow.

---

## Build stages

1. ✅ UI mockup (the engine switch + transcript).
2. ✅ **This** — real loop + mock engine + mock tools, server-driven, offline.
3. ⬜ Wire the UI's Run/Step to `POST /run` (small change to `index.html`).
4. ⬜ Live engines: drop in `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, flip the switch.
5. ⬜ Swap one mock tool for a real outbound call.

## Adding a new engine

Write one class implementing `reason(messages, tools) -> Decision`, then add one
line to the factory in `providers/base.py`. The loop and the tools don't change.
