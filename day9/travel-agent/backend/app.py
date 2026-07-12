"""
app.py — the HTTP surface.

The frontend POSTs to /run with the chosen engine; we run the loop once and
return the turns it produced. The goal is FIXED (option A), so the only knob the
client sends is which engine drives the REASON step.

Run locally:   uvicorn app:app --reload --port 8000
Then open:     http://localhost:8000/        (the UI is served from /frontend)
Or test:       curl -s localhost:8000/run -H 'content-type: application/json' \
                    -d '{"provider":"mock"}' | python -m json.tool
"""
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from providers.base import get_provider
from tools.mock_directories import build_registry
from loop import run_loop

app = FastAPI(title="Travel Agent · Loop Runner")

# Option A — the goal is fixed; the engine is chosen per run.
FIXED_GOAL = "Goa · round-trip + 5★ hotel · 5 nights · under $5,000"


class RunRequest(BaseModel):
    provider: str = "mock"        # "mock" | "openai" | "anthropic"


@app.post("/run")
def run(req: RunRequest):
    provider = get_provider(req.provider)        # pick the engine
    registry = build_registry()                  # mock flight/hotel tools
    turns = run_loop(FIXED_GOAL, provider, registry)
    return {
        "goal": FIXED_GOAL,
        "provider": provider.name,
        "turns": [asdict(t) for t in turns],
    }


# Serve the static UI at "/".  (Mounted last so /run wins over the catch-all.)
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
