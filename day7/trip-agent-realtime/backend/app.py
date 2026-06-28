"""
app.py — POST /run streams the loop's events as Server-Sent Events.

Streaming is the point: the browser shows each real tool call and its latency
the instant it happens, instead of waiting for the whole plan. Keys stay here on
the server; the browser only ever sees events.

    uvicorn app:app --reload --port 8000   ->   http://localhost:8000/
"""
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import loop as agent
from engine import get_provider
from notify import notify_plan_ready

app = FastAPI(title="Trip Agent · live")


class RunReq(BaseModel):
    goal: str
    engine: str = "openai"       # "openai" | "anthropic"
    notify_email: str = ""       # optional — emails the plan here when done


@app.post("/run")
def run(req: RunReq):
    def gen():
        try:
            provider = get_provider(req.engine)
        except Exception as e:
            yield _sse({"type": "error", "where": "engine", "message": str(e)})
            return
        plan_text = None
        for event in agent.run(req.goal, provider):
            if event.get("type") == "final":
                plan_text = event.get("text")
            yield _sse(event)
        if plan_text:
            print("notify:", notify_plan_ready(req.goal, plan_text, req.notify_email))
    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# Serve the UI at "/" (mounted last so /run wins).
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
